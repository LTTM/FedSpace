import copy
import logging
import os
import numpy as np
import torch
import wandb
import pickle
from client.client_CL import Client
from data.data_loader_CL import get_dataloader_test
from client.my_model_trainer_classification_CL import LOSS_KEYS
import time
from client.proto_queue import ProtoQueue


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_local_num_dict, train_data_local_dict, train_data_task_dict, class_num, task_classes] = dataset

        self.task_classes = task_classes
        self.class_num = class_num
        self.cumulative_output_classes = np.cumsum([len(self.task_classes[task]) for task in self.task_classes])
        self.task_size = args.total_nc if self.args.one_task else int((self.class_num - self.args.num_classes_first_task) / self.args.task_num)
        self.file_name = f'{self.args.name}'

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict

        self.client_protos = dict()
        self.global_radiuses = dict()

        self.old_model = None

        self.model_trainer = model_trainer

        if self.args.model == "mobilenet":
            self.feature_size = 1024
        elif self.args.model == "resnet_18":
            self.feature_size = 512

        self.proto_global = None
        self.class_label = None

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, model_trainer)

        self.global_discovered_tasks = set()
        self.global_discovered_classes = []
        self.rounds_per_task = None
        self.task_order = None
        self.global_task_id = None
        self.global_current_tasks = []
        self.global_current_tasks_per_round = []
        self.aggregated_proto_flag = False

        self.proto_queue = ProtoQueue(n_classes=args.total_nc, max_length=args.proto_queue_length)
        self.current_client_indexes = None

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                       self.args, self.device, model_trainer, self.feature_size)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def centralized_fractal_pretraining(self):
        self.model_trainer.fractal_pretrain(client=None, args=self.args, sec_device=self.device)
        return self.model_trainer.model

    def federated_fractal_pretraining(self):
        w_global = self.model_trainer.get_model_params()

        for round_idx in range(self.args.fractal_pretrain_rounds):

            print(f"################Pretrain - Communication round {round_idx}")
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            clients_loss = []
            w_locals = []
            for idx in client_indexes:
                client = self.client_list[idx]
                w, client_loss = client.fractal_pretrain(
                    copy.deepcopy(w_global),
                )
                clients_loss.append(client_loss)
                w_locals.append((10, copy.deepcopy(w), idx))


            print(f"Aggregating model weights...")
            w_global = self._aggregate_w(w_locals, None, round_idx)
            self.model_trainer.set_model_params(w_global)

        torch.save(self.model_trainer.model, "federated_fractal_pretrain_model.pth")

        print(f"End federated fractal pretraining...")

    def train_protos_global_local(self):

        initial_lr = self.args.lr

        self.model_trainer.update_output_dim(100)

        w_global = self.model_trainer.get_model_params()

        mask = None
        # list storing the indexes of the clients that contributed in the federated training for the current task
        workers = list()

        if self.args.same_time:
            mean = int(self.args.comm_rounds / (self.args.task_num + 1))
            self.rounds_per_task = [mean] * self.args.task_num + [self.args.comm_rounds - mean * self.args.task_num]
            self.task_order = np.arange(self.args.task_num + 1)
            self.global_task_id = 0

        old_classes = 0

        selected_clients_per_round = {}
        initial_time = time.time()

        for round_idx in range(self.args.comm_rounds):

            if round_idx == self.args.stop_at_round:
                exit()
            proto_locals = dict()
            radius_locals = dict()
            client_losses = []

            print(f"################Communication round {round_idx} - time {round(time.time()-initial_time, 2)}")
            w_locals = list()

            if self.args.same_time and round_idx + 1 > np.cumsum(self.rounds_per_task)[self.global_task_id]:
                self.global_task_id += 1
                if self.args.setup == 'federated':
                    self.args.lr = initial_lr

            if (round_idx + 1) % self.args.n_rounds_scheduling == 0:
                self.args.lr *= self.args.lr_scheduler_multiplier

            # sample client_num_per_round that will contribute to the federated training
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            # for each selected client, store True if task was updated during this round, False if no update occurred
            update_client_task = {}
            self.global_current_tasks = []

            selected_clients_per_round[round_idx] = client_indexes
            self.current_client_indexes = client_indexes

            # update server list of discovered tasks / classes
            for idx in client_indexes:
                client = self.client_list[idx]

                update_task = client.update_local_dataset(round_idx, self.train_data_local_dict[idx],
                                                            self.train_data_local_num_dict[idx],
                                                            global_task=self.global_task_id)
                update_client_task[idx] = update_task

                client_tasks = client.task_order[:client.task_index+1]
                client.class_label = [class_id for task in client_tasks for class_id in self.task_classes[task]]
                client.current_classes = [class_id for class_id in self.task_classes[client.current_task]]

                if self.args.same_time:
                    self.global_discovered_tasks = self.task_order[:self.global_task_id+1]
                else:
                    self.global_discovered_tasks = set.union(set(self.global_discovered_tasks), set(client_tasks))
                    self.global_current_tasks.append(client_tasks[-1])

            if self.global_current_tasks:
                global_current_discovered_classes = \
                    [class_id for task in self.global_current_tasks for class_id in self.task_classes[task]]
                print(f"server current tasks: {self.global_current_tasks}")

                self.global_current_tasks = np.sort(self.global_current_tasks).tolist()
                self.global_current_tasks_per_round.append(self.global_current_tasks)

            self.global_discovered_tasks = np.sort(list(self.global_discovered_tasks))
            self.global_discovered_classes = \
                [class_id for task in self.global_discovered_tasks for class_id in self.task_classes[task]]

            print(f"server total tasks: {self.global_discovered_tasks}")

            self.model_trainer.current_classes = sorted(list(set([class_id for task in self.global_current_tasks
                                                                   for class_id in self.task_classes[task]])))

            for idx in client_indexes:
                client = self.client_list[idx]

                if self.args.mask_model:
                    mask = sorted([a * 4 + i for a in client.class_label for i in range(4)])

                # before training, update clients with latest global prototypes and radius
                if self.proto_global:
                    self._update_proto_radius_labels(client)

                # train on new dataset
                w, client_loss, num_sample_class = client.train(
                    copy.deepcopy(w_global),
                    old_classes,
                    self.old_model,
                    mask=mask,
                    round=round_idx,
                    proto_queue=self.proto_queue)

                client_losses.append(client_loss)

                if self.args.same_time:
                    current_task = self.global_task_id
                else:
                    current_task = client.current_task

                radius, prototype, class_label = client.proto_save(self.task_classes[current_task])
                proto_locals[idx] = {'sample_num': client.get_sample_number(),
                                     'prototype': prototype,
                                     'num_samples_class': num_sample_class}
                radius_locals[idx] = {'sample_num': client.get_sample_number(),
                                      'radius': radius}

                client.prototype["local"] = {**client.prototype["local"], **prototype}
                client.radius["local"] = radius

                self.proto_queue.insert(prototype, radius, num_sample_class)

                client.num_sample_class = num_sample_class

                if w is not None:
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w), idx))
                    if idx not in workers:
                        workers.append(idx)

            for key in LOSS_KEYS:
                wandb.log({f"{key}": np.mean([c_loss[key] for c_loss in client_losses]), "round": round_idx})

            # update global weights
            print(f"Aggregating model weights...")
            old_w_global = copy.deepcopy(w_global)
            w_global = self._aggregate_w(w_locals, proto_locals, round_idx)
            if self.args.aggregate_with_global_model:
                for k in w_global.keys():
                    w_global[k] = w_global[k]*self.args.global_weight + old_w_global[k]*(1-self.args.global_weight)
            self.model_trainer.set_model_params(w_global)


            if self.args.same_time and self.args.setup == 'federated' \
                    and (((round_idx + 1) * self.args.epochs) % self.args.step_size == 0):
                self.args.lr *= 0.1

            if self.args.proto_queue is False and \
                    ((proto_locals and self.args.aggregate_proto and ((round_idx + 1) % self.args.aggr_proto_step == 0)) or
                        (round_idx >= self.args.aggr_proto_after_round)):
                logging.info(f"Aggregating local prototypes to produce global prototypes...")
                if self.args.aggregate_proto_by_class:
                    self.proto_global = self._aggregate_proto_by_class(proto_locals)
                else:
                    self.proto_global = self._aggregate_proto(proto_locals)
                self.radius_global = self._aggregate_radius(radius_locals)

            if self.args.test_every_n_rounds and round_idx % self.args.test_every_n_rounds == 0:
                accuracy_total = self.test_all_classes()
                wandb.log({f"Accuracy_all_classes": accuracy_total, "round": round_idx})
                accuracy_tasks = self.test_for_up_now_global_tasks()
                wandb.log({f"Mean_accuracy_task": accuracy_tasks, "round": round_idx})


            if self.args.test_clients_every_n_rounds and round_idx % self.args.test_clients_every_n_rounds == 0:
                accuracy_clients_tasks = self.test_clients_for_current_and_past_tasks()
                for client_id, client_acc in accuracy_clients_tasks.items():
                    for task_name, accuracy in client_acc.items():
                        wandb.log({f"client_{client_id}/Accuracy_{task_name}": accuracy, "round": round_idx})

            print(f"proto locals len: {len(proto_locals)} \t")

            if self.args.update_teacher_step == 0 or (round_idx % self.args.update_teacher_step == 0 and round_idx > 0):
                old_classes = copy.deepcopy(self.global_discovered_classes)
                if self.old_model:
                    old_model_params = copy.deepcopy(self.model_trainer.model.state_dict())
                    if self.args.update_teacher_ema < 1.:
                        old_old_model_params = copy.deepcopy(self.old_model.state_dict())
                        for k,v in old_old_model_params.items():
                            old_model_params[k] = self.args.update_teacher_ema * old_model_params[k] + (1-self.args.update_teacher_ema) * v
                    self.old_model.load_state_dict(old_model_params)
                else:
                    self.old_model = copy.deepcopy(self.model_trainer.model)
                self.old_model.eval()

            if round_idx % 100 == 0:
                self._after_train(round_idx)

        print(np.array(self.global_current_tasks_per_round))
        print("CLIENTS per round")
        print(selected_clients_per_round)
        print("end of training, saving the model")
        self._after_train(round_idx)

    def _compute_avg_radius(self, workers):
        training_num = 0
        avg_radius = 0
        for idx in workers:
            client = self.client_list[idx]
            training_num += client.get_sample_number()

        for idx in workers:
            client = self.client_list[idx]
            w = client.get_sample_number() / training_num
            avg_radius += w * client.get_radius()

        return avg_radius

    def _check_local_protos(self, client, len_protos, current_task):
        if not client.prototype["local"] or len(client.prototype["local"]) < len_protos:
            if isinstance(client.prototype["local"], dict):
                for c in self.task_classes[current_task]:
                    if c not in client.prototype["local"].keys():
                        client.prototype["local"][c] = np.zeros((1, self.feature_size))

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_indexes = [client_index for client_index in range(client_num_in_total)]
        if client_num_in_total != client_num_per_round:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            selectable_clients = copy.deepcopy(client_indexes)
            client_indexes = []
            client_idx = -1
            for _ in range(num_clients):
                cond = True
                while cond:
                    client_idx = np.random.choice(selectable_clients)
                    client = self.client_list[client_idx]
                    task_id = client.task_order[client.task_index]
                    if self.args.same_time:
                        task_id = self.global_task_id
                        if self.args.same_order is False:
                            task_id = client.task_order[task_id]
                    elif self.args.same_time is False and \
                            round_idx + 1 > np.cumsum(client.rounds_per_task)[client.task_index]:
                        task_id = client.task_order[client.task_index + 1]

                    if self.train_data_local_dict[client_idx][task_id] is None:
                        cond = True
                    else:
                        cond = self.train_data_local_num_dict[client_idx][task_id] == 0

                selectable_clients.remove(client_idx)
                client_indexes.append(client_idx)

        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate_w(self, w_locals, proto_local, round_idx):

        training_num = 0
        for idx in range(len(w_locals)):
            sample_num, _, _ = w_locals[idx]
            training_num += sample_num

        sample_num, averaged_params, _ = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params, _ = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def _aggregate_proto_by_class(self, proto_locals):
        global_classes = set()

        for client in proto_locals.keys():
            global_classes = set.union(global_classes, set(proto_locals[client]["prototype"].keys()))
        global_classes = list(global_classes)
        proto_global = {k: np.zeros(self.feature_size) for k in global_classes}

        weights_sums = {k: 0 for k in global_classes}

        for client in proto_locals.keys():
            local_proto = proto_locals[client]['prototype']
            for j in global_classes:
                if j in local_proto.keys() and not np.all(local_proto[j] == 0):
                    w = proto_locals[client]["num_samples_class"][j]
                    proto_global[j] += local_proto[j] * w
                    weights_sums[j] += w

        for j in global_classes:
            if 0 < weights_sums[j] < 1:
                proto_global[j] /= weights_sums[j]

        if self.proto_global is not None:
            for k in self.proto_global.keys():
                if k in proto_global.keys():
                    proto_global[k] = proto_global[k] * self.args.ema_global + self.proto_global[k] * (
                            1 - self.args.ema_global)

        return proto_global

    def _aggregate_proto(self, proto_locals):
        training_num = 0
        global_classes = set()
        for client in proto_locals.keys():
            sample_num = proto_locals[client]['sample_num']
            training_num += sample_num
            global_classes = set.union(global_classes, set(proto_locals[client]["prototype"].keys()))

        global_classes = list(global_classes)
        proto_global = {k: np.zeros(self.feature_size) for k in global_classes}

        weights_sums = {k: 0 for k in global_classes}

        for client in proto_locals.keys():
            local_sample_number = proto_locals[client]['sample_num']
            local_proto = proto_locals[client]['prototype']
            w = local_sample_number / training_num

            for j in global_classes:
                if j in local_proto.keys() and not np.all(local_proto[j] == 0):
                    proto_global[j] += local_proto[j] * w
                    weights_sums[j] += w

        for j in global_classes:
            if 0 < weights_sums[j] < 1:
                proto_global[j] /= weights_sums[j]

        if self.proto_global is not None:
            for k in self.proto_global.keys():
                if k in proto_global.keys():
                    proto_global[k] = proto_global[k] * self.args.ema_global + self.proto_global[k] * (
                            1 - self.args.ema_global)

        return proto_global

    def _aggregate_radius(self, radius_locals):
        radius_global = 0
        training_num = 0
        for client in radius_locals.keys():
            training_num += radius_locals[client]['sample_num']

        for client in radius_locals.keys():
            local_sample_number = radius_locals[client]['sample_num']
            local_radius = radius_locals[client]['radius']
            w = local_sample_number / training_num
            radius_global += local_radius * w

        if self.args.aggregate_mean_radius:
            return np.mean([radius_locals[c]["radius"] for c in radius_locals.keys()])
        else:
            return radius_global

    def _update_proto_radius_labels(self, client):
        client.radius["global"] = copy.deepcopy(self.radius_global)
        client.prototype["global"] = copy.deepcopy(self.proto_global)

    def _update_class_labels(self, client, current_task):
        client.class_label = list()
        for task in range(current_task + 1):
            client.class_label = self.task_classes[task] + client.class_label

    def _after_train(self, round):
        path = os.path.join('model_saved_check', self.file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, f"server_model_round_{round}.pkl")
        torch.save(self.model_trainer.model, filename)
        wandb.save(filename)


    def fetch_data(self, current_task, data_dict, data):
        data_dict[f'task {current_task}'] = data

    def save_protos(self, data, folder, path):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, path), 'wb') as file:
            pickle.dump(data, file)
            file.close()


    def test_clients_for_current_and_past_tasks(self):
        print("############# Test each client in this round on their classes and other classes #############")
        acc_per_client = {}

        for idx in self.current_client_indexes:
            client = self.client_list[idx]
            acc_per_client[idx] = {}
            client.model_trainer.model.to(self.device)
            client.model_trainer.model.eval()

            past_tasks = np.sort(list(set(self.global_discovered_tasks).difference({client.current_task})))

            task_group = {
                "current": client.current_task,
                "past": past_tasks
            }

            for key, task in task_group.items():
                if key == "current":
                    if task == 0:
                        classes = list(range(self.args.num_classes_first_task))
                    else:
                        classes = list(range(self.args.num_classes_first_task + (task - 1) * self.task_size,
                                         self.args.num_classes_first_task + task * self.task_size))
                elif key == "past":
                    for i in task:
                        classes = list(range(self.args.num_classes_first_task + (i - 1) * self.task_size,
                                             self.args.num_classes_first_task + i * self.task_size))

                test_loader = get_dataloader_test(self.args.data_dir, self.args.batch_size, classes)

                correct, total = 0.0, 0.0
                for batch_idx, (imgs, labels) in enumerate(test_loader):
                    labels = labels.long()
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        outputs = client.model_trainer.model(imgs)
                    outputs = outputs[:, ::4]

                    if self.args.mask_model:
                        # mask outputs with only the global discovered classes
                        outputs = outputs[:, self.global_discovered_classes]

                    predicts = torch.max(outputs, dim=1)[1].cpu()

                    if self.args.mask_model:
                        # map predictions in the "total" class scale compatible with the labels
                        predicts = torch.tensor([self.global_discovered_classes[p] for p in predicts])

                    correct += (predicts == labels.cpu()).sum()
                    total += len(labels)
                accuracy = correct.item() / total
                acc_per_client[idx][key] = accuracy
            client.model_trainer.model.train()
        print(acc_per_client)
        return acc_per_client


    def test_for_up_now_global_tasks(self):
        print("############# Test for each global discovered task #############")
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.eval()
        classes = []
        for i in self.global_discovered_tasks:
            if i == 0:
                classes = classes + list(range(self.args.num_classes_first_task))
            else:
                classes = classes + list(range(self.args.num_classes_first_task + (i - 1) * self.task_size,
                                     self.args.num_classes_first_task + i * self.task_size))

        test_loader = get_dataloader_test(self.args.data_dir, self.args.batch_size, classes)
        correct, total = 0.0, 0.0
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            labels = labels.long()
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model_trainer.model(imgs)
            outputs = outputs[:, ::4]

            if self.args.mask_model:
                # mask outputs with only the global discovered classes
                outputs = outputs[:, self.global_discovered_classes]

            predicts = torch.max(outputs, dim=1)[1].cpu()

            if self.args.mask_model:
                # map predictions in the "total" class scale compatible with the labels
                predicts = torch.tensor([self.global_discovered_classes[p] for p in predicts])

            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model_trainer.model.train()
        return accuracy


    def test_all_classes(self):
        print("############# Test on all classes #############")
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.eval()
        classes = np.arange(self.args.total_nc)
        test_loader = get_dataloader_test(self.args.data_dir, self.args.batch_size, classes)
        correct, total = 0.0, 0.0
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            labels = labels.long()
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model_trainer.model(imgs)
            outputs = outputs[:, ::4]

            if self.args.mask_model:
                # mask outputs with only the global discovered classes
                outputs = outputs[:, self.global_discovered_classes]

            predicts = torch.max(outputs, dim=1)[1].cpu()

            if self.args.mask_model:
                # map predictions in the "total" class scale compatible with the labels
                predicts = torch.tensor([self.global_discovered_classes[p] for p in predicts])

            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total

        self.model_trainer.model.train()
        return accuracy

