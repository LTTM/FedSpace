import copy
import logging
import numpy as np
import torch
import math
import colorama
from fractal_learning.training import datamodule


class Client:

    def __init__(self, client_idx, local_training_data, local_sample_number, args, device,
                 model_trainer, feature_size):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.feature_size = feature_size

        self.prototype = {"global": {}, "local": {}}
        self.radius = {"global": 0, "local": 0}
        self.class_label = list()

        self.task_index = 0
        self.current_task = None
        self.current_classes = None
        self.round_counter = 0

        if args.same_order:
            self.task_order = np.arange(args.task_num + 1)
        else:
            self.task_order = np.random.permutation(np.arange(args.task_num + 1))

        if self.args.repeat_tasks_n_times:
            self.task_order = np.array([k for _ in range(self.args.repeat_tasks_n_times) for k in self.task_order])

        self.rounds_per_task = []
        self.partition_round_per_task()

        self.num_sample_class = None

        if args.fractal_pretrain_rounds:
            self.pretrain_loader = datamodule.FractalClassDataModule().train_dataloader()

    def partition_round_per_task(self, v=0.1):
        if self.args.more_asynch:
            v = 0.3

        mean = round(self.args.comm_rounds / (self.args.task_num + 1))

        if self.args.force_rounds_per_task != -1:
            self.rounds_per_task = [self.args.force_rounds_per_task for _ in range(self.args.task_num + 1)]
        elif self.args.same_time:
            self.rounds_per_task = [mean] * self.args.task_num + [self.args.comm_rounds - mean * self.args.task_num]
        else:
            var = round(mean * v)

            partition = [round(np.random.uniform(mean - var, mean + var))]
            for i in range(self.args.task_num - 1):
                partition.append(round(np.random.uniform(mean - var, mean + var)))

            partition.append(self.args.comm_rounds - np.sum(partition))
            if self.args.more_asynch:
                self.rounds_per_task = np.random.permutation(partition).tolist()
            else:
                self.rounds_per_task = partition

        if self.args.repeat_tasks_n_times:
            self.rounds_per_task = np.array([k for _ in range(self.args.repeat_tasks_n_times) for k in self.rounds_per_task])


    def update_classes(self, task_classes):
        self.class_label = list()
        for task in range(self.task_index + 1):
            self.class_label = task_classes[self.task_order[task]] + self.class_label
        self.class_label = np.sort(self.class_label)

    def update_local_dataset(self, r, local_training_data, local_sample_number, global_task=None):

        update_task = False
        if self.args.same_time:
            self.task_index = global_task

        if self.args.same_order and self.args.same_time and (self.current_task != global_task):
            self.current_task = global_task
            update_task = True
        else:
            self.current_task = self.task_order[self.task_index]

            if self.args.same_time is False and r + 1 > np.cumsum(self.rounds_per_task)[self.task_index]:
                self.task_index += 1
                self.current_task = self.task_order[self.task_index]
                update_task = True

        message = f"Client {self.client_idx} task: " + " ".join(
            [f'{colorama.Fore.RED}{el}{colorama.Style.RESET_ALL}'
             if el == self.current_task else str(el) for el in self.task_order])

        self.local_training_data = local_training_data[self.current_task]
        if self.local_training_data is None:
            self.local_sample_number = 0
        else:
            self.local_sample_number = local_sample_number[self.current_task]

        print(f"{message}, \t sample number: {self.local_sample_number}, \t rounds per task:{self.rounds_per_task}")
        return update_task

    def get_sample_number(self):
        return self.local_sample_number

    def fractal_pretrain(self, w_global):
        self.model_trainer.set_model_params(w_global)
        loss = self.model_trainer.fractal_pretrain(self, self.args)
        weights = self.model_trainer.get_model_params()
        return weights, loss


    def train(self, w_global, old_classes, old_model, mask=None, round=None, proto_queue=None):
        self.model_trainer.set_model_params(w_global)
        if self.local_sample_number == 0 or len(self.local_training_data) == 0:
            print("There is no training data for task {} of client {}! TRAINING ABORTED".format(self.current_task, self.client_idx))
            return None
        loss, num_sample_class = self.model_trainer.train(self, self.args, old_model, old_classes, self.current_task, mask, round, proto_queue)
        weights = self.model_trainer.get_model_params()
        return weights, loss, num_sample_class

    def proto_save(self, current_task_classes):
        features = []
        labels = []
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.eval()
        with torch.no_grad():
            for batch_idx, (images, target) in enumerate(self.local_training_data):
                feature = self.model_trainer.model.feature(images.to(self.device))
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())

        labels = np.concatenate([label_vector for label_vector in labels])
        features = np.concatenate([feature_vector for feature_vector in features], axis=0)
        feature_dim = features.shape[1]

        prototype = {}
        radius = {}
        class_label = []
        for item in current_task_classes:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            if np.size(feature_classwise) == 0:
                prototype[item] = np.zeros(self.feature_size,)
            else:
                prototype[item] = np.mean(feature_classwise, axis=0)

            if not self.prototype["local"] or (self.args.proto_queue and self.args.multi_radius):
                cov = np.cov(feature_classwise.T)
                if not math.isnan(np.trace(cov)):
                    radius[item] = np.trace(cov) / feature_dim
                else:
                    radius[item] = 0

        if self.radius["local"] and ((self.args.proto_queue and self.args.multi_radius) is False):
            radius = copy.deepcopy(self.radius["local"])
        else:
            radius = np.sqrt(np.mean(list(radius.values())))

        self.model_trainer.model.train()
        return radius, prototype, class_label

    def get_proto(self):
        return self.prototype

    def get_radius(self):
        return self.radius

    def get_class_label(self):
        return self.class_label
