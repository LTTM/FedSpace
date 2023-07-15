import logging
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch import nn
from tqdm import tqdm

try:
    from .model_trainer import ModelTrainer
except ImportError:
    from .model_trainer import ModelTrainer

LOSS_KEYS = ["Proto_aug_loss"]


class MyModelTrainer(ModelTrainer):
    def update_output_dim(self, class_num):
        weight = self.model.fc.weight.data
        bias = self.model.fc.bias.data
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features

        self.model.fc = nn.Linear(in_features, class_num * 4)
        self.model.fc.weight.data[:out_features] = weight[:out_features]
        self.model.fc.bias.data[:out_features] = bias[:out_features]

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def fractal_pretrain(self, client, args, sec_device=None):
        if client:
            device = client.device
            pretrain_loader = client.pretrain_loader
        else:
            device = sec_device
            pretrain_loader = self.pretrain_loader
        model = self.model
        model.to(device)
        model.train()
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=0.9, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd)
        for epoch in range(args.epochs):
            for batch_idx, (images, labels) in enumerate(tqdm(pretrain_loader)):
                # centralized pretraining will run till the end of the dataloader
                if (batch_idx == args.federated_fractal_pretrain_steps and client) or \
                        (batch_idx == args.centralized_fractal_pretrain_steps and client is None):
                    break
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = nn.CrossEntropyLoss()(output / args.temp, labels)
                loss.backward()
        return loss


    def train(self, client, args, old_model, old_classes, task, mask, round=None, proto_queue=None):

        model = self.model

        model.to(client.device)
        model.train()

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=0.9, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd)

        # Learning rate schedule on PASS paper
        if args.setup == 'centralized':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

        epoch_loss = []

        loss_terms = epoch_loss_terms = {name: [] for name in LOSS_KEYS}

        num_sample_class = {k: 0 for k in range(100)}

        for epoch in range(args.epochs):
            batch_loss = []
            batch_loss_terms = {name: [] for name in LOSS_KEYS}
            for batch_idx, (images, labels) in enumerate(client.local_training_data):

                labels = labels.long()

                for lab in labels.tolist():
                    num_sample_class[lab] += 1

                images, labels = images.to(client.device), labels.to(client.device)

                # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)

                optimizer.zero_grad()
                loss = self._compute_loss(images, labels, client, args, old_model, old_classes, mask, round, proto_queue)
                optimizer.zero_grad()
                loss["Total_loss"].backward()

                optimizer.step()
                batch_loss.append(loss["Total_loss"].item())
                
                if batch_idx % 100 == 0:
                    wandb.log({f"task{task}/{k}": v for k, v in loss.items()})

                for key in LOSS_KEYS:
                    batch_loss_terms[key].append(loss[key].item())

            if args.setup == 'centralized':
                scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            for key in LOSS_KEYS:
                epoch_loss_terms[key].append(np.mean(batch_loss_terms[key]))

        for key in LOSS_KEYS:
            loss_terms[key].append(np.mean(epoch_loss_terms[key]))

        return loss_terms, num_sample_class

    def _compute_loss(self, images, labels, client, args, old_model, old_class, mask=None, round=None, proto_queue=None):
        feature = self.model.feature(images)
        output = self.model.fc(feature)
        if mask:
            labels = torch.stack([torch.tensor(mask.index(lab)) for lab in labels])
            output = output[:, mask]

        output, labels = output.to(client.device), labels.long().to(client.device)
        loss_cls = nn.CrossEntropyLoss()(output / args.temp, labels)
        loss_dict = {"CE_loss": loss_cls,
                     "Proto_aug_loss": torch.zeros(1).to(client.device),
                     "Repr_learn_loss": torch.zeros(1).to(client.device),
                     "Total_loss": 0}


        proto_aug = []
        proto_aug_label = []
        location = args.location_proto_aug

        # if local protos are aggregated after N rounds, during the first N-1 rounds use local proto
        if args.location_proto_aug == "global" and not client.prototype["global"]:
            location = "local"

        prototype = client.prototype[location]
        radius = client.radius[location]

        if args.proto_queue and args.mean_proto_queue:
            prototype = proto_queue.compute_mean()

        index = [k for k, v in prototype.items() if np.sum(v) != 0 and (k not in client.current_classes or args.proto_loss_curr_classes)]
        if index and args.lambda_proto_aug and (prototype is not None) and radius:
            # select only the` indexes with old (non-empty) prototypes
            for _ in range(args.batch_size):
                np.random.shuffle(index)
                if args.proto_queue and not args.mean_proto_queue:
                    choice = np.random.choice(len(proto_queue.queue[index[0]]))
                    p, r, num_samples_class = proto_queue.queue[index[0]][choice]
                    temp = p + np.random.normal(0, 1, client.feature_size) * r
                else:
                    temp = prototype[index[0]] + np.random.normal(0, 1, client.feature_size) * radius
                proto_aug.append(temp)
                proto_aug_label.append(4 * index[0])
            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(client.device)
            if len(proto_aug.shape) > 2:
                proto_aug = proto_aug.squeeze()
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).long().to(client.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_dict["Proto_aug_loss"] = nn.CrossEntropyLoss()(soft_feat_aug / args.temp, proto_aug_label)

        # REPR LEARNING LOSS
        index = [k for k, v in prototype.items() if np.sum(v) != 0]
        proto_aug_feat = []
        proto_aug_lab = []
        if index and args.lambda_repr_loss > 0. and (prototype is not None) and radius:
            # select only the` indexes with old (non-empty) prototypes
            for _ in range(args.batch_size):
                np.random.shuffle(index)
                proto_aug_feat.append(prototype[index[0]] + np.random.normal(0, 1, client.feature_size) * radius)
                proto_aug_lab.append(index[0])
            proto_aug_feat = torch.from_numpy(np.float32(np.asarray(proto_aug_feat))).float().to(client.device)
            if len(proto_aug_feat.shape) > 2:
                proto_aug_feat = proto_aug_feat.squeeze()
            proto_aug_lab = torch.from_numpy(np.asarray(proto_aug_lab)).long().to(client.device)

            wo_aug = args.repr_loss_wo_aug
            if wo_aug:
                slc = slice(0, -1, 4)  # W/O AUG
            else:
                slc = slice(None)  # W/ AUG
            curr_cls_feat = feature[slc]
            curr_cls_lab = torch.tensor([mask[i]/4 for i in labels][slc]).int().to(client.device)
            feat, lab = torch.cat([curr_cls_feat, proto_aug_feat], dim=0), torch.cat([curr_cls_lab, proto_aug_lab], dim=0)      # N x D, N
            feat = F.normalize(feat, p=2., dim=1)
            loss_tot = 0.
            for c in client.current_classes:
                Nc = (lab==c).sum()
                if Nc<=1: continue
                feat_c = feat[lab==c]                                                       # Nc x D
                feat_not_c = feat[lab!=c]                                                   # Nnc x D
                pos = feat_c @ feat_c.T / args.repr_loss_temp                                                  # Nc x Nc
                pos[torch.eye(Nc).bool()] *= 0.
                neg = feat_c @ feat_not_c.T / args.repr_loss_temp                                              # Nc x Nnc
                loss = pos - torch.logsumexp(torch.cat([pos,neg], dim=1), dim=1).unsqueeze(1)   # Nc x Nc
                loss_tot += -1 * loss[~torch.eye(Nc).bool()].sum() / (Nc-1)
            loss_dict["Repr_learn_loss"] = loss_tot / lab.size(0)

        if old_model is None and self.args.client_num_in_total == 1 and self.args.proto_loss_after_first_task:
            loss_dict["Proto_aug_loss"] = torch.zeros(1).to(client.device)

        loss_dict["Total_loss"] = loss_dict["CE_loss"] + \
            loss_dict["Proto_aug_loss"] * args.lambda_proto_aug + \
            loss_dict["Repr_learn_loss"] * args.lambda_repr_loss

        return loss_dict
