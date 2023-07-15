'''Based off of https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
'''
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .base import _BaseModule


class SimSiam(_BaseModule):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dim: int = 2048,
        pred_dim: int = 512,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        warmup: float = 0.1,
        training_steps: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        optim_name: str = 'SGD',
        optim_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            dim=dim,
            pred_dim=pred_dim,
            lr=lr,
            weight_decay=weight_decay,
            warmup=warmup,
            training_steps=training_steps,
            loss_fn=loss_fn,
            optim_name=optim_name,
            optim_kwargs=optim_kwargs,
            **kwargs
        )

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.model = model

        # build a 3-layer projector
        prev_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), # second layer
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), # hidden layer
            nn.Linear(pred_dim, dim)) # output layer

    def setup_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn or torch.nn.CosineSimilarity(dim=1)

    def setup_metrics(self):
        self.metric_hist = {
            "train/loss": [],
        }

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.model(x1) # NxC
        z2 = self.model(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        p1, p2, z1, z2 = self(x1, x2)
        loss = -0.5 * (self.loss_fn(p1, z2).mean() + self.loss_fn(p2, z1).mean())

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss}

    def on_after_backward(self):
        for pg in self.optimizers()._optimizer.param_groups:
            if pg.get('fix_lr', False):
                pg['lr'] = self.hparams.lr
