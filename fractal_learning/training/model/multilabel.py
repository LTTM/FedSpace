from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
from torchmetrics import Accuracy, F1

from .base import _BaseModule


class MultiLabelModel(_BaseModule):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
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
            lr=lr,
            weight_decay=weight_decay,
            warmup=warmup,
            training_steps=training_steps,
            loss_fn=loss_fn,
            optim_name=optim_name,
            optim_kwargs=optim_kwargs,
        )

    def setup_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn or torch.nn.functional.binary_cross_entropy_with_logits

    def setup_metrics(self):
        self.train_accuracy = Accuracy()
        self.train_f1 = F1()
        self.val_accuracy = Accuracy()
        self.val_f1 = F1()
        self.metric_hist = {
            "train/acc": [],
            "train/loss": [],
            "train/f1": [],
            "val/acc": [],
            "val/loss": [],
            "val/f1": [],
        }

    def step(self, batch: Any):
        x, yy = batch
        logits = self.forward(x)
        mask = yy[:, 0].eq(-1)
        yy[mask] = 0
        y = torch.zeros(logits.shape, dtype=x.dtype, device=x.device).scatter_(1, yy, 1)
        y[mask, 0] = 0
        # w = torch.empty(logits.shape[1], dtype=x.dtype, device=x.device).fill_(logits.shape[1] / 5)
        w = y.count_nonzero(dim=1)
        w[w == 0] = 1
        w = (logits.shape[1] / w)[:, None].expand(*logits.shape)
        loss = self.loss_fn(logits, y, pos_weight=w)
        with torch.no_grad():
            preds = logits.sigmoid()
        return loss, preds, y.int()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/f1", f1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        f1 = self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", f1, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

