from collections.abc import Callable
import random
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Metric
import wandb

from .base import _BaseModule


class GlaSSegmentationModel(_BaseModule):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.1,
        weight_decay: float = 0.0001,
        warmup: float = 0.1,
        training_steps: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        optim_name: str = 'SGD',
        optim_kwargs: Optional[Dict] = None,
        mixup: bool = True,
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
            mixup=mixup,
        )

    def setup_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn if loss_fn is not None else SoftDiceLoss()

    def setup_metrics(self):
        self.train_iou = MeanIoU(1)
        self.val_iou = MeanIoU(1)
        self.metric_hist = {
            "train/iou": [],
            "train/loss": [],
            "val/iou": [],
            "val/loss": [],
        }

    def forward(self, x: torch.Tensor):
        return self.model(x).sigmoid()

    def step(self, batch: Any):
        x, y = batch
        probs = self.forward(x)
        loss = self.loss_fn(probs, y)
        return loss, probs, y

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y = y.gt(0).byte()
        if self.hparams.mixup:
            x, y = mixup(x, y)
        loss, probs, targets = self.step([x, y])

        # log train metrics
        iou = self.train_iou(probs, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/iou", iou, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "probs": probs, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y = y.gt(0).byte()
        h, w = x.shape[-2:]
        x = torch.cat([x[:,:,:448,:448], x[:,:,h-448:,:448], x[:,:,:448,w-448:], x[:,:,h-448:,w-448:]])
        y = torch.cat([y[:,:,:448,:448], y[:,:,h-448:,:448], y[:,:,:448,w-448:], y[:,:,h-448:,w-448:]])
        loss, probs, targets = self.step([x, y])

        # log val metrics
        iou = self.val_iou(probs, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        # log example segmentations
        if batch_idx == 0:
            self.log_masks(x[::4], y[::4], probs[::4])

        return {"loss": loss, "probs": probs, "targets": targets}

    @rank_zero_only
    def log_masks(self, x, y, probs):
        preds = probs.ge(0.5)
        masked = [
            wandb.Image(
                xx.permute(1,2,0).cpu().numpy(),
                masks={
                'predicted': {'mask_data': p.squeeze().cpu().numpy()},
                'truth': {'mask_data': yy.squeeze().cpu().numpy()}
            })
            for xx, yy, p in zip(x, y, preds)
        ]
        self.trainer.logger.experiment[0].log({
            'val/examples': masked, 'global_step': self.trainer.global_step
        })


class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # breakpoint()
        b = pred.shape[0]
        pred = pred.view(b, -1)
        target = target.view(b, -1)
        intersect = pred.mul(target).sum(1)
        p_sum = pred.sum(1)
        t_sum = target.sum(1)
        dice = intersect.mul(2).div(p_sum + t_sum + 1e-6).mean()
        return 1 - dice


class MeanIoU(Metric):
    def __init__(
        self,
        num_classes: int = 1,
        normalize: Optional[str] = None,
        threshold: float = 0.5,
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.normalize = normalize
        self.threshold = threshold
        self.multilabel = multilabel

        allowed_normalize = ('true', 'pred', 'all', 'none', None)
        if self.normalize not in allowed_normalize:
            raise ValueError(f"Argument average needs to one of the following: {allowed_normalize}")

        default = torch.zeros(num_classes)
        self.add_state("iou", default=default, dist_reduce_fx="sum")
        self.add_state("total", default=default, dist_reduce_fx="sum")

    def update(self, preds, target):
        if preds.dtype == torch.float32:
            preds = preds.ge(self.threshold).byte()
        if target.dtype == torch.float32:
            target = target.ge(self.threshold).byte()
        dim = 2 if self.multilabel else 1
        intersection = preds.mul(target).flatten(dim).sum(-1)
        union = preds.logical_or(target).flatten(dim).sum(-1)
        mask = union.eq(0)
        iou = intersection.float() / union
        iou[mask] = 0
        mask = mask.logical_not()
        iou = iou.sum(0)
        total = mask.sum(0)
        self.iou += iou
        self.total += total

    def compute(self):
        return self.iou.div(self.total).mean()

    @property
    def is_differentiable(self):
        return False


def mixup(imgs, masks, alpha=0.5):
    a = random.betavariate(alpha, alpha)
    idx = torch.randperm(imgs.shape[0], device=imgs.device)
    imgs_mix = a * imgs + (1 - a) * imgs[idx]
    masks_mix = a * masks + (1 - a) * masks[idx]
    return imgs_mix, masks_mix