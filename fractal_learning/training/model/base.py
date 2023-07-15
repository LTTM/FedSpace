from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


class _BaseModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        warmup: float = 0.1,
        training_steps: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        optim_name: str = 'AdamW',
        optim_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()

        # add to self.hparams and save to ckpt
        self.save_hyperparameters(ignore='model loss_fn'.split())
        if isinstance(model, torch.nn.Module):
            self.hparams.model = model.__class__.__name__
        if hasattr(loss_fn, '__call__'):
            self.hparams.loss_fn = getattr(loss_fn, '__name__', loss_fn.__class__.__name__)

        self.model = model
        self.setup_loss_fn(loss_fn)
        self.setup_metrics()

    def setup_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    def setup_metrics(self):
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.metric_hist = {
            "train/acc": [],
            "train/loss": [],
            "val/acc": [],
            "val/loss": [],
        }

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def epoch_end_log(self, prefix='train'):
        for metric in self.metric_hist:
            if metric.startswith(prefix):
                self.metric_hist[metric].append(self.trainer.callback_metrics[metric])
                decide = min if 'loss' in metric else max
                self.log(metric+'_best', decide(self.metric_hist[metric]), prog_bar=False)

    def training_epoch_end(self, outputs: List[Any]):
        self.epoch_end_log('train')

    def validation_epoch_end(self, outputs: List[Any]):
        self.epoch_end_log('val')

    def _parameter_groups(self):
        hp = self.hparams
        lr = hp.lr

        params = []

        # handle manually specified param groups
        pgs = hp.get('param_groups', {})
        names = set()
        for pg in pgs:
            name = pg['name']
            names.add(name)
            parts = name.split('.')
            param = self.model
            for p in parts:
                param = getattr(param, p, getattr(self, p))
            if isinstance(param, torch.Tensor):
                param = [param]
            else:
                param = param.parameters()
            group = {
                'params': param,
                'lr': lr * pg.get('lr_factor', 1),
                'weight_decay': pg.get('weight_decay', hp.weight_decay)
            }
            group.update({k: v for k, v in pg.items() if k not in ('lr_factor, weight_decay')})
            params.append(group)

        # create parameter groups for not decaying bias and normalizaton parameters
        decay, no_decay = [], []
        for mname, m in self.model.named_modules():
            if mname in names: continue  # already handled above
            if 'Norm' in m.__class__.__name__:  # normalization layers, such as BatchNorm2d
                no_decay.extend(list(m.parameters()))
            else:
                for name, param in m.named_parameters(recurse=False):
                    if f'{mname}.{name}' in names: continue  # already handled above
                    if 'bias' in name: no_decay.append(param)  # bias parameters
                    else: decay.append(param)

        params.extend([
            {'params': decay, 'weight_decay': hp.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ])

        return params

    def configure_optimizers(self):
        hp = self.hparams
        lr = hp.lr

        params = self._parameter_groups()
        
        # create optimizer
        optim = getattr(torch.optim, hp.optim_name, 'AdamW')(
            params=params, lr=lr, weight_decay=hp.weight_decay, **(hp.optim_kwargs or {})
        )

        # optionally add a OneCycle scheduler
        if self.hparams.training_steps is not None:
            lrs = [pg.get('lr', lr) for pg in params]
            steps = hp.training_steps
            warmup = hp.warmup if hp.warmup < 1 else hp.warmup / steps
            final_div = hp.get('final_div_factor', 1e4)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=lrs, total_steps=steps, pct_start=warmup, final_div_factor=final_div)
            return [optim], [{'scheduler': sched, 'interval': 'step', 'name': 'lr'}]

        return optim


class ClassificationModel(_BaseModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        warmup: float = 0.1,
        training_steps: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        optim_name: str = 'AdamW',
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
            optim_kwargs=optim_kwargs
        )

    def step(self, batch: Any):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

