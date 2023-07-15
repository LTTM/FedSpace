import logging
import math
import os
import re
from typing import List

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch

######################
### Python logging ###
def get_logger(name=__name__, level=logging.INFO):
    """Initializes python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, pl.utilities.rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)
######################

class SaveConfig(pl.callbacks.Callback):
    '''Saves the config file for the run into the log directory'''
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = trainer.logger
        for lg in logger:
            dir = None
            if hasattr(lg, 'log_dir'): dir = lg.log_dir
            elif hasattr(lg._experiment, 'dir'): dir = lg._experiment.dir
            if dir:
                with open(f'{dir}/hydra-config.yaml', 'w') as f:
                    f.write(OmegaConf.to_yaml(self.cfg))


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __resolve_ckpt_dir(self, trainer):
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:
        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path
        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        # Todo: required argument `pl_module` is not used
        if self.dirpath is not None:
            return  # short circuit

        # breakpoint()
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                # modified to behave like I want it to with wandb
                if isinstance(trainer.logger, pl.loggers.base.LoggerCollection):
                    logger = trainer.logger[0]
                else:
                    logger = trainer.logger
                if isinstance(logger, pl.loggers.WandbLogger):
                    save_dir = logger.save_dir
                    name = logger.version
                    ckpt_path = os.path.join(save_dir, 'checkpoints')
                    if name:
                        ckpt_path = os.path.join(ckpt_path, name)
                else:
                    save_dir = trainer.logger.save_dir or trainer.default_root_dir

                    version = (
                        trainer.logger.version
                        if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
                    )
                    ckpt_path = os.path.join(save_dir, str(trainer.logger.name), version, "checkpoints")
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        ckpt_path = trainer.training_type_plugin.broadcast(ckpt_path)

        self.dirpath = ckpt_path

        if not trainer.fast_dev_run and trainer.is_global_zero:
            self._fs.makedirs(self.dirpath, exist_ok=True)


def empty(*args, **kwargs):
    pass


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    '''Makes sure everything closed properly.'''

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.WandbLogger):
            import wandb
            wandb.finish()


def is_rank_zero():
    '''Return True if this is the rank_zero process'''
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    return (local_rank == 0 and node_rank == 0)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    '''This method controls which parameters from Hydra config are saved by Lightning loggers.
    '''

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams['trainer'] = config['trainer']
    hparams['model'] = config['model']
    hparams['data'] = config['data']
    if 'optimizer' in config:
        hparams['optimizer'] = config['optimizer']
    if 'callbacks' in config:
        hparams['callbacks'] = config['callbacks']
    if 'model_weights' in config:
        hparams['model_weights'] = config['model_weights']

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty


def ngpu(cfg):
    '''Return the number of GPUs that are being used'''
    if cfg.trainer.gpus is None: return 0
    p = pl.utilities.device_parser
    gpus = cfg.trainer.gpus
    return len(p._normalize_parse_gpu_input_to_list(p._normalize_parse_gpu_string_input(gpus)))


def resolve_computed(cfg, computed):
    '''This is an added utility patched into the Hydra system, to allow for computing
    values and updating the config at runtime with values that can't be determined
    before parts of the code run. The special syntax "$$computed[val]" is used in the
    configs to indicate a value that needs to be filled in.
    '''
    for k, v in cfg.items():
        if hasattr(v, 'items'):
            resolve_computed(v, computed)
        elif isinstance(v, str) and v.strip().startswith('$$computed'):
            cv = re.search(r'\[(.+)\]', v).groups()[0]
            cfg[k] = computed[cv]


def restore_compatible_weights(model, state):
    '''Restores weights from a saved checkpoint. Any layers that have aren't compatible,
    such as classifier layers that have been adjusted for a new set of classes, are not restored,
    but left randomly initialized.
    '''
    net = model
    net_state = net.state_dict()
    new_state = {}
    incompatible = []
    for k in state:
        kk = k[6:] if k.startswith('model.') else k
        if kk in net_state:
            if net_state[kk].shape == state[k].shape:
                new_state[kk] = state[k]
            else:
                incompatible.append(kk)
    missing, extra = net.load_state_dict(new_state, strict=False)
    missing = [x for x in missing if x not in set(incompatible)]

    log.info('Model weights loaded:')
    log.info(f'  Missing keys: {", ".join(missing)}')
    log.info(f'  Extra keys: {", ".join(extra)}')
    log.info(f'  Incompatible sizes: {", ".join(incompatible)}')


class CrossEntropy(torch.nn.Module):
    '''Cross entropy loss function accepting arbitrary distribution for target.
    Useful for, e.g., label smoothing cross entropy. Predictions should be logits,
    targets should be a valid probability distribution. Also works for normal class-
    label targets.
    '''
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def _soft_label_loss(self, pred, label):
        pred = pred.log_softmax(dim=1)
        loss = label.mul(pred).sum(1).neg()
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
    def _hard_label_loss(self, pred, label):
        return torch.nn.functional.cross_entropy(pred, label, reduction=self.reduction)

    def forward(self, pred, label):
        if label.ndim == pred.ndim:
            return self._soft_label_loss(pred, label)
        else:
            return self._hard_label_loss(pred, label)
