import logging
import math
import os
import re

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch

import utils


log = utils.get_logger(__name__)


def sub_instantiate(cfg):
    override = {}
    for k in cfg.keys():
        if isinstance(cfg[k], (dict, DictConfig)) and '_target_' in cfg[k]:
            override[k] = hydra.utils.instantiate(cfg[k])
    return override


def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    computed = {}
    computed['ngpu'] = utils.ngpu(cfg)
    computed['nodes'] = getattr(cfg.trainer, 'num_nodes', 1)

    # INITIALIZE: data
    # datamodule = hydra.utils.instantiate(cfg.data, **sub_instantiate(cfg.data))
    datamodule = hydra.utils.instantiate(cfg.data)
    train_loader = datamodule.train_dataloader()

    computed['train_batches'] = len(train_loader)
    computed['train_steps'] = computed['train_batches'] // ((computed['ngpu'] or 1) * computed['nodes'])
    computed['total_train_steps'] = cfg.trainer.max_epochs * computed['train_steps']

    # INITIALIZE: model
    utils.resolve_computed(cfg.model, computed)
    model = hydra.utils.instantiate(cfg.model)

    if 'model_weights' in cfg:  # possibly load pre-trained weights
        if cfg.model_weights == 'imagenet':
            state = hydra.utils.instantiate(cfg.model.model, pretrained=True, num_classes=1000).state_dict()
        else:
            state = torch.load(hydra.utils.to_absolute_path(cfg.model_weights), map_location='cpu')
            state = state.get('state_dict', state)
        to_restore = getattr(model.model, 'encoder', model.model)
        utils.restore_compatible_weights(to_restore, state)

    # INITIALIZE: logger
    if 'logger' in cfg:
        logger = []
        for _, log_conf in cfg['logger'].items():
            if '_target_' in log_conf:
                logger.append(hydra.utils.instantiate(log_conf))
    else:
        logger = True

    # INITIALIZE: callbacks
    callbacks = []
    if 'callback' in cfg:
        utils.resolve_computed(cfg.callback, computed)
        for _, cb_conf in cfg.callback.items():
            if '_target_' in cb_conf:
                log.info(f'Instantiating callback <{cb_conf._target_}>')
                callbacks.append(hydra.utils.instantiate(cb_conf))
    if not getattr(cfg.trainer, 'fast_dev_run', False):
        callbacks.append(utils.SaveConfig(cfg))

    if cfg.trainer.accelerator == 'ddp':
        plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    else:
        plugins = None

    # INITIALIZE: trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks, plugins=plugins)

    # log hyperparameters
    utils.log_hyperparameters(
        cfg, model=model, datamodule=datamodule, trainer=trainer, callbacks=callbacks, logger=logger
    )

    # Train
    log.info('BEGIN TRAINING')
    trainer.fit(model, datamodule=datamodule)

    # Shutdown
    utils.finish(
        cfg, model=model, datamodule=datamodule, trainer=trainer, callbacks=callbacks, logger=logger
    )
