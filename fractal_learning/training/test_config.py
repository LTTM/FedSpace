import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
