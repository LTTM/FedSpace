import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    import train

    if train.utils.is_rank_zero():
        print(OmegaConf.to_yaml(cfg))

    train.train(cfg)


if __name__ == '__main__':
    main()