import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from bin.create_csv import create_csv
from bin.light_run import get_image, make_prediction
from bin.train_ctc import train_ctc
from src.utils.common import get_config_path


def hard_run(cfg: DictConfig):
    # loading data
    loader = instantiate(cfg.data_loader)
    loader.load()
    create_csv(cfg=cfg)

    # train model
    train_ctc(cfg=cfg)

    # getting image
    get_image(cfg=cfg)

    # prediction
    make_prediction(cfg=cfg)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()), config_name="config", version_base="1.2"
    )(hard_run)()
