import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.common import get_config_path


def main(cfg: DictConfig):
    loader = instantiate(cfg.data_loader)
    loader.load()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()), config_name="config", version_base="1.2"
    )(main)()
