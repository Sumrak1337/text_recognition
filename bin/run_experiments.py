import hydra
from omegaconf import DictConfig

from src.utils.common import get_config_path


def main(cfg: DictConfig):
    ...


if __name__ == '__main__':
    hydra.main(
        config_path=str(get_config_path()),
        config_name='config',
        version_base='1.2'
    )(main)()


