import os
import cv2
import numpy as np
import csv
import hydra
from omegaconf import DictConfig
from src.utils.common import get_config_path
from pathlib import Path
from tqdm import tqdm


def create_csv(cfg: DictConfig):
    os.makedirs(cfg.csv_path, exist_ok=True)

    rs = np.random.RandomState(42)
    imgs = [file for file in os.listdir(cfg.processed_words_path)]
    imgs.sort()
    rs.shuffle(imgs)

    length = len(imgs)
    sp1 = int((1 - cfg.test_set - cfg.validation_set) * length)
    sp2 = int((1 - cfg.test_set) * length)
    img_paths = {'train': imgs[:sp1], 'dev': imgs[sp1:sp2], 'test': imgs[sp2:]}

    datadir = Path(cfg.csv_path)
    processed_words_path = Path(cfg.processed_words_path)
    print('Converting word images to CSV...')

    for split in ['train', 'dev', 'test']:
        if split == "train":
            continue
        labels = np.array([name for name in img_paths[split]])
        length = len(img_paths[split])
        images = np.empty(length, dtype=object)

        for i, img in tqdm(enumerate(img_paths[split]), total=length, desc=f"{split}"):
            images[i] = (cv2.imread(str(processed_words_path / img), 0), "None")

        with open(datadir / f'{split}.csv', 'w') as csvfile:
            fieldnames = ['label', 'shape', 'image', 'gaplines']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in tqdm(range(length)):
                if images[i][0] is None:
                    continue
                writer.writerow({
                    fieldnames[0]: labels[i],
                    fieldnames[1]: str(images[i][0].shape)[1:-1],
                    fieldnames[2]: str(list(images[i][0].flatten()))[1:-1],
                    fieldnames[3]: images[i][1]
                })

    print('\tCSV files created!')


if __name__ == '__main__':
    hydra.main(
        config_path=str(get_config_path()),
        config_name="config",
        version_base="1.2"
    )(create_csv)()
