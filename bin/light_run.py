import os
import tkinter as tk
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.ocr import page, words
from src.ocr.datahelpers import idx2char
from src.ocr.normalization import word_normalization
from src.ocr.tfhelpers import Model
from src.utils.common import get_config_path


def main(cfg: DictConfig):
    get_image(cfg=cfg)

    model_path = Path(cfg.models_output)
    model_loc_ctc = model_path / cfg.model_params.model_name

    ctc_model = Model(model_loc_ctc, operation="word_prediction")

    img = Path(cfg.save_image_dir) / cfg.image_name
    image = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)

    # crop
    crop = page.detection(image)
    boxes = words.detection(crop)
    lines = words.sort_words(boxes)

    s = ""
    for line in lines:
        s = " ".join(
            [
                ctc_recognition(crop[y1:y2, x1:x2], ctc_model=ctc_model)
                for (x1, y1, x2, y2) in line
            ]
        )
    with open(Path(cfg.save_image_dir) / "prediction_text.txt", "w") as file:
        file.write(s)


def ctc_recognition(img, ctc_model):
    """Recognising words using CTC Model."""
    img = word_normalization(img, 64, border=False, tilt=False, hyst_norm=False)
    length = img.shape[1]
    # Input has shape [batch_size, height, width, 1]
    input_imgs = np.zeros((1, 64, length, 1), dtype=np.uint8)
    input_imgs[0][:, :length, 0] = img

    pred = ctc_model.eval_feed(
        {"inputs:0": input_imgs, "inputs_length:0": [length], "keep_prob:0": 1}
    )[0]

    word = ""
    for i in pred:
        word += idx2char(i + 1)
    return word


def get_image(cfg):
    os.makedirs(cfg.save_image_dir, exist_ok=True)
    application = tk.Tk()
    application.title("Painter")
    instantiate(cfg.painter, application=application)
    application.mainloop()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()), config_name="config", version_base="1.2"
    )(main)()
