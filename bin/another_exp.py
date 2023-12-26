import os
import tkinter as tk
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate

from src.ocr.normalization import word_normalization, letter_normalization
from src.ocr import page, words, characters
from src.ocr.tfhelpers import Model
from src.ocr.datahelpers import idx2char

from src.utils.common import get_config_path


def main(cfg: DictConfig):
    get_image(cfg=cfg)
    model_path = Path(cfg.models_output)
    model_loc_chars = model_path / 'char-clas' / 'en' / 'CharClassifier'
    model_loc_ctc = model_path / 'word-clas' / 'CTC' / 'Classifier1'

    character_model = Model(model_loc_chars)
    ctc_model = Model(model_loc_ctc, operation='word_prediction')

    img = Path(cfg.save_image_dir) / cfg.image_name
    image = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)

    # crop
    crop = page.detection(image)
    boxes = words.detection(crop)
    lines = words.sort_words(boxes)
    # TODO: add comparing from dictionary
    # implt(crop)
    # for line in lines:
    #     print(" ".join([char_recognition(crop[y1:y2, x1:x2],
    #                                      character_model=character_model) for (x1, y1, x2, y2) in line]))

    s = ''
    for line in lines:
        s = " ".join([ctc_recognition(crop[y1:y2, x1:x2],
                                      ctc_model=ctc_model) for (x1, y1, x2, y2) in line])
    print(f'model prediction: {s}')


def char_recognition(img, character_model):
    """Recognition using character model"""
    # Pre-processing the word
    img = word_normalization(
        img,
        100,
        border=False,
        tilt=True,
        hyst_norm=True)

    # Separate letters
    img = cv2.copyMakeBorder(
        img,
        0, 0, 30, 30,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0])
    gaps = characters.segment(img, rnn=True)

    chars = []
    for i in range(len(gaps)-1):
        char = img[:, gaps[i]:gaps[i+1]]
        char, dim = letter_normalization(char, is_thresh=True, dim=True)
        # TODO Test different values
        if dim[0] > 4 and dim[1] > 4:
            chars.append(char.flatten())

    chars = np.array(chars)
    word = ''
    if len(chars) != 0:
        pred = character_model.run(chars)
        for c in pred:
            word += idx2char(c)

    return word


def ctc_recognition(img, ctc_model):
    """Recognising words using CTC Model."""
    img = word_normalization(
        img,
        64,
        border=False,
        tilt=False,
        hyst_norm=False)
    length = img.shape[1]
    # Input has shape [batch_size, height, width, 1]
    input_imgs = np.zeros(
            (1, 64, length, 1), dtype=np.uint8)
    input_imgs[0][:, :length, 0] = img

    pred = ctc_model.eval_feed({
        'inputs:0': input_imgs,
        'inputs_length:0': [length],
        'keep_prob:0': 1})[0]

    word = ''
    for i in pred:
        word += idx2char(i + 1)
    return word


def get_image(cfg):
    os.makedirs(cfg.painter.save_root, exist_ok=True)
    application = tk.Tk()
    application.title("Painter")
    # save_root = Path(f'{cfg.save_image_dir}')
    # os.makedirs(save_root, exist_ok=True)
    instantiate(cfg.painter, application=application)
    # Painter(application=application, save_root=save_root, image_name=cfg.image_name)
    application.mainloop()


if __name__ == '__main__':
    hydra.main(config_path=str(get_config_path()),
               config_name="config",
               version_base="1.2")(main)()
