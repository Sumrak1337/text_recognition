import csv
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import simplejson
import unidecode
from tqdm import tqdm

CHARS = [
    "",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    "-",
    "+",
    "'",
]
CHAR_SIZE = len(CHARS)
idxs = [i for i in range(CHAR_SIZE)]
idx_2_chars = dict(zip(idxs, CHARS))
chars_2_idx = dict(zip(CHARS, idxs))


def idx2char(idx, sequence=False):
    if sequence:
        return idx_2_chars[idx - 1]
    return idx_2_chars[idx]


def load_words_data(dataloc="", is_csv=False, load_gaplines=False):
    # TODO: refresh
    """
    Load word images with corresponding labels and gaplines (if load_gaplines == True).
    Args:
        dataloc: image folder location/CSV file - can be list of multiple locations
        is_csv: using CSV files
        load_gaplines: wheter or not load gaplines positions files
    Returns:
        (images, labels (, gaplines))
    """
    print("Loading words...")
    if is_csv:
        with open(dataloc, "r") as file:
            length = len(file.readlines())

        labels = []
        images = []

        with open(dataloc) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, total=length):
                shape = np.fromstring(row["shape"], sep=",", dtype=int)
                img = np.fromstring(row["image"], sep=", ", dtype=np.uint8).reshape(
                    shape
                )
                labels.append(row["label"])
                images.append(img)
    else:
        img_list = []
        tmp_labels = []
        for loc in dataloc:
            tmp_list = glob.glob(os.path.join(loc, "*.png"))
            img_list += tmp_list
            tmp_labels += [name[len(loc) :].split("_")[0] for name in tmp_list]

        labels = np.array(tmp_labels)
        images = np.empty(len(img_list), dtype=object)

        # Load grayscaled images
        for i, img in tqdm(enumerate(img_list[:100]), total=len(img_list[:100])):
            images[i] = cv2.imread(img, 0)

        # Load gaplines (lines separating letters) from txt files
        if load_gaplines:
            gaplines = np.empty(len(img_list), dtype=object)
            for i, name in enumerate(img_list):
                with open(name[:-3] + "txt", "r") as fp:
                    gaplines[i] = np.array(simplejson.load(fp))

    if load_gaplines:
        assert len(labels) == len(images) == len(gaplines)
    else:
        assert len(labels) == len(images)
    print("-> Number of words:", len(labels))

    if load_gaplines:
        return (images, labels, gaplines)
    return images, labels


def char2idx(c, sequence=False):
    if sequence:
        return chars_2_idx[c] + 1
    return chars_2_idx[c]


def sequences_to_sparse(sequences):
    """
    Create a sparse representention of sequences.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray(
        [len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64
    )

    return indices, values, shape
