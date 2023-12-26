import sys
import os
import glob
import unidecode
import simplejson
import csv
import cv2
from tqdm import tqdm

import numpy as np

from src.ocr.normalization import letter_normalization
from src.ocr.viz import print_progress_bar

CHARS = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
         'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', '.', '-', '+', "'"]

idxs = [i for i in range(len(CHARS))]
idx_2_chars = dict(zip(idxs, CHARS))
chars_2_idx = dict(zip(CHARS, idxs))


def idx2char(idx, sequence=False):
    if sequence:
        return idx_2_chars[idx-1]
    return idx_2_chars[idx]


def load_chars_data(charloc='data/charclas/', wordloc='data/words/', lang='en'):
    """
    Load chars images with corresponding labels.
    Args:
        charloc: char images FOLDER LOCATION
        wordloc: word images with gaplines FOLDER LOCATION
    Returns:
        (images, labels)
    """
    print("Loading chars...")
    images = np.zeros((1, 4096))
    labels = []

    if charloc != '':
        # Get subfolders with chars
        dir_list = glob.glob(os.path.join(charloc, lang, "*/"))
        dir_list.sort()

        # if lang == 'en':
        chars = CHARS[:53]

        assert [d[-2] if d[-2] != '0' else '' for d in dir_list] == chars

        # For every label load images and create corresponding labels
        # cv2.imread(img, 0) - for loading images in grayscale
        # Images are scaled to 64x64 = 4096 px
        for i in range(len(chars)):
            img_list = glob.glob(os.path.join(dir_list[i], '*.jpg'))
            imgs = np.array([letter_normalization(cv2.imread(img, 0)) for img in img_list])
            images = np.concatenate([images, imgs.reshape(len(imgs), 4096)])
            labels.extend([i] * len(imgs))

    if wordloc != '':
        imgs, words, gaplines = load_words_data(wordloc, load_gaplines=True)
        if lang != 'cz':
            words = np.array([unidecode.unidecode(w) for w in words])
        imgs, chars = _words2chars(imgs, words, gaplines)

        labels.extend(chars)
        images2 = np.zeros((len(imgs), 4096))
        for i in range(len(imgs)):
            images2[i] = letter_normalization(imgs[i]).reshape(1, 4096)

        images = np.concatenate([images, images2])

    images = images[1:]
    labels = np.array(labels)

    print("-> Number of chars:", len(labels))
    return images, labels


def load_words_data(dataloc='', is_csv=False, load_gaplines=False):
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
    if type(dataloc) is not list:
        dataloc = [dataloc]

    if is_csv:
        csv.field_size_limit(sys.maxsize)
        length = 0
        for loc in dataloc:
            with open(loc) as csvfile:
                reader = csv.reader(csvfile)
                length += max(sum(1 for row in csvfile) - 1, 0)

        labels = np.empty(length, dtype=object)
        images = np.empty(length, dtype=object)
        i = 0
        for loc in dataloc:
            with open(loc) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    shape = np.fromstring(
                        row['shape'],
                        sep=',',
                        dtype=int)
                    img = np.fromstring(
                        row['image'],
                        sep=', ',
                        dtype=np.uint8).reshape(shape)
                    labels[i] = row['label']
                    images[i] = img

                    print_progress_bar(i, length)
                    i += 1
    else:
        img_list = []
        tmp_labels = []
        for loc in dataloc:
            tmp_list = glob.glob(os.path.join(loc, '*.png'))
            img_list += tmp_list
            tmp_labels += [name[len(loc):].split("_")[0] for name in tmp_list]

        labels = np.array(tmp_labels)
        images = np.empty(len(img_list), dtype=object)

        # Load grayscaled images
        for i, img in tqdm(enumerate(img_list[:100]), total=len(img_list[:100])):
            images[i] = cv2.imread(img, 0)

        # Load gaplines (lines separating letters) from txt files
        if load_gaplines:
            gaplines = np.empty(len(img_list), dtype=object)
            for i, name in enumerate(img_list):
                with open(name[:-3] + 'txt', 'r') as fp:
                    gaplines[i] = np.array(simplejson.load(fp))

    if load_gaplines:
        assert len(labels) == len(images) == len(gaplines)
    else:
        assert len(labels) == len(images)
    print("-> Number of words:", len(labels))

    if load_gaplines:
        return (images, labels, gaplines)
    return (images, labels)


def _words2chars(images, labels, gaplines):
    """Transform word images with gaplines into individual chars."""
    # Total number of chars
    length = sum([len(l) for l in labels])

    imgs = np.empty(length, dtype=object)
    new_labels = []

    height = images[0].shape[0]

    idx = 0
    for i, gaps in enumerate(gaplines):
        for pos in range(len(gaps) - 1):
            imgs[idx] = images[i][0:height, gaps[pos]:gaps[pos + 1]]
            new_labels.append(char2idx(labels[i][pos]))
            idx += 1

    print("Loaded chars from words:", length)
    return imgs, new_labels


def char2idx(c, sequence=False):
    if sequence:
        return chars_2_idx[c] + 1
    return chars_2_idx[c]


def corresponding_shuffle(a):
    """
    Shuffle array of numpy arrays such that
    each pair a[x][i] and a[y][i] remains the same.
    Args:
        a: array of same length numpy arrays
    Returns:
        Array a with shuffled numpy arrays
    """
    assert all([len(a[0]) == len(a[i]) for i in range(len(a))])
    p = np.random.permutation(len(a[0]))
    for i in range(len(a)):
        a[i] = a[i][p]
    return a
