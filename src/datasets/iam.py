import time
import os
from shutil import copyfile
import hydra
from omegaconf import DictConfig
from src.utils.common import get_config_path
from tqdm import tqdm


# Words with these characters are removed
# you have to extend the alphabet in order to use them (ocr/datahelpers.py)
prohibited = [',', '(', ')', ';', ':', '/', '\\',
              '#', '"', '?', '!', '*', '_', '&']


def extract(cfg: DictConfig):
    location = cfg.iam
    output = cfg.output
    err_output = os.path.join(location, 'words_with_error')
    os.makedirs(output, exist_ok=True)
    os.makedirs(err_output, exist_ok=True)

    folder = os.path.join(location, 'words')
    label_file = os.path.join(location, 'words.txt')
    length = len(open(label_file).readlines())

    with open(label_file) as fp:
        for i, line in tqdm(enumerate(fp), total=length):
            if line[0] != '#':
                l = line.strip().split(" ")
                impath = os.path.join(
                    folder,
                    l[0].split('-')[0],
                    l[0].split('-')[0] + '-' + l[0].split('-')[1],
                    l[0] + '.png')
                word = l[-1]

                if (os.stat(impath).st_size != 0
                    and word not in ['.', '-', "'"]
                    and not any(i in word for i in prohibited)):

                    out = output if l[1] == 'ok' else err_output
                    outpath = os.path.join(
                        out, "%s_%s_%s.png" % (word, 2, time.time()))
                    copyfile(impath, outpath)

    print("\tNumber of words:", len([n for n in os.listdir(output)]))


if __name__ == '__main__':
    hydra.main(config_path=str(get_config_path()),
               config_name="config",
               version_base="1.2")(extract)()
