import os
import time
from pathlib import Path
from shutil import copyfile

from tqdm import tqdm


class Loader:
    def __init__(
        self,
        prohibited: list[str],
        words_path: str,
        labels_file: str,
        output_path: str,
        processed_words_path: str,
        err_words_path: str,
    ):
        self.prohibited = set(prohibited)
        self.words_path = words_path
        self.labels_file = labels_file
        self.processed_words_path = Path(processed_words_path)
        self.err_words_path = Path(err_words_path)

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(self.processed_words_path, exist_ok=True)
        os.makedirs(self.err_words_path, exist_ok=True)

    def load(self):
        tabu_set = set(list([".", "-", "'"]))
        length = len(open(self.labels_file).readlines())

        with open(self.labels_file) as fp:
            for i, line in tqdm(enumerate(fp), total=length):
                if line[0] != "#":
                    pic_id_split = line.strip().split(" ")
                    pic_name, check = pic_id_split[0:2]

                    out = (
                        self.processed_words_path
                        if check == "ok"
                        else self.err_words_path
                    )

                    word = pic_id_split[-1]
                    if word in tabu_set:
                        continue

                    trigger = False
                    for ch in word:
                        if ch in self.prohibited:
                            trigger = True
                            break
                    if trigger:
                        continue

                    folder_name, suffix = pic_name.split("-")[0:2]
                    impath = (
                        Path(self.words_path)
                        / folder_name
                        / "-".join([folder_name, suffix])
                        / f"{pic_name}.png"
                    )
                    if os.stat(impath).st_size != 0:
                        outpath = out / f"{word}_2_{time.time()}.png"
                        copyfile(impath, outpath)

        print(
            "\tNumber of words:",
            len([n for n in os.listdir(self.processed_words_path)]),
        )
