from itertools import chain
import argparse
from pathlib import Path
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()
    return args


class LabelMaker:
    def __init__(self, length=37, value_range=(-1, 1)):
        self.length = length
        self.value_range = value_range

    def make_random_label(self):
        random_array = np.random.uniform(
            self.value_range[0], self.value_range[1], self.length
        )
        return random_array

    def write(self, path, label):
        with open(path, "w", encoding="utf-8") as f:
            for line in label:
                f.write(f"{line}\n")


def main(args):
    label_maker = LabelMaker()
    image_files = []
    for ext in ["*.jpg", "*.png"]:
        globed = Path(args.root_dir).rglob(ext)
        image_files.extend(sorted(globed))
    image_files = sorted(image_files)
    for file in image_files:
        random_label = label_maker.make_random_label()
        label_path = file.with_suffix(".txt")
        label_maker.write(str(label_path), random_label)


if __name__ == "__main__":
    args = get_args()
    args.root_dir = "/Users/kangnam/project/face_to_parameter/data"
    main(args)
