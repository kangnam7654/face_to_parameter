from itertools import chain
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()
    return args


class LabelMaker:
    def __init__(self, length=37, value_range=(-1, 1)):
        self.length = length
        self.value_range = value_range
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval()
        self.prerprocess_transform = v2.Compose(
            [
                v2.Resize(256, interpolation=v2.InterpolationMode.LANCZOS),
                v2.ToTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def make_random_label(self):
        random_array = np.random.uniform(
            self.value_range[0], self.value_range[1], self.length
        )
        return random_array

    def write(self, path, label):
        with open(path, "w", encoding="utf-8") as f:
            for line in label:
                f.write(f"{line}\n")

    def prerprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.prerprocess_transform(image)
        return image

    def embed(self, image_path):
        data = self.prerprocess(image_path)
        data = data.unsqueeze(0)
        embed = self.encoder(data)
        return embed.squeeze(0)


def main(args):
    maker = LabelMaker()
    image_files = []
    for ext in ["*.jpg", "*.png"]:
        globed = Path(args.root_dir).rglob(ext)
        image_files.extend(sorted(globed))
    image_files = sorted(image_files)
    for file in tqdm(image_files):
        label = maker.embed(file)
        label_path = file.with_suffix(".txt")
        maker.write(str(label_path), label)


if __name__ == "__main__":
    args = get_args()
    args.root_dir = "/Users/kangnam/project/face_to_parameter/data"
    main(args)
