from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image


class SimpleDatamodule(Dataset):
    def __init__(self, root_dir, label_dir=None, return_label=False):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        
        temp = []
        for ext in ["*.jpg", "*png"]:
            globed = Path(self.root_dir).rglob(ext)
            temp.extend(globed)
        self.data = sorted(temp)
        
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.return_label = return_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = self.preprocess(image_file)
        
        if self.return_label:
            label = self.get_label(image_file)
            return image, label
        return image

    def preprocess(self, image):
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.transform(image)
        return image

    def get_label(self, image_file):
        if self.label_dir is None:
            self.label_dir = self.root_dir

        label = Path(image_file).with_suffix(".txt")
        if not label.is_file():
            raise FileExistsError("The label is not exist!")

        new = []
        with open(label, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "").strip()
            new.append(float(line))
        to_return = torch.tensor(new)
        return to_return
