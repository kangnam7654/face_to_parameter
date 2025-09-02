from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class SimpleDatamodule(Dataset):
    def __init__(
        self,
        csv_or_parquet: str | Path,
        return_label=False,
        flip=True,
        resolution=512,
    ):

        super().__init__()
        self.data = (
            pl.read_csv(csv_or_parquet)
            if str(csv_or_parquet).endswith(".csv")
            else pl.read_parquet(csv_or_parquet)
        )

        self.resolution = resolution
        self.transform = self.default_transform()
        if flip:
            self.transform.insert(0, v2.RandomHorizontalFlip())
        self.return_label = return_label

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image_file, label_file = self.data.row(idx)
        image = self.preprocess(image_file)

        label = torch.tensor([0])
        if self.return_label:
            label = self.get_label_npy(label_file)
        return (image, label)

    def preprocess(self, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        return image

    def get_label_npy(self, label_path) -> torch.Tensor:
        label = np.load(label_path)
        return torch.from_numpy(label)

    def default_transform(self):
        return v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(
                    (self.resolution, self.resolution),
                    # interpolation=v2.InterpolationMode.LANCZOS,
                ),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
