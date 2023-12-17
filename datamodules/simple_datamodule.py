from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SimpleDatamodule(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data = sorted(
            [Path(self.root_dir).rglob(ext) for ext in ["*.jpg", "*png"]]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
