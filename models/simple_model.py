import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
