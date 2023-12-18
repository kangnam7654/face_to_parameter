import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, 8, 8)

        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten()
        x = self.fc1(x)
        return x
