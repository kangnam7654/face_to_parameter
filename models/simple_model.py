import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 37)

    def forward(self, x):
        x = x.flatten()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
