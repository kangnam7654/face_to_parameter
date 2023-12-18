import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()    
        self.mse = nn.MSELoss()
    
    def forward(self, x1, x2):
        loss = self.mse(x1, x2)
        return torch.sqrt(loss)