from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl


class SimplePipeline(pl.LightningModule):
    def __init__(self, model: nn.Module, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self.loop(batch)
        return loss

    def loop(self, batch):
        data, label = batch
        out = self.model(data)
        loss = self.compute_criterion(out, label)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def compute_criterion(self, out, label):
        loss = self.criterion(out, label)
        return loss
