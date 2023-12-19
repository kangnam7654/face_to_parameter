from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x1, x2):
        loss = self.mse(x1, x2)
        return torch.sqrt(loss)


class SimplePipeline(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        model: nn.Module,
        lr,
        style_transfer: bool = False,
        style_transfer_model: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.lr = lr
        self.criterion = RMSELoss()
        self.style_transfer = style_transfer
        self.style_transfer_model = style_transfer_model

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch)
        self.log("Train Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self.loop(batch)
        self.log("Valid Loss", loss, prog_bar=True)
        return loss

    def loop(self, batch):
        image, label = batch
        if self.style_transfer:
            transfered_image = self.style_transfer_model(image)
        vector = self.encoder(transfered_image)
        out = self.model(vector)
        loss = self.compute_criterion(out, label)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def compute_criterion(self, out, label):
        loss = self.criterion(out, label)
        return loss
