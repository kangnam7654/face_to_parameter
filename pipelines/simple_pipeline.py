from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimplePipeline(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        model: nn.Module,
        lr,
        style_transfer: bool = False,
        style_transfer_model: nn.Module = None,
        imitator: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.imitator = imitator
        self.encoder = encoder
        self.lr = lr
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.l1_loss = nn.L1Loss()
        self.w_param = 1
        self.w_embed = 1
        
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
        real_image, _ = batch
        transfered_image = self.style_transfer_model(real_image)
        real_param, real_embed = self.model(transfered_image)
        
        fake_image = self.imitator(real_param)
        fake_param, fake_embed = self.model(fake_image)
        
        param_loss = self.l1_loss(fake_param, real_param)
        embed_loss = self.cos_loss(fake_embed, real_embed, 1)
        loss = self.w_param * param_loss + self.w_embed * embed_loss
        return loss
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
