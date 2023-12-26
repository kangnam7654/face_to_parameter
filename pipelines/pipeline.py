from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Pipeline(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        imitator: nn.Module = None,
        style_transfer_model: nn.Module = None,
        lr=1e-3,
    ):
        super().__init__()
        # | Models |
        self.model = model
        self.imitator = imitator
        self.style_transfer_model = style_transfer_model

        self.lr = lr
        # | Loss |
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.l1_loss = nn.L1Loss()
        self.w_param = 1
        self.w_embed = 1

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch, step="Train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self.loop(batch, step="Valid")
        return loss

    def loop(self, batch, step="Train"):
        # | Prediction |
        real_image, _ = batch
        transfered_image = self.style_transfer_model(real_image)  # Style Transfer
        real_param, real_embed = self.model(transfered_image)

        fake_image = self.imitator(real_param)
        fake_param, fake_embed = self.model(fake_image)

        # | Loss |
        param_loss = self.l1_loss(fake_param, real_param)
        embed_loss = self.cos_loss(fake_embed, real_embed, torch.ones(real_embed.size(0)).type_as(real_embed))
        loss = self.w_param * param_loss + self.w_embed * embed_loss

        # | Logging |
        to_log = {
            f"{step} Param Loss": param_loss,
            f"{step} Embed Loss": embed_loss,
            f"{step} Loss": loss,
        }
        self.log_dict(to_log, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
