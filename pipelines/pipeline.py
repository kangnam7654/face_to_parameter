from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Pipeline(pl.LightningModule):
    def __init__(
        self,
        predictor: nn.Module,
        imitator: nn.Module = None,
        style_transfer: nn.Module = None,
        lr=1e-3,
        show: bool = False,
    ):
        super().__init__()
        # | Models |
        self.predictor = predictor
        self.imitator = imitator
        self.style_transfer_model = style_transfer

        self.lr = lr
        # | Loss |
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.l1_loss = nn.L1Loss()

        # Loss Weights
        self.w_idt = 1
        self.w_loop = 1

    def forward(self, image: torch.Tensor):
        slide_value, encoded = self.predictor(image)
        return slide_value, encoded

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch, step="Train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self.loop(batch, step="Valid")
        return loss

    def loop(self, batch, step="Train"):
        # | Prediction |
        real_image = batch
        transfered_image = self.style_transfer_model(real_image)  # Style Transfer
        real_param, real_embed = self.forward(transfered_image)

        fake_image = self.imitator(real_param)
        fake_param, fake_embed = self.predictor(fake_image)

        # | Loss |
        param_loss = self.l1_loss(fake_param, real_param)
        embed_loss = self.cos_loss(
            fake_embed, real_embed, torch.ones(real_embed.size(0)).type_as(real_embed)
        )
        loss = self.w_idt * param_loss + self.w_loop * embed_loss

        # | Logging |
        to_log = {
            f"{step} Identity Loss": param_loss,
            f"{step} Loopback Loss": embed_loss,
            f"{step} Loss": loss,
        }
        self.log_dict(to_log, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
