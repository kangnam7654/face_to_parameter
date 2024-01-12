from pathlib import Path
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import save_image


class Pipeline(pl.LightningModule):
    def __init__(
        self,
        predictor: nn.Module,
        imitator: nn.Module,
        style_transfer: nn.Module,
        save_dir="./logs",
        lr=1e-3,
        image_save_interval=100,
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

        # Save Args
        self.image_save_interval = image_save_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # else
        self.training_step_counter = 0

    def forward(self, image: torch.Tensor):
        slide_value, encoded = self.predictor(image)
        return slide_value, encoded

    def training_step(self, batch, batch_idx):
        loss, real_image, transfered_image, fake_image = self.loop(batch, step="Train")

        # | Image Save |
        if self.training_step_counter % self.image_save_interval == 0:
            full_path = self.save_dir.joinpath(f"{self.training_step_counter}".zfill(8) + ".jpg")
            save_image([real_image, transfered_image, fake_image], full_path)

        self.training_step_counter += 1
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss, _, _, _ = self.loop(batch, step="Valid")
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

        return loss, real_image, transfered_image, fake_image

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        return opt
