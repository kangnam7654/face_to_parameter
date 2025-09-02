import copy
from pathlib import Path

import lightning as L
import torch

from utils.concat_tensor_images import save_image


class ImitatorPipeline(L.LightningModule):
    def __init__(
        self,
        imitator,
        discriminator,
        recon_loss,
        adv_loss,
        G_lr=1e-3,
        D_lr=1e-3,
        gamma=0.3,
        checkpoint_dir="./checkpoints",
        image_save_dir="./log_images",
        save_every=1000,
    ):
        super().__init__()
        self.imitator = imitator
        self.imitator_ema = copy.deepcopy(imitator)

        for p in self.imitator_ema.parameters():
            p.requires_grad_(False)

        self.discriminator = discriminator
        self.recon_loss = recon_loss
        self.adv_loss = adv_loss
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.gamma = gamma
        self.automatic_optimization = False
        self.save_hyperparameters({"G_lr": G_lr, "D_lr": D_lr, "gamma": gamma})
        self._training_step_count = 0
        self._checkpoint_dir = Path(checkpoint_dir)
        self._image_save_dir = Path(image_save_dir)
        self._save_every = save_every
        self.ema_decay = 0.999

    def training_step(self, batch, batch_idx):
        images, labels = batch
        G_optimizers, D_optimizers = self.optimizers()

        # ======================
        # | Discriminator Step |
        # ======================
        self.toggle_optimizer(D_optimizers)

        self.imitator.eval()
        self.discriminator.train()

        fake_image_with_noise = self.imitator(labels, noise_sigma=0.1)

        D_logits_real, _ = self.discriminator(images, labels)
        D_logits_fake, _ = self.discriminator(fake_image_with_noise.detach(), labels)

        D_loss_real = self.adv_loss(D_logits_real, torch.ones_like(D_logits_real))
        D_loss_fake = self.adv_loss(D_logits_fake, torch.zeros_like(D_logits_fake))
        D_loss = (D_loss_real + D_loss_fake) / 2

        D_optimizers.zero_grad()
        self.manual_backward(D_loss)
        D_optimizers.step()

        self.log("train/d_real", D_loss_real)
        self.log("train/d_fake", D_loss_fake)
        self.log("train/d_loss", D_loss, prog_bar=True)
        self.untoggle_optimizer(D_optimizers)

        # ==================
        # | Generator Step |
        # ==================
        self.toggle_optimizer(G_optimizers)
        self.imitator.train()
        self.discriminator.eval()

        fake_image = self.imitator(labels)
        fake_image_with_noise = self.imitator(labels, noise_sigma=0.1)

        G_logits_fake, _ = self.discriminator(fake_image_with_noise, labels)

        G_loss_reconstruction = self.recon_loss(fake_image, images)
        G_loss_adversarial = self.adv_loss(
            G_logits_fake, torch.ones_like(G_logits_fake)
        )
        G_loss = G_loss_reconstruction + self.gamma * G_loss_adversarial

        G_optimizers.zero_grad()
        self.manual_backward(G_loss)
        G_optimizers.step()

        self.log("train/g_recon", G_loss_reconstruction, prog_bar=True)
        self.log("train/g_adv", G_loss_adversarial)
        self.log("train/g_loss", G_loss, prog_bar=True)
        self.untoggle_optimizer(G_optimizers)

        if self._training_step_count % self._save_every == 0:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._image_save_dir.mkdir(parents=True, exist_ok=True)

            name = str(self._training_step_count).zfill(10)
            ckpt_path = self._checkpoint_dir.joinpath(f"ckpt_{name}.ckpt")

            checkpoints = {
                "G": self.imitator.state_dict(),
                "G_ema": self.imitator_ema.state_dict(),
                "D": self.discriminator.state_dict(),
            }
            torch.save(checkpoints, ckpt_path)

            image_path = self._image_save_dir.joinpath(f"image_{name}.png").as_posix()
            save_image(images, fake_image, fake_image_with_noise, save_path=image_path)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update EMA
        with torch.no_grad():
            for p_ema, p in zip(
                self.imitator_ema.parameters(), self.imitator.parameters()
            ):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
        self._training_step_count += 1

    def configure_optimizers(self):
        G_optimizers = torch.optim.AdamW(self.imitator.parameters(), lr=self.G_lr)
        D_optimizers = torch.optim.AdamW(self.discriminator.parameters(), lr=self.D_lr)
        return (G_optimizers, D_optimizers)
