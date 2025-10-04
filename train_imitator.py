import argparse
import logging

import cv2
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from datamodules.simple_datamodule import SimpleDatamodule
from models.imitator import Imitator, ProjectionDiscriminator
from pipelines.imitator_pipeline import ImitatorPipeline
from utils.concat_tensor_images import concat_tensor_images

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_or_parquet", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def image_show(*args, no_convert_indices=None):
    image = concat_tensor_images(*args, no_convert_indices=no_convert_indices)
    cv2.imshow("concatenated image", image)
    cv2.waitKey(1)


def main(args):
    dataset = SimpleDatamodule(args.csv_or_parquet, return_label=True, flip=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Model Load
    model = Imitator(960)
    discriminator = ProjectionDiscriminator(in_ch=3, cond_in=960)

    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt["G_ema"])
        discriminator.load_state_dict(ckpt["D"])

    pipeline = ImitatorPipeline(
        imitator=model,
        discriminator=discriminator,
        recon_loss=nn.MSELoss(),
        adv_loss=nn.MSELoss(),
        G_lr=args.lr,
        D_lr=args.lr,
        gamma=0.05,
    )

    wandb_logger = WandbLogger(project="face_to_parameter", log_model="all")
    trainer = pl.Trainer(logger=wandb_logger, max_steps=args.iteration)

    trainer.fit(pipeline, loader)


if __name__ == "__main__":
    args = get_args()
    main(args)
