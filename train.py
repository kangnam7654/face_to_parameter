import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pytorch_lightning import Trainer
import pytorch_lightning as pl

from models.translator import Translator
from models.style_transfer import Generator
from models.imitator import Imitator

from pipelines.pipeline import Pipeline
from datamodules.simple_datamodule import SimpleDatamodule

# | Set Seed |
pl.seed_everything(2024)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=float, default=10000000)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main(args):
    # | Dataset |
    dataset = SimpleDatamodule(args.data_dir, return_label=True)
    train_dataset, valid_dataset = random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = Translator(out_features=512)
    style_transfer_model = Generator().eval()
    imitator = Imitator(latent_dim=512).eval()

    pipeline = Pipeline(
        model=model,
        style_transfer_model=style_transfer_model,
        imitator=imitator,
        lr=args.lr,
    )

    trainer = Trainer(max_steps=args.max_steps)
    trainer.fit(
        model=pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
