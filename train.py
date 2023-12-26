import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pytorch_lightning import Trainer
import pytorch_lightning as pl
from facenet_pytorch import InceptionResnetV1
from models.simple_model import SimpleRegressor
from models.style_transfer import Generator
from pipelines.simple_pipeline import SimplePipeline
from datamodules.simple_datamodule import SimpleDatamodule

# | Set Seed |
pl.seed_everything(2024)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=float, default=10000000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--style_transfer", action="store_true")
    return parser.parse_args()


def main(args):
    # | Dataset |
    dataset = SimpleDatamodule(args.data_dir, return_label=True)
    train_dataset, valid_dataset = random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    encoder = InceptionResnetV1()
    regressor = SimpleRegressor()
    style_transfer_model = Generator().eval()
    pipeline = SimplePipeline(
        model=regressor,
        lr=args.lr,
        style_transfer=args.style_transfer,
        style_transfer_model=style_transfer_model,
    )

    trainer = Trainer(max_steps=args.max_steps)
    trainer.fit(
        model=pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
