import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer
import pytorch_lightning as pl

from models.translator import Translator
from models.animegan import Generator
from models.imitator import Imitator
from pipelines.pipeline import Pipeline
from datamodules.simple_datamodule import SimpleDatamodule

# | Set Seed |
pl.seed_everything(2024)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_steps", type=float, default=10000000)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--image_save_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./logs")

    # Loss weight args
    parser.add_argument("--w_idt", type=float, default=1, help="Identity loss weight")
    parser.add_argument("--w_loop", type=float, default=1, help="Loopback loss weight")

    # Weight load args
    parser.add_argument(
        "--weight_imitator",
        type=str,
        default=None,
        help="Weight file (e.g. *.pt, *.pth) for imitator",
    )
    parser.add_argument(
        "--weight_predictor",
        default=None,
        help="Weight file (e.g. *.pt, *.pth) for predictor",
    )
    parser.add_argument(
        "--weight_style_transfer",
        type=str,
        default=None,
        help="Weight file (e.g. *.pt, *.pth) for style transfer",
    )

    return parser.parse_args()


def main(args):
    # ===========
    # | Dataset |
    # ===========
    dataset = SimpleDatamodule(
        args.root_dir, return_label=False, resolution=args.resolution
    )
    train_dataset, valid_dataset = random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # ===============
    # | Models Load |
    # ===============
    predictor = Translator(out_features=512)
    style_transfer_model = Generator().eval()
    imitator = Imitator(latent_dim=512).eval()

    # =====================
    # | Load Model Weight |
    # =====================
    if args.weight_imitator is not None:
        weight_imitator = torch.load(args.weight_imitator)
        imitator.load_state_dict(weight_imitator)

    if args.weight_predictor is not None:
        weight_predictor = torch.load(args.weight_predictor)
        predictor.load_state_dict(weight_predictor)

    if args.weight_style_transfer is not None:
        weight_style_transfer = torch.load(args.weight_style_trransfer)
        style_transfer_model.load_state_dict(weight_style_transfer)

    # ================
    # | Pipeline Set |
    # ================
    pipeline = Pipeline(
        predictor=predictor,
        style_transfer=style_transfer_model,
        imitator=imitator,
        lr=args.lr,
        image_save_interval=args.image_save_interval,
        save_dir=args.save_dir
    )
    # =======
    # | Run |
    # =======
    trainer = Trainer(max_steps=args.max_steps)
    trainer.fit(
        model=pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
