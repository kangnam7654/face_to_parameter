import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from models.simple_model import SimpleRegressor
from pipelines.simple_pipeline import SimplePipeline
from datamodules.simple_datamodule import SimpleDatamodule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=float, default=10000000)

def main(args):
    dataset = SimpleDatamodule()
    model = SimpleRegressor()
    pipeline = SimplePipeline(model=model, lr=args.lr)

    trainer = Trainer(max_steps=args.max_steps)
    trainer.fit(model=pipeline)


if __name__ == "__main__":
    main()
