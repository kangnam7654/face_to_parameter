import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datamodules.simple_datamodule import SimpleDatamodule
from models.animegan import Generator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--iteration", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()    


def main(args):
    dataset = SimpleDatamodule(args.root_dir)
    loader = DataLoader(dataset, batch_size=2)
    model = Generator()
    model = model.cuda()
    generators = iter(loader)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.iteration):
        try:
            images = next(generators)
        except:
            generators = iter(loader)
            images = next(generators)

        images = images.cuda()

        out = model(images)
        loss = criterion(out, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{i+1}/{args.iteration} Train Loss : {loss}")


if __name__ == "__main__":
    args = get_args()
    args.root_dir = "/home/kangnam/datasets/ffhq"
    main(args)
