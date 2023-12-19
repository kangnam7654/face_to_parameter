from typing import Optional, List
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datamodules.simple_datamodule import SimpleDatamodule
from models.animegan import Generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--iteration", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def concat_tensor_images(
    *tensor_images: torch.Tensor,
    no_convert_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Concatenate images.

    Args:
        tensor_images: Input reconstructed images. They will be concatenated in the order provided.
        no_convert_indices: List of indices for images that will skip the BGR conversion.

    Returns:
        np.ndarray: Concatenated grid of images.
    """
    images = list(tensor_images)  # tensor_images == tuple
    nrow = len(images)

    # Handle None case
    if no_convert_indices is None:
        no_convert_indices = []

    # Handle -1 as last index
    if -1 in no_convert_indices:
        no_convert_indices.remove(-1)
        no_convert_indices.append(nrow - 1)

    # Image calibration
    aligned = []
    batch_size = tensor_images[0].shape[0]
    for idx in range(batch_size):
        for image_idx, image in enumerate(images):
            if image_idx in no_convert_indices:
                aligned.append(image[idx])
            else:
                aligned.append(image[idx][[2, 1, 0], :, :])  # RGB to BGR

    grid = (
        make_grid(
            aligned,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return grid


def image_show(*args, no_convert_indices=None):
    image = concat_tensor_images(*args, no_convert_indices=no_convert_indices)
    cv2.imshow("concatenated image", image)
    cv2.waitKey(1)


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
        image_show(images, out)


if __name__ == "__main__":
    args = get_args()
    args.root_dir = "/home/kangnam/datasets/ffhq"
    main(args)
