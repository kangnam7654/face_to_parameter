import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def concat_tensor_images(
    *tensor_images: torch.Tensor,
    no_convert_indices: list[int] | None = None,
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
    grid = np.clip(grid, 0, 1)
    grid = (grid) * 255
    grid = np.array(grid, dtype=np.uint8)
    return grid


def save_image(*args, save_path, no_convert_indieces=None):
    image = concat_tensor_images(*args, no_convert_indices=no_convert_indieces)
    cv2.imwrite(save_path, image)
