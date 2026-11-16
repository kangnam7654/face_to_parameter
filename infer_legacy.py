"""Run the tracked legacy 37-D Imitator checkpoint on a synthetic vector.

This is NOT face-to-parameter inference. No face image is read. A
deterministic, seeded, synthetic 37-D vector (uniform in [0, 1)) is fed to the
legacy generator on CPU and the 256x256 RGB result is written as one PNG to an
explicitly supplied output path.

The checkpoint is loaded with ``torch.load(..., map_location="cpu",
weights_only=True)`` and strictly validated; no arbitrary pickle code is run,
nothing is downloaded and no external service is contacted.

Example::

    python infer_legacy.py \
        --checkpoint weigths/imitator.pt \
        --seed 0 \
        --output /tmp/legacy_imitator_seed0.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from models.legacy_imitator import (
    LEGACY_LATENT_DIM,
    LEGACY_OUTPUT_RESOLUTION,
    LegacyCheckpointError,
    LegacyImitator,
    load_legacy_imitator,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weigths" / "imitator.pt"
MAX_SEED = 2**32 - 1
# Tanh output should already be in [-1, 1]; allow tiny float slack only.
RANGE_TOLERANCE = 1e-5


def make_synthetic_vector(seed: int, latent_dim: int = LEGACY_LATENT_DIM) -> torch.Tensor:
    """Deterministic synthetic parameter vector of shape [1, latent_dim].

    Values are uniform in [0, 1). This is an arbitrary synthetic probe, not a
    parameter predicted from any face.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(1, latent_dim, generator=generator, dtype=torch.float32)


def validate_output(image: torch.Tensor) -> None:
    """Check shape, finiteness and Tanh range before anything is saved."""
    expected = (1, 3, LEGACY_OUTPUT_RESOLUTION, LEGACY_OUTPUT_RESOLUTION)
    if tuple(image.shape) != expected:
        raise ValueError(
            f"Unexpected output shape {list(image.shape)}, expected {list(expected)}"
        )
    if not image.is_floating_point():
        raise ValueError(f"Unexpected output dtype {image.dtype}")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("Output contains NaN or Inf")
    lo, hi = float(image.min()), float(image.max())
    if lo < -1.0 - RANGE_TOLERANCE or hi > 1.0 + RANGE_TOLERANCE:
        raise ValueError(f"Output outside [-1, 1]: min={lo}, max={hi}")


def to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] -> [H, W, 3] uint8."""
    validate_output(image)
    x = image[0].detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).contiguous().numpy()


@torch.no_grad()
def generate(model: LegacyImitator, seed: int) -> torch.Tensor:
    model.eval()
    vector = make_synthetic_vector(seed, latent_dim=model.latent_dim)
    image = model(vector)
    validate_output(image)
    return image


def save_png(array: np.ndarray, output: Path, overwrite: bool) -> None:
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 array, got {array.dtype} {array.shape}")
    mode = "wb" if overwrite else "xb"
    with open(output, mode) as fp:
        Image.fromarray(array).convert("RGB").save(fp, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one 256x256 PNG from the legacy 37-D Imitator checkpoint "
            "using a seeded synthetic parameter vector (no face input)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Plain legacy state dict (default: weigths/imitator.pt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=f"Seed for the synthetic 37-D vector, integer in [0, {MAX_SEED}].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path (must end in .png; parent directory must exist).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be in [0, {MAX_SEED}], got {args.seed}")
    output = args.output.expanduser()
    if output.suffix.lower() != ".png":
        parser.error(f"--output must end with .png, got {output}")
    if not output.parent.is_dir():
        parser.error(f"--output parent directory does not exist: {output.parent}")
    if output.exists() and not args.overwrite:
        parser.error(f"--output already exists (use --overwrite): {output}")
    if output.exists() and not output.is_file():
        parser.error(f"--output exists and is not a regular file: {output}")

    try:
        model = load_legacy_imitator(args.checkpoint.expanduser())
        image = generate(model, args.seed)
        array = to_rgb_uint8(image)
        save_png(array, output, overwrite=args.overwrite)
    except LegacyCheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {array.shape[1]}x{array.shape[0]} RGB PNG to {output} "
        f"(legacy checkpoint, synthetic seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
