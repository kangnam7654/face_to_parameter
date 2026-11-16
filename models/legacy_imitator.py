"""Legacy 37-D Imitator generator (Dec 2023 architecture).

This module restores the architecture that matches the tracked
``weigths/imitator.pt`` checkpoint: a plain 12-key state dict whose
``deconv1.conv.weight`` has shape ``[37, 512, 4, 4]``.

It is intentionally separate from :class:`models.imitator.Imitator`, which is
the newer 960-D FiLM-conditioned 512px generator. The legacy checkpoint does
**not** belong to that newer model and must not be loaded into it.

Architecture::

    [B, 37] -> view [B, 37, 1, 1]
    deconv1  TransUpConv 37  -> 512, 4x4   (ConvT k4 s1 p0, BatchNorm, LeakyReLU)
    deconv2  UpConv      512 -> 512, 8     (bilinear x2, Conv3x3, InstanceNorm, LeakyReLU)
    deconv3  UpConv      512 -> 512, 16
    deconv4  UpConv      512 -> 256, 32
    deconv5  UpConv      256 -> 128, 64
    deconv6  UpConv      128 -> 64, 128
    deconv7  OutConv     64  -> 3, 256     (bilinear x2, Conv3x3, Tanh)
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

LEGACY_LATENT_DIM = 37
LEGACY_OUTPUT_RESOLUTION = 256


class LegacyCheckpointError(ValueError):
    """Raised when a checkpoint is missing, unsafe to load, or incompatible."""


class LegacyTransUpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=1, padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LegacyUpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.instance_norm = nn.InstanceNorm2d(out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.instance_norm(self.conv(self.upsample(x))))


class LegacyOutConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.nonlinear = nn.Tanh()

    def forward(self, x):
        return self.nonlinear(self.conv(self.upsample(x)))


class LegacyImitator(nn.Module):
    """37-D parameter vector -> 256x256 RGB image in [-1, 1]."""

    def __init__(self, latent_dim: int = LEGACY_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.deconv1 = LegacyTransUpConv(latent_dim, 512)  # 4
        self.deconv2 = LegacyUpConv(512, 512)  # 8
        self.deconv3 = LegacyUpConv(512, 512)  # 16
        self.deconv4 = LegacyUpConv(512, 256)  # 32
        self.deconv5 = LegacyUpConv(256, 128)  # 64
        self.deconv6 = LegacyUpConv(128, 64)  # 128
        self.deconv7 = LegacyOutConv(64, 3)  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.latent_dim:
            raise ValueError(
                f"LegacyImitator expects input of shape [B, {self.latent_dim}], "
                f"got {list(x.shape)}"
            )
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        return x


def expected_state_dict_schema(latent_dim: int = LEGACY_LATENT_DIM):
    """Return ``{key: (shape, dtype)}`` expected for the checkpoint."""
    reference = LegacyImitator(latent_dim=latent_dim).state_dict()
    return {k: (tuple(v.shape), v.dtype) for k, v in reference.items()}


def validate_legacy_state_dict(state_dict, latent_dim: int = LEGACY_LATENT_DIM):
    """Validate a loaded object against the exact legacy schema.

    Raises :class:`LegacyCheckpointError` listing every problem found; never
    performs partial loading.
    """
    if not isinstance(state_dict, Mapping):
        raise LegacyCheckpointError(
            "Checkpoint is not a plain state dict "
            f"(got {type(state_dict).__name__})."
        )

    schema = expected_state_dict_schema(latent_dim)
    problems = []

    non_str = [k for k in state_dict.keys() if not isinstance(k, str)]
    if non_str:
        problems.append(f"non-string keys: {non_str!r}")

    keys = {k for k in state_dict.keys() if isinstance(k, str)}
    missing = sorted(set(schema) - keys)
    unexpected = sorted(keys - set(schema))
    if missing:
        problems.append(f"missing keys: {missing}")
    if unexpected:
        problems.append(f"unexpected keys: {unexpected}")

    for key in sorted(set(schema) & keys):
        value = state_dict[key]
        shape, expected_dtype = schema[key]
        if not isinstance(value, torch.Tensor):
            problems.append(f"{key}: expected Tensor, got {type(value).__name__}")
            continue
        if tuple(value.shape) != shape:
            problems.append(
                f"{key}: shape mismatch, expected {list(shape)}, "
                f"got {list(value.shape)}"
            )
            continue
        if value.dtype != expected_dtype:
            problems.append(f"{key}: expected {expected_dtype}, got {value.dtype}")
            continue
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            problems.append(f"{key}: contains NaN or Inf")

    if problems:
        raise LegacyCheckpointError(
            "Checkpoint is incompatible with LegacyImitator "
            f"(latent_dim={latent_dim}):\n  - " + "\n  - ".join(problems)
        )


def load_legacy_state_dict(path: Union[str, Path]):
    """Safely load a plain state dict on CPU with ``weights_only=True``."""
    path = Path(path)
    if not path.is_file():
        raise LegacyCheckpointError(f"Checkpoint file not found: {path}")
    try:
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # unpickling refusal, corrupt archive, etc.
        raise LegacyCheckpointError(
            f"Failed to load checkpoint {path} with weights_only=True: {exc}"
        ) from exc
    return state_dict


def load_legacy_imitator(
    path: Union[str, Path], latent_dim: int = LEGACY_LATENT_DIM
) -> LegacyImitator:
    """Build a :class:`LegacyImitator` and strictly load ``path`` into it.

    The returned model is on CPU in eval mode.
    """
    state_dict = load_legacy_state_dict(path)
    validate_legacy_state_dict(state_dict, latent_dim=latent_dim)
    model = LegacyImitator(latent_dim=latent_dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise LegacyCheckpointError(
            f"Strict load of {path} into LegacyImitator failed: {exc}"
        ) from exc
    return model.eval()
