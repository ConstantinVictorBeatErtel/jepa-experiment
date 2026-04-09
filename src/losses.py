"""Loss functions for JEPA-style prediction and masked reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_latent_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cosine distance between predicted and target latent tokens."""
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return 1.0 - (predictions * targets).sum(dim=-1).mean()


def normalized_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE after L2-normalization, a stable latent prediction alternative."""
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return F.mse_loss(predictions, targets)


def masked_patch_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE on masked patch pixels only."""
    return F.mse_loss(predictions, targets)


def get_latent_loss(name: str):
    """Resolve the configured JEPA latent loss."""
    choices = {
        "cosine": cosine_latent_loss,
        "normalized_mse": normalized_mse_loss,
    }
    if name not in choices:
        raise ValueError(f"Unknown latent loss '{name}'. Expected one of {sorted(choices)}.")
    return choices[name]

