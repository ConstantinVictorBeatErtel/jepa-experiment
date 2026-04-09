"""Patch creation and masking helpers."""

from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images of shape [B, C, H, W] into flattened patches [B, N, patch_dim]."""
    batch_size, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image size must be divisible by patch size.")
    patches_h = height // patch_size
    patches_w = width // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(batch_size, patches_h * patches_w, channels * patch_size * patch_size)


def batch_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select a batch-specific subset along dimension 1."""
    if values.ndim != 3:
        raise ValueError("Expected values with shape [B, N, D].")
    gather_index = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return torch.gather(values, dim=1, index=gather_index)


def random_masking(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample visible and masked patch indices for each item in a batch."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1.")
    num_masked = max(1, int(num_patches * mask_ratio))
    num_visible = num_patches - num_masked
    if num_visible <= 0:
        raise ValueError("Mask ratio is too high for the number of patches.")
    noise = torch.rand(batch_size, num_patches, device=device)
    shuffled = torch.argsort(noise, dim=1)
    visible_idx = shuffled[:, :num_visible]
    masked_idx = shuffled[:, num_visible:]
    return visible_idx, masked_idx


def full_patch_indices(batch_size: int, num_patches: int, device: str | torch.device) -> torch.Tensor:
    """Return per-batch indices for all patches in order."""
    return torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)

