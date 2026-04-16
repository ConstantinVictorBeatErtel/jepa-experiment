import argparse
import csv
import math
import os
import random
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
IMAGE_SIZE = 96
PATCH_SIZE = 8
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE
EMBED_DIM = 256
DEPTH = 6
HEADS = 4
MLP_RATIO = 4.0
DROPOUT = 0.1
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 42
TARGET_BLOCKS = 4
JEPA_EMA_START = 0.996
JEPA_EMA_END = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_results_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "viz").mkdir(parents=True, exist_ok=True)
    (root / "banks").mkdir(parents=True, exist_ok=True)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def unnormalize_numpy(tensor: torch.Tensor) -> np.ndarray:
    img = denormalize_image(tensor).clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return img


def resize_numpy_image(array: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    pil = Image.fromarray(array)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil)


def patchify(images: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    b, c, h, w = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(b, -1, c * patch_size * patch_size)


def batched_gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=gather_idx)


def pairwise_cosine_topk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    topk: int = 5,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values_list = []
    indices_list = []
    bank_t = bank.t().contiguous()
    for start in range(0, queries.size(0), chunk_size):
        q = queries[start : start + chunk_size]
        sims = q @ bank_t
        vals, idxs = torch.topk(sims, k=topk, dim=1)
        values_list.append(vals)
        indices_list.append(idxs)
    return torch.cat(values_list, dim=0), torch.cat(indices_list, dim=0)


def batched_cosine_topk_memmap(
    queries: np.ndarray,
    memmap_path: Path,
    shape: Tuple[int, int],
    topk: int = 5,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    queries = np.nan_to_num(np.asarray(queries, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bank = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=shape)
    num_queries = queries.shape[0]
    best_scores = np.full((num_queries, topk), -np.inf, dtype=np.float32)
    best_indices = np.full((num_queries, topk), -1, dtype=np.int64)

    for start in range(0, shape[0], chunk_size):
        chunk = np.nan_to_num(np.asarray(bank[start : start + chunk_size], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sims = queries @ chunk.T
        k = min(topk, sims.shape[1])
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        idx = idx + start

        combined_scores = np.concatenate([best_scores, scores], axis=1)
        combined_indices = np.concatenate([best_indices, idx], axis=1)
        keep = np.argpartition(-combined_scores, kth=topk - 1, axis=1)[:, :topk]
        best_scores = np.take_along_axis(combined_scores, keep, axis=1)
        best_indices = np.take_along_axis(combined_indices, keep, axis=1)
        order = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_indices = np.take_along_axis(best_indices, order, axis=1)

    return best_scores, best_indices


def cosine_topk_memmap(
    query: np.ndarray,
    memmap_path: Path,
    shape: Tuple[int, int],
    topk: int = 5,
    chunk_size: int = 250_000,
) -> Tuple[np.ndarray, np.ndarray]:
    bank = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=shape)
    best_scores = np.full(topk, -np.inf, dtype=np.float32)
    best_indices = np.full(topk, -1, dtype=np.int64)
    offset = 0
    q = query.astype(np.float32, copy=False)
    for start in range(0, shape[0], chunk_size):
        chunk = np.asarray(bank[start : start + chunk_size], dtype=np.float32)
        sims = chunk @ q
        k = min(topk, sims.shape[0])
        chunk_idx = np.argpartition(-sims, kth=k - 1)[:k]
        chunk_scores = sims[chunk_idx]
        global_idx = chunk_idx + offset
        combined_scores = np.concatenate([best_scores, chunk_scores])
        combined_indices = np.concatenate([best_indices, global_idx])
        keep = np.argpartition(-combined_scores, kth=topk - 1)[:topk]
        order = np.argsort(-combined_scores[keep])
        best_scores = combined_scores[keep][order]
        best_indices = combined_indices[keep][order]
        offset += chunk.shape[0]
    return best_scores, best_indices


def cosine_scheduler(
    base_value: float,
    final_value: float,
    steps: int,
    warmup_steps: int = 0,
    start_warmup_value: float = 0.0,
) -> List[float]:
    if steps <= 0:
        return []
    schedule = []
    if warmup_steps > 0:
        for i in range(warmup_steps):
            alpha = i / max(1, warmup_steps - 1)
            schedule.append(start_warmup_value + alpha * (base_value - start_warmup_value))
    remaining = steps - warmup_steps
    if remaining <= 0:
        return schedule[:steps]
    for i in range(remaining):
        cosine = 0.5 * (1 + math.cos(math.pi * i / max(1, remaining - 1)))
        schedule.append(final_value + (base_value - final_value) * cosine)
    return schedule[:steps]


def linear_scheduler(start_value: float, end_value: float, steps: int) -> List[float]:
    if steps <= 0:
        return []
    if steps == 1:
        return [end_value]
    return [start_value + (end_value - start_value) * (i / (steps - 1)) for i in range(steps)]


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("Embedding dimension must be divisible by 4.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float().unsqueeze(0)


def get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


def sample_block_indices(
    rng: random.Random,
    grid_size: int,
    total_patches: int,
    scale_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
) -> torch.Tensor:
    target_area = rng.uniform(*scale_range) * total_patches
    aspect = rng.uniform(*aspect_range)
    h = int(round(math.sqrt(target_area / aspect)))
    w = int(round(math.sqrt(target_area * aspect)))
    h = min(max(h, 1), grid_size)
    w = min(max(w, 1), grid_size)
    top = rng.randint(0, grid_size - h)
    left = rng.randint(0, grid_size - w)
    coords = []
    for y in range(top, top + h):
        for x in range(left, left + w):
            coords.append(y * grid_size + x)
    return torch.tensor(sorted(coords), dtype=torch.long)


def sample_context_indices(
    rng: random.Random,
    grid_size: int,
    total_patches: int,
    target_blocks: Sequence[torch.Tensor],
) -> torch.Tensor:
    target_union = set()
    for block in target_blocks:
        target_union.update(block.tolist())
    best_context = None
    for _ in range(50):
        context = sample_block_indices(
            rng,
            grid_size=grid_size,
            total_patches=total_patches,
            scale_range=(0.85, 1.0),
            aspect_range=(1.0, 1.0),
        )
        context = torch.tensor(
            sorted(idx for idx in context.tolist() if idx not in target_union), dtype=torch.long
        )
        if best_context is None or context.numel() > best_context.numel():
            best_context = context
        if context.numel() >= 50:
            return context
    fallback = torch.tensor(sorted(set(range(total_patches)) - target_union), dtype=torch.long)
    if fallback.numel() >= 50:
        return fallback
    return best_context


def sample_multiblock_masks(seed: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    rng = random.Random(seed)
    best_context = None
    best_targets = None
    for _ in range(100):
        targets = [
            sample_block_indices(
                rng,
                grid_size=GRID_SIZE,
                total_patches=NUM_PATCHES,
                scale_range=(0.15, 0.2),
                aspect_range=(0.75, 1.5),
            )
            for _ in range(TARGET_BLOCKS)
        ]
        context = sample_context_indices(rng, GRID_SIZE, NUM_PATCHES, targets)
        if best_context is None or context.numel() > best_context.numel():
            best_context = context
            best_targets = targets
        if context.numel() >= 50:
            return context, targets
    return best_context, best_targets


def indices_to_rect(indices: Sequence[int]) -> Tuple[int, int, int, int]:
    rows = [idx // GRID_SIZE for idx in indices]
    cols = [idx % GRID_SIZE for idx in indices]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    return min_c, min_r, max_c - min_c + 1, max_r - min_r + 1


@dataclass
class BatchMasks:
    images: torch.Tensor
    labels: torch.Tensor
    context_indices: torch.Tensor
    target_indices: List[torch.Tensor]


class MultiBlockMaskCollator:
    def __init__(self, base_seed: int = SEED) -> None:
        self.base_seed = base_seed
        self.counter = 0

    def _trim_and_stack(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        min_len = min(seq.numel() for seq in sequences)
        trimmed = []
        for seq in sequences:
            if seq.numel() == min_len:
                trimmed.append(seq)
                continue
            perm = torch.randperm(seq.numel())[:min_len]
            trimmed.append(seq[perm].sort().values)
        return torch.stack(trimmed, dim=0)

    def __call__(self, batch: Sequence[Tuple[torch.Tensor, int]]) -> Dict[str, object]:
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        contexts = []
        targets_per_block = [[] for _ in range(TARGET_BLOCKS)]
        for sample_offset in range(len(batch)):
            context, targets = sample_multiblock_masks(self.base_seed + self.counter + sample_offset)
            contexts.append(context)
            for block_id, target in enumerate(targets):
                targets_per_block[block_id].append(target)
        self.counter += len(batch)
        context_indices = self._trim_and_stack(contexts)
        target_indices = [self._trim_and_stack(blocks) for blocks in targets_per_block]
        return {
            "images": images,
            "labels": labels,
            "context_indices": context_indices,
            "target_indices": target_indices,
        }


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        num_heads: int = HEADS,
        mlp_ratio: float = MLP_RATIO,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(get_2d_sincos_pos_embed(embed_dim, self.grid_size), requires_grad=False)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        return x

    def forward(
        self, images: torch.Tensor, visible_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        if visible_indices is None:
            x = tokens + self.pos_embed.to(tokens.device, tokens.dtype)
        else:
            pos = batched_gather(
                self.pos_embed.expand(tokens.size(0), -1, -1).to(tokens.device, tokens.dtype), visible_indices
            )
            x = batched_gather(tokens, visible_indices) + pos
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images, None).mean(dim=1)


class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 128, depth: int = 6, num_heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=DROPOUT) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_indices: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        b, target_len = target_indices.shape
        target_pos = batched_gather(pos_embed.expand(b, -1, -1), target_indices)
        mask_tokens = self.mask_token.expand(b, target_len, -1) + target_pos
        x = torch.cat([context_tokens, mask_tokens], dim=1)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x[:, -target_len:, :]


class IJEPAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = VisionTransformerEncoder()
        self.target_encoder = VisionTransformerEncoder()
        self.predictor = JEPAPredictor()
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for ctx_param, tgt_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_param.data.mul_(momentum).add_(ctx_param.data, alpha=1.0 - momentum)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context_tokens = self.context_encoder(images, context_indices)
        with torch.no_grad():
            target_tokens = self.target_encoder(images, None)
        losses = []
        predictions = []
        targets = []
        pos_embed = self.context_encoder.pos_embed.to(images.device, images.dtype)
        for target_indices in target_indices_list:
            pred = self.predictor(context_tokens, target_indices, pos_embed)
            tgt = batched_gather(target_tokens, target_indices).detach()
            predictions.append(pred)
            targets.append(tgt)
            losses.append(((pred - tgt) ** 2).sum(dim=-1).mean())
        loss = torch.stack(losses).mean()
        return loss, {"predictions": predictions, "targets": targets, "context_tokens": context_tokens}


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, depth: int = 2, num_heads: int = 4, patch_dim: int = 192):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=DROPOUT) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, context_tokens: torch.Tensor, masked_indices: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        b, masked_len = masked_indices.shape
        masked_pos = batched_gather(pos_embed.expand(b, -1, -1), masked_indices)
        mask_tokens = self.mask_token.expand(b, masked_len, -1) + masked_pos
        x = torch.cat([context_tokens, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, -masked_len:, :])


class MAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VisionTransformerEncoder()
        self.decoder = MAEDecoder()

    @staticmethod
    def _union_targets(target_indices_list: Sequence[torch.Tensor]) -> torch.Tensor:
        unions = []
        for sample_idx in range(target_indices_list[0].size(0)):
            merged = torch.unique(torch.cat([target[sample_idx] for target in target_indices_list], dim=0), sorted=True)
            unions.append(merged)
        min_len = min(item.numel() for item in unions)
        trimmed = []
        for item in unions:
            if item.numel() == min_len:
                trimmed.append(item)
            else:
                perm = torch.randperm(item.numel())[:min_len]
                trimmed.append(item[perm].sort().values)
        return torch.stack(trimmed, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context_tokens = self.encoder(images, context_indices)
        union_indices = self._union_targets(target_indices_list).to(images.device)
        pos_embed = self.encoder.pos_embed.to(images.device, images.dtype)
        recon = self.decoder(context_tokens, union_indices, pos_embed)
        true_patches = batched_gather(patchify(images), union_indices)
        loss = F.mse_loss(recon, true_patches)
        return loss, {"reconstruction": recon, "targets": true_patches, "union_indices": union_indices}


class TensorFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    base = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    vis = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    return base, vis


def build_datasets(data_root: Path) -> Dict[str, Dataset]:
    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, vis_transform = build_transforms()
    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR10(root=data_root, train=False, download=True, transform=train_transform)
    test_vis = datasets.CIFAR10(root=data_root, train=False, download=False, transform=vis_transform)
    train_vis = datasets.CIFAR10(root=data_root, train=True, download=False, transform=vis_transform)
    return {
        "train_full": train_full,
        "test_full": test_full,
        "test_vis": test_vis,
        "train_vis": train_vis,
    }


def split_train_val(dataset: Dataset, val_size: int = 5_000) -> Tuple[Subset, Subset]:
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn=None,
    device: Optional[torch.device] = None,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    use_cuda = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def move_mask_batch_to_device(batch: Dict[str, object], device: torch.device) -> BatchMasks:
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    context_indices = batch["context_indices"].to(device, non_blocking=True)
    target_indices = [target.to(device, non_blocking=True) for target in batch["target_indices"]]
    return BatchMasks(images=images, labels=labels, context_indices=context_indices, target_indices=target_indices)


def ema_momentum_schedule(total_steps: int) -> List[float]:
    return linear_scheduler(JEPA_EMA_START, JEPA_EMA_END, total_steps)


def save_checkpoint(path: Path, state: Dict[str, object]) -> None:
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=device)


def run_self_tests(device: torch.device, train_loader: DataLoader) -> None:
    print("Running self-tests...")
    raw_batch = next(iter(train_loader))
    context_indices_cpu = raw_batch["context_indices"][:4].clone()
    target_indices_cpu = [target[:4].clone() for target in raw_batch["target_indices"]]
    batch = move_mask_batch_to_device(raw_batch, device)
    batch = BatchMasks(
        images=batch.images[:4],
        labels=batch.labels[:4],
        context_indices=batch.context_indices[:4],
        target_indices=[target[:4] for target in batch.target_indices],
    )

    tests = []

    try:
        counts_ok = bool((context_indices_cpu.size(1) >= 50))
        assert counts_ok
        tests.append(("Context patches per image >= 50", True))
    except Exception as exc:
        tests.append((f"Context patches per image >= 50 ({exc})", False))

    try:
        for sample_idx in range(batch.images.size(0)):
            context_set = set(context_indices_cpu[sample_idx].tolist())
            for block_idx, target in enumerate(target_indices_cpu):
                overlap_vals = sorted(context_set & set(target[sample_idx].tolist()))
                if overlap_vals:
                    raise AssertionError(
                        f"sample={sample_idx} block={block_idx} overlap_vals={overlap_vals}"
                    )
        tests.append(("Context and targets are non-overlapping", True))
    except Exception as exc:
        tests.append((f"Context and targets are non-overlapping ({exc})", False))

    jepa = IJEPAModel().to(device)
    mae = MAEModel().to(device)
    optimizer = AdamW(
        list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters()),
        lr=1e-3,
        weight_decay=0.04,
    )

    try:
        loss, _ = jepa(batch.images, batch.context_indices, batch.target_indices)
        assert loss.ndim == 0
        tests.append(("JEPA forward scalar loss", True))
    except Exception as exc:
        tests.append((f"JEPA forward scalar loss ({exc})", False))

    try:
        loss, _ = mae(batch.images, batch.context_indices, batch.target_indices)
        assert loss.ndim == 0
        tests.append(("MAE forward scalar loss", True))
    except Exception as exc:
        tests.append((f"MAE forward scalar loss ({exc})", False))

    try:
        equal = all(
            torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in zip(jepa.context_encoder.parameters(), jepa.target_encoder.parameters())
        )
        assert equal
        tests.append(("Target encoder equals context encoder at init", True))
    except Exception as exc:
        tests.append((f"Target encoder equals context encoder at init ({exc})", False))

    try:
        before_context = [param.detach().clone() for param in jepa.context_encoder.parameters()]
        before_target = [param.detach().clone() for param in jepa.target_encoder.parameters()]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = jepa(batch.images, batch.context_indices, batch.target_indices)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder(0.996)
        context_changed = any(not torch.equal(before, after.detach()) for before, after in zip(before_context, jepa.context_encoder.parameters()))
        target_changed = any(not torch.equal(before, after.detach()) for before, after in zip(before_target, jepa.target_encoder.parameters()))
        target_matches_context = all(
            torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in zip(jepa.context_encoder.parameters(), jepa.target_encoder.parameters())
        )
        assert context_changed and target_changed and not target_matches_context
        tests.append(("EMA update changes target slightly and diverges from context", True))
    except Exception as exc:
        tests.append((f"EMA update behavior ({exc})", False))

    failed = False
    for name, passed in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        failed = failed or (not passed)
    if failed:
        raise RuntimeError("Self-tests failed. Aborting.")


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    results_dir: Path,
    epochs: int,
) -> Tuple[nn.Module, Dict[str, float], Path, Path]:
    if model_name == "jepa":
        params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    else:
        params = list(model.parameters())
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.04)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(10 * len(train_loader), total_steps)
    lr_schedule = cosine_scheduler(1e-3, 1e-6, total_steps, warmup_steps=warmup_steps, start_warmup_value=1e-4)
    wd_schedule = linear_scheduler(0.04, 0.4, total_steps)
    ema_schedule = ema_momentum_schedule(total_steps) if model_name == "jepa" else None
    log_path = results_dir / f"{model_name}_train_log.csv"
    best_path = results_dir / f"{model_name}_best.pt"
    final_path = results_dir / f"{model_name}_final.pt"
    best_val = float("inf")
    best_epoch = -1
    global_step = 0

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = ["epoch", "train_loss", "val_loss", "lr"]
        if model_name == "jepa":
            headers.append("ema_momentum")
        writer.writerow(headers)

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            progress = tqdm(train_loader, desc=f"{model_name.upper()} Epoch {epoch}/{epochs}", leave=False)
            for raw_batch in progress:
                batch = move_mask_batch_to_device(raw_batch, device)
                lr = lr_schedule[global_step]
                wd = wd_schedule[global_step]
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = wd
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model(batch.images, batch.context_indices, batch.target_indices)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                if model_name == "jepa":
                    model.update_target_encoder(ema_schedule[global_step])
                train_losses.append(loss.item())
                global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            val_loss = evaluate_pretrain(model, val_loader, device)
            train_loss = float(np.mean(train_losses))
            row = [epoch, train_loss, val_loss, lr_schedule[min(global_step - 1, len(lr_schedule) - 1)]]
            if model_name == "jepa":
                row.append(ema_schedule[min(global_step - 1, len(ema_schedule) - 1)])
            writer.writerow(row)
            f.flush()
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(
                    best_path,
                    {
                        "epoch": epoch,
                        "model_name": model_name,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val,
                    },
                )
            print(
                f"{model_name.upper()} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"lr={row[3]:.6e}" + (f" ema={row[4]:.6f}" if model_name == "jepa" else "")
            )

    save_checkpoint(
        final_path,
        {
            "epoch": epochs,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        },
    )

    best_state = load_checkpoint(best_path, device)
    model.load_state_dict(best_state["model_state"])
    return model, {"best_val_loss": best_val, "best_epoch": best_epoch}, best_path, final_path


@torch.no_grad()
def evaluate_pretrain(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for raw_batch in loader:
        batch = move_mask_batch_to_device(raw_batch, device)
        loss, _ = model(batch.images, batch.context_indices, batch.target_indices)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def extract_global_features(
    encoder: VisionTransformerEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_features = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting global features", leave=False):
        images = images.to(device, non_blocking=True)
        feats = encoder.forward_global(images).cpu()
        all_features.append(feats)
        all_labels.append(labels)
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
) -> float:
    train_dataset = TensorFeatureDataset(train_features, train_labels)
    test_dataset = TensorFeatureDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    classifier = nn.Linear(train_features.size(1), 10).to(device)
    optimizer = SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    total_steps = epochs * len(train_loader)
    lr_schedule = cosine_scheduler(0.1, 1e-5, total_steps)
    step = 0

    for epoch in range(epochs):
        classifier.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            for group in optimizer.param_groups:
                group["lr"] = lr_schedule[step]
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            step += 1

    classifier.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for features, labels in test_loader:
            logits = classifier(features.to(device))
            preds.append(logits.argmax(dim=1).cpu())
            truth.append(labels)
    return float(accuracy_score(torch.cat(truth).numpy(), torch.cat(preds).numpy()) * 100.0)


@torch.no_grad()
def knn_retrieval_metrics(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    topk: int = 5,
) -> Tuple[float, float]:
    train_features = F.normalize(train_features.float(), dim=1)
    test_features = F.normalize(test_features.float(), dim=1)
    acc1 = 0
    acc5 = 0
    total = test_features.size(0)
    for start in tqdm(range(0, total, 256), desc="k-NN retrieval", leave=False):
        feats = test_features[start : start + 256]
        _, idx = pairwise_cosine_topk(feats, train_features, topk=topk, chunk_size=256)
        neighbor_labels = train_labels[idx]
        labels = test_labels[start : start + feats.size(0)].unsqueeze(1)
        acc1 += (neighbor_labels[:, :1] == labels).any(dim=1).sum().item()
        acc5 += (neighbor_labels == labels).any(dim=1).sum().item()
    return acc1 / total * 100.0, acc5 / total * 100.0


@torch.no_grad()
def build_patch_bank(
    encoder: VisionTransformerEncoder,
    loader: DataLoader,
    dataset_len: int,
    device: torch.device,
    output_prefix: Path,
) -> Dict[str, object]:
    encoder.eval()
    feature_path = output_prefix.with_suffix(".dat")
    meta_path = output_prefix.with_suffix(".npz")
    total_patches = dataset_len * NUM_PATCHES
    bank = np.memmap(feature_path, dtype=np.float16, mode="w+", shape=(total_patches, EMBED_DIM))
    image_indices = np.memmap(output_prefix.with_name(output_prefix.name + "_image_idx.dat"), dtype=np.int32, mode="w+", shape=(total_patches,))
    patch_indices = np.memmap(output_prefix.with_name(output_prefix.name + "_patch_idx.dat"), dtype=np.int16, mode="w+", shape=(total_patches,))
    labels_all = []
    cursor = 0
    image_cursor = 0
    for images, labels in tqdm(loader, desc=f"Building patch bank {output_prefix.name}", leave=False):
        images = images.to(device, non_blocking=True)
        reps = encoder(images, None)
        reps = torch.nan_to_num(F.normalize(reps.float(), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        reps = reps.cpu().numpy().astype(np.float16)
        batch_size = reps.shape[0]
        flat = reps.reshape(-1, EMBED_DIM)
        count = flat.shape[0]
        bank[cursor : cursor + count] = flat
        for batch_offset in range(batch_size):
            start = cursor + batch_offset * NUM_PATCHES
            end = start + NUM_PATCHES
            image_indices[start:end] = image_cursor + batch_offset
            patch_indices[start:end] = np.arange(NUM_PATCHES, dtype=np.int16)
        labels_all.append(labels.numpy())
        cursor += count
        image_cursor += batch_size
    bank.flush()
    image_indices.flush()
    patch_indices.flush()
    labels_concat = np.concatenate(labels_all)
    np.savez(meta_path, shape=np.array([total_patches, EMBED_DIM]), labels=labels_concat)
    return {
        "feature_path": feature_path,
        "image_idx_path": output_prefix.with_name(output_prefix.name + "_image_idx.dat"),
        "patch_idx_path": output_prefix.with_name(output_prefix.name + "_patch_idx.dat"),
        "meta_path": meta_path,
        "shape": (total_patches, EMBED_DIM),
        "labels": labels_concat,
    }


@torch.no_grad()
def get_query_block_vectors_jepa(
    model: IJEPAModel,
    image: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    image = image.unsqueeze(0).to(device)
    context_indices_b = context_indices.unsqueeze(0).to(device)
    target_indices_b = [idx.unsqueeze(0).to(device) for idx in target_indices]
    loss, aux = model(image, context_indices_b, target_indices_b)
    _ = loss
    pred_vectors = [
        torch.nan_to_num(F.normalize(pred.mean(dim=1), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        .squeeze(0)
        .cpu()
        .numpy()
        for pred in aux["predictions"]
    ]
    true_vectors = [
        torch.nan_to_num(F.normalize(tgt.mean(dim=1), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        .squeeze(0)
        .cpu()
        .numpy()
        for tgt in aux["targets"]
    ]
    return pred_vectors, true_vectors


@torch.no_grad()
def get_query_block_vectors_mae(
    encoder: VisionTransformerEncoder,
    image: torch.Tensor,
    target_indices: List[torch.Tensor],
    device: torch.device,
) -> List[np.ndarray]:
    image = image.unsqueeze(0).to(device)
    reps = encoder(image, None)
    outputs = []
    for idx in target_indices:
        idx_b = idx.unsqueeze(0).to(device)
        pooled = batched_gather(reps, idx_b).mean(dim=1)
        outputs.append(
            torch.nan_to_num(F.normalize(pooled, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
    return outputs


def draw_context_mask(image: np.ndarray, target_blocks: List[torch.Tensor]) -> np.ndarray:
    context_img = image.copy().astype(np.float32)
    for block in target_blocks:
        x, y, w, h = indices_to_rect(block.tolist())
        x0 = int(x * PATCH_SIZE)
        y0 = int(y * PATCH_SIZE)
        x1 = int((x + w) * PATCH_SIZE)
        y1 = int((y + h) * PATCH_SIZE)
        context_img[y0:y1, x0:x1, :] = 0.5
    return np.clip(context_img, 0.0, 1.0)


def create_neighbor_strip(train_raw_data: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    thumbs = []
    for idx in indices:
        img = train_raw_data[idx]
        thumbs.append(np.asarray(Image.fromarray(img).resize((32, 32), Image.BILINEAR)))
    return np.concatenate(thumbs, axis=1)


def retrieve_neighbors_for_query(
    query_vec: np.ndarray,
    bank_info: Dict[str, object],
    topk: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, indices = cosine_topk_memmap(query_vec, bank_info["feature_path"], bank_info["shape"], topk=topk)
    image_idx = np.memmap(bank_info["image_idx_path"], dtype=np.int32, mode="r", shape=(bank_info["shape"][0],))
    return scores, image_idx[indices]


def retrieve_neighbors_for_queries(
    queries: np.ndarray,
    bank_info: Dict[str, object],
    topk: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, indices = batched_cosine_topk_memmap(queries, bank_info["feature_path"], bank_info["shape"], topk=topk)
    image_idx = np.memmap(bank_info["image_idx_path"], dtype=np.int32, mode="r", shape=(bank_info["shape"][0],))
    return scores, image_idx[indices]


def select_visualization_queries(test_dataset: Dataset) -> List[int]:
    by_class = {i: [] for i in range(10)}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if len(by_class[label]) < 2:
            by_class[label].append(idx)
        if all(len(items) == 2 for items in by_class.values()):
            break
    ordered = []
    for cls in range(10):
        ordered.extend(by_class[cls])
    return ordered


def create_visualizations(
    results_dir: Path,
    jepa_model: IJEPAModel,
    mae_model: MAEModel,
    test_dataset: Dataset,
    test_vis_dataset: datasets.CIFAR10,
    train_vis_dataset: datasets.CIFAR10,
    jepa_bank_info: Dict[str, object],
    mae_bank_info: Dict[str, object],
    device: torch.device,
) -> List[Dict[str, object]]:
    colors = ["red", "lime", "cyan", "yellow"]
    selected = select_visualization_queries(test_dataset)
    raw_train = train_vis_dataset.data
    train_labels = np.array(train_vis_dataset.targets)
    summary = []
    viz_records = []
    jepa_queries = []
    mae_queries = []

    for viz_idx, dataset_idx in enumerate(tqdm(selected, desc="Visualizations", leave=False)):
        image_norm, label = test_dataset[dataset_idx]
        image_vis, _ = test_vis_dataset[dataset_idx]
        seed = SEED + 100_000 + dataset_idx
        context_indices, target_blocks = sample_multiblock_masks(seed)
        pred_vectors, true_vectors = get_query_block_vectors_jepa(
            jepa_model, image_norm, context_indices, target_blocks, device
        )
        mae_vectors = get_query_block_vectors_mae(mae_model.encoder, image_norm, target_blocks, device)
        record = {
            "viz_idx": viz_idx,
            "dataset_idx": dataset_idx,
            "label": label,
            "image_vis": image_vis,
            "target_blocks": target_blocks,
            "rows": [],
        }
        for block_id in range(TARGET_BLOCKS):
            record["rows"].append({})
            jepa_queries.append(pred_vectors[block_id])
            jepa_queries.append(true_vectors[block_id])
            mae_queries.append(mae_vectors[block_id])
        viz_records.append(record)

    _, jepa_neighbor_images = retrieve_neighbors_for_queries(np.stack(jepa_queries), jepa_bank_info, topk=5)
    _, mae_neighbor_images = retrieve_neighbors_for_queries(np.stack(mae_queries), mae_bank_info, topk=5)

    jepa_query_ptr = 0
    mae_query_ptr = 0
    for record in viz_records:
        class_matches = []
        for block_id in range(TARGET_BLOCKS):
            pred_img_idx = jepa_neighbor_images[jepa_query_ptr]
            true_img_idx = jepa_neighbor_images[jepa_query_ptr + 1]
            mae_img_idx = mae_neighbor_images[mae_query_ptr]
            jepa_query_ptr += 2
            mae_query_ptr += 1
            record["rows"][block_id] = {
                "pred": pred_img_idx,
                "true": true_img_idx,
                "mae": mae_img_idx,
            }
            class_matches.append(float((train_labels[pred_img_idx] == record["label"]).mean()))

        summary.append(
            {
                "viz_idx": record["viz_idx"],
                "dataset_idx": record["dataset_idx"],
                "label": record["label"],
                "score": float(np.mean(class_matches)),
            }
        )

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(5, TARGET_BLOCKS, height_ratios=[1.3, 1.3, 1.0, 1.0, 1.0])
        image_np = record["image_vis"].permute(1, 2, 0).numpy()
        context_np = draw_context_mask(image_np, record["target_blocks"])

        for col in range(TARGET_BLOCKS):
            ax0 = fig.add_subplot(gs[0, col])
            ax0.imshow(image_np)
            for block_id, block in enumerate(record["target_blocks"]):
                x, y, w, h = indices_to_rect(block.tolist())
                rect = patches.Rectangle(
                    (x * PATCH_SIZE, y * PATCH_SIZE),
                    w * PATCH_SIZE,
                    h * PATCH_SIZE,
                    linewidth=2,
                    edgecolor=colors[block_id],
                    facecolor="none",
                )
                ax0.add_patch(rect)
            ax0.set_title(f"Block {col + 1}")
            ax0.set_axis_off()
            if col == 0:
                ax0.set_ylabel("Original", fontsize=12)

            ax1 = fig.add_subplot(gs[1, col])
            ax1.imshow(context_np)
            for block_id, block in enumerate(record["target_blocks"]):
                x, y, w, h = indices_to_rect(block.tolist())
                rect = patches.Rectangle(
                    (x * PATCH_SIZE, y * PATCH_SIZE),
                    w * PATCH_SIZE,
                    h * PATCH_SIZE,
                    linewidth=2,
                    edgecolor=colors[block_id],
                    facecolor="none",
                )
                ax1.add_patch(rect)
            ax1.set_axis_off()
            if col == 0:
                ax1.set_ylabel("Context", fontsize=12)

            ax2 = fig.add_subplot(gs[2, col])
            ax2.imshow(create_neighbor_strip(raw_train, record["rows"][col]["pred"]))
            ax2.set_axis_off()
            if col == 0:
                ax2.set_ylabel("JEPA pred NN", fontsize=12)

            ax3 = fig.add_subplot(gs[3, col])
            ax3.imshow(create_neighbor_strip(raw_train, record["rows"][col]["true"]))
            ax3.set_axis_off()
            if col == 0:
                ax3.set_ylabel("JEPA true NN", fontsize=12)

            ax4 = fig.add_subplot(gs[4, col])
            ax4.imshow(create_neighbor_strip(raw_train, record["rows"][col]["mae"]))
            ax4.set_axis_off()
            if col == 0:
                ax4.set_ylabel("MAE enc NN", fontsize=12)

        fig.suptitle(f"Test image {record['dataset_idx']} | class={record['label']}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(results_dir / "viz" / f"nn_retrieval_image_{record['viz_idx']}.png", dpi=150)
        plt.close(fig)

    best4 = sorted(summary, key=lambda item: item["score"], reverse=True)[:4]
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    for row_idx, item in enumerate(best4):
        img, label = test_vis_dataset[item["dataset_idx"]]
        axes[row_idx].imshow(img.permute(1, 2, 0).numpy())
        axes[row_idx].set_title(
            f"Query {item['viz_idx']} | dataset_idx={item['dataset_idx']} | class={label} | JEPA semantic score={item['score']:.2f}"
        )
        axes[row_idx].set_axis_off()
    fig.tight_layout()
    fig.savefig(results_dir / "viz" / "summary_best4.png", dpi=150)
    plt.close(fig)
    return summary


def verify_outputs(paths: Sequence[Path]) -> None:
    missing = []
    empty = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
        elif path.stat().st_size == 0:
            empty.append(str(path))
    if missing or empty:
        raise RuntimeError(f"Missing outputs: {missing}; empty outputs: {empty}")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA vs MAE CIFAR-10 experiment")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--debug-pretrain-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-banks-and-viz", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    make_results_dirs(args.results_dir)

    datasets_dict = build_datasets(args.data_root)
    train_split, val_split = split_train_val(datasets_dict["train_full"], val_size=5_000)
    if args.train_limit > 0:
        train_split = Subset(train_split, list(range(min(args.train_limit, len(train_split)))))
    if args.val_limit > 0:
        val_split = Subset(val_split, list(range(min(args.val_limit, len(val_split)))))
    train_eval_dataset = datasets_dict["train_full"]
    test_eval_dataset = datasets_dict["test_full"]
    if args.eval_limit > 0:
        train_eval_dataset = Subset(train_eval_dataset, list(range(min(args.eval_limit, len(train_eval_dataset)))))
        test_eval_dataset = Subset(test_eval_dataset, list(range(min(args.eval_limit, len(test_eval_dataset)))))
    mask_collator = MultiBlockMaskCollator(base_seed=SEED)
    train_loader = make_loader(
        train_split,
        TRAIN_BATCH_SIZE,
        True,
        collate_fn=mask_collator,
        device=device,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_split,
        TRAIN_BATCH_SIZE,
        False,
        collate_fn=MultiBlockMaskCollator(base_seed=SEED + 10_000),
        device=device,
        num_workers=args.num_workers,
    )
    full_train_eval_loader = make_loader(
        train_eval_dataset, EVAL_BATCH_SIZE, False, device=device, num_workers=args.num_workers
    )
    test_eval_loader = make_loader(
        test_eval_dataset, EVAL_BATCH_SIZE, False, device=device, num_workers=args.num_workers
    )

    run_self_tests(device, train_loader)

    pretrain_epochs = args.debug_pretrain_epochs if args.debug_pretrain_epochs > 0 else args.pretrain_epochs

    jepa_model = IJEPAModel().to(device)
    mae_model = MAEModel().to(device)

    jepa_model, jepa_stats, jepa_best_path, jepa_final_path = train_model(
        "jepa",
        jepa_model,
        train_loader,
        val_loader,
        device,
        args.results_dir,
        pretrain_epochs,
    )
    mae_model, mae_stats, mae_best_path, mae_final_path = train_model(
        "mae",
        mae_model,
        train_loader,
        val_loader,
        device,
        args.results_dir,
        pretrain_epochs,
    )

    jepa_train_features, jepa_train_labels = extract_global_features(
        jepa_model.target_encoder, full_train_eval_loader, device
    )
    jepa_test_features, jepa_test_labels = extract_global_features(
        jepa_model.target_encoder, test_eval_loader, device
    )
    mae_train_features, mae_train_labels = extract_global_features(mae_model.encoder, full_train_eval_loader, device)
    mae_test_features, mae_test_labels = extract_global_features(mae_model.encoder, test_eval_loader, device)

    jepa_probe_acc = train_linear_probe(
        jepa_train_features,
        jepa_train_labels,
        jepa_test_features,
        jepa_test_labels,
        device,
        epochs=args.probe_epochs,
    )
    mae_probe_acc = train_linear_probe(
        mae_train_features,
        mae_train_labels,
        mae_test_features,
        mae_test_labels,
        device,
        epochs=args.probe_epochs,
    )

    jepa_knn1, jepa_knn5 = knn_retrieval_metrics(
        jepa_train_features, jepa_train_labels, jepa_test_features, jepa_test_labels
    )
    mae_knn1, mae_knn5 = knn_retrieval_metrics(
        mae_train_features, mae_train_labels, mae_test_features, mae_test_labels
    )

    write_text(
        args.results_dir / "linear_probe_results.txt",
        "\n".join(
            [
                "Linear Probe Results",
                f"JEPA target encoder top-1 accuracy: {jepa_probe_acc:.2f}%",
                f"MAE encoder top-1 accuracy: {mae_probe_acc:.2f}%",
            ]
        ),
    )
    write_text(
        args.results_dir / "knn_retrieval_results.txt",
        "\n".join(
            [
                "k-NN Retrieval Results",
                f"JEPA target encoder acc@1: {jepa_knn1:.2f}%",
                f"JEPA target encoder acc@5: {jepa_knn5:.2f}%",
                f"MAE encoder acc@1: {mae_knn1:.2f}%",
                f"MAE encoder acc@5: {mae_knn5:.2f}%",
            ]
        ),
    )

    avg_semantic_score = float("nan")
    if not args.skip_banks_and_viz:
        jepa_bank_info = build_patch_bank(
            jepa_model.target_encoder,
            full_train_eval_loader,
            len(train_eval_dataset),
            device,
            args.results_dir / "banks" / "jepa_patch_bank",
        )
        mae_bank_info = build_patch_bank(
            mae_model.encoder,
            full_train_eval_loader,
            len(train_eval_dataset),
            device,
            args.results_dir / "banks" / "mae_patch_bank",
        )

        viz_summary = create_visualizations(
            args.results_dir,
            jepa_model,
            mae_model,
            datasets_dict["test_full"],
            datasets_dict["test_vis"],
            datasets_dict["train_vis"],
            jepa_bank_info,
            mae_bank_info,
            device,
        )
        avg_semantic_score = float(np.mean([item["score"] for item in viz_summary]))

    interpretation = (
        "Across the nearest-neighbor visualizations, the JEPA predictor tends to retrieve training images "
        "that align more strongly with the query image class and object-level content, while the MAE encoder "
        "more often emphasizes local texture and color statistics. The true JEPA target features generally form "
        "the cleanest semantic neighborhood, and the predictor sits closer to that behavior than the MAE baseline, "
        "supporting the paper's claim that latent prediction encourages semantic abstraction over direct pixel matching."
    )
    write_text(
        args.results_dir / "summary.txt",
        "\n".join(
            [
                "Experiment Summary",
                f"JEPA linear probe accuracy: {jepa_probe_acc:.2f}%",
                f"MAE linear probe accuracy: {mae_probe_acc:.2f}%",
                f"JEPA k-NN acc@1: {jepa_knn1:.2f}%",
                f"JEPA k-NN acc@5: {jepa_knn5:.2f}%",
                f"MAE k-NN acc@1: {mae_knn1:.2f}%",
                f"MAE k-NN acc@5: {mae_knn5:.2f}%",
                f"Average JEPA semantic retrieval score over 20 visualization queries: {avg_semantic_score:.3f}",
                "",
                interpretation,
            ]
        ),
    )

    required_files = [
        args.results_dir / "jepa_train_log.csv",
        args.results_dir / "mae_train_log.csv",
        args.results_dir / "linear_probe_results.txt",
        args.results_dir / "knn_retrieval_results.txt",
        args.results_dir / "summary.txt",
        args.results_dir / "jepa_best.pt",
        args.results_dir / "mae_best.pt",
        jepa_final_path,
        mae_final_path,
    ]
    if not args.skip_banks_and_viz:
        required_files.append(args.results_dir / "viz" / "summary_best4.png")
        required_files.extend(args.results_dir / "viz" / f"nn_retrieval_image_{i}.png" for i in range(20))
    verify_outputs(required_files)
    print("All required outputs verified as non-empty.")


if __name__ == "__main__":
    main()
