"""Utility helpers for configuration, logging, checkpoints, and reproducibility."""

from __future__ import annotations

import copy
import csv
import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def detect_device(requested: str = "auto") -> str:
    """Auto-detect the best available device."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def use_mixed_precision(device: str, enabled: bool) -> bool:
    """Enable AMP only on CUDA, where it is stable and broadly supported."""
    return bool(enabled and device == "cuda")


def get_autocast_context(device: str, enabled: bool):
    """Return a device-aware autocast context manager or a no-op context."""
    if use_mixed_precision(device, enabled):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to a YAML file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to a JSON file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Save a list of dictionaries to CSV."""
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_by_dotted_key(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value with a dotted key path."""
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Apply CLI overrides like model.mask_ratio=0.8."""
    resolved = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_key(resolved, key, value)
    return resolved


def prepare_config(config_path: str | Path, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load and resolve a config file plus optional CLI overrides."""
    config = load_yaml(config_path)
    return apply_overrides(config, overrides)


def timestamp() -> str:
    """Return a compact timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def create_run_dir(config: Dict[str, Any], objective: str, mask_ratio: float) -> Path:
    """Create a timestamped output directory under runs/."""
    runs_dir = get_project_root() / config["paths"]["runs_dir"]
    ensure_dir(runs_dir)
    dataset_name = config["dataset"]["name"]
    base_name = config.get("logging", {}).get("run_name", objective)
    run_name = f"{base_name}_{objective}_{dataset_name}_mr{mask_ratio:.1f}_{timestamp()}"
    run_dir = runs_dir / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "artifacts")
    return run_dir


def get_eval_output_dir(checkpoint_path: str | Path, name: str, output_dir: Optional[str] = None) -> Path:
    """Create a default evaluation output directory next to a checkpoint."""
    if output_dir is not None:
        return ensure_dir(output_dir)
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.resolve().parents[1]
    return ensure_dir(run_dir / name)


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    """Save a training checkpoint."""
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint from disk."""
    return torch.load(Path(path), map_location=map_location)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def move_batch_to_device(batch: Iterable[Any], device: str) -> List[Any]:
    """Move tensor-like items in a batch to the target device."""
    moved = []
    for item in batch:
        if torch.is_tensor(item):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return moved


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from logits and labels."""
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    dataloader,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Extract image embeddings, labels, and indices for a full dataloader."""
    encoder.eval()
    embeddings = []
    labels = []
    indices = []
    for images, batch_labels, batch_indices in dataloader:
        images = images.to(device)
        batch_embeddings = encoder.encode(images).detach().cpu()
        embeddings.append(batch_embeddings)
        labels.append(batch_labels.cpu())
        indices.append(torch.as_tensor(batch_indices).cpu())
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": torch.cat(labels, dim=0),
        "indices": torch.cat(indices, dim=0),
    }


def plot_curves(history: List[Dict[str, float]], path: str | Path, title: str, ylabel: str) -> None:
    """Plot one or more scalar curves from an epoch history list."""
    if not history:
        return
    keys = [key for key in history[0].keys() if key != "epoch"]
    plt.figure(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    for key in keys:
        plt.plot(epochs, [row[key] for row in history], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_mask_ablation(rows: List[Dict[str, float]], path: str | Path, title: str) -> None:
    """Plot a compact mask-ratio ablation chart."""
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: row["mask_ratio"])
    plt.figure(figsize=(6, 4))
    plt.plot(
        [row["mask_ratio"] for row in ordered],
        [row["best_loss"] for row in ordered],
        marker="o",
    )
    plt.xlabel("Mask Ratio")
    plt.ylabel("Best Pretraining Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def resolve_split_name(split: str) -> str:
    """Normalize human-friendly split aliases."""
    aliases = {
        "train": "probe_train",
        "val": "probe_val",
        "valid": "probe_val",
        "validation": "probe_val",
        "test": "test",
        "pretrain": "pretrain",
    }
    if split not in aliases:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {sorted(aliases)}.")
    return aliases[split]
