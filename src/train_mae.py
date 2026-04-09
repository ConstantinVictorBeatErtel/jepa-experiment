"""Train a tiny masked-patch reconstruction baseline on CIFAR-10 or STL-10."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from src.data import build_dataloader, build_datasets
from src.losses import masked_patch_mse_loss
from src.models import build_mae_model
from src.utils import (
    count_parameters,
    create_run_dir,
    detect_device,
    get_autocast_context,
    plot_curves,
    plot_mask_ablation,
    prepare_config,
    save_checkpoint,
    save_csv,
    save_json,
    save_yaml,
    set_seed,
    use_mixed_precision,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like training.epochs=10. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Optional mask-ratio ablation. Example: --mask-ratios 0.3 0.6 0.8",
    )
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    """Warmup + cosine decay scheduler."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_run(config: Dict, mask_ratio: float) -> Dict:
    """Train one masked reconstruction run."""
    run_dir = create_run_dir(config, objective="mae", mask_ratio=mask_ratio)
    device = detect_device(config.get("device", "auto"))
    set_seed(int(config["seed"]))

    datasets = build_datasets(config)
    config["dataset"]["channels"] = datasets.channels
    train_loader = build_dataloader(
        datasets.pretrain,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        drop_last=True,
    )

    model = build_mae_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        total_epochs=int(config["training"]["epochs"]),
    )
    amp_enabled = use_mixed_precision(device, bool(config["training"]["mixed_precision"]))
    scaler = GradScaler(enabled=amp_enabled)

    save_yaml(config, run_dir / "resolved_config.yaml")
    summary = {
        "objective": "mae",
        "device": device,
        "mask_ratio": mask_ratio,
        "num_parameters": count_parameters(model),
    }
    save_json(summary, run_dir / "run_summary.json")

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        progress_bar = tqdm(train_loader, desc=f"MAE Epoch {epoch}", leave=False)
        for images, _, _ in progress_bar:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with get_autocast_context(device, amp_enabled):
                outputs = model(images, mask_ratio=mask_ratio)
                loss = masked_patch_mse_loss(outputs["predictions"], outputs["targets"])

            scaler.scale(loss).backward()
            if float(config["training"]["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / total_examples
        epoch_row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        save_csv(history, run_dir / "metrics.csv")
        save_json({"history": history}, run_dir / "metrics.json")
        plot_curves(
            [{"epoch": row["epoch"], "train_loss": row["train_loss"]} for row in history],
            run_dir / "plots" / "pretrain_loss.png",
            "Masked Reconstruction",
            "Loss",
        )

        checkpoint = {
            "objective": "mae",
            "model_type": "mae",
            "epoch": epoch,
            "config": copy.deepcopy(config),
            "history": history,
            "encoder_config": model.context_encoder.spec.__dict__,
            "encoder_state_dict": model.context_encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "mask_ratio": mask_ratio,
        }
        save_checkpoint(checkpoint, run_dir / "checkpoints" / "last.ckpt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            save_checkpoint(checkpoint, run_dir / "checkpoints" / "best.ckpt")

    final_checkpoint = copy.deepcopy(checkpoint)
    final_checkpoint["best_loss"] = best_loss
    final_checkpoint["best_epoch"] = best_epoch
    save_checkpoint(final_checkpoint, run_dir / "checkpoints" / "final.ckpt")
    save_json(
        {
            **summary,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / "checkpoints" / "best.ckpt"),
        },
        run_dir / "final_summary.json",
    )
    return {
        "mask_ratio": mask_ratio,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "checkpoints" / "best.ckpt"),
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    mask_ratios = args.mask_ratios or [float(config["model"]["mask_ratio"])]

    results = []
    for mask_ratio in mask_ratios:
        run_config = copy.deepcopy(config)
        run_config["model"]["mask_ratio"] = float(mask_ratio)
        results.append(train_one_run(run_config, mask_ratio=float(mask_ratio)))

    if len(results) > 1:
        runs_dir = Path(__file__).resolve().parents[1] / config["paths"]["runs_dir"]
        ablation_dir = runs_dir / f"mask_ablation_mae_{config['dataset']['name']}"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        save_csv(results, ablation_dir / "mask_ablation.csv")
        plot_mask_ablation(
            results,
            ablation_dir / "mask_ablation.png",
            title="MAE Mask Ratio Ablation",
        )
        save_json({"results": results}, ablation_dir / "mask_ablation.json")

    for result in results:
        print(
            f"[MAE] mask_ratio={result['mask_ratio']:.1f} "
            f"best_loss={result['best_loss']:.4f} checkpoint={result['checkpoint']}"
        )


if __name__ == "__main__":
    main()
