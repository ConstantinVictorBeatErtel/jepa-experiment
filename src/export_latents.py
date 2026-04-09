"""Export frozen encoder embeddings for later temporal modeling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import (
    detect_device,
    extract_embeddings,
    get_eval_output_dir,
    load_checkpoint,
    prepare_config,
    resolve_split_name,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="test", help="Split: train, val, test, or pretrain.")
    parser.add_argument("--output-path", type=str, default=None, help="Optional .pt output path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like dataset.name=stl10.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    device = detect_device(config.get("device", "auto"))
    set_seed(int(config["seed"]))

    split_name = resolve_split_name(args.split)
    export_dir = get_eval_output_dir(args.checkpoint, "latents")
    output_path = Path(args.output_path) if args.output_path else export_dir / f"latents_{split_name}.pt"

    datasets = build_datasets(config)
    dataset = getattr(datasets, split_name)
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    encoder = get_encoder(args.checkpoint, device=device, eval_mode=True)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    bundle = extract_embeddings(encoder, dataloader, device=device)

    payload = {
        "embeddings": bundle["embeddings"],
        "labels": bundle["labels"],
        "indices": bundle["indices"],
        "metadata": {
            "checkpoint": args.checkpoint,
            "objective": checkpoint.get("objective", "unknown"),
            "dataset": config["dataset"]["name"],
            "split": split_name,
            "encoder_config": checkpoint["encoder_config"],
        },
    }
    torch.save(payload, output_path)
    save_json(payload["metadata"], output_path.with_suffix(".json"))
    print(f"Exported latents to {output_path}")


if __name__ == "__main__":
    main()
