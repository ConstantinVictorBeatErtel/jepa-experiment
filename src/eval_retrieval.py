"""Evaluate nearest-neighbor retrieval quality of pretrained image embeddings."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import (
    detect_device,
    extract_embeddings,
    get_eval_output_dir,
    prepare_config,
    resolve_split_name,
    save_csv,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="test", help="Split: train, val, or test.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like evaluation.retrieval_k=10.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    device = detect_device(config.get("device", "auto"))
    set_seed(int(config["seed"]))

    output_dir = get_eval_output_dir(args.checkpoint, "retrieval", args.output_dir)
    split_name = resolve_split_name(args.split)
    datasets = build_datasets(config)
    dataset = getattr(datasets, split_name)
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    encoder = get_encoder(args.checkpoint, device=device, eval_mode=True)
    bundle = extract_embeddings(encoder, dataloader, device=device)
    embeddings = F.normalize(bundle["embeddings"], dim=1)
    labels = bundle["labels"]
    indices = bundle["indices"]

    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-1e9)
    retrieval_k = int(config["evaluation"]["retrieval_k"])
    top1 = similarity.argmax(dim=1)
    topk = similarity.topk(k=retrieval_k, dim=1).indices

    acc_at_1 = (labels[top1] == labels).float().mean().item()
    acc_at_k = (labels[topk] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    example_rows = []
    for row in range(min(25, len(labels))):
        nn_idx = top1[row].item()
        example_rows.append(
            {
                "query_dataset_index": int(indices[row].item()),
                "query_label": int(labels[row].item()),
                "neighbor_dataset_index": int(indices[nn_idx].item()),
                "neighbor_label": int(labels[nn_idx].item()),
                "match": bool(labels[row].item() == labels[nn_idx].item()),
            }
        )

    save_csv(example_rows, output_dir / "retrieval_examples.csv")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "split": split_name,
            "acc_at_1": acc_at_1,
            "acc_at_k": acc_at_k,
            "retrieval_k": retrieval_k,
        },
        output_dir / "retrieval_metrics.json",
    )
    torch.save(bundle, output_dir / f"{split_name}_embeddings.pt")

    print(
        f"Retrieval complete. split={split_name} acc@1={acc_at_1:.4f} "
        f"acc@{retrieval_k}={acc_at_k:.4f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

