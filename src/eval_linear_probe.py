"""Freeze a pretrained encoder and train a linear classifier on top of its embeddings."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import (
    accuracy,
    detect_device,
    extract_embeddings,
    get_eval_output_dir,
    plot_curves,
    prepare_config,
    save_csv,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like evaluation.linear_probe_epochs=50.",
    )
    return parser.parse_args()


def build_tensor_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap precomputed features in a dataloader."""
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_head(
    head: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Evaluate linear probe loss and accuracy."""
    criterion = torch.nn.CrossEntropyLoss()
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        logits = head(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * features.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += features.size(0)
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    device = detect_device(config.get("device", "auto"))
    set_seed(int(config["seed"]))

    output_dir = get_eval_output_dir(args.checkpoint, "linear_probe", args.output_dir)
    datasets = build_datasets(config)
    batch_size = int(config["evaluation"]["batch_size"])
    encoder = get_encoder(args.checkpoint, device=device, eval_mode=True)

    probe_train_loader = build_dataloader(
        datasets.probe_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    probe_val_loader = build_dataloader(
        datasets.probe_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    test_loader = build_dataloader(
        datasets.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    train_bundle = extract_embeddings(encoder, probe_train_loader, device=device)
    val_bundle = extract_embeddings(encoder, probe_val_loader, device=device)
    test_bundle = extract_embeddings(encoder, test_loader, device=device)

    torch.save(train_bundle, output_dir / "train_embeddings.pt")
    torch.save(val_bundle, output_dir / "val_embeddings.pt")
    torch.save(test_bundle, output_dir / "test_embeddings.pt")

    train_features = train_bundle["embeddings"]
    train_labels = train_bundle["labels"]
    val_features = val_bundle["embeddings"]
    val_labels = val_bundle["labels"]
    test_features = test_bundle["embeddings"]
    test_labels = test_bundle["labels"]

    probe_batch_size = int(config["evaluation"]["linear_probe_batch_size"])
    train_loader = build_tensor_loader(train_features, train_labels, probe_batch_size, shuffle=True)
    val_loader = build_tensor_loader(val_features, val_labels, probe_batch_size, shuffle=False)
    heldout_test_loader = build_tensor_loader(test_features, test_labels, probe_batch_size, shuffle=False)

    head = torch.nn.Linear(train_features.size(1), datasets.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["evaluation"]["linear_probe_lr"]),
        weight_decay=float(config["evaluation"]["linear_probe_weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_state = copy.deepcopy(head.state_dict())
    best_val_acc = -1.0

    for epoch in range(1, int(config["evaluation"]["linear_probe_epochs"]) + 1):
        head.train()
        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_acc += accuracy(logits.detach(), labels) * features.size(0)
            total_examples += features.size(0)

        train_loss = total_loss / total_examples
        train_acc = total_acc / total_examples
        val_loss, val_acc = evaluate_head(head, val_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    test_loss, test_acc = evaluate_head(head, heldout_test_loader, device=device)
    torch.save(best_state, output_dir / "best_linear_probe.pt")
    save_csv(history, output_dir / "linear_probe_history.csv")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
        },
        output_dir / "linear_probe_metrics.json",
    )
    plot_curves(
        [
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
            }
            for row in history
        ],
        output_dir / "linear_probe_losses.png",
        "Linear Probe Loss",
        "Loss",
    )
    plot_curves(
        [
            {
                "epoch": row["epoch"],
                "train_acc": row["train_acc"],
                "val_acc": row["val_acc"],
            }
            for row in history
        ],
        output_dir / "linear_probe_accuracy.png",
        "Linear Probe Accuracy",
        "Accuracy",
    )

    print(
        f"Linear probe complete. best_val_acc={best_val_acc:.4f} "
        f"test_acc={test_acc:.4f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
