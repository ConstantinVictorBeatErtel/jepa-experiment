"""Evaluate simple class-centroid anomaly scores in frozen embedding space.

This is an exploratory bridge toward the later temporal anomaly-detection
project. It treats each class as "normal" in turn, builds a centroid for that
class in embedding space, and scores examples by distance from the centroid.
For a stricter evaluation, use a train split for centroids and test split for
scoring. For reproducing quick Colab geometry checks, the default is test/test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import (
    detect_device,
    ensure_dir,
    extract_embeddings,
    get_eval_output_dir,
    get_project_root,
    prepare_config,
    resolve_split_name,
    save_csv,
    save_json,
    set_seed,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Path to a pretrained checkpoint. Repeat to compare models.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Optional display name for each checkpoint, e.g. --name JEPA --name MAE.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--fit-split", type=str, default="test", help="Split used to fit normal centroids.")
    parser.add_argument("--eval-split", type=str, default="test", help="Split used for anomaly scoring.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like evaluation.batch_size=512.",
    )
    return parser.parse_args()


def default_output_dir(checkpoints: List[str], output_dir: str | None) -> Path:
    """Choose a stable output directory for one-model or multi-model runs."""
    if output_dir is not None:
        return ensure_dir(output_dir)
    if len(checkpoints) == 1:
        return get_eval_output_dir(checkpoints[0], "anomaly")
    return ensure_dir(get_project_root() / "runs" / f"anomaly_comparison_{timestamp()}")


def extract_split_embeddings(
    checkpoint: str,
    config: Dict,
    split: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Extract normalized embeddings for a dataset split."""
    datasets = build_datasets(config)
    dataset = getattr(datasets, resolve_split_name(split))
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    encoder = get_encoder(checkpoint, device=device, eval_mode=True)
    bundle = extract_embeddings(encoder, dataloader, device=device)
    bundle["embeddings"] = F.normalize(bundle["embeddings"].float(), dim=1)
    return bundle


def class_centroid_auc(
    model_name: str,
    fit_bundle: Dict[str, torch.Tensor],
    eval_bundle: Dict[str, torch.Tensor],
) -> List[Dict[str, float | int | str]]:
    """Compute one-vs-rest anomaly AUC for every class with fit examples."""
    fit_embeddings = fit_bundle["embeddings"]
    fit_labels = fit_bundle["labels"]
    eval_embeddings = eval_bundle["embeddings"]
    eval_labels = eval_bundle["labels"]

    rows: List[Dict[str, float | int | str]] = []
    for class_id in sorted(fit_labels.unique().tolist()):
        normal_mask = fit_labels == class_id
        if int(normal_mask.sum().item()) == 0:
            continue
        centroid = fit_embeddings[normal_mask].mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, dim=1)

        anomaly_targets = (eval_labels != class_id).numpy().astype(np.int64)
        if len(np.unique(anomaly_targets)) < 2:
            continue

        cosine_similarity = (eval_embeddings @ centroid.T).squeeze(1)
        anomaly_scores = (1.0 - cosine_similarity).numpy()
        auc = roc_auc_score(anomaly_targets, anomaly_scores)
        rows.append(
            {
                "model": model_name,
                "normal_class": int(class_id),
                "anomaly_auc": float(auc),
            }
        )
    return rows


def add_winners(rows: List[Dict[str, float | int | str]]) -> None:
    """Annotate class-wise winners when multiple models are evaluated."""
    by_class: Dict[int, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        by_class.setdefault(int(row["normal_class"]), []).append(row)

    for class_rows in by_class.values():
        if len(class_rows) < 2:
            continue
        best_auc = max(float(row["anomaly_auc"]) for row in class_rows)
        winners = [str(row["model"]) for row in class_rows if float(row["anomaly_auc"]) == best_auc]
        for row in class_rows:
            row["winner"] = ",".join(winners)


def summarize(rows: List[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    """Summarize mean AUC and class-wise wins per model."""
    models = sorted({str(row["model"]) for row in rows})
    summary = []
    for model_name in models:
        model_rows = [row for row in rows if row["model"] == model_name]
        wins = sum(str(row.get("winner", "")) == model_name for row in model_rows)
        summary.append(
            {
                "model": model_name,
                "mean_anomaly_auc": float(np.mean([float(row["anomaly_auc"]) for row in model_rows])),
                "wins": int(wins),
            }
        )
    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    set_seed(int(config["seed"]))
    device = detect_device(config.get("device", "auto"))

    checkpoints = list(args.checkpoint)
    names = args.name or [Path(path).resolve().parents[1].name for path in checkpoints]
    if len(names) != len(checkpoints):
        raise ValueError("--name must be provided the same number of times as --checkpoint.")

    output_dir = default_output_dir(checkpoints, args.output_dir)
    rows: List[Dict[str, float | int | str]] = []
    for checkpoint, model_name in zip(checkpoints, names):
        fit_bundle = extract_split_embeddings(checkpoint, config, args.fit_split, device)
        if args.fit_split == args.eval_split:
            eval_bundle = fit_bundle
        else:
            eval_bundle = extract_split_embeddings(checkpoint, config, args.eval_split, device)
        rows.extend(class_centroid_auc(model_name, fit_bundle, eval_bundle))

    add_winners(rows)
    summary = summarize(rows)

    save_csv(rows, output_dir / "anomaly_class_auc.csv")
    save_csv(summary, output_dir / "anomaly_summary.csv")
    save_json(
        {
            "device": device,
            "fit_split": args.fit_split,
            "eval_split": args.eval_split,
            "checkpoints": checkpoints,
            "summary": summary,
            "class_auc": rows,
        },
        output_dir / "anomaly_metrics.json",
    )
    print(f"Anomaly evaluation complete. output_dir={output_dir}")
    for row in summary:
        print(
            f"{row['model']}: mean_auc={float(row['mean_anomaly_auc']):.4f} "
            f"wins={int(row['wins'])}"
        )


if __name__ == "__main__":
    main()
