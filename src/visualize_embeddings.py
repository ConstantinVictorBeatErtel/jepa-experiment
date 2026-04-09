"""Visualize pretrained embeddings with PCA and t-SNE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import (
    detect_device,
    extract_embeddings,
    load_checkpoint,
    prepare_config,
    resolve_split_name,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path. Pass multiple times to compare JEPA vs MAE.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="test", help="Split: train, val, or test.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values like evaluation.visualization_max_points=1000.",
    )
    return parser.parse_args()


def subsample(embeddings: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    """Randomly subsample embeddings for faster visualization."""
    if len(embeddings) <= max_points:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(embeddings), size=max_points, replace=False)
    return embeddings[chosen], labels[chosen]


def prepare_embeddings_for_viz(embeddings: np.ndarray) -> np.ndarray:
    """Clean and standardize embeddings before PCA/t-SNE visualization."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    return embeddings.astype(np.float32)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = prepare_config(args.config, args.override)
    device = detect_device(config.get("device", "auto"))
    set_seed(int(config["seed"]))
    split_name = resolve_split_name(args.split)

    if args.output_dir is None:
        run_dir = Path(args.checkpoint[0]).resolve().parents[1]
        output_dir = run_dir / "embedding_viz"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(config)
    dataset = getattr(datasets, split_name)
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    checkpoints: List[str] = args.checkpoint
    num_rows = len(checkpoints)
    figure, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])

    metadata = []
    for row_index, checkpoint_path in enumerate(checkpoints):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        encoder = get_encoder(checkpoint_path, device=device, eval_mode=True)
        bundle = extract_embeddings(encoder, dataloader, device=device)
        embeddings = bundle["embeddings"].numpy()
        labels = bundle["labels"].numpy()
        embeddings, labels = subsample(
            embeddings,
            labels,
            max_points=int(config["evaluation"]["visualization_max_points"]),
            seed=int(config["seed"]),
        )
        embeddings = prepare_embeddings_for_viz(embeddings)

        pca_projection = PCA(n_components=2, svd_solver="full").fit_transform(embeddings)
        tsne_projection = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            perplexity=min(30, max(5, len(embeddings) // 20)),
            random_state=int(config["seed"]),
        ).fit_transform(embeddings)

        label_name = f"{checkpoint.get('objective', 'model').upper()} | {Path(checkpoint_path).stem}"
        for column_index, (projection, title) in enumerate(
            [(pca_projection, "PCA"), (tsne_projection, "t-SNE")]
        ):
            axis = axes[row_index, column_index]
            scatter = axis.scatter(
                projection[:, 0],
                projection[:, 1],
                c=labels,
                cmap="tab10",
                s=10,
                alpha=0.75,
            )
            axis.set_title(f"{label_name} - {title}")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0 and column_index == 1:
                figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)

        metadata.append(
            {
                "checkpoint": checkpoint_path,
                "objective": checkpoint.get("objective", "unknown"),
                "num_points": int(len(embeddings)),
                "split": split_name,
            }
        )

    figure.tight_layout()
    plot_path = output_dir / "embedding_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    save_json({"plots": str(plot_path), "models": metadata}, output_dir / "embedding_viz_manifest.json")
    print(f"Saved embedding visualization to {plot_path}")


if __name__ == "__main__":
    main()
