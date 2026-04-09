"""Visual JEPA-vs-MAE comparison with robotics-style image perturbations.

This is not a real robot benchmark. It is a lightweight visual sanity test that
asks whether two frozen encoders are stable under common robot-camera nuisances:
occlusion, blur, sensor noise, lighting shift, and small viewpoint rotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from src.data import build_dataloader, build_datasets
from src.models import get_encoder
from src.utils import detect_device, ensure_dir, extract_embeddings, prepare_config, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-checkpoint", type=str, required=True, help="Path to JEPA checkpoint.")
    parser.add_argument("--mae-checkpoint", type=str, required=True, help="Path to MAE checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default="runs/visual_robotics_test")
    parser.add_argument("--num-images", type=int, default=256, help="Number of test images for perturbation scores.")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of visual query examples.")
    return parser.parse_args()


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor in [0, 1] to a HWC NumPy image."""
    image = image.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def occlude(images: torch.Tensor) -> torch.Tensor:
    """Apply a centered black square occlusion."""
    out = images.clone()
    _, _, height, width = out.shape
    patch = max(4, height // 4)
    y0 = height // 2 - patch // 2
    x0 = width // 2 - patch // 2
    out[:, :, y0 : y0 + patch, x0 : x0 + patch] = 0.0
    return out


def add_noise(images: torch.Tensor) -> torch.Tensor:
    """Add deterministic Gaussian-like sensor noise."""
    generator = torch.Generator(device=images.device).manual_seed(123)
    noise = torch.randn(images.shape, generator=generator, device=images.device) * 0.12
    return (images + noise).clamp(0, 1)


def darken(images: torch.Tensor) -> torch.Tensor:
    """Simulate a lighting drop."""
    return (images * 0.45).clamp(0, 1)


def blur(images: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur."""
    return TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[1.5, 1.5])


def rotate(images: torch.Tensor) -> torch.Tensor:
    """Apply a small camera/viewpoint rotation."""
    return TF.rotate(images, angle=12, interpolation=TF.InterpolationMode.BILINEAR)


PERTURBATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "occlusion": occlude,
    "noise": add_noise,
    "dark": darken,
    "blur": blur,
    "rotation": rotate,
}


@torch.no_grad()
def encode(encoder: torch.nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    """Encode a batch of images and return normalized CPU embeddings."""
    embeddings = encoder.encode(images.to(device)).detach().cpu().float()
    return F.normalize(embeddings, dim=1)


def collect_images(config: Dict, num_images: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a small deterministic slice of the test set."""
    datasets = build_datasets(config)
    dataloader = build_dataloader(
        datasets.test,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    images = []
    labels = []
    for batch_images, batch_labels, _ in dataloader:
        images.append(batch_images)
        labels.append(batch_labels)
        if sum(chunk.size(0) for chunk in images) >= num_images:
            break
    return torch.cat(images, dim=0)[:num_images], torch.cat(labels, dim=0)[:num_images]


def perturbation_scores(
    name: str,
    encoder: torch.nn.Module,
    images: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    """Compute embedding cosine similarity under each perturbation."""
    original = encode(encoder, images, device)
    row = {"model": name}
    for perturbation_name, transform in PERTURBATIONS.items():
        changed = encode(encoder, transform(images), device)
        row[f"{perturbation_name}_cosine"] = (original * changed).sum(dim=1).mean().item()
    row["mean_cosine"] = float(np.mean([row[f"{key}_cosine"] for key in PERTURBATIONS]))
    return row


def plot_perturbation_scores(rows: List[Dict[str, float]], path: Path) -> None:
    """Plot stability scores for both encoders."""
    labels = list(PERTURBATIONS.keys()) + ["mean"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    for offset, row in zip([-width / 2, width / 2], rows):
        values = [row[f"{label}_cosine"] for label in PERTURBATIONS] + [row["mean_cosine"]]
        plt.bar(x + offset, values, width=width, label=row["model"])
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Cosine similarity original vs perturbed")
    plt.title("Robotics-Style Perturbation Stability")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def nearest_neighbors(
    encoder: torch.nn.Module,
    dataloader,
    device: str,
    query_indices: List[int],
) -> Tuple[Dict[int, int], Dict[str, torch.Tensor]]:
    """Compute nearest neighbor indices for selected query positions."""
    bundle = extract_embeddings(encoder, dataloader, device)
    embeddings = F.normalize(bundle["embeddings"].float(), dim=1)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-1e9)
    nn_positions = similarity.argmax(dim=1)
    return {idx: nn_positions[idx].item() for idx in query_indices}, bundle


def plot_neighbor_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    jepa_neighbors: Dict[int, int],
    mae_neighbors: Dict[int, int],
    output_path: Path,
) -> None:
    """Save a grid of query, JEPA nearest neighbor, and MAE nearest neighbor."""
    query_indices = list(jepa_neighbors.keys())
    figure, axes = plt.subplots(len(query_indices), 3, figsize=(8, 2.2 * len(query_indices)))
    if len(query_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, query_idx in enumerate(query_indices):
        jepa_idx = jepa_neighbors[query_idx]
        mae_idx = mae_neighbors[query_idx]
        entries = [
            ("Query", query_idx),
            ("JEPA NN", jepa_idx),
            ("MAE NN", mae_idx),
        ]
        for col_idx, (title, image_idx) in enumerate(entries):
            axis = axes[row_idx, col_idx]
            axis.imshow(tensor_to_image(images[image_idx]))
            axis.set_title(f"{title}\nlabel={int(labels[image_idx])}")
            axis.axis("off")

    figure.suptitle("Nearest-Neighbor Visual Comparison", y=1.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_perturbation_grid(images: torch.Tensor, labels: torch.Tensor, output_path: Path) -> None:
    """Save a small grid showing the perturbations used in this test."""
    image = images[0:1]
    variants = [("original", image)] + [(name, transform(image)) for name, transform in PERTURBATIONS.items()]
    figure, axes = plt.subplots(1, len(variants), figsize=(2.2 * len(variants), 2.4))
    for axis, (name, variant) in zip(axes, variants):
        axis.imshow(tensor_to_image(variant[0]))
        axis.set_title(name)
        axis.axis("off")
    figure.suptitle(f"Robotics-Style Perturbations, label={int(labels[0])}", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the visual robotics-style comparison."""
    args = parse_args()
    config = prepare_config(args.config)
    set_seed(int(config["seed"]))
    device = detect_device(config.get("device", "auto"))
    output_dir = ensure_dir(args.output_dir)

    images, labels = collect_images(config, args.num_images)
    jepa_encoder = get_encoder(args.jepa_checkpoint, device=device, eval_mode=True)
    mae_encoder = get_encoder(args.mae_checkpoint, device=device, eval_mode=True)

    rows = [
        perturbation_scores("JEPA", jepa_encoder, images, device),
        perturbation_scores("MAE", mae_encoder, images, device),
    ]
    with (output_dir / "perturbation_stability.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    plot_perturbation_scores(rows, output_dir / "perturbation_stability.png")
    plot_perturbation_grid(images, labels, output_dir / "perturbation_examples.png")

    datasets = build_datasets(config)
    dataloader = build_dataloader(
        datasets.test,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    query_indices = list(range(min(args.num_examples, args.num_images)))
    jepa_neighbors, _ = nearest_neighbors(jepa_encoder, dataloader, device, query_indices)
    mae_neighbors, _ = nearest_neighbors(mae_encoder, dataloader, device, query_indices)
    plot_neighbor_grid(
        images,
        labels,
        jepa_neighbors,
        mae_neighbors,
        output_dir / "nearest_neighbor_comparison.png",
    )

    print(f"Saved robotics-style visual comparison to {output_dir}")
    print(json.dumps({"perturbation_stability": rows}, indent=2))


if __name__ == "__main__":
    main()

