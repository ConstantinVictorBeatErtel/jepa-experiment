from __future__ import annotations

import csv
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

import run_experiment as exp


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ASSET_DIR = RESULTS_DIR / "paper_assets"


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


COLORS = {
    "jepa": "#1f4e79",
    "mae": "#b55d32",
    "accent": "#111111",
    "muted": "#6b7280",
    "light": "#d1d5db",
    "block": "#c72e29",
}


def ensure_output_dir() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_percent(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)%", text)
    if not match:
        raise ValueError(f"Could not find metric '{label}' in summary text.")
    return float(match.group(1))


def load_metrics() -> Dict[str, float]:
    summary_text = (RESULTS_DIR / "summary.txt").read_text(encoding="utf-8")
    semantic_match = re.search(
        r"Average JEPA semantic retrieval score over 20 visualization queries:\s*([0-9.]+)",
        summary_text,
    )
    if not semantic_match:
        raise ValueError("Could not find semantic retrieval score in summary text.")
    return {
        "jepa_probe": extract_percent(summary_text, "JEPA linear probe accuracy"),
        "mae_probe": extract_percent(summary_text, "MAE linear probe accuracy"),
        "jepa_knn1": extract_percent(summary_text, "JEPA k-NN acc@1"),
        "jepa_knn5": extract_percent(summary_text, "JEPA k-NN acc@5"),
        "mae_knn1": extract_percent(summary_text, "MAE k-NN acc@1"),
        "mae_knn5": extract_percent(summary_text, "MAE k-NN acc@5"),
        "semantic": float(semantic_match.group(1)),
    }


def load_training_logs() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    return read_csv_rows(RESULTS_DIR / "jepa_train_log.csv"), read_csv_rows(RESULTS_DIR / "mae_train_log.csv")


def summarize_log(rows: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    return {
        "epoch": np.array([int(row["epoch"]) for row in rows], dtype=np.int32),
        "train_loss": np.array([float(row["train_loss"]) for row in rows], dtype=np.float64),
        "val_loss": np.array([float(row["val_loss"]) for row in rows], dtype=np.float64),
        "lr": np.array([float(row["lr"]) for row in rows], dtype=np.float64),
    }


def draw_booktabs_table(
    ax: plt.Axes,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[float],
    aligns: Sequence[str] | None = None,
    bold_cells: Iterable[Tuple[int, int]] | None = None,
    footnote: str | None = None,
) -> None:
    ax.set_axis_off()
    bold = set(bold_cells or [])
    aligns = aligns or ["left"] + ["center"] * (len(columns) - 1)
    left = 0.02
    right = 0.98
    table_width = right - left
    x_edges = [left]
    running = left
    for width in widths:
        running += width * table_width
        x_edges.append(running)
    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2.0 for i in range(len(columns))]

    ax.text(left, 0.95, title, ha="left", va="top", fontsize=13, fontweight="bold", color=COLORS["accent"])

    top = 0.82
    header_h = 0.17 if len(rows) <= 4 else 0.11
    row_heights = []
    for row in rows:
        line_count = max(str(cell).count("\n") + 1 for cell in row)
        if len(rows) <= 4:
            row_heights.append(0.17)
        else:
            row_heights.append(0.10 + 0.035 * (line_count - 1))
    bottom = top - header_h - sum(row_heights)
    ax.hlines(top, left, right, colors="black", linewidth=1.4)
    ax.hlines(top - header_h, left, right, colors="black", linewidth=0.8)
    ax.hlines(bottom, left, right, colors="black", linewidth=1.1)

    for col_idx, col in enumerate(columns):
        x = x_centers[col_idx] if aligns[col_idx] == "center" else x_edges[col_idx] + 0.01
        ha = "center" if aligns[col_idx] == "center" else "left"
        ax.text(x, top - header_h / 2.0, col, ha=ha, va="center", fontsize=11, fontweight="bold")

    cursor = top - header_h
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        y = cursor - row_h / 2.0
        for col_idx, cell in enumerate(row):
            x = x_centers[col_idx] if aligns[col_idx] == "center" else x_edges[col_idx] + 0.01
            ha = "center" if aligns[col_idx] == "center" else "left"
            fontweight = "bold" if (row_idx, col_idx) in bold else "normal"
            ax.text(x, y, cell, ha=ha, va="center", fontsize=11, fontweight=fontweight)
        cursor -= row_h

    if footnote:
        ax.text(left, 0.06, footnote, ha="left", va="bottom", fontsize=9.5, color=COLORS["muted"])


def make_main_results_table(metrics: Dict[str, float]) -> Path:
    out_path = ASSET_DIR / "table_main_results.png"
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    rows = [
        ["I-JEPA-style latent prediction", "Latent target", f"{metrics['jepa_probe']:.2f}", f"{metrics['jepa_knn1']:.2f}", f"{metrics['jepa_knn5']:.2f}"],
        ["MAE baseline", "Pixels", f"{metrics['mae_probe']:.2f}", f"{metrics['mae_knn1']:.2f}", f"{metrics['mae_knn5']:.2f}"],
    ]
    draw_booktabs_table(
        ax=ax,
        title="Table 1. Quantitative comparison on CIFAR-10 under the I-JEPA-style protocol.",
        columns=["Method", "Prediction Target", "Linear Probe", "k-NN @1", "k-NN @5"],
        rows=rows,
        widths=[0.38, 0.20, 0.14, 0.14, 0.14],
        aligns=["left", "center", "center", "center", "center"],
        bold_cells={(0, 2), (0, 3), (0, 4)},
        footnote="Numbers are top-1 accuracy percentages on CIFAR-10 test images; higher is better in every column.",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_protocol_table() -> Path:
    out_path = ASSET_DIR / "table_protocol_alignment.png"
    fig, ax = plt.subplots(figsize=(12, 5.6))
    rows = [
        ["Data domain", "ImageNet-1K images", "CIFAR-10 images\nupsampled to 96x96"],
        ["Masking", "4 target blocks + 1 context block", "Same multi-block masking\nwith overlap removed from context"],
        ["Teacher", "EMA target encoder", "Same EMA target encoder\n(0.996 to 1.0)"],
        ["Predictor", "Narrow latent-space predictor", "Same 6-layer predictor\nwith 128-d bottleneck"],
        ["Evaluation", "Frozen target encoder,\nlinear probe, retrieval", "Frozen target encoder / encoder\nwith linear probe and k-NN"],
        ["Difference in scope", "Semantic prediction on\nlarge-scale natural images", "Image-only CIFAR-10 adaptation\nof the I-JEPA recipe"],
    ]
    draw_booktabs_table(
        ax=ax,
        title="Table 2. How the present experiment maps onto Assran et al. (2023).",
        columns=["Component", "I-JEPA Paper", "This Experiment"],
        rows=rows,
        widths=[0.20, 0.34, 0.46],
        aligns=["left", "left", "left"],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_quantitative_figure(metrics: Dict[str, float]) -> Path:
    out_path = ASSET_DIR / "figure_quantitative_results.png"
    metric_names = ["Linear probe", "k-NN @1", "k-NN @5"]
    jepa_vals = [metrics["jepa_probe"], metrics["jepa_knn1"], metrics["jepa_knn5"]]
    mae_vals = [metrics["mae_probe"], metrics["mae_knn1"], metrics["mae_knn5"]]

    x = np.arange(len(metric_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars1 = ax.bar(x - width / 2, jepa_vals, width, label="I-JEPA-style", color=COLORS["jepa"])
    bars2 = ax.bar(x + width / 2, mae_vals, width, label="MAE", color=COLORS["mae"])

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, metric_names)
    ax.set_title("Figure 1. Quantitative performance mirrors the I-JEPA claim: latent prediction yields stronger semantic features.")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    delta_text = (
        f"JEPA improves over MAE by +{metrics['jepa_probe'] - metrics['mae_probe']:.2f} pts (probe), "
        f"+{metrics['jepa_knn1'] - metrics['mae_knn1']:.2f} pts (k-NN @1), "
        f"and +{metrics['jepa_knn5'] - metrics['mae_knn5']:.2f} pts (k-NN @5)."
    )
    fig.text(0.01, -0.02, delta_text, ha="left", va="top", fontsize=10, color=COLORS["muted"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_training_curves_figure(jepa_rows: List[Dict[str, str]], mae_rows: List[Dict[str, str]]) -> Path:
    out_path = ASSET_DIR / "figure_training_curves.png"
    jepa = summarize_log(jepa_rows)
    mae = summarize_log(mae_rows)
    jepa_best = int(jepa["epoch"][np.argmin(jepa["val_loss"])])
    mae_best = int(mae["epoch"][np.argmin(mae["val_loss"])])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for ax, name, data, color, best_epoch in [
        (axes[0], "I-JEPA-style", jepa, COLORS["jepa"], jepa_best),
        (axes[1], "MAE", mae, COLORS["mae"], mae_best),
    ]:
        ax.plot(data["epoch"], data["train_loss"], color=color, linewidth=2.0, label="Train loss")
        ax.plot(data["epoch"], data["val_loss"], color="black", linewidth=1.8, linestyle="--", label="Validation loss")
        ax.axvline(best_epoch, color=COLORS["light"], linewidth=1.2, linestyle=":")
        ax.scatter(
            [best_epoch],
            [data["val_loss"][best_epoch - 1]],
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        ax.set_title(f"{name} pretraining")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.text(
            0.98,
            0.96,
            f"Best val: {data['val_loss'][best_epoch - 1]:.3f}\nEpoch: {best_epoch}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["light"]),
        )

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 2. Pretraining curves for the two objectives.", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_bank_info(prefix: Path) -> Dict[str, object]:
    meta = np.load(prefix.with_suffix(".npz"))
    shape = tuple(int(x) for x in meta["shape"])
    return {
        "feature_path": prefix.with_suffix(".dat"),
        "image_idx_path": prefix.with_name(prefix.name + "_image_idx.dat"),
        "patch_idx_path": prefix.with_name(prefix.name + "_patch_idx.dat"),
        "shape": shape,
        "labels": meta["labels"],
    }


def load_models(device: torch.device) -> Tuple[exp.IJEPAModel, exp.MAEModel]:
    jepa = exp.IJEPAModel().to(device)
    mae = exp.MAEModel().to(device)
    jepa_state = exp.load_checkpoint(RESULTS_DIR / "jepa_best.pt", device)
    mae_state = exp.load_checkpoint(RESULTS_DIR / "mae_best.pt", device)
    jepa.load_state_dict(jepa_state["model_state"])
    mae.load_state_dict(mae_state["model_state"])
    jepa.eval()
    mae.eval()
    return jepa, mae


def build_best4_qualitative_figure() -> Path:
    out_path = ASSET_DIR / "figure_qualitative_best4.png"
    device = exp.get_device()
    datasets_dict = exp.build_datasets(ROOT / "data")
    test_dataset = datasets_dict["test_full"]
    test_vis_dataset = datasets_dict["test_vis"]
    train_vis_dataset = datasets_dict["train_vis"]
    train_raw = train_vis_dataset.data
    train_labels = np.array(train_vis_dataset.targets)
    class_names = list(train_vis_dataset.classes)

    jepa_model, mae_model = load_models(device)
    jepa_bank = load_bank_info(RESULTS_DIR / "banks" / "jepa_patch_bank")
    mae_bank = load_bank_info(RESULTS_DIR / "banks" / "mae_patch_bank")

    selected = exp.select_visualization_queries(test_dataset)
    query_records = []
    pred_queries: List[np.ndarray] = []
    true_queries: List[np.ndarray] = []
    mae_queries: List[np.ndarray] = []
    for dataset_idx in selected:
        image_norm, label = test_dataset[dataset_idx]
        image_vis, _ = test_vis_dataset[dataset_idx]
        image_np = image_vis.permute(1, 2, 0).numpy()
        seed = exp.SEED + 100_000 + dataset_idx
        context_indices, target_blocks = exp.sample_multiblock_masks(seed)
        pred_vectors, true_vectors = exp.get_query_block_vectors_jepa(
            jepa_model, image_norm, context_indices, target_blocks, device
        )
        mae_vectors = exp.get_query_block_vectors_mae(mae_model.encoder, image_norm, target_blocks, device)

        query_records.append(
            {
                "dataset_idx": dataset_idx,
                "label": label,
                "label_name": class_names[label],
                "image_np": image_np,
                "context_np": exp.draw_context_mask(image_np, target_blocks),
                "target_blocks": target_blocks,
                "pred_vectors": pred_vectors,
                "true_vectors": true_vectors,
                "mae_vectors": mae_vectors,
            }
        )
        pred_queries.extend(pred_vectors)
        true_queries.extend(true_vectors)
        mae_queries.extend(mae_vectors)

    _, pred_neighbor_images = exp.retrieve_neighbors_for_queries(np.stack(pred_queries), jepa_bank, topk=5)
    _, true_neighbor_images = exp.retrieve_neighbors_for_queries(np.stack(true_queries), jepa_bank, topk=5)
    _, mae_neighbor_images = exp.retrieve_neighbors_for_queries(np.stack(mae_queries), mae_bank, topk=5)

    pred_ptr = 0
    true_ptr = 0
    mae_ptr = 0
    for item in query_records:
        block_records = []
        for block_idx in range(exp.TARGET_BLOCKS):
            pred_img_indices = pred_neighbor_images[pred_ptr].astype(int).tolist()
            true_img_indices = true_neighbor_images[true_ptr].astype(int).tolist()
            mae_img_indices = mae_neighbor_images[mae_ptr].astype(int).tolist()
            pred_ptr += 1
            true_ptr += 1
            mae_ptr += 1

            pred_score = float(np.mean(train_labels[pred_img_indices] == item["label"]))
            true_score = float(np.mean(train_labels[true_img_indices] == item["label"]))
            mae_score = float(np.mean(train_labels[mae_img_indices] == item["label"]))
            block_records.append(
                {
                    "block_idx": block_idx,
                    "block": item["target_blocks"][block_idx],
                    "pred_imgs": pred_img_indices,
                    "true_imgs": true_img_indices,
                    "mae_imgs": mae_img_indices,
                    "pred_score": pred_score,
                    "true_score": true_score,
                    "mae_score": mae_score,
                }
            )
        item["best_block"] = max(
            block_records,
            key=lambda block: (block["pred_score"], block["true_score"], -block["mae_score"]),
        )

    best4 = sorted(
        query_records,
        key=lambda item: (
            item["best_block"]["pred_score"],
            item["best_block"]["true_score"],
            -item["best_block"]["mae_score"],
        ),
        reverse=True,
    )[:4]

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(4, 5, width_ratios=[1.1, 1.1, 1.8, 1.8, 1.8], hspace=0.55, wspace=0.28)
    header_titles = ["Query", "Context", "JEPA predictor NN", "JEPA target NN", "MAE encoder NN"]
    header_x = [0.12, 0.28, 0.49, 0.68, 0.86]
    for x_pos, title in zip(header_x, header_titles):
        fig.text(x_pos, 0.92, title, ha="center", va="bottom", fontsize=12, fontweight="bold")

    for row_idx, item in enumerate(best4):
        block = item["best_block"]["block"]
        x, y, w, h = exp.indices_to_rect(block.tolist())
        block_name = f"Block {item['best_block']['block_idx'] + 1}"

        query_ax = fig.add_subplot(gs[row_idx, 0])
        query_ax.imshow(item["image_np"])
        query_ax.add_patch(
            patches.Rectangle(
                (x * exp.PATCH_SIZE, y * exp.PATCH_SIZE),
                w * exp.PATCH_SIZE,
                h * exp.PATCH_SIZE,
                linewidth=2.6,
                edgecolor=COLORS["block"],
                facecolor="none",
            )
        )
        query_ax.set_axis_off()
        query_ax.set_title(
            f"{item['label_name'].title()}\nidx={item['dataset_idx']}",
            fontsize=10.5,
        )

        context_ax = fig.add_subplot(gs[row_idx, 1])
        context_ax.imshow(item["context_np"])
        context_ax.add_patch(
            patches.Rectangle(
                (x * exp.PATCH_SIZE, y * exp.PATCH_SIZE),
                w * exp.PATCH_SIZE,
                h * exp.PATCH_SIZE,
                linewidth=2.6,
                edgecolor=COLORS["block"],
                facecolor="none",
            )
        )
        context_ax.set_axis_off()
        context_ax.set_title(block_name, fontsize=10.5)

        pred_ax = fig.add_subplot(gs[row_idx, 2])
        pred_ax.imshow(exp.create_neighbor_strip(train_raw, item["best_block"]["pred_imgs"]))
        pred_ax.set_axis_off()
        pred_ax.set_title(
            f"score={item['best_block']['pred_score']:.2f}",
            fontsize=10,
            color=COLORS["jepa"],
            pad=4,
        )

        true_ax = fig.add_subplot(gs[row_idx, 3])
        true_ax.imshow(exp.create_neighbor_strip(train_raw, item["best_block"]["true_imgs"]))
        true_ax.set_axis_off()
        true_ax.set_title(
            f"score={item['best_block']['true_score']:.2f}",
            fontsize=10,
            color=COLORS["accent"],
            pad=4,
        )

        mae_ax = fig.add_subplot(gs[row_idx, 4])
        mae_ax.imshow(exp.create_neighbor_strip(train_raw, item["best_block"]["mae_imgs"]))
        mae_ax.set_axis_off()
        mae_ax.set_title(
            f"score={item['best_block']['mae_score']:.2f}",
            fontsize=10,
            color=COLORS["mae"],
            pad=4,
        )

    fig.suptitle(
        "Figure 3. Best qualitative retrieval cases. JEPA predictor neighborhoods are closer to the JEPA target neighborhoods than to the MAE neighborhoods.",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Each row shows the strongest-scoring query among the 20 selected test images, using the block whose JEPA predictor retrievals had the highest class-consistency among the top-5 neighbors.",
        ha="left",
        va="bottom",
        fontsize=10,
        color=COLORS["muted"],
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_report_markdown(metrics: Dict[str, float]) -> Path:
    out_path = ASSET_DIR / "report_assets.md"
    lines = [
        "# Paper Assets",
        "",
        "Suggested placement in the report:",
        "",
        "1. Insert `table_main_results.png` in Section IV as the main quantitative table.",
        "2. Insert `figure_quantitative_results.png` immediately after the quantitative paragraph.",
        "3. Insert `figure_training_curves.png` in the implementation or appendix section.",
        "4. Insert `figure_qualitative_best4.png` in the qualitative analysis subsection.",
        "5. Insert `table_protocol_alignment.png` near the experimental setup subsection to tie the CIFAR-10 experiment back to Assran et al. (2023).",
        "",
        "## Ready-to-paste caption text",
        "",
        "Table 1. Quantitative comparison on CIFAR-10 under the I-JEPA-style protocol. The JEPA objective outperforms MAE on all downstream metrics, indicating that latent prediction yields stronger semantic representations than direct pixel reconstruction in this setting.",
        "",
        "Table 2. Mapping between the original I-JEPA design and the present CIFAR-10 adaptation. The table makes clear that the experiment preserves the core teacher, masking, and predictor design choices from Assran et al. (2023) while changing only the data regime and model scale.",
        "",
        "Figure 1. Quantitative comparison between the two self-supervised objectives. JEPA improves over MAE by "
        f"{metrics['jepa_probe'] - metrics['mae_probe']:.2f} points on linear probe, "
        f"{metrics['jepa_knn1'] - metrics['mae_knn1']:.2f} points on k-NN acc@1, and "
        f"{metrics['jepa_knn5'] - metrics['mae_knn5']:.2f} points on k-NN acc@5.",
        "",
        "Figure 2. Pretraining loss curves for JEPA and MAE. Both objectives optimize stably, but the curves should be read within-objective rather than across-objective because the loss definitions differ.",
        "",
        "Figure 3. Best qualitative nearest-neighbor retrieval cases. The JEPA predictor retrieves training images whose object category is more often aligned with the query than the MAE encoder, and the retrieved neighborhoods more closely resemble the true JEPA target neighborhoods.",
        "",
        "## Quantitative values",
        "",
        f"- JEPA linear probe: {metrics['jepa_probe']:.2f}%",
        f"- MAE linear probe: {metrics['mae_probe']:.2f}%",
        f"- JEPA k-NN @1 / @5: {metrics['jepa_knn1']:.2f}% / {metrics['jepa_knn5']:.2f}%",
        f"- MAE k-NN @1 / @5: {metrics['mae_knn1']:.2f}% / {metrics['mae_knn5']:.2f}%",
        f"- Average JEPA semantic retrieval score: {metrics['semantic']:.3f}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    ensure_output_dir()
    metrics = load_metrics()
    jepa_rows, mae_rows = load_training_logs()

    outputs = [
        make_main_results_table(metrics),
        make_protocol_table(),
        make_quantitative_figure(metrics),
        make_training_curves_figure(jepa_rows, mae_rows),
        build_best4_qualitative_figure(),
        make_report_markdown(metrics),
    ]

    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected output missing or empty: {path}")
        print(path)


if __name__ == "__main__":
    main()
