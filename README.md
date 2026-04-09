# JEPA Mini Project

This repo is a small, runnable PyTorch research project that compares two closely matched self-supervised vision objectives:

- `JEPA-style latent prediction`: predict masked target latents from visible context
- `Masked-patch reconstruction`: predict masked patch pixels from visible context

The backbone, masking setup, data pipeline, checkpointing, and downstream evaluation are intentionally shared as much as possible so the main difference is the learning objective.

## Why This Project Exists

The central question is:

Do latent-prediction objectives learn different or better image representations than reconstruction objectives when the encoder and masking scheme are otherwise similar?

This mini-project is also designed as stage one of a larger follow-on project:

- freeze a visual encoder
- extract latent states `z_t`
- model temporal dynamics over `z_t` with a causal transformer
- score anomalies using predicted-vs-actual latent mismatch

That is why the code emphasizes:

- a reusable encoder interface via `src.models.get_encoder`
- a shared latent extraction path used by probe, retrieval, visualization, and export
- clean checkpoint payloads that save encoder weights independently of the pretraining head
- config-driven runs for both Apple Silicon and Colab

## Repo Layout

```text
jepa_mini_project/
  README.md
  requirements.txt
  configs/
    mac.yaml
    colab.yaml
    colab_pro.yaml
    colab_pro_jepa.yaml
  src/
    __init__.py
    data.py
    patching.py
    models.py
    losses.py
    train_jepa.py
    train_mae.py
    eval_linear_probe.py
    eval_retrieval.py
    eval_anomaly.py
    visualize_embeddings.py
    visual_robotics_test.py
    export_latents.py
    utils.py
  notebooks/
    colab_runner.ipynb
    colab_pro_runner.ipynb
  scripts/
    run_mac_jepa.sh
    run_mac_mae.sh
    run_mac_probe.sh
    run_colab_jepa.sh
    run_colab_mae.sh
    run_colab_probe.sh
    run_colab_pro_jepa.sh
    run_colab_pro_mae.sh
    run_colab_jepa_sweep.sh
  runs/
    .gitkeep
```

## Method Summary

### JEPA-style model

- patchify the image
- sample visible and masked patch subsets
- encode visible patches with a tiny ViT context encoder
- encode masked patches with a stop-gradient target encoder
- predict masked target latents from context plus masked positions
- optimize cosine or normalized-MSE latent loss

Default mode is `shared` target encoder with stop-gradient. Optional EMA target mode is also supported through config.

### Reconstruction baseline

- uses the same tiny ViT encoder over visible patches
- uses the same masking scheme
- replaces latent prediction with masked pixel reconstruction
- optimizes masked-patch MSE only

## Default Experimental Setup

- dataset: `CIFAR-10`
- optional dataset: `STL-10`
- image size: `32`
- patch size: `4`
- embed dim: `128`
- transformer depth: `4`
- heads: `4`
- default mask ratio: `0.6`
- supported ablations: `0.3`, `0.6`, `0.8`

Augmentations are intentionally modest:

- random crop
- horizontal flip
- light color jitter

## Setup

### Local Apple Silicon MacBook

Create and activate an environment, then install requirements:

```bash
cd jepa_mini_project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Device selection is automatic:

- `mps` if available
- else `cuda`
- else `cpu`

Notes for Apple Silicon:

- mixed precision is disabled in `configs/mac.yaml` because MPS autocast can still be inconsistent across PyTorch versions
- batch size defaults are conservative
- `num_workers` stays low to avoid common macOS dataloader overhead issues

### Google Colab

In Colab, choose a GPU runtime, clone or upload the repo, then either use the notebook or run:

```bash
pip install -r requirements.txt
python -m src.train_jepa --config configs/colab.yaml
```

Notes for Colab CUDA:

- mixed precision is enabled in `configs/colab.yaml`
- larger batch sizes and longer schedules are used by default
- checkpoints are saved under `runs/`

### Google Colab Pro

For a stronger GPU follow-up, use the Colab Pro files:

- `configs/colab_pro.yaml`: stronger generic GPU config
- `configs/colab_pro_jepa.yaml`: JEPA-focused GPU config with longer training, EMA target encoder, and normalized-MSE latent loss
- `notebooks/colab_pro_runner.ipynb`: practical Colab Pro notebook with Drive mount and GPU workflow

Recommended order on Colab Pro:

1. Run `MAE` with `configs/colab_pro.yaml`
2. Run `JEPA` with `configs/colab_pro_jepa.yaml`
3. If JEPA is still behind, run `scripts/run_colab_jepa_sweep.sh`

Direct commands:

```bash
python3 -m src.train_mae --config configs/colab_pro.yaml
python3 -m src.train_jepa --config configs/colab_pro_jepa.yaml
```

JEPA sweep:

```bash
bash scripts/run_colab_jepa_sweep.sh
```

Good Colab Pro runtime settings:

- Runtime type: `Python 3`
- Hardware accelerator: `GPU`
- Keep the repo under `/content/jepa_mini_project`
- If you want persistent results, copy `runs/` to Drive after training or work from a Drive-backed clone

## Main Commands

### Pretraining

JEPA:

```bash
python -m src.train_jepa --config configs/mac.yaml
```

MAE-style reconstruction baseline:

```bash
python -m src.train_mae --config configs/mac.yaml
```

Shell helpers:

```bash
./scripts/run_mac_jepa.sh
./scripts/run_mac_mae.sh
./scripts/run_colab_jepa.sh
./scripts/run_colab_mae.sh
./scripts/run_colab_pro_jepa.sh
./scripts/run_colab_pro_mae.sh
```

### Command-Line Overrides

Overrides use dotted keys:

```bash
python -m src.train_jepa --config configs/mac.yaml --override training.epochs=5 --override model.mask_ratio=0.8
python -m src.train_mae --config configs/mac.yaml --override dataset.name=stl10
```

### Mask-Ratio Ablation

Run a compact ablation over `0.3`, `0.6`, and `0.8`:

```bash
python -m src.train_jepa --config configs/mac.yaml --mask-ratios 0.3 0.6 0.8
```

This saves:

- per-run checkpoints and metrics under `runs/...`
- an ablation CSV/JSON/PNG summary under `runs/mask_ablation_jepa_<dataset>/`

### Linear Probe

```bash
python -m src.eval_linear_probe --checkpoint runs/<run_name>/checkpoints/best.ckpt --config configs/mac.yaml
```

Helper script:

```bash
./scripts/run_mac_probe.sh runs/<run_name>/checkpoints/best.ckpt
```

### Retrieval

```bash
python -m src.eval_retrieval --checkpoint runs/<run_name>/checkpoints/best.ckpt --config configs/mac.yaml
```

### Anomaly-Style Geometry Check

Compare class-centroid anomaly scores for two frozen encoders:

```bash
python -m src.eval_anomaly \
  --checkpoint runs/<jepa_run>/checkpoints/best.ckpt --name JEPA \
  --checkpoint runs/<mae_run>/checkpoints/best.ckpt --name MAE \
  --config configs/mac.yaml
```

By default this reproduces the lightweight test/test geometry check used in the
current report. For a stricter split, add:

```bash
--fit-split train --eval-split test
```

### Embedding Visualization

Single model:

```bash
python -m src.visualize_embeddings --checkpoint runs/<run_name>/checkpoints/best.ckpt --config configs/mac.yaml
```

Compare JEPA vs reconstruction:

```bash
python -m src.visualize_embeddings \
  --checkpoint runs/<jepa_run>/checkpoints/best.ckpt \
  --checkpoint runs/<mae_run>/checkpoints/best.ckpt \
  --config configs/mac.yaml
```

### Robotics-Style Visual Sanity Test

This compares nearest neighbors and simple camera perturbations such as
occlusion, noise, darkening, blur, and small rotation:

```bash
python -m src.visual_robotics_test \
  --jepa-checkpoint runs/<jepa_run>/checkpoints/best.ckpt \
  --mae-checkpoint runs/<mae_run>/checkpoints/best.ckpt \
  --config configs/mac.yaml \
  --output-dir runs/visual_robotics_test
```

### Latent Export

```bash
python -m src.export_latents --checkpoint runs/<run_name>/checkpoints/best.ckpt --config configs/mac.yaml --split test
```

## Expected Outputs

Each pretraining run creates a folder under `runs/` with:

- `resolved_config.yaml`
- `run_summary.json`
- `metrics.csv`
- `metrics.json`
- `plots/pretrain_loss.png`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`
- `checkpoints/final.ckpt`

Downstream evaluations save their own folders near the checkpoint:

- `linear_probe/`
- `retrieval/`
- `embedding_viz/`
- `latents/`

## Current Results

See [RESULTS.md](RESULTS.md) for the current CIFAR-10 experiment summary, including the local Mac/MPS run, the Colab Pro GPU sweep, and suggested report figures. In short, the masked-patch reconstruction baseline is stronger on linear probing and nearest-neighbor retrieval, while the JEPA-style model is closer on a simple latent anomaly-style test and more invariant under several camera-style perturbations. This makes MAE the stronger encoder for the current static CIFAR-10 benchmark, while JEPA remains worth testing in a more realistic temporal anomaly-detection setting.

## Reusing The Encoder

The encoder can be loaded independently of the training objective:

```python
from src.models import get_encoder

encoder = get_encoder("runs/my_run/checkpoints/best.ckpt", device="cpu")
```

The returned encoder exposes:

- `encoder.encode(images)` for one embedding per image
- `encoder.forward_full(images)` if you also want token-level outputs

## How This Extends To A World Model

Later, a frozen encoder from this project can be used to produce latent states `z_t` for image sequences. Those latents can then feed a causal transformer that predicts future latent states. An anomaly detector can compare:

- predicted latent `\hat{z}_{t+1}`
- observed latent `z_{t+1}`

Large mismatches can be treated as anomalous events. The `export_latents.py` script is included specifically to make that next stage easy to prototype.

## Runtime Expectations

These are rough estimates and depend on PyTorch version and hardware.

- MacBook with MPS, JEPA or MAE, 20 epochs, batch size 64: roughly 10 to 30 minutes
- Colab GPU, JEPA or MAE, 40 epochs, batch size 256: usually much faster, often well under 30 minutes
- linear probe and retrieval: typically much quicker than pretraining

## Common Failure Modes And Fixes

- `Out of memory`
  Reduce `training.batch_size` or `evaluation.batch_size`.

- `MPS seems unstable or slow`
  Keep `mixed_precision: false`, reduce batch size, and try fewer dataloader workers.

- `Colab runtime restarts`
  Lower batch size to `128` and rerun.

- `Dataset download issues`
  Retry once or set `paths.data_dir` to a writable location.

- `t-SNE is slow`
  Lower `evaluation.visualization_max_points`.

## Educational Simplifications

This is intentionally a tiny, clear implementation rather than a large-scale JEPA reproduction.

- the predictor is a small MLP conditioned on masked positions
- the reconstruction baseline uses the same visible-context encoder and a small decoder head
- CIFAR-10 is the default to keep experiments lightweight and reproducible

Those simplifications make it easier to understand the objective comparison and easier to extend later into a temporal latent-dynamics project.
