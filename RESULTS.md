# Experiment Results

This page summarizes the completed CIFAR-10 JEPA-vs-MAE replication that was run end-to-end on Mac hardware using the I-JEPA-style masking and evaluation recipe.

## Final Setup

- Dataset: CIFAR-10
- Input size: 96 x 96 via bilinear upsampling in the data transform
- Patch size: 8 x 8
- Encoder: small ViT, 256-dim embeddings, depth 6, 4 heads
- JEPA objective: I-JEPA-style latent prediction with four target blocks, one context block, EMA target encoder, and a narrow predictor
- MAE objective: masked patch pixel reconstruction on the same masking pattern
- Training budget: 100 pretrain epochs per model, 50 epochs for linear probing
- Evaluation: linear probe, k-NN retrieval, and qualitative nearest-neighbor visualization

## Main Results

| Method | Prediction target | Linear probe (%) | k-NN @1 (%) | k-NN @5 (%) |
| --- | --- | ---: | ---: | ---: |
| I-JEPA-style latent prediction | Latent target | **71.08** | **66.15** | **89.13** |
| MAE baseline | Pixels | 56.29 | 50.59 | 83.59 |

In this revised experiment, JEPA outperformed MAE on all downstream metrics. The result flipped the earlier draft and strongly suggests that the main difference came from implementing the I-JEPA recipe more faithfully: multi-block masking, excluded masked tokens, EMA teacher updates, target-encoder evaluation, and the narrow predictor bottleneck.

## Interpretation

The qualitative retrievals point in the same direction as the quantitative scores. JEPA predictor outputs tended to retrieve training images that matched the query class and object identity more often than MAE, while MAE neighbors were more likely to align on texture or color. Across 20 selected visualization queries, the average JEPA semantic retrieval score was `0.502`.

This is consistent with the claim in Assran et al. (2023): latent prediction encourages more semantic representations than direct pixel reconstruction, even in this reduced image-only CIFAR-10 setting.

## Tracked Artifacts In This Branch

The branch includes:

- the exact runnable single-file training pipeline: `run_experiment.py`
- a CUDA-prioritized copy: `run_experiment_cuda.py`
- the figure generator used to build paper-style assets: `make_paper_figures.py`
- the revised report generator: `build_revised_draft.py`
- compact result artifacts under `results/`
- polished tables and figures under `results/paper_assets/`
- qualitative retrieval figures under `results/viz/`

Large raw artifacts such as the dataset cache, memmapped patch banks, and model checkpoints are intentionally excluded from Git tracking.

## Suggested Figures

Use these first if space is limited:

1. `results/paper_assets/table_main_results.png`
2. `results/paper_assets/figure_quantitative_results.png`
3. `results/paper_assets/figure_qualitative_best4.png`
4. `results/paper_assets/table_protocol_alignment.png`
