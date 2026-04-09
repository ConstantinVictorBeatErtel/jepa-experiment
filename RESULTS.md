# Experiment Results

This page summarizes the first JEPA-vs-masked-reconstruction experiment on CIFAR-10.

## Setup

- Dataset: CIFAR-10
- Image size: 32 x 32
- Patch size: 4
- Encoder: tiny ViT-style encoder
- JEPA objective: latent prediction from visible patches to masked target patches
- MAE objective: masked patch pixel reconstruction
- Main GPU comparison:
  - JEPA run: `colab_pro_jepa_jepa_cifar10_mr0.3_20260409_201133`
  - MAE run: `colab_pro_mae_cifar10_mr0.6_20260409_184049`

The JEPA run used the stronger Colab Pro configuration with EMA target encoder, normalized-MSE latent loss, and mask ratio 0.3. The MAE run used the Colab Pro reconstruction baseline with mask ratio 0.6.

## What Was Run

The experiment had two stages.

First, we ran the small local Apple Silicon/MPS experiment from `configs/mac.yaml`. This verified that the full pipeline worked end-to-end on a Mac: pretraining, checkpointing, linear probe, retrieval, visualization, and latent export.

Then, we ran a stronger Colab Pro GPU follow-up. The MAE baseline used the stronger generic GPU config, while JEPA used a JEPA-focused config with longer training, EMA target encoder, normalized-MSE latent loss, and a mask-ratio sweep. The best JEPA result came from mask ratio 0.3.

## Main Representation Results

| Model | Probe test accuracy | Probe validation accuracy | Retrieval acc@1 | Retrieval acc@5 |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 0.2740 | 0.2738 | 0.2090 | 0.5814 |
| MAE | 0.4002 | 0.4018 | 0.3361 | 0.7016 |

The masked-reconstruction baseline learned more class-separable CIFAR-10 representations. It outperformed the JEPA-style model on linear probing and nearest-neighbor retrieval.

## Mac vs Colab JEPA

The Colab JEPA sweep improved the linear probe result compared with the original Mac run. For context, the local Mac MAE baseline was already much stronger than the Mac JEPA model.

| Model/run | Mask ratio | Probe test accuracy | Retrieval acc@1 | Retrieval acc@5 |
| --- | ---: | ---: | ---: | ---: |
| Mac JEPA | 0.6 | 0.1861 | 0.2070 | 0.5819 |
| Mac MAE | 0.6 | 0.4000 | 0.3278 | 0.7056 |
| Colab Pro JEPA, best sweep run | 0.3 | 0.2740 | 0.2090 | 0.5814 |
| Colab Pro MAE | 0.6 | 0.4002 | 0.3361 | 0.7016 |

The GPU-tuned JEPA configuration improved classification accuracy from 0.1861 to 0.2740, but retrieval barely changed. The MAE result was stable across Mac and Colab, around 0.40 probe accuracy and around 0.70 retrieval acc@5. This suggests the JEPA sweep helped somewhat, but did not fully fix the representation gap versus MAE.

The Colab Pro JEPA sweep results were:

| JEPA run | Mask ratio | Epochs | Probe test accuracy | Retrieval acc@1 | Retrieval acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `colab_pro_jepa_jepa_cifar10_mr0.3_20260409_201133` | 0.3 | 80 | 0.2740 | 0.2090 | 0.5814 |
| `colab_pro_jepa_jepa_cifar10_mr0.6_20260409_190159` | 0.6 | 80 | 0.2568 | 0.1830 | 0.5640 |
| `colab_pro_jepa_jepa_cifar10_mr0.6_20260409_194522` | 0.6 | 80 | 0.2568 | 0.1830 | 0.5640 |
| `colab_pro_jepa_jepa_cifar10_mr0.8_20260409_203726` | 0.8 | 12 | not evaluated | not evaluated | not evaluated |

The 0.8 run was stopped early to conserve Colab GPU quota and was not used in the final comparison.

## Simple Latent Anomaly Test

We also tested a simple anomaly-style score: for each CIFAR-10 class, treat that class as the normal class, compute the normalized embedding centroid, and score all examples by distance from that centroid.

| Model | Mean anomaly AUC | Class-wise wins |
| --- | ---: | ---: |
| JEPA | 0.5728 | 6 / 10 |
| MAE | 0.5815 | 4 / 10 |

MAE was slightly better on average, but JEPA won more individual one-vs-rest class splits. The average AUC difference was small, so this result is best interpreted as a near tie. This is more encouraging for JEPA than the linear-probe result, because the intended follow-on project is anomaly detection rather than CIFAR-10 classification.

## Robotics-Style Perturbation Test

We ran a lightweight visual robustness test using common robot-camera nuisance transformations: occlusion, noise, low light, blur, and small rotation. The metric is cosine similarity between the original embedding and the perturbed-image embedding. Higher means the representation changes less under that perturbation.

| Model | Occlusion | Noise | Dark | Blur | Rotation | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JEPA | 0.999973 | 0.986278 | 0.999299 | 0.999759 | 0.999328 | 0.996927 |
| MAE | 0.989105 | 0.996559 | 0.928765 | 0.996674 | 0.968855 | 0.975992 |

JEPA was more invariant to occlusion, darkening, blur, and rotation. MAE was more invariant to noise.

However, this should be interpreted carefully. Very high invariance is not automatically good. Since JEPA had weaker classification and retrieval metrics, its high perturbation stability may indicate that the representation is too insensitive or partially collapsed, not necessarily that it is better for robotics. The result is still useful: it shows that the objectives produce measurably different embedding geometry, and it motivates a more realistic temporal anomaly experiment.

## Interpretation

The main conclusion is:

> In this small CIFAR-10 mini-project, the masked-patch reconstruction baseline learned stronger class-separable representations than the simplified JEPA-style latent prediction model. In the final Colab Pro comparison, MAE reached 0.4002 linear-probe test accuracy and 0.3361 retrieval acc@1, while JEPA reached 0.2740 and 0.2090. JEPA improved over the original Mac JEPA probe result, from 0.1861 to 0.2740, and looked closer on a simple anomaly-style score, 0.5728 mean AUC versus MAE's 0.5815, but it did not beat MAE overall.

This does not contradict the JEPA literature. Full I-JEPA/V-JEPA systems use larger models, more data, carefully designed context/target masking, and longer training. This repo intentionally uses a tiny educational setup. On CIFAR-10, pixel reconstruction is a strong baseline because preserving local color, texture, and shape information is already useful for class labels.

For the intended robot anomaly-detection direction, the important next step is not to continue optimizing CIFAR-10 classification. The better next step is to move from static images to sequences, freeze the encoder, extract latent states, and train a small causal transformer to predict future latents. Anomaly scores should then be based on predicted-vs-observed latent mismatch.

## Suggested Report Figures

If space is limited, use these figures in this order:

1. Main metrics bar chart: linear probe accuracy and retrieval acc@1/acc@5 for JEPA vs MAE. This is the clearest quantitative result and should be the main figure.
2. Robotics-style perturbation stability bar chart. This is useful because it connects the mini-project to robot-camera nuisance changes, but describe it as a sanity check rather than a real robotics benchmark.
3. One compact nearest-neighbor visual grid, only if you have room. It gives intuition for embedding behavior, but it is less rigorous than the metrics.

Skip t-SNE/PCA if space is tight. Those plots show that the embeddings look different, but they are harder to interpret quantitatively and weaker as evidence than the probe, retrieval, anomaly AUC, and perturbation tables.

## Next Steps

Recommended follow-up work:

- Re-run `eval_anomaly.py` with `--fit-split train --eval-split test` as a stricter anomaly check.
- Fix and polish the nearest-neighbor visual comparison so the saved grids are easier to interpret.
- Try a real sequence dataset or a sampled-frame robotics/autonomous-driving dataset.
- Train a small causal transformer over exported latents to create a first temporal world-model prototype.
- Revisit JEPA design if needed: stronger predictor, block-based target masks, better target encoder design, and anti-collapse diagnostics.
