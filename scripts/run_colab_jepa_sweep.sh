#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m src.train_jepa --config configs/colab_pro_jepa.yaml --override model.mask_ratio=0.6 --override training.latent_loss=normalized_mse "$@"
"${PYTHON_BIN}" -m src.train_jepa --config configs/colab_pro_jepa.yaml --override model.mask_ratio=0.3 --override training.latent_loss=normalized_mse "$@"
"${PYTHON_BIN}" -m src.train_jepa --config configs/colab_pro_jepa.yaml --override model.mask_ratio=0.8 --override training.latent_loss=normalized_mse "$@"
"${PYTHON_BIN}" -m src.train_jepa --config configs/colab_pro.yaml --override model.mask_ratio=0.6 --override training.latent_loss=cosine --override model.target_encoder_mode=shared "$@"

