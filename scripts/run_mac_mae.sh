#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "$(dirname "$0")/.."
"${PYTHON_BIN}" -m src.train_mae --config configs/mac.yaml "$@"
