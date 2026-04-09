#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "$(dirname "$0")/.."
CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 path/to/checkpoint.ckpt [extra linear-probe args]"
  exit 1
fi
shift
"${PYTHON_BIN}" -m src.eval_linear_probe --checkpoint "${CHECKPOINT}" --config configs/mac.yaml "$@"
