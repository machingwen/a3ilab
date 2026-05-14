#!/usr/bin/env bash
set -euo pipefail

ROOT="/work-pvc/macw1030/projects/a3ilab/Projects/Modulated_Prior_Diffusion"

cd "$ROOT"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" scripts/preprocess_isic_128.py \
  --src-root /work-pvc/macw1030/isic_mpd_png \
  --dst-root /work-pvc/macw1030/isic_mpd_png_128 \
  --size 128 \
  --overwrite
