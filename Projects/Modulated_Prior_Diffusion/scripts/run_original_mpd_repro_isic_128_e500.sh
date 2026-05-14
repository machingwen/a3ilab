#!/usr/bin/env bash
set -euo pipefail

ROOT="/work-pvc/macw1030/projects/a3ilab/Projects/Modulated_Prior_Diffusion"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

DATA_ROOT="/work-pvc/macw1030/isic_mpd_png_128"
EXP="isic_original_mpd_repro_128_e500"
IMG_SIZE=128
EPOCHS=500
EVAL_INTERVAL=50
LR=5e-6
STEPS=100
BATCH_SIZE=32
# Use 0 workers here to stay within the shared-memory limit of this environment.
# For the final faithful run in a larger-shm environment, switch this back to 10.
WORKERS=10

cd "$ROOT"

"$PYTHON_BIN" train_main.py \
  --exp "$EXP" \
  --train_data_root "$DATA_ROOT" \
  --eval_data_root "$DATA_ROOT" \
  --img_size "$IMG_SIZE" \
  --mpd_w 0.9 \
  --epochs "$EPOCHS" \
  --eval_interval "$EVAL_INTERVAL" \
  --lr "$LR" \
  --num_workers "$WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$BATCH_SIZE" \
  --steps "$STEPS"
