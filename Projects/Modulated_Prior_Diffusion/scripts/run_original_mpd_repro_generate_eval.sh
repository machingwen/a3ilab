#!/usr/bin/env bash
set -euo pipefail

ROOT="/work-pvc/macw1030/projects/a3ilab/Projects/Modulated_Prior_Diffusion"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

DATA_ROOT="/work-pvc/macw1030/isic_mpd_png_128"
CHECKPOINT_ROOT="${ROOT}/checkpoint/isic_original_mpd_repro_128_e500"
OUTPUT_ROOT="${ROOT}/generate_result/repro_mpd_postprocess_densecrf"
EVAL_OUTPUT_DIR="${ROOT}/result/isic_original_mpd_repro_128_e500/eval_generated_masks_densecrf"

EPOCH=500
CKPT_LR="5.0e-06"
IMG_SIZE=128
STEPS=100
EVAL_BATCH_SIZE=8
WORKERS=0
SAMPLER_W=3.0
MPD_W=0.9

cd "$ROOT"

"$PYTHON_BIN" generate_main.py \
  --method mpd \
  --data_root "$DATA_ROOT" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --output_dir "$OUTPUT_ROOT" \
  --epoch "$EPOCH" \
  --ckpt_lr "$CKPT_LR" \
  --img_size "$IMG_SIZE" \
  --steps "$STEPS" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$WORKERS" \
  --sampler_w "$SAMPLER_W" \
  --mpd_w_values "$MPD_W"

"$PYTHON_BIN" scripts/eval_original_mpd_repro_densecrf_isic.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --pred_mask_dir "${OUTPUT_ROOT}/mpd_w_${MPD_W}/masks_mpd_w_${MPD_W}" \
  --mpd_w "$MPD_W"
