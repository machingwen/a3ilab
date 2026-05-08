#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/macw1030/anaconda3/bin/python}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/work/macw1030/isic/val}"

MPD_EXP="${MPD_EXP:-wandb_curve_mpd_w07_e5}"
GPCD_EXP="${GPCD_EXP:-wandb_curve_gpcd_short}"

MPD_CHECKPOINT_ROOT="${MPD_CHECKPOINT_ROOT:-checkpoint/$MPD_EXP}"
GPCD_CHECKPOINT_ROOT="${GPCD_CHECKPOINT_ROOT:-checkpoint/$GPCD_EXP}"

MPD_EPOCH="${MPD_EPOCH:-5}"
GPCD_EPOCH="${GPCD_EPOCH:-5}"

MPD_W="${MPD_W:-0.7}"

IMG_SIZE="${IMG_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"

STEP_VALUES=(${STEP_VALUES:-1 3 5 10 20 50})

COMMON_ARGS=(
  --eval_data_root "$EVAL_DATA_ROOT"
  --img_size "$IMG_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --max_eval_samples "$MAX_EVAL_SAMPLES"
  --step_values "${STEP_VALUES[@]}"
)

echo "Evaluating MPD..."
echo "  exp: $MPD_EXP"
echo "  checkpoint_root: $MPD_CHECKPOINT_ROOT"
echo "  epoch: $MPD_EPOCH"
echo "  w: $MPD_W"
echo "  steps: ${STEP_VALUES[*]}"

"$PYTHON_BIN" eval_isic_metrics.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint_root "$MPD_CHECKPOINT_ROOT" \
  --exp "$MPD_EXP" \
  --epoch "$MPD_EPOCH" \
  --w "$MPD_W" \
  --methods mpd

echo "Evaluating GPCD..."
echo "  exp: $GPCD_EXP"
echo "  checkpoint_root: $GPCD_CHECKPOINT_ROOT"
echo "  epoch: $GPCD_EPOCH"
echo "  steps: ${STEP_VALUES[*]}"

"$PYTHON_BIN" eval_isic_metrics.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint_root "$GPCD_CHECKPOINT_ROOT" \
  --exp "$GPCD_EXP" \
  --epoch "$GPCD_EPOCH" \
  --methods gpcd_concat

echo "Step evaluation complete."
echo "Metrics:"
echo "  result/$MPD_EXP/w_${MPD_W}_lr1.0e-04/metrics.csv"
echo "  result/$GPCD_EXP/gpcd_concat_lr1.0e-04/metrics.csv"
