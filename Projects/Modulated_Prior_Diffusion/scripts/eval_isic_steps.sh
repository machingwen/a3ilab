#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/macw1030/anaconda3/bin/python}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/work/macw1030/isic/val}"
EXP="${EXP:-isic_core}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoint/$EXP}"
EPOCH="${EPOCH:-50}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

ARGS=(
  --eval_data_root "$EVAL_DATA_ROOT"
  --checkpoint_root "$CHECKPOINT_ROOT"
  --exp "$EXP"
  --epoch "$EPOCH"
  --w 0.9
  --img_size 256
  --eval_batch_size 4
  --step_values 5 10 20 50 100
  --methods mpd gpcd_concat
)

if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
fi

"$PYTHON_BIN" eval_isic_metrics.py "${ARGS[@]}"

echo "Step evaluation complete."
echo "Metrics:"
echo "  result/$EXP/w_0.9_lr1.0e-04/metrics.csv"
echo "  result/$EXP/gpcd_concat_lr1.0e-04/metrics.csv"
