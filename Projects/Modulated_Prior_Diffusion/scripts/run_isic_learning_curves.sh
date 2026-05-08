#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/macw1030/anaconda3/bin/python}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/work/macw1030/isic/train}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/work/macw1030/isic/val}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-isic-learning-curves}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VAL_STEPS="${VAL_STEPS:-20}"
RUN_WHICH="${RUN_WHICH:-both}"

MPD_EXP="${MPD_EXP:-wandb_curve_mpd_w07_e5}"
GPCD_EXP="${GPCD_EXP:-wandb_curve_gpcd_short}"
MPD_RUN_NAME="${MPD_RUN_NAME:-mpd-w0.7-e5}"
GPCD_RUN_NAME="${GPCD_RUN_NAME:-gpcd-e5}"

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  fi
else
  WANDB_ARGS+=(--no_wandb)
fi

COMMON_ARGS=(
  --train_data_root "$TRAIN_DATA_ROOT"
  --eval_data_root "$EVAL_DATA_ROOT"
  --img_size 256
  --batch_size 2
  --micro_batch_size 1
  --epochs 5
  --eval_interval 1
  --steps "$VAL_STEPS"
  --max_train_samples 1000
  --max_eval_samples 50
  --num_workers 0
  "${WANDB_ARGS[@]}"
)

if [[ "$RUN_WHICH" == "mpd" || "$RUN_WHICH" == "both" ]]; then
  "$PYTHON_BIN" train_main.py \
    --method mpd \
    --w 0.7 \
    --exp "$MPD_EXP" \
    --wandb_run_name "$MPD_RUN_NAME" \
    "${COMMON_ARGS[@]}"
fi

if [[ "$RUN_WHICH" == "gpcd" || "$RUN_WHICH" == "both" ]]; then
  "$PYTHON_BIN" train_main.py \
    --method gpcd_concat \
    --exp "$GPCD_EXP" \
    --wandb_run_name "$GPCD_RUN_NAME" \
    "${COMMON_ARGS[@]}"
fi

echo "Learning-curve runs complete."
echo "Metrics:"
echo "  result/$MPD_EXP/w_0.7_lr1.0e-04/metrics.csv"
echo "  result/$GPCD_EXP/gpcd_concat_lr1.0e-04/metrics.csv"
