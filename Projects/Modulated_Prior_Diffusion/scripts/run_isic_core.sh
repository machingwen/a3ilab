#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/macw1030/anaconda3/bin/python}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/work/macw1030/isic/train}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/work/macw1030/isic/val}"
EXP="${EXP:-isic_core}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-$EXP}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

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
  --batch_size 4
  --eval_batch_size 4
  --epochs 50
  --eval_interval 10
  --steps 50
  --sample_method ddim
  --exp "$EXP"
  "${WANDB_ARGS[@]}"
)

"$PYTHON_BIN" train_main.py \
  --method mpd \
  --w 0.9 \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" train_main.py \
  --method gpcd_concat \
  "${COMMON_ARGS[@]}"

echo "Core ISIC runs complete."
echo "Metrics:"
echo "  result/$EXP/w_0.9_lr1.0e-04/metrics.csv"
echo "  result/$EXP/gpcd_concat_lr1.0e-04/metrics.csv"
