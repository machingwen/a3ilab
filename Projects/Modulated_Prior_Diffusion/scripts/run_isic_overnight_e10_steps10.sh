#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/macw1030/anaconda3/bin/python}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/work/macw1030/isic/train}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/work/macw1030/isic/val}"

WANDB_PROJECT="${WANDB_PROJECT:-isic-learning-curves}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-10}"
VAL_STEPS="${VAL_STEPS:-10}"
IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"

MPD_W="${MPD_W:-0.7}"

MPD_EXP="${MPD_EXP:-overnight_mpd_w07_lr${LR}_e${EPOCHS}_steps${VAL_STEPS}}"
GPCD_EXP="${GPCD_EXP:-overnight_gpcd_lr${LR}_e${EPOCHS}_steps${VAL_STEPS}}"

RUN_WHICH="${RUN_WHICH:-both}"

WANDB_ARGS=(--wandb_project "$WANDB_PROJECT")
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

COMMON_ARGS=(
  --train_data_root "$TRAIN_DATA_ROOT"
  --eval_data_root "$EVAL_DATA_ROOT"
  --img_size "$IMG_SIZE"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --epochs "$EPOCHS"
  --eval_interval 1
  --steps "$VAL_STEPS"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_eval_samples "$MAX_EVAL_SAMPLES"
  --num_workers "$NUM_WORKERS"
  --lr "$LR"
  "${WANDB_ARGS[@]}"
)

echo "============================================================"
echo "Overnight ISIC run"
echo "LR=$LR EPOCHS=$EPOCHS VAL_STEPS=$VAL_STEPS"
echo "TRAIN=$TRAIN_DATA_ROOT"
echo "EVAL=$EVAL_DATA_ROOT"
echo "RUN_WHICH=$RUN_WHICH"
echo "============================================================"

if [[ "$RUN_WHICH" == "mpd" || "$RUN_WHICH" == "both" ]]; then
  echo "Running MPD..."
  "$PYTHON_BIN" train_main.py \
    --method mpd \
    --w "$MPD_W" \
    --exp "$MPD_EXP" \
    --wandb_run_name "mpd-w${MPD_W}-lr${LR}-e${EPOCHS}-steps${VAL_STEPS}" \
    "${COMMON_ARGS[@]}"
fi

if [[ "$RUN_WHICH" == "gpcd" || "$RUN_WHICH" == "both" ]]; then
  echo "Running GPCD..."
  "$PYTHON_BIN" train_main.py \
    --method gpcd_concat \
    --exp "$GPCD_EXP" \
    --wandb_run_name "gpcd-lr${LR}-e${EPOCHS}-steps${VAL_STEPS}" \
    "${COMMON_ARGS[@]}"
fi

echo "============================================================"
echo "Runs complete."
echo "Metrics:"
echo "  MPD : result/$MPD_EXP/w_${MPD_W}_lr${LR}/metrics.csv"
echo "  GPCD: result/$GPCD_EXP/gpcd_concat_lr${LR}/metrics.csv"
echo "Checkpoints:"
echo "  MPD : checkpoint/$MPD_EXP/w_${MPD_W}_lr${LR}"
echo "  GPCD: checkpoint/$GPCD_EXP/gpcd_concat_lr${LR}"
echo "============================================================"
