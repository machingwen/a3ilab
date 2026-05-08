#!/usr/bin/env bash
set -euo pipefail
cd /home/macw1030/a3ilab/Projects/Modulated_Prior_Diffusion
mkdir -p logs

PYTHON=/home/macw1030/anaconda3/bin/python
EXP=match128_mpd_gpcd_lr5e-6_e100
COMMON=(
  --train_data_root /work/macw1030/isic_mpd_png
  --eval_data_root /work/macw1030/isic_mpd_png
  --img_size 128
  --batch_size 2
  --eval_batch_size 8
  --epochs 100
  --eval_interval 50
  --steps 100
  --num_workers 4
  --lr 5e-6
  --exp "$EXP"
  --no_wandb
)

echo "===== MPD match run: $(date) ====="
"$PYTHON" train_main.py --method mpd --w 0.9 "${COMMON[@]}" 2>&1 | tee "logs/${EXP}_mpd.log"

echo "===== GPCD match run: $(date) ====="
"$PYTHON" train_main.py --method gpcd_concat "${COMMON[@]}" 2>&1 | tee "logs/${EXP}_gpcd.log"

echo "===== DONE: $(date) ====="
echo "MPD metrics:  result/${EXP}/w_0.9_lr5.0e-06/metrics.csv"
echo "GPCD metrics: result/${EXP}/gpcd_concat_lr5.0e-06/metrics.csv"
