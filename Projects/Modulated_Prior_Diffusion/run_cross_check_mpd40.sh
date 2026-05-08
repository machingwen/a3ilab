#!/usr/bin/env bash
set -euo pipefail
cd /home/macw1030/a3ilab/Projects/Modulated_Prior_Diffusion
mkdir -p logs
/home/macw1030/anaconda3/bin/python eval_isic_metrics.py \
  --methods mpd \
  --eval_data_root /work/macw1030/isic_mpd_png \
  --img_size 128 \
  --eval_batch_size 8 \
  --num_workers 0 \
  --step_values 100 \
  --epoch 500 \
  --lr 5e-6 \
  --w 0.9 \
  --exp cross_check_a3ilab_mpd_ckpt \
  --checkpoint_root /home/macw1030/a3ilab-mpd/Projects/Modulated_Prior_Diffusion/checkpoint/MED_pth_w_ISIC \
  --max_eval_samples 40 \
  2>&1 | tee logs/cross_check_mpd40.log
