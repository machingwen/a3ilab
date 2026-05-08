#!/usr/bin/env bash
set -euo pipefail

cd /home/macw1030/a3ilab-mpd/Projects/Modulated_Prior_Diffusion
mkdir -p logs result/MED_pth_w_ISIC

/home/macw1030/anaconda3/bin/python evaluate_generate_style.py \
  --batch_size 4 \
  --num_workers 0 \
  --steps 100 \
  --num_generations 3 \
  --crf \
  --largest_component \
  --fill_holes \
  --output_csv result/MED_pth_w_ISIC/generate_style_full_epoch500.csv \
  2>&1 | tee logs/evaluate_generate_style_full_epoch500.log
