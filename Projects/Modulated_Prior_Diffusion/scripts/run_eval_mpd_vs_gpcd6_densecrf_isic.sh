#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python scripts/eval_mpd_vs_gpcd6_densecrf_isic.py \
  --data_root /work-pvc/macw1030/isic_mpd_png \
  --mpd_checkpoint checkpoint/isic_mpd_matched_128_e500/w_0.9_lr5.0e-06/model_epoch500_lr5.0e-06_w0.9.pth \
  --gpcd6_checkpoint checkpoint/isic_mpd_matched_128_e500/gpcd_6ch_lr5.0e-06/model_epoch500_lr5.0e-06_gpcd_6ch.pth \
  --methods mpd gpcd_6ch \
  --seeds 0 1 2 \
  --steps 100 \
  --img_size 128 \
  --eval_batch_size 8 \
  --num_workers 4 \
  --mpd_w 0.9
