# Modulated Prior Diffusion (MPD) Reproduction Review

This branch is prepared for review by the original MPD author.  Its purpose is to make the current MPD reproduction workflow explicit and easy to check before any public release or downstream method comparison.

New method work is intentionally out of scope for this branch.  The README and scripts here should describe only the original MPD reproduction path.

## Scope

- MPD training on a preprocessed ISIC split.
- MPD generation from a trained checkpoint.
- Optional DenseCRF post-processing and metric evaluation.
- Reproduction scripts under `scripts/` for the exact local workflow.

## Method Summary

MPD injects condition information by modulating the prior distribution instead of adding a separate conditioning module inside the denoising network.  The reverse process is initialized from noise mixed with the reference image:

```text
x_T = (1 - mpd_w) * noise + mpd_w * reference_image
```

In this reproduction workflow, `mpd_w` controls the MPD prior mixing strength.  `sampler_w` is kept separate and controls DDIM sampler guidance where supported.

## Dataset

The scripts expect an ISIC split with this directory layout:

```text
<data_root>/
  train_original_80/
  train_crop_80/
  test_original_20/
  test_crop_20/
```

The current reproduction uses images resized to `128 x 128` before training.

Preprocess the ISIC split:

```bash
bash scripts/run_preprocess_isic_128.sh
```

Or run the preprocessing script directly:

```bash
python scripts/preprocess_isic_128.py \
  --src-root /path/to/isic_mpd_png \
  --dst-root /path/to/isic_mpd_png_128 \
  --size 128 \
  --overwrite
```

## Training

Run the local reproduction training script:

```bash
bash scripts/run_original_mpd_repro_isic_128_e500.sh
```

Equivalent direct command:

```bash
python train_main.py \
  --exp isic_original_mpd_repro_128_e500 \
  --train_data_root /path/to/isic_mpd_png_128 \
  --eval_data_root /path/to/isic_mpd_png_128 \
  --img_size 128 \
  --mpd_w 0.9 \
  --epochs 500 \
  --eval_interval 50 \
  --lr 5e-6 \
  --num_workers 4 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --steps 100
```

Checkpoints are written under:

```text
checkpoint/isic_original_mpd_repro_128_e500/mpd_w_0.9_lr5.0e-06/
```

## Generation And Evaluation

Run generation and generated-mask evaluation:

```bash
bash scripts/run_original_mpd_repro_generate_eval.sh
```

Equivalent generation command:

```bash
python generate_main.py \
  --method mpd \
  --data_root /path/to/isic_mpd_png_128 \
  --checkpoint_root checkpoint/isic_original_mpd_repro_128_e500 \
  --output_dir generate_result/repro_mpd_postprocess_densecrf \
  --epoch 500 \
  --ckpt_lr 5.0e-06 \
  --img_size 128 \
  --steps 100 \
  --eval_batch_size 8 \
  --num_workers 0 \
  --sampler_w 3.0 \
  --mpd_w_values 0.9
```

Evaluate generated masks with DenseCRF:

```bash
python scripts/eval_original_mpd_repro_densecrf_isic.py \
  --data_root /path/to/isic_mpd_png_128 \
  --output_dir result/isic_original_mpd_repro_128_e500/eval_generated_masks_densecrf \
  --pred_mask_dir generate_result/repro_mpd_postprocess_densecrf/mpd_w_0.9/masks_mpd_w_0.9 \
  --mpd_w 0.9
```

## Review Notes

- `train_main.py` is the MPD reproduction entry point for this branch.
- `generate_main.py` is the MPD generation entry point.
- The reproduction scripts use local default paths from the current environment; reviewers should replace them with their dataset/checkpoint paths when needed.
- Generated outputs, checkpoints, datasets, virtual environments, and notebook checkpoint folders should not be committed.
