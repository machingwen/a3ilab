# Three-condition CCDM extension (CelebA)

This extends `Projects/Conditional_Diffusion-main` without changing its
two-condition code. It covers the archived **Add U-Net** and its `--compose`
variant. Conditions are always `(hairstyle, hair color, sex)`; the color map
matches the archived `config.py` (`Brown=0, Blond=1, Gray=2, Black=3`).

The existing two-condition `UNet` blocks, embeddings, diffusion coefficients,
loss scaling, classifier-free guidance formula, and DDIM time grid are reused.
The third embedding is added to the sum or concatenated to the learned
`1536 -> 4096 -> 512` projection. All three labels receive the source's
independent 0.1 dropout during training and are dropped for unconditional
guidance. This is an extension of the archived code, **not** proof that it
recreates the thesis's unpublished three-condition checkpoint.

From `Projects/Conditional_Diffusion-main`, with the same Python dependencies
as its original `train.py`:

```bash
python train_triple.py \
  --data /path/to/celeba_condition_folders \
  --held-out 'Wavy_Hair Black_Hair Female' \
  --output /path/to/new_run --compose

python sample_triple.py \
  --checkpoint /path/to/new_run/model_100.pth \
  --condition 'Wavy_Hair Black_Hair Female' \
  --output /path/to/new_samples --seed 42
```

The additive baseline uses the same training command without `--compose` and
must write to a different output directory. To make a paired comparison, set
the same training and generation seeds, dataset root, held-out tuple, and
evaluation procedure for both runs. Generation records the requested tuple and
seed; it does not assign ground-truth hair colors to generated images.

The implementation requires CUDA and creates a new output directory rather
than overwriting a previous run. It saves checkpoint state and training tuple
counts. Before a substantial training run, run a one-epoch engineering smoke
on a small, separate data copy and verify model loading and DDIM output.
