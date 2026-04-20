# Modulated Prior Diffusion (MPD)

MPD is a diffusion model that injects condition information by modulating the prior distribution, rather than using conditioning modules inside the network.  It learns to generate segmentation results by initializing the reverse process from noise that contains information from the reference image, instead of starting from pure Gaussian noise.  The neural network architecture is based on U-Net, while the conditioning mechanism is simplified by removing additional encoders or feature fusion modules.  The diffusion process follows the standard [Denoising Diffusion Probabilistic Model (DDPM)](https://arxiv.org/abs/2006.11239).  This implementation is modified from an existing [conditional diffusion repository](https://github.com/machingwen/a3ilab/tree/main/Projects/Compositional%20Conditional%20Diffusion%20Model), and extended to support the Modulated Prior Diffusion (MPD) framework.

## Key Idea

- The reverse process is initialized from a modulated prior whose mean is shifted toward the reference image:

$$
x_T \sim \mathcal{N}(x_r, I)
$$

- The reverse denoising process starts from a noisy sample that already contains conditioning information.

![MPD_process](MPD_process.png)

---

