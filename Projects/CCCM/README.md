# CCCM
## Compositional Conditioning Consistency Model
### Introduction
Compositional Conditional Consistency Model (CCCM) is a fast and flexible generative model for compositional conditional image synthesis. CCCM distills the strong compositional generalization abilities of Compositional Conditional Diffusion Models (CCDM) into a consistency-model framework, enabling high-fidelity image generation in just 2–4 inference steps (10-20x faster than CCDM)—while maintaining zero-shot generation on unseen attribute-object pairs.

<img width="627" height="297" alt="image" src="https://github.com/user-attachments/assets/d3d38739-d960-4f0c-ad3c-f5b89ba88bd1" />

This repository provides the full codebase for CCCM, including novel consistency distillation strategies—Step Fuse, Loss Fuse, and Switch—which blend teacher predictions and diffusion-formulated signals to achieve optimal trade-offs between image quality and compositional accuracy.
All experiments are conducted on the CelebA dataset.

### Method Overview
Teacher Model: pretrained CCDM (U-Net), trained for compositional conditional generation.

Student Model: CCCM, initialized from CCDM weights, but trained with consistency loss as proposed in Consistency Models (Song et al., 2023).

Modified Consistency Distillation:
Leverages teacher predictions (via ODE/DDIM solver) and diffusion-formulated supervision.
Three supervision fusion strategies: StepFuse, LossFuse, Switch, each with scheduler-controlled weighting.

<img width="636" height="274" alt="switch_ppt" src="https://github.com/user-attachments/assets/2a93e821-2537-4d84-9578-2f4e6fdfe104" />
<img width="636" height="274" alt="stepfuse_ppt" src="https://github.com/user-attachments/assets/77847c5a-f256-4e5f-b977-d7ed1c02786e" />
<img width="636" height="274" alt="lossfuse_ppt" src="https://github.com/user-attachments/assets/21ed0653-27e8-4d6b-81fe-1f9032951c6a" />

### Installation
1. Clone this repository
2. Set up Python environment: This project runs in Python 3.8.10.  
   - If you do not already have Python 3.8.10 installed, run the setup script:  
     `./py38venv_install.sh`  
   - If you already have Python 3.8.10, you can skip the script and manually create a virtual environment:  
     `python3.8 -m venv CCCM_venv`
   - Download Pytorch manually:
     `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
   - Make yourself a `checkpoints` and `data` directory for saving ckpts and ur training dataset.
   
### Usage
1. Dataset preparation:
   - Each image should be placed into a subdirectory named according to its attribute-object condition (e.g., BlondeHair_female, BlackHair_male, etc.)
   - If you are using a new dataset, please define a new class in `config.py` and follow the structure of the existing implementations.
2. Download pretrained weights of the teacher model using `download_from_hub.py`.
3. Start training by `accelerate launch train_script.py --data "" --dataset_nums_cond "" --save_path "" --epochs "" --train_batch_size "" --fuse_method "" --fuse_schedule "" --fuse_args ""`
4. This script is modified from https://github.com/luosiallen/latent-consistency-model/tree/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py
5. For inference, use
   - `from models.engine import ConsisctencySampler`
   - `sampler = ConsisctencySampler(model, noise_scheduler, n_samples, args, device=device)`
   - `images = sampler(c1, c2, x_t = torch.randn(n_samples, 3, 128, 128), num_inference_steps = num_inference_steps, guidance_scale = omega)`
