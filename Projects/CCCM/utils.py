import argparse
import functools
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF


def args_string_to_dict(dict_stepfuse_args):
    """  ['0.25:0.6', '0.5:0.3', '0.75:0.1'] to {0.25: 0.6, 0.5: 0.3, 0.75: 0.1} """
    segment_dict = {}
    for segment in dict_stepfuse_args:
        key, value = segment.split(":")
        segment_dict[float(key)] = float(value)
        
    return segment_dict

class StepFuseScheduler:
    def __init__(self, method, total_epochs, stepfuse_args=None):
        """ piecewise_dict example: 
            {0:0.5} for 0.5 constant c_t.
            {1:0} for fully linear decreasing c_t.
            {0.2:0.5, 0.8:0.1} will gradually decreases to 0.5 at 20% epochs, 0.1 at 80%, and maintain 0.1.
        """
        self.method = method
        self.total_epochs = total_epochs
        
        # parse stepfuse_args based on method
        if self.method == "piecewise":
            try:
                self.piecewise_dict = args_string_to_dict(stepfuse_args)
            except ValueError as err:
                raise ValueError("For stepwise method, args.stepfuse_args should be a dict in str. Ex:'0.2:0.5'\n" + err.args[0]) from None
            self.decay_rate = None

        elif self.method == "exponential":
            assert len(stepfuse_args) == 1, "For exponential method, args.stepfuse_args should be a float for decay rate"
            self.decay_rate = float(stepfuse_args[0])
            self.piecewise_dict = None
            self.switch_threshold = None
            
        elif self.method == "only_teacher":
            self.piecewise_dict = None
            self.decay_rate = None
            self.switch_threshold = None
            
        elif self.method == "only_ode":
            self.piecewise_dict = None
            self.decay_rate = None
            self.switch_threshold = None
        
        elif self.method == "switch":
            assert len(stepfuse_args) == 1, "For switch, args.stepfuse_args should be a float between 0 and 1."
            self.switch_threshold = float(stepfuse_args[0])
            self.piecewise_dict = None
            self.decay_rate = None
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        self._current_c_t = 1.0  # Default to 1 at the beginning

    def update_c_t(self, current_epoch):
        """ compute c_t once per epoch"""
        assert current_epoch <= self.total_epochs, "current_epoch > total_epochs?"
        progress = current_epoch / self.total_epochs

        if self.method == "piecewise":
            self._current_c_t = self._compute_piecewise(progress)
        elif self.method == "exponential":
            self._current_c_t = self._compute_exponential(progress)
        elif self.method == "only_teacher":
            self._current_c_t = 1.0
        elif self.method == "only_ode":
            self._current_c_t = 0.0
        elif self.method == "switch":
            self._current_c_t = self._compute_switch(progress)
        else:
            raise ValueError(f"Unsupported stepfuse_method: {self.method}")

    def _compute_piecewise(self, progress):
        """ compute c_t with piecewise linear"""
        segments_keys = sorted(self.piecewise_dict.keys())
        segments_values = [self.piecewise_dict[k] for k in segments_keys]
        # Add start (1.0) points
        segments_keys = [0.0] + segments_keys
        segments_values = [1.0] + segments_values

        # Find the current segment
        for i in range(len(segments_keys) - 1):
            if segments_keys[i] <= progress < segments_keys[i + 1]:
                # Compute linear interpolation in the current segment
                c_t = segments_values[i] + (segments_values[i + 1] - segments_values[i]) * \
                      (progress - segments_keys[i]) / (segments_keys[i + 1] - segments_keys[i])
                c_t = round(c_t, 4)
                break
            else:
                c_t = segments_values[-1]
        return c_t

    def _compute_exponential(self, progress):
        """ compute c_t with exponential decay. """
        c_t = math.exp(-self.decay_rate * progress)
        c_t = c_t * (1 - progress)
        return c_t
    
    def _compute_switch(self, progress):
        return 1 if progress < self.switch_threshold else 0
    
    def get_c_t(self):
        """ get c_t of current epoch，used for each step(batch)."""
        return self._current_c_t


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=100):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


# For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
def get_ddim_timesteps(scheduler, solver, bsz, num_ddim_timesteps, device='cpu'):
    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
    index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=device).long()
    start_timesteps = solver.ddim_timesteps[index]
    timesteps = start_timesteps - topk
    timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
    
    return start_timesteps, timesteps

def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out

# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract(alphas, timesteps, sample.shape)
    sigmas = extract(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract(alphas, timesteps, sample.shape)
    sigmas = extract(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if is_torch_npu_available():
    #     torch.npu.manual_seed_all(seed)
    # else:
    #     torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
        
def load_checkpoint(model, path, optimizer=None):
    checkpoint = torch.load(path)
    
    new_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith("module"):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    try:
        model.load_state_dict(new_dict)
        print("All keys successfully match")
    except:
        print("some keys are missing!")    
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
    return epoch
