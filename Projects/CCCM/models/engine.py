from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
from diffusers.schedulers import LCMScheduler
from diffusers.utils.torch_utils import randn_tensor

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


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)



class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, c):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, c)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
        loss = torch.sum(loss)
        return loss
    
class ConditionalGaussianDiffusionTrainer(nn.Module):
    
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, c1, c2):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, c1, c2)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
#         loss = torch.sum(loss)
        return loss

class ConditionalDiffusionEncoderTrainer(nn.Module):
    
    def __init__(self, encoder: nn.Module, model: nn.Module, beta: Tuple[int, int], T: int, drop_prob: float = 0.1, only_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.T = T
        self.drop_prob = drop_prob
        self.only_encoder = only_encoder

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, c1, c2):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        
        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)
        
        # get embedding from conditional encoder
        context = self.encoder.class_encoder(c1, c2)
        if self.only_encoder:
            context = context.view(x_0.shape[0], -1, self.encoder.class_emb_dim) # [B, emb_dim * 2] -> [B, 2, emb_dim]
        else:
            context = self.encoder.class_projection(context) # [B, context_dim] -> [B, 1, context_dim]
        
        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, context)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
#         loss = torch.sum(loss)
        return loss
    
class ConditionalGaussianDiffusionTrainerOneCond(nn.Module):
    
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, c1):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, c1)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
        loss = torch.sum(loss)
        return loss


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int, w: float, schedule="linear"):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w
        # generate T steps of beta
        if schedule == "linear":
            beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        elif schedule == "cosine":
            beta_t = betas_for_alpha_bar(num_diffusion_timesteps=T)
        self.register_buffer("beta_t", beta_t)

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t, obj, atr):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$
        """
        epsilon_theta = self.model(x_t, t, obj, atr)
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta

        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, obj, atr, time_step: int):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        mean, var = self.cal_mean_variance(x_t, t)

        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, only_return_x_0: bool = True, interval: int = 1, **kwargs):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.
            kwargs: no meaning, just for compatibility.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        x = [x_t]
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
            for time_step in sampling_steps:
                x_t = self.sample_one_step(x_t, time_step)

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int, w: float, schedule="linear"):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w

        # generate T steps of beta
        if schedule == "linear":
            beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        elif schedule == "cosine":
            beta_t = betas_for_alpha_bar(num_diffusion_timesteps=T)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, atr: torch.LongTensor, obj: torch.LongTensor, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict conditional noise and unconditional noise using model
        epsilon_theta_t = self.model(x_t, t, atr, obj)
        unc_epsilon_theta_t = self.model(x_t, t, obj, atr, force_drop_ids=True)
        
        # classifier-free guidance
        epsilon_theta_t = (1 + self.w) * epsilon_theta_t - self.w * unc_epsilon_theta_t
        
        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, atr: torch.LongTensor, obj: torch.LongTensor, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int32)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], atr, obj, time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
    


class DDIMSamplerEncoder(nn.Module):
    def __init__(self, model, encoder, beta: Tuple[int, int], T: int, w: float, schedule="linear", only_encoder=False):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.T = T
        self.w = w
        self.only_encoder = only_encoder

        # generate T steps of beta
        if schedule == "linear":
            beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        elif schedule == "cosine":
            beta_t = betas_for_alpha_bar(num_diffusion_timesteps=T)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, atr: torch.LongTensor, obj: torch.LongTensor, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict conditional noise and unconditional noise using model
        context = self.encoder.class_encoder(atr, obj)
        if self.only_encoder:
            context = context.view(x_t.shape[0], -1, self.encoder.class_emb_dim) # [B, emb_dim * 2] -> [B, 2, emb_dim]
        else:
            context = self.encoder.class_projection(context) # [B, context_dim] -> [B, 1, context_dim]
        
        unc_atr = torch.full((x_t.shape[0],), self.encoder.num_atr, device=x_t.device, dtype=torch.long)
        unc_obj = torch.full((x_t.shape[0],), self.encoder.num_obj, device=x_t.device, dtype=torch.long)
        unc_context = self.encoder.class_encoder(unc_atr, unc_obj)
        if self.only_encoder:
            unc_context = unc_context.view(x_t.shape[0], -1, self.encoder.class_emb_dim) # [B, emb_dim * 2] -> [B, 2, emb_dim]
        else:
            unc_context = self.encoder.class_projection(unc_context) # [B, context_dim] -> [B, 1, context_dim]
        
        epsilon_theta_t = self.model(x_t, t, context)
        unc_epsilon_theta_t = self.model(x_t, t, unc_context)
        
        # classifier-free guidance
        epsilon_theta_t = (1 + self.w) * epsilon_theta_t - self.w * unc_epsilon_theta_t
        
        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, atr: torch.LongTensor, obj: torch.LongTensor, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int32)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], atr, obj, time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
    
def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Denormalize an image array to [0,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to denormalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The denormalized image array.
    """
    return (images / 2 + 0.5).clamp(0, 1)

class ConsisctencySampler(nn.Module):
    def __init__(self, model, noise_scheduler, n_samples, args, device):
        super().__init__()
        self.time_cond_proj_dim = args.time_cond_proj_dim
        self.img_size = args.img_size
        self.model = model.eval()
        self.scheduler = LCMScheduler(
                            beta_schedule = noise_scheduler.config.beta_schedule, 
                            beta_start = noise_scheduler.beta_start, 
                            beta_end = noise_scheduler.beta_end, 
                            clip_sample = True
                        )
        self.x_shape = (n_samples, 3, self.img_size, self.img_size)
        # self.guidance_scale = 
        self.device = device

    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles custom timesteps. 
        Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. 
                If `timesteps` is passed, `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. 
                If `sigmas` is passed, `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the second element is the number of inference steps.

        """
        
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            self.scheduler.set_timesteps(timesteps=timesteps, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
        return timesteps, num_inference_steps
    
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 256, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
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
    
    @torch.no_grad()
    def forward(self,
                c1: torch.LongTensor, 
                c2: torch.LongTensor,
                x_t: Optional[torch.Tensor] = None,
                num_inference_steps: int = 4,
                timesteps: List[int] = None,
                height:Optional[int] = 64,
                width: Optional[int] = 64,
                guidance_scale: float = 3.5,
                num_images_per_cond: Optional[int] = 1,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                return_dict: bool = False,
                ):
        """
        Args:
            c1 (`torch.LongTensor`): 
                attribute condition.
            c2 (`torch.LongTensor`): 
                object condition.
            x_t (`torch.Tensor`): 
                Noisy images at timestep t with shape (batch_size, channels, height, width).
            num_inference_steps (`int`, *optional*, defaults to 4):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to 64):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 64):
                The width in pixels of the generated image.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                scheduler's `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps on the original LCM training/distillation timestep schedule are used. Must be in descending
                order.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                Note that the original latent consistency models paper uses a different CFG formulation where the
                guidance scales are decreased by 1 (so in the paper formulation CFG is enabled when `guidance_scale >
                0`).
            num_images_per_cond (`int`, *optional*, defaults to 1):
                The number of images to generate per condition.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`] to make generation deterministic.

        Returns:
            will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.

        """
        # 0. Default height and width to unet
        # height = height or args.img_size
        # width = width or args.img_size
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps, timesteps)
        # 5. Prepare noise
        if x_t is None:
            if generator is None:
                print("No noise or seed provided for sampler, going to sample random noises.")
            x_t = randn_tensor(self.x_shape, generator=generator, device=self.device, dtype=torch.float32)
            x_t = x_t * self.scheduler.init_noise_sigma
        
        else:
            x_t = randn_tensor(self.x_shape, generator=generator, device=self.device, dtype=torch.float32) if generator else x_t
            
        
        bs = self.x_shape[0] * num_images_per_cond
        
        # 6. Get Guidance Scale Embedding
        # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
        # CFG formulation, so we need to subtract 1 from the input guidance_scale.
        # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
        w = torch.tensor(guidance_scale - 1).repeat(bs)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.time_cond_proj_dim).to(
            device=self.device, dtype=x_t.dtype
        )
        
        # 8. LCM MultiStep Sampling Loop: implements Algorithm 1 in the paper
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        
        with tqdm(reversed(range(num_inference_steps)), colour="#6565b5", total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                x_t = x_t.to(dtype=torch.float32)
                
                # model prediction (eps)
                noise_pred = self.model(
                    x_t,
                    t,
                    c1,
                    c2,
                    timestep_cond=w_embedding,
                )
                # compute the previous noisy sample x_t -> x_t-1
                x_t, denoised = self.scheduler.step(noise_pred, t, x_t, return_dict=False)
                
                if i == len(timesteps)-1 or ((i+1) > num_warmup_steps and (i+1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        image = denormalize(denoised)       
        
        return image