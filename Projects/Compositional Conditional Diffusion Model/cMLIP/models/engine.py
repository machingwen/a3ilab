from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


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

    def forward(self, x_0):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t)

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
        elif scheduler == "cosine":
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
    def cal_mean_variance(self, x_t, t, label, atr):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$
        """
        epsilon_theta = self.model(x_t, t, label, atr)
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta

        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, label, atr, time_step: int):
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
        elif scheduler == "cosine":
            beta_t = betas_for_alpha_bar(num_diffusion_timesteps=T)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, label: torch.LongTensor, atr: torch.LongTensor, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict conditional noise and unconditional noise using model
        epsilon_theta_t = self.model(x_t, t, label, atr)
        unc_epsilon_theta_t = self.model(x_t, t, label, atr, force_drop_ids=True)
        
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
    def forward(self, x_t, label: torch.LongTensor, atr: torch.LongTensor, steps: int = 1, method="linear", eta=0.0,
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
                x_t = self.sample_one_step(x_t, time_steps[i], label, atr, time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]