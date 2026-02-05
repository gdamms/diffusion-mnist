"""
Diffusion process utilities.
Contains forward and reverse diffusion functions.
"""

import torch
from .config import DEVICE, ALPHA, ALPHA_BAR, BETA, DIFFU_STEPS


def q_xt_xt_1(xt_1: torch.Tensor, t: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion step: q(x_t | x_{t-1}).
    Adds noise to image at step t-1 to get image at step t.

    Args:
        xt_1: Image at timestep t-1 [B, C, H, W]
        t: Timestep (int or tensor)

    Returns:
        xt: Noisy image at timestep t
        eps: The noise that was added
    """
    if isinstance(t, int):
        t_ind = torch.tensor(t, dtype=torch.long, device=DEVICE)
    else:
        t_ind = t.to(dtype=torch.long, device=DEVICE)

    alpha = ALPHA[t_ind]
    mean = torch.sqrt(alpha) * xt_1
    std = torch.sqrt(1 - alpha)

    eps = torch.randn(xt_1.shape, device=DEVICE)
    xt = mean + std * eps

    return xt, eps


def q_xt_x0(x0: torch.Tensor, t: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion: q(x_t | x_0).
    Directly compute noisy image at any timestep t from clean image x0.

    Args:
        x0: Clean image [B, C, H, W] or [C, H, W]
        t: Timestep tensor [B, 1] or [B]

    Returns:
        xt: Noisy image at timestep t
        eps: The noise that was added
    """
    if isinstance(t, int):
        t_ind = torch.tensor(t, dtype=torch.long, device=DEVICE)
    else:
        t_ind = t.to(dtype=torch.long, device=DEVICE)

    # Handle both batched and single images
    reshaped = len(x0.shape) == 3
    if reshaped:
        c, w, h = x0.shape
        b = 1
        x0 = x0.view(b, c, w, h)
    else:
        b, c, w, h = x0.shape

    # Reshape t for broadcasting
    t_ind = t_ind.view(b, 1, 1, 1)
    t_ind = t_ind.expand(b, c, w, h)

    alpha_bar = ALPHA_BAR[t_ind]
    mean = torch.sqrt(alpha_bar) * x0
    std = torch.sqrt(1 - alpha_bar)

    eps = torch.randn(x0.shape, device=DEVICE)
    xt = mean + std * eps

    if reshaped:
        xt = xt.view(c, w, h)

    return xt, eps


def p_xt_1_xt(model: torch.nn.Module, xt: torch.Tensor, t: torch.Tensor,
              vec: torch.Tensor) -> torch.Tensor:
    """
    Reverse diffusion step: p(x_{t-1} | x_t).
    Denoise image at step t to get image at step t-1.
    Model predicts the noise.

    Args:
        model: UNet model that predicts noise
        xt: Noisy image at timestep t [B, C, H, W]
        t: Timestep tensor [B, 1]
        vec: Label one-hot vector [B, NB_LABEL]

    Returns:
        xt_1: Denoised image at timestep t-1
    """
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar_t = ALPHA_BAR[t_ind].view(-1, 1, 1, 1)
    alpha_bar_t_1 = ALPHA_BAR[t_ind - 1].view(-1, 1, 1, 1)
    alpha_t = ALPHA[t_ind].view(-1, 1, 1, 1)
    beta_t = BETA[t_ind].view(-1, 1, 1, 1)

    beta_tilde = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

    # Model predicts the noise
    epsilon_theta = model(xt, t, vec)

    sigma_theta = torch.sqrt(beta_tilde)
    mu_theta = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_t)

    # Don't add noise at t=1
    mask_t0 = (t > 1).to(dtype=torch.float32).view(-1, 1, 1, 1)
    noise = torch.randn(xt.shape, device=DEVICE) * mask_t0

    return mu_theta + sigma_theta * noise


def p_xt_1_xt_x0_pred(model: torch.nn.Module, xt: torch.Tensor, t: torch.Tensor,
                      vec: torch.Tensor) -> torch.Tensor:
    """
    Reverse diffusion step where model predicts x0 directly.

    Args:
        model: UNet model that predicts clean image x0
        xt: Noisy image at timestep t [B, C, H, W]
        t: Timestep tensor [B, 1]
        vec: Label one-hot vector [B, NB_LABEL]

    Returns:
        xt_1: Denoised image at timestep t-1
    """
    x0_pred = model(xt, t, vec)
    xt_1, _ = q_xt_x0(x0_pred, t - 1)
    return xt_1


def forward_diffusion(x0: torch.Tensor) -> list[torch.Tensor]:
    """
    Run full forward diffusion process.

    Args:
        x0: Clean image [C, H, W] or [B, C, H, W]

    Returns:
        List of images at each timestep [x0, x1, ..., xT]
    """
    x = x0.clone()
    xs = [x]
    for t in range(1, DIFFU_STEPS + 1):
        x, _ = q_xt_xt_1(x, t)
        xs.append(x)
    return xs
