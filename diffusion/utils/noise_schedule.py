"""Noise schedules for DDPM (beta) and masked diffusion (mask probability)."""
import torch
import math


def make_beta_schedule(
    steps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Build a beta noise schedule.

    Args:
        steps: number of diffusion steps K
        schedule: "linear" or "cosine"
        beta_start: minimum beta
        beta_end: maximum beta (for linear)

    Returns:
        betas: (K,) float tensor
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, steps)
    elif schedule == "cosine":
        # Cosine schedule from Nichol & Dhariwal (2021)
        s = 0.008
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        ft = torch.cos(((t / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bars = ft[1:] / ft[0]  # ratio relative to f(0)
        # Clip to ensure numerical stability
        betas = 1.0 - alpha_bars / torch.clamp(ft[:-1] / ft[0], min=1e-8)
        return torch.clamp(betas.float(), max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def make_mask_schedule(
    steps: int,
    schedule: str = "linear",
) -> torch.Tensor:
    """Build a mask probability schedule for discrete diffusion.

    Args:
        steps: number of diffusion steps K
        schedule: "linear" or "cosine"

    Returns:
        mask_probs: (K,) float tensor, m_k ∈ [0, 1], increasing
    """
    if schedule == "linear":
        return torch.linspace(0.0, 1.0, steps)
    elif schedule == "cosine":
        t = torch.linspace(0, steps, steps, dtype=torch.float64)
        return (1.0 - torch.cos(t / steps * math.pi * 0.5)).float()
    else:
        raise ValueError(f"Unknown mask schedule: {schedule}")


def get_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of alphas from betas.

    Args:
        betas: (K,) noise schedule

    Returns:
        alpha_bars: (K,) cumulative product of (1 - beta_i)
    """
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def compute_snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    """Compute SNR_k = alpha_bar_k / (1 - alpha_bar_k)."""
    return alpha_bars / (1.0 - alpha_bars)


def min_snr_weight(
    k: torch.Tensor,
    alpha_bars: torch.Tensor,
    gamma: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-SNR loss weighting: w_k = min(SNR_k, gamma) / SNR_k.

    Args:
        k: (B,) timestep indices
        alpha_bars: (K,) cumulative alpha product
        gamma: clip threshold
        eps: numerical stability

    Returns:
        weights: (B,) per-sample weights
    """
    snr = alpha_bars[k] / (1.0 - alpha_bars[k] + eps)
    w = torch.clamp(snr, max=gamma) / (snr + eps)
    return w
