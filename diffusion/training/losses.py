"""Loss functions for Mask-DDPM training."""
from typing import List, Optional
import torch
import torch.nn.functional as F


def epsilon_mse(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """Standard epsilon-prediction MSE loss for DDPM."""
    return F.mse_loss(eps_pred, eps_true)


def weighted_epsilon_mse(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    k: torch.Tensor,
    alpha_bars: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Min-SNR weighted epsilon-prediction MSE.

    w_k = min(SNR_k, gamma) / SNR_k
    """
    eps = 1e-8
    snr = alpha_bars[k] / (1.0 - alpha_bars[k] + eps)
    w = torch.clamp(snr, max=gamma) / (snr + eps)  # (B,)
    # Expand weight to match tensor dimensions
    while w.dim() < eps_pred.dim():
        w = w.unsqueeze(-1)
    return (w * F.mse_loss(eps_pred, eps_true, reduction="none")).mean()


def masked_cross_entropy(
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy loss computed only on masked positions.

    Args:
        logits_list: list of (B, L, |V_j|) logits per discrete variable
        targets_list: list of (B, L) target token indices
        mask_list: list of (B, L) boolean masks (True = position was masked)

    Returns:
        scalar average CE loss over all masked positions
    """
    total_loss = 0.0
    total_count = 0
    for logits, targets, mask in zip(logits_list, targets_list, mask_list):
        if mask.sum() == 0:
            continue
        logits_masked = logits[mask]      # (N_masked, |V_j|)
        targets_masked = targets[mask]     # (N_masked,)
        total_loss += F.cross_entropy(logits_masked, targets_masked, reduction="sum")
        total_count += mask.sum().item()
    if total_count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return total_loss / total_count


def combined_loss(
    loss_cont: torch.Tensor,
    loss_disc: torch.Tensor,
    lambda_bal: float,
) -> torch.Tensor:
    """Weighted sum of continuous and discrete losses.

    L = lambda * L_cont + (1 - lambda) * L_disc
    """
    return lambda_bal * loss_cont + (1.0 - lambda_bal) * loss_disc


def quantile_loss(
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    n_quantiles: int = 20,
) -> torch.Tensor:
    """Quantile alignment loss: L1 distance between real and generated quantiles.

    Helps reduce KS by aligning distribution tails.
    """
    qs = torch.linspace(0.05, 0.95, n_quantiles, device=x_real.device)
    loss = 0.0
    for q in qs:
        q_real = torch.quantile(x_real, q.item())
        q_gen = torch.quantile(x_gen, q.item())
        loss += F.l1_loss(q_gen, q_real)
    return loss / n_quantiles


def stat_loss(
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
) -> torch.Tensor:
    """Mean/std alignment to prevent residual distribution collapse."""
    return (
        F.mse_loss(x_gen.mean(), x_real.mean())
        + F.mse_loss(x_gen.std(), x_real.std())
    )
