"""DDPM for continuous residual generation, conditioned on trend S.

ResidualDDPM models R = X - S via:
- Forward: r_k = sqrt(alpha_bar_k)*r_0 + sqrt(1-alpha_bar_k)*epsilon
- Reverse: denoiser predicts epsilon, DDPM update step
- Optional Min-SNR weighting for training stability
"""
from typing import Tuple
import torch
import torch.nn as nn

from .denoiser import TransformerDenoiser
from ..utils.noise_schedule import make_beta_schedule, get_alpha_bars


class ResidualDDPM(nn.Module):
    """Denoising diffusion on the residual R = X - S.

    Takes a (noised residual, timestep, trend condition S) and predicts the noise.
    """

    def __init__(
        self,
        d_c: int,
        d_cond: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        diffusion_steps: int = 600,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.d_c = d_c
        self.K = diffusion_steps

        # Noise schedule
        self.register_buffer("betas", make_beta_schedule(
            diffusion_steps, beta_schedule, beta_start, beta_end))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", get_alpha_bars(self.betas))

        # Denoiser backbone
        self.denoiser = TransformerDenoiser(
            d_in=d_c,
            d_cond=d_cond,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_head = nn.Linear(d_model, d_c)

    def forward_noise(
        self, r0: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to the clean residual (forward diffusion).

        Args:
            r0: (B, L, d_c) clean residual
            k:  (B,) timestep indices

        Returns:
            r_k: (B, L, d_c) noised residual
            eps: (B, L, d_c) the noise that was added
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[k])[:, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bars[k])[:, None, None]
        eps = torch.randn_like(r0)
        r_k = sqrt_alpha_bar * r0 + sqrt_one_minus * eps
        return r_k, eps

    def forward(
        self, r0: torch.Tensor, s_cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass: sample timesteps, noise, predict epsilon.

        Args:
            r0:     (B, L, d_c) clean residual
            s_cond: (B, L, d_cond) trend conditioning signal

        Returns:
            loss: scalar MSE
            k: timestep indices used (for optional external weighting)
            eps_pred: predicted noise (for combined loss weighting)
        """
        B = r0.shape[0]
        k = torch.randint(0, self.K, (B,), device=r0.device)
        r_k, eps = self.forward_noise(r0, k)
        h = self.denoiser(r_k, k, s_cond)
        eps_pred = self.output_head(h)
        loss = nn.functional.mse_loss(eps_pred, eps)
        return loss, k, eps_pred

    @torch.no_grad()
    def sample(self, s_cond: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion: sample residual from noise.

        Args:
            s_cond: (B, L, d_cond) trend conditioning

        Returns:
            r_hat: (B, L, d_c) sampled residual
        """
        B, L, _ = s_cond.shape
        device = s_cond.device
        r = torch.randn(B, L, self.d_c, device=device)

        for k in reversed(range(self.K)):
            k_t = torch.full((B,), k, device=device, dtype=torch.long)
            h = self.denoiser(r, k_t, s_cond)
            eps_pred = self.output_head(h)

            alpha = self.alphas[k]
            alpha_bar = self.alpha_bars[k]
            beta = self.betas[k]

            # DDPM reverse update
            coef = 1.0 / torch.sqrt(alpha)
            drift = beta / torch.sqrt(1.0 - alpha_bar)
            mean = coef * (r - drift * eps_pred)

            if k > 0:
                noise = torch.randn_like(r)
                var = torch.sqrt(beta) * noise
            else:
                var = 0.0
            r = mean + var

        return r
