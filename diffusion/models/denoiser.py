"""Shared Transformer denoiser backbone used by both ResidualDDPM and MaskedDiffusion.

Architecture: input proj → timestep embedding inject → condition inject →
               bidirectional Transformer encoder → hidden output.
"""
import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding, projects to d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Convert integer timesteps to sinusoidal embeddings.

        Args:
            t: (B,) long tensor of timestep indices

        Returns:
            emb: (B, d_model) float tensor
        """
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


import torch.nn.functional as F


class TransformerDenoiser(nn.Module):
    """Bidirectional Transformer denoising backbone.

    Shared by ResidualDDPM and MaskedDiffusion. Both need:
    - timestep conditioning via sinusoidal embedding
    - trend conditioning signal injection
    - bidirectional attention over the full window
    - output hidden states for task-specific heads

    The only difference between DDPM and Mask use is the input dimension (d_in)
    and the output head attached after this backbone.
    """

    def __init__(
        self,
        d_in: int,
        d_cond: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)
        self.cond_proj = nn.Linear(d_cond, d_model) if d_cond != d_model else nn.Identity()
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Denoiser forward pass.

        Args:
            x:    (B, L, d_in)  noisy/masked input
            t:    (B,)          diffusion timestep indices
            cond: (B, L, d_cond) conditioning signal (trend, optionally + X_hat)

        Returns:
            h: (B, L, d_model) hidden representation for downstream heads
        """
        h = self.input_proj(x)                          # (B, L, d_model)
        t_emb = self.time_embed(t).unsqueeze(1)          # (B, 1, d_model)
        c = self.cond_proj(cond)                          # (B, L, d_model)
        h = h + t_emb + c                                 # inject both
        h = self.encoder(h)                               # (B, L, d_model)
        return h
