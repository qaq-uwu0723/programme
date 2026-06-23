"""Causal Transformer for temporal trend extraction.

X = S + R decomposition: the trend module learns the smooth backbone S,
and residual R is handed off to the DDPM for distributional refinement.
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, additive to input features."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerTrend(nn.Module):
    """Causal Transformer for temporal trend extraction.

    Input:  X of shape (B, L, d_c)
    Output: S of shape (B, L, d_c) — the predicted trend

    Uses causal (triangular) attention so position t can only attend to [0..t].
    Trained with teacher forcing: predict X_{t+1} from X_{0..t}, loss = MSE.
    """

    def __init__(
        self,
        d_c: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.d_c = d_c
        self.d_model = d_model

        self.input_proj = nn.Linear(d_c, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_c)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: position t cannot attend to > t."""
        return torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal attention.

        Args:
            x: (B, L, d_c) input continuous features

        Returns:
            s: (B, L, d_c) trend prediction (same length as input)
        """
        B, L, _ = x.shape
        h = self.input_proj(x)          # (B, L, d_model)
        h = self.pos_enc(h)
        mask = self._causal_mask(L, x.device)
        h = self.encoder(h, mask=mask)   # (B, L, d_model)
        s = self.output_proj(h)          # (B, L, d_c)
        return s

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing MSE loss.

        Predicts each x_t from x_{0..t-1}. The causal mask handles the
        temporal ordering; we shift the target by 1 to create the
        next-step prediction task.

        Args:
            x: (B, L, d_c) input continuous features

        Returns:
            loss: scalar MSE
        """
        s_hat = self.forward(x)         # (B, L, d_c)
        # Input positions 0..L-1, targets are positions 1..L
        # So s_hat[:, :-1, :] should predict x[:, 1:, :]
        pred = s_hat[:, :-1, :]          # (B, L-1, d_c)
        target = x[:, 1:, :]             # (B, L-1, d_c)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def generate_trend(self, seed: torch.Tensor, total_len: int) -> torch.Tensor:
        """Autoregressively roll out a trend sequence.

        Args:
            seed: (B, seed_len, d_c) initial context
            total_len: desired total output length L

        Returns:
            s: (B, total_len, d_c) full trend sequence
        """
        B, seed_len, d_c = seed.shape
        s = torch.zeros(B, total_len, d_c, device=seed.device, dtype=seed.dtype)
        s[:, :seed_len] = seed

        for t in range(seed_len, total_len):
            # Use all past context (causal attention handles masking)
            s_t = self.forward(s[:, :t, :])[:, -1:, :]
            s[:, t:t + 1] = s_t

        return s
