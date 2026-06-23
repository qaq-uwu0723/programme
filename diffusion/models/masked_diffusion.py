"""Masked (absorbing-state) diffusion for discrete Modbus variables.

Forward process: randomly replace tokens with [MASK] according to schedule m_k.
Reverse process: denoiser predicts original tokens for masked positions (CE loss).
Sampling: start from all [MASK], iteratively unmask.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoiser import TransformerDenoiser
from ..utils.noise_schedule import make_mask_schedule


class MaskedDiffusion(nn.Module):
    """Discrete diffusion via random masking / unmasking.

    Each discrete variable j has vocabulary V_j. The special [MASK] token is
    assigned index |V_j| (one past the last valid token).
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        d_cond: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        diffusion_steps: int = 600,
        mask_schedule: str = "linear",
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes      # list of |V_j|
        self.K = diffusion_steps
        self.d_d = len(vocab_sizes)

        # Mask probability schedule: m_k ∈ [0, 1], increasing with k
        self.register_buffer("mask_probs", make_mask_schedule(diffusion_steps, mask_schedule))

        # Per-variable embeddings (vocab_size + 1 for [MASK] token)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs + 1, d_model, padding_idx=vs)  # [MASK] is at index vs
            for vs in vocab_sizes
        ])

        # Denoiser: takes concatenated embeddings, predicts original tokens
        d_in = d_model * self.d_d
        self.denoiser = TransformerDenoiser(
            d_in=d_in,
            d_cond=d_cond,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # Per-variable output heads: predict logits over vocabulary
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vs) for vs in vocab_sizes
        ])

    def _embed(self, y_list: List[torch.Tensor]) -> torch.Tensor:
        """Embed each discrete variable and concatenate along feature dim.

        Args:
            y_list: list of (B, L) tensors (token indices, [MASK] = vocab_size)

        Returns:
            (B, L, d_model * d_d) concatenated embeddings
        """
        embeds = []
        for y, emb_layer in zip(y_list, self.embeddings):
            embeds.append(emb_layer(y))  # (B, L, d_model)
        return torch.cat(embeds, dim=-1)  # (B, L, d_model * d_d)

    def forward_mask(
        self, y0: List[torch.Tensor], k: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward corruption: independently mask tokens with probability m_k.

        Args:
            y0: list of (B, L) clean token sequences
            k:  (B,) diffusion timestep indices

        Returns:
            y_masked: list of (B, L) with some tokens replaced by [MASK]
            mask_pos: list of (B, L) boolean (True = position was masked)
        """
        B = y0[0].shape[0]
        L = y0[0].shape[1]
        device = y0[0].device
        m_k = self.mask_probs[k]  # (B,)

        y_masked = []
        mask_pos = []
        for y, vs in zip(y0, self.vocab_sizes):
            mask_prob = m_k[:, None, None].expand(B, L, 1)  # (B, L, 1)
            rand = torch.rand(B, L, 1, device=device)
            is_masked = rand < mask_prob
            ym = y.clone()
            ym[is_masked.squeeze(-1)] = vs  # [MASK] = vocab_size
            y_masked.append(ym)
            mask_pos.append(is_masked.squeeze(-1))  # (B, L) bool
        return y_masked, mask_pos

    def forward(
        self,
        y0: List[torch.Tensor],
        s_cond: torch.Tensor,
        x_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward: mask tokens, predict originals, compute CE loss.

        Args:
            y0:     list of (B, L) clean discrete sequences
            s_cond: (B, L, d_cond) trend conditioning
            x_hat:  (B, L, d_c) optional generated continuous for cross-conditioning

        Returns:
            loss: scalar cross-entropy averaged over masked positions
        """
        B = y0[0].shape[0]
        k = torch.randint(0, self.K, (B,), device=y0[0].device)

        y_masked, mask_pos = self.forward_mask(y0, k)
        y_embed = self._embed(y_masked)

        # Build conditioning: trend + optional generated continuous
        if x_hat is not None:
            cond = torch.cat([s_cond, x_hat], dim=-1)
        else:
            cond = s_cond

        h = self.denoiser(y_embed, k, cond)  # (B, L, d_model)

        # Compute CE loss per variable, only on masked positions
        total_loss = 0.0
        total_count = 0
        for j, head in enumerate(self.output_heads):
            logits = head(h)               # (B, L, |V_j|)
            mask = mask_pos[j]             # (B, L)
            if mask.sum() == 0:
                continue
            logits_m = logits[mask]         # (N_masked, |V_j|)
            targets_m = y0[j][mask]         # (N_masked,)
            total_loss += F.cross_entropy(logits_m, targets_m, reduction="sum")
            total_count += mask.sum().item()

        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        s_cond: torch.Tensor,
        x_hat: Optional[torch.Tensor] = None,
        num_unmask_steps: int = 50,
    ) -> List[torch.Tensor]:
        """Generate discrete sequences by iterative unmasking.

        Strategy: start all-mask, at each reverse step predict the most
        confident unmasked tokens and commit them.

        Args:
            B: batch size
            L: sequence length
            s_cond: (B, L, d_cond) trend conditioning
            x_hat: (B, L, d_c) optional continuous context
            num_unmask_steps: how many reverse steps (subset of K)

        Returns:
            list of (B, L) discrete sequences (token indices)
        """
        device = s_cond.device

        # Start from all-mask
        y_list = [
            torch.full((B, L), vs, device=device, dtype=torch.long)
            for vs in self.vocab_sizes
        ]

        # Build conditioning once
        if x_hat is not None:
            cond = torch.cat([s_cond, x_hat], dim=-1)
        else:
            cond = s_cond

        # Reverse steps: map k_rev ∈ [0, num_unmask_steps) → actual K steps
        step_indices = torch.linspace(self.K - 1, 0, num_unmask_steps, dtype=torch.long)

        for step_idx in range(num_unmask_steps):
            k = step_indices[step_idx]
            k_t = torch.full((B,), k.item(), device=device, dtype=torch.long)
            m_k = self.mask_probs[k].item()

            # Current mask positions
            mask_now = [(y == vs) for y, vs in zip(y_list, self.vocab_sizes)]

            if not any(m.any() for m in mask_now):
                break  # all tokens unmasked

            # Predict
            y_embed = self._embed(y_list)
            h = self.denoiser(y_embed, k_t, cond)

            for j, head in enumerate(self.output_heads):
                logits = head(h)           # (B, L, |V_j|)
                probs = F.softmax(logits, dim=-1)
                max_prob, pred_token = probs.max(dim=-1)  # (B, L)

                # Only update positions that are currently masked
                m = mask_now[j]
                if not m.any():
                    continue

                # Unmask a fraction of positions determined by the schedule
                # At step k, we expect m_k fraction still masked after this step.
                # The number to unmask now is proportional to the schedule.
                num_masked = m.sum().item()
                # Target: after this step, we want m_{k-1} fraction masked
                # (or 0 if k is the first reverse step)
                target_masked_frac = self.mask_probs[k - 1].item() if k > 0 else 0.0
                num_to_keep = max(0, int(target_masked_frac * B * L))
                num_to_unmask = max(1, num_masked - num_to_keep)

                # Select positions with highest prediction confidence
                conf_masked = max_prob.clone()
                conf_masked[~m] = -float("inf")
                _, top_indices = conf_masked.reshape(-1).topk(
                    min(num_to_unmask, num_masked)
                )

                # Flatten and update
                flat_y = y_list[j].reshape(-1)
                flat_y[top_indices] = pred_token.reshape(-1)[top_indices]
                y_list[j] = flat_y.reshape(B, L)

        return y_list
