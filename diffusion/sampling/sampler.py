"""Mask-DDPM multi-stage generation pipeline.

1. Trend rollout: autoregressively generate S_hat from TransformerTrend
2. Residual sampling: DDPM reverse diffusion → R_hat
3. Continuous assembly: X_hat = S_hat + R_hat
4. Discrete sampling: Masked diffusion reverse unmasking → Y_hat
5. Type-aware post-processing
6. Denormalization
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..models.trend_transformer import TransformerTrend
from ..models.residual_ddpm import ResidualDDPM
from ..models.masked_diffusion import MaskedDiffusion
from ..utils.normalisation import Normalizer
from extractor.schema import FeatureSchema


# ---------------------------------------------------------------------------
# Conditional payload lookup table
# ---------------------------------------------------------------------------

class StubSampler:
    """Empirical distribution sampler for continuous features unsuitable for DDPM.

    Stores training values per feature and can generate new values by random
    sampling from the empirical distribution. Used for:
    - Low-cardinality features (e.g. 3-value setpoints) that Gaussian DDPM cannot model
    - Dead features (std≈0) filled with training mean

    The replacement indices are stored internally and applied independently of
    the Schema's TypeRouter classification (which controls training, not sampling).
    """

    def __init__(self):
        self._distributions: Dict[int, np.ndarray] = {}  # feature_index → values
        self._indices: List[int] = []  # features to replace during sampling

    def fit(self, X_cont_raw: np.ndarray, stub_indices: List[int]) -> "StubSampler":
        """Store empirical distributions for features to replace post-hoc.

        Args:
            X_cont_raw: (N, d_c) continuous features in RAW units
            stub_indices: list of feature indices to sample from
        """
        if X_cont_raw.ndim == 3:
            X_cont_raw = X_cont_raw.reshape(-1, X_cont_raw.shape[-1])
        for idx in stub_indices:
            if idx < X_cont_raw.shape[1]:
                self._distributions[idx] = X_cont_raw[:, idx].copy()
        self._indices = stub_indices
        return self

    def sample(self, feature_idx: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample from the empirical distribution for a feature.

        Args:
            feature_idx: which continuous feature to sample
            shape: output tensor shape e.g. (B, L)

        Returns:
            tensor of sampled values in RAW units
        """
        if feature_idx in self._distributions and len(self._distributions[feature_idx]) > 0:
            vals = self._distributions[feature_idx]
            samples = np.random.choice(vals, size=shape)
            return torch.from_numpy(samples).float()
        return torch.zeros(shape, dtype=torch.float32)


class PayloadLookup:
    """Conditional payload_size distribution lookup.

    Builds a mapping from (function_code, direction, quantity) → list of payload
    values observed in the training data. During sampling, for each generated
    (function_code, direction, quantity), we randomly sample from the
    corresponding list to produce a realistic payload_size distribution.

    quantity is included because Modbus payload size depends on register count:
    e.g. FC3 response = 3 + quantity × 2 bytes.
    """

    def __init__(self):
        self._table: Dict[Tuple[int, int, int], List[float]] = {}

    def fit(
        self,
        X_cont: np.ndarray,
        Y_disc: np.ndarray,
        fc_vocab: List[int] = None,
    ) -> "PayloadLookup":
        """Build conditional distributions from training data.

        Args:
            X_cont: (N, d_c) or (N, L, d_c) continuous features (RAW units)
            Y_disc: (N, d_d) or (N, L, d_d) discrete features (vocabulary indices)
            fc_vocab: function code vocabulary list
        """
        if fc_vocab is None:
            fc_vocab = [1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 17, 43]
        if X_cont.ndim == 3:
            X_cont = X_cont.reshape(-1, X_cont.shape[-1])
        if Y_disc.ndim == 3:
            Y_disc = Y_disc.reshape(-1, Y_disc.shape[-1])

        payload_idx = 4   # C_PAYLOAD_SIZE
        quantity_idx = 6   # C_QUANTITY
        for i in range(Y_disc.shape[0]):
            fc_idx = int(Y_disc[i, 0])
            if fc_idx >= len(fc_vocab):
                continue
            fc = fc_vocab[fc_idx]
            direction = int(Y_disc[i, 1])
            quantity = int(X_cont[i, quantity_idx])
            key = (fc, direction, quantity)
            ps = float(X_cont[i, payload_idx])
            self._table.setdefault(key, []).append(ps)
        return self

    def sample(
        self, fc: torch.Tensor, direction: torch.Tensor, quantity: torch.Tensor
    ) -> torch.Tensor:
        """Sample payload_size given function_code, direction, and quantity.

        Args:
            fc: (B, L) function code values (actual codes, not vocab indices)
            direction: (B, L) direction (0=c2s, 1=s2c)
            quantity: (B, L) quantity values (register count)

        Returns:
            payload: (B, L) sampled payload values in raw units
        """
        device = fc.device
        shape = fc.shape
        payload = torch.zeros(shape, device=device, dtype=torch.float32)

        unique_keys = set()
        for b in range(shape[0]):
            for t in range(shape[1]):
                key = (
                    int(fc[b, t].item()),
                    int(direction[b, t].item()),
                    int(quantity[b, t].item()),
                )
                unique_keys.add(key)

        cache: Dict[Tuple[int, int, int], float] = {}
        for key in unique_keys:
            if key in self._table and len(self._table[key]) > 0:
                cache[key] = float(np.random.choice(self._table[key]))
            else:
                # fallback: infer from protocol rules when lookup misses
                fc_val, direction_val, qty_val = key
                if not hasattr(self, '_fallback_cache'):
                    self._fallback_cache = {}
                if key not in self._fallback_cache:
                    self._fallback_cache[key] = self._compute_fallback(
                        fc_val, direction_val, qty_val
                    )
                cache[key] = self._fallback_cache[key]

        for b in range(shape[0]):
            for t in range(shape[1]):
                key = (
                    int(fc[b, t].item()),
                    int(direction[b, t].item()),
                    int(quantity[b, t].item()),
                )
                payload[b, t] = cache[key]

        return payload

    @staticmethod
    def _compute_fallback(fc: int, direction: int, quantity: int) -> float:
        """Protocol-aware fallback when no matching sample exists in training data."""
        if direction == 0:  # request
            if fc in (1, 2, 3, 4):     return 6.0   # read: MBAP(7) + func(1) + addr(2) + qty(2) = 12 → wait, ADU only
            if fc in (5, 6):            return 6.0   # write single
            if fc in (15, 16):          return 7.0 + quantity * 2  # write multiple
            return 6.0
        else:  # response
            if fc in (1, 2, 3, 4):      return 3.0 + quantity * 2  # read response: func(1) + byte_count(1) + regs(qty*2)
            if fc in (5, 6):            return 6.0   # write single ack
            if fc in (15, 16):          return 6.0   # write multiple ack
            return 6.0


class MaskDDPMSampler:
    """End-to-end sampling from the trained Mask-DDPM pipeline."""

    def __init__(
        self,
        trend_model: TransformerTrend,
        ddpm: ResidualDDPM,
        mask_diff: MaskedDiffusion,
        normalizer: Normalizer,
        schema: FeatureSchema,
        payload_lookup: Optional[PayloadLookup] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.trend_model = trend_model.to(device).eval()
        self.ddpm = ddpm.to(device).eval()
        self.mask_diff = mask_diff.to(device).eval()
        self.normalizer = normalizer
        self.schema = schema
        self.payload_lookup = payload_lookup
        self.stub_sampler: Optional[StubSampler] = None
        self.device = device

        self.L = schema.window_length

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seed_seq: Optional[torch.Tensor] = None,
        num_unmask_steps: int = 50,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate synthetic feature windows.

        Args:
            num_samples: number of windows to generate (B)
            seed_seq: optional (B, seed_len, d_c) to prime the trend model
            num_unmask_steps: reverse steps for masked diffusion

        Returns:
            X_hat: (B, L, d_c) generated continuous features (denormalized, ALL features)
            Y_hat: list of (B, L) generated discrete features
        """
        B = num_samples
        device = self.device
        d_c_active = self.ddpm.d_c  # only DDPM-routed features
        active_mask = self._get_active_mask(device)

        # --- Step 1: Trend rollout (active features only) ---
        if seed_seq is not None:
            seed = seed_seq[:, :, active_mask].to(device) if seed_seq.shape[-1] != d_c_active else seed_seq.to(device)
            S_hat = self.trend_model.generate_trend(seed, self.L)
        else:
            seed = torch.randn(B, 1, d_c_active, device=device) * 0.1
            S_hat = self.trend_model.generate_trend(seed, self.L)

        # --- Step 2: Residual sampling ---
        R_hat = self.ddpm.sample(S_hat)

        # --- Step 3: Continuous assembly (active only) ---
        X_active_norm = S_hat + R_hat  # (B, L, d_c_active)

        # --- Step 4: Discrete sampling ---
        Y_hat = self.mask_diff.sample(
            B, self.L, S_hat, x_hat=X_active_norm,
            num_unmask_steps=num_unmask_steps,
        )

        # --- Step 5: Reconstruct full feature vector (all d_c features) ---
        X_hat_norm_full = self._build_full_tensor(X_active_norm, active_mask, B, self.L, device)

        # --- Step 6: Fill quantity (Type6, needed for payload_size lookup) ---
        quantity_idx = 6
        quantity_raw = None
        if self.stub_sampler is not None and quantity_idx in self.stub_sampler._indices:
            shape = (B, self.L)
            quantity_raw = self.stub_sampler.sample(quantity_idx, shape).to(device)
            # Put quantity into z-scored space for consistency
            mean_q = self.normalizer.mean[quantity_idx]
            std_q = self.normalizer.std[quantity_idx]
            X_hat_norm_full[:, :, quantity_idx] = (quantity_raw.float() - mean_q) / std_q.clamp(min=1e-8)
        else:
            # fallback: denormalize whatever is there (DDPM-generated)
            mean_q = self.normalizer.mean[quantity_idx]
            std_q = self.normalizer.std[quantity_idx]
            quantity_raw = X_hat_norm_full[:, :, quantity_idx] * std_q + mean_q

        # --- Step 7: Fill Type5 (payload_size) using (fc, direction, quantity) ---
        X_hat_norm_full = self._fill_payload_size(
            X_hat_norm_full, Y_hat, quantity_raw, device
        )

        # --- Step 8: Denormalize all features together ---
        X_hat = self.normalizer.inverse_transform(X_hat_norm_full)

        # --- Step 9: Fill remaining Type6 stub features (skip quantity = already filled) ---
        if self.stub_sampler is not None:
            X_hat = self._fill_stub_features(X_hat, device, skip_indices={6})

        # --- Step 9b: Inverse log transform for log1p-compressed features ---
        X_hat = self._inverse_log_transform(X_hat)

        # --- Step 9: Clamp to valid ranges ---
        for i, spec in enumerate(self.schema.continuous):
            if spec.min_val is not None:
                X_hat[:, :, i] = torch.clamp(X_hat[:, :, i], min=spec.min_val)
            if spec.max_val is not None:
                X_hat[:, :, i] = torch.clamp(X_hat[:, :, i], max=spec.max_val)

        return X_hat, Y_hat

    def _inverse_log_transform(self, X_hat: torch.Tensor) -> torch.Tensor:
        """Apply expm1 to features that were log1p-transformed before training.
        Skips features handled by StubSampler (already in raw units)."""
        log_indices = [3]  # inter_arrival_ns
        stub_indices = set(self.stub_sampler._indices) if self.stub_sampler else set()
        for idx in log_indices:
            if idx not in stub_indices:
                X_hat[:, :, idx] = torch.expm1(X_hat[:, :, idx])
        return X_hat

    def _fill_stub_features(self, X_hat: torch.Tensor, device: torch.device,
                            skip_indices: set = None) -> torch.Tensor:
        """Replace low-cardinality / dead features with empirical samples."""
        if self.stub_sampler is None:
            return X_hat
        skip = skip_indices or set()
        shape = (X_hat.shape[0], X_hat.shape[1])
        for i in self.stub_sampler._indices:
            if i in skip:
                continue
            X_hat[:, :, i] = self.stub_sampler.sample(i, shape).to(device)
        return X_hat

    def _get_active_mask(self, device) -> torch.Tensor:
        """Boolean mask of shape (d_c_all,) — True for DDPM-routed features."""
        routes = self.schema.continuous
        mask = torch.zeros(len(routes), dtype=torch.bool, device=device)
        from extractor.schema import VariableType
        for i, spec in enumerate(routes):
            if spec.var_type == VariableType.TYPE4:
                mask[i] = True
        return mask

    def _build_full_tensor(
        self, X_active: torch.Tensor, active_mask: torch.Tensor,
        B: int, L: int, device: torch.device,
    ) -> torch.Tensor:
        """Place active generated features into their original positions.
        Dead features (Type6) are filled with 0 (z-scored mean)."""
        d_c_all = len(active_mask)
        X_full = torch.zeros(B, L, d_c_all, device=device, dtype=X_active.dtype)
        active_idx = 0
        for i in range(d_c_all):
            if active_mask[i]:
                X_full[:, :, i] = X_active[:, :, active_idx]
                active_idx += 1
        return X_full

    def _fill_payload_size(
        self, X_hat: torch.Tensor, Y_hat: List[torch.Tensor],
        quantity: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Overwrite payload_size (index 4) using conditional sampling.

        Samples from the empirical distribution of payload_size conditioned on
        (function_code, direction, quantity). quantity is pre-filled from
        StubSampler to ensure consistency with the protocol."""
        FC_VOCAB = [1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 17, 43]
        func_idx = Y_hat[0].long()
        fc = torch.tensor(FC_VOCAB, device=device)[func_idx.clamp(0, 11)]
        direction = Y_hat[1]

        if self.payload_lookup is not None:
            payload_raw = self.payload_lookup.sample(fc, direction, quantity)
        else:
            is_request = (direction == 0)
            payload_raw = torch.full_like(fc, 12.0, dtype=torch.float32)
            payload_raw = torch.where((fc == 3) & ~is_request, 28.0, payload_raw)
            payload_raw = torch.where((fc == 16) & is_request, 15.0, payload_raw)

        # Convert to z-scored space to match other features
        mean_ps = self.normalizer.mean[4]
        std_ps = self.normalizer.std[4].clamp(min=1e-8)
        X_hat[:, :, 4] = (payload_raw - mean_ps) / std_ps
        return X_hat


@torch.no_grad()
def generate_long_sequence(
    sampler: MaskDDPMSampler,
    total_steps: int,
    overlap: int = 8,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate a long sequence by stitching overlapping windows.

    Uses the last `overlap` steps of each window as seed for the next.

    Args:
        sampler: configured MaskDDPMSampler
        total_steps: desired total length
        overlap: steps to overlap between consecutive windows

    Returns:
        X_long: (1, total_steps, d_c)
        Y_long: list of (1, total_steps)
    """
    L = sampler.L
    device = sampler.device
    d_c_all = sampler.schema.d_c
    d_d = sampler.schema.d_d

    X_parts = []
    Y_parts = [[] for _ in range(d_d)]

    # First window (no seed)
    x1, y1 = sampler.generate(num_samples=1)
    X_parts.append(x1)
    for j in range(d_d):
        Y_parts[j].append(y1[j])

    steps_generated = L
    current_seed = x1[:, -overlap:, :]  # last `overlap` steps as seed

    while steps_generated < total_steps:
        x_w, y_w = sampler.generate(num_samples=1, seed_seq=current_seed)
        # Drop the seed portion (already covered by overlap)
        X_parts.append(x_w[:, overlap:, :])
        for j in range(d_d):
            Y_parts[j].append(y_w[j][:, overlap:])
        steps_generated += L - overlap
        current_seed = x_w[:, -overlap:, :]

    X_long = torch.cat(X_parts, dim=1)[:, :total_steps, :]
    Y_long = [torch.cat(parts, dim=1)[:, :total_steps] for parts in Y_parts]

    return X_long, Y_long
