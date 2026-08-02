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


class StubSampler:
    """Empirical distribution sampler for continuous features unsuitable for DDPM.

    Stores training values per feature and can generate new values by random
    sampling from the empirical distribution. Used for:
    - Low-cardinality features (e.g. 3-value setpoints) that Gaussian DDPM cannot model
    - Dead features (std≈0) filled with training mean
    """

    def __init__(self):
        self._distributions: Dict[int, np.ndarray] = {}
        self._indices: List[int] = []

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

    def save(self, path: str) -> None:
        """Persist empirical distributions to npz for generation-time reuse."""
        np.savez(path, **{f"feat_{idx}": self._distributions[idx] for idx in self._indices})

    @staticmethod
    def load(path: str) -> "StubSampler":
        """Load empirical distributions saved by save()."""
        s = StubSampler()
        data = np.load(path, allow_pickle=False)
        for key in sorted(data.files, key=lambda k: int(k.split("_")[1])):
            idx = int(key.split("_")[1])
            s._distributions[idx] = data[key]
            s._indices.append(idx)
        return s


class MaskDDPMSampler:
    """End-to-end sampling from the trained Mask-DDPM pipeline."""

    def __init__(
        self,
        trend_model: TransformerTrend,
        ddpm: ResidualDDPM,
        mask_diff: MaskedDiffusion,
        normalizer: Normalizer,
        schema: FeatureSchema,
        device: torch.device = torch.device("cpu"),
    ):
        self.trend_model = trend_model.to(device).eval()
        self.ddpm = ddpm.to(device).eval()
        self.mask_diff = mask_diff.to(device).eval()
        self.normalizer = normalizer
        self.schema = schema
        self.stub_sampler: Optional[StubSampler] = None
        self.device = device

        self.L = schema.window_length
        self._log_indices = list(normalizer.log_features) if hasattr(normalizer, 'log_features') and normalizer.log_features else []
        self._fc_vocab = schema.discrete[0].vocab if schema.discrete[0].vocab else list(range(256))

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
        d_c_active = self.ddpm.d_c
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

        # --- Step 4b: Override transaction_id with random values ---
        # MaskedDiffusion collapses txid to 1-2 values.
        # Use wide random txid (0..65535) to minimise cross-window collisions.
        txid_idx = 3
        Y_hat[txid_idx] = torch.randint(0, 65536, (B, self.L), device=device)

        # --- Step 5: Reconstruct full feature vector (all d_c features) ---
        X_hat_norm_full = self._build_full_tensor(X_active_norm, active_mask, B, self.L, device)

        # --- Step 6: Fill quantity (Type6 stub, in z-scored space) ---
        quantity_idx = 6
        if self.stub_sampler is not None and quantity_idx in self.stub_sampler._indices:
            shape = (B, self.L)
            quantity_raw = self.stub_sampler.sample(quantity_idx, shape).to(device)
            mean_q = self.normalizer.mean[quantity_idx]
            std_q = self.normalizer.std[quantity_idx]
            X_hat_norm_full[:, :, quantity_idx] = (quantity_raw.float() - mean_q) / std_q.clamp(min=1e-8)

        # --- Step 7: Denormalize all features together ---
        X_hat = self.normalizer.inverse_transform(X_hat_norm_full)

        # --- Step 8: Fill remaining Type6 stub features (skip quantity = already filled) ---
        if self.stub_sampler is not None:
            X_hat = self._fill_stub_features(X_hat, device, skip_indices={6})

        # --- Step 9: Inverse log transform for log1p-compressed features ---
        X_hat = self._inverse_log_transform(X_hat)

        # --- Step 10: Clamp to valid ranges ---
        for i, spec in enumerate(self.schema.continuous):
            if spec.min_val is not None:
                X_hat[:, :, i] = torch.clamp(X_hat[:, :, i], min=spec.min_val)
            if spec.max_val is not None:
                X_hat[:, :, i] = torch.clamp(X_hat[:, :, i], max=spec.max_val)

        return X_hat, Y_hat

    def _inverse_log_transform(self, X_hat: torch.Tensor) -> torch.Tensor:
        """Apply expm1 to features that were log1p-transformed before training.
        Skips features handled by StubSampler (already in raw units).
        Clamps input using observed data log-bounds (falls back to μ±3σ if unavailable)
        to prevent float32 overflow and astronomical values from heavy tails."""
        stub_indices = set(self.stub_sampler._indices) if self.stub_sampler else set()
        bounds = getattr(self.normalizer, "log_bounds", {})
        for idx in self._log_indices:
            if idx not in stub_indices:
                if idx in bounds:
                    min_log, max_log = bounds[idx]
                else:
                    log_mean = float(self.normalizer.mean[idx])
                    log_std = float(self.normalizer.std[idx])
                    max_log = log_mean + 3.0 * log_std
                    min_log = max(log_mean - 3.0 * log_std, 0.0)
                X_hat[:, :, idx] = torch.expm1(X_hat[:, :, idx].clamp(min=min_log, max=max_log))
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
    current_seed = x1[:, -overlap:, :]

    while steps_generated < total_steps:
        x_w, y_w = sampler.generate(num_samples=1, seed_seq=current_seed)
        X_parts.append(x_w[:, overlap:, :])
        for j in range(d_d):
            Y_parts[j].append(y_w[j][:, overlap:])
        steps_generated += L - overlap
        current_seed = x_w[:, -overlap:, :]

    X_long = torch.cat(X_parts, dim=1)[:, :total_steps, :]
    Y_long = [torch.cat(parts, dim=1)[:, :total_steps] for parts in Y_parts]

    return X_long, Y_long
