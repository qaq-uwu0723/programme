"""Per-feature z-score normalisation tracker for continuous variables."""
from typing import Dict, List, Optional, Tuple
import torch


class Normalizer:
    """Tracks per-feature mean/std for z-score normalisation of continuous features.

    Supports both numpy arrays (for data loading) and torch tensors (for training).
    """

    def __init__(self, num_features: int, eps: float = 1e-8):
        self.num_features = num_features
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None   # (num_features,)
        self.std: Optional[torch.Tensor] = None     # (num_features,)

    def fit(self, data: torch.Tensor) -> "Normalizer":
        """Compute mean/std from data.

        Args:
            data: (N, L, d_c) or (N, d_c) float tensor
        """
        if data.dim() == 3:
            # Flatten N and L
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=self.eps)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to zero-mean unit-variance."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return (data - mean) / std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize from zero-mean unit-variance back to original scale."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean

    def fit_from_numpy(self, data: "np.ndarray") -> "Normalizer":
        """Convenience: fit from numpy array."""
        import numpy as np
        flat = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        self.mean = torch.from_numpy(flat.mean(axis=0).astype("float32"))
        self.std = torch.from_numpy(flat.std(axis=0).astype("float32")).clamp(min=self.eps)
        return self

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "std": self.std, "num_features": self.num_features}

    def load_state_dict(self, state: Dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.num_features = state["num_features"]

    def save(self, path: str) -> None:
        import json
        d = {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "num_features": self.num_features,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path: str) -> "Normalizer":
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Support both formats: diffusion save (num_features) and extractor save (mean/std only)
        nf = d.get("num_features", len(d["mean"]))
        n = Normalizer(nf)
        n.mean = torch.tensor(d["mean"], dtype=torch.float32) if d.get("mean") else None
        n.std = torch.tensor(d["std"], dtype=torch.float32) if d.get("std") else None
        return n
