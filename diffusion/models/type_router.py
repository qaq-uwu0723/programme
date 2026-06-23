"""Type-aware variable routing — determines which modeling mechanism handles each variable.

Initial version: Type4 variables use the full pipeline (trend+DDPM/mask).
Other types use stub strategies (empirical resampling from training distribution).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

from extractor.schema import FeatureSchema, FeatureSpec, VariableType, FeatureKind


class Mechanism(Enum):
    TREND_DDPM = "trend_ddpm"       # full pipeline: trend Transformer + DDPM
    MASK = "mask"                    # masked diffusion for discrete
    STUB = "stub"                    # empirical resampling (not learned)


@dataclass
class VarRoute:
    name: str
    kind: FeatureKind
    var_type: VariableType
    mechanism: Mechanism
    index: int  # position in the continuous or discrete tensor


@dataclass
class Routing:
    """Routing table for all variables."""
    routes: List[VarRoute]

    @property
    def ddpm_indices(self) -> List[int]:
        """Indices of continuous variables routed to DDPM."""
        return [r.index for r in self.routes if r.mechanism == Mechanism.TREND_DDPM]

    @property
    def mask_indices(self) -> List[int]:
        """Indices of discrete variables routed to masked diffusion."""
        return [r.index for r in self.routes if r.mechanism == Mechanism.MASK]

    @property
    def ddpm_count(self) -> int:
        return len(self.ddpm_indices)

    @property
    def mask_count(self) -> int:
        return len(self.mask_indices)

    @property
    def ddpm_var_names(self) -> List[str]:
        return [r.name for r in self.routes if r.mechanism == Mechanism.TREND_DDPM]

    @property
    def mask_var_names(self) -> List[str]:
        return [r.name for r in self.routes if r.mechanism == Mechanism.MASK]


class TypeRouter:
    """Determines which mechanism handles each variable based on its type.

    v0 strategy:
      - Type4 continuous  → TREND_DDPM
      - Type4 discrete    → MASK
      - All other types   → STUB (empirical resampling)
    """

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
        self.routing = self._build_routing(schema)

    def _build_routing(self, schema: FeatureSchema) -> Routing:
        routes = []
        for i, spec in enumerate(schema.continuous):
            mech = Mechanism.TREND_DDPM if spec.var_type == VariableType.TYPE4 else Mechanism.STUB
            routes.append(VarRoute(spec.name, FeatureKind.CONTINUOUS, spec.var_type, mech, i))
        for i, spec in enumerate(schema.discrete):
            mech = Mechanism.MASK if spec.var_type == VariableType.TYPE4 else Mechanism.STUB
            routes.append(VarRoute(spec.name, FeatureKind.DISCRETE, spec.var_type, mech, i))
        return Routing(routes)

    def get_ddpm_mask(self, d_c: int, device) -> torch.Tensor:
        """Boolean mask of shape (d_c,) — True for DDPM-routed continuous features."""
        import torch
        mask = torch.zeros(d_c, dtype=torch.bool, device=device)
        for idx in self.routing.ddpm_indices:
            mask[idx] = True
        return mask
