"""Shared feature schema — defines the Modbus feature space used by extractor, diffusion, and assembler."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class VariableType(Enum):
    TYPE1 = 1  # program-driven / setpoint-like (step-dwell)
    TYPE2 = 2  # controller outputs (feedback-coupled)
    TYPE3 = 3  # actuator states/positions (saturation, dwell)
    TYPE4 = 4  # process variables (inertia-dominated) — PRIMARY FOCUS
    TYPE5 = 5  # derived/deterministic (algebraic functions)
    TYPE6 = 6  # auxiliary/low-impact


class FeatureKind(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class FeatureSpec:
    name: str
    kind: FeatureKind
    var_type: VariableType
    vocab: Optional[List[int]] = None  # for discrete fields
    min_val: Optional[float] = None    # for continuous clamp
    max_val: Optional[float] = None


@dataclass
class FeatureSchema:
    trace_id: str
    continuous: List[FeatureSpec] = field(default_factory=list)
    discrete: List[FeatureSpec] = field(default_factory=list)
    window_length: int = 128

    @property
    def d_c(self) -> int:
        return len(self.continuous)

    @property
    def d_d(self) -> int:
        return len(self.discrete)

    @property
    def vocab_sizes(self) -> List[int]:
        return [len(spec.vocab) if spec.vocab else 256 for spec in self.discrete]

    def adapt_to_data(
        self,
        X_cont_flat: "np.ndarray",
        dead_threshold: float = 1e-4,
    ) -> "FeatureSchema":
        """Return a new schema with dead features automatically reclassified as Type6.

        - Features with std < threshold → Type6 (skipped in training)
        - payload_size always stays Type5 (deterministic)
        - All others keep their original type

        Args:
            X_cont_flat: (N, d_c) or (N, L, d_c) continuous feature array
            dead_threshold: features with std below this are considered dead
        """
        import numpy as np
        # Accept both torch tensor and numpy array
        if hasattr(X_cont_flat, "cpu"):
            X_cont_flat = X_cont_flat.cpu().numpy()
        if X_cont_flat.ndim == 3:
            flat = X_cont_flat.reshape(-1, X_cont_flat.shape[-1])
        else:
            flat = X_cont_flat
        stds = flat.std(axis=0)
        # Detect low-cardinality "continuous" features (e.g. step-and-dwell setpoints)
        # These are fundamentally discrete and cannot be modelled by Gaussian DDPM
        cardinality = np.array([len(np.unique(flat[:, i])) for i in range(flat.shape[1])])

        new_cont = []
        for i, spec in enumerate(self.continuous):
            new_spec = FeatureSpec(
                name=spec.name, kind=spec.kind, var_type=spec.var_type,
                vocab=spec.vocab, min_val=spec.min_val, max_val=spec.max_val,
            )
            if spec.name == "payload_size":
                new_spec.var_type = VariableType.TYPE5  # permanent: deterministic
            elif spec.kind == FeatureKind.CONTINUOUS and spec.var_type == VariableType.TYPE4:
                if stds[i] < dead_threshold:
                    new_spec.var_type = VariableType.TYPE6  # dead → stub
                elif cardinality[i] < 10:
                    new_spec.var_type = VariableType.TYPE6  # low-cardinality → stub (not suitable for Gaussian DDPM)
            new_cont.append(new_spec)

        return FeatureSchema(
            trace_id=f"{self.trace_id}_adapted",
            continuous=new_cont,
            discrete=list(self.discrete),
            window_length=self.window_length,
        )

    @staticmethod
    def default_modbus() -> "FeatureSchema":
        return FeatureSchema(
            trace_id="modbus_default",
            continuous=[
                FeatureSpec("register_value_0", FeatureKind.CONTINUOUS, VariableType.TYPE4),
                FeatureSpec("register_value_1", FeatureKind.CONTINUOUS, VariableType.TYPE4),
                FeatureSpec("register_value_2", FeatureKind.CONTINUOUS, VariableType.TYPE4),
                FeatureSpec("inter_arrival_ns", FeatureKind.CONTINUOUS, VariableType.TYPE4,
                            min_val=0),
                FeatureSpec("payload_size", FeatureKind.CONTINUOUS, VariableType.TYPE5,  # deterministic
                            min_val=7),
                FeatureSpec("register_address", FeatureKind.CONTINUOUS, VariableType.TYPE4,
                            min_val=0, max_val=65535),
                FeatureSpec("quantity", FeatureKind.CONTINUOUS, VariableType.TYPE4, min_val=1),
            ],
            discrete=[
                FeatureSpec("function_code", FeatureKind.DISCRETE, VariableType.TYPE4,
                            vocab=[1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 17, 43]),
                FeatureSpec("direction", FeatureKind.DISCRETE, VariableType.TYPE4,
                            vocab=[0, 1]),
                FeatureSpec("unit_id", FeatureKind.DISCRETE, VariableType.TYPE4,
                            vocab=list(range(0, 248))),
                FeatureSpec("transaction_id", FeatureKind.DISCRETE, VariableType.TYPE6,
                            vocab=None),
                FeatureSpec("is_exception", FeatureKind.DISCRETE, VariableType.TYPE6,
                            vocab=[0, 1]),
                FeatureSpec("exception_code", FeatureKind.DISCRETE, VariableType.TYPE6,
                            vocab=list(range(0, 256))),
            ],
            window_length=128,
        )
