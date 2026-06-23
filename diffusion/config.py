"""Diffusion model configuration — JSON-driven, similar to checker/config.py pattern."""
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class TrendConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    epochs: int = 200
    batch_size: int = 64


@dataclass
class DDPMConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    diffusion_steps: int = 600
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_target: str = "epsilon"
    use_min_snr: bool = True
    snr_gamma: float = 5.0
    learning_rate: float = 1e-4
    epochs: int = 300
    batch_size: int = 64
    ema_decay: float = 0.999


@dataclass
class MaskDiffusionConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    diffusion_steps: int = 600
    mask_schedule: str = "linear"
    learning_rate: float = 1e-4
    epochs: int = 300
    batch_size: int = 64


@dataclass
class DiffusionConfig:
    trend: TrendConfig = field(default_factory=TrendConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    mask: MaskDiffusionConfig = field(default_factory=MaskDiffusionConfig)
    lambda_balance: float = 0.7
    window_length: int = 128
    seed: int = 42

    @staticmethod
    def load(path: str) -> "DiffusionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DiffusionConfig(
            trend=TrendConfig(**data.get("trend", {})),
            ddpm=DDPMConfig(**data.get("ddpm", {})),
            mask=MaskDiffusionConfig(**data.get("mask", {})),
            lambda_balance=data.get("lambda_balance", 0.7),
            window_length=data.get("window_length", 128),
            seed=data.get("seed", 42),
        )
