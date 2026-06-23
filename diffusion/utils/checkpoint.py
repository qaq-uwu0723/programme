"""Model checkpoint save/load utilities."""
from typing import Dict, Any, Optional
from pathlib import Path
import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    ema_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    loss: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model": model_state,
        "epoch": epoch,
        "loss": loss,
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    if ema_state is not None:
        checkpoint["ema"] = ema_state
    if extra is not None:
        checkpoint["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """Load a training checkpoint."""
    return torch.load(path, map_location=device, weights_only=False)


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save only model state dict."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device = torch.device("cpu")) -> None:
    """Load state dict into an existing model."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
