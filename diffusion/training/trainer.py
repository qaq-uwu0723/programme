"""Mask-DDPM staged training orchestrator.

Stage 1: Train TransformerTrend with next-step MSE
Stage 2: Freeze trend, compute residuals, jointly train DDPM + MaskedDiffusion
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import DiffusionConfig
from ..models.trend_transformer import TransformerTrend
from ..models.residual_ddpm import ResidualDDPM
from ..models.masked_diffusion import MaskedDiffusion
from ..models.type_router import TypeRouter
from ..utils.noise_schedule import get_alpha_bars
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from extractor.schema import FeatureSchema


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply(self):
        """Copy shadow params into the model (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params (for continuing training)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


class MaskDDPMTrainer:
    """Orchestrates staged training of the Mask-DDPM pipeline."""

    def __init__(
        self,
        config: DiffusionConfig,
        schema: FeatureSchema,
        device: torch.device = torch.device("cpu"),
    ):
        self.config = config
        self.schema = schema
        self.device = device
        self.router = TypeRouter(schema)

        # Use only DDPM-routed features for training (skip stub/deterministic)
        self.d_c_active = self.router.routing.ddpm_count
        self.active_indices = self.router.routing.ddpm_indices
        self.d_c_all = schema.d_c
        d_d = schema.d_d

        # Build models with active feature count
        self.trend_model = TransformerTrend(
            d_c=self.d_c_active,
            d_model=config.trend.d_model,
            nhead=config.trend.nhead,
            num_layers=config.trend.num_layers,
            dim_feedforward=config.trend.dim_feedforward,
            dropout=config.trend.dropout,
        ).to(device)

        self.ddpm = ResidualDDPM(
            d_c=self.d_c_active,
            d_cond=self.d_c_active,     # conditioned on trend S_hat (same dim as active X)
            d_model=config.ddpm.d_model,
            nhead=config.ddpm.nhead,
            num_layers=config.ddpm.num_layers,
            dim_feedforward=config.ddpm.dim_feedforward,
            dropout=config.ddpm.dropout,
            diffusion_steps=config.ddpm.diffusion_steps,
            beta_schedule=config.ddpm.beta_schedule,
            beta_start=config.ddpm.beta_start,
            beta_end=config.ddpm.beta_end,
        ).to(device)

        self.mask_diff = MaskedDiffusion(
            vocab_sizes=schema.vocab_sizes,
            d_cond=self.d_c_active + self.d_c_active,  # trend + generated continuous
            d_model=config.mask.d_model,
            nhead=config.mask.nhead,
            num_layers=config.mask.num_layers,
            dim_feedforward=config.mask.dim_feedforward,
            dropout=config.mask.dropout,
            diffusion_steps=config.mask.diffusion_steps,
            mask_schedule=config.mask.mask_schedule,
        ).to(device)

        # EMA for DDPM (used at sampling time)
        self.ddpm_ema = EMAModel(self.ddpm, config.ddpm.ema_decay)

        # Optimizers (created during training stages)
        self.trend_optimizer: Optional[torch.optim.Optimizer] = None
        self.diffusion_optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Stage 1: Trend Training
    # ------------------------------------------------------------------

    def _slice_active(self, X: torch.Tensor) -> torch.Tensor:
        """Extract only DDPM-routed features from the full feature tensor."""
        idx = torch.tensor(self.active_indices, device=X.device, dtype=torch.long)
        return X.index_select(-1, idx)

    def train_trend(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Stage 1: Train TransformerTrend with next-step MSE prediction.

        Args:
            train_data: (N, L, d_c) continuous features (already normalized)
            val_data: optional validation tensor

        Returns:
            dict with training history
        """
        cfg = self.config.trend
        train_data = self._slice_active(train_data)
        if val_data is not None:
            val_data = self._slice_active(val_data)
        dataset = TensorDataset(train_data)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self.trend_optimizer = torch.optim.Adam(
            self.trend_model.parameters(), lr=cfg.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.trend_optimizer, factor=0.5, patience=20
        )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(cfg.epochs):
            self.trend_model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                self.trend_optimizer.zero_grad()
                loss = self.trend_model.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trend_model.parameters(), 1.0)
                self.trend_optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(train_data)
            history["train_loss"].append(epoch_loss)

            if val_data is not None:
                self.trend_model.eval()
                with torch.no_grad():
                    val_loss = self.trend_model.compute_loss(val_data.to(self.device))
                history["val_loss"].append(val_loss.item())
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Trend epoch {epoch+1}/{cfg.epochs}  loss={epoch_loss:.6f}")

        return history

    # ------------------------------------------------------------------
    # Stage 2: Diffusion Training
    # ------------------------------------------------------------------

    def train_diffusion(
        self,
        train_x: torch.Tensor,
        train_y: List[torch.Tensor],
        val_x: Optional[torch.Tensor] = None,
        val_y: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Stage 2: Joint training of ResidualDDPM + MaskedDiffusion.

        Trend model is frozen. Residuals R = X - S_hat are computed first.

        Args:
            train_x: (N, L, d_c) continuous features
            train_y: list of (N, L) discrete features
            val_x, val_y: optional validation tensors

        Returns:
            dict with training history
        """
        cfg = self.config
        L = cfg.window_length
        lambda_bal = cfg.lambda_balance

        # Slice to only DDPM-routed features
        train_x = self._slice_active(train_x)

        # Freeze trend model
        self.trend_model.eval()
        for p in self.trend_model.parameters():
            p.requires_grad_(False)

        # Precompute trend and residuals — batched to avoid OOM on large datasets
        print("Computing trend predictions for training data...")
        S_hat_train = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(train_x), batch_size):
                batch = train_x[i:i+batch_size].to(self.device)
                S_hat_train.append(self.trend_model(batch))
        S_hat_train = torch.cat(S_hat_train, dim=0)
        R_train = train_x.to(self.device) - S_hat_train  # residuals

        # Precompute validation residuals if validation data provided
        val_loader = None
        if val_x is not None and val_y is not None:
            val_x = self._slice_active(val_x).to(self.device)
            val_y = [y.to(self.device) for y in val_y]
            with torch.no_grad():
                S_hat_val = self.trend_model(val_x)
            R_val = val_x - S_hat_val
            val_dataset = TensorDataset(R_val, S_hat_val, *val_y)
            val_loader = DataLoader(val_dataset, batch_size=cfg.ddpm.batch_size, shuffle=False)

        # Build datasets
        train_dataset = TensorDataset(R_train, S_hat_train, *[y.to(self.device) for y in train_y])
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.ddpm.batch_size, shuffle=True
        )

        # Optimizer: joint params of ddpm + mask_diff
        params = list(self.ddpm.parameters()) + list(self.mask_diff.parameters())
        self.diffusion_optimizer = torch.optim.AdamW(
            params, lr=cfg.ddpm.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.diffusion_optimizer, T_0=50, T_mult=2
        )

        history = {"train_loss_cont": [], "train_loss_disc": [], "train_loss_total": [],
                   "val_loss_total": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        patience = 20  # stop after 20 epochs without improvement
        best_state = None

        for epoch in range(cfg.ddpm.epochs):
            self.ddpm.train()
            self.mask_diff.train()
            epoch_cont = 0.0
            epoch_disc = 0.0
            epoch_total = 0.0
            n_samples = 0

            for batch in train_loader:
                r0 = batch[0].to(self.device)       # (B, L, d_c)
                s_hat = batch[1].to(self.device)     # (B, L, d_c)
                y_list = [b.to(self.device) for b in batch[2:]]  # list of (B, L)

                B = r0.size(0)

                # Continuous: DDPM forward
                loss_cont, k, eps_pred = self.ddpm(r0, s_hat)

                # Discrete: Mask forward (condition on S_hat + X_hat from trend)
                # X_hat = S_hat (trend only; during training we don't have residuals yet)
                loss_disc = self.mask_diff(y_list, s_hat, x_hat=s_hat)

                total = loss_cont * lambda_bal + loss_disc * (1.0 - lambda_bal)

                self.diffusion_optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                self.diffusion_optimizer.step()
                self.ddpm_ema.update()

                epoch_cont += loss_cont.item() * B
                epoch_disc += loss_disc.item() * B
                epoch_total += total.item() * B
                n_samples += B

            scheduler.step()

            epoch_cont /= n_samples
            epoch_disc /= n_samples
            epoch_total /= n_samples
            history["train_loss_cont"].append(epoch_cont)
            history["train_loss_disc"].append(epoch_disc)
            history["train_loss_total"].append(epoch_total)

            # --- Validation + Early Stopping (every 3 epochs) ---
            val_total = None
            if val_loader is not None and (epoch + 1) % 3 == 0:
                self.ddpm.eval()
                self.mask_diff.eval()
                val_cont_sum = 0.0
                val_disc_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        v_r0 = vb[0].to(self.device)
                        v_sh = vb[1].to(self.device)
                        v_yl = [b.to(self.device) for b in vb[2:]]
                        v_lc, _, _ = self.ddpm(v_r0, v_sh)
                        v_ld = self.mask_diff(v_yl, v_sh, x_hat=v_sh)
                        nv = v_r0.size(0)
                        val_cont_sum += v_lc.item() * nv
                        val_disc_sum += v_ld.item() * nv
                        val_n += nv
                val_cont = val_cont_sum / val_n
                val_disc = val_disc_sum / val_n
                val_total = val_cont * lambda_bal + val_disc * (1.0 - lambda_bal)
                history["val_loss_total"].append(val_total)

                # Early stopping check
                if val_total < best_val_loss * 0.995:  # improved by >0.5%
                    best_val_loss = val_total
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model state
                    best_state = {
                        "ddpm": {k: v.cpu().clone() for k, v in self.ddpm.state_dict().items()},
                        "mask": {k: v.cpu().clone() for k, v in self.mask_diff.state_dict().items()},
                    }
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch+1}, "
                          f"val_loss={best_val_loss:.6f})")
                    break

                self.ddpm.train()
                self.mask_diff.train()

            if (epoch + 1) % 20 == 0:
                val_str = f"  val={val_total:.6f}" if val_total is not None else ""
                print(
                    f"Diff epoch {epoch+1}/{cfg.ddpm.epochs}  "
                    f"cont={epoch_cont:.6f}  disc={epoch_disc:.6f}  "
                    f"total={epoch_total:.6f}{val_str}"
                )

        # Restore best model if early stopping was triggered
        if best_state is not None and best_epoch < cfg.ddpm.epochs - 1:
            self.ddpm.load_state_dict(best_state["ddpm"])
            self.mask_diff.load_state_dict(best_state["mask"])
            print(f"Restored best model from epoch {best_epoch+1} "
                  f"(val_loss={best_val_loss:.6f})")

        return history

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.trend_model.state_dict(), f"{path}/trend_model.pt")
        torch.save(self.ddpm.state_dict(), f"{path}/ddpm_model.pt")
        torch.save(self.mask_diff.state_dict(), f"{path}/mask_diff_model.pt")
        if self.ddpm_ema:
            torch.save(self.ddpm_ema.shadow, f"{path}/ddpm_ema.pt")
        print(f"Models saved to {path}/")

    def load(self, path: str) -> None:
        self.trend_model.load_state_dict(
            torch.load(f"{path}/trend_model.pt", map_location=self.device, weights_only=True))
        self.ddpm.load_state_dict(
            torch.load(f"{path}/ddpm_model.pt", map_location=self.device, weights_only=True))
        self.mask_diff.load_state_dict(
            torch.load(f"{path}/mask_diff_model.pt", map_location=self.device, weights_only=True))
        ema_path = f"{path}/ddpm_ema.pt"
        if Path(ema_path).exists():
            self.ddpm_ema.shadow = torch.load(ema_path, map_location=self.device)
        print(f"Models loaded from {path}/")
