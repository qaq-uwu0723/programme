"""Automated experiment runner — trains Mask-DDPM and records results to EXPERIMENT_LOG.md.

Usage:
    python experiments/run_experiment.py --name "baseline" --data data/ics_clean/ --output checkpoints/exp01/
"""
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler
from diffusion.utils.normalisation import Normalizer
from diffusion.utils.metrics import evaluate_all, compute_ks, compute_jsd, compute_lag1_diff


def load_data(data_dir: str, schema: FeatureSchema):
    """Load training tensors from directory."""
    p = Path(data_dir)
    X = torch.from_numpy(np.load(str(p / "train_X.npy"))).float()
    Y = [torch.from_numpy(np.load(str(p / f"train_Y_{j}.npy"))).long()
         for j in range(schema.d_d)]
    # Load normalizer stats
    with open(p / "normalizer.json") as f:
        stats = json.load(f)
    return X, Y, stats


def split_data(X, Y, train_frac=0.70, val_frac=0.15):
    """Split into train/val/test by window index (no shuffle to keep temporal order)."""
    N = X.shape[0]
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    X_train = X[:n_train]
    Y_train = [y[:n_train] for y in Y]
    X_val = X[n_train:n_train + n_val]
    Y_val = [y[n_train:n_train + n_val] for y in Y]
    X_test = X[n_train + n_val:]
    Y_test = [y[n_train + n_val:] for y in Y]
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def train_and_evaluate(args):
    schema = FeatureSchema.default_modbus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load & Prepare Data ---
    X, Y, norm_stats = load_data(args.data, schema)
    print(f"Data loaded: X={X.shape}, Y=6×{Y[0].shape}")

    # Auto-adapt schema based on data statistics
    schema = schema.adapt_to_data(X)
    active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
    dead   = [s.name for s in schema.continuous if s.var_type.name == "TYPE6"]
    det    = [s.name for s in schema.continuous if s.var_type.name == "TYPE5"]
    print(f"Schema adapted: {len(active)} active [{', '.join(active)}]")
    if dead:
        print(f"               {len(dead)} dead   [{', '.join(dead)}]")
    if det:
        print(f"               {len(det)} det    [{', '.join(det)}]")

    (X_tr, Y_tr), (X_val, Y_val), (X_te, Y_te) = split_data(X, Y)
    print(f"Split: train={X_tr.shape[0]}, val={X_val.shape[0]}, test={X_te.shape[0]} windows")

    # Use ORIGINAL normalization stats from feature extraction
    # (data is already z-scored, so we load stats for denormalization)
    from diffusion.utils.normalisation import Normalizer
    norm = Normalizer.load(args.data + "/normalizer.json")
    # Data is already normalized — use as-is
    X_tr_norm = X_tr
    X_val_norm = X_val
    X_te_norm = X_te

    # --- Load Config ---
    if args.config:
        cfg = DiffusionConfig.load(args.config)
    else:
        cfg = DiffusionConfig()
    cfg.window_length = X_tr.shape[1]  # use actual window length from data

    # --- Train ---
    trainer = MaskDDPMTrainer(cfg, schema, device=device)
    start_time = time.time()

    print("\n" + "="*60)
    print("STAGE 1: Trend Training")
    print("="*60)
    trend_hist = trainer.train_trend(X_tr_norm.to(device), X_val_norm.to(device) if len(X_val) > 0 else None)

    print("\n" + "="*60)
    print("STAGE 2: Diffusion Training")
    print("="*60)
    diff_hist = trainer.train_diffusion(X_tr_norm.to(device), [y.to(device) for y in Y_tr])

    train_time = time.time() - start_time

    # Save model
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_dir))
    # Save normalizer
    norm_mean = norm.mean.tolist() if norm.mean is not None else []
    norm_std = norm.std.tolist() if norm.std is not None else []
    with open(out_dir / "normalizer.json", "w") as f:
        json.dump({"mean": norm_mean, "std": norm_std}, f, indent=2)

    # --- Build conditional payload lookup + stub sampler from training data ---
    from diffusion.sampling.sampler import PayloadLookup, StubSampler
    payload_lu = PayloadLookup()
    stub_sampler = StubSampler()
    # Use raw-unit training data
    X_tr_raw = norm.inverse_transform(X_tr.reshape(-1, schema.d_c)).reshape(X_tr.shape)
    Y_tr_flat = torch.stack(Y_tr, dim=-1).reshape(-1, schema.d_d)
    payload_lu.fit(X_tr_raw.cpu().numpy(), Y_tr_flat.cpu().numpy())
    # Detect low-cardinality + dead features for post-hoc empirical replacement
    flat_raw = X_tr_raw.reshape(-1, schema.d_c).cpu().numpy()
    cardinality = [len(np.unique(flat_raw[:, i])) for i in range(schema.d_c)]
    dead_idx = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
    low_card_idx = [i for i, c in enumerate(cardinality) if 1 < c < 10
                    and schema.continuous[i].var_type.name == "TYPE4"]
    stub_idx = sorted(set(dead_idx + low_card_idx))
    if stub_idx:
        stub_sampler.fit(flat_raw, stub_idx)
        names = [schema.continuous[i].name for i in stub_idx]
        print(f"Payload lookup: {len(payload_lu._table)} pairs  |  Stub sampler: {names}")
    else:
        print(f"Payload lookup: {len(payload_lu._table)} pairs  |  No stub features")

    # --- Generate ---
    print("\n" + "="*60)
    print("GENERATION")
    print("="*60)
    trainer.ddpm_ema.apply()
    sampler = MaskDDPMSampler(
        trainer.trend_model, trainer.ddpm, trainer.mask_diff,
        norm, schema, payload_lookup=payload_lu, device=device,
    )
    sampler.stub_sampler = stub_sampler
    X_gen, Y_gen = sampler.generate(num_samples=min(50, len(X_te)), num_unmask_steps=50)
    print(f"Generated: X={list(X_gen.shape)}, Y={[list(y.shape) for y in Y_gen]}")

    # --- Evaluate ---
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    gen_x = X_gen.cpu().numpy()
    gen_y = np.stack([y.cpu().numpy() for y in Y_gen], axis=-1)
    # Denormalize test data to raw units for fair comparison
    real_x_raw = norm.inverse_transform(X_te[:len(gen_x)]).cpu().numpy()
    real_y = np.stack([y[:len(gen_x)].numpy() for y in Y_te], axis=-1)

    results = evaluate_all(real_x_raw, gen_x, real_y, gen_y, schema.vocab_sizes)

    # --- Assemble & Check ---
    from assembler.packet_builder import PacketAssembler
    assembler = PacketAssembler(schema)
    pcap_out = str(out_dir / "gen_trace.pcapng")
    meta_out = str(out_dir / "gen_trace.meta.jsonl")
    assembler.assemble(X_gen, Y_gen, pcap_out, meta_out, trace_id=f"exp-{args.name}")

    from checker.validate import validate
    from checker.config import Config
    report = validate(pcap_out, meta_out, Config(), mode="mvp")
    checker_s = report.summary

    # --- Print Summary ---
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train time:       {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"Trend loss:       {trend_hist['train_loss'][0]:.4f} → {trend_hist['train_loss'][-1]:.4f}")
    print(f"Diff cont loss:   {diff_hist['train_loss_cont'][0]:.4f} → {diff_hist['train_loss_cont'][-1]:.4f}")
    print(f"Diff disc loss:   {diff_hist['train_loss_disc'][0]:.4f} → {diff_hist['train_loss_disc'][-1]:.4f}")
    print(f"Diff total loss:  {diff_hist['train_loss_total'][0]:.4f} → {diff_hist['train_loss_total'][-1]:.4f}")
    print(f"Mean KS:          {results['ks']['mean_ks']:.4f}")
    print(f"Max KS:           {results['ks']['max_ks']:.4f}")
    print(f"Mean JSD:         {results['jsd']['mean_jsd']:.4f}")
    if results.get('lag1_diff'):
        print(f"Mean Lag-1 Diff:  {results['lag1_diff']['mean_lag1_diff']:.4f}")
    print(f"Checker:          fatal={checker_s.by_severity.get('fatal',0)}, "
          f"error={checker_s.by_severity.get('error',0)}, "
          f"warn={checker_s.by_severity.get('warn',0)}")

    # --- Per-feature KS ---
    print("\nPer-feature KS:")
    for i, spec in enumerate(schema.continuous):
        ks_v = results['ks'].get('per_feature', [0]*schema.d_c)[i]
        print(f"  {spec.name:25s}  KS={ks_v:.4f}")

    # --- Update EXPERIMENT_LOG.md ---
    update_log(args, cfg, norm_stats, trend_hist, diff_hist, results, checker_s, train_time, schema)

    return results


def update_log(args, cfg, norm_stats, trend_hist, diff_hist, results, checker_s, train_time, schema):
    """Append results to EXPERIMENT_LOG.md."""
    log_path = Path(__file__).parent / "EXPERIMENT_LOG.md"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build per-feature KS lines
    ks_lines = ""
    for i, spec in enumerate(schema.continuous):
        ks_v = results['ks'].get('per_feature', [0]*schema.d_c)[i]
        ks_lines += f"| {spec.name} | {ks_v:.4f} |\n"

    per_feat_ks = results['ks'].get('per_feature', [])
    ks_str = ", ".join([f"{v:.4f}" for v in per_feat_ks]) if per_feat_ks else "-"

    entry = f"""
### 实验 #{args.name}

**时间**: {date_str}  |  **训练耗时**: {train_time:.0f}s ({train_time/60:.1f}min)  |  **设备**: {"GPU" if torch.cuda.is_available() else "CPU"}

#### 数据
| 指标 | 值 |
|------|-----|
| 总窗口数 | {norm_stats.get('num_windows', '-')} |
| 训练/验证/测试 | 70% / 15% / 15% |
| 窗口长度 | {cfg.window_length} |
| d_c / d_d | {schema.d_c} / {schema.d_d} |

#### 训练曲线
| 阶段 | 指标 | 初始 | 最终 |
|------|------|------|------|
| Stage 1 | Trend Loss | {trend_hist['train_loss'][0]:.4f} | {trend_hist['train_loss'][-1]:.4f} |
| Stage 2 | Loss_cont | {diff_hist['train_loss_cont'][0]:.4f} | {diff_hist['train_loss_cont'][-1]:.4f} |
| Stage 2 | Loss_disc | {diff_hist['train_loss_disc'][0]:.4f} | {diff_hist['train_loss_disc'][-1]:.4f} |
| Stage 2 | Loss_total | {diff_hist['train_loss_total'][0]:.4f} | {diff_hist['train_loss_total'][-1]:.4f} |

#### 评估指标
| 指标 | 值 |
|------|-----|
| Mean KS | {results['ks']['mean_ks']:.4f} |
| Max KS | {results['ks']['max_ks']:.4f} |
| Mean JSD | {results['jsd']['mean_jsd']:.4f} |
| Checker: fatal / error / warn | {checker_s.by_severity.get('fatal',0)} / {checker_s.by_severity.get('error',0)} / {checker_s.by_severity.get('warn',0)} |

#### Per-feature KS
| 特征 | KS |
|------|-----|
{ks_lines}
"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"\nExperiment log updated: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mask-DDPM experiment")
    parser.add_argument("--name", required=True, help="Experiment name/number")
    parser.add_argument("--data", required=True, help="Path to training data directory")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--output", default="checkpoints/exp01/", help="Output directory")
    args = parser.parse_args()
    train_and_evaluate(args)
