"""Universal experiment runner — works with any dataset, fully self-contained.

Usage:
    # Pre-extracted tensors
    python experiments/run_experiment.py --name "exp01" --data data/ics_clean/ --output checkpoints/exp01/

    # Raw FARAONIC CSV (auto-extracts)
    python experiments/run_experiment.py --name "exp02" --csv dataset/FARAONIC/training.csv --csv-rows 500000

    # Raw PCAP
    python experiments/run_experiment.py --name "exp03" --pcap dataset/ICS_PACPS/clean/traffic.pcap
"""
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, PayloadLookup, StubSampler
from diffusion.utils.normalisation import Normalizer
from diffusion.utils.metrics import evaluate_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tensors(data_dir):
    """Load pre-extracted .npy tensors from a directory."""
    p = Path(data_dir)
    schema = FeatureSchema.default_modbus()
    X = torch.from_numpy(np.load(str(p / "train_X.npy"))).float()
    Y = [torch.from_numpy(np.load(str(p / f"train_Y_{j}.npy"))).long()
         for j in range(schema.d_d)]
    with open(p / "normalizer.json") as f:
        stats = json.load(f)
    return X, Y, stats


def load_csv(path, max_rows):
    """Extract from FARAONIC CSV on-the-fly."""
    from extractor.faraonic_reader import read_faraonic_csv
    from extractor.feature_builder import build_training_data
    records = read_faraonic_csv(path, max_rows=max_rows, label_filter="NORMAL")
    schema = FeatureSchema.default_modbus()
    X_w, Y_w, stats = build_training_data(records, schema, window_length=128, stride=16)
    X = torch.from_numpy(X_w).float()
    Y = [torch.from_numpy(Y_w[j]).long() for j in range(schema.d_d)]
    return X, Y, stats


def load_pcap(path):
    """Extract from PCAP on-the-fly."""
    from extractor.pcap_reader import extract_packets
    from extractor.feature_builder import build_training_data
    records = extract_packets(path)
    schema = FeatureSchema.default_modbus()
    X_w, Y_w, stats = build_training_data(records, schema, window_length=128, stride=16)
    X = torch.from_numpy(X_w).float()
    Y = [torch.from_numpy(Y_w[j]).long() for j in range(schema.d_d)]
    return X, Y, stats


def split(X, Y, train_frac=0.70, val_frac=0.15):
    N = X.shape[0]
    nt = int(N * train_frac)
    nv = int(N * val_frac)
    return (X[:nt], [y[:nt] for y in Y]), \
           (X[nt:nt + nv], [y[nt:nt + nv] for y in Y]), \
           (X[nt + nv:], [y[nt + nv:] for y in Y])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    schema = FeatureSchema.default_modbus()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Log to both console and file
    log_file = out / "training.log"

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== TADS-ICS Training: {args.name} ===")
    log(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 1. Load data ---
    t0 = time.time()
    if args.csv:
        log(f"Loading CSV: {args.csv}" + (f" (max {args.csv_rows} rows)" if args.csv_rows else ""))
        X, Y, stats = load_csv(args.csv, args.csv_rows)
    elif args.pcap:
        log(f"Loading PCAP: {args.pcap}")
        X, Y, stats = load_pcap(args.pcap)
    else:
        log(f"Loading tensors: {args.data}")
        X, Y, stats = load_tensors(args.data)
    log(f"  {X.shape[0]} windows | {time.time() - t0:.1f}s")

    # --- 2. Adapt schema ---
    schema = schema.adapt_to_data(X)
    active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
    stub   = [s.name for s in schema.continuous if s.var_type.name == "TYPE6"]
    det    = [s.name for s in schema.continuous if s.var_type.name == "TYPE5"]
    log(f"  Active: {active}")
    if stub: log(f"  Stub:   {stub}")
    if det:  log(f"  Det:    {det}")

    # --- 3. Split ---
    (X_tr, Y_tr), (X_val, Y_val), (X_te, Y_te) = split(X, Y)
    log(f"  Split: train={X_tr.shape[0]} val={X_val.shape[0]} test={X_te.shape[0]}")

    # --- 4. Normalizer ---
    norm = Normalizer(schema.d_c)
    norm.mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm.std  = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-8)

    # --- 5. Config ---
    cfg = DiffusionConfig.load(args.config) if args.config else DiffusionConfig()
    cfg.window_length = X_tr.shape[1]
    if args.batch_size:
        cfg.trend.batch_size = args.batch_size
        cfg.ddpm.batch_size = args.batch_size
        cfg.mask.batch_size = args.batch_size
    if args.epochs:
        cfg.ddpm.epochs = args.epochs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"  Config: batch={cfg.ddpm.batch_size} ep={cfg.ddpm.epochs} d={cfg.ddpm.d_model} K={cfg.ddpm.diffusion_steps} device={device}")

    # --- 6. Train ---
    trainer = MaskDDPMTrainer(cfg, schema, device=device)

    log("--- STAGE 1: Trend ---")
    t0 = time.time()
    h1 = trainer.train_trend(X_tr.to(device), log_fn=log)
    log(f"  {time.time()-t0:.0f}s | loss {h1['train_loss'][0]:.4f}->{h1['train_loss'][-1]:.4f}")

    log("--- STAGE 2: Diffusion ---")
    t0 = time.time()
    h2 = trainer.train_diffusion(X_tr.to(device), Y_tr, val_x=X_val.to(device), val_y=Y_val, log_fn=log)
    t_train = time.time() - t0
    n_ep = len(h2["train_loss_total"])
    log(f"  {t_train:.0f}s ({t_train/60:.1f}min) | {n_ep} epochs")
    log(f"  cont: {h2['train_loss_cont'][0]:.4f}->{h2['train_loss_cont'][-1]:.4f}")
    log(f"  disc: {h2['train_loss_disc'][0]:.4f}->{h2['train_loss_disc'][-1]:.4f}")
    if h2.get("val_loss_total") and len(h2["val_loss_total"]) > 0:
        vh = h2["val_loss_total"]
        log(f"  val: {vh[0]:.6f}->{vh[-1]:.6f} | best={min(vh):.6f} @ ep{vh.index(min(vh))+1}")

    # --- 7. SAVE MODEL (before eval — survives eval crash) ---
    trainer.ddpm_ema.apply()
    trainer.save(str(out))
    with open(out / "normalizer.json", "w") as f:
        json.dump({"mean": norm.mean.tolist(), "std": norm.std.tolist(),
                   "log_features": stats.get("log_features", [])}, f, indent=2)
    log("-- Model saved --")

    # --- 8. Build helpers ---
    X_tr_raw = norm.inverse_transform(X_tr.float()).cpu().numpy()
    Y_tr_flat = torch.stack(Y_tr, dim=-1).reshape(-1, schema.d_d).numpy()
    payload_lu = PayloadLookup()
    payload_lu.fit(X_tr_raw.reshape(-1, schema.d_c), Y_tr_flat)

    card = [len(np.unique(X_tr_raw.reshape(-1, schema.d_c)[:, i])) for i in range(schema.d_c)]
    dead = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
    low_card = [i for i, c in enumerate(card) if 1 < c < 10 and schema.continuous[i].var_type.name == "TYPE4"]
    stub_idx = sorted(set(dead + low_card))
    stub_s = StubSampler()
    if stub_idx: stub_s.fit(X_tr_raw.reshape(-1, schema.d_c), stub_idx)

    # --- 9. Generate & Evaluate ---
    log("--- Evaluation ---")
    log_features = stats.get("log_features", [])
    sampler = MaskDDPMSampler(trainer.trend_model, trainer.ddpm, trainer.mask_diff,
                              norm, schema, payload_lookup=payload_lu, device=device)
    if stub_s: sampler.stub_sampler = stub_s
    n_gen = min(50, X_tr.shape[0])
    X_gen, Y_gen = sampler.generate(num_samples=n_gen, num_unmask_steps=50)
    gen_x = X_gen.cpu().numpy()
    gen_y = np.stack([y.cpu().numpy() for y in Y_gen], axis=-1)

    def eval_set(real_x_tensor, real_y_list, name):
        x = norm.inverse_transform(real_x_tensor.float()).cpu().numpy()
        for idx in log_features:
            x[:, :, idx] = np.expm1(x[:, :, idx])
        y = np.stack([y[:n_gen].numpy() for y in real_y_list], axis=-1)
        return evaluate_all(x[:n_gen], gen_x, y, gen_y, schema.vocab_sizes)

    tr_res = eval_set(X_tr, Y_tr, "train")
    te_res = eval_set(X_te, Y_te, "test")

    ks_r = te_res["ks"]["mean_ks"] / tr_res["ks"]["mean_ks"]
    log(f"KS  Train={tr_res['ks']['mean_ks']:.4f}  Test={te_res['ks']['mean_ks']:.4f}  Ratio={ks_r:.2f}  {'NO OVERFIT' if ks_r < 1.5 else 'OVERFIT'}")
    log(f"Max KS: {te_res['ks']['max_ks']:.4f} | JSD Train={tr_res['jsd']['mean_jsd']:.4f} Test={te_res['jsd']['mean_jsd']:.4f}")
    for i, spec in enumerate(schema.continuous):
        log(f"  {spec.name:20s} Train KS={tr_res['ks']['per_feature'][i]:.4f}  Test KS={te_res['ks']['per_feature'][i]:.4f}")

    # --- 10. Save everything ---
    with open(out / "results.json", "w") as f:
        json.dump({
            "ks_train": tr_res["ks"], "ks_test": te_res["ks"],
            "jsd_train": tr_res["jsd"], "jsd_test": te_res["jsd"],
            "overfitting_ratio": float(ks_r),
            "num_windows": int(X.shape[0]),
            "trend_loss": [float(h1["train_loss"][0]), float(h1["train_loss"][-1])],
            "diff_cont":  [float(h2["train_loss_cont"][0]), float(h2["train_loss_cont"][-1])],
            "diff_disc":  [float(h2["train_loss_disc"][0]), float(h2["train_loss_disc"][-1])],
            "epochs_run": n_ep,
        }, f, indent=2)

    log("=== DONE ===")
    return ks_r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TADS-ICS Universal Experiment Runner")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints + results")

    # Data sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Pre-extracted tensor directory")
    src.add_argument("--csv", help="FARAONIC CSV file path")
    src.add_argument("--pcap", help="PCAP file path")

    parser.add_argument("--csv-rows", type=int, default=500000, help="Max CSV rows (default: 500K)")
    parser.add_argument("--config", help="Config JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)

    args = parser.parse_args()
    run(args)
