"""Evaluate exp_1m_type4 checkpoint — generation quality vs training data."""
import sys, time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema
from extractor.faraonic_reader import read_faraonic_csv
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, StubSampler
from diffusion.utils.normalisation import Normalizer

OUT = Path("tests/eval_1m")
OUT.mkdir(parents=True, exist_ok=True)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)

log("Loading 10K reference data...")
schema = FeatureSchema.default_modbus()
records = read_faraonic_csv(
    "dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv",
    max_rows=10000, label_filter=None
)
X_w, Y_w, stats = build_training_data(records, schema, window_length=128, stride=16)
log(f"  {X_w.shape[0]} windows (all labels)")
X_flat = X_w.reshape(-1, schema.d_c)
Y_flat = np.stack(Y_w, axis=-1).reshape(-1, schema.d_d)

# Restore schema configuration from checkpoint (NOT fresh adapt_to_data)
import json
from extractor.schema import VariableType
with open("checkpoints/exp_1m_type4/schema_info.json") as f:
    info = json.load(f)
active_indices = set(info["active_indices"])
for i, spec in enumerate(schema.continuous):
    if i in active_indices:
        spec.var_type = VariableType.TYPE4
    else:
        spec.var_type = VariableType.TYPE6

active = [s for s in schema.continuous if s.var_type.name == "TYPE4"]
log(f"  Active ({len(active)}): {[s.name for s in active]} (from checkpoint)")
log(f"  active_indices: {sorted(active_indices)}")

# Load normalizer, denorm training data
normalizer = Normalizer.load("checkpoints/exp_1m_type4/normalizer.json")
X_raw = normalizer.inverse_transform(torch.from_numpy(X_flat)).numpy()
for idx in normalizer.log_features:
    # clamp in log space using observed data bounds (see sampler._inverse_log_transform)
    bounds = normalizer.log_bounds.get(idx)
    if bounds:
        lo, hi = bounds
        X_raw[:, idx] = np.expm1(np.clip(X_raw[:, idx], lo, hi))
    else:
        X_raw[:, idx] = np.expm1(X_raw[:, idx].clip(max=80.0))

# All features are in comparison — no adapt_to_data needed

# Build model
log("Loading model...")
config = DiffusionConfig()
device = torch.device("cuda")
trainer = MaskDDPMTrainer(config, schema, device=device)
trainer.load("checkpoints/exp_1m_type4")
trainer.ddpm_ema.apply()

# Stub sampler (fit on raw data)
dead = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
stub_s = StubSampler()
if dead:
    stub_s.fit(X_raw, dead)

# Sampler (no PayloadLookup)
sampler = MaskDDPMSampler(
    trainer.trend_model, trainer.ddpm, trainer.mask_diff,
    normalizer, schema, device=device,
)
if dead:
    sampler.stub_sampler = stub_s

# Generate
N_ROUNDS = 5
N_PER_ROUND = 20
log(f"Generating {N_ROUNDS}x{N_PER_ROUND}={N_ROUNDS*N_PER_ROUND} windows...")

all_X = []
all_Y = {spec.name: [] for spec in schema.discrete}
for r in range(N_ROUNDS):
    t0 = time.time()
    X_gen, Y_gen = sampler.generate(num_samples=N_PER_ROUND, num_unmask_steps=50)
    dt = time.time() - t0
    x_np = X_gen.cpu().numpy()
    all_X.append(x_np)
    for j, spec in enumerate(schema.discrete):
        all_Y[spec.name].append(Y_gen[j].cpu().numpy())
    nan_c = np.isnan(x_np).sum()
    inf_c = np.isinf(x_np).sum()
    log(f"  Round {r+1}: {dt:.1f}s  {'OK' if nan_c==0 and inf_c==0 else f'NAN={nan_c} INF={inf_c}'}")

# Aggregate
X_gen_flat = np.concatenate(all_X, axis=0).reshape(-1, schema.d_c)

log("\n" + "=" * 70)
log("CONTINUOUS FEATURES")
log(f"{'Feature':<22s} {'Gen Mean':>12s} {'Train Mean':>12s} {'Gen Std':>12s} {'Train Std':>12s} {'Gen Min':>12s} {'Gen Max':>12s}")
log("-" * 90)
for i, spec in enumerate(schema.continuous):
    gcol = X_gen_flat[:, i]
    tcol = X_raw[:, i]
    log(f"{spec.name:<22s} {np.mean(gcol):>12.4f} {np.mean(tcol):>12.4f} {np.std(gcol):>12.4f} {np.std(tcol):>12.4f} {np.min(gcol):>12.4f} {np.max(gcol):>12.4f}")

log("\nDISCRETE FEATURES")
for j, spec in enumerate(schema.discrete):
    gcol = np.concatenate(all_Y[spec.name], axis=0).flatten()
    tcol = Y_flat[:, j]
    g_vals, g_counts = np.unique(gcol, return_counts=True)
    t_vals, t_counts = np.unique(tcol, return_counts=True)
    g_total = len(gcol)
    t_total = len(tcol)
    log(f"\n{spec.name}: gen_unique={len(g_vals)} train_unique={len(t_vals)}")
    log(f"  {'Val':>6s}  {'Gen%':>8s}  {'Train%':>8s}")
    all_vals = sorted(set(g_vals.tolist()) | set(t_vals.tolist()))
    for v in all_vals[:12]:
        gp = np.sum(gcol == v) / g_total * 100
        tp = np.sum(tcol == v) / t_total * 100
        log(f"  {v:>6d}  {gp:>7.2f}%  {tp:>7.2f}%")

# Save
np.save(OUT / "gen_X.npy", np.concatenate(all_X, axis=0))
for j, spec in enumerate(schema.discrete):
    np.save(OUT / f"gen_Y_{j}.npy", np.concatenate(all_Y[spec.name], axis=0))
log(f"\nSaved to {OUT.resolve()}/")
