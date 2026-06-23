"""FARAONIC 大样本正式训练 (500K rows, d=128, 200+300 epochs)."""
import torch, numpy as np, json, time, sys, os
sys.path.insert(0, "D:/programme")

from extractor.schema import FeatureSchema
from extractor.faraonic_reader import read_faraonic_csv
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, PayloadLookup, StubSampler
from diffusion.utils.normalisation import Normalizer
from diffusion.utils.metrics import evaluate_all

LOG_PATH = "D:/programme/experiments/training_progress.log"

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# --- Init ---
with open(LOG_PATH, "w") as f:
    f.write(f"FARAONIC Training Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n")

log("Reading FARAONIC (500K NORMAL rows)...")
t0 = time.time()
records = read_faraonic_csv(
    "D:/programme/dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv",
    max_rows=500000, label_filter="NORMAL")
log(f"  {len(records)} records in {time.time()-t0:.1f}s")

log("Building training windows (L=128, stride=16)...")
t0 = time.time()
schema = FeatureSchema.default_modbus()
X_w, Y_w, stats = build_training_data(records, schema, window_length=128, stride=16)
log(f"  X={X_w.shape}  {stats['num_windows']} windows in {time.time()-t0:.1f}s")
log(f"  log_features: {stats.get('log_features')}")
log(f"  inter_arrival (log1p): mean={stats['mean'][3]:.2f} std={stats['std'][3]:.2f}")

X_w_t = torch.from_numpy(X_w).float()
schema = schema.adapt_to_data(X_w_t)
active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
log(f"  Active features: {active} (d_c={len(active)})")

# Split
N = X_w.shape[0]; nt = int(N * 0.70); nv = int(N * 0.15)
X_tr = X_w_t[:nt]; Y_tr = [torch.from_numpy(Y_w[j][:nt]).long() for j in range(schema.d_d)]
X_val = X_w_t[nt:nt+nv]; X_te_np = X_w[nt+nv:]
Y_te_np = [Y_w[j][nt+nv:] for j in range(schema.d_d)]
log(f"  Split: train={nt} val={nv} test={N-nt-nv}")

# Normalizer
norm = Normalizer(schema.d_c)
norm.mean = torch.tensor(stats["mean"], dtype=torch.float32)
norm.std  = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-8)

# Config
cfg = DiffusionConfig()
cfg.trend.epochs = 200; cfg.ddpm.epochs = 300
cfg.trend.d_model = 128; cfg.trend.num_layers = 4
cfg.ddpm.d_model = 128; cfg.ddpm.num_layers = 4; cfg.ddpm.diffusion_steps = 600
cfg.mask.d_model = 128; cfg.mask.num_layers = 4; cfg.mask.diffusion_steps = 600
cfg.window_length = 128; cfg.seed = 42
device = torch.device("cuda")
log(f"  Device: {device} | d_model=128, 4 layers, K=600, 200+300 epochs")

# --- Stage 1 ---
log("="*60)
log("STAGE 1: Trend Training (200 epochs)")
log("="*60)
trainer = MaskDDPMTrainer(cfg, schema, device=device)
t0 = time.time()
hist1 = trainer.train_trend(X_tr.to(device))
elapsed = time.time() - t0
log(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
log(f"  Loss: {hist1['train_loss'][0]:.4f} -> {hist1['train_loss'][-1]:.4f}")

# --- Stage 2 ---
log("="*60)
log("STAGE 2: Diffusion Training (300 epochs)")
log("="*60)
t0 = time.time()
hist2 = trainer.train_diffusion(X_tr.to(device), Y_tr)
elapsed = time.time() - t0
log(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
log(f"  Cont: {hist2['train_loss_cont'][0]:.4f} -> {hist2['train_loss_cont'][-1]:.4f}")
log(f"  Disc: {hist2['train_loss_disc'][0]:.4f} -> {hist2['train_loss_disc'][-1]:.4f}")
log(f"  Total:{hist2['train_loss_total'][0]:.4f} -> {hist2['train_loss_total'][-1]:.4f}")

# Save model
out_dir = "D:/programme/checkpoints/exp04_faraonic"
os.makedirs(out_dir, exist_ok=True)
trainer.ddpm_ema.apply()
trainer.save(out_dir)
with open(f"{out_dir}/normalizer.json", "w") as f:
    json.dump({
        "mean": norm.mean.tolist(), "std": norm.std.tolist(),
        "log_features": stats.get("log_features", []),
    }, f)
log(f"Model saved to {out_dir}/")

# --- Generate ---
log("="*60)
log("GENERATION & EVALUATION")
log("="*60)
# Build lookup/stub from raw training data
X_tr_raw = norm.inverse_transform(X_tr.float()).cpu().numpy()
Y_tr_flat = torch.stack(Y_tr, dim=-1).reshape(-1, schema.d_d).numpy()
payload_lu = PayloadLookup()
payload_lu.fit(X_tr_raw.reshape(-1, schema.d_c), Y_tr_flat)

card = [len(np.unique(X_tr_raw.reshape(-1, schema.d_c)[:, i])) for i in range(schema.d_c)]
dead = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
low_card = [i for i, c in enumerate(card) if 1 < c < 10 and schema.continuous[i].var_type.name == "TYPE4"]
stub_idx = sorted(set(dead + low_card))
stub_s = StubSampler()
if stub_idx:
    stub_s.fit(X_tr_raw.reshape(-1, schema.d_c), stub_idx)

sampler = MaskDDPMSampler(trainer.trend_model, trainer.ddpm, trainer.mask_diff,
                          norm, schema, payload_lookup=payload_lu, device=device)
if stub_s:
    sampler.stub_sampler = stub_s

n_gen = min(50, nt)
X_gen, Y_gen = sampler.generate(num_samples=n_gen, num_unmask_steps=50)
gen_x = X_gen.cpu().numpy()
gen_y = np.stack([y.cpu().numpy() for y in Y_gen], axis=-1)

# Evaluate on TRAIN
tr_x = norm.inverse_transform(X_tr[:n_gen].float()).cpu().numpy()
tr_y = np.stack([y[:n_gen].numpy() for y in Y_tr], axis=-1)
tr_res = evaluate_all(tr_x, gen_x, tr_y, gen_y, schema.vocab_sizes)

# Evaluate on TEST
X_te_t = torch.from_numpy(X_te_np[:n_gen]).float()
te_x = norm.inverse_transform(X_te_t).cpu().numpy()
te_y = np.stack([y[:n_gen] for y in Y_te_np], axis=-1)
te_res = evaluate_all(te_x, gen_x, te_y, gen_y, schema.vocab_sizes)

# --- Results ---
log("="*60)
log("RESULTS")
log("="*60)
ks_r = te_res["ks"]["mean_ks"] / tr_res["ks"]["mean_ks"]
log(f"KS  Train={tr_res['ks']['mean_ks']:.4f}  Test={te_res['ks']['mean_ks']:.4f}  Ratio={ks_r:.2f}  {'NO OVERFIT' if ks_r<1.5 else 'OVERFIT'}")
log(f"JSD Train={tr_res['jsd']['mean_jsd']:.4f}  Test={te_res['jsd']['mean_jsd']:.4f}")
log(f"Mean KS: {te_res['ks']['mean_ks']:.4f}  Max KS: {te_res['ks']['max_ks']:.4f}")
for i, spec in enumerate(schema.continuous):
    log(f"  {spec.name:20s} KS={te_res['ks']['per_feature'][i]:.4f}")

log("="*60)
log("TRAINING COMPLETE")

# Save full results
with open(f"{out_dir}/results.json", "w") as f:
    json.dump({
        "ks_train": tr_res["ks"], "ks_test": te_res["ks"],
        "jsd_train": tr_res["jsd"], "jsd_test": te_res["jsd"],
        "trend_loss": [hist1["train_loss"][0], hist1["train_loss"][-1]],
        "diff_loss": [hist2["train_loss_cont"][-1], hist2["train_loss_disc"][-1]],
        "overfitting_ks_ratio": ks_r,
        "num_windows": N, "train_windows": nt,
    }, f, indent=2)
