"""Quick FARAONIC training + overfitting check."""
import torch, numpy as np, json, time, sys
sys.path.insert(0, "D:/programme")
from extractor.schema import FeatureSchema
from extractor.faraonic_reader import read_faraonic_csv
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, PayloadLookup, StubSampler
from diffusion.utils.normalisation import Normalizer
from diffusion.utils.metrics import evaluate_all

print("=== FARAONIC + log fix (200K NORMAL) ===")
t0 = time.time()
records = read_faraonic_csv(
    "D:/programme/dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv",
    max_rows=200000, label_filter="NORMAL")
print(f"{len(records)} records in {time.time()-t0:.1f}s")

schema = FeatureSchema.default_modbus()
X_w, Y_w, stats = build_training_data(records, schema, window_length=64, stride=16)
X_w_t = torch.from_numpy(X_w).float()
schema = schema.adapt_to_data(X_w_t)
print(f"Windows: {X_w.shape}  log: {stats.get('log_features')}  NaN: {torch.isnan(X_w_t).any().item()}")

N = X_w.shape[0]; nt = int(N*0.7); nv = int(N*0.15)
X_tr_np = X_w[:nt]; Y_tr_np = [Y_w[j][:nt] for j in range(schema.d_d)]
X_te_np = X_w[nt+nv:]; Y_te_np = [Y_w[j][nt+nv:] for j in range(schema.d_d)]

X_tr = torch.from_numpy(X_tr_np).float()
Y_tr = [torch.from_numpy(y).long() for y in Y_tr_np]

norm = Normalizer(schema.d_c)
norm.mean = torch.tensor(stats["mean"], dtype=torch.float32)
norm.std  = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-8)

cfg = DiffusionConfig()
cfg.trend.epochs = 20; cfg.ddpm.epochs = 20; cfg.window_length = 64
cfg.trend.d_model = 64; cfg.trend.num_layers = 2
cfg.ddpm.d_model = 64; cfg.ddpm.num_layers = 2; cfg.ddpm.diffusion_steps = 100
cfg.mask.d_model = 64; cfg.mask.num_layers = 2; cfg.mask.diffusion_steps = 100
device = torch.device("cuda")

trainer = MaskDDPMTrainer(cfg, schema, device=device)
trainer.train_trend(X_tr.to(device))
trainer.train_diffusion(X_tr.to(device), Y_tr)
trainer.ddpm_ema.apply()

# Build lookup/stub from RAW numpy
X_tr_raw = norm.inverse_transform(X_tr.float()).cpu().numpy()
Y_tr_flat = np.stack([y.numpy() for y in Y_tr], axis=-1).reshape(-1, schema.d_d)
payload_lu = PayloadLookup()
payload_lu.fit(X_tr_raw.reshape(-1, schema.d_c), Y_tr_flat)

card = [len(np.unique(X_tr_raw.reshape(-1, schema.d_c)[:, i])) for i in range(schema.d_c)]
dead = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
low_card = [i for i, c in enumerate(card) if 1 < c < 10 and schema.continuous[i].var_type.name == "TYPE4"]
stub_idx = sorted(set(dead + low_card))
stub_s = StubSampler()
if stub_idx:
    stub_s.fit(X_tr_raw.reshape(-1, schema.d_c), stub_idx)
print(f"Stub features: {[schema.continuous[i].name for i in stub_idx]}")

sampler = MaskDDPMSampler(trainer.trend_model, trainer.ddpm, trainer.mask_diff,
                          norm, schema, payload_lookup=payload_lu, device=device)
if stub_s:
    sampler.stub_sampler = stub_s

X_gen, Y_gen = sampler.generate(num_samples=min(30, nt), num_unmask_steps=30)
gen_x = X_gen.cpu().numpy()
gen_y = np.stack([y.cpu().numpy() for y in Y_gen], axis=-1)

# Evaluate on TRAIN (numpy all the way)
tr_x = norm.inverse_transform(X_tr[:len(gen_x)].float()).cpu().numpy()
tr_y = np.stack([y[:len(gen_x)].numpy() for y in Y_tr], axis=-1)
tr_res = evaluate_all(tr_x, gen_x, tr_y, gen_y, schema.vocab_sizes)

# Evaluate on TEST
X_te_t = torch.from_numpy(X_te_np[:len(gen_x)]).float()
te_x = norm.inverse_transform(X_te_t).cpu().numpy()
te_y = np.stack([y[:len(gen_x)] for y in Y_te_np], axis=-1)
te_res = evaluate_all(te_x, gen_x, te_y, gen_y, schema.vocab_sizes)

# Results
print(f"\n=== OVERFITTING CHECK ===")
ks_r = te_res["ks"]["mean_ks"] / tr_res["ks"]["mean_ks"]
jsd_r = te_res["jsd"]["mean_jsd"] / tr_res["jsd"]["mean_jsd"]
print(f"KS  Train={tr_res['ks']['mean_ks']:.4f}  Test={te_res['ks']['mean_ks']:.4f}  Ratio={ks_r:.2f}  {'NO OVERFIT' if ks_r<1.5 else 'OVERFIT'}")
print(f"JSD Train={tr_res['jsd']['mean_jsd']:.4f}  Test={te_res['jsd']['mean_jsd']:.4f}  Ratio={jsd_r:.2f}  {'NO OVERFIT' if jsd_r<1.5 else 'OVERFIT'}")
print(f"\nPer-feature KS:")
for i, spec in enumerate(schema.continuous):
    tr = tr_res["ks"]["per_feature"][i]
    te = te_res["ks"]["per_feature"][i]
    r = te / tr if tr > 0.01 else 999
    print(f"  {spec.name:20s} train={tr:.4f} test={te:.4f} ratio={r:4.1f}  {'OVERFIT' if r>1.5 else 'OK'}")
