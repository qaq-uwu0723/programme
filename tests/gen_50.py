"""Generate 50 windows for pipeline test with V2.9 checkpoint."""
import sys, time, json
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from extractor.schema import FeatureSchema, VariableType
from extractor.faraonic_reader import read_faraonic_csv
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, StubSampler
from diffusion.utils.normalisation import Normalizer

OUT = Path("tests/pipeline_test/generated")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading model...", flush=True)
schema = FeatureSchema.default_modbus()
with open("checkpoints/exp_1m_type4/schema_info.json") as f:
    info = json.load(f)
for i, spec in enumerate(schema.continuous):
    spec.var_type = VariableType.TYPE4 if i in set(info["active_indices"]) else VariableType.TYPE6

normalizer = Normalizer.load("checkpoints/exp_1m_type4/normalizer.json")
device = torch.device("cuda")
trainer = MaskDDPMTrainer(DiffusionConfig(), schema, device=device)
trainer.load("checkpoints/exp_1m_type4")
trainer.ddpm_ema.apply()

# StubSampler
records = read_faraonic_csv("dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv", max_rows=50000, label_filter=None)
X_w, Y_w, _ = build_training_data(records, schema, window_length=128, stride=16)
X_raw = normalizer.inverse_transform(torch.from_numpy(X_w.reshape(-1, schema.d_c))).numpy()
for idx in normalizer.log_features:
    X_raw[:, idx] = np.expm1(X_raw[:, idx].clip(max=80.0))
dead = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
stub_s = StubSampler()
if dead: stub_s.fit(X_raw, dead)

sampler = MaskDDPMSampler(trainer.trend_model, trainer.ddpm, trainer.mask_diff, normalizer, schema, device=device)
if dead: sampler.stub_sampler = stub_s

print(f"Generating 50 windows...", flush=True)
t0 = time.time()
X_gen, Y_gen = sampler.generate(num_samples=50, num_unmask_steps=50)
print(f"Done in {time.time()-t0:.1f}s", flush=True)

x_np = X_gen.cpu().numpy()
print(f"NaN={np.isnan(x_np).sum()} Inf={np.isinf(x_np).sum()}")
print(f"payload_size: mean={x_np[:,:,4].mean():.1f} std={x_np[:,:,4].std():.1f} min={x_np[:,:,4].min():.1f} max={x_np[:,:,4].max():.1f}")
for j, spec in enumerate(schema.discrete):
    yn = Y_gen[j].cpu().numpy()
    print(f"{spec.name}: unique={len(np.unique(yn))}")

np.save(OUT / "gen_X.npy", x_np)
for j in range(len(Y_gen)):
    np.save(OUT / f"gen_Y_{j}.npy", Y_gen[j].cpu().numpy())
print(f"Saved to {OUT}/")
