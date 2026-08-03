"""Evaluate V3.0 checkpoint — KS (continuous) + JSD (discrete) vs real data.

Usage:
    .venv/Scripts/python.exe tests/eval_v30_metrics.py
"""
import sys, time, random, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema, VariableType
from extractor.feature_builder import packet_to_features
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.sampling.sampler import MaskDDPMSampler, StubSampler
from diffusion.utils.normalisation import Normalizer
from diffusion.utils.metrics import evaluate_all

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/exp_1m_type4"
CSV = "dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv"
N_REAL_ROWS = 300_000
N_GEN_WINDOWS = 300
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    schema = FeatureSchema.default_modbus()

    # --- Real raw features (fixed direction-aware reader) ---
    log(f"Loading {N_REAL_ROWS} real rows...")
    from tests.backfill_v30 import reservoir_sample  # noqa: E402
    from tests.train_1m import records_from_rows  # noqa: E402
    headers, col, rows = reservoir_sample(CSV, N_REAL_ROWS)
    records = records_from_rows(headers, col, rows)
    X_real_flat, Y_real_flat = packet_to_features(records, schema)
    X_real_flat[:, 3] = np.expm1(X_real_flat[:, 3])  # log1p -> raw ns
    log(f"  real: {len(records)} records, X {X_real_flat.shape}")

    # --- Restore schema routing from checkpoint ---
    with open(f"{CKPT}/schema_info.json") as f:
        info = json.load(f)
    active_indices = set(info["active_indices"])
    for i, spec in enumerate(schema.continuous):
        spec.var_type = VariableType.TYPE4 if i in active_indices else VariableType.TYPE6
    log(f"  active: {[s.name for s in schema.continuous if s.var_type.name=='TYPE4']}")

    # --- Load model ---
    log("Loading model...")
    normalizer = Normalizer.load(f"{CKPT}/normalizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MaskDDPMTrainer(DiffusionConfig(), schema, device=device)
    trainer.load(CKPT)
    trainer.ddpm_ema.apply()

    stub_path = Path(f"{CKPT}/stub_distributions.npz")
    sampler = MaskDDPMSampler(
        trainer.trend_model, trainer.ddpm, trainer.mask_diff,
        normalizer, schema, device=device,
    )
    if stub_path.exists():
        sampler.stub_sampler = StubSampler.load(str(stub_path))

    # --- Generate ---
    log(f"Generating {N_GEN_WINDOWS} windows...")
    X_gen, Y_gen = sampler.generate(num_samples=N_GEN_WINDOWS, num_unmask_steps=50)
    X_gen_flat = X_gen.cpu().numpy().reshape(-1, schema.d_c)
    Y_gen_flat = np.stack([y.cpu().numpy() for y in Y_gen], axis=-1).reshape(-1, schema.d_d)
    log(f"  gen: {X_gen_flat.shape[0]} packets")

    # --- Evaluate ---
    vocab = schema.vocab_sizes
    res = evaluate_all(
        X_real_flat.reshape(1, -1, schema.d_c),
        X_gen_flat.reshape(1, -1, schema.d_c),
        Y_real_flat.reshape(1, -1, schema.d_d),
        Y_gen_flat.reshape(1, -1, schema.d_d),
        vocab,
    )

    names = [s.name for s in schema.continuous]
    print("\n=== KS (continuous, raw units) ===")
    for i, n in enumerate(names):
        print(f"  {n:<18s} KS={res['ks']['per_feature'][i]:.4f}")
    print(f"  Mean KS={res['ks']['mean_ks']:.4f}  Max KS={res['ks']['max_ks']:.4f}")

    dnames = [s.name for s in schema.discrete]
    jsd_all = res['jsd']['per_feature']
    print("\n=== JSD (discrete) ===")
    for i, n in enumerate(dnames):
        print(f"  {n:<18s} JSD={jsd_all[i]:.4f}")
    # txid (idx 3) is sampler-overridden, not model-learned — exclude from mean
    learned_idx = [i for i, n in enumerate(dnames) if n != "transaction_id"]
    mean_jsd_learned = np.mean([jsd_all[i] for i in learned_idx])
    print(f"  Mean JSD (5 learned, excl txid)={mean_jsd_learned:.4f}")
    print(f"  Mean JSD (all 6)={res['jsd']['mean_jsd']:.4f}")

    # --- Save report ---
    out = Path("tests/eval_1m/v30_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ks": res["ks"], "jsd": res["jsd"],
        "mean_jsd_learned": float(mean_jsd_learned),
        "n_real_records": len(records), "n_gen_packets": int(X_gen_flat.shape[0]),
    }, indent=2), encoding="utf-8")
    log(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
