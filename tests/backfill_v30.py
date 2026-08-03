"""Backfill V3.0 checkpoint with correct normalizer.json + stub distributions.

The V3.0 training script double-normalized: build_training_data already z-scores
X_w, then train_1m.py re-fitted a Normalizer on the standardized data (mean≈0/std≈1,
log_features=[]). The model itself trained on z-scored data so its weights are valid —
only the normalizer metadata is wrong. This recomputes raw stats from a fresh CSV
sample and writes:
  checkpoints/exp_1m_type4/normalizer.json          (raw mean/std + log_features=[3])
  checkpoints/exp_1m_type4/stub_distributions.npz   (Type6 empirical distributions)

Usage:
    .venv/Scripts/python.exe tests/backfill_v30.py
"""
import sys, time, random, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema
from extractor.feature_builder import packet_to_features
from diffusion.sampling.sampler import StubSampler
from diffusion.utils.normalisation import Normalizer

CSV_PATH = "dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv"
OUT = "checkpoints/exp_1m_type4"
SAMPLE_ROWS = 500_000
SEED = 42
# Exclude inter_arrival session-gap artifacts (>1s) from the empirical
# distribution. The raw data has ~48% of values > 10s (CSV session-stitching gaps,
# e.g. 20 days) which are data-collection artifacts, not real packet timing.
# Reproducing them breaks request-response pairing in the checker (requests time
# out). Excluding >1s keeps only sub-second real timing (see MQ2 in TO_DEBUG_LIST);
# inter_arrival KS is already pinned ~0.49 by removing the 20-day mode, so this
# tightening costs negligible extra fidelity.
INTER_ARRIVAL_MAX_NS = 1_000_000_000


def reservoir_sample(path: str, n: int):
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)
        col = {h: i for i, h in enumerate(headers)}
        reservoir = []
        for i, row in enumerate(reader):
            if i < n:
                reservoir.append(row)
            else:
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = row
    return headers, col, reservoir


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t0 = time.time()
    print("Reservoir sampling...")
    headers, col, rows = reservoir_sample(CSV_PATH, SAMPLE_ROWS)
    print(f"  sampled {len(rows)} rows")

    # Reconstruct PacketRecords using the FIXED direction-aware reader
    from extractor.faraonic_reader import read_faraonic_csv
    # Reuse reader logic via a temp file-free path: import records_from_rows from train_1m
    sys.path.insert(0, str(Path(__file__).parent))
    from train_1m import records_from_rows
    records = records_from_rows(headers, col, rows)
    print(f"  {len(records)} valid records")

    # Raw feature matrix (log1p applied to inter_arrival_ns by packet_to_features)
    schema = FeatureSchema.default_modbus()
    X_flat, _ = packet_to_features(records, schema)
    print(f"  X_flat: {X_flat.shape}")

    # Classify Type4/Type6 as training did (std<1e-4 or cardinality<15 → Type6)
    schema = schema.adapt_to_data(X_flat)
    active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
    stub = [s.name for s in schema.continuous if s.var_type.name == "TYPE6"]
    print(f"  Active: {active}")
    print(f"  Stub: {stub}")

    # Raw stats (matching build_training_data normalization)
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0).clip(min=1e-8)
    normalizer = Normalizer(schema.d_c)
    normalizer.mean = torch.tensor(mean, dtype=torch.float32)
    normalizer.std = torch.tensor(std, dtype=torch.float32)
    normalizer.log_features = [3]
    normalizer.log_bounds = {3: (float(X_flat[:, 3].min()), float(X_flat[:, 3].max()))}
    normalizer.save(str(Path(OUT) / "normalizer.json"))
    print(f"  normalizer.json -> mean[3]={mean[3]:.4f} std[3]={std[3]:.4f} "
          f"log_features=[3] log_bounds={normalizer.log_bounds}")

    # Stub distributions (Type6 features + inter_arrival_ns, raw units).
    # inter_arrival_ns is routed to empirical sampling too: its raw distribution is
    # degenerate (49% at 1ns floor, ~30% session-gap spike) which Gaussian DDPM
    # structurally cannot represent. Override it at generation with raw ns samples,
    # capped to INTER_ARRIVAL_MAX_NS to avoid session-stitch 20-day gaps.
    stub_idx = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
    if 3 not in stub_idx:
        stub_idx = sorted(stub_idx + [3])
    print(f"  stub_idx: {stub_idx}")
    if stub_idx:
        # column 3 is log1p'd in X_flat — store RAW ns for empirical replacement
        X_stub_fit = X_flat.copy()
        X_stub_fit[:, 3] = np.expm1(X_stub_fit[:, 3])
        stub_s = StubSampler()
        stub_s.fit(X_stub_fit, stub_idx)
        # Remove session-gap artifacts (inter_arrival > checker timeout 10s)
        ia_vals = stub_s._distributions[3]
        stub_s._distributions[3] = ia_vals[ia_vals <= INTER_ARRIVAL_MAX_NS]
        stub_s.save(str(Path(OUT) / "stub_distributions.npz"))
        for i in stub_idx:
            vals = stub_s._distributions[i]
            print(f"    feat[{i}] {schema.continuous[i].name}: {len(vals)} vals, "
                  f"unique={len(np.unique(vals))}, range=[{vals.min():.3g}, {vals.max():.3g}]")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
