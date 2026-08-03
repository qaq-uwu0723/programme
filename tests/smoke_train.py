"""Quick smoke test for train_1m.py — 10K rows, 2 epochs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np
from extractor.schema import FeatureSchema
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.utils.normalisation import Normalizer
from tests.train_1m import reservoir_sample_csv, records_from_rows

CSV = "dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv"

print("1. Sample 10K rows...")
headers, col, rows = reservoir_sample_csv(CSV, 10_000)
print("2. Convert to records...")
records = records_from_rows(headers, col, rows)
print(f"   {len(records)} records")

print("3. Build windows...")
schema = FeatureSchema.default_modbus()
schema.window_length = 128
X_w, Y_w, stats = build_training_data(records, schema, window_length=128, stride=16)
print(f"   {X_w.shape[0]} windows")

schema = schema.adapt_to_data(X_w)
active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
print(f"   Active: {active} ({len(active)})")

print("4. Normalize...")
train_x = torch.from_numpy(X_w).float()
train_y = [torch.from_numpy(y).long() for y in Y_w]
normalizer = Normalizer(schema.d_c).fit(train_x)
train_x_norm = normalizer.transform(train_x)

print("5. Quick train (2 epochs each)...")
config = DiffusionConfig()
config.trend.epochs = 2
config.ddpm.epochs = 2
config.mask.epochs = 2
config.trend.batch_size = 32
config.ddpm.batch_size = 32
config.mask.batch_size = 32

device = torch.device("cuda")
trainer = MaskDDPMTrainer(config, schema, device=device)
print(f"   d_c_active={trainer.d_c_active}, indices={trainer.active_indices}")

trainer.train_trend(train_x_norm.to(device))
trainer.train_diffusion(train_x_norm.to(device), train_y)
print("   OK — smoke test passed")
