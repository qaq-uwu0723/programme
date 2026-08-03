"""Train Mask-DDPM on 1M randomly-sampled rows from FARAONIC dataset.

Usage:
    python -m tests.train_1m
    # or: .venv/Scripts/python.exe tests/train_1m.py
"""
import sys, os, time, random, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor.schema import FeatureSchema
from extractor.faraonic_reader import read_faraonic_csv
from extractor.feature_builder import build_training_data
from diffusion.config import DiffusionConfig
from diffusion.training.trainer import MaskDDPMTrainer
from diffusion.utils.normalisation import Normalizer


CSV_PATH = "dataset/FARAONIC/Modbus_TCP_ Cybersecurity_Dataset_Training.csv"
OUT_DIR = "checkpoints/exp_15m_type4"
TARGET_ROWS = 1_500_000
WINDOW_LENGTH = 128
STRIDE = 16
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def reservoir_sample_csv(path: str, n: int) -> list:
    """Reservoir sample n rows from a semicolon-delimited CSV."""
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

    print(f"Reservoir sampled {len(reservoir)} rows from {i+1} total")
    return headers, col, reservoir


def records_from_rows(headers, col, rows):
    """Convert CSV rows to PacketRecord list (same logic as read_faraonic_csv)."""
    from extractor.pcap_reader import PacketRecord

    records = []
    prev_ts = None

    for row in rows:
        try:
            ts = float(row[col["timestamp"]])
            src_ip = row[col["IP_src"]]
            dst_ip = row[col["IP_dst"]]
            src_port = int(row[col["TCP_sport"]])
            dst_port = int(row[col["TCP_dport"]])
        except (ValueError, IndexError):
            continue

        direction = "c2s" if dst_port == 502 else "s2c"

        if prev_ts is not None:
            inter_arrival_ns = int(max(1, (ts - prev_ts) * 1_000_000_000))
        else:
            inter_arrival_ns = 50_000_000
        prev_ts = ts

        try:
            prefix = "ModbusTCPRequest_" if direction == "c2s" else "ModbusTCPResponse_"
            func_code = int(row[col[f"{prefix}func_code"]] or 0)
            unit_id = int(row[col[f"{prefix}unit_id"]] or 1) % 248
            txid = int(row[col[f"{prefix}trans_id"]] or 0) % 65536
        except (ValueError, IndexError):
            continue

        reg_addr = 0
        for rc in ["ModbusReadDiscreteInputsRequest_reference_number",
                    "ModbusWriteMultipleCoilsRequest_reference_number"]:
            if rc in col and row[col[rc]]:
                reg_addr = int(row[col[rc]])
                break

        quantity = 1
        for qc in ["ModbusReadDiscreteInputsRequest_bit_count",
                    "ModbusWriteMultipleCoilsRequest_bit_count"]:
            if qc in col and row[col[qc]]:
                quantity = int(row[col[qc]])
                break

        reg_val_0 = 0
        for rc in ["ModbusReadDiscreteInputsResponse_input_status",
                    "ModbusWriteMultipleCoilsResponse_bit_count"]:
            if rc in col and row[col[rc]]:
                val_str = row[col[rc]]
                if val_str:
                    try:
                        reg_val_0 = int(val_str, 16) if val_str.startswith("0x") else int(val_str)
                    except ValueError:
                        reg_val_0 = 0
                break

        payload_size = int(row[col["IP_len"]] or 40)

        records.append(PacketRecord(
            ts_ns=int(ts * 1_000_000_000),
            inter_arrival_ns=inter_arrival_ns,
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            direction=direction,
            transaction_id=txid,
            protocol_id=0,
            unit_id=unit_id,
            function_code=func_code,
            is_exception=False,
            exception_code=0,
            pdu_data=b"",
            payload_size=payload_size,
            register_address=reg_addr,
            register_values=[reg_val_0, 0, 0],
            quantity=max(1, quantity),
        ))

    return records


def main():
    t0 = time.time()

    # Create output dir + training log early (for monitor)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(str(out / "training.log"), "w", buffering=1)

    def tee(msg: str) -> None:
        """Write to both stdout and training.log with [HH:MM:SS] timestamp."""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    # ---- Step 1: Randomly sample 1M rows ----
    tee(f"Step 1: Reservoir sampling {TARGET_ROWS} rows from CSV...")
    headers, col, rows = reservoir_sample_csv(CSV_PATH, TARGET_ROWS)

    # Show label distribution
    cls_idx = col["Classification"]
    labels = {}
    for r in rows:
        lbl = r[cls_idx]
        labels[lbl] = labels.get(lbl, 0) + 1
    tee(f"  Label distribution: {labels}")

    # ---- Step 2: Convert to records ----
    tee("Step 2: Converting to PacketRecords...")
    records = records_from_rows(headers, col, rows)
    tee(f"  {len(records)} valid records")

    # ---- Step 3: Build windows ----
    tee(f"Step 3: Building windows (L={WINDOW_LENGTH}, stride={STRIDE})...")
    schema = FeatureSchema.default_modbus()
    schema.window_length = WINDOW_LENGTH
    X_w, Y_w, stats = build_training_data(records, schema, window_length=WINDOW_LENGTH, stride=STRIDE)
    tee(f"  {X_w.shape[0]} windows, d_c={schema.d_c}, d_d={schema.d_d}")

    # ---- Step 4: Adapt schema ----
    schema = schema.adapt_to_data(X_w)
    active = [s.name for s in schema.continuous if s.var_type.name == "TYPE4"]
    stub_names = [s.name for s in schema.continuous if s.var_type.name == "TYPE6"]
    tee(f"  Active ({len(active)}): {active}")
    tee(f"  Stub ({len(stub_names)}): {stub_names}")
    tee(f"  d_c_active={len(active)}, d_c_all={schema.d_c}")

    # ---- Step 5: Normalize (X_w already z-scored by build_training_data) ----
    tee("Step 4: Normalizing...")
    train_x = torch.from_numpy(X_w).float()
    train_y = [torch.from_numpy(y).long() for y in Y_w]

    # Normalizer stats come from build_training_data (raw feature scale).
    # X_w is already standardized — do NOT re-fit on it (would give mean≈0/std≈1).
    normalizer = Normalizer(schema.d_c)
    normalizer.mean = torch.tensor(stats["mean"], dtype=torch.float32)
    normalizer.std = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-8)
    normalizer.log_features = stats.get("log_features", [])
    normalizer.log_bounds = {int(k): tuple(v) for k, v in stats.get("log_bounds", {}).items()}
    train_x_norm = train_x  # already z-scored
    tee(f"  X: {train_x_norm.shape}, Y: [{', '.join(f'{y.shape}' for y in train_y)}]")
    tee(f"  log_features: {normalizer.log_features}")

    # ---- Step 6: Train ----
    config = DiffusionConfig()
    config.seed = SEED
    # Scale config for 1M data
    config.trend.epochs = 150
    config.trend.batch_size = 256
    config.ddpm.epochs = 150
    config.ddpm.batch_size = 256
    config.mask.epochs = 0
    config.mask.batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tee(f"Step 5: Training on {device}...")
    tee(f"  Config: trend_ep={config.trend.epochs}, ddpm_ep={config.ddpm.epochs}, mask_ep={config.mask.epochs}, bs={config.trend.batch_size}")

    trainer = MaskDDPMTrainer(config, schema, device=device)
    tee(f"  Split: train={train_x_norm.shape[0]} (no val/test split)")
    tee(f"  Config: batch={config.trend.batch_size} ep={config.trend.epochs}+{config.ddpm.epochs}+{config.mask.epochs} d_c_active={trainer.d_c_active} device={device}")
    tee("--- STAGE 1: Trend ---")
    trend_history = trainer.train_trend(train_x_norm.to(device), log_fn=tee)

    tee("--- STAGE 2: Diffusion ---")
    diff_history = trainer.train_diffusion(train_x_norm.to(device), train_y, log_fn=tee)

    # ---- Step 7: Save ----
    out = Path(OUT_DIR)
    trainer.save(str(out))
    normalizer.save(str(out / "normalizer.json"))

    # Save empirical distributions for Type6 stub features + inter_arrival_ns
    # (generation-time replacement; inter_arrival is degenerate — see backfill_v30).
    # Fit on RAW packet features (exact integers) — denormalized windows carry
    # float noise (e.g. 2.29e-09 vs exact 0) that blows up KS against real data.
    from extractor.feature_builder import packet_to_features
    raw_flat, _ = packet_to_features(records, schema)
    raw_flat[:, 3] = np.expm1(raw_flat[:, 3])  # col 3 is log1p'd — store raw ns
    stub_idx = [i for i, s in enumerate(schema.continuous) if s.var_type.name == "TYPE6"]
    if 3 not in stub_idx:
        stub_idx = sorted(stub_idx + [3])
    if stub_idx:
        from diffusion.sampling.sampler import StubSampler
        stub_s = StubSampler()
        stub_s.fit(raw_flat, stub_idx)
        # Exclude session-gap artifacts (>1s) — reproducing them breaks request-
        # response pairing. Must match backfill_v30.py.
        ia_vals = stub_s._distributions[3]
        stub_s._distributions[3] = ia_vals[ia_vals <= 1_000_000_000]
        stub_s.save(str(out / "stub_distributions.npz"))
        tee(f"  Saved stub distributions for {stub_idx}")

    elapsed = time.time() - t0
    tee(f"\nTraining complete in {elapsed/60:.1f} min. Output saved to {out}/")
    tee(f"d_c_active={trainer.d_c_active}, active_indices={trainer.active_indices}")
    log_file.close()


if __name__ == "__main__":
    main()
