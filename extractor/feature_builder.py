"""Feature builder — converts per-packet records to normalized training tensors.

Maps PacketRecord → 7 continuous + 6 discrete features (per FeatureSchema),
then slices into windows of length L and normalizes.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .schema import FeatureSchema, FeatureKind
from .pcap_reader import PacketRecord


# Modbus function code → vocabulary index mapping
FC_TO_IDX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 6, 11: 7, 15: 8, 16: 9, 17: 10, 43: 11,
}
IDX_TO_FC = {v: k for k, v in FC_TO_IDX.items()}
NUM_FC = 12

# Direction → vocabulary index
DIR_TO_IDX = {"c2s": 0, "s2c": 1}


def packet_to_features(
    records: List[PacketRecord],
    schema: Optional[FeatureSchema] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert packet records to raw feature arrays.

    Args:
        records: list of PacketRecord from pcap_reader
        schema: optional FeatureSchema (uses default_modbus if None)

    Returns:
        X_cont: (N, d_c) float32 continuous features (unnormalized)
        Y_disc: (N, d_d) int64 discrete features (vocabulary indices)
    """
    if schema is None:
        schema = FeatureSchema.default_modbus()

    N = len(records)
    d_c = schema.d_c
    d_d = schema.d_d

    X = np.zeros((N, d_c), dtype=np.float32)
    Y = np.zeros((N, d_d), dtype=np.int64)

    for i, rec in enumerate(records):
        # --- Continuous features ---
        # [C0..C2] register values
        X[i, 0] = rec.register_values[0]
        X[i, 1] = rec.register_values[1]
        X[i, 2] = rec.register_values[2]

        # [C3] inter-arrival time (ns) — log1p to compress scale
        X[i, 3] = np.log1p(float(rec.inter_arrival_ns))

        # [C4] payload size (bytes)
        X[i, 4] = float(rec.payload_size)

        # [C5] register address
        X[i, 5] = float(rec.register_address)

        # [C6] quantity
        X[i, 6] = float(max(1, rec.quantity))

        # --- Discrete features ---
        # [D0] function code → vocab index
        base_fc = rec.function_code & 0x7F
        Y[i, 0] = FC_TO_IDX.get(base_fc, 2)  # default: FC3

        # [D1] direction
        Y[i, 1] = DIR_TO_IDX.get(rec.direction, 0)

        # [D2] unit_id
        Y[i, 2] = min(rec.unit_id % 248, 247)

        # [D3] transaction_id (clamped to vocab range; default vocab=256)
        Y[i, 3] = min(rec.transaction_id % 65536, 255)

        # [D4] is_exception
        Y[i, 4] = 1 if rec.is_exception else 0

        # [D5] exception_code
        Y[i, 5] = rec.exception_code % 256

    return X, Y


def build_windows(
    X_cont: np.ndarray,
    Y_disc: np.ndarray,
    window_length: int = 128,
    stride: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Slice flat feature arrays into fixed-length windows.

    Args:
        X_cont: (N, d_c) continuous features
        Y_disc: (N, d_d) discrete features
        window_length: L, number of steps per window
        stride: step between consecutive window starts

    Returns:
        X_windows: (N_windows, L, d_c) float32
        Y_windows: list of (N_windows, L) int64, one per discrete variable
    """
    N = X_cont.shape[0]
    d_c = X_cont.shape[1]
    d_d = Y_disc.shape[1]

    if N < window_length:
        # Pad with repetition if too short
        repeats = (window_length // N) + 1
        X_cont = np.tile(X_cont, (repeats, 1))
        Y_disc = np.tile(Y_disc, (repeats, 1))
        N = X_cont.shape[0]

    num_windows = max(1, (N - window_length) // stride + 1)

    X_w = np.zeros((num_windows, window_length, d_c), dtype=np.float32)
    Y_w = [np.zeros((num_windows, window_length), dtype=np.int64) for _ in range(d_d)]

    for w in range(num_windows):
        start = w * stride
        end = start + window_length
        X_w[w] = X_cont[start:end, :]
        for j in range(d_d):
            Y_w[j][w] = Y_disc[start:end, j]

    return X_w, Y_w


def build_training_data(
    records: List[PacketRecord],
    schema: Optional[FeatureSchema] = None,
    window_length: int = 128,
    stride: int = 1,
    normalizer_stats_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Full feature extraction pipeline: records → windowed tensors.

    Args:
        records: per-packet records from pcap_reader
        schema: feature specification
        window_length: L
        stride: window stride
        normalizer_stats_path: optional path to save normalization stats

    Returns:
        X_train: (N_windows, L, d_c) normalized continuous
        Y_train: list of (N_windows, L) discrete
        stats: dict with mean/std per continuous feature
    """
    if schema is None:
        schema = FeatureSchema.default_modbus()

    # Step 1: records → flat features
    X_flat, Y_flat = packet_to_features(records, schema)

    # Step 2: normalize continuous (z-score)
    mean = X_flat.mean(axis=0, keepdims=True)
    std = X_flat.std(axis=0, keepdims=True).clip(min=1e-8)
    X_norm = (X_flat - mean) / std

    # Step 3: slice into windows
    X_w, Y_w = build_windows(X_norm, Y_flat, window_length, stride)

    stats = {
        "mean": mean.squeeze(0).tolist(),
        "std": std.squeeze(0).tolist(),
        "num_records": len(records),
        "num_windows": X_w.shape[0],
        "window_length": window_length,
        "log_features": [3],  # inter_arrival_ns was log1p-transformed
    }

    if normalizer_stats_path is not None:
        import json
        with open(normalizer_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return X_w, Y_w, stats


def save_training_data(
    X_train: np.ndarray,
    Y_train: List[np.ndarray],
    output_dir: str,
) -> None:
    """Save training tensors to disk in the format expected by the diffusion trainer.

    Output files:
        train_X.npy — (N_windows, L, d_c) float32
        train_Y_0.npy ... train_Y_{d_d-1}.npy — (N_windows, L) int64
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train_X.npy"), X_train)
    for j, y in enumerate(Y_train):
        np.save(os.path.join(output_dir, f"train_Y_{j}.npy"), y)

    print(f"Saved to {output_dir}/")
    print(f"  train_X.npy: {X_train.shape} {X_train.dtype}")
    for j, y in enumerate(Y_train):
        print(f"  train_Y_{j}.npy: {y.shape} {y.dtype}")
