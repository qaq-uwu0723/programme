"""CLI entry point for the assembler.

Usage:
    python -m assembler --data generated/ --output traces/
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from .packet_builder import PacketAssembler
from extractor.schema import FeatureSchema


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Modbus Packet Assembler — converts diffusion output to PCAP + JSONL"
    )
    parser.add_argument("--data", required=True, help="Directory containing gen_X.npy and gen_Y_*.npy")
    parser.add_argument("--output", default="traces/", help="Output directory for PCAP + JSONL")
    parser.add_argument("--client-ip", default="10.0.0.10")
    parser.add_argument("--server-ip", default="10.0.0.20")
    parser.add_argument("--client-port", type=int, default=51000)
    parser.add_argument("--server-port", type=int, default=502)
    parser.add_argument("--trace-id", default="generated-trace-001")

    args = parser.parse_args(argv)

    schema = FeatureSchema.default_modbus()
    data_dir = Path(args.data)

    # Load generated tensors
    X_hat = torch.from_numpy(np.load(data_dir / "gen_X.npy"))
    Y_hat = [torch.from_numpy(np.load(data_dir / f"gen_Y_{j}.npy")) for j in range(schema.d_d)]

    print(f"Loaded X: {X_hat.shape}, Y: {[y.shape for y in Y_hat]}")

    # Assemble
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    assembler = PacketAssembler(
        schema,
        client_ip=args.client_ip,
        server_ip=args.server_ip,
        client_port=args.client_port,
        server_port=args.server_port,
    )
    assembler.assemble(
        X_hat, Y_hat,
        output_pcap=str(out_dir / f"{args.trace_id}.pcapng"),
        output_meta=str(out_dir / f"{args.trace_id}.meta.jsonl"),
        trace_id=args.trace_id,
    )


if __name__ == "__main__":
    main()
