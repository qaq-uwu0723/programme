"""CLI entry point for the feature extraction pipeline.

Usage:
    # From existing synthetic Modbus PCAP
    python -m extractor --pcap data/synthetic/trace.pcapng --output data/processed/

    # Generate synthetic training data + extract features in one step
    python -m extractor --generate --num-packets 2000 --output data/processed/
"""
import argparse
import sys
from pathlib import Path

from .schema import FeatureSchema
from .pcap_reader import extract_packets
from .feature_builder import build_training_data, save_training_data


def cmd_extract(args) -> None:
    """Extract features from existing PCAP → training tensors."""
    print(f"Reading {args.pcap} …")
    records = extract_packets(args.pcap)
    if not records:
        print("Error: no Modbus packets found in PCAP", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(records)} Modbus packets")

    schema = FeatureSchema.default_modbus()
    stats_path = str(Path(args.output) / "normalizer.json")
    X_train, Y_train, stats = build_training_data(
        records, schema,
        window_length=args.window_length,
        stride=args.stride,
        normalizer_stats_path=stats_path,
    )

    save_training_data(X_train, Y_train, args.output)
    print(f"Stats: mean={stats['mean'][:3]}...  std={stats['std'][:3]}...")
    print(f"Records: {stats['num_records']} → Windows: {stats['num_windows']}")


def cmd_generate_and_extract(args) -> None:
    """Generate synthetic Modbus PCAP + extract features in one step."""
    from .generate_synthetic import generate_traffic

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pcap_path = str(out / "trace.pcapng")
    meta_path = str(out / "trace.meta.jsonl")

    print(f"Generating {args.num_packets} synthetic Modbus packets …")
    generate_traffic(
        num_packets=args.num_packets,
        output_pcap=pcap_path,
        output_meta=meta_path,
        seed=args.seed,
    )

    # Now extract from the generated PCAP
    print(f"\nExtracting features …")
    records = extract_packets(pcap_path)
    print(f"Extracted {len(records)} packets")

    schema = FeatureSchema.default_modbus()
    stats_path = str(out / "normalizer.json")
    X_train, Y_train, stats = build_training_data(
        records, schema,
        window_length=args.window_length,
        stride=args.stride,
        normalizer_stats_path=stats_path,
    )

    save_training_data(X_train, Y_train, args.output)
    print(f"Done. {stats['num_windows']} windows of length {args.window_length}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Modbus Feature Extractor — PCAP → training tensors"
    )
    parser.add_argument("--pcap", help="Path to input PCAP/PCAPNG file")
    parser.add_argument("--generate", action="store_true",
                       help="Generate synthetic traffic instead of reading PCAP")
    parser.add_argument("--num-packets", type=int, default=2000,
                       help="Number of packets if generating (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/",
                       help="Output directory for training tensors")
    parser.add_argument("--window-length", type=int, default=128,
                       help="Window length L (default: 128)")
    parser.add_argument("--stride", type=int, default=1,
                       help="Window stride (default: 1)")

    args = parser.parse_args(argv)

    if args.generate:
        cmd_generate_and_extract(args)
    elif args.pcap:
        cmd_extract(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
