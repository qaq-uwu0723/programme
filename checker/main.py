"""CLI entry point for the Modbus/TCP traffic checker."""
import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .validate import validate
from .report import Severity


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Modbus/TCP Traffic Checker — validate generated Modbus traces",
    )
    parser.add_argument(
        "pcap",
        help="Path to PCAP/PCAPNG file containing generated packets",
    )
    parser.add_argument(
        "meta",
        help="Path to JSONL sidecar metadata file",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to Modbus descriptor config JSON (default: built-in descriptors)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["mvp", "strict"],
        default="mvp",
        help="Validation mode (default: mvp)",
    )
    parser.add_argument(
        "--output", "-o",
        default="report.json",
        help="Output JSON report path (default: report.json)",
    )
    parser.add_argument(
        "--text-report",
        action="store_true",
        help="Also emit a human-readable text report alongside the JSON report",
    )
    parser.add_argument(
        "--fail-on",
        choices=["fatal", "error", "warn"],
        default="error",
        help="Exit with non-zero when findings at this severity or higher exist (default: error)",
    )

    args = parser.parse_args(argv)

    # --- Load descriptor config ---
    if args.config:
        config = Config.load(args.config)
    else:
        builtin = Path(__file__).parent / "configs" / "modbus_default.json"
        if builtin.exists():
            config = Config.load(str(builtin))
        else:
            config = Config()

    # --- Run validation ---
    print(f"Checking {args.pcap} …")
    report = validate(args.pcap, args.meta, config, mode=args.mode)

    # --- Write JSON report ---
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Report written to {output_path.resolve()}")

    # --- Summary ---
    s = report.summary
    print(f"\nSummary:")
    print(f"  Total packets scanned : {s.total_packets}")
    print(f"  Total findings        : {s.total_findings}")
    for sev in Severity:
        count = s.by_severity.get(sev.value, 0)
        if count:
            print(f"  {sev.value:<5}              : {count}")

    # --- Text report ---
    if args.text_report:
        text_path = output_path.with_suffix(".txt")
        _write_text_report(report, text_path)
        print(f"\nText report written to {text_path.resolve()}")

    # --- Exit code ---
    fail_map = {
        "fatal": {Severity.FATAL},
        "error": {Severity.FATAL, Severity.ERROR},
        "warn":  {Severity.FATAL, Severity.ERROR, Severity.WARN},
    }
    for sev in fail_map[args.fail_on]:
        if s.by_severity.get(sev.value, 0) > 0:
            sys.exit(1)

    sys.exit(0)


def _write_text_report(report, path: Path) -> None:
    """Write a human-readable text report."""
    lines = [
        "=" * 60,
        "Modbus/TCP Checker — Report",
        "=" * 60,
    ]
    s = report.summary
    lines.append(f"\nTotal packets : {s.total_packets}")
    lines.append(f"Total findings: {s.total_findings}")
    for sev in Severity:
        count = s.by_severity.get(sev.value, 0)
        lines.append(f"  {sev.value:<5} : {count}")

    if report.findings:
        lines.append("\n" + "-" * 60)
        lines.append("Findings")
        lines.append("-" * 60)
        for f in report.findings:
            lines.append(
                f"\n[{f.severity.value}] #{f.pcap_index}  {f.code}"
            )
            lines.append(f"  {f.message}")
            if f.observed:
                lines.append(f"  Observed: {json.dumps(f.observed)}")
            if f.expected:
                lines.append(f"  Expected: {json.dumps(f.expected)}")
    else:
        lines.append("\nNo findings — traffic looks clean.")

    path.write_text("\n".join(lines), encoding="utf-8")
