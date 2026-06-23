"""JSONL sidecar writer — produces metadata lines for the checker."""
import json
from typing import Dict, List, Optional, Any


def write_meta_line(
    fp,
    trace_id: str,
    event_id: int,
    pcap_index: int,
    ts_ns: int,
    direction: str,
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    expected_modbus: Optional[Dict[str, int]] = None,
    expected_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one JSONL line."""
    line: Dict[str, Any] = {
        "trace_id": trace_id,
        "event_id": event_id,
        "pcap_index": pcap_index,
        "ts_ns": ts_ns,
        "direction": direction,
        "flow": {
            "src_ip": src_ip,
            "src_port": src_port,
            "dst_ip": dst_ip,
            "dst_port": dst_port,
        },
    }
    if expected_modbus is not None or expected_fields is not None:
        line["expected"] = {}
        if expected_modbus is not None:
            line["expected"]["modbus"] = expected_modbus
        if expected_fields is not None:
            line["expected"]["fields"] = expected_fields

    fp.write(json.dumps(line, ensure_ascii=False) + "\n")
