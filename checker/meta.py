"""JSONL sidecar metadata reader."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
import json


@dataclass
class FlowKey:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    ip_version: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
        }


@dataclass
class ExpectedModbus:
    transaction_id: Optional[int] = None
    unit_id: Optional[int] = None
    function_code: Optional[int] = None


@dataclass
class ExpectedFields:
    modbus: Optional[ExpectedModbus] = None
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PacketMeta:
    trace_id: str
    event_id: int
    pcap_index: int
    ts_ns: int
    direction: str  # "c2s" or "s2c"
    flow: FlowKey
    expected: Optional[ExpectedFields] = None

    @staticmethod
    def from_dict(data: Dict[str, Any], line_index: int) -> "PacketMeta":
        flow_data = data.get("flow", {})
        flow = FlowKey(
            src_ip=flow_data.get("src_ip", ""),
            src_port=flow_data.get("src_port", 0),
            dst_ip=flow_data.get("dst_ip", ""),
            dst_port=flow_data.get("dst_port", 0),
        )

        expected = None
        if "expected" in data:
            exp = data["expected"]
            exp_modbus = None
            if "modbus" in exp:
                mb = exp["modbus"]
                exp_modbus = ExpectedModbus(
                    transaction_id=mb.get("transaction_id"),
                    unit_id=mb.get("unit_id"),
                    function_code=mb.get("function_code"),
                )
            expected = ExpectedFields(
                modbus=exp_modbus,
                fields=exp.get("fields", {}),
            )

        return PacketMeta(
            trace_id=data.get("trace_id", ""),
            event_id=data.get("event_id", line_index),
            pcap_index=data.get("pcap_index", line_index),
            ts_ns=data.get("ts_ns", 0),
            direction=data.get("direction", "c2s"),
            flow=flow,
            expected=expected,
        )


def read_meta(path: str) -> Iterator[PacketMeta]:
    """Read JSONL metadata file line by line, yielding PacketMeta objects."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield PacketMeta.from_dict(data, i)


def read_all_meta(path: str) -> List[PacketMeta]:
    """Read all metadata lines into a list."""
    return list(read_meta(path))
