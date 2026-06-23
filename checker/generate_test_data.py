"""Generate synthetic Modbus/TCP PCAP + JSONL for checker testing."""
import json
import struct
import sys
from pathlib import Path

# Add project root so we can import scapy
sys.path.insert(0, str(Path(__file__).parent.parent))

from scapy.all import (
    Ether, IP, TCP, Raw,
    wrpcap,
)


def build_modbus_request(
    transaction_id: int,
    unit_id: int,
    function_code: int,
    pdu_data: bytes,
) -> bytes:
    """Build a Modbus/TCP request ADU."""
    pdu = bytes([function_code]) + pdu_data
    length = 1 + len(pdu)  # unit_id + PDU
    mbap = struct.pack(">HHHB", transaction_id, 0, length, unit_id)
    return mbap + pdu


def build_read_holding_registers_request(
    transaction_id: int,
    unit_id: int,
    start_addr: int,
    quantity: int,
) -> bytes:
    pdu_data = struct.pack(">HH", start_addr, quantity)
    return build_modbus_request(transaction_id, unit_id, 3, pdu_data)


def build_read_holding_registers_response(
    transaction_id: int,
    unit_id: int,
    register_values: bytes,
) -> bytes:
    byte_count = len(register_values)
    pdu_data = bytes([byte_count]) + register_values
    return build_modbus_request(transaction_id, unit_id, 3, pdu_data)


def build_write_single_register_request(
    transaction_id: int,
    unit_id: int,
    reg_addr: int,
    reg_value: int,
) -> bytes:
    pdu_data = struct.pack(">HH", reg_addr, reg_value)
    return build_modbus_request(transaction_id, unit_id, 6, pdu_data)


def build_write_single_register_response(
    transaction_id: int,
    unit_id: int,
    reg_addr: int,
    reg_value: int,
) -> bytes:
    pdu_data = struct.pack(">HH", reg_addr, reg_value)
    return build_modbus_request(transaction_id, unit_id, 6, pdu_data)


def build_exception_response(
    transaction_id: int,
    unit_id: int,
    function_code: int,
    exception_code: int,
) -> bytes:
    fc = function_code | 0x80
    pdu = bytes([fc, exception_code])
    length = 1 + len(pdu)  # unit_id + PDU
    mbap = struct.pack(">HHHB", transaction_id, 0, length, unit_id)
    return mbap + pdu


def make_tcp_packet(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    payload: bytes,
    seq: int = 1000,
    ack: int = 0,
) -> Ether:
    """Build an Ethernet / IP / TCP / Raw packet."""
    return (
        Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb")
        / IP(src=src_ip, dst=dst_ip)
        / TCP(sport=src_port, dport=dst_port, seq=seq, ack=ack, flags="PA")
        / Raw(load=payload)
    )


def main():
    packets = []
    meta_lines = []
    trace_id = "test-trace-001"

    client_ip = "10.0.0.10"
    server_ip = "10.0.0.20"
    client_port = 51000
    server_port = 502

    ts_base = 1736451234567890123
    seq = 1000

    # --- Packet 0: Read Holding Registers request (regs 0–9) ---
    txid = 1
    payload = build_read_holding_registers_request(txid, unit_id=1, start_addr=0, quantity=10)
    pkt = make_tcp_packet(client_ip, server_ip, client_port, server_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 0,
        "pcap_index": 0,
        "ts_ns": ts_base,
        "direction": "c2s",
        "flow": {"src_ip": client_ip, "src_port": client_port, "dst_ip": server_ip, "dst_port": server_port},
        "expected": {
            "modbus": {"transaction_id": txid, "unit_id": 1, "function_code": 3},
            "fields": {"starting_address": 0, "quantity_of_registers": 10},
        },
    }))
    seq += len(payload)

    # --- Packet 1: Read Holding Registers response ---
    reg_values = bytes(range(20))  # 10 registers × 2 bytes
    payload = build_read_holding_registers_response(txid, unit_id=1, register_values=reg_values)
    pkt = make_tcp_packet(server_ip, client_ip, server_port, client_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 1,
        "pcap_index": 1,
        "ts_ns": ts_base + 500_000,
        "direction": "s2c",
        "flow": {"src_ip": server_ip, "src_port": server_port, "dst_ip": client_ip, "dst_port": client_port},
        "expected": {
            "modbus": {"transaction_id": txid, "unit_id": 1, "function_code": 3},
            "fields": {"byte_count": 20},
        },
    }))
    seq += len(payload)

    # --- Packet 2: Write Single Register request ---
    txid = 2
    payload = build_write_single_register_request(txid, unit_id=1, reg_addr=100, reg_value=42)
    pkt = make_tcp_packet(client_ip, server_ip, client_port, server_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 2,
        "pcap_index": 2,
        "ts_ns": ts_base + 1_000_000,
        "direction": "c2s",
        "flow": {"src_ip": client_ip, "src_port": client_port, "dst_ip": server_ip, "dst_port": server_port},
        "expected": {
            "modbus": {"transaction_id": txid, "unit_id": 1, "function_code": 6},
            "fields": {"register_address": 100, "register_value": 42},
        },
    }))
    seq += len(payload)

    # --- Packet 3: Write Single Register response ---
    payload = build_write_single_register_response(txid, unit_id=1, reg_addr=100, reg_value=42)
    pkt = make_tcp_packet(server_ip, client_ip, server_port, client_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 3,
        "pcap_index": 3,
        "ts_ns": ts_base + 1_500_000,
        "direction": "s2c",
        "flow": {"src_ip": server_ip, "src_port": server_port, "dst_ip": client_ip, "dst_port": client_port},
        "expected": {
            "modbus": {"transaction_id": txid, "unit_id": 1, "function_code": 6},
            "fields": {"register_address": 100, "register_value": 42},
        },
    }))
    seq += len(payload)

    # --- Packet 4: Read Holding Registers request (unmatched) ---
    txid = 3
    payload = build_read_holding_registers_request(txid, unit_id=1, start_addr=50, quantity=5)
    pkt = make_tcp_packet(client_ip, server_ip, client_port, server_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 4,
        "pcap_index": 4,
        "ts_ns": ts_base + 2_000_000,
        "direction": "c2s",
        "flow": {"src_ip": client_ip, "src_port": client_port, "dst_ip": server_ip, "dst_port": server_port},
        "expected": {
            "modbus": {"transaction_id": txid, "unit_id": 1, "function_code": 3},
            "fields": {"starting_address": 50, "quantity_of_registers": 5},
        },
    }))
    seq += len(payload)

    # --- Packet 5: Exception response (illegal function) ---
    txid = 4
    payload = build_exception_response(txid, unit_id=1, function_code=1, exception_code=1)
    pkt = make_tcp_packet(server_ip, client_ip, server_port, client_port, payload, seq=seq)
    packets.append(pkt)
    meta_lines.append(json.dumps({
        "trace_id": trace_id,
        "event_id": 5,
        "pcap_index": 5,
        "ts_ns": ts_base + 2_500_000,
        "direction": "s2c",
        "flow": {"src_ip": server_ip, "src_port": server_port, "dst_ip": client_ip, "dst_port": client_port},
    }))

    # --- Write output ---
    out_dir = Path(__file__).parent.parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    pcap_path = str(out_dir / "test_trace.pcapng")
    meta_path = str(out_dir / "test_trace.meta.jsonl")

    wrpcap(pcap_path, packets)
    print(f"Wrote {len(packets)} packets to {pcap_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for line in meta_lines:
            f.write(line + "\n")
    print(f"Wrote {len(meta_lines)} metadata lines to {meta_path}")


if __name__ == "__main__":
    main()
