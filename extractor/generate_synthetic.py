"""Synthetic Modbus/TCP training data generator.

Generates realistic Modbus traffic as PCAP + JSONL, usable for both
checker validation testing and diffusion model training.
"""
import json
import os
import random
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


# ---------------------------------------------------------------------------
# Modbus ADU builders (minimal, no dependency on assembler)
# ---------------------------------------------------------------------------

def _build_mbap(txid: int, unit_id: int, pdu: bytes) -> bytes:
    length = 1 + len(pdu)
    return struct.pack(">HHHB", txid, 0, length, unit_id) + pdu


def _build_read_req(txid: int, unit_id: int, start_addr: int, quantity: int) -> bytes:
    pdu = struct.pack(">BHH", 3, start_addr, quantity)
    return _build_mbap(txid, unit_id, pdu)


def _build_read_resp(txid: int, unit_id: int, byte_count: int, values: bytes) -> bytes:
    pdu = bytes([3, byte_count]) + values
    return _build_mbap(txid, unit_id, pdu)


def _build_write_single_req(txid: int, unit_id: int, addr: int, val: int) -> bytes:
    pdu = struct.pack(">BHH", 6, addr, val)
    return _build_mbap(txid, unit_id, pdu)


def _build_write_single_resp(txid: int, unit_id: int, addr: int, val: int) -> bytes:
    pdu = struct.pack(">BHH", 6, addr, val)
    return _build_mbap(txid, unit_id, pdu)


def _build_write_multi_req(txid: int, unit_id: int, addr: int, qty: int, vals: bytes) -> bytes:
    bc = len(vals)
    pdu = struct.pack(">BHHB", 16, addr, qty, bc) + vals
    return _build_mbap(txid, unit_id, pdu)


def _build_write_multi_resp(txid: int, unit_id: int, addr: int, qty: int) -> bytes:
    pdu = struct.pack(">BHH", 16, addr, qty)
    return _build_mbap(txid, unit_id, pdu)


# ---------------------------------------------------------------------------
# Process simulation: generates realistic register value trajectories
# ---------------------------------------------------------------------------

class ProcessSimulator:
    """Simulates a simple industrial process with physical inertia.

    Three registers:
      - Reg 0: temperature (slow sine + noise, 20-80°C)
      - Reg 1: pressure (correlated with temp, 1-10 bar)
      - Reg 2: flow rate (independent, 0-100 L/min)
    """

    def __init__(self):
        self.temp = 45.0   # °C
        self.press = 4.5   # bar
        self.flow = 50.0   # L/min

    def step(self) -> Tuple[float, float, float]:
        # Temperature: slow random walk + sine oscillation
        self.temp += random.gauss(0, 0.3)
        self.temp += 0.1 * 0.02  # slight drift
        self.temp = max(20, min(80, self.temp))

        # Pressure: correlated with temp + own noise
        target_press = 1.0 + (self.temp - 20) * 0.12
        self.press += 0.3 * (target_press - self.press) + random.gauss(0, 0.05)
        self.press = max(1.0, min(10.0, self.press))

        # Flow: independent random walk
        self.flow += random.gauss(0, 1.0)
        self.flow = max(0, min(100, self.flow))

        return self.temp, self.press, self.flow

    def register_values(self) -> Tuple[int, int, int]:
        """Return current register values as 16-bit integers."""
        t = int(self.temp * 256)      # scale: 0.0039°C/LSB
        p = int(self.press * 6553)    # scale: 0.00015 bar/LSB
        f = int(self.flow * 655)      # scale: 0.0015 L/min/LSB
        return t, p, f


# ---------------------------------------------------------------------------
# Traffic generation
# ---------------------------------------------------------------------------

def generate_traffic(
    num_packets: int = 2000,
    output_pcap: str = "synthetic_modbus.pcapng",
    output_meta: str = "synthetic_modbus.meta.jsonl",
    client_ip: str = "10.0.0.10",
    server_ip: str = "10.0.0.20",
    client_port: int = 51000,
    server_port: int = 502,
    trace_id: str = "synthetic-001",
    seed: int = 42,
) -> None:
    """Generate synthetic Modbus/TCP traffic with realistic register dynamics.

    Args:
        num_packets: total number of Modbus packets to generate
        output_pcap: output PCAPNG file path
        output_meta: output JSONL sidecar file path
        client_ip/server_ip: IP addresses for client and server
        client_port/server_port: ports (server typically 502)
        trace_id: identifier for the trace
        seed: random seed for reproducibility
    """
    if not HAS_SCAPY:
        raise ImportError("scapy is required")

    random.seed(seed)
    process = ProcessSimulator()
    packets = []
    next_seq_client = 1000
    next_seq_server = 2000
    ack_client = 2000
    ack_server = 1000

    # Function code distribution (read-heavy, as typical ICS)
    fc_weights = {3: 60, 6: 25, 16: 15}  # FC3/6/16 percentages
    fc_choices = []
    for fc, w in fc_weights.items():
        fc_choices.extend([fc] * w)

    # Address pools
    addr_pool = [0, 10, 20, 50, 100]

    ts_ns = 1_736_451_234_567_890_123
    inter_arrival_us = 50_000  # 50ms polling interval

    with open(output_meta, "w", encoding="utf-8") as meta_fp:
        i = 0
        while i < num_packets:
            # Decide function code
            if i + 1 >= num_packets:
                break
            fc = random.choice(fc_choices)
            unit_id = random.randint(1, 5)
            txid = i // 2 + 1
            addr = random.choice(addr_pool)
            qty = random.choice([1, 2, 3, 5, 10])

            # Simulate process step
            t_val, p_val, f_val = process.register_values()

            # --- Request ---
            if fc == 3:
                payload = _build_read_req(txid, unit_id, addr, qty)
            elif fc == 6:
                addr = random.choice(addr_pool[:3])
                payload = _build_write_single_req(txid, unit_id, addr, t_val)
            elif fc == 16:
                qty = min(3, qty)
                vals = struct.pack(">HHH", t_val, p_val, f_val)[:qty * 2]
                payload = _build_write_multi_req(txid, unit_id, addr, qty, vals)

            # Client → Server packet
            pkt_req = (
                Ether()
                / IP(src=client_ip, dst=server_ip)
                / TCP(sport=client_port, dport=server_port, flags="PA",
                      seq=next_seq_client, ack=ack_client)
                / Raw(load=payload)
            )
            packets.append(pkt_req)
            write_meta_line(meta_fp, trace_id, i, i, ts_ns, "c2s",
                          client_ip, client_port, server_ip, server_port,
                          txid, unit_id, fc, addr, qty)
            i += 1
            next_seq_client += len(payload)

            # Small time gap
            ts_ns += inter_arrival_us * 1000
            inter_arrival_us += random.randint(-5000, 5000)
            inter_arrival_us = max(5000, min(200000, inter_arrival_us))

            # --- Response ---
            if fc == 3:
                reg_values = process.register_values()
                vals = struct.pack(">HHH", *reg_values)[:qty * 2]
                payload = _build_read_resp(txid, unit_id, qty * 2, vals)
            elif fc == 6:
                payload = _build_write_single_resp(txid, unit_id, addr, t_val)
            elif fc == 16:
                payload = _build_write_multi_resp(txid, unit_id, addr, qty)

            # Server → Client packet
            pkt_resp = (
                Ether()
                / IP(src=server_ip, dst=client_ip)
                / TCP(sport=server_port, dport=client_port, flags="PA",
                      seq=next_seq_server, ack=ack_server)
                / Raw(load=payload)
            )
            packets.append(pkt_resp)
            write_meta_line(meta_fp, trace_id, i, i, ts_ns, "s2c",
                          server_ip, server_port, client_ip, client_port,
                          txid, unit_id, fc, addr, qty)
            i += 1
            next_seq_server += len(payload)
            ack_server += len(pkt_req[Raw].load) if Raw in pkt_req else 0
            ack_client += len(payload)

            ts_ns += random.randint(1000, 50000) * 1000

    wrpcap(output_pcap, packets)
    print(f"Generated {len(packets)} packets → {output_pcap}")
    print(f"Generated {len(packets)//2} request-response pairs")


def write_meta_line(
    fp,
    trace_id: str, event_id: int, pcap_index: int, ts_ns: int,
    direction: str, src_ip: str, src_port: int,
    dst_ip: str, dst_port: int,
    txid: int, unit_id: int, fc: int, addr: int, qty: int,
) -> None:
    line = {
        "trace_id": trace_id,
        "event_id": event_id,
        "pcap_index": pcap_index,
        "ts_ns": ts_ns,
        "direction": direction,
        "flow": {
            "src_ip": src_ip, "src_port": src_port,
            "dst_ip": dst_ip, "dst_port": dst_port,
        },
    }
    fp.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Modbus/TCP traffic")
    parser.add_argument("--num-packets", type=int, default=2000)
    parser.add_argument("--output-dir", default="data/synthetic/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generate_traffic(
        num_packets=args.num_packets,
        output_pcap=str(out / "trace.pcapng"),
        output_meta=str(out / "trace.meta.jsonl"),
        seed=args.seed,
    )
