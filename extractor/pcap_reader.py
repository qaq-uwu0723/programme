"""PCAP reader — extracts per-packet Modbus feature records from raw PCAP.

Reuses checker's decode/mbap/meta modules for packet parsing.
Outputs a list of dicts, one per Modbus packet, for the feature builder.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import struct

# Reuse checker internals for parsing
import sys
from pathlib import Path
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from checker.decode import decode_packet
from checker.mbap import parse_mbap, parse_modbus_adu


@dataclass
class PacketRecord:
    """Per-packet feature record extracted from a single Modbus/TCP packet."""
    ts_ns: int                    # absolute timestamp in nanoseconds
    inter_arrival_ns: int         # inter-arrival time (ns) from previous packet
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    direction: str                # "c2s" or "s2c"
    # Modbus ADU fields
    transaction_id: int
    protocol_id: int
    unit_id: int
    function_code: int            # raw (may have 0x80 for exceptions)
    is_exception: bool
    exception_code: int
    pdu_data: bytes
    payload_size: int
    # Parsed register-level fields
    register_address: int
    register_values: List[int]    # up to 3 values extracted
    quantity: int


def extract_packets(pcap_path: str) -> List[PacketRecord]:
    """Extract per-packet Modbus feature records from a PCAP file.

    Args:
        pcap_path: path to a PCAP or PCAPNG file containing Modbus/TCP traffic

    Returns:
        list of PacketRecord, one per valid Modbus packet
    """
    from scapy.all import rdpcap

    try:
        packets = rdpcap(pcap_path)
    except Exception:
        from scapy.utils import RawPcapReader
        packets = []
        for pkt, _ in RawPcapReader(pcap_path):
            from scapy.all import Ether
            packets.append(Ether(pkt))
    records: List[PacketRecord] = []
    prev_ts_ns: Optional[int] = None

    for pkt in packets:
        decoded = decode_packet(pkt)
        if decoded is None:
            continue

        # Only process Modbus port traffic
        if decoded.dst_port != 502 and decoded.src_port != 502:
            continue

        payload = decoded.tcp_payload
        if len(payload) < 8:  # MBAP (7) + PDU function code (1)
            continue

        # Parse Modbus ADU
        try:
            adu = parse_modbus_adu(payload)
        except ValueError:
            continue

        # Determine direction: c2s = client→server (dst=502), s2c = server→client (src=502)
        if decoded.dst_port == 502:
            direction = "c2s"
        else:
            direction = "s2c"

        # Timestamp
        ts_ns = int(pkt.time * 1_000_000_000)
        if prev_ts_ns is not None:
            inter_arrival_ns = max(1, ts_ns - prev_ts_ns)
        else:
            inter_arrival_ns = 10_000_000  # 10ms default for first packet
        prev_ts_ns = ts_ns

        # Extract register-level fields from PDU (direction-aware: responses and
        # requests have different PDU layouts)
        reg_addr, reg_vals, quantity = _extract_pdu_fields(
            adu.function_code, adu.pdu_data, adu.is_exception, direction
        )

        records.append(PacketRecord(
            ts_ns=ts_ns,
            inter_arrival_ns=inter_arrival_ns,
            src_ip=decoded.src_ip,
            dst_ip=decoded.dst_ip,
            src_port=decoded.src_port,
            dst_port=decoded.dst_port,
            direction=direction,
            transaction_id=adu.transaction_id,
            protocol_id=adu.protocol_id,
            unit_id=adu.unit_id,
            function_code=adu.function_code,
            is_exception=adu.is_exception,
            exception_code=adu.exception_code or 0,
            pdu_data=adu.pdu_data,
            # payload_size: FARAONIC CSV defines it as IP total length (IP_len),
            # which includes IP (20B) + TCP (20B) headers. Approximate the same
            # definition here for cross-source consistency.
            payload_size=len(payload) + 40,
            register_address=reg_addr,
            register_values=reg_vals,
            quantity=quantity,
        ))

    return records


def _extract_pdu_fields(
    raw_function_code: int,
    pdu_data: bytes,
    is_exception: bool,
    direction: str,
) -> Tuple[int, List[int], int]:
    """Extract register address, values, and quantity from PDU data.

    Handles the most common Modbus function codes (3, 6, 16). Read-type
    requests and responses have different PDU layouts (2B addr+qty vs
    1B byte_count+values), so direction disambiguates them — a read response
    with >=2 registers would otherwise be misparsed as a request.

    Returns:
        (register_address, register_values_list, quantity)
    """
    register_address = 0
    register_values: List[int] = [0, 0, 0]
    quantity = 1

    if is_exception or len(pdu_data) < 2:
        return register_address, register_values, quantity

    base_fc = raw_function_code & 0x7F

    try:
        if base_fc in (1, 2, 3, 4):
            if direction == "c2s":
                # Read request: 2B address + 2B quantity
                if len(pdu_data) >= 4:
                    register_address = int.from_bytes(pdu_data[0:2], "big")
                    quantity = int.from_bytes(pdu_data[2:4], "big")
            else:
                # Read response: 1B byte_count + N×2B values
                if len(pdu_data) >= 1:
                    byte_count = pdu_data[0]
                    quantity = byte_count // 2
                    for j in range(min(3, quantity)):
                        off = 1 + j * 2
                        if off + 2 <= len(pdu_data):
                            register_values[j] = int.from_bytes(pdu_data[off:off+2], "big")

        elif base_fc in (5, 6):
            # Write single: 2B address + 2B value (same layout both directions)
            if len(pdu_data) >= 4:
                register_address = int.from_bytes(pdu_data[0:2], "big")
                register_values[0] = int.from_bytes(pdu_data[2:4], "big")
                quantity = 1

        elif base_fc in (15, 16):
            if direction == "c2s":
                # Write-multiple request: 2B addr + 2B qty + 1B byte_count + values
                if len(pdu_data) >= 5:
                    register_address = int.from_bytes(pdu_data[0:2], "big")
                    quantity = int.from_bytes(pdu_data[2:4], "big")
                    byte_count = pdu_data[4]
                    for j in range(min(3, quantity)):
                        off = 5 + j * 2
                        if off + 2 <= len(pdu_data):
                            register_values[j] = int.from_bytes(pdu_data[off:off+2], "big")
            else:
                # Write-multiple response: 2B addr + 2B qty
                if len(pdu_data) >= 4:
                    register_address = int.from_bytes(pdu_data[0:2], "big")
                    quantity = int.from_bytes(pdu_data[2:4], "big")

    except (IndexError, struct.error):
        pass

    return register_address, register_values, quantity
