"""Packet decoding: extract TCP payload from raw packets via scapy."""
from dataclasses import dataclass
from typing import Optional

from scapy.all import IP, IPv6, TCP, Raw
from scapy.packet import Packet


@dataclass
class DecodedPacket:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    ip_version: int
    tcp_payload: bytes


def decode_packet(pkt: Packet) -> Optional[DecodedPacket]:
    """Decode a scapy Packet and extract IP/TCP header fields + TCP payload.

    Returns None if the packet has no IP or TCP layer.
    """
    if IP in pkt:
        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        ip_version = 4
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        src_ip = ip.src
        dst_ip = ip.dst
        ip_version = 6
    else:
        return None

    if TCP not in pkt:
        return None

    tcp = pkt[TCP]
    src_port = tcp.sport
    dst_port = tcp.dport

    payload = bytes(tcp.payload) if Raw in tcp else b""

    return DecodedPacket(
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        ip_version=ip_version,
        tcp_payload=payload,
    )
