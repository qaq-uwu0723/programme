"""PCAP/PCAPNG file reading and alignment with JSONL metadata."""
from typing import Iterator, List, Optional, Tuple

from scapy.all import sniff
from scapy.packet import Packet

from .meta import PacketMeta, read_all_meta
from .decode import decode_packet, DecodedPacket


class PcapReader:
    """Read PCAP/PCAPNG files and provide indexed access to packets."""

    def __init__(self, pcap_path: str):
        self.pcap_path = pcap_path
        self._packets: Optional[List[Packet]] = None

    def _load(self) -> List[Packet]:
        if self._packets is None:
            self._packets = sniff(offline=self.pcap_path)
        return self._packets

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Tuple[int, Packet]]:
        for i, pkt in enumerate(self._load()):
            yield i, pkt

    def get(self, index: int) -> Optional[Packet]:
        packets = self._load()
        if 0 <= index < len(packets):
            return packets[index]
        return None


def iter_aligned(
    pcap_path: str,
    meta_path: str,
) -> Iterator[Tuple[int, Optional[Packet], Optional[PacketMeta], Optional[DecodedPacket], Optional[str]]]:
    """Iterate over aligned PCAP packets and JSONL metadata.

    Yields (pcap_index, raw_packet, meta, decoded_packet, error).
    Exactly one of (decoded_packet, error) will be set per iteration:
    - If both PCAP and meta are present and decoding succeeds: decoded_packet
      is set, error is None.
    - Otherwise: decoded_packet is None, error describes the problem.
    """
    pcap = PcapReader(pcap_path)
    metas = read_all_meta(meta_path)

    pcap_len = len(pcap)
    meta_len = len(metas)
    max_idx = max(pcap_len, meta_len)

    for i in range(max_idx):
        if i >= pcap_len:
            yield i, None, metas[i], None, "Missing PCAP packet (meta line has no matching packet)"
            continue

        if i >= meta_len:
            yield i, pcap.get(i), None, None, "Missing metadata line (PCAP packet has no matching meta)"
            continue

        pkt = pcap.get(i)
        meta = metas[i]

        decoded = decode_packet(pkt)
        if decoded is None:
            yield i, pkt, meta, None, "Failed to decode packet (no IP/TCP layer found)"
            continue

        yield i, pkt, meta, decoded, None
