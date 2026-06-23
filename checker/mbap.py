"""Modbus/TCP MBAP header parsing and ADU decoding."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MbapHeader:
    transaction_id: int
    protocol_id: int
    length: int
    unit_id: int


@dataclass
class DecodedModbus:
    transaction_id: int
    protocol_id: int
    length: int
    unit_id: int
    function_code: int
    is_exception: bool
    exception_code: Optional[int]
    pdu_data: bytes
    parsed_fields: Optional[dict] = None


def parse_mbap(data: bytes) -> Tuple[MbapHeader, bytes]:
    """Parse MBAP header (7 bytes) and return header + remaining PDU bytes.

    MBAP structure (7 bytes):
      Transaction ID: 2 bytes
      Protocol ID:    2 bytes (0x0000 for Modbus)
      Length:         2 bytes (number of following bytes: unit_id + PDU)
      Unit ID:        1 byte

    Returns (MbapHeader, remaining_bytes) where remaining_bytes is the PDU
    (function code + PDU data), starting immediately after the MBAP header.
    """
    if len(data) < 7:
        raise ValueError(
            f"Data too short for MBAP header: {len(data)} bytes (need >= 7)"
        )

    transaction_id = int.from_bytes(data[0:2], byteorder="big")
    protocol_id = int.from_bytes(data[2:4], byteorder="big")
    length = int.from_bytes(data[4:6], byteorder="big")
    unit_id = data[6]

    header = MbapHeader(
        transaction_id=transaction_id,
        protocol_id=protocol_id,
        length=length,
        unit_id=unit_id,
    )

    remaining = data[7:]
    return header, remaining


def parse_modbus_adu(data: bytes) -> DecodedModbus:
    """Parse a full Modbus/TCP ADU (MBAP + PDU) from raw TCP payload bytes.

    Returns a DecodedModbus with all parsed fields.
    Raises ValueError if the data is too short.
    """
    mbap, pdu = parse_mbap(data)

    if len(pdu) < 1:
        raise ValueError("PDU too short: no function code")

    function_code = pdu[0]
    pdu_data = pdu[1:]

    is_exception = (function_code & 0x80) != 0
    exception_code = None
    if is_exception and len(pdu_data) >= 1:
        exception_code = pdu_data[0]

    return DecodedModbus(
        transaction_id=mbap.transaction_id,
        protocol_id=mbap.protocol_id,
        length=mbap.length,
        unit_id=mbap.unit_id,
        function_code=function_code,
        is_exception=is_exception,
        exception_code=exception_code,
        pdu_data=pdu_data,
    )
