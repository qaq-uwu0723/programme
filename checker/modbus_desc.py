"""Parse Modbus PDU data using function descriptors."""
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, FieldDescriptor, FieldType


def parse_pdu_with_descriptor(
    pdu_data: bytes,
    unit_id: int,
    function_code: int,
    fields: List[FieldDescriptor],
) -> Dict[str, Any]:
    """Parse PDU data bytes (AFTER function code) using field descriptors.

    This is the core reusable entry point — equivalent to the design doc's
    suggested `parse_with_descriptor(pdu_data, unit, function, fields)`.

    Returns a dict with 'unit', 'function', optional 'exception', and 'fields'.
    """
    result: Dict[str, Any] = {
        "unit": unit_id,
        "function": function_code,
        "fields": {},
    }

    is_exception = (function_code & 0x80) != 0
    if is_exception:
        result["exception"] = True
        if len(pdu_data) >= 1:
            result["exception_code"] = pdu_data[0]
        return result

    offset = 0
    parsed_fields: Dict[str, Any] = {}

    for fd in fields:
        if offset >= len(pdu_data):
            break

        value, consumed = _parse_field(pdu_data, offset, fd, parsed_fields)
        parsed_fields[fd.name] = value
        offset += consumed

    result["fields"] = parsed_fields
    return result


def parse_full_pdu(
    pdu: bytes,
    unit_id: int,
    config: Config,
    direction: str,
) -> Dict[str, Any]:
    """Parse a full Modbus PDU (function_code + data) using config descriptors.

    This is a convenience wrapper that extracts the function code from the PDU
    and delegates to parse_pdu_with_descriptor.
    """
    if len(pdu) < 1:
        return {
            "unit": unit_id,
            "function": 0,
            "fields": {},
            "_parse_error": "PDU too short (no function code)",
        }

    function_code = pdu[0]
    data = pdu[1:]

    fields = config.get_fields(function_code & 0x7F, direction)
    result = parse_pdu_with_descriptor(data, unit_id, function_code, fields)
    return result


def _parse_field(
    data: bytes,
    offset: int,
    fd: FieldDescriptor,
    parsed: Dict[str, Any],
) -> Tuple[Any, int]:
    """Parse a single field from data at offset.

    Returns (value, bytes_consumed).
    """
    ft = fd.field_type

    if ft == FieldType.BYTES:
        length = fd.length
        if length is None and fd.length_from:
            length = parsed.get(fd.length_from, 0)
            if isinstance(length, dict):
                length = 0
        if length is None:
            length = len(data) - offset
        end = min(offset + length, len(data))
        return list(data[offset:end]), end - offset

    if ft == FieldType.BITS:
        length = fd.length or 1
        end = min(offset + length, len(data))
        return list(data[offset:end]), end - offset

    size = ft.byte_size()
    if size is None:
        size = fd.length or 0

    if size == 0:
        return None, 0

    end = min(offset + size, len(data))
    raw = data[offset:end]

    if len(raw) < size:
        return None, len(raw)

    value = int.from_bytes(raw, byteorder="big", signed=ft.value.startswith("i"))

    if fd.scale:
        value = value * fd.scale

    if fd.enum_map and str(value) in fd.enum_map:
        value = fd.enum_map[str(value)]

    return value, size
