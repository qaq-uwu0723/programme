"""Protocol constraint enforcement for the assembler.

Deterministic rules that the diffusion model does NOT learn — these are
applied at packet assembly time to guarantee protocol validity.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModbusADU:
    """Internal representation of a Modbus ADU before serialization."""
    transaction_id: int
    protocol_id: int = 0         # always 0 for Modbus
    unit_id: int = 1
    function_code: int = 3       # default: read holding registers
    is_exception: bool = False
    exception_code: int = 0
    pdu_data: bytes = b""

    @property
    def pdu(self) -> bytes:
        """Assemble the PDU (function code + data or exception)."""
        if self.is_exception:
            return bytes([self.function_code | 0x80, self.exception_code])
        return bytes([self.function_code]) + self.pdu_data

    @property
    def mbap(self) -> bytes:
        """Assemble the 7-byte MBAP header."""
        length = 1 + len(self.pdu)  # unit_id + PDU
        return (
            self.transaction_id.to_bytes(2, "big")
            + self.protocol_id.to_bytes(2, "big")
            + length.to_bytes(2, "big")
            + self.unit_id.to_bytes(1, "big")
        )

    @property
    def raw(self) -> bytes:
        """Full Modbus/TCP ADU as bytes."""
        return self.mbap + self.pdu


def build_read_registers_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=3 (read holding registers) request."""
    pdu = (
        start_addr.to_bytes(2, "big")
        + quantity.to_bytes(2, "big")
    )
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=3, pdu_data=pdu,
    )


def build_read_registers_response(
    txid: int, unit_id: int, register_values: bytes,
) -> ModbusADU:
    """Build FC=3 response. register_values contains 2*N bytes."""
    byte_count = len(register_values)
    pdu = bytes([byte_count]) + register_values
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=3, pdu_data=pdu,
    )


def build_write_single_register_request(
    txid: int, unit_id: int, reg_addr: int, reg_value: int,
) -> ModbusADU:
    """Build FC=6 (write single register) request."""
    pdu = reg_addr.to_bytes(2, "big") + reg_value.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=6, pdu_data=pdu,
    )


def build_write_single_register_response(
    txid: int, unit_id: int, reg_addr: int, reg_value: int,
) -> ModbusADU:
    """Build FC=6 response (echoes the request)."""
    pdu = reg_addr.to_bytes(2, "big") + reg_value.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=6, pdu_data=pdu,
    )


def build_write_multiple_registers_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
    register_values: bytes,
) -> ModbusADU:
    """Build FC=16 (write multiple registers) request."""
    byte_count = len(register_values)
    pdu = (
        start_addr.to_bytes(2, "big")
        + quantity.to_bytes(2, "big")
        + bytes([byte_count])
        + register_values
    )
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=16, pdu_data=pdu,
    )


def build_write_multiple_registers_response(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=16 response."""
    pdu = start_addr.to_bytes(2, "big") + quantity.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=16, pdu_data=pdu,
    )


def build_exception_response(
    txid: int, unit_id: int, function_code: int, exception_code: int,
) -> ModbusADU:
    """Build an exception response."""
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id,
        function_code=function_code, is_exception=True,
        exception_code=exception_code,
    )


# Mapping from function code to builder functions
FUNC_BUILDERS = {
    # Request builders
    3: ("request", build_read_registers_request),
    6: ("request", build_write_single_register_request),
    16: ("request", build_write_multiple_registers_request),
    # Response builders
    "response_3": build_read_registers_response,
    "response_6": build_write_single_register_response,
    "response_16": build_write_multiple_registers_response,
}
