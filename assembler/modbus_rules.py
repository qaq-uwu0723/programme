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


def build_read_coils_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=1 (read coils) request."""
    pdu = start_addr.to_bytes(2, "big") + quantity.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=1, pdu_data=pdu,
    )


def build_read_coils_response(
    txid: int, unit_id: int, coil_status: bytes,
) -> ModbusADU:
    """Build FC=1 response. coil_status contains ceil(quantity/8) bytes."""
    pdu = bytes([len(coil_status)]) + coil_status
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=1, pdu_data=pdu,
    )


def build_read_discrete_inputs_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=2 (read discrete inputs) request."""
    pdu = start_addr.to_bytes(2, "big") + quantity.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=2, pdu_data=pdu,
    )


def build_read_discrete_inputs_response(
    txid: int, unit_id: int, input_status: bytes,
) -> ModbusADU:
    """Build FC=2 response. input_status contains ceil(quantity/8) bytes."""
    pdu = bytes([len(input_status)]) + input_status
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=2, pdu_data=pdu,
    )


def build_read_input_registers_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=4 (read input registers) request."""
    pdu = start_addr.to_bytes(2, "big") + quantity.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=4, pdu_data=pdu,
    )


def build_read_input_registers_response(
    txid: int, unit_id: int, register_values: bytes,
) -> ModbusADU:
    """Build FC=4 response. register_values contains 2*N bytes."""
    pdu = bytes([len(register_values)]) + register_values
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=4, pdu_data=pdu,
    )


def build_write_single_coil_request(
    txid: int, unit_id: int, output_addr: int, output_value: int,
) -> ModbusADU:
    """Build FC=5 (write single coil) request. output_value: 0 or 0xFF00."""
    pdu = output_addr.to_bytes(2, "big") + output_value.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=5, pdu_data=pdu,
    )


def build_write_single_coil_response(
    txid: int, unit_id: int, output_addr: int, output_value: int,
) -> ModbusADU:
    """Build FC=5 response (echoes the request)."""
    pdu = output_addr.to_bytes(2, "big") + output_value.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=5, pdu_data=pdu,
    )


def build_diagnostics_request(
    txid: int, unit_id: int, sub_function: int, data: bytes,
) -> ModbusADU:
    """Build FC=8 (diagnostics) request."""
    pdu = sub_function.to_bytes(2, "big") + data
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=8, pdu_data=pdu,
    )


def build_diagnostics_response(
    txid: int, unit_id: int, sub_function: int, data: bytes,
) -> ModbusADU:
    """Build FC=8 response (echoes sub_function + data)."""
    pdu = sub_function.to_bytes(2, "big") + data
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=8, pdu_data=pdu,
    )


def build_event_counter_request(txid: int, unit_id: int) -> ModbusADU:
    """Build FC=11 (get comm event counter) request — no data."""
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=11, pdu_data=b"",
    )


def build_event_counter_response(
    txid: int, unit_id: int, event_count: int = 0,
) -> ModbusADU:
    """Build FC=11 response: status + event_count."""
    pdu = (0).to_bytes(2, "big") + (event_count & 0xFFFF).to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=11, pdu_data=pdu,
    )


def build_write_multiple_coils_request(
    txid: int, unit_id: int, start_addr: int, quantity: int,
    output_values: bytes,
) -> ModbusADU:
    """Build FC=15 (write multiple coils) request."""
    byte_count = len(output_values)
    pdu = (
        start_addr.to_bytes(2, "big")
        + quantity.to_bytes(2, "big")
        + bytes([byte_count])
        + output_values
    )
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=15, pdu_data=pdu,
    )


def build_write_multiple_coils_response(
    txid: int, unit_id: int, start_addr: int, quantity: int,
) -> ModbusADU:
    """Build FC=15 response: echoes start_addr + quantity."""
    pdu = start_addr.to_bytes(2, "big") + quantity.to_bytes(2, "big")
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=15, pdu_data=pdu,
    )


def build_report_server_id_request(txid: int, unit_id: int) -> ModbusADU:
    """Build FC=17 (report server id) request — no data."""
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=17, pdu_data=b"",
    )


def build_report_server_id_response(
    txid: int, unit_id: int, server_data: bytes = b"\x01\xff",
) -> ModbusADU:
    """Build FC=17 response: byte_count + server_data."""
    pdu = bytes([len(server_data)]) + server_data
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=17, pdu_data=pdu,
    )


def build_mei_request(
    txid: int, unit_id: int, mei_type: int, mei_data: bytes,
) -> ModbusADU:
    """Build FC=43 (encapsulated interface transport) request."""
    pdu = bytes([mei_type & 0xFF]) + mei_data
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=43, pdu_data=pdu,
    )


def build_mei_response(
    txid: int, unit_id: int, mei_type: int, mei_data: bytes,
) -> ModbusADU:
    """Build FC=43 response (echoes mei_type + data)."""
    pdu = bytes([mei_type & 0xFF]) + mei_data
    return ModbusADU(
        transaction_id=txid, unit_id=unit_id, function_code=43, pdu_data=pdu,
    )


# Mapping from function code to builder functions
FUNC_BUILDERS = {
    # Request builders
    1: ("request", build_read_coils_request),
    2: ("request", build_read_discrete_inputs_request),
    3: ("request", build_read_registers_request),
    4: ("request", build_read_input_registers_request),
    5: ("request", build_write_single_coil_request),
    6: ("request", build_write_single_register_request),
    8: ("request", build_diagnostics_request),
    11: ("request", build_event_counter_request),
    15: ("request", build_write_multiple_coils_request),
    16: ("request", build_write_multiple_registers_request),
    17: ("request", build_report_server_id_request),
    43: ("request", build_mei_request),
    # Response builders
    "response_1": build_read_coils_response,
    "response_2": build_read_discrete_inputs_response,
    "response_3": build_read_registers_response,
    "response_4": build_read_input_registers_response,
    "response_5": build_write_single_coil_response,
    "response_6": build_write_single_register_response,
    "response_8": build_diagnostics_response,
    "response_11": build_event_counter_response,
    "response_15": build_write_multiple_coils_response,
    "response_16": build_write_multiple_registers_response,
    "response_17": build_report_server_id_response,
    "response_43": build_mei_response,
}
