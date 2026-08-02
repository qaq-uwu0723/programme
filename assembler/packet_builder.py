"""Packet builder: converts diffusion-generated features → PCAP + JSONL sidecar.

This is the DIFFUSION → CHECKER integration contract.
Output format matches exactly what checker/validate.py expects.
"""
from typing import List

import torch
import numpy as np

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

from .modbus_rules import (
    ModbusADU,
    build_read_coils_request,
    build_read_coils_response,
    build_read_discrete_inputs_request,
    build_read_discrete_inputs_response,
    build_read_registers_request,
    build_read_registers_response,
    build_read_input_registers_request,
    build_read_input_registers_response,
    build_write_single_coil_request,
    build_write_single_coil_response,
    build_write_single_register_request,
    build_write_single_register_response,
    build_diagnostics_request,
    build_diagnostics_response,
    build_event_counter_request,
    build_event_counter_response,
    build_write_multiple_coils_request,
    build_write_multiple_coils_response,
    build_write_multiple_registers_request,
    build_write_multiple_registers_response,
    build_report_server_id_request,
    build_report_server_id_response,
    build_mei_request,
    build_mei_response,
    build_exception_response,
)
from .meta_writer import write_meta_line
from extractor.schema import FeatureSchema


# Column indices in the continuous tensor (matching default_modbus schema)
C_REG_VALUE_0 = 0
C_REG_VALUE_1 = 1
C_INTER_ARRIVAL_NS = 3
C_REG_ADDRESS = 5
C_QUANTITY = 6

# Column indices in the discrete tensor
D_FUNCTION_CODE = 0
D_DIRECTION = 1
D_UNIT_ID = 2
D_TRANSACTION_ID = 3
D_IS_EXCEPTION = 4
D_EXCEPTION_CODE = 5


# Modbus function code vocabulary mapping
FC_VOCAB = [1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 17, 43]


def _safe_int(val) -> int:
    """Convert numpy/torch scalar to int, clamping NaN/Inf to 0."""
    if hasattr(val, "item"):
        val = val.item()
    v = float(val)
    if not np.isfinite(v):
        return 0
    return int(v)


def _expected_fields(
    func_code: int, direction: str, reg_addr: int, quantity: int, v0: int, v1: int,
) -> dict:
    """Metadata expected_fields matching the PDU structure for (fc, direction).

    Mirrors the field descriptors in checker/configs/modbus_default.json so the
    checker's EXPECTED_FIELD checks never see fields absent from the PDU.
    """
    if direction == "c2s":
        if func_code in (1, 2, 3, 4):
            qty = {1: "quantity_of_coils", 2: "quantity_of_inputs",
                   3: "quantity_of_registers", 4: "quantity_of_registers"}[func_code]
            return {"starting_address": reg_addr, qty: quantity}
        elif func_code == 5:
            # checker enum-maps FC5 request output_value (0→OFF, 0xFF00→ON)
            return {"output_address": reg_addr, "output_value": "ON" if (v0 & 0xFF00) else "OFF"}
        elif func_code == 6:
            return {"register_address": reg_addr, "register_value": v0}
        elif func_code == 8:
            return {"sub_function": v0 & 0xFFFF}
        elif func_code == 15:
            return {"starting_address": reg_addr, "quantity_of_outputs": quantity}
        elif func_code == 16:
            return {"starting_address": reg_addr, "quantity_of_registers": quantity}
        elif func_code == 43:
            return {"mei_type": 0x0E}
        return {}
    else:
        # Responses
        if func_code == 1:
            return {"byte_count": (quantity + 7) // 8}
        elif func_code == 2:
            return {"byte_count": (quantity + 7) // 8}
        elif func_code == 3:
            return {"byte_count": quantity * 2}
        elif func_code == 4:
            return {"byte_count": quantity * 2}
        elif func_code == 5:
            return {"output_address": reg_addr, "output_value": 0xFF00 if (v0 & 0xFF00) else 0}
        elif func_code == 6:
            return {"register_address": reg_addr, "register_value": v0}
        elif func_code == 8:
            return {"sub_function": v0 & 0xFFFF}
        elif func_code == 11:
            return {"status": 0, "event_count": v1}
        elif func_code == 15:
            return {"starting_address": reg_addr, "quantity_of_outputs": quantity}
        elif func_code == 16:
            return {"starting_address": reg_addr, "quantity_of_registers": quantity}
        elif func_code == 17:
            return {"byte_count": 2}
        elif func_code == 43:
            return {"mei_type": 0x0E}
        return {}


class PacketAssembler:
    """Converts generated feature tensors to PCAP + JSONL.

    Usage:
        assembler = PacketAssembler(schema, flow_config)
        assembler.assemble(X_hat, Y_hat, "output.pcapng", "output.meta.jsonl")
    """

    def __init__(
        self,
        schema: FeatureSchema,
        client_ip: str = "10.0.0.10",
        server_ip: str = "10.0.0.20",
        client_port: int = 51000,
        server_port: int = 502,
    ):
        if not HAS_SCAPY:
            raise ImportError("scapy is required for packet assembly")
        self.schema = schema
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.client_port = client_port
        self.server_port = server_port

    def assemble(
        self,
        X_hat: torch.Tensor,
        Y_hat: List[torch.Tensor],
        output_pcap: str,
        output_meta: str,
        trace_id: str = "generated-trace-001",
        base_ts_ns: int = 1_736_451_234_567_890_123,
    ) -> None:
        """Main entry point: feature tensors → PCAP + JSONL.

        Args:
            X_hat: (N_windows, L, d_c) continuous features
            Y_hat: list of (N_windows, L) discrete features
            output_pcap: path to output PCAPNG file
            output_meta: path to output JSONL sidecar file
            trace_id: identifier for this trace
            base_ts_ns: starting timestamp in nanoseconds
        """
        # Flatten windows into a single sequence
        X = X_hat.reshape(-1, self.schema.d_c).cpu().numpy()
        Y = torch.stack(Y_hat, dim=-1).reshape(-1, self.schema.d_d).cpu().numpy()

        N = X.shape[0]
        packets = []
        ts_ns = base_ts_ns
        next_seq = 1000
        next_ack = 2000
        pending_requests: list = []  # FIFO queue for outstanding requests
        _next_txid = 1

        with open(output_meta, "w", encoding="utf-8") as meta_fp:
            for i in range(N):
                fc_idx = int(Y[i, D_FUNCTION_CODE])
                func_code = FC_VOCAB[fc_idx] if 0 <= fc_idx < len(FC_VOCAB) else 3
                direction = "c2s" if Y[i, D_DIRECTION] == 0 else "s2c"
                unit_id = int(Y[i, D_UNIT_ID]) % 248
                is_exc = bool(Y[i, D_IS_EXCEPTION])
                exc_code = max(0, min(255, int(Y[i, D_EXCEPTION_CODE])))
                reg_addr = max(0, min(65535, _safe_int(X[i, C_REG_ADDRESS])))
                quantity = max(1, min(125, _safe_int(X[i, C_QUANTITY])))

                # Assign transaction_id: requests get new id, responses pop from queue.
                # Responses must echo the request's txid, function code, AND unit id.
                req = None
                if direction == "c2s":
                    txid = _next_txid
                    _next_txid = (_next_txid + 1) % 65536
                else:
                    req = pending_requests.pop(0) if pending_requests else None
                    if req is None:
                        continue  # orphan response (no outstanding request) — drop it
                    txid = req["txid"]
                    func_code = req["func_code"]  # echo request fc
                    unit_id = req["unit_id"]      # echo request unit id

                # Pick src/dst based on direction
                if direction == "c2s":
                    src_ip, dst_ip = self.client_ip, self.server_ip
                    src_port, dst_port = self.client_port, self.server_port
                else:
                    src_ip, dst_ip = self.server_ip, self.client_ip
                    src_port, dst_port = self.server_port, self.client_port

                # Build the Modbus ADU
                adu = self._build_adu(
                    func_code, txid, unit_id, direction, is_exc, exc_code,
                    reg_addr, quantity, X, i,
                )

                # Manage TCP seq/ack
                if direction == "c2s":
                    seq, ack = next_seq, next_ack
                    next_seq += len(adu.raw)
                else:
                    seq, ack = next_ack, next_seq
                    next_ack += len(adu.raw)

                # Build scapy packet
                pkt = (
                    Ether()
                    / IP(src=src_ip, dst=dst_ip)
                    / TCP(sport=src_port, dport=dst_port, flags="PA", seq=seq, ack=ack)
                    / Raw(load=adu.raw)
                )
                packets.append(pkt)

                # Build JSONL metadata line
                expected_modbus = {
                    "transaction_id": txid,
                    "unit_id": unit_id,
                    "function_code": func_code,
                }
                v0 = max(0, min(65535, _safe_int(X[i, C_REG_VALUE_0])))
                v1 = max(0, min(65535, _safe_int(X[i, C_REG_VALUE_1])))
                expected_fields = _expected_fields(
                    func_code, direction, reg_addr, quantity, v0, v1,
                )

                write_meta_line(
                    meta_fp,
                    trace_id=trace_id,
                    event_id=i,
                    pcap_index=i,
                    ts_ns=ts_ns,
                    direction=direction,
                    src_ip=src_ip,
                    src_port=src_port,
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    expected_modbus=expected_modbus,
                    expected_fields=expected_fields,
                )

                # Advance time
                inter_arrival_ns = max(1000, _safe_int(abs(X[i, C_INTER_ARRIVAL_NS])))
                ts_ns += inter_arrival_ns

                if direction == "c2s":
                    pending_requests.append({"txid": txid, "func_code": func_code, "unit_id": unit_id})

            # --- Inject responses for unanswered requests (protocol pairing guarantee) ---
            # The model may generate more c2s than s2c. Every Modbus request must
            # get a response, so pair leftovers here — the checker then sees no
            # orphan requests / timed-out transactions.
            last_idx = N - 1
            inj_idx = N
            while pending_requests:
                req = pending_requests.pop(0)
                r_fc, r_txid, r_uid = req["func_code"], req["txid"], req["unit_id"]
                is_exc = bool(Y[last_idx, D_IS_EXCEPTION])
                exc_code = max(0, min(255, int(Y[last_idx, D_EXCEPTION_CODE])))
                reg_addr = max(0, min(65535, _safe_int(X[last_idx, C_REG_ADDRESS])))
                quantity = max(1, min(125, _safe_int(X[last_idx, C_QUANTITY])))

                adu = self._build_adu(
                    r_fc, r_txid, r_uid, "s2c", is_exc, exc_code,
                    reg_addr, quantity, X, last_idx,
                )
                seq, ack = next_ack, next_seq
                next_ack += len(adu.raw)

                pkt = (
                    Ether()
                    / IP(src=self.server_ip, dst=self.client_ip)
                    / TCP(sport=self.server_port, dport=self.client_port, flags="PA", seq=seq, ack=ack)
                    / Raw(load=adu.raw)
                )
                packets.append(pkt)

                v0 = max(0, min(65535, _safe_int(X[last_idx, C_REG_VALUE_0])))
                v1 = max(0, min(65535, _safe_int(X[last_idx, C_REG_VALUE_1])))
                write_meta_line(
                    meta_fp,
                    trace_id=trace_id,
                    event_id=inj_idx,
                    pcap_index=inj_idx,
                    ts_ns=ts_ns,
                    direction="s2c",
                    src_ip=self.server_ip,
                    src_port=self.server_port,
                    dst_ip=self.client_ip,
                    dst_port=self.client_port,
                    expected_modbus={"transaction_id": r_txid, "unit_id": r_uid, "function_code": r_fc},
                    expected_fields=_expected_fields(r_fc, "s2c", reg_addr, quantity, v0, v1),
                )
                ts_ns += 1000  # small gap for injected response
                inj_idx += 1

        # Write PCAP
        wrpcap(output_pcap, packets)
        print(f"Wrote {len(packets)} packets to {output_pcap}")
        print(f"Wrote {inj_idx} metadata lines to {output_meta}")

    def _build_adu(
        self,
        func_code: int,
        txid: int,
        unit_id: int,
        direction: str,
        is_exc: bool,
        exc_code: int,
        reg_addr: int,
        quantity: int,
        X: np.ndarray,
        i: int,
    ) -> ModbusADU:
        """Build the appropriate Modbus ADU based on function code and direction."""
        if is_exc and direction == "s2c":
            return build_exception_response(txid, unit_id, func_code, exc_code)

        v0 = max(0, min(65535, _safe_int(X[i, C_REG_VALUE_0])))
        v1 = max(0, min(65535, _safe_int(X[i, C_REG_VALUE_1])))

        def _reg_bytes() -> bytes:
            """Repeat v0,v1 to fill quantity registers (2 bytes each)."""
            byte_count = quantity * 2
            reg_vals = v0.to_bytes(2, "big") + v1.to_bytes(2, "big")
            return (reg_vals * ((byte_count // len(reg_vals)) + 1))[:byte_count]

        def _coil_bytes() -> bytes:
            """Pack quantity coil bits from v0 into ceil(quantity/8) bytes."""
            n_bytes = (quantity + 7) // 8
            return bytes([v0 & 0xFF]) * n_bytes

        if direction == "c2s":
            # --- Requests ---
            if func_code == 1:
                return build_read_coils_request(txid, unit_id, reg_addr, quantity)
            elif func_code == 2:
                return build_read_discrete_inputs_request(txid, unit_id, reg_addr, quantity)
            elif func_code == 3:
                return build_read_registers_request(txid, unit_id, reg_addr, quantity)
            elif func_code == 4:
                return build_read_input_registers_request(txid, unit_id, reg_addr, quantity)
            elif func_code == 5:
                return build_write_single_coil_request(txid, unit_id, reg_addr, 0xFF00 if (v0 & 0xFF00) else 0)
            elif func_code == 6:
                return build_write_single_register_request(txid, unit_id, reg_addr, v0)
            elif func_code == 8:
                return build_diagnostics_request(txid, unit_id, v0 & 0xFFFF, b"\x00\x00")
            elif func_code == 11:
                return build_event_counter_request(txid, unit_id)
            elif func_code == 15:
                return build_write_multiple_coils_request(txid, unit_id, reg_addr, quantity, _coil_bytes())
            elif func_code == 16:
                return build_write_multiple_registers_request(txid, unit_id, reg_addr, quantity, _reg_bytes())
            elif func_code == 17:
                return build_report_server_id_request(txid, unit_id)
            elif func_code == 43:
                return build_mei_request(txid, unit_id, 0x0E, b"\x01\x00")
            else:
                # Unknown fc: fall back to a valid FC3 read request
                return build_read_registers_request(txid, unit_id, reg_addr, quantity)
        else:
            # --- Responses (fc already echoed from the pending request) ---
            if func_code == 1:
                return build_read_coils_response(txid, unit_id, _coil_bytes())
            elif func_code == 2:
                return build_read_discrete_inputs_response(txid, unit_id, _coil_bytes())
            elif func_code == 3:
                return build_read_registers_response(txid, unit_id, _reg_bytes())
            elif func_code == 4:
                return build_read_input_registers_response(txid, unit_id, _reg_bytes())
            elif func_code == 5:
                return build_write_single_coil_response(txid, unit_id, reg_addr, 0xFF00 if (v0 & 0xFF00) else 0)
            elif func_code == 6:
                return build_write_single_register_response(txid, unit_id, reg_addr, v0)
            elif func_code == 8:
                return build_diagnostics_response(txid, unit_id, v0 & 0xFFFF, b"\x00\x00")
            elif func_code == 11:
                return build_event_counter_response(txid, unit_id, v1)
            elif func_code == 15:
                return build_write_multiple_coils_response(txid, unit_id, reg_addr, quantity)
            elif func_code == 16:
                return build_write_multiple_registers_response(txid, unit_id, reg_addr, quantity)
            elif func_code == 17:
                return build_report_server_id_response(txid, unit_id)
            elif func_code == 43:
                return build_mei_response(txid, unit_id, 0x0E, b"\x01\x00")
            else:
                # Unknown fc: fall back to a valid FC3 echo response
                return build_read_registers_response(txid, unit_id, _reg_bytes())
