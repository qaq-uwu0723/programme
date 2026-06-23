"""Packet builder: converts diffusion-generated features → PCAP + JSONL sidecar.

This is the DIFFUSION → CHECKER integration contract.
Output format matches exactly what checker/validate.py expects.
"""
from typing import Dict, List, Optional, Tuple
import struct
from pathlib import Path

import torch
import numpy as np

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

from .modbus_rules import (
    ModbusADU,
    build_read_registers_request,
    build_read_registers_response,
    build_write_single_register_request,
    build_write_single_register_response,
    build_write_multiple_registers_request,
    build_write_multiple_registers_response,
    build_exception_response,
)
from .meta_writer import write_meta_line
from extractor.schema import FeatureSchema


# Column indices in the continuous tensor (matching default_modbus schema)
C_REG_VALUE_0 = 0
C_REG_VALUE_1 = 1
C_REG_VALUE_2 = 2
C_INTER_ARRIVAL_NS = 3
C_PAYLOAD_SIZE = 4
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
        pending_request: Optional[Dict] = None

        with open(output_meta, "w", encoding="utf-8") as meta_fp:
            for i in range(N):
                fc_idx = int(Y[i, D_FUNCTION_CODE])
                func_code = FC_VOCAB[fc_idx] if fc_idx < len(FC_VOCAB) else 3
                direction = "c2s" if Y[i, D_DIRECTION] == 0 else "s2c"
                unit_id = int(Y[i, D_UNIT_ID]) % 248
                txid = int(Y[i, D_TRANSACTION_ID]) % 65536
                is_exc = bool(Y[i, D_IS_EXCEPTION])
                exc_code = int(Y[i, D_EXCEPTION_CODE])
                reg_addr = max(0, min(65535, int(X[i, C_REG_ADDRESS])))
                quantity = max(1, min(125, int(X[i, C_QUANTITY])))

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
                    seq, ack = next_ack, next_seq + (len(adu.raw) if pending_request else 0)
                    next_ack += len(adu.raw) if pending_request else 0

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
                expected_fields = {
                    "starting_address": reg_addr,
                    "quantity_of_registers": quantity,
                }

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
                inter_arrival_ns = max(1000, int(abs(X[i, C_INTER_ARRIVAL_NS])))
                ts_ns += inter_arrival_ns

                if direction == "c2s":
                    pending_request = {"txid": txid, "func_code": func_code, "size": len(adu.raw)}
                else:
                    pending_request = None

        # Write PCAP
        wrpcap(output_pcap, packets)
        print(f"Wrote {len(packets)} packets to {output_pcap}")
        print(f"Wrote {N} metadata lines to {output_meta}")

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

        if direction == "c2s":
            # Request
            if func_code == 3:
                return build_read_registers_request(txid, unit_id, reg_addr, quantity)
            elif func_code == 6:
                reg_val = max(0, min(65535, int(X[i, C_REG_VALUE_0])))
                return build_write_single_register_request(txid, unit_id, reg_addr, reg_val)
            elif func_code == 16:
                byte_count = quantity * 2
                reg_vals = (
                    max(0, min(65535, int(X[i, C_REG_VALUE_0]))).to_bytes(2, "big")
                    + max(0, min(65535, int(X[i, C_REG_VALUE_1]))).to_bytes(2, "big")
                )
                reg_vals = (reg_vals * ((byte_count // len(reg_vals)) + 1))[:byte_count]
                return build_write_multiple_registers_request(
                    txid, unit_id, reg_addr, quantity, reg_vals,
                )
            else:
                # Fallback: FC=3 read holding registers
                return build_read_registers_request(txid, unit_id, reg_addr, quantity)
        else:
            # Response
            if func_code == 3:
                byte_count = quantity * 2
                reg_vals = (
                    max(0, min(65535, int(X[i, C_REG_VALUE_0]))).to_bytes(2, "big")
                    + max(0, min(65535, int(X[i, C_REG_VALUE_1]))).to_bytes(2, "big")
                )
                reg_vals = (reg_vals * ((byte_count // len(reg_vals)) + 1))[:byte_count]
                return build_read_registers_response(txid, unit_id, reg_vals)
            elif func_code == 6:
                reg_val = max(0, min(65535, int(X[i, C_REG_VALUE_0])))
                return build_write_single_register_response(txid, unit_id, reg_addr, reg_val)
            elif func_code == 16:
                return build_write_multiple_registers_response(txid, unit_id, reg_addr, quantity)
            else:
                # Fallback: FC=3 echo with empty registers
                return build_read_registers_response(txid, unit_id, b"\x00\x00")
