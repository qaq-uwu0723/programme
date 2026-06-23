"""FARAONIC CSV dataset reader — maps Modbus/TCP CSV features to PacketRecord."""
import csv
from typing import Dict, Iterator, List, Optional
from .pcap_reader import PacketRecord

# FARAONIC function codes → vocabulary index
FARAONIC_FC_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: 8}


def read_faraonic_csv(
    path: str,
    max_rows: Optional[int] = None,
    label_filter: Optional[str] = "NORMAL",
) -> List[PacketRecord]:
    """Read FARAONIC CSV and convert to PacketRecord list.

    Args:
        path: CSV file path (semicolon-delimited)
        max_rows: max rows to read (useful for sampling)
        label_filter: only keep rows with this Classification (None = all)

    Returns:
        list of PacketRecord
    """
    records: List[PacketRecord] = []
    prev_ts: Optional[float] = None

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)
        col = {h: i for i, h in enumerate(headers)}

        for row in reader:
            if max_rows and len(records) >= max_rows:
                break
            if label_filter and row[col["Classification"]] != label_filter:
                continue

            # --- Parse core fields ---
            try:
                ts = float(row[col["timestamp"]])
                src_ip = row[col["IP_src"]]
                dst_ip = row[col["IP_dst"]]
                src_port = int(row[col["TCP_sport"]])
                dst_port = int(row[col["TCP_dport"]])
            except (ValueError, IndexError):
                continue

            # Direction
            direction = "c2s" if dst_port == 502 else "s2c"

            # Inter-arrival time
            if prev_ts is not None:
                inter_arrival_ns = int(max(1, (ts - prev_ts) * 1_000_000_000))
            else:
                inter_arrival_ns = 50_000_000  # 50ms default
            prev_ts = ts

            # --- Modbus fields ---
            func_code = int(row[col["ModbusTCPRequest_func_code"]] or 0)
            unit_id = int(row[col["ModbusTCPRequest_unit_id"]] or 1) % 248
            txid = int(row[col["ModbusTCPRequest_trans_id"]] or 0) % 65536

            # Register address (reference_number)
            ref_cols = [
                "ModbusReadDiscreteInputsRequest_reference_number",
                "ModbusWriteMultipleCoilsRequest_reference_number",
            ]
            reg_addr = 0
            for rc in ref_cols:
                if rc in col and row[col[rc]]:
                    reg_addr = int(row[col[rc]])
                    break

            # Quantity (bit_count or byte_count)
            qty_cols = [
                "ModbusReadDiscreteInputsRequest_bit_count",
                "ModbusWriteMultipleCoilsRequest_bit_count",
            ]
            quantity = 1
            for qc in qty_cols:
                if qc in col and row[col[qc]]:
                    quantity = int(row[col[qc]])
                    break

            # Register value from response
            reg_val_0 = 0
            resp_cols = [
                "ModbusReadDiscreteInputsResponse_input_status",
                "ModbusWriteMultipleCoilsResponse_bit_count",
            ]
            for rc in resp_cols:
                if rc in col and row[col[rc]]:
                    # input_status/coil_status is a bytes-like hex string
                    val_str = row[col[rc]]
                    if val_str:
                        try:
                            reg_val_0 = int(val_str, 16) if val_str.startswith("0x") else int(val_str)
                        except ValueError:
                            reg_val_0 = 0
                    break

            # Payload size from IP_len
            payload_size = int(row[col["IP_len"]] or 40)

            records.append(PacketRecord(
                ts_ns=int(ts * 1_000_000_000),
                inter_arrival_ns=inter_arrival_ns,
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=src_port, dst_port=dst_port,
                direction=direction,
                transaction_id=txid,
                protocol_id=0,
                unit_id=unit_id,
                function_code=func_code,
                is_exception=False,
                exception_code=0,
                pdu_data=b"",
                payload_size=payload_size,
                register_address=reg_addr,
                register_values=[reg_val_0, 0, 0],
                quantity=max(1, quantity),
            ))

    return records
