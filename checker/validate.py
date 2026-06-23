"""Main validation pipeline — runs all checker rules."""
from typing import Dict, List

from .report import Finding, Report, Severity
from .meta import PacketMeta
from .decode import DecodedPacket
from .config import Config
from .mbap import parse_mbap, parse_modbus_adu, DecodedModbus
from .modbus_desc import parse_full_pdu
from .state import CheckerState, OutstandingRequest


def validate(
    pcap_path: str,
    meta_path: str,
    config: Config,
    mode: str = "mvp",
) -> Report:
    """Run the full validation pipeline over a PCAP + JSONL trace pair.

    Args:
        pcap_path: Path to the PCAP/PCAPNG file.
        meta_path: Path to the JSONL sidecar metadata file.
        config: Modbus descriptor configuration.
        mode: Validation mode — 'mvp' or 'strict'.

    Returns:
        A Report containing all findings and a summary.
    """
    from .pcap_in import iter_aligned

    report = Report()
    state = CheckerState()
    packet_count = 0

    for pcap_index, raw_pkt, meta, decoded, error in iter_aligned(pcap_path, meta_path):
        packet_count += 1

        flow_dict: Dict = {}
        if meta is not None:
            flow_dict = meta.flow.to_dict()

        # --- Step 0–1: alignment errors ---
        if error is not None:
            report.add_finding(Finding(
                pcap_index=pcap_index,
                event_id=meta.event_id if meta else -1,
                flow=flow_dict,
                severity=Severity.FATAL,
                code="ALIGNMENT_ERROR",
                message=error,
            ))
            continue

        # Both decoded and meta are non-None after this point
        assert decoded is not None and meta is not None

        # Use actual decoded src/dst for accurate flow reporting
        flow_dict = {
            "src_ip": decoded.src_ip,
            "src_port": decoded.src_port,
            "dst_ip": decoded.dst_ip,
            "dst_port": decoded.dst_port,
        }

        # --- Step 2: port sanity ---
        if decoded.dst_port != 502 and decoded.src_port != 502:
            report.add_finding(Finding(
                pcap_index=pcap_index,
                event_id=meta.event_id,
                flow=flow_dict,
                severity=Severity.WARN,
                code="NON_STANDARD_PORT",
                message=(
                    f"No port 502 on either side "
                    f"(src={decoded.src_port}, dst={decoded.dst_port})"
                ),
            ))

        # --- Step 2: empty payload ---
        payload = decoded.tcp_payload
        if len(payload) == 0:
            report.add_finding(Finding(
                pcap_index=pcap_index,
                event_id=meta.event_id,
                flow=flow_dict,
                severity=Severity.ERROR,
                code="EMPTY_TCP_PAYLOAD",
                message="TCP payload is empty (no Modbus data)",
            ))
            continue

        # --- Step 3: MBAP validation ---
        mbap_findings = _check_mbap(payload, pcap_index, meta.event_id, flow_dict)
        for f in mbap_findings:
            report.add_finding(f)

        # --- Step 3: parse MBAP for downstream use ---
        try:
            mbap, pdu = parse_mbap(payload)
        except ValueError as e:
            report.add_finding(Finding(
                pcap_index=pcap_index,
                event_id=meta.event_id,
                flow=flow_dict,
                severity=Severity.FATAL,
                code="MBAP_PARSE_ERROR",
                message=str(e),
            ))
            continue

        # --- Step 3: PDU validation + descriptor parse ---
        pdu_findings = _check_modbus_pdu(
            pdu, mbap.unit_id, meta.direction, config,
            pcap_index, meta.event_id, flow_dict, mode,
        )
        for f in pdu_findings:
            report.add_finding(f)

        # --- Parse full ADU for state tracking & feature-fidelity ---
        try:
            adu = parse_modbus_adu(payload)
        except ValueError:
            continue

        # Attach descriptor-parsed fields to the ADU for downstream checks
        if not adu.is_exception:
            try:
                parsed = parse_full_pdu(pdu, mbap.unit_id, config, meta.direction)
                adu.parsed_fields = parsed
            except Exception:
                pass

        # --- Step 4: transaction pairing ---
        tx_findings = _check_transaction(
            adu, meta, state, decoded, pcap_index, flow_dict,
        )
        for f in tx_findings:
            report.add_finding(f)

        # --- Step 5: feature-fidelity checks ---
        if meta.expected is not None:
            fidelity_findings = _check_expected(
                adu, meta, pcap_index, flow_dict,
            )
            for f in fidelity_findings:
                report.add_finding(f)

        # --- timeout sweep ---
        timeout_findings = _check_timeouts(state, meta.ts_ns)
        for f in timeout_findings:
            report.add_finding(f)

    report.summary.total_packets = packet_count
    return report


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def _check_mbap(
    payload: bytes, pcap_index: int, event_id: int, flow: dict
) -> List[Finding]:
    """RuleMbapValid: check MBAP header invariants."""
    findings: List[Finding] = []

    if len(payload) < 7:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=event_id, flow=flow,
            severity=Severity.FATAL,
            code="MBAP_TOO_SHORT",
            message=f"Payload too short for MBAP: {len(payload)} bytes (need >= 7)",
        ))
        return findings

    protocol_id = int.from_bytes(payload[2:4], byteorder="big")
    length = int.from_bytes(payload[4:6], byteorder="big")
    actual_remaining = len(payload) - 6  # bytes after the length field

    if protocol_id != 0:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=event_id, flow=flow,
            severity=Severity.WARN,
            code="MBAP_INVALID_PROTOCOL_ID",
            message=f"MBAP protocol_id should be 0, got {protocol_id}",
            observed={"protocol_id": protocol_id},
            expected={"protocol_id": 0},
        ))

    if length != actual_remaining:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=event_id, flow=flow,
            severity=Severity.ERROR,
            code="MBAP_LENGTH_MISMATCH",
            message=(
                f"MBAP length field ({length}) does not match "
                f"remaining bytes ({actual_remaining})"
            ),
            observed={"mbap_length": length, "actual_remaining": actual_remaining},
        ))

    return findings


def _check_modbus_pdu(
    pdu: bytes,
    unit_id: int,
    direction: str,
    config: Config,
    pcap_index: int,
    event_id: int,
    flow: dict,
    mode: str,
) -> List[Finding]:
    """RuleFunctionPduWellFormed: validate PDU structure and parse with descriptor."""
    findings: List[Finding] = []

    if len(pdu) < 1:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=event_id, flow=flow,
            severity=Severity.ERROR,
            code="PDU_EMPTY",
            message="Modbus PDU is empty (no function code)",
        ))
        return findings

    function_code = pdu[0]
    is_exception = (function_code & 0x80) != 0
    base_fc = function_code & 0x7F
    data = pdu[1:]

    desc = config.get_descriptor(base_fc)
    if desc is None:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=event_id, flow=flow,
            severity=Severity.WARN,
            code="PDU_UNKNOWN_FUNCTION_CODE",
            message=f"Unknown Modbus function code: {base_fc}",
            observed={"function_code": base_fc, "is_exception": is_exception},
        ))

    if is_exception:
        if len(data) < 1:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=event_id, flow=flow,
                severity=Severity.ERROR,
                code="PDU_EXCEPTION_NO_CODE",
                message="Exception response has no exception code byte",
            ))
        return findings

    # Descriptor parse
    if desc is not None:
        try:
            result = parse_full_pdu(pdu, unit_id, config, direction)
            if "_parse_error" in result:
                findings.append(Finding(
                    pcap_index=pcap_index, event_id=event_id, flow=flow,
                    severity=Severity.ERROR,
                    code="PDU_DESCRIPTOR_PARSE_ERROR",
                    message=result["_parse_error"],
                ))
        except Exception as e:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=event_id, flow=flow,
                severity=Severity.ERROR,
                code="PDU_PARSE_ERROR",
                message=str(e),
            ))

    # --- Strict-mode extras ---
    if mode == "strict":
        if unit_id > 247:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=event_id, flow=flow,
                severity=Severity.WARN,
                code="PDU_UNIT_ID_RANGE",
                message=f"Unit ID {unit_id} outside typical range 0–247",
                observed={"unit_id": unit_id},
            ))

    return findings


def _check_transaction(
    adu: DecodedModbus,
    meta: PacketMeta,
    state: CheckerState,
    decoded: DecodedPacket,
    pcap_index: int,
    flow: dict,
) -> List[Finding]:
    """RuleTxIdPairing: match requests and responses by (txid, unit_id)."""
    findings: List[Finding] = []

    flow_state = state.get_flow(
        decoded.src_ip, decoded.src_port,
        decoded.dst_ip, decoded.dst_port,
    )

    is_request = meta.direction == "c2s"

    if is_request:
        req = OutstandingRequest(
            transaction_id=adu.transaction_id,
            unit_id=adu.unit_id,
            flow_key=f"{decoded.src_ip}:{decoded.src_port}-{decoded.dst_ip}:{decoded.dst_port}",
            pcap_index=pcap_index,
            event_id=meta.event_id,
            ts_ns=meta.ts_ns,
        )
        err = flow_state.add_request(req)
        if err:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
                severity=Severity.WARN,
                code="TX_DUPLICATE_OUTSTANDING",
                message=err,
            ))
    else:
        matched = flow_state.match_response(adu.transaction_id, adu.unit_id)
        if matched is None:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
                severity=Severity.ERROR,
                code="TX_UNMATCHED_RESPONSE",
                message=(
                    f"No outstanding request for "
                    f"transaction_id={adu.transaction_id} unit_id={adu.unit_id}"
                ),
                observed={
                    "transaction_id": adu.transaction_id,
                    "unit_id": adu.unit_id,
                },
            ))

    return findings


def _check_expected(
    adu: DecodedModbus,
    meta: PacketMeta,
    pcap_index: int,
    flow: dict,
) -> List[Finding]:
    """RuleExpectedMatch: compare decoded values against JSONL expected block."""
    findings: List[Finding] = []
    exp = meta.expected
    if exp is None or exp.modbus is None:
        return findings

    emb = exp.modbus

    if emb.transaction_id is not None and emb.transaction_id != adu.transaction_id:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
            severity=Severity.ERROR,
            code="EXPECTED_TXID_MISMATCH",
            message="Transaction ID mismatch",
            observed={"transaction_id": adu.transaction_id},
            expected={"transaction_id": emb.transaction_id},
        ))

    if emb.unit_id is not None and emb.unit_id != adu.unit_id:
        findings.append(Finding(
            pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
            severity=Severity.ERROR,
            code="EXPECTED_UNIT_ID_MISMATCH",
            message="Unit ID mismatch",
            observed={"unit_id": adu.unit_id},
            expected={"unit_id": emb.unit_id},
        ))

    if emb.function_code is not None:
        if not adu.is_exception and emb.function_code != adu.function_code:
            findings.append(Finding(
                pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
                severity=Severity.ERROR,
                code="EXPECTED_FC_MISMATCH",
                message="Function code mismatch",
                observed={"function_code": adu.function_code},
                expected={"function_code": emb.function_code},
            ))

    if exp.fields and adu.parsed_fields:
        parsed = adu.parsed_fields.get("fields", {})
        for field_name, expected_val in exp.fields.items():
            actual_val = parsed.get(field_name)
            if actual_val is not None and actual_val != expected_val:
                findings.append(Finding(
                    pcap_index=pcap_index, event_id=meta.event_id, flow=flow,
                    severity=Severity.WARN,
                    code="EXPECTED_FIELD_MISMATCH",
                    message=f"Field '{field_name}' mismatch",
                    observed={field_name: actual_val},
                    expected={field_name: expected_val},
                ))

    return findings


def _check_timeouts(state: CheckerState, current_ts_ns: int) -> List[Finding]:
    """Sweep outstanding requests for timeouts."""
    findings: List[Finding] = []
    for t in state.check_timeouts(current_ts_ns):
        elapsed_ns = current_ts_ns - t["request_ts_ns"]
        findings.append(Finding(
            pcap_index=t["request_pcap_index"],
            event_id=t.get("request_event_id", -1),
            flow={"flow": t["flow"]},
            severity=Severity.WARN,
            code="TX_TIMEOUT",
            message=(
                f"Request timed out: txid={t['transaction_id']} "
                f"uid={t['unit_id']} (elapsed {elapsed_ns / 1e9:.3f}s)"
            ),
            observed={
                "transaction_id": t["transaction_id"],
                "unit_id": t["unit_id"],
                "elapsed_ns": elapsed_ns,
            },
        ))
    return findings
