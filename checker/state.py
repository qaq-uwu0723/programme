"""Stateful tracking for Modbus transaction pairing and timeout detection."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OutstandingRequest:
    transaction_id: int
    unit_id: int
    flow_key: str
    pcap_index: int
    event_id: int
    ts_ns: int


@dataclass
class FlowState:
    outstanding: Dict[Tuple[int, int], OutstandingRequest] = field(default_factory=dict)
    unmatched_responses: List[dict] = field(default_factory=list)

    def add_request(self, req: OutstandingRequest) -> Optional[str]:
        """Register an outstanding request.

        Returns an error message string if a duplicate (txid, uid) exists,
        otherwise None.
        """
        key = (req.transaction_id, req.unit_id)
        if key in self.outstanding:
            return (
                f"Duplicate outstanding transaction_id={req.transaction_id} "
                f"unit_id={req.unit_id}"
            )
        self.outstanding[key] = req
        return None

    def match_response(
        self, transaction_id: int, unit_id: int
    ) -> Optional[OutstandingRequest]:
        """Match a response to an outstanding request. Returns the matched
        request if found, None otherwise."""
        key = (transaction_id, unit_id)
        return self.outstanding.pop(key, None)


@dataclass
class CheckerState:
    flows: Dict[str, FlowState] = field(default_factory=dict)
    timeout_ns: int = 10_000_000_000  # 10 second default

    @staticmethod
    def _flow_key(
        src_ip: str, src_port: int, dst_ip: str, dst_port: int
    ) -> str:
        """Normalize flow key so client↔server pairs map to the same key."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
        return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"

    def get_flow(
        self, src_ip: str, src_port: int, dst_ip: str, dst_port: int
    ) -> FlowState:
        key = self._flow_key(src_ip, src_port, dst_ip, dst_port)
        if key not in self.flows:
            self.flows[key] = FlowState()
        return self.flows[key]

    def check_timeouts(self, current_ts_ns: int) -> List[dict]:
        """Check for timed-out outstanding requests.

        Returns a list of timeout info dicts. Timed-out entries are removed
        from the state.
        """
        timeouts = []
        for flow_key, flow_state in self.flows.items():
            for (txid, uid), req in list(flow_state.outstanding.items()):
                if current_ts_ns - req.ts_ns > self.timeout_ns:
                    timeouts.append({
                        "flow": flow_key,
                        "transaction_id": txid,
                        "unit_id": uid,
                        "request_pcap_index": req.pcap_index,
                        "request_event_id": req.event_id,
                        "request_ts_ns": req.ts_ns,
                        "current_ts_ns": current_ts_ns,
                    })
                    del flow_state.outstanding[(txid, uid)]
        return timeouts
