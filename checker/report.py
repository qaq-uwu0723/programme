"""Report data structures for checker findings."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    FATAL = "Fatal"
    ERROR = "Error"
    WARN = "Warn"
    INFO = "Info"


@dataclass
class Finding:
    pcap_index: int
    event_id: int
    flow: Dict[str, Any]
    severity: Severity
    code: str
    message: str
    observed: Optional[Dict[str, Any]] = None
    expected: Optional[Dict[str, Any]] = None


@dataclass
class ReportSummary:
    total_packets: int = 0
    total_findings: int = 0
    by_severity: Dict[str, int] = field(default_factory=lambda: {s.value: 0 for s in Severity})


@dataclass
class Report:
    summary: ReportSummary = field(default_factory=ReportSummary)
    findings: List[Finding] = field(default_factory=list)

    def add_finding(self, finding: Finding) -> None:
        self.findings.append(finding)
        self.summary.total_findings += 1
        self.summary.by_severity[finding.severity.value] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_packets": self.summary.total_packets,
                "total_findings": self.summary.total_findings,
                "by_severity": self.summary.by_severity,
            },
            "findings": [
                {
                    "pcap_index": f.pcap_index,
                    "event_id": f.event_id,
                    "flow": f.flow,
                    "severity": f.severity.value,
                    "code": f.code,
                    "message": f.message,
                    "observed": f.observed,
                    "expected": f.expected,
                }
                for f in self.findings
            ],
        }
