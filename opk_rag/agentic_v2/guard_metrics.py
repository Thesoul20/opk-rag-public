from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentGuardMetrics:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **record: Any) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        latencies = sorted(float(r.get("latency_ms", 0.0)) for r in self.records)
        return {
            "guard_validation_count": len(self.records),
            "allow_count": self._count("allow"),
            "modify_count": self._count("modify"),
            "reject_count": self._count("reject"),
            "terminate_count": self._count("terminate"),
            "budget_rejection_count": sum(1 for r in self.records if "budget_exhausted" in str(r.get("reason_code", ""))),
            "duplicate_action_rejection_count": sum(1 for r in self.records if str(r.get("reason_code", "")).startswith("duplicate_") or r.get("reason_code") == "graph_recovery_already_used"),
            "graph_rejection_count": sum(1 for r in self.records if "graph" in str(r.get("reason_code", "")) and r.get("outcome") in {"reject", "terminate"}),
            "query_rewrite_rejection_count": sum(1 for r in self.records if "rewrite" in str(r.get("reason_code", "")) and r.get("outcome") == "reject"),
            "finish_rejection_count": sum(1 for r in self.records if str(r.get("reason_code", "")).startswith("finish_") and r.get("outcome") == "reject"),
            "policy_guard_disagreement_count": sum(1 for r in self.records if r.get("outcome") in {"modify", "reject", "terminate"}),
            "guard_p50_latency_ms": _percentile(latencies, 0.50),
            "guard_p95_latency_ms": _percentile(latencies, 0.95),
        }

    def _count(self, outcome: str) -> int:
        return sum(1 for r in self.records if r.get("outcome") == outcome)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return values[round((len(values) - 1) * q)]
