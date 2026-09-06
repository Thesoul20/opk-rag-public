from __future__ import annotations

from statistics import median
from typing import Any


class AgentStateTransitionMetrics:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, *, action: str, transition_status: str, reason_code: str, tool_status: str, latency_ms: float) -> None:
        self.records.append({
            "action": action,
            "transition_status": transition_status,
            "reason_code": reason_code,
            "tool_status": tool_status,
            "latency_ms": float(latency_ms),
        })

    def summary(self) -> dict[str, Any]:
        latencies = sorted(float(row["latency_ms"]) for row in self.records)
        def count(field: str, value: str) -> int:
            return sum(1 for row in self.records if row.get(field) == value)
        def percentile95(values: list[float]) -> float:
            if not values:
                return 0.0
            index = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
            return values[index]
        return {
            "transition_count": len(self.records),
            "applied_count": count("transition_status", "applied"),
            "rejected_count": count("transition_status", "rejected"),
            "terminated_count": count("transition_status", "terminated"),
            "hybrid_transition_count": count("action", "hybrid_search"),
            "structure_transition_count": count("action", "structure_search"),
            "graph_transition_count": count("action", "graph_search"),
            "rewrite_transition_count": count("action", "rewrite_query"),
            "inspect_transition_count": count("action", "inspect_evidence"),
            "finish_transition_count": count("action", "finish"),
            "abstain_transition_count": count("action", "abstain"),
            "tool_failure_transition_count": count("tool_status", "failed"),
            "no_result_transition_count": count("tool_status", "no_result"),
            "duplicate_execution_rejection_count": count("reason_code", "duplicate_execution_transition"),
            "invariant_violation_count": sum(1 for row in self.records if "invariant_violation" in str(row.get("reason_code", ""))),
            "p50_transition_latency_ms": median(latencies) if latencies else 0.0,
            "p95_transition_latency_ms": percentile95(latencies),
        }
