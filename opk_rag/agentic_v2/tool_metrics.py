from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentToolMetrics:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **record: Any) -> None:
        self.records.append(dict(record))

    def summary(self) -> dict[str, Any]:
        rows = self.records
        latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
        result = {
            "tool_execution_count": len(rows),
            "success_count": sum(row.get("status") == "success" for row in rows),
            "no_result_count": sum(row.get("status") == "no_result" for row in rows),
            "rejected_count": sum(row.get("status") == "rejected" for row in rows),
            "failed_count": sum(row.get("status") == "failed" for row in rows),
            "p50_tool_latency_ms": _percentile(latencies, 0.50),
            "p95_tool_latency_ms": _percentile(latencies, 0.95),
        }
        for action in ("hybrid_search","structure_search","graph_search","rewrite_query","inspect_evidence","finish","abstain"):
            result[f"{action}_count"] = sum(row.get("action") == action for row in rows)
        result["graph_candidate_count"] = sum(int(row.get("candidate_count") or 0) for row in rows if row.get("action") == "graph_search")
        return result


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return values[round((len(values) - 1) * q)]
