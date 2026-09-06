from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class AgentPolicyMetrics:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **record: Any) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        records = self.records
        latencies = sorted(int(row.get("latency_ms") or 0) for row in records)
        inputs = [int(x) for x in (row.get("input_tokens") for row in records) if x is not None]
        outputs = [int(x) for x in (row.get("output_tokens") for row in records) if x is not None]
        return {
            "decision_count": sum(1 for row in records if row.get("status") == "success"),
            "valid_first_attempt_count": sum(1 for row in records if row.get("status") == "success" and not row.get("repair_attempt")),
            "repair_count": sum(1 for row in records if row.get("repair_attempt")),
            "repair_success_count": sum(1 for row in records if row.get("status") == "success" and row.get("repair_attempt")),
            "invalid_output_count": sum(1 for row in records if row.get("status") == "failure" and str(row.get("failure_code", "")).startswith("policy_")),
            "unknown_action_count": sum(1 for row in records if row.get("failure_code") == "policy_schema_validation_failed" and row.get("failure_detail") == "unknown_action"),
            "forbidden_argument_count": sum(1 for row in records if row.get("failure_code") == "policy_schema_validation_failed" and row.get("failure_detail") == "forbidden_argument"),
            "provider_failure_count": sum(1 for row in records if row.get("failure_code") == "agent_policy_provider_failure"),
            "timeout_count": sum(1 for row in records if row.get("failure_code") == "agent_policy_timeout"),
            "p50_policy_latency_ms": _percentile(latencies, 0.50),
            "p95_policy_latency_ms": _percentile(latencies, 0.95),
            "average_input_tokens": mean(inputs) if inputs else None,
            "average_output_tokens": mean(outputs) if outputs else None,
        }


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    index = round((len(values) - 1) * q)
    return values[index]
