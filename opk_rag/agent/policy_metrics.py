from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any


@dataclass
class PolicyCallRecord:
    provider_name: str
    model_name: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    repair_attempt: bool
    status: str
    failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "repair_attempt": self.repair_attempt,
            "status": self.status,
            "failure_code": self.failure_code,
        }


@dataclass
class PolicyMetricsRecorder:
    calls: list[PolicyCallRecord] = field(default_factory=list)
    repair_call_count: int = 0
    repair_success_count: int = 0

    def record(self, record: PolicyCallRecord) -> None:
        self.calls.append(record)
        if record.repair_attempt:
            self.repair_call_count += 1
            if record.status == "success":
                self.repair_success_count += 1

    def summary(self) -> dict[str, Any]:
        failures = [call for call in self.calls if call.status != "success"]
        latencies = sorted(call.latency_ms for call in self.calls)
        return {
            "policy_call_count": len(self.calls),
            "first_pass_valid_count": sum(1 for call in self.calls if call.status == "success" and not call.repair_attempt),
            "repair_call_count": self.repair_call_count,
            "repair_success_count": self.repair_success_count,
            "repair_success_rate": self.repair_success_count / self.repair_call_count if self.repair_call_count else None,
            "provider_failure_count": sum(1 for call in failures if call.failure_code == "agent_policy_provider_failure"),
            "timeout_count": sum(1 for call in failures if call.failure_code == "agent_policy_timeout"),
            "contract_failure_count": sum(1 for call in failures if call.failure_code == "agent_policy_contract_failure"),
            "malformed_json_count": sum(1 for call in failures if call.failure_code == "agent_policy_malformed_json"),
            "unknown_action_count": sum(1 for call in failures if call.failure_code == "agent_policy_unknown_action"),
            "illegal_transition_count": sum(1 for call in failures if call.failure_code == "agent_policy_illegal_transition"),
            "argument_failure_count": sum(1 for call in failures if call.failure_code == "agent_policy_argument_failure"),
            "total_input_tokens": sum(call.input_tokens or 0 for call in self.calls),
            "total_output_tokens": sum(call.output_tokens or 0 for call in self.calls),
            "token_source": "provider_reported" if any(call.input_tokens is not None or call.output_tokens is not None for call in self.calls) else "unavailable",
            "total_latency_ms": sum(call.latency_ms for call in self.calls),
            "average_latency_ms": sum(call.latency_ms for call in self.calls) / len(self.calls) if self.calls else 0,
            "p50_latency_ms": median(latencies) if latencies else 0,
            "p95_latency_ms": _percentile(latencies, 0.95) if latencies else 0,
        }


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * percentile))
    return sorted_values[index]
