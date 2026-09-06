from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentErrorCode = Literal[
    "agent_policy_failure",
    "agent_policy_provider_failure",
    "agent_policy_timeout",
    "agent_policy_malformed_json",
    "agent_policy_contract_failure",
    "agent_policy_unknown_action",
    "agent_policy_illegal_transition",
    "agent_policy_argument_failure",
    "agent_policy_repair_exhausted",
    "agent_policy_prompt_leakage",
    "agent_policy_gold_leakage",
    "agent_policy_budget_violation",
    "agent_policy_prompt_injection_attempt",
    "agent_invalid_action",
    "agent_state_transition_failure",
    "agent_budget_exhausted",
    "agent_unknown_tool",
    "agent_tool_input_contract_failure",
    "core_tool_failure",
    "core_tool_timeout",
    "answerability_abstention",
    "grounding_rejection",
    "trace_integrity_failure",
    "infrastructure_failure",
]


@dataclass(frozen=True)
class AgentFailure:
    code: AgentErrorCode
    message: str
    origin: Literal["agent_policy", "agent_runtime", "tool_adapter", "core_rag", "provider", "infrastructure"]
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "origin": self.origin,
            "detail": _stable(self.detail),
        }


class AgentRuntimeError(RuntimeError):
    def __init__(
        self,
        code: AgentErrorCode,
        message: str,
        *,
        origin: Literal["agent_policy", "agent_runtime", "tool_adapter", "core_rag", "provider", "infrastructure"],
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = AgentFailure(code=code, message=message, origin=origin, detail=detail or {})


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(child) for child in value]
    return value
