from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.base import StrictContract, stable_digest
from opk_rag.agentic_v2.decision import AgentDecision
from opk_rag.agentic_v2.termination import AgentTermination

AGENTIC_V2_GUARD_TRACE_VERSION = "opk-rag.agentic-v2.guard-trace.v1"


class AgentGuardTrace(StrictContract):
    contract_version: Literal[AGENTIC_V2_GUARD_TRACE_VERSION] = AGENTIC_V2_GUARD_TRACE_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    decision_id: str = Field(min_length=1, max_length=128)
    state_digest: str
    observation_digest: str
    decision_digest: str
    allowed_actions: tuple[str, ...]
    budget_snapshot: dict[str, Any]
    guard_outcome: Literal["allow", "modify", "reject", "terminate"]
    guard_reason_code: str
    original_decision: dict[str, Any]
    validated_action: dict[str, Any] | None = None
    parameter_modifications: dict[str, Any]
    termination: dict[str, Any] | None = None
    latency_ms: float = Field(ge=0)


def public_decision(decision: AgentDecision) -> dict[str, Any]:
    return _redact_argument_text(decision.model_dump(mode="json"))


def public_action(action: AgentAction | None) -> dict[str, Any] | None:
    return None if action is None else _redact_argument_text(action.model_dump(mode="json"))


def _redact_argument_text(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"query", "focus"} and isinstance(child, str):
                redacted[key] = {"digest": stable_digest(child), "length": len(child)}
            else:
                redacted[key] = _redact_argument_text(child)
        return redacted
    if isinstance(value, list):
        return [_redact_argument_text(child) for child in value]
    return value


def public_termination(termination: AgentTermination | None) -> dict[str, Any] | None:
    return None if termination is None else termination.model_dump(mode="json")
