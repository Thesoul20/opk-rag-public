from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.decision import AgentDecision

AGENTIC_V2_POLICY_TRACE_VERSION = "opk-rag.agentic-v2.policy-trace.v1"


class AgentPolicyTrace(StrictContract):
    contract_version: Literal[AGENTIC_V2_POLICY_TRACE_VERSION] = AGENTIC_V2_POLICY_TRACE_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    decision_id: str = Field(min_length=1, max_length=128)
    policy_version: str
    provider_name: str
    model_name: str
    prompt_version: str
    prompt_digest: str
    observation_digest: str
    allowed_actions: tuple[str, ...]
    decision: dict[str, Any] | None = None
    validation_status: Literal["passed", "failed"]
    repair_attempt_count: int = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    failure_code: str | None = None


def public_decision(decision: AgentDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return decision.model_dump(mode="json")
