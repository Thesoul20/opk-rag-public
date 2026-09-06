from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract, stable_digest
from opk_rag.agentic_v2.termination import AgentTermination

AGENTIC_V2_STATE_TRANSITION_TRACE_VERSION = "opk-rag.agentic-v2.state-transition-trace.v1"


class AgentStateTransitionTrace(StrictContract):
    contract_version: Literal[AGENTIC_V2_STATE_TRANSITION_TRACE_VERSION] = AGENTIC_V2_STATE_TRANSITION_TRACE_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    execution_id: str = Field(min_length=1, max_length=128)
    action: ActionName
    tool_status: str = Field(min_length=1, max_length=32)
    previous_state_digest: str = Field(min_length=64, max_length=64)
    next_state_digest: str = Field(min_length=64, max_length=64)
    next_observation_digest: str = Field(min_length=64, max_length=64)
    query_digest_before: str = Field(min_length=64, max_length=64)
    query_digest_after: str = Field(min_length=64, max_length=64)
    query_length_before: int = Field(ge=0)
    query_length_after: int = Field(ge=0)
    step_index_before: int = Field(ge=0, le=3)
    step_index_after: int = Field(ge=0, le=3)
    retrieval_calls_before: int = Field(ge=0, le=2)
    retrieval_calls_after: int = Field(ge=0, le=2)
    structure_calls_before: int = Field(ge=0, le=2)
    structure_calls_after: int = Field(ge=0, le=2)
    rewrite_count_before: int = Field(ge=0, le=1)
    rewrite_count_after: int = Field(ge=0, le=1)
    graph_calls_before: int = Field(ge=0, le=1)
    graph_calls_after: int = Field(ge=0, le=1)
    graph_hops_before: int = Field(ge=0, le=1)
    graph_hops_after: int = Field(ge=0, le=1)
    candidate_count_before: int = Field(ge=0)
    candidate_count_after: int = Field(ge=0)
    evidence_count_before: int = Field(ge=0)
    evidence_count_after: int = Field(ge=0)
    status_before: str = Field(min_length=1, max_length=32)
    status_after: str = Field(min_length=1, max_length=32)
    transition_status: str = Field(min_length=1, max_length=32)
    transition_reason_code: str = Field(min_length=1, max_length=96)
    termination: dict[str, Any] | None = None
    latency_ms: float = Field(ge=0.0)


def query_digest(query: str) -> str:
    return stable_digest(query)


def public_termination(termination: AgentTermination | None) -> dict[str, Any] | None:
    return termination.model_dump(mode="json") if termination is not None else None
