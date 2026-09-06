from __future__ import annotations

from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.budget import AgentBudget, RemainingBudget
from opk_rag.agentic_v2.state import AgentState

AGENTIC_V2_OBSERVATION_CONTRACT_VERSION = "opk-rag.agentic-v2.observation.v1"


class AgentObservation(StrictContract):
    contract_version: Literal[AGENTIC_V2_OBSERVATION_CONTRACT_VERSION] = AGENTIC_V2_OBSERVATION_CONTRACT_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    current_query: str = Field(min_length=1, max_length=512)
    candidate_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    top_rerank_score: float | None = None
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    answerability_status: str = Field(max_length=64)
    structure_context_available: bool = False
    graph_relation_available: bool = False
    previous_action: ActionName | None = None
    previous_action_status: Literal["not_run", "ok", "rejected", "failed"] = "not_run"
    remaining_budget: RemainingBudget


def build_agent_observation(
    state: AgentState,
    *,
    authority: AgentBudget | None = None,
    structure_context_available: bool = False,
    graph_relation_available: bool = False,
    previous_action_status: Literal["not_run", "ok", "rejected", "failed"] = "not_run",
) -> AgentObservation:
    return AgentObservation(
        run_id=state.run_id,
        current_query=state.current_query,
        candidate_count=state.candidate_count,
        evidence_count=state.evidence_count,
        source_count=state.source_count,
        top_rerank_score=state.top_rerank_score,
        evidence_coverage=state.evidence_coverage,
        answerability_status=state.answerability_status,
        structure_context_available=structure_context_available,
        graph_relation_available=graph_relation_available,
        previous_action=state.previous_actions[-1] if state.previous_actions else None,
        previous_action_status=previous_action_status,
        remaining_budget=state.remaining_budget(authority),
    )
