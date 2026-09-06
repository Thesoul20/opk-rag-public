from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.budget import AgentBudget, RemainingBudget

AGENTIC_V2_STATE_CONTRACT_VERSION = "opk-rag.agentic-v2.state.v1"
AgentStateStatus = Literal["running", "finished", "abstained", "failed"]


class AgentState(StrictContract):
    contract_version: Literal[AGENTIC_V2_STATE_CONTRACT_VERSION] = AGENTIC_V2_STATE_CONTRACT_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    original_query: str = Field(min_length=1, max_length=512)
    current_query: str = Field(min_length=1, max_length=512)
    step_index: int = Field(default=0, ge=0, le=3)
    previous_actions: tuple[ActionName, ...] = ()
    retrieval_call_count: int = Field(default=0, ge=0, le=2)
    structure_call_count: int = Field(default=0, ge=0, le=2)
    graph_call_count: int = Field(default=0, ge=0, le=1)
    rewrite_count: int = Field(default=0, ge=0, le=1)
    graph_hop_count: int = Field(default=0, ge=0, le=1)
    candidate_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    top_rerank_score: float | None = None
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    answerability_status: Literal["unknown", "answerable", "partially_answerable", "insufficient_evidence", "unanswerable"] = "unknown"
    status: AgentStateStatus = "running"
    termination_reason: str | None = Field(default=None, max_length=160)

    @computed_field
    @property
    def remaining_step_budget(self) -> int:
        return max(0, 3 - self.step_index)

    @computed_field
    @property
    def remaining_retrieval_budget(self) -> int:
        return max(0, 2 - self.retrieval_call_count)

    @computed_field
    @property
    def remaining_rewrite_budget(self) -> int:
        return max(0, 1 - self.rewrite_count)

    @computed_field
    @property
    def remaining_graph_budget(self) -> int:
        return max(0, 1 - self.graph_call_count)

    @computed_field
    @property
    def remaining_graph_hop_budget(self) -> int:
        return max(0, 1 - self.graph_hop_count)

    def remaining_budget(self, authority: AgentBudget | None = None) -> RemainingBudget:
        authority = authority or AgentBudget()
        return RemainingBudget(
            steps=max(0, authority.max_steps - self.step_index),
            retrieval_calls=max(0, authority.max_retrieval_calls - self.retrieval_call_count),
            query_rewrites=max(0, authority.max_query_rewrites - self.rewrite_count),
            graph_calls=max(0, authority.max_graph_calls - self.graph_call_count),
            graph_hops=max(0, authority.max_graph_hops - self.graph_hop_count),
        )
