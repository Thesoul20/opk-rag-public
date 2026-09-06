from __future__ import annotations

from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.termination import AgentTermination

AGENTIC_V2_STATE_TRANSITION_RESULT_VERSION = "opk-rag.agentic-v2.state-transition-result.v1"
AGENTIC_V2_TRANSITION_OBSERVATION_SIGNALS_VERSION = "opk-rag.agentic-v2.transition-observation-signals.v1"
AGENTIC_V2_APPLIED_EXECUTION_REGISTRY_VERSION = "opk-rag.agentic-v2.applied-execution-registry.v1"
TransitionStatus = Literal["applied", "rejected", "terminated"]


class AgentTransitionObservationSignals(StrictContract):
    """Optional governed runtime signals not represented by AgentToolResult V1.

    All fields are execution-bound. Missing fields remain unavailable; the transition
    runtime never invents rerank, coverage, answerability, source, structure, or graph facts.
    """

    contract_version: Literal[AGENTIC_V2_TRANSITION_OBSERVATION_SIGNALS_VERSION] = AGENTIC_V2_TRANSITION_OBSERVATION_SIGNALS_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    execution_id: str = Field(min_length=1, max_length=128)
    action: ActionName
    source_count: int | None = Field(default=None, ge=0)
    top_rerank_score: float | None = None
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    answerability_status: Literal["unknown", "answerable", "partially_answerable", "insufficient_evidence", "unanswerable"] | None = None
    structure_context_available: bool | None = None
    graph_relation_available: bool | None = None


class AppliedExecutionRegistry(StrictContract):
    contract_version: Literal[AGENTIC_V2_APPLIED_EXECUTION_REGISTRY_VERSION] = AGENTIC_V2_APPLIED_EXECUTION_REGISTRY_VERSION
    execution_ids: tuple[str, ...] = ()

    def contains(self, execution_id: str) -> bool:
        return execution_id in self.execution_ids

    def with_applied(self, execution_id: str) -> "AppliedExecutionRegistry":
        if execution_id in self.execution_ids:
            return self
        return self.model_copy(update={"execution_ids": (*self.execution_ids, execution_id)})


class AgentStateTransitionResult(StrictContract):
    contract_version: Literal[AGENTIC_V2_STATE_TRANSITION_RESULT_VERSION] = AGENTIC_V2_STATE_TRANSITION_RESULT_VERSION
    previous_state_digest: str = Field(min_length=64, max_length=64)
    next_state: AgentState
    next_state_digest: str = Field(min_length=64, max_length=64)
    execution_id: str = Field(min_length=1, max_length=128)
    action: ActionName
    transition_status: TransitionStatus
    reason_code: str = Field(min_length=1, max_length=96)
    termination: AgentTermination | None = None
    next_observation: AgentObservation
    next_observation_digest: str = Field(min_length=64, max_length=64)
    allowed_actions: tuple[ActionName, ...] = ()
    applied_execution_registry: AppliedExecutionRegistry
