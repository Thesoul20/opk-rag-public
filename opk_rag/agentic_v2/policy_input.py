from __future__ import annotations

from typing import Literal

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.budget import RemainingBudget
from opk_rag.agentic_v2.observation import AgentObservation

AGENTIC_V2_POLICY_INPUT_CONTRACT_VERSION = "opk-rag.agentic-v2.policy-input.v1"
AGENTIC_V2_ACTION_SPACE: tuple[ActionName, ...] = (
    "hybrid_search",
    "structure_search",
    "graph_search",
    "rewrite_query",
    "inspect_evidence",
    "finish",
    "abstain",
)


class AgentPolicyInput(StrictContract):
    contract_version: Literal[AGENTIC_V2_POLICY_INPUT_CONTRACT_VERSION] = AGENTIC_V2_POLICY_INPUT_CONTRACT_VERSION
    run_id: str
    current_query: str
    candidate_count: int
    evidence_count: int
    source_count: int
    top_rerank_score: float | None = None
    evidence_coverage: float | None = None
    answerability_status: str
    structure_context_available: bool
    graph_relation_available: bool
    previous_action: ActionName | None = None
    previous_action_status: Literal["not_run", "ok", "rejected", "failed"]
    remaining_budget: RemainingBudget
    allowed_actions: tuple[ActionName, ...] = AGENTIC_V2_ACTION_SPACE


def build_policy_input(
    observation: AgentObservation,
    *,
    allowed_actions: tuple[ActionName, ...] | None = None,
) -> AgentPolicyInput:
    """Build model-visible input from Observation, optionally using Guard-derived actions."""
    return AgentPolicyInput(
        run_id=observation.run_id,
        current_query=observation.current_query,
        candidate_count=observation.candidate_count,
        evidence_count=observation.evidence_count,
        source_count=observation.source_count,
        top_rerank_score=observation.top_rerank_score,
        evidence_coverage=observation.evidence_coverage,
        answerability_status=observation.answerability_status,
        structure_context_available=observation.structure_context_available,
        graph_relation_available=observation.graph_relation_available,
        previous_action=observation.previous_action,
        previous_action_status=observation.previous_action_status,
        remaining_budget=observation.remaining_budget,
        allowed_actions=allowed_actions if allowed_actions is not None else AGENTIC_V2_ACTION_SPACE,
    )
