from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.errors import AgentFailure

AGENT_RECOVERY_STATE_CONTRACT_VERSION = "opk-rag.agent-recovery-state.v1"

RecoveryStateName = Literal[
    "START",
    "INITIAL_RETRIEVAL",
    "INITIAL_EVIDENCE_INSPECTION",
    "RECOVERY_ELIGIBILITY",
    "QUERY_REFORMULATION",
    "RECOVERY_RETRIEVAL",
    "EVIDENCE_COMPARISON",
    "FINAL_ANSWERABILITY",
    "GENERATION",
    "GENERATION_RETRY_ELIGIBILITY",
    "GENERATION_RETRY",
    "RETRY_CITATION_VALIDATION",
    "RETRY_GROUNDING_VALIDATION",
    "CITATION_VALIDATION",
    "GROUNDING_VALIDATION",
    "FINAL_ANSWER",
    "FINAL_ABSTENTION",
    "TERMINAL_ERROR",
]

FinalRecoveryAction = Literal["answer", "abstain", "error"]


@dataclass(frozen=True)
class RecoveryEligibilityDecision:
    eligible: bool
    recovery_reason: str
    terminal_safety: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "recovery_reason": self.recovery_reason,
            "terminal_safety": self.terminal_safety,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AgentRecoveryState:
    run_id: str
    original_query: str
    active_query: str
    sample_id: str | None = None
    state_name: RecoveryStateName = "START"
    transition_index: int = 0
    retrieval_attempt_count: int = 0
    reformulation_attempt_count: int = 0
    generation_attempt_count: int = 0
    generation_retry_count: int = 0
    generation_retry_eligibility_evaluated: bool = False
    generation_retry_eligible: bool = False
    generation_retry_decision_reason: str | None = None
    generation_retry_rule_id: str | None = None
    generation_retry_started: bool = False
    generation_retry_completed: bool = False
    generation_retry_outcome: str | None = None
    generation_request_identity_digest_attempt_1: str | None = None
    generation_request_identity_digest_attempt_2: str | None = None
    tool_call_count: int = 0
    initial_retrieval_summary: dict[str, Any] | None = None
    recovery_retrieval_summary: dict[str, Any] | None = None
    evidence_comparison: dict[str, Any] | None = None
    answerability_decision: dict[str, Any] | None = None
    recovery_eligibility: dict[str, Any] | None = None
    recovery_reason: str | None = None
    reformulated_query: str | None = None
    generation_result: dict[str, Any] | None = None
    citation_result: dict[str, Any] | None = None
    grounding_result: dict[str, Any] | None = None
    final_action: FinalRecoveryAction | None = None
    termination_reason: str | None = None
    error_type: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    contract_version: str = AGENT_RECOVERY_STATE_CONTRACT_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.state_name in {"FINAL_ANSWER", "FINAL_ABSTENTION", "TERMINAL_ERROR"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "original_query_digest": stable_digest(self.original_query),
            "active_query_digest": stable_digest(self.active_query),
            "original_query_length": len(self.original_query),
            "active_query_length": len(self.active_query),
            "state_name": self.state_name,
            "transition_index": self.transition_index,
            "retrieval_attempt_count": self.retrieval_attempt_count,
            "reformulation_attempt_count": self.reformulation_attempt_count,
            "generation_attempt_count": self.generation_attempt_count,
            "generation_retry_count": self.generation_retry_count,
            "generation_retry_eligibility_evaluated": self.generation_retry_eligibility_evaluated,
            "generation_retry_eligible": self.generation_retry_eligible,
            "generation_retry_decision_reason": self.generation_retry_decision_reason,
            "generation_retry_rule_id": self.generation_retry_rule_id,
            "generation_retry_started": self.generation_retry_started,
            "generation_retry_completed": self.generation_retry_completed,
            "generation_retry_outcome": self.generation_retry_outcome,
            "generation_request_identity_digest_attempt_1": self.generation_request_identity_digest_attempt_1,
            "generation_request_identity_digest_attempt_2": self.generation_request_identity_digest_attempt_2,
            "tool_call_count": self.tool_call_count,
            "initial_retrieval_summary": self.initial_retrieval_summary,
            "recovery_retrieval_summary": self.recovery_retrieval_summary,
            "evidence_comparison": self.evidence_comparison,
            "answerability_decision": self.answerability_decision,
            "recovery_eligibility": self.recovery_eligibility,
            "recovery_reason": self.recovery_reason,
            "reformulated_query_digest": None if self.reformulated_query is None else stable_digest(self.reformulated_query),
            "reformulated_query_length": None if self.reformulated_query is None else len(self.reformulated_query),
            "generation_result": self.generation_result,
            "citation_result": self.citation_result,
            "grounding_result": self.grounding_result,
            "final_action": self.final_action,
            "termination_reason": self.termination_reason,
            "error_type": self.error_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @property
    def identity_digest(self) -> str:
        return stable_digest(self.to_dict())


def transition_state(state: AgentRecoveryState, to_state: RecoveryStateName, **updates: Any) -> AgentRecoveryState:
    return replace(state, state_name=to_state, transition_index=state.transition_index + 1, **updates)


def failure_to_state(state: AgentRecoveryState, failure: AgentFailure, *, completed_at: str) -> AgentRecoveryState:
    return transition_state(
        state,
        "TERMINAL_ERROR",
        final_action="error",
        termination_reason=failure.code,
        error_type=failure.origin,
        completed_at=completed_at,
    )
