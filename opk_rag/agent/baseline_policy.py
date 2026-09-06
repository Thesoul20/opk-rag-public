from __future__ import annotations

from opk_rag.agent.contracts import AgentAction, AgentRuntimeConfig, AgentState
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.policy import AgentPolicy
from opk_rag.agent.transition import resolve_available_actions


class DeterministicBaselinePolicy(AgentPolicy):
    name = "deterministic_baseline_v1"
    version = "1.0.0"

    def __init__(self, config: AgentRuntimeConfig | None = None) -> None:
        self.config = config or AgentRuntimeConfig()

    def next_action(self, state: AgentState) -> AgentAction:
        if state.is_terminal:
            raise AgentRuntimeError(
                "agent_policy_failure",
                "Policy was asked for an action after terminal state.",
                origin="agent_policy",
            )
        available = resolve_available_actions(state, self.config)
        if not available:
            raise AgentRuntimeError("agent_policy_failure", "No available action for non-terminal state.", origin="agent_policy")
        action = available[0]
        if action == "search":
            return AgentAction(
                action="search",
                reason_code="initial_retrieval",
                arguments={"query": state.question, "top_k": self.config.search_top_k},
                expected_state_transition="evidence_available",
            )
        if action == "evaluate_answerability":
            return AgentAction(
                action="evaluate_answerability",
                reason_code="initial_answerability_check",
                expected_state_transition="answerability_decided",
            )
        if action == "generate_grounded_answer":
            return AgentAction(
                action="generate_grounded_answer",
                reason_code="answerability_permits_generation",
                expected_state_transition="generation_available",
            )
        if action == "verify_grounding":
            return AgentAction(
                action="verify_grounding",
                reason_code="post_generation_grounding_required",
                expected_state_transition="grounding_verified",
            )
        if action == "finish_answer":
            return AgentAction(action="finish_answer", reason_code="grounding_verified", expected_state_transition="terminal_answer")
        if action == "expand_evidence":
            chunk_ids = state.evidence_state.get("chunk_ids") or []
            return AgentAction(
                action="expand_evidence",
                reason_code="insufficient_evidence_expand_once",
                arguments={"chunk_id": chunk_ids[0] if chunk_ids else None, "max_items": self.config.expansion_max_items},
                expected_state_transition="evidence_expanded",
            )
        if action == "finish_failure":
            return AgentAction(action="finish_failure", reason_code="execution_budget_exhausted", expected_state_transition="terminal_failure")
        status = state.answerability_state.get("status")
        reason = "grounding_rejection" if state.generation_state.get("attempted") else (status or "unanswerable")
        return AgentAction(action="finish_abstain", reason_code=reason, expected_state_transition="terminal_abstain")
