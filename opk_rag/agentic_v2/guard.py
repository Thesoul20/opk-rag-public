from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from opk_rag.agentic_v2.action import AgentAction, validate_agent_action
from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.budget import AgentBudget
from opk_rag.agentic_v2.decision import AgentDecision
from opk_rag.agentic_v2.guard_contracts import AgentGuardDecision
from opk_rag.agentic_v2.guard_metrics import AgentGuardMetrics
from opk_rag.agentic_v2.guard_trace import AgentGuardTrace, public_action, public_decision, public_termination
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.termination import AgentTermination

MAX_RUNTIME_HYBRID_TOP_K = 10


@dataclass(frozen=True)
class GuardValidationResult:
    guard_decision: AgentGuardDecision
    termination: AgentTermination | None
    trace: AgentGuardTrace

    @property
    def validated_action(self) -> AgentAction | None:
        return self.guard_decision.validated_action


class AgentGuard:
    """Deterministic proposal-to-authorization boundary. It never executes tools or mutates state."""

    def __init__(self, *, resolver: AllowedActionResolver | None = None) -> None:
        self.resolver = resolver or AllowedActionResolver()
        self.metrics = AgentGuardMetrics()

    def validate(
        self,
        *,
        state: AgentState,
        observation: AgentObservation,
        decision: AgentDecision,
        budget: AgentBudget | None = None,
        action_history: tuple[AgentAction, ...] = (),
    ) -> GuardValidationResult:
        started = perf_counter()
        budget = budget or AgentBudget()
        try:
            return self._validate(
                state=state,
                observation=observation,
                decision=decision,
                budget=budget,
                action_history=action_history,
                started=started,
            )
        except Exception:
            # Guard failure must never increase authority.
            return self._result(
                state=state, observation=observation, decision=decision, budget=budget,
                allowed_actions=(), outcome="reject", reason_code="guard_internal_error",
                action=None, termination=None, modifications={}, started=started,
            )

    def _validate(
        self,
        *,
        state: AgentState,
        observation: AgentObservation,
        decision: AgentDecision,
        budget: AgentBudget,
        action_history: tuple[AgentAction, ...],
        started: float,
    ) -> GuardValidationResult:
        if state.run_id != observation.run_id or state.current_query != observation.current_query:
            return self._result(state, observation, decision, budget, (), "reject", "state_observation_mismatch", None, None, {}, started)

        if state.status != "running":
            termination = _terminal_state_termination(state)
            return self._result(state, observation, decision, budget, (), "terminate", "agent_already_terminal", None, termination, {}, started)

        allowed = self.resolver.resolve(state=state, observation=observation, budget=budget, action_history=action_history)
        action_name = decision.proposed_action
        remaining = state.remaining_budget(budget)

        # A safe abstain proposal remains available even when ordinary step budget is exhausted.
        if remaining.steps <= 0 and action_name != "abstain":
            return self._result(state, observation, decision, budget, allowed, "terminate", "agent_step_budget_exhausted", None,
                                AgentTermination(kind="budget_exhausted", reason_code="agent_step_budget_exhausted", is_runtime_failure=False), {}, started)

        if action_name not in allowed:
            reason = _not_allowed_reason(state, decision, remaining, action_history)
            return self._result(state, observation, decision, budget, allowed, "reject", reason, None, None, {}, started)

        args = decision.root.arguments
        if action_name in {"hybrid_search", "structure_search", "graph_search"}:
            if _normalize_query(args.query) != _normalize_query(state.current_query):
                return self._result(state, observation, decision, budget, allowed, "reject", "unauthorized_query_change", None, None, {}, started)
            if remaining.retrieval_calls <= 0:
                return self._result(state, observation, decision, budget, allowed, "reject", "retrieval_budget_exhausted", None, None, {}, started)

        if action_name == "graph_search":
            if state.graph_call_count >= budget.max_graph_calls:
                return self._result(state, observation, decision, budget, allowed, "reject", "graph_call_budget_exhausted", None, None, {}, started)
            if state.graph_hop_count >= budget.max_graph_hops or remaining.graph_hops <= 0:
                return self._result(state, observation, decision, budget, allowed, "reject", "graph_hop_budget_exhausted", None, None, {}, started)
            if "graph_search" in state.previous_actions or any(a.action == "graph_search" for a in action_history):
                return self._result(state, observation, decision, budget, allowed, "reject", "graph_recovery_already_used", None, None, {}, started)

        if action_name == "rewrite_query":
            if state.rewrite_count >= budget.max_query_rewrites or remaining.query_rewrites <= 0:
                return self._result(state, observation, decision, budget, allowed, "reject", "query_rewrite_budget_exhausted", None, None, {}, started)
            if _normalize_query(args.query) == _normalize_query(state.current_query):
                return self._result(state, observation, decision, budget, allowed, "reject", "rewrite_did_not_change_query", None, None, {}, started)

        if action_name == "hybrid_search" and _is_duplicate_retrieval(action_name, args.query, state, action_history):
            return self._result(state, observation, decision, budget, allowed, "reject", "duplicate_retrieval_without_query_change", None, None, {}, started)
        if action_name == "structure_search" and _is_duplicate_retrieval(action_name, args.query, state, action_history):
            return self._result(state, observation, decision, budget, allowed, "reject", "duplicate_structure_recovery", None, None, {}, started)

        if action_name == "finish":
            if observation.evidence_count <= 0 or observation.answerability_status not in {"answerable", "partially_answerable"}:
                return self._result(state, observation, decision, budget, allowed, "reject", "finish_not_authorized_by_answerability", None, None, {}, started)

        action = _decision_to_action(decision)
        modifications: dict[str, Any] = {}
        outcome = "allow"
        reason_code = "allowed"
        if action_name == "hybrid_search" and action.arguments.top_k > MAX_RUNTIME_HYBRID_TOP_K:
            original = action.arguments.top_k
            payload = action.model_dump(mode="json")
            payload["arguments"]["top_k"] = MAX_RUNTIME_HYBRID_TOP_K
            action = validate_agent_action(payload)
            modifications = {"top_k": {"from": original, "to": MAX_RUNTIME_HYBRID_TOP_K}}
            outcome = "modify"
            reason_code = "hybrid_top_k_clamped"

        return self._result(state, observation, decision, budget, allowed, outcome, reason_code, action, None, modifications, started)

    def _result(
        self,
        state: AgentState,
        observation: AgentObservation,
        decision: AgentDecision,
        budget: AgentBudget,
        allowed_actions: tuple[str, ...],
        outcome: str,
        reason_code: str,
        action: AgentAction | None,
        termination: AgentTermination | None,
        modifications: dict[str, Any],
        started: float,
    ) -> GuardValidationResult:
        guard_decision = AgentGuardDecision(decision=outcome, reason_code=reason_code, validated_action=action)
        latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        decision_digest = stable_digest(decision)
        trace = AgentGuardTrace(
            run_id=state.run_id,
            decision_id=decision_digest[:32],
            state_digest=stable_digest(state),
            observation_digest=stable_digest(observation),
            decision_digest=decision_digest,
            allowed_actions=tuple(allowed_actions),
            budget_snapshot=budget.model_dump(mode="json"),
            guard_outcome=outcome,
            guard_reason_code=reason_code,
            original_decision=public_decision(decision),
            validated_action=public_action(action),
            parameter_modifications=modifications,
            termination=public_termination(termination),
            latency_ms=latency_ms,
        )
        self.metrics.record(outcome=outcome, reason_code=reason_code, latency_ms=latency_ms)
        return GuardValidationResult(guard_decision=guard_decision, termination=termination, trace=trace)


def _decision_to_action(decision: AgentDecision) -> AgentAction:
    payload = decision.model_dump(mode="json")
    return validate_agent_action({
        "contract_version": "opk-rag.agentic-v2.action.v1",
        "action": payload["proposed_action"],
        "arguments": payload["arguments"],
        "reason_code": payload["reason_code"],
    })


def _normalize_query(value: str) -> str:
    return " ".join(value.split()).strip()


def _is_duplicate_retrieval(action_name: str, query: str, state: AgentState, history: tuple[AgentAction, ...]) -> bool:
    normalized = _normalize_query(query)
    if history:
        last = history[-1]
        if last.action == action_name and hasattr(last.arguments, "query") and _normalize_query(last.arguments.query) == normalized:
            return True
    # Coarse fallback when only frozen State action names are available.
    return bool(state.previous_actions and state.previous_actions[-1] == action_name and _normalize_query(state.current_query) == normalized)


def _not_allowed_reason(state: AgentState, decision: AgentDecision, remaining, history: tuple[AgentAction, ...]) -> str:
    action = decision.proposed_action
    if action in {"hybrid_search", "structure_search", "graph_search"} and remaining.retrieval_calls <= 0:
        return "retrieval_budget_exhausted"
    if action == "rewrite_query" and remaining.query_rewrites <= 0:
        return "query_rewrite_budget_exhausted"
    if action == "graph_search":
        if state.graph_call_count >= 1:
            return "graph_call_budget_exhausted"
        if state.graph_hop_count >= 1 or remaining.graph_hops <= 0:
            return "graph_hop_budget_exhausted"
        if "graph_search" in state.previous_actions or any(a.action == "graph_search" for a in history):
            return "graph_recovery_already_used"
    if action == "finish":
        return "finish_not_authorized_by_answerability"
    return "action_not_currently_allowed"


def _terminal_state_termination(state: AgentState) -> AgentTermination:
    if state.status == "finished":
        return AgentTermination(kind="finished_answer", reason_code="agent_already_terminal", is_runtime_failure=False)
    if state.status == "abstained":
        return AgentTermination(kind="abstained", reason_code="agent_already_terminal", is_runtime_failure=False)
    return AgentTermination(kind="runtime_failure", reason_code="agent_already_terminal", is_runtime_failure=True)
