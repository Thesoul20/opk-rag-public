from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.budget import AgentBudget
from opk_rag.agentic_v2.observation import AgentObservation, build_agent_observation
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.state_transition_contracts import (
    AgentStateTransitionResult,
    AgentTransitionObservationSignals,
    AppliedExecutionRegistry,
)
from opk_rag.agentic_v2.state_transition_metrics import AgentStateTransitionMetrics
from opk_rag.agentic_v2.state_transition_trace import AgentStateTransitionTrace, public_termination, query_digest
from opk_rag.agentic_v2.termination import AgentTermination
from opk_rag.agentic_v2.tool_contracts import AgentToolResult

RETRIEVAL_ACTIONS = {"hybrid_search", "structure_search", "graph_search"}


@dataclass(frozen=True)
class StateTransitionExecution:
    result: AgentStateTransitionResult
    trace: AgentStateTransitionTrace


class AgentStateTransitionEngine:
    """Pure deterministic ToolResult -> AgentState -> AgentObservation authority."""

    def __init__(self, *, resolver: AllowedActionResolver | None = None) -> None:
        self.resolver = resolver or AllowedActionResolver()
        self.metrics = AgentStateTransitionMetrics()

    def transition(
        self,
        *,
        state: AgentState,
        action: AgentAction,
        tool_result: AgentToolResult,
        budget: AgentBudget | None = None,
        applied_execution_registry: AppliedExecutionRegistry | None = None,
        observation_signals: AgentTransitionObservationSignals | None = None,
    ) -> StateTransitionExecution:
        started = perf_counter()
        budget = budget or AgentBudget()
        registry = applied_execution_registry or AppliedExecutionRegistry()
        before_digest = stable_digest(state)
        try:
            return self._transition(
                state=state,
                action=action,
                tool_result=tool_result,
                budget=budget,
                registry=registry,
                observation_signals=observation_signals,
                started=started,
            )
        except Exception:
            # Transition failures never increase authority and never mutate the input state.
            return self._finalize(
                previous_state=state,
                next_state=state,
                action=action,
                tool_result=tool_result,
                budget=budget,
                registry=registry,
                observation_signals=None,
                transition_status="rejected",
                reason_code="state_transition_internal_error",
                termination=None,
                started=started,
                previous_state_digest=before_digest,
                previous_action_status="rejected",
            )

    def _transition(
        self,
        *,
        state: AgentState,
        action: AgentAction,
        tool_result: AgentToolResult,
        budget: AgentBudget,
        registry: AppliedExecutionRegistry,
        observation_signals: AgentTransitionObservationSignals | None,
        started: float,
    ) -> StateTransitionExecution:
        before_digest = stable_digest(state)
        if state.status != "running":
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "terminal_state_transition_rejected", None, started, before_digest, "rejected")
        if tool_result.run_id != state.run_id:
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "run_id_mismatch", None, started, before_digest, "rejected")
        if tool_result.action != action.action:
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "action_tool_result_mismatch", None, started, before_digest, "rejected")
        if registry.contains(tool_result.execution_id):
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "duplicate_execution_transition", None, started, before_digest, "rejected")
        if observation_signals is not None and (
            observation_signals.run_id != state.run_id
            or observation_signals.execution_id != tool_result.execution_id
            or observation_signals.action != action.action
        ):
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "observation_signal_identity_mismatch", None, started, before_digest, "rejected")

        if tool_result.status == "rejected":
            failed_state = state.model_copy(update={"status": "failed", "termination_reason": "tool_execution_rejected"})
            termination = AgentTermination(kind="runtime_failure", reason_code="tool_execution_rejected", is_runtime_failure=True)
            return self._finalize(state, failed_state, action, tool_result, budget, registry.with_applied(tool_result.execution_id), observation_signals, "terminated", "tool_execution_rejected", termination, started, before_digest, "rejected")

        if action.action == "rewrite_query" and tool_result.status == "success":
            if tool_result.rewritten_query is None or tool_result.rewritten_query != action.arguments.query:
                return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "rewrite_result_mismatch", None, started, before_digest, "rejected")

        counters = _next_counters(state, action)
        invariant = _budget_invariant_reason(counters, budget)
        if invariant is not None:
            termination = AgentTermination(kind="budget_exhausted", reason_code=invariant, is_runtime_failure=False)
            exhausted = state.model_copy(update={"status": "failed", "termination_reason": invariant}) if tool_result.status == "failed" else state
            return self._finalize(state, exhausted, action, tool_result, budget, registry, observation_signals, "terminated", invariant, termination, started, before_digest, "failed" if tool_result.status == "failed" else "rejected")

        updates = dict(counters)
        updates["previous_actions"] = (*state.previous_actions, action.action)
        termination: AgentTermination | None = None
        transition_status = "applied"
        reason_code = "transition_applied"

        if action.action in RETRIEVAL_ACTIONS:
            updates["candidate_count"] = tool_result.candidate_count
            updates["evidence_count"] = tool_result.evidence_count
            updates["source_count"] = observation_signals.source_count if observation_signals and observation_signals.source_count is not None else 0
            updates["top_rerank_score"] = observation_signals.top_rerank_score if observation_signals else None
            updates["evidence_coverage"] = observation_signals.evidence_coverage if observation_signals else None
            updates["answerability_status"] = observation_signals.answerability_status if observation_signals and observation_signals.answerability_status is not None else "unknown"
        elif action.action == "inspect_evidence":
            updates["candidate_count"] = tool_result.candidate_count
            updates["evidence_count"] = tool_result.evidence_count
            if observation_signals and observation_signals.source_count is not None:
                updates["source_count"] = observation_signals.source_count
            if observation_signals and observation_signals.top_rerank_score is not None:
                updates["top_rerank_score"] = observation_signals.top_rerank_score
            if observation_signals and observation_signals.evidence_coverage is not None:
                updates["evidence_coverage"] = observation_signals.evidence_coverage
            if observation_signals and observation_signals.answerability_status is not None:
                updates["answerability_status"] = observation_signals.answerability_status
        elif action.action == "rewrite_query" and tool_result.status == "success":
            updates["current_query"] = tool_result.rewritten_query
        elif action.action == "finish" and tool_result.status == "success":
            updates["status"] = "finished"
            updates["termination_reason"] = action.arguments.reason_code
            termination = AgentTermination(kind="finished_answer", reason_code=action.arguments.reason_code, is_runtime_failure=False)
            transition_status = "terminated"
            reason_code = "finish_transition_applied"
        elif action.action == "abstain" and tool_result.status == "success":
            updates["status"] = "abstained"
            updates["termination_reason"] = action.arguments.reason_code
            termination = AgentTermination(kind="abstained", reason_code=action.arguments.reason_code, is_runtime_failure=False)
            transition_status = "terminated"
            reason_code = "abstain_transition_applied"

        if tool_result.status == "failed" and counters["step_index"] >= budget.max_steps:
            updates["status"] = "failed"
            updates["termination_reason"] = "agent_step_budget_exhausted"
            termination = AgentTermination(kind="budget_exhausted", reason_code="agent_step_budget_exhausted", is_runtime_failure=False)
            transition_status = "terminated"
            reason_code = "agent_step_budget_exhausted"

        try:
            next_state = state.model_copy(update=updates)
            # force validation because model_copy does not validate update by default
            next_state = AgentState.model_validate({name: getattr(next_state, name) for name in AgentState.model_fields})
        except ValidationError:
            return self._finalize(state, state, action, tool_result, budget, registry, None, "rejected", "state_transition_internal_error", None, started, before_digest, "rejected")

        previous_action_status = _previous_action_status(tool_result.status)
        return self._finalize(
            state,
            next_state,
            action,
            tool_result,
            budget,
            registry.with_applied(tool_result.execution_id),
            observation_signals,
            transition_status,
            reason_code,
            termination,
            started,
            before_digest,
            previous_action_status,
        )

    def _finalize(
        self,
        previous_state: AgentState,
        next_state: AgentState,
        action: AgentAction,
        tool_result: AgentToolResult,
        budget: AgentBudget,
        registry: AppliedExecutionRegistry,
        observation_signals: AgentTransitionObservationSignals | None,
        transition_status: str,
        reason_code: str,
        termination: AgentTermination | None,
        started: float,
        previous_state_digest: str,
        previous_action_status: str,
    ) -> StateTransitionExecution:
        structure_available = bool(observation_signals.structure_context_available) if observation_signals and observation_signals.structure_context_available is not None else False
        graph_available = bool(observation_signals.graph_relation_available) if observation_signals and observation_signals.graph_relation_available is not None else False
        next_observation = build_agent_observation(
            next_state,
            authority=budget,
            structure_context_available=structure_available,
            graph_relation_available=graph_available,
            previous_action_status=previous_action_status,
        )
        allowed_actions = self.resolver.resolve(state=next_state, observation=next_observation, budget=budget)
        next_state_digest = stable_digest(next_state)
        next_observation_digest = stable_digest(next_observation)
        result = AgentStateTransitionResult(
            previous_state_digest=previous_state_digest,
            next_state=next_state,
            next_state_digest=next_state_digest,
            execution_id=tool_result.execution_id,
            action=action.action,
            transition_status=transition_status,
            reason_code=reason_code,
            termination=termination,
            next_observation=next_observation,
            next_observation_digest=next_observation_digest,
            allowed_actions=allowed_actions,
            applied_execution_registry=registry,
        )
        latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        trace = AgentStateTransitionTrace(
            run_id=previous_state.run_id,
            execution_id=tool_result.execution_id,
            action=action.action,
            tool_status=tool_result.status,
            previous_state_digest=previous_state_digest,
            next_state_digest=next_state_digest,
            next_observation_digest=next_observation_digest,
            query_digest_before=query_digest(previous_state.current_query),
            query_digest_after=query_digest(next_state.current_query),
            query_length_before=len(previous_state.current_query),
            query_length_after=len(next_state.current_query),
            step_index_before=previous_state.step_index,
            step_index_after=next_state.step_index,
            retrieval_calls_before=previous_state.retrieval_call_count,
            retrieval_calls_after=next_state.retrieval_call_count,
            structure_calls_before=previous_state.structure_call_count,
            structure_calls_after=next_state.structure_call_count,
            rewrite_count_before=previous_state.rewrite_count,
            rewrite_count_after=next_state.rewrite_count,
            graph_calls_before=previous_state.graph_call_count,
            graph_calls_after=next_state.graph_call_count,
            graph_hops_before=previous_state.graph_hop_count,
            graph_hops_after=next_state.graph_hop_count,
            candidate_count_before=previous_state.candidate_count,
            candidate_count_after=next_state.candidate_count,
            evidence_count_before=previous_state.evidence_count,
            evidence_count_after=next_state.evidence_count,
            status_before=previous_state.status,
            status_after=next_state.status,
            transition_status=transition_status,
            transition_reason_code=reason_code,
            termination=public_termination(termination),
            latency_ms=latency_ms,
        )
        self.metrics.record(action=action.action, transition_status=transition_status, reason_code=reason_code, tool_status=tool_result.status, latency_ms=latency_ms)
        return StateTransitionExecution(result=result, trace=trace)


def _next_counters(state: AgentState, action: AgentAction) -> dict[str, int]:
    updates = {
        "step_index": state.step_index + 1,
        "retrieval_call_count": state.retrieval_call_count,
        "structure_call_count": state.structure_call_count,
        "graph_call_count": state.graph_call_count,
        "rewrite_count": state.rewrite_count,
        "graph_hop_count": state.graph_hop_count,
    }
    if action.action in RETRIEVAL_ACTIONS:
        updates["retrieval_call_count"] += 1
    if action.action == "structure_search":
        updates["structure_call_count"] += 1
    if action.action == "graph_search":
        updates["graph_call_count"] += 1
        updates["graph_hop_count"] += 1
    if action.action == "rewrite_query":
        updates["rewrite_count"] += 1
    return updates


def _budget_invariant_reason(counters: dict[str, int], budget: AgentBudget) -> str | None:
    if counters["step_index"] > budget.max_steps:
        return "step_budget_invariant_violation"
    if counters["retrieval_call_count"] > budget.max_retrieval_calls:
        return "retrieval_budget_invariant_violation"
    if counters["rewrite_count"] > budget.max_query_rewrites:
        return "rewrite_budget_invariant_violation"
    if counters["graph_call_count"] > budget.max_graph_calls or counters["graph_hop_count"] > budget.max_graph_hops:
        return "graph_budget_invariant_violation"
    return None


def _previous_action_status(tool_status: str) -> str:
    if tool_status in {"success", "no_result"}:
        return "ok"
    if tool_status == "failed":
        return "failed"
    return "rejected"
