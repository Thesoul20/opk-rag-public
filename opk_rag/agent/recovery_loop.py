from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.evidence_comparison import compare_retrieval_evidence, evidence_summary_from_tool
from opk_rag.agent.generation_retry import (
    classify_retry_outcome,
    decide_generation_retry_eligibility,
    generation_request_identity_digest,
    validate_retry_request_identity,
)
from opk_rag.agent.query_reformulation import GovernedQueryReformulator
from opk_rag.agent.recovery_contracts import validate_recovery_transition
from opk_rag.agent.recovery_state import AgentRecoveryState, RecoveryEligibilityDecision, failure_to_state, transition_state
from opk_rag.agent.recovery_trace import AgentRecoveryTraceEvent, serialize_agent_trace
from opk_rag.agent.tool_registry import AgentCoreToolExecutor

AGENT_RECOVERY_LOOP_CONTRACT_VERSION = "opk-rag.governed-agent-recovery-loop-runtime.v1"

ANSWERABLE_STATUSES = {"answerable", "partially_answerable"}
RECOVERABLE_STATUSES = {"insufficient_evidence"}
SAFETY_TERMINAL_STATUSES = {"unanswerable", "false_premise", "disallowed", "forbidden"}


@dataclass(frozen=True)
class AgentRecoveryConfig:
    max_retrieval_calls: int = 2
    max_reformulation_calls: int = 1
    max_generation_calls: int = 1
    generation_retry_enabled: bool = False
    max_generation_retries: int = 1
    max_transitions: int = 14
    max_total_tool_calls: int = 6
    search_top_k: int = 10
    max_reformulated_query_length: int = 256
    contract_version: str = AGENT_RECOVERY_LOOP_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_reformulation_calls": self.max_reformulation_calls,
            "max_generation_calls": self.max_generation_calls,
            "generation_retry_enabled": self.generation_retry_enabled,
            "max_generation_retries": self.max_generation_retries,
            "max_transitions": self.max_transitions,
            "max_total_tool_calls": self.max_total_tool_calls,
            "search_top_k": self.search_top_k,
            "max_reformulated_query_length": self.max_reformulated_query_length,
        }


@dataclass(frozen=True)
class AgentRecoveryRunResult:
    state: AgentRecoveryState
    trace_events: tuple[AgentRecoveryTraceEvent, ...]
    final_answer: str | None = None
    final_refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "opk-rag.governed-agent-recovery-result.v1",
            "state": self.state.to_dict(),
            "trace": serialize_agent_trace(self.trace_events),
            "final_answer_hash": None if self.final_answer is None else stable_digest(self.final_answer),
            "final_refusal_reason": self.final_refusal_reason,
        }


def run_agent_recovery_loop(
    *,
    question: str,
    executor: AgentCoreToolExecutor,
    reformulator: GovernedQueryReformulator,
    sample_id: str | None = None,
    run_id: str | None = None,
    config: AgentRecoveryConfig | None = None,
) -> AgentRecoveryRunResult:
    config = config or AgentRecoveryConfig()
    now = _utc_now()
    state = AgentRecoveryState(run_id=run_id or str(uuid4()), sample_id=sample_id, original_query=question, active_query=question, started_at=now)
    memory: dict[str, Any] = {}
    trace: list[AgentRecoveryTraceEvent] = []
    final_answer: str | None = None
    final_refusal_reason: str | None = None

    def move(to_state: str, *, decision_type: str, decision_reason: str, tool_name: str | None = None, tool_call_index: int | None = None, input_identity: str | None = None, output_identity: str | None = None, status: str = "ok", error_type: str | None = None, started: str | None = None, started_monotonic: float | None = None, **updates: Any) -> None:
        nonlocal state
        if state.transition_index >= config.max_transitions:
            raise AgentRuntimeError("agent_budget_exhausted", "Maximum Agent transition budget exhausted.", origin="agent_runtime")
        validate_recovery_transition(state.state_name, to_state)
        from_state = state.state_name
        completed = _utc_now()
        state = transition_state(state, to_state, **updates)
        trace.append(
            AgentRecoveryTraceEvent(
                transition_index=state.transition_index,
                from_state=from_state,
                to_state=state.state_name,
                decision_type=decision_type,
                decision_reason=decision_reason,
                tool_name=tool_name,
                tool_call_index=tool_call_index,
                input_identity=input_identity,
                output_identity=output_identity,
                started_at=started or completed,
                completed_at=completed,
                duration_ms=0 if started_monotonic is None else max(0, int((time.monotonic() - started_monotonic) * 1000)),
                status=status,
                error_type=error_type,
            )
        )

    def tool(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, str, float]:
        _check_tool_budget(state, config, tool_name)
        started = _utc_now()
        started_monotonic = time.monotonic()
        payload, summary = executor.execute(tool_name, arguments, memory)
        return payload, summary, state.tool_call_count + 1, started, started_monotonic

    try:
        move("INITIAL_RETRIEVAL", decision_type="transition", decision_reason="start_initial_retrieval")
        _payload, summary, call_index, started, started_monotonic = tool("search_knowledge_base", {"query": state.active_query, "top_k": config.search_top_k})
        initial_summary = evidence_summary_from_tool(summary)
        move(
            "INITIAL_EVIDENCE_INSPECTION",
            decision_type="tool",
            decision_reason="initial_retrieval_completed",
            tool_name="search_knowledge_base",
            tool_call_index=call_index,
            input_identity=stable_digest({"query": state.active_query, "top_k": config.search_top_k}),
            output_identity=initial_summary["identity_digest"],
            started=started,
            started_monotonic=started_monotonic,
            retrieval_attempt_count=state.retrieval_attempt_count + 1,
            tool_call_count=state.tool_call_count + 1,
            initial_retrieval_summary=initial_summary,
        )

        _payload, answerability, call_index, started, started_monotonic = tool("assess_answerability", {})
        answerability_payload = answerability.get("answerability") or answerability
        status = answerability_payload.get("status")
        if status in ANSWERABLE_STATUSES:
            move(
                "GENERATION",
                decision_type="tool",
                decision_reason="initial_evidence_answerable",
                tool_name="assess_answerability",
                tool_call_index=call_index,
                output_identity=stable_digest(answerability_payload),
                started=started,
                started_monotonic=started_monotonic,
                tool_call_count=state.tool_call_count + 1,
                answerability_decision=answerability_payload,
            )
            return _generate_validate_and_finish(state, trace, executor, memory, config, final_answer, final_refusal_reason)

        eligibility = decide_recovery_eligibility(state, answerability_payload, config)
        if not eligibility.eligible:
            move(
                "FINAL_ABSTENTION",
                decision_type="eligibility",
                decision_reason=eligibility.recovery_reason,
                tool_name="assess_answerability",
                tool_call_index=call_index,
                output_identity=stable_digest(answerability_payload),
                started=started,
                started_monotonic=started_monotonic,
                tool_call_count=state.tool_call_count + 1,
                answerability_decision=answerability_payload,
                recovery_eligibility=eligibility.to_dict(),
                recovery_reason=eligibility.recovery_reason,
                final_action="abstain",
                termination_reason=eligibility.recovery_reason,
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason=eligibility.recovery_reason)

        move(
            "RECOVERY_ELIGIBILITY",
            decision_type="eligibility",
            decision_reason=eligibility.recovery_reason,
            tool_name="assess_answerability",
            tool_call_index=call_index,
            output_identity=stable_digest(answerability_payload),
            started=started,
            started_monotonic=started_monotonic,
            tool_call_count=state.tool_call_count + 1,
            answerability_decision=answerability_payload,
            recovery_eligibility=eligibility.to_dict(),
            recovery_reason=eligibility.recovery_reason,
        )

        started = _utc_now()
        started_monotonic = time.monotonic()
        plan = reformulator.reformulate(
            question=state.original_query,
            initial_query=state.active_query,
            summary={
                "answerability_status": status,
                "documents_found": (state.initial_retrieval_summary or {}).get("candidate_count", 0),
                "evidence_identities_found": (state.initial_retrieval_summary or {}).get("candidate_count", 0),
                "scope_localization_status": eligibility.recovery_reason,
            },
        )
        if not plan.queries:
            move(
                "FINAL_ABSTENTION",
                decision_type="query_reformulation",
                decision_reason="reformulation_not_permitted",
                started=started,
                started_monotonic=started_monotonic,
                reformulation_attempt_count=state.reformulation_attempt_count + 1,
                final_action="abstain",
                termination_reason="query_reformulation_failure",
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="query_reformulation_failure")
        reformulated_query = plan.queries[0].query
        move(
            "QUERY_REFORMULATION",
            decision_type="query_reformulation",
            decision_reason=plan.reason_code,
            input_identity=stable_digest({"query": state.active_query, "summary": state.initial_retrieval_summary}),
            output_identity=stable_digest(plan.to_dict(redact_queries=True)),
            started=started,
            started_monotonic=started_monotonic,
            reformulation_attempt_count=state.reformulation_attempt_count + 1,
            reformulated_query=reformulated_query,
            active_query=reformulated_query,
        )

        _payload, recovery_summary_raw, call_index, started, started_monotonic = tool("search_knowledge_base", {"query": state.active_query, "top_k": config.search_top_k})
        recovery_summary = evidence_summary_from_tool(recovery_summary_raw)
        move(
            "RECOVERY_RETRIEVAL",
            decision_type="tool",
            decision_reason="recovery_retrieval_completed",
            tool_name="search_knowledge_base",
            tool_call_index=call_index,
            input_identity=stable_digest({"query": state.active_query, "top_k": config.search_top_k}),
            output_identity=recovery_summary["identity_digest"],
            started=started,
            started_monotonic=started_monotonic,
            retrieval_attempt_count=state.retrieval_attempt_count + 1,
            tool_call_count=state.tool_call_count + 1,
            recovery_retrieval_summary=recovery_summary,
        )

        _payload, recovery_answerability, call_index, started, started_monotonic = tool("assess_answerability", {})
        recovery_answerability_payload = recovery_answerability.get("answerability") or recovery_answerability
        comparison = compare_retrieval_evidence(
            state.initial_retrieval_summary or {},
            state.recovery_retrieval_summary or {},
            initial_answerability=status,
            recovery_answerability=recovery_answerability_payload.get("status"),
        )
        move(
            "EVIDENCE_COMPARISON",
            decision_type="evidence_comparison",
            decision_reason=comparison.outcome,
            tool_name="assess_answerability",
            tool_call_index=call_index,
            output_identity=comparison.identity_digest,
            started=started,
            started_monotonic=started_monotonic,
            tool_call_count=state.tool_call_count + 1,
            evidence_comparison=comparison.to_dict(),
            answerability_decision=recovery_answerability_payload,
        )
        if comparison.outcome in {"evidence_unchanged", "evidence_degraded", "evidence_comparison_invalid"} and recovery_answerability_payload.get("status") not in ANSWERABLE_STATUSES:
            move(
                "FINAL_ABSTENTION",
                decision_type="evidence_comparison",
                decision_reason=comparison.outcome,
                final_action="abstain",
                termination_reason=comparison.outcome,
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason=comparison.outcome)
        if recovery_answerability_payload.get("status") not in ANSWERABLE_STATUSES:
            move(
                "FINAL_ABSTENTION",
                decision_type="answerability",
                decision_reason=recovery_answerability_payload.get("status") or "final_answerability_abstention",
                final_action="abstain",
                termination_reason="final_answerability_abstention",
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="final_answerability_abstention")

        move("FINAL_ANSWERABILITY", decision_type="answerability", decision_reason="recovery_evidence_answerable")
        move("GENERATION", decision_type="transition", decision_reason="answerability_permits_generation")
        return _generate_validate_and_finish(state, trace, executor, memory, config, final_answer, final_refusal_reason)
    except AgentRuntimeError as exc:
        completed = _utc_now()
        error_state = failure_to_state(state, exc.failure, completed_at=completed)
        trace.append(
            AgentRecoveryTraceEvent(
                transition_index=error_state.transition_index,
                from_state=state.state_name,
                to_state=error_state.state_name,
                decision_type="error",
                decision_reason=exc.failure.code,
                tool_name=None,
                tool_call_index=None,
                input_identity=state.identity_digest,
                output_identity=error_state.identity_digest,
                started_at=completed,
                completed_at=completed,
                duration_ms=0,
                status="error",
                error_type=exc.failure.origin,
            )
        )
        return AgentRecoveryRunResult(state=error_state, trace_events=tuple(trace), final_refusal_reason=exc.failure.code)


def decide_recovery_eligibility(state: AgentRecoveryState, answerability_payload: dict[str, Any], config: AgentRecoveryConfig) -> RecoveryEligibilityDecision:
    status = answerability_payload.get("status")
    reason = answerability_payload.get("reason_code") or status or "unknown"
    if answerability_payload.get("forbidden_scope_present") is True:
        return RecoveryEligibilityDecision(False, "safety_terminal", terminal_safety=True, detail={"reason_code": reason})
    if status in SAFETY_TERMINAL_STATUSES:
        return RecoveryEligibilityDecision(False, "safety_terminal", terminal_safety=True, detail={"answerability_status": status})
    if state.reformulation_attempt_count >= config.max_reformulation_calls or state.tool_call_count >= config.max_total_tool_calls:
        return RecoveryEligibilityDecision(False, "budget_exhausted")
    if status in RECOVERABLE_STATUSES:
        mapped = _map_recovery_reason(reason)
        return RecoveryEligibilityDecision(True, mapped, detail={"answerability_status": status, "reason_code": reason})
    return RecoveryEligibilityDecision(False, "recovery_not_permitted", detail={"answerability_status": status, "reason_code": reason})


def _generate_validate_and_finish(
    state: AgentRecoveryState,
    trace: list[AgentRecoveryTraceEvent],
    executor: AgentCoreToolExecutor,
    memory: dict[str, Any],
    config: AgentRecoveryConfig,
    final_answer: str | None,
    final_refusal_reason: str | None,
) -> AgentRecoveryRunResult:
    def local_move(to_state: str, *, decision_type: str, decision_reason: str, tool_name: str | None = None, tool_call_index: int | None = None, input_identity: str | None = None, output_identity: str | None = None, status: str = "ok", error_type: str | None = None, started: str | None = None, started_monotonic: float | None = None, **updates: Any) -> None:
        nonlocal state
        if state.transition_index >= config.max_transitions:
            raise AgentRuntimeError("agent_budget_exhausted", "Maximum Agent transition budget exhausted.", origin="agent_runtime")
        validate_recovery_transition(state.state_name, to_state)
        from_state = state.state_name
        completed = _utc_now()
        state = transition_state(state, to_state, **updates)
        trace.append(
            AgentRecoveryTraceEvent(
                transition_index=state.transition_index,
                from_state=from_state,
                to_state=state.state_name,
                decision_type=decision_type,
                decision_reason=decision_reason,
                tool_name=tool_name,
                tool_call_index=tool_call_index,
                input_identity=input_identity,
                output_identity=output_identity,
                started_at=started or completed,
                completed_at=completed,
                duration_ms=0 if started_monotonic is None else max(0, int((time.monotonic() - started_monotonic) * 1000)),
                status=status,
                error_type=error_type,
            )
        )

    def tool(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, str, float]:
        _check_tool_budget(state, config, tool_name)
        started = _utc_now()
        started_monotonic = time.monotonic()
        payload, summary = executor.execute(tool_name, arguments, memory)
        return payload, summary, state.tool_call_count + 1, started, started_monotonic

    try:
        _payload, generation_summary, call_index, started, started_monotonic = tool("generate_grounded_answer", {})
        generation_payload = generation_summary.get("answer_response") or generation_summary
        attempt_1_request = _generation_request_identity(state, attempt_index=1)
        attempt_1_digest = generation_request_identity_digest(attempt_1_request)
        if generation_payload.get("status") != "answered":
            if config.generation_retry_enabled:
                observation = _generation_retry_observation(state, generation_payload, attempt_index=1, retry_budget_remaining=max(0, config.max_generation_retries - state.generation_retry_count))
                decision = decide_generation_retry_eligibility(observation)
                local_move(
                    "GENERATION_RETRY_ELIGIBILITY",
                    decision_type="generation_retry_eligibility",
                    decision_reason=decision.decision_reason,
                    tool_name="generate_grounded_answer",
                    tool_call_index=call_index,
                    output_identity=stable_digest(generation_payload),
                    started=started,
                    started_monotonic=started_monotonic,
                    generation_attempt_count=state.generation_attempt_count + 1,
                    tool_call_count=state.tool_call_count + 1,
                    generation_result=generation_payload,
                    generation_retry_eligibility_evaluated=True,
                    generation_retry_eligible=decision.eligible,
                    generation_retry_decision_reason=decision.decision_reason,
                    generation_retry_rule_id=decision.matched_rule_id,
                    generation_request_identity_digest_attempt_1=attempt_1_digest,
                )
                if not decision.eligible:
                    local_move(
                        "FINAL_ABSTENTION",
                        decision_type="generation_retry_eligibility",
                        decision_reason=decision.decision_reason,
                        generation_retry_outcome="retry_not_eligible",
                        final_action="abstain",
                        termination_reason="generation_refusal",
                        completed_at=_utc_now(),
                    )
                    return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="generation_refusal")
                attempt_2_request = _generation_request_identity(state, attempt_index=2, retry_of_invocation_id=f"{state.run_id}:generation:1")
                identity = validate_retry_request_identity(attempt_1_request, attempt_2_request)
                if not identity["valid"]:
                    local_move(
                        "FINAL_ABSTENTION",
                        decision_type="generation_retry_identity",
                        decision_reason="request_identity_mismatch",
                        generation_request_identity_digest_attempt_2=identity["generation_request_identity_digest_attempt_2"],
                        generation_retry_outcome="retry_request_identity_mismatch",
                        final_action="abstain",
                        termination_reason="request_identity_mismatch",
                        completed_at=_utc_now(),
                    )
                    return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="request_identity_mismatch")
                local_move(
                    "GENERATION_RETRY",
                    decision_type="generation_retry",
                    decision_reason="eligible_for_single_generation_retry",
                    generation_retry_started=True,
                    generation_request_identity_digest_attempt_2=identity["generation_request_identity_digest_attempt_2"],
                )
                _payload, retry_generation_summary, retry_call_index, retry_started, retry_started_monotonic = tool("generate_grounded_answer", {})
                retry_generation_payload = retry_generation_summary.get("answer_response") or retry_generation_summary
                if retry_generation_payload.get("status") != "answered":
                    retry_outcome = classify_retry_outcome(_generation_retry_observation(state, retry_generation_payload, attempt_index=2, retry_budget_remaining=0))
                    local_move(
                        "FINAL_ABSTENTION",
                        decision_type="generation_retry",
                        decision_reason=retry_generation_payload.get("refusal_reason_code") or retry_outcome,
                        tool_name="generate_grounded_answer",
                        tool_call_index=retry_call_index,
                        output_identity=stable_digest(retry_generation_payload),
                        started=retry_started,
                        started_monotonic=retry_started_monotonic,
                        generation_attempt_count=state.generation_attempt_count + 1,
                        generation_retry_count=state.generation_retry_count + 1,
                        tool_call_count=state.tool_call_count + 1,
                        generation_result=retry_generation_payload,
                        generation_retry_completed=True,
                        generation_retry_outcome=retry_outcome,
                        final_action="abstain",
                        termination_reason=retry_outcome,
                        completed_at=_utc_now(),
                    )
                    return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason=retry_outcome)
                local_move(
                    "RETRY_CITATION_VALIDATION",
                    decision_type="generation_retry",
                    decision_reason="retry_generation_completed",
                    tool_name="generate_grounded_answer",
                    tool_call_index=retry_call_index,
                    output_identity=stable_digest(retry_generation_payload),
                    started=retry_started,
                    started_monotonic=retry_started_monotonic,
                    generation_attempt_count=state.generation_attempt_count + 1,
                    generation_retry_count=state.generation_retry_count + 1,
                    tool_call_count=state.tool_call_count + 1,
                    generation_result=retry_generation_payload,
                    generation_retry_completed=True,
                )
                retry_citation_result = _validate_citations(retry_generation_payload, state)
                if not retry_citation_result["valid"]:
                    local_move(
                        "FINAL_ABSTENTION",
                        decision_type="retry_citation_validation",
                        decision_reason=retry_citation_result["reason_code"],
                        output_identity=stable_digest(retry_citation_result),
                        citation_result=retry_citation_result,
                        generation_retry_outcome="retry_answer_draft_citation_failure",
                        final_action="abstain",
                        termination_reason="citation_failure",
                        completed_at=_utc_now(),
                    )
                    return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="citation_failure")
                local_move("RETRY_GROUNDING_VALIDATION", decision_type="retry_citation_validation", decision_reason="citation_valid", output_identity=stable_digest(retry_citation_result), citation_result=retry_citation_result)
                _payload, retry_grounding_summary, grounding_call_index, grounding_started, grounding_started_monotonic = tool("verify_grounding", {})
                retry_grounding_payload = retry_grounding_summary.get("grounding") or retry_grounding_summary
                if retry_grounding_payload.get("valid") is not True:
                    retry_outcome = "retry_answer_draft_unsupported_claim" if retry_grounding_payload.get("reason_code") == "unsupported_claims" else "retry_answer_draft_grounding_failure"
                    local_move(
                        "FINAL_ABSTENTION",
                        decision_type="retry_grounding_validation",
                        decision_reason=retry_grounding_payload.get("reason_code") or "grounding_failure",
                        tool_name="verify_grounding",
                        tool_call_index=grounding_call_index,
                        output_identity=stable_digest(retry_grounding_payload),
                        started=grounding_started,
                        started_monotonic=grounding_started_monotonic,
                        tool_call_count=state.tool_call_count + 1,
                        grounding_result=retry_grounding_payload,
                        generation_retry_outcome=retry_outcome,
                        final_action="abstain",
                        termination_reason="grounding_failure",
                        completed_at=_utc_now(),
                    )
                    return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="grounding_failure")
                answer = memory.get("answer_response")
                final_answer = getattr(answer, "answer", None)
                local_move(
                    "FINAL_ANSWER",
                    decision_type="retry_grounding_validation",
                    decision_reason="grounding_valid",
                    tool_name="verify_grounding",
                    tool_call_index=grounding_call_index,
                    output_identity=stable_digest(retry_grounding_payload),
                    started=grounding_started,
                    started_monotonic=grounding_started_monotonic,
                    tool_call_count=state.tool_call_count + 1,
                    grounding_result=retry_grounding_payload,
                    generation_retry_outcome="retry_answer_draft_grounded",
                    final_action="answer",
                    termination_reason="grounding_valid",
                    completed_at=_utc_now(),
                )
                return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_answer=final_answer, final_refusal_reason=final_refusal_reason)
            local_move(
                "FINAL_ABSTENTION",
                decision_type="generation",
                decision_reason=generation_payload.get("refusal_reason_code") or "generation_refusal",
                tool_name="generate_grounded_answer",
                tool_call_index=call_index,
                output_identity=stable_digest(generation_payload),
                started=started,
                started_monotonic=started_monotonic,
                generation_attempt_count=state.generation_attempt_count + 1,
                tool_call_count=state.tool_call_count + 1,
                generation_result=generation_payload,
                generation_request_identity_digest_attempt_1=attempt_1_digest,
                final_action="abstain",
                termination_reason="generation_refusal",
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="generation_refusal")
        local_move(
            "CITATION_VALIDATION",
            decision_type="generation",
            decision_reason="generation_completed",
            tool_name="generate_grounded_answer",
            tool_call_index=call_index,
            output_identity=stable_digest(generation_payload),
            started=started,
            started_monotonic=started_monotonic,
            generation_attempt_count=state.generation_attempt_count + 1,
            tool_call_count=state.tool_call_count + 1,
            generation_result=generation_payload,
            generation_request_identity_digest_attempt_1=attempt_1_digest,
        )

        citation_result = _validate_citations(generation_payload, state)
        if not citation_result["valid"]:
            local_move(
                "FINAL_ABSTENTION",
                decision_type="citation_validation",
                decision_reason=citation_result["reason_code"],
                output_identity=stable_digest(citation_result),
                citation_result=citation_result,
                final_action="abstain",
                termination_reason="citation_failure",
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="citation_failure")
        local_move("GROUNDING_VALIDATION", decision_type="citation_validation", decision_reason="citation_valid", output_identity=stable_digest(citation_result), citation_result=citation_result)

        _payload, grounding_summary, call_index, started, started_monotonic = tool("verify_grounding", {})
        grounding_payload = grounding_summary.get("grounding") or grounding_summary
        if grounding_payload.get("valid") is not True:
            local_move(
                "FINAL_ABSTENTION",
                decision_type="grounding_validation",
                decision_reason=grounding_payload.get("reason_code") or "grounding_failure",
                tool_name="verify_grounding",
                tool_call_index=call_index,
                output_identity=stable_digest(grounding_payload),
                started=started,
                started_monotonic=started_monotonic,
                tool_call_count=state.tool_call_count + 1,
                grounding_result=grounding_payload,
                final_action="abstain",
                termination_reason="grounding_failure",
                completed_at=_utc_now(),
            )
            return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_refusal_reason="grounding_failure")
        answer = memory.get("answer_response")
        final_answer = getattr(answer, "answer", None)
        local_move(
            "FINAL_ANSWER",
            decision_type="grounding_validation",
            decision_reason="grounding_valid",
            tool_name="verify_grounding",
            tool_call_index=call_index,
            output_identity=stable_digest(grounding_payload),
            started=started,
            started_monotonic=started_monotonic,
            tool_call_count=state.tool_call_count + 1,
            grounding_result=grounding_payload,
            final_action="answer",
            termination_reason="grounding_valid",
            completed_at=_utc_now(),
        )
        return AgentRecoveryRunResult(state=state, trace_events=tuple(trace), final_answer=final_answer, final_refusal_reason=final_refusal_reason)
    except AgentRuntimeError as exc:
        completed = _utc_now()
        error_state = failure_to_state(state, exc.failure, completed_at=completed)
        trace.append(
            AgentRecoveryTraceEvent(
                transition_index=error_state.transition_index,
                from_state=state.state_name,
                to_state=error_state.state_name,
                decision_type="error",
                decision_reason=exc.failure.code,
                tool_name=None,
                tool_call_index=None,
                input_identity=state.identity_digest,
                output_identity=error_state.identity_digest,
                started_at=completed,
                completed_at=completed,
                duration_ms=0,
                status="error",
                error_type=exc.failure.origin,
            )
        )
        return AgentRecoveryRunResult(state=error_state, trace_events=tuple(trace), final_refusal_reason=exc.failure.code)


def _validate_citations(generation_payload: dict[str, Any], state: AgentRecoveryState) -> dict[str, Any]:
    citations = [item.get("citation_id") for item in generation_payload.get("citations", []) if isinstance(item, dict)]
    summary = state.recovery_retrieval_summary or state.initial_retrieval_summary or {}
    available = {candidate.get("citation_id") for candidate in summary.get("candidates", []) if isinstance(candidate, dict)}
    invalid = sorted(str(citation) for citation in citations if citation not in available)
    return {
        "schema_version": "opk-rag.agent-citation-validation.v1",
        "valid": bool(citations) and not invalid,
        "cited_ids": citations,
        "available_ids": sorted(str(value) for value in available if value is not None),
        "invalid_ids": invalid,
        "reason_code": None if citations and not invalid else "citation_failure",
    }


def _check_tool_budget(state: AgentRecoveryState, config: AgentRecoveryConfig, tool_name: str) -> None:
    if state.tool_call_count >= config.max_total_tool_calls:
        raise AgentRuntimeError("agent_budget_exhausted", "Maximum total tool call budget exhausted.", origin="agent_runtime")
    if tool_name == "search_knowledge_base" and state.retrieval_attempt_count >= config.max_retrieval_calls:
        raise AgentRuntimeError("agent_budget_exhausted", "Maximum retrieval call budget exhausted.", origin="agent_runtime")
    if tool_name == "generate_grounded_answer" and state.generation_attempt_count >= config.max_generation_calls:
        raise AgentRuntimeError("agent_budget_exhausted", "Maximum generation call budget exhausted.", origin="agent_runtime")
    if tool_name == "generate_grounded_answer" and state.generation_retry_count >= config.max_generation_retries and state.generation_attempt_count > 0:
        raise AgentRuntimeError("agent_budget_exhausted", "Maximum generation retry budget exhausted.", origin="agent_runtime")


def _map_recovery_reason(reason_code: str) -> str:
    mapping = {
        "insufficient_evidence": "insufficient_required_evidence",
        "no_evidence": "no_relevant_candidate",
        "low_evidence": "insufficient_required_evidence",
        "missing_scope": "insufficient_scope_localization",
        "semantic_mismatch": "retrieval_semantic_mismatch",
        "lexical_mismatch": "retrieval_lexical_mismatch",
    }
    return mapping.get(reason_code, "insufficient_required_evidence")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _generation_request_identity(state: AgentRecoveryState, *, attempt_index: int, retry_of_invocation_id: str | None = None) -> dict[str, Any]:
    retrieval = state.recovery_retrieval_summary or state.initial_retrieval_summary or {}
    return {
        "original_query_digest": stable_digest(state.original_query),
        "active_query_digest": stable_digest(state.active_query),
        "selected_evidence_identity_digest": retrieval.get("identity_digest") or stable_digest(retrieval),
        "selected_evidence_ordering_digest": stable_digest([candidate.get("citation_id") for candidate in retrieval.get("candidates", []) if isinstance(candidate, dict)]),
        "evidence_budget": {
            "candidate_count": retrieval.get("candidate_count"),
            "context_token_budget": retrieval.get("context_token_budget"),
            "context_token_count": retrieval.get("context_token_count"),
        },
        "generation_prompt_identity": "answer-prompt-v4",
        "generation_contract_id": "answer-response-v3",
        "provider_identity": "reference_runtime_provider",
        "model_identity": "reference_runtime_model",
        "provider_parameter_identity": "reference_runtime_answer_config",
        "answerability_result_digest": stable_digest(state.answerability_decision or {}),
        "generation_invocation_id": f"{state.run_id}:generation:{attempt_index}",
        "generation_attempt_index": attempt_index,
        "retry_of_invocation_id": retry_of_invocation_id,
    }


def _generation_retry_observation(state: AgentRecoveryState, generation_payload: dict[str, Any], *, attempt_index: int, retry_budget_remaining: int) -> dict[str, Any]:
    status = generation_payload.get("status")
    response_contract_valid = status in {"answered", "refused"}
    response_mode = "answer" if status == "answered" else "abstain" if status == "refused" else "empty" if not generation_payload else "malformed"
    refusal_reason = generation_payload.get("refusal_reason_code")
    answerability = state.answerability_decision or {}
    runtime_failure_class = "refusal_model_conservative"
    if status == "answered":
        runtime_failure_class = "answer_draft_grounded"
    elif not response_contract_valid:
        runtime_failure_class = "generation_contract_invalid"
    elif refusal_reason in {"prompt_injection_detected", "false_premise", "forbidden", "disallowed"}:
        runtime_failure_class = "refusal_safety_terminal"
    elif refusal_reason in {"no_evidence", "insufficient_evidence", "no_relevant_context", "partial_evidence"}:
        runtime_failure_class = "refusal_claimed_insufficient_evidence"
    elif refusal_reason in {"unsupported_claims", "ungrounded_answer", "invalid_citations", "model_abstained", "answerable_generation_abstained", None}:
        runtime_failure_class = "refusal_model_conservative"
    return {
        "generation_invoked": True,
        "provider_call_completed": bool(generation_payload),
        "response_contract_valid": response_contract_valid,
        "response_mode": response_mode,
        "provider_refusal_detected": status == "refused" or bool(refusal_reason),
        "original_refusal_reason_code": refusal_reason or "model_abstained",
        "runtime_failure_class": runtime_failure_class,
        "runtime_evidence_class": "runtime_evidence_appears_sufficient" if answerability.get("status") in ANSWERABLE_STATUSES and int(answerability.get("evidence_count") or 0) > 0 else "runtime_evidence_appears_insufficient",
        "answerability_label": answerability.get("status"),
        "generation_invocation_count": attempt_index,
        "generation_retries": state.generation_retry_count,
        "retry_budget_remaining": retry_budget_remaining,
        "observability_complete": True,
        "safety_terminal_signal": runtime_failure_class == "refusal_safety_terminal",
        "correct_unanswerable_signal": runtime_failure_class == "refusal_correct_unanswerable",
        "forbidden_claim_signal": runtime_failure_class == "refusal_forbidden_claim_risk",
        "unsupported_claim_signal": False,
        "answer_draft_present": status == "answered",
        "citation_validation_passed": None,
        "grounding_validation_passed": None,
    }
