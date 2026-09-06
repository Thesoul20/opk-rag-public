from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from opk_rag.agent.baseline_policy import DeterministicBaselinePolicy
from opk_rag.agent.contracts import AgentAction, AgentRuntimeConfig, AgentState, state_digest
from opk_rag.agent.errors import AgentFailure, AgentRuntimeError
from opk_rag.agent.policy import AgentPolicy
from opk_rag.agent.tool_registry import ACTION_TO_TOOL, AgentToolRegistry
from opk_rag.agent.trace import AgentTraceEvent, summarize_for_trace, utc_now
from opk_rag.agent.policy_validation import action_argument_diagnostics
from opk_rag.agent.transition import answerability_result_digest, transition_trace
from opk_rag.evaluation.post_generation_failure_classification import (
    build_generation_boundary_snapshot,
    build_generation_outcome_snapshot,
    build_observability_record,
    build_validation_snapshot,
)


@dataclass(frozen=True)
class AgentRunResult:
    state: AgentState
    trace_events: tuple[AgentTraceEvent, ...]
    final_answer: str | None = None
    final_refusal_reason: str | None = None
    post_generation_observability: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.state.run_id,
            "sample_id": self.state.sample_id,
            "question_hash": _text_digest(self.state.question),
            "status": self.state.status,
            "final_action": self.state.final_action,
            "failure": None if self.state.failure is None else self.state.failure.to_dict(),
            "tool_call_count": self.state.tool_call_count,
            "expansion_count": self.state.expansion_count,
            "search_round_count": self.state.search_round_count,
            "step_count": self.state.step_index,
            "evidence_state": self.state.evidence_state,
            "answerability_state": self.state.answerability_state,
            "generation_state": self.state.generation_state,
            "verification_state": self.state.verification_state,
            "final_answer_hash": None if self.final_answer is None else _text_digest(self.final_answer),
            "final_refusal_reason": self.final_refusal_reason,
            "post_generation_observability": list(self.post_generation_observability),
        }


class AgentRuntime:
    def __init__(
        self,
        *,
        tool_registry: AgentToolRegistry,
        policy: AgentPolicy | None = None,
        config: AgentRuntimeConfig | None = None,
        post_generation_observability_enabled: bool = False,
        observability_split: str | None = None,
        observability_replicate_id: str | int | None = None,
        observability_variant: str = "a1",
    ) -> None:
        self.config = config or AgentRuntimeConfig()
        self.policy = policy or DeterministicBaselinePolicy(self.config)
        self.tool_registry = tool_registry
        self.post_generation_observability_enabled = post_generation_observability_enabled
        self.observability_split = observability_split
        self.observability_replicate_id = observability_replicate_id
        self.observability_variant = observability_variant

    def run(self, *, question: str, sample_id: str | None = None, run_id: str | None = None) -> AgentRunResult:
        state = AgentState(run_id=run_id or str(uuid4()), sample_id=sample_id, question=question)
        memory: dict[str, Any] = {}
        trace: list[AgentTraceEvent] = []
        final_answer = None
        final_refusal_reason = None
        boundary_by_invocation: dict[str, dict[str, Any]] = {}
        outcome_by_invocation: dict[str, dict[str, Any]] = {}
        generation_order: list[str] = []

        while not state.is_terminal:
            if state.step_index >= self.config.max_steps:
                state = self._terminal_failure(state, AgentFailure("agent_budget_exhausted", "Maximum step budget exhausted.", "agent_runtime"))
                break
            before = state_digest(state)
            started = time.monotonic()
            failure = None
            tool_request: dict[str, Any] = {}
            tool_summary: dict[str, Any] = {}
            try:
                state_before = state
                action = self.policy.next_action(state)
                policy_decision = getattr(self.policy, "last_decision_trace", None)
                self.tool_registry.validate_action(action, state, self.config)
                if action.action in ACTION_TO_TOOL:
                    tool_name = ACTION_TO_TOOL[action.action]
                    tool_request = {
                        "tool_name": tool_name,
                        "argument_diagnostics": action_argument_diagnostics(action.action, action.arguments),
                    }
                    generation_invocation_id = None
                    if self.post_generation_observability_enabled and action.action == "generate_grounded_answer":
                        generation_invocation_id = f"{state.run_id}:generation:{state.step_index + 1}"
                        boundary_by_invocation[generation_invocation_id] = build_generation_boundary_snapshot(
                            run_id=state.run_id,
                            sample_id=state.sample_id,
                            split=self.observability_split,
                            replicate_id=self.observability_replicate_id,
                            variant=self.observability_variant,
                            generation_invocation_id=generation_invocation_id,
                            question=state.question,
                            evidence_state=state.evidence_state,
                            answerability_state=state.answerability_state,
                        )
                        generation_order.append(generation_invocation_id)
                    _payload, tool_summary = self.tool_registry.execute(action, state, memory, self.config)
                    state = self._apply_tool_result(state, action, tool_summary, memory)
                    if self.post_generation_observability_enabled and generation_invocation_id is not None:
                        outcome_by_invocation[generation_invocation_id] = build_generation_outcome_snapshot(
                            generation_invocation_id=generation_invocation_id,
                            tool_summary=tool_summary,
                            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                        )
                else:
                    state, final_answer, final_refusal_reason = self._apply_finish(state, action, memory)
            except AgentRuntimeError as exc:
                state_before = state
                failure = exc.failure
                state = self._terminal_failure(state, failure)
                action = AgentAction(action="finish_failure", reason_code=failure.code, expected_state_transition="terminal_failure")
                policy_decision = getattr(self.policy, "last_decision_trace", None)
                tool_summary = {"failure": failure.to_dict()}
                if self.post_generation_observability_enabled and action.action == "finish_failure":
                    generation_invocation_id = f"{state_before.run_id}:generation:{state_before.step_index + 1}"
                    if state_before.answerability_state.get("status") in {"answerable", "partially_answerable"}:
                        boundary_by_invocation[generation_invocation_id] = build_generation_boundary_snapshot(
                            run_id=state_before.run_id,
                            sample_id=state_before.sample_id,
                            split=self.observability_split,
                            replicate_id=self.observability_replicate_id,
                            variant=self.observability_variant,
                            generation_invocation_id=generation_invocation_id,
                            question=state_before.question,
                            evidence_state=state_before.evidence_state,
                            answerability_state=state_before.answerability_state,
                        )
                        outcome_by_invocation[generation_invocation_id] = build_generation_outcome_snapshot(
                            generation_invocation_id=generation_invocation_id,
                            failure=failure.to_dict(),
                            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                        )
                        generation_order.append(generation_invocation_id)
            after = state_digest(state)
            runtime_transition = transition_trace(
                step_index=state.step_index - 1,
                state_before=state_before,
                state_after=state,
                selected_action=action.action,
                action_source=self.policy.name,
                runtime_validation="failed" if failure is not None else "passed",
                state_before_digest=before,
                state_after_digest=after,
                config=self.config,
            )
            trace.append(
                AgentTraceEvent(
                    run_id=state.run_id,
                    sample_id=state.sample_id,
                    policy_name=self.policy.name,
                    policy_version=self.policy.version,
                    runtime_contract_version=self.config.contract_version,
                    tool_registry_version=self.tool_registry.version,
                    step_index=state.step_index - 1,
                    state_before_digest=before,
                    action=action,
                    tool_request=summarize_for_trace(tool_request),
                    tool_response_summary=summarize_for_trace(tool_summary),
                    state_after_digest=after,
                    latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                    failure=failure,
                    policy_decision=policy_decision,
                    runtime_transition=runtime_transition,
                    timestamp=utc_now(),
                )
            )
        observability_records: list[dict[str, Any]] = []
        for invocation_id in generation_order:
            boundary = boundary_by_invocation.get(invocation_id)
            outcome = outcome_by_invocation.get(invocation_id)
            if boundary is None or outcome is None:
                continue
            validation = build_validation_snapshot(
                generation_invocation_id=invocation_id,
                generation_state=state.generation_state,
                verification_state=state.verification_state,
                final_action=state.final_action,
                final_termination_reason=final_refusal_reason or state.status,
            )
            observability_records.append(build_observability_record(boundary, outcome, validation).to_dict())
        return AgentRunResult(
            state=state,
            trace_events=tuple(trace),
            final_answer=final_answer,
            final_refusal_reason=final_refusal_reason,
            post_generation_observability=tuple(observability_records),
        )

    def _apply_tool_result(self, state: AgentState, action: AgentAction, summary: dict[str, Any], memory: dict[str, Any]) -> AgentState:
        next_state = replace(state, step_index=state.step_index + 1, tool_call_count=state.tool_call_count + 1)
        if action.action == "search":
            trusted = summary.get("trusted_evidence") or {}
            return replace(
                next_state,
                search_round_count=state.search_round_count + 1,
                evidence_state={
                    "searched": True,
                    "search_round_count": state.search_round_count + 1,
                    "trusted_evidence_digest": trusted.get("evidence_identity_digest"),
                    "citation_ids": trusted.get("citation_ids", []),
                    "chunk_ids": trusted.get("chunk_ids", []),
                    "document_ids": trusted.get("document_ids", []),
                    "result_count": summary.get("result_count"),
                },
                answerability_state={},
                generation_state={},
                verification_state={},
            )
        if action.action == "expand_evidence":
            trusted = summary.get("trusted_evidence") or {}
            return replace(
                next_state,
                expansion_count=state.expansion_count + 1,
                evidence_state={
                    **state.evidence_state,
                    "expanded": True,
                    "trusted_evidence_digest": trusted.get("evidence_identity_digest"),
                    "citation_ids": trusted.get("citation_ids", state.evidence_state.get("citation_ids", [])),
                    "chunk_ids": trusted.get("chunk_ids", state.evidence_state.get("chunk_ids", [])),
                    "document_ids": trusted.get("document_ids", state.evidence_state.get("document_ids", [])),
                },
                answerability_state={},
            )
        if action.action == "evaluate_answerability":
            payload = summary.get("answerability") or {}
            evidence_digest = state.evidence_state.get("trusted_evidence_digest")
            return replace(
                next_state,
                answerability_state={
                    "status": payload.get("status"),
                    "reason_code": payload.get("reason_code"),
                    "evidence_count": payload.get("evidence_count"),
                    "evidence_chunk_ids": payload.get("evidence_chunk_ids", []),
                    "evidence_bundle_digest": evidence_digest,
                    "answerability_result_digest": answerability_result_digest(payload, evidence_digest),
                    "forbidden_scope_present": payload.get("forbidden_scope_present", False),
                },
            )
        if action.action == "generate_grounded_answer":
            payload = summary.get("answer_response") or {}
            return replace(
                next_state,
                generation_state={
                    "attempted": True,
                    "status": payload.get("status"),
                    "answer_hash": payload.get("answer_hash"),
                    "citation_ids": [citation.get("citation_id") for citation in payload.get("citations", [])],
                    "refusal_reason_code": payload.get("refusal_reason_code"),
                },
            )
        if action.action == "verify_grounding":
            payload = summary.get("grounding") or {}
            return replace(
                next_state,
                verification_state={
                    "attempted": True,
                    "valid": payload.get("valid"),
                    "status": payload.get("status"),
                    "reason_code": payload.get("reason_code"),
                    "cited_ids": payload.get("cited_ids", []),
                },
            )
        raise AgentRuntimeError("agent_state_transition_failure", f"Unhandled tool action: {action.action}", origin="agent_runtime")

    def _apply_finish(self, state: AgentState, action: AgentAction, memory: dict[str, Any]) -> tuple[AgentState, str | None, str | None]:
        next_state = replace(state, step_index=state.step_index + 1)
        if action.action == "finish_answer":
            answer = memory.get("answer_response")
            return replace(next_state, status="answered", final_action="answer"), getattr(answer, "answer", None), None
        if action.action == "finish_abstain":
            reason = action.reason_code
            answer = memory.get("answer_response")
            reason = getattr(answer, "refusal_reason_code", None) or reason
            return replace(next_state, status="abstained", final_action="abstain"), None, reason
        if action.action == "finish_failure":
            return self._terminal_failure(state, AgentFailure("agent_policy_failure", action.reason_code, "agent_policy")), None, action.reason_code
        raise AgentRuntimeError("agent_invalid_action", f"Unknown finish action: {action.action}", origin="agent_policy")

    def _terminal_failure(self, state: AgentState, failure: AgentFailure) -> AgentState:
        return replace(state, step_index=state.step_index + 1, status="failed", final_action="failure", failure=failure)


def _text_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
