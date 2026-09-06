from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from opk_rag.agent.contracts import AgentAction, AgentRuntimeConfig, AgentState, state_digest
from opk_rag.agent.errors import AgentFailure, AgentRuntimeError
from opk_rag.agent.evidence_merge import merge_evidence_rounds, records_from_trusted_evidence
from opk_rag.agent.query_privacy import diagnose_query_privacy, query_digest
from opk_rag.agent.query_reformulation import GovernedQueryReformulator
from opk_rag.agent.retrieval_round import build_retrieval_round
from opk_rag.agent.tool_registry import ACTION_TO_TOOL, AgentToolRegistry
from opk_rag.agent.trace import AgentTraceEvent, summarize_for_trace, utc_now

RETRIEVAL_PLANNING_POLICY_NAME = "model_policy_query_reformulation_v1"
RETRIEVAL_PLANNING_POLICY_VERSION = "1.0.0"


class GovernedRetrievalPlanningRunner:
    """Bounded two-round retrieval planner for TASK-0067 experiments."""

    def __init__(
        self,
        *,
        tool_registry: AgentToolRegistry,
        reformulator: GovernedQueryReformulator,
        config: AgentRuntimeConfig | None = None,
    ) -> None:
        self.config = config or AgentRuntimeConfig(
            model_planning_enabled=True,
            policy_name=RETRIEVAL_PLANNING_POLICY_NAME,
            policy_version=RETRIEVAL_PLANNING_POLICY_VERSION,
            max_steps=12,
            max_tool_calls=8,
            max_search_rounds=2,
            max_reformulations=1,
            max_queries_per_reformulation=2,
        )
        self.tool_registry = tool_registry
        self.reformulator = reformulator

    def run_retrieval(self, *, question: str, sample_id: str | None = None, run_id: str = "task0067-run") -> dict[str, Any]:
        state = AgentState(run_id=run_id, sample_id=sample_id, question=question)
        memory: dict[str, Any] = {}
        traces: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        query_privacy: list[dict[str, Any]] = []
        reformulation_count = 0

        state, trace, first_summary, first_latency = self._execute_action(state, memory, AgentAction("search", "initial_retrieval_required", {"query": question, "top_k": self.config.search_top_k}, expected_state_transition="evidence_available"))
        traces.append(trace)
        state, trace, answerability_summary, _latency = self._execute_action(state, memory, AgentAction("evaluate_answerability", "initial_answerability_check", expected_state_transition="answerability_decided"))
        traces.append(trace)
        first_trusted = first_summary.get("trusted_evidence") or {}
        first_answerability = (answerability_summary.get("answerability") or {}).get("status")
        first_records = records_from_trusted_evidence(first_trusted, source_round=1, query_digest=query_digest(question.strip()))
        rounds.append(
            build_retrieval_round(
                round_index=1,
                query=question,
                query_source="original_question",
                result_count=int(first_summary.get("result_count") or 0),
                document_ids=first_trusted.get("document_ids") or [],
                evidence_ids=[record.evidence_identity for record in first_records],
                answerability_after=first_answerability,
                latency_ms=first_latency,
            ).to_dict()
        )

        second_records = []
        plan_trace = None
        if first_answerability in {"insufficient_evidence", "partial_evidence"} and self._can_reformulate(state, reformulation_count):
            try:
                plan = self.reformulator.reformulate(
                    question=question,
                    initial_query=question,
                    summary={
                        "answerability_status": first_answerability,
                        "documents_found": len(set(first_trusted.get("document_ids") or [])),
                        "evidence_identities_found": len(first_records),
                        "scope_localization_status": "weak",
                        "duplicate_ratio": 0.0,
                    },
                )
                reformulation_count += 1
                plan_trace = self.reformulator.last_trace
                for query in plan.queries[: self.config.max_queries_per_reformulation]:
                    query_privacy.append({"source": "model_reformulation", "round": 2, **diagnose_query_privacy(query.query).to_dict()})
                    if state.search_round_count >= self.config.max_search_rounds or state.tool_call_count >= self.config.max_tool_calls:
                        break
                    before_answerability = state.answerability_state.get("status")
                    state, trace, search_summary, search_latency = self._execute_action(state, memory, AgentAction("search", "query_reformulation_search", {"query": query.query, "top_k": self.config.search_top_k}, expected_state_transition="evidence_available"))
                    traces.append(trace)
                    state, trace, answerability_summary, _ = self._execute_action(state, memory, AgentAction("evaluate_answerability", "post_reformulation_answerability_check", expected_state_transition="answerability_decided"))
                    traces.append(trace)
                    trusted = search_summary.get("trusted_evidence") or {}
                    current_records = records_from_trusted_evidence(trusted, source_round=2, query_digest=query_digest(query.query.strip()))
                    second_records.extend(current_records)
                    rounds.append(
                        build_retrieval_round(
                            round_index=2,
                            query=query.query,
                            query_source="model_reformulation",
                            result_count=int(search_summary.get("result_count") or 0),
                            previous_document_ids={record.document_id for record in first_records},
                            previous_evidence_ids={record.evidence_identity for record in first_records},
                            document_ids=trusted.get("document_ids") or [],
                            evidence_ids=[record.evidence_identity for record in current_records],
                            answerability_before=before_answerability,
                            answerability_after=(answerability_summary.get("answerability") or {}).get("status"),
                            latency_ms=search_latency,
                        ).to_dict()
                    )
            except AgentRuntimeError as exc:
                plan_trace = self.reformulator.last_trace
                state, trace = self._finish_abstain(state, reason_code=exc.failure.code, failure=exc.failure)
                traces.append(trace)

        merge = merge_evidence_rounds(first_records, second_records, max_items=5)
        if state.status == "running":
            state, trace = self._finish_abstain(state, reason_code="retrieval_planning_experiment_terminal")
            traces.append(trace)
        return {
            "schema_version": "opk-rag.task0067-c2-run-result.v1",
            "run_id": run_id,
            "sample_id": sample_id,
            "status": "completed" if state.status == "running" else state.status,
            "final_action": "abstain" if state.status == "abstained" else None,
            "state": state.to_dict(),
            "trace_events": traces,
            "retrieval_rounds": rounds,
            "query_reformulation_trace": plan_trace,
            "query_privacy_diagnostics": query_privacy,
            "evidence_merge": merge,
            "reformulation_count": reformulation_count,
            "second_round_search_count": sum(1 for row in rounds if row["round_index"] == 2),
        }

    def _execute_action(self, state: AgentState, memory: dict[str, Any], action: AgentAction) -> tuple[AgentState, dict[str, Any], dict[str, Any], int]:
        if state.step_index >= self.config.max_steps:
            raise AgentRuntimeError("agent_budget_exhausted", "Maximum step budget exhausted.", origin="agent_runtime")
        before = state_digest(state)
        started = time.monotonic()
        failure: AgentFailure | None = None
        summary: dict[str, Any] = {}
        try:
            self.tool_registry.validate_action(action, state, self.config)
            _payload, summary = self.tool_registry.execute(action, state, memory, self.config)
            state = self._apply_tool_result(state, action, summary)
        except AgentRuntimeError as exc:
            failure = exc.failure
            state = replace(state, status="failed", final_action="failure", failure=failure, step_index=state.step_index + 1)
            summary = {"failure": failure.to_dict()}
            raise
        finally:
            latency_ms = max(0, int((time.monotonic() - started) * 1000))
        trace = AgentTraceEvent(
            run_id=state.run_id,
            sample_id=state.sample_id,
            policy_name=RETRIEVAL_PLANNING_POLICY_NAME,
            policy_version=RETRIEVAL_PLANNING_POLICY_VERSION,
            runtime_contract_version=self.config.contract_version,
            tool_registry_version=self.tool_registry.version,
            step_index=state.step_index - 1,
            state_before_digest=before,
            action=action,
            tool_request=summarize_for_trace({"tool_name": ACTION_TO_TOOL.get(action.action, action.action)}),
            tool_response_summary=summarize_for_trace(summary),
            state_after_digest=state_digest(state),
            latency_ms=latency_ms,
            failure=failure,
            timestamp=utc_now(),
        ).to_dict()
        return state, trace, summary, latency_ms

    def _apply_tool_result(self, state: AgentState, action: AgentAction, summary: dict[str, Any]) -> AgentState:
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
        if action.action == "evaluate_answerability":
            payload = summary.get("answerability") or {}
            return replace(
                next_state,
                answerability_state={
                    "status": payload.get("status"),
                    "reason_code": payload.get("reason_code"),
                    "evidence_count": payload.get("evidence_count"),
                    "evidence_chunk_ids": payload.get("evidence_chunk_ids", []),
                },
            )
        raise AgentRuntimeError("agent_state_transition_failure", f"Unhandled retrieval planning action: {action.action}", origin="agent_runtime")

    def _can_reformulate(self, state: AgentState, reformulation_count: int) -> bool:
        return (
            not state.is_terminal
            and reformulation_count < self.config.max_reformulations
            and state.search_round_count < self.config.max_search_rounds
            and state.tool_call_count < self.config.max_tool_calls
            and state.step_index < self.config.max_steps
        )

    def _finish_abstain(self, state: AgentState, *, reason_code: str, failure: AgentFailure | None = None) -> tuple[AgentState, dict[str, Any]]:
        before = state_digest(state)
        action = AgentAction("finish_abstain", reason_code, expected_state_transition="terminal_abstain")
        next_state = replace(state, step_index=state.step_index + 1, status="abstained", final_action="abstain", failure=failure or state.failure)
        trace = AgentTraceEvent(
            run_id=next_state.run_id,
            sample_id=next_state.sample_id,
            policy_name=RETRIEVAL_PLANNING_POLICY_NAME,
            policy_version=RETRIEVAL_PLANNING_POLICY_VERSION,
            runtime_contract_version=self.config.contract_version,
            tool_registry_version=self.tool_registry.version,
            step_index=next_state.step_index - 1,
            state_before_digest=before,
            action=action,
            tool_request={},
            tool_response_summary={},
            state_after_digest=state_digest(next_state),
            latency_ms=0,
            timestamp=utc_now(),
        ).to_dict()
        return next_state, trace
