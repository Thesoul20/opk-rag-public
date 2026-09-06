from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from opk_rag.agentic_v2.action import AgentAction
from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.guard import AgentGuard
from opk_rag.agentic_v2.observation import AgentObservation, build_agent_observation
from opk_rag.agentic_v2.policy_errors import AgentPolicyRuntimeError
from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime
from opk_rag.agentic_v2.run_contracts import AgentRunResult, AgentRunStatus, AgentStepRecord
from opk_rag.agentic_v2.run_metrics import AgentRunMetrics
from opk_rag.agentic_v2.run_trace import AgentRunTrace, run_semantic_digest
from opk_rag.agentic_v2.runtime_context import AgentRuntimeContext
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.state_transition import AgentStateTransitionEngine
from opk_rag.agentic_v2.state_transition_contracts import AppliedExecutionRegistry
from opk_rag.agentic_v2.termination import AgentTermination
from opk_rag.agentic_v2.tool_executor import GuardedToolExecutor
from opk_rag.agentic_v2.tools import EvidenceInspectionSnapshot


@dataclass(frozen=True)
class AgenticV2RunExecution:
    result: AgentRunResult
    trace: AgentRunTrace


class AgenticV2Runtime:
    """Bounded orchestration only: Observe -> Decide -> Guard -> Act -> Transition -> Observe."""

    def __init__(
        self,
        *,
        policy: LLMAgentPolicyRuntime,
        guard: AgentGuard,
        tool_executor: GuardedToolExecutor,
        state_transition: AgentStateTransitionEngine,
        resolver: AllowedActionResolver | None = None,
    ) -> None:
        self.policy = policy
        self.guard = guard
        self.tool_executor = tool_executor
        self.state_transition = state_transition
        self.resolver = resolver or AllowedActionResolver()
        self.metrics = AgentRunMetrics()

    def run(self, *, initial_state: AgentState, context: AgentRuntimeContext) -> AgenticV2RunExecution:
        started = perf_counter()
        initial_state_digest = stable_digest(initial_state)
        state = initial_state
        if context.tool_context.run_id != state.run_id:
            return self._finish(
                initial_state_digest=initial_state_digest,
                state=state,
                observation=build_agent_observation(state, authority=context.budget),
                termination=AgentTermination(kind="runtime_failure", reason_code="runtime_identity_mismatch", is_runtime_failure=True),
                run_status="runtime_failure",
                step_records=(),
                action_sequence=(),
                policy_decision_count=0,
                tool_execution_count=0,
                transition_count=0,
                started=started,
            )
        if state.status != "running":
            termination, status = _termination_from_terminal_state(state)
            return self._finish(
                initial_state_digest=initial_state_digest,
                state=state,
                observation=build_agent_observation(state, authority=context.budget),
                termination=termination,
                run_status=status,
                step_records=(),
                action_sequence=(),
                policy_decision_count=0,
                tool_execution_count=0,
                transition_count=0,
                started=started,
            )

        observation = build_agent_observation(state, authority=context.budget)
        registry = AppliedExecutionRegistry()
        action_history: tuple[AgentAction, ...] = ()
        action_sequence: list[str] = []
        step_records: list[AgentStepRecord] = []
        policy_decision_count = 0
        tool_execution_count = 0
        transition_count = 0
        last_tool_result = None

        while True:
            if state.status != "running":
                termination, status = _termination_from_terminal_state(state)
                return self._finish(initial_state_digest, state, observation, termination, status, tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            if state.run_id != observation.run_id or state.current_query != observation.current_query:
                termination = AgentTermination(kind="runtime_failure", reason_code="state_observation_mismatch", is_runtime_failure=True)
                return self._finish(initial_state_digest, state, observation, termination, "runtime_failure", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            if state.remaining_budget(context.budget).steps <= 0:
                termination = AgentTermination(kind="budget_exhausted", reason_code="agent_step_budget_exhausted", is_runtime_failure=False)
                return self._finish(initial_state_digest, state, observation, termination, "budget_exhausted", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)

            allowed_actions = self.resolver.resolve(state=state, observation=observation, budget=context.budget, action_history=action_history)
            observation_digest = stable_digest(observation)
            step_index = state.step_index

            try:
                policy_result = self.policy.decide(observation=observation, allowed_actions=allowed_actions)
            except AgentPolicyRuntimeError as exc:
                policy_decision_count += 1
                step_records.append(AgentStepRecord(
                    step_index=step_index,
                    observation_digest=observation_digest,
                    allowed_actions=allowed_actions,
                    policy_validation_status="failed",
                    policy_failure_code=exc.failure.code,
                ))
                termination, status = _termination_from_policy_failure(exc.failure.code)
                return self._finish(initial_state_digest, state, observation, termination, status, tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            policy_decision_count += 1
            decision = policy_result.decision
            decision_digest = stable_digest(decision)
            policy_ids = tuple(trace.decision_id for trace in policy_result.trace)

            authorization = self.guard.validate(
                state=state,
                observation=observation,
                decision=decision,
                budget=context.budget,
                action_history=action_history,
            )
            guard_outcome = authorization.guard_decision.decision
            if guard_outcome == "reject":
                step_records.append(AgentStepRecord(
                    step_index=step_index,
                    observation_digest=observation_digest,
                    allowed_actions=allowed_actions,
                    policy_decision_digest=decision_digest,
                    policy_decision_ids=policy_ids,
                    policy_validation_status="passed",
                    proposed_action=decision.proposed_action,
                    guard_outcome="reject",
                    guard_reason_code=authorization.guard_decision.reason_code,
                    guard_decision_id=authorization.trace.decision_id,
                ))
                termination = AgentTermination(kind="guard_rejected", reason_code=authorization.guard_decision.reason_code, is_runtime_failure=False)
                return self._finish(initial_state_digest, state, observation, termination, "guard_rejected", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            if guard_outcome == "terminate":
                termination = authorization.termination or AgentTermination(kind="runtime_failure", reason_code="guard_terminated_without_contract", is_runtime_failure=True)
                step_records.append(AgentStepRecord(
                    step_index=step_index,
                    observation_digest=observation_digest,
                    allowed_actions=allowed_actions,
                    policy_decision_digest=decision_digest,
                    policy_decision_ids=policy_ids,
                    policy_validation_status="passed",
                    proposed_action=decision.proposed_action,
                    guard_outcome="terminate",
                    guard_reason_code=authorization.guard_decision.reason_code,
                    guard_decision_id=authorization.trace.decision_id,
                ))
                return self._finish(initial_state_digest, state, observation, termination, _run_status_from_termination(termination), tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)

            action = authorization.validated_action
            if action is None:
                termination = AgentTermination(kind="runtime_failure", reason_code="guard_authorized_without_action", is_runtime_failure=True)
                return self._finish(initial_state_digest, state, observation, termination, "runtime_failure", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)

            step_tool_context = _tool_context_for_step(context.tool_context, state=state, action=action, last_tool_result=last_tool_result)
            try:
                tool_execution = self.tool_executor.execute(authorization=authorization, state=state, context=step_tool_context)
            except Exception:
                step_records.append(AgentStepRecord(
                    step_index=step_index,
                    observation_digest=observation_digest,
                    allowed_actions=allowed_actions,
                    policy_decision_digest=decision_digest,
                    policy_decision_ids=policy_ids,
                    policy_validation_status="passed",
                    proposed_action=decision.proposed_action,
                    guard_outcome=guard_outcome,
                    guard_reason_code=authorization.guard_decision.reason_code,
                    guard_decision_id=authorization.trace.decision_id,
                    validated_action_digest=stable_digest(action),
                ))
                termination = AgentTermination(kind="runtime_failure", reason_code="tool_executor_failure", is_runtime_failure=True)
                return self._finish(initial_state_digest, state, observation, termination, "runtime_failure", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            tool_execution_count += 1
            action_sequence.append(action.action)
            tool_result = tool_execution.result
            signals = context.observation_signal_resolver(state, action, tool_result) if context.observation_signal_resolver else None
            transition = self.state_transition.transition(
                state=state,
                action=action,
                tool_result=tool_result,
                budget=context.budget,
                applied_execution_registry=registry,
                observation_signals=signals,
            )
            transition_count += 1
            step_records.append(AgentStepRecord(
                step_index=step_index,
                observation_digest=observation_digest,
                allowed_actions=allowed_actions,
                policy_decision_digest=decision_digest,
                policy_decision_ids=policy_ids,
                policy_validation_status="passed",
                proposed_action=decision.proposed_action,
                guard_outcome=guard_outcome,
                guard_reason_code=authorization.guard_decision.reason_code,
                guard_decision_id=authorization.trace.decision_id,
                validated_action_digest=stable_digest(action),
                execution_id=tool_result.execution_id,
                tool_status=tool_result.status,
                transition_status=transition.result.transition_status,
                next_state_digest=transition.result.next_state_digest,
                next_observation_digest=transition.result.next_observation_digest,
            ))

            if transition.result.transition_status == "rejected":
                termination = AgentTermination(kind="runtime_failure", reason_code=transition.result.reason_code, is_runtime_failure=True)
                return self._finish(initial_state_digest, state, observation, termination, "runtime_failure", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)

            state = transition.result.next_state
            observation = transition.result.next_observation
            registry = transition.result.applied_execution_registry
            action_history = (*action_history, action)
            last_tool_result = tool_result

            if transition.result.termination is not None:
                termination = transition.result.termination
                return self._finish(initial_state_digest, state, observation, termination, _run_status_from_termination(termination), tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)
            if state.remaining_budget(context.budget).steps <= 0:
                termination = AgentTermination(kind="budget_exhausted", reason_code="agent_step_budget_exhausted", is_runtime_failure=False)
                return self._finish(initial_state_digest, state, observation, termination, "budget_exhausted", tuple(step_records), tuple(action_sequence), policy_decision_count, tool_execution_count, transition_count, started)

    def _finish(
        self,
        initial_state_digest: str,
        state: AgentState,
        observation: AgentObservation,
        termination: AgentTermination,
        run_status: AgentRunStatus,
        step_records: tuple[AgentStepRecord, ...],
        action_sequence: tuple[str, ...],
        policy_decision_count: int,
        tool_execution_count: int,
        transition_count: int,
        started: float,
    ) -> AgenticV2RunExecution:
        final_state_digest = stable_digest(state)
        final_observation_digest = stable_digest(observation)
        semantic_digest = run_semantic_digest(
            initial_state_digest=initial_state_digest,
            final_state_digest=final_state_digest,
            final_observation_digest=final_observation_digest,
            run_status=run_status,
            termination=termination,
            step_records=step_records,
        )
        result = AgentRunResult(
            run_id=state.run_id,
            initial_state_digest=initial_state_digest,
            final_state=state,
            final_state_digest=final_state_digest,
            final_observation=observation,
            final_observation_digest=final_observation_digest,
            termination=termination,
            step_count=state.step_index,
            policy_decision_count=policy_decision_count,
            tool_execution_count=tool_execution_count,
            transition_count=transition_count,
            action_sequence=action_sequence,
            run_status=run_status,
            step_records=step_records,
            agent_run_digest=semantic_digest,
        )
        trace = AgentRunTrace(
            run_id=state.run_id,
            initial_state_digest=initial_state_digest,
            final_state_digest=final_state_digest,
            final_observation_digest=final_observation_digest,
            run_status=run_status,
            termination=termination,
            step_records=step_records,
            semantic_digest=semantic_digest,
        )
        latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        self.metrics.record(
            run_status=run_status,
            steps=state.step_index,
            policy_decisions=policy_decision_count,
            tool_executions=tool_execution_count,
            transitions=transition_count,
            action_sequence=action_sequence,
            latency_ms=latency_ms,
        )
        return AgenticV2RunExecution(result=result, trace=trace)


def _tool_context_for_step(base_context, *, state: AgentState, action: AgentAction, last_tool_result):
    if action.action != "inspect_evidence" or last_tool_result is None:
        return base_context
    snapshot = EvidenceInspectionSnapshot(
        candidate_ids=last_tool_result.candidate_ids,
        evidence_ids=last_tool_result.evidence_ids,
        source_count=state.source_count,
        coverage=state.evidence_coverage,
    )
    return replace(base_context, evidence_snapshot=snapshot)


def _termination_from_policy_failure(code: str) -> tuple[AgentTermination, AgentRunStatus]:
    if code in {"agent_policy_timeout", "agent_policy_provider_failure"}:
        return AgentTermination(kind="runtime_failure", reason_code=code, is_runtime_failure=True), "runtime_failure"
    return AgentTermination(kind="invalid_policy_output", reason_code=code, is_runtime_failure=False), "invalid_policy_output"


def _termination_from_terminal_state(state: AgentState) -> tuple[AgentTermination, AgentRunStatus]:
    reason = state.termination_reason or "agent_already_terminal"
    if state.status == "finished":
        return AgentTermination(kind="finished_answer", reason_code=reason, is_runtime_failure=False), "finished"
    if state.status == "abstained":
        return AgentTermination(kind="abstained", reason_code=reason, is_runtime_failure=False), "abstained"
    return AgentTermination(kind="runtime_failure", reason_code=reason, is_runtime_failure=True), "runtime_failure"


def _run_status_from_termination(termination: AgentTermination) -> AgentRunStatus:
    mapping: dict[str, AgentRunStatus] = {
        "finished_answer": "finished",
        "abstained": "abstained",
        "budget_exhausted": "budget_exhausted",
        "invalid_policy_output": "invalid_policy_output",
        "guard_rejected": "guard_rejected",
        "runtime_failure": "runtime_failure",
        "insufficient_evidence": "abstained",
    }
    return mapping.get(termination.kind, "runtime_failure")
