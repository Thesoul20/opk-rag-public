from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.allowed_actions import AllowedActionResolver
from opk_rag.agentic_v2.base import StrictContract
from opk_rag.agentic_v2.guard import AgentGuard
from opk_rag.agentic_v2.observation import AgentObservation
from opk_rag.agentic_v2.policy_errors import AgentPolicyRuntimeError
from opk_rag.agentic_v2.policy_runtime import LLMAgentPolicyRuntime
from opk_rag.agentic_v2.runtime_context import AgentRuntimeContext
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.state_transition import AgentStateTransitionEngine
from opk_rag.agentic_v2.state_transition_contracts import AppliedExecutionRegistry
from opk_rag.agentic_v2.tool_executor import GuardedToolExecutor

AGENT_INVOCATION_CONTRACT_VERSION = "opk-rag.agentic-v2.invocation-decision.v1"
RECOVERY_ACTIONS: tuple[ActionName, ...] = (
    "hybrid_search",
    "structure_search",
    "graph_search",
    "rewrite_query",
)
InvocationReason = Literal[
    "query_ambiguity_detected",
    "retrieval_evidence_insufficient",
    "retrieval_signal_conflict",
    "structure_recovery_opportunity",
    "graph_recovery_opportunity",
    "query_rewrite_opportunity",
    "clear_answerable_no_agent_needed",
    "clear_abstain_no_agent_needed",
    "budget_prevents_agent_recovery",
]
HardTerminalSignal = Literal["answerable", "abstain", "none"]


class AgentInvocationDecision(StrictContract):
    contract_version: Literal[AGENT_INVOCATION_CONTRACT_VERSION] = AGENT_INVOCATION_CONTRACT_VERSION
    invoke_llm: bool
    reason_code: InvocationReason
    runtime_signal_summary: tuple[str, ...] = Field(default=(), max_length=12)
    eligible_recovery_actions: tuple[ActionName, ...] = ()
    hard_terminal_signal: HardTerminalSignal = "none"
    ambiguity_detected: bool = False
    recovery_opportunity_detected: bool = False


class NecessaryLLMInvocationGate:
    """TASK-0247 evaluation-only deterministic runtime gate; evaluator Gold is not an input."""

    _ambiguity_markers = (
        "之前那个", "前面那个", "上面那个", "这个问题", "那个问题",
        "后来怎么样", "这个呢", "那个呢", "它呢", "这件事",
    )
    _graph_markers = ("链接", "关联", "显式关系", "跨文档", "关系图", "路线图")
    _structure_markers = ("章节", "同一文档", "上下文", "相关测试笔记", "技术路线")

    def decide(self, *, observation: AgentObservation, allowed_actions: tuple[ActionName, ...] | None = None) -> AgentInvocationDecision:
        allowed = allowed_actions if allowed_actions is not None else RECOVERY_ACTIONS
        eligible = tuple(action for action in RECOVERY_ACTIONS if action in allowed)
        query = observation.current_query
        ambiguity = any(marker in query for marker in self._ambiguity_markers)
        partial = observation.answerability_status == "partially_answerable"
        insufficient = observation.answerability_status in {"unknown", "insufficient_evidence", "unanswerable"} or observation.evidence_count <= 0
        graph_opportunity = any(marker in query for marker in self._graph_markers) and not observation.graph_relation_available
        structure_opportunity = any(marker in query for marker in self._structure_markers) and not observation.structure_context_available
        recovery = partial or insufficient or graph_opportunity or structure_opportunity
        signals = (
            f"answerability={observation.answerability_status}",
            f"candidates={observation.candidate_count}",
            f"evidence={observation.evidence_count}",
            f"structure_context={str(observation.structure_context_available).lower()}",
            f"graph_relation={str(observation.graph_relation_available).lower()}",
            f"remaining_steps={observation.remaining_budget.steps}",
            f"remaining_retrieval={observation.remaining_budget.retrieval_calls}",
        )
        if observation.remaining_budget.steps <= 0 or not eligible:
            return AgentInvocationDecision(invoke_llm=False, reason_code="budget_prevents_agent_recovery", runtime_signal_summary=signals, eligible_recovery_actions=eligible, hard_terminal_signal="none", ambiguity_detected=ambiguity, recovery_opportunity_detected=recovery)
        if ambiguity:
            return AgentInvocationDecision(invoke_llm=True, reason_code="query_ambiguity_detected", runtime_signal_summary=signals, eligible_recovery_actions=eligible, ambiguity_detected=True, recovery_opportunity_detected=True)
        if partial:
            return AgentInvocationDecision(invoke_llm=True, reason_code="retrieval_signal_conflict", runtime_signal_summary=signals, eligible_recovery_actions=eligible, recovery_opportunity_detected=True)
        if graph_opportunity and "graph_search" in eligible:
            return AgentInvocationDecision(invoke_llm=True, reason_code="graph_recovery_opportunity", runtime_signal_summary=signals, eligible_recovery_actions=eligible, recovery_opportunity_detected=True)
        if structure_opportunity and "structure_search" in eligible:
            return AgentInvocationDecision(invoke_llm=True, reason_code="structure_recovery_opportunity", runtime_signal_summary=signals, eligible_recovery_actions=eligible, recovery_opportunity_detected=True)
        if insufficient:
            if observation.evidence_count == 0 and observation.answerability_status in {"insufficient_evidence", "unanswerable"} and not (graph_opportunity or structure_opportunity):
                return AgentInvocationDecision(invoke_llm=False, reason_code="clear_abstain_no_agent_needed", runtime_signal_summary=signals, eligible_recovery_actions=eligible, hard_terminal_signal="abstain", recovery_opportunity_detected=False)
            return AgentInvocationDecision(invoke_llm=True, reason_code="retrieval_evidence_insufficient", runtime_signal_summary=signals, eligible_recovery_actions=eligible, recovery_opportunity_detected=True)
        return AgentInvocationDecision(invoke_llm=False, reason_code="clear_answerable_no_agent_needed", runtime_signal_summary=signals, eligible_recovery_actions=eligible, hard_terminal_signal="answerable", ambiguity_detected=False, recovery_opportunity_detected=False)


class BudgetAwareAllowedActionResolver(AllowedActionResolver):
    """TASK-0247 A2 resolver that reserves the final Agent step for terminal action."""
    def resolve(self, *, state, observation, budget=None, action_history=()):
        allowed = super().resolve(state=state, observation=observation, budget=budget, action_history=action_history)
        if state.remaining_budget(budget).steps <= 1:
            return tuple(action for action in allowed if action in {"finish", "abstain"})
        return allowed


class RecoveryOnlyAllowedActionResolver(AllowedActionResolver):
    """TASK-0247 A3 resolver: model proposes recovery only; terminal authority is governed."""
    def resolve(self, *, state, observation, budget=None, action_history=()):
        if state.remaining_budget(budget).steps <= 0:
            return ()
        allowed = super().resolve(state=state, observation=observation, budget=budget, action_history=action_history)
        return tuple(action for action in RECOVERY_ACTIONS if action in allowed)


@dataclass(frozen=True)
class SelectiveRecoveryExecution:
    invocation: AgentInvocationDecision
    terminal: str
    baseline_terminal: str
    policy_called: bool
    policy_failure_code: str | None
    fallback_to_governed: bool
    action_sequence: tuple[str, ...]
    recovery_selected: str | None
    tool_execution_count: int
    retrieval_call_count: int
    graph_call_count: int
    rewrite_count: int
    recovered_answerability_status: str | None
    policy_input_tokens: int
    policy_output_tokens: int
    policy_latency_ms: int


class SelectiveRecoveryController:
    """Evaluation-only one-call Recovery Policy, followed by deterministic governed handoff."""
    def __init__(self, *, policy: LLMAgentPolicyRuntime, gate: NecessaryLLMInvocationGate | None = None, resolver: RecoveryOnlyAllowedActionResolver | None = None, guard: AgentGuard | None = None, tool_executor: GuardedToolExecutor, state_transition: AgentStateTransitionEngine | None = None) -> None:
        self.policy=policy; self.gate=gate or NecessaryLLMInvocationGate(); self.resolver=resolver or RecoveryOnlyAllowedActionResolver(); self.guard=guard or AgentGuard(resolver=self.resolver); self.tool_executor=tool_executor; self.state_transition=state_transition or AgentStateTransitionEngine(resolver=self.resolver)

    def run(self, *, state: AgentState, observation: AgentObservation, context: AgentRuntimeContext, baseline_terminal: str) -> SelectiveRecoveryExecution:
        allowed=self.resolver.resolve(state=state,observation=observation,budget=context.budget); invocation=self.gate.decide(observation=observation,allowed_actions=allowed)
        if not invocation.invoke_llm: return self._result(invocation,baseline_terminal,baseline_terminal,False,None,False,(),None,state,None,0,0,0)
        before=len(self.policy.metrics.records)
        try: decision_result=self.policy.decide(observation=observation,allowed_actions=allowed)
        except AgentPolicyRuntimeError as exc:
            pi,po,pl=self._policy_metrics(before); return self._result(invocation,baseline_terminal,baseline_terminal,True,exc.failure.code,True,(),None,state,None,pi,po,pl)
        decision=decision_result.decision; authorization=self.guard.validate(state=state,observation=observation,decision=decision,budget=context.budget)
        if authorization.guard_decision.decision not in {"allow","modify"} or authorization.validated_action is None:
            pi,po,pl=self._policy_metrics(before); return self._result(invocation,baseline_terminal,baseline_terminal,True,authorization.guard_decision.reason_code,True,(),None,state,None,pi,po,pl)
        action=authorization.validated_action
        try: execution=self.tool_executor.execute(authorization=authorization,state=state,context=context.tool_context)
        except Exception:
            pi,po,pl=self._policy_metrics(before); return self._result(invocation,baseline_terminal,baseline_terminal,True,"tool_executor_failure",True,(),action.action,state,None,pi,po,pl)
        signals=context.observation_signal_resolver(state,action,execution.result) if context.observation_signal_resolver else None
        transition=self.state_transition.transition(state=state,action=action,tool_result=execution.result,budget=context.budget,applied_execution_registry=AppliedExecutionRegistry(),observation_signals=signals)
        if transition.result.transition_status=="rejected":
            pi,po,pl=self._policy_metrics(before); return self._result(invocation,baseline_terminal,baseline_terminal,True,transition.result.reason_code,True,(action.action,),action.action,state,None,pi,po,pl)
        adapter=context.tool_context.retrieval_adapter; recovered=None
        if action.action=="rewrite_query" and adapter is not None and hasattr(adapter,"run_rule_governed"): recovered=adapter.run_rule_governed(transition.result.next_state.current_query)
        elif adapter is not None: recovered=getattr(adapter,"last_rule_result",None)
        terminal=baseline_terminal if recovered is None else recovered.status; fallback=recovered is None; recovered_status=None if recovered is None else recovered.answerability_status
        pi,po,pl=self._policy_metrics(before)
        return self._result(invocation,terminal,baseline_terminal,True,None,fallback,(action.action,),action.action,transition.result.next_state,recovered_status,pi,po,pl)

    def _policy_metrics(self,before):
        rows=self.policy.metrics.records[before:]; return sum(int(x.get("input_tokens") or 0) for x in rows),sum(int(x.get("output_tokens") or 0) for x in rows),sum(int(x.get("latency_ms") or 0) for x in rows)

    @staticmethod
    def _result(invocation,terminal,baseline_terminal,policy_called,failure,fallback,sequence,selected,state,recovered_status,pi,po,pl):
        return SelectiveRecoveryExecution(invocation=invocation,terminal=terminal,baseline_terminal=baseline_terminal,policy_called=policy_called,policy_failure_code=failure,fallback_to_governed=fallback,action_sequence=sequence,recovery_selected=selected,tool_execution_count=len(sequence),retrieval_call_count=state.retrieval_call_count+(1 if selected=="rewrite_query" and recovered_status is not None else 0),graph_call_count=state.graph_call_count,rewrite_count=state.rewrite_count,recovered_answerability_status=recovered_status,policy_input_tokens=pi,policy_output_tokens=po,policy_latency_ms=pl)
