from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from uuid import uuid4

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.agentic_v2.guard import GuardValidationResult
from opk_rag.agentic_v2.state import AgentState
from opk_rag.agentic_v2.tool_contracts import AgentToolResult
from opk_rag.agentic_v2.tool_metrics import AgentToolMetrics
from opk_rag.agentic_v2.tool_registry import AgentToolRegistry, ToolRegistryError
from opk_rag.agentic_v2.tool_trace import AgentToolExecutionTrace, ids_digest
from opk_rag.agentic_v2.tools import AgentToolContext


@dataclass(frozen=True)
class AgentToolExecutionResult:
    result: AgentToolResult
    trace: AgentToolExecutionTrace


class GuardedToolExecutor:
    """Execute exactly one Guard-authorized AgentAction. No retries, fallback, or state transition."""

    def __init__(self, *, registry: AgentToolRegistry) -> None:
        self.registry = registry
        self.metrics = AgentToolMetrics()

    def execute(self, *, authorization: GuardValidationResult, state: AgentState, context: AgentToolContext) -> AgentToolExecutionResult:
        if not isinstance(authorization, GuardValidationResult):
            raise TypeError("GuardedToolExecutor requires GuardValidationResult; AgentDecision/raw action bypass is forbidden")
        guard = authorization.guard_decision
        if guard.decision not in {"allow", "modify"} or guard.validated_action is None:
            raise PermissionError("guard_authorization_missing")
        action = guard.validated_action
        if state.run_id != context.run_id:
            raise PermissionError("runtime_identity_mismatch")
        execution_id = str(uuid4())
        bound_context = replace(context, execution_id=execution_id)
        before = stable_digest(state)
        started = perf_counter()
        try:
            tool = self.registry.resolve(action.action)
            result = tool.execute(action=action, context=bound_context)
        except ToolRegistryError:
            result = AgentToolResult(run_id=state.run_id, execution_id=execution_id, action=action.action, status="failed", failure_code="tool_not_registered")
            tool_name, tool_version = "unresolved", "unknown"
        except Exception:
            result = AgentToolResult(run_id=state.run_id, execution_id=execution_id, action=action.action, status="failed", failure_code="tool_runtime_failure")
            tool_name, tool_version = getattr(locals().get("tool"), "definition", None).tool_name if locals().get("tool") else "unresolved", getattr(locals().get("tool"), "definition", None).tool_version if locals().get("tool") else "unknown"
        else:
            tool_name, tool_version = tool.definition.tool_name, tool.definition.tool_version
        latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        after = stable_digest(state)
        if after != before:
            # The executor never authorizes state mutation; fail closed if an adapter violated the boundary.
            result = AgentToolResult(run_id=state.run_id, execution_id=execution_id, action=action.action, status="failed", failure_code="tool_state_mutation_detected")
        trace = AgentToolExecutionTrace(
            run_id=state.run_id, execution_id=execution_id, action=action.action, action_digest=stable_digest(action),
            tool_name=tool_name, tool_version=tool_version, execution_status=result.status,
            candidate_count=result.candidate_count, evidence_count=result.evidence_count,
            candidate_ids_digest=ids_digest(result.candidate_ids), evidence_ids_digest=ids_digest(result.evidence_ids),
            latency_ms=latency_ms, failure_code=result.failure_code, state_mutated=False,
        )
        self.metrics.record(action=action.action, status=result.status, candidate_count=result.candidate_count, latency_ms=latency_ms)
        return AgentToolExecutionResult(result=result, trace=trace)
