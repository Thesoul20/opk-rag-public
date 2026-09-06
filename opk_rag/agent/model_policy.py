from __future__ import annotations

import hashlib
from typing import Any

from opk_rag.agent.contracts import AgentAction, AgentRuntimeConfig, AgentState
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.policy import AgentPolicy
from opk_rag.agent.policy_contracts import AGENT_POLICY_PROMPT_VERSION, POLICY_DECISION_SCHEMA
from opk_rag.agent.policy_metrics import PolicyCallRecord, PolicyMetricsRecorder
from opk_rag.agent.policy_prompt import build_governed_state_view, build_policy_prompt, build_policy_repair_prompt, policy_prompt_digest
from opk_rag.agent.policy_provider import AgentPolicyProvider
from opk_rag.agent.policy_validation import (
    action_argument_diagnostics,
    decision_to_agent_action,
    is_structural_policy_error,
    parse_policy_decision,
    policy_argument_diagnostics,
    redacted_policy_output,
    validate_policy_decision,
)


class ModelDrivenAgentPolicy(AgentPolicy):
    name = "model_driven_policy_v1"
    version = "1.0.0"

    def __init__(
        self,
        *,
        provider: AgentPolicyProvider,
        config: AgentRuntimeConfig | None = None,
        max_structural_repairs: int = 1,
    ) -> None:
        self.provider = provider
        self.config = config or AgentRuntimeConfig(model_planning_enabled=True, policy_name=self.name, policy_version=self.version)
        self.max_structural_repairs = max_structural_repairs
        self.metrics = PolicyMetricsRecorder()
        self.last_decision_trace: dict[str, Any] | None = None
        self._current_decision_traces: list[dict[str, Any]] = []

    def next_action(self, state: AgentState) -> AgentAction:
        self._current_decision_traces = []
        state_view = build_governed_state_view(state, self.config)
        prompt = build_policy_prompt(state_view=state_view)
        repair_context = None
        last_error: AgentRuntimeError | None = None
        for attempt in range(self.max_structural_repairs + 1):
            is_repair = attempt > 0
            result = self.provider.decide(
                prompt=prompt,
                state_view=state_view,
                output_schema=POLICY_DECISION_SCHEMA,
                repair_context=repair_context,
            )
            if result.error_code is not None:
                error = AgentRuntimeError(result.error_code, result.error_message or "Policy provider failed.", origin="provider")
                self._record(result, is_repair=is_repair, status="failure", failure_code=error.failure.code, prompt=prompt, state_view=state_view)
                raise error
            diagnostics: dict[str, Any] = {
                "provider_raw_response_arguments": _raw_output_argument_diagnostics(result.raw_output),
            }
            parsed: dict[str, Any] | None = None
            try:
                parsed = parse_policy_decision(result.raw_output)
                diagnostics["parsed_output_arguments"] = policy_argument_diagnostics(parsed)
                decision = validate_policy_decision(parsed, available_actions=list(state_view["available_actions"]))
                diagnostics["policy_decision_arguments"] = policy_argument_diagnostics(decision.to_dict())
                action = decision_to_agent_action(decision)
                diagnostics["agent_action_arguments"] = action_argument_diagnostics(action.action, action.arguments)
            except AgentRuntimeError as exc:
                last_error = exc
                self._record(result, is_repair=is_repair, status="failure", failure_code=exc.failure.code, prompt=prompt, state_view=state_view, diagnostics=diagnostics)
                if is_structural_policy_error(exc) and attempt < self.max_structural_repairs:
                    failure = exc.failure.to_dict()
                    previous_output = redacted_policy_output(parsed) if parsed is not None else _redacted_raw_failure(result.raw_output)
                    prompt = build_policy_repair_prompt(state_view=state_view, previous_output=previous_output, failure=failure)
                    repair_context = {"failure": failure, "instruction": "Return exactly one repaired policy decision JSON object."}
                    continue
                if is_structural_policy_error(exc) and attempt >= self.max_structural_repairs:
                    raise AgentRuntimeError("agent_policy_repair_exhausted", "Policy structural repair budget exhausted.", origin="agent_policy", detail={"last_failure": exc.failure.to_dict()}) from exc
                raise
            self._record(result, is_repair=is_repair, status="success", failure_code=None, prompt=prompt, state_view=state_view, decision=decision.to_dict(), diagnostics=diagnostics)
            return action
        assert last_error is not None
        raise last_error

    def _record(
        self,
        result,
        *,
        is_repair: bool,
        status: str,
        failure_code: str | None,
        prompt: str,
        state_view: dict[str, Any],
        decision: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.metrics.record(
            PolicyCallRecord(
                provider_name=result.provider_name,
                model_name=result.model_name,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                repair_attempt=is_repair,
                status=status,
                failure_code=failure_code,
            )
        )
        record = {
            "policy_provider": result.to_trace_dict(),
            "policy_prompt_version": AGENT_POLICY_PROMPT_VERSION,
            "policy_prompt_digest": policy_prompt_digest(prompt),
            "state_view_contract_version": state_view.get("contract_version"),
            "available_actions": state_view.get("available_actions"),
            "repair_attempt": is_repair,
            "decision": redacted_policy_output(decision) if decision is not None else None,
            "argument_diagnostics": diagnostics or {},
            "failure_code": failure_code,
        }
        self._current_decision_traces.append(record)
        self.last_decision_trace = {**record, "attempt_history": list(self._current_decision_traces)}


def _raw_output_argument_diagnostics(raw_output: str) -> dict[str, Any]:
    try:
        parsed = parse_policy_decision(raw_output)
    except AgentRuntimeError:
        return {
            "parseable_json_object": False,
            "raw_output_length": len(raw_output),
            "raw_output_digest": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        }
    return {"parseable_json_object": True, **policy_argument_diagnostics(parsed)}


def _redacted_raw_failure(raw_output: str) -> dict[str, Any]:
    return {
        "parseable_json_object": False,
        "raw_output_length": len(raw_output),
        "raw_output_digest": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
    }
