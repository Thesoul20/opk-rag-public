from __future__ import annotations

from typing import Any

from opk_rag.agent.contracts import AGENT_TRACE_CONTRACT_VERSION
from opk_rag.agent.errors import AgentFailure
from opk_rag.agent.tool_registry import ACTION_TO_TOOL, ALLOWED_ARGS
from opk_rag.agent.transition import RUNTIME_TRANSITION_TRACE_VERSION


def replay_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    terminal_count = 0
    previous_after = None
    previous_step = None
    tool_calls = 0
    max_tool_calls = 6
    max_steps = 8

    for index, event in enumerate(events):
        if event.get("policy_name") == "model_policy_query_reformulation_v1":
            max_tool_calls = 8
            max_steps = 12
        if event.get("schema_version") != AGENT_TRACE_CONTRACT_VERSION:
            findings.append(_finding(index, "trace_integrity_failure", "Unknown trace schema version."))
        step = event.get("step_index")
        if previous_step is not None and step != previous_step + 1:
            findings.append(_finding(index, "trace_integrity_failure", "Trace step order is not continuous."))
        if previous_after is not None and event.get("state_before_digest") != previous_after:
            findings.append(_finding(index, "trace_integrity_failure", "State digest chain is not continuous."))
        action = event.get("action") or {}
        action_name = action.get("action")
        transition = event.get("runtime_transition") or {}
        if transition:
            _validate_runtime_transition(findings, index, event, transition, action_name)
        if action_name not in ALLOWED_ARGS:
            findings.append(_finding(index, "agent_invalid_action", "Unknown action in trace."))
        else:
            args = action.get("arguments") or {}
            unknown = _unknown_argument_fields(action_name, args)
            if unknown:
                findings.append(_finding(index, "agent_tool_input_contract_failure", "Action arguments violate replay contract."))
            if action_name in ACTION_TO_TOOL:
                tool_calls += 1
                request = event.get("tool_request") or {}
                if request.get("tool_name") != ACTION_TO_TOOL[action_name]:
                    findings.append(_finding(index, "trace_integrity_failure", "Tool request does not match action mapping."))
                diagnostics = ((event.get("action") or {}).get("arguments") or {}).get("diagnostics") or {}
                if action_name == "search" and "query_digest" not in diagnostics:
                    findings.append(_finding(index, "trace_integrity_failure", "Search trace is missing query digest diagnostics."))
                if action_name == "search" and "query" in ((event.get("action") or {}).get("arguments") or {}):
                    findings.append(_finding(index, "trace_privacy_failure", "Search trace stores a full query."))
            if action_name in {"finish_answer", "finish_abstain", "finish_failure"}:
                terminal_count += 1
                if index != len(events) - 1:
                    findings.append(_finding(index, "trace_integrity_failure", "Terminal action is followed by more trace events."))
        if action_name == "finish_answer":
            response = events[index - 1].get("tool_response_summary", {}) if index > 0 else {}
            if not _previous_grounding_valid(events, index):
                findings.append(_finding(index, "agent_state_transition_failure", "Final answer was returned without grounding verification."))
        if step is not None and step >= max_steps:
            findings.append(_finding(index, "agent_budget_exhausted", "Trace exceeded maximum step budget."))
        previous_after = event.get("state_after_digest")
        previous_step = step

    if not events:
        findings.append(_finding(0, "trace_integrity_failure", "Trace is empty."))
    if terminal_count != 1:
        findings.append(_finding(len(events), "trace_integrity_failure", "Trace must contain exactly one terminal action."))
    if tool_calls > max_tool_calls:
        findings.append(_finding(len(events), "agent_budget_exhausted", "Trace exceeded maximum tool-call budget."))

    return {
        "schema_version": "opk-rag.agent-trace-replay-report.v1",
        "status": "pass" if not findings else "fail",
        "event_count": len(events),
        "tool_call_count": tool_calls,
        "terminal_state_count": terminal_count,
        "remote_provider_called": False,
        "findings": findings,
    }


def _validate_runtime_transition(
    findings: list[dict[str, Any]],
    index: int,
    event: dict[str, Any],
    transition: dict[str, Any],
    action_name: str | None,
) -> None:
    if transition.get("contract_version") != RUNTIME_TRANSITION_TRACE_VERSION:
        findings.append(_finding(index, "runtime_transition_trace_failure", "Unknown runtime transition trace version."))
    if transition.get("state_before_digest") != event.get("state_before_digest") or transition.get("state_after_digest") != event.get("state_after_digest"):
        findings.append(_finding(index, "runtime_transition_trace_failure", "Transition trace digest does not match event digest."))
    available = transition.get("available_actions")
    if not isinstance(available, list):
        findings.append(_finding(index, "runtime_transition_trace_failure", "Transition trace is missing available_actions."))
        available = []
    if action_name and action_name not in available and action_name != "finish_failure":
        findings.append(_finding(index, "agent_policy_illegal_transition", "Selected action is not in runtime available_actions."))
    generation_eligible = transition.get("generation_eligible") is True
    if generation_eligible and "generate_grounded_answer" not in available and transition.get("phase_before") == "generation_pending":
        findings.append(_finding(index, "runtime_transition_exposure_failure", "Eligible generation state did not expose generate_grounded_answer."))
    if not generation_eligible and "generate_grounded_answer" in available:
        findings.append(_finding(index, "runtime_transition_exposure_failure", "Non-eligible state exposed generate_grounded_answer."))
    if generation_eligible and action_name == "finish_abstain" and ((event.get("action") or {}).get("reason_code") not in {"budget_exhausted", "provider_unavailable", "runtime_contract_failure", "explicit_safety_block"}):
        findings.append(_finding(index, "premature_terminal_transition", "Eligible state finished abstain without a structured safety reason."))
    if transition.get("phase_before") == "verification_pending" and action_name != "verify_grounding":
        findings.append(_finding(index, "agent_state_transition_failure", "Generation completed state must verify grounding next."))


def _previous_grounding_valid(events: list[dict[str, Any]], index: int) -> bool:
    for event in reversed(events[:index]):
        action = event.get("action") or {}
        if action.get("action") != "verify_grounding":
            continue
        grounding = (event.get("tool_response_summary") or {}).get("grounding") or {}
        return grounding.get("valid") is True
    return False


def _unknown_argument_fields(action_name: str, args: dict[str, Any]) -> set[str]:
    diagnostics = args.get("diagnostics") if isinstance(args, dict) else None
    if isinstance(diagnostics, dict):
        fields = set(str(field) for field in diagnostics.get("argument_field_names") or [])
        if action_name == "search" and diagnostics.get("query_non_empty") is not True:
            fields.add("<missing_non_empty_query>")
        return fields - ALLOWED_ARGS[action_name]
    return set(args) - ALLOWED_ARGS[action_name]


def _finding(index: int, code: str, message: str) -> dict[str, Any]:
    failure = AgentFailure(code=code, message=message, origin="agent_runtime")
    return {"event_index": index, **failure.to_dict()}
