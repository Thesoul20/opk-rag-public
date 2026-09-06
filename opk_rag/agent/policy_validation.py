from __future__ import annotations

import json
import re
import hashlib
from typing import Any

from opk_rag.agent.contracts import AgentAction
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.agent.policy_contracts import (
    AGENT_POLICY_DECISION_CONTRACT_VERSION,
    ALLOWED_POLICY_REASON_CODES,
    EXPECTED_TRANSITION_BY_ACTION,
    PolicyDecision,
)
from opk_rag.agent.tool_registry import ALLOWED_ARGS, FORBIDDEN_ARGS

STRUCTURAL_ERROR_CODES = {
    "agent_policy_malformed_json",
    "agent_policy_contract_failure",
    "agent_policy_argument_failure",
}

_SQL_OR_FILE_PATTERNS = (
    re.compile(r"\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b", re.IGNORECASE),
    re.compile(r"\b(read|open|cat)\s+/(?:home|data|mnt|var|tmp)\b", re.IGNORECASE),
)


def parse_policy_decision(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentRuntimeError("agent_policy_malformed_json", "Policy provider returned malformed JSON.", origin="agent_policy", detail={"json_error": str(exc)}) from exc
    if not isinstance(value, dict):
        raise AgentRuntimeError("agent_policy_contract_failure", "Policy decision must be a JSON object.", origin="agent_policy")
    return value


def validate_policy_decision(value: dict[str, Any], *, available_actions: list[str]) -> PolicyDecision:
    allowed_fields = {"contract_version", "action", "reason_code", "arguments", "decision_summary", "expected_state_transition"}
    unknown_fields = set(value) - allowed_fields
    if unknown_fields:
        raise AgentRuntimeError("agent_policy_contract_failure", "Policy decision contains unknown fields.", origin="agent_policy", detail={"unknown_fields": sorted(unknown_fields)})
    missing = allowed_fields - set(value)
    if missing:
        raise AgentRuntimeError("agent_policy_contract_failure", "Policy decision is missing required fields.", origin="agent_policy", detail={"missing_fields": sorted(missing)})
    if value["contract_version"] != AGENT_POLICY_DECISION_CONTRACT_VERSION:
        raise AgentRuntimeError("agent_policy_contract_failure", "Unknown policy decision contract version.", origin="agent_policy")
    action = value["action"]
    if action not in ALLOWED_ARGS:
        raise AgentRuntimeError("agent_policy_unknown_action", "Policy selected an unknown action.", origin="agent_policy", detail={"action": action})
    if action not in available_actions:
        raise AgentRuntimeError("agent_policy_illegal_transition", "Policy selected an action unavailable in the current governed state.", origin="agent_policy", detail={"action": action, "available_actions": available_actions})
    reason_code = value["reason_code"]
    if reason_code not in ALLOWED_POLICY_REASON_CODES:
        raise AgentRuntimeError("agent_policy_contract_failure", "Policy reason_code is not in the versioned enum.", origin="agent_policy", detail={"reason_code": reason_code})
    arguments = value["arguments"]
    if not isinstance(arguments, dict):
        raise AgentRuntimeError("agent_policy_argument_failure", "Policy arguments must be an object.", origin="agent_policy")
    _validate_arguments(action, arguments)
    decision_summary = value["decision_summary"]
    if not isinstance(decision_summary, str) or len(decision_summary) > 200:
        raise AgentRuntimeError("agent_policy_contract_failure", "Policy decision_summary must be a short string.", origin="agent_policy")
    _validate_summary(decision_summary)
    expected = EXPECTED_TRANSITION_BY_ACTION[action]
    if value["expected_state_transition"] != expected:
        raise AgentRuntimeError("agent_policy_illegal_transition", "Policy expected_state_transition does not match the action contract.", origin="agent_policy", detail={"expected": expected, "actual": value["expected_state_transition"]})
    return PolicyDecision(
        contract_version=value["contract_version"],
        action=action,
        reason_code=reason_code,
        arguments=arguments,
        decision_summary=decision_summary,
        expected_state_transition=expected,
    )


def decision_to_agent_action(decision: PolicyDecision) -> AgentAction:
    return AgentAction(
        action=decision.action,
        reason_code=decision.reason_code,
        arguments=decision.arguments,
        expected_state_transition=decision.expected_state_transition,
    )


def policy_argument_diagnostics(value: dict[str, Any] | None) -> dict[str, Any]:
    action = value.get("action") if isinstance(value, dict) else None
    arguments = value.get("arguments") if isinstance(value, dict) and isinstance(value.get("arguments"), dict) else {}
    return action_argument_diagnostics(action if isinstance(action, str) else None, arguments)


def action_argument_diagnostics(action: str | None, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments if isinstance(arguments, dict) else {}
    query = args.get("query")
    query_present = "query" in args
    query_non_empty = isinstance(query, str) and bool(query.strip())
    return {
        "action": action,
        "argument_field_names": sorted(str(key) for key in args),
        "query_present": query_present,
        "query_non_empty": query_non_empty,
        "query_length": len(query) if isinstance(query, str) else None,
        "query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest() if isinstance(query, str) else None,
    }


def redacted_policy_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"parseable_json_object": False, "value_type": type(value).__name__}
    redacted = {str(key): _redact_policy_value(key, child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value.get("arguments"), dict):
        redacted["argument_diagnostics"] = policy_argument_diagnostics(value)
    return redacted


def is_structural_policy_error(exc: AgentRuntimeError) -> bool:
    return exc.failure.code in STRUCTURAL_ERROR_CODES


def _validate_arguments(action: str, arguments: dict[str, Any]) -> None:
    unknown = set(arguments) - ALLOWED_ARGS[action]
    forbidden = set(arguments) & FORBIDDEN_ARGS
    budget_fields = {"max_steps", "max_tool_calls", "max_expansions", "remaining_budget", "budget"} & set(arguments)
    if budget_fields:
        raise AgentRuntimeError("agent_policy_budget_violation", "Policy attempted to modify runtime budget.", origin="agent_policy", detail={"budget_fields": sorted(budget_fields)})
    if unknown or forbidden:
        raise AgentRuntimeError("agent_policy_argument_failure", "Policy arguments violate the closed action schema.", origin="agent_policy", detail={"unknown_arguments": sorted(unknown), "forbidden_arguments": sorted(forbidden)})
    if action == "search":
        query = arguments.get("query")
        top_k = arguments.get("top_k")
        if not isinstance(query, str) or not query.strip():
            raise AgentRuntimeError("agent_policy_argument_failure", "search.query must be a non-empty string.", origin="agent_policy")
        _reject_unsafe_text(query)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1 or top_k > 20):
            raise AgentRuntimeError("agent_policy_argument_failure", "search.top_k must be between 1 and 20.", origin="agent_policy")
    if action == "expand_evidence":
        if not isinstance(arguments.get("chunk_id"), str) or not arguments["chunk_id"]:
            raise AgentRuntimeError("agent_policy_argument_failure", "expand_evidence.chunk_id is required.", origin="agent_policy")
        max_items = arguments.get("max_items")
        if max_items is not None and (not isinstance(max_items, int) or max_items < 1 or max_items > 5):
            raise AgentRuntimeError("agent_policy_argument_failure", "expand_evidence.max_items must be between 1 and 5.", origin="agent_policy")


def _validate_summary(summary: str) -> None:
    lowered = summary.lower()
    blocked = ("api_key", "authorization", "system prompt", "gold", "required_claims", "forbidden_claims")
    if any(term in lowered for term in blocked):
        raise AgentRuntimeError("agent_policy_prompt_leakage", "Policy decision summary contains prohibited content.", origin="agent_policy")
    _reject_unsafe_text(summary)


def _reject_unsafe_text(value: str) -> None:
    if any(pattern.search(value) for pattern in _SQL_OR_FILE_PATTERNS):
        raise AgentRuntimeError("agent_policy_prompt_injection_attempt", "Policy decision contains unsafe tool-bypass text.", origin="agent_policy")


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _redact_policy_value(key: Any, value: Any) -> Any:
    if str(key) == "query" and isinstance(value, str):
        return {
            "query_present": True,
            "query_non_empty": bool(value.strip()),
            "query_length": len(value),
            "query_digest": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(child_key): _redact_policy_value(child_key, child) for child_key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_redact_policy_value("", child) for child in value]
    return value
