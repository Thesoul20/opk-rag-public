from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from opk_rag.agent.contracts import AGENT_TRACE_CONTRACT_VERSION, AgentAction, stable_digest
from opk_rag.agent.errors import AgentFailure
from opk_rag.agent.policy_validation import action_argument_diagnostics


@dataclass(frozen=True)
class AgentTraceEvent:
    run_id: str
    sample_id: str | None
    policy_name: str
    policy_version: str
    runtime_contract_version: str
    tool_registry_version: str
    step_index: int
    state_before_digest: str
    action: AgentAction
    tool_request: dict[str, Any]
    tool_response_summary: dict[str, Any]
    state_after_digest: str
    latency_ms: int
    failure: AgentFailure | None = None
    policy_decision: dict[str, Any] | None = None
    runtime_transition: dict[str, Any] | None = None
    timestamp: str = ""
    schema_version: str = AGENT_TRACE_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "runtime_contract_version": self.runtime_contract_version,
            "tool_registry_version": self.tool_registry_version,
            "step_index": self.step_index,
            "state_before_digest": self.state_before_digest,
            "action": _action_for_trace(self.action),
            "tool_request": _redact(self.tool_request),
            "tool_response_summary": _redact(self.tool_response_summary),
            "state_after_digest": self.state_after_digest,
            "latency_ms": self.latency_ms,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "policy_decision": _redact(self.policy_decision or {}),
            "runtime_transition": _redact(self.runtime_transition or {}),
            "timestamp": self.timestamp or utc_now(),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def summarize_for_trace(value: dict[str, Any]) -> dict[str, Any]:
    return _redact(value)


def payload_digest(value: Any) -> str:
    return stable_digest(_redact(value))


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {
            "api_key",
            "authorization",
            "database_url",
            "token",
            "password",
            "content",
            "answer",
            "raw_text",
            "raw_generation_text",
            "prompt",
            "messages",
            "query",
        }
        return {str(key): _redact(child) for key, child in sorted(value.items(), key=lambda item: str(item[0])) if str(key).lower() not in blocked}
    if isinstance(value, (tuple, list)):
        return [_redact(child) for child in value]
    if isinstance(value, str) and (value.startswith("/") or "authorization:" in value.lower() or "api_key" in value.lower()):
        return "<redacted>"
    return value


def _action_for_trace(action: AgentAction) -> dict[str, Any]:
    payload = action.to_dict()
    payload["arguments"] = {
        "diagnostics": action_argument_diagnostics(action.action, action.arguments),
    }
    return payload
