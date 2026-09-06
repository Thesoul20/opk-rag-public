from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.recovery_state import RecoveryStateName

AGENT_RECOVERY_TRACE_CONTRACT_VERSION = "opk-rag.agent-recovery-trace.v1"


@dataclass(frozen=True)
class AgentRecoveryTraceEvent:
    transition_index: int
    from_state: RecoveryStateName
    to_state: RecoveryStateName
    decision_type: str
    decision_reason: str
    tool_name: str | None
    tool_call_index: int | None
    input_identity: str | None
    output_identity: str | None
    started_at: str
    completed_at: str
    duration_ms: int
    status: str
    error_type: str | None = None
    contract_version: str = AGENT_RECOVERY_TRACE_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "transition_index": self.transition_index,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "decision_type": self.decision_type,
            "decision_reason": self.decision_reason,
            "tool_name": self.tool_name,
            "tool_call_index": self.tool_call_index,
            "input_identity": self.input_identity,
            "output_identity": self.output_identity,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_type": self.error_type,
        }

    @property
    def identity_digest(self) -> str:
        return stable_digest(self.to_dict())


def serialize_agent_trace(events: tuple[AgentRecoveryTraceEvent, ...]) -> list[dict[str, Any]]:
    payload = [event.to_dict() for event in events]
    text = repr(payload).lower()
    forbidden = ("api_key", "authorization", "/data/envs", "private vault raw body", "postgres://")
    if any(term in text for term in forbidden):
        raise ValueError("trace_serialization_failure")
    return payload
