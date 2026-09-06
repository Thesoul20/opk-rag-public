from __future__ import annotations

from typing import Any, Literal
from pydantic import Field

from opk_rag.agentic_v2.base import StrictContract, stable_digest

AGENTIC_V2_TOOL_TRACE_VERSION = "opk-rag.agentic-v2.tool-execution-trace.v1"


class AgentToolExecutionTrace(StrictContract):
    contract_version: Literal[AGENTIC_V2_TOOL_TRACE_VERSION] = AGENTIC_V2_TOOL_TRACE_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    execution_id: str = Field(min_length=1, max_length=128)
    action: str
    action_digest: str
    tool_name: str
    tool_version: str
    execution_status: str
    candidate_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    candidate_ids_digest: str
    evidence_ids_digest: str
    latency_ms: float = Field(ge=0.0)
    failure_code: str | None = None
    read_only: Literal[True] = True
    state_mutated: Literal[False] = False


def ids_digest(ids: tuple[str, ...]) -> str:
    return stable_digest(list(ids))
