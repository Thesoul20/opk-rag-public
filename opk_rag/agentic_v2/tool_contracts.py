from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator

from opk_rag.agentic_v2.action import ActionName
from opk_rag.agentic_v2.base import StrictContract

AGENTIC_V2_TOOL_RESULT_CONTRACT_VERSION = "opk-rag.agentic-v2.tool-result.v1"
AGENTIC_V2_TOOL_DEFINITION_VERSION = "opk-rag.agentic-v2.tool-definition.v1"
ToolStatus = Literal["success", "no_result", "rejected", "failed"]
TerminalIntent = Literal["finish", "abstain"]


class AgentToolResult(StrictContract):
    contract_version: Literal[AGENTIC_V2_TOOL_RESULT_CONTRACT_VERSION] = AGENTIC_V2_TOOL_RESULT_CONTRACT_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    execution_id: str = Field(min_length=1, max_length=128)
    action: ActionName
    status: ToolStatus
    candidate_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    candidate_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    result_summary: str | None = Field(default=None, max_length=240)
    rewritten_query: str | None = Field(default=None, min_length=1, max_length=512)
    terminal_intent: TerminalIntent | None = None
    failure_code: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def validate_semantics(self) -> "AgentToolResult":
        if self.candidate_count != len(self.candidate_ids):
            raise ValueError("candidate_count must equal candidate_ids length")
        if self.evidence_count != len(self.evidence_ids):
            raise ValueError("evidence_count must equal evidence_ids length")
        if self.status == "failed" and not self.failure_code:
            raise ValueError("failed tool result requires failure_code")
        if self.status != "failed" and self.failure_code is not None:
            raise ValueError("only failed tool result may carry failure_code")
        if self.action == "rewrite_query" and self.status == "success" and self.rewritten_query is None:
            raise ValueError("successful rewrite_query requires rewritten_query")
        if self.action in {"finish", "abstain"} and self.status == "success" and self.terminal_intent != self.action:
            raise ValueError("terminal action requires matching terminal_intent")
        if self.action not in {"finish", "abstain"} and self.terminal_intent is not None:
            raise ValueError("non-terminal action cannot carry terminal_intent")
        return self


class AgentToolDefinition(StrictContract):
    contract_version: Literal[AGENTIC_V2_TOOL_DEFINITION_VERSION] = AGENTIC_V2_TOOL_DEFINITION_VERSION
    action: ActionName
    tool_name: str = Field(min_length=1, max_length=96)
    tool_version: str = Field(min_length=1, max_length=64)
    capability: str = Field(min_length=1, max_length=160)
    read_only: Literal[True] = True
    dynamic_registration_allowed: Literal[False] = False
