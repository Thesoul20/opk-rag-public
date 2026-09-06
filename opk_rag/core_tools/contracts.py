from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TOOL_CONTRACT_ID = "core-rag-agent-tools-v1"
TOOL_SCHEMA_VERSION = "core-rag-agent-tool-schema-v1"
TOOL_TRACE_SCHEMA_VERSION = "core-rag-agent-tool-trace-v1"
CORE_RAG_AGENT_TOOLS_FEATURE_FLAG = "OPK_RAG_CORE_AGENT_TOOLS_ENABLED"

ToolName = Literal[
    "search_knowledge_base",
    "expand_document_section",
    "assess_answerability",
    "generate_grounded_answer",
    "verify_grounding",
]

ToolErrorCode = Literal[
    "invalid_query",
    "invalid_tool_name",
    "invalid_tool_sequence",
    "invalid_evidence_identity",
    "knowledge_base_not_found",
    "document_not_found",
    "evidence_not_found",
    "candidate_limit_exceeded",
    "answerability_contract_violation",
    "generation_not_permitted",
    "provider_failure",
    "citation_validation_failure",
    "grounding_validation_failure",
    "runtime_identity_mismatch",
    "privacy_policy_violation",
]


@dataclass(frozen=True)
class ToolError:
    code: ToolErrorCode
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": _stable(self.detail)}


class CoreToolError(RuntimeError):
    def __init__(self, code: ToolErrorCode, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error = ToolError(code=code, message=message, detail=detail or {})


@dataclass(frozen=True)
class ToolDefinition:
    name: ToolName
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    sequence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": _stable(self.input_schema),
            "output_schema": _stable(self.output_schema),
            "sequence": _stable(self.sequence),
        }


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: ToolName
    arguments: dict[str, Any]
    invocation_id: str
    runtime_identity: dict[str, Any]
    caller: str = "fixed_core_pipeline_adapter"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": redact_private_payload(self.arguments),
            "invocation_id": self.invocation_id,
            "runtime_identity": redact_private_payload(self.runtime_identity),
            "caller": self.caller,
        }


@dataclass(frozen=True)
class ToolResult:
    tool_name: ToolName
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": redact_private_payload(self.output),
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True)
class ToolTraceEvent:
    schema_version: str
    invocation: ToolInvocation
    result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "invocation": self.invocation.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class TrustedEvidence:
    knowledge_base_id: str
    query: str
    evidence_identity_digest: str
    citation_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    context_token_count: int
    context_token_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "query_hash": stable_digest(self.query),
            "evidence_identity_digest": self.evidence_identity_digest,
            "citation_ids": list(self.citation_ids),
            "chunk_ids": list(self.chunk_ids),
            "document_ids": list(self.document_ids),
            "context_token_count": self.context_token_count,
            "context_token_budget": self.context_token_budget,
        }


def stable_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_private_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): redact_private_payload(child) for key, child in sorted(value.items(), key=lambda item: str(item[0])) if str(key) not in {"content", "answer", "raw_generation_text", "raw_text", "database_url", "api_key", "authorization"}}
    if isinstance(value, (list, tuple)):
        return [redact_private_payload(child) for child in value]
    if isinstance(value, str):
        if "://" in value and ("postgres" in value.lower() or "key=" in value.lower() or "token" in value.lower()):
            return "<redacted>"
        if value.startswith("/"):
            return "<redacted-absolute-path>"
    return value


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_stable(child) for child in value]
    if isinstance(value, list):
        return [_stable(child) for child in value]
    return value
