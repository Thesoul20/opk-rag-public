from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SHOWCASE_API_VERSION = "opk-rag.showcase-api.v1"
TRACE_EVENT_SCHEMA_VERSION = "opk-rag.runtime-trace-event.v1"

MAX_QUERY_CHARS = 2000
MAX_TOP_K = 20
DEFAULT_TOP_K = None


class ErrorResponse(BaseModel):
    schema_version: str = "opk-rag.showcase-api-error.v1"
    code: str
    message: str
    trace_id: str | None = None
    stage: str | None = None
    retryable: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class ConversationSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)


class ConversationAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    client_request_id: str | None = Field(default=None, max_length=128)
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class TraceEnvelope(BaseModel):
    schema_version: str = SHOWCASE_API_VERSION
    trace_id: str
    status: Literal["completed", "failed", "refused", "partial"]
    trace: dict[str, Any]


class RuntimeTraceEvent(BaseModel):
    event_schema_version: str = TRACE_EVENT_SCHEMA_VERSION
    event_id: str
    trace_id: str
    sequence: int
    event_type: str
    stage: str
    timestamp: str | None
    payload: dict[str, Any]
    terminal: bool
