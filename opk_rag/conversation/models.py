from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from opk_rag.answer.models import AnswerResponse

ConversationSessionStatus = Literal["active", "archived", "deleted"]
RewriteStatus = Literal["not_needed", "resolved", "failed"]
TurnStatus = Literal["pending", "rewriting", "retrieving", "generating", "completed", "abstained", "failed"]
AnswerDecision = Literal["answer", "abstain", "system_error"]
CitationSourceStatus = Literal["source_current", "source_changed", "source_deleted"]

CONVERSATION_PROMPT_VERSION = "conversation-context-v1"
FOLLOWUP_REWRITE_VERSION = "followup-query-resolution-v1"
STANDALONE_QUERY_SCHEMA_VERSION = "standalone-query-v1"
DEFAULT_CONTEXT_MAX_TURNS = 6
DEFAULT_CONTEXT_TOKEN_BUDGET = 1200


@dataclass(frozen=True)
class ConversationSession:
    id: UUID
    knowledge_base_id: UUID
    title: str | None
    status: ConversationSessionStatus
    created_at: datetime
    updated_at: datetime
    last_turn_at: datetime | None
    turn_count: int
    conversation_prompt_version: str
    followup_rewrite_version: str
    metadata: dict


@dataclass(frozen=True)
class ConversationTurn:
    id: UUID
    session_id: UUID
    turn_number: int
    client_request_id: str | None
    user_query: str
    standalone_query: str | None
    rewrite_status: RewriteStatus
    answer_decision: AnswerDecision | None
    answer_text: str | None
    abstention_reason: str | None
    turn_status: TurnStatus
    retrieval_mode: str | None
    reranking_enabled: bool | None
    provider_id: str | None
    model_id: str | None
    model_version: str | None
    prompt_fingerprint: str | None
    input_token_count: int | None
    output_token_count: int | None
    latency_ms: int | None
    error_code: str | None
    created_at: datetime
    completed_at: datetime | None
    metadata: dict


@dataclass(frozen=True)
class ConversationTurnCitation:
    turn_id: UUID
    citation_id: str
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None
    content_hash: str
    context_rank: int
    created_at: datetime | None = None
    source_status: CitationSourceStatus | None = None


@dataclass(frozen=True)
class ConversationContextTurn:
    turn_number: int
    user_query: str
    standalone_query: str | None
    answer_text: str | None
    answer_decision: AnswerDecision | None
    abstention_reason: str | None
    citations: tuple[ConversationTurnCitation, ...]
    token_count: int


@dataclass(frozen=True)
class ConversationContext:
    session_id: UUID
    knowledge_base_id: UUID
    prompt_version: str
    max_turns: int
    token_budget: int
    token_count: int
    turns: tuple[ConversationContextTurn, ...]


@dataclass(frozen=True)
class FollowupResolution:
    standalone_query: str
    is_followup: bool
    referenced_turn_numbers: tuple[int, ...]
    referenced_citation_ids: tuple[str, ...]
    resolution_reason: str
    model_id: str
    model_revision: str | None
    prompt_version: str
    output_schema_version: str
    input_token_count: int | None
    output_token_count: int | None
    latency_ms: int | None


@dataclass(frozen=True)
class ConversationTurnResponse:
    session: ConversationSession
    turn: ConversationTurn
    resolution: FollowupResolution
    answer: AnswerResponse | None
    citations: tuple[ConversationTurnCitation, ...]
    idempotent_replay: bool = False


class FollowupQueryResolver(Protocol):
    def resolve(
        self,
        *,
        current_query: str,
        conversation_context: ConversationContext,
    ) -> FollowupResolution:
        ...
