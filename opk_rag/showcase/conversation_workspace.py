from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider
from opk_rag.conversation.models import ConversationSession, ConversationTurn, ConversationTurnCitation
from opk_rag.conversation.repository import SessionRepository
from opk_rag.conversation.resolver import OpenAICompatibleFollowupQueryResolver
from opk_rag.conversation.service import ConversationService
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import KnowledgeBaseRepository
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.demo import SHOWCASE_ROOT
from opk_rag.showcase.runtime_trace import RuntimeTraceContext

SCHEMA_VERSION = "opk-rag.control-center-conversation.v1"


@dataclass(frozen=True)
class ConversationWorkspaceAskResult:
    payload: dict[str, Any]
    trace: dict[str, Any] | None


class ConversationWorkspace:
    """Bounded adapter from Control Center API requests to existing conversation authority."""

    def __init__(self, executor) -> None:
        self.executor = executor

    def list_sessions(self) -> dict[str, Any]:
        database_url = self._database_url()
        with connect_postgres(database_url) as connection:
            kb_id = self._active_kb_id(connection)
            rows = SessionRepository(connection).list_sessions(knowledge_base_id=kb_id)
            return {"schema_version": SCHEMA_VERSION, "sessions": [_session_payload(row) for row in rows]}

    def create_session(self, *, title: str | None = None) -> dict[str, Any]:
        database_url = self._database_url()
        with connect_postgres(database_url) as connection:
            kb_id = self._active_kb_id(connection)
            row = SessionRepository(connection).create_session(knowledge_base_id=kb_id, title=(title or "").strip() or None)
            return {"schema_version": SCHEMA_VERSION, "session": _session_payload(row)}

    def get_session(self, session_id: UUID) -> dict[str, Any] | None:
        database_url = self._database_url()
        with connect_postgres(database_url) as connection:
            repo = SessionRepository(connection)
            session = repo.get_session(session_id)
            if session is None:
                return None
            if session.knowledge_base_id != self._active_kb_id(connection):
                return None
            turns = repo.list_turns(session_id)
            citations = repo.list_citations_for_turns(tuple(turn.id for turn in turns))
            return {
                "schema_version": SCHEMA_VERSION,
                "session": _session_payload(session),
                "turns": [_turn_payload(turn, citations.get(turn.id, ())) for turn in turns],
            }

    def list_turns(self, session_id: UUID) -> dict[str, Any] | None:
        payload = self.get_session(session_id)
        if payload is None:
            return None
        return {
            "schema_version": SCHEMA_VERSION,
            "session": payload["session"],
            "turns": payload["turns"],
        }

    async def ask(
        self,
        *,
        session_id: UUID,
        query: str,
        client_request_id: str | None = None,
        top_k: int | None = None,
    ) -> ConversationWorkspaceAskResult:
        # Share the exact API execution budget used by Search/Ask so UI conversation
        # cannot create an independent unbounded model-execution lane.
        async with self.executor._semaphore:
            return await asyncio.to_thread(
                self._ask_sync,
                session_id,
                query,
                client_request_id,
                top_k,
            )

    def _ask_sync(
        self,
        session_id: UUID,
        query: str,
        client_request_id: str | None,
        top_k: int | None,
    ) -> ConversationWorkspaceAskResult:
        database_url = self._database_url()
        with connect_postgres(database_url) as connection:
            repo = SessionRepository(connection)
            session = repo.get_session(session_id)
            if session is None or session.knowledge_base_id != self._active_kb_id(connection):
                raise ValueError("conversation_session_not_found")

            parts = self.executor._runtime_parts(execution_scope="ask", top_k=top_k)
            if str(session.knowledge_base_id) != str(parts.kb_id):
                raise ValueError("conversation_knowledge_base_mismatch")
            answer_config = load_answer_generation_config()
            api_key = os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None
            answer_provider = OpenAICompatibleLocalChatProvider(answer_config, api_key=api_key)
            resolver = OpenAICompatibleFollowupQueryResolver(answer_config, api_key=api_key)

            def traced_search(search_query: str, context: object):
                if not isinstance(context, RuntimeTraceContext):
                    raise RuntimeError("runtime_trace_context_required")
                return search_knowledge_base(
                    parts.database_url,
                    knowledge_base_id=session.knowledge_base_id,
                    query=search_query,
                    provider=parts.embedding_provider,
                    embedding_config=parts.embedding_config,
                    search_config=parts.search_config,
                    reranker_provider=parts.reranker_provider,
                    context_token_counter=parts.token_counter,
                    execution_scope="ask",
                    runtime_trace_context=context,
                )

            service = ConversationService(
                repository=repo,
                resolver=resolver,
                answer_provider=answer_provider,
                answer_config=answer_config,
                # Compatibility runner is intentionally retained; trace-aware Control
                # Center execution always selects traced_search_runner below.
                search_runner=lambda search_query: traced_search(
                    search_query,
                    RuntimeTraceContext(query_text=search_query, execution_scope="ask", enabled=True),
                ),
                traced_search_runner=traced_search,
                runtime_trace_context_factory=lambda search_query: RuntimeTraceContext(
                    query_text=search_query,
                    execution_scope="ask",
                    enabled=True,
                ),
                context_token_counter=parts.token_counter,
            )
            result = service.ask(
                session_id=session_id,
                user_query=query,
                client_request_id=(client_request_id or "").strip() or None,
            )
            trace = dict(result.answer.runtime_trace or {}) if result.answer is not None else None
            payload = {
                "schema_version": SCHEMA_VERSION,
                "session": _session_payload(result.session),
                "turn": _turn_payload(result.turn, result.citations),
                "resolution": {
                    "standalone_query": result.resolution.standalone_query,
                    "is_followup": result.resolution.is_followup,
                    "referenced_turn_numbers": list(result.resolution.referenced_turn_numbers),
                    "referenced_citation_ids": list(result.resolution.referenced_citation_ids),
                    "model_id": result.resolution.model_id,
                    "prompt_version": result.resolution.prompt_version,
                },
                "answer": _answer_payload(result.answer) if result.answer is not None else None,
                "idempotent_replay": result.idempotent_replay,
            }
            return ConversationWorkspaceAskResult(payload=payload, trace=trace)

    @staticmethod
    def _database_url() -> str:
        value = os.environ.get("DATABASE_URL", "").strip()
        if not value:
            raise RuntimeError("runtime_unavailable")
        return value

    @staticmethod
    def _active_kb_id(connection) -> UUID:
        kb = KnowledgeBaseRepository(connection).get_by_root_path(str(SHOWCASE_ROOT))
        if kb is None:
            raise RuntimeError("knowledge_base_unavailable")
        return kb.id


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _session_payload(session: ConversationSession) -> dict[str, Any]:
    return {
        "id": str(session.id),
        "knowledge_base_id": str(session.knowledge_base_id),
        "title": session.title,
        "status": session.status,
        "created_at": _iso(session.created_at),
        "updated_at": _iso(session.updated_at),
        "last_turn_at": _iso(session.last_turn_at),
        "turn_count": session.turn_count,
    }


def _citation_payload(citation: ConversationTurnCitation) -> dict[str, Any]:
    return {
        "citation_id": citation.citation_id,
        "document_id": str(citation.document_id),
        "chunk_id": str(citation.chunk_id),
        "relative_path": citation.relative_path,
        "heading_path": list(citation.heading_path),
        "start_line": citation.start_line,
        "end_line": citation.end_line,
        "context_rank": citation.context_rank,
        "source_status": citation.source_status,
    }


def _turn_payload(turn: ConversationTurn, citations: tuple[ConversationTurnCitation, ...]) -> dict[str, Any]:
    resolver = turn.metadata.get("resolver") if isinstance(turn.metadata, dict) else {}
    trace_meta = turn.metadata.get("runtime_trace") if isinstance(turn.metadata, dict) else {}
    return {
        "id": str(turn.id),
        "session_id": str(turn.session_id),
        "turn_number": turn.turn_number,
        "client_request_id": turn.client_request_id,
        "user_query": turn.user_query,
        "standalone_query": turn.standalone_query,
        "is_followup": bool(resolver.get("is_followup")) if isinstance(resolver, dict) else False,
        "rewrite_status": turn.rewrite_status,
        "answer_decision": turn.answer_decision,
        "answer_text": turn.answer_text,
        "abstention_reason": turn.abstention_reason,
        "turn_status": turn.turn_status,
        "retrieval_mode": turn.retrieval_mode,
        "reranking_enabled": turn.reranking_enabled,
        "provider_id": turn.provider_id,
        "model_id": turn.model_id,
        "latency_ms": turn.latency_ms,
        "error_code": turn.error_code,
        "created_at": _iso(turn.created_at),
        "completed_at": _iso(turn.completed_at),
        "trace_id": trace_meta.get("trace_id") if isinstance(trace_meta, dict) else None,
        "citations": [_citation_payload(row) for row in citations],
    }


def _answer_payload(answer) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answerable": answer.answerable,
        "answer": answer.answer,
        "refusal_reason_code": answer.refusal_reason_code,
        "provider_id": answer.provider_id,
        "model_id": answer.model_id,
        "generation_latency_ms": answer.generation_latency_ms,
        "grounding_valid": answer.grounding.valid,
        "citation_count": len(answer.citations),
    }
