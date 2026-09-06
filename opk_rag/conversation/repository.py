from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from opk_rag.answer.models import AnswerResponse
from opk_rag.conversation.models import (
    CONVERSATION_PROMPT_VERSION,
    FOLLOWUP_REWRITE_VERSION,
    ConversationSession,
    ConversationTurn,
    ConversationTurnCitation,
    FollowupResolution,
)
from opk_rag.db.repositories import RepositoryError

ACTIVE_TURN_STATUSES = ("pending", "rewriting", "retrieving", "generating")


class ConversationConcurrencyError(RepositoryError):
    pass


class ConversationNotFoundError(RepositoryError):
    pass


@dataclass(frozen=True)
class CreateTurnResult:
    turn: ConversationTurn
    created: bool


class SessionRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def create_session(
        self,
        *,
        knowledge_base_id: UUID,
        title: str | None = None,
        metadata: dict | None = None,
        conversation_prompt_version: str = CONVERSATION_PROMPT_VERSION,
        followup_rewrite_version: str = FOLLOWUP_REWRITE_VERSION,
    ) -> ConversationSession:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute("select id from public.knowledge_bases where id = %s", (knowledge_base_id,))
                if cursor.fetchone() is None:
                    raise ConversationNotFoundError(f"Knowledge base not found: {knowledge_base_id}")
                cursor.execute(
                    """
                    insert into public.conversation_sessions (
                      knowledge_base_id,
                      title,
                      conversation_prompt_version,
                      followup_rewrite_version,
                      metadata
                    )
                    values (%s, %s, %s, %s, %s::jsonb)
                    returning id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                              turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                    """,
                    (knowledge_base_id, title, conversation_prompt_version, followup_rewrite_version, _json(metadata or {})),
                )
                return _map_session(cursor.fetchone())

    def get_session(self, session_id: UUID) -> ConversationSession | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                       turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                from public.conversation_sessions
                where id = %s and status <> 'deleted'
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        return _map_session(row) if row is not None else None

    def list_sessions(self, *, knowledge_base_id: UUID | None = None, include_archived: bool = False) -> tuple[ConversationSession, ...]:
        filters = ["status <> 'deleted'"]
        params: list[object] = []
        if knowledge_base_id is not None:
            filters.append("knowledge_base_id = %s")
            params.append(knowledge_base_id)
        if not include_archived:
            filters.append("status = 'active'")
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                select id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                       turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                from public.conversation_sessions
                where {' and '.join(filters)}
                order by coalesce(last_turn_at, created_at) desc, created_at desc
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
        return tuple(_map_session(row) for row in rows)

    def archive_session(self, session_id: UUID) -> ConversationSession:
        return self._set_session_status(session_id, "archived")

    def delete_session(self, session_id: UUID) -> ConversationSession:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    delete from public.conversation_sessions
                    where id = %s
                    returning id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                              turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation session not found: {session_id}")
        return _map_session(row)

    def create_turn(
        self,
        *,
        session_id: UUID,
        user_query: str,
        client_request_id: str | None = None,
        metadata: dict | None = None,
    ) -> CreateTurnResult:
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")
        with self.connection.transaction():
            session = self._lock_session(session_id)
            if session.status != "active":
                raise ConversationConcurrencyError(f"Session is not active: {session.status}")
            if client_request_id:
                existing = self._get_turn_by_client_request_id(session_id, client_request_id)
                if existing is not None:
                    return CreateTurnResult(existing, created=False)
            latest = self._get_latest_turn_for_update(session_id)
            if latest is not None and latest.turn_status in ACTIVE_TURN_STATUSES:
                raise ConversationConcurrencyError(f"Previous turn is still {latest.turn_status}.")
            turn_number = session.turn_count + 1
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into public.conversation_turns (
                      session_id, turn_number, client_request_id, user_query, turn_status, metadata
                    )
                    values (%s, %s, %s, %s, 'pending', %s::jsonb)
                    returning id, session_id, turn_number, client_request_id, user_query, standalone_query,
                              rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                              retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                              prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                              error_code, created_at, completed_at, metadata
                    """,
                    (session_id, turn_number, client_request_id, user_query, _json(metadata or {})),
                )
                row = cursor.fetchone()
                cursor.execute(
                    """
                    update public.conversation_sessions
                    set turn_count = turn_count + 1
                    where id = %s
                    """,
                    (session_id,),
                )
        return CreateTurnResult(_map_turn(row), created=True)

    def update_turn_status(
        self,
        turn_id: UUID,
        status: Literal["rewriting", "retrieving", "generating"],
        *,
        standalone_query: str | None = None,
        rewrite_status: str | None = None,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        updates = ["turn_status = %s", "metadata = metadata || %s::jsonb"]
        params: list[object] = [status, _json(metadata or {})]
        if standalone_query is not None:
            updates.append("standalone_query = %s")
            params.append(standalone_query)
        if rewrite_status is not None:
            updates.append("rewrite_status = %s")
            params.append(rewrite_status)
        params.append(turn_id)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    update public.conversation_turns
                    set {', '.join(updates)}
                    where id = %s
                    returning id, session_id, turn_number, client_request_id, user_query, standalone_query,
                              rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                              retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                              prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                              error_code, created_at, completed_at, metadata
                    """,
                    tuple(params),
                )
                return _map_turn(cursor.fetchone())

    def complete_turn(
        self,
        *,
        turn_id: UUID,
        answer: AnswerResponse,
        resolution: FollowupResolution,
        latency_ms: int | None = None,
    ) -> ConversationTurn:
        status = "completed" if answer.answerable else "abstained"
        decision = "answer" if answer.answerable else "abstain"
        metadata = {
            "resolver": _resolution_metadata(resolution),
            "search": {
                "candidate_count": answer.search_response.candidate_count,
                "result_count": answer.search_response.result_count,
                "context_token_budget": answer.search_response.context_token_budget,
                "context_token_count": answer.search_response.context_token_count,
                "evidence_serialization": answer.evidence_serialization,
            },
            "model_decision": answer.model_decision,
            "answerability": {
                "answerable": answer.answerability.answerable,
                "status": answer.answerability.status,
                "reason_code": answer.answerability.reason_code,
                "reason": answer.answerability.reason,
                "confidence": answer.answerability.confidence,
                "evidence_chunk_ids": list(answer.answerability.evidence_chunk_ids),
                "evidence_score": answer.answerability.evidence_score,
                "evidence_count": answer.answerability.evidence_count,
                "considered_evidence_count": answer.answerability.considered_evidence_count,
                "diagnostics": answer.answerability.diagnostics,
            },
            "runtime_trace": {
                "trace_id": ((answer.runtime_trace or {}).get("trace") or {}).get("trace_id"),
                "trace_schema_version": ((answer.runtime_trace or {}).get("trace") or {}).get("trace_schema_version"),
            },
            "grounding": {
                "valid": answer.grounding.valid,
                "status": answer.grounding.status,
                "reason_code": answer.grounding.reason_code,
                "reason": answer.grounding.reason,
                "cited_ids": list(answer.grounding.cited_ids),
                "valid_cited_ids": list(answer.grounding.valid_cited_ids),
                "invalid_cited_ids": list(answer.grounding.invalid_cited_ids),
                "available_evidence_ids": list(answer.grounding.available_evidence_ids),
                "citation_coverage": answer.grounding.citation_coverage,
                "diagnostics": answer.grounding.diagnostics,
            },
        }
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    update public.conversation_turns
                    set standalone_query = %s,
                        rewrite_status = %s,
                        answer_decision = %s,
                        answer_text = %s,
                        abstention_reason = %s,
                        turn_status = %s,
                        retrieval_mode = %s,
                        reranking_enabled = %s,
                        provider_id = %s,
                        model_id = %s,
                        model_version = %s,
                        prompt_fingerprint = %s,
                        input_token_count = %s,
                        output_token_count = %s,
                        latency_ms = %s,
                        completed_at = now(),
                        metadata = metadata || %s::jsonb
                    where id = %s
                    returning id, session_id, turn_number, client_request_id, user_query, standalone_query,
                              rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                              retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                              prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                              error_code, created_at, completed_at, metadata
                    """,
                    (
                        resolution.standalone_query,
                        "resolved" if resolution.is_followup else "not_needed",
                        decision,
                        answer.answer,
                        answer.refusal_reason_code,
                        status,
                        answer.search_response.retrieval_mode,
                        answer.search_response.reranker_enabled,
                        answer.provider_id,
                        answer.model_id,
                        answer.model_revision,
                        answer.prompt_version,
                        (answer.prompt_tokens or 0) + (resolution.input_token_count or 0),
                        (answer.completion_tokens or 0) + (resolution.output_token_count or 0),
                        latency_ms if latency_ms is not None else answer.generation_latency_ms,
                        _json(metadata),
                        turn_id,
                    ),
                )
                turn = _map_turn(cursor.fetchone())
                cursor.execute("update public.conversation_sessions set last_turn_at = now() where id = %s", (turn.session_id,))
        return turn

    def fail_turn(
        self,
        *,
        turn_id: UUID,
        error_code: str,
        error_message: str,
        resolution: FollowupResolution | None = None,
    ) -> ConversationTurn:
        metadata = {"error_message": error_message}
        if resolution is not None:
            metadata["resolver"] = _resolution_metadata(resolution)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    update public.conversation_turns
                    set turn_status = 'failed',
                        answer_decision = 'system_error',
                        error_code = %s,
                        completed_at = now(),
                        metadata = metadata || %s::jsonb
                    where id = %s
                    returning id, session_id, turn_number, client_request_id, user_query, standalone_query,
                              rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                              retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                              prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                              error_code, created_at, completed_at, metadata
                    """,
                    (error_code, _json(metadata), turn_id),
                )
                turn = _map_turn(cursor.fetchone())
                cursor.execute("update public.conversation_sessions set last_turn_at = now() where id = %s", (turn.session_id,))
        return turn

    def list_turns(self, session_id: UUID) -> tuple[ConversationTurn, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, session_id, turn_number, client_request_id, user_query, standalone_query,
                       rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                       retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                       prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                       error_code, created_at, completed_at, metadata
                from public.conversation_turns
                where session_id = %s
                order by turn_number
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
        return tuple(_map_turn(row) for row in rows)

    def get_recent_turns(self, session_id: UUID, *, limit: int) -> tuple[ConversationTurn, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select *
                from (
                  select id, session_id, turn_number, client_request_id, user_query, standalone_query,
                         rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                         retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                         prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                         error_code, created_at, completed_at, metadata
                  from public.conversation_turns
                  where session_id = %s and turn_status in ('completed', 'abstained')
                  order by turn_number desc
                  limit %s
                ) recent
                order by turn_number
                """,
                (session_id, max(1, limit)),
            )
            rows = cursor.fetchall()
        return tuple(_map_turn(row) for row in rows)

    def save_turn_citations(self, *, turn_id: UUID, answer: AnswerResponse) -> tuple[ConversationTurnCitation, ...]:
        if not answer.citations:
            return ()
        item_by_chunk_id = {item.chunk_id: item for item in answer.search_response.evidence_bundle.items} if answer.search_response.evidence_bundle else {}
        rows = []
        for citation in answer.citations:
            item = item_by_chunk_id.get(UUID(citation.chunk_id))
            if item is None:
                raise RepositoryError(f"Citation {citation.citation_id} does not belong to current evidence.")
            rows.append(
                (
                    turn_id,
                    citation.citation_id,
                    citation.document_id,
                    citation.chunk_id,
                    citation.relative_path,
                    _json_array(list(citation.heading_path)),
                    citation.start_line,
                    citation.end_line,
                    _content_hash(item.content),
                    item.context_rank,
                )
            )
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    insert into public.conversation_turn_citations (
                      turn_id, citation_id, document_id, chunk_id, relative_path, heading_path,
                      start_line, end_line, content_hash, context_rank
                    )
                    values (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    on conflict (turn_id, citation_id) do nothing
                    """,
                    rows,
                )
        return self.list_turn_citations(turn_id)

    def list_turn_citations(self, turn_id: UUID) -> tuple[ConversationTurnCitation, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select turn_id, citation_id, document_id, chunk_id, relative_path, heading_path,
                       start_line, end_line, content_hash, context_rank, created_at
                from public.conversation_turn_citations
                where turn_id = %s
                order by context_rank, citation_id
                """,
                (turn_id,),
            )
            rows = cursor.fetchall()
        return tuple(_map_citation(row) for row in rows)

    def list_citations_for_turns(self, turn_ids: tuple[UUID, ...]) -> dict[UUID, tuple[ConversationTurnCitation, ...]]:
        if not turn_ids:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select c.turn_id, c.citation_id, c.document_id, c.chunk_id, c.relative_path, c.heading_path,
                       c.start_line, c.end_line, c.content_hash, c.context_rank, c.created_at,
                       case
                         when ch.id is null then 'source_deleted'
                         when ch.content_hash is distinct from c.content_hash then 'source_changed'
                         else 'source_current'
                       end as source_status
                from public.conversation_turn_citations c
                left join public.chunks ch on ch.id = c.chunk_id
                where c.turn_id = any(%s)
                order by c.context_rank, c.citation_id
                """,
                (list(turn_ids),),
            )
            rows = cursor.fetchall()
        grouped: dict[UUID, list[ConversationTurnCitation]] = {}
        for row in rows:
            citation = _map_citation(row)
            grouped.setdefault(citation.turn_id, []).append(citation)
        return {turn_id: tuple(values) for turn_id, values in grouped.items()}

    def _set_session_status(self, session_id: UUID, status: Literal["archived", "deleted"]) -> ConversationSession:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    update public.conversation_sessions
                    set status = %s
                    where id = %s and status <> 'deleted'
                    returning id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                              turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                    """,
                    (status, session_id),
                )
                row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation session not found: {session_id}")
        return _map_session(row)

    def _lock_session(self, session_id: UUID) -> ConversationSession:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, knowledge_base_id, title, status, created_at, updated_at, last_turn_at,
                       turn_count, conversation_prompt_version, followup_rewrite_version, metadata
                from public.conversation_sessions
                where id = %s and status <> 'deleted'
                for update
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation session not found: {session_id}")
        return _map_session(row)

    def _get_latest_turn_for_update(self, session_id: UUID) -> ConversationTurn | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, session_id, turn_number, client_request_id, user_query, standalone_query,
                       rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                       retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                       prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                       error_code, created_at, completed_at, metadata
                from public.conversation_turns
                where session_id = %s
                order by turn_number desc
                limit 1
                for update
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        return _map_turn(row) if row is not None else None

    def _get_turn_by_client_request_id(self, session_id: UUID, client_request_id: str) -> ConversationTurn | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, session_id, turn_number, client_request_id, user_query, standalone_query,
                       rewrite_status, answer_decision, answer_text, abstention_reason, turn_status,
                       retrieval_mode, reranking_enabled, provider_id, model_id, model_version,
                       prompt_fingerprint, input_token_count, output_token_count, latency_ms,
                       error_code, created_at, completed_at, metadata
                from public.conversation_turns
                where session_id = %s and client_request_id = %s
                """,
                (session_id, client_request_id),
            )
            row = cursor.fetchone()
        return _map_turn(row) if row is not None else None


def _map_session(row) -> ConversationSession:
    return ConversationSession(
        id=row[0],
        knowledge_base_id=row[1],
        title=row[2],
        status=row[3],
        created_at=row[4],
        updated_at=row[5],
        last_turn_at=row[6],
        turn_count=row[7],
        conversation_prompt_version=row[8],
        followup_rewrite_version=row[9],
        metadata=dict(row[10]) if row[10] is not None else {},
    )


def _map_turn(row) -> ConversationTurn:
    return ConversationTurn(
        id=row[0],
        session_id=row[1],
        turn_number=row[2],
        client_request_id=row[3],
        user_query=row[4],
        standalone_query=row[5],
        rewrite_status=row[6],
        answer_decision=row[7],
        answer_text=row[8],
        abstention_reason=row[9],
        turn_status=row[10],
        retrieval_mode=row[11],
        reranking_enabled=row[12],
        provider_id=row[13],
        model_id=row[14],
        model_version=row[15],
        prompt_fingerprint=row[16],
        input_token_count=row[17],
        output_token_count=row[18],
        latency_ms=row[19],
        error_code=row[20],
        created_at=row[21],
        completed_at=row[22],
        metadata=dict(row[23]) if row[23] is not None else {},
    )


def _map_citation(row) -> ConversationTurnCitation:
    return ConversationTurnCitation(
        turn_id=row[0],
        citation_id=row[1],
        document_id=row[2],
        chunk_id=row[3],
        relative_path=row[4],
        heading_path=tuple(row[5] or ()),
        start_line=row[6],
        end_line=row[7],
        content_hash=row[8],
        context_rank=row[9],
        created_at=row[10],
        source_status=row[11] if len(row) > 11 else None,
    )


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_array(value: list) -> str:
    return json.dumps(value, ensure_ascii=False)


def _content_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolution_metadata(resolution: FollowupResolution) -> dict:
    return {
        "is_followup": resolution.is_followup,
        "referenced_turn_numbers": list(resolution.referenced_turn_numbers),
        "referenced_citation_ids": list(resolution.referenced_citation_ids),
        "resolution_reason": resolution.resolution_reason,
        "model_id": resolution.model_id,
        "model_revision": resolution.model_revision,
        "prompt_version": resolution.prompt_version,
        "output_schema_version": resolution.output_schema_version,
        "latency_ms": resolution.latency_ms,
    }
