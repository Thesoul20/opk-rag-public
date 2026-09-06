from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.models import AnswerGeneratorProvider, AnswerResponse, AnswerSystemError
from opk_rag.answer.service import _contains_prompt_injection_query, answer_knowledge_base
from opk_rag.conversation.context import build_conversation_context
from opk_rag.conversation.models import (
    DEFAULT_CONTEXT_MAX_TURNS,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    ConversationTurnResponse,
    FollowupResolution,
    FollowupQueryResolver,
)
from opk_rag.conversation.repository import SessionRepository
from opk_rag.search.models import SearchResponse

SearchRunner = Callable[[str], SearchResponse]
TracedSearchRunner = Callable[[str, object], SearchResponse]
RuntimeTraceContextFactory = Callable[[str], object]


class ConversationService:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        resolver: FollowupQueryResolver,
        answer_provider: AnswerGeneratorProvider,
        answer_config: AnswerGenerationConfig,
        search_runner: SearchRunner,
        context_token_counter,
        traced_search_runner: TracedSearchRunner | None = None,
        runtime_trace_context_factory: RuntimeTraceContextFactory | None = None,
        context_max_turns: int = DEFAULT_CONTEXT_MAX_TURNS,
        context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        self.repository = repository
        self.resolver = resolver
        self.answer_provider = answer_provider
        self.answer_config = answer_config
        self.search_runner = search_runner
        self.context_token_counter = context_token_counter
        self.traced_search_runner = traced_search_runner
        self.runtime_trace_context_factory = runtime_trace_context_factory
        self.context_max_turns = context_max_turns
        self.context_token_budget = context_token_budget

    def ask(
        self,
        *,
        session_id: UUID,
        user_query: str,
        client_request_id: str | None = None,
    ) -> ConversationTurnResponse:
        session = self.repository.get_session(session_id)
        if session is None:
            raise ValueError(f"Conversation session not found: {session_id}")
        created = self.repository.create_turn(
            session_id=session_id,
            user_query=user_query,
            client_request_id=client_request_id,
        )
        turn = created.turn
        if not created.created and turn.turn_status in {"completed", "abstained"}:
            citations = self.repository.list_turn_citations(turn.id)
            resolution = _resolution_from_turn(turn)
            return ConversationTurnResponse(session=session, turn=turn, resolution=resolution, answer=None, citations=citations, idempotent_replay=True)
        if not created.created:
            raise ValueError(f"Existing idempotent turn is still {turn.turn_status}.")

        started = time.perf_counter()
        resolution = None
        try:
            recent_turns = self.repository.get_recent_turns(session_id, limit=self.context_max_turns)
            citations_by_turn = self.repository.list_citations_for_turns(tuple(t.id for t in recent_turns))
            context = build_conversation_context(
                session=session,
                turns=recent_turns,
                citations_by_turn=citations_by_turn,
                token_counter=self.context_token_counter,
                max_turns=self.context_max_turns,
                token_budget=self.context_token_budget,
            )
            self.repository.update_turn_status(turn.id, "rewriting")
            if _contains_prompt_injection_query(user_query):
                resolution = FollowupResolution(
                    standalone_query=user_query.strip(),
                    is_followup=False,
                    referenced_turn_numbers=(),
                    referenced_citation_ids=(),
                    resolution_reason="Current query matched deterministic prompt-injection guard; rewrite disabled.",
                    model_id="deterministic_prompt_injection_guard",
                    model_revision=None,
                    prompt_version=session.followup_rewrite_version,
                    output_schema_version="standalone-query-v1",
                    input_token_count=0,
                    output_token_count=0,
                    latency_ms=0,
                )
            else:
                resolution = self.resolver.resolve(current_query=user_query, conversation_context=context)
            self.repository.update_turn_status(
                turn.id,
                "retrieving",
                standalone_query=resolution.standalone_query,
                rewrite_status="resolved" if resolution.is_followup else "not_needed",
                metadata={"resolver": _resolution_metadata(resolution), "conversation_context": _context_metadata(context)},
            )
            runtime_trace_context = (
                self.runtime_trace_context_factory(resolution.standalone_query)
                if self.runtime_trace_context_factory is not None
                else None
            )
            search_response = (
                self.traced_search_runner(resolution.standalone_query, runtime_trace_context)
                if self.traced_search_runner is not None and runtime_trace_context is not None
                else self.search_runner(resolution.standalone_query)
            )
            if search_response.knowledge_base_id != session.knowledge_base_id:
                raise ValueError("Search response knowledge_base_id does not match the conversation session.")
            self.repository.update_turn_status(turn.id, "generating")
            answer = answer_knowledge_base(
                search_response,
                provider=self.answer_provider,
                config=self.answer_config,
                runtime_trace_context=runtime_trace_context,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            completed_turn = self.repository.complete_turn(
                turn_id=turn.id,
                answer=answer,
                resolution=resolution,
                latency_ms=elapsed_ms,
            )
            citations = self.repository.save_turn_citations(turn_id=turn.id, answer=answer)
            refreshed_session = self.repository.get_session(session_id) or session
            return ConversationTurnResponse(
                session=refreshed_session,
                turn=completed_turn,
                resolution=resolution,
                answer=answer,
                citations=citations,
            )
        except AnswerSystemError as exc:
            self.repository.fail_turn(
                turn_id=turn.id,
                error_code=exc.code,
                error_message=str(exc),
                resolution=resolution,
            )
            raise
        except Exception as exc:
            self.repository.fail_turn(
                turn_id=turn.id,
                error_code=type(exc).__name__,
                error_message=str(exc),
                resolution=resolution,
            )
            raise


def _resolution_from_turn(turn):
    from opk_rag.conversation.models import FollowupResolution

    resolver = turn.metadata.get("resolver") if isinstance(turn.metadata, dict) else None
    if not isinstance(resolver, dict):
        resolver = {}
    return FollowupResolution(
        standalone_query=turn.standalone_query or turn.user_query,
        is_followup=bool(resolver.get("is_followup", turn.standalone_query not in {None, turn.user_query})),
        referenced_turn_numbers=tuple(int(value) for value in resolver.get("referenced_turn_numbers", ())),
        referenced_citation_ids=tuple(str(value) for value in resolver.get("referenced_citation_ids", ())),
        resolution_reason=str(resolver.get("resolution_reason", "idempotent replay")),
        model_id=str(resolver.get("model_id", "unknown")),
        model_revision=resolver.get("model_revision"),
        prompt_version=str(resolver.get("prompt_version", "unknown")),
        output_schema_version=str(resolver.get("output_schema_version", "unknown")),
        input_token_count=None,
        output_token_count=None,
        latency_ms=resolver.get("latency_ms") if isinstance(resolver.get("latency_ms"), int) else None,
    )


def _resolution_metadata(resolution) -> dict:
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


def _context_metadata(context) -> dict:
    return {
        "prompt_version": context.prompt_version,
        "max_turns": context.max_turns,
        "token_budget": context.token_budget,
        "token_count": context.token_count,
        "included_turn_numbers": [turn.turn_number for turn in context.turns],
    }
