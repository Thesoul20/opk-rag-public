from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from opk_rag.showcase.api.events import encode_sse_event
from opk_rag.showcase.api.execution import ShowcaseExecutor, health_payload, safe_runtime_authority
from opk_rag.showcase.api.models import (
    AskRequest,
    ConversationAskRequest,
    ConversationSessionCreateRequest,
    SearchRequest,
    SHOWCASE_API_VERSION,
    TraceEnvelope,
)
from opk_rag.showcase.api.registry import TraceRegistry, after_sequence_from_last_event_id
from opk_rag.showcase.api.safety import body_size_guard, safe_error
from opk_rag.showcase.demo import list_showcase_scenarios
from opk_rag.showcase.runtime_trace import TRACE_SCHEMA_VERSION, validate_runtime_trace
from opk_rag.showcase.runtime_status import collect_runtime_status
from opk_rag.showcase.conversation_workspace import ConversationWorkspace


API_PREFIX = "/api/showcase/v1"
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def create_app(
    *,
    registry: TraceRegistry | None = None,
    executor: ShowcaseExecutor | None = None,
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS,
    conversation_workspace: ConversationWorkspace | None = None,
) -> FastAPI:
    trace_registry = registry or TraceRegistry()
    showcase_executor = executor or ShowcaseExecutor(max_concurrent_executions=1)
    conversation = conversation_workspace or ConversationWorkspace(showcase_executor)
    app = FastAPI(
        title="OPK-RAG Showcase API",
        version=SHOWCASE_API_VERSION,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Last-Event-ID", "Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request, _exc):
        return safe_error("invalid_request", status_code=422, stage="transport")

    @app.middleware("http")
    async def _body_size_middleware(request, call_next):
        return await body_size_guard(request, call_next)

    @app.get(f"{API_PREFIX}/health")
    async def health() -> dict[str, Any]:
        payload = health_payload()
        payload["registry"] = trace_registry.stats()
        payload["concurrency"] = showcase_executor.concurrency_status()
        return payload

    @app.get(f"{API_PREFIX}/scenarios")
    async def scenarios():
        try:
            return list_showcase_scenarios()
        except Exception:
            return safe_error("authority_unavailable", status_code=503, stage="authority", retryable=True)

    @app.get(f"{API_PREFIX}/runtime")
    async def runtime():
        try:
            return safe_runtime_authority()
        except Exception:
            return safe_error("runtime_unavailable", status_code=503, stage="runtime", retryable=True)

    @app.get(f"{API_PREFIX}/system")
    async def system_status():
        try:
            return collect_runtime_status()
        except Exception:
            return safe_error("system_status_unavailable", status_code=503, stage="system_status", retryable=True)

    @app.get(f"{API_PREFIX}/sessions")
    async def list_sessions():
        try:
            return conversation.list_sessions()
        except Exception:
            return safe_error("conversation_unavailable", status_code=503, stage="conversation", retryable=True)

    @app.post(f"{API_PREFIX}/sessions")
    async def create_session(request: ConversationSessionCreateRequest):
        try:
            return conversation.create_session(title=request.title)
        except Exception:
            return safe_error("conversation_unavailable", status_code=503, stage="conversation", retryable=True)

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}")
    async def get_session(session_id: UUID):
        try:
            payload = conversation.get_session(session_id)
            if payload is None:
                return safe_error("conversation_session_not_found", status_code=404, stage="conversation")
            return payload
        except Exception:
            return safe_error("conversation_unavailable", status_code=503, stage="conversation", retryable=True)

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}/turns")
    async def list_session_turns(session_id: UUID):
        try:
            payload = conversation.list_turns(session_id)
            if payload is None:
                return safe_error("conversation_session_not_found", status_code=404, stage="conversation")
            return payload
        except Exception:
            return safe_error("conversation_unavailable", status_code=503, stage="conversation", retryable=True)

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/ask")
    async def ask_session(session_id: UUID, request: ConversationAskRequest):
        try:
            result = await conversation.ask(
                session_id=session_id,
                query=request.query.strip(),
                client_request_id=request.client_request_id,
                top_k=request.top_k,
            )
            payload = dict(result.payload)
            payload["trace"] = _store_trace(trace_registry, result.trace) if result.trace else None
            return payload
        except ValueError:
            return safe_error("conversation_session_not_found", status_code=404, stage="conversation")
        except Exception:
            return safe_error("conversation_execution_failed", status_code=503, stage="conversation", retryable=True)

    @app.post(f"{API_PREFIX}/search")
    async def search(request: SearchRequest):
        try:
            trace = await showcase_executor.run_search(request.query.strip(), top_k=request.top_k)
            return _store_trace(trace_registry, trace)
        except ValueError:
            return safe_error("execution_rejected", status_code=400, stage="search")
        except Exception:
            return safe_error("execution_failed", status_code=503, stage="search", retryable=True)

    @app.post(f"{API_PREFIX}/ask")
    async def ask(request: AskRequest):
        try:
            run_with_result = getattr(showcase_executor, "run_ask_result", None)
            if callable(run_with_result):
                execution = await run_with_result(request.query.strip(), top_k=request.top_k)
                payload = _store_trace(trace_registry, execution.trace)
                payload["result"] = _answer_result_payload(execution.answer)
                return payload
            trace = await showcase_executor.run_ask(request.query.strip(), top_k=request.top_k)
            return _store_trace(trace_registry, trace)
        except ValueError:
            return safe_error("execution_rejected", status_code=400, stage="ask")
        except Exception:
            return safe_error("execution_failed", status_code=503, stage="ask", retryable=True)

    @app.post(f"{API_PREFIX}/scenarios/{{scenario_id}}/run")
    async def run_scenario(scenario_id: str):
        try:
            authority = list_showcase_scenarios()
            valid_ids = {str(row.get("scenario_id")) for row in authority.get("scenarios", [])}
            if scenario_id.upper() not in valid_ids:
                return safe_error("scenario_not_found", status_code=404, stage="scenario")
            trace = await showcase_executor.run_scenario(scenario_id.upper())
            return _store_trace(trace_registry, trace)
        except Exception:
            return safe_error("execution_failed", status_code=503, stage="scenario", retryable=True)

    @app.get(f"{API_PREFIX}/traces/{{trace_id}}")
    async def get_trace(trace_id: str):
        record = trace_registry.get(trace_id)
        if record is None:
            return safe_error("trace_not_found", status_code=404, trace_id=trace_id, stage="trace_lookup")
        status = str((record.trace.get("trace") or {}).get("status") or "failed")
        return TraceEnvelope(trace_id=trace_id, status=status, trace=record.trace).model_dump()

    @app.get(f"{API_PREFIX}/traces/{{trace_id}}/events")
    async def get_trace_events(
        trace_id: str,
        after_sequence: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        replay_after = after_sequence_from_last_event_id(last_event_id, fallback=after_sequence)
        events = trace_registry.replay(trace_id, after_sequence=replay_after)
        if events is None:
            return safe_error("trace_not_found", status_code=404, trace_id=trace_id, stage="event_stream")

        async def stream():
            for event in events:
                yield encode_sse_event(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    app.state.trace_registry = trace_registry
    app.state.showcase_executor = showcase_executor
    app.state.conversation_workspace = conversation
    return app


def _store_trace(registry: TraceRegistry, trace: dict[str, Any]) -> dict[str, Any]:
    checks = validate_runtime_trace(trace)
    if not all(checks.values()):
        raise RuntimeError("runtime_trace_validation_failed")
    record = registry.put(trace)
    status = str((trace.get("trace") or {}).get("status") or "failed")
    return TraceEnvelope(trace_id=record.trace_id, status=status, trace=record.trace).model_dump()


def _answer_result_payload(answer) -> dict[str, Any]:
    """Bounded product result; Runtime Trace remains execution telemetry authority."""
    return {
        "status": answer.status,
        "answerable": answer.answerable,
        "answer": answer.answer,
        "refusal_reason_code": answer.refusal_reason_code,
        "generation_latency_ms": answer.generation_latency_ms,
        "grounding_valid": answer.grounding.valid,
        "citations": [
            {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "relative_path": citation.relative_path,
                "heading_path": list(citation.heading_path),
                "start_line": citation.start_line,
                "end_line": citation.end_line,
                "snippet": citation.snippet,
            }
            for citation in answer.citations
        ],
    }


app = create_app()
