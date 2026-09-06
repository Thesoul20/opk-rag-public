from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.models import AnswerSystemError
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider, RemoteLLMNotAllowedError
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.conversation.repository import ConversationConcurrencyError, ConversationNotFoundError, SessionRepository
from opk_rag.conversation.resolver import HeuristicFollowupQueryResolver, OpenAICompatibleFollowupQueryResolver
from opk_rag.conversation.service import ConversationService
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.indexing.orchestrator import EndToEndIndexReport, index_vault_end_to_end
from opk_rag.lexical.config import LexicalIndexConfig
from opk_rag.lexical.service import index_knowledge_base_lexical
from opk_rag.doctor.service import render_doctor_text, run_doctor
from opk_rag.evaluation.task0177_candidate_retrieval_machine_readable_output_contract_remediation import with_search_output_contract
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import load_reranker_config
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.runtime_trace import RuntimeTraceContext


def main(argv: list[str] | None = None) -> int:
    env_load_result = load_project_env()
    parser = argparse.ArgumentParser(prog="opk-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect reference runtime readiness",
        description="Safe, read-only health checks for the reference runtime. Remote checks are opt-in; --remote-deepseek may incur a small API cost.",
    )
    doctor_parser.add_argument("--remote", action="store_true", help="Enable both remote Supabase and DeepSeek checks.")
    doctor_parser.add_argument("--remote-supabase", action="store_true", help="Enable only the remote Supabase checks.")
    doctor_parser.add_argument("--remote-deepseek", action="store_true", help="Enable only the remote DeepSeek checks.")
    doctor_parser.add_argument("--format", choices=("text", "json"), default="text", help="Render the report as text or JSON.")
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("vault_path", nargs="?")
    index_parser.add_argument("--knowledge-base-name")
    index_parser.add_argument("--knowledge-base-description")
    index_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    index_parser.add_argument("--embedding-local-files-only", action="store_true")
    index_parser.add_argument("--embedding-cache-dir")
    index_parser.add_argument("--lexical-batch-size", type=int, default=100)
    index_parser.add_argument("--format", choices=("text", "json"), default="text")
    search_parser = subparsers.add_parser("search")
    _add_retrieval_args(search_parser)
    search_parser.add_argument("--format", choices=("text", "json"), default="text")
    search_parser.add_argument("--trace", action="store_true", help="Return Runtime Trace V1 JSON for this live Search execution.")
    ask_parser = subparsers.add_parser("ask")
    _add_retrieval_args(ask_parser)
    ask_parser.add_argument("--llm-base-url")
    ask_parser.add_argument("--llm-model-id")
    ask_parser.add_argument("--llm-timeout-seconds", type=float)
    ask_parser.add_argument("--allow-remote-llm", action="store_true")
    ask_parser.add_argument("--format", choices=("text", "json"), default="text")
    ask_parser.add_argument("--trace", action="store_true", help="Return Runtime Trace V1 JSON for this live Ask execution.")
    session_parser = subparsers.add_parser("session")
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)
    session_create = session_subparsers.add_parser("create")
    session_create.add_argument("--knowledge-base-id", required=True)
    session_create.add_argument("--title")
    session_create.add_argument("--format", choices=("text", "json"), default="text")
    session_list = session_subparsers.add_parser("list")
    session_list.add_argument("--knowledge-base-id")
    session_list.add_argument("--include-archived", action="store_true")
    session_list.add_argument("--format", choices=("text", "json"), default="text")
    session_show = session_subparsers.add_parser("show")
    session_show.add_argument("--session-id", required=True)
    session_show.add_argument("--format", choices=("text", "json"), default="text")
    session_delete = session_subparsers.add_parser("delete")
    session_delete.add_argument("--session-id", required=True)
    session_delete.add_argument("--format", choices=("text", "json"), default="text")
    session_ask = session_subparsers.add_parser("ask")
    _add_retrieval_args(session_ask, require_knowledge_base=False)
    session_ask.add_argument("--session-id", required=True)
    session_ask.add_argument("--client-request-id")
    session_ask.add_argument("--llm-base-url")
    session_ask.add_argument("--llm-model-id")
    session_ask.add_argument("--llm-timeout-seconds", type=float)
    session_ask.add_argument("--allow-remote-llm", action="store_true")
    session_ask.add_argument("--resolver", choices=("local-llm", "heuristic"), default="local-llm")
    session_ask.add_argument("--context-max-turns", type=int)
    session_ask.add_argument("--conversation-context-token-budget", type=int)
    session_ask.add_argument("--format", choices=("text", "json"), default="text")
    lexical_parser = subparsers.add_parser("lexical-index")
    lexical_parser.add_argument("--knowledge-base-id", required=True)
    lexical_parser.add_argument("--batch-size", type=int, default=100)
    lexical_parser.add_argument("--dry-run", action="store_true")
    lexical_parser.add_argument("--force-rebuild", action="store_true")
    lexical_parser.add_argument("--format", choices=("text", "json"), default="text")
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the frozen project showcase scenarios from TASK-0218 authority",
        description="Execute the authoritative Showcase Demo Query Set V1 through the existing production Search/Ask runtime.",
    )
    demo_parser.add_argument("--scenario", help="Run one approved frozen scenario, for example S03.")
    demo_parser.add_argument("--list", action="store_true", help="List approved frozen showcase scenarios without running them.")
    demo_parser.add_argument("--format", choices=("text", "json"), default="text")
    demo_parser.add_argument("--trace", action="store_true", help="Return live Runtime Trace V1 JSON for executed scenario(s).")
    api_parser = subparsers.add_parser(
        "showcase-api",
        help="Serve the local Showcase API and Runtime Trace SSE transport.",
        description="Start the local-only Showcase API transport. Default bind is 127.0.0.1:8766.",
    )
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8766)

    args = parser.parse_args(argv)
    if args.command == "index":
        return _index(args)
    if args.command == "doctor":
        return _doctor(args, env_load_result)
    if args.command == "search":
        return _search(args)
    if args.command == "ask":
        return _ask(args)
    if args.command == "session":
        return _session(args)
    if args.command == "lexical-index":
        return _lexical_index(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "showcase-api":
        return _showcase_api(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _showcase_api(args) -> int:
    try:
        import uvicorn

        uvicorn.run(
            "opk_rag.showcase.api.app:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except Exception as exc:
        print(f"Showcase API failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


def _demo(args) -> int:
    from opk_rag.showcase.demo import (
        execute_showcase,
        list_showcase_scenarios,
        render_scenario_list_text,
        render_showcase_text,
    )

    try:
        if args.list:
            payload = list_showcase_scenarios()
            if args.format == "json":
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(render_scenario_list_text(payload))
            return 0
        payload = execute_showcase(args.scenario, trace=args.trace)
    except Exception as exc:
        print(f"Showcase demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        trace_payload = payload.get("runtime_traces") if args.trace and args.scenario is None else (payload.get("runtime_traces") or [None])[0] if args.trace else payload
        print(json.dumps(trace_payload, ensure_ascii=False, sort_keys=True))
    else:
        print(render_showcase_text(payload))
    return 0 if payload.get("execution_status") == "passed" else 1


def _doctor(args, env_load_result) -> int:
    remote_supabase = bool(args.remote or args.remote_supabase)
    remote_deepseek = bool(args.remote or args.remote_deepseek)
    try:
        report = run_doctor(
            env=os.environ,
            load_result=env_load_result,
            remote_supabase=remote_supabase,
            remote_deepseek=remote_deepseek,
        )
    except Exception as exc:
        print(f"Doctor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(render_doctor_text(report))
    if report.overall_status == "healthy":
        return 0
    if report.overall_status == "degraded":
        return 0
    return 1


def _index(args) -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is required.", file=sys.stderr)
        return 2

    vault_arg = args.vault_path or os.environ.get("OPK_RAG_VAULT_PATH", "").strip()
    if not vault_arg:
        print("vault_path is required unless OPK_RAG_VAULT_PATH is set.", file=sys.stderr)
        return 2

    vault_path = Path(vault_arg).expanduser()
    if not vault_path.exists():
        print(f"Vault path does not exist: {vault_path}", file=sys.stderr)
        return 2
    if not vault_path.is_dir():
        print(f"Vault path is not a directory: {vault_path}", file=sys.stderr)
        return 2
    if args.lexical_batch_size < 1:
        print("--lexical-batch-size must be >= 1.", file=sys.stderr)
        return 2

    try:
        embedding_config = load_embedding_config()
        overrides = {}
        if args.device is not None:
            overrides["device"] = args.device
        if args.embedding_local_files_only:
            overrides["local_files_only"] = True
        if args.embedding_cache_dir:
            overrides["cache_dir"] = Path(args.embedding_cache_dir).expanduser()
        if overrides:
            embedding_config = _replace_dataclass(embedding_config, **overrides)
        provider = QwenLocalEmbeddingProvider(embedding_config)
        report = index_vault_end_to_end(
            database_url,
            vault_path,
            embedding_config=embedding_config,
            embedding_provider=provider,
            lexical_batch_size=args.lexical_batch_size,
            knowledge_base_name=args.knowledge_base_name,
            knowledge_base_description=args.knowledge_base_description,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Index failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(_index_report_to_json(report), ensure_ascii=False, sort_keys=True))
    else:
        print(_index_report_to_text(report))
    return 0 if report.successful else 1


def _add_retrieval_args(parser, *, require_knowledge_base: bool = True) -> None:
    parser.add_argument("--knowledge-base-id", required=require_knowledge_base)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--rerank-top-n", type=int)
    parser.add_argument("--bm25-candidate-k", type=int)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"))
    parser.add_argument("--bm25-k1", type=float)
    parser.add_argument("--bm25-b", type=float)
    parser.add_argument("--rrf-k", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--rerank", action="store_true", help="Enable the default rank-fusion reranker.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking and use original retrieval ranking.")
    parser.add_argument("--reranker-device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--reranker-local-files-only", action="store_true")
    parser.add_argument("--reranker-cache-dir")
    parser.add_argument("--reranker-max-pair-tokens", type=int)
    parser.add_argument("--search-reranker-precision", choices=("fp32", "fp16_autocast"))
    parser.add_argument("--ask-reranker-precision", choices=("fp32", "fp16_autocast"))
    parser.add_argument("--context-token-budget", type=int)
    parser.add_argument("--context-max-chunks", type=int)
    parser.add_argument("--context-max-chunks-per-document", type=int)


def _search(args) -> int:
    if args.format == "json":
        with redirect_stdout(sys.stderr):
            response, code = _run_search_from_args(args, execution_scope="search")
    else:
        response, code = _run_search_from_args(args, execution_scope="search")
    if code != 0:
        return code

    if args.format == "json":
        payload = response.runtime_trace if getattr(args, "trace", False) else with_search_output_contract(_response_to_json(response))
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(_response_to_text(response))
    return 0


def _ask(args) -> int:
    if args.format == "json":
        with redirect_stdout(sys.stderr):
            response, code = _run_search_from_args(args, execution_scope="ask")
    else:
        response, code = _run_search_from_args(args, execution_scope="ask")
    if code != 0:
        return code
    try:
        answer_config = load_answer_generation_config()
        answer_overrides = {}
        if args.llm_base_url:
            answer_overrides["base_url"] = args.llm_base_url
        if args.llm_model_id:
            answer_overrides["model_id"] = args.llm_model_id
        if args.llm_timeout_seconds is not None:
            answer_overrides["timeout_seconds"] = args.llm_timeout_seconds
        if args.allow_remote_llm:
            answer_overrides["allow_remote"] = True
        if answer_overrides:
            answer_config = _replace_dataclass(answer_config, **answer_overrides)
        provider = OpenAICompatibleLocalChatProvider(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
        answer = answer_knowledge_base(response, provider=provider, config=answer_config, runtime_trace_context=getattr(args, "_runtime_trace_context", None))
    except RemoteLLMNotAllowedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except AnswerSystemError as exc:
        print(f"Answer system error ({exc.code}): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Answer system error: {type(exc).__name__}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(answer.runtime_trace if getattr(args, "trace", False) else _answer_to_json(answer), ensure_ascii=False, sort_keys=True))
    else:
        print(_answer_to_text(answer))
    return 0


def _session(args) -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is required.", file=sys.stderr)
        return 2
    try:
        with connect_postgres(database_url) as connection:
            repo = SessionRepository(connection)
            if args.session_command == "create":
                knowledge_base_id = _parse_uuid_arg(args.knowledge_base_id, "--knowledge-base-id")
                session = repo.create_session(knowledge_base_id=knowledge_base_id, title=args.title)
                _print_session_result(args.format, {"session": _session_to_json(session)}, _session_to_text(session))
                return 0
            if args.session_command == "list":
                knowledge_base_id = _parse_uuid_arg(args.knowledge_base_id, "--knowledge-base-id") if args.knowledge_base_id else None
                sessions = repo.list_sessions(knowledge_base_id=knowledge_base_id, include_archived=args.include_archived)
                _print_session_result(args.format, {"sessions": [_session_to_json(session) for session in sessions]}, "\n".join(_session_to_text(session) for session in sessions) or "No sessions found.")
                return 0
            if args.session_command == "show":
                session_id = _parse_uuid_arg(args.session_id, "--session-id")
                session = repo.get_session(session_id)
                if session is None:
                    print("Conversation session not found.", file=sys.stderr)
                    return 1
                turns = repo.list_turns(session_id)
                citations = repo.list_citations_for_turns(tuple(turn.id for turn in turns))
                payload = {"session": _session_to_json(session), "turns": [_turn_to_json(turn, citations.get(turn.id, ())) for turn in turns]}
                _print_session_result(args.format, payload, _conversation_to_text(session, turns, citations))
                return 0
            if args.session_command == "delete":
                session_id = _parse_uuid_arg(args.session_id, "--session-id")
                session = repo.delete_session(session_id)
                _print_session_result(args.format, {"session": _session_to_json(session)}, f"deleted session: {session.id}")
                return 0
            if args.session_command == "ask":
                return _session_ask(args, database_url, connection, repo)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConversationConcurrencyError, ConversationNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Session command failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Unknown session command: {args.session_command}", file=sys.stderr)
    return 2


def _session_ask(args, database_url: str, connection, repo: SessionRepository) -> int:
    session_id = _parse_uuid_arg(args.session_id, "--session-id")
    session = repo.get_session(session_id)
    if session is None:
        print("Conversation session not found.", file=sys.stderr)
        return 1
    answer_config = load_answer_generation_config()
    answer_overrides = {}
    if args.llm_base_url:
        answer_overrides["base_url"] = args.llm_base_url
    if args.llm_model_id:
        answer_overrides["model_id"] = args.llm_model_id
    if args.llm_timeout_seconds is not None:
        answer_overrides["timeout_seconds"] = args.llm_timeout_seconds
    if args.allow_remote_llm:
        answer_overrides["allow_remote"] = True
    if answer_overrides:
        answer_config = _replace_dataclass(answer_config, **answer_overrides)
    try:
        answer_provider = OpenAICompatibleLocalChatProvider(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
        resolver = (
            HeuristicFollowupQueryResolver()
            if args.resolver == "heuristic"
            else OpenAICompatibleFollowupQueryResolver(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
        )
    except RemoteLLMNotAllowedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    embedding_config = load_embedding_config()
    if args.device is not None:
        embedding_config = _replace_dataclass(embedding_config, device=args.device)
    context_token_counter = QwenContextTokenCounter(embedding_config)

    def run_search(query: str):
        search_args = SimpleNamespace(**vars(args))
        search_args.knowledge_base_id = str(session.knowledge_base_id)
        search_args.query = query
        response, code = _run_search_from_args(search_args, execution_scope="ask")
        if code != 0:
            raise RuntimeError("Search failed for conversation turn.")
        return response

    service = ConversationService(
        repository=repo,
        resolver=resolver,
        answer_provider=answer_provider,
        answer_config=answer_config,
        search_runner=run_search,
        context_token_counter=context_token_counter,
        context_max_turns=args.context_max_turns or 6,
        context_token_budget=args.conversation_context_token_budget or 1200,
    )
    try:
        result = service.ask(session_id=session_id, user_query=args.query, client_request_id=args.client_request_id)
    except AnswerSystemError as exc:
        print(f"Answer system error ({exc.code}): {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(_conversation_turn_response_to_json(result), ensure_ascii=False, sort_keys=True))
    else:
        print(_conversation_turn_response_to_text(result))
    return 0


def _run_search_from_args(args, *, execution_scope: str = "search"):
    try:
        knowledge_base_id = UUID(args.knowledge_base_id)
    except ValueError:
        print("Invalid --knowledge-base-id; expected UUID.", file=sys.stderr)
        return None, 2

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is required.", file=sys.stderr)
        return None, 2

    try:
        base_search_config = load_vector_search_config()
        overrides = {}
        if args.top_k is not None:
            overrides["top_k"] = args.top_k
            overrides["candidate_k"] = max(base_search_config.candidate_k, args.top_k)
            overrides["rerank_top_n"] = max(base_search_config.rerank_top_n, args.top_k)
            overrides["bm25_candidate_k"] = max(base_search_config.bm25_candidate_k, args.top_k)
        if args.candidate_k is not None:
            overrides["candidate_k"] = args.candidate_k
        if args.rerank_top_n is not None:
            overrides["rerank_top_n"] = args.rerank_top_n
        if args.bm25_candidate_k is not None:
            overrides["bm25_candidate_k"] = args.bm25_candidate_k
        if args.min_similarity is not None:
            overrides["min_similarity"] = args.min_similarity
        if args.mode is not None:
            overrides["mode"] = args.mode
        if args.bm25_k1 is not None:
            overrides["bm25_k1"] = args.bm25_k1
        if args.bm25_b is not None:
            overrides["bm25_b"] = args.bm25_b
        if args.rrf_k is not None:
            overrides["rrf_k"] = args.rrf_k
        if args.rerank:
            overrides["rerank_enabled"] = True
        if args.no_rerank:
            overrides["rerank_enabled"] = False
        if args.context_token_budget is not None:
            overrides["context_token_budget"] = args.context_token_budget
        if args.context_max_chunks is not None:
            overrides["context_max_chunks"] = args.context_max_chunks
        if args.context_max_chunks_per_document is not None:
            overrides["context_max_chunks_per_document"] = args.context_max_chunks_per_document
        search_config = _replace_dataclass(base_search_config, **overrides)
        embedding_config = load_embedding_config()
        if args.device is not None:
            embedding_config = _replace_dataclass(embedding_config, device=args.device)
        context_token_counter = QwenContextTokenCounter(embedding_config)
        if search_config.mode in {"vector", "hybrid"}:
            provider = QwenLocalEmbeddingProvider(embedding_config)
        else:
            provider = None
        reranker_provider = None
        if search_config.rerank_enabled:
            reranker_config = load_reranker_config()
            reranker_overrides = {}
            if args.reranker_device is not None:
                reranker_overrides["device"] = args.reranker_device
            if args.reranker_local_files_only:
                reranker_overrides["local_files_only"] = True
            if args.reranker_cache_dir:
                from pathlib import Path

                reranker_overrides["cache_dir"] = Path(args.reranker_cache_dir).expanduser()
            if args.reranker_max_pair_tokens is not None:
                reranker_overrides["max_pair_tokens"] = args.reranker_max_pair_tokens
            if getattr(args, "search_reranker_precision", None) is not None:
                reranker_overrides["search_precision"] = args.search_reranker_precision
            if getattr(args, "ask_reranker_precision", None) is not None:
                reranker_overrides["ask_precision"] = args.ask_reranker_precision
            reranker_provider = BgeLocalRerankerProvider(_replace_dataclass(reranker_config, **reranker_overrides))
        trace_context = (
            RuntimeTraceContext(
                query_text=args.query,
                execution_scope=execution_scope,
                enabled=True,
            )
            if getattr(args, "trace", False)
            else None
        )
        response = search_knowledge_base(
            database_url,
            knowledge_base_id=knowledge_base_id,
            query=args.query,
            provider=provider,
            embedding_config=embedding_config,
            search_config=search_config,
            reranker_provider=reranker_provider,
            context_token_counter=context_token_counter,
            execution_scope=execution_scope,
            runtime_trace_context=trace_context,
        )
        if getattr(args, "trace", False):
            args._runtime_trace_context = trace_context
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None, 2
    except Exception as exc:
        print(f"Search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, 1
    return response, 0


def _lexical_index(args) -> int:
    try:
        knowledge_base_id = UUID(args.knowledge_base_id)
    except ValueError:
        print("Invalid --knowledge-base-id; expected UUID.", file=sys.stderr)
        return 2

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is required.", file=sys.stderr)
        return 2

    try:
        result = index_knowledge_base_lexical(
            database_url,
            knowledge_base_id=knowledge_base_id,
            config=LexicalIndexConfig(),
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            force_rebuild=args.force_rebuild,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Lexical indexing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = {
        "total_chunks": result.total_chunks,
        "pending_chunks": result.pending_chunks,
        "indexed_chunks": result.indexed_chunks,
        "skipped_chunks": result.skipped_chunks,
        "failed_chunks": result.failed_chunks,
        "indexed_chunk_count": result.indexed_chunk_count,
        "unique_term_count": result.unique_term_count,
        "average_document_length": result.average_document_length,
        "tokenizer_id": result.tokenizer_id,
        "tokenizer_version": result.tokenizer_version,
        "configuration_fingerprint": result.configuration_fingerprint,
        "lexical_ready": result.ready,
    }
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "\n".join(
                (
                    f"total_chunks: {payload['total_chunks']}",
                    f"pending_chunks: {payload['pending_chunks']}",
                    f"indexed_chunks: {payload['indexed_chunks']}",
                    f"skipped_chunks: {payload['skipped_chunks']}",
                    f"failed_chunks: {payload['failed_chunks']}",
                    f"indexed_chunk_count: {payload['indexed_chunk_count']}",
                    f"unique_term_count: {payload['unique_term_count']}",
                    f"average_document_length: {payload['average_document_length']:.4f}",
                    f"tokenizer_id: {payload['tokenizer_id']}",
                    f"tokenizer_version: {payload['tokenizer_version']}",
                    f"configuration_fingerprint: {payload['configuration_fingerprint']}",
                    f"lexical_ready: {str(payload['lexical_ready']).lower()}",
                )
            )
        )
    return 1 if result.failed_chunks else 0


class _BM25OnlyProvider:
    dimension = 1024

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed_query(self, query: str, *, instruction: str | None = None):
        raise RuntimeError("BM25 mode does not use query embeddings.")


def _replace_dataclass(instance, **changes):
    return replace(instance, **changes)


def _response_to_json(response) -> dict:
    return {
        "query": response.query,
        "normalized_query": response.normalized_query,
        "knowledge_base_id": str(response.knowledge_base_id),
        "model_id": response.model_id,
        "retrieval_mode": response.retrieval_mode,
        "query_template_version": response.query_template_version,
        "requested_top_k": response.requested_top_k,
        "candidate_k": response.candidate_k,
        "candidate_count": response.candidate_count,
        "vector_candidate_count": response.vector_candidate_count,
        "bm25_candidate_count": response.bm25_candidate_count,
        "threshold_filtered_count": response.threshold_filtered_count,
        "deduplicated_count": response.deduplicated_count,
        "result_count": response.result_count,
        "query_token_count": response.query_token_count,
        "query_input_token_count": response.query_input_token_count,
        "lexical_query_terms": list(response.lexical_query_terms),
        "lexical_ready": response.lexical_ready,
        "retrieval_degraded": response.retrieval_degraded,
        "reranker_enabled": response.reranker_enabled,
        "reranker_model_id": response.reranker_model_id,
        "reranker_model_revision": response.reranker_model_revision,
        "reranker_input_template_version": response.reranker_input_template_version,
        "reranker_score_semantics": response.reranker_score_semantics,
        "rerank_top_n": response.rerank_top_n,
        "rerank_candidate_count": response.rerank_candidate_count,
        "final_top_k": response.final_top_k,
        "context_token_budget": response.context_token_budget,
        "context_token_count": response.context_token_count,
        "evidence_bundle": None
        if response.evidence_bundle is None
        else {
            "query": response.evidence_bundle.query,
            "normalized_query": response.evidence_bundle.normalized_query,
            "knowledge_base_id": str(response.evidence_bundle.knowledge_base_id),
            "context_token_budget": response.evidence_bundle.context_token_budget,
            "context_token_count": response.evidence_bundle.context_token_count,
            "total_token_count": response.evidence_bundle.total_token_count,
            "context_tokenizer_id": response.evidence_bundle.context_tokenizer_id,
            "context_tokenizer_revision": response.evidence_bundle.context_tokenizer_revision,
            "items": [
                {
                    "context_rank": item.context_rank,
                    "result_rank": item.result_rank,
                    "chunk_id": str(item.chunk_id),
                    "document_id": str(item.document_id),
                    "relative_path": item.relative_path,
                    "heading_path": list(item.heading_path),
                    "content": item.content,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "token_count": item.context_token_count,
                    "context_token_count": item.context_token_count,
                    "reranker_pair_token_count": item.reranker_pair_token_count,
                    "reranker_original_pair_token_count": item.reranker_original_pair_token_count,
                    "reranker_input_truncated": item.reranker_input_truncated,
                    "rerank_score": item.rerank_score,
                    "retrieval_sources": list(item.retrieval_sources),
                }
                for item in response.evidence_bundle.items
            ],
        },
        "evidence_signals": None
        if response.evidence_signals is None
        else {
            "top_reranker_score": response.evidence_signals.top_reranker_score,
            "second_reranker_score": response.evidence_signals.second_reranker_score,
            "top1_top2_margin": response.evidence_signals.top1_top2_margin,
            "max_vector_similarity": response.evidence_signals.max_vector_similarity,
            "max_bm25_score": response.evidence_signals.max_bm25_score,
            "max_rrf_score": response.evidence_signals.max_rrf_score,
            "dual_channel_candidate_count": response.evidence_signals.dual_channel_candidate_count,
            "selected_source_document_count": response.evidence_signals.selected_source_document_count,
            "selected_chunk_count": response.evidence_signals.selected_chunk_count,
            "selected_context_token_count": response.evidence_signals.selected_context_token_count,
            "query_term_coverage": response.evidence_signals.query_term_coverage,
            "exact_identifier_match": response.evidence_signals.exact_identifier_match,
            "any_reranker_input_truncated": response.evidence_signals.any_reranker_input_truncated,
            "reranker_score_min": response.evidence_signals.reranker_score_min,
            "reranker_score_max": response.evidence_signals.reranker_score_max,
            "reranker_score_mean": response.evidence_signals.reranker_score_mean,
        },
        "graph_trace": dict(response.graph_trace or {}),
        "guard_trace": dict(response.guard_trace or {}),
        "results": [
            {
                "rank": result.rank,
                "document_id": str(result.document_id),
                "chunk_id": str(result.chunk_id),
                "relative_path": result.relative_path,
                "heading_path": list(result.heading_path),
                "content": result.content,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "similarity": result.similarity,
                "vector_similarity": result.vector_similarity,
                "bm25_score": result.bm25_score,
                "vector_rank": result.vector_rank,
                "bm25_rank": result.bm25_rank,
                "rrf_score": result.rrf_score,
                "original_rank": result.original_rank,
                "rerank_score": result.rerank_score,
                "rerank_rank": result.rerank_rank,
                "pair_token_count": result.pair_token_count,
                "reranker_pair_token_count": result.reranker_pair_token_count,
                "reranker_original_pair_token_count": result.reranker_original_pair_token_count,
                "reranker_input_truncated": result.reranker_input_truncated,
                "context_token_count": result.context_token_count,
                "selected_for_context": result.selected_for_context,
                "context_rank": result.context_rank,
                "retrieval_sources": list(result.retrieval_sources),
                "matched_terms": list(result.matched_terms),
            }
            for result in response.results
        ],
    }


def _response_to_text(response) -> str:
    if not response.results:
        return "No matching chunks found."
    blocks = []
    for result in response.results:
        heading = result.heading or "(no heading)"
        lines = _format_lines(result.start_line, result.end_line)
        content = _truncate_content(result.content)
        blocks.append(
            f"{result.rank}. score: {_format_score(response.retrieval_mode, result)}\n"
            f"   rerank: {_format_rerank(result)}\n"
            f"   context: {_format_context(result)}\n"
            f"   sources: {', '.join(result.retrieval_sources) or 'unknown'}\n"
            f"   matched_terms: {', '.join(result.matched_terms) if result.matched_terms else '(none)'}\n"
            f"   source: {result.relative_path}\n"
            f"   heading: {heading}\n"
            f"   lines: {lines}\n\n"
            f"   {content.replace(chr(10), chr(10) + '   ')}"
        )
    return "\n\n".join(blocks)


def _format_lines(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return "unknown"
    return f"{start_line}-{end_line}"


def _format_score(mode: str, result) -> str:
    if mode == "bm25":
        return f"bm25_score={result.bm25_score or 0.0:.4f}"
    if mode == "hybrid":
        similarity = f"{result.similarity:.4f}" if result.vector_rank is not None else "null"
        bm25_score = f"{result.bm25_score:.4f}" if result.bm25_score is not None else "null"
        return (
            f"rrf={result.rrf_score or 0.0:.6f}, "
            f"vector_rank={result.vector_rank}, bm25_rank={result.bm25_rank}, "
            f"similarity={similarity}, bm25_score={bm25_score}"
        )
    return f"similarity={result.similarity:.4f}"


def _format_rerank(result) -> str:
    if result.rerank_score is None:
        return "disabled"
    return f"rank={result.rerank_rank}, score={result.rerank_score:.6f}, original_rank={result.original_rank}"


def _format_context(result) -> str:
    selected = "selected" if result.selected_for_context else "not_selected"
    return (
        f"{selected}, context_rank={result.context_rank}, "
        f"context_tokens={result.context_token_count}, reranker_pair_tokens={result.reranker_pair_token_count}"
    )


def _truncate_content(content: str, limit: int = 1200) -> str:
    if len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n[truncated]"


def _answer_to_json(answer) -> dict:
    search_payload = _response_to_json(answer.search_response)
    guard_trace = dict(search_payload.get("guard_trace") or {})
    guard_trace.update(
        {
            "final_decision": "answer" if answer.status == "answered" else "fail_closed",
            "answer_status": answer.status,
            "fail_closed": answer.status == "refused",
            "refusal_reason_code": answer.refusal_reason_code,
            "bounded_recovery_trace_valid": int(guard_trace.get("recovery_attempt_count", 0)) <= int(guard_trace.get("maximum_recovery_attempt_count", 1)),
        }
    )
    search_payload["guard_trace"] = guard_trace
    return {
        "status": answer.status,
        "answerable": answer.answerable,
        "answer": answer.answer,
        "refusal_reason_code": answer.refusal_reason_code,
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
        "provider_id": answer.provider_id,
        "model_id": answer.model_id,
        "model_revision": answer.model_revision,
        "model_license": answer.model_license,
        "runtime_base_url": answer.runtime_base_url,
        "runtime_endpoint_type": answer.runtime_endpoint_type,
        "prompt_version": answer.prompt_version,
        "output_schema_version": answer.output_schema_version,
        "evidence_serialization": answer.evidence_serialization,
        "generation_latency_ms": answer.generation_latency_ms,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "total_tokens": answer.total_tokens,
        "unsupported_claims": list(answer.unsupported_claims),
        "model_decision": answer.model_decision,
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
        "search": search_payload,
    }


def _answer_to_text(answer) -> str:
    if not answer.answerable:
        return (
            f"{answer.answer}\n\n"
            f"status: {answer.status}\n"
            f"refusal_reason_code: {answer.refusal_reason_code}\n"
            f"answerability_reason_code: {answer.answerability.reason_code}\n"
            f"grounding_reason_code: {answer.grounding.reason_code}"
        )
    citation_lines = [
        f"[{citation.citation_id}] {citation.relative_path}:{_format_lines(citation.start_line, citation.end_line)}"
        for citation in answer.citations
    ]
    return answer.answer + "\n\nstatus: answered\ngrounding: " + answer.grounding.reason_code + "\n" + "\n".join(citation_lines)


def _index_report_to_json(report: EndToEndIndexReport) -> dict:
    return {
        "status": "completed" if report.successful else "partial",
        "vault": report.vault_path,
        "knowledge_base_id": str(report.knowledge_base_id),
        "index_run_id": str(report.index_run_id) if report.index_run_id else None,
        "configuration_fingerprint": report.configuration_fingerprint,
        "scanned_files": report.scanned_files,
        "added_files": report.added_files,
        "updated_files": report.updated_files,
        "unchanged_files": report.unchanged_files,
        "deleted_files": report.deleted_files,
        "restored_files": report.restored_files,
        "failed_files": report.failed_files,
        "documents_written": report.documents_written,
        "chunks_written": report.chunks_written,
        "chunks_deleted": report.chunks_deleted,
        "embeddings_written": report.embeddings_written,
        "embeddings_skipped": report.embeddings_skipped,
        "lexical_records_updated": report.lexical_records_updated,
        "lexical_records_removed": report.lexical_records_removed,
        "lexical_ready": report.lexical_ready,
        "duration_seconds": round(report.duration_seconds, 3),
        "failures": [
            {"relative_path": failure.relative_path, "stage": failure.stage, "message": failure.message}
            for failure in report.failures
        ],
    }


def _index_report_to_text(report: EndToEndIndexReport) -> str:
    lines = [
        "Index completed" if report.successful else "Index completed with failures",
        "",
        f"Vault: {report.vault_path}",
        f"Knowledge base: {report.knowledge_base_id}",
        f"Index run: {report.index_run_id}",
        "",
        f"Scanned files: {report.scanned_files}",
        f"Added files: {report.added_files}",
        f"Updated files: {report.updated_files}",
        f"Unchanged files: {report.unchanged_files}",
        f"Deleted files: {report.deleted_files}",
        f"Restored files: {report.restored_files}",
        f"Failed files: {report.failed_files}",
        "",
        f"Documents written: {report.documents_written}",
        f"Chunks written: {report.chunks_written}",
        f"Chunks deleted: {report.chunks_deleted}",
        f"Embeddings written: {report.embeddings_written}",
        f"Embeddings skipped: {report.embeddings_skipped}",
        f"Lexical records updated: {report.lexical_records_updated}",
        f"Lexical records removed: {report.lexical_records_removed}",
        f"Lexical ready: {str(report.lexical_ready).lower()}",
        "",
        f"Duration: {report.duration_seconds:.3f}s",
    ]
    if report.failures:
        lines.extend(("", "Failures:"))
        lines.extend(f"- {failure.relative_path} [{failure.stage}]: {failure.message}" for failure in report.failures)
    return "\n".join(lines)


def _parse_uuid_arg(value: str, name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}; expected UUID.") from exc


def _print_session_result(output_format: str, payload: dict, text: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(text)


def _session_to_json(session) -> dict:
    return {
        "id": str(session.id),
        "knowledge_base_id": str(session.knowledge_base_id),
        "title": session.title,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "last_turn_at": session.last_turn_at.isoformat() if session.last_turn_at else None,
        "turn_count": session.turn_count,
        "conversation_prompt_version": session.conversation_prompt_version,
        "followup_rewrite_version": session.followup_rewrite_version,
        "metadata": session.metadata,
    }


def _turn_to_json(turn, citations=()) -> dict:
    return {
        "id": str(turn.id),
        "session_id": str(turn.session_id),
        "turn_number": turn.turn_number,
        "client_request_id": turn.client_request_id,
        "user_query": turn.user_query,
        "standalone_query": turn.standalone_query,
        "rewrite_status": turn.rewrite_status,
        "answer_decision": turn.answer_decision,
        "answer_text": turn.answer_text,
        "abstention_reason": turn.abstention_reason,
        "turn_status": turn.turn_status,
        "retrieval_mode": turn.retrieval_mode,
        "reranking_enabled": turn.reranking_enabled,
        "provider_id": turn.provider_id,
        "model_id": turn.model_id,
        "model_version": turn.model_version,
        "prompt_fingerprint": turn.prompt_fingerprint,
        "input_token_count": turn.input_token_count,
        "output_token_count": turn.output_token_count,
        "latency_ms": turn.latency_ms,
        "error_code": turn.error_code,
        "created_at": turn.created_at.isoformat(),
        "completed_at": turn.completed_at.isoformat() if turn.completed_at else None,
        "metadata": turn.metadata,
        "citations": [_conversation_citation_to_json(citation) for citation in citations],
    }


def _conversation_citation_to_json(citation) -> dict:
    return {
        "turn_id": str(citation.turn_id),
        "citation_id": citation.citation_id,
        "document_id": str(citation.document_id),
        "chunk_id": str(citation.chunk_id),
        "relative_path": citation.relative_path,
        "heading_path": list(citation.heading_path),
        "start_line": citation.start_line,
        "end_line": citation.end_line,
        "content_hash": citation.content_hash,
        "context_rank": citation.context_rank,
        "created_at": citation.created_at.isoformat() if citation.created_at else None,
        "source_status": citation.source_status,
    }


def _conversation_to_text(session, turns, citations_by_turn) -> str:
    lines = [_session_to_text(session)]
    for turn in turns:
        lines.append("")
        lines.append(f"Turn {turn.turn_number}: {turn.turn_status}")
        lines.append(f"Q: {turn.user_query}")
        if turn.standalone_query:
            lines.append(f"Standalone: {turn.standalone_query}")
        if turn.answer_text:
            lines.append(f"A: {turn.answer_text}")
        if turn.abstention_reason:
            lines.append(f"abstention_reason: {turn.abstention_reason}")
        for citation in citations_by_turn.get(turn.id, ()):
            status = f" ({citation.source_status})" if citation.source_status else ""
            lines.append(f"[{citation.citation_id}] {citation.relative_path}:{_format_lines(citation.start_line, citation.end_line)}{status}")
    return "\n".join(lines)


def _session_to_text(session) -> str:
    title = f" title={session.title}" if session.title else ""
    return f"session: {session.id} kb={session.knowledge_base_id} status={session.status} turns={session.turn_count}{title}"


def _conversation_turn_response_to_json(result) -> dict:
    return {
        "session": _session_to_json(result.session),
        "turn": _turn_to_json(result.turn, result.citations),
        "resolution": {
            "standalone_query": result.resolution.standalone_query,
            "is_followup": result.resolution.is_followup,
            "referenced_turn_numbers": list(result.resolution.referenced_turn_numbers),
            "referenced_citation_ids": list(result.resolution.referenced_citation_ids),
            "resolution_reason": result.resolution.resolution_reason,
            "model_id": result.resolution.model_id,
            "model_revision": result.resolution.model_revision,
            "prompt_version": result.resolution.prompt_version,
            "output_schema_version": result.resolution.output_schema_version,
            "input_token_count": result.resolution.input_token_count,
            "output_token_count": result.resolution.output_token_count,
            "latency_ms": result.resolution.latency_ms,
        },
        "answer": None if result.answer is None else _answer_to_json(result.answer),
        "idempotent_replay": result.idempotent_replay,
    }


def _conversation_turn_response_to_text(result) -> str:
    lines = [
        f"session: {result.session.id}",
        f"turn: {result.turn.turn_number}",
        f"standalone_query: {result.resolution.standalone_query}",
        f"is_followup: {str(result.resolution.is_followup).lower()}",
    ]
    if result.idempotent_replay:
        lines.append("idempotent_replay: true")
        if result.turn.answer_text:
            lines.append("")
            lines.append(result.turn.answer_text)
    elif result.answer is not None:
        lines.append("")
        lines.append(_answer_to_text(result.answer))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
