from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from uuid import UUID
import math

from opk_rag.db.connection import connect_postgres
from opk_rag.db.models import BM25SearchRow, ChunkSearchRow
from opk_rag.db.repositories import (
    BM25SearchRepository,
    ChunkSearchRepository,
    IndexConfigurationRepository,
    KnowledgeBaseRepository,
    LexicalIndexRepository,
)
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.embedding.provider import EmbeddingOutputValidationError, EmbeddingProvider
from opk_rag.embedding.query import prepare_query_input
from opk_rag.embedding.service import _validate_vectors
from opk_rag.lexical.config import LexicalIndexConfig, build_lexical_configuration_fingerprint
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer, normalize_lexical_text
from opk_rag.reranking.config import RerankerExecutionScope
from opk_rag.reranking.input import RERANKER_SCORE_SEMANTICS, render_reranker_document
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerProvider
from opk_rag.runtime_v2.rank_fusion import RankFusionParameters, rank_fusion_order
from opk_rag.runtime_v2.public_search_runtime import integrate_public_search_runtime
from opk_rag.search.config import VectorSearchConfig
from opk_rag.search.context_tokens import ContextTokenCounter, SimpleContextTokenCounter
from opk_rag.search.models import EvidenceBundle, EvidenceItem, EvidenceSignals, SearchResponse, SearchResult
from opk_rag.search.retrieval_timing import RetrievalTimingObserver
from opk_rag.showcase.runtime_trace import RuntimeTraceContext, build_runtime_trace_from_live_execution
from opk_rag.vector_backends.shadow import maybe_observe_qdrant_shadow
from opk_rag.vector_backends.base import VectorBackendCandidate, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config


class SearchError(RuntimeError):
    pass


class KnowledgeBaseNotFoundError(SearchError):
    pass


class LexicalIndexNotReadyError(SearchError):
    pass


def search_knowledge_base(
    database_url: str,
    *,
    knowledge_base_id: UUID,
    query: str,
    provider: EmbeddingProvider | None,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    reranker_provider: RerankerProvider | None = None,
    context_token_counter: ContextTokenCounter | None = None,
    timing_observer: RetrievalTimingObserver | None = None,
    execution_scope: RerankerExecutionScope = "search",
    runtime_trace_context: RuntimeTraceContext | None = None,
) -> SearchResponse:
    observer = timing_observer or RetrievalTimingObserver(enabled=bool(runtime_trace_context and runtime_trace_context.enabled))
    observer.increment("database_connection_creation_count")
    with connect_postgres(database_url) as connection:
        return search_knowledge_base_connection(
            connection,
            knowledge_base_id=knowledge_base_id,
            query=query,
            provider=provider,
            embedding_config=embedding_config,
            search_config=search_config,
            reranker_provider=reranker_provider,
            context_token_counter=context_token_counter,
            timing_observer=observer,
            execution_scope=execution_scope,
            runtime_trace_context=runtime_trace_context,
        )


def search_knowledge_base_connection(
    connection,
    *,
    knowledge_base_id: UUID,
    query: str,
    provider: EmbeddingProvider | None,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    reranker_provider: RerankerProvider | None = None,
    context_token_counter: ContextTokenCounter | None = None,
    timing_observer: RetrievalTimingObserver | None = None,
    execution_scope: RerankerExecutionScope = "search",
    runtime_trace_context: RuntimeTraceContext | None = None,
) -> SearchResponse:
    observer = timing_observer or RetrievalTimingObserver(enabled=bool(runtime_trace_context and runtime_trace_context.enabled))
    with observer.time_stage("request_validation"):
        if search_config.mode in {"vector", "hybrid"} and provider is None:
            raise SearchError(f"{search_config.mode} mode requires an embedding provider.")
        if context_token_counter is None:
            context_token_counter = _provider_context_counter(provider, embedding_config)
    with observer.time_stage("query_preparation"):
        observer.increment("postgresql_query_count")
        if KnowledgeBaseRepository(connection).get_by_id(knowledge_base_id) is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge base not found: {knowledge_base_id}")
    index_configuration = None
    if search_config.mode in {"vector", "hybrid"}:
        with observer.time_stage("query_preparation"):
            observer.increment("postgresql_query_count")
            index_configuration = IndexConfigurationRepository(connection).require_by_fingerprint(
                build_configuration_fingerprint(embedding_config)
            )

    prepared = None
    shadow_observation = None
    vector_rows: tuple[ChunkSearchRow, ...] = ()
    thresholded_vector: tuple[ChunkSearchRow, ...] = ()
    lexical_terms: tuple[str, ...] = ()
    bm25_rows: tuple[BM25SearchRow, ...] = ()
    lexical_ready = False

    if search_config.mode in {"vector", "hybrid"}:
        assert provider is not None
        timed_provider = _timed_embedding_provider(provider, observer)
        with observer.time_stage("query_preprocessing"):
            prepared = prepare_query_input(
                query,
                embedding_config,
                count_tokens=timed_provider.count_tokens,
                instruction=search_config.query_instruction,
                template_version=search_config.query_template_version,
                max_query_tokens=search_config.max_query_tokens,
            )
        with observer.time_stage("query_embedding"):
            observer.increment("embedding_call_count")
            with observer.time_stage("embedding_forward"):
                vector = provider.embed_query(prepared.query, instruction=prepared.instruction)
        _validate_query_vector(vector, provider=provider, config=embedding_config)
        embedding_revision = build_configuration_fingerprint(embedding_config)
        with observer.time_stage("retrieval_total"):
            with observer.time_stage("vector_search"):
                observer.increment("qdrant_request_count" if _load_vector_backend_id(os.environ) == "qdrant" else "pgvector_request_count")
                vector_rows = _search_vector_candidates(
                    connection,
                    knowledge_base_id=knowledge_base_id,
                    query_embedding=vector,
                    match_count=search_config.candidate_k,
                    index_configuration_id=index_configuration.id if index_configuration is not None else None,
                    embedding_revision=embedding_revision,
                    vector_size=embedding_config.dimension,
                    env=os.environ,
                    timing_observer=observer,
                )
        if _load_vector_backend_id(os.environ) == "postgres_pgvector":
            shadow_observation = maybe_observe_qdrant_shadow(
                env=os.environ,
                query_id=prepared.text_hash,
                query_embedding=vector,
                authoritative_rows=vector_rows,
                top_k=search_config.candidate_k,
                embedding_revision=embedding_revision,
                vector_size=embedding_config.dimension,
            )
        with observer.time_stage("candidate_fusion"):
            sorted_rows = _stable_sort(_require_valid_similarity(vector_rows))
            if search_config.min_similarity is None:
                thresholded_vector = sorted_rows
            else:
                thresholded_vector = tuple(row for row in sorted_rows if row.similarity >= search_config.min_similarity)

    if search_config.mode in {"bm25", "hybrid"}:
        lexical_config = LexicalIndexConfig(bm25_k1=search_config.bm25_k1, bm25_b=search_config.bm25_b)
        tokenizer = JiebaLexicalTokenizer()
        lexical_fingerprint = build_lexical_configuration_fingerprint(lexical_config)
        with observer.time_stage("retrieval_total"):
            with observer.time_stage("lexical_search"):
                observer.increment("lexical_request_count")
                observer.increment("postgresql_query_count")
                readiness = LexicalIndexRepository(connection).get_lexical_readiness(
                    knowledge_base_id=knowledge_base_id,
                    tokenizer_id=tokenizer.tokenizer_id,
                    tokenizer_version=tokenizer.version,
                    configuration_fingerprint=lexical_fingerprint,
                )
                lexical_ready = readiness.ready
                if not readiness.ready:
                    raise LexicalIndexNotReadyError(
                        "Lexical index is incomplete for this knowledge base. "
                        "Run the lexical indexing step before hybrid search."
                    )
                lexical_terms = tokenizer.tokenize_query(query)
                observer.increment("postgresql_query_count")
                bm25_rows = BM25SearchRepository(connection).search_chunks_by_bm25(
                    query_terms=lexical_terms,
                    knowledge_base_id=knowledge_base_id,
                    match_count=search_config.bm25_candidate_k,
                    k1=search_config.bm25_k1,
                    b=search_config.bm25_b,
                    configuration_fingerprint=lexical_fingerprint,
                )

    with observer.time_stage("retrieval_fusion"):
        if search_config.mode == "vector":
            base_results = tuple(_to_vector_result(index + 1, row, vector_rank=index + 1) for index, row in enumerate(thresholded_vector))
        elif search_config.mode == "bm25":
            base_results = tuple(_to_bm25_result(index + 1, row, bm25_rank=index + 1) for index, row in enumerate(bm25_rows))
        else:
            fused = _rrf_fuse(thresholded_vector, bm25_rows, search_config)
            base_results = tuple(_fused_to_result(rank=index + 1, item=item) for index, item in enumerate(fused))

    normalized_query = prepared.query if prepared is not None else normalize_lexical_text(query).strip()
    runtime_integration = None
    with observer.time_stage("agent_control"):
        with observer.time_stage("retriever_selection"):
            observer.set_detail("selected_retrieval_lane", search_config.mode)
        with observer.time_stage("guard_evaluation"):
            runtime_integration = integrate_public_search_runtime(
                connection,
                knowledge_base_id=knowledge_base_id,
                query=query,
                base_results=base_results,
                timing_observer=observer,
            )
            observer.set_detail("guard_triggered", bool(runtime_integration.guard_trace.get("guard_triggered")))
            observer.set_detail("guard_reason", runtime_integration.guard_trace.get("guard_reason"))
            observer.set_detail("structure_lane_invoked", bool(runtime_integration.guard_trace.get("structure_lane_invoked")))
            observer.set_detail("recovery_activated", bool(runtime_integration.guard_trace.get("recovery_activated")))
        with observer.time_stage("recovery_decision"):
            observer.set_detail("recovery_attempt_count", int(runtime_integration.guard_trace.get("recovery_attempt_count", 0)))
            observer.set_detail("maximum_recovery_attempt_count", int(runtime_integration.guard_trace.get("maximum_recovery_attempt_count", 1)))
        selected_results = _select_results(
            runtime_integration.candidates,
            search_config,
            normalized_query=normalized_query if search_config.rerank_enabled else None,
            reranker_provider=reranker_provider,
            count_context_tokens=context_token_counter.count_tokens,
            timing_observer=observer,
            execution_scope=execution_scope,
        )
    query_token_count = prepared.query_token_count if prepared is not None else len(lexical_terms)
    query_input_token_count = prepared.input_token_count if prepared is not None else len(lexical_terms)
    query_input_hash = prepared.text_hash if prepared is not None else ""
    with observer.time_stage("candidate_materialization"):
        threshold_filtered_count = len(_stable_sort(_require_valid_similarity(vector_rows))) - len(thresholded_vector) if vector_rows else 0
        pre_dedup_count = len(thresholded_vector) if search_config.mode == "vector" else len(bm25_rows) if search_config.mode == "bm25" else len(_rrf_fuse(thresholded_vector, bm25_rows, search_config))
    with observer.time_stage("evidence_composition"):
        evidence_bundle = _build_evidence_bundle(
            query=query,
            normalized_query=normalized_query,
            knowledge_base_id=knowledge_base_id,
            results=selected_results,
            context_token_budget=search_config.context_token_budget,
            context_tokenizer_id=context_token_counter.tokenizer_id,
            context_tokenizer_revision=context_token_counter.tokenizer_revision,
        )
    with observer.time_stage("citation_preparation"):
        evidence_signals = _build_evidence_signals(selected_results, evidence_bundle, lexical_terms=lexical_terms)
    observer.set_detail("initial_candidate_count", len(vector_rows) + len(bm25_rows))
    observer.set_detail("expanded_candidate_count", len(runtime_integration.candidates) if runtime_integration is not None else len(base_results))
    observer.set_detail("reranker_input_count", min(pre_dedup_count, search_config.rerank_top_n) if search_config.rerank_enabled else 0)
    observer.set_detail("evidence_input_count", len(selected_results))
    observer.set_detail("final_evidence_count", len(evidence_bundle.items))

    with observer.time_stage("response_serialization"):
        graph_trace = dict(runtime_integration.graph_trace) if runtime_integration is not None else _build_graph_trace(results=selected_results)
        guard_trace = dict(runtime_integration.guard_trace) if runtime_integration is not None else _build_guard_trace(search_config=search_config, observer=observer)
        response = SearchResponse(
            query=query,
            normalized_query=normalized_query,
            knowledge_base_id=knowledge_base_id,
            model_id=provider.model_id if provider is not None else context_token_counter.tokenizer_id,
            retrieval_mode=search_config.mode,
            query_template_version=prepared.template_version if prepared is not None else "",
            query_instruction=prepared.instruction if prepared is not None else "",
            requested_top_k=search_config.top_k,
            candidate_k=search_config.candidate_k,
            candidate_count=len(vector_rows) + len(bm25_rows),
            vector_candidate_count=len(vector_rows),
            bm25_candidate_count=len(bm25_rows),
            threshold_filtered_count=threshold_filtered_count,
            deduplicated_count=max(pre_dedup_count - len(selected_results), 0),
            result_count=len(selected_results),
            query_token_count=query_token_count,
            query_input_token_count=query_input_token_count,
            query_input_hash=query_input_hash,
            lexical_query_terms=lexical_terms,
            lexical_ready=lexical_ready,
            retrieval_degraded=False,
            results=selected_results,
            reranker_enabled=search_config.rerank_enabled,
            reranker_model_id=reranker_provider.model_id if search_config.rerank_enabled and reranker_provider else None,
            reranker_model_revision=reranker_provider.model_revision if search_config.rerank_enabled and reranker_provider else None,
            reranker_input_template_version=reranker_provider.input_template_version
            if search_config.rerank_enabled and reranker_provider
            else None,
            reranker_score_semantics=RERANKER_SCORE_SEMANTICS if search_config.rerank_enabled else None,
            rerank_top_n=search_config.rerank_top_n,
            rerank_candidate_count=min(pre_dedup_count, search_config.rerank_top_n) if search_config.rerank_enabled else 0,
            final_top_k=search_config.top_k,
            context_token_budget=search_config.context_token_budget,
            context_token_count=evidence_bundle.context_token_count,
            evidence_bundle=evidence_bundle,
            evidence_signals=evidence_signals,
            shadow_observation=shadow_observation,
            graph_trace=graph_trace,
            guard_trace=guard_trace,
        )
        if runtime_trace_context is not None and runtime_trace_context.enabled:
            trace = build_runtime_trace_from_live_execution(
                runtime_trace_context,
                search_response=response,
                timing_snapshot=observer.snapshot(retrieval_total_latency_ms=runtime_trace_context.elapsed_ms),
                embedding_config=embedding_config,
                reranker_config=getattr(reranker_provider, "config", None),
                qdrant_config=load_qdrant_config(os.environ) if _load_vector_backend_id(os.environ) == "qdrant" else None,
                vector_backend=_load_vector_backend_id(os.environ),
                reranker_device=getattr(reranker_provider, "device", None),
            )
            response = _replace_search_response(response, runtime_trace=trace)
        return response


def _validate_query_vector(vector: Sequence[float], *, provider: EmbeddingProvider, config: EmbeddingConfig) -> None:
    if provider.dimension != config.dimension:
        raise EmbeddingOutputValidationError(
            f"Provider dimension {provider.dimension} does not match configured dimension {config.dimension}."
        )
    _validate_vectors((vector,), expected_count=1, dimension=config.dimension, normalize=config.normalize)


def _load_vector_backend_id(env: Mapping[str, str]) -> str:
    backend = env.get("OPK_RAG_VECTOR_BACKEND", "qdrant").strip() or "qdrant"
    if backend not in {"postgres_pgvector", "qdrant"}:
        raise SearchError(f"Unsupported OPK_RAG_VECTOR_BACKEND: {backend}")
    return backend


def _search_vector_candidates(
    connection,
    *,
    knowledge_base_id: UUID,
    query_embedding: Sequence[float],
    match_count: int,
    index_configuration_id: UUID | None,
    embedding_revision: str,
    vector_size: int,
    env: Mapping[str, str],
    timing_observer: RetrievalTimingObserver | None = None,
) -> tuple[ChunkSearchRow, ...]:
    observer = timing_observer or RetrievalTimingObserver(enabled=False)
    backend = _load_vector_backend_id(env)
    repository = ChunkSearchRepository(connection)
    if backend == "postgres_pgvector":
        observer.increment("vector_search_call_count")
        observer.increment("postgresql_query_count")
        return repository.search_chunks_by_vector(
            query_embedding=query_embedding,
            knowledge_base_id=knowledge_base_id,
            match_count=match_count,
            index_configuration_id=index_configuration_id,
        )
    with observer.time_stage("qdrant_request_build"):
        qdrant = QdrantVectorBackend(load_qdrant_config(env))
        observer.increment("qdrant_client_init_count")
        observer.set_detail("qdrant_server_timing_available", False)
    try:
        observer.increment("vector_search_call_count")
        with observer.time_stage("qdrant_network_round_trip"):
            qdrant_candidates = qdrant.search(
                query_embedding,
                top_k=match_count,
                search_filter=VectorBackendSearchFilter(
                    knowledge_base_id=str(knowledge_base_id),
                    embedding_revision=embedding_revision,
                ),
            )
    except Exception as exc:
        raise SearchError("Qdrant authoritative vector backend failed; explicit rollback or fail-closed is required.") from exc
    finally:
        qdrant.close()
    observer.increment("postgresql_query_count")
    with observer.time_stage("qdrant_response_parse"):
        return _qdrant_candidates_to_chunk_rows(repository, qdrant_candidates, knowledge_base_id=knowledge_base_id)


def _qdrant_candidates_to_chunk_rows(
    repository: ChunkSearchRepository,
    candidates: Sequence[VectorBackendCandidate],
    *,
    knowledge_base_id: UUID,
) -> tuple[ChunkSearchRow, ...]:
    chunk_ids = []
    similarity_by_chunk_id: dict[UUID, float] = {}
    for candidate in candidates:
        try:
            chunk_id = UUID(str(candidate.chunk_id))
        except ValueError as exc:
            raise SearchError(f"Qdrant returned invalid chunk_id: {candidate.chunk_id}") from exc
        chunk_ids.append(chunk_id)
        similarity_by_chunk_id[chunk_id] = float(candidate.canonical_similarity_score)
    rows = repository.get_chunks_by_ids_for_search(
        chunk_ids=tuple(chunk_ids),
        knowledge_base_id=knowledge_base_id,
        similarity_by_chunk_id=similarity_by_chunk_id,
    )
    if len(rows) != len(tuple(dict.fromkeys(chunk_ids))):
        raise SearchError("Qdrant returned candidates that are missing from PostgreSQL relational authority.")
    return rows


def _build_graph_trace(*, results: Sequence[SearchResult]) -> Mapping[str, Any]:
    return {
        "schema_version": "opk-rag.search.graph-trace.v1",
        "graph_execution_hook_available": False,
        "graph_enabled": False,
        "graph_activated": False,
        "hop_depth": 1,
        "seed_chunk_ids": [str(result.chunk_id) for result in results],
        "seed_document_ids": sorted({str(result.document_id) for result in results}),
        "relations": [],
        "expanded_chunk_ids": [],
        "expanded_document_ids": [],
        "expanded_candidate_count": 0,
        "graph_trace_is_observational_only": True,
        "graph_trace_changes_runtime_decision": False,
        "graph_trace_changes_candidate_set": False,
        "runtime_gold_metadata_usage": False,
    }


def _build_guard_trace(*, search_config: VectorSearchConfig, observer: RetrievalTimingObserver) -> Mapping[str, Any]:
    return {
        "schema_version": "opk-rag.search.guard-trace.v1",
        "agent_type": "guarded_agent",
        "selected_retrieval_lane": observer.details.get("selected_retrieval_lane", search_config.mode),
        "guard_evaluated": True,
        "recovery_activated": bool(observer.details.get("recovery_activated", False)),
        "recovery_attempt_count": int(observer.details.get("recovery_attempt_count", 0)),
        "maximum_recovery_attempt_count": int(observer.details.get("maximum_recovery_attempt_count", 0)),
        "final_decision": "search_completed",
        "guard_trace_observational_only": True,
        "guard_decision_policy_unchanged": True,
        "recovery_budget_unchanged": True,
    }


def _provider_context_counter(provider: EmbeddingProvider | None, embedding_config: EmbeddingConfig) -> ContextTokenCounter:
    if provider is None:
        return SimpleContextTokenCounter()

    class _ProviderContextTokenCounter:
        tokenizer_id = provider.model_id
        tokenizer_revision = embedding_config.model_revision

        def count_tokens(self, text: str) -> int:
            return provider.count_tokens(text)

    return _ProviderContextTokenCounter()


def _require_valid_similarity(rows: Sequence[ChunkSearchRow]) -> tuple[ChunkSearchRow, ...]:
    for row in rows:
        if not math.isfinite(float(row.similarity)):
            raise SearchError(f"Database returned non-finite similarity for chunk {row.chunk_id}.")
    return tuple(rows)


def _stable_sort(rows: Sequence[ChunkSearchRow]) -> tuple[ChunkSearchRow, ...]:
    return tuple(sorted(rows, key=lambda row: (-row.similarity, row.relative_path, row.start_line or 0, str(row.chunk_id))))


def _deduplicate(rows: Sequence[ChunkSearchRow], config: VectorSearchConfig) -> tuple[ChunkSearchRow, ...]:
    kept: list[ChunkSearchRow] = []
    seen_chunk_ids: set[UUID] = set()
    seen_ranges: set[tuple[UUID, int | None, int | None]] = set()
    seen_content: set[tuple[UUID, str]] = set()
    for row in rows:
        if row.chunk_id in seen_chunk_ids:
            continue
        range_key = (row.document_id, row.start_line, row.end_line)
        if range_key in seen_ranges:
            continue
        content_key = (row.document_id, _content_hash(row.content))
        if content_key in seen_content:
            continue
        if any(_is_overlapping_duplicate(row, kept_row, config.overlap_deduplication_threshold) for kept_row in kept):
            continue
        kept.append(row)
        seen_chunk_ids.add(row.chunk_id)
        seen_ranges.add(range_key)
        seen_content.add(content_key)
    return tuple(kept)


def _is_overlapping_duplicate(left: ChunkSearchRow, right: ChunkSearchRow, threshold: float) -> bool:
    if left.document_id != right.document_id:
        return False
    if left.start_line is None or left.end_line is None or right.start_line is None or right.end_line is None:
        return False
    overlap_start = max(left.start_line, right.start_line)
    overlap_end = min(left.end_line, right.end_line)
    if overlap_end < overlap_start:
        return False
    overlap = overlap_end - overlap_start + 1
    smaller = min(left.end_line - left.start_line + 1, right.end_line - right.start_line + 1)
    return smaller > 0 and (overlap / smaller) >= threshold


def _content_hash(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _select_results(
    results: Sequence[SearchResult],
    config: VectorSearchConfig,
    *,
    normalized_query: str | None,
    reranker_provider: RerankerProvider | None,
    count_context_tokens,
    timing_observer: RetrievalTimingObserver | None = None,
    execution_scope: RerankerExecutionScope = "search",
) -> tuple[SearchResult, ...]:
    observer = timing_observer or RetrievalTimingObserver(enabled=False)
    ranked = tuple(_with_original_rank(result) for result in results)
    if config.rerank_enabled and normalized_query is not None and reranker_provider is not None:
        try:
            with observer.time_stage("reranking_total"):
                ranked = _rerank_results(
                    normalized_query,
                    ranked[: config.rerank_top_n],
                    reranker_provider,
                    config,
                    timing_observer=observer,
                    execution_scope=execution_scope,
                )
        except (RerankerInferenceError, RerankerModelLoadError) as exc:
            raise SearchError("Reranker failed; fail-closed prevents returning unverified ranking results.") from exc
    with observer.time_stage("candidate_deduplication"):
        deduplicated = _deduplicate_results(ranked, config) if config.deduplicate else ranked
    truncated = tuple(_replace_result_rank(result, index + 1) for index, result in enumerate(deduplicated[: config.top_k]))
    with observer.time_stage("evidence_composition"):
        return _mark_context_selection(truncated, config, count_context_tokens=count_context_tokens)


def _with_original_rank(result: SearchResult) -> SearchResult:
    if result.original_rank is not None:
        return result
    return _replace_search_result(result, original_rank=result.rank)


def _rerank_results(
    normalized_query: str,
    results: Sequence[SearchResult],
    reranker_provider: RerankerProvider,
    config: VectorSearchConfig,
    timing_observer: RetrievalTimingObserver | None = None,
    execution_scope: RerankerExecutionScope = "search",
) -> tuple[SearchResult, ...]:
    observer = timing_observer or RetrievalTimingObserver(enabled=False)
    documents = tuple(render_reranker_document(result.heading_path, result.content) for result in results)
    with observer.time_stage("reranking_inference"):
        observer.increment("reranker_call_count")
        batch_size = int(getattr(getattr(reranker_provider, "config", None), "batch_size", len(documents) or 1) or 1)
        observer.increment("reranker_batch_count", max(1, (len(documents) + batch_size - 1) // batch_size) if documents else 0)
        scores = tuple(_score_reranker_pairs(reranker_provider, normalized_query, documents, execution_scope=execution_scope))
    if len(scores) != len(results):
        raise SearchError(f"Expected {len(results)} reranker scores, got {len(scores)}.")
    scored = []
    for result, document, score in zip(results, documents, scores, strict=True):
        with observer.time_stage("reranking_tokenization"):
            pair_metadata = reranker_provider.prepare_pair_metadata(normalized_query, document)
        scored.append(
            _replace_search_result(
                result,
                rerank_score=float(score),
                pair_token_count=pair_metadata.pair_token_count,
                reranker_pair_token_count=pair_metadata.pair_token_count,
                reranker_original_pair_token_count=pair_metadata.original_pair_token_count,
                reranker_input_truncated=pair_metadata.input_truncated,
            )
        )
    reranked = sorted(
        scored,
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else float("-inf")),
            item.original_rank or item.rank,
            item.relative_path,
            item.start_line or 0,
            str(item.chunk_id),
        ),
    )
    reranker_rank_by_identity = {str(result.chunk_id): index for index, result in enumerate(reranked, start=1)}
    fused = rank_fusion_order(
        tuple(scored),
        identity=lambda item: str(item.chunk_id),
        retrieval_rank=lambda item: item.original_rank or item.rank,
        reranker_rank_by_identity=reranker_rank_by_identity,
        parameters=RankFusionParameters(k=config.rank_fusion_k, lambda_weight=config.rank_fusion_lambda),
    )
    return tuple(
        _replace_search_result(scored_item.item, rerank_rank=scored_item.reranker_rank)
        for scored_item in fused
    )


def _score_reranker_pairs(
    reranker_provider: RerankerProvider,
    normalized_query: str,
    documents: Sequence[str],
    *,
    execution_scope: RerankerExecutionScope,
) -> Sequence[float]:
    try:
        return reranker_provider.score_pairs(normalized_query, documents, execution_scope=execution_scope)
    except TypeError as exc:
        if "execution_scope" not in str(exc):
            raise
        return reranker_provider.score_pairs(normalized_query, documents)


def _timed_embedding_provider(provider: EmbeddingProvider, observer: RetrievalTimingObserver):
    class _TimedEmbeddingProvider:
        def count_tokens(self, text: str) -> int:
            with observer.time_stage("query_tokenization"):
                return provider.count_tokens(text)

    return _TimedEmbeddingProvider()


def _deduplicate_results(results: Sequence[SearchResult], config: VectorSearchConfig) -> tuple[SearchResult, ...]:
    kept: list[SearchResult] = []
    seen_chunk_ids: set[UUID] = set()
    seen_ranges: set[tuple[UUID, int | None, int | None]] = set()
    seen_content: set[tuple[UUID, str]] = set()
    for result in results:
        if result.chunk_id in seen_chunk_ids:
            continue
        range_key = (result.document_id, result.start_line, result.end_line)
        if range_key in seen_ranges:
            continue
        content_key = (result.document_id, _content_hash(result.content))
        if content_key in seen_content:
            continue
        if any(_is_overlapping_duplicate(_result_as_chunk_row(result), _result_as_chunk_row(kept_result), config.overlap_deduplication_threshold) for kept_result in kept):
            continue
        kept.append(result)
        seen_chunk_ids.add(result.chunk_id)
        seen_ranges.add(range_key)
        seen_content.add(content_key)
    return tuple(kept)


def _mark_context_selection(results: Sequence[SearchResult], config: VectorSearchConfig, *, count_context_tokens) -> tuple[SearchResult, ...]:
    selected: list[SearchResult] = []
    used = 0
    context_rank = 1
    selected_by_document: dict[UUID, int] = {}
    for result in results:
        token_count = _count_context_tokens(result, count_context_tokens)
        document_selected = selected_by_document.get(result.document_id, 0)
        can_select = (
            context_rank <= config.context_max_chunks
            and document_selected < config.context_max_chunks_per_document
            and used + token_count <= config.context_token_budget
        )
        if can_select:
            selected.append(
                _replace_search_result(
                    result,
                    context_token_count=token_count,
                    selected_for_context=True,
                    context_rank=context_rank,
                )
            )
            used += token_count
            selected_by_document[result.document_id] = document_selected + 1
            context_rank += 1
        else:
            selected.append(
                _replace_search_result(
                    result,
                    context_token_count=token_count,
                    selected_for_context=False,
                    context_rank=None,
                )
            )
    return tuple(selected)


def _count_context_tokens(result: SearchResult, count_context_tokens) -> int:
    text = render_reranker_document(result.heading_path, result.content)
    return max(1, int(count_context_tokens(text)))


def _replace_result_rank(result: SearchResult, rank: int) -> SearchResult:
    return _replace_search_result(result, rank=rank)


def _replace_search_result(result: SearchResult, **changes) -> SearchResult:
    from dataclasses import replace

    return replace(result, **changes)


def _replace_search_response(response: SearchResponse, **changes) -> SearchResponse:
    from dataclasses import replace

    return replace(response, **changes)


def _result_as_chunk_row(result: SearchResult) -> ChunkSearchRow:
    return ChunkSearchRow(
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        relative_path=result.relative_path,
        heading_path=result.heading_path,
        content=result.content,
        start_line=result.start_line,
        end_line=result.end_line,
        similarity=result.similarity,
    )


def _build_evidence_bundle(
    *,
    query: str,
    normalized_query: str,
    knowledge_base_id: UUID,
    results: Sequence[SearchResult],
    context_token_budget: int,
    context_tokenizer_id: str,
    context_tokenizer_revision: str | None,
) -> EvidenceBundle:
    items = tuple(
        EvidenceItem(
            context_rank=result.context_rank or 0,
            result_rank=result.rank,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            relative_path=result.relative_path,
            heading_path=result.heading_path,
            content=result.content,
            start_line=result.start_line,
            end_line=result.end_line,
            context_token_count=result.context_token_count or 0,
            reranker_pair_token_count=result.reranker_pair_token_count,
            reranker_original_pair_token_count=result.reranker_original_pair_token_count,
            reranker_input_truncated=result.reranker_input_truncated,
            rerank_score=result.rerank_score,
            retrieval_sources=result.retrieval_sources,
        )
        for result in results
        if result.selected_for_context
    )
    return EvidenceBundle(
        query=query,
        normalized_query=normalized_query,
        knowledge_base_id=knowledge_base_id,
        context_token_budget=context_token_budget,
        context_token_count=sum(item.context_token_count for item in items),
        total_token_count=sum(item.context_token_count for item in items),
        context_tokenizer_id=context_tokenizer_id,
        context_tokenizer_revision=context_tokenizer_revision,
        items=items,
    )


def _build_evidence_signals(
    results: Sequence[SearchResult],
    evidence_bundle: EvidenceBundle,
    *,
    lexical_terms: Sequence[str],
) -> EvidenceSignals:
    rerank_scores = [result.rerank_score for result in results if result.rerank_score is not None]
    sorted_scores = sorted(rerank_scores, reverse=True)
    top_score = sorted_scores[0] if sorted_scores else None
    second_score = sorted_scores[1] if len(sorted_scores) >= 2 else None
    margin = (top_score - second_score) if top_score is not None and second_score is not None else None
    selected_documents = {item.document_id for item in evidence_bundle.items}
    selected_text = "\n".join(f"{' '.join(item.heading_path)}\n{item.content}".lower() for item in evidence_bundle.items)
    covered_terms = {term for term in lexical_terms if term and term.lower() in selected_text}
    identifier_terms = tuple(term for term in lexical_terms if _looks_like_identifier(term))
    exact_identifier_match = any(term.lower() in selected_text for term in identifier_terms)
    return EvidenceSignals(
        top_reranker_score=top_score,
        second_reranker_score=second_score,
        top1_top2_margin=margin,
        max_vector_similarity=_max_or_none(result.vector_similarity for result in results),
        max_bm25_score=_max_or_none(result.bm25_score for result in results),
        max_rrf_score=_max_or_none(result.rrf_score for result in results),
        dual_channel_candidate_count=sum(1 for result in results if "vector" in result.retrieval_sources and "bm25" in result.retrieval_sources),
        selected_source_document_count=len(selected_documents),
        selected_chunk_count=len(evidence_bundle.items),
        selected_context_token_count=evidence_bundle.context_token_count,
        query_term_coverage=(len(covered_terms) / len(set(lexical_terms))) if lexical_terms else 0.0,
        exact_identifier_match=exact_identifier_match,
        any_reranker_input_truncated=any(result.reranker_input_truncated for result in results),
        reranker_score_min=min(rerank_scores) if rerank_scores else None,
        reranker_score_max=max(rerank_scores) if rerank_scores else None,
        reranker_score_mean=(sum(rerank_scores) / len(rerank_scores)) if rerank_scores else None,
    )


def _max_or_none(values) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _looks_like_identifier(term: str) -> bool:
    return any(marker in term for marker in ("_", "-", ".", "::")) or any(char.isdigit() for char in term)


def _to_vector_result(rank: int, row: ChunkSearchRow, *, vector_rank: int | None = None) -> SearchResult:
    return SearchResult(
        rank=rank,
        document_id=row.document_id,
        chunk_id=row.chunk_id,
        relative_path=row.relative_path,
        heading_path=row.heading_path,
        content=row.content,
        start_line=row.start_line,
        end_line=row.end_line,
        similarity=row.similarity,
        vector_similarity=row.similarity,
        vector_rank=vector_rank,
        retrieval_sources=("vector",),
    )


def _to_bm25_result(rank: int, row: BM25SearchRow, *, bm25_rank: int | None = None) -> SearchResult:
    return SearchResult(
        rank=rank,
        document_id=row.document_id,
        chunk_id=row.chunk_id,
        relative_path=row.relative_path,
        heading_path=row.heading_path,
        content=row.content,
        start_line=row.start_line,
        end_line=row.end_line,
        similarity=0.0,
        vector_similarity=None,
        bm25_score=row.bm25_score,
        bm25_rank=bm25_rank,
        retrieval_sources=("bm25",),
        matched_terms=row.matched_terms,
    )


def _deduplicate_bm25(rows: Sequence[BM25SearchRow], config: VectorSearchConfig) -> tuple[BM25SearchRow, ...]:
    kept: list[BM25SearchRow] = []
    seen_chunk_ids: set[UUID] = set()
    seen_ranges: set[tuple[UUID, int | None, int | None]] = set()
    seen_content: set[tuple[UUID, str]] = set()
    for row in rows:
        if row.chunk_id in seen_chunk_ids:
            continue
        range_key = (row.document_id, row.start_line, row.end_line)
        if range_key in seen_ranges:
            continue
        content_key = (row.document_id, _content_hash(row.content))
        if content_key in seen_content:
            continue
        row_as_chunk = _bm25_as_chunk_row(row)
        if any(_is_overlapping_duplicate(row_as_chunk, _bm25_as_chunk_row(kept_row), config.overlap_deduplication_threshold) for kept_row in kept):
            continue
        kept.append(row)
        seen_chunk_ids.add(row.chunk_id)
        seen_ranges.add(range_key)
        seen_content.add(content_key)
    return tuple(kept)


def _bm25_as_chunk_row(row: BM25SearchRow) -> ChunkSearchRow:
    return ChunkSearchRow(
        document_id=row.document_id,
        chunk_id=row.chunk_id,
        relative_path=row.relative_path,
        heading_path=row.heading_path,
        content=row.content,
        start_line=row.start_line,
        end_line=row.end_line,
        similarity=row.bm25_score,
    )


@dataclass(frozen=True)
class _FusedCandidate:
    row: ChunkSearchRow | BM25SearchRow
    vector_rank: int | None
    bm25_rank: int | None
    similarity: float | None
    bm25_score: float | None
    matched_terms: tuple[str, ...]
    rrf_score: float


def _rrf_fuse(
    vector_rows: Sequence[ChunkSearchRow],
    bm25_rows: Sequence[BM25SearchRow],
    config: VectorSearchConfig,
) -> tuple[_FusedCandidate, ...]:
    by_chunk: dict[UUID, _FusedCandidate] = {}
    for rank, row in enumerate(vector_rows, start=1):
        by_chunk[row.chunk_id] = _FusedCandidate(
            row=row,
            vector_rank=rank,
            bm25_rank=None,
            similarity=row.similarity,
            bm25_score=None,
            matched_terms=(),
            rrf_score=config.rrf_vector_weight / (config.rrf_k + rank),
        )
    for rank, row in enumerate(bm25_rows, start=1):
        existing = by_chunk.get(row.chunk_id)
        contribution = config.rrf_bm25_weight / (config.rrf_k + rank)
        if existing is None:
            by_chunk[row.chunk_id] = _FusedCandidate(
                row=row,
                vector_rank=None,
                bm25_rank=rank,
                similarity=None,
                bm25_score=row.bm25_score,
                matched_terms=row.matched_terms,
                rrf_score=contribution,
            )
        else:
            by_chunk[row.chunk_id] = _FusedCandidate(
                row=existing.row,
                vector_rank=existing.vector_rank,
                bm25_rank=rank,
                similarity=existing.similarity,
                bm25_score=row.bm25_score,
                matched_terms=row.matched_terms,
                rrf_score=existing.rrf_score + contribution,
            )
    return tuple(
        sorted(
            by_chunk.values(),
            key=lambda item: (-item.rrf_score, item.row.relative_path, item.row.start_line or 0, str(item.row.chunk_id)),
        )
    )


def _deduplicate_fused(rows: Sequence[_FusedCandidate], config: VectorSearchConfig) -> tuple[_FusedCandidate, ...]:
    kept: list[_FusedCandidate] = []
    for item in rows:
        if any(_is_overlapping_duplicate(_as_chunk_row(item), _as_chunk_row(kept_item), config.overlap_deduplication_threshold) for kept_item in kept):
            continue
        if item.row.chunk_id in {kept_item.row.chunk_id for kept_item in kept}:
            continue
        kept.append(item)
    return tuple(kept)


def _as_chunk_row(item: _FusedCandidate) -> ChunkSearchRow:
    if isinstance(item.row, ChunkSearchRow):
        return item.row
    return _bm25_as_chunk_row(item.row)


def _fused_to_result(*, rank: int, item: _FusedCandidate) -> SearchResult:
    row = item.row
    return SearchResult(
        rank=rank,
        document_id=row.document_id,
        chunk_id=row.chunk_id,
        relative_path=row.relative_path,
        heading_path=row.heading_path,
        content=row.content,
        start_line=row.start_line,
        end_line=row.end_line,
        similarity=item.similarity or 0.0,
        vector_similarity=item.similarity,
        bm25_score=item.bm25_score,
        vector_rank=item.vector_rank,
        bm25_rank=item.bm25_rank,
        rrf_score=item.rrf_score,
        retrieval_sources=tuple(source for source, present in (("vector", item.vector_rank is not None), ("bm25", item.bm25_rank is not None)) if present),
        matched_terms=item.matched_terms,
    )
