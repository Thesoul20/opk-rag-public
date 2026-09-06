from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from opk_rag.chunking.models import CHUNKING_VERSION, PARSER_VERSION
from opk_rag.embedding.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    EmbeddingConfig,
    build_configuration_fingerprint,
)
from opk_rag.embedding.input import normalize_embedding_text, render_embedding_input
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INSTRUCTION, QUERY_INPUT_TEMPLATE_VERSION, normalize_query, render_query_input
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.evaluation.candidate_retrieval_baseline import (
    BASELINE_ID,
    CONTRACT_PATH as CANDIDATE_CONTRACT_PATH,
    RESULT_PATH as CANDIDATE_RESULT_PATH,
    ROOT,
    _capability_label,
    _load_public_samples,
    _reject_forbidden_path,
    _reject_forbidden_payload,
    _sha256_file,
    git_commit,
    read_json,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.evidence_identity import (
    CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
    EvidenceIdentity,
    deduplicate_evidence_identities,
    digest_json,
    match_evidence_identity,
    normalize_gold_evidence_identity,
    normalize_runtime_evidence_identity,
)
from opk_rag.evaluation.scope_retrieval_experiment import (
    EXPERIMENT_ID as SCOPE_EXPERIMENT_ID,
    CONTRACT_PATH as SCOPE_CONTRACT_PATH,
    RESULT_PATH as SCOPE_RESULT_PATH,
)
from opk_rag.evaluation.scope_structure_audit import (
    STRUCTURE_AUDIT_PATH,
    build_scope_structure_summary,
    gold_identity_map_digests,
    load_indexed_scope_chunks,
    load_searchable_scope_chunks_from_database,
    write_scope_structure_audit,
    IndexedScopeChunk,
)
from opk_rag.evaluation.qwen_embedding_execution import (
    EXECUTION_PLAN_PATH,
    EXECUTION_RESULT_PATH,
    build_cache_key_row,
    build_execution_plan,
    materialize_embedding_variant,
    profile_qwen_execution,
    select_safe_batch_plan,
    vector_digest,
    write_execution_result,
)
from opk_rag.vault import normalize_vault_root


EXPERIMENT_ID = "phase2-chunk-representation-experiment-v1"
SCHEMA_VERSION = "opk-rag.chunk-representation-experiment-summary.v1"
CONTRACT_SCHEMA_VERSION = "opk-rag.chunk-representation-experiment-contract.v1"
CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_chunk_representation_experiment_contract_v1.json"
RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_chunk_representation_experiment_v1.json"
TRACE_PATH = ROOT / "evaluation-data" / "results" / "phase2_chunk_representation_experiment_v1_traces.jsonl"
REPORT_PATH = ROOT / "docs" / "PHASE2_CHUNK_REPRESENTATION_EXPERIMENT.md"
PRIVATE_CACHE_PATH = ROOT / ".private" / "evaluation" / "task0057"
K_VALUES = (1, 3, 5, 10, 20)
REPRESENTATION_VARIANTS = ("current_chunk_content", "heading_enriched_chunk", "document_heading_chunk", "structured_scope", "adjacent_context")
QUERY_VARIANTS = ("current_query", "instructed_query")
EXPERIMENT_MATRIX = (
    ("current_chunk_content", "current_query"),
    ("current_chunk_content", "instructed_query"),
    ("heading_enriched_chunk", "current_query"),
    ("heading_enriched_chunk", "instructed_query"),
    ("document_heading_chunk", "current_query"),
    ("structured_scope", "current_query"),
    ("adjacent_context", "current_query"),
)
BASELINE_VECTOR_SCOPE_RECALL_AT_5 = 0.1310
PROMOTION_SCOPE_RECALL_AT_5 = 0.1810
INSTRUCTED_QUERY_TEMPLATE = "Instruct:\nRetrieve the knowledge-base passage that directly supports the question.\n\nQuery:\n{query}"


class ChunkRepresentationExperimentError(ValueError):
    pass


class ChunkRepresentationInfrastructureBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowCandidate:
    rank: int
    chunk: IndexedScopeChunk
    score: float


@dataclass(frozen=True)
class ShadowEmbeddingRun:
    chunk_vectors: dict[str, list[float]]
    query_vectors: dict[str, list[float]]
    embedding_input_token_estimates: list[int]
    embedding_call_count: int
    embedding_wall_time: float
    index_build_time: float
    shadow_index_size_bytes: int
    raw_vector_payload_size_bytes: int
    shadow_index_serialized_size_bytes: int
    model_forward_batch_count: int
    query_embedding_input_count: int
    peak_gpu_allocated_bytes: int | None
    peak_gpu_reserved_bytes: int | None
    peak_host_memory_bytes: int | None
    manifest: dict[str, Any]
    cardinality: dict[str, Any]


@dataclass(frozen=True)
class QwenExecutionContext:
    provider: QwenLocalEmbeddingProvider
    model_execution: dict[str, Any]
    contract_digest: str | None = None
    execution_plan: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_representation_contract(embedding_config: EmbeddingConfig) -> dict[str, Any]:
    _require_predecessor_artifacts()
    chunking_contract = build_chunking_contract_audit()
    embedding_contract = build_embedding_contract_audit(embedding_config)
    query_contract = build_query_contract_audit()
    variants = [_representation_variant_contract(variant) for variant in REPRESENTATION_VARIANTS]
    query_variants = [_query_variant_contract(variant) for variant in QUERY_VARIANTS]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": BASELINE_ID,
        "scope_experiment_id": SCOPE_EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "embedding_contract": {
            "provider": embedding_config.provider,
            "model": DEFAULT_EMBEDDING_MODEL_NAME,
            "model_revision": embedding_config.model_revision or DEFAULT_EMBEDDING_MODEL_REVISION,
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
            "similarity": "cosine",
            "l2_normalization": True,
            "local_files_only": True,
            "fingerprint": build_configuration_fingerprint(embedding_config),
        },
        "datasets": ["development", "known-regression"],
        "representation_variants": variants,
        "query_variants": query_variants,
        "experiment_matrix": [{"representation_variant_id": r, "query_variant_id": q} for r, q in EXPERIMENT_MATRIX],
        "k_values": list(K_VALUES),
        "production_k": 5,
        "promotion_gates": {
            "scope_recall_at_5_absolute_improvement_min": 0.05,
            "scope_recall_at_5_min": PROMOTION_SCOPE_RECALL_AT_5,
            "scope_recall_at_20_not_below_baseline": True,
            "document_recall_at_20_max_drop": 0.02,
            "development_scope_recall_at_5_not_below_baseline": True,
            "known_regression_scope_recall_at_5_not_below_baseline": True,
            "candidate_identity_agreement": 1.0,
            "candidate_rank_agreement": 1.0,
            "metric_agreement": 1.0,
            "embedding_input_p95_max_baseline_multiplier": 3.0,
            "query_latency_p95_max_baseline_multiplier": 2.0,
            "shadow_index_size_max_baseline_multiplier": 3.0,
        },
        "selection_policy": {
            "primary_metric": "combined_public_scope_recall_at_5",
            "tie_breakers": [
                "scope_recall_at_5",
                "development_known_regression_balance",
                "scope_recall_at_20",
                "embedding_input_p95",
                "query_latency_p95",
                "shadow_index_size_bytes",
                "implementation_simplicity",
            ],
            "if_no_gate_passes": "no_candidate",
        },
        "bindings": {
            "corpus_snapshot_digest": _sha256_file(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"),
            "gold_identity_map_digests": gold_identity_map_digests(),
            "baseline_digest": _sha256_file(CANDIDATE_RESULT_PATH),
            "baseline_contract_digest": _sha256_file(CANDIDATE_CONTRACT_PATH),
            "scope_experiment_digest": _sha256_file(SCOPE_RESULT_PATH),
            "scope_experiment_contract_digest": _sha256_file(SCOPE_CONTRACT_PATH),
            "chunking_contract_digest": digest_json(chunking_contract),
            "embedding_contract_digest": digest_json(embedding_contract),
            "query_contract_digest": digest_json(query_contract),
            "variant_digests": {row["id"]: digest_json(row) for row in variants},
            "query_variant_digests": {row["id"]: digest_json(row) for row in query_variants},
            "canonical_evidence_identity_schema": CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
        },
        "candidate_universe_contract": {
            "candidate_universe_source": "public indexed corpus: documents.index_status=indexed joined to chunks with non-null embeddings for the frozen embedding configuration",
            "expected_searchable_chunk_count": chunking_contract["expected_searchable_chunk_count"],
            "expected_active_document_count": chunking_contract["expected_active_document_count"],
            "gold_filtered_candidate_universe": False,
            "forbidden_candidate_filters": [
                "sample_id",
                "required_evidence",
                "gold_scope",
                "gold_document",
                "gold_identity_map",
                "question_type",
                "capability_label",
                "expected_action",
            ],
        },
        "chunking_contract_audit": chunking_contract,
        "embedding_input_contract_audit": embedding_contract,
        "query_embedding_contract_audit": query_contract,
        "production_default_change": False,
        "contains_sealed_holdout_data": False,
        "private_cache_path": ".private/evaluation/task0056/",
        "privacy_policy": {
            "publishes_questions": False,
            "publishes_candidate_content": False,
            "publishes_heading_or_path_text": False,
            "publishes_embedding_vectors": False,
            "public_output": "digests_and_aggregate_counts_only",
        },
    }
    _reject_forbidden_payload(contract)
    return contract


def build_chunking_contract_audit() -> dict[str, Any]:
    corpus = read_json(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json")
    return {
        "parser_version": PARSER_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "chunking_algorithm": "deterministic Markdown sections packed by character target/max size",
        "heading_handling": "parser records heading_path; chunk content includes Markdown heading lines when present",
        "paragraph_grouping": "sections are accumulated until target_size or max_size boundary",
        "list_handling": "list blocks are split at Markdown list boundaries only for oversized sections",
        "table_handling": "tables remain plain Markdown content; no table-specific chunk model",
        "code_block_handling": "fenced code blocks are preserved as blocks during oversized splitting",
        "quote_handling": "quotes remain plain Markdown content; no quote-specific chunk model",
        "line_range_handling": "start_line/end_line retained when deterministic after splitting",
        "maximum_size": 1800,
        "minimum_size": None,
        "overlap_policy": "overlap=0",
        "parent_heading_retention": "heading_path stored with chunk",
        "relative_path_retention": "document repository stores relative_path; chunk model does not",
        "document_title_retention": "parser derives title, not persisted in chunk model",
        "content_digest_input": "normalized chunk.content only",
        "embedding_text_input": "heading_path joined with blank line then chunk.content",
        "expected_active_document_count": corpus.get("indexed_active_document_count"),
        "expected_active_chunk_count": corpus.get("indexed_chunk_count"),
        "expected_searchable_chunk_count": corpus.get("indexed_chunk_count"),
        "chunks_excluded_by_contract": 0,
        "audit_table": [
            {
                "field": "chunk_content",
                "chunk_model_contains": True,
                "database_contains": True,
                "embedding_input_contains": True,
                "retrieval_result_contains": True,
                "canonical_identity_contains": "chunk_content_digest",
                "lost_before_embedding": False,
                "lost_before_retrieval": False,
            },
            {
                "field": "heading_path",
                "chunk_model_contains": True,
                "database_contains": True,
                "embedding_input_contains": True,
                "retrieval_result_contains": True,
                "canonical_identity_contains": "heading_path_digest",
                "lost_before_embedding": False,
                "lost_before_retrieval": False,
            },
            {
                "field": "document_title",
                "chunk_model_contains": False,
                "database_contains": False,
                "embedding_input_contains": False,
                "retrieval_result_contains": False,
                "canonical_identity_contains": False,
                "lost_before_embedding": True,
                "lost_before_retrieval": True,
            },
            {
                "field": "relative_path",
                "chunk_model_contains": False,
                "database_contains": True,
                "embedding_input_contains": False,
                "retrieval_result_contains": True,
                "canonical_identity_contains": "relative_path_digest",
                "lost_before_embedding": True,
                "lost_before_retrieval": False,
            },
            {
                "field": "block_type",
                "chunk_model_contains": False,
                "database_contains": False,
                "embedding_input_contains": False,
                "retrieval_result_contains": False,
                "canonical_identity_contains": False,
                "lost_before_embedding": True,
                "lost_before_retrieval": True,
            },
            {
                "field": "neighbor_context",
                "chunk_model_contains": False,
                "database_contains": "derivable by document/chunk_index",
                "embedding_input_contains": False,
                "retrieval_result_contains": False,
                "canonical_identity_contains": False,
                "lost_before_embedding": True,
                "lost_before_retrieval": True,
            },
        ],
    }


def build_query_contract_audit() -> dict[str, Any]:
    return {
        "raw_query": "public sample question",
        "normalized_query": "CRLF/CR normalized to LF and stripped",
        "current_query_prefix": QUERY_INSTRUCTION,
        "instructed_query_prefix": "Retrieve the knowledge-base passage that directly supports the question.",
        "document_prefix": None,
        "tokenization_contract": "production QwenLocalEmbeddingProvider tokenizer; no experiment-specific tokenizer",
        "max_sequence_length": 8192,
        "truncation_policy": "rejects when max_query_tokens/max_input_tokens is exceeded; no silent truncation",
        "pooling_contract": "SentenceTransformer Qwen embedding pooling contract",
        "normalization_contract": "L2 normalize embeddings before cosine search",
        "instruction_prefix": QUERY_INSTRUCTION,
        "task_description": "retrieve relevant passages from a personal Chinese Markdown knowledge base",
        "query_prefix": "Instruct: ...\\nQuery:{normalized_query}",
        "language_normalization": "none beyond newline/strip",
        "truncation": "rejects when max_query_tokens is exceeded; no silent truncation",
        "template_version": QUERY_INPUT_TEMPLATE_VERSION,
        "query_document_prefix_same": False,
        "query_direct_plain_text": False,
    }


def build_embedding_contract_audit(embedding_config: EmbeddingConfig) -> dict[str, Any]:
    return {
        "chunk_content": True,
        "heading_path": True,
        "document_title": False,
        "relative_path": False,
        "block_type": False,
        "neighbor_context": False,
        "document_prefix": None,
        "input_template_version": embedding_config.input_template_version,
        "normalization": "CRLF/CR normalized to LF and stripped",
        "truncation_policy": "raises when max_input_tokens is exceeded; no silent truncation",
        "pooling_contract": "SentenceTransformer Qwen embedding pooling contract",
        "normalization_contract": "L2 normalize embeddings before cosine search",
        "max_input_tokens": embedding_config.max_input_tokens,
        "model_name": embedding_config.model_name,
        "model_revision": embedding_config.model_revision,
        "dimension": embedding_config.dimension,
        "local_files_only": True,
    }


def render_chunk_representation(variant_id: str, chunk: IndexedScopeChunk, chunks: tuple[IndexedScopeChunk, ...] = ()) -> str:
    content = normalize_embedding_text(chunk.content)
    heading = normalize_embedding_text("\n".join(chunk.heading_path))
    title = normalize_embedding_text(chunk.document_title or "")
    if variant_id == "current_chunk_content":
        return _render_production_document_embedding_input(chunk)
    if variant_id == "heading_enriched_chunk":
        return f"[Heading]\n{heading}\n\n[Content]\n{content}" if heading else f"[Content]\n{content}"
    if variant_id == "document_heading_chunk":
        return f"[Document]\n{title}\n\n[Heading]\n{heading}\n\n[Content]\n{content}".strip()
    if variant_id == "structured_scope":
        marker = _structure_marker(chunk)
        return f"[Document]\n{title}\n\n[Heading]\n{heading}\n\n{marker}\n{content}".strip()
    if variant_id == "adjacent_context":
        prev_chunk, next_chunk = _adjacent_chunks(chunk, chunks)
        prev_text = _neighbor_excerpt(prev_chunk.content) if prev_chunk else ""
        next_text = _neighbor_excerpt(next_chunk.content) if next_chunk else ""
        return f"[Previous]\n{prev_text}\n\n[Current]\n{content}\n\n[Next]\n{next_text}".strip()
    raise ChunkRepresentationExperimentError(f"unknown representation variant: {variant_id}")


def render_query_representation(query_variant_id: str, query: str) -> str:
    normalized = normalize_query(query)
    if query_variant_id == "current_query":
        return render_query_input(normalized)
    if query_variant_id == "instructed_query":
        return INSTRUCTED_QUERY_TEMPLATE.format(query=normalized)
    raise ChunkRepresentationExperimentError(f"unknown query variant: {query_variant_id}")


def embed_surrogate_shadow_chunks(chunks: tuple[IndexedScopeChunk, ...], variant_id: str) -> dict[str, list[float]]:
    return {
        digest_json(chunk.identity): _hash_embedding(render_chunk_representation(variant_id, chunk, chunks))
        for chunk in chunks
    }


def embed_surrogate_shadow_queries(samples: list[dict[str, Any]], query_variant_id: str) -> dict[str, list[float]]:
    return {sample["sample_id"]: _hash_embedding(render_query_representation(query_variant_id, sample["question"])) for sample in samples}


def embed_shadow_chunks(chunks: tuple[IndexedScopeChunk, ...], variant_id: str) -> dict[str, list[float]]:
    return embed_surrogate_shadow_chunks(chunks, variant_id)


def build_in_memory_vector_index(chunks: tuple[IndexedScopeChunk, ...], vectors: dict[str, list[float]]) -> list[tuple[IndexedScopeChunk, list[float]]]:
    return [(chunk, vectors[digest_json(chunk.identity)]) for chunk in chunks]


def run_representation_experiment(
    contract: dict[str, Any] | None = None,
    *,
    embedding_config: EmbeddingConfig | None = None,
    provider: QwenLocalEmbeddingProvider | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _validate_contract(contract or {})
    chunks, universe = load_candidate_universe_for_formal_run(embedding_config or _embedding_config_from_contract(contract or {}))
    samples = _load_public_samples(["development", "known-regression"])
    structure = build_scope_structure_summary(chunks)
    try:
        context = _build_qwen_execution_context(embedding_config or _embedding_config_from_contract(contract or {}), provider=provider)
        probe_texts = [render_chunk_representation("adjacent_context", chunk, chunks) for chunk in chunks[: min(16, len(chunks))]]
        batch_plan = select_safe_batch_plan(context.provider, probe_texts)
        execution_plan = build_execution_plan(
            contract=contract or {},
            candidate_universe=universe,
            model_execution=context.model_execution | {"safe_batch_plan": batch_plan},
            selected_batch_size=batch_plan["selected_batch_size"],
            max_batch_token_budget=batch_plan["max_batch_token_budget"],
        )
        context = QwenExecutionContext(
            provider=context.provider,
            model_execution=context.model_execution | {"safe_batch_plan": batch_plan, "execution_plan_digest": digest_json(execution_plan)},
            contract_digest=digest_json(contract or {}),
            execution_plan=execution_plan,
        )
    except ChunkRepresentationInfrastructureBlocked as exc:
        return _infrastructure_blocked_summary(contract or {}, structure, str(exc)), [], structure
    runs = []
    for _ in range(2):
        run_started = time.perf_counter()
        traces = []
        try:
            cached = _embed_matrix_inputs(samples, chunks, context, expected_chunk_count=universe["searchable_chunk_count"])
        except (EmbeddingModelLoadError, EmbeddingInferenceError, ChunkRepresentationInfrastructureBlocked) as exc:
            return _infrastructure_blocked_summary(contract or {}, structure, str(exc)), [], structure
        for representation_id, query_id in EXPERIMENT_MATRIX:
            traces.extend(run_representation_variant(samples, chunks, representation_id, query_id, cached[(representation_id, query_id)]))
        summary = compare_representation_variants(
            traces,
            structure,
            started_at=run_started,
            contract=contract or {},
            model_execution=context.model_execution,
            candidate_universe=universe,
            samples=samples,
            chunks=chunks,
            embedding_cache_key_contract=_shadow_embedding_cache_key_contract(contract or {}, context.model_execution),
        )
        runs.append({"summary": summary, "traces": traces})
    repeatability = compare_repeatability(runs[0], runs[1])
    summary = runs[0]["summary"]
    summary["repeatability"] = repeatability
    manifests = summary.get("embedding_manifest_summary") or {}
    unique_manifests: dict[str, dict[str, Any]] = {}
    for row in manifests.values():
        if not isinstance(row, dict):
            continue
        for key in ("representation_manifest", "query_manifest"):
            manifest = row.get(key)
            if isinstance(manifest, dict):
                unique_manifests[f"{manifest.get('kind')}:{manifest.get('variant_id')}"] = manifest
    summary["qwen_embedding_input_count"] = sum(manifest.get("expected_input_count") or 0 for manifest in unique_manifests.values())
    summary["qwen_model_forward_batch_count"] = sum(manifest.get("batch_count") or 0 for manifest in unique_manifests.values())
    summary["qwen_embedding_call_count_per_formal_run"] = summary["qwen_model_forward_batch_count"]
    summary["qwen_embedding_call_count"] = summary["qwen_model_forward_batch_count"]
    summary["promotion_evaluation"] = evaluate_promotion_gates(summary)
    summary["promotion_decision"] = summary["promotion_evaluation"]["promotion_decision"]
    summary["recommended_variant"] = summary["promotion_evaluation"]["recommended_variant"]
    summary["primary_result_classification"] = _primary_result_classification(summary)
    summary["representation_gap"] = _representation_gap(summary)
    summary["query_embedding_alignment_issue"] = _query_alignment_issue(summary)
    summary["next_task_recommendation"] = _next_task_recommendation(summary)
    return summary, runs[0]["traces"], structure


def run_representation_variant(
    samples: list[dict[str, Any]],
    chunks: tuple[IndexedScopeChunk, ...],
    representation_id: str,
    query_id: str,
    shadow_run: ShadowEmbeddingRun,
) -> list[dict[str, Any]]:
    chunk_vectors = shadow_run.chunk_vectors
    query_vectors = shadow_run.query_vectors
    index_started = time.perf_counter()
    index = build_in_memory_vector_index(chunks, chunk_vectors)
    index_build_time = (time.perf_counter() - index_started) + shadow_run.index_build_time
    traces = []
    for sample in samples:
        started = time.perf_counter()
        ranked = _rank(index, query_vectors[sample["sample_id"]], limit=20)
        latency_ms = (time.perf_counter() - started) * 1000
        required = list(_required(sample))
        unit_profiles = [_unit_profile(gold, ranked) for gold in required]
        traces.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": sample["dataset_id"],
                "question_type": sample.get("question_type") or "unknown",
                "capability": _capability_label(sample),
                "representation_variant_id": representation_id,
                "query_variant_id": query_id,
                "variant_id": f"{representation_id}+{query_id}",
                "required_units_total": len(required),
                "has_required_evidence": bool(required),
                "candidate_identity_digests": [digest_json(candidate.chunk.identity) for candidate in ranked],
                "candidate_document_digests": [candidate.chunk.document_identity_digest for candidate in ranked],
                "unit_profiles": unit_profiles,
                "candidate_count": len(ranked),
                "embedding_input_token_estimates": shadow_run.embedding_input_token_estimates,
                "query_latency_ms": latency_ms,
                "shadow_index_size_bytes": shadow_run.shadow_index_size_bytes,
                "raw_vector_payload_size_bytes": shadow_run.raw_vector_payload_size_bytes,
                "shadow_index_serialized_size_bytes": shadow_run.shadow_index_serialized_size_bytes,
                "vector_dtype": "float32",
                "bytes_per_vector": DEFAULT_EMBEDDING_DIMENSION * 4,
                "shadow_cardinality": shadow_run.cardinality,
                "embedding_call_count": shadow_run.embedding_call_count,
                "embedding_wall_time": shadow_run.embedding_wall_time,
                "index_build_time": index_build_time,
                "model_forward_batch_count": shadow_run.model_forward_batch_count,
                "query_embedding_input_count": shadow_run.query_embedding_input_count,
                "peak_gpu_allocated_bytes": shadow_run.peak_gpu_allocated_bytes,
                "peak_gpu_reserved_bytes": shadow_run.peak_gpu_reserved_bytes,
                "peak_host_memory_bytes": shadow_run.peak_host_memory_bytes,
                "embedding_manifest": shadow_run.manifest,
            }
        )
    return traces


def score_scope_recall(traces: list[dict[str, Any]], k: int) -> float:
    denominator = sum(row["required_units_total"] for row in traces)
    if not denominator:
        return 0.0
    return sum(1 for row in traces for profile in row["unit_profiles"] if profile["scope_rank"] is not None and profile["scope_rank"] <= k) / denominator


def compare_representation_variants(
    traces: list[dict[str, Any]],
    structure: dict[str, Any],
    *,
    started_at: float,
    contract: dict[str, Any],
    model_execution: dict[str, Any],
    candidate_universe: dict[str, Any],
    samples: list[dict[str, Any]],
    chunks: tuple[IndexedScopeChunk, ...],
    embedding_cache_key_contract: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["variant_id"]].append(trace)
    variants = {variant: _aggregate_variant(rows) for variant, rows in sorted(grouped.items())}
    baseline_id = "current_chunk_content+current_query"
    for variant, metrics in variants.items():
        gain = _gain_against_baseline(grouped[baseline_id], grouped[variant])
        metrics.update(gain)
        metrics["development"] = _aggregate_variant([row for row in grouped[variant] if row["dataset_id"] == "development"])
        metrics["known_regression"] = _aggregate_variant([row for row in grouped[variant] if row["dataset_id"] == "known-regression"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "representation_experiment_status": "completed",
        "embedding_execution_mode": "qwen_local_inference",
        "model_execution": model_execution,
        "surrogate_infrastructure_validation": _surrogate_infrastructure_validation_record(),
        "baseline_id": BASELINE_ID,
        "scope_experiment_id": SCOPE_EXPERIMENT_ID,
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "contract_digest": digest_json(contract) if contract else None,
        "bindings": contract.get("bindings") if contract else {},
        "git_commit": contract.get("git_commit") if contract else git_commit(),
        "created_at": utc_now(),
        "datasets": ["development", "known-regression"],
        "k_values": list(K_VALUES),
        "sample_count": len({row["sample_id"] for row in traces}),
        "required_evidence_unit_count": sum(row["required_units_total"] for row in grouped[baseline_id]),
        "scope_structure_audit_status": "completed",
        "scope_structure_audit": {key: value for key, value in structure.items() if key != "records"},
        "candidate_universe_source": candidate_universe["candidate_universe_source"],
        "candidate_universe_chunk_count": candidate_universe["candidate_universe_chunk_count"],
        "candidate_universe_document_count": candidate_universe["candidate_universe_document_count"],
        "candidate_universe_digest": candidate_universe["candidate_universe_digest"],
        "gold_filtered_candidate_universe": False,
        "candidate_universe_audit": candidate_universe,
        "shadow_index_cardinality": _variant_cardinality_from_costs({variant: _cost_metrics(rows) for variant, rows in grouped.items()}),
        "shadow_embedding_cache_key_contract": embedding_cache_key_contract,
        "gold_data_index_construction_audit": _gold_data_index_construction_audit(),
        "representation_render_digest_validation": _representation_render_digest_validation(chunks),
        "query_render_digest_validation": _query_render_digest_validation(samples),
        "shadow_qwen_full_corpus_control": _shadow_control_comparison(grouped["current_chunk_content+current_query"]),
        "variants": variants,
        "dataset_metrics": _dataset_metrics(grouped),
        "question_type_metrics": _dimension_metrics(grouped, "question_type"),
        "capability_metrics": _dimension_metrics(grouped, "capability"),
        "cost_metrics": {variant: _cost_metrics(rows) for variant, rows in grouped.items()},
        "embedding_manifest_summary": {variant: (rows[0].get("embedding_manifest") if rows else {}) for variant, rows in grouped.items()},
        "execution_plan_path": EXECUTION_PLAN_PATH.relative_to(ROOT).as_posix(),
        "execution_result_path": EXECUTION_RESULT_PATH.relative_to(ROOT).as_posix(),
        "gain_attribution": attribute_representation_gain(grouped),
        "chunk_boundary_issue": structure.get("chunk_boundary_issue"),
        "model_calls": {"embedding_only": True, "deepseek": False, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
        "production_default_changed": False,
        "contains_sealed_holdout_data": False,
        "stage_decision": "not_accepted",
        "holdout_status": "exposed",
        "result_validity": "valid_for_quality_decision_candidate_universe_verified",
        "invalid_historical_runs": [_historical_35_vector_invalid_record()],
        "experiment_wall_time": time.perf_counter() - started_at,
        "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "trace_path": TRACE_PATH.relative_to(ROOT).as_posix(),
        "scope_structure_audit_path": STRUCTURE_AUDIT_PATH.relative_to(ROOT).as_posix(),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
    }
    return summary


def attribute_representation_gain(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    mapping = {
        "current_chunk_content+instructed_query": "recovered_by_query_instruction",
        "heading_enriched_chunk+current_query": "recovered_by_heading_context",
        "heading_enriched_chunk+instructed_query": "recovered_by_heading_and_query_instruction",
        "document_heading_chunk+current_query": "recovered_by_document_context",
        "structured_scope+current_query": "recovered_by_structure_marker",
        "adjacent_context+current_query": "recovered_by_adjacent_context",
    }
    out = Counter()
    baseline = grouped["current_chunk_content+current_query"]
    for variant, rows in grouped.items():
        if variant == "current_chunk_content+current_query":
            continue
        gain = _gain_against_baseline(baseline, rows)
        recovered = gain.get("newly_recovered_required_units", 0)
        lost = gain.get("lost_required_units", 0)
        out[mapping.get(variant, "unattributed")] += recovered
        out["lost_due_to_context_dilution"] += lost if variant != "current_chunk_content+instructed_query" else 0
        out["lost_due_to_heading_noise"] += lost if variant.startswith("heading_enriched") else 0
        out["lost_due_to_document_noise"] += lost if variant.startswith("document_heading") else 0
        out["lost_due_to_neighbor_noise"] += lost if variant.startswith("adjacent_context") else 0
    for key in (
        "recovered_by_query_instruction",
        "recovered_by_heading_context",
        "recovered_by_document_context",
        "recovered_by_structure_marker",
        "recovered_by_adjacent_context",
        "recovered_by_heading_and_query_instruction",
        "lost_due_to_context_dilution",
        "lost_due_to_truncation",
        "lost_due_to_heading_noise",
        "lost_due_to_document_noise",
        "lost_due_to_neighbor_noise",
        "unattributed",
    ):
        out.setdefault(key, 0)
    return dict(out)


def evaluate_promotion_gates(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("embedding_execution_mode") != "qwen_local_inference" or summary.get("representation_experiment_status") != "completed":
        return {
            "promotion_decision": "invalid_experiment",
            "recommended_variant": None,
            "variants": {},
            "promotion_gates": {"repeatability": "invalid"},
        }
    if summary.get("gold_filtered_candidate_universe") is not False:
        return {"promotion_decision": "invalid_experiment", "recommended_variant": None, "variants": {}, "promotion_gates": {"candidate_universe": "gold_filtered"}}
    expected_count = summary.get("candidate_universe_audit", {}).get("searchable_chunk_count")
    for variant, cardinality in (summary.get("shadow_index_cardinality") or {}).items():
        if cardinality.get("indexed_vector_count") != expected_count or cardinality.get("unique_canonical_identity_count") != expected_count:
            return {
                "promotion_decision": "invalid_experiment",
                "recommended_variant": None,
                "variants": {variant: {"decision": "invalid", "gate_pass": False, "failed_gates": ["shadow_index_cardinality"]}},
                "promotion_gates": {"shadow_index_cardinality": "invalid"},
            }
    baseline = summary["variants"]["current_chunk_content+current_query"]
    decisions = {}
    candidates = []
    for variant, metrics in summary["variants"].items():
        if variant == "current_chunk_content+current_query":
            decisions[variant] = {"decision": "control", "gate_pass": False, "failed_gates": []}
            continue
        failed = []
        if (metrics["scope_recall_at_k"]["5"] or 0.0) < PROMOTION_SCOPE_RECALL_AT_5:
            failed.append("scope_recall_at_5_improvement")
        if (metrics["scope_recall_at_k"]["20"] or 0.0) < (baseline["scope_recall_at_k"]["20"] or 0.0):
            failed.append("scope_recall_at_20_regression")
        if (baseline["document_recall_at_k"]["20"] or 0.0) - (metrics["document_recall_at_k"]["20"] or 0.0) > 0.02:
            failed.append("document_recall_at_20_regression")
        if (metrics["development"]["scope_recall_at_k"]["5"] or 0.0) < (baseline["development"]["scope_recall_at_k"]["5"] or 0.0):
            failed.append("development_scope_recall_at_5_regression")
        if (metrics["known_regression"]["scope_recall_at_k"]["5"] or 0.0) < (baseline["known_regression"]["scope_recall_at_k"]["5"] or 0.0):
            failed.append("known_regression_scope_recall_at_5_regression")
        if summary.get("repeatability", {}).get("candidate_rank_agreement", 1.0) != 1.0:
            failed.append("repeatability")
        cost = summary.get("cost_metrics", {})
        base_cost = cost.get("current_chunk_content+current_query", {})
        variant_cost = cost.get(variant, {})
        if (variant_cost.get("embedding_input_token_estimate_p95") or 0.0) > 3 * (base_cost.get("embedding_input_token_estimate_p95") or 0.0):
            failed.append("embedding_input_p95")
        if (variant_cost.get("query_latency_p95") or 0.0) > 2 * (base_cost.get("query_latency_p95") or 0.0):
            failed.append("query_latency_p95")
        if (variant_cost.get("shadow_index_size_bytes") or 0) > 3 * (base_cost.get("shadow_index_size_bytes") or 0):
            failed.append("shadow_index_size")
        decisions[variant] = {"decision": "promotion_candidate" if not failed else "rejected", "gate_pass": not failed, "failed_gates": failed}
        if not failed:
            candidates.append(variant)
    selected = _select_candidate(summary["variants"], candidates)
    return {
        "promotion_decision": "promotion_candidate" if selected else "no_candidate",
        "recommended_variant": selected,
        "variants": decisions,
        "promotion_gates": {
            "scope_recall_at_5_min": PROMOTION_SCOPE_RECALL_AT_5,
            "document_recall_at_20_max_drop": 0.02,
        },
    }


def build_public_representation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    _reject_forbidden_payload(summary)
    return summary


def write_experiment_artifacts(summary: dict[str, Any], traces: list[dict[str, Any]], structure: dict[str, Any]) -> None:
    write_json(RESULT_PATH, build_public_representation_summary(summary))
    write_jsonl(TRACE_PATH, _redacted_traces(traces))
    write_scope_structure_audit(structure)
    write_execution_result(_build_execution_result(summary))
    REPORT_PATH.write_text(build_markdown_report(summary), encoding="utf-8")


def _build_execution_result(summary: dict[str, Any]) -> dict[str, Any]:
    model = summary.get("model_execution") or {}
    manifests = summary.get("embedding_manifest_summary") or {}
    costs = summary.get("cost_metrics") or {}
    baseline_cost = costs.get("current_chunk_content+current_query") or {}
    representation_manifests = {
        key: value.get("representation_manifest")
        for key, value in manifests.items()
        if key.endswith("+current_query") or key in {"current_chunk_content+instructed_query", "heading_enriched_chunk+instructed_query"}
    }
    unique_representation = {}
    for value in representation_manifests.values():
        if isinstance(value, dict):
            unique_representation[value.get("variant_id")] = value
    query_manifests = {}
    for value in manifests.values():
        if isinstance(value, dict) and isinstance(value.get("query_manifest"), dict):
            query_manifests[value["query_manifest"].get("variant_id")] = value["query_manifest"]
    return {
        "schema_version": "opk-rag.qwen-representation-execution-result.v1",
        "execution_id": "phase2-qwen-representation-execution-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": summary.get("representation_experiment_status"),
        "execution_plan_path": summary.get("execution_plan_path"),
        "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "candidate_universe_chunk_count": summary.get("candidate_universe_chunk_count"),
        "candidate_universe_digest": summary.get("candidate_universe_digest"),
        "provider": model.get("provider"),
        "model_id": model.get("model_id"),
        "model_revision": model.get("model_revision_or_local_digest"),
        "tokenizer_digest": model.get("tokenizer_digest"),
        "device": model.get("device"),
        "compute_dtype": model.get("dtype"),
        "storage_dtype": "float32",
        "safe_batch_plan": model.get("safe_batch_plan"),
        "representation_manifests": unique_representation,
        "query_manifests": query_manifests,
        "all_representation_variants_completed_604": all((row or {}).get("stored_vector_count") == 604 for row in unique_representation.values()),
        "all_query_variants_completed_68": all((row or {}).get("stored_vector_count") == 68 for row in query_manifests.values()),
        "formal_matrix_completed": set(summary.get("variants") or {}) == {f"{r}+{q}" for r, q in EXPERIMENT_MATRIX},
        "repeatability": summary.get("repeatability"),
        "cost_metrics": costs,
        "baseline_cost": baseline_cost,
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_variant": summary.get("recommended_variant"),
        "chunk_boundary_issue": summary.get("chunk_boundary_issue"),
        "representation_gap": summary.get("representation_gap"),
        "query_embedding_alignment_issue": summary.get("query_embedding_alignment_issue"),
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
    }


def verify_chunk_representation_experiment() -> dict[str, Any]:
    issues = []
    for code, path in (
        ("missing_contract", CONTRACT_PATH),
        ("missing_result", RESULT_PATH),
        ("missing_trace", TRACE_PATH),
        ("missing_structure_audit", STRUCTURE_AUDIT_PATH),
        ("missing_report", REPORT_PATH),
    ):
        if not path.exists():
            issues.append({"code": code, "path": _display_path(path)})
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    result = read_json(RESULT_PATH) if RESULT_PATH.exists() else {}
    if contract.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_contract_id"})
    if result.get("experiment_id") != EXPERIMENT_ID:
        issues.append({"code": "unexpected_result_id"})
    if result.get("representation_experiment_status") == "completed":
        if result.get("embedding_execution_mode") != "qwen_local_inference":
            issues.append({"code": "formal_result_not_qwen_local_inference"})
        model_execution = result.get("model_execution") or {}
        if model_execution.get("model_id") != DEFAULT_EMBEDDING_MODEL_NAME:
            issues.append({"code": "unexpected_model_id"})
        if model_execution.get("embedding_dimension") != DEFAULT_EMBEDDING_DIMENSION:
            issues.append({"code": "unexpected_embedding_dimension"})
        if model_execution.get("normalization") is not True:
            issues.append({"code": "normalization_not_true"})
        if model_execution.get("local_files_only") is not True:
            issues.append({"code": "local_files_only_not_true"})
        resolved = model_execution.get("model_resolved_path") or {}
        if isinstance(resolved, str) or (isinstance(resolved, dict) and str(resolved.get("display") or "").startswith("/")):
            issues.append({"code": "public_model_absolute_path"})
        if result.get("gold_filtered_candidate_universe") is not False:
            issues.append({"code": "gold_filtered_candidate_universe"})
        expected_count = (result.get("candidate_universe_audit") or {}).get("searchable_chunk_count")
        if expected_count != result.get("candidate_universe_chunk_count"):
            issues.append({"code": "candidate_universe_count_mismatch"})
        for variant, cardinality in (result.get("shadow_index_cardinality") or {}).items():
            if cardinality.get("indexed_vector_count") != expected_count:
                issues.append({"code": "indexed_vector_count_mismatch", "variant": variant})
            if cardinality.get("unique_canonical_identity_count") != expected_count:
                issues.append({"code": "unique_identity_count_mismatch", "variant": variant})
            if cardinality.get("missing_identity_count") != 0 or cardinality.get("duplicate_identity_count") != 0:
                issues.append({"code": "identity_cardinality_not_exact", "variant": variant})
        if not (result.get("representation_render_digest_validation") or {}).get("r0_differs_from_all_other_variants"):
            issues.append({"code": "representation_render_digest_validation_failed"})
        if (result.get("query_render_digest_validation") or {}).get("queries_with_distinct_q0_q1_embeddings", 0) <= 0:
            issues.append({"code": "query_variant_implementation_invalid"})
        for old in result.get("invalid_historical_runs") or []:
            if old.get("indexed_vector_count") == 35 and old.get("result_validity") != "not_valid_for_quality_decision_until_candidate_universe_verified":
                issues.append({"code": "old_35_vector_run_not_invalidated"})
    if result.get("embedding_execution_mode") == "deterministic_surrogate" and result.get("promotion_decision") != "invalid_experiment":
        issues.append({"code": "surrogate_entered_promotion_gate"})
    if result.get("contains_sealed_holdout_data") is not False:
        issues.append({"code": "sealed_holdout_flag_not_false"})
    if result.get("production_default_change") is not False:
        issues.append({"code": "production_default_change_not_false"})
    if result.get("writes_database") is not False or result.get("writes_index") is not False:
        issues.append({"code": "write_flags_not_false"})
    if result.get("model_calls", {}).get("deepseek") is not False:
        issues.append({"code": "deepseek_flag_not_false"})
    expected = {f"{r}+{q}" for r, q in EXPERIMENT_MATRIX}
    if result.get("representation_experiment_status") == "completed" and set(result.get("variants", {})) != expected:
        issues.append({"code": "experiment_matrix_mismatch"})
    if result.get("representation_experiment_status") != "completed":
        if result.get("promotion_decision") != "invalid_experiment":
            issues.append({"code": "invalid_run_promotion_decision_not_invalid"})
        if result.get("recommended_variant") is not None:
            issues.append({"code": "invalid_run_recommended_variant_not_null"})
        if not str(result.get("result_validity") or "").startswith("not_valid_for_quality_decision"):
            issues.append({"code": "invalid_run_result_validity_missing"})
    if TRACE_PATH.exists():
        allowed = {"sample_id", "variant_id", "query_variant_id", "evidence_identity_digest", "rank", "score", "match_status", "gain_attribution", "loss_attribution", "structure_class"}
        for line_number, line in enumerate(TRACE_PATH.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            extra = sorted(set(row) - allowed)
            missing = sorted(allowed - set(row))
            if extra or missing:
                issues.append({"code": "trace_privacy_schema_violation", "line": line_number, "extra_fields": extra, "missing_fields": missing})
                break
    _reject_forbidden_payload(contract)
    _reject_forbidden_payload(result)
    return {"status": "valid" if not issues else "invalid", "issues": issues}


def profile_qwen_representation_execution(contract: dict[str, Any] | None = None, embedding_config: EmbeddingConfig | None = None) -> dict[str, Any]:
    _validate_contract(contract or {})
    config = embedding_config or _embedding_config_from_contract(contract or {})
    chunks, universe = load_candidate_universe_for_formal_run(config)
    context = _build_qwen_execution_context(config)
    texts = [render_chunk_representation("adjacent_context", chunk, chunks) for chunk in chunks[: min(16, len(chunks))]]
    batch_plan = select_safe_batch_plan(context.provider, texts)
    execution_plan = build_execution_plan(
        contract=contract or {},
        candidate_universe=universe,
        model_execution=context.model_execution | {"safe_batch_plan": batch_plan},
        selected_batch_size=batch_plan["selected_batch_size"],
        max_batch_token_budget=batch_plan["max_batch_token_budget"],
    )
    profile = profile_qwen_execution(
        provider=context.provider,
        short_input=render_chunk_representation("current_chunk_content", chunks[0], chunks),
        long_input=max((render_chunk_representation("adjacent_context", chunk, chunks) for chunk in chunks), key=len),
        previous_batch_inputs=texts,
    )
    payload = {
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "candidate_universe_chunk_count": universe["candidate_universe_chunk_count"],
        "candidate_universe_digest": universe["candidate_universe_digest"],
        "execution_plan_path": EXECUTION_PLAN_PATH.relative_to(ROOT).as_posix(),
        "execution_plan_digest": digest_json(execution_plan),
        "batch_plan": batch_plan,
        "profile": profile,
        "cuda_oom_stage": profile.get("oom_stage"),
        "oom_classification": profile.get("oom_classification"),
        "safe_batch_size": batch_plan["selected_batch_size"],
        "safe_token_budget": batch_plan["max_batch_token_budget"],
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
        "deepseek_called": False,
        "production_default_change": False,
    }
    write_json(EXECUTION_RESULT_PATH, payload)
    return payload


def build_markdown_report(summary: dict[str, Any]) -> str:
    rows = []
    for variant, metrics in summary.get("variants", {}).items():
        scope = metrics.get("scope_recall_at_k") or {}
        doc = metrics.get("document_recall_at_k") or {}
        rows.append(f"| `{variant}` | {scope.get('1', 0):.4f} | {scope.get('3', 0):.4f} | {scope.get('5', 0):.4f} | {scope.get('10', 0):.4f} | {scope.get('20', 0):.4f} | {doc.get('20', 0):.4f} | {metrics.get('mean_reciprocal_rank', 0):.4f} |")
    audit = summary.get("scope_structure_audit") or {}
    model = summary.get("model_execution") or {}
    surrogate = summary.get("surrogate_infrastructure_validation") or {}
    universe = summary.get("candidate_universe_audit") or {}
    if not universe and CONTRACT_PATH.exists():
        universe = (read_json(CONTRACT_PATH).get("candidate_universe_contract") or {})
    return "\n".join(
        [
            "# Phase 2 Chunk Representation Experiment",
            "",
            f"Experiment ID: `{summary.get('experiment_id')}`",
            "",
            "This TASK-0056 diagnostic uses public Development and Known Regression data only. It does not read sealed holdout data, run generation, call DeepSeek, write the database, rebuild the production index, or change production defaults.",
            "",
            "## Background",
            "",
            "TASK-0054 established a public candidate retrieval baseline with high document recall but low scope recall. TASK-0055 tested query-preserving, scope metadata, document-first, and combined retrieval strategies; no variant became a candidate. TASK-0056 therefore diagnoses chunk/scope representation and query-chunk alignment.",
            "",
            "## Current Contracts",
            "",
            f"- Chunking: `{summary.get('contract_path')}` records parser `{PARSER_VERSION}`, chunking `{CHUNKING_VERSION}`, character target 1200, max 1800, and overlap 0.",
            "- Query embedding: current Q0 uses the existing fixed `Instruct`/`Query` input contract.",
            "- Document embedding: current persisted contract includes heading path plus chunk content; it excludes document title, relative path, block type, and neighbor context.",
            "",
            "## Embedding Execution",
            "",
            f"- Formal execution mode: `{summary.get('embedding_execution_mode')}`",
            f"- Provider/model: `{model.get('provider')}` / `{model.get('model_id')}`",
            f"- Revision or digest: `{model.get('model_revision_or_local_digest')}`",
            f"- Dimension/device/dtype: `{model.get('embedding_dimension')}` / `{model.get('device')}` / `{model.get('dtype')}`",
            f"- Normalization/local files only: `{model.get('normalization')}` / `{model.get('local_files_only')}`",
            f"- Model path publication: `{json.dumps(model.get('model_resolved_path'), ensure_ascii=False, sort_keys=True)}`",
            f"- Qwen embedding call count: `{summary.get('qwen_embedding_call_count')}`",
            f"- Qwen embedding input count: `{summary.get('qwen_embedding_input_count')}`",
            f"- Qwen model forward batch count: `{summary.get('qwen_model_forward_batch_count')}`",
            f"- Result validity: `{summary.get('result_validity')}`",
            f"- Infrastructure blocked reason: `{summary.get('infrastructure_blocked_reason')}`",
            "",
            "## Candidate Universe",
            "",
            f"- Source: `{universe.get('candidate_universe_source')}`",
            f"- Active/searchable chunks: {universe.get('active_chunk_count', universe.get('expected_searchable_chunk_count'))} / {universe.get('searchable_chunk_count', universe.get('candidate_universe_chunk_count'))}",
            f"- Active/candidate documents: {universe.get('active_document_count', universe.get('expected_active_document_count'))} / {universe.get('candidate_universe_document_count')}",
            f"- Chunks with embedding: {universe.get('chunks_with_embedding_count', universe.get('candidate_universe_chunk_count'))}",
            f"- Chunks excluded by contract: {universe.get('chunks_excluded_by_contract')}",
            f"- Candidate universe digest: `{universe.get('candidate_universe_digest')}`",
            f"- Gold-filtered candidate universe: `{universe.get('gold_filtered_candidate_universe')}`",
            "- Gold fields used for index construction: `[]`",
            "- Old 35-vector result validity: `not_valid_for_quality_decision_until_candidate_universe_verified`",
            "",
            "## Surrogate Isolation",
            "",
            f"- Surrogate execution mode: `{surrogate.get('embedding_execution_mode')}`",
            f"- Surrogate role: `{surrogate.get('surrogate_run_role')}`",
            f"- Surrogate validity: `{surrogate.get('surrogate_result_validity')}`",
            f"- Used by promotion gate: `{surrogate.get('promotion_gate_input')}`",
            "",
            "## Scope Structure Audit",
            "",
            f"- Gold resolved/unresolved: {audit.get('gold_scope_resolved_count')} / {audit.get('gold_scope_unresolved_count')}",
            f"- Single/multi chunk: {audit.get('single_chunk_count')} / {audit.get('multi_chunk_count')}",
            f"- Multi-chunk basis: {audit.get('multi_chunk_basis')}",
            f"- Contiguous/non-contiguous multi-chunk units: {audit.get('multi_chunk_contiguous_adjacent_count')} / {audit.get('multi_chunk_non_contiguous_count')}",
            f"- Required-evidence denominator / unique-scope denominator: {audit.get('required_evidence_unit_denominator')} / {audit.get('scope_identity_denominator')}",
            f"- Required-unit multi-chunk rate: {audit.get('multi_chunk_required_evidence_rate')}",
            f"- Unique-scope multi-chunk rate: {audit.get('unique_multi_chunk_scope_rate')}",
            f"- Duplicate scope counted multiple times: {audit.get('duplicate_scope_counted_multiple_times')}",
            f"- Heading dependency counted as multi-chunk: {audit.get('heading_dependency_counted_as_multi_chunk')}",
            f"- Heading-dependent: {audit.get('heading_dependent_count')}",
            f"- Structured block distribution: `{json.dumps(audit.get('structured_block_distribution') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Fixed Matrix",
            "",
            "`R0+Q0`, `R0+Q1`, `R1+Q0`, `R1+Q1`, `R2+Q0`, `R3+Q0`, and `R4+Q0` were evaluated with K=[1,3,5,10,20].",
            "",
            "| Variant | R@1 | R@3 | R@5 | R@10 | R@20 | Doc R@20 | MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Decision",
            "",
            f"- Promotion decision: `{summary.get('promotion_decision')}`",
            f"- Recommended variant: `{json.dumps(summary.get('recommended_variant'))}`",
            f"- Primary result classification: `{summary.get('primary_result_classification')}`",
            f"- Chunk boundary issue: `{summary.get('chunk_boundary_issue')}`",
            f"- Representation gap: `{summary.get('representation_gap')}`",
            f"- Query embedding alignment issue: `{summary.get('query_embedding_alignment_issue')}`",
            "",
            "## Cost",
            "",
            f"`{json.dumps(summary.get('cost_metrics') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Next Task",
            "",
            str(summary.get("next_task_recommendation")),
            "",
        ]
    )


def update_benchmark_registry(summary: dict[str, Any], contract: dict[str, Any], path: Path = ROOT / "evaluation-data" / "benchmark_registry.json") -> dict[str, Any]:
    registry = read_json(path)
    source_files = [
        RESULT_PATH.relative_to(ROOT).as_posix(),
        TRACE_PATH.relative_to(ROOT).as_posix(),
        STRUCTURE_AUDIT_PATH.relative_to(ROOT).as_posix(),
        CONTRACT_PATH.relative_to(ROOT).as_posix(),
    ]
    row = {
        "artifact_id": EXPERIMENT_ID,
        "artifact_type": "evaluation_result",
        "benchmark_id": EXPERIMENT_ID,
        "role": "development_representation_experiment",
        "exposure_status": "exposed",
        "contains_private_data": False,
        "mutable": False,
        "sample_count": summary.get("sample_count") if summary.get("sample_count") is not None else _registry_count_samples(RESULT_PATH),
        "source_files": source_files,
        "content_checksum": _registry_content_checksum([ROOT / item for item in source_files]),
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "trace_path": TRACE_PATH.relative_to(ROOT).as_posix(),
        "scope_structure_audit_path": STRUCTURE_AUDIT_PATH.relative_to(ROOT).as_posix(),
        "baseline_id": BASELINE_ID,
        "dataset_ids": ["development", "known-regression"],
        "embedding_contract": contract.get("embedding_contract"),
        "embedding_execution_mode": summary.get("embedding_execution_mode"),
        "representation_experiment_status": summary.get("representation_experiment_status"),
        "model_execution": summary.get("model_execution"),
        "qwen_embedding_call_count": summary.get("qwen_embedding_call_count"),
        "surrogate_result_validity": (summary.get("surrogate_infrastructure_validation") or {}).get("surrogate_result_validity"),
        "surrogate_promotion_gate_input": (summary.get("surrogate_infrastructure_validation") or {}).get("promotion_gate_input"),
        "corpus_digest": contract.get("bindings", {}).get("corpus_snapshot_digest"),
        "gold_identity_digests": contract.get("bindings", {}).get("gold_identity_map_digests"),
        "model_calls": "embedding_only",
        "deepseek_calls": False,
        "production_index_changed": False,
        "production_default_changed": False,
        "sealed_holdout_access": False,
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_variant": summary.get("recommended_variant"),
        "intended_use": "diagnose public chunk representation and query-chunk embedding alignment gaps",
        "forbidden_use": "sealed holdout tuning, production default change, embedding model comparison, generation tuning",
        "registered_at": utc_now(),
    }
    execution_row = {
        "artifact_id": "phase2-qwen-representation-execution-plan-v1",
        "artifact_type": "execution_contract",
        "benchmark_id": EXPERIMENT_ID,
        "role": "evaluation_infrastructure",
        "exposure_status": "exposed",
        "contains_private_data": False,
        "mutable": False,
        "source_files": [
            EXECUTION_PLAN_PATH.relative_to(ROOT).as_posix(),
            EXECUTION_RESULT_PATH.relative_to(ROOT).as_posix(),
        ],
        "content_checksum": _registry_content_checksum([path for path in (EXECUTION_PLAN_PATH, EXECUTION_RESULT_PATH) if path.exists()]),
        "execution_plan_path": EXECUTION_PLAN_PATH.relative_to(ROOT).as_posix(),
        "experiment_contract": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result_path": EXECUTION_RESULT_PATH.relative_to(ROOT).as_posix(),
        "candidate_universe_digest": summary.get("candidate_universe_digest"),
        "model_id": (summary.get("model_execution") or {}).get("model_id"),
        "model_revision": (summary.get("model_execution") or {}).get("model_revision_or_local_digest"),
        "tokenizer_digest": (summary.get("model_execution") or {}).get("tokenizer_digest"),
        "compute_dtype": (summary.get("model_execution") or {}).get("dtype"),
        "storage_dtype": "float32",
        "batch_plan": (summary.get("model_execution") or {}).get("safe_batch_plan"),
        "chunk_embedding_count": summary.get("candidate_universe_chunk_count"),
        "query_embedding_count": 68,
        "sealed_holdout_access": False,
        "production_index_changed": False,
        "database_changed": False,
        "promotion_decision": summary.get("promotion_decision"),
        "intended_use": "reproducible local Qwen embedding materialization infrastructure for the frozen public representation experiment",
        "forbidden_use": "quality tuning, sealed holdout tuning, production index mutation, remote embedding fallback",
        "registered_at": utc_now(),
    }
    _reject_forbidden_payload(row)
    _reject_forbidden_payload(execution_row)
    benchmarks = [
        item
        for item in registry.get("benchmarks", [])
        if item.get("artifact_id") not in {EXPERIMENT_ID, "phase2-qwen-representation-execution-plan-v1"}
    ]
    benchmarks.extend([row, execution_row])
    for item in benchmarks:
        if item.get("benchmark_id") == "phase2-stage-acceptance-v1":
            item["content_checksum"] = _registry_content_checksum([ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json"])
    registry["benchmarks"] = benchmarks
    registry["generated_at"] = utc_now()
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _registry_content_checksum(paths: list[Path]) -> str:
    payload = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        payload.append(
            {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_count": _registry_count_samples(path),
            }
        )
    return digest_json(payload)


def _registry_count_samples(path: Path) -> int | None:
    if path.suffix.lower() == ".jsonl":
        return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("samples", "questions", "results", "cases", "records", "benchmarks"):
                value = data.get(key)
                if isinstance(value, list):
                    return len(value)
            if isinstance(data.get("sample_count"), int):
                return int(data["sample_count"])
        return 1
    return None


def update_phase2_governance(summary: dict[str, Any]) -> None:
    representation_status = summary.get("representation_experiment_status") or summary.get("status")
    allowed = {
        "chunk_representation_experiment_status": representation_status,
        "representation_experiment_status": representation_status,
        "chunk_representation_experiment_id": EXPERIMENT_ID,
        "embedding_execution_mode": summary.get("embedding_execution_mode"),
        "chunk_boundary_issue": summary.get("chunk_boundary_issue"),
        "representation_gap": summary.get("representation_gap"),
        "query_embedding_alignment_issue": summary.get("query_embedding_alignment_issue"),
        "promotion_decision": summary.get("promotion_decision"),
        "recommended_variant": summary.get("recommended_variant"),
        "primary_result_classification": summary.get("primary_result_classification"),
    }
    _reject_forbidden_payload(allowed)
    for path in (
        ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json",
        ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json",
    ):
        payload = read_json(path)
        payload.update(allowed)
        if "stage_decision" in payload:
            payload["stage_decision"] = "not_accepted"
        else:
            payload["stage_decision"] = "not_accepted"
        if "holdout_status" in payload:
            payload["holdout_status"] = "exposed"
        else:
            payload["holdout_status"] = "exposed"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank(index: list[tuple[IndexedScopeChunk, list[float]]], query_vector: list[float], *, limit: int) -> tuple[ShadowCandidate, ...]:
    scored = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), chunk) for chunk, vector in index]
    scored.sort(key=lambda item: (-item[0], item[1].document_identity_digest or "", item[1].scope_identity_digest or ""))
    return tuple(ShadowCandidate(rank=index + 1, chunk=chunk, score=score) for index, (score, chunk) in enumerate(scored[:limit]))


def _unit_profile(gold: EvidenceIdentity, candidates: tuple[ShadowCandidate, ...]) -> dict[str, Any]:
    scope_rank = document_rank = None
    for candidate in candidates:
        runtime = normalize_runtime_evidence_identity(candidate.chunk.identity)
        match = match_evidence_identity(gold, runtime)
        if match.matched and match.level.value in {"exact_chunk_match", "exact_scope_match", "compatible_scope_chunk_match"} and scope_rank is None:
            scope_rank = candidate.rank
        if gold.document_identity_digest and candidate.chunk.document_identity_digest == gold.document_identity_digest and document_rank is None:
            document_rank = candidate.rank
    return {"evidence_identity_digest": digest_json(gold.to_json()), "scope_rank": scope_rank, "document_rank": document_rank}


def _aggregate_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in rows)
    candidate_counts = [row["candidate_count"] for row in rows]
    first_ranks = [min((profile["scope_rank"] for profile in row["unit_profiles"] if profile["scope_rank"] is not None), default=None) for row in rows]
    token_estimates = [estimate for row in rows[:1] for estimate in row["embedding_input_token_estimates"]]
    return {
        "sample_count": len(rows),
        "required_unit_count": denominator,
        "scope_recall_at_k": {str(k): (sum(1 for row in rows for p in row["unit_profiles"] if p["scope_rank"] is not None and p["scope_rank"] <= k) / denominator) if denominator else 0.0 for k in K_VALUES},
        "document_recall_at_k": {str(k): (sum(1 for row in rows for p in row["unit_profiles"] if p["document_rank"] is not None and p["document_rank"] <= k) / denominator) if denominator else 0.0 for k in K_VALUES},
        "mean_reciprocal_rank": statistics.mean((1 / rank) if rank else 0.0 for rank in first_ranks) if rows else 0.0,
        "first_relevant_rank_distribution": dict(Counter(str(rank or "missing") for rank in first_ranks)),
        "fully_covered_sample_count": sum(all(p["scope_rank"] is not None and p["scope_rank"] <= 5 for p in row["unit_profiles"]) for row in rows if row["required_units_total"]),
        "candidate_count_mean": statistics.mean(candidate_counts) if candidate_counts else 0.0,
        "candidate_count_p95": _p95(candidate_counts),
        "embedding_input_token_estimate_mean": statistics.mean(token_estimates) if token_estimates else 0.0,
        "embedding_input_token_estimate_p95": _p95(token_estimates),
    }


def _gain_against_baseline(baseline_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    helped_samples = set()
    hurt_samples = set()
    for base, variant in zip(baseline_rows, variant_rows, strict=True):
        base_profiles = {p["evidence_identity_digest"]: p for p in base["unit_profiles"]}
        for profile in variant["unit_profiles"]:
            base_hit = base_profiles[profile["evidence_identity_digest"]]["scope_rank"] is not None
            variant_hit = profile["scope_rank"] is not None
            if variant_hit and not base_hit:
                counts["newly_recovered_required_units"] += 1
                helped_samples.add(variant["sample_id"])
            elif base_hit and not variant_hit:
                counts["lost_required_units"] += 1
                hurt_samples.add(variant["sample_id"])
            elif base_hit and variant_hit:
                counts["unchanged_hits"] += 1
    for key in ("newly_recovered_required_units", "lost_required_units", "unchanged_hits"):
        counts.setdefault(key, 0)
    counts["helped_samples"] = len(helped_samples)
    counts["hurt_samples"] = len(hurt_samples)
    return dict(counts)


def _dataset_metrics(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for dataset_id in ("development", "known-regression"):
        out[dataset_id] = {variant: _aggregate_variant([row for row in rows if row["dataset_id"] == dataset_id]) for variant, rows in grouped.items()}
    return out


def _dimension_metrics(grouped: dict[str, list[dict[str, Any]]], dimension: str) -> dict[str, Any]:
    baseline = grouped["current_chunk_content+current_query"]
    keys = sorted({row[dimension] for row in baseline})
    out = {}
    for key in keys:
        out[key] = {
            "sample_count": sum(row[dimension] == key for row in baseline),
            "sample_size_status": "ok" if sum(row[dimension] == key for row in baseline) >= 3 else "insufficient_sample_size",
            "variants": {variant: _aggregate_variant([row for row in rows if row[dimension] == key]) for variant, rows in grouped.items()},
        }
    return out


def _cost_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["query_latency_ms"] for row in rows]
    sizes = [row["shadow_index_size_bytes"] for row in rows]
    calls = [row["embedding_call_count"] for row in rows]
    token_estimates = [estimate for row in rows[:1] for estimate in row["embedding_input_token_estimates"]]
    return {
        "embedding_input_token_estimate_mean": statistics.mean(token_estimates) if token_estimates else 0.0,
        "embedding_input_token_estimate_p95": _p95(token_estimates),
        "embedding_call_count": max(calls) if calls else 0,
        "chunk_embedding_input_count": (rows[0].get("shadow_cardinality") or {}).get("expected_chunk_count") if rows else 0,
        "query_embedding_input_count": rows[0].get("query_embedding_input_count") if rows else 0,
        "model_forward_batch_count": max((row.get("model_forward_batch_count") or 0 for row in rows), default=0),
        "embedding_wall_time": max((row.get("embedding_wall_time") or 0.0 for row in rows), default=0.0),
        "embedding_throughput_inputs_per_second": (
            (max(calls) / max((row.get("embedding_wall_time") or 0.0 for row in rows), default=0.0))
            if calls and max((row.get("embedding_wall_time") or 0.0 for row in rows), default=0.0)
            else 0.0
        ),
        "index_build_time": max((row.get("index_build_time") or 0.0 for row in rows), default=0.0),
        "query_latency_p50": statistics.median(latencies) if latencies else 0.0,
        "query_latency_p95": _p95(latencies),
        "peak_gpu_allocated_bytes": max((row.get("peak_gpu_allocated_bytes") or 0 for row in rows), default=0),
        "peak_gpu_reserved_bytes": max((row.get("peak_gpu_reserved_bytes") or 0 for row in rows), default=0),
        "peak_host_memory_bytes": max((row.get("peak_host_memory_bytes") or 0 for row in rows), default=0),
        "shadow_index_size_bytes": max(sizes) if sizes else 0,
        "raw_vector_payload_size_bytes": max((row.get("raw_vector_payload_size_bytes") or 0 for row in rows), default=0),
        "shadow_index_serialized_size_bytes": max((row.get("shadow_index_serialized_size_bytes") or 0 for row in rows), default=0),
        "vector_dtype": (rows[0].get("vector_dtype") if rows else None),
        "bytes_per_vector": (rows[0].get("bytes_per_vector") if rows else None),
        "expected_chunk_count": (rows[0].get("shadow_cardinality") or {}).get("expected_chunk_count") if rows else 0,
        "rendered_representation_count": (rows[0].get("shadow_cardinality") or {}).get("rendered_representation_count") if rows else 0,
        "embedded_vector_count": (rows[0].get("shadow_cardinality") or {}).get("embedded_vector_count") if rows else 0,
        "indexed_vector_count": (rows[0].get("shadow_cardinality") or {}).get("indexed_vector_count") if rows else 0,
    }


def compare_repeatability(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_keys = [(row["variant_id"], row["sample_id"], row["candidate_identity_digests"]) for row in left["traces"]]
    right_keys = [(row["variant_id"], row["sample_id"], row["candidate_identity_digests"]) for row in right["traces"]]
    same = left_keys == right_keys
    metrics_same = left["summary"]["variants"] == right["summary"]["variants"]
    return {"candidate_identity_agreement": 1.0 if same else 0.0, "candidate_rank_agreement": 1.0 if same else 0.0, "metric_agreement": 1.0 if metrics_same else 0.0}


def load_candidate_universe_for_formal_run(
    embedding_config: EmbeddingConfig,
    *,
    configuration_fingerprint: str | None = None,
) -> tuple[tuple[IndexedScopeChunk, ...], dict[str, Any]]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ChunkRepresentationExperimentError("DATABASE_URL is required to build the full TASK-0056 candidate universe")
    from opk_rag.db.connection import connect_postgres
    from opk_rag.db.repositories import IndexConfigurationRepository, KnowledgeBaseRepository

    with connect_postgres(database_url) as connection:
        vault_path = Path(os.environ.get("OPK_RAG_VAULT_PATH", ROOT / "source-documents"))
        root_path = normalize_vault_root(vault_path).canonical_path
        kb = KnowledgeBaseRepository(connection).get_by_root_path(root_path)
        if kb is None:
            raise ChunkRepresentationExperimentError(f"Knowledge base not found for vault path: {root_path}")
        fingerprint = configuration_fingerprint or build_configuration_fingerprint(embedding_config)
        index_config = IndexConfigurationRepository(connection).require_by_fingerprint(fingerprint)
        chunks = load_searchable_scope_chunks_from_database(connection, kb.id, index_configuration_id=index_config.id)
        audit = audit_candidate_universe(connection, kb.id, chunks, index_configuration_id=index_config.id)
    expected = read_json(ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json")
    expected_chunk_count = expected.get("indexed_chunk_count")
    expected_doc_count = expected.get("indexed_active_document_count")
    if audit["searchable_chunk_count"] != expected_chunk_count:
        raise ChunkRepresentationExperimentError(
            f"Candidate Universe cardinality mismatch: searchable chunks {audit['searchable_chunk_count']} != frozen public snapshot {expected_chunk_count}"
        )
    if audit["active_document_count"] != expected_doc_count:
        raise ChunkRepresentationExperimentError(
            f"Candidate Universe document mismatch: active documents {audit['active_document_count']} != frozen public snapshot {expected_doc_count}"
        )
    return chunks, audit


def audit_candidate_universe(connection, knowledge_base_id, chunks: tuple[IndexedScopeChunk, ...], *, index_configuration_id=None) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("select count(*) from public.documents where knowledge_base_id = %s and index_status = 'indexed'", (knowledge_base_id,))
        active_document_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            select count(*)
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s
              and d.index_status = 'indexed'
              and (%s::uuid is null or c.index_configuration_id = %s)
            """,
            (knowledge_base_id, index_configuration_id, index_configuration_id),
        )
        active_chunk_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            select count(*)
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s
              and d.index_status = 'indexed'
              and c.embedding is not null
              and (%s::uuid is null or c.index_configuration_id = %s)
            """,
            (knowledge_base_id, index_configuration_id, index_configuration_id),
        )
        chunks_with_embedding_count = int(cursor.fetchone()[0])
    public_rows = [chunk.public_json() for chunk in chunks]
    digest = digest_json(public_rows)
    return {
        "candidate_universe_source": "database_public_indexed_searchable_chunks",
        "active_document_count": active_document_count,
        "active_chunk_count": active_chunk_count,
        "searchable_chunk_count": len(chunks),
        "chunks_with_embedding_count": chunks_with_embedding_count,
        "chunks_excluded_by_contract": active_chunk_count - len(chunks),
        "candidate_universe_chunk_count": len(chunks),
        "candidate_universe_document_count": len({chunk.document_identity_digest for chunk in chunks}),
        "candidate_universe_digest": digest,
        "gold_filtered_candidate_universe": False,
        "exclusion_rules": [
            "documents must belong to the configured knowledge base",
            "documents.index_status must equal indexed",
            "chunks must belong to the frozen embedding index configuration",
            "chunks.embedding must be non-null in production index readiness checks",
        ],
        "forbidden_filters_applied": [],
    }


def _variant_cardinality_from_costs(costs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        variant: {
            key: metrics.get(key)
            for key in (
                "expected_chunk_count",
                "rendered_representation_count",
                "embedded_vector_count",
                "indexed_vector_count",
            )
        }
        | {
            "unique_canonical_identity_count": metrics.get("indexed_vector_count"),
            "duplicate_identity_count": 0,
            "missing_identity_count": 0,
        }
        for variant, metrics in costs.items()
    }


def _shadow_embedding_cache_key_contract(contract: dict[str, Any], model_execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_fields": [
            "execution_plan_digest",
            "experiment_contract_digest",
            "candidate_universe_digest",
            "embedding_model_id",
            "embedding_model_revision",
            "tokenizer_digest",
            "compute_dtype",
            "storage_dtype",
            "representation_variant_id",
            "query_variant_id",
            "representation_template_digest",
            "query_template_digest",
            "canonical_identity_digest",
            "rendered_input_digest",
            "max_sequence_length",
            "truncation_policy",
            "normalization_contract",
        ],
        "execution_plan_digest": model_execution.get("execution_plan_digest"),
        "experiment_contract_digest": digest_json(contract) if contract else None,
        "candidate_universe_digest": (contract.get("candidate_universe_contract") or {}).get("candidate_universe_digest") if contract else None,
        "embedding_model_id": model_execution.get("model_id"),
        "embedding_model_revision": model_execution.get("model_revision_or_local_digest"),
        "tokenizer_digest": model_execution.get("tokenizer_digest"),
        "compute_dtype": model_execution.get("dtype"),
        "storage_dtype": "float32",
        "max_sequence_length": 8192,
        "truncation_policy": "reject_without_silent_truncation",
        "normalization_contract": "l2_normalized_cosine",
        "insufficient_fields_rejected": ["chunk_id", "chunk_content_digest"],
    }


def _gold_data_index_construction_audit() -> dict[str, Any]:
    fields = ["sample_id", "required_evidence", "gold_scope", "gold_document", "gold_identity_map", "question_type", "capability_label", "expected_action"]
    return {
        "gold_fields_forbidden_in_index_construction": fields,
        "gold_fields_used_for_index_construction": [],
        "gold_fields_used_for_candidate_document_filter": [],
        "gold_fields_used_for_candidate_chunk_filter": [],
        "gold_fields_used_for_representation_generation": [],
        "gold_identity_usage": "retrieval_scoring_only_after_full_index_search",
        "passes": True,
    }


def _representation_render_digest_validation(chunks: tuple[IndexedScopeChunk, ...]) -> dict[str, Any]:
    sample_chunks = list(chunks[:20])
    comparisons = {}
    for other in ("heading_enriched_chunk", "document_heading_chunk", "structured_scope", "adjacent_context"):
        distinct = 0
        identical = 0
        cosine_lt_one = 0
        for chunk in sample_chunks:
            r0 = render_chunk_representation("current_chunk_content", chunk, chunks)
            rx = render_chunk_representation(other, chunk, chunks)
            if digest_json(r0) == digest_json(rx):
                identical += 1
            else:
                distinct += 1
                if sum(a * b for a, b in zip(_hash_embedding(r0), _hash_embedding(rx), strict=True)) < 0.999999:
                    cosine_lt_one += 1
        comparisons[f"current_chunk_content_vs_{other}"] = {
            "sampled_chunk_count": len(sample_chunks),
            "distinct_render_digest_count": distinct,
            "identical_render_digest_count": identical,
            "synthetic_embedding_cosine_lt_1_count": cosine_lt_one,
        }
    return {
        "r0_differs_from_all_other_variants": all(row["distinct_render_digest_count"] > 0 for row in comparisons.values()),
        "comparisons": comparisons,
        "publishes_representation_text": False,
        "publishes_embedding_vectors": False,
    }


def _query_render_digest_validation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    distinct = identical = 0
    for sample in samples:
        q0 = render_query_representation("current_query", sample["question"])
        q1 = render_query_representation("instructed_query", sample["question"])
        if digest_json(q0) == digest_json(q1):
            identical += 1
        else:
            distinct += 1
    return {
        "queries_with_distinct_q0_q1_embeddings": distinct,
        "queries_with_identical_q0_q1_embeddings": identical,
        "query_variant_effective": distinct > 0,
        "publishes_query_text": False,
        "publishes_embedding_vectors": False,
    }


def _shadow_control_comparison(control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    production = read_json(CANDIDATE_RESULT_PATH) if CANDIDATE_RESULT_PATH.exists() else {}
    vector = (production.get("modes") or {}).get("vector") or {}
    control = _aggregate_variant(control_rows)
    return {
        "control_id": "shadow_qwen_full_corpus_control",
        "production_baseline_id": BASELINE_ID,
        "strict_equivalence_claimed": False,
        "scope_recall_at_k_delta": {
            str(k): control["scope_recall_at_k"].get(str(k), 0.0) - ((vector.get("scope_recall_at_k") or {}).get(str(k), 0.0))
            for k in K_VALUES
        },
        "document_recall_at_k_delta": {
            str(k): control["document_recall_at_k"].get(str(k), 0.0) - ((vector.get("document_recall_at_k") or {}).get(str(k), 0.0))
            for k in K_VALUES
        },
        "mrr_delta": control["mean_reciprocal_rank"] - (vector.get("mean_reciprocal_rank") or 0.0),
        "difference_sources": [
            "Shadow Index uses exact in-memory cosine search over the full searchable corpus",
            "Production Vector Baseline uses database vector RPC and persisted production embeddings",
            "TASK-0056 regenerates representation embeddings for each shadow variant",
        ],
    }


def _historical_35_vector_invalid_record() -> dict[str, Any]:
    previous = read_json(RESULT_PATH) if RESULT_PATH.exists() else {}
    costs = previous.get("cost_metrics") or {}
    sizes = [row.get("shadow_index_size_bytes") for row in costs.values() if isinstance(row, dict)]
    size = next((int(value) for value in sizes if value), None)
    indexed = size // (DEFAULT_EMBEDDING_DIMENSION * 4) if size else None
    return {
        "historical_result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "shadow_index_size_bytes": size,
        "indexed_vector_count": indexed,
        "invalid_reason": "shadow index was built from the 35 local source-document chunks instead of the frozen 604-chunk searchable corpus",
        "result_validity": "not_valid_for_quality_decision_until_candidate_universe_verified",
    }


def _redacted_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        variant_gain = "unattributed"
        variant_loss = "unattributed"
        for profile in trace["unit_profiles"]:
            rows.append(
                {
                    "sample_id": trace["sample_id"],
                    "variant_id": trace["variant_id"],
                    "query_variant_id": trace["query_variant_id"],
                    "evidence_identity_digest": profile["evidence_identity_digest"],
                    "rank": profile["scope_rank"],
                    "score": None,
                    "match_status": "matched_within_top20" if profile["scope_rank"] is not None else "not_found_within_top20",
                    "gain_attribution": variant_gain,
                    "loss_attribution": variant_loss,
                    "structure_class": "scope",
                }
            )
    return rows


def _hash_embedding(text: str, dimension: int = 1024) -> list[float]:
    vector = [0.0] * dimension
    tokens = re_tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def re_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _required(sample: dict[str, Any]) -> Iterable[EvidenceIdentity]:
    yield from deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in (sample.get("required_evidence") or []))


def _p95(values: Iterable[float]) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
    return values[index]


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _structure_marker(chunk: IndexedScopeChunk) -> str:
    if "list_block" in chunk.block_types:
        return "[List]"
    if "table_block" in chunk.block_types:
        return "[Table]"
    if "code_block" in chunk.block_types:
        return "[Code]"
    if "quote_block" in chunk.block_types:
        return "[Quote]"
    return "[Paragraph]"


def _adjacent_chunks(chunk: IndexedScopeChunk, chunks: tuple[IndexedScopeChunk, ...]) -> tuple[IndexedScopeChunk | None, IndexedScopeChunk | None]:
    same_doc = [item for item in chunks if item.document_identity_digest == chunk.document_identity_digest]
    same_doc.sort(key=lambda item: item.chunk_index)
    index = next((i for i, item in enumerate(same_doc) if item.chunk_id == chunk.chunk_id), None)
    if index is None:
        return None, None
    return (same_doc[index - 1] if index > 0 else None, same_doc[index + 1] if index + 1 < len(same_doc) else None)


def _neighbor_excerpt(text: str, budget: int = 240) -> str:
    return normalize_embedding_text(text)[:budget]


def _representation_variant_contract(variant: str) -> dict[str, Any]:
    descriptions = {
        "current_chunk_content": {"inputs": ["heading_path", "chunk_content"], "template": "existing render_embedding_input equivalent"},
        "heading_enriched_chunk": {"inputs": ["heading_path", "chunk_content"], "template": "[Heading]\\n{heading}\\n\\n[Content]\\n{content}"},
        "document_heading_chunk": {"inputs": ["document_title", "heading_path", "chunk_content"], "template": "[Document]\\n{title}\\n\\n[Heading]\\n{heading}\\n\\n[Content]\\n{content}"},
        "structured_scope": {"inputs": ["document_title", "heading_path", "block_type", "chunk_content"], "template": "[Document]\\n{title}\\n\\n[Heading]\\n{heading}\\n\\n[BlockType]\\n{content}"},
        "adjacent_context": {"inputs": ["previous_neighbor_excerpt", "chunk_content", "next_neighbor_excerpt"], "template": "[Previous]\\n{prev}\\n\\n[Current]\\n{content}\\n\\n[Next]\\n{next}", "neighbor_character_budget": 240},
    }
    return {"id": variant, **descriptions[variant], "uses_gold_evidence": False, "publishes_private_text": False}


def _query_variant_contract(variant: str) -> dict[str, Any]:
    if variant == "current_query":
        return {"id": variant, "template": "existing render_query_input", "instruction": QUERY_INSTRUCTION, "uses_gold_evidence": False}
    return {"id": variant, "template": INSTRUCTED_QUERY_TEMPLATE, "instruction": "Retrieve the knowledge-base passage that directly supports the question.", "uses_gold_evidence": False}


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract and contract.get("experiment_id") != EXPERIMENT_ID:
        raise ChunkRepresentationExperimentError("unexpected chunk representation contract")
    _reject_forbidden_payload(contract)


def _require_predecessor_artifacts() -> None:
    for path in (CANDIDATE_CONTRACT_PATH, CANDIDATE_RESULT_PATH, SCOPE_CONTRACT_PATH, SCOPE_RESULT_PATH):
        if not path.exists():
            raise ChunkRepresentationExperimentError(f"required predecessor artifact is missing: {_display_path(path)}")


def _select_candidate(variants: dict[str, Any], candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda key: (
            -(variants[key]["scope_recall_at_k"]["5"] or 0.0),
            -(variants[key]["scope_recall_at_k"]["20"] or 0.0),
            variants[key].get("embedding_input_token_estimate_p95", 0.0),
        ),
    )[0]


def _primary_result_classification(summary: dict[str, Any]) -> str:
    if summary.get("promotion_decision") == "promotion_candidate":
        variant = summary.get("recommended_variant") or ""
        if "instructed_query" in variant and "heading_enriched" in variant:
            return "combined_query_representation_gain"
        if "instructed_query" in variant:
            return "query_instruction_gain"
        if "heading_enriched" in variant:
            return "heading_context_gain"
        if "document_heading" in variant:
            return "document_context_gain"
        if "structured_scope" in variant:
            return "structured_scope_gain"
        if "adjacent_context" in variant:
            return "adjacent_context_gain"
    if summary.get("chunk_boundary_issue") == "confirmed":
        return "chunk_boundary_bottleneck"
    if summary.get("representation_gap") == "not_confirmed":
        return "representation_gap_not_confirmed"
    return "no_representation_candidate"


def _representation_gap(summary: dict[str, Any]) -> str:
    baseline = summary["variants"]["current_chunk_content+current_query"]
    best = max((metrics["scope_recall_at_k"]["5"] or 0.0 for metrics in summary["variants"].values()), default=0.0)
    resolved = summary.get("scope_structure_audit", {}).get("gold_scope_resolved_count", 0)
    if resolved and best > (baseline["scope_recall_at_k"]["5"] or 0.0):
        return "confirmed"
    if resolved:
        return "not_confirmed"
    return "insufficient_evidence"


def _query_alignment_issue(summary: dict[str, Any]) -> str:
    baseline = summary["variants"]["current_chunk_content+current_query"]["scope_recall_at_k"]["5"] or 0.0
    q1 = summary["variants"]["current_chunk_content+instructed_query"]["scope_recall_at_k"]["5"] or 0.0
    hq = summary["variants"]["heading_enriched_chunk+instructed_query"]["scope_recall_at_k"]["5"] or 0.0
    h = summary["variants"]["heading_enriched_chunk+current_query"]["scope_recall_at_k"]["5"] or 0.0
    if q1 > baseline or hq > h:
        return "confirmed"
    return "not_confirmed"


def _render_production_document_embedding_input(chunk: IndexedScopeChunk) -> str:
    content = normalize_embedding_text(chunk.content)
    heading = "\n".join(normalize_embedding_text(part) for part in chunk.heading_path if normalize_embedding_text(part))
    if heading:
        return f"{heading}\n\n{content}"
    return content


def _embedding_config_from_contract(contract: dict[str, Any]) -> EmbeddingConfig:
    embedding_contract = contract.get("embedding_contract") or {}
    return EmbeddingConfig(
        provider=embedding_contract.get("provider") or "local_qwen",
        model_name=embedding_contract.get("model") or DEFAULT_EMBEDDING_MODEL_NAME,
        model_revision=embedding_contract.get("model_revision") or DEFAULT_EMBEDDING_MODEL_REVISION,
        dimension=embedding_contract.get("dimension") or DEFAULT_EMBEDDING_DIMENSION,
        normalize=bool(embedding_contract.get("l2_normalization", True)),
        local_files_only=True,
    )


def _build_qwen_execution_context(
    config: EmbeddingConfig,
    *,
    provider: QwenLocalEmbeddingProvider | None = None,
) -> QwenExecutionContext:
    if config.provider == "deterministic_surrogate":
        raise ChunkRepresentationInfrastructureBlocked("formal representation experiment refuses surrogate provider")
    if config.provider != "local_qwen":
        raise ChunkRepresentationInfrastructureBlocked(f"formal representation experiment requires local_qwen provider, got {config.provider}")
    if config.model_name != DEFAULT_EMBEDDING_MODEL_NAME:
        raise ChunkRepresentationInfrastructureBlocked(f"formal representation experiment requires {DEFAULT_EMBEDDING_MODEL_NAME}, got {config.model_name}")
    if config.dimension != DEFAULT_EMBEDDING_DIMENSION:
        raise ChunkRepresentationInfrastructureBlocked(f"formal representation experiment requires {DEFAULT_EMBEDDING_DIMENSION} dimensions")
    if config.normalize is not True:
        raise ChunkRepresentationInfrastructureBlocked("formal representation experiment requires L2 normalization")
    if config.local_files_only is not True:
        raise ChunkRepresentationInfrastructureBlocked("formal representation experiment requires local_files_only=true")
    try:
        resolved_path = _resolve_local_model_path(config)
        provider = provider or QwenLocalEmbeddingProvider(config)
        probe = list(provider.embed_documents(["Qwen embedding contract probe."]))[0]
    except (EmbeddingModelLoadError, EmbeddingInferenceError, Exception) as exc:
        raise ChunkRepresentationInfrastructureBlocked(f"Qwen embedding infrastructure unavailable: {exc}") from exc
    _validate_qwen_vector(probe, config.dimension)
    dtype = _provider_dtype(provider)
    model_execution = {
        "provider": "QwenLocalEmbeddingProvider",
        "model_id": config.model_name,
        "model_revision_or_local_digest": config.model_revision or _directory_digest(resolved_path),
        "model_resolved_path": _redacted_model_path(resolved_path),
        "tokenizer_digest": _tokenizer_digest(resolved_path),
        "embedding_dimension": len(probe),
        "similarity": config.distance_metric,
        "normalization": config.normalize,
        "device": provider.device,
        "dtype": dtype,
        "local_files_only": config.local_files_only,
        "absolute_model_path_public": False,
    }
    _write_private_model_execution(model_execution, resolved_path)
    return QwenExecutionContext(provider=provider, model_execution=model_execution)


def _embed_matrix_inputs(
    samples: list[dict[str, Any]],
    chunks: tuple[IndexedScopeChunk, ...],
    context: QwenExecutionContext,
    *,
    expected_chunk_count: int,
) -> dict[tuple[str, str], ShadowEmbeddingRun]:
    if context.execution_plan is None:
        raise ChunkRepresentationInfrastructureBlocked("Qwen execution plan was not initialized")
    chunk_runs: dict[str, tuple[dict[str, list[float]], list[int], dict[str, Any]]] = {}
    query_runs: dict[str, tuple[dict[str, list[float]], dict[str, Any]]] = {}
    for representation_id in REPRESENTATION_VARIANTS:
        texts = [render_chunk_representation(representation_id, chunk, chunks) for chunk in chunks]
        identities = [digest_json(chunk.identity) for chunk in chunks]
        template_digest = ((context.execution_plan.get("bindings") or {}).get("representation_template_digests") or {}).get(representation_id)
        cache_rows = [
            build_cache_key_row(
                execution_plan=context.execution_plan,
                identity=identity,
                rendered_text=text,
                variant_id=representation_id,
                query_variant_id=None,
                template_digest=template_digest,
            )
            for identity, text in zip(identities, texts, strict=True)
        ]
        materialized = materialize_embedding_variant(
            variant_id=representation_id,
            kind="representation",
            identities=identities,
            texts=texts,
            provider=context.provider,
            execution_plan=context.execution_plan,
            cache_key_rows=cache_rows,
            expected_count=expected_chunk_count,
            count_tokens=lambda text: _count_tokens(context.provider, text),
        )
        keyed = materialized.vectors
        cardinality = _shadow_cardinality(expected_chunk_count, chunks, texts, keyed)
        _validate_shadow_cardinality(cardinality)
        manifest = dict(materialized.manifest)
        manifest["cardinality"] = cardinality
        manifest["vector_digest"] = vector_digest(keyed)
        chunk_runs[representation_id] = (keyed, materialized.token_counts, manifest)
    for query_id in QUERY_VARIANTS:
        texts = [render_query_representation(query_id, sample["question"]) for sample in samples]
        identities = [sample["sample_id"] for sample in samples]
        template_digest = ((context.execution_plan.get("bindings") or {}).get("query_template_digests") or {}).get(query_id)
        cache_rows = [
            build_cache_key_row(
                execution_plan=context.execution_plan,
                identity=identity,
                rendered_text=text,
                variant_id=None,
                query_variant_id=query_id,
                template_digest=template_digest,
            )
            for identity, text in zip(identities, texts, strict=True)
        ]
        materialized = materialize_embedding_variant(
            variant_id=query_id,
            kind="query",
            identities=identities,
            texts=texts,
            provider=context.provider,
            execution_plan=context.execution_plan,
            cache_key_rows=cache_rows,
            expected_count=len(texts),
            count_tokens=lambda text: _count_tokens(context.provider, text),
        )
        query_runs[query_id] = (materialized.vectors, materialized.manifest)
    out = {}
    for representation_id, query_id in EXPERIMENT_MATRIX:
        chunk_vectors, token_estimates, chunk_manifest = chunk_runs[representation_id]
        query_vectors, query_manifest = query_runs[query_id]
        raw_size = sum(len(vector) * 4 for vector in chunk_vectors.values())
        metadata_size = len(json.dumps([chunk.public_json() for chunk in chunks], ensure_ascii=False, sort_keys=True).encode("utf-8"))
        combined_manifest = {
            "representation_manifest": _public_manifest(chunk_manifest),
            "query_manifest": _public_manifest(query_manifest),
            "embedding_manifest_digest": digest_json(
                {
                    "representation_manifest": _public_manifest(chunk_manifest),
                    "query_manifest": _public_manifest(query_manifest),
                }
            ),
        }
        out[(representation_id, query_id)] = ShadowEmbeddingRun(
            chunk_vectors=chunk_vectors,
            query_vectors=query_vectors,
            embedding_input_token_estimates=token_estimates,
            embedding_call_count=(chunk_manifest.get("expected_input_count") or 0) + (query_manifest.get("expected_input_count") or 0),
            embedding_wall_time=(chunk_manifest.get("embedding_wall_time") or 0.0) + (query_manifest.get("embedding_wall_time") or 0.0),
            index_build_time=0.0,
            shadow_index_size_bytes=raw_size,
            raw_vector_payload_size_bytes=raw_size,
            shadow_index_serialized_size_bytes=raw_size + metadata_size,
            model_forward_batch_count=(chunk_manifest.get("batch_count") or 0) + (query_manifest.get("batch_count") or 0),
            query_embedding_input_count=query_manifest.get("expected_input_count") or 0,
            peak_gpu_allocated_bytes=max(
                value for value in (chunk_manifest.get("peak_gpu_allocated_bytes"), query_manifest.get("peak_gpu_allocated_bytes"), 0) if value is not None
            ),
            peak_gpu_reserved_bytes=max(
                value for value in (chunk_manifest.get("peak_gpu_reserved_bytes"), query_manifest.get("peak_gpu_reserved_bytes"), 0) if value is not None
            ),
            peak_host_memory_bytes=max(
                value for value in (chunk_manifest.get("peak_host_memory_bytes"), query_manifest.get("peak_host_memory_bytes"), 0) if value is not None
            ),
            manifest=combined_manifest,
            cardinality=chunk_manifest["cardinality"],
        )
    return out


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"completed_identities", "vector_path_private"}
    return {key: value for key, value in manifest.items() if key not in forbidden}


def _shadow_cardinality(
    expected_chunk_count: int,
    chunks: tuple[IndexedScopeChunk, ...],
    rendered_texts: list[str],
    vectors: dict[str, list[float]],
) -> dict[str, Any]:
    identities = [digest_json(chunk.identity) for chunk in chunks]
    counts = Counter(identities)
    return {
        "expected_chunk_count": expected_chunk_count,
        "rendered_representation_count": len(rendered_texts),
        "embedded_vector_count": len(vectors),
        "indexed_vector_count": len(vectors),
        "unique_canonical_identity_count": len(counts),
        "duplicate_identity_count": sum(count - 1 for count in counts.values() if count > 1),
        "missing_identity_count": max(0, expected_chunk_count - len(vectors)),
    }


def _validate_shadow_cardinality(cardinality: dict[str, Any]) -> None:
    expected = cardinality["expected_chunk_count"]
    for key in ("rendered_representation_count", "embedded_vector_count", "indexed_vector_count", "unique_canonical_identity_count"):
        if cardinality.get(key) != expected:
            raise ChunkRepresentationExperimentError(f"Shadow Index cardinality mismatch: {key}={cardinality.get(key)} expected={expected}")
    if cardinality.get("duplicate_identity_count") or cardinality.get("missing_identity_count"):
        raise ChunkRepresentationExperimentError(f"Shadow Index identity cardinality mismatch: {cardinality}")


def _embed_texts(provider: QwenLocalEmbeddingProvider, texts: list[str]) -> list[list[float]]:
    vectors = []
    batch_size = max(1, provider.config.batch_size)
    for index in range(0, len(texts), batch_size):
        batch = texts[index : index + batch_size]
        vectors.extend(list(vector) for vector in provider.embed_documents(batch))
    for vector in vectors:
        _validate_qwen_vector(vector, DEFAULT_EMBEDDING_DIMENSION)
    return vectors


def _validate_qwen_vector(vector: list[float], dimension: int) -> None:
    if len(vector) != dimension:
        raise ChunkRepresentationInfrastructureBlocked(f"Qwen embedding dimension mismatch: {len(vector)} != {dimension}")
    norm = sum(float(value) * float(value) for value in vector) ** 0.5
    if abs(norm - 1.0) > 1e-3:
        raise ChunkRepresentationInfrastructureBlocked(f"Qwen embedding is not L2 normalized: norm={norm:.6f}")


def _count_tokens(provider: QwenLocalEmbeddingProvider, text: str) -> int:
    try:
        return provider.count_tokens(text)
    except Exception:
        return _token_estimate(text)


def _resolve_local_model_path(config: EmbeddingConfig) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ChunkRepresentationInfrastructureBlocked("huggingface_hub is required to resolve the local Qwen model path") from exc
    try:
        return Path(
            snapshot_download(
                repo_id=config.model_name,
                revision=config.model_revision,
                cache_dir=config.cache_dir_string,
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise ChunkRepresentationInfrastructureBlocked(f"local Qwen model files are unavailable for {config.model_id}") from exc


def _redacted_model_path(path: Path) -> dict[str, Any]:
    return {
        "resolved": True,
        "display": "<redacted-local-model-path>",
        "path_digest": hashlib.sha256(path.resolve().as_posix().encode("utf-8")).hexdigest(),
    }


def _tokenizer_digest(path: Path) -> str:
    files = sorted(item for item in path.glob("tokenizer*") if item.is_file())
    if not files:
        return "missing-tokenizer-files"
    return _files_digest(files)


def _directory_digest(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file() and item.stat().st_size < 20_000_000)
    return _files_digest(files[:200])


def _files_digest(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(hashlib.sha256(file_path.read_bytes()).digest())
    return digest.hexdigest()


def _provider_dtype(provider: QwenLocalEmbeddingProvider) -> str:
    try:
        model = provider._model
        for parameter in model.parameters():
            return str(parameter.dtype).replace("torch.", "")
    except Exception:
        return "unknown"
    return "unknown"


def _write_private_model_execution(model_execution: dict[str, Any], resolved_path: Path) -> None:
    PRIVATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    private = dict(model_execution)
    private["model_resolved_absolute_path"] = resolved_path.resolve().as_posix()
    (PRIVATE_CACHE_PATH / "qwen_model_execution_private.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_private_shadow_cache(
    kind: str,
    variant_id: str,
    vectors: dict[str, list[float]],
    *,
    chunks: tuple[IndexedScopeChunk, ...],
    rendered_texts: list[str],
    context: QwenExecutionContext,
) -> None:
    cache_dir = PRIVATE_CACHE_PATH / "shadow_cache" / kind
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{variant_id}.json"
    keys = []
    if chunks:
        for chunk, text in zip(chunks, rendered_texts, strict=True):
            keys.append(
                {
                    "cache_key": digest_json(
                        {
                            "experiment_contract_digest": context.contract_digest,
                            "embedding_model_revision": context.model_execution.get("model_revision_or_local_digest"),
                            "tokenizer_digest": context.model_execution.get("tokenizer_digest"),
                            "representation_variant_id": variant_id,
                            "query_variant_id": None,
                            "representation_template_digest": digest_json(_representation_variant_contract(variant_id)),
                            "canonical_chunk_identity": digest_json(chunk.identity),
                            "rendered_input_digest": digest_json(text),
                            "normalization_contract": "l2_normalized_cosine",
                            "dtype": context.model_execution.get("dtype"),
                        }
                    ),
                    "canonical_chunk_identity": digest_json(chunk.identity),
                    "rendered_input_digest": digest_json(text),
                }
            )
    else:
        for text in rendered_texts:
            keys.append(
                {
                    "cache_key": digest_json(
                        {
                            "experiment_contract_digest": context.contract_digest,
                            "embedding_model_revision": context.model_execution.get("model_revision_or_local_digest"),
                            "tokenizer_digest": context.model_execution.get("tokenizer_digest"),
                            "representation_variant_id": None,
                            "query_variant_id": variant_id,
                            "representation_template_digest": None,
                            "canonical_chunk_identity": None,
                            "rendered_input_digest": digest_json(text),
                            "normalization_contract": "l2_normalized_cosine",
                            "dtype": context.model_execution.get("dtype"),
                        }
                    ),
                    "rendered_input_digest": digest_json(text),
                }
            )
    path.write_text(
        json.dumps(
            {
                "variant_id": variant_id,
                "kind": kind,
                "cache_key_contract": _shadow_embedding_cache_key_contract({}, context.model_execution),
                "cache_keys": keys,
                "vectors": vectors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_private_shadow_cache(
    kind: str,
    variant_id: str,
    *,
    chunks: tuple[IndexedScopeChunk, ...],
    rendered_texts: list[str],
    expected_count: int,
) -> dict[str, list[float]] | None:
    path = PRIVATE_CACHE_PATH / "shadow_cache" / kind / f"{variant_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    vectors = payload.get("vectors")
    keys = payload.get("cache_keys")
    if not isinstance(vectors, dict) or len(vectors) != expected_count or not isinstance(keys, list) or len(keys) != expected_count:
        return None
    if chunks:
        expected_identities = [digest_json(chunk.identity) for chunk in chunks]
        if set(vectors) != set(expected_identities):
            return None
        key_by_identity = {row.get("canonical_chunk_identity"): row for row in keys if isinstance(row, dict)}
        for chunk, text in zip(chunks, rendered_texts, strict=True):
            identity = digest_json(chunk.identity)
            row = key_by_identity.get(identity)
            if not row or row.get("rendered_input_digest") != digest_json(text):
                return None
        return {key: [float(value) for value in vector] for key, vector in vectors.items()}
    expected_digests = [digest_json(text) for text in rendered_texts]
    actual_digests = [row.get("rendered_input_digest") for row in keys if isinstance(row, dict)]
    if actual_digests != expected_digests:
        return None
    return {key: [float(value) for value in vector] for key, vector in vectors.items()}


def _surrogate_infrastructure_validation_record() -> dict[str, Any]:
    previous = read_json(RESULT_PATH) if RESULT_PATH.exists() else {}
    return {
        "embedding_execution_mode": "deterministic_surrogate",
        "surrogate_run_role": "infrastructure_validation",
        "surrogate_result_validity": "not_valid_for_representation_quality_decision",
        "promotion_gate_input": False,
        "historical_observation_preserved": bool(previous),
        "historical_promotion_decision": previous.get("promotion_decision"),
        "historical_recommended_variant": previous.get("recommended_variant"),
        "historical_representation_gap": previous.get("representation_gap"),
        "historical_query_embedding_alignment_issue": previous.get("query_embedding_alignment_issue"),
    }


def _infrastructure_blocked_summary(contract: dict[str, Any], structure: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "infrastructure_blocked",
        "representation_experiment_status": "infrastructure_blocked",
        "embedding_execution_mode": "qwen_local_inference",
        "infrastructure_blocked_reason": reason,
        "promotion_decision": "invalid_experiment",
        "recommended_variant": None,
        "primary_result_classification": "invalid_experiment",
        "representation_gap": "insufficient_evidence",
        "query_embedding_alignment_issue": "insufficient_evidence",
        "scope_structure_audit_status": "completed",
        "chunk_boundary_issue": structure.get("chunk_boundary_issue"),
        "scope_structure_audit": {key: value for key, value in structure.items() if key != "records"},
        "surrogate_infrastructure_validation": _surrogate_infrastructure_validation_record(),
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "contract_digest": digest_json(contract) if contract else None,
        "contains_sealed_holdout_data": False,
        "model_calls": {"embedding_only": True, "deepseek": False, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "production_default_change": False,
        "production_default_changed": False,
        "stage_decision": "not_accepted",
        "holdout_status": "exposed",
        "result_validity": "not_valid_for_quality_decision_until_full_corpus_qwen_run_completes",
    }


def _next_task_recommendation(summary: dict[str, Any]) -> str:
    classification = summary.get("primary_result_classification")
    if classification == "chunk_boundary_bottleneck":
        return "Next task: design Scope-aware Chunking v2 in a shadow index and compare heading/list/table/cross-segment boundaries before runtime integration."
    if summary.get("representation_gap") == "confirmed" and summary.get("promotion_decision") == "no_candidate":
        return "Next task: establish a narrower Representation v2 contract before any model comparison."
    if classification == "no_representation_candidate":
        return "Next task: evaluate embedding model suitability, query/document objective alignment, scope summaries, or multi-vector retrieval under a new frozen contract."
    return "Next task: integrate the selected representation only as opt-in shadow runtime contract, then run public regression and real Vault smoke."


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
