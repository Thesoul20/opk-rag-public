from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Literal
from uuid import UUID

from opk_rag.db.models import BM25SearchRow, ChunkSearchRow
from opk_rag.db.repositories import (
    BM25SearchRepository,
    ChunkSearchRepository,
    IndexConfigurationRepository,
    KnowledgeBaseRepository,
    LexicalIndexRepository,
)
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.embedding.query import build_query_input_hash, normalize_query, prepare_query_input, render_query_input
from opk_rag.evaluation.evidence_identity import (
    CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
    EvidenceIdentity,
    MatchLevel,
    deduplicate_evidence_identities,
    digest_json,
    identity_from_runtime_evidence_item,
    match_evidence_identity,
    normalize_gold_evidence_identity,
    normalize_runtime_evidence_identity,
)
from opk_rag.evaluation.required_evidence_diagnostics import load_diagnostic_dataset
from opk_rag.lexical.config import LexicalIndexConfig, build_lexical_configuration_fingerprint
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer, normalize_lexical_text
from opk_rag.search.config import VectorSearchConfig
from opk_rag.search.service import (
    _fused_to_result,
    _require_valid_similarity,
    _rrf_fuse,
    _stable_sort,
    _to_bm25_result,
    _to_vector_result,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_candidate_retrieval_contract_v1.json"
RESULT_PATH = ROOT / "evaluation-data" / "results" / "phase2_candidate_retrieval_baseline_v1.json"
RANK_TRACE_PATH = ROOT / "evaluation-data" / "results" / "phase2_candidate_retrieval_baseline_v1_ranks.jsonl"
PUBLIC_CORPUS_SNAPSHOT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
BENCHMARK_REGISTRY_PATH = ROOT / "evaluation-data" / "benchmark_registry.json"
DATASET_SOURCE_PATHS = {
    "development": (ROOT / "evaluation-data" / "results" / "task0027_retrieval_dev_results.jsonl",),
    "known-regression": (
        ROOT / "evaluation-data" / "results" / "task0027_retrieval_dev_from_fixture_results.jsonl",
        ROOT / "evaluation-data" / "answerability_test.jsonl",
    ),
}
GOLD_IDENTITY_PATHS = {
    "development": ROOT / "evaluation-data" / "diagnostics" / "phase2_development_gold_identity_v1.json",
    "known-regression": ROOT / "evaluation-data" / "diagnostics" / "phase2_known_regression_gold_identity_v1.json",
}
FORBIDDEN_DATASETS = {"sealed_holdout", "holdout", "phase2-holdout-v1"}
RETRIEVAL_MODES = ("lexical", "vector", "hybrid")
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
SCHEMA_VERSION = "opk-rag.candidate-retrieval-baseline-summary.v1"
CONTRACT_SCHEMA_VERSION = "opk-rag.candidate-retrieval-contract.v1"
BASELINE_ID = "phase2-candidate-retrieval-baseline-v1"


class CandidateRetrievalBaselineError(ValueError):
    pass


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    identity: dict[str, Any]
    document_identity_digest: str | None
    scope_identity_digest: str | None
    source: Literal["lexical_only", "vector_only", "both"]
    lexical_rank: int | None
    vector_rank: int | None
    hybrid_rank: int | None
    fusion_score: float | None
    score_type: str
    score: float | None

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def read_json(path: Path) -> Any:
    _reject_forbidden_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    _reject_forbidden_path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    _reject_forbidden_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    _reject_forbidden_payload(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build_retrieval_contract(
    *,
    dataset_ids: list[str],
    search_config: VectorSearchConfig,
    embedding_config: EmbeddingConfig,
    environment_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_ids = _validate_dataset_ids(dataset_ids)
    corpus = read_json(PUBLIC_CORPUS_SNAPSHOT_PATH)
    k_values = [k for k in DEFAULT_K_VALUES if k <= max(search_config.candidate_k, search_config.bm25_candidate_k)]
    if not k_values:
        k_values = [search_config.top_k]
    config_payload = _retrieval_config_payload(search_config, embedding_config)
    query_payload = {
        "raw_question": "used_without_manual_rewrite",
        "vector_normalization": "opk_rag.embedding.query.normalize_query",
        "lexical_normalization": "opk_rag.lexical.tokenizer.normalize_lexical_text",
        "query_input_template_version": search_config.query_template_version,
        "max_query_tokens": search_config.max_query_tokens,
        "query_text_publication": "digests_only",
    }
    sample_counts = _sample_counts(dataset_ids)
    gold_digests = {
        dataset_id: _sha256_file(GOLD_IDENTITY_PATHS[dataset_id])
        for dataset_id in dataset_ids
        if dataset_id in GOLD_IDENTITY_PATHS and GOLD_IDENTITY_PATHS[dataset_id].exists()
    }
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": BASELINE_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "dataset_ids": dataset_ids,
        "sample_counts": sample_counts,
        "corpus_snapshot_id": corpus.get("snapshot_id"),
        "corpus_snapshot_digest": _sha256_file(PUBLIC_CORPUS_SNAPSHOT_PATH),
        "indexed_corpus_contract_digest": corpus.get("indexed_contract_digest"),
        "public_corpus_private_manifest_digest": corpus.get("private_manifest_digest"),
        "canonical_identity_schema": CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION,
        "gold_identity_map_digests": gold_digests,
        "retrieval_modes": list(RETRIEVAL_MODES),
        "k_values": k_values,
        "production_top_k": search_config.top_k,
        "matching_levels": ["scope", "document"],
        "metrics": [
            "first_relevant_rank",
            "matched_required_units",
            "required_units_total",
            "recall_at_k",
            "mean_reciprocal_rank",
            "candidate_count",
            "filtered_candidate_count",
        ],
        "query_contract": query_payload,
        "query_normalization_digest": digest_json(query_payload),
        "filter_contract": {
            "knowledge_base_filter": "required",
            "document_status_filter": "indexed",
            "vector_index_configuration_filter": "embedding_fingerprint",
            "vector_null_embedding_filter": "inside public.match_chunks RPC",
            "lexical_readiness_filter": "indexed lexical chunk count must match indexed chunks",
            "private_path_publication": "forbidden",
        },
        "aggregation_contract": {
            "positive_denominator": "required evidence samples only",
            "required_evidence_unit_denominator": "deduplicated verifiable required evidence units per sample",
            "recall_unit_level": True,
            "mrr_sample_level": True,
            "hybrid_hurt_definition": "required evidence in lexical or vector at K but absent from hybrid at the same K",
        },
        "retrieval_configuration": config_payload,
        "retrieval_configuration_digest": digest_json(config_payload),
        "environment_sources": environment_sources or {},
        "privacy_policy": {
            "contains_sealed_holdout_data": False,
            "publishes_questions": False,
            "publishes_candidate_content": False,
            "publishes_private_paths": False,
            "query_publication": "sha256 digests and lengths only",
        },
    }


def build_pipeline_audit(search_config: VectorSearchConfig, embedding_config: EmbeddingConfig) -> list[dict[str, Any]]:
    return [
        {
            "stage": "raw user question",
            "implementation": "dataset question field passed unchanged to retrieval",
            "input": "sample.question",
            "output": "query string",
            "candidate_limit": None,
            "score_type": None,
            "normalization": "none",
            "filter": "dataset role allow-list",
            "stable_identity_available": False,
            "configuration_source": "public dataset loader",
        },
        {
            "stage": "query normalization",
            "implementation": "normalize_query for vector, normalize_lexical_text/tokenize_query for lexical",
            "input": "raw query",
            "output": "normalized query, lexical terms, embedding input digest",
            "candidate_limit": search_config.max_query_tokens,
            "score_type": None,
            "normalization": "newline normalization + strip; lexical lower/unicode/tokenizer normalization",
            "filter": "max_query_tokens",
            "stable_identity_available": False,
            "configuration_source": "VectorSearchConfig",
        },
        {
            "stage": "lexical candidates",
            "implementation": "BM25SearchRepository.search_chunks_by_bm25",
            "input": "jieba query terms",
            "output": "BM25SearchRow",
            "candidate_limit": search_config.bm25_candidate_k,
            "score_type": "bm25_score",
            "normalization": "JiebaLexicalTokenizer.tokenize_query",
            "filter": "knowledge_base_id, indexed lexical readiness/config fingerprint",
            "stable_identity_available": True,
            "configuration_source": "OPK_RAG_SEARCH_BM25_CANDIDATE_K, OPK_RAG_BM25_K1, OPK_RAG_BM25_B or defaults",
        },
        {
            "stage": "vector candidates",
            "implementation": "ChunkSearchRepository.search_chunks_by_vector via public.match_chunks RPC",
            "input": "Qwen query embedding",
            "output": "ChunkSearchRow",
            "candidate_limit": search_config.candidate_k,
            "score_type": "cosine similarity",
            "normalization": "prepare_query_input",
            "filter": "knowledge_base_id, index_configuration_id, RPC embedding availability",
            "stable_identity_available": True,
            "configuration_source": "EmbeddingConfig and VectorSearchConfig env/defaults",
        },
        {
            "stage": "hybrid fusion",
            "implementation": "opk_rag.search.service._rrf_fuse",
            "input": "thresholded vector rows + BM25 rows",
            "output": "RRF-ranked fused candidates",
            "candidate_limit": search_config.candidate_k + search_config.bm25_candidate_k,
            "score_type": "rrf_score",
            "normalization": "inherits lexical/vector query normalization",
            "filter": "min_similarity before fusion if configured",
            "stable_identity_available": True,
            "configuration_source": "OPK_RAG_RRF_K and OPK_RAG_RRF_*_WEIGHT or defaults",
        },
        {
            "stage": "deduplication/final candidates",
            "implementation": "_select_results uses chunk id, line range, content hash, overlap dedupe, final top_k",
            "input": "ranked candidates",
            "output": "SearchResponse.results",
            "candidate_limit": search_config.top_k,
            "score_type": "mode score after optional rerank",
            "normalization": "none",
            "filter": "deduplicate flag and context selection limits",
            "stable_identity_available": True,
            "configuration_source": "OPK_RAG_SEARCH_TOP_K, OPK_RAG_SEARCH_DEDUPLICATE, context env/defaults",
        },
    ]


def run_public_candidate_retrieval_baseline(
    *,
    connection,
    knowledge_base_id: UUID,
    provider,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    dataset_ids: list[str],
    modes: list[str],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_ids = _validate_dataset_ids(dataset_ids)
    modes = _validate_modes(modes)
    samples = _load_public_samples(dataset_ids)
    rows = [sample for sample in samples if sample["required_evidence"]]
    corpus_visibility = _load_corpus_visibility(connection, knowledge_base_id)
    traces: list[dict[str, Any]] = []
    for sample in rows:
        mode_candidates: dict[str, tuple[RankedCandidate, ...]] = {}
        for mode in modes:
            mode_candidates[mode] = capture_ranked_candidates(
                connection,
                knowledge_base_id=knowledge_base_id,
                query=sample["question"],
                mode=mode,
                provider=provider,
                embedding_config=embedding_config,
                search_config=search_config,
            )
        traces.append(score_sample_modes(sample, mode_candidates, contract["k_values"], corpus_visibility=corpus_visibility))
    summary = aggregate_retrieval_baseline(
        traces,
        contract=contract,
        sample_count=len(samples),
        required_evidence_sample_count=len(rows),
        modes=modes,
    )
    return summary, traces


def capture_ranked_candidates(
    connection,
    *,
    knowledge_base_id: UUID,
    query: str,
    mode: str,
    provider,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
) -> tuple[RankedCandidate, ...]:
    if KnowledgeBaseRepository(connection).get_by_id(knowledge_base_id) is None:
        raise CandidateRetrievalBaselineError(f"Knowledge base not found: {knowledge_base_id}")
    vector_rows: tuple[ChunkSearchRow, ...] = ()
    bm25_rows: tuple[BM25SearchRow, ...] = ()
    if mode in {"vector", "hybrid"}:
        if provider is None:
            raise CandidateRetrievalBaselineError(f"{mode} mode requires an embedding provider")
        index_configuration = IndexConfigurationRepository(connection).require_by_fingerprint(
            build_configuration_fingerprint(embedding_config)
        )
        prepared = prepare_query_input(
            query,
            embedding_config,
            count_tokens=provider.count_tokens,
            instruction=search_config.query_instruction,
            template_version=search_config.query_template_version,
            max_query_tokens=search_config.max_query_tokens,
        )
        vector = provider.embed_query(prepared.query, instruction=prepared.instruction)
        vector_rows = ChunkSearchRepository(connection).search_chunks_by_vector(
            query_embedding=vector,
            knowledge_base_id=knowledge_base_id,
            match_count=search_config.candidate_k,
            index_configuration_id=index_configuration.id,
        )
        sorted_rows = _stable_sort(_require_valid_similarity(vector_rows))
        vector_rows = (
            sorted_rows
            if search_config.min_similarity is None
            else tuple(row for row in sorted_rows if row.similarity >= search_config.min_similarity)
        )
    if mode in {"lexical", "hybrid"}:
        tokenizer = JiebaLexicalTokenizer()
        lexical_config = LexicalIndexConfig(bm25_k1=search_config.bm25_k1, bm25_b=search_config.bm25_b)
        lexical_fingerprint = build_lexical_configuration_fingerprint(lexical_config)
        readiness = LexicalIndexRepository(connection).get_lexical_readiness(
            knowledge_base_id=knowledge_base_id,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_version=tokenizer.version,
            configuration_fingerprint=lexical_fingerprint,
        )
        if not readiness.ready:
            raise CandidateRetrievalBaselineError("Lexical index is incomplete for this knowledge base.")
        bm25_rows = BM25SearchRepository(connection).search_chunks_by_bm25(
            query_terms=tokenizer.tokenize_query(query),
            knowledge_base_id=knowledge_base_id,
            match_count=search_config.bm25_candidate_k,
            k1=search_config.bm25_k1,
            b=search_config.bm25_b,
            configuration_fingerprint=lexical_fingerprint,
        )
    if mode == "vector":
        results = tuple(_to_vector_result(index + 1, row, vector_rank=index + 1) for index, row in enumerate(vector_rows))
    elif mode == "lexical":
        results = tuple(_to_bm25_result(index + 1, row, bm25_rank=index + 1) for index, row in enumerate(bm25_rows))
    elif mode == "hybrid":
        results = tuple(_fused_to_result(rank=index + 1, item=item) for index, item in enumerate(_rrf_fuse(vector_rows, bm25_rows, search_config)))
    else:
        raise CandidateRetrievalBaselineError(f"unknown retrieval mode: {mode}")
    return tuple(_candidate_from_result(result, mode=mode) for result in results)


def score_sample_modes(
    sample: dict[str, Any],
    mode_candidates: dict[str, tuple[RankedCandidate, ...]],
    k_values: list[int],
    *,
    corpus_visibility: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = list(deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in sample["required_evidence"]))
    unit_rows = []
    for index, gold in enumerate(required, start=1):
        per_mode = {}
        for mode, candidates in mode_candidates.items():
            per_mode[mode] = _unit_match_profile(gold, candidates, k_values)
        primary, secondary = _attribute_unit(per_mode, gold, corpus_visibility, k_values)
        unit_rows.append(
            {
                "unit_index": index,
                "gold_identity_digest": digest_json(gold.to_json()),
                "granularity": gold.granularity,
                "lexical_rank": per_mode.get("lexical", {}).get("scope_rank"),
                "vector_rank": per_mode.get("vector", {}).get("scope_rank"),
                "hybrid_rank": per_mode.get("hybrid", {}).get("scope_rank"),
                "lexical_document_rank": per_mode.get("lexical", {}).get("document_rank"),
                "vector_document_rank": per_mode.get("vector", {}).get("document_rank"),
                "hybrid_document_rank": per_mode.get("hybrid", {}).get("document_rank"),
                "primary_attribution": primary,
                "secondary_attributions": secondary,
                "filter_status": _filter_status(gold, corpus_visibility),
            }
        )
    query = sample["question"]
    normalized = normalize_query(query)
    lexical_normalized = normalize_lexical_text(query).strip()
    return {
        "sample_id": sample["sample_id"],
        "dataset_id": sample["dataset_id"],
        "dataset_role": sample["dataset_role"],
        "question_type": sample.get("question_type"),
        "capability": _capability_label(sample),
        "query_pattern": _query_pattern(query),
        "query_contract": _query_contract(query),
        "normalization_audit": {
            "normalization_changed": normalized != query or lexical_normalized != query,
            "vector_normalization_changed": normalized != query,
            "lexical_normalization_changed": lexical_normalized != query,
            "critical_token_change": _critical_token_changed(query, normalized, lexical_normalized),
        },
        "required_units_total": len(required),
        "modes": {
            mode: _score_mode(candidates, required, k_values)
            for mode, candidates in mode_candidates.items()
        },
        "required_units": unit_rows,
        "candidate_counts": {mode: len(candidates) for mode, candidates in mode_candidates.items()},
    }


def aggregate_retrieval_baseline(
    traces: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    sample_count: int,
    required_evidence_sample_count: int,
    modes: list[str],
) -> dict[str, Any]:
    k_values = [int(k) for k in contract["k_values"]]
    required_units = sum(row["required_units_total"] for row in traces)
    mode_summary = {mode: _aggregate_mode(traces, mode, k_values) for mode in modes}
    overlap = _route_overlap(traces, k_values)
    fusion = _fusion_analysis(traces, contract["production_top_k"], k_values)
    failure = Counter(unit["primary_attribution"] for row in traces for unit in row["required_units"])
    capability_metrics = _aggregate_by_dimension(traces, "capability", modes, k_values)
    question_type_metrics = _aggregate_by_dimension(traces, "question_type", modes, k_values)
    query_pattern_metrics = _aggregate_by_dimension(traces, "query_pattern", modes, k_values)
    reranker = _reranker_assessment(overlap, fusion, required_units)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "status": "completed",
        "contract_id": contract["contract_id"],
        "git_commit": contract["git_commit"],
        "created_at": utc_now(),
        "datasets": contract["dataset_ids"],
        "sample_count": sample_count,
        "required_evidence_sample_count": required_evidence_sample_count,
        "required_evidence_unit_count": required_units,
        "k_values": k_values,
        "production_top_k": contract["production_top_k"],
        "modes": mode_summary,
        "route_overlap": overlap,
        "fusion_analysis": fusion,
        "failure_attribution": dict(sorted(failure.items())),
        "capability_metrics": capability_metrics,
        "question_type_metrics": question_type_metrics,
        "query_pattern_metrics": query_pattern_metrics,
        "query_normalization_audit": _aggregate_query_normalization(traces),
        "metadata_filter_audit": _aggregate_filter_status(traces),
        "pipeline_audit": contract.get("pipeline_audit") or [],
        "reranker_assessment": reranker,
        "next_task_recommendation": _next_task_recommendation(mode_summary, overlap, fusion, reranker),
        "model_calls": {"embedding_only": True, "deepseek": False, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "contains_sealed_holdout_data": False,
        "rank_trace_path": RANK_TRACE_PATH.relative_to(ROOT).as_posix(),
    }


def write_contract(contract: dict[str, Any], path: Path = CONTRACT_PATH) -> dict[str, Any]:
    write_json(path, contract)
    return contract


def write_baseline(summary: dict[str, Any], traces: list[dict[str, Any]], *, result_path: Path = RESULT_PATH, trace_path: Path = RANK_TRACE_PATH) -> dict[str, Any]:
    write_json(result_path, summary)
    write_jsonl(trace_path, _redacted_traces(traces))
    return summary


def verify_baseline(contract_path: Path = CONTRACT_PATH, result_path: Path = RESULT_PATH) -> dict[str, Any]:
    issues = []
    if not contract_path.exists():
        issues.append({"code": "missing_contract", "path": _display_path(contract_path)})
    if not result_path.exists():
        issues.append({"code": "missing_result", "path": _display_path(result_path)})
    contract = read_json(contract_path) if contract_path.exists() else {}
    result = read_json(result_path) if result_path.exists() else {}
    if contract.get("contract_id") != BASELINE_ID:
        issues.append({"code": "unexpected_contract_id"})
    if result.get("baseline_id") != BASELINE_ID:
        issues.append({"code": "unexpected_baseline_id"})
    if result.get("contains_sealed_holdout_data") is not False:
        issues.append({"code": "sealed_holdout_flag_not_false"})
    if result.get("model_calls", {}).get("deepseek") is not False:
        issues.append({"code": "deepseek_call_flag_not_false"})
    if result.get("writes_database") is not False or result.get("writes_index") is not False:
        issues.append({"code": "write_flags_not_false"})
    _reject_forbidden_payload(contract)
    _reject_forbidden_payload(result)
    return {"status": "valid" if not issues else "invalid", "issues": issues}


def compare_reproducibility(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_traces = left.get("traces") or []
    right_traces = right.get("traces") or []
    if len(left_traces) != len(right_traces):
        return {"candidate_identity_agreement": 0.0, "candidate_rank_agreement": 0.0, "metric_agreement": 0.0}
    total = rank_same = identity_same = 0
    for a, b in zip(left_traces, right_traces, strict=True):
        for mode in set((a.get("modes") or {}).keys()) | set((b.get("modes") or {}).keys()):
            ar = (a.get("modes") or {}).get(mode, {})
            br = (b.get("modes") or {}).get(mode, {})
            total += 1
            identity_same += ar.get("candidate_identity_digests") == br.get("candidate_identity_digests")
            rank_same += ar.get("candidate_identity_digests") == br.get("candidate_identity_digests")
    metric_agreement = 1.0 if left.get("summary", {}).get("modes") == right.get("summary", {}).get("modes") else 0.0
    return {
        "candidate_identity_agreement": (identity_same / total) if total else 1.0,
        "candidate_rank_agreement": (rank_same / total) if total else 1.0,
        "metric_agreement": metric_agreement,
    }


def update_benchmark_registry(summary: dict[str, Any], contract: dict[str, Any], path: Path = BENCHMARK_REGISTRY_PATH) -> dict[str, Any]:
    registry = read_json(path)
    benchmarks = list(registry.get("benchmarks") or [])
    row = {
        "artifact_id": BASELINE_ID,
        "artifact_type": "evaluation_result",
        "role": "development_retrieval_baseline",
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "dataset_ids": contract["dataset_ids"],
        "sample_count": summary["sample_count"],
        "required_evidence_sample_count": summary["required_evidence_sample_count"],
        "required_evidence_unit_count": summary["required_evidence_unit_count"],
        "gold_identity_digests": contract["gold_identity_map_digests"],
        "retrieval_config_digest": contract["retrieval_configuration_digest"],
        "corpus_snapshot": contract["corpus_snapshot_id"],
        "indexed_corpus_contract": contract["indexed_corpus_contract_digest"],
        "sealed_holdout_access": False,
        "model_calls": "embedding_only",
        "deepseek_calls": False,
        "intended_use": "diagnose public candidate retrieval recall gaps across lexical, vector, and hybrid modes",
        "forbidden_use": "sealed holdout tuning, retrieval parameter tuning, prompt tuning, model comparison",
        "registered_at": utc_now(),
    }
    _reject_forbidden_payload(row)
    benchmarks = [item for item in benchmarks if item.get("artifact_id") != BASELINE_ID]
    benchmarks.append(row)
    registry["benchmarks"] = benchmarks
    registry["generated_at"] = utc_now()
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def _candidate_from_result(result, *, mode: str) -> RankedCandidate:
    identity = identity_from_runtime_evidence_item(
        {
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "relative_path": result.relative_path,
            "heading_path": result.heading_path,
            "content": result.content,
            "start_line": result.start_line,
            "end_line": result.end_line,
        }
    ).to_json()
    source = "both" if result.vector_rank is not None and result.bm25_rank is not None else "vector_only" if result.vector_rank is not None else "lexical_only"
    return RankedCandidate(
        rank=result.rank,
        identity=identity,
        document_identity_digest=identity.get("document_identity_digest"),
        scope_identity_digest=identity.get("scope_identity_digest"),
        source=source,
        lexical_rank=result.bm25_rank,
        vector_rank=result.vector_rank,
        hybrid_rank=result.rank if mode == "hybrid" else None,
        fusion_score=result.rrf_score,
        score_type="rrf_score" if mode == "hybrid" else "bm25_score" if mode == "lexical" else "cosine_similarity",
        score=result.rrf_score if mode == "hybrid" else result.bm25_score if mode == "lexical" else result.vector_similarity,
    )


def _unit_match_profile(gold: EvidenceIdentity, candidates: tuple[RankedCandidate, ...], k_values: list[int]) -> dict[str, Any]:
    scope_rank = None
    document_rank = None
    scope_hits = {str(k): False for k in k_values}
    document_hits = {str(k): False for k in k_values}
    for candidate in candidates:
        runtime = normalize_runtime_evidence_identity(candidate.identity)
        match = match_evidence_identity(gold, runtime)
        doc_match = _document_match(gold, runtime, match)
        scope_match = _scope_match(match)
        if scope_match and scope_rank is None:
            scope_rank = candidate.rank
        if doc_match and document_rank is None:
            document_rank = candidate.rank
        for k in k_values:
            if candidate.rank <= k and scope_match:
                scope_hits[str(k)] = True
            if candidate.rank <= k and doc_match:
                document_hits[str(k)] = True
    return {
        "scope_rank": scope_rank,
        "document_rank": document_rank,
        "scope_hit_at": scope_hits,
        "document_hit_at": document_hits,
    }


def _score_mode(candidates: tuple[RankedCandidate, ...], required: list[EvidenceIdentity], k_values: list[int]) -> dict[str, Any]:
    profiles = [_unit_match_profile(gold, candidates, k_values) for gold in required]
    denominator = len(required)
    first_rank = min((profile["scope_rank"] for profile in profiles if profile["scope_rank"] is not None), default=None)
    return {
        "candidate_count": len(candidates),
        "filtered_candidate_count": 0,
        "first_relevant_rank": first_rank,
        "matched_required_units": sum(1 for profile in profiles if profile["scope_rank"] is not None),
        "required_units_total": denominator,
        "mean_reciprocal_rank": (1 / first_rank) if first_rank else 0.0,
        "scope_recall_at_k": {
            str(k): (sum(1 for profile in profiles if profile["scope_hit_at"][str(k)]) / denominator) if denominator else None
            for k in k_values
        },
        "document_recall_at_k": {
            str(k): (sum(1 for profile in profiles if profile["document_hit_at"][str(k)]) / denominator) if denominator else None
            for k in k_values
        },
        "candidate_identity_digests": [digest_json(candidate.identity) for candidate in candidates],
    }


def _aggregate_mode(traces: list[dict[str, Any]], mode: str, k_values: list[int]) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in traces)
    mode_rows = [row["modes"].get(mode, {}) for row in traces]
    return {
        "sample_count": len(mode_rows),
        "candidate_count_mean": (sum(row.get("candidate_count", 0) for row in mode_rows) / len(mode_rows)) if mode_rows else 0.0,
        "first_relevant_rank_mean": _mean(row.get("first_relevant_rank") for row in mode_rows if row.get("first_relevant_rank")),
        "mean_reciprocal_rank": _mean(row.get("mean_reciprocal_rank", 0.0) for row in mode_rows),
        "scope_recall_at_k": {
            str(k): (sum(row.get("scope_recall_at_k", {}).get(str(k), 0.0) * trace["required_units_total"] for row, trace in zip(mode_rows, traces, strict=True)) / denominator) if denominator else None
            for k in k_values
        },
        "document_recall_at_k": {
            str(k): (sum(row.get("document_recall_at_k", {}).get(str(k), 0.0) * trace["required_units_total"] for row, trace in zip(mode_rows, traces, strict=True)) / denominator) if denominator else None
            for k in k_values
        },
    }


def _route_overlap(traces: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    k = str(max(k_values))
    counts = Counter()
    sample_counts = Counter()
    for row in traces:
        helped_lexical = helped_vector = helped_hybrid = hurt_hybrid = False
        for unit in row["required_units"]:
            lex = unit.get("lexical_rank") is not None and unit["lexical_rank"] <= int(k)
            vec = unit.get("vector_rank") is not None and unit["vector_rank"] <= int(k)
            hyb = unit.get("hybrid_rank") is not None and unit["hybrid_rank"] <= int(k)
            if lex and vec:
                counts["required_units_found_by_both"] += 1
            elif lex:
                counts["required_units_found_by_lexical_only"] += 1
            elif vec:
                counts["required_units_found_by_vector_only"] += 1
            else:
                counts["required_units_found_by_neither"] += 1
            helped_lexical |= lex
            helped_vector |= vec
            helped_hybrid |= hyb
            hurt_hybrid |= (lex or vec) and not hyb
        sample_counts["samples_helped_by_lexical"] += bool(helped_lexical)
        sample_counts["samples_helped_by_vector"] += bool(helped_vector)
        sample_counts["samples_helped_by_hybrid"] += bool(helped_hybrid)
        sample_counts["samples_hurt_by_hybrid"] += bool(hurt_hybrid)
    return {**dict(counts), **dict(sample_counts), "analysis_k": int(k)}


def _fusion_analysis(traces: list[dict[str, Any]], production_top_k: int, k_values: list[int]) -> dict[str, Any]:
    max_k = max(k_values)
    counts = Counter()
    for row in traces:
        for unit in row["required_units"]:
            single_rank = min([rank for rank in (unit.get("lexical_rank"), unit.get("vector_rank")) if rank is not None], default=None)
            hybrid_rank = unit.get("hybrid_rank")
            if single_rank is not None and single_rank <= max_k and hybrid_rank is None:
                counts["single_route_hit_but_hybrid_miss"] += 1
            if single_rank is not None and hybrid_rank is not None:
                if hybrid_rank > single_rank:
                    counts["hybrid_rank_degradation"] += 1
                elif hybrid_rank < single_rank:
                    counts["hybrid_rank_improvement"] += 1
            if hybrid_rank is not None and hybrid_rank > production_top_k:
                counts["hybrid_hit_beyond_production_k"] += 1
            if single_rank is not None and single_rank <= max_k and (hybrid_rank is None or hybrid_rank > production_top_k):
                counts["hybrid_candidate_budget_overflow"] += 1
    return {
        **dict(counts),
        "fusion_score_collision": "not_separately_measurable",
        "duplicate_merge_effect": "not_separately_measurable",
        "analysis_k": max_k,
    }


def _aggregate_by_dimension(traces: list[dict[str, Any]], dimension: str, modes: list[str], k_values: list[int]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        groups[str(row.get(dimension) or "unknown")].append(row)
    out = {}
    for key, rows in sorted(groups.items()):
        out[key] = {
            "sample_count": len(rows),
            "sample_size_status": "ok" if len(rows) >= 3 else "insufficient_sample_size",
            "modes": {mode: _aggregate_mode(rows, mode, k_values) for mode in modes},
        }
    return out


def _attribute_unit(
    per_mode: dict[str, dict[str, Any]],
    gold: EvidenceIdentity,
    corpus_visibility: dict[str, dict[str, Any]],
    k_values: list[int],
) -> tuple[str, list[str]]:
    max_k = max(k_values)
    lex = (per_mode.get("lexical") or {}).get("scope_rank")
    vec = (per_mode.get("vector") or {}).get("scope_rank")
    hyb = (per_mode.get("hybrid") or {}).get("scope_rank")
    secondary = []
    if _filter_status(gold, corpus_visibility) == "filter_exclusion_confirmed":
        return "metadata_filter_exclusion", []
    if lex and vec:
        primary = "lexical_hit_vector_hit"
    elif lex:
        primary = "lexical_hit_vector_miss"
    elif vec:
        primary = "lexical_miss_vector_hit"
    else:
        primary = "both_routes_miss"
    if (lex or vec) and not hyb:
        secondary.append("single_route_hit_hybrid_miss")
    if hyb and hyb > max_k:
        secondary.append("hybrid_hit_beyond_production_k")
    return primary, secondary


def _scope_match(match) -> bool:
    return match.matched and match.level in {
        MatchLevel.EXACT_CHUNK,
        MatchLevel.EXACT_SCOPE,
        MatchLevel.COMPATIBLE_SCOPE_CHUNK,
    }


def _document_match(gold: EvidenceIdentity, runtime: EvidenceIdentity, match) -> bool:
    if _scope_match(match) or match.level in {
        MatchLevel.EXACT_DOCUMENT,
        MatchLevel.COMPATIBLE_DOCUMENT_CHUNK,
        MatchLevel.SOURCE_DIGEST,
    }:
        return bool(match.matched)
    if gold.document_identity_digest and runtime.document_identity_digest:
        return gold.document_identity_digest == runtime.document_identity_digest
    if gold.relative_path_digest and runtime.relative_path_digest:
        return gold.relative_path_digest == runtime.relative_path_digest
    if gold.source_digest and runtime.source_digest:
        return gold.source_digest == runtime.source_digest
    return False


def _load_public_samples(dataset_ids: list[str]) -> list[dict[str, Any]]:
    samples = []
    for dataset_id in dataset_ids:
        diagnostic_rows = load_diagnostic_dataset(dataset_id)
        source_rows = _source_rows_for_dataset(dataset_id)
        for index, row in enumerate(diagnostic_rows):
            source = source_rows[index] if index < len(source_rows) else {}
            samples.append(
                {
                    "sample_id": f"{dataset_id}:{row['sample_id']}:{index + 1:03d}",
                    "source_sample_id": row["sample_id"],
                    "dataset_id": dataset_id,
                    "dataset_role": row["dataset_role"],
                    "question_type": row.get("question_type"),
                    "question": source.get("question") or row.get("question") or "",
                    "expected_action": source.get("expected_action"),
                    "required_evidence": row.get("required_evidence") or [],
                }
            )
    missing = [row["sample_id"] for row in samples if not row["question"]]
    if missing:
        raise CandidateRetrievalBaselineError(f"missing public questions for samples: {missing[:5]}")
    return samples


def _source_rows_for_dataset(dataset_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in DATASET_SOURCE_PATHS[dataset_id]:
        rows.extend(read_jsonl(path))
    return rows[: len(load_diagnostic_dataset(dataset_id))]


def _sample_counts(dataset_ids: list[str]) -> dict[str, Any]:
    counts = {}
    for dataset_id in dataset_ids:
        rows = load_diagnostic_dataset(dataset_id)
        unit_counts = [
            len(deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in (row.get("required_evidence") or [])))
            for row in rows
        ]
        counts[dataset_id] = {
            "sample_count": len(rows),
            "required_evidence_sample_count": sum(1 for row in rows if row.get("required_evidence")),
            "required_evidence_unit_count": sum(unit_counts),
        }
    return counts


def _retrieval_config_payload(search_config: VectorSearchConfig, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    return {
        "search": asdict(search_config),
        "embedding": {
            "provider": embedding_config.provider,
            "model_name": embedding_config.model_name,
            "model_revision": embedding_config.model_revision,
            "dimension": embedding_config.dimension,
            "normalize": embedding_config.normalize,
            "distance_metric": embedding_config.distance_metric,
            "input_template_version": embedding_config.input_template_version,
            "fingerprint": build_configuration_fingerprint(embedding_config),
        },
        "lexical": {
            "backend": "PostgreSQL BM25 tables/RPC with Jieba tokenizer",
            "tokenizer_id": JiebaLexicalTokenizer().tokenizer_id,
            "tokenizer_version": JiebaLexicalTokenizer().version,
            "configuration_fingerprint": build_lexical_configuration_fingerprint(
                LexicalIndexConfig(bm25_k1=search_config.bm25_k1, bm25_b=search_config.bm25_b)
            ),
        },
        "hybrid": {
            "algorithm": "reciprocal_rank_fusion",
            "rrf_k": search_config.rrf_k,
            "vector_weight": search_config.rrf_vector_weight,
            "bm25_weight": search_config.rrf_bm25_weight,
            "weighted_scores": False,
        },
    }


def _query_contract(query: str) -> dict[str, Any]:
    normalized = normalize_query(query)
    lexical_normalized = normalize_lexical_text(query).strip()
    token_payload = JiebaLexicalTokenizer().tokenize_query(query)
    return {
        "raw_query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "normalized_query_digest": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "lexical_normalized_query_digest": hashlib.sha256(lexical_normalized.encode("utf-8")).hexdigest(),
        "token_digest": digest_json(list(token_payload)),
        "embedding_input_digest": hashlib.sha256(render_query_input(normalized).encode("utf-8")).hexdigest(),
        "query_length": len(query),
        "normalization_changed": normalized != query or lexical_normalized != query,
    }


def _critical_token_changed(raw: str, normalized: str, lexical_normalized: str) -> bool:
    markers = ("TASK-", "OPK_RAG_", ".md", ".toml", "/", "_", "-", "`")
    return any(marker in raw and marker not in normalized for marker in markers) or any(marker in raw and marker.lower() not in lexical_normalized for marker in markers)


def _query_pattern(query: str) -> str:
    lowered = query.lower()
    if "task-" in lowered or "task" in lowered and any(char.isdigit() for char in lowered):
        return "task_number"
    if "opk_rag_" in lowered or "uv_" in lowered or "=" in query:
        return "configuration_key"
    if "/" in query or ".md" in lowered or ".toml" in lowered:
        return "path_or_filename"
    if any(ch.isdigit() for ch in query) and any(marker in query for marker in ("-", "_", ".")):
        return "exact_identifier"
    if any(term in query for term in ("对比", "区别", "相比")):
        return "conceptual_paraphrase"
    if any(term in query for term in ("跨", "之间", "关系")):
        return "cross_document"
    if any(term in query for term in ("之前", "后来", "多久", "什么时候")):
        return "temporal"
    if any(term in query for term in ("冲突", "矛盾", "不一致")):
        return "conflict"
    return "natural_language_fact"


def _capability_label(sample: dict[str, Any]) -> str:
    question_type = str(sample.get("question_type") or "")
    pattern = _query_pattern(sample["question"])
    if question_type == "false_premise":
        return "false_assumption"
    if pattern == "configuration_key":
        return "configuration_lookup"
    if pattern == "path_or_filename":
        return "source_location"
    if pattern == "cross_document":
        return "cross_document_synthesis"
    if pattern == "temporal":
        return "temporal_evolution"
    if pattern == "conceptual_paraphrase":
        return "concept_relationship"
    if pattern == "conflict":
        return "conflict_detection"
    if question_type == "partially_answerable":
        return "partial_information"
    return "exact_fact_lookup"


def _load_corpus_visibility(connection, knowledge_base_id: UUID) -> dict[str, dict[str, Any]]:
    visibility = {}
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select d.id, c.id, d.relative_path, d.index_status, c.heading_path, c.content, c.start_line, c.end_line
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s
            """,
            (knowledge_base_id,),
        )
        for row in cursor.fetchall():
            payload = identity_from_runtime_evidence_item(
                {
                    "document_id": row[0],
                    "chunk_id": row[1],
                    "relative_path": row[2],
                    "heading_path": row[4],
                    "content": row[5],
                    "start_line": row[6],
                    "end_line": row[7],
                },
                knowledge_base_id=knowledge_base_id,
            ).to_json()
            for key in ("document_identity_digest", "relative_path_digest", "source_digest", "scope_identity_digest", "chunk_content_digest"):
                if payload.get(key):
                    visibility.setdefault(f"{key}:{payload[key]}", {"document_status": row[3], "visible": row[3] == "indexed"})
    return visibility


def _filter_status(gold: EvidenceIdentity, corpus_visibility: dict[str, dict[str, Any]]) -> str:
    keys = []
    for field in ("scope_identity_digest", "chunk_content_digest", "document_identity_digest", "relative_path_digest", "source_digest"):
        value = getattr(gold, field)
        if value:
            keys.append(f"{field}:{value}")
    if not keys:
        return "filter_exclusion_unverifiable"
    observed = [corpus_visibility.get(key) for key in keys if corpus_visibility.get(key)]
    if not observed:
        return "filter_exclusion_unverifiable"
    if any(item.get("visible") for item in observed):
        return "filter_exclusion_not_observed"
    return "filter_exclusion_confirmed"


def _aggregate_filter_status(traces: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(unit.get("filter_status") for row in traces for unit in row["required_units"]))


def _aggregate_query_normalization(traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(traces),
        "normalization_changed_count": sum(row["normalization_audit"]["normalization_changed"] for row in traces),
        "critical_token_change_count": sum(row["normalization_audit"]["critical_token_change"] for row in traces),
    }


def _reranker_assessment(overlap: dict[str, Any], fusion: dict[str, Any], required_units: int) -> str:
    beyond = int(fusion.get("hybrid_hit_beyond_production_k") or 0)
    neither = int(overlap.get("required_units_found_by_neither") or 0)
    if required_units == 0:
        return "insufficient_evidence"
    if beyond / required_units >= 0.2:
        return "reranker_strongly_justified"
    if beyond:
        return "reranker_candidate"
    if neither / required_units >= 0.5:
        return "reranker_not_justified"
    return "insufficient_evidence"


def _next_task_recommendation(mode_summary: dict[str, Any], overlap: dict[str, Any], fusion: dict[str, Any], reranker: str) -> str:
    if overlap.get("required_units_found_by_neither", 0) > max(overlap.get("required_units_found_by_lexical_only", 0), overlap.get("required_units_found_by_vector_only", 0)):
        return "repair query expression and single-route recall before reranking"
    if fusion.get("single_route_hit_but_hybrid_miss", 0):
        return "diagnose RRF fusion and final candidate budget"
    if reranker in {"reranker_candidate", "reranker_strongly_justified"}:
        return "design reranker experiment on public candidate top20"
    return "inspect lexical/vector route-specific misses"


def _redacted_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in traces:
        for unit in row["required_units"]:
            for mode in RETRIEVAL_MODES:
                rank = unit.get(f"{mode}_rank")
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "evidence_identity_digest": unit["gold_identity_digest"],
                        "route": mode,
                        "rank": rank,
                        "score": None,
                        "match_level": "scope" if rank is not None else None,
                        "match_status": "matched_within_top20" if rank is not None else "not_found_within_top20",
                        "failure_attribution": unit.get("primary_attribution"),
                    }
                )
    return rows


def _validate_dataset_ids(dataset_ids: list[str]) -> list[str]:
    ids = [item.strip() for item in dataset_ids if item.strip()]
    if not ids:
        raise CandidateRetrievalBaselineError("at least one dataset is required")
    if any(item in FORBIDDEN_DATASETS for item in ids):
        raise CandidateRetrievalBaselineError("sealed holdout datasets are not allowed")
    unknown = [item for item in ids if item not in DATASET_SOURCE_PATHS]
    if unknown:
        raise CandidateRetrievalBaselineError(f"unknown public dataset: {unknown[0]}")
    return ids


def _validate_modes(modes: list[str]) -> list[str]:
    values = [item.strip() for item in modes if item.strip()]
    unknown = [item for item in values if item not in RETRIEVAL_MODES]
    if unknown:
        raise CandidateRetrievalBaselineError(f"unknown retrieval mode: {unknown[0]}")
    return values


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return (sum(values) / len(values)) if values else None


def _sha256_file(path: Path) -> str:
    _reject_forbidden_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _reject_forbidden_path(path: Path) -> None:
    text = path.as_posix()
    if "phase2_holdout_v1" in text or "phase2_sealed_baseline_v1" in text:
        raise CandidateRetrievalBaselineError(f"refusing sealed holdout path: {text}")


def _reject_forbidden_payload(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for marker in ("phase2_holdout_v1", "phase2_sealed_baseline_v1"):
        if marker in text:
            raise CandidateRetrievalBaselineError(f"payload contains forbidden sealed holdout marker: {marker}")
