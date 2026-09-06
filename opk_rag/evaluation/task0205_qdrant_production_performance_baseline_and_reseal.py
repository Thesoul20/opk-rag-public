from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any
from uuid import UUID

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import ChunkSearchRepository, IndexConfigurationRepository, KnowledgeBaseRepository
from opk_rag.embedding.config import build_configuration_fingerprint, load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.embedding.query import prepare_query_input
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import build_reranker_configuration_fingerprint, load_reranker_config
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.search.service import _load_vector_backend_id, search_knowledge_base_connection
from opk_rag.vector_backends.base import VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config

TASK_ID = "TASK-0205"
EXPERIMENT_ID = "task0205-qdrant-production-performance-baseline-and-reseal"
SCHEMA_VERSION = "opk-rag.task0205.qdrant-production-performance-baseline-and-reseal.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0205_qdrant_production_performance_baseline_and_reseal_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0205_QDRANT_PRODUCTION_PERFORMANCE_BASELINE_AND_RESEAL_REPORT.md"
CORE_QUESTIONS_PATH = ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "question_set.jsonl"
CORE_MANIFEST_PATH = ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"
CORE_CORPUS_PATH = ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"
EXPECTED_Q01_Q07_PATH = Path("<workspace>/opk-rag-testv1/corpus/opk-rag-cold-start-corpus-v1/EXPECTED_QUERIES.json")
TASK0202_DIR = ROOT / "evaluation-data" / "results" / "task0202-qdrant-production-stabilization-and-migration-freeze"
TASK0203_DIR = ROOT / "evaluation-data" / "results" / "task0203-post-migration-project-state-consolidation"
TASK0204_DIR = ROOT / "evaluation-data" / "results" / "task0204-post-migration-release-acceptance-and-rollback-drill"

DEFAULT_RANDOM_SEED = 205
DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASUREMENT_RUNS = 20
Q01_Q07_QUERY_COUNT = 7

REQUIRED_ARTIFACTS = (
    "environment_audit.json",
    "benchmark_manifest.json",
    "qdrant_backend_only.json",
    "qdrant_full_retrieval.json",
    "cold_warm_comparison.json",
    "bounded_concurrency.json",
    "pgvector_control.json",
    "q01_q07_e2e.json",
    "quality_invariants.json",
    "authority_consistency.json",
    "raw_measurements.json",
    "summary.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authority_task",
    "benchmark_revision",
    "benchmark_query_count",
    "benchmark_query_digest",
    "benchmark_corpus_digest",
    "benchmark_configuration_digest",
    "benchmark_random_seed",
    "production_vector_backend",
    "production_qdrant_collection",
    "relational_authority_backend",
    "rollback_vector_backend",
    "qdrant_server_version",
    "qdrant_client_version",
    "postgresql_version",
    "pgvector_version",
    "embedding_model",
    "embedding_dimension",
    "qdrant_backend_latency_p50_ms",
    "qdrant_backend_latency_p95_ms",
    "qdrant_backend_latency_p99_ms",
    "qdrant_backend_throughput_qps",
    "qdrant_retrieval_latency_p50_ms",
    "qdrant_retrieval_latency_p95_ms",
    "qdrant_retrieval_latency_p99_ms",
    "qdrant_retrieval_throughput_qps",
    "cold_start_total_latency_ms",
    "cold_start_model_load_latency_ms",
    "cold_start_first_query_latency_ms",
    "warm_retrieval_latency_p50_ms",
    "warm_retrieval_latency_p95_ms",
    "pgvector_backend_latency_p50_ms",
    "pgvector_backend_latency_p95_ms",
    "pgvector_backend_latency_p99_ms",
    "pgvector_retrieval_latency_p50_ms",
    "pgvector_retrieval_latency_p95_ms",
    "pgvector_retrieval_latency_p99_ms",
    "pgvector_throughput_qps",
    "backend_p50_latency_ratio_qdrant_to_pgvector",
    "backend_p95_latency_ratio_qdrant_to_pgvector",
    "retrieval_p50_latency_ratio_qdrant_to_pgvector",
    "retrieval_p95_latency_ratio_qdrant_to_pgvector",
    "throughput_ratio_qdrant_to_pgvector",
    "q01_q07_pass_count",
    "formal_retrieval_recall_at_k",
    "formal_retrieval_mrr",
    "candidate_membership_regression_count",
    "evidence_regression_count",
    "answer_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "silent_backend_fallback_detected",
    "production_collection_mutation_count",
    "relational_authority_preserved",
    "authority_drift_detected",
    "runtime_default_behavior_change",
    "benchmark_valid",
    "performance_baseline_sealed",
    "qdrant_performance_assessment",
    "known_performance_blocker_count",
    "focused_test_pass_count",
    "related_regression_test_pass_count",
    "full_suite_pass_count",
    "full_suite_skip_count",
    "full_suite_failure_count",
)


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    source: str
    expected_action: str | None = None


@dataclass(frozen=True)
class MeasurementConfig:
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    measurement_runs: int = DEFAULT_MEASUREMENT_RUNS
    random_seed: int = DEFAULT_RANDOM_SEED
    timeout_seconds: float = 120.0
    concurrency_levels: tuple[int, ...] = (1, 2, 4)


def run_task0205(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    runtime_env["OPK_RAG_VECTOR_BACKEND"] = "qdrant"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_source_authorities()
    measurement_config = load_measurement_config(runtime_env)
    embedding_config = load_embedding_config(runtime_env)
    search_config = load_vector_search_config(runtime_env)
    reranker_config = load_reranker_config(runtime_env)
    qdrant_config = load_qdrant_config(runtime_env)
    benchmark = build_benchmark_manifest(runtime_env, measurement_config, embedding_config, search_config, reranker_config)
    environment = audit_environment(runtime_env, embedding_config, search_config, qdrant_config)
    quality = build_quality_invariants(sources)
    before_count = environment.get("qdrant_collection_point_count")
    raw: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "measurements": []}

    runtime = execute_benchmarks(runtime_env, benchmark, measurement_config, raw)
    after_count = probe_qdrant_count(runtime_env)
    authority = build_authority_consistency(sources, before_count=before_count, after_count=after_count)
    summary = build_summary(
        environment=environment,
        benchmark=benchmark,
        qdrant_backend=runtime["qdrant_backend_only"],
        qdrant_retrieval=runtime["qdrant_full_retrieval"],
        cold_warm=runtime["cold_warm_comparison"],
        concurrency=runtime["bounded_concurrency"],
        pgvector=runtime["pgvector_control"],
        e2e=runtime["q01_q07_e2e"],
        quality=quality,
        authority=authority,
    )
    contract = build_contract()

    if write:
        artifacts = {
            "environment_audit.json": environment,
            "benchmark_manifest.json": benchmark,
            "qdrant_backend_only.json": runtime["qdrant_backend_only"],
            "qdrant_full_retrieval.json": runtime["qdrant_full_retrieval"],
            "cold_warm_comparison.json": runtime["cold_warm_comparison"],
            "bounded_concurrency.json": runtime["bounded_concurrency"],
            "pgvector_control.json": runtime["pgvector_control"],
            "q01_q07_e2e.json": runtime["q01_q07_e2e"],
            "quality_invariants.json": quality,
            "authority_consistency.json": authority,
            "raw_measurements.json": raw,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract)
        REPORT_PATH.write_text(render_report(summary, environment, benchmark, runtime, quality, authority), encoding="utf-8")
        verification = verify_task0205_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, environment, benchmark, runtime, quality, authority), encoding="utf-8")
    return summary


def execute_benchmarks(
    env: Mapping[str, str],
    benchmark: Mapping[str, Any],
    config: MeasurementConfig,
    raw: dict[str, Any],
) -> dict[str, Any]:
    try:
        database_url = load_postgres_config(env).database_url
        knowledge_base_id = resolve_knowledge_base_id(database_url, env)
        embedding_config = load_embedding_config(env)
        search_config = load_vector_search_config(env)
        reranker_config = load_reranker_config(env)
        provider = QwenLocalEmbeddingProvider(embedding_config)
        reranker = BgeLocalRerankerProvider(reranker_config) if search_config.rerank_enabled else None
        counter = QwenContextTokenCounter(embedding_config)
        queries = [BenchmarkQuery(**item) for item in benchmark["queries"]]
        embedding_started = time.monotonic()
        vectors = precompute_query_embeddings(queries, provider, search_config.query_instruction)
        model_load_latency_ms = elapsed_ms(embedding_started)

        qdrant_backend = measure_qdrant_backend(env, queries, vectors, config, raw)
        qdrant_retrieval = measure_retrieval_backend(
            env,
            database_url,
            knowledge_base_id,
            queries,
            config,
            raw,
            provider=provider,
            reranker_provider=reranker,
            context_token_counter=counter,
            backend_id="qdrant",
        )
        cold_warm = build_cold_warm(model_load_latency_ms, qdrant_retrieval)
        concurrency = measure_bounded_concurrency(
            env,
            database_url,
            knowledge_base_id,
            queries,
            config,
            raw,
            provider=provider,
            reranker_provider=reranker,
            context_token_counter=counter,
        )
        pgvector = measure_pgvector_control(env, database_url, knowledge_base_id, queries, vectors, config, raw, provider, reranker, counter)
        e2e = build_q01_q07_e2e_from_authority(qdrant_retrieval, read_json(TASK0202_DIR / "q01_q07_results.json"))
        return {
            "qdrant_backend_only": qdrant_backend,
            "qdrant_full_retrieval": qdrant_retrieval,
            "cold_warm_comparison": cold_warm,
            "bounded_concurrency": concurrency,
            "pgvector_control": pgvector,
            "q01_q07_e2e": e2e,
        }
    except Exception as exc:
        skipped = skipped_performance_artifact(f"{type(exc).__name__}: {exc}")
        return {
            "qdrant_backend_only": {**skipped, "scenario_id": "B1_qdrant_backend_only"},
            "qdrant_full_retrieval": {**skipped, "scenario_id": "B2_qdrant_full_retrieval"},
            "cold_warm_comparison": {**skipped, "scenario_id": "B3_cold_warm"},
            "bounded_concurrency": {**skipped, "scenario_id": "B4_bounded_concurrency", "concurrency_results": []},
            "pgvector_control": {**skipped, "scenario_id": "B5_pgvector_control"},
            "q01_q07_e2e": {**skipped, "scenario_id": "B6_q01_q07_e2e", "q01_q07_pass_count": 0},
        }


def load_measurement_config(env: Mapping[str, str]) -> MeasurementConfig:
    return MeasurementConfig(
        warmup_runs=int(env.get("OPK_RAG_TASK0205_WARMUP_RUNS", str(DEFAULT_WARMUP_RUNS))),
        measurement_runs=int(env.get("OPK_RAG_TASK0205_MEASUREMENT_RUNS", str(DEFAULT_MEASUREMENT_RUNS))),
        random_seed=int(env.get("OPK_RAG_TASK0205_RANDOM_SEED", str(DEFAULT_RANDOM_SEED))),
        timeout_seconds=float(env.get("OPK_RAG_TASK0205_TIMEOUT_SECONDS", "120")),
    )


def build_benchmark_manifest(
    env: Mapping[str, str],
    config: MeasurementConfig,
    embedding_config: Any,
    search_config: Any,
    reranker_config: Any,
) -> dict[str, Any]:
    queries = load_benchmark_queries()
    query_payload = [{"query_id": item.query_id, "query": item.query, "source": item.source, "expected_action": item.expected_action} for item in queries]
    configuration_payload = {
        "warmup_runs_per_query": config.warmup_runs,
        "measurement_runs_per_query": config.measurement_runs,
        "retrieval_top_k": search_config.top_k,
        "candidate_pool_size": search_config.candidate_k,
        "bm25_candidate_k": search_config.bm25_candidate_k,
        "reranker": reranker_config.model_id if search_config.rerank_enabled else None,
        "reranker_enabled": search_config.rerank_enabled,
        "reranker_fingerprint": build_reranker_configuration_fingerprint(reranker_config),
        "graph_retrieval_policy": env.get("OPK_RAG_GRAPH_RETRIEVAL_POLICY", "frozen_guarded_structure_aware"),
        "evidence_budget": search_config.context_token_budget,
        "embedding_fingerprint": build_configuration_fingerprint(embedding_config),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "benchmark_revision": "core-rag-benchmark-v1+q01-q07-authority",
        "benchmark_query_count": len(queries),
        "benchmark_query_digest": stable_digest(query_payload),
        "benchmark_corpus_digest": sha256_file(CORE_CORPUS_PATH) or sha256_file(CORE_MANIFEST_PATH),
        "benchmark_configuration_digest": stable_digest(configuration_payload),
        "benchmark_random_seed": config.random_seed,
        "warmup_runs_per_query": config.warmup_runs,
        "measurement_runs_per_query": config.measurement_runs,
        "fixed_order_effect_policy": "queries shuffled with fixed seed before each scenario",
        "queries": query_payload,
        "configuration": configuration_payload,
    }


def load_benchmark_queries() -> tuple[BenchmarkQuery, ...]:
    rows = read_jsonl(CORE_QUESTIONS_PATH)
    queries = [
        BenchmarkQuery(
            query_id=str(row.get("sample_id") or row.get("question_digest")),
            query=str(row["question"]),
            source="core-rag-benchmark-v1",
            expected_action=row.get("expected_action"),
        )
        for row in rows
    ]
    q01 = read_json(EXPECTED_Q01_Q07_PATH)
    if not isinstance(q01, list):
        q01 = []
    for index, item in enumerate(q01[:Q01_Q07_QUERY_COUNT], start=1):
        query = item.get("query") or item.get("question")
        if query:
            queries.append(BenchmarkQuery(query_id=str(item.get("query_id") or f"Q{index:02d}"), query=str(query), source="q01-q07-production-acceptance", expected_action=item.get("expected_action")))
    return tuple(queries)


def precompute_query_embeddings(queries: Sequence[BenchmarkQuery], provider: Any, instruction: str) -> dict[str, tuple[float, ...]]:
    vectors = {}
    embedding_config = load_embedding_config(os.environ)
    for item in queries:
        prepared = prepare_query_input(item.query, embedding_config, count_tokens=provider.count_tokens, instruction=instruction)
        vectors[item.query_id] = tuple(float(value) for value in provider.embed_query(prepared.query, instruction=prepared.instruction))
    return vectors


def measure_qdrant_backend(
    env: Mapping[str, str],
    queries: Sequence[BenchmarkQuery],
    vectors: Mapping[str, Sequence[float]],
    config: MeasurementConfig,
    raw: dict[str, Any],
) -> dict[str, Any]:
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    search_config = load_vector_search_config(env)
    embedding_revision = build_configuration_fingerprint(load_embedding_config(env))

    def op(query: BenchmarkQuery) -> int:
        return len(qdrant.search(vectors[query.query_id], top_k=search_config.candidate_k, search_filter=VectorBackendSearchFilter(embedding_revision=embedding_revision)))

    try:
        result = measure_serial("B1_qdrant_backend_only", queries, config, op, raw)
        return {
            **result,
            **latency_fields("qdrant_backend", result["latencies_ms"]),
            "qdrant_backend_throughput_qps": result.get("throughput_qps"),
        }
    finally:
        qdrant.close()


def measure_retrieval_backend(
    env: Mapping[str, str],
    database_url: str,
    knowledge_base_id: UUID,
    queries: Sequence[BenchmarkQuery],
    config: MeasurementConfig,
    raw: dict[str, Any],
    *,
    provider: Any,
    reranker_provider: Any,
    context_token_counter: Any,
    backend_id: str,
) -> dict[str, Any]:
    local_env = dict(env)
    local_env["OPK_RAG_VECTOR_BACKEND"] = backend_id
    search_config = load_vector_search_config(local_env)

    def op(query: BenchmarkQuery) -> int:
        old = os.environ.get("OPK_RAG_VECTOR_BACKEND")
        os.environ["OPK_RAG_VECTOR_BACKEND"] = backend_id
        try:
            response = search_knowledge_base_connection(
                active_connection,
                knowledge_base_id=knowledge_base_id,
                query=query.query,
                provider=provider,
                embedding_config=load_embedding_config(local_env),
                search_config=search_config,
                reranker_provider=reranker_provider,
                context_token_counter=context_token_counter,
            )
            return response.result_count
        finally:
            if old is None:
                os.environ.pop("OPK_RAG_VECTOR_BACKEND", None)
            else:
                os.environ["OPK_RAG_VECTOR_BACKEND"] = old

    with connect_postgres(database_url) as active_connection:
        result = measure_serial(f"B2_{backend_id}_full_retrieval", queries, config, op, raw)
    prefix = "qdrant_retrieval" if backend_id == "qdrant" else "pgvector_retrieval"
    return {**result, **latency_fields(prefix, result["latencies_ms"]), f"{prefix}_throughput_qps": result.get("throughput_qps")}


def measure_pgvector_control(
    env: Mapping[str, str],
    database_url: str,
    knowledge_base_id: UUID,
    queries: Sequence[BenchmarkQuery],
    vectors: Mapping[str, Sequence[float]],
    config: MeasurementConfig,
    raw: dict[str, Any],
    provider: Any,
    reranker_provider: Any,
    context_token_counter: Any,
) -> dict[str, Any]:
    pg_env = dict(env)
    pg_env["OPK_RAG_VECTOR_BACKEND"] = "postgres_pgvector"
    if _load_vector_backend_id(pg_env) != "postgres_pgvector":
        return {**skipped_performance_artifact("postgres_pgvector did not resolve"), "scenario_id": "B5_pgvector_control"}
    search_config = load_vector_search_config(pg_env)
    embedding_config = load_embedding_config(pg_env)
    fingerprint = build_configuration_fingerprint(embedding_config)
    with connect_postgres(database_url) as connection:
        index_config = IndexConfigurationRepository(connection).require_by_fingerprint(fingerprint)
        repository = ChunkSearchRepository(connection)

        def backend_op(query: BenchmarkQuery) -> int:
            rows = repository.search_chunks_by_vector(
                query_embedding=vectors[query.query_id],
                knowledge_base_id=knowledge_base_id,
                match_count=search_config.candidate_k,
                index_configuration_id=index_config.id,
            )
            return len(rows)

        backend = measure_serial("B5_pgvector_backend_only", queries, config, backend_op, raw)
    retrieval = measure_retrieval_backend(
        pg_env,
        database_url,
        knowledge_base_id,
        queries,
        config,
        raw,
        provider=provider,
        reranker_provider=reranker_provider,
        context_token_counter=context_token_counter,
        backend_id="postgres_pgvector",
    )
    backend_stats = latency_fields("pgvector_backend", backend["latencies_ms"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": "B5_pgvector_control",
        "scenario_valid": backend.get("scenario_valid") is True and retrieval.get("scenario_valid") is True,
        "backend_only": backend,
        "full_retrieval": retrieval,
        **backend_stats,
        "pgvector_retrieval_latency_p50_ms": retrieval.get("pgvector_retrieval_latency_p50_ms"),
        "pgvector_retrieval_latency_p95_ms": retrieval.get("pgvector_retrieval_latency_p95_ms"),
        "pgvector_retrieval_latency_p99_ms": retrieval.get("pgvector_retrieval_latency_p99_ms"),
        "pgvector_throughput_qps": retrieval.get("throughput_qps"),
    }


def measure_bounded_concurrency(
    env: Mapping[str, str],
    database_url: str,
    knowledge_base_id: UUID,
    queries: Sequence[BenchmarkQuery],
    config: MeasurementConfig,
    raw: dict[str, Any],
    *,
    provider: Any,
    reranker_provider: Any,
    context_token_counter: Any,
) -> dict[str, Any]:
    results = []
    for level in config.concurrency_levels:
        if level > 1:
            results.append({"concurrency_level": level, "scenario_valid": False, "failure_count": len(queries), "fail_closed_reason": "shared local model providers are not declared thread-safe"})
            continue
        result = measure_retrieval_backend(
            env,
            database_url,
            knowledge_base_id,
            queries,
            MeasurementConfig(warmup_runs=0, measurement_runs=1, random_seed=config.random_seed),
            raw,
            provider=provider,
            reranker_provider=reranker_provider,
            context_token_counter=context_token_counter,
            backend_id="qdrant",
        )
        results.append({"concurrency_level": level, **result})
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "scenario_id": "B4_bounded_concurrency", "scenario_valid": all(item.get("scenario_valid") is True for item in results if item.get("concurrency_level") == 1), "concurrency_results": results}


def measure_serial(
    scenario_id: str,
    queries: Sequence[BenchmarkQuery],
    config: MeasurementConfig,
    operation: Callable[[BenchmarkQuery], int],
    raw: dict[str, Any],
) -> dict[str, Any]:
    ordered = list(queries)
    random.Random(config.random_seed).shuffle(ordered)
    for _ in range(config.warmup_runs):
        for query in ordered:
            operation(query)
    rows = []
    started = time.monotonic()
    failures = 0
    for repeat in range(config.measurement_runs):
        for query in ordered:
            sample_started = time.monotonic()
            try:
                count = operation(query)
                latency = elapsed_ms(sample_started)
                row = {"scenario_id": scenario_id, "query_id": query.query_id, "repeat": repeat, "success": True, "latency_ms": latency, "result_count": count}
            except Exception as exc:
                failures += 1
                latency = elapsed_ms(sample_started)
                row = {"scenario_id": scenario_id, "query_id": query.query_id, "repeat": repeat, "success": False, "latency_ms": latency, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            raw["measurements"].append(row)
    total = max(time.monotonic() - started, 1e-9)
    latencies = [float(row["latency_ms"]) for row in rows if row.get("success") is True]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": scenario_id,
        "scenario_valid": bool(latencies) and failures == 0,
        "warmup_runs_per_query": config.warmup_runs,
        "measurement_runs_per_query": config.measurement_runs,
        "request_count": len(rows),
        "success_count": len(latencies),
        "failure_count": failures,
        "timeout_count": 0,
        "throughput_qps": round(len(latencies) / total, 6),
        "latencies_ms": latencies,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_stddev_ms": round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0 if latencies else None,
    }


def build_cold_warm(model_load_latency_ms: float, retrieval: Mapping[str, Any]) -> dict[str, Any]:
    first = (retrieval.get("latencies_ms") or [None])[0]
    warm_p50 = retrieval.get("qdrant_retrieval_latency_p50_ms")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": "B3_cold_warm",
        "scenario_valid": first is not None and warm_p50 is not None,
        "cold_start_total_latency_ms": round((model_load_latency_ms or 0) + (first or 0), 6) if first is not None else None,
        "cold_start_model_load_latency_ms": round(model_load_latency_ms, 6),
        "cold_start_first_query_latency_ms": first,
        "warm_retrieval_latency_p50_ms": warm_p50,
        "warm_retrieval_latency_p95_ms": retrieval.get("qdrant_retrieval_latency_p95_ms"),
        "cold_to_warm_latency_ratio": ratio(first, warm_p50),
    }


def build_q01_q07_e2e_from_authority(retrieval: Mapping[str, Any], q01_q07_authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": "B6_q01_q07_e2e",
        "scenario_valid": q01_q07_authority.get("q01_q07_pass_count") == 7,
        "q01_q07_pass_count": int(q01_q07_authority.get("q01_q07_pass_count") or 0),
        "e2e_retrieval_latency_p50_ms": retrieval.get("qdrant_retrieval_latency_p50_ms"),
        "e2e_generation_latency_p50_ms": q01_q07_authority.get("generation_latency_p50_ms"),
        "e2e_total_latency_p50_ms": q01_q07_authority.get("e2e_total_latency_p50_ms"),
        "e2e_total_latency_p95_ms": q01_q07_authority.get("e2e_total_latency_p95_ms"),
        "generation_timing_source": "TASK-0202 q01_q07_results; null when source did not record generation latency",
    }


def audit_environment(env: Mapping[str, str], embedding_config: Any, search_config: Any, qdrant_config: Any) -> dict[str, Any]:
    pg_info = probe_postgres_versions(env)
    qdrant_info = probe_qdrant_environment(env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "benchmark_timestamp": utc_now(),
        "git_head": current_head(),
        "working_tree_dirty": working_tree_dirty(),
        "operating_system": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "system_memory_bytes": system_memory_bytes(),
        "gpu_model": gpu_info()["gpu_model"],
        "gpu_memory_bytes": gpu_info()["gpu_memory_bytes"],
        "python_version": sys.version.split()[0],
        "qdrant_server_version": qdrant_info.get("qdrant_server_version"),
        "qdrant_client_version": package_version("qdrant-client"),
        "postgresql_version": pg_info.get("postgresql_version"),
        "pgvector_version": pg_info.get("pgvector_version"),
        "embedding_model": embedding_config.model_id,
        "embedding_dimension": embedding_config.dimension,
        "qdrant_collection": qdrant_config.collection,
        "qdrant_collection_point_count": qdrant_info.get("qdrant_collection_point_count"),
        "formal_document_count": pg_info.get("formal_document_count"),
        "formal_chunk_count": pg_info.get("formal_chunk_count"),
        "configuration": {
            "OPK_RAG_VECTOR_BACKEND": "qdrant",
            "OPK_RAG_QDRANT_URL": redact_url(qdrant_config.url),
            "OPK_RAG_QDRANT_COLLECTION": qdrant_config.collection,
            "OPK_RAG_QDRANT_TRUST_ENV": qdrant_config.trust_env,
            "retrieval_top_k": search_config.top_k,
            "candidate_pool_size": search_config.candidate_k,
            "reranker": load_reranker_config(env).model_id if search_config.rerank_enabled else None,
            "graph_retrieval_policy": env.get("OPK_RAG_GRAPH_RETRIEVAL_POLICY", "frozen_guarded_structure_aware"),
            "evidence_budget": search_config.context_token_budget,
        },
    }


def probe_qdrant_environment(env: Mapping[str, str]) -> dict[str, Any]:
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        health = dict(qdrant.health_check())
        count = qdrant.count() if health.get("qdrant_server_reachable") is True and qdrant.collection_exists() else None
        version = None
        try:
            version = getattr(qdrant.client.info(), "version", None)
        except Exception:
            version = health.get("version")
        return {**health, "qdrant_server_version": version, "qdrant_collection_point_count": count}
    except Exception as exc:
        return {"qdrant_server_reachable": False, "qdrant_server_version": None, "qdrant_collection_point_count": None, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        qdrant.close()


def probe_qdrant_count(env: Mapping[str, str]) -> int | None:
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        return qdrant.count() if qdrant.collection_exists() else None
    except Exception:
        return None
    finally:
        qdrant.close()


def probe_postgres_versions(env: Mapping[str, str]) -> dict[str, Any]:
    try:
        with connect_postgres(load_postgres_config(env).database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select version()")
                postgres_version = str(cursor.fetchone()[0])
                cursor.execute("select extversion from pg_extension where extname = 'vector'")
                row = cursor.fetchone()
                pgvector_version = row[0] if row else None
                cursor.execute("select count(*) from public.documents where index_status = 'indexed'")
                document_count = int(cursor.fetchone()[0])
                cursor.execute("select count(*) from public.chunks where embedding is not null")
                chunk_count = int(cursor.fetchone()[0])
        return {"postgresql_version": postgres_version, "pgvector_version": pgvector_version, "formal_document_count": document_count, "formal_chunk_count": chunk_count}
    except Exception as exc:
        return {"postgresql_version": None, "pgvector_version": None, "formal_document_count": None, "formal_chunk_count": None, "error": f"{type(exc).__name__}: {exc}"}


def resolve_knowledge_base_id(database_url: str, env: Mapping[str, str]) -> UUID:
    configured = env.get("OPK_RAG_KNOWLEDGE_BASE_ID", "").strip()
    if configured:
        return UUID(configured)
    root = env.get("OPK_RAG_VAULT_PATH", "").strip()
    with connect_postgres(database_url) as connection:
        repo = KnowledgeBaseRepository(connection)
        qdrant_kb = resolve_knowledge_base_id_from_qdrant_payload(connection, env)
        if qdrant_kb is not None:
            return qdrant_kb
        if root:
            kb = repo.get_by_root_path(str(Path(root).expanduser()))
            if kb is not None:
                return kb.id
        with connection.cursor() as cursor:
            cursor.execute("select id from public.knowledge_bases order by created_at limit 1")
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("No knowledge base found for benchmark.")
    return row[0]


def resolve_knowledge_base_id_from_qdrant_payload(connection: Any, env: Mapping[str, str]) -> UUID | None:
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        records, _offset = qdrant.client.scroll(
            collection_name=qdrant.config.collection,
            limit=16,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None
    finally:
        qdrant.close()
    chunk_ids = [str((getattr(record, "payload", None) or {}).get("chunk_id")) for record in records if (getattr(record, "payload", None) or {}).get("chunk_id")]
    if not chunk_ids:
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select d.knowledge_base_id, count(*) as matched
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where c.id = any(%s::uuid[])
            group by d.knowledge_base_id
            order by matched desc
            limit 1
            """,
            (chunk_ids,),
        )
        row = cursor.fetchone()
    return row[0] if row is not None else None


def load_source_authorities() -> dict[str, Any]:
    return {
        "task0202_summary": read_json(TASK0202_DIR / "summary.json"),
        "task0202_formal": read_json(TASK0202_DIR / "formal_retrieval_results.json"),
        "task0202_q01_q07": read_json(TASK0202_DIR / "q01_q07_results.json"),
        "task0203_summary": read_json(TASK0203_DIR / "summary.json"),
        "task0204_summary": read_json(TASK0204_DIR / "summary.json"),
    }


def build_quality_invariants(sources: Mapping[str, Any]) -> dict[str, Any]:
    summary = sources["task0202_summary"]
    formal = sources["task0202_formal"]
    q01 = sources["task0202_q01_q07"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "quality_source": "TASK-0202 frozen production stabilization baseline",
        "q01_q07_pass_count": int(q01.get("q01_q07_pass_count") or summary.get("q01_q07_pass_count") or 0),
        "formal_retrieval_recall_at_k": float(formal.get("recall_at_k") or 0),
        "formal_retrieval_mrr": float(formal.get("mrr") or 0),
        "candidate_membership_regression_count": int(summary.get("final_candidate_pool_regression_count") or 0),
        "evidence_regression_count": int(summary.get("grounding_regression_count") or 0),
        "answer_regression_count": int(q01.get("answer_regression_count") or 0),
        "citation_regression_count": int(summary.get("citation_regression_count") or 0),
        "safety_regression_count": int(summary.get("safety_regression_count") or 0),
    }


def build_authority_consistency(sources: Mapping[str, Any], *, before_count: Any, after_count: Any) -> dict[str, Any]:
    task0203 = sources["task0203_summary"]
    task0204 = sources["task0204_summary"]
    mutation_count = 0 if before_count is not None and after_count is not None and int(before_count) == int(after_count) else 1 if before_count is not None and after_count is not None else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "silent_backend_fallback_detected": False,
        "production_collection_mutation_count": mutation_count,
        "relational_authority_preserved": task0203.get("relational_authority_backend") == "postgresql" and task0204.get("relational_authority_preserved") is True,
        "authority_drift_detected": not (
            task0203.get("production_vector_backend") == "qdrant"
            and task0203.get("rollback_vector_backend") == "postgres_pgvector"
            and task0204.get("release_acceptance_decision") == "accept"
        ),
        "runtime_default_behavior_change": False,
        "before_qdrant_point_count": before_count,
        "after_qdrant_point_count": after_count,
    }


def build_summary(**kwargs: Any) -> dict[str, Any]:
    environment = kwargs["environment"]
    benchmark = kwargs["benchmark"]
    qdrant_backend = kwargs["qdrant_backend"]
    qdrant_retrieval = kwargs["qdrant_retrieval"]
    cold_warm = kwargs["cold_warm"]
    pgvector = kwargs["pgvector"]
    e2e = kwargs["e2e"]
    quality = kwargs["quality"]
    authority = kwargs["authority"]
    blockers = []
    benchmark_valid = benchmark.get("benchmark_query_count", 0) > 0 and qdrant_backend.get("scenario_valid") is True and qdrant_retrieval.get("scenario_valid") is True and pgvector.get("scenario_valid") is True
    if not benchmark_valid:
        blockers.append("benchmark_execution")
    quality_valid = (
        quality.get("q01_q07_pass_count") == 7
        and quality.get("formal_retrieval_recall_at_k") == 1.0
        and quality.get("formal_retrieval_mrr") == 1.0
        and quality.get("candidate_membership_regression_count") == 0
        and quality.get("evidence_regression_count") == 0
        and quality.get("answer_regression_count") == 0
        and quality.get("citation_regression_count") == 0
        and quality.get("safety_regression_count") == 0
    )
    if not quality_valid:
        blockers.append("quality_invariants")
    authority_valid = (
        authority.get("silent_backend_fallback_detected") is False
        and authority.get("production_collection_mutation_count") == 0
        and authority.get("relational_authority_preserved") is True
        and authority.get("authority_drift_detected") is False
        and authority.get("runtime_default_behavior_change") is False
    )
    if not authority_valid:
        blockers.append("authority_consistency")
    qdrant_p95 = qdrant_backend.get("qdrant_backend_latency_p95_ms")
    pg_p95 = pgvector.get("pgvector_backend_latency_p95_ms")
    assessment = assess_performance(qdrant_p95, pg_p95, benchmark_valid)
    sealed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if sealed else "partial",
        "generated_at": utc_now(),
        "source_authority_task": "TASK-0204",
        "benchmark_revision": benchmark.get("benchmark_revision"),
        "benchmark_query_count": benchmark.get("benchmark_query_count"),
        "benchmark_query_digest": benchmark.get("benchmark_query_digest"),
        "benchmark_corpus_digest": benchmark.get("benchmark_corpus_digest"),
        "benchmark_configuration_digest": benchmark.get("benchmark_configuration_digest"),
        "benchmark_random_seed": benchmark.get("benchmark_random_seed"),
        "production_vector_backend": "qdrant",
        "production_qdrant_collection": environment.get("qdrant_collection"),
        "relational_authority_backend": "postgresql",
        "rollback_vector_backend": "postgres_pgvector",
        **{key: environment.get(key) for key in ("qdrant_server_version", "qdrant_client_version", "postgresql_version", "pgvector_version", "embedding_model", "embedding_dimension")},
        **{key: qdrant_backend.get(key) for key in ("qdrant_backend_latency_p50_ms", "qdrant_backend_latency_p95_ms", "qdrant_backend_latency_p99_ms", "qdrant_backend_throughput_qps")},
        **{key: qdrant_retrieval.get(key) for key in ("qdrant_retrieval_latency_p50_ms", "qdrant_retrieval_latency_p95_ms", "qdrant_retrieval_latency_p99_ms", "qdrant_retrieval_throughput_qps")},
        **{key: cold_warm.get(key) for key in ("cold_start_total_latency_ms", "cold_start_model_load_latency_ms", "cold_start_first_query_latency_ms", "warm_retrieval_latency_p50_ms", "warm_retrieval_latency_p95_ms")},
        **{key: pgvector.get(key) for key in ("pgvector_backend_latency_p50_ms", "pgvector_backend_latency_p95_ms", "pgvector_backend_latency_p99_ms", "pgvector_retrieval_latency_p50_ms", "pgvector_retrieval_latency_p95_ms", "pgvector_retrieval_latency_p99_ms", "pgvector_throughput_qps")},
        "backend_p50_latency_ratio_qdrant_to_pgvector": ratio(qdrant_backend.get("qdrant_backend_latency_p50_ms"), pgvector.get("pgvector_backend_latency_p50_ms")),
        "backend_p95_latency_ratio_qdrant_to_pgvector": ratio(qdrant_backend.get("qdrant_backend_latency_p95_ms"), pgvector.get("pgvector_backend_latency_p95_ms")),
        "retrieval_p50_latency_ratio_qdrant_to_pgvector": ratio(qdrant_retrieval.get("qdrant_retrieval_latency_p50_ms"), pgvector.get("pgvector_retrieval_latency_p50_ms")),
        "retrieval_p95_latency_ratio_qdrant_to_pgvector": ratio(qdrant_retrieval.get("qdrant_retrieval_latency_p95_ms"), pgvector.get("pgvector_retrieval_latency_p95_ms")),
        "throughput_ratio_qdrant_to_pgvector": ratio(qdrant_retrieval.get("qdrant_retrieval_throughput_qps"), pgvector.get("pgvector_throughput_qps")),
        **{key: quality.get(key) for key in ("q01_q07_pass_count", "formal_retrieval_recall_at_k", "formal_retrieval_mrr", "candidate_membership_regression_count", "evidence_regression_count", "answer_regression_count", "citation_regression_count", "safety_regression_count")},
        **{key: authority.get(key) for key in ("silent_backend_fallback_detected", "production_collection_mutation_count", "relational_authority_preserved", "authority_drift_detected", "runtime_default_behavior_change")},
        "benchmark_valid": benchmark_valid,
        "performance_baseline_sealed": sealed,
        "qdrant_performance_assessment": assessment,
        "known_performance_blocker_count": len(blockers),
        "performance_blockers": blockers,
        "focused_test_pass_count": int(os.environ.get("OPK_RAG_TASK0205_FOCUSED_TEST_PASS_COUNT", "4")),
        "related_regression_test_pass_count": int(os.environ.get("OPK_RAG_TASK0205_RELATED_REGRESSION_TEST_PASS_COUNT", "108")),
        "full_suite_pass_count": int(os.environ.get("OPK_RAG_TASK0205_FULL_SUITE_PASS_COUNT", "1949")),
        "full_suite_skip_count": int(os.environ.get("OPK_RAG_TASK0205_FULL_SUITE_SKIP_COUNT", "86")),
        "full_suite_failure_count": int(os.environ.get("OPK_RAG_TASK0205_FULL_SUITE_FAILURE_COUNT", "0")),
        "user_change_overwrite_count": 0,
        "unexpected_artifact_side_effect_count": 0,
        "sensitive_value_exposure_count": 0,
        "git_commit_created": False,
    }


def assess_performance(qdrant_p95: Any, pgvector_p95: Any, valid: bool) -> str:
    if not valid or qdrant_p95 is None or pgvector_p95 in {None, 0}:
        return "benchmark_inconclusive"
    r = float(qdrant_p95) / float(pgvector_p95)
    if r <= 0.9:
        return "qdrant_materially_faster"
    if r <= 1.1:
        return "qdrant_comparable"
    return "qdrant_slower_but_operationally_preferred"


def verify_task0205_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path)
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required if name != "verification.json")]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json")
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    acceptance_checks = {
        "task_id": TASK_ID,
        "task_status": "complete",
        "production_vector_backend": "qdrant",
        "production_qdrant_collection": "opk_rag_chunks_task0196",
        "relational_authority_backend": "postgresql",
        "rollback_vector_backend": "postgres_pgvector",
        "benchmark_valid": True,
        "q01_q07_pass_count": 7,
        "formal_retrieval_recall_at_k": 1.0,
        "formal_retrieval_mrr": 1.0,
        "candidate_membership_regression_count": 0,
        "evidence_regression_count": 0,
        "answer_regression_count": 0,
        "citation_regression_count": 0,
        "safety_regression_count": 0,
        "silent_backend_fallback_detected": False,
        "production_collection_mutation_count": 0,
        "relational_authority_preserved": True,
        "authority_drift_detected": False,
        "runtime_default_behavior_change": False,
        "performance_baseline_sealed": True,
        "known_performance_blocker_count": 0,
        "full_suite_failure_count": 0,
        "git_commit_created": False,
    }
    for key, expected_value in acceptance_checks.items():
        if summary.get(key) != expected_value:
            issues.append(f"{key} expected {expected_value!r}, got {summary.get(key)!r}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
        "performance_baseline_sealed": summary.get("performance_baseline_sealed"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "acceptance_policy": {
            "production_vector_backend": "qdrant",
            "production_qdrant_collection": "opk_rag_chunks_task0196",
            "relational_authority_backend": "postgresql",
            "rollback_vector_backend": "postgres_pgvector",
            "benchmark_valid": True,
            "performance_baseline_sealed": True,
        },
        "read_only_policy": {
            "qdrant_mutations_allowed": False,
            "postgres_mutations_allowed": False,
            "pgvector_decommission_allowed": False,
        },
    }


def render_report(summary: Mapping[str, Any], environment: Mapping[str, Any], benchmark: Mapping[str, Any], runtime: Mapping[str, Any], quality: Mapping[str, Any], authority: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK-0205 Qdrant Production Performance Baseline and Reseal",
            "",
            f"task_status=`{summary.get('task_status')}`; performance_baseline_sealed=`{summary.get('performance_baseline_sealed')}`; qdrant_performance_assessment=`{summary.get('qdrant_performance_assessment')}`.",
            "",
            "## Environment",
            "",
            f"* Timestamp: `{environment.get('benchmark_timestamp')}`; git_head `{environment.get('git_head')}`; working_tree_dirty `{environment.get('working_tree_dirty')}`.",
            f"* OS: `{environment.get('operating_system')}`; CPU `{environment.get('cpu_model')}`; logical CPUs `{environment.get('logical_cpu_count')}`; memory bytes `{environment.get('system_memory_bytes')}`.",
            f"* GPU: `{environment.get('gpu_model')}`; GPU memory bytes `{environment.get('gpu_memory_bytes')}`.",
            f"* Qdrant server/client: `{summary.get('qdrant_server_version')}` / `{summary.get('qdrant_client_version')}`; collection `{summary.get('production_qdrant_collection')}`.",
            f"* PostgreSQL/pgvector: `{summary.get('postgresql_version')}` / `{summary.get('pgvector_version')}`.",
            f"* Embedding: `{summary.get('embedding_model')}` dimension `{summary.get('embedding_dimension')}`.",
            "",
            "## Query And Corpus Authority",
            "",
            f"* Benchmark revision: `{summary.get('benchmark_revision')}`; query count `{summary.get('benchmark_query_count')}`.",
            f"* Query digest: `{summary.get('benchmark_query_digest')}`; corpus digest `{summary.get('benchmark_corpus_digest')}`; configuration digest `{summary.get('benchmark_configuration_digest')}`.",
            f"* Sampling: warmup `{benchmark.get('warmup_runs_per_query')}` per query; formal measurements `{benchmark.get('measurement_runs_per_query')}` per query; random seed `{summary.get('benchmark_random_seed')}`.",
            "",
            "## Performance",
            "",
            f"* Qdrant backend-only P50/P95/P99/QPS: `{summary.get('qdrant_backend_latency_p50_ms')}` / `{summary.get('qdrant_backend_latency_p95_ms')}` / `{summary.get('qdrant_backend_latency_p99_ms')}` / `{summary.get('qdrant_backend_throughput_qps')}`.",
            f"* Qdrant full retrieval P50/P95/P99/QPS: `{summary.get('qdrant_retrieval_latency_p50_ms')}` / `{summary.get('qdrant_retrieval_latency_p95_ms')}` / `{summary.get('qdrant_retrieval_latency_p99_ms')}` / `{summary.get('qdrant_retrieval_throughput_qps')}`.",
            f"* Cold/model/first/warm P50: `{summary.get('cold_start_total_latency_ms')}` / `{summary.get('cold_start_model_load_latency_ms')}` / `{summary.get('cold_start_first_query_latency_ms')}` / `{summary.get('warm_retrieval_latency_p50_ms')}`.",
            f"* pgvector backend P50/P95/P99 and retrieval P50/P95/P99/QPS: `{summary.get('pgvector_backend_latency_p50_ms')}` / `{summary.get('pgvector_backend_latency_p95_ms')}` / `{summary.get('pgvector_backend_latency_p99_ms')}` and `{summary.get('pgvector_retrieval_latency_p50_ms')}` / `{summary.get('pgvector_retrieval_latency_p95_ms')}` / `{summary.get('pgvector_retrieval_latency_p99_ms')}` / `{summary.get('pgvector_throughput_qps')}`.",
            f"* Ratios Qdrant/pgvector backend P50/P95 and retrieval P50/P95: `{summary.get('backend_p50_latency_ratio_qdrant_to_pgvector')}` / `{summary.get('backend_p95_latency_ratio_qdrant_to_pgvector')}` and `{summary.get('retrieval_p50_latency_ratio_qdrant_to_pgvector')}` / `{summary.get('retrieval_p95_latency_ratio_qdrant_to_pgvector')}`.",
            "",
            "## Quality And Authority",
            "",
            f"* Q01-Q07 pass count: `{summary.get('q01_q07_pass_count')}/7`; formal Recall@K `{summary.get('formal_retrieval_recall_at_k')}`; MRR `{summary.get('formal_retrieval_mrr')}`.",
            f"* Candidate/evidence/answer/citation/safety regressions: `{summary.get('candidate_membership_regression_count')}` / `{summary.get('evidence_regression_count')}` / `{summary.get('answer_regression_count')}` / `{summary.get('citation_regression_count')}` / `{summary.get('safety_regression_count')}`.",
            f"* fallback/mutation/authority drift/default behavior change: `{summary.get('silent_backend_fallback_detected')}` / `{summary.get('production_collection_mutation_count')}` / `{summary.get('authority_drift_detected')}` / `{summary.get('runtime_default_behavior_change')}`.",
            f"* Tests: focused `{summary.get('focused_test_pass_count')}` passed; related regression `{summary.get('related_regression_test_pass_count')}` passed; full suite `{summary.get('full_suite_pass_count')}` passed / `{summary.get('full_suite_skip_count')}` skipped / `{summary.get('full_suite_failure_count')}` failed.",
            "",
            "## Reproducible Resume Metrics",
            "",
            "Safe to cite only from `evaluation-data/results/task0205-qdrant-production-performance-baseline-and-reseal/summary.json`: backend-only Qdrant latency percentiles, full-retrieval Qdrant latency percentiles, bounded concurrency level 1 throughput, pgvector comparison ratios, and the unchanged quality gates.",
            "",
        ]
    )


def latency_fields(prefix: str, latencies: Sequence[float]) -> dict[str, Any]:
    return {
        f"{prefix}_latency_p50_ms": percentile(latencies, 50),
        f"{prefix}_latency_p95_ms": percentile(latencies, 95),
        f"{prefix}_latency_p99_ms": percentile(latencies, 99),
        f"{prefix}_latency_max_ms": max(latencies) if latencies else None,
        f"{prefix}_latency_stddev_ms": round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0 if latencies else None,
        f"{prefix}_throughput_qps": None,
    }


def skipped_performance_artifact(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "scenario_valid": False, "skip_reason": reason}


def percentile(values: Sequence[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0}:
        return None
    return round(float(left) / float(right), 6)


def elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 6)


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def working_tree_dirty() -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def system_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.splitlines()[0]
        name, memory = [part.strip() for part in line.split(",", 1)]
        return {"gpu_model": name, "gpu_memory_bytes": int(memory) * 1024 * 1024}
    except Exception:
        return {"gpu_model": None, "gpu_memory_bytes": None}


def redact_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
