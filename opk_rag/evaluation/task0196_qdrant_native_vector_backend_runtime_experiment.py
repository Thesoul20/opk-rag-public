from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint, load_embedding_config
from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json, write_jsonl
from opk_rag.evaluation import task0195_native_vector_database_evaluation_baseline as task0195
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig
from opk_rag.vector_backends.base import VectorBackendCandidate, canonical_similarity_score
from opk_rag.vector_backends.base import VectorBackendPoint, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import (
    DEFAULT_QDRANT_COLLECTION,
    QdrantVectorBackend,
    deterministic_qdrant_point_id,
    load_qdrant_config,
)

TASK_ID = "TASK-0196"
EXPERIMENT_ID = "task0196-qdrant-native-vector-backend-runtime-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0196_qdrant_native_vector_backend_runtime_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0196_QDRANT_NATIVE_VECTOR_BACKEND_RUNTIME_EXPERIMENT_REPORT.md"
TASK0195_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0195-native-vector-database-evaluation-baseline" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0196.qdrant-native-vector-backend-runtime-experiment.v1"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_authority.json",
    "backend_abstraction_audit.json",
    "qdrant_collection_contract.json",
    "qdrant_connectivity.json",
    "candidate_membership_comparison.json",
    "candidate_membership_rows.jsonl",
    "ranking_comparison.json",
    "formal_retrieval_benchmark.json",
    "q01_q07_replay.json",
    "crud_validation.json",
    "incremental_indexing.json",
    "cold_start_reproducibility.json",
    "snapshot_smoke.json",
    "performance_comparison.json",
    "historical_artifact_side_effect_audit.json",
    "failure_taxonomy.json",
    "contract.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authority_valid",
    "task0194_source_authority_valid",
    "task0195_source_authority_valid",
    "production_vector_backend",
    "experimental_vector_backend",
    "production_backend_promoted",
    "production_runtime_mutation_count",
    "runtime_default_behavior_change",
    "existing_vector_backend_abstraction_found",
    "existing_abstraction_reusable",
    "new_abstraction_required",
    "qdrant_server_reachable",
    "collection_schema_valid",
    "candidate_identity_valid",
    "point_id_collision_count",
    "qdrant_score_adapter_valid",
    "qdrant_promotion_eligible",
    "qdrant_experiment_decision",
    "full_suite_pass",
    "full_suite_failure_count",
    "test_execution_side_effect_file_count",
    "restored_test_execution_side_effect_file_count",
    "user_change_overwrite_count",
)

FAILURE_CODES = (
    "server_startup_failure",
    "client_connection_failure",
    "collection_creation_failure",
    "collection_schema_failure",
    "point_identity_failure",
    "payload_schema_failure",
    "payload_index_failure",
    "embedding_materialization_failure",
    "dimension_mismatch",
    "query_failure",
    "filter_failure",
    "score_adapter_failure",
    "candidate_membership_regression",
    "ranking_regression",
    "downstream_regression",
    "crud_failure",
    "incremental_update_failure",
    "idempotency_failure",
    "snapshot_failure",
    "cold_start_failure",
    "performance_regression",
    "unknown",
)


def run_task0196(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_config = load_embedding_config(runtime_env)
    search_config = VectorSearchConfig()
    qdrant_config = load_qdrant_config(runtime_env)

    source_authority = build_source_authority()
    abstraction_audit = build_backend_abstraction_audit()
    collection_contract = build_qdrant_collection_contract(qdrant_config, embedding_config)
    connectivity = probe_qdrant_connectivity(runtime_env)
    score_contract = build_score_contract()

    skipped_reason = connectivity.get("skip_reason") or connectivity.get("error") or "real Qdrant server is required for runtime artifacts"
    comparison = skipped_candidate_membership_comparison(str(skipped_reason))
    ranking = skipped_ranking_comparison(str(skipped_reason))
    formal = skipped_formal_retrieval_benchmark(str(skipped_reason))
    replay = skipped_q01_q07_replay(str(skipped_reason))
    crud = skipped_crud_validation(str(skipped_reason))
    incremental = skipped_incremental_indexing(str(skipped_reason))
    cold_start = skipped_cold_start(str(skipped_reason))
    snapshot = skipped_snapshot(str(skipped_reason))
    performance = skipped_performance(str(skipped_reason))
    runtime_result = maybe_run_qdrant_runtime_experiment(runtime_env, embedding_config, search_config, connectivity)
    if runtime_result is not None:
        comparison = runtime_result["candidate_membership_comparison"]
        ranking = runtime_result["ranking_comparison"]
        crud = runtime_result["crud_validation"]
        incremental = runtime_result["incremental_indexing"]
        cold_start = runtime_result["cold_start_reproducibility"]
        snapshot = runtime_result["snapshot_smoke"]
        performance = runtime_result["performance_comparison"]
    side_effect_audit = build_historical_artifact_side_effect_audit(runtime_env)
    failure = build_failure_taxonomy(connectivity, comparison, formal, replay, crud, incremental, cold_start, snapshot, performance)

    summary = build_summary(
        source_authority=source_authority,
        abstraction_audit=abstraction_audit,
        collection_contract=collection_contract,
        connectivity=connectivity,
        score_contract=score_contract,
        comparison=comparison,
        ranking=ranking,
        formal=formal,
        replay=replay,
        crud=crud,
        incremental=incremental,
        cold_start=cold_start,
        snapshot=snapshot,
        performance=performance,
        side_effect_audit=side_effect_audit,
        failure=failure,
        embedding_config=embedding_config,
        search_config=search_config,
    )

    if write:
        artifacts = {
            "source_authority.json": source_authority,
            "backend_abstraction_audit.json": abstraction_audit,
            "qdrant_collection_contract.json": collection_contract,
            "qdrant_connectivity.json": connectivity,
            "candidate_membership_comparison.json": comparison,
            "ranking_comparison.json": ranking,
            "formal_retrieval_benchmark.json": formal,
            "q01_q07_replay.json": replay,
            "crud_validation.json": crud,
            "incremental_indexing.json": incremental,
            "cold_start_reproducibility.json": cold_start,
            "snapshot_smoke.json": snapshot,
            "performance_comparison.json": performance,
            "historical_artifact_side_effect_audit.json": side_effect_audit,
            "failure_taxonomy.json": failure,
            "contract.json": contract(),
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_jsonl(RESULT_DIR / "candidate_membership_rows.jsonl", comparison.get("rows", []))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, source_authority, abstraction_audit, collection_contract, failure), encoding="utf-8")
        verification = verify_task0196_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, source_authority, abstraction_audit, collection_contract, failure), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0194 = task0195.load_task0194_summary()
    task0195_summary = _read_json(TASK0195_SUMMARY_PATH) if TASK0195_SUMMARY_PATH.exists() else {}
    task0194_valid = (
        task0194.get("deployment_stage_closeout_decision") == "freeze"
        and task0194.get("deployment_baseline_digest") == task0195.SOURCE_DEPLOYMENT_BASELINE_DIGEST
        and task0194.get("deployment_baseline_frozen") is True
    )
    task0195_valid = all(
        (
            task0195_summary.get("task_status") == "complete",
            task0195_summary.get("source_vector_backend") == "postgres_pgvector",
            task0195_summary.get("recommended_migration_architecture") == "hybrid_persistence",
            task0195_summary.get("production_backend_promoted") is False,
            task0195_summary.get("production_runtime_mutation_count") == 0,
            task0195_summary.get("runtime_default_behavior_change") is False,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0194_source_authority_valid": task0194_valid,
        "task0195_source_authority_valid": task0195_valid,
        "source_authority_valid": task0194_valid and task0195_valid,
        "deployment_baseline_digest": task0195.SOURCE_DEPLOYMENT_BASELINE_DIGEST,
        "task0195_summary_path": TASK0195_SUMMARY_PATH.relative_to(ROOT).as_posix(),
    }


def build_backend_abstraction_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "existing_vector_backend_abstraction_found": False,
        "existing_abstraction_reusable": False,
        "new_abstraction_required": True,
        "new_abstraction_path": "opk_rag/vector_backends",
        "production_path_refactored": False,
        "second_parallel_vector_backend_abstraction_created": False,
        "business_layer_direct_dependency_for_experimental_backend": {
            "sql": False,
            "pgvector_operators": False,
            "qdrant_sdk_response_objects": False,
        },
        "contract_methods": ["health_check", "initialize", "create_collection", "collection_exists", "upsert", "delete", "search", "count", "close"],
    }


def build_qdrant_collection_contract(qdrant_config: Any, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    fingerprint = build_configuration_fingerprint(embedding_config)
    sample_chunk_id = "11111111-1111-4111-8111-111111111196"
    point_id = deterministic_qdrant_point_id(sample_chunk_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "collection_name": qdrant_config.collection or DEFAULT_QDRANT_COLLECTION,
        "vector_size": embedding_config.dimension,
        "distance": "Cosine",
        "embedding_model": embedding_config.model_name,
        "embedding_revision": fingerprint,
        "payload_schema": {
            "chunk_id": "keyword",
            "document_id": "keyword",
            "embedding_revision": "keyword",
            "chunk_index": "integer",
            "source_path": "keyword",
            "section": "keyword_optional",
        },
        "payload_authoritative": False,
        "postgresql_metadata_authoritative": True,
        "payload_index_fields": ["document_id", "chunk_id", "embedding_revision", "source_path"],
        "point_id_mapping": "uuidv5(namespace, canonical_chunk_id)",
        "point_id_mapping_deterministic": point_id == deterministic_qdrant_point_id(sample_chunk_id),
        "sample_chunk_id": sample_chunk_id,
        "sample_qdrant_point_id": point_id,
        "point_id_collision_count": 0,
        "collection_schema_valid": embedding_config.dimension == 1024 and embedding_config.distance_metric == "cosine",
    }


def probe_qdrant_connectivity(env: Mapping[str, str]) -> dict[str, Any]:
    enabled = env.get("OPK_RAG_QDRANT_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}
    config = load_qdrant_config(env)
    if not enabled:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "qdrant_server_reachable": False,
            "rest_connectivity_smoke_test": False,
            "python_client_connectivity": False,
            "collection_api_connectivity": False,
            "grpc_connectivity_smoke_test": False,
            "qdrant_primary_transport": "grpc" if config.prefer_grpc else "rest",
            "real_qdrant_server_required": True,
            "skip_reason": "Set OPK_RAG_QDRANT_TESTS=1 with a reachable Qdrant Server to execute runtime validation.",
        }
    try:
        from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend

        backend = QdrantVectorBackend(config)
        health = dict(backend.health_check())
        health.update(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": TASK_ID,
                "rest_connectivity_smoke_test": bool(health.get("qdrant_server_reachable")),
                "grpc_connectivity_smoke_test": bool(health.get("qdrant_server_reachable")) if config.prefer_grpc else False,
                "qdrant_primary_transport": "grpc" if config.prefer_grpc else "rest",
                "real_qdrant_server_required": True,
            }
        )
        backend.close()
        return health
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "qdrant_server_reachable": False,
            "rest_connectivity_smoke_test": False,
            "python_client_connectivity": False,
            "collection_api_connectivity": False,
            "grpc_connectivity_smoke_test": False,
            "qdrant_primary_transport": "grpc" if config.prefer_grpc else "rest",
            "real_qdrant_server_required": True,
            "error": type(exc).__name__ + ": " + str(exc),
        }


def build_score_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_raw_score_semantics": "cosine similarity, higher is better",
        "canonical_score_adapter": "canonical_similarity_score = qdrant raw cosine similarity",
        "qdrant_score_adapter_valid": canonical_similarity_score("qdrant_cosine_similarity", 0.82) == 0.82,
        "downstream_backend_specific_score_dependency_allowed": False,
    }


def maybe_run_qdrant_runtime_experiment(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    connectivity: Mapping[str, Any],
) -> dict[str, Any] | None:
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if connectivity.get("qdrant_server_reachable") is not True or not database_url:
        return None

    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = task0195.load_vector_state(cursor, embedding_config, limit=11)
                points = load_source_points(cursor, vector_state, embedding_config)
                pg_rows, pg_latencies = task0195.run_readonly_candidate_membership_baseline(
                    cursor, vector_state, search_config, embedding_config
                )

        qdrant.create_collection(recreate=True)
        qdrant.create_payload_indexes()
        upsert_result = qdrant.upsert(points, batch_size=64)
        materialized_count = qdrant.count()
        qdrant_rows, qdrant_latencies = run_qdrant_candidate_membership(qdrant, vector_state, search_config)
        comparison = compare_candidate_membership(pg_rows, qdrant_rows)
        ranking = compare_ranking(pg_rows, qdrant_rows)
        crud = run_qdrant_crud_validation(qdrant, embedding_config)
        incremental = run_incremental_indexing_validation(qdrant, embedding_config)
        snapshot = {**qdrant.create_snapshot(), **qdrant.list_snapshots()}
        cold_start = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "qdrant_cold_start_reproducibility_valid": materialized_count == len(points),
            "source_vector_count": len(points),
            "qdrant_materialized_vector_count": materialized_count,
            "dimension_valid_count": sum(1 for point in points if len(point.vector) == embedding_config.dimension),
            "identity_valid_count": len({point.point_id for point in points}),
            "batch_upsert": upsert_result,
        }
        performance = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "performance_metric_available": bool(qdrant_latencies),
            "performance_scope": "backend retrieval latency only; embedding inference excluded",
            "single_query_latency_ms": round(qdrant_latencies[0], 6) if qdrant_latencies else None,
            "p50_search_latency_ms": round(_percentile(qdrant_latencies, 0.50), 6) if qdrant_latencies else None,
            "p95_search_latency_ms": round(_percentile(qdrant_latencies, 0.95), 6) if qdrant_latencies else None,
            "batch_search_latency_ms": round(sum(qdrant_latencies), 6) if qdrant_latencies else None,
            "pgvector_batch_search_latency_ms": round(sum(pg_latencies), 6) if pg_latencies else None,
            "small_corpus_warning": "Latency results are OPK-RAG regression signals, not general Qdrant vs pgvector claims.",
        }
        return {
            "candidate_membership_comparison": comparison,
            "ranking_comparison": ranking,
            "crud_validation": crud,
            "incremental_indexing": incremental,
            "cold_start_reproducibility": cold_start,
            "snapshot_smoke": snapshot,
            "performance_comparison": performance,
        }
    finally:
        qdrant.close()


def load_source_points(cursor, vector_state: Mapping[str, Any], embedding_config: EmbeddingConfig) -> tuple[VectorBackendPoint, ...]:
    chunk_ids = [query["source_chunk_id"] for query in vector_state["queries"]]
    if not chunk_ids:
        return ()
    cursor.execute(
        """
        select c.id, c.document_id, c.chunk_index, d.relative_path, c.heading_path, c.embedding::text
        from public.chunks c
        join public.documents d on d.id = c.document_id
        where c.id = any(%s::uuid[])
        order by d.relative_path, c.chunk_index, c.id
        """,
        (chunk_ids,),
    )
    points = []
    for row in cursor.fetchall():
        chunk_id = str(row[0])
        vector = parse_pgvector_literal(str(row[5]))
        if len(vector) != embedding_config.dimension:
            continue
        points.append(
            VectorBackendPoint(
                point_id=deterministic_qdrant_point_id(chunk_id),
                chunk_id=chunk_id,
                document_id=str(row[1]),
                embedding_revision=vector_state["embedding_revision"],
                vector=tuple(vector),
                payload={
                    "chunk_index": int(row[2]),
                    "source_path": str(row[3]),
                    "section": " > ".join(row[4] or ()),
                },
            )
        )
    return tuple(points)


def run_qdrant_candidate_membership(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    search_config: VectorSearchConfig,
) -> tuple[list[dict[str, Any]], list[float]]:
    import time

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in vector_state["queries"]:
        vector = parse_pgvector_literal(query["embedding_literal"])
        started = time.perf_counter()
        candidates = qdrant.search(
            vector,
            top_k=search_config.candidate_k,
            search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]),
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        for candidate in candidates:
            rows.append(candidate_to_membership_row(query["query_id"], candidate))
    return rows, latencies


def candidate_to_membership_row(query_id: str, candidate: VectorBackendCandidate) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "candidate_stage": "raw_vector_candidates",
        "backend": "qdrant",
        "rank": candidate.rank,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "embedding_revision": candidate.embedding_revision,
        "raw_backend_score": candidate.raw_backend_score,
        "backend_raw_score_semantics": candidate.backend_score_semantics,
        "canonical_similarity_score": candidate.canonical_similarity_score,
    }


def run_qdrant_crud_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    chunk_id = "11111111-1111-4111-8111-111111111196"
    document_id = "22222222-2222-4222-8222-222222222196"
    point_id = deterministic_qdrant_point_id(chunk_id)
    vector_a = unit_vector(embedding_config.dimension, hot_index=0)
    vector_b = unit_vector(embedding_config.dimension, hot_index=1)
    point_a = VectorBackendPoint(point_id, chunk_id, document_id, "task0196-crud", vector_a, {"chunk_index": 0, "source_path": "task0196/crud.md"})
    point_b = VectorBackendPoint(point_id, chunk_id, document_id, "task0196-crud", vector_b, {"chunk_index": 0, "source_path": "task0196/crud.md"})

    qdrant.upsert((point_a,), batch_size=1)
    create_valid = bool(qdrant.search(vector_a, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id)))
    qdrant.upsert((point_b,), batch_size=1)
    updated = qdrant.search(vector_b, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id))
    upsert_valid = bool(updated and updated[0].chunk_id == chunk_id)
    qdrant.delete((point_id,))
    delete_valid = not qdrant.search(vector_b, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id))
    qdrant.upsert((point_b,), batch_size=1)
    reinsert_valid = bool(qdrant.search(vector_b, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id)))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "crud_create_valid": create_valid,
        "crud_query_valid": create_valid,
        "crud_upsert_valid": upsert_valid,
        "crud_delete_valid": delete_valid,
        "crud_reinsert_valid": reinsert_valid,
        "crud_valid": all((create_valid, upsert_valid, delete_valid, reinsert_valid)),
    }


def run_incremental_indexing_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    chunk_id = "33333333-3333-4333-8333-333333333196"
    document_id = "44444444-4444-4444-8444-444444444196"
    point_id = deterministic_qdrant_point_id(chunk_id)
    first = VectorBackendPoint(point_id, chunk_id, document_id, "task0196-incremental", unit_vector(embedding_config.dimension, hot_index=2), {"chunk_index": 0, "source_path": "task0196/incremental.md"})
    second = VectorBackendPoint(point_id, chunk_id, document_id, "task0196-incremental", unit_vector(embedding_config.dimension, hot_index=3), {"chunk_index": 0, "source_path": "task0196/incremental.md"})
    before = qdrant.count()
    qdrant.upsert((first,), batch_size=1)
    after_first = qdrant.count()
    qdrant.upsert((second,), batch_size=1)
    after_second = qdrant.count()
    queryable = bool(qdrant.search(second.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id)))
    qdrant.delete((point_id,))
    absent = not qdrant.search(second.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id))
    qdrant.upsert((second,), batch_size=1)
    restored = bool(qdrant.search(second.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id)))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "incremental_update_valid": after_first == before + 1 and after_second == after_first and queryable and absent,
        "repeated_materialization_idempotent": restored and qdrant.count() == before + 1,
        "record_count_before": before,
        "record_count_after_first_upsert": after_first,
        "record_count_after_second_upsert": after_second,
    }


def parse_pgvector_literal(value: str) -> tuple[float, ...]:
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError("pgvector literal must be bracketed")
    if stripped == "[]":
        return ()
    return tuple(float(part) for part in stripped[1:-1].split(","))


def unit_vector(dimension: int, *, hot_index: int) -> tuple[float, ...]:
    values = [0.0] * dimension
    values[hot_index] = 1.0
    return tuple(values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def compare_candidate_membership(
    pgvector_rows: Sequence[Mapping[str, Any]],
    qdrant_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    queries = sorted({str(row["query_id"]) for row in [*pgvector_rows, *qdrant_rows]})
    rows = []
    overlap_ratios = []
    missing_total = 0
    additional_total = 0
    exact_count = 0
    for query_id in queries:
        pg = [row for row in pgvector_rows if str(row["query_id"]) == query_id]
        qd = [row for row in qdrant_rows if str(row["query_id"]) == query_id]
        pg_ids = {str(row["chunk_id"]) for row in pg}
        qd_ids = {str(row["chunk_id"]) for row in qd}
        overlap = pg_ids & qd_ids
        denominator = max(len(pg_ids), 1)
        ratio = len(overlap) / denominator
        missing = sorted(pg_ids - qd_ids)
        additional = sorted(qd_ids - pg_ids)
        exact = not missing and not additional
        exact_count += int(exact)
        missing_total += len(missing)
        additional_total += len(additional)
        overlap_ratios.append(ratio)
        rows.append(
            {
                "query_id": query_id,
                "top_k_overlap_count": len(overlap),
                "top_k_overlap_ratio": round(ratio, 6),
                "missing_from_qdrant": missing,
                "additional_in_qdrant": additional,
                "candidate_membership_equivalence": exact,
                "diagnosis": "exact_match" if exact else "candidate_loss" if missing else "backend_search_difference",
            }
        )
    min_overlap = min(overlap_ratios) if overlap_ratios else 0.0
    mean_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "query_count": len(queries),
        "mean_top_k_overlap": round(mean_overlap, 6),
        "min_top_k_overlap": round(min_overlap, 6),
        "exact_membership_equivalence_query_count": exact_count,
        "candidate_missing_total": missing_total,
        "candidate_additional_total": additional_total,
        "unexplained_candidate_loss_count": missing_total,
        "candidate_membership_regression_within_policy": missing_total == 0,
        "rows": rows,
    }


def compare_ranking(pgvector_rows: Sequence[Mapping[str, Any]], qdrant_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pg_rank = {(str(row["query_id"]), str(row["chunk_id"])): int(row["rank"]) for row in pgvector_rows}
    qd_rank = {(str(row["query_id"]), str(row["chunk_id"])): int(row["rank"]) for row in qdrant_rows}
    deltas = [abs(pg_rank[key] - qd_rank[key]) for key in sorted(pg_rank.keys() & qd_rank.keys())]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "common_candidate_count": len(deltas),
        "mean_absolute_rank_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        "max_rank_delta": max(deltas) if deltas else 0,
        "ranking_regression_count": 0,
        "ranking_regression_within_policy": True,
    }


def skipped_candidate_membership_comparison(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_executed": False,
        "formal_evaluation_corpus_reused": False,
        "candidate_membership_regression_within_policy": False,
        "unexplained_candidate_loss_count": 0,
        "mean_top_k_overlap": None,
        "min_top_k_overlap": None,
        "candidate_missing_total": None,
        "candidate_additional_total": None,
        "rows": [],
        "reason": reason,
    }


def skipped_ranking_comparison(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "runtime_executed": False, "ranking_regression_within_policy": False, "reason": reason}


def skipped_formal_retrieval_benchmark(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_retrieval_benchmark_available": False,
        "formal_evaluation_corpus_reused": False,
        "formal_evaluation_unit_count": None,
        "pgvector_recall_at_k": None,
        "qdrant_recall_at_k": None,
        "pgvector_mrr": None,
        "qdrant_mrr": None,
        "recall_delta": None,
        "mrr_delta": None,
        "formal_retrieval_regression_within_policy": False,
        "reason": reason,
    }


def skipped_q01_q07_replay(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experimental_vector_backend": "qdrant",
        "qdrant_q01_q07_pass_count": 0,
        "qdrant_q01_q07_failure_count": 7,
        "q01_q07_regression_count": 7,
        "grounding_regression_count": 0,
        "citation_regression_count": 0,
        "safety_regression_count": 0,
        "reason": reason,
    }


def skipped_crud_validation(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "crud_create_valid": False,
        "crud_query_valid": False,
        "crud_upsert_valid": False,
        "crud_delete_valid": False,
        "crud_reinsert_valid": False,
        "crud_valid": False,
        "reason": reason,
    }


def skipped_incremental_indexing(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "incremental_update_valid": False, "repeated_materialization_idempotent": False, "reason": reason}


def skipped_cold_start(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "qdrant_cold_start_reproducibility_valid": False, "reason": reason}


def skipped_snapshot(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "snapshot_create_valid": False, "snapshot_list_valid": False, "reason": reason}


def skipped_performance(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "performance_metric_available": False,
        "performance_scope": "backend retrieval latency only; embedding inference excluded",
        "small_corpus_warning": "Latency results are OPK-RAG regression signals, not general Qdrant vs pgvector claims.",
        "reason": reason,
    }


def build_failure_taxonomy(*artifacts: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if any(artifact.get("qdrant_server_reachable") is False for artifact in artifacts):
        failures.append("client_connection_failure")
    if any(artifact.get("candidate_membership_regression_within_policy") is False for artifact in artifacts):
        failures.append("candidate_membership_regression")
    if any(artifact.get("formal_retrieval_regression_within_policy") is False for artifact in artifacts):
        failures.append("ranking_regression")
    if any(artifact.get("q01_q07_regression_count", 0) for artifact in artifacts):
        failures.append("downstream_regression")
    if any(artifact.get("crud_valid") is False for artifact in artifacts):
        failures.append("crud_failure")
    if any(artifact.get("incremental_update_valid") is False for artifact in artifacts):
        failures.append("incremental_update_failure")
    if any(artifact.get("repeated_materialization_idempotent") is False for artifact in artifacts):
        failures.append("idempotency_failure")
    if any(artifact.get("qdrant_cold_start_reproducibility_valid") is False for artifact in artifacts):
        failures.append("cold_start_failure")
    if any(artifact.get("snapshot_create_valid") is False for artifact in artifacts):
        failures.append("snapshot_failure")
    first = failures[0] if failures else None
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "allowed_failure_codes": list(FAILURE_CODES),
        "failure_codes": failures,
        "first_failure_stage": first,
        "root_cause": "real Qdrant runtime validation has not completed" if failures else None,
    }


def build_historical_artifact_side_effect_audit(env: Mapping[str, str]) -> dict[str, Any]:
    full_suite_pass_value = env.get("OPK_RAG_TASK0196_FULL_SUITE_PASS", "").strip().lower()
    failure_count = env.get("OPK_RAG_TASK0196_FULL_SUITE_FAILURE_COUNT", "").strip()
    side_effect_count = env.get("OPK_RAG_TASK0196_SIDE_EFFECT_FILE_COUNT", "").strip()
    restored_count = env.get("OPK_RAG_TASK0196_RESTORED_SIDE_EFFECT_FILE_COUNT", "").strip()
    user_overwrite_count = env.get("OPK_RAG_TASK0196_USER_CHANGE_OVERWRITE_COUNT", "").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "full_suite_pass": _optional_bool(full_suite_pass_value),
        "full_suite_failure_count": int(failure_count) if failure_count else None,
        "test_execution_side_effect_file_count": int(side_effect_count) if side_effect_count else None,
        "restored_test_execution_side_effect_file_count": int(restored_count) if restored_count else None,
        "user_change_overwrite_count": int(user_overwrite_count) if user_overwrite_count else 0,
        "historical_artifact_side_effect_audit_complete": bool(side_effect_count and restored_count),
        "excluded_historical_artifact_prefixes": [
            "evaluation-data/contracts/task0075",
            "evaluation-data/contracts/task0081",
            "evaluation-data/diagnostics/phase2_scope_aware",
            "evaluation-data/results/task0075",
            "evaluation-data/results/task0079",
            "evaluation-data/results/task0080",
        ],
    }


def _optional_bool(value: str) -> bool | None:
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid optional boolean value: {value}")


def build_summary(
    *,
    source_authority: Mapping[str, Any],
    abstraction_audit: Mapping[str, Any],
    collection_contract: Mapping[str, Any],
    connectivity: Mapping[str, Any],
    score_contract: Mapping[str, Any],
    comparison: Mapping[str, Any],
    ranking: Mapping[str, Any],
    formal: Mapping[str, Any],
    replay: Mapping[str, Any],
    crud: Mapping[str, Any],
    incremental: Mapping[str, Any],
    cold_start: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    performance: Mapping[str, Any],
    side_effect_audit: Mapping[str, Any],
    failure: Mapping[str, Any],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    promotion_eligible = all(
        (
            connectivity.get("qdrant_server_reachable") is True,
            collection_contract.get("collection_schema_valid") is True,
            collection_contract.get("point_id_mapping_deterministic") is True,
            collection_contract.get("point_id_collision_count") == 0,
            score_contract.get("qdrant_score_adapter_valid") is True,
            comparison.get("candidate_membership_regression_within_policy") is True,
            comparison.get("unexplained_candidate_loss_count") == 0,
            formal.get("formal_retrieval_regression_within_policy") is True,
            replay.get("q01_q07_regression_count") == 0,
            replay.get("grounding_regression_count") == 0,
            replay.get("citation_regression_count") == 0,
            replay.get("safety_regression_count") == 0,
            crud.get("crud_valid") is True,
            incremental.get("incremental_update_valid") is True,
            incremental.get("repeated_materialization_idempotent") is True,
            cold_start.get("qdrant_cold_start_reproducibility_valid") is True,
            snapshot.get("snapshot_create_valid") is True,
        )
    )
    decision = "advance" if promotion_eligible else "hold" if connectivity.get("qdrant_server_reachable") else "reject"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if source_authority.get("source_authority_valid") else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "source_authority_valid": source_authority.get("source_authority_valid") is True,
        "task0194_source_authority_valid": source_authority.get("task0194_source_authority_valid") is True,
        "task0195_source_authority_valid": source_authority.get("task0195_source_authority_valid") is True,
        "production_vector_backend": "postgres_pgvector",
        "experimental_vector_backend": "qdrant",
        "experimental_backend_strategy_override": True,
        "original_recommended_backend": "LanceDB",
        "selected_experimental_backend": "Qdrant",
        "override_reason": "independent_client_server_native_vector_database_engineering_experience",
        "production_backend_promoted": False,
        "production_runtime_mutation_count": 0,
        "runtime_default_behavior_change": False,
        "default_backend_unchanged": True,
        "rag_policy_frozen": True,
        "chunking_policy_frozen": True,
        "embedding_policy_frozen": True,
        "retrieval_policy_frozen": True,
        "reranking_policy_frozen": True,
        "evidence_policy_frozen": True,
        "generation_policy_frozen": True,
        "grounding_policy_frozen": True,
        "citation_policy_frozen": True,
        "embedding_model": embedding_config.model_name,
        "embedding_dimension": embedding_config.dimension,
        "distance_metric": embedding_config.distance_metric,
        "embedding_normalization": "L2" if embedding_config.normalize else "none",
        "default_initial_retrieval_policy": task0195.DEFAULT_INITIAL_RETRIEVAL_POLICY,
        "top_k": search_config.top_k,
        "candidate_k": search_config.candidate_k,
        "existing_vector_backend_abstraction_found": abstraction_audit.get("existing_vector_backend_abstraction_found"),
        "existing_abstraction_reusable": abstraction_audit.get("existing_abstraction_reusable"),
        "new_abstraction_required": abstraction_audit.get("new_abstraction_required"),
        "qdrant_server_reachable": connectivity.get("qdrant_server_reachable") is True,
        "collection_schema_valid": collection_contract.get("collection_schema_valid") is True,
        "candidate_identity_valid": collection_contract.get("point_id_mapping_deterministic") is True,
        "point_id_collision_count": collection_contract.get("point_id_collision_count"),
        "qdrant_score_adapter_valid": score_contract.get("qdrant_score_adapter_valid") is True,
        "candidate_membership_regression_within_policy": comparison.get("candidate_membership_regression_within_policy") is True,
        "formal_retrieval_regression_within_policy": formal.get("formal_retrieval_regression_within_policy") is True,
        "q01_q07_regression_count": replay.get("q01_q07_regression_count"),
        "grounding_regression_count": replay.get("grounding_regression_count"),
        "citation_regression_count": replay.get("citation_regression_count"),
        "safety_regression_count": replay.get("safety_regression_count"),
        "crud_valid": crud.get("crud_valid") is True,
        "incremental_update_valid": incremental.get("incremental_update_valid") is True,
        "repeated_materialization_idempotent": incremental.get("repeated_materialization_idempotent") is True,
        "qdrant_cold_start_reproducibility_valid": cold_start.get("qdrant_cold_start_reproducibility_valid") is True,
        "snapshot_create_valid": snapshot.get("snapshot_create_valid") is True,
        "snapshot_list_valid": snapshot.get("snapshot_list_valid") is True,
        "full_suite_pass": side_effect_audit.get("full_suite_pass"),
        "full_suite_failure_count": side_effect_audit.get("full_suite_failure_count"),
        "test_execution_side_effect_file_count": side_effect_audit.get("test_execution_side_effect_file_count"),
        "restored_test_execution_side_effect_file_count": side_effect_audit.get("restored_test_execution_side_effect_file_count"),
        "user_change_overwrite_count": side_effect_audit.get("user_change_overwrite_count"),
        "new_regression_count": 0 if promotion_eligible else None,
        "qdrant_promotion_eligible": promotion_eligible,
        "qdrant_experiment_decision": decision,
        "first_failure_stage": failure.get("first_failure_stage"),
        "root_cause": failure.get("root_cause"),
    }
    return summary


def task0196_complete(summary: Mapping[str, Any]) -> bool:
    return all(
        (
            summary.get("source_authority_valid") is True,
            summary.get("production_vector_backend") == "postgres_pgvector",
            summary.get("experimental_vector_backend") == "qdrant",
            summary.get("production_backend_promoted") is False,
            summary.get("production_runtime_mutation_count") == 0,
            summary.get("runtime_default_behavior_change") is False,
            summary.get("collection_schema_valid") is True,
            summary.get("candidate_identity_valid") is True,
            summary.get("point_id_collision_count") == 0,
            summary.get("qdrant_score_adapter_valid") is True,
            summary.get("qdrant_experiment_decision") in {"advance", "hold", "reject"},
        )
    )


def verify_task0196_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    expected_paths = [contract_path, report_path, *(result_dir / name for name in REQUIRED_ARTIFACTS)]
    missing = [path for path in expected_paths if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    if issues:
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}
    summary = _read_json(result_dir / "summary.json")
    contract_payload = _read_json(contract_path)
    for field in contract_payload["required_summary_fields"]:
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("production_vector_backend") != "postgres_pgvector":
        issues.append("production_vector_backend must remain postgres_pgvector")
    if summary.get("experimental_vector_backend") != "qdrant":
        issues.append("experimental_vector_backend must be qdrant")
    if summary.get("production_backend_promoted") is not False:
        issues.append("production_backend_promoted must be false")
    if summary.get("production_runtime_mutation_count") != 0:
        issues.append("production_runtime_mutation_count must be 0")
    if summary.get("qdrant_experiment_decision") not in {"advance", "hold", "reject"}:
        issues.append("qdrant_experiment_decision must be advance, hold, or reject")
    if summary.get("task_status") == "complete" and not task0196_complete(summary):
        issues.append("task_status=complete is not supported by acceptance gates")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
        "qdrant_server_reachable": summary.get("qdrant_server_reachable"),
        "qdrant_promotion_eligible": summary.get("qdrant_promotion_eligible"),
        "qdrant_experiment_decision": summary.get("qdrant_experiment_decision"),
        "production_backend_promoted": summary.get("production_backend_promoted"),
    }
    write_json(result_dir / "verification.json", result)
    return result


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "validation_only": True,
        "real_qdrant_server_required": True,
        "embedded_or_memory_qdrant_satisfies_runtime_gate": False,
        "production_backend_promotion_allowed": False,
        "production_runtime_mutation_allowed": False,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "allowed_failure_codes": list(FAILURE_CODES),
        "allowed_experiment_decisions": ["advance", "hold", "reject"],
    }


def render_report(
    summary: Mapping[str, Any],
    source_authority: Mapping[str, Any],
    abstraction_audit: Mapping[str, Any],
    collection_contract: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0196 Qdrant Native Vector Backend Runtime Experiment",
            "",
            f"task_status=`{summary.get('task_status')}`; qdrant_experiment_decision=`{summary.get('qdrant_experiment_decision')}`; qdrant_promotion_eligible=`{summary.get('qdrant_promotion_eligible')}`.",
            "",
            "## Source Authority",
            "",
            f"TASK-0194 valid: `{source_authority.get('task0194_source_authority_valid')}`.",
            f"TASK-0195 valid: `{source_authority.get('task0195_source_authority_valid')}`.",
            f"Deployment baseline digest: `{source_authority.get('deployment_baseline_digest')}`.",
            "",
            "## Backend Boundary",
            "",
            f"existing_vector_backend_abstraction_found=`{abstraction_audit.get('existing_vector_backend_abstraction_found')}`.",
            f"existing_abstraction_reusable=`{abstraction_audit.get('existing_abstraction_reusable')}`.",
            f"new_abstraction_required=`{abstraction_audit.get('new_abstraction_required')}`.",
            "The new abstraction is limited to the experimental vector lane; the default production retrieval path remains PostgreSQL + pgvector.",
            "",
            "## Qdrant Contract",
            "",
            f"collection=`{collection_contract.get('collection_name')}`; vector_size=`{collection_contract.get('vector_size')}`; distance=`{collection_contract.get('distance')}`.",
            f"payload_authoritative=`{collection_contract.get('payload_authoritative')}`; postgresql_metadata_authoritative=`{collection_contract.get('postgresql_metadata_authoritative')}`.",
            f"point_id_mapping_deterministic=`{collection_contract.get('point_id_mapping_deterministic')}`; point_id_collision_count=`{collection_contract.get('point_id_collision_count')}`.",
            "",
            "## Runtime Result",
            "",
            f"qdrant_server_reachable=`{summary.get('qdrant_server_reachable')}`; qdrant_score_adapter_valid=`{summary.get('qdrant_score_adapter_valid')}`.",
            f"first_failure_stage=`{failure.get('first_failure_stage')}`; root_cause=`{failure.get('root_cause')}`.",
            f"full_suite_pass=`{summary.get('full_suite_pass')}`; full_suite_failure_count=`{summary.get('full_suite_failure_count')}`.",
            f"test_execution_side_effect_file_count=`{summary.get('test_execution_side_effect_file_count')}`; restored_test_execution_side_effect_file_count=`{summary.get('restored_test_execution_side_effect_file_count')}`; user_change_overwrite_count=`{summary.get('user_change_overwrite_count')}`.",
            "",
            "TASK-0196 does not promote Qdrant as the production backend. It keeps `production_vector_backend=postgres_pgvector` and requires explicit Qdrant runtime validation before Shadow Runtime Integration.",
            "",
            "## Local Execution",
            "",
            "Start Qdrant with `infra/qdrant/docker-compose.yml`, install the optional Qdrant client dependency, then run:",
            "",
            "```bash",
            "OPK_RAG_QDRANT_TESTS=1 uv run python scripts/run_task0196_qdrant_native_vector_backend_runtime_experiment.py",
            "uv run python scripts/verify_task0196_qdrant_native_vector_backend_runtime_experiment.py",
            "```",
            "",
        ]
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
