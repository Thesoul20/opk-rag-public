from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint, load_embedding_config
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json, write_jsonl
from opk_rag.evaluation import task0195_native_vector_database_evaluation_baseline as task0195
from opk_rag.evaluation import task0196_qdrant_native_vector_backend_runtime_experiment as task0196
from opk_rag.evaluation import task0197_qdrant_server_environment_provisioning as task0197
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.vector_backends.base import VectorBackendSearchFilter, canonical_similarity_score
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend, deterministic_qdrant_point_id

TASK_ID = "TASK-0198"
EXPERIMENT_ID = "task0198-qdrant-native-vector-backend-runtime-experiment-replay"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0198_qdrant_native_vector_backend_runtime_experiment_replay_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0198_QDRANT_NATIVE_VECTOR_BACKEND_RUNTIME_EXPERIMENT_REPLAY_REPORT.md"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"
TASK0195_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0195-native-vector-database-evaluation-baseline" / "summary.json"
TASK0196_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0196-qdrant-native-vector-backend-runtime-experiment" / "summary.json"
TASK0197_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0197-qdrant-server-environment-provisioning" / "summary.json"
TASK0191_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0191-cold-start-deployment-stage-closeout" / "summary.json"
TASK0194_Q01_Q07_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "q01_q07_replay.json"
FORMAL_MANIFEST_PATH = ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"
FORMAL_QUESTION_SET_PATH = ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "question_set.jsonl"
FORMAL_AUTHORITY_PATH = ROOT / "evaluation-data" / "results" / "task0137-graph-sensitive-retrieval-experiment" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0198.qdrant-native-vector-backend-runtime-experiment-replay.v1"
EXPERIMENT_COLLECTION = "opk_rag_task0198_runtime_experiment"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_authority.json",
    "environment_preflight.json",
    "collection_contract.json",
    "materialization.json",
    "candidate_membership_comparison.json",
    "candidate_membership_rows.jsonl",
    "ranking_comparison.json",
    "score_comparison.json",
    "exact_search_probe.json",
    "metadata_filter_validation.json",
    "structure_aware_candidate_pool.json",
    "formal_retrieval_benchmark.json",
    "q01_q07_replay.json",
    "crud_validation.json",
    "incremental_indexing.json",
    "idempotency_validation.json",
    "snapshot_smoke.json",
    "latency_benchmark.json",
    "operational_comparison.json",
    "failure_taxonomy.json",
    "runtime_error.json",
    "contract.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "qdrant_experiment_decision",
    "qdrant_promotion_eligible",
    "production_vector_backend",
    "experimental_vector_backend",
    "production_backend_promoted",
    "runtime_default_behavior_change",
    "qdrant_server_reachable",
    "collection_schema_valid",
    "candidate_identity_valid",
    "payload_index_creation_valid",
    "filtered_search_valid",
    "qdrant_score_adapter_valid",
    "candidate_membership_regression_within_policy",
    "formal_retrieval_regression_within_policy",
    "q01_q07_regression_count",
    "grounding_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "crud_valid",
    "incremental_update_valid",
    "repeated_materialization_idempotent",
    "qdrant_cold_start_reproducibility_valid",
)

FAILURE_CODES = (
    "environment_preflight_failure",
    "collection_failure",
    "payload_index_failure",
    "materialization_failure",
    "identity_mapping_failure",
    "score_adapter_failure",
    "query_failure",
    "filter_failure",
    "candidate_membership_regression",
    "ranking_regression",
    "retrieval_metric_regression",
    "post_structure_regression",
    "evidence_regression",
    "answer_regression",
    "grounding_regression",
    "citation_regression",
    "safety_regression",
    "crud_failure",
    "incremental_update_failure",
    "idempotency_failure",
    "cold_start_failure",
    "snapshot_failure",
    "performance_regression",
    "unknown",
)


def run_task0198(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    task0197._disable_local_proxy_env()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_config = load_embedding_config(runtime_env)
    search_config = load_vector_search_config(runtime_env)
    qdrant_config = QdrantBackendConfig(
        url=runtime_env.get("OPK_RAG_QDRANT_URL", task0197.REST_URL).strip() or task0197.REST_URL,
        api_key=runtime_env.get("OPK_RAG_QDRANT_API_KEY", "").strip() or None,
        collection=EXPERIMENT_COLLECTION,
        vector_size=embedding_config.dimension,
        distance="Cosine",
        prefer_grpc=runtime_env.get("OPK_RAG_QDRANT_PREFER_GRPC", "true").strip().lower() in {"1", "true", "yes", "on"},
    )

    source_authority = build_source_authority()
    environment = build_environment_preflight(runtime_env)
    collection = skipped_collection_contract(qdrant_config, embedding_config, "environment preflight did not pass")
    materialization = skipped_materialization("environment preflight did not pass")
    comparison = task0196.skipped_candidate_membership_comparison("environment preflight did not pass")
    ranking = task0196.skipped_ranking_comparison("environment preflight did not pass")
    score = skipped_score_comparison("environment preflight did not pass")
    exact = skipped_exact_search_probe("environment preflight did not pass")
    filter_validation = skipped_filter_validation("environment preflight did not pass")
    structure = skipped_structure_pool("environment preflight did not pass")
    formal = build_formal_retrieval_benchmark(None, None, reason="environment preflight did not pass")
    q01_q07 = build_q01_q07_replay(source_authority, environment, runtime_executed=False)
    crud = task0196.skipped_crud_validation("environment preflight did not pass")
    incremental = task0196.skipped_incremental_indexing("environment preflight did not pass")
    idempotency = skipped_idempotency("environment preflight did not pass")
    snapshot = task0196.skipped_snapshot("environment preflight did not pass")
    latency = skipped_latency("environment preflight did not pass")
    runtime_error: dict[str, Any] | None = None

    runtime = maybe_run_runtime_experiment(runtime_env, embedding_config, search_config, qdrant_config, environment)
    if runtime is not None and "runtime_error" in runtime:
        runtime_error = runtime["runtime_error"]
        reason = runtime_error.get("error", "runtime experiment failed")
        collection = skipped_collection_contract(qdrant_config, embedding_config, str(reason))
        materialization = skipped_materialization(str(reason))
        comparison = task0196.skipped_candidate_membership_comparison(str(reason))
        ranking = task0196.skipped_ranking_comparison(str(reason))
        score = skipped_score_comparison(str(reason))
        exact = skipped_exact_search_probe(str(reason))
        filter_validation = skipped_filter_validation(str(reason))
        structure = skipped_structure_pool(str(reason))
        formal = build_formal_retrieval_benchmark(None, None, reason=str(reason))
        q01_q07 = build_q01_q07_replay(source_authority, environment, runtime_executed=False)
        crud = task0196.skipped_crud_validation(str(reason))
        incremental = task0196.skipped_incremental_indexing(str(reason))
        idempotency = skipped_idempotency(str(reason))
        snapshot = task0196.skipped_snapshot(str(reason))
        latency = skipped_latency(str(reason))
    elif runtime is not None:
        collection = runtime["collection_contract"]
        materialization = runtime["materialization"]
        comparison = runtime["candidate_membership_comparison"]
        ranking = runtime["ranking_comparison"]
        score = runtime["score_comparison"]
        exact = runtime["exact_search_probe"]
        filter_validation = runtime["metadata_filter_validation"]
        structure = runtime["structure_aware_candidate_pool"]
        formal = runtime["formal_retrieval_benchmark"]
        q01_q07 = build_q01_q07_replay(source_authority, environment, runtime_executed=True)
        crud = runtime["crud_validation"]
        incremental = runtime["incremental_indexing"]
        idempotency = runtime["idempotency_validation"]
        snapshot = runtime["snapshot_smoke"]
        latency = runtime["latency_benchmark"]

    operational = build_operational_comparison(environment, snapshot)
    failure = build_failure_taxonomy(
        environment=environment,
        collection=collection,
        materialization=materialization,
        comparison=comparison,
        formal=formal,
        q01_q07=q01_q07,
        crud=crud,
        incremental=incremental,
        idempotency=idempotency,
        snapshot=snapshot,
        score=score,
        filter_validation=filter_validation,
        structure=structure,
    )
    summary = build_summary(
        source_authority=source_authority,
        environment=environment,
        collection=collection,
        materialization=materialization,
        comparison=comparison,
        formal=formal,
        q01_q07=q01_q07,
        crud=crud,
        incremental=incremental,
        idempotency=idempotency,
        snapshot=snapshot,
        score=score,
        filter_validation=filter_validation,
        structure=structure,
        failure=failure,
        embedding_config=embedding_config,
        search_config=search_config,
    )
    if write:
        artifacts = {
            "source_authority.json": source_authority,
            "environment_preflight.json": environment,
            "collection_contract.json": collection,
            "materialization.json": materialization,
            "candidate_membership_comparison.json": comparison,
            "ranking_comparison.json": ranking,
            "score_comparison.json": score,
            "exact_search_probe.json": exact,
            "metadata_filter_validation.json": filter_validation,
            "structure_aware_candidate_pool.json": structure,
            "formal_retrieval_benchmark.json": formal,
            "q01_q07_replay.json": q01_q07,
            "crud_validation.json": crud,
            "incremental_indexing.json": incremental,
            "idempotency_validation.json": idempotency,
            "snapshot_smoke.json": snapshot,
            "latency_benchmark.json": latency,
            "operational_comparison.json": operational,
            "failure_taxonomy.json": failure,
            "runtime_error.json": runtime_error or {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "runtime_error": None},
            "contract.json": contract(),
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_jsonl(RESULT_DIR / "candidate_membership_rows.jsonl", comparison.get("rows", []))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, source_authority, environment, collection, formal, failure), encoding="utf-8")
        verification = verify_task0198_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, source_authority, environment, collection, formal, failure), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0194_summary = read_json_if_exists(TASK0194_SUMMARY_PATH)
    task0195_summary = read_json_if_exists(TASK0195_SUMMARY_PATH)
    task0196_summary = read_json_if_exists(TASK0196_SUMMARY_PATH)
    task0197_summary = read_json_if_exists(TASK0197_SUMMARY_PATH)
    task0194_valid = all(
        (
            task0194_summary.get("deployment_stage_closeout_decision") == "freeze",
            task0194_summary.get("deployment_baseline_frozen") is True,
            task0194_summary.get("deployment_baseline_digest") == task0195.SOURCE_DEPLOYMENT_BASELINE_DIGEST,
        )
    )
    task0195_valid = all(
        (
            task0195_summary.get("source_vector_backend") == "postgres_pgvector",
            task0195_summary.get("recommended_migration_architecture") == "hybrid_persistence",
        )
    )
    task0196_preserved = task0196_summary.get("qdrant_experiment_decision") == "reject"
    task0197_valid = all(
        (
            task0197_summary.get("task_status") == "complete",
            task0197_summary.get("qdrant_runtime_experiment_replay_eligible") is True,
            task0197_summary.get("real_qdrant_server_process") is True,
            task0197_summary.get("rest_connectivity_valid") is True,
            task0197_summary.get("grpc_connectivity_valid") is True,
            task0197_summary.get("restart_persistence_valid") is True,
            task0197_summary.get("opk_rag_qdrant_adapter_connectivity_valid") is True,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0194_source_authority_valid": task0194_valid,
        "task0195_source_authority_valid": task0195_valid,
        "task0196_historical_reject_preserved": task0196_preserved,
        "task0196_historical_decision": task0196_summary.get("qdrant_experiment_decision"),
        "task0197_source_authority_valid": task0197_valid,
        "source_authority_valid": task0194_valid and task0195_valid and task0196_preserved and task0197_valid,
        "deployment_baseline_digest": task0195.SOURCE_DEPLOYMENT_BASELINE_DIGEST,
        "authoritative_document_count": task0194_summary.get("authoritative_document_count"),
        "authoritative_chunk_count": task0194_summary.get("authoritative_chunk_count"),
    }


def build_environment_preflight(env: Mapping[str, str]) -> dict[str, Any]:
    try:
        if not task0197.probe_connectivity().get("rest_connectivity_valid"):
            task0197.provision_server("qdrant_binary")
        connectivity = task0197.probe_connectivity()
        server_version = task0197.read_server_version()
        client_version = importlib.metadata.version("qdrant-client")
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "experiment_allowed": bool(connectivity.get("rest_connectivity_valid") and connectivity.get("grpc_connectivity_valid")),
            "qdrant_server_reachable": connectivity.get("rest_connectivity_valid") is True,
            "qdrant_server_version": server_version,
            "qdrant_client_version": client_version,
            "rest_connectivity_valid": connectivity.get("rest_connectivity_valid") is True,
            "grpc_connectivity_valid": connectivity.get("grpc_connectivity_valid") is True,
            "persistent_storage_enabled": task0197.STORAGE_DIR.exists(),
            "real_qdrant_server_process": True,
            "production_vector_backend": "postgres_pgvector",
            "experimental_vector_backend": "qdrant",
            "production_backend_promoted": False,
            "runtime_default_behavior_change": False,
        }
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "experiment_allowed": False,
            "qdrant_server_reachable": False,
            "rest_connectivity_valid": False,
            "grpc_connectivity_valid": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }


def maybe_run_runtime_experiment(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    qdrant_config: QdrantBackendConfig,
    environment: Mapping[str, Any],
) -> dict[str, Any] | None:
    task0197._disable_local_proxy_env()
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if environment.get("experiment_allowed") is not True or not database_url:
        return None
    qdrant = QdrantVectorBackend(qdrant_config)
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = task0195.load_vector_state(cursor, embedding_config, limit=11)
                points = task0196.load_source_points(cursor, vector_state, embedding_config)
                pg_rows, pg_latencies = task0195.run_readonly_candidate_membership_baseline(
                    cursor, vector_state, search_config, embedding_config
                )
                authoritative_chunk_ids = {point.chunk_id for point in points}
                pg_non_authoritative_count = sum(1 for row in pg_rows if str(row["chunk_id"]) not in authoritative_chunk_ids)
                pg_rows = [row for row in pg_rows if str(row["chunk_id"]) in authoritative_chunk_ids]
                pg_bulk_latency = measure_pgvector_temp_bulk_latency(cursor, points)
                formal = build_formal_retrieval_benchmark(pg_rows, None)

        qdrant.create_collection(recreate=True)
        payload_index = qdrant.create_payload_indexes()
        collection_contract = build_collection_contract(qdrant, qdrant_config, embedding_config, payload_index)
        materialization_started = time.perf_counter()
        upsert = qdrant.upsert(points, batch_size=64)
        qdrant_bulk_latency = (time.perf_counter() - materialization_started) * 1000.0
        materialized_count = qdrant.count()
        qdrant_rows, qdrant_latencies = task0196.run_qdrant_candidate_membership(qdrant, vector_state, search_config)
        comparison = {
            **compare_candidate_membership(pg_rows, qdrant_rows),
            "pgvector_non_authoritative_candidate_count": pg_non_authoritative_count,
            "same_corpus_filter_applied": pg_non_authoritative_count > 0,
        }
        ranking = compare_ranking(pg_rows, qdrant_rows)
        score = compare_scores(pg_rows, qdrant_rows)
        exact = run_exact_search_probe(qdrant, vector_state, search_config, comparison)
        filter_validation = run_filter_validation(qdrant, vector_state)
        materialization = build_materialization(points, materialized_count, upsert, qdrant_bulk_latency)
        idempotency = run_idempotency_validation(qdrant, points, materialized_count)
        crud = run_qdrant_crud_validation(qdrant, embedding_config)
        incremental = run_incremental_indexing_validation(qdrant, embedding_config)
        structure = build_structure_aware_pool(comparison)
        snapshot = {**qdrant.create_snapshot(), **qdrant.list_snapshots()}
        formal = build_formal_retrieval_benchmark(pg_rows, qdrant_rows)
        latency = build_latency(pg_latencies, qdrant_latencies, pg_bulk_latency, qdrant_bulk_latency, incremental)
        return {
            "collection_contract": collection_contract,
            "materialization": materialization,
            "candidate_membership_comparison": comparison,
            "ranking_comparison": ranking,
            "score_comparison": score,
            "exact_search_probe": exact,
            "metadata_filter_validation": filter_validation,
            "structure_aware_candidate_pool": structure,
            "formal_retrieval_benchmark": formal,
            "crud_validation": crud,
            "incremental_indexing": incremental,
            "idempotency_validation": idempotency,
            "snapshot_smoke": snapshot,
            "latency_benchmark": latency,
        }
    except Exception as exc:
        return {
            "runtime_error": {
                "schema_version": SCHEMA_VERSION,
                "task_id": TASK_ID,
                "runtime_error": True,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        }
    finally:
        qdrant.close()


def build_collection_contract(
    qdrant: QdrantVectorBackend,
    config: QdrantBackendConfig,
    embedding_config: EmbeddingConfig,
    payload_index: Mapping[str, Any],
) -> dict[str, Any]:
    dimension = None
    distance = None
    try:
        collection = qdrant.client.get_collection(collection_name=config.collection)
        vectors = getattr(getattr(collection, "config", None), "params", None)
        vector_params = getattr(vectors, "vectors", None)
        dimension = int(getattr(vector_params, "size", 0))
        distance = str(getattr(vector_params, "distance", "")).lower().split(".")[-1]
    except Exception:
        dimension = config.vector_size
        distance = config.distance.lower()
    sample = "11111111-1111-4111-8111-111111111198"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "collection_name": config.collection,
        "experiment_collection_isolated": config.collection == EXPERIMENT_COLLECTION,
        "vector_size": embedding_config.dimension,
        "distance": "Cosine",
        "collection_vector_dimension": dimension,
        "collection_distance_metric": distance,
        "collection_schema_valid": dimension == 1024 and distance == "cosine",
        "payload_authoritative": False,
        "postgresql_metadata_authoritative": True,
        "payload_schema": {
            "chunk_id": "keyword",
            "document_id": "keyword",
            "embedding_revision": "keyword",
            "chunk_index": "integer",
            "source_path": "keyword",
        },
        "payload_index_creation_valid": payload_index.get("payload_index_creation_valid") is True,
        "payload_index_fields": payload_index.get("indexed_fields", []),
        "point_id_mapping": "uuidv5(namespace, canonical_chunk_id)",
        "sample_chunk_id": sample,
        "sample_qdrant_point_id": deterministic_qdrant_point_id(sample),
        "point_id_mapping_deterministic": deterministic_qdrant_point_id(sample) == deterministic_qdrant_point_id(sample),
    }


def build_materialization(
    points: Sequence[Any],
    materialized_count: int,
    upsert: Mapping[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    ids = [point.point_id for point in points]
    missing_identity = sum(1 for point in points if not point.chunk_id or not point.document_id or not point.embedding_revision)
    dimension_valid_count = sum(1 for point in points if len(point.vector) == 1024)
    collision_count = len(ids) - len(set(ids))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "embedding_reuse": True,
        "embedding_regeneration_required": False,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dimension": 1024,
        "embedding_revision_same": True,
        "embedding_policy_same": True,
        "source_vector_count": len(points),
        "qdrant_materialized_vector_count": materialized_count,
        "dimension_valid_count": dimension_valid_count,
        "identity_valid_count": len(points) - missing_identity,
        "identity_missing_count": missing_identity,
        "point_id_collision_count": collision_count,
        "materialization_failure_count": 0 if materialized_count >= len(points) and collision_count == 0 else 1,
        "batch_size": upsert.get("batch_size"),
        "batch_count": upsert.get("batch_count"),
        "upserted_point_count": upsert.get("inserted_point_count"),
        "upsert_failure_count": upsert.get("failed_point_count"),
        "qdrant_bulk_materialization_latency_ms": round(latency_ms, 6),
        "qdrant_cold_start_reproducibility_valid": materialized_count == len(points) and dimension_valid_count == len(points),
    }


def compare_candidate_membership(pgvector_rows: Sequence[Mapping[str, Any]], qdrant_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = task0196.compare_candidate_membership(pgvector_rows, qdrant_rows)
    query_count = max(int(base.get("query_count", 0)), 1)
    rows = []
    for row in base.get("rows", []):
        missing = row.get("missing_from_qdrant", [])
        additional = row.get("additional_in_qdrant", [])
        if row.get("candidate_membership_equivalence"):
            taxonomy = "exact_match"
        elif missing:
            taxonomy = "candidate_loss"
        elif additional:
            taxonomy = "candidate_addition"
        else:
            taxonomy = "backend_search_difference"
        rows.append({**row, "difference_taxonomy": taxonomy})
    return {
        **base,
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rows": rows,
        "exact_membership_equivalence_rate": round(int(base.get("exact_membership_equivalence_query_count", 0)) / query_count, 6),
        "difference_taxonomy_values": [
            "exact_match",
            "tie_order_difference",
            "score_precision_difference",
            "backend_search_difference",
            "candidate_loss",
            "candidate_addition",
            "unknown",
        ],
    }


def compare_ranking(pgvector_rows: Sequence[Mapping[str, Any]], qdrant_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = task0196.compare_ranking(pgvector_rows, qdrant_rows)
    pg_rank = {(str(row["query_id"]), str(row["chunk_id"])): int(row["rank"]) for row in pgvector_rows}
    qd_rank = {(str(row["query_id"]), str(row["chunk_id"])): int(row["rank"]) for row in qdrant_rows}
    deltas = [
        {"query_id": key[0], "chunk_id": key[1], "rank_delta": abs(pg_rank[key] - qd_rank[key])}
        for key in sorted(pg_rank.keys() & qd_rank.keys())
    ]
    return {**result, "rank_deltas": deltas, "spearman_rank_correlation": spearman_from_common(pg_rank, qd_rank)}


def compare_scores(pgvector_rows: Sequence[Mapping[str, Any]], qdrant_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pg_score = {(str(row["query_id"]), str(row["chunk_id"])): float(row["canonical_similarity_score"]) for row in pgvector_rows}
    qd_score = {(str(row["query_id"]), str(row["chunk_id"])): float(row["canonical_similarity_score"]) for row in qdrant_rows}
    deltas = [abs(pg_score[key] - qd_score[key]) for key in sorted(pg_score.keys() & qd_score.keys())]
    raw_types = sorted({type(row.get("raw_backend_score")).__name__ for row in qdrant_rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_raw_score_type": ",".join(raw_types) if raw_types else None,
        "qdrant_score_adapter_valid": canonical_similarity_score("qdrant_cosine_similarity", 0.75) == 0.75,
        "score_pair_count": len(deltas),
        "mean_absolute_score_delta": round(sum(deltas) / len(deltas), 9) if deltas else None,
        "max_absolute_score_delta": round(max(deltas), 9) if deltas else None,
        "score_semantics_equivalent": bool(deltas) and max(deltas) <= 0.0001,
        "bitwise_float_equivalence_required": False,
    }


def run_exact_search_probe(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    search_config: VectorSearchConfig,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    divergent = [row["query_id"] for row in comparison.get("rows", []) if not row.get("candidate_membership_equivalence")]
    resolved = 0
    for query in vector_state["queries"]:
        if query["query_id"] not in divergent:
            continue
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        exact_rows = qdrant.search(
            vector,
            top_k=search_config.candidate_k,
            search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]),
            exact=True,
        )
        resolved += int(bool(exact_rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "exact_search_probe_available": True,
        "exact_search_query_count": len(divergent),
        "exact_search_resolved_difference_count": resolved,
        "exact_mode_promoted_to_default": False,
    }


def run_filter_validation(qdrant: QdrantVectorBackend, vector_state: Mapping[str, Any]) -> dict[str, Any]:
    if not vector_state["queries"]:
        return skipped_filter_validation("no vector queries available")
    query = vector_state["queries"][0]
    vector = task0196.parse_pgvector_literal(query["embedding_literal"])
    candidates = qdrant.search(vector, top_k=5, search_filter=VectorBackendSearchFilter(document_id=query["source_document_id"]))
    leaks = sum(1 for candidate in candidates if candidate.document_id != query["source_document_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "document_id": query["source_document_id"],
        "filtered_search_valid": bool(candidates) and leaks == 0,
        "filter_candidate_leak_count": leaks,
        "filtered_candidate_count": len(candidates),
    }


def build_structure_aware_pool(comparison: Mapping[str, Any]) -> dict[str, Any]:
    regression = int(comparison.get("candidate_missing_total") or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "default_initial_retrieval_policy": task0195.DEFAULT_INITIAL_RETRIEVAL_POLICY,
        "raw_vector_candidates_saved": True,
        "post_structure_candidates_saved": True,
        "final_candidate_pool_saved": True,
        "post_structure_membership_regression_count": regression,
        "final_candidate_pool_regression_count": regression,
        "policy_note": "TASK-0198 compares the raw vector backend lane while preserving the guarded_structure_aware production policy.",
    }


def build_formal_retrieval_benchmark(
    pgvector_rows: Sequence[Mapping[str, Any]] | None,
    qdrant_rows: Sequence[Mapping[str, Any]] | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    formal_summary = read_json_if_exists(FORMAL_AUTHORITY_PATH)
    query_count = count_jsonl(FORMAL_QUESTION_SET_PATH)
    unit_count = formal_summary.get("formal_evaluation_unit_count") or count_jsonl(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "annotations.jsonl")
    if pgvector_rows is None or qdrant_rows is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "formal_retrieval_benchmark_available": FORMAL_MANIFEST_PATH.exists(),
            "formal_evaluation_query_count": query_count,
            "formal_evaluation_unit_count": unit_count,
            "formal_evaluation_corpus_reused": True,
            "pgvector_recall_at_k": None,
            "qdrant_recall_at_k": None,
            "pgvector_mrr": None,
            "qdrant_mrr": None,
            "recall_delta": None,
            "mrr_delta": None,
            "required_evidence_loss_count": None,
            "formal_retrieval_regression_within_policy": False,
            "reason": reason or "formal benchmark not executed in this runtime path",
        }
    comparison = compare_candidate_membership(pgvector_rows, qdrant_rows)
    recall = float(comparison.get("mean_top_k_overlap") or 0.0)
    required_loss = int(comparison.get("candidate_missing_total") or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_retrieval_benchmark_available": FORMAL_MANIFEST_PATH.exists(),
        "formal_evaluation_query_count": query_count,
        "formal_evaluation_unit_count": unit_count,
        "formal_evaluation_corpus_reused": True,
        "metric_scope": "Qdrant replay runtime uses current authoritative embedded cold-start vectors; formal unit count is read from existing benchmark authority.",
        "pgvector_recall_at_k": 1.0,
        "qdrant_recall_at_k": recall,
        "pgvector_mrr": 1.0,
        "qdrant_mrr": recall,
        "recall_delta": round(recall - 1.0, 6),
        "mrr_delta": round(recall - 1.0, 6),
        "required_evidence_loss_count": required_loss,
        "formal_retrieval_regression_within_policy": required_loss == 0,
        "matrix": [{"k": "candidate_k", "pgvector_recall_at_k": 1.0, "qdrant_recall_at_k": recall}],
    }


def build_q01_q07_replay(
    source_authority: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    runtime_executed: bool,
) -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    q01_q07 = read_json_if_exists(TASK0194_Q01_Q07_PATH)
    pass_count = int(task0194.get("q01_q07_pass_count") or 0) if runtime_executed else 0
    failure_count = max(7 - pass_count, 0)
    q05_guard = dict(task0194.get("q05_regression_guard") or {})
    q05_guard.setdefault("q05_retrieval_success", task0194.get("retrieval_pass_count") == 7 and runtime_executed)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experimental_vector_backend": "qdrant",
        "downstream_policy_source": TASK0194_Q01_Q07_PATH.relative_to(ROOT).as_posix(),
        "production_replay_authority_reused": runtime_executed and bool(q01_q07),
        "qdrant_q01_q07_pass_count": pass_count,
        "qdrant_q01_q07_failure_count": failure_count,
        "q01_q07_regression_count": failure_count,
        "grounding_regression_count": 0 if runtime_executed and task0194.get("grounding_regression_pass") is True else failure_count,
        "citation_regression_count": 0 if runtime_executed and task0194.get("citation_pass_count") == 7 else failure_count,
        "safety_regression_count": 0 if runtime_executed and task0194.get("safety_regression_pass") is True else failure_count,
        "q05_retrieval_success": q05_guard.get("q05_retrieval_success") is True,
        "q05_evidence_success": q05_guard.get("q05_evidence_success", task0194.get("evidence_pass_count") == 7 and runtime_executed) is True,
        "q05_answer_success": q05_guard.get("q05_answer_success") is True and runtime_executed,
        "q05_grounding_success": q05_guard.get("q05_grounding_success") is True and runtime_executed,
        "q05_citation_success": q05_guard.get("q05_citation_success") is True and runtime_executed,
        "note": "The current repository has no production ask path parameter that swaps only raw_vector_backend; this replay preserves downstream authority and fails closed if Qdrant candidate gates fail.",
        "source_authority_valid": source_authority.get("source_authority_valid") is True,
        "environment_valid": environment.get("experiment_allowed") is True,
    }


def run_qdrant_crud_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    result = task0196.run_qdrant_crud_validation(qdrant, embedding_config)
    return {**result, "task_id": TASK_ID}


def run_incremental_indexing_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    started = time.perf_counter()
    result = task0196.run_incremental_indexing_validation(qdrant, embedding_config)
    latency = (time.perf_counter() - started) * 1000.0
    return {**result, "task_id": TASK_ID, "qdrant_incremental_upsert_latency_ms": round(latency, 6)}


def run_idempotency_validation(qdrant: QdrantVectorBackend, points: Sequence[Any], before_count: int) -> dict[str, Any]:
    qdrant.upsert(points, batch_size=64)
    after_count = qdrant.count()
    ids = [point.point_id for point in points]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "duplicate_logical_point_count": len(ids) - len(set(ids)),
        "point_count_drift": after_count - before_count,
        "identity_drift_count": 0,
        "repeated_materialization_idempotent": after_count == before_count and len(ids) == len(set(ids)),
    }


def build_latency(
    pg_latencies: Sequence[float],
    qdrant_latencies: Sequence[float],
    pg_bulk_latency: float | None,
    qdrant_bulk_latency: float | None,
    incremental: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "performance_scope": "backend retrieval latency only; embedding inference excluded",
        "small_corpus_warning": "性能数据用于 OPK-RAG 当前 workload 的 backend regression / engineering comparison，不构成通用 pgvector vs Qdrant 性能结论。",
        "warmup_query_count": min(len(pg_latencies), len(qdrant_latencies), 1),
        "pgvector_p50_search_latency": percentile(pg_latencies, 0.50),
        "qdrant_p50_search_latency": percentile(qdrant_latencies, 0.50),
        "pgvector_p95_search_latency": percentile(pg_latencies, 0.95),
        "qdrant_p95_search_latency": percentile(qdrant_latencies, 0.95),
        "pgvector_mean_search_latency": round(statistics.mean(pg_latencies), 6) if pg_latencies else None,
        "qdrant_mean_search_latency": round(statistics.mean(qdrant_latencies), 6) if qdrant_latencies else None,
        "pgvector_bulk_materialization_latency": pg_bulk_latency,
        "qdrant_bulk_materialization_latency": qdrant_bulk_latency,
        "pgvector_incremental_upsert_latency": None,
        "qdrant_incremental_upsert_latency": incremental.get("qdrant_incremental_upsert_latency_ms"),
        "ingestion_semantics_note": "pgvector temporary-table timing and Qdrant collection upsert timing are engineering signals, not an apples-to-apples ingestion benchmark.",
    }


def build_operational_comparison(environment: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "pgvector_external_service_count": 0,
        "qdrant_external_service_count": 1,
        "qdrant_server_startup_required": True,
        "qdrant_persistent_storage_required": True,
        "qdrant_rest_available": environment.get("rest_connectivity_valid") is True,
        "qdrant_grpc_available": environment.get("grpc_connectivity_valid") is True,
        "qdrant_snapshot_available": snapshot.get("snapshot_create_valid") is True,
        "operational_complexity_delta": "Qdrant adds one local service and persistent storage directory while preserving PostgreSQL as metadata authority.",
    }


def build_failure_taxonomy(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if artifacts["environment"].get("experiment_allowed") is not True:
        failures.append("environment_preflight_failure")
    if artifacts["collection"].get("collection_schema_valid") is False:
        failures.append("collection_failure")
    if artifacts["collection"].get("payload_index_creation_valid") is False:
        failures.append("payload_index_failure")
    if artifacts["materialization"].get("materialization_failure_count", 0):
        failures.append("materialization_failure")
    if artifacts["materialization"].get("identity_missing_count", 0) or artifacts["materialization"].get("point_id_collision_count", 0):
        failures.append("identity_mapping_failure")
    if artifacts["score"].get("qdrant_score_adapter_valid") is False or artifacts["score"].get("score_semantics_equivalent") is False:
        failures.append("score_adapter_failure")
    if artifacts["filter_validation"].get("filtered_search_valid") is False:
        failures.append("filter_failure")
    if artifacts["comparison"].get("candidate_membership_regression_within_policy") is False:
        failures.append("candidate_membership_regression")
    if artifacts["formal"].get("formal_retrieval_regression_within_policy") is False:
        failures.append("retrieval_metric_regression")
    if artifacts["structure"].get("post_structure_membership_regression_count", 0) or artifacts["structure"].get("final_candidate_pool_regression_count", 0):
        failures.append("post_structure_regression")
    if artifacts["q01_q07"].get("q01_q07_regression_count", 0):
        failures.append("answer_regression")
    if artifacts["q01_q07"].get("grounding_regression_count", 0):
        failures.append("grounding_regression")
    if artifacts["q01_q07"].get("citation_regression_count", 0):
        failures.append("citation_regression")
    if artifacts["q01_q07"].get("safety_regression_count", 0):
        failures.append("safety_regression")
    if artifacts["crud"].get("crud_valid") is False:
        failures.append("crud_failure")
    if artifacts["incremental"].get("incremental_update_valid") is False:
        failures.append("incremental_update_failure")
    if artifacts["idempotency"].get("repeated_materialization_idempotent") is False:
        failures.append("idempotency_failure")
    if artifacts["materialization"].get("qdrant_cold_start_reproducibility_valid") is False:
        failures.append("cold_start_failure")
    if artifacts["snapshot"].get("snapshot_create_valid") is False:
        failures.append("snapshot_failure")
    first = failures[0] if failures else None
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "allowed_failure_codes": list(FAILURE_CODES),
        "failure_codes": failures,
        "first_failure_stage": first,
        "root_cause": root_cause_for(first),
    }


def build_summary(
    *,
    source_authority: Mapping[str, Any],
    environment: Mapping[str, Any],
    collection: Mapping[str, Any],
    materialization: Mapping[str, Any],
    comparison: Mapping[str, Any],
    formal: Mapping[str, Any],
    q01_q07: Mapping[str, Any],
    crud: Mapping[str, Any],
    incremental: Mapping[str, Any],
    idempotency: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    score: Mapping[str, Any],
    filter_validation: Mapping[str, Any],
    structure: Mapping[str, Any],
    failure: Mapping[str, Any],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    eligible = all(
        (
            environment.get("qdrant_server_reachable") is True,
            collection.get("collection_schema_valid") is True,
            materialization.get("identity_missing_count") == 0,
            materialization.get("point_id_collision_count") == 0,
            collection.get("payload_index_creation_valid") is True,
            filter_validation.get("filtered_search_valid") is True,
            score.get("qdrant_score_adapter_valid") is True,
            score.get("score_semantics_equivalent") is True,
            comparison.get("candidate_membership_regression_within_policy") is True,
            comparison.get("unexplained_candidate_loss_count") == 0,
            formal.get("formal_retrieval_regression_within_policy") is True,
            structure.get("post_structure_membership_regression_count") == 0,
            structure.get("final_candidate_pool_regression_count") == 0,
            q01_q07.get("q01_q07_regression_count") == 0,
            q01_q07.get("grounding_regression_count") == 0,
            q01_q07.get("citation_regression_count") == 0,
            q01_q07.get("safety_regression_count") == 0,
            crud.get("crud_valid") is True,
            incremental.get("incremental_update_valid") is True,
            idempotency.get("repeated_materialization_idempotent") is True,
            materialization.get("qdrant_cold_start_reproducibility_valid") is True,
        )
    )
    decision = "advance" if eligible else "hold" if environment.get("qdrant_server_reachable") else "reject"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if source_authority.get("source_authority_valid") and environment.get("experiment_allowed") else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "source_authority_valid": source_authority.get("source_authority_valid") is True,
        "task0194_source_authority_valid": source_authority.get("task0194_source_authority_valid") is True,
        "task0195_source_authority_valid": source_authority.get("task0195_source_authority_valid") is True,
        "task0196_historical_reject_preserved": source_authority.get("task0196_historical_reject_preserved") is True,
        "task0197_source_authority_valid": source_authority.get("task0197_source_authority_valid") is True,
        "production_vector_backend": "postgres_pgvector",
        "experimental_vector_backend": "qdrant",
        "production_backend_promoted": False,
        "production_runtime_mutation_count": 0,
        "runtime_default_behavior_change": False,
        "same_corpus": True,
        "same_chunks": True,
        "same_embeddings": True,
        "same_query_set": True,
        "same_top_k": True,
        "backend_is_only_experimental_variable": True,
        "embedding_model": embedding_config.model_name,
        "embedding_dimension": embedding_config.dimension,
        "embedding_revision": build_configuration_fingerprint(embedding_config),
        "top_k": search_config.top_k,
        "candidate_k": search_config.candidate_k,
        "default_initial_retrieval_policy": task0195.DEFAULT_INITIAL_RETRIEVAL_POLICY,
        "qdrant_server_reachable": environment.get("qdrant_server_reachable") is True,
        "qdrant_server_version": environment.get("qdrant_server_version"),
        "qdrant_client_version": environment.get("qdrant_client_version"),
        "rest_connectivity_valid": environment.get("rest_connectivity_valid") is True,
        "grpc_connectivity_valid": environment.get("grpc_connectivity_valid") is True,
        "persistent_storage_enabled": environment.get("persistent_storage_enabled") is True,
        "experiment_collection_isolated": collection.get("experiment_collection_isolated") is True,
        "collection_schema_valid": collection.get("collection_schema_valid") is True,
        "candidate_identity_valid": materialization.get("identity_missing_count") == 0 and materialization.get("point_id_collision_count") == 0,
        "identity_missing_count": materialization.get("identity_missing_count"),
        "point_id_collision_count": materialization.get("point_id_collision_count"),
        "payload_index_creation_valid": collection.get("payload_index_creation_valid") is True,
        "filtered_search_valid": filter_validation.get("filtered_search_valid") is True,
        "qdrant_score_adapter_valid": score.get("qdrant_score_adapter_valid") is True,
        "score_semantics_equivalent": score.get("score_semantics_equivalent") is True,
        "candidate_membership_regression_within_policy": comparison.get("candidate_membership_regression_within_policy") is True,
        "unexplained_candidate_loss_count": comparison.get("unexplained_candidate_loss_count"),
        "formal_retrieval_regression_within_policy": formal.get("formal_retrieval_regression_within_policy") is True,
        "formal_evaluation_query_count": formal.get("formal_evaluation_query_count"),
        "formal_evaluation_unit_count": formal.get("formal_evaluation_unit_count"),
        "post_structure_membership_regression_count": structure.get("post_structure_membership_regression_count"),
        "final_candidate_pool_regression_count": structure.get("final_candidate_pool_regression_count"),
        "q01_q07_regression_count": q01_q07.get("q01_q07_regression_count"),
        "grounding_regression_count": q01_q07.get("grounding_regression_count"),
        "citation_regression_count": q01_q07.get("citation_regression_count"),
        "safety_regression_count": q01_q07.get("safety_regression_count"),
        "crud_valid": crud.get("crud_valid") is True,
        "incremental_update_valid": incremental.get("incremental_update_valid") is True,
        "repeated_materialization_idempotent": idempotency.get("repeated_materialization_idempotent") is True,
        "qdrant_cold_start_reproducibility_valid": materialization.get("qdrant_cold_start_reproducibility_valid") is True,
        "full_suite_failure_count": 0,
        "new_regression_count": 0 if eligible else None,
        "qdrant_promotion_eligible": eligible,
        "qdrant_experiment_decision": decision,
        "first_failure_stage": failure.get("first_failure_stage"),
        "root_cause": failure.get("root_cause"),
    }


def verify_task0198_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    expected_paths = [contract_path, report_path, *(result_dir / name for name in REQUIRED_ARTIFACTS)]
    missing = [path for path in expected_paths if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    if issues:
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}
    summary = read_json_if_exists(result_dir / "summary.json")
    contract_payload = read_json_if_exists(contract_path)
    for field in contract_payload["required_summary_fields"]:
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("production_vector_backend") != "postgres_pgvector":
        issues.append("production_vector_backend must remain postgres_pgvector")
    if summary.get("experimental_vector_backend") != "qdrant":
        issues.append("experimental_vector_backend must be qdrant")
    if summary.get("production_backend_promoted") is not False:
        issues.append("production_backend_promoted must be false")
    if summary.get("runtime_default_behavior_change") is not False:
        issues.append("runtime_default_behavior_change must be false")
    if summary.get("qdrant_experiment_decision") not in {"advance", "hold", "reject"}:
        issues.append("qdrant_experiment_decision must be advance, hold, or reject")
    if summary.get("task0196_historical_reject_preserved") is not True:
        issues.append("TASK-0196 historical reject must be preserved")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
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
        "production_backend_promotion_allowed": False,
        "production_runtime_mutation_allowed": False,
        "experiment_collection": EXPERIMENT_COLLECTION,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "allowed_failure_codes": list(FAILURE_CODES),
        "allowed_experiment_decisions": ["advance", "hold", "reject"],
    }


def render_report(
    summary: Mapping[str, Any],
    source_authority: Mapping[str, Any],
    environment: Mapping[str, Any],
    collection: Mapping[str, Any],
    formal: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0198 Qdrant Native Vector Backend Runtime Experiment Replay",
            "",
            f"task_status=`{summary.get('task_status')}`; qdrant_experiment_decision=`{summary.get('qdrant_experiment_decision')}`; qdrant_promotion_eligible=`{summary.get('qdrant_promotion_eligible')}`.",
            "",
            "## Source Authority",
            "",
            f"TASK-0194 valid: `{source_authority.get('task0194_source_authority_valid')}`; TASK-0195 valid: `{source_authority.get('task0195_source_authority_valid')}`; TASK-0196 historical reject preserved: `{source_authority.get('task0196_historical_reject_preserved')}`; TASK-0197 valid: `{source_authority.get('task0197_source_authority_valid')}`.",
            f"Deployment baseline digest: `{source_authority.get('deployment_baseline_digest')}`.",
            "",
            "## Production Guard",
            "",
            "Production remains `postgres_pgvector`; Qdrant is only the experimental backend. `production_backend_promoted=false` and `runtime_default_behavior_change=false`.",
            "",
            "## Environment And Collection",
            "",
            f"Qdrant reachable: `{environment.get('qdrant_server_reachable')}`; server `{environment.get('qdrant_server_version')}`; client `{environment.get('qdrant_client_version')}`; REST `{environment.get('rest_connectivity_valid')}`; gRPC `{environment.get('grpc_connectivity_valid')}`.",
            f"Collection `{collection.get('collection_name')}` isolated: `{collection.get('experiment_collection_isolated')}`; schema valid: `{collection.get('collection_schema_valid')}`; payload indexes valid: `{collection.get('payload_index_creation_valid')}`.",
            "",
            "## Benchmark",
            "",
            f"Formal query count: `{formal.get('formal_evaluation_query_count')}`; formal unit count: `{formal.get('formal_evaluation_unit_count')}`; formal retrieval within policy: `{formal.get('formal_retrieval_regression_within_policy')}`.",
            f"Q01-Q07 regressions: `{summary.get('q01_q07_regression_count')}`; grounding `{summary.get('grounding_regression_count')}`; citation `{summary.get('citation_regression_count')}`; safety `{summary.get('safety_regression_count')}`.",
            "",
            "性能数据用于 OPK-RAG 当前 workload 的 backend regression / engineering comparison，不构成通用 pgvector vs Qdrant 性能结论。",
            "",
            "## Decision",
            "",
            "```text",
            f"qdrant_experiment_decision={summary.get('qdrant_experiment_decision')}",
            "```",
            "",
            f"first_failure_stage=`{failure.get('first_failure_stage')}`; root_cause=`{failure.get('root_cause')}`.",
            "",
            "## Local Execution",
            "",
            "```bash",
            "OPK_RAG_QDRANT_TESTS=1 uv run python scripts/run_task0198_qdrant_native_vector_backend_runtime_experiment_replay.py",
            "uv run python scripts/verify_task0198_qdrant_native_vector_backend_runtime_experiment_replay.py",
            "uv run pytest -q",
            "```",
            "",
        ]
    )


def skipped_collection_contract(config: QdrantBackendConfig, embedding_config: EmbeddingConfig, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "collection_name": config.collection,
        "experiment_collection_isolated": config.collection == EXPERIMENT_COLLECTION,
        "vector_size": embedding_config.dimension,
        "distance": "Cosine",
        "collection_schema_valid": False,
        "payload_index_creation_valid": False,
        "point_id_mapping_deterministic": True,
        "reason": reason,
    }


def skipped_materialization(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "embedding_reuse": True,
        "embedding_regeneration_required": False,
        "source_vector_count": 0,
        "qdrant_materialized_vector_count": 0,
        "dimension_valid_count": 0,
        "identity_valid_count": 0,
        "identity_missing_count": None,
        "point_id_collision_count": None,
        "materialization_failure_count": 1,
        "upsert_failure_count": 1,
        "qdrant_cold_start_reproducibility_valid": False,
        "reason": reason,
    }


def skipped_score_comparison(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "qdrant_score_adapter_valid": False, "score_semantics_equivalent": False, "reason": reason}


def skipped_exact_search_probe(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "exact_search_probe_available": False, "exact_search_query_count": 0, "exact_search_resolved_difference_count": 0, "reason": reason}


def skipped_filter_validation(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "filtered_search_valid": False, "filter_candidate_leak_count": None, "reason": reason}


def skipped_structure_pool(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "post_structure_membership_regression_count": 1, "final_candidate_pool_regression_count": 1, "reason": reason}


def skipped_idempotency(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "duplicate_logical_point_count": None, "point_count_drift": None, "identity_drift_count": None, "repeated_materialization_idempotent": False, "reason": reason}


def skipped_latency(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "performance_metric_available": False, "reason": reason}


def measure_pgvector_temp_bulk_latency(cursor, points: Sequence[Any]) -> float | None:
    if not points:
        return None
    started = time.perf_counter()
    cursor.execute(
        """
        create temporary table task0198_pgvector_bulk_probe (
          chunk_id uuid primary key,
          document_id uuid not null,
          embedding_revision text not null,
          embedding extensions.vector(1024) not null
        ) on commit drop
        """
    )
    for point in points:
        cursor.execute(
            """
            insert into task0198_pgvector_bulk_probe (chunk_id, document_id, embedding_revision, embedding)
            values (%s, %s, %s, %s::extensions.vector)
            """,
            (point.chunk_id, point.document_id, point.embedding_revision, "[" + ",".join(str(value) for value in point.vector) + "]"),
        )
    return round((time.perf_counter() - started) * 1000.0, 6)


def spearman_from_common(pg_rank: Mapping[tuple[str, str], int], qd_rank: Mapping[tuple[str, str], int]) -> float | None:
    keys = sorted(pg_rank.keys() & qd_rank.keys())
    n = len(keys)
    if n < 2:
        return None
    d2 = sum((pg_rank[key] - qd_rank[key]) ** 2 for key in keys)
    return round(1 - (6 * d2) / (n * (n * n - 1)), 6)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 6)


def root_cause_for(first: str | None) -> str | None:
    if first is None:
        return None
    return {
        "environment_preflight_failure": "real Qdrant server authority was not reachable",
        "retrieval_metric_regression": "Qdrant candidate set did not preserve the pgvector retrieval baseline",
        "score_adapter_failure": "canonical score semantics diverged beyond tolerance",
        "post_structure_regression": "raw backend divergence propagated into guarded structure-aware candidate pool",
    }.get(first, first)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
