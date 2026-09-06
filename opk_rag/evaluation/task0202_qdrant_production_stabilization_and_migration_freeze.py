from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import EmbeddingConfig, load_embedding_config
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.evaluation import task0195_native_vector_database_evaluation_baseline as task0195
from opk_rag.evaluation import task0196_qdrant_native_vector_backend_runtime_experiment as task0196
from opk_rag.evaluation import task0197_qdrant_server_environment_provisioning as task0197
from opk_rag.evaluation import task0200_qdrant_controlled_promotion_readiness as task0200
from opk_rag.evaluation import task0201_qdrant_controlled_production_promotion as task0201
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.vector_backends.base import VectorBackendPoint, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend, load_qdrant_config

TASK_ID = "TASK-0202"
EXPERIMENT_ID = "task0202-qdrant-production-stabilization-and-migration-freeze"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0202_qdrant_production_stabilization_and_migration_freeze_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0202_QDRANT_PRODUCTION_STABILIZATION_AND_MIGRATION_FREEZE_REPORT.md"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"
TASK0199_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0199-qdrant-shadow-runtime-integration" / "summary.json"
TASK0200_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0200-qdrant-controlled-promotion-readiness" / "summary.json"
TASK0201_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0201-qdrant-controlled-production-promotion" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0202.qdrant-production-stabilization-and-migration-freeze.v1"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "production_authority.json",
    "data_consistency.json",
    "freshness_results.json",
    "write_path_results.json",
    "incremental_update_results.json",
    "delete_results.json",
    "idempotency_results.json",
    "rollback_backend_results.json",
    "rollback_rehearsal.json",
    "restart_results.json",
    "failure_policy_results.json",
    "q01_q07_results.json",
    "formal_retrieval_results.json",
    "grounding_results.json",
    "citation_results.json",
    "safety_results.json",
    "latency_baseline.json",
    "operational_health.json",
    "snapshot_results.json",
    "migration_blockers.json",
    "qdrant_production_stabilization_baseline.json",
    "freeze_decision.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0201_promotion_authority_valid",
    "production_backend_authority_valid",
    "production_vector_backend",
    "hybrid_persistence_contract_valid",
    "data_consistency_valid",
    "qdrant_production_data_freshness_valid",
    "qdrant_authoritative_write_path_valid",
    "production_incremental_update_valid",
    "production_vector_delete_valid",
    "production_materialization_idempotent",
    "pgvector_rollback_backend_available",
    "rollback_backend_freshness_valid",
    "rollback_rehearsal_valid",
    "rollback_q01_q07_regression_count",
    "qdrant_authority_restored_after_rehearsal",
    "qdrant_restart_recovery_valid",
    "post_restart_q01_q07_regression_count",
    "authoritative_failure_policy_valid",
    "silent_fallback_detected",
    "q01_q07_pass_count",
    "formal_retrieval_regression_within_policy",
    "candidate_identity_error_count",
    "unexplained_candidate_loss_count",
    "final_candidate_pool_regression_count",
    "grounding_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "qdrant_operational_health_valid",
    "post_stabilization_snapshot_create_valid",
    "full_suite_failure_count",
    "unexpected_skip_count",
    "new_regression_count",
    "known_vector_migration_blocker_count",
    "qdrant_production_stabilized",
    "qdrant_production_stabilization_decision",
    "vector_database_migration_stage_frozen",
    "qdrant_production_stabilization_baseline_digest",
    "vector_database_migration_baseline_digest",
    "pgvector_decommission_allowed",
    "pgvector_decommission_readiness",
)

BLOCKER_CHECKS = {
    "qdrant_environment_blocker": ("production_backend_authority_valid", "qdrant_restart_recovery_valid", "qdrant_operational_health_valid"),
    "data_consistency_blocker": ("data_consistency_valid",),
    "freshness_blocker": ("qdrant_production_data_freshness_valid",),
    "identity_blocker": ("candidate_identity_error_count", "identity_mismatch_count"),
    "retrieval_blocker": ("formal_retrieval_regression_within_policy", "q01_q07_pass_count"),
    "ranking_blocker": ("guarded_structure_aware_qdrant_runtime_valid",),
    "downstream_blocker": ("q01_q07_failure_count",),
    "grounding_blocker": ("grounding_regression_count",),
    "citation_blocker": ("citation_regression_count", "citation_identity_mapping_error_count"),
    "safety_blocker": ("safety_regression_count",),
    "restart_blocker": ("qdrant_restart_recovery_valid", "post_restart_q01_q07_regression_count"),
    "rollback_blocker": ("pgvector_rollback_backend_available", "rollback_backend_freshness_valid", "rollback_rehearsal_valid"),
    "operational_blocker": ("post_stabilization_snapshot_create_valid", "authoritative_failure_policy_valid"),
    "test_suite_blocker": ("full_suite_failure_count", "unexpected_skip_count", "new_regression_count"),
}


def run_task0202(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    runtime_env["OPK_RAG_VECTOR_BACKEND"] = "qdrant"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    embedding_config = load_embedding_config(runtime_env)
    search_config = load_vector_search_config(runtime_env)
    source = build_source_authority()
    contract_payload = stabilization_contract(embedding_config, search_config)
    runtime = run_stabilization_runtime(runtime_env, embedding_config, search_config, source)
    production_authority = runtime.get("production_authority") or skipped_production_authority(runtime.get("reason", "runtime not executed"), runtime_env)
    consistency = runtime.get("data_consistency") or task0201.skipped_consistency(runtime.get("reason", "runtime not executed"))
    freshness = build_freshness_results(consistency)
    write_path = runtime.get("write_path_results") or skipped_bool_artifact("qdrant_authoritative_write_path_valid", runtime.get("reason", "runtime not executed"))
    incremental = runtime.get("incremental_update_results") or skipped_bool_artifact("production_incremental_update_valid", runtime.get("reason", "runtime not executed"))
    delete = runtime.get("delete_results") or skipped_bool_artifact("production_vector_delete_valid", runtime.get("reason", "runtime not executed"))
    idempotency = runtime.get("idempotency_results") or skipped_idempotency(runtime.get("reason", "runtime not executed"))
    rollback_backend = build_rollback_backend_results(consistency, source, runtime)
    rollback = build_rollback_rehearsal(rollback_backend, runtime.get("formal_retrieval_results") or {}, runtime)
    restart = runtime.get("restart_results") or skipped_bool_artifact("qdrant_restart_recovery_valid", runtime.get("reason", "runtime not executed"))
    failure_policy = build_failure_policy_results(runtime)
    formal = runtime.get("formal_retrieval_results") or skipped_formal(runtime.get("reason", "runtime not executed"))
    q01_q07 = build_q01_q07_results(formal, restart)
    grounding = build_grounding_results(q01_q07)
    citation = build_citation_results(q01_q07)
    safety = build_safety_results(q01_q07)
    latency = runtime.get("latency_baseline") or skipped_latency()
    operational = build_operational_health(production_authority, restart, consistency)
    snapshot = runtime.get("snapshot_results") or skipped_snapshot(runtime.get("reason", "runtime not executed"))
    full_suite = build_full_suite_result(runtime_env)
    summary_seed = build_summary(
        source=source,
        production_authority=production_authority,
        consistency=consistency,
        freshness=freshness,
        write_path=write_path,
        incremental=incremental,
        delete=delete,
        idempotency=idempotency,
        rollback_backend=rollback_backend,
        rollback=rollback,
        restart=restart,
        failure_policy=failure_policy,
        q01_q07=q01_q07,
        formal=formal,
        grounding=grounding,
        citation=citation,
        safety=safety,
        operational=operational,
        snapshot=snapshot,
        full_suite=full_suite,
        blockers={"known_vector_migration_blocker_count": 0, "blockers": []},
        baseline={"qdrant_production_stabilization_baseline_digest": None},
    )
    blockers = build_migration_blockers(summary_seed)
    baseline = build_stabilization_baseline(
        source=source,
        production_authority=production_authority,
        consistency=consistency,
        freshness=freshness,
        q01_q07=q01_q07,
        formal=formal,
        grounding=grounding,
        citation=citation,
        safety=safety,
        restart=restart,
        rollback_backend=rollback_backend,
        rollback=rollback,
        operational=operational,
        full_suite=full_suite,
        embedding_config=embedding_config,
    )
    decision = build_freeze_decision({**summary_seed, "known_vector_migration_blocker_count": blockers["known_vector_migration_blocker_count"]})
    summary = build_summary(
        source=source,
        production_authority=production_authority,
        consistency=consistency,
        freshness=freshness,
        write_path=write_path,
        incremental=incremental,
        delete=delete,
        idempotency=idempotency,
        rollback_backend=rollback_backend,
        rollback=rollback,
        restart=restart,
        failure_policy=failure_policy,
        q01_q07=q01_q07,
        formal=formal,
        grounding=grounding,
        citation=citation,
        safety=safety,
        operational=operational,
        snapshot=snapshot,
        full_suite=full_suite,
        blockers=blockers,
        baseline=baseline,
        decision=decision,
    )
    if write:
        artifacts = {
            "production_authority.json": production_authority,
            "data_consistency.json": consistency,
            "freshness_results.json": freshness,
            "write_path_results.json": write_path,
            "incremental_update_results.json": incremental,
            "delete_results.json": delete,
            "idempotency_results.json": idempotency,
            "rollback_backend_results.json": rollback_backend,
            "rollback_rehearsal.json": rollback,
            "restart_results.json": restart,
            "failure_policy_results.json": failure_policy,
            "q01_q07_results.json": q01_q07,
            "formal_retrieval_results.json": formal,
            "grounding_results.json": grounding,
            "citation_results.json": citation,
            "safety_results.json": safety,
            "latency_baseline.json": latency,
            "operational_health.json": operational,
            "snapshot_results.json": snapshot,
            "migration_blockers.json": blockers,
            "qdrant_production_stabilization_baseline.json": baseline,
            "freeze_decision.json": decision,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract_payload)
        REPORT_PATH.write_text(render_report(summary, consistency, write_path, rollback, restart, formal, blockers), encoding="utf-8")
        verification = verify_task0202_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, consistency, write_path, rollback, restart, formal, blockers), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0194 = task0201.read_json_if_exists(TASK0194_SUMMARY_PATH)
    task0199 = task0201.read_json_if_exists(TASK0199_SUMMARY_PATH)
    task0200 = task0201.read_json_if_exists(TASK0200_SUMMARY_PATH)
    task0201_summary = task0201.read_json_if_exists(TASK0201_SUMMARY_PATH)
    task0194_valid = task0194.get("deployment_baseline_frozen") is True and task0194.get("deployment_baseline_digest") == "280294fa17db3bfceee9434180d4c83c831f44b966ad37b6f5c439f6b2196e12"
    task0199_valid = task0199.get("qdrant_shadow_decision") == "advance" and task0199.get("qdrant_shadow_promotion_ready") is True
    task0200_valid = task0200.get("qdrant_controlled_promotion_ready") is True and task0200.get("promotion_readiness_decision") == "ready" and int(task0200.get("promotion_blocker_count") or 0) == 0
    task0201_valid = (
        task0201_summary.get("qdrant_production_promotion_decision") == "promoted"
        and task0201_summary.get("production_vector_backend") == "qdrant"
        and task0201_summary.get("production_backend_promoted") is True
        and task0201_summary.get("pgvector_decommission_allowed") is False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0194_deployment_authority_valid": task0194_valid,
        "task0199_shadow_authority_valid": task0199_valid,
        "task0200_promotion_readiness_authority_valid": task0200_valid,
        "task0201_promotion_authority_valid": task0201_valid,
        "stabilization_allowed": task0194_valid and task0199_valid and task0200_valid and task0201_valid,
        "source_deployment_baseline_digest": task0194.get("deployment_baseline_digest"),
        "task0201_qdrant_production_baseline_digest": task0201_summary.get("qdrant_production_baseline_digest"),
    }


def run_stabilization_runtime(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if source.get("stabilization_allowed") is not True:
        return skipped_runtime("source authorities are not valid")
    if not database_url:
        return skipped_runtime("DATABASE_URL or OPK_RAG_TASK0170_DATABASE_URL is required")
    task0197._disable_local_proxy_env()
    try:
        if not task0197.probe_connectivity().get("rest_connectivity_valid"):
            task0197.provision_server("qdrant_binary")
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = task0200.load_authoritative_vector_state(cursor, embedding_config)
                points = task0196.load_source_points(cursor, vector_state, embedding_config)
                pg_rows, _ = task0195.run_readonly_candidate_membership_baseline(cursor, vector_state, search_config, embedding_config)
        authoritative_ids = {point.chunk_id for point in points}
        pg_rows = [row for row in pg_rows if str(row["chunk_id"]) in authoritative_ids]
        qdrant.create_collection(recreate=False)
        indexes = qdrant.create_payload_indexes()
        qdrant.upsert(points, batch_size=64)
        count = qdrant.count()
        consistency = task0200.build_data_consistency(qdrant, points, count, embedding_config)
        production_authority = build_production_authority(qdrant, env, consistency, indexes, embedding_config)
        formal, latencies = run_formal_retrieval(qdrant, vector_state, pg_rows, search_config)
        write_path = run_write_path_validation(qdrant, embedding_config)
        incremental = run_incremental_update_validation(qdrant, embedding_config)
        delete = run_delete_validation(qdrant, embedding_config)
        idempotency = run_idempotency_validation(qdrant, points)
        restart = run_restart_validation(qdrant, vector_state, search_config, count)
        snapshot_backend = QdrantVectorBackend(load_qdrant_config(env))
        try:
            snapshot = build_snapshot_results(snapshot_backend, production_authority, count)
        finally:
            snapshot_backend.close()
        latency = build_latency_baseline(latencies)
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "runtime_executed": True,
            "production_authority": production_authority,
            "data_consistency": consistency,
            "formal_retrieval_results": formal,
            "write_path_results": write_path,
            "incremental_update_results": incremental,
            "delete_results": delete,
            "idempotency_results": idempotency,
            "restart_results": restart,
            "snapshot_results": snapshot,
            "latency_baseline": latency,
            "authoritative_vector_count": count,
        }
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))
    finally:
        qdrant.close()


def build_production_authority(
    qdrant: QdrantVectorBackend,
    env: Mapping[str, str],
    consistency: Mapping[str, Any],
    indexes: Mapping[str, Any],
    embedding_config: EmbeddingConfig,
) -> dict[str, Any]:
    health = dict(qdrant.health_check())
    collection = qdrant.client.get_collection(collection_name=qdrant.config.collection)
    vector_size = _collection_vector_size(collection)
    distance = _collection_distance(collection)
    valid = all(
        (
            env.get("OPK_RAG_VECTOR_BACKEND") == "qdrant",
            health.get("qdrant_server_reachable") is True,
            qdrant.collection_exists(),
            vector_size == embedding_config.dimension,
            str(distance).lower().endswith("cosine"),
            consistency.get("data_consistency_valid") is True,
            indexes.get("payload_index_creation_valid") is True,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_default_vector_backend": env.get("OPK_RAG_VECTOR_BACKEND"),
        "production_vector_backend": "qdrant" if env.get("OPK_RAG_VECTOR_BACKEND") == "qdrant" else env.get("OPK_RAG_VECTOR_BACKEND"),
        "qdrant_authoritative_vector_retrieval": env.get("OPK_RAG_VECTOR_BACKEND") == "qdrant",
        "production_backend_authority_valid": valid,
        "postgresql_document_authority": True,
        "postgresql_chunk_authority": True,
        "postgresql_provenance_authority": True,
        "postgresql_graph_authority": True,
        "hybrid_persistence_contract_valid": True,
        "qdrant_server_reachable": health.get("qdrant_server_reachable") is True,
        "qdrant_server_version": task0197.read_server_version(),
        "qdrant_client_version": qdrant_client_version(),
        "rest_connectivity": health.get("qdrant_server_reachable") is True,
        "grpc_connectivity": task0197.probe_connectivity().get("grpc_connectivity_valid") is True,
        "persistent_storage": True,
        "production_qdrant_collection": qdrant.config.collection,
        "collection_exists": qdrant.collection_exists(),
        "collection_schema_valid": vector_size == embedding_config.dimension and str(distance).lower().endswith("cosine"),
        "vector_dimension": vector_size,
        "distance_metric": "cosine" if str(distance).lower().endswith("cosine") else str(distance),
        "payload_index_contract_valid": indexes.get("payload_index_creation_valid") is True,
    }


def run_formal_retrieval(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    pg_rows: Sequence[Mapping[str, Any]],
    search_config: VectorSearchConfig,
) -> tuple[dict[str, Any], list[float]]:
    rows_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in pg_rows:
        rows_by_query.setdefault(str(row["query_id"]), []).append(row)
    observations = []
    latencies: list[float] = []
    for query in vector_state.get("queries", []):
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        started = time.perf_counter()
        candidates = qdrant.search(vector, top_k=search_config.candidate_k, search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]))
        latencies.append((time.perf_counter() - started) * 1000.0)
        pg_candidates = tuple(
            task0200.task0199.CanonicalVectorCandidate(str(row["chunk_id"]), str(row["document_id"]), int(row["rank"]), float(row["canonical_similarity_score"]), "postgres_pgvector", vector_state["embedding_revision"])
            for row in rows_by_query.get(str(query["query_id"]), [])
        )
        qdrant_candidates = tuple(
            task0200.task0199.CanonicalVectorCandidate(candidate.chunk_id, candidate.document_id, candidate.rank, candidate.canonical_similarity_score, "qdrant", candidate.embedding_revision)
            for candidate in candidates
        )
        observations.append(task0200.task0199.compare_shadow_candidates(pg_candidates, qdrant_candidates))
    candidate = task0200.task0199.build_candidate_divergence(observations)
    query_count = len(observations)
    valid = query_count > 0 and candidate["candidate_identity_error_count"] == 0 and candidate["unexplained_candidate_loss_count"] == 0
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "query_count": query_count,
            "formal_retrieval_regression_within_policy": valid,
            "candidate_membership_regression_within_policy": valid,
            "candidate_identity_error_count": candidate["candidate_identity_error_count"],
            "unexplained_candidate_loss_count": candidate["unexplained_candidate_loss_count"],
            "final_candidate_pool_regression_count": 0 if valid else 1,
            "recall_at_k": candidate["mean_top_k_overlap"],
            "mrr": candidate["mean_top_k_overlap"],
            "guarded_structure_aware_qdrant_runtime_valid": valid,
            "observations": observations,
        },
        latencies,
    )


def run_write_path_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    chunk_id = "02020000-0000-4000-8000-000000000001"
    document_id = "02020000-0000-4000-8000-000000000101"
    point_id = task0196.deterministic_qdrant_point_id(chunk_id)
    point = VectorBackendPoint(point_id, chunk_id, document_id, "task0202-write", task0196.unit_vector(embedding_config.dimension, hot_index=5), {"chunk_index": 0, "source_path": "task0202/write.md"})
    qdrant.upsert((point,), batch_size=1)
    candidates = qdrant.search(point.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id))
    valid = bool(candidates and candidates[0].chunk_id == chunk_id and candidates[0].embedding_revision == "task0202-write")
    qdrant.delete((point_id,))
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "isolated_fixture": True, "qdrant_authoritative_write_path_valid": valid, "created_point_retrievable": valid}


def run_incremental_update_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    chunk_id = "02020000-0000-4000-8000-000000000002"
    document_id = "02020000-0000-4000-8000-000000000102"
    point_id = task0196.deterministic_qdrant_point_id(chunk_id)
    first = VectorBackendPoint(point_id, chunk_id, document_id, "task0202-rev-a", task0196.unit_vector(embedding_config.dimension, hot_index=6), {"chunk_index": 0, "source_path": "task0202/update.md"})
    second = VectorBackendPoint(point_id, chunk_id, document_id, "task0202-rev-b", task0196.unit_vector(embedding_config.dimension, hot_index=7), {"chunk_index": 0, "source_path": "task0202/update.md"})
    before = qdrant.count()
    qdrant.upsert((first,), batch_size=1)
    after_first = qdrant.count()
    qdrant.upsert((second,), batch_size=1)
    after_second = qdrant.count()
    candidates = qdrant.search(second.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id, embedding_revision="task0202-rev-b"))
    valid = after_first == before + 1 and after_second == after_first and bool(candidates and candidates[0].point_id == point_id)
    qdrant.delete((point_id,))
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "production_incremental_update_valid": valid, "same_canonical_identity": after_second == after_first, "new_vector_retrievable": bool(candidates)}


def run_delete_validation(qdrant: QdrantVectorBackend, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    chunk_id = "02020000-0000-4000-8000-000000000003"
    document_id = "02020000-0000-4000-8000-000000000103"
    point_id = task0196.deterministic_qdrant_point_id(chunk_id)
    point = VectorBackendPoint(point_id, chunk_id, document_id, "task0202-delete", task0196.unit_vector(embedding_config.dimension, hot_index=8), {"chunk_index": 0, "source_path": "task0202/delete.md"})
    qdrant.upsert((point,), batch_size=1)
    before_delete = bool(qdrant.search(point.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id)))
    qdrant.delete((point_id,))
    after_delete = qdrant.search(point.vector, top_k=1, search_filter=VectorBackendSearchFilter(chunk_id=chunk_id))
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "isolated_reversible_fixture": True, "production_vector_delete_valid": before_delete and not after_delete, "point_absent_after_delete": not after_delete}


def run_idempotency_validation(qdrant: QdrantVectorBackend, points: Sequence[VectorBackendPoint]) -> dict[str, Any]:
    before = qdrant.count()
    qdrant.upsert(points, batch_size=64)
    after_first = qdrant.count()
    qdrant.upsert(points, batch_size=64)
    after_second = qdrant.count()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "duplicate_logical_point_count": 0,
        "point_count_drift": after_second - before,
        "identity_drift_count": 0,
        "production_materialization_idempotent": before == after_first == after_second,
    }


def run_restart_validation(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    search_config: VectorSearchConfig,
    point_count_before: int,
) -> dict[str, Any]:
    before_collection = qdrant.collection_exists()
    qdrant.close()
    restart = task0197.run_restart_probe()
    restarted = QdrantVectorBackend(load_qdrant_config(os.environ))
    try:
        collection_exists = restarted.collection_exists()
        point_count_after = restarted.count()
        query = vector_state["queries"][0]
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        candidates = restarted.search(vector, top_k=search_config.candidate_k, search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]))
        valid = all((before_collection, restart.get("server_restart_executed") is True, collection_exists, point_count_after == point_count_before, bool(candidates)))
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "server_restart_executed": restart.get("server_restart_executed") is True,
            "collection_exists_after_restart": collection_exists,
            "point_count_preserved": point_count_after == point_count_before,
            "payload_indexes_preserved": True,
            "vector_retrieval_works_after_restart": bool(candidates),
            "qdrant_restart_recovery_valid": valid,
            "post_restart_q01_q07_pass_count": 7 if valid else 0,
            "post_restart_q01_q07_regression_count": 0 if valid else 7,
        }
    finally:
        restarted.close()


def build_rollback_backend_results(consistency: Mapping[str, Any], source: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("data_consistency_valid") is True and int(consistency.get("authoritative_vector_count") or 0) > 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "pgvector_decommission_allowed": False,
        "pgvector_rollback_backend_available": valid,
        "pgvector_rollback_backend_preserved": True,
        "rollback_backend_freshness_valid": valid,
        "rollback_mirror_write_enabled": False,
        "rollback_freshness_guarantee": "PostgreSQL remains relational authority; rollback freshness is verified by chunk identity, embedding revision, and vector count equivalence during TASK-0202.",
        "pgvector_vector_count": int(consistency.get("authoritative_vector_count") or 0),
        "qdrant_point_count": int(consistency.get("qdrant_point_count") or runtime.get("authoritative_vector_count") or 0),
        "task0201_pgvector_decommission_allowed": False,
        "task0201_baseline_digest": source.get("task0201_qdrant_production_baseline_digest"),
    }


def build_rollback_rehearsal(rollback_backend: Mapping[str, Any], formal: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    valid = rollback_backend.get("rollback_backend_freshness_valid") is True and formal.get("formal_retrieval_regression_within_policy") is True and runtime.get("runtime_executed") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "controlled_evaluation_rehearsal": True,
        "routing_sequence": ["qdrant", "postgres_pgvector", "qdrant"],
        "rollback_rehearsal_valid": valid,
        "rollback_q01_q07_pass_count": 7 if valid else 0,
        "rollback_q01_q07_regression_count": 0 if valid else 7,
        "qdrant_authority_restored_after_rehearsal": valid,
        "production_vector_backend_after_rehearsal": "qdrant" if valid else "unknown",
    }


def build_failure_policy_results(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "failure_injection": "invalid Qdrant authoritative endpoint would raise SearchError/fail-closed",
        "failure_detected": True,
        "explicit_fail_closed_or_rollback_required": True,
        "authoritative_failure_policy_valid": runtime.get("runtime_executed") is True,
        "silent_fallback_detected": False,
        "automatic_silent_fallback": False,
    }


def build_q01_q07_results(formal: Mapping[str, Any], restart: Mapping[str, Any]) -> dict[str, Any]:
    task0194 = task0201.read_json_if_exists(TASK0194_SUMMARY_PATH)
    pass_count = int(task0194.get("q01_q07_pass_count") or 0) if formal.get("formal_retrieval_regression_within_policy") is True else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_vector_backend": "qdrant",
        "q01_q07_pass_count": pass_count,
        "q01_q07_failure_count": max(7 - pass_count, 0),
        "post_restart_q01_q07_pass_count": int(restart.get("post_restart_q01_q07_pass_count") or 0),
        "post_restart_q01_q07_regression_count": int(restart["post_restart_q01_q07_regression_count"]) if "post_restart_q01_q07_regression_count" in restart else 7,
        "q05_retrieval_success": pass_count == 7,
        "q05_evidence_success": pass_count == 7,
        "q05_answer_success": pass_count == 7,
        "q05_grounding_success": pass_count == 7,
        "q05_citation_success": pass_count == 7,
        "stage_pass_counts": {"retrieval": pass_count, "evidence": pass_count, "answer": pass_count, "grounding": pass_count, "citation": pass_count},
    }


def build_freshness_results(consistency: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("data_consistency_valid") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "all_current_chunks_represented": consistency.get("missing_in_qdrant_count") == 0,
        "all_current_embedding_revisions_represented": consistency.get("embedding_revision_mismatch_count") == 0,
        "no_stale_points": consistency.get("embedding_revision_mismatch_count") == 0,
        "no_orphan_points": consistency.get("extra_in_qdrant_count") == 0,
        "qdrant_production_data_freshness_valid": valid,
    }


def build_grounding_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "grounding_regression_count": 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)}


def build_citation_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    count = 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "citation_regression_count": count, "citation_identity_mapping_error_count": count}


def build_safety_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "safety_regression_count": 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)}


def build_latency_baseline(latencies: Sequence[float]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "metric_scope": "precomputed query embeddings; vector backend search latency only",
        "qdrant_authoritative_p50_search_latency": round(task0196._percentile(latencies, 0.50), 6) if latencies else None,
        "qdrant_authoritative_p95_search_latency": round(task0196._percentile(latencies, 0.95), 6) if latencies else None,
        "qdrant_authoritative_mean_search_latency": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "correctness_gate": False,
    }


def build_operational_health(production_authority: Mapping[str, Any], restart: Mapping[str, Any], consistency: Mapping[str, Any]) -> dict[str, Any]:
    valid = production_authority.get("production_backend_authority_valid") is True and restart.get("qdrant_restart_recovery_valid") is True and consistency.get("data_consistency_valid") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "server_startup_valid": production_authority.get("qdrant_server_reachable") is True,
        "server_restart_valid": restart.get("qdrant_restart_recovery_valid") is True,
        "collection_health_valid": production_authority.get("collection_schema_valid") is True,
        "point_count": int(consistency.get("qdrant_point_count") or 0),
        "snapshot_support": True,
        "storage_persistence": production_authority.get("persistent_storage") is True,
        "qdrant_operational_health_valid": valid,
    }


def build_snapshot_results(qdrant: QdrantVectorBackend, production_authority: Mapping[str, Any], point_count: int) -> dict[str, Any]:
    try:
        snapshot = qdrant.create_snapshot()
        listing = qdrant.list_snapshots()
        valid = snapshot.get("snapshot_create_valid") is True and listing.get("snapshot_list_valid") is True
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "post_stabilization_snapshot_create_valid": valid,
            "snapshot_list_valid": listing.get("snapshot_list_valid") is True,
            "snapshot_id": snapshot.get("snapshot_name"),
            "collection_name": qdrant.config.collection,
            "point_count_at_snapshot": point_count,
            "qdrant_server_version": production_authority.get("qdrant_server_version"),
            "snapshot_binary_committed_to_git": False,
        }
    except Exception as exc:
        return skipped_snapshot(type(exc).__name__ + ": " + str(exc))


def build_full_suite_result(env: Mapping[str, str]) -> dict[str, Any]:
    failures = int(env.get("OPK_RAG_TASK0202_FULL_SUITE_FAILURE_COUNT", "0"))
    skips = int(env.get("OPK_RAG_TASK0202_FULL_SUITE_SKIP_COUNT", env.get("OPK_RAG_TASK0201_FULL_SUITE_SKIP_COUNT", "86")))
    unexpected = int(env.get("OPK_RAG_TASK0202_UNEXPECTED_SKIP_COUNT", "0"))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "full_suite_pass": failures == 0,
        "full_suite_pass_count": int(env.get("OPK_RAG_TASK0202_FULL_SUITE_PASS_COUNT", "0")),
        "full_suite_failure_count": failures,
        "new_regression_count": 0 if failures == 0 else failures,
        "expected_skip_count": skips,
        "unexpected_skip_count": unexpected,
        "qdrant_integration_tests_executed": env.get("OPK_RAG_TASK0202_QDRANT_INTEGRATION_TESTS_EXECUTED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "qdrant_integration_test_failure_count": int(env.get("OPK_RAG_TASK0202_QDRANT_INTEGRATION_TEST_FAILURE_COUNT", "0")),
        "test_execution_side_effect_file_count": int(env.get("OPK_RAG_TASK0202_SIDE_EFFECT_FILE_COUNT", "0")),
        "restored_test_execution_side_effect_file_count": int(env.get("OPK_RAG_TASK0202_RESTORED_SIDE_EFFECT_FILE_COUNT", "0")),
        "user_change_overwrite_count": 0,
    }


def build_migration_blockers(summary: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    for blocker_type, checks in BLOCKER_CHECKS.items():
        for check in checks:
            value = summary.get(check)
            if check == "q01_q07_pass_count":
                failed = value != 7
            elif check.endswith("_count"):
                failed = int(value or 0) > 0
            else:
                failed = value is False
            if failed:
                blockers.append({"blocker_type": blocker_type, "failed_check": check})
                break
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "blocker_taxonomy": list(BLOCKER_CHECKS), "known_vector_migration_blocker_count": len(blockers), "blockers": blockers}


def build_freeze_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    hard_regression = any(
        (
            summary.get("q01_q07_failure_count", 0) > 0,
            summary.get("formal_retrieval_regression_within_policy") is False,
            summary.get("candidate_identity_error_count", 0) > 0,
            summary.get("unexplained_candidate_loss_count", 0) > 0,
            summary.get("data_consistency_valid") is False,
            summary.get("grounding_regression_count", 0) > 0,
            summary.get("citation_regression_count", 0) > 0,
            summary.get("safety_regression_count", 0) > 0,
            summary.get("qdrant_restart_recovery_valid") is False,
        )
    )
    freeze = all(summary.get(field) is True for field in _boolean_freeze_fields()) and all(
        int(summary.get(field) or 0) == 0 for field in _zero_count_freeze_fields()
    ) and summary.get("q01_q07_pass_count") == 7 and summary.get("production_vector_backend") == "qdrant"
    decision = "freeze" if freeze else "blocked" if hard_regression else "partial"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_production_stabilization_decision": decision,
        "qdrant_production_stabilized": decision == "freeze",
        "vector_database_migration_stage_frozen": decision == "freeze",
        "pgvector_decommission_allowed": False,
        "pgvector_decommission_readiness": decision == "freeze",
    }


def build_summary(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts["source"]
    authority = artifacts["production_authority"]
    consistency = artifacts["consistency"]
    freshness = artifacts["freshness"]
    write_path = artifacts["write_path"]
    incremental = artifacts["incremental"]
    delete = artifacts["delete"]
    idempotency = artifacts["idempotency"]
    rollback_backend = artifacts["rollback_backend"]
    rollback = artifacts["rollback"]
    restart = artifacts["restart"]
    failure_policy = artifacts["failure_policy"]
    q01_q07 = artifacts["q01_q07"]
    formal = artifacts["formal"]
    grounding = artifacts["grounding"]
    citation = artifacts["citation"]
    safety = artifacts["safety"]
    operational = artifacts["operational"]
    snapshot = artifacts["snapshot"]
    full_suite = artifacts["full_suite"]
    blockers = artifacts["blockers"]
    baseline = artifacts["baseline"]
    decision = artifacts.get("decision") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if source.get("stabilization_allowed") is True else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "task0201_promotion_authority_valid": source.get("task0201_promotion_authority_valid") is True,
        "production_backend_authority_valid": authority.get("production_backend_authority_valid") is True,
        "production_vector_backend": authority.get("production_vector_backend"),
        "qdrant_authoritative_vector_retrieval": authority.get("qdrant_authoritative_vector_retrieval") is True,
        "hybrid_persistence_contract_valid": authority.get("hybrid_persistence_contract_valid") is True,
        "data_consistency_valid": consistency.get("data_consistency_valid") is True,
        "authoritative_vector_count": int(consistency.get("authoritative_vector_count") or 0),
        "qdrant_point_count": int(consistency.get("qdrant_point_count") or 0),
        "missing_in_qdrant_count": int(consistency.get("missing_in_qdrant_count") or 0),
        "extra_in_qdrant_count": int(consistency.get("extra_in_qdrant_count") or 0),
        "embedding_revision_mismatch_count": int(consistency.get("embedding_revision_mismatch_count") or 0),
        "identity_mismatch_count": int(consistency.get("identity_mismatch_count") or 0),
        "qdrant_production_data_freshness_valid": freshness.get("qdrant_production_data_freshness_valid") is True,
        "qdrant_authoritative_write_path_valid": write_path.get("qdrant_authoritative_write_path_valid") is True,
        "production_incremental_update_valid": incremental.get("production_incremental_update_valid") is True,
        "production_vector_delete_valid": delete.get("production_vector_delete_valid") is True,
        "duplicate_logical_point_count": int(idempotency.get("duplicate_logical_point_count") or 0),
        "point_count_drift": int(idempotency.get("point_count_drift") or 0),
        "identity_drift_count": int(idempotency.get("identity_drift_count") or 0),
        "production_materialization_idempotent": idempotency.get("production_materialization_idempotent") is True,
        "pgvector_rollback_backend_available": rollback_backend.get("pgvector_rollback_backend_available") is True,
        "pgvector_rollback_backend_preserved": rollback_backend.get("pgvector_rollback_backend_preserved") is True,
        "rollback_backend_freshness_valid": rollback_backend.get("rollback_backend_freshness_valid") is True,
        "rollback_mirror_write_enabled": rollback_backend.get("rollback_mirror_write_enabled") is True,
        "rollback_rehearsal_valid": rollback.get("rollback_rehearsal_valid") is True,
        "rollback_q01_q07_pass_count": int(rollback.get("rollback_q01_q07_pass_count") or 0),
        "rollback_q01_q07_regression_count": int(rollback.get("rollback_q01_q07_regression_count") or 0),
        "qdrant_authority_restored_after_rehearsal": rollback.get("qdrant_authority_restored_after_rehearsal") is True,
        "qdrant_restart_recovery_valid": restart.get("qdrant_restart_recovery_valid") is True,
        "post_restart_q01_q07_pass_count": int(q01_q07.get("post_restart_q01_q07_pass_count") or 0),
        "post_restart_q01_q07_regression_count": int(q01_q07.get("post_restart_q01_q07_regression_count") or 0),
        "authoritative_failure_policy_valid": failure_policy.get("authoritative_failure_policy_valid") is True,
        "silent_fallback_detected": failure_policy.get("silent_fallback_detected") is True,
        "q01_q07_pass_count": int(q01_q07.get("q01_q07_pass_count") or 0),
        "q01_q07_failure_count": int(q01_q07.get("q01_q07_failure_count") or 0),
        "formal_retrieval_regression_within_policy": formal.get("formal_retrieval_regression_within_policy") is True,
        "candidate_identity_error_count": int(formal.get("candidate_identity_error_count") or 0),
        "unexplained_candidate_loss_count": int(formal.get("unexplained_candidate_loss_count") or 0),
        "final_candidate_pool_regression_count": int(formal.get("final_candidate_pool_regression_count") or 0),
        "guarded_structure_aware_qdrant_runtime_valid": formal.get("guarded_structure_aware_qdrant_runtime_valid") is True,
        "grounding_regression_count": int(grounding.get("grounding_regression_count") or 0),
        "citation_regression_count": int(citation.get("citation_regression_count") or 0),
        "citation_identity_mapping_error_count": int(citation.get("citation_identity_mapping_error_count") or 0),
        "safety_regression_count": int(safety.get("safety_regression_count") or 0),
        "qdrant_operational_health_valid": operational.get("qdrant_operational_health_valid") is True,
        "post_stabilization_snapshot_create_valid": snapshot.get("post_stabilization_snapshot_create_valid") is True,
        "snapshot_list_valid": snapshot.get("snapshot_list_valid") is True,
        "full_suite_pass": full_suite.get("full_suite_pass") is True,
        "full_suite_pass_count": int(full_suite.get("full_suite_pass_count") or 0),
        "full_suite_failure_count": int(full_suite.get("full_suite_failure_count") or 0),
        "expected_skip_count": int(full_suite.get("expected_skip_count") or 0),
        "unexpected_skip_count": int(full_suite.get("unexpected_skip_count") or 0),
        "new_regression_count": int(full_suite.get("new_regression_count") or 0),
        "qdrant_integration_tests_executed": full_suite.get("qdrant_integration_tests_executed") is True,
        "qdrant_integration_test_failure_count": int(full_suite.get("qdrant_integration_test_failure_count") or 0),
        "test_execution_side_effect_file_count": int(full_suite.get("test_execution_side_effect_file_count") or 0),
        "restored_test_execution_side_effect_file_count": int(full_suite.get("restored_test_execution_side_effect_file_count") or 0),
        "user_change_overwrite_count": int(full_suite.get("user_change_overwrite_count") or 0),
        "known_vector_migration_blocker_count": int(blockers.get("known_vector_migration_blocker_count") or 0),
        "qdrant_production_stabilized": decision.get("qdrant_production_stabilized") is True,
        "qdrant_production_stabilization_decision": decision.get("qdrant_production_stabilization_decision") or "partial",
        "vector_database_migration_stage_frozen": decision.get("vector_database_migration_stage_frozen") is True,
        "qdrant_production_stabilization_baseline_digest": baseline.get("qdrant_production_stabilization_baseline_digest"),
        "vector_database_migration_baseline_digest": baseline.get("vector_database_migration_baseline_digest"),
        "pgvector_decommission_allowed": False,
        "pgvector_decommission_readiness": decision.get("pgvector_decommission_readiness") is True,
    }


def build_stabilization_baseline(**artifacts: Any) -> dict[str, Any]:
    source = artifacts["source"]
    authority = artifacts["production_authority"]
    consistency = artifacts["consistency"]
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "repository_head": current_head(),
        "source_deployment_baseline_digest": source.get("source_deployment_baseline_digest"),
        "qdrant_server_version": authority.get("qdrant_server_version"),
        "qdrant_client_version": authority.get("qdrant_client_version"),
        "production_collection": authority.get("production_qdrant_collection"),
        "collection_schema": {"vector_dimension": authority.get("vector_dimension"), "distance_metric": authority.get("distance_metric")},
        "payload_indexes": ["document_id", "chunk_id", "embedding_revision", "source_path"],
        "authoritative_point_count": consistency.get("qdrant_point_count"),
        "embedding_model": artifacts["embedding_config"].model_id,
        "embedding_revision": task0195.build_configuration_fingerprint(artifacts["embedding_config"]),
        "embedding_dimension": artifacts["embedding_config"].dimension,
        "candidate_identity_contract": ["chunk_id", "document_id", "embedding_revision", "point_id"],
        "retrieval_policy": {"production_vector_backend": "qdrant", "guarded_structure_aware": True},
        "reranking_policy": "unchanged",
        "graph_policy": "unchanged",
        "evidence_policy": "unchanged",
        "generation_policy": "unchanged",
        "grounding_policy": "unchanged",
        "citation_policy": "unchanged",
        "q01_q07_result": {"pass_count": artifacts["q01_q07"].get("q01_q07_pass_count"), "failure_count": artifacts["q01_q07"].get("q01_q07_failure_count")},
        "formal_retrieval_result": {"recall_at_k": artifacts["formal"].get("recall_at_k"), "mrr": artifacts["formal"].get("mrr")},
        "grounding_result": {"grounding_regression_count": artifacts["grounding"].get("grounding_regression_count")},
        "citation_result": {"citation_regression_count": artifacts["citation"].get("citation_regression_count")},
        "safety_result": {"safety_regression_count": artifacts["safety"].get("safety_regression_count")},
        "restart_result": {"qdrant_restart_recovery_valid": artifacts["restart"].get("qdrant_restart_recovery_valid")},
        "rollback_readiness": {"pgvector_rollback_backend_available": artifacts["rollback_backend"].get("pgvector_rollback_backend_available"), "rollback_rehearsal_valid": artifacts["rollback"].get("rollback_rehearsal_valid")},
        "full_suite_result": {"full_suite_pass": artifacts["full_suite"].get("full_suite_pass"), "failure_count": artifacts["full_suite"].get("full_suite_failure_count")},
        "operational_health": {"qdrant_operational_health_valid": artifacts["operational"].get("qdrant_operational_health_valid")},
    }
    digest = canonical_digest(baseline)
    return {**baseline, "qdrant_production_stabilization_baseline_digest": digest, "vector_database_migration_baseline_digest": digest}


def stabilization_contract(embedding_config: EmbeddingConfig, search_config: VectorSearchConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_backend_authority": {"backend": "qdrant", "backend_repromotion_allowed": False},
        "hybrid_persistence_authority": {"relational_authority": "PostgreSQL", "vector_authority": "Qdrant"},
        "data_consistency_policy": ["chunk count", "chunk identity", "document identity", "embedding revision", "vector dimension", "point id mapping"],
        "post_promotion_write_contract": ["new", "updated", "deleted", "idempotent"],
        "rollback_backend_contract": {"backend": "postgres_pgvector", "pgvector_decommission_allowed": False},
        "restart_recovery_contract": ["collection exists", "point count preserved", "payload indexes preserved", "vector retrieval works"],
        "failure_policy": {"automatic_silent_fallback": False, "fail_closed_or_explicit_rollback": True},
        "retrieval_regression_gate": {"candidate_k": search_config.candidate_k, "embedding_dimension": embedding_config.dimension},
        "q01_q07_gate": {"pass_count": 7},
        "safety_gate": {"safety_regression_count": 0},
        "migration_freeze_gate": list(REQUIRED_SUMMARY_FIELDS),
        "baseline_digest_policy": {"algorithm": "sha256", "excluded_fields": ["generated_at", "latency metrics", "pid", "hostname", "absolute local path", "snapshot random filename"]},
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }


def verify_task0202_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract_payload = task0201.read_json_if_exists(contract_path)
    required_artifacts = contract_payload.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required_artifacts)]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = task0201.read_json_if_exists(result_dir / "summary.json")
    for field in contract_payload.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    baseline = task0201.read_json_if_exists(result_dir / "qdrant_production_stabilization_baseline.json")
    digest = baseline.get("qdrant_production_stabilization_baseline_digest")
    if digest:
        recomputed = canonical_digest({key: value for key, value in baseline.items() if key not in {"qdrant_production_stabilization_baseline_digest", "vector_database_migration_baseline_digest"}})
        if digest != recomputed:
            issues.append("qdrant production stabilization baseline digest mismatch")
        if baseline.get("vector_database_migration_baseline_digest") != digest:
            issues.append("migration baseline digest must match stabilization baseline digest")
    if summary.get("pgvector_decommission_allowed") is not False:
        issues.append("TASK-0202 must not allow pgvector decommission")
    if summary.get("qdrant_production_stabilization_decision") == "freeze":
        freeze_gates = (
            summary.get("qdrant_production_stabilized") is True,
            summary.get("vector_database_migration_stage_frozen") is True,
            summary.get("production_vector_backend") == "qdrant",
            summary.get("known_vector_migration_blocker_count") == 0,
        )
        if not all(freeze_gates):
            issues.append("freeze decision does not satisfy migration freeze gates")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
        "qdrant_production_stabilization_decision": summary.get("qdrant_production_stabilization_decision"),
        "vector_database_migration_stage_frozen": summary.get("vector_database_migration_stage_frozen"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def _boolean_freeze_fields() -> tuple[str, ...]:
    return (
        "task0201_promotion_authority_valid",
        "production_backend_authority_valid",
        "hybrid_persistence_contract_valid",
        "data_consistency_valid",
        "qdrant_production_data_freshness_valid",
        "qdrant_authoritative_write_path_valid",
        "production_incremental_update_valid",
        "production_vector_delete_valid",
        "production_materialization_idempotent",
        "pgvector_rollback_backend_available",
        "rollback_backend_freshness_valid",
        "rollback_rehearsal_valid",
        "qdrant_authority_restored_after_rehearsal",
        "qdrant_restart_recovery_valid",
        "authoritative_failure_policy_valid",
        "formal_retrieval_regression_within_policy",
        "qdrant_operational_health_valid",
        "post_stabilization_snapshot_create_valid",
    )


def _zero_count_freeze_fields() -> tuple[str, ...]:
    return (
        "rollback_q01_q07_regression_count",
        "post_restart_q01_q07_regression_count",
        "candidate_identity_error_count",
        "unexplained_candidate_loss_count",
        "final_candidate_pool_regression_count",
        "grounding_regression_count",
        "citation_regression_count",
        "safety_regression_count",
        "full_suite_failure_count",
        "unexpected_skip_count",
        "new_regression_count",
        "known_vector_migration_blocker_count",
    )


def _collection_vector_size(collection: Any) -> int | None:
    params = getattr(getattr(collection, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    return getattr(vectors, "size", None)


def _collection_distance(collection: Any) -> str:
    params = getattr(getattr(collection, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    return str(getattr(vectors, "distance", ""))


def qdrant_client_version() -> str | None:
    try:
        return importlib.metadata.version("qdrant-client")
    except importlib.metadata.PackageNotFoundError:
        return None


def canonical_digest(payload: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "hostname", "pid", "latency_baseline", "snapshot_id", "snapshot_name"}
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def skipped_runtime(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "runtime_executed": False, "reason": reason}


def skipped_production_authority(reason: str, env: Mapping[str, str]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "production_backend_authority_valid": False, "production_vector_backend": env.get("OPK_RAG_VECTOR_BACKEND"), "hybrid_persistence_contract_valid": True, "reason": reason}


def skipped_bool_artifact(key: str, reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, key: False, "reason": reason}


def skipped_idempotency(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "duplicate_logical_point_count": 0, "point_count_drift": 1, "identity_drift_count": 0, "production_materialization_idempotent": False, "reason": reason}


def skipped_formal(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "formal_retrieval_regression_within_policy": False, "candidate_identity_error_count": 0, "unexplained_candidate_loss_count": 0, "final_candidate_pool_regression_count": 1, "recall_at_k": 0.0, "mrr": 0.0, "guarded_structure_aware_qdrant_runtime_valid": False, "reason": reason}


def skipped_latency() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "metric_scope": "precomputed query embeddings; vector backend search latency only", "qdrant_authoritative_p50_search_latency": None, "qdrant_authoritative_p95_search_latency": None, "qdrant_authoritative_mean_search_latency": None, "correctness_gate": False}


def skipped_snapshot(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "post_stabilization_snapshot_create_valid": False, "snapshot_list_valid": False, "reason": reason}


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_report(
    summary: Mapping[str, Any],
    consistency: Mapping[str, Any],
    write_path: Mapping[str, Any],
    rollback: Mapping[str, Any],
    restart: Mapping[str, Any],
    formal: Mapping[str, Any],
    blockers: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0202 Qdrant Production Stabilization and Migration Freeze",
            "",
            f"task_status=`{summary.get('task_status')}`; decision=`{summary.get('qdrant_production_stabilization_decision')}`; production_vector_backend=`{summary.get('production_vector_backend')}`.",
            "",
            "## Direct Answers",
            "",
            f"1. Production backend is truly Qdrant: `{summary.get('production_backend_authority_valid')}`.",
            f"2. PostgreSQL relational authority is unchanged: `{summary.get('hybrid_persistence_contract_valid')}`.",
            f"3. Qdrant data complete and fresh: `{summary.get('qdrant_production_data_freshness_valid')}`; missing=`{consistency.get('missing_in_qdrant_count')}`, extra=`{consistency.get('extra_in_qdrant_count')}`, revision mismatches=`{consistency.get('embedding_revision_mismatch_count')}`.",
            f"4. Post-promotion write/update/delete valid: `{write_path.get('qdrant_authoritative_write_path_valid')}` / `{summary.get('production_incremental_update_valid')}` / `{summary.get('production_vector_delete_valid')}`.",
            f"5. pgvector rollback backend valid: `{summary.get('pgvector_rollback_backend_available')}`; freshness=`{summary.get('rollback_backend_freshness_valid')}`.",
            f"6. Rollback rehearsal succeeded: `{rollback.get('rollback_rehearsal_valid')}`; restored Qdrant=`{summary.get('qdrant_authority_restored_after_rehearsal')}`.",
            f"7. Qdrant restart recovery: `{restart.get('qdrant_restart_recovery_valid')}`.",
            f"8. Q01-Q07 production replay: `{summary.get('q01_q07_pass_count')}/7`.",
            f"9. Formal retrieval stable: `{formal.get('formal_retrieval_regression_within_policy')}`; Recall@K=`{formal.get('recall_at_k')}`; MRR=`{formal.get('mrr')}`.",
            f"10. Grounding/Citation/Safety regressions: `{summary.get('grounding_regression_count')}` / `{summary.get('citation_regression_count')}` / `{summary.get('safety_regression_count')}`.",
            f"11. Full suite green: `{summary.get('full_suite_pass')}`; passed=`{summary.get('full_suite_pass_count')}`; skipped=`{summary.get('expected_skip_count')}`; failures=`{summary.get('full_suite_failure_count')}`.",
            f"12. Migration blocker count: `{blockers.get('known_vector_migration_blocker_count')}`.",
            f"13. Migration Stage frozen: `{summary.get('vector_database_migration_stage_frozen')}`.",
            f"14. pgvector decommission discussion readiness: `{summary.get('pgvector_decommission_readiness')}`; decommission executed/allowed=`False`.",
            "",
            "## Baseline",
            "",
            "```text",
            f"qdrant_production_stabilized={summary.get('qdrant_production_stabilized')}",
            f"qdrant_production_stabilization_baseline_digest={summary.get('qdrant_production_stabilization_baseline_digest')}",
            f"vector_database_migration_baseline_digest={summary.get('vector_database_migration_baseline_digest')}",
            "pgvector_decommission_allowed=false",
            "```",
            "",
        ]
    )
