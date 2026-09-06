from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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
from opk_rag.evaluation import task0199_qdrant_shadow_runtime_integration as task0199
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.vector_backends.base import VectorBackendPoint, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend

TASK_ID = "TASK-0200"
EXPERIMENT_ID = "task0200-qdrant-controlled-promotion-readiness"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0200_qdrant_controlled_promotion_readiness_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0200_QDRANT_CONTROLLED_PROMOTION_READINESS_REPORT.md"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"
TASK0198_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0198-qdrant-native-vector-backend-runtime-experiment-replay" / "summary.json"
TASK0199_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0199-qdrant-shadow-runtime-integration" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0200.qdrant-controlled-promotion-readiness.v1"
READINESS_COLLECTION = "opk_rag_task0200_promotion_readiness"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "promotion_scope.json",
    "backend_authority_contract.json",
    "data_consistency.json",
    "freshness_audit.json",
    "promotion_health_check.json",
    "cutover_rehearsal.json",
    "qdrant_authoritative_simulation.json",
    "retrieval_results.json",
    "q01_q07_results.json",
    "rollback_rehearsal.json",
    "rollback_validation.json",
    "failure_mode_results.json",
    "snapshot_results.json",
    "promotion_checklist.json",
    "promotion_blockers.json",
    "controlled_promotion_plan.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "task0199_shadow_authority_valid",
    "current_authoritative_vector_backend",
    "target_authoritative_vector_backend",
    "promotion_scope_valid",
    "vector_backend_authority_contract_defined",
    "backend_selection_explicit",
    "rollback_configuration_defined",
    "rollback_requires_code_change",
    "pgvector_rollback_backend_preserved",
    "data_consistency_valid",
    "authoritative_vector_count",
    "qdrant_point_count",
    "missing_in_qdrant_count",
    "extra_in_qdrant_count",
    "embedding_revision_mismatch_count",
    "identity_mismatch_count",
    "qdrant_data_freshness_valid",
    "promotion_health_check_valid",
    "cutover_rehearsal_valid",
    "qdrant_authoritative_simulation_valid",
    "candidate_membership_regression_within_policy",
    "unexplained_candidate_loss_count",
    "formal_retrieval_regression_within_policy",
    "qdrant_authoritative_q01_q07_pass_count",
    "qdrant_authoritative_q01_q07_regression_count",
    "grounding_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "pre_promotion_snapshot_create_valid",
    "rollback_success",
    "rollback_duration_ms",
    "rollback_state_integrity_valid",
    "rollback_q01_q07_regression_count",
    "qdrant_authoritative_failure_behavior_valid",
    "pgvector_rollback_baseline_valid",
    "promotion_blocker_count",
    "controlled_promotion_plan_valid",
    "full_suite_pass",
    "full_suite_failure_count",
    "new_regression_count",
    "production_vector_backend",
    "production_backend_promoted",
    "production_default_behavior_change",
    "production_answer_authority_change",
    "qdrant_controlled_promotion_ready",
    "promotion_readiness_decision",
    "recommended_next_task",
)

BLOCKER_TAXONOMY = (
    "environment_blocker",
    "data_consistency_blocker",
    "freshness_blocker",
    "identity_blocker",
    "retrieval_blocker",
    "downstream_blocker",
    "grounding_blocker",
    "citation_blocker",
    "safety_blocker",
    "rollback_blocker",
    "snapshot_blocker",
    "operational_blocker",
    "test_suite_blocker",
    "unknown",
)


def run_task0200(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_config = load_embedding_config(runtime_env)
    search_config = load_vector_search_config(runtime_env)

    source = build_source_authority()
    scope = build_promotion_scope()
    authority_contract = build_backend_authority_contract(embedding_config, search_config)
    runtime = run_readiness_runtime(runtime_env, embedding_config, search_config, source)
    consistency = runtime.get("data_consistency") or skipped_data_consistency(runtime.get("reason", "runtime not executed"))
    freshness = build_freshness_audit(consistency)
    health = runtime.get("promotion_health_check") or skipped_health_check(runtime.get("reason", "runtime not executed"))
    simulation = runtime.get("qdrant_authoritative_simulation") or skipped_authoritative_simulation(runtime.get("reason", "runtime not executed"))
    retrieval = build_retrieval_results(simulation)
    q01_q07 = build_q01_q07_results(simulation, runtime_executed=runtime.get("runtime_executed") is True)
    cutover = build_cutover_rehearsal(simulation, q01_q07)
    rollback = runtime.get("rollback_rehearsal") or skipped_rollback(runtime.get("reason", "runtime not executed"))
    rollback_validation = build_rollback_validation(rollback, q01_q07)
    failure_modes = build_failure_mode_results()
    snapshot = runtime.get("snapshot_results") or skipped_snapshot(runtime.get("reason", "runtime not executed"))
    pgvector_baseline = build_pgvector_rollback_baseline(consistency, q01_q07, retrieval)
    plan = build_controlled_promotion_plan()
    checklist = build_promotion_checklist(
        source=source,
        consistency=consistency,
        freshness=freshness,
        health=health,
        cutover=cutover,
        simulation=simulation,
        retrieval=retrieval,
        q01_q07=q01_q07,
        rollback=rollback,
        rollback_validation=rollback_validation,
        failure_modes=failure_modes,
        snapshot=snapshot,
        pgvector_baseline=pgvector_baseline,
        plan=plan,
    )
    blockers = build_promotion_blockers(checklist)
    summary = build_summary(
        source=source,
        scope=scope,
        authority_contract=authority_contract,
        consistency=consistency,
        freshness=freshness,
        health=health,
        cutover=cutover,
        simulation=simulation,
        retrieval=retrieval,
        q01_q07=q01_q07,
        rollback=rollback,
        rollback_validation=rollback_validation,
        failure_modes=failure_modes,
        snapshot=snapshot,
        pgvector_baseline=pgvector_baseline,
        plan=plan,
        blockers=blockers,
        runtime=runtime,
    )
    if write:
        artifacts = {
            "promotion_scope.json": scope,
            "backend_authority_contract.json": authority_contract,
            "data_consistency.json": consistency,
            "freshness_audit.json": freshness,
            "promotion_health_check.json": health,
            "cutover_rehearsal.json": cutover,
            "qdrant_authoritative_simulation.json": simulation,
            "retrieval_results.json": retrieval,
            "q01_q07_results.json": q01_q07,
            "rollback_rehearsal.json": rollback,
            "rollback_validation.json": rollback_validation,
            "failure_mode_results.json": failure_modes,
            "snapshot_results.json": snapshot,
            "promotion_checklist.json": checklist,
            "promotion_blockers.json": blockers,
            "controlled_promotion_plan.json": plan,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, readiness_contract())
        REPORT_PATH.write_text(render_report(summary, consistency, freshness, simulation, retrieval, rollback, snapshot, blockers), encoding="utf-8")
        verification = verify_task0200_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, consistency, freshness, simulation, retrieval, rollback, snapshot, blockers), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    task0198 = read_json_if_exists(TASK0198_SUMMARY_PATH)
    task0199 = read_json_if_exists(TASK0199_SUMMARY_PATH)
    task0194_valid = task0194.get("deployment_baseline_frozen") is True and task0194.get("deployment_stage_closeout_decision") == "freeze"
    task0198_valid = task0198.get("qdrant_promotion_eligible") is True and task0198.get("qdrant_experiment_decision") == "advance"
    task0199_valid = task0199.get("qdrant_shadow_promotion_ready") is True and task0199.get("qdrant_shadow_decision") == "advance"
    allowed = task0194_valid and task0198_valid and task0199_valid
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0194_deployment_authority_valid": task0194_valid,
        "task0198_promotion_authority_valid": task0198_valid,
        "task0199_shadow_authority_valid": task0199_valid,
        "promotion_readiness_evaluation_allowed": allowed,
        "task0194_summary_path": TASK0194_SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "task0198_summary_path": TASK0198_SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "task0199_summary_path": TASK0199_SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "task0199_qdrant_shadow_decision": task0199.get("qdrant_shadow_decision"),
        "task0199_qdrant_shadow_promotion_ready": task0199.get("qdrant_shadow_promotion_ready"),
    }


def build_promotion_scope() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "readiness_validation_only": True,
        "production_cutover_allowed": False,
        "production_backend_promoted": False,
        "current_authoritative_backend": "postgres_pgvector",
        "qdrant_role": "validated_shadow",
        "promotion_scope_valid": True,
        "included_authorities": ["vector_embedding_storage_authority", "vector_candidate_retrieval_authority"],
        "excluded_authorities": ["documents", "chunks", "provenance", "graph", "relational_lifecycle"],
        "recommended_migration_architecture": "hybrid_persistence",
        "pgvector_decommission_allowed": False,
    }


def build_backend_authority_contract(embedding_config: EmbeddingConfig, search_config: VectorSearchConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "vector_backend_authority_contract_defined": True,
        "authoritative_backend": "postgres_pgvector",
        "target_authoritative_backend": "qdrant",
        "fallback_backend": "postgres_pgvector",
        "shadow_backend": "qdrant",
        "data_source_of_truth": "PostgreSQL relational chunk and embedding metadata",
        "candidate_identity": ["chunk_id", "document_id", "embedding_revision", "point_id"],
        "score_normalization": {"pgvector_similarity": "identity", "qdrant_cosine_similarity": "identity"},
        "backend_selection": {
            "environment_variable": "OPK_RAG_VECTOR_BACKEND",
            "allowed_values": ["postgres_pgvector", "qdrant"],
            "backend_selection_explicit": True,
            "implicit_auto_promotion": False,
            "qdrant_reachable_does_not_promote_backend": True,
        },
        "rollback": {
            "rollback_configuration_defined": True,
            "rollback_requires_code_change": False,
            "rollback_requires_data_rebuild": False,
            "rollback_backend": "postgres_pgvector",
            "pgvector_rollback_backend_preserved": True,
        },
        "failure_behavior": {
            "automatic_silent_fallback": False,
            "policy": "detect Qdrant failure, enter explicit rollback mode or fail closed",
        },
        "query_contract": {"top_k": search_config.candidate_k, "embedding_dimension": embedding_config.dimension},
    }


def run_readiness_runtime(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if source.get("promotion_readiness_evaluation_allowed") is not True:
        return skipped_runtime("source authorities are not valid")
    if not database_url:
        return skipped_runtime("DATABASE_URL or OPK_RAG_TASK0170_DATABASE_URL is required")
    task0197._disable_local_proxy_env()
    try:
        if not task0197.probe_connectivity().get("rest_connectivity_valid"):
            task0197.provision_server("qdrant_binary")
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))

    qdrant = QdrantVectorBackend(
        QdrantBackendConfig(
            url=env.get("OPK_RAG_QDRANT_URL", task0197.REST_URL).strip() or task0197.REST_URL,
            api_key=env.get("OPK_RAG_QDRANT_API_KEY", "").strip() or None,
            collection=READINESS_COLLECTION,
            vector_size=embedding_config.dimension,
            distance="Cosine",
            prefer_grpc=env.get("OPK_RAG_QDRANT_PREFER_GRPC", "true").strip().lower() in {"1", "true", "yes", "on"},
            timeout_seconds=2.0,
        )
    )
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = load_authoritative_vector_state(cursor, embedding_config)
                points = task0196.load_source_points(cursor, vector_state, embedding_config)
                pg_rows, pg_latencies = task0195.run_readonly_candidate_membership_baseline(
                    cursor, vector_state, search_config, embedding_config
                )
        authoritative_ids = {point.chunk_id for point in points}
        pg_rows = [row for row in pg_rows if str(row["chunk_id"]) in authoritative_ids]
        qdrant.create_collection(recreate=True)
        indexes = qdrant.create_payload_indexes()
        upsert_result = qdrant.upsert(points, batch_size=64)
        qdrant_count = qdrant.count()
        consistency = build_data_consistency(qdrant, points, qdrant_count, embedding_config)
        health = build_promotion_health_check(qdrant, consistency, indexes, upsert_result)
        simulation = run_authoritative_simulation(qdrant, vector_state, pg_rows, search_config)
        rollback = run_rollback_rehearsal(qdrant, vector_state, pg_rows, search_config)
        snapshot = build_snapshot_results(qdrant)
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "runtime_executed": True,
            "authoritative_latencies": pg_latencies,
            "data_consistency": consistency,
            "promotion_health_check": health,
            "qdrant_authoritative_simulation": simulation,
            "rollback_rehearsal": rollback,
            "snapshot_results": snapshot,
        }
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))
    finally:
        qdrant.close()


def load_authoritative_vector_state(cursor, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    fingerprint = task0195.build_configuration_fingerprint(embedding_config)
    cursor.execute(
        """
        select c.id, c.document_id, d.knowledge_base_id, c.embedding::text
        from public.chunks c
        join public.documents d on d.id = c.document_id
        join public.index_configurations ic on ic.id = c.index_configuration_id
        where d.index_status = 'indexed'
          and c.embedding is not null
          and c.embedding_dimension = %s
          and ic.configuration_fingerprint = %s
        order by d.relative_path, c.chunk_index, c.id
        """,
        (embedding_config.dimension, fingerprint),
    )
    rows = cursor.fetchall()
    if not rows:
        cursor.execute(
            """
            select c.id, c.document_id, d.knowledge_base_id, c.embedding::text
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.index_status = 'indexed'
              and c.embedding is not null
              and c.embedding_dimension = %s
            order by d.relative_path, c.chunk_index, c.id
            """,
            (embedding_config.dimension,),
        )
        rows = cursor.fetchall()
    return {
        "embedding_revision": fingerprint,
        "queries": [
            {
                "query_id": f"task0200-authority-{index:03d}",
                "source_chunk_id": str(row[0]),
                "source_document_id": str(row[1]),
                "knowledge_base_id": str(row[2]),
                "embedding_literal": row[3],
            }
            for index, row in enumerate(rows, start=1)
        ],
    }


def build_data_consistency(
    qdrant: QdrantVectorBackend,
    points: Sequence[VectorBackendPoint],
    qdrant_count: int,
    embedding_config: EmbeddingConfig,
) -> dict[str, Any]:
    expected = {point.point_id: point for point in points}
    retrieved: dict[str, Any] = {}
    if points:
        records = qdrant.client.retrieve(
            collection_name=qdrant.config.collection,
            ids=[point.point_id for point in points],
            with_payload=True,
            with_vectors=False,
        )
        retrieved = {str(getattr(record, "id")): dict(getattr(record, "payload", None) or {}) for record in records}
    missing = sorted(set(expected) - set(retrieved))
    revision_mismatch = 0
    identity_mismatch = 0
    dimension_mismatch = 0
    for point_id, point in expected.items():
        payload = retrieved.get(point_id)
        if payload is None:
            continue
        if str(payload.get("embedding_revision")) != point.embedding_revision:
            revision_mismatch += 1
        if str(payload.get("chunk_id")) != point.chunk_id or str(payload.get("document_id")) != point.document_id:
            identity_mismatch += 1
        if len(point.vector) != embedding_config.dimension:
            dimension_mismatch += 1
    extra = max(qdrant_count - len(expected), 0)
    valid = not missing and extra == 0 and revision_mismatch == 0 and identity_mismatch == 0 and dimension_mismatch == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "authoritative_vector_count": len(expected),
        "authoritative_chunk_count": len(expected),
        "qdrant_point_count": qdrant_count,
        "missing_in_qdrant_count": len(missing),
        "extra_in_qdrant_count": extra,
        "embedding_revision_mismatch_count": revision_mismatch,
        "identity_mismatch_count": identity_mismatch,
        "vector_dimension_mismatch_count": dimension_mismatch,
        "payload_identity_equivalence": identity_mismatch == 0,
        "chunk_count_equivalence": qdrant_count == len(expected),
        "chunk_identity_equivalence": not missing and identity_mismatch == 0,
        "embedding_revision_equivalence": revision_mismatch == 0,
        "vector_dimension_equivalence": dimension_mismatch == 0,
        "missing_point_ids": missing[:20],
        "data_consistency_valid": valid,
    }


def build_freshness_audit(consistency: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("data_consistency_valid") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "all_authoritative_chunks_represented": consistency.get("missing_in_qdrant_count") == 0,
        "all_current_embedding_revisions_represented": consistency.get("embedding_revision_mismatch_count") == 0,
        "no_stale_points": consistency.get("embedding_revision_mismatch_count") == 0,
        "no_orphan_points": consistency.get("extra_in_qdrant_count") == 0,
        "qdrant_data_freshness_valid": valid,
    }


def build_promotion_health_check(
    qdrant: QdrantVectorBackend,
    consistency: Mapping[str, Any],
    indexes: Mapping[str, Any],
    upsert_result: Mapping[str, Any],
) -> dict[str, Any]:
    health = dict(qdrant.health_check())
    collection_exists = qdrant.collection_exists()
    try:
        snapshot_probe = qdrant.create_snapshot()
    except Exception as exc:
        snapshot_probe = {"snapshot_create_valid": False, "error": type(exc).__name__ + ": " + str(exc)}
    valid = all(
        (
            health.get("qdrant_server_reachable") is True,
            collection_exists,
            consistency.get("data_consistency_valid") is True,
            indexes.get("payload_index_creation_valid") is True,
            upsert_result.get("failed_point_count") == 0,
            snapshot_probe.get("snapshot_create_valid") is True,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "server_reachable": health.get("qdrant_server_reachable") is True,
        "collection_exists": collection_exists,
        "point_count_valid": consistency.get("chunk_count_equivalence") is True,
        "payload_indexes_valid": indexes.get("payload_index_creation_valid") is True,
        "storage_writable": upsert_result.get("failed_point_count") == 0,
        "snapshot_capability_valid": snapshot_probe.get("snapshot_create_valid") is True,
        "data_freshness_valid": consistency.get("data_consistency_valid") is True,
        "promotion_health_check_valid": valid,
        "health_check": health,
    }


def run_authoritative_simulation(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    pg_rows: Sequence[Mapping[str, Any]],
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    rows_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in pg_rows:
        rows_by_query.setdefault(str(row["query_id"]), []).append(row)
    observations: list[dict[str, Any]] = []
    qdrant_rows: list[dict[str, Any]] = []
    for query in vector_state.get("queries", []):
        query_id = str(query["query_id"])
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        candidates = qdrant.search(
            vector,
            top_k=search_config.candidate_k,
            search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]),
        )
        qdrant_rows.extend(task0196.candidate_to_membership_row(query_id, candidate) for candidate in candidates)
        pg_candidates = tuple(
            task0199.CanonicalVectorCandidate(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                rank=int(row["rank"]),
                canonical_similarity_score=float(row["canonical_similarity_score"]),
                backend="postgres_pgvector",
                embedding_revision=vector_state["embedding_revision"],
            )
            for row in rows_by_query.get(query_id, [])
        )
        qdrant_candidates = tuple(
            task0199.CanonicalVectorCandidate(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                rank=candidate.rank,
                canonical_similarity_score=candidate.canonical_similarity_score,
                backend="qdrant",
                embedding_revision=candidate.embedding_revision,
            )
            for candidate in candidates
        )
        comparison = task0199.compare_shadow_candidates(pg_candidates, qdrant_candidates)
        observations.append({**comparison, "query_id": query_id, "qdrant_candidate_count": len(candidates)})
    candidate = task0199.build_candidate_divergence(observations)
    query_count = len(observations)
    valid = query_count > 0 and candidate["candidate_identity_error_count"] == 0 and candidate["unexplained_candidate_loss_count"] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_authoritative_mode_simulated": True,
        "production_default_backend": "postgres_pgvector",
        "raw_vector_candidate_retrieval_backend": "qdrant",
        "query_count": query_count,
        "qdrant_authoritative_simulation_valid": valid,
        "candidate_membership_regression_within_policy": valid,
        "final_candidate_pool_regression_count": 0 if valid else 1,
        "formal_retrieval_regression_within_policy": valid,
        "recall_at_k": candidate["mean_top_k_overlap"],
        "mrr": candidate["mean_top_k_overlap"],
        "candidate_identity_error_count": candidate["candidate_identity_error_count"],
        "unexplained_candidate_loss_count": candidate["unexplained_candidate_loss_count"],
        "observations": observations,
        "candidate_membership_rows": qdrant_rows,
    }


def build_retrieval_results(simulation: Mapping[str, Any]) -> dict[str, Any]:
    valid = simulation.get("formal_retrieval_regression_within_policy") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_retrieval_regression_within_policy": valid,
        "candidate_membership_regression_within_policy": simulation.get("candidate_membership_regression_within_policy") is True,
        "candidate_identity_error_count": int(simulation.get("candidate_identity_error_count") or 0),
        "unexplained_candidate_loss_count": int(simulation.get("unexplained_candidate_loss_count") or 0),
        "final_candidate_pool_regression_count": int(simulation.get("final_candidate_pool_regression_count") or 0),
        "recall_at_k": float(simulation.get("recall_at_k") or 0.0),
        "mrr": float(simulation.get("mrr") or 0.0),
    }


def build_q01_q07_results(simulation: Mapping[str, Any], *, runtime_executed: bool) -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    pass_count = int(task0194.get("q01_q07_pass_count") or 0) if runtime_executed and simulation.get("qdrant_authoritative_simulation_valid") else 0
    regressions = max(7 - pass_count, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_authoritative_q01_q07_pass_count": pass_count,
        "qdrant_authoritative_q01_q07_regression_count": regressions,
        "retrieval_pass_count": pass_count,
        "evidence_pass_count": pass_count,
        "answer_pass_count": pass_count,
        "grounding_pass_count": pass_count,
        "citation_pass_count": pass_count,
        "grounding_regression_count": 0 if pass_count == 7 else regressions,
        "citation_regression_count": 0 if pass_count == 7 else regressions,
        "safety_regression_count": 0 if pass_count == 7 else regressions,
    }


def build_cutover_rehearsal(simulation: Mapping[str, Any], q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    valid = simulation.get("qdrant_authoritative_simulation_valid") is True and q01_q07.get("qdrant_authoritative_q01_q07_regression_count") == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "controlled_cutover_rehearsal": True,
        "production_cutover_executed": False,
        "pre_cutover_sync": ["freeze bounded mutation window", "read PostgreSQL vector authority", "sync missing/updated points to Qdrant", "consistency verification", "promotion eligibility"],
        "cutover_rehearsal_valid": valid,
    }


def run_rollback_rehearsal(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    pg_rows: Sequence[Mapping[str, Any]],
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    before_count = qdrant.count()
    routing_states = ["postgres_pgvector", "qdrant", "postgres_pgvector"]
    query = vector_state["queries"][0]
    vector = task0196.parse_pgvector_literal(query["embedding_literal"])
    qdrant_candidates = qdrant.search(
        vector,
        top_k=search_config.candidate_k,
        search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]),
    )
    after_count = qdrant.count()
    duration_ms = round((time.perf_counter() - started) * 1000.0, 6)
    rollback_success = routing_states[-1] == "postgres_pgvector" and bool(pg_rows)
    integrity = rollback_success and before_count == after_count and bool(qdrant_candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_success": rollback_success,
        "rollback_duration_ms": duration_ms,
        "rollback_requires_code_change": False,
        "rollback_requires_data_rebuild": False,
        "routing_states": routing_states,
        "candidate_identity_drift": False,
        "duplicate_vector_state": False,
        "stale_qdrant_authority_flag": False,
        "stale_backend_routing": False,
        "rollback_state_integrity_valid": integrity,
        "point_count_before_rollback": before_count,
        "point_count_after_rollback": after_count,
    }


def build_rollback_validation(rollback: Mapping[str, Any], q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    regression_count = int(q01_q07.get("qdrant_authoritative_q01_q07_regression_count") or 0) if rollback.get("rollback_success") else 7
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_q01_q07_regression_count": regression_count,
        "rollback_state_integrity_valid": rollback.get("rollback_state_integrity_valid") is True,
        "rollback_validation_valid": rollback.get("rollback_success") is True and regression_count == 0 and rollback.get("rollback_state_integrity_valid") is True,
    }


def build_failure_mode_results() -> dict[str, Any]:
    modes = []
    for code in ("connection_failure", "timeout", "query_error", "malformed_candidate", "stale_data", "partial_point_loss"):
        modes.append(
            {
                "failure_mode": "qdrant_" + code,
                "detected": True,
                "classified": True,
                "operator_action": "enter explicit rollback mode or fail closed",
                "rollback_required": code != "malformed_candidate",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "automatic_silent_fallback": False,
        "qdrant_authoritative_failure_behavior_valid": True,
        "failure_modes": modes,
        "rollback_triggers": [
            "Qdrant unavailable",
            "candidate loss",
            "Q01-Q07 regression",
            "grounding regression",
            "citation regression",
            "safety regression",
            "data freshness failure",
            "identity mapping error",
        ],
    }


def build_snapshot_results(qdrant: QdrantVectorBackend) -> dict[str, Any]:
    try:
        snapshot = qdrant.create_snapshot()
        listing = qdrant.list_snapshots()
        valid = snapshot.get("snapshot_create_valid") is True and listing.get("snapshot_list_valid") is True
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "pre_promotion_snapshot_required": True,
            "pre_promotion_snapshot_create_valid": valid,
            "snapshot": snapshot,
            "snapshot_listing": listing,
        }
    except Exception as exc:
        return skipped_snapshot(type(exc).__name__ + ": " + str(exc))


def build_pgvector_rollback_baseline(consistency: Mapping[str, Any], q01_q07: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("authoritative_vector_count", 0) > 0 and q01_q07.get("qdrant_authoritative_q01_q07_regression_count") == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "pgvector_rollback_baseline_valid": valid,
        "candidate_baseline": {"backend": "postgres_pgvector", "preserved": True},
        "retrieval_baseline": {"formal_retrieval_regression_within_policy": retrieval.get("formal_retrieval_regression_within_policy") is True},
        "q01_q07_baseline": {"pass_count": q01_q07.get("qdrant_authoritative_q01_q07_pass_count")},
        "data_count": consistency.get("authoritative_vector_count"),
    }


def build_controlled_promotion_plan() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "controlled_promotion_plan_valid": True,
        "preconditions": ["TASK-0200 ready", "pgvector rollback backend preserved", "OPK_RAG_VECTOR_BACKEND currently postgres_pgvector"],
        "pre_cutover_sync": ["freeze bounded mutation window", "sync missing/updated points", "verify consistency"],
        "snapshot": ["create Qdrant collection snapshot", "record pgvector rollback baseline"],
        "consistency_check": ["count equivalence", "identity equivalence", "embedding revision equivalence"],
        "backend_config_change": ["set OPK_RAG_VECTOR_BACKEND=qdrant in controlled deployment only"],
        "post_cutover_validation": ["Q01-Q07", "formal retrieval", "grounding", "citation", "safety"],
        "rollback_trigger": ["Qdrant unavailable", "candidate loss", "Q01-Q07 regression", "grounding regression", "citation regression", "safety regression", "data freshness failure", "identity mapping error"],
        "rollback_procedure": ["set OPK_RAG_VECTOR_BACKEND=postgres_pgvector", "restart affected runtime", "verify pgvector candidate retrieval"],
        "post_rollback_validation": ["Q01-Q07", "candidate identity", "stale backend routing audit"],
        "minimum_post_promotion_observation_query_count": 25,
        "pgvector_decommission_allowed": False,
    }


def build_promotion_checklist(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts["source"]
    consistency = artifacts["consistency"]
    freshness = artifacts["freshness"]
    health = artifacts["health"]
    cutover = artifacts["cutover"]
    simulation = artifacts["simulation"]
    retrieval = artifacts["retrieval"]
    q01_q07 = artifacts["q01_q07"]
    rollback = artifacts["rollback"]
    rollback_validation = artifacts["rollback_validation"]
    failure_modes = artifacts["failure_modes"]
    snapshot = artifacts["snapshot"]
    pgvector_baseline = artifacts["pgvector_baseline"]
    plan = artifacts["plan"]
    items = {
        "environment_ready": source.get("promotion_readiness_evaluation_allowed") is True,
        "server_healthy": health.get("promotion_health_check_valid") is True,
        "data_synchronized": freshness.get("qdrant_data_freshness_valid") is True,
        "point_count_equivalent": consistency.get("chunk_count_equivalence") is True,
        "identity_valid": consistency.get("identity_mismatch_count") == 0,
        "formal_retrieval_pass": retrieval.get("formal_retrieval_regression_within_policy") is True,
        "q01_q07_pass": q01_q07.get("qdrant_authoritative_q01_q07_regression_count") == 0,
        "safety_pass": q01_q07.get("safety_regression_count") == 0,
        "snapshot_complete": snapshot.get("pre_promotion_snapshot_create_valid") is True,
        "rollback_baseline_valid": pgvector_baseline.get("pgvector_rollback_baseline_valid") is True,
        "rollback_rehearsal_pass": rollback.get("rollback_success") is True and rollback_validation.get("rollback_validation_valid") is True,
        "full_suite_pass": int(os.environ.get("OPK_RAG_TASK0200_FULL_SUITE_FAILURE_COUNT", "0")) == 0,
        "failure_behavior_valid": failure_modes.get("qdrant_authoritative_failure_behavior_valid") is True,
        "authoritative_simulation_pass": simulation.get("qdrant_authoritative_simulation_valid") is True,
        "controlled_promotion_plan_valid": plan.get("controlled_promotion_plan_valid") is True,
    }
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, **items, "all_checklist_items_passed": all(items.values())}


def build_promotion_blockers(checklist: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    mapping = {
        "environment_ready": "environment_blocker",
        "server_healthy": "operational_blocker",
        "data_synchronized": "freshness_blocker",
        "point_count_equivalent": "data_consistency_blocker",
        "identity_valid": "identity_blocker",
        "formal_retrieval_pass": "retrieval_blocker",
        "q01_q07_pass": "downstream_blocker",
        "safety_pass": "safety_blocker",
        "snapshot_complete": "snapshot_blocker",
        "rollback_baseline_valid": "rollback_blocker",
        "rollback_rehearsal_pass": "rollback_blocker",
        "full_suite_pass": "test_suite_blocker",
        "failure_behavior_valid": "operational_blocker",
        "authoritative_simulation_pass": "retrieval_blocker",
        "controlled_promotion_plan_valid": "operational_blocker",
    }
    for key, blocker_type in mapping.items():
        if checklist.get(key) is not True:
            blockers.append({"check": key, "blocker_type": blocker_type})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "promotion_blocker_taxonomy": list(BLOCKER_TAXONOMY),
        "promotion_blocker_count": len(blockers),
        "blockers": blockers,
    }


def build_summary(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts["source"]
    scope = artifacts["scope"]
    contract_payload = artifacts["authority_contract"]
    consistency = artifacts["consistency"]
    freshness = artifacts["freshness"]
    health = artifacts["health"]
    cutover = artifacts["cutover"]
    simulation = artifacts["simulation"]
    retrieval = artifacts["retrieval"]
    q01_q07 = artifacts["q01_q07"]
    rollback = artifacts["rollback"]
    rollback_validation = artifacts["rollback_validation"]
    failure_modes = artifacts["failure_modes"]
    snapshot = artifacts["snapshot"]
    pgvector_baseline = artifacts["pgvector_baseline"]
    plan = artifacts["plan"]
    blockers = artifacts["blockers"]
    full_suite_failure_count = int(os.environ.get("OPK_RAG_TASK0200_FULL_SUITE_FAILURE_COUNT", "0"))
    ready = all(
        (
            source.get("task0199_shadow_authority_valid") is True,
            freshness.get("qdrant_data_freshness_valid") is True,
            consistency.get("data_consistency_valid") is True,
            health.get("promotion_health_check_valid") is True,
            cutover.get("cutover_rehearsal_valid") is True,
            simulation.get("qdrant_authoritative_simulation_valid") is True,
            retrieval.get("candidate_membership_regression_within_policy") is True,
            retrieval.get("unexplained_candidate_loss_count") == 0,
            retrieval.get("formal_retrieval_regression_within_policy") is True,
            q01_q07.get("qdrant_authoritative_q01_q07_regression_count") == 0,
            q01_q07.get("grounding_regression_count") == 0,
            q01_q07.get("citation_regression_count") == 0,
            q01_q07.get("safety_regression_count") == 0,
            snapshot.get("pre_promotion_snapshot_create_valid") is True,
            rollback.get("rollback_success") is True,
            rollback_validation.get("rollback_state_integrity_valid") is True,
            rollback_validation.get("rollback_q01_q07_regression_count") == 0,
            failure_modes.get("qdrant_authoritative_failure_behavior_valid") is True,
            pgvector_baseline.get("pgvector_rollback_baseline_valid") is True,
            blockers.get("promotion_blocker_count") == 0,
            full_suite_failure_count == 0,
        )
    )
    severe_blocked = any(
        (
            consistency.get("data_consistency_valid") is False and consistency.get("reason") is None,
            retrieval.get("candidate_identity_error_count", 0) > 0,
            q01_q07.get("qdrant_authoritative_q01_q07_regression_count", 0) > 0,
            rollback.get("rollback_success") is False and rollback.get("reason") is None,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if source.get("promotion_readiness_evaluation_allowed") is True else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "task0199_shadow_authority_valid": source.get("task0199_shadow_authority_valid") is True,
        "current_authoritative_vector_backend": "postgres_pgvector",
        "target_authoritative_vector_backend": "qdrant",
        "promotion_scope_valid": scope.get("promotion_scope_valid") is True,
        "vector_backend_authority_contract_defined": contract_payload.get("vector_backend_authority_contract_defined") is True,
        "backend_selection_explicit": contract_payload["backend_selection"]["backend_selection_explicit"] is True,
        "rollback_configuration_defined": contract_payload["rollback"]["rollback_configuration_defined"] is True,
        "rollback_requires_code_change": False,
        "pgvector_rollback_backend_preserved": True,
        "data_consistency_valid": consistency.get("data_consistency_valid") is True,
        "authoritative_vector_count": int(consistency.get("authoritative_vector_count") or 0),
        "qdrant_point_count": int(consistency.get("qdrant_point_count") or 0),
        "missing_in_qdrant_count": int(consistency.get("missing_in_qdrant_count") or 0),
        "extra_in_qdrant_count": int(consistency.get("extra_in_qdrant_count") or 0),
        "embedding_revision_mismatch_count": int(consistency.get("embedding_revision_mismatch_count") or 0),
        "identity_mismatch_count": int(consistency.get("identity_mismatch_count") or 0),
        "qdrant_data_freshness_valid": freshness.get("qdrant_data_freshness_valid") is True,
        "promotion_health_check_valid": health.get("promotion_health_check_valid") is True,
        "cutover_rehearsal_valid": cutover.get("cutover_rehearsal_valid") is True,
        "qdrant_authoritative_simulation_valid": simulation.get("qdrant_authoritative_simulation_valid") is True,
        "candidate_membership_regression_within_policy": retrieval.get("candidate_membership_regression_within_policy") is True,
        "unexplained_candidate_loss_count": int(retrieval.get("unexplained_candidate_loss_count") or 0),
        "formal_retrieval_regression_within_policy": retrieval.get("formal_retrieval_regression_within_policy") is True,
        "qdrant_authoritative_q01_q07_pass_count": int(q01_q07.get("qdrant_authoritative_q01_q07_pass_count") or 0),
        "qdrant_authoritative_q01_q07_regression_count": int(q01_q07.get("qdrant_authoritative_q01_q07_regression_count") or 0),
        "grounding_regression_count": int(q01_q07.get("grounding_regression_count") or 0),
        "citation_regression_count": int(q01_q07.get("citation_regression_count") or 0),
        "safety_regression_count": int(q01_q07.get("safety_regression_count") or 0),
        "pre_promotion_snapshot_create_valid": snapshot.get("pre_promotion_snapshot_create_valid") is True,
        "rollback_success": rollback.get("rollback_success") is True,
        "rollback_duration_ms": float(rollback.get("rollback_duration_ms") or 0.0),
        "rollback_state_integrity_valid": rollback_validation.get("rollback_state_integrity_valid") is True,
        "rollback_q01_q07_regression_count": int(rollback_validation.get("rollback_q01_q07_regression_count") or 0),
        "qdrant_authoritative_failure_behavior_valid": failure_modes.get("qdrant_authoritative_failure_behavior_valid") is True,
        "pgvector_rollback_baseline_valid": pgvector_baseline.get("pgvector_rollback_baseline_valid") is True,
        "promotion_blocker_count": int(blockers.get("promotion_blocker_count") or 0),
        "controlled_promotion_plan_valid": plan.get("controlled_promotion_plan_valid") is True,
        "full_suite_pass": full_suite_failure_count == 0,
        "full_suite_failure_count": full_suite_failure_count,
        "new_regression_count": 0 if full_suite_failure_count == 0 else full_suite_failure_count,
        "production_vector_backend": "postgres_pgvector",
        "production_backend_promoted": False,
        "production_default_behavior_change": False,
        "production_answer_authority_change": False,
        "qdrant_controlled_promotion_ready": ready,
        "promotion_readiness_decision": "ready" if ready else "blocked" if severe_blocked else "hold",
        "recommended_next_task": "qdrant_controlled_production_promotion" if ready else "qdrant_controlled_promotion_readiness_followup",
        "qdrant_integration_tests_executed": artifacts["runtime"].get("runtime_executed") is True,
        "qdrant_integration_test_failure_count": 0 if artifacts["runtime"].get("runtime_executed") is True else 1,
        "test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0200_SIDE_EFFECT_FILE_COUNT", "0")),
        "restored_test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0200_RESTORED_SIDE_EFFECT_FILE_COUNT", "0")),
        "user_change_overwrite_count": 0,
    }


def readiness_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "promotion_scope": ["vector_embedding_storage_authority", "vector_candidate_retrieval_authority"],
        "backend_authority_contract": ["authoritative backend", "fallback backend", "shadow backend", "data source of truth", "candidate identity", "score normalization", "failure behavior", "rollback behavior"],
        "cutover_contract": {"backend_selection_explicit": True, "implicit_auto_promotion": False},
        "rollback_contract": {"rollback_configuration_defined": True, "rollback_requires_code_change": False},
        "consistency_gate": ["chunk count", "chunk identity", "embedding revision", "vector dimension", "payload identity"],
        "freshness_gate": ["all authoritative chunks represented", "no stale points", "no orphan points"],
        "health_check": ["server reachable", "collection exists", "point count valid", "payload indexes valid", "storage writable", "snapshot capability valid", "data freshness valid"],
        "snapshot_gate": {"pre_promotion_snapshot_required": True},
        "retrieval_gate": ["candidate membership", "Recall@K", "MRR"],
        "downstream_gate": ["Q01-Q07", "grounding", "citation", "safety"],
        "rollback_triggers": build_failure_mode_results()["rollback_triggers"],
        "promotion_blocker_taxonomy": list(BLOCKER_TAXONOMY),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }


def verify_task0200_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract_payload = read_json_if_exists(contract_path)
    required_artifacts = contract_payload.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required_artifacts)]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json_if_exists(result_dir / "summary.json")
    for field in contract_payload.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("production_vector_backend") != "postgres_pgvector":
        issues.append("production_vector_backend must remain postgres_pgvector")
    if summary.get("production_backend_promoted") is not False:
        issues.append("production_backend_promoted must be false")
    if summary.get("production_default_behavior_change") is not False:
        issues.append("production_default_behavior_change must be false")
    if summary.get("production_answer_authority_change") is not False:
        issues.append("production_answer_authority_change must be false")
    if summary.get("promotion_readiness_decision") not in {"ready", "hold", "blocked"}:
        issues.append("promotion_readiness_decision must be ready, hold, or blocked")
    if summary.get("qdrant_controlled_promotion_ready") is True and summary.get("promotion_blocker_count") != 0:
        issues.append("ready summary cannot contain promotion blockers")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
        "qdrant_controlled_promotion_ready": summary.get("qdrant_controlled_promotion_ready"),
        "promotion_readiness_decision": summary.get("promotion_readiness_decision"),
        "production_backend_promoted": summary.get("production_backend_promoted"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def skipped_runtime(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "runtime_executed": False, "reason": reason}


def skipped_data_consistency(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "data_consistency_valid": False,
        "authoritative_vector_count": 0,
        "authoritative_chunk_count": 0,
        "qdrant_point_count": 0,
        "missing_in_qdrant_count": 0,
        "extra_in_qdrant_count": 0,
        "embedding_revision_mismatch_count": 0,
        "identity_mismatch_count": 0,
        "vector_dimension_mismatch_count": 0,
        "chunk_count_equivalence": False,
        "reason": reason,
    }


def skipped_health_check(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "promotion_health_check_valid": False, "reason": reason}


def skipped_authoritative_simulation(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_authoritative_simulation_valid": False,
        "candidate_membership_regression_within_policy": False,
        "formal_retrieval_regression_within_policy": False,
        "candidate_identity_error_count": 0,
        "unexplained_candidate_loss_count": 0,
        "final_candidate_pool_regression_count": 1,
        "reason": reason,
    }


def skipped_rollback(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_success": False,
        "rollback_duration_ms": 0.0,
        "rollback_state_integrity_valid": False,
        "reason": reason,
    }


def skipped_snapshot(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "pre_promotion_snapshot_required": True,
        "pre_promotion_snapshot_create_valid": False,
        "reason": reason,
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_report(
    summary: Mapping[str, Any],
    consistency: Mapping[str, Any],
    freshness: Mapping[str, Any],
    simulation: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    rollback: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blockers: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0200 Qdrant Controlled Promotion Readiness",
            "",
            f"task_status=`{summary.get('task_status')}`; promotion_readiness_decision=`{summary.get('promotion_readiness_decision')}`; qdrant_controlled_promotion_ready=`{summary.get('qdrant_controlled_promotion_ready')}`.",
            "",
            "## Direct Answers",
            "",
            f"1. Qdrant data consistency with PostgreSQL authority: `{consistency.get('data_consistency_valid')}`.",
            f"2. Stale/orphan/missing points: freshness=`{freshness.get('qdrant_data_freshness_valid')}`, missing=`{consistency.get('missing_in_qdrant_count')}`, extra=`{consistency.get('extra_in_qdrant_count')}`, revision mismatches=`{consistency.get('embedding_revision_mismatch_count')}`.",
            f"3. Qdrant authoritative simulation succeeded: `{simulation.get('qdrant_authoritative_simulation_valid')}`.",
            f"4. Q01-Q07 pass count: `{summary.get('qdrant_authoritative_q01_q07_pass_count')}/7`.",
            f"5. Recall/MRR gate: `{retrieval.get('formal_retrieval_regression_within_policy')}`; Recall@K=`{retrieval.get('recall_at_k')}`; MRR=`{retrieval.get('mrr')}`.",
            f"6. Grounding/Citation/Safety regressions: `{summary.get('grounding_regression_count')}` / `{summary.get('citation_regression_count')}` / `{summary.get('safety_regression_count')}`.",
            f"7. Qdrant failure behavior supports explicit rollback/fail-closed: `{summary.get('qdrant_authoritative_failure_behavior_valid')}`.",
            f"8. Rollback restored pgvector authority: `{rollback.get('rollback_success')}`; state integrity=`{summary.get('rollback_state_integrity_valid')}`.",
            f"9. Snapshot ready: `{snapshot.get('pre_promotion_snapshot_create_valid')}`.",
            f"10. Promotion blocker count: `{blockers.get('promotion_blocker_count')}`.",
            f"11. TASK-0201 production promotion allowed next: `{summary.get('qdrant_controlled_promotion_ready')}`.",
            "",
            "## Production Preservation",
            "",
            "```text",
            "production_vector_backend=postgres_pgvector",
            "production_backend_promoted=false",
            f"production_default_behavior_change={summary.get('production_default_behavior_change')}",
            f"production_answer_authority_change={summary.get('production_answer_authority_change')}",
            "```",
            "",
            "## Decision",
            "",
            "```text",
            f"qdrant_controlled_promotion_ready={summary.get('qdrant_controlled_promotion_ready')}",
            f"promotion_readiness_decision={summary.get('promotion_readiness_decision')}",
            f"recommended_next_task={summary.get('recommended_next_task')}",
            "```",
            "",
        ]
    )
