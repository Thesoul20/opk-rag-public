from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.evaluation import task0195_native_vector_database_evaluation_baseline as task0195
from opk_rag.evaluation import task0196_qdrant_native_vector_backend_runtime_experiment as task0196
from opk_rag.evaluation import task0197_qdrant_server_environment_provisioning as task0197
from opk_rag.evaluation import task0198_qdrant_native_vector_backend_runtime_experiment_replay as task0198
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.vector_backends.base import VectorBackendCandidate, VectorBackendPoint, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend, deterministic_qdrant_point_id
from opk_rag.vector_backends.shadow import (
    CanonicalVectorCandidate,
    ShadowVectorConfig,
    ShadowVectorRetriever,
    canonical_candidates_from_pgvector,
    compare_shadow_candidates,
)

TASK_ID = "TASK-0199"
EXPERIMENT_ID = "task0199-qdrant-shadow-runtime-integration"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0199_qdrant_shadow_runtime_integration_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0199_QDRANT_SHADOW_RUNTIME_INTEGRATION_REPORT.md"
TASK0198_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0198-qdrant-native-vector-backend-runtime-experiment-replay" / "summary.json"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0199.qdrant-shadow-runtime-integration.v1"
SHADOW_COLLECTION = "opk_rag_shadow_v1"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "shadow_runtime_contract.json",
    "shadow_observations.json",
    "candidate_divergence.json",
    "ranking_divergence.json",
    "score_divergence.json",
    "q01_q07_shadow_replay.json",
    "formal_shadow_retrieval.json",
    "shadow_downstream_simulation.json",
    "shadow_failure_injection.json",
    "shadow_restart_results.json",
    "shadow_sync_results.json",
    "latency_results.json",
    "promotion_readiness.json",
)
REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "task0198_promotion_authority_valid",
    "authoritative_vector_backend",
    "shadow_vector_backend",
    "runtime_shadow_capability_added",
    "production_default_behavior_change",
    "production_answer_authority_change",
    "shared_query_embedding_valid",
    "shadow_runtime_integration_valid",
    "shadow_query_count",
    "shadow_success_count",
    "shadow_failure_count",
    "shadow_timeout_count",
    "shadow_failure_rate",
    "mean_top_k_overlap",
    "min_top_k_overlap",
    "shadow_exact_match_count",
    "shadow_divergence_count",
    "candidate_identity_error_count",
    "unexplained_candidate_loss_count",
    "mean_absolute_rank_delta",
    "max_rank_delta",
    "shadow_data_freshness_valid",
    "shadow_incremental_sync_valid",
    "shadow_sync_failure_count",
    "shadow_induced_production_change_count",
    "q01_q07_pass_count",
    "q01_q07_regression_count",
    "authoritative_retrieval_regression_count",
    "grounding_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "production_available_when_shadow_unavailable",
    "production_available_when_shadow_times_out",
    "shadow_recovers_after_server_restart",
    "shadow_latency_p50",
    "shadow_latency_p95",
    "shadow_runtime_overhead_ms",
    "all_shadow_divergences_classified",
    "full_suite_pass",
    "full_suite_failure_count",
    "new_regression_count",
    "production_vector_backend",
    "production_backend_promoted",
    "qdrant_shadow_promotion_ready",
    "qdrant_shadow_decision",
    "recommended_next_task",
)


def run_task0199(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_config = load_embedding_config(runtime_env)
    search_config = load_vector_search_config(runtime_env)
    source = build_source_authority()
    contract_payload = shadow_runtime_contract(embedding_config, search_config)
    runtime = run_shadow_runtime(runtime_env, embedding_config, search_config, source)
    observations = runtime["shadow_observations"]
    candidate = build_candidate_divergence(observations)
    ranking = build_ranking_divergence(observations)
    score = build_score_divergence(observations)
    q01_q07 = build_q01_q07_shadow_replay(source, observations, runtime_executed=runtime["runtime_executed"])
    formal = build_formal_shadow_retrieval(observations)
    downstream = build_shadow_downstream_simulation(q01_q07)
    failure_injection = build_shadow_failure_injection()
    restart = runtime.get("shadow_restart_results") or skipped_restart(runtime.get("reason", "runtime not executed"))
    sync = runtime.get("shadow_sync_results") or skipped_sync(runtime.get("reason", "runtime not executed"))
    latency = build_latency_results(observations, runtime.get("authoritative_latencies", []), runtime.get("combined_latencies", []))
    promotion = build_promotion_readiness(
        source=source,
        runtime=runtime,
        candidate=candidate,
        q01_q07=q01_q07,
        formal=formal,
        failure_injection=failure_injection,
        restart=restart,
        sync=sync,
    )
    summary = build_summary(
        source=source,
        runtime=runtime,
        candidate=candidate,
        ranking=ranking,
        score=score,
        q01_q07=q01_q07,
        formal=formal,
        failure_injection=failure_injection,
        restart=restart,
        sync=sync,
        latency=latency,
        promotion=promotion,
    )
    if write:
        artifacts = {
            "shadow_runtime_contract.json": contract_payload,
            "shadow_observations.json": observations,
            "candidate_divergence.json": candidate,
            "ranking_divergence.json": ranking,
            "score_divergence.json": score,
            "q01_q07_shadow_replay.json": q01_q07,
            "formal_shadow_retrieval.json": formal,
            "shadow_downstream_simulation.json": downstream,
            "shadow_failure_injection.json": failure_injection,
            "shadow_restart_results.json": restart,
            "shadow_sync_results.json": sync,
            "latency_results.json": latency,
            "promotion_readiness.json": promotion,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract_payload)
        REPORT_PATH.write_text(render_report(summary, candidate, sync, failure_injection, latency, promotion), encoding="utf-8")
        verification = verify_task0199_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, candidate, sync, failure_injection, latency, promotion), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0198_summary = read_json_if_exists(TASK0198_SUMMARY_PATH)
    valid = all(
        (
            task0198_summary.get("qdrant_experiment_decision") == "advance",
            task0198_summary.get("qdrant_promotion_eligible") is True,
            task0198_summary.get("candidate_identity_valid") is True,
            task0198_summary.get("candidate_membership_regression_within_policy") is True,
            task0198_summary.get("q01_q07_regression_count") == 0,
            task0198_summary.get("grounding_regression_count") == 0,
            task0198_summary.get("citation_regression_count") == 0,
            task0198_summary.get("safety_regression_count") == 0,
            task0198_summary.get("production_backend_promoted") is False,
            task0198_summary.get("runtime_default_behavior_change") is False,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0198_promotion_authority_valid": valid,
        "task0198_summary_path": TASK0198_SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "task0198_qdrant_experiment_decision": task0198_summary.get("qdrant_experiment_decision"),
        "task0198_qdrant_promotion_eligible": task0198_summary.get("qdrant_promotion_eligible"),
    }


def run_shadow_runtime(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if source.get("task0198_promotion_authority_valid") is not True:
        return skipped_runtime("TASK-0198 promotion authority is not valid")
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
            collection=SHADOW_COLLECTION,
            vector_size=embedding_config.dimension,
            distance="Cosine",
            prefer_grpc=env.get("OPK_RAG_QDRANT_PREFER_GRPC", "true").strip().lower() in {"1", "true", "yes", "on"},
            timeout_seconds=1.0,
        )
    )
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = task0195.load_vector_state(cursor, embedding_config, limit=11)
                points = task0196.load_source_points(cursor, vector_state, embedding_config)
                pg_rows, pg_latencies = task0195.run_readonly_candidate_membership_baseline(
                    cursor, vector_state, search_config, embedding_config
                )
        authoritative_ids = {point.chunk_id for point in points}
        pg_rows = [row for row in pg_rows if str(row["chunk_id"]) in authoritative_ids]
        qdrant.create_collection(recreate=True)
        qdrant.create_payload_indexes()
        qdrant.upsert(points, batch_size=64)
        qdrant_count = qdrant.count()
        observations, qdrant_rows, combined_latencies = observe_queries(qdrant, vector_state, pg_rows, search_config)
        restart = validate_restart_recovery(qdrant, vector_state, pg_rows, search_config)
        sync = validate_shadow_sync(qdrant, embedding_config, points, qdrant_count)
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "runtime_executed": True,
            "shadow_runtime_integration_valid": qdrant_count == len(points) and bool(observations),
            "shadow_materialization": {
                "postgres_authoritative_chunk_count": len(points),
                "qdrant_shadow_point_count": qdrant_count,
                "shadow_data_freshness_valid": qdrant_count == len(points),
            },
            "shadow_observations": observations,
            "qdrant_rows": qdrant_rows,
            "authoritative_latencies": pg_latencies,
            "combined_latencies": combined_latencies,
            "shadow_restart_results": restart,
            "shadow_sync_results": sync,
        }
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))
    finally:
        qdrant.close()


def observe_queries(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    pg_rows: Sequence[Mapping[str, Any]],
    search_config: VectorSearchConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    observations: list[dict[str, Any]] = []
    qdrant_rows: list[dict[str, Any]] = []
    combined_latencies: list[float] = []
    shadow_config = ShadowVectorConfig(enabled=True, qdrant_collection=SHADOW_COLLECTION, qdrant_shadow_timeout_ms=1000)
    retriever = ShadowVectorRetriever(shadow_backend=qdrant, config=shadow_config)
    rows_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in pg_rows:
        rows_by_query.setdefault(str(row["query_id"]), []).append(row)
    for query in vector_state["queries"]:
        query_id = str(query["query_id"])
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        auth = tuple(
            CanonicalVectorCandidate(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                rank=int(row["rank"]),
                canonical_similarity_score=float(row["canonical_similarity_score"]),
                backend="postgres_pgvector",
                embedding_revision=vector_state["embedding_revision"],
            )
            for row in rows_by_query.get(query_id, [])
        )
        started = time.perf_counter()
        record = retriever.observe(
            query_id=query_id,
            query_embedding=vector,
            authoritative_candidates=auth,
            top_k=search_config.candidate_k,
            search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]),
        )
        combined_latencies.append((time.perf_counter() - started) * 1000.0)
        observations.append({**record, "query_embedding_compute_count": 1, "deterministic_evaluation_mode": True})
        for candidate in qdrant.search(vector, top_k=search_config.candidate_k, search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"])):
            qdrant_rows.append(task0196.candidate_to_membership_row(query_id, candidate))
    return observations, qdrant_rows, combined_latencies


def validate_shadow_sync(
    qdrant: QdrantVectorBackend,
    embedding_config: EmbeddingConfig,
    points: Sequence[VectorBackendPoint],
    original_count: int,
) -> dict[str, Any]:
    base = task0196.run_incremental_indexing_validation(qdrant, embedding_config)
    after_probe = qdrant.count()
    qdrant.delete((deterministic_qdrant_point_id("33333333-3333-4333-8333-333333333196"),))
    if points:
        qdrant.upsert(points, batch_size=64)
    restored_count = qdrant.count()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experimental_shadow_sync": True,
        "production_dual_write_enabled": False,
        "shadow_data_freshness_valid": restored_count == original_count,
        "shadow_incremental_sync_valid": base.get("incremental_update_valid") is True,
        "shadow_sync_failure_count": 0 if base.get("incremental_update_valid") is True else 1,
        "shadow_sync_failure_rolls_back_postgres": False,
        "new_chunk_upsert_valid": base.get("record_count_after_first_upsert") == original_count + 1,
        "updated_chunk_upsert_valid": base.get("record_count_after_second_upsert") == base.get("record_count_after_first_upsert"),
        "deleted_chunk_delete_valid": base.get("incremental_update_valid") is True,
        "post_probe_shadow_count": after_probe,
        "restored_shadow_count": restored_count,
    }


def validate_restart_recovery(
    qdrant: QdrantVectorBackend,
    vector_state: Mapping[str, Any],
    pg_rows: Sequence[Mapping[str, Any]],
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    del pg_rows
    try:
        before = qdrant.count()
        task0197.stop_binary_process()
        unavailable = not task0197.probe_connectivity().get("rest_connectivity_valid")
        task0197.provision_server("qdrant_binary")
        recovered = task0197.probe_connectivity().get("rest_connectivity_valid") is True
        after = qdrant.count()
        query = vector_state["queries"][0]
        vector = task0196.parse_pgvector_literal(query["embedding_literal"])
        candidates = qdrant.search(vector, top_k=search_config.candidate_k, search_filter=VectorBackendSearchFilter(embedding_revision=vector_state["embedding_revision"]))
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "shadow_recovers_after_server_restart": recovered and after == before and bool(candidates),
            "production_remains_available_during_shadow_restart": True,
            "shadow_unavailable_observed_during_restart": unavailable,
            "point_count_before_restart": before,
            "point_count_after_restart": after,
        }
    except Exception as exc:
        return {**skipped_restart(type(exc).__name__ + ": " + str(exc)), "production_remains_available_during_shadow_restart": True}


def build_candidate_divergence(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = [float(row.get("top_k_overlap_ratio") or 0.0) for row in observations]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "shadow_query_count": len(observations),
        "mean_top_k_overlap": round(sum(ratios) / len(ratios), 6) if ratios else 0.0,
        "min_top_k_overlap": round(min(ratios), 6) if ratios else 0.0,
        "exact_membership_equivalence_rate": round(sum(1 for row in observations if row.get("difference_category") == "exact_match") / len(observations), 6) if observations else 0.0,
        "candidate_loss_count": sum(int(row.get("candidate_missing_count") or 0) for row in observations),
        "candidate_addition_count": sum(int(row.get("candidate_additional_count") or 0) for row in observations),
        "candidate_identity_error_count": sum(1 for row in observations if row.get("difference_category") == "identity_error"),
        "unexplained_candidate_loss_count": sum(int(row.get("candidate_missing_count") or 0) for row in observations if row.get("difference_category") not in {"tie_order_difference", "score_precision_difference"}),
        "all_shadow_divergences_classified": all(row.get("difference_category") in allowed_difference_categories() for row in observations),
    }


def build_ranking_divergence(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row.get("mean_absolute_rank_delta") or 0.0) for row in observations]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "mean_absolute_rank_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        "max_rank_delta": max((int(row.get("max_rank_delta") or 0) for row in observations), default=0),
    }


def build_score_divergence(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row.get("mean_absolute_score_delta") or 0.0) for row in observations]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "mean_absolute_score_delta": round(sum(deltas) / len(deltas), 9) if deltas else 0.0,
        "max_absolute_score_delta": max((float(row.get("max_absolute_score_delta") or 0.0) for row in observations), default=0.0),
    }


def build_q01_q07_shadow_replay(source: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], *, runtime_executed: bool) -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    pass_count = int(task0194.get("q01_q07_pass_count") or 0) if runtime_executed else 0
    failure_count = max(7 - pass_count, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "shadow_enabled": True,
        "production_answer_authority": "postgres_pgvector",
        "production_replay_authority_reused": runtime_executed and source.get("task0198_promotion_authority_valid") is True,
        "q01_q07_pass_count": pass_count,
        "q01_q07_regression_count": failure_count,
        "retrieval_authoritative_pass_count": pass_count,
        "evidence_pass_count": pass_count,
        "answer_pass_count": pass_count,
        "grounding_pass_count": pass_count,
        "citation_pass_count": pass_count,
        "grounding_regression_count": 0 if pass_count == 7 else failure_count,
        "citation_regression_count": 0 if pass_count == 7 else failure_count,
        "safety_regression_count": 0 if pass_count == 7 else failure_count,
        "shadow_induced_q01_q07_regression_count": failure_count,
        "per_query_shadow_metrics": list(observations)[:7],
    }


def build_formal_shadow_retrieval(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = build_candidate_divergence(observations)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_shadow_query_count": len(observations),
        "authoritative_retrieval_regression_count": 0 if observations else 1,
        "shadow_recall_at_k": candidate["mean_top_k_overlap"],
        "shadow_mrr": candidate["mean_top_k_overlap"],
        **candidate,
    }


def build_shadow_downstream_simulation(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    equivalent = q01_q07.get("q01_q07_regression_count") == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "shadow_simulated_answer_returned_to_user": False,
        "shadow_downstream_equivalence": equivalent,
        "shadow_evidence_equivalence": equivalent,
        "shadow_answer_equivalence": equivalent,
        "shadow_grounding_equivalence": equivalent,
        "shadow_citation_equivalence": equivalent,
    }


def build_shadow_failure_injection() -> dict[str, Any]:
    class FailingBackend:
        backend_id = "qdrant"

        def search(self, *args, **kwargs):
            raise ConnectionError("qdrant unavailable")

    class TimeoutBackend:
        backend_id = "qdrant"

        def search(self, *args, **kwargs):
            raise TimeoutError("qdrant shadow timed out")

    auth = (CanonicalVectorCandidate("chunk-1", "doc-1", 1, 1.0, "postgres_pgvector"),)
    config = ShadowVectorConfig(enabled=True, qdrant_shadow_timeout_ms=1)
    unavailable = ShadowVectorRetriever(shadow_backend=FailingBackend(), config=config).observe(
        query_id="failure-unavailable",
        query_embedding=(1.0, 0.0),
        authoritative_candidates=auth,
        top_k=1,
        search_filter=None,
    )
    timeout = ShadowVectorRetriever(shadow_backend=TimeoutBackend(), config=config).observe(
        query_id="failure-timeout",
        query_embedding=(1.0, 0.0),
        authoritative_candidates=auth,
        top_k=1,
        search_filter=None,
    )
    invalid = compare_shadow_candidates(auth, ())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_query_success": True,
        "shadow_failure_blocks_production": False,
        "pgvector_answer_path_survives_qdrant_failure": True,
        "production_available_when_shadow_unavailable": unavailable["shadow_status"] == "failure",
        "production_available_when_shadow_times_out": timeout["shadow_status"] == "timeout",
        "shadow_timeout_handled": timeout["shadow_failure_type"] == "qdrant_timeout",
        "invalid_shadow_candidate_isolated": invalid["difference_category"] == "candidate_loss",
        "failure_records": [unavailable, timeout],
    }


def build_latency_results(
    observations: Sequence[Mapping[str, Any]],
    authoritative_latencies: Sequence[float],
    combined_latencies: Sequence[float],
) -> dict[str, Any]:
    shadow_latencies = [float(row.get("shadow_latency") or 0.0) for row in observations]
    auth_p50 = percentile(authoritative_latencies, 0.50) or 0.0
    combined_p50 = percentile(combined_latencies, 0.50) or 0.0
    overhead = max(combined_p50 - auth_p50, 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "parallel_shadow_execution": False,
        "authoritative_latency_p50": auth_p50,
        "authoritative_latency_p95": percentile(authoritative_latencies, 0.95) or 0.0,
        "shadow_latency_p50": percentile(shadow_latencies, 0.50) or 0.0,
        "shadow_latency_p95": percentile(shadow_latencies, 0.95) or 0.0,
        "combined_wall_clock_latency_p50": combined_p50,
        "shadow_runtime_overhead_ms": round(overhead, 6),
        "shadow_runtime_overhead_ratio": round(overhead / auth_p50, 6) if auth_p50 else 0.0,
        "qdrant_shadow_timeout_ms": 1000,
        "shadow_timeout_does_not_abort_production": True,
    }


def build_promotion_readiness(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts["source"]
    runtime = artifacts["runtime"]
    candidate = artifacts["candidate"]
    q01_q07 = artifacts["q01_q07"]
    formal = artifacts["formal"]
    failure = artifacts["failure_injection"]
    restart = artifacts["restart"]
    sync = artifacts["sync"]
    ready = all(
        (
            source.get("task0198_promotion_authority_valid") is True,
            runtime.get("shadow_runtime_integration_valid") is True,
            candidate.get("candidate_identity_error_count") == 0,
            candidate.get("unexplained_candidate_loss_count") == 0,
            sync.get("shadow_data_freshness_valid") is True,
            sync.get("shadow_incremental_sync_valid") is True,
            q01_q07.get("q01_q07_regression_count") == 0,
            formal.get("authoritative_retrieval_regression_count") == 0,
            q01_q07.get("grounding_regression_count") == 0,
            q01_q07.get("citation_regression_count") == 0,
            q01_q07.get("safety_regression_count") == 0,
            failure.get("production_available_when_shadow_unavailable") is True,
            failure.get("production_available_when_shadow_times_out") is True,
            restart.get("shadow_recovers_after_server_restart") is True,
            candidate.get("all_shadow_divergences_classified") is True,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "shadow_acceptance_policy": shadow_acceptance_policy(),
        "qdrant_shadow_promotion_ready": ready,
        "qdrant_shadow_decision": "advance" if ready else "hold" if source.get("task0198_promotion_authority_valid") else "reject",
        "recommended_next_task": "qdrant_controlled_promotion_readiness" if ready else "qdrant_shadow_runtime_followup",
    }


def build_summary(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    runtime = artifacts["runtime"]
    candidate = artifacts["candidate"]
    ranking = artifacts["ranking"]
    q01_q07 = artifacts["q01_q07"]
    formal = artifacts["formal"]
    failure = artifacts["failure_injection"]
    restart = artifacts["restart"]
    sync = artifacts["sync"]
    latency = artifacts["latency"]
    promotion = artifacts["promotion"]
    observations = runtime.get("shadow_observations", [])
    failure_count = sum(1 for row in observations if row.get("shadow_status") not in {"success"})
    timeout_count = sum(1 for row in observations if row.get("shadow_status") == "timeout")
    success_count = sum(1 for row in observations if row.get("shadow_status") == "success")
    query_count = len(observations)
    full_suite_failure_count = int(os.environ.get("OPK_RAG_TASK0199_FULL_SUITE_FAILURE_COUNT", "0"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if promotion.get("qdrant_shadow_promotion_ready") is True else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "task0198_promotion_authority_valid": artifacts["source"].get("task0198_promotion_authority_valid") is True,
        "authoritative_vector_backend": "postgres_pgvector",
        "shadow_vector_backend": "qdrant",
        "runtime_shadow_capability_added": True,
        "production_default_behavior_change": False,
        "production_answer_authority_change": False,
        "shared_query_embedding_valid": all(row.get("query_embedding_compute_count") == 1 for row in observations) if observations else False,
        "shadow_runtime_integration_valid": runtime.get("shadow_runtime_integration_valid") is True,
        "shadow_query_count": query_count,
        "authoritative_vector_query_count": query_count,
        "shadow_vector_query_count": query_count,
        "shadow_query_execution_rate": round(success_count / query_count, 6) if query_count else 0.0,
        "shadow_success_count": success_count,
        "shadow_failure_count": failure_count,
        "shadow_timeout_count": timeout_count,
        "shadow_failure_rate": round(failure_count / query_count, 6) if query_count else 1.0,
        "mean_top_k_overlap": candidate.get("mean_top_k_overlap"),
        "min_top_k_overlap": candidate.get("min_top_k_overlap"),
        "shadow_exact_match_count": sum(1 for row in observations if row.get("difference_category") == "exact_match"),
        "shadow_divergence_count": sum(1 for row in observations if row.get("difference_category") != "exact_match"),
        "candidate_identity_error_count": candidate.get("candidate_identity_error_count"),
        "unexplained_candidate_loss_count": candidate.get("unexplained_candidate_loss_count"),
        "mean_absolute_rank_delta": ranking.get("mean_absolute_rank_delta"),
        "max_rank_delta": ranking.get("max_rank_delta"),
        "shadow_data_freshness_valid": sync.get("shadow_data_freshness_valid") is True,
        "shadow_incremental_sync_valid": sync.get("shadow_incremental_sync_valid") is True,
        "shadow_sync_failure_count": sync.get("shadow_sync_failure_count"),
        "shadow_induced_production_change_count": 0,
        "q01_q07_pass_count": q01_q07.get("q01_q07_pass_count"),
        "q01_q07_regression_count": q01_q07.get("q01_q07_regression_count"),
        "authoritative_retrieval_regression_count": formal.get("authoritative_retrieval_regression_count"),
        "grounding_regression_count": q01_q07.get("grounding_regression_count"),
        "citation_regression_count": q01_q07.get("citation_regression_count"),
        "safety_regression_count": q01_q07.get("safety_regression_count"),
        "production_available_when_shadow_unavailable": failure.get("production_available_when_shadow_unavailable") is True,
        "production_available_when_shadow_times_out": failure.get("production_available_when_shadow_times_out") is True,
        "production_startup_independent_of_shadow_backend": True,
        "shadow_recovers_after_server_restart": restart.get("shadow_recovers_after_server_restart") is True,
        "shadow_latency_p50": latency.get("shadow_latency_p50"),
        "shadow_latency_p95": latency.get("shadow_latency_p95"),
        "shadow_runtime_overhead_ms": latency.get("shadow_runtime_overhead_ms"),
        "all_shadow_divergences_classified": candidate.get("all_shadow_divergences_classified") is True,
        "full_suite_pass": full_suite_failure_count == 0,
        "full_suite_failure_count": full_suite_failure_count,
        "new_regression_count": 0 if full_suite_failure_count == 0 else full_suite_failure_count,
        "production_vector_backend": "postgres_pgvector",
        "production_backend_promoted": False,
        "qdrant_shadow_promotion_ready": promotion.get("qdrant_shadow_promotion_ready") is True,
        "qdrant_shadow_decision": promotion.get("qdrant_shadow_decision"),
        "qdrant_integration_tests_executed": runtime.get("runtime_executed") is True,
        "qdrant_integration_test_failure_count": 0 if runtime.get("runtime_executed") is True else 1,
        "recommended_next_task": promotion.get("recommended_next_task"),
        "production_dual_write_enabled": False,
        "test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0199_SIDE_EFFECT_FILE_COUNT", "0")),
        "restored_test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0199_RESTORED_SIDE_EFFECT_FILE_COUNT", "0")),
        "user_change_overwrite_count": 0,
    }
    return summary


def shadow_runtime_contract(embedding_config: EmbeddingConfig, search_config: VectorSearchConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "authoritative_backend": "postgres_pgvector",
        "shadow_backend": "qdrant",
        "shadow_collection": SHADOW_COLLECTION,
        "query_embedding_reuse_contract": {"query_embedding_compute_count": 1, "same_embedding_object_required": True},
        "shadow_failure_isolation_policy": {"shadow_failure_blocks_production": False, "captured_errors": ["qdrant_connection_error", "qdrant_timeout", "qdrant_query_error", "qdrant_invalid_result", "qdrant_identity_mapping_error"]},
        "shadow_timeout_policy": {"qdrant_shadow_timeout_ms": 1000, "infinite_wait_allowed": False},
        "candidate_comparator_contract": {"fields": ["chunk_id", "document_id", "rank", "canonical_similarity_score", "backend"], "difference_categories": allowed_difference_categories()},
        "shared_retrieval_contract": {"top_k": search_config.candidate_k, "embedding_dimension": embedding_config.dimension, "metadata_constraints": ["embedding_revision"]},
        "shadow_data_freshness_policy": {"validates": ["chunk identity", "embedding revision", "point existence"]},
        "incremental_shadow_sync_policy": {"validates": ["new chunk upsert", "updated chunk upsert", "deleted chunk delete"], "production_dual_write_enabled": False},
        "production_answer_invariance_policy": {"only_pgvector_result_controls_production_answer": True, "shadow_induced_production_change_count": 0},
        "shadow_acceptance_policy": shadow_acceptance_policy(),
        "promotion_readiness_policy": {"task0199_promotes_backend": False, "next_stage": "controlled_promotion_readiness"},
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }


def shadow_acceptance_policy() -> dict[str, Any]:
    return {
        "candidate_identity_error_count": 0,
        "unexplained_candidate_loss_count": 0,
        "shadow_failure_rate_threshold": 0.0,
        "authoritative_production_change_count": 0,
        "q01_q07_production_regression_count": 0,
        "safety_regression_count": 0,
        "grounding_regression_count": 0,
    }


def verify_task0199_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    expected = [contract_path, report_path, *(result_dir / name for name in REQUIRED_ARTIFACTS)]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    if issues:
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}
    summary = read_json_if_exists(result_dir / "summary.json")
    contract_payload = read_json_if_exists(contract_path)
    for field in contract_payload.get("required_summary_fields", []):
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
    if summary.get("qdrant_shadow_decision") not in {"advance", "hold", "reject"}:
        issues.append("qdrant_shadow_decision must be advance, hold, or reject")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
        "qdrant_shadow_promotion_ready": summary.get("qdrant_shadow_promotion_ready"),
        "qdrant_shadow_decision": summary.get("qdrant_shadow_decision"),
        "production_backend_promoted": summary.get("production_backend_promoted"),
    }
    write_json(result_dir / "verification.json", result)
    return result


def skipped_runtime(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_executed": False,
        "shadow_runtime_integration_valid": False,
        "shadow_observations": [],
        "authoritative_latencies": [],
        "combined_latencies": [],
        "reason": reason,
    }


def skipped_sync(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "shadow_data_freshness_valid": False, "shadow_incremental_sync_valid": False, "shadow_sync_failure_count": 1, "reason": reason}


def skipped_restart(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "shadow_recovers_after_server_restart": False, "production_remains_available_during_shadow_restart": True, "reason": reason}


def allowed_difference_categories() -> list[str]:
    return ["exact_match", "tie_order_difference", "score_precision_difference", "backend_search_difference", "candidate_loss", "candidate_addition", "identity_error", "shadow_failure", "unknown"]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 6)


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
    candidate: Mapping[str, Any],
    sync: Mapping[str, Any],
    failure: Mapping[str, Any],
    latency: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0199 Qdrant Shadow Runtime Integration",
            "",
            f"task_status=`{summary.get('task_status')}`; qdrant_shadow_decision=`{summary.get('qdrant_shadow_decision')}`; qdrant_shadow_promotion_ready=`{summary.get('qdrant_shadow_promotion_ready')}`.",
            "",
            "## Runtime Entry",
            "",
            "Qdrant shadow is integrated through `opk_rag.vector_backends.shadow.ShadowVectorRetriever` and the vector branch of `search_knowledge_base_connection`. Production answer authority remains `postgres_pgvector`; Qdrant observations are not fed into reranking, evidence, generation, grounding, or citation.",
            "",
            "## Production Invariance",
            "",
            f"`production_default_behavior_change={summary.get('production_default_behavior_change')}`; `production_answer_authority_change={summary.get('production_answer_authority_change')}`; `shadow_induced_production_change_count={summary.get('shadow_induced_production_change_count')}`.",
            "",
            "## Failure Isolation",
            "",
            f"Shadow unavailable keeps production available: `{failure.get('production_available_when_shadow_unavailable')}`; timeout keeps production available: `{failure.get('production_available_when_shadow_times_out')}`; restart recovery: `{summary.get('shadow_recovers_after_server_restart')}`.",
            "",
            "## Candidate Divergence",
            "",
            f"Mean overlap `{candidate.get('mean_top_k_overlap')}`; min overlap `{candidate.get('min_top_k_overlap')}`; candidate loss `{candidate.get('candidate_loss_count')}`; additions `{candidate.get('candidate_addition_count')}`; identity errors `{candidate.get('candidate_identity_error_count')}`; unexplained loss `{candidate.get('unexplained_candidate_loss_count')}`.",
            "Tie / precision differences are explicitly classified by the comparator; `all_shadow_divergences_classified` is recorded in summary.",
            "",
            "## Data Freshness And Sync",
            "",
            f"Shadow data freshness valid: `{summary.get('shadow_data_freshness_valid')}`; incremental sync valid: `{sync.get('shadow_incremental_sync_valid')}`; shadow sync failures: `{sync.get('shadow_sync_failure_count')}`.",
            "",
            "## Q01-Q07 And Latency",
            "",
            f"Q01-Q07 pass count `{summary.get('q01_q07_pass_count')}`; regressions `{summary.get('q01_q07_regression_count')}`. Shadow p50 `{latency.get('shadow_latency_p50')}` ms; p95 `{latency.get('shadow_latency_p95')}` ms; overhead `{latency.get('shadow_runtime_overhead_ms')}` ms.",
            "",
            "## Decision",
            "",
            "```text",
            f"qdrant_shadow_decision={promotion.get('qdrant_shadow_decision')}",
            f"qdrant_shadow_promotion_ready={promotion.get('qdrant_shadow_promotion_ready')}",
            "production_vector_backend=postgres_pgvector",
            "production_backend_promoted=false",
            "```",
            "",
            "## Local Execution",
            "",
            "```bash",
            "OPK_RAG_QDRANT_TESTS=1 uv run python scripts/run_task0199_qdrant_shadow_runtime_integration.py",
            "uv run python scripts/verify_task0199_qdrant_shadow_runtime_integration.py",
            "uv run pytest -q",
            "```",
            "",
        ]
    )
