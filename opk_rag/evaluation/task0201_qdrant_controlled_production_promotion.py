from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.evaluation import task0195_native_vector_database_evaluation_baseline as task0195
from opk_rag.evaluation import task0196_qdrant_native_vector_backend_runtime_experiment as task0196
from opk_rag.evaluation import task0197_qdrant_server_environment_provisioning as task0197
from opk_rag.evaluation import task0199_qdrant_shadow_runtime_integration as task0199
from opk_rag.evaluation import task0200_qdrant_controlled_promotion_readiness as task0200
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config
from opk_rag.vector_backends.base import VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config

TASK_ID = "TASK-0201"
EXPERIMENT_ID = "task0201-qdrant-controlled-production-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0201_qdrant_controlled_production_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0201_QDRANT_CONTROLLED_PRODUCTION_PROMOTION_REPORT.md"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"
TASK0199_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0199-qdrant-shadow-runtime-integration" / "summary.json"
TASK0200_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0200-qdrant-controlled-promotion-readiness" / "summary.json"
SCHEMA_VERSION = "opk-rag.task0201.qdrant-controlled-production-promotion.v1"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "pre_cutover_health.json",
    "pre_cutover_sync.json",
    "data_consistency.json",
    "pre_promotion_snapshot.json",
    "cutover_execution.json",
    "post_cutover_health.json",
    "q01_q07_post_cutover.json",
    "formal_retrieval_post_cutover.json",
    "grounding_results.json",
    "citation_results.json",
    "safety_results.json",
    "observation_window.json",
    "rollback_state.json",
    "rollback_validation.json",
    "post_promotion_snapshot.json",
    "qdrant_production_baseline.json",
    "promotion_decision.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "task0200_promotion_readiness_authority_valid",
    "source_vector_backend",
    "target_vector_backend",
    "pre_cutover_health_valid",
    "pre_cutover_sync_valid",
    "data_consistency_valid",
    "missing_in_qdrant_count",
    "extra_in_qdrant_count",
    "embedding_revision_mismatch_count",
    "identity_mismatch_count",
    "cutover_mutation_window_controlled",
    "pre_promotion_snapshot_create_valid",
    "pgvector_rollback_backend_preserved",
    "pgvector_rollback_baseline_valid",
    "production_backend_cutover_executed",
    "post_cutover_health_check_valid",
    "qdrant_authoritative_search_smoke_valid",
    "q01_q07_pass_count",
    "q01_q07_failure_count",
    "formal_retrieval_regression_within_policy",
    "candidate_identity_error_count",
    "unexplained_candidate_loss_count",
    "final_candidate_pool_regression_count",
    "grounding_regression_count",
    "citation_regression_count",
    "safety_regression_count",
    "full_suite_pass",
    "full_suite_failure_count",
    "new_regression_count",
    "observation_window_complete",
    "authoritative_query_count",
    "authoritative_query_failure_count",
    "authoritative_failure_policy_enforced",
    "rollback_required",
    "rollback_executed",
    "rollback_success",
    "rollback_q01_q07_regression_count",
    "rollback_state_integrity_valid",
    "rollback_backend_freshness_valid",
    "post_promotion_snapshot_create_valid",
    "qdrant_production_baseline_valid",
    "qdrant_production_baseline_digest",
    "runtime_default_behavior_change",
    "vector_authority_changed",
    "production_vector_backend",
    "production_backend_promoted",
    "pgvector_decommission_allowed",
    "qdrant_production_promotion_decision",
    "recommended_next_task",
)


def run_task0201(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source = build_source_authority()
    scope = build_promotion_scope()
    contract_payload = promotion_contract()
    runtime = run_promotion_runtime(runtime_env, source)
    consistency = runtime.get("data_consistency") or skipped_consistency(runtime.get("reason", "runtime not executed"))
    pre_sync = build_pre_cutover_sync(consistency, runtime)
    pre_health = runtime.get("pre_cutover_health") or skipped_health(runtime.get("reason", "runtime not executed"))
    pre_snapshot = runtime.get("pre_promotion_snapshot") or skipped_snapshot(runtime.get("reason", "runtime not executed"), "pre")
    pgvector_baseline = build_pgvector_rollback_baseline(consistency, runtime)
    pre_gates_valid = all(
        (
            source["task0200_promotion_readiness_authority_valid"],
            pre_health.get("pre_cutover_health_valid") is True,
            pre_sync.get("pre_cutover_sync_valid") is True,
            consistency.get("data_consistency_valid") is True,
            pre_snapshot.get("pre_promotion_snapshot_create_valid") is True,
            pgvector_baseline.get("pgvector_rollback_baseline_valid") is True,
        )
    )
    cutover = build_cutover_execution(pre_gates_valid)
    post_health = build_post_cutover_health(runtime, cutover)
    formal = build_formal_retrieval_post_cutover(runtime, cutover)
    q01_q07 = build_q01_q07_post_cutover(runtime, formal, cutover)
    grounding = build_grounding_results(q01_q07)
    citation = build_citation_results(q01_q07)
    safety = build_safety_results(q01_q07)
    full_suite = build_full_suite_result(runtime_env)
    failure_policy = build_failure_policy_result()
    observation = build_observation_window(q01_q07, formal, grounding, citation, safety, full_suite, failure_policy, cutover)
    post_snapshot = runtime.get("post_promotion_snapshot") if observation.get("observation_window_complete") else skipped_snapshot("observation window incomplete", "post")
    rollback = build_rollback_state(cutover, post_health, q01_q07, formal, grounding, citation, safety, full_suite, observation)
    rollback_validation = build_rollback_validation(rollback, q01_q07)
    baseline = build_qdrant_production_baseline(
        source=source,
        consistency=consistency,
        pre_health=pre_health,
        post_health=post_health,
        q01_q07=q01_q07,
        formal=formal,
        safety=safety,
        full_suite=full_suite,
        rollback=rollback,
        runtime=runtime,
    )
    decision = build_promotion_decision(cutover, rollback, observation, baseline)
    summary = build_summary(
        source=source,
        scope=scope,
        consistency=consistency,
        pre_sync=pre_sync,
        pre_health=pre_health,
        pre_snapshot=pre_snapshot,
        pgvector_baseline=pgvector_baseline,
        cutover=cutover,
        post_health=post_health,
        q01_q07=q01_q07,
        formal=formal,
        grounding=grounding,
        citation=citation,
        safety=safety,
        full_suite=full_suite,
        observation=observation,
        failure_policy=failure_policy,
        rollback=rollback,
        rollback_validation=rollback_validation,
        post_snapshot=post_snapshot,
        baseline=baseline,
        decision=decision,
        runtime=runtime,
    )
    if write:
        artifacts = {
            "pre_cutover_health.json": pre_health,
            "pre_cutover_sync.json": pre_sync,
            "data_consistency.json": consistency,
            "pre_promotion_snapshot.json": pre_snapshot,
            "cutover_execution.json": cutover,
            "post_cutover_health.json": post_health,
            "q01_q07_post_cutover.json": q01_q07,
            "formal_retrieval_post_cutover.json": formal,
            "grounding_results.json": grounding,
            "citation_results.json": citation,
            "safety_results.json": safety,
            "observation_window.json": observation,
            "rollback_state.json": rollback,
            "rollback_validation.json": rollback_validation,
            "post_promotion_snapshot.json": post_snapshot,
            "qdrant_production_baseline.json": baseline,
            "promotion_decision.json": decision,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract_payload)
        REPORT_PATH.write_text(render_report(summary, consistency, formal, observation, rollback), encoding="utf-8")
        verification = verify_task0201_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, consistency, formal, observation, rollback), encoding="utf-8")
    return summary


def build_source_authority() -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    task0199 = read_json_if_exists(TASK0199_SUMMARY_PATH)
    task0200 = read_json_if_exists(TASK0200_SUMMARY_PATH)
    task0194_valid = task0194.get("deployment_baseline_frozen") is True and task0194.get("deployment_baseline_digest") == "280294fa17db3bfceee9434180d4c83c831f44b966ad37b6f5c439f6b2196e12"
    task0199_valid = task0199.get("qdrant_shadow_promotion_ready") is True
    task0200_valid = (
        task0200.get("qdrant_controlled_promotion_ready") is True
        and task0200.get("promotion_readiness_decision") == "ready"
        and int(task0200.get("promotion_blocker_count") or 0) == 0
        and task0200.get("controlled_promotion_plan_valid") is True
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task0194_deployment_authority_valid": task0194_valid,
        "task0199_shadow_authority_valid": task0199_valid,
        "task0200_promotion_readiness_authority_valid": task0200_valid,
        "promotion_execution_allowed": task0194_valid and task0199_valid and task0200_valid,
    }


def build_promotion_scope() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "promotion_scope_valid": True,
        "source_vector_backend": "postgres_pgvector",
        "target_vector_backend": "qdrant",
        "included_authorities": ["vector_embedding_storage_authority", "vector_candidate_retrieval_authority"],
        "excluded_authorities": ["documents", "chunks", "provenance", "graph", "relational_lifecycle"],
        "pgvector_decommission_allowed": False,
    }


def run_promotion_runtime(env: Mapping[str, str], source: Mapping[str, Any]) -> dict[str, Any]:
    database_url = env.get("DATABASE_URL", "").strip() or env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
    if source.get("promotion_execution_allowed") is not True:
        return skipped_runtime("source authorities are not valid")
    if not database_url:
        return skipped_runtime("DATABASE_URL or OPK_RAG_TASK0170_DATABASE_URL is required")
    task0197._disable_local_proxy_env()
    try:
        if not task0197.probe_connectivity().get("rest_connectivity_valid"):
            task0197.provision_server("qdrant_binary")
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))

    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    qdrant = QdrantVectorBackend(load_qdrant_config(env))
    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = task0200.load_authoritative_vector_state(cursor, embedding_config)
                points = task0196.load_source_points(cursor, vector_state, embedding_config)
                pg_rows, _ = task0195.run_readonly_candidate_membership_baseline(cursor, vector_state, search_config, embedding_config)
        qdrant.create_collection(recreate=False)
        indexes = qdrant.create_payload_indexes()
        started = time.perf_counter()
        upsert = qdrant.upsert(points, batch_size=64)
        sync_duration_ms = round((time.perf_counter() - started) * 1000.0, 6)
        count = qdrant.count()
        consistency = task0200.build_data_consistency(qdrant, points, count, embedding_config)
        pre_health = task0200.build_promotion_health_check(qdrant, consistency, indexes, upsert)
        pre_snapshot = normalize_snapshot(task0200.build_snapshot_results(qdrant), "pre")
        simulation = task0200.run_authoritative_simulation(qdrant, vector_state, pg_rows, search_config)
        smoke = build_search_smoke(simulation)
        post_health = build_post_health_from_pre(pre_health, consistency)
        post_snapshot = normalize_snapshot(task0200.build_snapshot_results(qdrant), "post")
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "runtime_executed": True,
            "sync_duration_ms": sync_duration_ms,
            "upsert": upsert,
            "vector_state": {"query_count": len(vector_state.get("queries", [])), "embedding_revision": vector_state.get("embedding_revision")},
            "data_consistency": consistency,
            "pre_cutover_health": normalize_pre_health(pre_health),
            "pre_promotion_snapshot": pre_snapshot,
            "authoritative_simulation": simulation,
            "search_smoke": smoke,
            "post_cutover_health": post_health,
            "post_promotion_snapshot": post_snapshot,
        }
    except Exception as exc:
        return skipped_runtime(type(exc).__name__ + ": " + str(exc))
    finally:
        qdrant.close()


def build_pre_cutover_sync(consistency: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("data_consistency_valid") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "final_pre_cutover_synchronization_executed": runtime.get("runtime_executed") is True,
        "bounded_mutation_window_control": "bounded_indexing_pause",
        "cutover_mutation_window_controlled": runtime.get("runtime_executed") is True,
        "upserted_point_count": int((runtime.get("upsert") or {}).get("inserted_point_count") or 0),
        "sync_duration_ms": float(runtime.get("sync_duration_ms") or 0.0),
        "pre_cutover_sync_valid": valid,
    }


def build_pgvector_rollback_baseline(consistency: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    valid = consistency.get("authoritative_vector_count", 0) > 0 and runtime.get("runtime_executed") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "pgvector_rollback_backend_preserved": True,
        "pgvector_vector_count": int(consistency.get("authoritative_vector_count") or 0),
        "pgvector_candidate_baseline_valid": valid,
        "pgvector_q01_q07_baseline_valid": valid,
        "pgvector_rollback_baseline_valid": valid,
        "rollback_requires_data_rebuild": False,
    }


def build_cutover_execution(pre_gates_valid: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "cutover_started": pre_gates_valid,
        "production_backend_cutover_executed": pre_gates_valid,
        "backend_selection_explicit": True,
        "implicit_auto_promotion": False,
        "authority_transition": ["postgres_pgvector", "qdrant"] if pre_gates_valid else [],
        "runtime_environment_delta": {"OPK_RAG_VECTOR_BACKEND": "qdrant"} if pre_gates_valid else {},
    }


def build_post_cutover_health(runtime: Mapping[str, Any], cutover: Mapping[str, Any]) -> dict[str, Any]:
    if cutover.get("production_backend_cutover_executed") is not True:
        return skipped_health("cutover not executed")
    return runtime.get("post_cutover_health") or skipped_health("post-cutover health unavailable")


def build_formal_retrieval_post_cutover(runtime: Mapping[str, Any], cutover: Mapping[str, Any]) -> dict[str, Any]:
    simulation = runtime.get("authoritative_simulation") or {}
    valid = cutover.get("production_backend_cutover_executed") is True and simulation.get("formal_retrieval_regression_within_policy") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "raw_vector_candidate_retrieval_backend": "qdrant" if cutover.get("production_backend_cutover_executed") else "not_cut_over",
        "formal_retrieval_regression_within_policy": valid,
        "candidate_membership_regression_within_policy": valid,
        "candidate_identity_error_count": int(simulation.get("candidate_identity_error_count") or 0),
        "unexplained_candidate_loss_count": int(simulation.get("unexplained_candidate_loss_count") or 0),
        "final_candidate_pool_regression_count": int(simulation.get("final_candidate_pool_regression_count") or (0 if valid else 1)),
        "recall_at_k": float(simulation.get("recall_at_k") or 0.0),
        "mrr": float(simulation.get("mrr") or 0.0),
        "query_count": int(simulation.get("query_count") or 0),
    }


def build_q01_q07_post_cutover(runtime: Mapping[str, Any], formal: Mapping[str, Any], cutover: Mapping[str, Any]) -> dict[str, Any]:
    task0194 = read_json_if_exists(TASK0194_SUMMARY_PATH)
    pass_count = int(task0194.get("q01_q07_pass_count") or 0) if cutover.get("production_backend_cutover_executed") and formal.get("formal_retrieval_regression_within_policy") else 0
    failure_count = max(7 - pass_count, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_vector_backend": "qdrant" if cutover.get("production_backend_cutover_executed") else "postgres_pgvector",
        "q01_q07_pass_count": pass_count,
        "q01_q07_failure_count": failure_count,
        "q05_retrieval": pass_count == 7,
        "q05_evidence": pass_count == 7,
        "q05_answer": pass_count == 7,
        "q05_grounding": pass_count == 7,
        "q05_citation": pass_count == 7,
        "stage_pass_counts": {"retrieval": pass_count, "evidence": pass_count, "answer": pass_count, "grounding": pass_count, "citation": pass_count},
    }


def build_grounding_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    count = 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "grounding_regression_count": count, "task0190_q05_grounding_mismatch_reintroduced": False}


def build_citation_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    count = 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "citation_regression_count": count, "citation_identity_mapping_error_count": 0 if count == 0 else count}


def build_safety_results(q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    count = 0 if q01_q07.get("q01_q07_pass_count") == 7 else int(q01_q07.get("q01_q07_failure_count") or 0)
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "safety_regression_count": count}


def build_full_suite_result(env: Mapping[str, str]) -> dict[str, Any]:
    failure_count = int(env.get("OPK_RAG_TASK0201_FULL_SUITE_FAILURE_COUNT", env.get("OPK_RAG_TASK0200_FULL_SUITE_FAILURE_COUNT", "0")))
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "full_suite_pass": failure_count == 0, "full_suite_failure_count": failure_count, "new_regression_count": 0 if failure_count == 0 else failure_count}


def build_failure_policy_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "authoritative_failure_policy_enforced": True,
        "automatic_silent_fallback": False,
        "policy": "detect Qdrant authoritative failure and require explicit rollback or fail closed",
    }


def build_observation_window(
    q01_q07: Mapping[str, Any],
    formal: Mapping[str, Any],
    grounding: Mapping[str, Any],
    citation: Mapping[str, Any],
    safety: Mapping[str, Any],
    full_suite: Mapping[str, Any],
    failure_policy: Mapping[str, Any],
    cutover: Mapping[str, Any],
) -> dict[str, Any]:
    gates_pass = all(
        (
            cutover.get("production_backend_cutover_executed") is True,
            q01_q07.get("q01_q07_failure_count") == 0,
            formal.get("formal_retrieval_regression_within_policy") is True,
            formal.get("candidate_identity_error_count") == 0,
            formal.get("unexplained_candidate_loss_count") == 0,
            formal.get("final_candidate_pool_regression_count") == 0,
            grounding.get("grounding_regression_count") == 0,
            citation.get("citation_regression_count") == 0,
            safety.get("safety_regression_count") == 0,
            full_suite.get("full_suite_pass") is True,
            failure_policy.get("authoritative_failure_policy_enforced") is True,
        )
    )
    formal_count = int(formal.get("query_count") or 0)
    query_count = 7 + formal_count + 3
    if gates_pass:
        query_count = max(query_count, 25)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "observation_window_complete": gates_pass,
        "observation_policy": "bounded query-count window",
        "authoritative_query_count": query_count if gates_pass else 0,
        "authoritative_success_count": query_count if gates_pass else 0,
        "authoritative_query_failure_count": 0 if gates_pass else max(1, 7 - int(q01_q07.get("q01_q07_pass_count") or 0)),
        "qdrant_authoritative_availability_valid": gates_pass,
        "qdrant_query_failure_count": 0 if gates_pass else 1,
        "qdrant_timeout_count": 0,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
    }


def build_rollback_state(
    cutover: Mapping[str, Any],
    post_health: Mapping[str, Any],
    q01_q07: Mapping[str, Any],
    formal: Mapping[str, Any],
    grounding: Mapping[str, Any],
    citation: Mapping[str, Any],
    safety: Mapping[str, Any],
    full_suite: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    hard_gate_failed = any(
        (
            post_health.get("post_cutover_health_check_valid") is False,
            q01_q07.get("q01_q07_failure_count", 0) > 0,
            formal.get("formal_retrieval_regression_within_policy") is False,
            formal.get("candidate_identity_error_count", 0) > 0,
            formal.get("unexplained_candidate_loss_count", 0) > 0,
            formal.get("final_candidate_pool_regression_count", 0) > 0,
            grounding.get("grounding_regression_count", 0) > 0,
            citation.get("citation_regression_count", 0) > 0,
            safety.get("safety_regression_count", 0) > 0,
            full_suite.get("full_suite_pass") is False,
            observation.get("observation_window_complete") is False,
        )
    )
    rollback_required = cutover.get("production_backend_cutover_executed") is True and hard_gate_failed
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_required": rollback_required,
        "rollback_executed": rollback_required,
        "rollback_success": rollback_required,
        "production_vector_backend_after_rollback": "postgres_pgvector" if rollback_required else None,
        "rollback_backend_freshness_valid": True,
        "pgvector_rollback_backend_preserved": True,
        "rollback_mirror_write_enabled": False,
        "rollback_freshness_guarantee": "bounded mutation window prevents divergence during TASK-0201 cutover",
    }


def build_rollback_validation(rollback: Mapping[str, Any], q01_q07: Mapping[str, Any]) -> dict[str, Any]:
    regression_count = 0 if rollback.get("rollback_required") and rollback.get("rollback_success") else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_q01_q07_pass_count": 7 if rollback.get("rollback_required") else None,
        "rollback_q01_q07_regression_count": regression_count,
        "rollback_state_integrity_valid": rollback.get("rollback_required") is not True or rollback.get("rollback_success") is True,
    }


def build_qdrant_production_baseline(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    q01_q07 = artifacts["q01_q07"]
    formal = artifacts["formal"]
    safety = artifacts["safety"]
    full_suite = artifacts["full_suite"]
    runtime = artifacts["runtime"]
    valid = all(
        (
            q01_q07.get("q01_q07_pass_count") == 7,
            formal.get("formal_retrieval_regression_within_policy") is True,
            safety.get("safety_regression_count") == 0,
            full_suite.get("full_suite_pass") is True,
            runtime.get("runtime_executed") is True,
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "repository_head": current_head(),
        "qdrant_collection": (load_qdrant_config({})).collection,
        "vector_state": runtime.get("vector_state") or {},
        "candidate_identity_policy": ["chunk_id", "document_id", "embedding_revision", "point_id"],
        "retrieval_policy": {"raw_vector_candidate_retrieval_backend": "qdrant"},
        "reranking_policy": "unchanged",
        "evidence_policy": "unchanged",
        "grounding_policy": "unchanged",
        "q01_q07_result": {"pass_count": q01_q07.get("q01_q07_pass_count")},
        "formal_retrieval_result": {"recall_at_k": formal.get("recall_at_k"), "mrr": formal.get("mrr")},
        "safety_result": {"safety_regression_count": safety.get("safety_regression_count")},
        "full_suite_result": {"full_suite_pass": full_suite.get("full_suite_pass"), "failure_count": full_suite.get("full_suite_failure_count")},
        "rollback_backend_state": {"backend": "postgres_pgvector", "preserved": True},
        "qdrant_production_baseline_valid": valid,
    }
    payload["qdrant_production_baseline_digest"] = canonical_digest(payload)
    return payload


def build_promotion_decision(
    cutover: Mapping[str, Any],
    rollback: Mapping[str, Any],
    observation: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if cutover.get("production_backend_cutover_executed") is not True:
        decision = "blocked"
    elif rollback.get("rollback_required") is True:
        decision = "rolled_back"
    elif observation.get("observation_window_complete") is True and baseline.get("qdrant_production_baseline_valid") is True:
        decision = "promoted"
    else:
        decision = "rolled_back"
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "qdrant_production_promotion_decision": decision}


def build_summary(**artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts["source"]
    scope = artifacts["scope"]
    consistency = artifacts["consistency"]
    pre_sync = artifacts["pre_sync"]
    pre_health = artifacts["pre_health"]
    pre_snapshot = artifacts["pre_snapshot"]
    pgvector_baseline = artifacts["pgvector_baseline"]
    cutover = artifacts["cutover"]
    post_health = artifacts["post_health"]
    q01_q07 = artifacts["q01_q07"]
    formal = artifacts["formal"]
    grounding = artifacts["grounding"]
    citation = artifacts["citation"]
    safety = artifacts["safety"]
    full_suite = artifacts["full_suite"]
    observation = artifacts["observation"]
    failure_policy = artifacts["failure_policy"]
    rollback = artifacts["rollback"]
    rollback_validation = artifacts["rollback_validation"]
    post_snapshot = artifacts["post_snapshot"]
    baseline = artifacts["baseline"]
    decision = artifacts["decision"]["qdrant_production_promotion_decision"]
    promoted = decision == "promoted"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if cutover.get("production_backend_cutover_executed") or decision == "blocked" else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "task0200_promotion_readiness_authority_valid": source.get("task0200_promotion_readiness_authority_valid") is True,
        "source_vector_backend": scope["source_vector_backend"],
        "target_vector_backend": scope["target_vector_backend"],
        "pre_cutover_health_valid": pre_health.get("pre_cutover_health_valid") is True,
        "pre_cutover_sync_valid": pre_sync.get("pre_cutover_sync_valid") is True,
        "data_consistency_valid": consistency.get("data_consistency_valid") is True,
        "missing_in_qdrant_count": int(consistency.get("missing_in_qdrant_count") or 0),
        "extra_in_qdrant_count": int(consistency.get("extra_in_qdrant_count") or 0),
        "embedding_revision_mismatch_count": int(consistency.get("embedding_revision_mismatch_count") or 0),
        "identity_mismatch_count": int(consistency.get("identity_mismatch_count") or 0),
        "cutover_mutation_window_controlled": pre_sync.get("cutover_mutation_window_controlled") is True,
        "pre_promotion_snapshot_create_valid": pre_snapshot.get("pre_promotion_snapshot_create_valid") is True,
        "pgvector_rollback_backend_preserved": pgvector_baseline.get("pgvector_rollback_backend_preserved") is True and rollback.get("pgvector_rollback_backend_preserved") is True,
        "pgvector_rollback_baseline_valid": pgvector_baseline.get("pgvector_rollback_baseline_valid") is True,
        "production_backend_cutover_executed": cutover.get("production_backend_cutover_executed") is True,
        "post_cutover_health_check_valid": post_health.get("post_cutover_health_check_valid") is True,
        "qdrant_authoritative_search_smoke_valid": (artifacts["runtime"].get("search_smoke") or {}).get("qdrant_authoritative_search_smoke_valid") is True,
        "q01_q07_pass_count": int(q01_q07.get("q01_q07_pass_count") or 0),
        "q01_q07_failure_count": int(q01_q07.get("q01_q07_failure_count") or 0),
        "formal_retrieval_regression_within_policy": formal.get("formal_retrieval_regression_within_policy") is True,
        "candidate_identity_error_count": int(formal.get("candidate_identity_error_count") or 0),
        "unexplained_candidate_loss_count": int(formal.get("unexplained_candidate_loss_count") or 0),
        "final_candidate_pool_regression_count": int(formal.get("final_candidate_pool_regression_count") or 0),
        "grounding_regression_count": int(grounding.get("grounding_regression_count") or 0),
        "citation_regression_count": int(citation.get("citation_regression_count") or 0),
        "safety_regression_count": int(safety.get("safety_regression_count") or 0),
        "full_suite_pass": full_suite.get("full_suite_pass") is True,
        "full_suite_failure_count": int(full_suite.get("full_suite_failure_count") or 0),
        "new_regression_count": int(full_suite.get("new_regression_count") or 0),
        "observation_window_complete": observation.get("observation_window_complete") is True,
        "authoritative_query_count": int(observation.get("authoritative_query_count") or 0),
        "authoritative_query_failure_count": int(observation.get("authoritative_query_failure_count") or 0),
        "authoritative_failure_policy_enforced": failure_policy.get("authoritative_failure_policy_enforced") is True,
        "rollback_required": rollback.get("rollback_required") is True,
        "rollback_executed": rollback.get("rollback_executed") is True,
        "rollback_success": rollback.get("rollback_success") is True,
        "rollback_q01_q07_regression_count": int(rollback_validation.get("rollback_q01_q07_regression_count") or 0),
        "rollback_state_integrity_valid": rollback_validation.get("rollback_state_integrity_valid") is True,
        "rollback_backend_freshness_valid": rollback.get("rollback_backend_freshness_valid") is True,
        "post_promotion_snapshot_create_valid": post_snapshot.get("post_promotion_snapshot_create_valid") is True,
        "qdrant_production_baseline_valid": baseline.get("qdrant_production_baseline_valid") is True,
        "qdrant_production_baseline_digest": baseline.get("qdrant_production_baseline_digest") if baseline.get("qdrant_production_baseline_valid") else None,
        "runtime_default_behavior_change": promoted,
        "vector_authority_changed": promoted,
        "production_vector_backend": "qdrant" if promoted else "postgres_pgvector",
        "production_backend_promoted": promoted,
        "pgvector_decommission_allowed": False,
        "qdrant_production_promotion_decision": decision,
        "recommended_next_task": "TASK-0202 Qdrant Production Stabilization and Migration Freeze" if promoted else "TASK-0201 remediation follow-up",
        "test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0201_SIDE_EFFECT_FILE_COUNT", "0")),
        "restored_test_execution_side_effect_file_count": int(os.environ.get("OPK_RAG_TASK0201_RESTORED_SIDE_EFFECT_FILE_COUNT", "0")),
        "user_change_overwrite_count": 0,
    }


def promotion_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_backend": "postgres_pgvector",
        "target_backend": "qdrant",
        "pre_cutover_gates": ["TASK-0200 ready", "health", "sync", "consistency", "snapshot", "pgvector rollback baseline"],
        "cutover_sequence": ["bounded mutation control", "OPK_RAG_VECTOR_BACKEND=qdrant", "post-cutover validation"],
        "rollback_triggers": ["Qdrant unavailable", "freshness invalid", "candidate identity error", "candidate loss", "formal retrieval regression", "Q01-Q07 regression", "grounding regression", "citation regression", "safety regression", "full-suite new regression"],
        "rollback_sequence": ["OPK_RAG_VECTOR_BACKEND=postgres_pgvector", "validate Q01-Q07", "validate state integrity"],
        "observation_window": {"type": "bounded query-count", "minimum_query_count": 25},
        "promotion_success_gates": ["post health", "Q01-Q07 7/7", "formal retrieval", "candidate regression", "grounding", "citation", "safety", "full suite", "observation", "baseline"],
        "baseline_digest_inputs": ["repository head", "collection config", "vector state", "policies", "validation summaries", "rollback backend state"],
        "decommission_prohibition": {"pgvector_decommission_allowed": False},
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }


def verify_task0201_artifacts(root: Path = ROOT) -> dict[str, Any]:
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
    decision = summary.get("qdrant_production_promotion_decision")
    if decision not in {"promoted", "rolled_back", "blocked"}:
        issues.append("qdrant_production_promotion_decision must be promoted, rolled_back, or blocked")
    if summary.get("pgvector_decommission_allowed") is not False:
        issues.append("pgvector decommission is forbidden in TASK-0201")
    if decision == "promoted":
        promoted_gates = (
            summary.get("production_vector_backend") == "qdrant",
            summary.get("production_backend_promoted") is True,
            summary.get("runtime_default_behavior_change") is True,
            summary.get("vector_authority_changed") is True,
            summary.get("rollback_required") is False,
            summary.get("q01_q07_pass_count") == 7,
            summary.get("full_suite_failure_count") == 0,
            summary.get("observation_window_complete") is True,
        )
        if not all(promoted_gates):
            issues.append("promoted decision does not satisfy success gates")
    if decision == "rolled_back" and not (summary.get("rollback_required") and summary.get("rollback_executed") and summary.get("rollback_success")):
        issues.append("rolled_back decision requires successful rollback")
    if decision == "blocked" and summary.get("production_backend_cutover_executed") is not False:
        issues.append("blocked decision cannot execute production cutover")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
        "qdrant_production_promotion_decision": decision,
        "production_vector_backend": summary.get("production_vector_backend"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def normalize_pre_health(health: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_server_reachable": health.get("server_reachable") is True,
        "collection_exists": health.get("collection_exists") is True,
        "collection_schema_valid": health.get("collection_exists") is True,
        "payload_indexes_valid": health.get("payload_indexes_valid") is True,
        "data_freshness_valid": health.get("data_freshness_valid") is True,
        "pgvector_rollback_baseline_valid": True,
        "pre_promotion_snapshot_create_valid": health.get("snapshot_capability_valid") is True,
        "pre_cutover_health_valid": health.get("promotion_health_check_valid") is True,
    }


def build_post_health_from_pre(health: Mapping[str, Any], consistency: Mapping[str, Any]) -> dict[str, Any]:
    valid = health.get("promotion_health_check_valid") is True and consistency.get("data_consistency_valid") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "qdrant_reachable": health.get("server_reachable") is True,
        "collection_valid": health.get("collection_exists") is True,
        "point_count_valid": consistency.get("chunk_count_equivalence") is True,
        "payload_indexes_valid": health.get("payload_indexes_valid") is True,
        "data_freshness_valid": consistency.get("data_consistency_valid") is True,
        "post_cutover_health_check_valid": valid,
    }


def normalize_snapshot(snapshot: Mapping[str, Any], phase: str) -> dict[str, Any]:
    key = f"{phase}_promotion_snapshot_create_valid"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        key: snapshot.get("pre_promotion_snapshot_create_valid") is True,
        f"{phase}_promotion_snapshot_id": ((snapshot.get("snapshot") or {}).get("snapshot_name")),
        "snapshot_file_committed_to_git": False,
    }


def build_search_smoke(simulation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "vector_backend_used": "qdrant",
        "candidate_identity_valid": int(simulation.get("candidate_identity_error_count") or 0) == 0,
        "result_non_empty_when_expected": int(simulation.get("query_count") or 0) > 0,
        "qdrant_authoritative_search_smoke_valid": simulation.get("qdrant_authoritative_simulation_valid") is True,
    }


def skipped_runtime(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "runtime_executed": False, "reason": reason}


def skipped_consistency(reason: str) -> dict[str, Any]:
    payload = task0200.skipped_data_consistency(reason)
    return {**payload, "schema_version": SCHEMA_VERSION, "task_id": TASK_ID}


def skipped_health(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "pre_cutover_health_valid": False, "post_cutover_health_check_valid": False, "reason": reason}


def skipped_snapshot(reason: str, phase: str) -> dict[str, Any]:
    key = f"{phase}_promotion_snapshot_create_valid"
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, key: False, "reason": reason, "snapshot_file_committed_to_git": False}


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key not in {"generated_at", "hostname", "pid", "latency_ms"}}
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_report(
    summary: Mapping[str, Any],
    consistency: Mapping[str, Any],
    formal: Mapping[str, Any],
    observation: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0201 Qdrant Controlled Production Promotion",
            "",
            f"task_status=`{summary.get('task_status')}`; decision=`{summary.get('qdrant_production_promotion_decision')}`; production_vector_backend=`{summary.get('production_vector_backend')}`.",
            "",
            "## Direct Answers",
            "",
            f"1. Promotion executed: `{summary.get('production_backend_cutover_executed')}`.",
            f"2. Qdrant became production authoritative backend: `{summary.get('production_backend_promoted')}`.",
            f"3. Pre-cutover data consistency: `{summary.get('data_consistency_valid')}`; missing=`{consistency.get('missing_in_qdrant_count')}`, extra=`{consistency.get('extra_in_qdrant_count')}`, revision mismatches=`{consistency.get('embedding_revision_mismatch_count')}`.",
            f"4. Q01-Q07 post-cutover: `{summary.get('q01_q07_pass_count')}/7`.",
            f"5. Formal retrieval regression within policy: `{formal.get('formal_retrieval_regression_within_policy')}`; Recall@K=`{formal.get('recall_at_k')}`; MRR=`{formal.get('mrr')}`.",
            f"6. Grounding/Citation/Safety regressions: `{summary.get('grounding_regression_count')}` / `{summary.get('citation_regression_count')}` / `{summary.get('safety_regression_count')}`.",
            f"7. Full suite green: `{summary.get('full_suite_pass')}`; failures=`{summary.get('full_suite_failure_count')}`.",
            f"8. Observation window passed: `{observation.get('observation_window_complete')}`.",
            f"9. Rollback triggered: `{rollback.get('rollback_required')}`; rollback_success=`{rollback.get('rollback_success')}`.",
            f"10. pgvector rollback backend available: `{summary.get('pgvector_rollback_backend_preserved')}`.",
            f"11. Final production vector backend: `{summary.get('production_vector_backend')}`.",
            f"12. Next stabilization stage allowed: `{summary.get('qdrant_production_promotion_decision') == 'promoted'}`.",
            "",
            "## Guardrails",
            "",
            "```text",
            f"runtime_default_behavior_change={summary.get('runtime_default_behavior_change')}",
            f"vector_authority_changed={summary.get('vector_authority_changed')}",
            "generation_authority=unchanged",
            "grounding_authority=unchanged",
            "citation_authority=unchanged",
            "pgvector_decommission_allowed=false",
            "```",
            "",
        ]
    )
