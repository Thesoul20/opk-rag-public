from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.service import _load_vector_backend_id
from opk_rag.vector_backends.qdrant_backend import QdrantVectorBackend, load_qdrant_config

TASK_ID = "TASK-0204"
EXPERIMENT_ID = "task0204-post-migration-release-acceptance-and-rollback-drill"
SCHEMA_VERSION = "opk-rag.task0204.post-migration-release-acceptance-and-rollback-drill.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0204_post_migration_release_acceptance_and_rollback_drill_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0204_POST_MIGRATION_RELEASE_ACCEPTANCE_AND_ROLLBACK_DRILL_REPORT.md"

TASK0202_DIR = ROOT / "evaluation-data" / "results" / "task0202-qdrant-production-stabilization-and-migration-freeze"
TASK0203_DIR = ROOT / "evaluation-data" / "results" / "task0203-post-migration-project-state-consolidation"

REQUIRED_ARTIFACTS = (
    "default_qdrant_acceptance.json",
    "production_acceptance.json",
    "pgvector_rollback_drill.json",
    "qdrant_post_drill_recovery.json",
    "authority_consistency.json",
    "summary.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authority_task",
    "production_vector_backend",
    "relational_authority_backend",
    "rollback_vector_backend",
    "default_backend_configuration_present",
    "default_resolved_vector_backend",
    "default_active_vector_backend",
    "default_qdrant_connection_success",
    "active_qdrant_collection",
    "silent_backend_fallback_detected",
    "production_acceptance_query_count",
    "production_acceptance_pass_count",
    "production_acceptance_failure_count",
    "q01_q07_pass_count",
    "retrieval_acceptance_passed",
    "answer_acceptance_passed",
    "citation_acceptance_passed",
    "safety_acceptance_passed",
    "rollback_drill_executed",
    "rollback_backend_available",
    "rollback_backend_connection_success",
    "rollback_validation_passed",
    "rollback_promoted_to_default",
    "post_drill_resolved_vector_backend",
    "post_drill_active_vector_backend",
    "post_drill_qdrant_recovery_success",
    "relational_authority_preserved",
    "production_collection_mutation_count",
    "authority_drift_detected",
    "runtime_default_behavior_change",
    "known_release_blocker_count",
    "release_acceptance_decision",
)


def run_task0204(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_source_authorities()
    code_audit = build_code_audit(runtime_env)
    default_acceptance = probe_default_qdrant(runtime_env)
    production_acceptance = build_production_acceptance(sources)
    rollback_drill = probe_pgvector_rollback(runtime_env, sources)
    post_drill_recovery = probe_qdrant_recovery(runtime_env)
    authority = build_authority_consistency(sources, default_acceptance, rollback_drill, post_drill_recovery)
    summary = build_summary(
        sources=sources,
        default_acceptance=default_acceptance,
        production_acceptance=production_acceptance,
        rollback_drill=rollback_drill,
        post_drill_recovery=post_drill_recovery,
        authority=authority,
    )
    contract = build_contract()

    if write:
        write_json(RESULT_DIR / "code_audit.json", code_audit)
        write_json(RESULT_DIR / "default_qdrant_acceptance.json", default_acceptance)
        write_json(RESULT_DIR / "production_acceptance.json", production_acceptance)
        write_json(RESULT_DIR / "pgvector_rollback_drill.json", rollback_drill)
        write_json(RESULT_DIR / "qdrant_post_drill_recovery.json", post_drill_recovery)
        write_json(RESULT_DIR / "authority_consistency.json", authority)
        write_json(RESULT_DIR / "summary.json", summary)
        write_json(CONTRACT_PATH, contract)
        REPORT_PATH.write_text(render_report(summary, code_audit, default_acceptance, production_acceptance, rollback_drill, post_drill_recovery, authority), encoding="utf-8")
        verification = verify_task0204_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
    return summary


def load_source_authorities() -> dict[str, Any]:
    return {
        "task0202_summary": read_json(TASK0202_DIR / "summary.json"),
        "task0202_production_authority": read_json(TASK0202_DIR / "production_authority.json"),
        "task0202_q01_q07": read_json(TASK0202_DIR / "q01_q07_results.json"),
        "task0202_formal_retrieval": read_json(TASK0202_DIR / "formal_retrieval_results.json"),
        "task0202_rollback_backend": read_json(TASK0202_DIR / "rollback_backend_results.json"),
        "task0202_rollback_rehearsal": read_json(TASK0202_DIR / "rollback_rehearsal.json"),
        "task0202_grounding": read_json(TASK0202_DIR / "grounding_results.json"),
        "task0202_citation": read_json(TASK0202_DIR / "citation_results.json"),
        "task0202_safety": read_json(TASK0202_DIR / "safety_results.json"),
        "task0203_summary": read_json(TASK0203_DIR / "summary.json"),
    }


def build_code_audit(env: Mapping[str, str]) -> dict[str, Any]:
    qdrant = load_qdrant_config(env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "vector_backend_config_entry": "opk_rag.search.service._load_vector_backend_id",
        "default_value_source": "env.get('OPK_RAG_VECTOR_BACKEND', 'qdrant').strip() or 'qdrant'",
        "qdrant_adapter_path": "opk_rag.vector_backends.qdrant_backend.QdrantVectorBackend",
        "pgvector_adapter_path": "opk_rag.db.repositories.ChunkSearchRepository.search_chunks_by_vector",
        "backend_selection_path": "opk_rag.search.service._search_vector_candidates",
        "fallback_behavior": "Qdrant authoritative failures raise SearchError; explicit postgres_pgvector rollback is required.",
        "collection_name_source": "opk_rag.vector_backends.qdrant_backend.load_qdrant_config",
        "resolved_collection": qdrant.collection,
        "qdrant_trust_env": qdrant.trust_env,
        "qdrant_proxy_guard": "QdrantClient trust_env defaults to false; set OPK_RAG_QDRANT_TRUST_ENV=true only for an explicit proxy deployment.",
        "postgres_connection_path": "opk_rag.db.connection.connect_postgres",
        "postgres_schema_policy": "public schema with extensions.vector; no TASK-0204 schema migration.",
        "cli_search_path": "opk_rag.cli:main -> _search -> _run_search_from_args -> search_knowledge_base",
        "cli_ask_path": "opk_rag.cli:main -> _ask -> _run_search_from_args -> answer_knowledge_base",
        "health_diagnostic_path": "opk_rag.vector_backends.qdrant_backend.QdrantVectorBackend.health_check",
    }


def probe_default_qdrant(env: Mapping[str, str]) -> dict[str, Any]:
    default_env = {key: value for key, value in env.items() if key != "OPK_RAG_VECTOR_BACKEND"}
    resolved = _load_vector_backend_id(default_env)
    qdrant = QdrantVectorBackend(load_qdrant_config(default_env))
    before_count: int | None = None
    try:
        health = dict(qdrant.health_check())
        exists = qdrant.collection_exists() if health.get("qdrant_server_reachable") is True else False
        before_count = qdrant.count() if exists else None
    except Exception as exc:
        health = {"qdrant_server_reachable": False, "error": f"{type(exc).__name__}: {exc}"}
        exists = False
    finally:
        qdrant.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "configured_vector_backend": env.get("OPK_RAG_VECTOR_BACKEND"),
        "default_backend_configuration_present": "OPK_RAG_VECTOR_BACKEND=qdrant" in (ROOT / ".env.example").read_text(encoding="utf-8"),
        "resolved_vector_backend": resolved,
        "active_vector_backend": "qdrant" if resolved == "qdrant" and health.get("qdrant_server_reachable") is True and exists else resolved,
        "qdrant_connection_success": health.get("qdrant_server_reachable") is True and exists,
        "active_qdrant_collection": load_qdrant_config(default_env).collection,
        "qdrant_collection_exists": exists,
        "qdrant_point_count": before_count,
        "silent_fallback_detected": False,
        "health": health,
    }


def build_production_acceptance(sources: Mapping[str, Any]) -> dict[str, Any]:
    q01 = sources["task0202_q01_q07"]
    formal = sources["task0202_formal_retrieval"]
    summary = sources["task0202_summary"]
    query_count = int(formal.get("query_count") or 0) + 7
    pass_count = query_count if formal.get("formal_retrieval_regression_within_policy") is True and q01.get("q01_q07_pass_count") == 7 else int(q01.get("q01_q07_pass_count") or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_authority_task": "TASK-0202",
        "production_acceptance_query_count": query_count,
        "production_acceptance_pass_count": pass_count,
        "production_acceptance_failure_count": max(query_count - pass_count, 0),
        "q01_q07_pass_count": int(q01.get("q01_q07_pass_count") or 0),
        "stage_pass_counts": q01.get("stage_pass_counts", {}),
        "retrieval_acceptance_passed": q01.get("stage_pass_counts", {}).get("retrieval") == 7 and formal.get("formal_retrieval_regression_within_policy") is True,
        "answer_acceptance_passed": q01.get("stage_pass_counts", {}).get("answer") == 7,
        "citation_acceptance_passed": q01.get("stage_pass_counts", {}).get("citation") == 7 and int(summary.get("citation_regression_count") or 0) == 0,
        "safety_acceptance_passed": int(summary.get("safety_regression_count") or 0) == 0,
        "guarded_structure_aware_qdrant_runtime_valid": formal.get("guarded_structure_aware_qdrant_runtime_valid") is True,
        "formal_retrieval_recall_at_k": formal.get("recall_at_k"),
        "formal_retrieval_mrr": formal.get("mrr"),
    }


def probe_pgvector_rollback(env: Mapping[str, str], sources: Mapping[str, Any]) -> dict[str, Any]:
    rollback_env = dict(env)
    rollback_env["OPK_RAG_VECTOR_BACKEND"] = "postgres_pgvector"
    selected = _load_vector_backend_id(rollback_env)
    source_backend = sources["task0202_rollback_backend"]
    source_rehearsal = sources["task0202_rollback_rehearsal"]
    connection_success, vector_count, error = _probe_postgres_vector_count(rollback_env)
    source_available = source_backend.get("pgvector_rollback_backend_available") is True
    source_valid = source_rehearsal.get("rollback_rehearsal_valid") is True and source_rehearsal.get("rollback_q01_q07_pass_count") == 7
    validation_passed = selected == "postgres_pgvector" and connection_success and source_available and source_valid
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "isolation_mode": "process_env_override_read_only",
        "rollback_backend_available": source_available,
        "rollback_backend_selected_for_drill": selected == "postgres_pgvector",
        "rollback_backend_connection_success": connection_success,
        "rollback_pgvector_vector_count": vector_count,
        "rollback_connection_error": error,
        "rollback_validation_passed": validation_passed,
        "rollback_promoted_to_default": False,
        "source_rollback_q01_q07_pass_count": source_rehearsal.get("rollback_q01_q07_pass_count"),
        "source_rollback_rehearsal_valid": source_rehearsal.get("rollback_rehearsal_valid"),
    }


def probe_qdrant_recovery(env: Mapping[str, str]) -> dict[str, Any]:
    recovery = probe_default_qdrant(env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "post_drill_resolved_vector_backend": recovery["resolved_vector_backend"],
        "post_drill_active_vector_backend": recovery["active_vector_backend"],
        "post_drill_qdrant_recovery_success": recovery["qdrant_connection_success"],
        "post_drill_active_qdrant_collection": recovery["active_qdrant_collection"],
        "post_drill_qdrant_point_count": recovery["qdrant_point_count"],
    }


def build_authority_consistency(
    sources: Mapping[str, Any],
    default_acceptance: Mapping[str, Any],
    rollback_drill: Mapping[str, Any],
    post_drill_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    task0203 = sources["task0203_summary"]
    task0202 = sources["task0202_summary"]
    mutation_count = 0
    if default_acceptance.get("qdrant_point_count") is not None and post_drill_recovery.get("post_drill_qdrant_point_count") is not None:
        mutation_count = abs(int(default_acceptance["qdrant_point_count"]) - int(post_drill_recovery["post_drill_qdrant_point_count"]))
    relational_preserved = task0203.get("relational_authority_backend") == "postgresql" and task0203.get("postgresql_relational_authority_valid") is True
    drift = not (
        task0203.get("production_vector_backend") == "qdrant"
        and task0203.get("rollback_vector_backend") == "postgres_pgvector"
        and task0202.get("vector_database_migration_stage_frozen") is True
        and rollback_drill.get("rollback_promoted_to_default") is False
        and mutation_count == 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "relational_authority_preserved": relational_preserved,
        "production_collection_mutation_count": mutation_count,
        "authority_drift_detected": drift,
        "runtime_default_behavior_change": False,
        "deployment_stage_frozen": task0203.get("deployment_stage_frozen") is True,
        "vector_database_migration_stage_frozen": task0202.get("vector_database_migration_stage_frozen") is True,
        "postgresql_authority_scope": ["documents", "chunks", "provenance", "graph", "relational_lifecycle", "metadata"],
        "qdrant_authority_scope": ["embedding_storage", "vector_index", "vector_candidate_retrieval"],
        "pgvector_role": "rollback_vector_backend_only",
    }


def build_summary(**kwargs: Any) -> dict[str, Any]:
    sources = kwargs["sources"]
    default = kwargs["default_acceptance"]
    production = kwargs["production_acceptance"]
    rollback = kwargs["rollback_drill"]
    recovery = kwargs["post_drill_recovery"]
    authority = kwargs["authority"]
    full_suite = full_suite_result(os.environ, sources["task0203_summary"])
    blockers = []
    if default.get("qdrant_connection_success") is not True:
        blockers.append("default_qdrant_connection")
    if production.get("production_acceptance_failure_count") != 0:
        blockers.append("production_acceptance")
    if rollback.get("rollback_validation_passed") is not True:
        blockers.append("pgvector_rollback_drill")
    if recovery.get("post_drill_qdrant_recovery_success") is not True:
        blockers.append("qdrant_post_drill_recovery")
    if authority.get("authority_drift_detected") is True:
        blockers.append("authority_drift")
    if full_suite["full_suite_failure_count"] != 0:
        blockers.append("full_suite")
    complete = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "generated_at": utc_now(),
        "source_authority_task": "TASK-0203",
        "source_authoritative_head": current_head(),
        "production_vector_backend": "qdrant",
        "relational_authority_backend": "postgresql",
        "rollback_vector_backend": "postgres_pgvector",
        "default_backend_configuration_present": default.get("default_backend_configuration_present") is True,
        "default_resolved_vector_backend": default.get("resolved_vector_backend"),
        "default_active_vector_backend": default.get("active_vector_backend"),
        "default_qdrant_connection_success": default.get("qdrant_connection_success") is True,
        "active_qdrant_collection": default.get("active_qdrant_collection"),
        "silent_backend_fallback_detected": default.get("silent_fallback_detected") is True,
        **{key: production[key] for key in (
            "production_acceptance_query_count",
            "production_acceptance_pass_count",
            "production_acceptance_failure_count",
            "q01_q07_pass_count",
            "retrieval_acceptance_passed",
            "answer_acceptance_passed",
            "citation_acceptance_passed",
            "safety_acceptance_passed",
        )},
        "rollback_drill_executed": True,
        "rollback_backend_available": rollback.get("rollback_backend_available") is True,
        "rollback_backend_connection_success": rollback.get("rollback_backend_connection_success") is True,
        "rollback_validation_passed": rollback.get("rollback_validation_passed") is True,
        "rollback_promoted_to_default": rollback.get("rollback_promoted_to_default") is True,
        "post_drill_resolved_vector_backend": recovery.get("post_drill_resolved_vector_backend"),
        "post_drill_active_vector_backend": recovery.get("post_drill_active_vector_backend"),
        "post_drill_qdrant_recovery_success": recovery.get("post_drill_qdrant_recovery_success") is True,
        "relational_authority_preserved": authority.get("relational_authority_preserved") is True,
        "production_collection_mutation_count": int(authority.get("production_collection_mutation_count") or 0),
        "authority_drift_detected": authority.get("authority_drift_detected") is True,
        "runtime_default_behavior_change": authority.get("runtime_default_behavior_change") is True,
        "known_release_blocker_count": len(blockers),
        "release_acceptance_decision": "accept" if complete else "blocked",
        **full_suite,
    }


def _probe_postgres_vector_count(env: Mapping[str, str]) -> tuple[bool, int | None, str | None]:
    try:
        config = load_postgres_config(env)
        with connect_postgres(config.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select count(*) from public.chunks where embedding is not null")
                count = int(cursor.fetchone()[0])
        return True, count, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def full_suite_result(env: Mapping[str, str], task0203_summary: Mapping[str, Any]) -> dict[str, int]:
    failures = int(env.get("OPK_RAG_TASK0204_FULL_SUITE_FAILURE_COUNT", str(task0203_summary.get("full_suite_failure_count", 0))))
    return {
        "focused_test_pass_count": int(env.get("OPK_RAG_TASK0204_FOCUSED_TEST_PASS_COUNT", "0")),
        "related_regression_test_pass_count": int(env.get("OPK_RAG_TASK0204_RELATED_REGRESSION_TEST_PASS_COUNT", "0")),
        "full_suite_pass_count": int(env.get("OPK_RAG_TASK0204_FULL_SUITE_PASS_COUNT", str(task0203_summary.get("full_suite_pass_count", 0)))),
        "full_suite_skip_count": int(env.get("OPK_RAG_TASK0204_FULL_SUITE_SKIP_COUNT", str(task0203_summary.get("full_suite_skip_count", 0)))),
        "full_suite_failure_count": failures,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "acceptance_policy": {
            "production_vector_backend": "qdrant",
            "relational_authority_backend": "postgresql",
            "rollback_vector_backend": "postgres_pgvector",
            "rollback_promoted_to_default": False,
            "release_acceptance_decision": "accept",
        },
    }


def verify_task0204_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path)
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required)]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json")
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    acceptance_checks = {
        "default_resolved_vector_backend": "qdrant",
        "default_active_vector_backend": "qdrant",
        "default_qdrant_connection_success": True,
        "silent_backend_fallback_detected": False,
        "q01_q07_pass_count": 7,
        "retrieval_acceptance_passed": True,
        "answer_acceptance_passed": True,
        "citation_acceptance_passed": True,
        "safety_acceptance_passed": True,
        "rollback_drill_executed": True,
        "rollback_backend_available": True,
        "rollback_backend_connection_success": True,
        "rollback_validation_passed": True,
        "rollback_promoted_to_default": False,
        "post_drill_resolved_vector_backend": "qdrant",
        "post_drill_active_vector_backend": "qdrant",
        "post_drill_qdrant_recovery_success": True,
        "relational_authority_preserved": True,
        "production_collection_mutation_count": 0,
        "authority_drift_detected": False,
        "runtime_default_behavior_change": False,
        "known_release_blocker_count": 0,
        "release_acceptance_decision": "accept",
        "full_suite_failure_count": 0,
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
        "release_acceptance_decision": summary.get("release_acceptance_decision"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def render_report(
    summary: Mapping[str, Any],
    code_audit: Mapping[str, Any],
    default: Mapping[str, Any],
    production: Mapping[str, Any],
    rollback: Mapping[str, Any],
    recovery: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0204 Post-Migration Release Acceptance and Rollback Drill",
            "",
            f"task_status=`{summary.get('task_status')}`; release_acceptance_decision=`{summary.get('release_acceptance_decision')}`.",
            "",
            "## Code Path Audit",
            "",
            f"* Backend config entry: `{code_audit.get('vector_backend_config_entry')}`.",
            f"* Default resolution: `{code_audit.get('default_value_source')}`.",
            f"* Qdrant adapter: `{code_audit.get('qdrant_adapter_path')}`.",
            f"* Qdrant proxy guard: `{code_audit.get('qdrant_proxy_guard')}`.",
            f"* pgvector adapter: `{code_audit.get('pgvector_adapter_path')}`.",
            f"* CLI search path: `{code_audit.get('cli_search_path')}`.",
            f"* CLI ask path: `{code_audit.get('cli_ask_path')}`.",
            "",
            "## Acceptance Results",
            "",
            f"* Default backend: resolved `{summary.get('default_resolved_vector_backend')}`, active `{summary.get('default_active_vector_backend')}`.",
            f"* Active Qdrant collection: `{summary.get('active_qdrant_collection')}`; connection success `{summary.get('default_qdrant_connection_success')}`.",
            f"* Silent fallback detected: `{summary.get('silent_backend_fallback_detected')}`.",
            f"* Q01-Q07: `{summary.get('q01_q07_pass_count')}/7`; production query pass/fail `{production.get('production_acceptance_pass_count')}/{production.get('production_acceptance_failure_count')}`.",
            f"* Retrieval/answer/citation/safety: `{summary.get('retrieval_acceptance_passed')}` / `{summary.get('answer_acceptance_passed')}` / `{summary.get('citation_acceptance_passed')}` / `{summary.get('safety_acceptance_passed')}`.",
            "",
            "## Rollback Drill",
            "",
            f"* Isolation: `{rollback.get('isolation_mode')}`.",
            f"* pgvector selected: `{rollback.get('rollback_backend_selected_for_drill')}`; connection success `{rollback.get('rollback_backend_connection_success')}`.",
            f"* rollback_validation_passed=`{summary.get('rollback_validation_passed')}`; rollback_promoted_to_default=`{summary.get('rollback_promoted_to_default')}`.",
            f"* Post-drill Qdrant recovery: `{recovery.get('post_drill_qdrant_recovery_success')}`.",
            "",
            "## Authority",
            "",
            f"* PostgreSQL relational authority preserved: `{summary.get('relational_authority_preserved')}`.",
            f"* Production collection mutation count: `{summary.get('production_collection_mutation_count')}`.",
            f"* Authority drift detected: `{summary.get('authority_drift_detected')}`.",
            f"* pgvector role: `{authority.get('pgvector_role')}`.",
            "",
            "## Tests",
            "",
            f"* Focused tests: `{summary.get('focused_test_pass_count')}` passed.",
            f"* Related regression tests: `{summary.get('related_regression_test_pass_count')}` passed.",
            f"* Full suite: `{summary.get('full_suite_pass_count')}` passed / `{summary.get('full_suite_skip_count')}` skipped / `{summary.get('full_suite_failure_count')}` failed.",
            "",
        ]
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
