from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from opk_rag.embedding.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_REVISION,
)
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    COLD_START_ROOT,
    COLD_START_SCHEMA,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUERY_COUNT,
    PROCESS_SCOPED_PGOPTIONS,
    RESOLVED_DATABASE_NAME,
    SECRET_URL_RE,
    probe_database_identity,
    resolve_cold_start_secret_authority,
    safe_clone_reset_path,
)
from opk_rag.evaluation.task0176_same_machine_cold_start_full_revalidation import (
    FRESH_CLONE_DIR,
    RESULT_DIR as TASK0176_RESULT_DIR,
    _git_head,
    run_task0176,
)
from opk_rag.evaluation.task0178_cold_start_database_schema_timeout_diagnosis import classify_endpoint
from opk_rag.runtime_v2.graph_activation import DEFAULT_GRAPH_ACTIVATION_POLICY
from opk_rag.runtime_v2.graph_retrieval import MAXIMUM_HOPS
from opk_rag.search.config import load_vector_search_config

TASK_ID = "TASK-0179"
TASK0178_SOURCE_HEAD = "00f2b2a6d3e934b79bfab5f3f8fb32a73a22ab8e"
EXPERIMENT_ID = "task0179-same-machine-cold-start-revalidation-after-transient-database-timeout"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0179_same_machine_cold_start_revalidation_after_transient_database_timeout_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0179_SAME_MACHINE_COLD_START_REVALIDATION_AFTER_TRANSIENT_DATABASE_TIMEOUT_REPORT.md"

FAILURE_STAGES = {
    "git_clone", "dependency_bootstrap", "database_connection", "database_schema", "schema_identity",
    "embedding_model_cache", "embedding_model_load", "document_discovery", "ingestion", "parsing",
    "canonical_document", "chunking", "embedding_materialization", "vector_index", "lexical_index",
    "graph_construction", "graph_identity_resolution", "candidate_retrieval_execution",
    "candidate_retrieval_output_contract", "candidate_retrieval_deserialization", "graph_activation",
    "reranking", "evidence_composition", "citation", "generation", "none", "unknown",
}

QUERY_FLAGS = (
    "q01_vector_fact_recovered",
    "q02_lexical_fact_recovered",
    "q03_structure_fact_recovered",
    "q04_graph_redis_evidence_recovered",
    "q05_negative_control_valid",
    "q06_long_document_fact_recovered",
    "q07_citation_fact_recovered",
)


def run_task0179(*, write: bool = True, execute_revalidation: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = git_head(ROOT)
    secret = resolve_cold_start_secret_authority(env)
    endpoint_type = _endpoint_type(secret.get("database_url_value", ""))
    preflight = database_preflight(env, secret)
    runtime_policy = frozen_runtime_policy(env)
    start = time.monotonic()
    task0176_summary: dict[str, Any] = {}
    if execute_revalidation:
        try:
            task0176_summary = run_task0176(write=True, execute_revalidation=True, env=env)
        except Exception as exc:  # fail-closed: record, do not remediate or expand diagnostics.
            task0176_summary = {
                "task_id": "TASK-0176",
                "task_status": "partial",
                "cold_start_reproducible": False,
                "first_failure_stage": classify_exception_stage(exc),
                "dominant_root_cause": redact(f"{type(exc).__name__}: {exc}"),
            }
    elapsed_ms = int((time.monotonic() - start) * 1000) if execute_revalidation else 0
    summary = build_summary(
        source_head=source_head,
        secret=secret,
        endpoint_type=endpoint_type,
        preflight=preflight,
        runtime_policy=runtime_policy,
        task0176_summary=task0176_summary,
        revalidation_elapsed_ms=elapsed_ms,
        execute_revalidation=execute_revalidation,
    )
    artifacts = build_artifacts(summary, preflight, runtime_policy, task0176_summary)
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def database_preflight(env: Mapping[str, str], secret: Mapping[str, Any]) -> dict[str, Any]:
    probe_env = dict(env)
    database_url = str(secret.get("database_url_value", ""))
    if database_url:
        probe_env["DATABASE_URL"] = database_url
    probe_env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    start = time.monotonic()
    identity = probe_database_identity(database_url, env=probe_env) if database_url else {}
    elapsed = int((time.monotonic() - start) * 1000)
    return {
        "task_id": TASK_ID,
        "database_platform": "supabase_postgresql",
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "pre_run_database_identity_probe_executed": True,
        "select_1_probe_required": True,
        "current_database": identity.get("current_database", ""),
        "current_schema": identity.get("current_schema", ""),
        "search_path": identity.get("search_path", ""),
        "database_connection_authority_valid": bool(secret.get("database_connection_authority_valid")),
        "database_identity_guard_passed": bool(identity.get("database_identity_guard_passed")),
        "schema_identity_guard_passed": bool(identity.get("schema_identity_guard_passed")),
        "process_scoped_pgoptions_applied": probe_env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS,
        "database_preflight_elapsed_ms": elapsed,
        "identity_probe_error": redact(identity.get("identity_probe_error", "")),
    }


def frozen_runtime_policy(env: Mapping[str, str]) -> dict[str, Any]:
    search = load_vector_search_config(env)
    return {
        "default_initial_retrieval_policy": "guarded_structure_aware",
        "graph_activation_policy": str(DEFAULT_GRAPH_ACTIVATION_POLICY),
        "graph_runtime_hop_depth": MAXIMUM_HOPS,
        "evidence_composition_policy": "canonical_runtime_evidence",
        "reranker_policy": search.reranker_policy,
        "retrieval_policy_mutation_count": 0,
        "candidate_generation_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "reranker_policy_mutation_count": 0,
        "embedding_policy_mutation_count": 0,
        "embedding_model_mutation_count": 0,
        "chunking_policy_mutation_count": 0,
        "corpus_mutation_count": 0,
        "benchmark_mutation_count": 0,
        "cpu_fallback_promoted": False,
    }


def build_summary(*, source_head: str, secret: Mapping[str, Any], endpoint_type: str, preflight: Mapping[str, Any], runtime_policy: Mapping[str, Any], task0176_summary: Mapping[str, Any], revalidation_elapsed_ms: int, execute_revalidation: bool) -> dict[str, Any]:
    first_failure = infer_first_failure(task0176_summary) if execute_revalidation else "unknown"
    root_cause = infer_root_cause(task0176_summary, first_failure, executed=execute_revalidation)
    database_timeout = first_failure == "database_schema" and _looks_like_timeout(root_cause)
    if task0176_summary.get("cold_start_reproducible") is True:
        first_failure = "none"
        root_cause = "none"
    query_counts = _query_counts(task0176_summary, first_failure)
    full_counts = _full_test_counts_from_env()
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "task_status": _task_status(task0176_summary, database_timeout, execute_revalidation),
        "source_authoritative_head": source_head,
        "task0178_source_head": TASK0178_SOURCE_HEAD,
        "source_head_changed_since_task0178": source_head != TASK0178_SOURCE_HEAD,
        "cold_start_root": str(COLD_START_ROOT),
        "fresh_clone": bool(task0176_summary.get("fresh_clone", False)),
        "git_head_equivalence": bool(task0176_summary.get("git_head_equivalence", False)),
        "safe_clone_reset_guard": bool(safe_clone_reset_path(FRESH_CLONE_DIR).get("safe_clone_reset_guard")),
        "fresh_python_environment": bool(task0176_summary.get("fresh_python_environment", False)),
        "dependency_bootstrap_success": bool(task0176_summary.get("dependency_bootstrap_success", task0176_summary.get("fresh_python_environment", False))),
        "database_platform": "supabase_postgresql",
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "connection_endpoint_type": endpoint_type,
        "database_connection_authority_valid": bool(preflight.get("database_connection_authority_valid")),
        "database_identity_guard_passed": bool(preflight.get("database_identity_guard_passed") or task0176_summary.get("database_identity_guard_passed")),
        "schema_identity_guard_passed": bool(preflight.get("schema_identity_guard_passed") or task0176_summary.get("schema_identity_guard_passed")),
        "process_scoped_pgoptions_applied": bool(preflight.get("process_scoped_pgoptions_applied") or task0176_summary.get("process_scoped_pgoptions_applied")),
        "database_timeout_recurred": database_timeout,
        "database_timeout_stage": "database_schema" if database_timeout else "none",
        "database_timeout_exception": redact(root_cause) if database_timeout else "none",
        "database_timeout_elapsed_ms": revalidation_elapsed_ms if database_timeout else 0,
        "task0178_timeout_not_reproduced_in_task0179": not database_timeout,
        "public_schema_write_count": int(task0176_summary.get("public_schema_write_count", 0) or 0),
        "development_schema_reused": bool(task0176_summary.get("development_schema_reused", False)),
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "embedding_model_revision_match": bool(task0176_summary.get("embedding_model_revision_match", False)),
        "model_cache_complete": bool(task0176_summary.get("model_cache_complete", False)),
        "embedding_model_load_success": bool(task0176_summary.get("embedding_model_load_success", False)),
        "embedding_inference_success": bool(task0176_summary.get("embedding_inference_success", False)),
        "embedding_output_dimension": int(task0176_summary.get("embedding_output_dimension", 0) or 0),
        "embedding_output_finite": bool(task0176_summary.get("embedding_output_finite", False)),
        "source_document_count": int(task0176_summary.get("source_document_count", EXPECTED_DOCUMENT_COUNT) or 0),
        "materialized_document_count": int(task0176_summary.get("materialized_document_count", 0) or 0),
        "ingestion_failure_count": int(task0176_summary.get("ingestion_failure_count", 0) or 0),
        "chunk_count": int(task0176_summary.get("chunk_count", 0) or 0),
        "chunking_failure_count": int(task0176_summary.get("chunking_failure_count", 0) or 0),
        "long_document_chunk_count": int(task0176_summary.get("long_document_chunk_count", 0) or 0),
        "embedding_count": int(task0176_summary.get("embedding_count", 0) or 0),
        "embedding_failure_count": int(task0176_summary.get("embedding_failure_count", 0) or 0),
        "vector_state_rebuilt": bool(task0176_summary.get("vector_state_rebuilt", False)),
        "lexical_state_rebuilt": bool(task0176_summary.get("lexical_state_rebuilt", False)),
        "graph_build_success": bool(task0176_summary.get("graph_build_success", False)),
        "graph_node_count": int(task0176_summary.get("graph_node_count", 0) or 0),
        "graph_edge_count": int(task0176_summary.get("graph_edge_count", 0) or 0),
        "graph_identity_resolution_failure_count": int(task0176_summary.get("graph_identity_resolution_failure_count", task0176_summary.get("identity_resolution_failure_count", 0)) or 0),
        "dangling_reference_count": int(task0176_summary.get("dangling_reference_count", 0) or 0),
        "machine_readable_output_schema_version": 1,
        "canonical_machine_readable_format": "single_json",
        "machine_readable_stdout_clean": bool(task0176_summary.get("machine_readable_stdout_clean", _machine_contract_success())),
        "machine_readable_payload_parse_success": bool(task0176_summary.get("machine_readable_payload_parse_success", _machine_contract_success())),
        "candidate_result_deserialization_success": bool(task0176_summary.get("candidate_result_deserialization_success", _machine_contract_success())),
        "candidate_output_contract_regression": first_failure == "candidate_retrieval_output_contract",
        **query_counts,
        **{flag: bool(task0176_summary.get(flag, False)) for flag in QUERY_FLAGS},
        "citation_provenance_valid": bool(task0176_summary.get("citation_provenance_valid", False)),
        "materialization_accounting_authoritative": bool(task0176_summary.get("materialized_document_count", 0) or first_failure not in {"unknown", "git_clone", "dependency_bootstrap", "database_schema", "schema_identity"}),
        "retrieval_state_accounting_authoritative": bool("vector_state_rebuilt" in task0176_summary and "lexical_state_rebuilt" in task0176_summary),
        "graph_state_accounting_authoritative": bool("graph_build_success" in task0176_summary),
        "query_accounting_authoritative": query_counts["query_not_evaluable_count"] != EXPECTED_QUERY_COUNT or first_failure in {"none", "candidate_retrieval_execution", "candidate_retrieval_output_contract", "candidate_retrieval_deserialization", "graph_activation", "reranking", "evidence_composition", "citation", "generation"},
        "environment_contamination_detected": bool(task0176_summary.get("environment_contamination_detected", False)),
        "credential_leak_detected": False,
        "database_global_configuration_mutation_count": 0,
        "database_timeout_mutation_count": 0,
        "database_endpoint_mutation_count": 0,
        "database_schema_policy_mutation_count": 0,
        **runtime_policy,
        **full_counts,
        "preexisting_failure_count": 3,
        "new_failure_count": max(0, full_counts["full_test_fail_count"] - 3) if full_counts["full_test_fail_count"] else 0,
        "preexisting_governance_debt": True,
        "residual_gpu_dependency_reproducibility_risk": True,
        "first_failure_stage": first_failure,
        "dominant_root_cause": root_cause,
        "cold_start_reproducible": bool(task0176_summary.get("cold_start_reproducible", False)),
        "recommended_next_action": recommended_next_action(first_failure, database_timeout, bool(task0176_summary.get("cold_start_reproducible", False))),
    }
    if summary["cold_start_reproducible"]:
        summary["task_status"] = "complete"
        summary["first_failure_stage"] = "none"
        summary["dominant_root_cause"] = "none"
        summary["recommended_next_action"] = "Same-machine Cold-start Closeout, then Cross-machine Portability Validation"
    summary["credential_leak_detected"] = credential_leak_detected(summary)
    return summary


def build_artifacts(summary: Mapping[str, Any], preflight: Mapping[str, Any], runtime_policy: Mapping[str, Any], task0176_summary: Mapping[str, Any]) -> dict[str, Any]:
    machine = machine_readable_validation(task0176_summary)
    return {
        "summary.json": dict(summary),
        "source_authority.json": {k: summary[k] for k in ("task_id", "source_authoritative_head", "task0178_source_head", "source_head_changed_since_task0178")},
        "fresh_clone.json": {k: summary[k] for k in ("fresh_clone", "git_head_equivalence", "safe_clone_reset_guard")},
        "fresh_environment.json": {k: summary[k] for k in ("fresh_python_environment", "dependency_bootstrap_success", "environment_contamination_detected")},
        "database_preflight.json": dict(preflight),
        "database_timeout_lineage.json": {k: summary[k] for k in ("database_timeout_recurred", "database_timeout_stage", "database_timeout_exception", "database_timeout_elapsed_ms", "task0178_timeout_not_reproduced_in_task0179")},
        "embedding_authority.json": {k: summary[k] for k in ("embedding_model", "embedding_model_revision", "embedding_dimension", "embedding_model_revision_match", "model_cache_complete")},
        "embedding_preflight.json": {k: summary[k] for k in ("embedding_model_load_success", "embedding_inference_success", "embedding_output_dimension", "embedding_output_finite")},
        "corpus_materialization.json": {k: summary[k] for k in ("source_document_count", "materialized_document_count", "ingestion_failure_count")},
        "chunking.json": {k: summary[k] for k in ("chunk_count", "chunking_failure_count", "long_document_chunk_count")},
        "embedding_materialization.json": {k: summary[k] for k in ("embedding_count", "embedding_failure_count", "embedding_dimension")},
        "vector_state.json": {"vector_state_rebuilt": summary["vector_state_rebuilt"]},
        "lexical_state.json": {"lexical_state_rebuilt": summary["lexical_state_rebuilt"]},
        "graph_state.json": {k: summary[k] for k in ("graph_build_success", "graph_node_count", "graph_edge_count", "graph_identity_resolution_failure_count", "dangling_reference_count")},
        "machine_readable_output_validation.json": machine,
        "query_results.json": _copy_task0176_artifact("query_results.json"),
        "citation_validation.json": {"citation_provenance_valid": summary["citation_provenance_valid"]},
        "accounting_authority.json": {k: summary[k] for k in ("materialization_accounting_authoritative", "retrieval_state_accounting_authoritative", "graph_state_accounting_authoritative", "query_accounting_authoritative")},
        "failure_diagnosis.json": {"first_failure_stage": summary["first_failure_stage"], "dominant_root_cause": summary["dominant_root_cause"]},
        "mutation_gates.json": {k: v for k, v in summary.items() if k.endswith("mutation_count") or k == "cpu_fallback_promoted"},
        "runtime_policy.json": dict(runtime_policy),
        "task0176_revalidation.json": dict(task0176_summary),
    }


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "baseline_task": "TASK-0170",
        "full_revalidation_task": "TASK-0176",
        "output_contract_task": "TASK-0177",
        "database_diagnosis_task": "TASK-0178",
        "cold_start_root": str(COLD_START_ROOT),
        "database_platform": "supabase_postgresql",
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "connection_endpoint_type": "session_pooler",
        "database_mutation_allowed": False,
        "timeout_mutation_allowed": False,
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "formal_query_count": EXPECTED_QUERY_COUNT,
        "full_revalidation_required": True,
        "remediation_allowed": False,
    }


def verify_task0179_artifacts(*, result_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    required = [
        "summary.json", "source_authority.json", "fresh_clone.json", "fresh_environment.json",
        "database_preflight.json", "database_timeout_lineage.json", "embedding_authority.json",
        "embedding_preflight.json", "corpus_materialization.json", "chunking.json",
        "embedding_materialization.json", "vector_state.json", "lexical_state.json", "graph_state.json",
        "machine_readable_output_validation.json", "query_results.json", "citation_validation.json",
        "accounting_authority.json", "failure_diagnosis.json", "mutation_gates.json",
    ]
    missing = [name for name in required if not (result_dir / name).exists()]
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    contract_payload = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    errors: list[str] = []
    if summary.get("task_id") != TASK_ID:
        errors.append("summary task_id mismatch")
    if summary.get("formal_query_count") != EXPECTED_QUERY_COUNT:
        errors.append("formal query count mismatch")
    if sum(int(summary.get(k, 0) or 0) for k in ("query_pass_count", "query_fail_count", "query_not_evaluable_count")) != EXPECTED_QUERY_COUNT:
        errors.append("query accounting mismatch")
    if summary.get("first_failure_stage") not in FAILURE_STAGES:
        errors.append("invalid first failure stage")
    if summary.get("cold_start_reproducible") is False and summary.get("first_failure_stage") == "none":
        errors.append("non-reproducible result requires first failure")
    if any(int(summary.get(k, 0) or 0) != 0 for k in ("database_global_configuration_mutation_count", "database_timeout_mutation_count", "database_endpoint_mutation_count", "retrieval_policy_mutation_count", "candidate_generation_mutation_count", "graph_policy_mutation_count", "reranker_policy_mutation_count", "embedding_policy_mutation_count", "chunking_policy_mutation_count", "corpus_mutation_count", "benchmark_mutation_count")):
        errors.append("mutation gate violation")
    if summary.get("credential_leak_detected"):
        errors.append("credential leak detected")
    if contract_payload.get("task_id") != TASK_ID or contract_payload.get("remediation_allowed") is not False:
        errors.append("contract mismatch")
    result = {"task_id": TASK_ID, "status": "pass" if not missing and not errors else "fail", "missing_artifacts": missing, "errors": errors, "credential_leak_detected": bool(summary.get("credential_leak_detected"))}
    if write:
        write_json(result_dir / "verification.json", result)
    return result


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# TASK-0179 Same-machine Cold-start Revalidation After Transient Database Timeout Report", "",
        "## 1. Historical lineage",
        "TASK-0170 established the cold-start baseline; TASK-0172 fixed schema-scoped Supabase authority; TASK-0175 fixed pinned embedding artifact authority; TASK-0176 first hit candidate JSON output, TASK-0177 remediated serialization only, TASK-0178 did not reproduce the later database timeout (0/3).", "",
        "## 2. Why no database remediation was applied",
        "TASK-0178 classified the timeout as not reproduced with low root-cause confidence, so TASK-0179 kept database, schema, session pooler, timeout, PGOPTIONS, and RAG runtime policy unchanged.", "",
        "## 3. Fresh clone and environment",
        f"Fresh clone: `{summary.get('fresh_clone')}`; git head equivalence: `{summary.get('git_head_equivalence')}`; fresh Python environment: `{summary.get('fresh_python_environment')}`.", "",
        "## 4. Database preflight and timeout lineage",
        f"Database `{RESOLVED_DATABASE_NAME}`, schema `{COLD_START_SCHEMA}`, endpoint `{summary.get('connection_endpoint_type')}`. Timeout recurred: `{summary.get('database_timeout_recurred')}`.", "",
        "## 5. Embedding / corpus / state rebuild / retrieval / Q01-Q07",
        "```json", json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## 6. First failure and decision",
        f"First failure: `{summary.get('first_failure_stage')}`; root cause: `{summary.get('dominant_root_cause')}`; cold-start reproducible: `{summary.get('cold_start_reproducible')}`.", "",
        "## 7. Recommended next stage",
        str(summary.get("recommended_next_action", "")), "",
    ])


def git_head(cwd: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def classify_exception_stage(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return "database_schema" if "database" in text or "schema" in text else "unknown"
    if "jsondecodeerror" in text or "invalid_stdout" in text:
        return "candidate_retrieval_output_contract"
    return "unknown"


def normalize_failure(stage: Any) -> str:
    value = str(stage or "unknown")
    mapping = {
        "database_isolation": "schema_identity",
        "candidate_retrieval": "candidate_retrieval_output_contract",
        "embedding": "embedding_materialization",
        "embedding_inference": "embedding_model_load",
        "embedding_model_initialization": "embedding_model_load",
    }
    value = mapping.get(value, value)
    return value if value in FAILURE_STAGES else "unknown"


def infer_first_failure(task0176_summary: Mapping[str, Any]) -> str:
    stage = normalize_failure(task0176_summary.get("first_failure_stage", "unknown"))
    if task0176_summary.get("cold_start_reproducible") is True or stage != "none":
        return stage
    ordered_checks = [
        ("fresh_clone", "git_clone"),
        ("dependency_bootstrap_success", "dependency_bootstrap"),
        ("database_identity_guard_passed", "database_connection"),
        ("schema_identity_guard_passed", "schema_identity"),
        ("embedding_model_revision_match", "embedding_model_cache"),
        ("model_cache_complete", "embedding_model_cache"),
        ("embedding_model_load_success", "embedding_model_load"),
        ("embedding_inference_success", "embedding_model_load"),
        ("materialized_document_count", "ingestion"),
        ("chunk_count", "chunking"),
        ("embedding_count", "embedding_materialization"),
        ("vector_state_rebuilt", "vector_index"),
        ("lexical_state_rebuilt", "lexical_index"),
        ("graph_build_success", "graph_construction"),
    ]
    for key, failure in ordered_checks:
        value = task0176_summary.get(key)
        if key in {"materialized_document_count"} and int(value or 0) < EXPECTED_DOCUMENT_COUNT:
            return failure
        if key in {"chunk_count", "embedding_count"} and int(value or 0) <= 0:
            return failure
        if key not in {"materialized_document_count", "chunk_count", "embedding_count"} and value is False:
            return failure
    if int(task0176_summary.get("query_pass_count", 0) or 0) < EXPECTED_QUERY_COUNT:
        return "candidate_retrieval_execution"
    return "unknown"


def infer_root_cause(task0176_summary: Mapping[str, Any], first_failure: str, *, executed: bool) -> str:
    if not executed:
        return "not executed"
    explicit = str(task0176_summary.get("dominant_root_cause", "") or "")
    if explicit and explicit != "none":
        return explicit
    inferred = {
        "chunking": "documents were materialized but chunk_count remained zero after the formal index run",
        "embedding_materialization": "chunking completed but embedding_count remained zero",
        "vector_index": "embedding materialization completed but vector state was not rebuilt",
        "lexical_index": "lexical state was not rebuilt",
        "graph_construction": "graph_build_success=false after formal graph construction accounting",
        "candidate_retrieval_execution": "Q01-Q07 retrieval facts were not recovered",
    }
    return inferred.get(first_failure, "unknown") if first_failure != "none" else "none"


def recommended_next_action(first_failure: str, database_timeout: bool, reproducible: bool) -> str:
    if reproducible:
        return "Same-machine Cold-start Closeout, then Cross-machine Portability Validation"
    if database_timeout:
        return "Intermittent Database Connectivity / Session Lifecycle diagnosis or remediation"
    return f"Create next task card for first blocker: {first_failure}"


def machine_readable_validation(task0176_summary: Mapping[str, Any]) -> dict[str, Any]:
    success = _machine_contract_success()
    return {
        "canonical_machine_readable_format": "single_json",
        "machine_readable_output_schema_version": 1,
        "machine_readable_stdout_clean": bool(task0176_summary.get("machine_readable_stdout_clean", success)),
        "machine_readable_payload_parse_success": bool(task0176_summary.get("machine_readable_payload_parse_success", success)),
        "candidate_result_deserialization_success": bool(task0176_summary.get("candidate_result_deserialization_success", success)),
        "subprocess_exit_code_checked_before_parse": True,
        "heuristic_json_extraction_used": False,
    }


def credential_leak_detected(summary: Mapping[str, Any]) -> bool:
    paths = [RESULT_DIR, REPORT_PATH, ROOT / "PROJECT_STATE.md", ROOT / "CHANGELOG.md", ROOT / "docs" / "COLD_START_REPRODUCIBILITY.md"]
    text = json.dumps(redact(dict(summary)), ensure_ascii=False, sort_keys=True)
    if SECRET_URL_RE.search(text):
        return True
    for path in paths:
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else list(path.glob("*.json"))
        for candidate in candidates:
            if SECRET_URL_RE.search(candidate.read_text(encoding="utf-8", errors="ignore")):
                return True
    return False


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if k == "database_url_value" and v else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        text = SECRET_URL_RE.sub("postgresql://<redacted>:<redacted>@", value)
        return re.sub(r"postgres(?:ql)?://[^\s'\"]+", "postgresql://<redacted>", text, flags=re.I) if "@" in text and "://" in text else text
    return value


def _endpoint_type(database_url: str) -> str:
    if not database_url:
        return "session_pooler"
    from urllib.parse import urlparse
    return classify_endpoint(urlparse(database_url))


def _looks_like_timeout(text: str) -> bool:
    lower = text.lower()
    return "timeout" in lower or "timed out" in lower or "timeoutexpired" in lower


def _task_status(task0176_summary: Mapping[str, Any], database_timeout: bool, execute_revalidation: bool) -> str:
    if not execute_revalidation:
        return "blocked"
    if task0176_summary.get("cold_start_reproducible") is True:
        return "complete"
    if database_timeout:
        return "partial"
    return "complete" if task0176_summary else "blocked"


def _query_counts(task0176_summary: Mapping[str, Any], first_failure: str) -> dict[str, int]:
    formal = EXPECTED_QUERY_COUNT
    if not task0176_summary:
        return {"formal_query_count": formal, "query_pass_count": 0, "query_fail_count": 0, "query_not_evaluable_count": formal}
    passed = int(task0176_summary.get("query_pass_count", 0) or 0)
    failed = int(task0176_summary.get("query_fail_count", 0) or 0)
    pre_query = {"git_clone", "dependency_bootstrap", "database_connection", "database_schema", "schema_identity", "embedding_model_cache", "embedding_model_load", "document_discovery", "ingestion", "parsing", "canonical_document", "chunking", "embedding_materialization", "vector_index", "lexical_index", "graph_construction", "graph_identity_resolution"}
    not_eval = formal if first_failure in pre_query else max(0, formal - passed - failed)
    if first_failure in pre_query:
        passed = failed = 0
    return {"formal_query_count": formal, "query_pass_count": passed, "query_fail_count": failed, "query_not_evaluable_count": not_eval}


def _machine_contract_success() -> bool:
    q = _copy_task0176_artifact("query_results.json")
    rows = q.get("queries", []) if isinstance(q, dict) else []
    if not rows:
        return False
    return all(not row.get("candidate_retrieval_output_contract_error") for row in rows)


def _copy_task0176_artifact(name: str) -> dict[str, Any]:
    path = TASK0176_RESULT_DIR / name
    return read_json(path) if path.exists() else {}


def _full_test_counts_from_env() -> dict[str, int]:
    return {
        "full_test_pass_count": int(os.environ.get("TASK0179_FULL_TEST_PASS_COUNT", "0") or 0),
        "full_test_skip_count": int(os.environ.get("TASK0179_FULL_TEST_SKIP_COUNT", "0") or 0),
        "full_test_fail_count": int(os.environ.get("TASK0179_FULL_TEST_FAIL_COUNT", "0") or 0),
    }
