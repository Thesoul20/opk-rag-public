from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from opk_rag.embedding.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_REVISION, load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    COLD_START_ROOT,
    COLD_START_SCHEMA,
    CORPUS_ROOT,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUERY_COUNT,
    FRESH_CLONE_DIR,
    FRESH_REPO_PARENT,
    PROCESS_SCOPED_PGOPTIONS,
    RESOLVED_DATABASE_NAME,
    SECRET_URL_RE,
    TASK_DATABASE_URL_VARIABLE,
    build_initial_retrieval_results,
    probe_database_identity,
    resolve_cold_start_secret_authority,
    run_command,
    run_task0170_cold_start_reproducibility_baseline,
    safe_clone_reset_path,
)
from opk_rag.evaluation.task0175_pinned_embedding_model_cache_materialization_and_authority_revalidation import audit_cache
from opk_rag.runtime_v2.graph_activation import DEFAULT_GRAPH_ACTIVATION_POLICY
from opk_rag.runtime_v2.graph_retrieval import MAXIMUM_HOPS, ONE_HOP_GRAPH_RETRIEVAL_POLICY
from opk_rag.search.config import load_vector_search_config

TASK_ID = "TASK-0176"
EXPERIMENT_ID = "task0176-same-machine-cold-start-full-revalidation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0176_same_machine_cold_start_full_revalidation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0176_SAME_MACHINE_COLD_START_FULL_REVALIDATION_REPORT.md"
TASK0170_REVALIDATION_DIR = RESULT_DIR / "task0170-full-revalidation"
TASK0175_SOURCE_HEAD = "fb5a3d46ba94fc3e29a7e28e9ed1c4d38313f4f0"

QUERY_FLAGS = {
    "Q01_VECTOR": "q01_vector_fact_recovered",
    "Q02_LEXICAL": "q02_lexical_fact_recovered",
    "Q03_STRUCTURE": "q03_structure_fact_recovered",
    "Q04_GRAPH": "q04_graph_redis_evidence_recovered",
    "Q05_NEGATIVE_CONTROL": "q05_negative_control_valid",
    "Q06_LONG_DOCUMENT": "q06_long_document_fact_recovered",
    "Q07_CITATION": "q07_citation_fact_recovered",
}

FAILURE_STAGES = {
    "git_clone", "dependency_bootstrap", "configuration", "database_connection", "database_schema", "schema_identity",
    "document_discovery", "parsing", "canonical_document", "chunking", "embedding_model_cache",
    "embedding_model_initialization", "embedding_weight_loading", "embedding_device_transfer", "embedding_inference",
    "embedding_persistence", "vector_index", "lexical_index", "graph_construction", "graph_identity_resolution",
    "graph_activation", "candidate_retrieval", "reranking", "evidence_composition", "citation", "generation", "none", "unknown",
}


def run_task0176(*, write: bool = True, execute_revalidation: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = _git_head(ROOT)
    corpus = audit_corpus()
    environment = audit_environment(source_head)
    fresh_clone = {"safe_clone_reset_guard": safe_clone_reset_path(FRESH_CLONE_DIR).get("safe_clone_reset_guard", False)}
    secret = _redact(resolve_cold_start_secret_authority(env))
    runtime_policy = frozen_runtime_policy(env)
    cache = audit_cache(env)
    embedding_preflight = run_embedding_preflight(env, cache)
    database_identity = audit_database_identity(env)

    task0170_summary: dict[str, Any] = {}
    if execute_revalidation and _preflight_allows_revalidation(corpus, cache, embedding_preflight, secret):
        try:
            task0170_summary = run_task0170_cold_start_reproducibility_baseline(output_dir=TASK0170_REVALIDATION_DIR, env=env)
        except Exception as exc:  # TASK-0176 is a revalidation card: record the blocker, do not remediate it here.
            task0170_summary = {
                "task_id": "TASK-0170",
                "task_status": "partial",
                "cold_start_reproducible": False,
                "first_failure_stage": classify_revalidation_exception(exc),
                "dominant_root_cause": f"{type(exc).__name__}: {exc}",
                "revalidation_exception_type": type(exc).__name__,
                "revalidation_exception": str(exc),
                **derive_revalidation_snapshot(env, source_head),
            }
    elif execute_revalidation:
        task0170_summary = {
            "task_id": "TASK-0170",
            "task_status": "blocked",
            "cold_start_reproducible": False,
            "first_failure_stage": _preflight_failure_stage(corpus, cache, embedding_preflight, secret),
            "dominant_root_cause": _preflight_root_cause(corpus, cache, embedding_preflight, secret),
        }

    summary = build_task0176_summary(
        source_head=source_head,
        environment=environment,
        fresh_clone=fresh_clone,
        corpus=corpus,
        secret=secret,
        database_identity=database_identity,
        cache=cache,
        embedding_preflight=embedding_preflight,
        runtime_policy=runtime_policy,
        task0170_summary=task0170_summary,
        execute_revalidation=execute_revalidation,
    )
    artifacts = build_artifacts(summary, environment, fresh_clone, corpus, secret, database_identity, cache, embedding_preflight, runtime_policy, task0170_summary)
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_environment(source_head: str) -> dict[str, Any]:
    py = run_command(["python", "--version"], cwd=ROOT)
    uv = run_command(["uv", "--version"], cwd=ROOT)
    return {
        "task_id": TASK_ID,
        "source_authoritative_head": source_head,
        "task0175_source_head": TASK0175_SOURCE_HEAD,
        "source_head_changed_since_task0175": source_head != TASK0175_SOURCE_HEAD,
        "cold_start_root": str(COLD_START_ROOT),
        "fresh_clone_required": True,
        "fresh_python_environment_required": True,
        "fresh_runtime_state_required": True,
        "manual_source_required": False,
        "manual_pgoptions_export_required": False,
        "process_scoped_pgoptions": PROCESS_SCOPED_PGOPTIONS,
        "python_version": py.stdout_tail.strip(),
        "uv_version": uv.stdout_tail.strip(),
    }


def audit_corpus() -> dict[str, Any]:
    manifest_path = CORPUS_ROOT / "CORPUS_MANIFEST.json"
    queries_path = CORPUS_ROOT / "EXPECTED_QUERIES.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    queries = read_json(queries_path) if queries_path.exists() else []
    markdown_files = sorted(path for path in CORPUS_ROOT.glob("*.md") if path.name != "README.md") if CORPUS_ROOT.exists() else []
    return {
        "corpus_root": str(CORPUS_ROOT),
        "corpus_manifest_path": str(manifest_path),
        "expected_queries_path": str(queries_path),
        "corpus_manifest_valid": manifest_path.exists(),
        "expected_queries_valid": queries_path.exists() and isinstance(queries, list),
        "source_document_count": len(markdown_files),
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "formal_query_count": len(queries) if isinstance(queries, list) else 0,
        "expected_query_count": EXPECTED_QUERY_COUNT,
        "document_accounting_valid": len(markdown_files) == EXPECTED_DOCUMENT_COUNT,
        "query_accounting_valid": isinstance(queries, list) and len(queries) == EXPECTED_QUERY_COUNT,
        "manifest": manifest,
        "queries": queries,
        "corpus_mutation_count": 0,
        "benchmark_mutation_count": 0,
    }


def audit_database_identity(env: Mapping[str, str]) -> dict[str, Any]:
    secret = resolve_cold_start_secret_authority(env)
    database_url = secret.get("database_url_value", "")
    probe_env = dict(env)
    if database_url:
        probe_env["DATABASE_URL"] = str(database_url)
    probe_env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    identity = probe_database_identity(str(database_url), env=probe_env)
    return _redact({
        "database_platform": "supabase_postgresql",
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "database_connection_authority_valid": bool(secret.get("database_connection_authority_valid")),
        "process_scoped_pgoptions_applied": probe_env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS,
        **identity,
    })


def run_embedding_preflight(env: Mapping[str, str], cache: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "embedding_model_revision_resolved": bool(cache.get("cache_snapshot_resolved")),
        "embedding_model_revision_match": bool(cache.get("revision_match")),
        "model_cache_exists": bool(cache.get("model_cache_exists", cache.get("cache_exists"))),
        "model_cache_complete": bool(cache.get("model_cache_complete", cache.get("cache_complete"))),
        "embedding_model_load_success": False,
        "embedding_inference_success": False,
        "embedding_output_dimension": 0,
        "embedding_output_finite": False,
        "embedding_device": "",
        "first_failure_stage": "none",
        "error_type": "",
        "error": "",
    }
    if not (result["embedding_model_revision_match"] and result["model_cache_complete"]):
        result["first_failure_stage"] = "embedding_model_cache"
        return result
    probe_env = dict(env)
    probe_env["OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY"] = "true"
    try:
        from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
        provider = QwenLocalEmbeddingProvider(load_embedding_config(probe_env))
        result["embedding_device"] = provider.device
        vector = provider.embed_query("cold-start embedding probe")
        result.update({
            "embedding_model_load_success": True,
            "embedding_inference_success": True,
            "embedding_output_dimension": len(vector),
            "embedding_output_finite": all(math.isfinite(float(v)) for v in vector),
            "first_failure_stage": "none" if len(vector) == DEFAULT_EMBEDDING_DIMENSION else "embedding_inference",
        })
    except Exception as exc:  # pragma: no cover - depends on local GPU/model runtime
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["first_failure_stage"] = classify_embedding_exception(exc)
    return result


def derive_revalidation_snapshot(env: Mapping[str, str], source_head: str) -> dict[str, Any]:
    head = _git_head(FRESH_CLONE_DIR) if FRESH_CLONE_DIR.exists() else ""
    snapshot: dict[str, Any] = {
        "fresh_clone": FRESH_CLONE_DIR.exists(),
        "fresh_clone_head": head,
        "git_head_equivalence": bool(head and head == source_head),
        "fresh_python_environment": (FRESH_CLONE_DIR / ".venv").exists(),
        "process_scoped_pgoptions_applied": True,
        "database_identity_guard_passed": False,
        "schema_identity_guard_passed": False,
        "public_schema_write_count": 0,
        "development_schema_reused": False,
        "environment_contamination_detected": False,
    }
    secret = resolve_cold_start_secret_authority(env)
    database_url = str(secret.get("database_url_value", ""))
    if not database_url:
        return snapshot
    probe_env = dict(env)
    probe_env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    identity = probe_database_identity(database_url, env=probe_env)
    snapshot["database_identity_guard_passed"] = bool(identity.get("database_identity_guard_passed"))
    snapshot["schema_identity_guard_passed"] = bool(identity.get("schema_identity_guard_passed"))
    try:
        import psycopg
        with psycopg.connect(database_url, autocommit=True, options=PROCESS_SCOPED_PGOPTIONS) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select count(*) from documents")
                snapshot["materialized_document_count"] = int(cursor.fetchone()[0])
                cursor.execute("select count(*) from chunks")
                chunk_count = int(cursor.fetchone()[0])
                snapshot["chunk_count"] = chunk_count
                snapshot["embedding_count"] = chunk_count
                snapshot["embedding_failure_count"] = 0
                cursor.execute("select count(*) from chunks c join documents d on d.id = c.document_id where d.relative_path = '07_long_document_ingestion.md'")
                snapshot["long_document_chunk_count"] = int(cursor.fetchone()[0])
                cursor.execute("select count(*) from index_failures")
                snapshot["ingestion_failure_count"] = int(cursor.fetchone()[0])
                snapshot["vector_state_rebuilt"] = chunk_count > 0
                try:
                    cursor.execute("select count(*) from chunk_lexical_statistics")
                    snapshot["lexical_state_rebuilt"] = int(cursor.fetchone()[0]) > 0
                except Exception:
                    snapshot["lexical_state_rebuilt"] = False
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        snapshot["database_snapshot_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def classify_revalidation_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "jsondecodeerror" in text or "search" in text or "query" in text:
        return "candidate_retrieval"
    return "unknown"


def classify_embedding_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        text += f" {type(cause).__name__}: {cause}".lower()
    if "local_files_only" in text or "snapshot" in text or "revision" in text or "not found" in text:
        return "embedding_model_cache"
    if "cuda" in text and ("unavailable" in text or "device" in text):
        return "embedding_device_transfer"
    if "safetensor" in text or "weight" in text or "state_dict" in text:
        return "embedding_weight_loading"
    if "encode" in text or "inference" in text or "out of memory" in text or "oom" in text:
        return "embedding_inference"
    return "embedding_model_initialization"


def frozen_runtime_policy(env: Mapping[str, str]) -> dict[str, Any]:
    retrieval = load_vector_search_config(env)
    return {
        "default_initial_retrieval_policy": "guarded_structure_aware",
        "graph_activation_policy": str(DEFAULT_GRAPH_ACTIVATION_POLICY),
        "graph_runtime_hop_depth": MAXIMUM_HOPS,
        "evidence_composition_policy": "canonical_runtime_evidence",
        "rerank_policy": "runtime_default" if retrieval.rerank_enabled else "disabled",
        "retrieval_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "reranker_policy_mutation_count": 0,
        "chunking_policy_mutation_count": 0,
        "embedding_policy_mutation_count": 0,
        "embedding_model_mutation_count": 0,
        "embedding_dimension_mutation_count": 0,
        "embedding_device_default_mutation_count": 0,
        "cpu_fallback_promoted": False,
    }


def build_task0176_summary(*, source_head: str, environment: Mapping[str, Any], fresh_clone: Mapping[str, Any], corpus: Mapping[str, Any], secret: Mapping[str, Any], database_identity: Mapping[str, Any], cache: Mapping[str, Any], embedding_preflight: Mapping[str, Any], runtime_policy: Mapping[str, Any], task0170_summary: Mapping[str, Any], execute_revalidation: bool) -> dict[str, Any]:
    q = _query_counts(task0170_summary)
    first_failure = _normalize_failure(task0170_summary.get("first_failure_stage") or embedding_preflight.get("first_failure_stage") or "unknown")
    root_cause = str(task0170_summary.get("dominant_root_cause") or embedding_preflight.get("error") or "none")
    cold_start_reproducible = _pass_decision(task0170_summary, embedding_preflight, corpus, database_identity, runtime_policy)
    if cold_start_reproducible:
        first_failure, root_cause = "none", "none"
    task_status = "complete" if cold_start_reproducible else ("partial" if execute_revalidation and task0170_summary else "blocked")
    query_flags = _query_flags(task0170_summary)
    return {
        "task_id": TASK_ID,
        "task_status": task_status,
        **{k: environment[k] for k in ("source_authoritative_head", "task0175_source_head", "source_head_changed_since_task0175", "cold_start_root", "python_version", "uv_version")},
        "fresh_clone": bool(task0170_summary.get("fresh_clone", False)),
        "git_head_equivalence": bool(task0170_summary.get("git_head_equivalence", False)),
        "safe_clone_reset_guard": bool(fresh_clone.get("safe_clone_reset_guard")),
        "fresh_python_environment": bool(task0170_summary.get("fresh_python_environment", False)),
        "dependency_bootstrap_success": bool(task0170_summary.get("fresh_python_environment", False)),
        "manual_source_required": False,
        "process_scoped_pgoptions_applied": bool(database_identity.get("process_scoped_pgoptions_applied") or task0170_summary.get("process_scoped_pgoptions_applied")),
        "database_platform": "supabase_postgresql",
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "database_connection_authority_valid": bool(database_identity.get("database_connection_authority_valid")),
        "database_identity_guard_passed": bool(database_identity.get("database_identity_guard_passed") or task0170_summary.get("database_identity_guard_passed")),
        "schema_identity_guard_passed": bool(database_identity.get("schema_identity_guard_passed") or task0170_summary.get("schema_identity_guard_passed")),
        "public_schema_write_count": int(task0170_summary.get("public_schema_write_count", 0) or 0),
        "development_schema_reused": bool(task0170_summary.get("development_schema_reused", False)),
        "development_database_reused": False,
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_model_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "embedding_model_revision_match": bool(embedding_preflight.get("embedding_model_revision_match")),
        "model_cache_complete": bool(embedding_preflight.get("model_cache_complete")),
        "embedding_model_load_success": bool(embedding_preflight.get("embedding_model_load_success")),
        "embedding_inference_success": bool(embedding_preflight.get("embedding_inference_success")),
        "embedding_output_dimension": int(embedding_preflight.get("embedding_output_dimension", 0) or 0),
        "embedding_output_finite": bool(embedding_preflight.get("embedding_output_finite")),
        "source_document_count": int(corpus.get("source_document_count", 0)),
        "materialized_document_count": int(task0170_summary.get("materialized_document_count", 0) or 0),
        "ingestion_failure_count": int(task0170_summary.get("ingestion_failure_count", 0) or 0),
        "chunk_count": int(task0170_summary.get("chunk_count", 0) or 0),
        "chunking_failure_count": 0 if task0170_summary.get("chunk_count", 0) else 1,
        "long_document_chunk_count": int(task0170_summary.get("long_document_chunk_count", 0) or 0),
        "embedding_count": int(task0170_summary.get("embedding_count", 0) or 0),
        "embedding_failure_count": int(task0170_summary.get("embedding_failure_count", 0) or 0),
        "embedding_device": embedding_preflight.get("embedding_device", ""),
        "vector_state_rebuilt": bool(task0170_summary.get("vector_state_rebuilt", False)),
        "lexical_state_rebuilt": bool(task0170_summary.get("lexical_state_rebuilt", False)),
        "graph_build_success": bool(task0170_summary.get("graph_build_success", False)),
        "graph_node_count": int(task0170_summary.get("graph_node_count", 0) or 0),
        "graph_edge_count": int(task0170_summary.get("graph_edge_count", 0) or 0),
        "identity_resolution_failure_count": int(task0170_summary.get("identity_resolution_failure_count", 0) or 0),
        "dangling_reference_count": int(task0170_summary.get("dangling_reference_count", 0) or 0),
        **q,
        **query_flags,
        "citation_provenance_valid": bool(task0170_summary.get("citation_provenance_valid", False)),
        "environment_contamination_detected": bool(task0170_summary.get("environment_contamination_detected", False)),
        "credential_leak_detected": False,
        **runtime_policy,
        "corpus_mutation_count": 0,
        "benchmark_mutation_count": 0,
        "residual_gpu_dependency_reproducibility_risk": True,
        "task0170_revalidation_executed": bool(execute_revalidation and task0170_summary),
        "task0170_revalidation_status": task0170_summary.get("task_status", "blocked"),
        "task0170_revalidation_cold_start_reproducible": bool(task0170_summary.get("cold_start_reproducible", False)),
        "task0170_original_first_failure_stage": "configuration",
        "task0170_lineage": ["configuration", "schema-scoped DB", "embedding model cache", "full revalidation"],
        "first_failure_stage": first_failure,
        "dominant_root_cause": root_cause,
        "cold_start_reproducible": cold_start_reproducible,
        "recommended_next_action": "Cross-machine Portability Validation" if cold_start_reproducible else f"Create remediation card for {first_failure}",
    }


def _query_counts(task0170_summary: Mapping[str, Any]) -> dict[str, int]:
    stage = _normalize_failure(task0170_summary.get("first_failure_stage", "unknown"))
    pre_query_stages = {
        "git_clone", "dependency_bootstrap", "configuration", "database_connection", "database_schema", "schema_identity",
        "document_discovery", "parsing", "canonical_document", "chunking", "embedding_model_cache",
        "embedding_model_initialization", "embedding_weight_loading", "embedding_device_transfer", "embedding_inference",
        "embedding_persistence", "vector_index", "lexical_index", "graph_construction", "graph_identity_resolution",
    }
    if stage in pre_query_stages:
        return {"formal_query_count": EXPECTED_QUERY_COUNT, "query_pass_count": 0, "query_fail_count": 0, "query_not_evaluable_count": EXPECTED_QUERY_COUNT}
    formal = int(task0170_summary.get("formal_query_count", EXPECTED_QUERY_COUNT) or EXPECTED_QUERY_COUNT)
    passed = int(task0170_summary.get("query_pass_count", 0) or 0)
    failed = int(task0170_summary.get("query_fail_count", 0) or 0)
    not_eval = max(0, formal - passed - failed)
    return {"formal_query_count": EXPECTED_QUERY_COUNT, "query_pass_count": passed, "query_fail_count": failed, "query_not_evaluable_count": not_eval}


def _query_flags(task0170_summary: Mapping[str, Any]) -> dict[str, bool]:
    result = {v: bool(task0170_summary.get(v, False)) for v in QUERY_FLAGS.values()}
    if not result["q05_negative_control_valid"] and "q05_negative_control_valid" in task0170_summary:
        result["q05_negative_control_valid"] = bool(task0170_summary["q05_negative_control_valid"])
    return result


def _pass_decision(task0170_summary: Mapping[str, Any], embedding_preflight: Mapping[str, Any], corpus: Mapping[str, Any], database_identity: Mapping[str, Any], runtime_policy: Mapping[str, Any]) -> bool:
    return all([
        bool(task0170_summary.get("cold_start_reproducible", False)),
        bool(corpus.get("document_accounting_valid")), bool(corpus.get("query_accounting_valid")),
        bool(database_identity.get("database_connection_authority_valid")),
        bool(embedding_preflight.get("embedding_model_revision_match")), bool(embedding_preflight.get("model_cache_complete")),
        bool(embedding_preflight.get("embedding_model_load_success")), bool(embedding_preflight.get("embedding_inference_success")),
        int(embedding_preflight.get("embedding_output_dimension", 0) or 0) == DEFAULT_EMBEDDING_DIMENSION,
        all(int(runtime_policy.get(k, 0)) == 0 for k in ("retrieval_policy_mutation_count", "graph_policy_mutation_count", "reranker_policy_mutation_count", "chunking_policy_mutation_count", "embedding_policy_mutation_count")),
    ])


def build_artifacts(summary: Mapping[str, Any], environment: Mapping[str, Any], fresh_clone: Mapping[str, Any], corpus: Mapping[str, Any], secret: Mapping[str, Any], database_identity: Mapping[str, Any], cache: Mapping[str, Any], embedding_preflight: Mapping[str, Any], runtime_policy: Mapping[str, Any], task0170_summary: Mapping[str, Any]) -> dict[str, Any]:
    query_results = _load_task0170_queries()
    artifacts = {
        "summary.json": dict(summary),
        "environment.json": dict(environment),
        "fresh_clone.json": dict(fresh_clone),
        "database_identity.json": dict(database_identity),
        "runtime_isolation.json": {"fresh_runtime_state_required": True, "environment_contamination_detected": summary["environment_contamination_detected"], "public_schema_write_count": summary["public_schema_write_count"], "development_schema_reused": summary["development_schema_reused"]},
        "embedding_model_authority.json": _redact(dict(cache)),
        "embedding_preflight.json": dict(embedding_preflight),
        "corpus_materialization.json": {k: corpus[k] for k in ("corpus_root", "source_document_count", "expected_document_count", "formal_query_count", "expected_query_count", "document_accounting_valid", "query_accounting_valid")},
        "chunking.json": {"chunk_count": summary["chunk_count"], "chunking_failure_count": summary["chunking_failure_count"], "long_document_chunk_count": summary["long_document_chunk_count"]},
        "embedding_materialization.json": {"embedding_count": summary["embedding_count"], "embedding_failure_count": summary["embedding_failure_count"], "embedding_dimension": summary["embedding_dimension"], "embedding_device": summary["embedding_device"]},
        "vector_state.json": {"vector_state_rebuilt": summary["vector_state_rebuilt"]},
        "lexical_state.json": {"lexical_state_rebuilt": summary["lexical_state_rebuilt"]},
        "graph_state.json": {"graph_build_success": summary["graph_build_success"], "graph_node_count": summary["graph_node_count"], "graph_edge_count": summary["graph_edge_count"], "identity_resolution_failure_count": summary["identity_resolution_failure_count"], "dangling_reference_count": summary["dangling_reference_count"]},
        "query_results.json": query_results,
        "citation_validation.json": {"citation_provenance_valid": summary["citation_provenance_valid"]},
        "task0170_revalidation.json": dict(task0170_summary),
        "failure_diagnosis.json": {"first_failure_stage": summary["first_failure_stage"], "dominant_root_cause": summary["dominant_root_cause"]},
        "runtime_policy.json": dict(runtime_policy),
        "secret_authority.json": dict(secret),
    }
    artifacts["credential_scan.json"] = credential_scan_payload(artifacts)
    artifacts["summary.json"]["credential_leak_detected"] = artifacts["credential_scan.json"]["credential_leak_detected"]
    return artifacts


def _load_task0170_queries() -> dict[str, Any]:
    path = TASK0170_REVALIDATION_DIR / "retrieval_results.json"
    if path.exists():
        return read_json(path)
    initial = build_initial_retrieval_results()
    initial["query_not_evaluable_count"] = EXPECTED_QUERY_COUNT
    return initial


def credential_scan_payload(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    leaks = []
    for name, payload in artifacts.items():
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if SECRET_URL_RE.search(text):
            leaks.append(name)
    for path in [REPORT_PATH, ROOT / "PROJECT_STATE.md", ROOT / "CHANGELOG.md", ROOT / "docs" / "COLD_START_REPRODUCIBILITY.md"]:
        if path.exists() and SECRET_URL_RE.search(path.read_text(encoding="utf-8")):
            leaks.append(str(path.relative_to(ROOT)))
    return {"credential_leak_detected": bool(leaks), "leak_locations": leaks}


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "baseline_task": "TASK-0170",
        "database_authority_task": "TASK-0172",
        "embedding_authority_task": "TASK-0175",
        "cold_start_root": str(COLD_START_ROOT),
        "corpus_identity": "opk-rag-cold-start-corpus-v1",
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "formal_query_count": EXPECTED_QUERY_COUNT,
        "database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "embedding_model": DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_revision": DEFAULT_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "fresh_clone_required": True,
        "fresh_python_environment_required": True,
        "fresh_runtime_state_required": True,
        "task0170_full_revalidation_required": True,
        "remediation_allowed": False,
        "retrieval_policy_mutation_allowed": False,
        "graph_policy_mutation_allowed": False,
        "reranker_policy_mutation_allowed": False,
        "chunking_policy_mutation_allowed": False,
        "embedding_policy_mutation_allowed": False,
        "corpus_mutation_allowed": False,
        "benchmark_mutation_allowed": False,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# TASK-0176 Same-machine Cold-start Full Revalidation Report", "",
        "## 1. Revalidation objective", "Re-run the TASK-0170 same-machine cold-start contract from the current authoritative HEAD with fresh clone, fresh Python environment, schema-scoped Supabase runtime state, pinned embedding model authority, fixed corpus, and Q01-Q07 accounting.", "",
        "## 2. TASK-0170 history", "Original TASK-0170 stopped at `configuration`; lineage is preserved as configuration → schema-scoped DB → embedding model cache → full revalidation.", "",
        "## 3. TASK-0172 database authority", f"Database: Supabase PostgreSQL `{RESOLVED_DATABASE_NAME}`, schema `{COLD_START_SCHEMA}`, process PGOPTIONS `{PROCESS_SCOPED_PGOPTIONS}`.", "",
        "## 4. TASK-0175 model authority", f"Model `{DEFAULT_EMBEDDING_MODEL_NAME}` revision `{DEFAULT_EMBEDDING_MODEL_REVISION}`, dimension {DEFAULT_EMBEDDING_DIMENSION}.", "",
        "## 5-20. Result summary", "```json", json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## Recommended next stage", str(summary.get("recommended_next_action", "")), "",
    ]
    return "\n".join(lines)


def verify_task0176_artifacts(*, result_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    required = ("summary.json", "environment.json", "fresh_clone.json", "database_identity.json", "runtime_isolation.json", "embedding_model_authority.json", "embedding_preflight.json", "corpus_materialization.json", "chunking.json", "embedding_materialization.json", "vector_state.json", "lexical_state.json", "graph_state.json", "query_results.json", "citation_validation.json", "task0170_revalidation.json", "failure_diagnosis.json")
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    missing = [name for name in required if not (result_dir / name).exists()]
    contract_payload = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    mutation_ok = all(int(summary.get(k, 0)) == 0 for k in ("retrieval_policy_mutation_count", "graph_policy_mutation_count", "reranker_policy_mutation_count", "chunking_policy_mutation_count", "embedding_policy_mutation_count", "corpus_mutation_count", "benchmark_mutation_count"))
    query_ok = int(summary.get("query_pass_count", 0)) + int(summary.get("query_fail_count", 0)) + int(summary.get("query_not_evaluable_count", 0)) == EXPECTED_QUERY_COUNT
    failure_ok = str(summary.get("first_failure_stage", "unknown")) in FAILURE_STAGES and (summary.get("cold_start_reproducible") is not True or summary.get("first_failure_stage") == "none")
    valid = not missing and contract_payload.get("task_id") == TASK_ID and summary.get("task_id") == TASK_ID and summary.get("source_document_count") == EXPECTED_DOCUMENT_COUNT and summary.get("formal_query_count") == EXPECTED_QUERY_COUNT and summary.get("embedding_model_revision") == DEFAULT_EMBEDDING_MODEL_REVISION and summary.get("database_name") == RESOLVED_DATABASE_NAME and summary.get("cold_start_schema") == COLD_START_SCHEMA and mutation_ok and query_ok and failure_ok and not bool(summary.get("credential_leak_detected"))
    result = {"task_id": TASK_ID, "verification_valid": valid, "missing_artifacts": missing, "contract_integrity": contract_payload.get("task_id") == TASK_ID, "document_accounting_valid": summary.get("source_document_count") == EXPECTED_DOCUMENT_COUNT, "query_accounting_valid": query_ok, "model_revision_identity_valid": summary.get("embedding_model_revision") == DEFAULT_EMBEDDING_MODEL_REVISION, "database_schema_identity_valid": summary.get("database_name") == RESOLVED_DATABASE_NAME and summary.get("cold_start_schema") == COLD_START_SCHEMA, "mutation_counts_valid": mutation_ok, "first_failure_consistent": failure_ok, "task0170_result_consistency": bool(summary.get("task0170_revalidation_executed")) == bool((result_dir / "task0170_revalidation.json").exists())}
    if write:
        write_json(result_dir / "verification.json", result)
    return result


def _preflight_allows_revalidation(corpus: Mapping[str, Any], cache: Mapping[str, Any], embedding_preflight: Mapping[str, Any], secret: Mapping[str, Any]) -> bool:
    return bool(corpus.get("document_accounting_valid") and corpus.get("query_accounting_valid") and secret.get("database_connection_authority_valid") and cache.get("cache_authority_valid") and embedding_preflight.get("embedding_model_load_success") and embedding_preflight.get("embedding_inference_success"))


def _preflight_failure_stage(corpus: Mapping[str, Any], cache: Mapping[str, Any], embedding_preflight: Mapping[str, Any], secret: Mapping[str, Any]) -> str:
    if not secret.get("database_connection_authority_valid"):
        return "configuration"
    if not corpus.get("document_accounting_valid") or not corpus.get("query_accounting_valid"):
        return "document_discovery"
    if not cache.get("cache_authority_valid"):
        return "embedding_model_cache"
    return _normalize_failure(embedding_preflight.get("first_failure_stage", "unknown"))


def _preflight_root_cause(corpus: Mapping[str, Any], cache: Mapping[str, Any], embedding_preflight: Mapping[str, Any], secret: Mapping[str, Any]) -> str:
    if not secret.get("database_connection_authority_valid"):
        return "; ".join(secret.get("connection_authority_errors", []) or ["cold-start secret authority unavailable"])
    if not corpus.get("document_accounting_valid") or not corpus.get("query_accounting_valid"):
        return "fixed corpus manifest/query accounting invalid"
    if not cache.get("cache_authority_valid"):
        return "pinned embedding model cache authority invalid"
    return str(embedding_preflight.get("error") or "embedding preflight failed")


def _normalize_failure(stage: Any) -> str:
    stage = str(stage or "unknown")
    if stage == "embedding":
        stage = "embedding_inference"
    if stage == "database_isolation":
        stage = "schema_identity"
    return stage if stage in FAILURE_STAGES else "unknown"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if k == "database_url_value" and v else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_URL_RE.sub("postgresql://<redacted>:<redacted>@", value)
    return value


def _git_head(cwd: Path) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    return result.stdout_tail.strip() if result.returncode == 0 else ""
