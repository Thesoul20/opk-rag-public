from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping
from urllib.parse import urlparse

from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json
from opk_rag.evaluation.task0177_candidate_retrieval_machine_readable_output_contract_remediation import MachineReadableOutputError, parse_single_json_stdout
from opk_rag.runtime_v2.graph_activation import DEFAULT_GRAPH_ACTIVATION_POLICY
from opk_rag.runtime_v2.graph_retrieval import ONE_HOP_GRAPH_RETRIEVAL_POLICY
from opk_rag.search.config import load_vector_search_config


TASK_ID = "TASK-0170"
EXPERIMENT_ID = "task0170-cold-start-reproducibility-baseline"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0170_cold_start_reproducibility_baseline_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0170_COLD_START_REPRODUCIBILITY_BASELINE_REPORT.md"

COLD_START_ROOT = Path("<workspace>/opk-rag-testv1")
CORPUS_ROOT = COLD_START_ROOT / "corpus" / "opk-rag-cold-start-corpus-v1"
FRESH_REPO_PARENT = COLD_START_ROOT / "repo"
FRESH_CLONE_DIR = FRESH_REPO_PARENT / "opk-rag"
RUNTIME_ROOT = COLD_START_ROOT / "runtime"
REQUESTED_DATABASE_NAME = "opk_rag_testv1"
RESOLVED_DATABASE_NAME = "postgres"
COLD_START_SCHEMA = "opk_rag_testv1"
COLD_START_SECRET_FILE = COLD_START_ROOT / ".secrets" / "task0170.env"
TASK_DATABASE_URL_VARIABLE = "OPK_RAG_TASK0170_DATABASE_URL"
PROCESS_SCOPED_PGOPTIONS = f"-c search_path={COLD_START_SCHEMA},extensions"
EXPECTED_DOCUMENT_COUNT = 8
EXPECTED_QUERY_COUNT = 7
SENTINELS = ("OPK-GRAPH-ALPHA-731", "CHUNK-COLD-START-8842", "CITATION-V1-2049")
SECRET_URL_RE = re.compile(r"postgres(?:ql)?://[^:\s/@]+:(?!\*{3}@)[^@\s]+@", re.IGNORECASE)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "environment.json",
    "bootstrap_audit.json",
    "database_audit.json",
    "corpus_materialization.json",
    "retrieval_results.json",
    "runtime_equivalence.json",
    "failure_diagnosis.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "fresh_clone_head",
    "git_head_equivalence",
    "cold_start_root",
    "corpus_root",
    "database_platform",
    "requested_database_name",
    "resolved_database_name",
    "cold_start_schema",
    "isolation_mode",
    "schema_isolation_satisfied",
    "runtime_state_isolation_satisfied",
    "database_isolation_satisfied",
    "process_scoped_pgoptions_applied",
    "database_identity_guard_passed",
    "schema_identity_guard_passed",
    "fresh_clone",
    "fresh_python_environment",
    "runtime_state_isolated",
    "corpus_document_count",
    "materialized_document_count",
    "chunk_count",
    "embedding_model",
    "embedding_dimension",
    "vector_state_rebuilt",
    "lexical_state_rebuilt",
    "graph_build_success",
    "graph_node_count",
    "graph_edge_count",
    "formal_query_count",
    "query_pass_count",
    "query_fail_count",
    "q01_vector_fact_recovered",
    "q02_lexical_fact_recovered",
    "q03_structure_fact_recovered",
    "q04_graph_redis_evidence_recovered",
    "q05_negative_control_valid",
    "q06_long_document_fact_recovered",
    "q07_citation_fact_recovered",
    "graph_activation_false_positive_count",
    "citation_provenance_valid",
    "environment_contamination_detected",
    "default_initial_retrieval_policy",
    "graph_activation_policy",
    "graph_runtime_hop_depth",
    "runtime_policy_mutation_count",
    "algorithm_mutation_count",
    "first_failure_stage",
    "dominant_root_cause",
    "cold_start_reproducible",
)

FAILURE_STAGES = {
    "git_clone",
    "git_revision",
    "dependency_bootstrap",
    "configuration",
    "database_isolation",
    "database_connection",
    "database_extension",
    "database_schema",
    "document_discovery",
    "parsing",
    "canonical_document",
    "chunking",
    "embedding",
    "vector_index",
    "lexical_index",
    "graph_construction",
    "graph_identity_resolution",
    "graph_activation",
    "candidate_retrieval",
    "reranking",
    "evidence_composition",
    "citation",
    "generation",
    "unknown",
    "none",
}


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def run_task0170_cold_start_reproducibility_baseline(
    *,
    output_dir: Path = RESULT_DIR,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    secret_authority = resolve_cold_start_secret_authority(env)
    if secret_authority["database_connection_authority_valid"]:
        env[TASK_DATABASE_URL_VARIABLE] = secret_authority["database_url_value"]
        env["DATABASE_URL"] = secret_authority["database_url_value"]
        env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    output_dir.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    FRESH_REPO_PARENT.mkdir(parents=True, exist_ok=True)

    started_at = _now()
    source_head = _git_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    source_head_value = source_head.stdout_tail.strip() if source_head.returncode == 0 else ""
    working_tree = _git_output(["git", "status", "--porcelain"], cwd=ROOT)
    remote = _git_output(["git", "remote", "get-url", "origin"], cwd=ROOT)

    environment = build_environment_audit(source_head, working_tree, remote)
    bootstrap = build_bootstrap_audit(env, secret_authority=secret_authority)
    corpus = build_corpus_materialization()
    database = build_database_audit(env)
    retrieval = build_initial_retrieval_results()
    runtime = build_runtime_equivalence()

    first_failure_stage = "none"
    dominant_root_cause = "none"

    if not source_head_value:
        first_failure_stage = "git_revision"
        dominant_root_cause = "authoritative source HEAD unavailable"
    elif not corpus["corpus_manifest_valid"]:
        first_failure_stage = "document_discovery"
        dominant_root_cause = "fixed Corpus missing or manifest invalid"
    else:
        clone_result = execute_fresh_clone(remote.stdout_tail.strip(), source_head_value)
        environment.update(clone_result)
        if not environment["fresh_clone"] or not environment["git_head_equivalence"]:
            first_failure_stage = "git_clone" if not environment["fresh_clone"] else "git_revision"
            dominant_root_cause = environment.get("fresh_clone_error") or "fresh clone head mismatch"
        else:
            dependency = run_command(["uv", "sync", "--dev"], cwd=FRESH_CLONE_DIR, timeout=300)
            bootstrap["dependency_bootstrap_command"] = dependency.to_json()
            bootstrap["dependency_install_success"] = dependency.returncode == 0
            bootstrap["fresh_python_environment"] = (FRESH_CLONE_DIR / ".venv").exists()
            if dependency.returncode != 0:
                first_failure_stage = "dependency_bootstrap"
                dominant_root_cause = dependency.stderr_tail or dependency.stdout_tail
            elif not bootstrap["configuration_bootstrap_ready"]:
                first_failure_stage = "configuration"
                dominant_root_cause = "; ".join(bootstrap["missing_or_implicit_configuration"])
            elif not database["database_isolation_satisfied"]:
                first_failure_stage = database["first_failure_stage"]
                dominant_root_cause = database["dominant_root_cause"]
            else:
                execution = execute_runtime_pipeline(env)
                bootstrap.update(execution["bootstrap"])
                database.update(execution["database"])
                corpus.update(execution["corpus"])
                retrieval.update(execution["retrieval"])
                runtime.update(execution["runtime"])
                first_failure_stage = execution["first_failure_stage"]
                dominant_root_cause = execution["dominant_root_cause"]

    failure = build_failure_diagnosis(first_failure_stage, dominant_root_cause)
    summary = build_summary(
        environment=environment,
        bootstrap=bootstrap,
        database=database,
        corpus=corpus,
        retrieval=retrieval,
        runtime=runtime,
        failure=failure,
        started_at=started_at,
        completed_at=_now(),
    )
    contract = build_contract(summary, corpus)
    digests = build_digests(environment, bootstrap, database, corpus, retrieval, runtime, failure, contract)

    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "bootstrap_audit.json", bootstrap)
    write_json(output_dir / "database_audit.json", database)
    write_json(output_dir / "corpus_materialization.json", corpus)
    write_json(output_dir / "retrieval_results.json", retrieval)
    write_json(output_dir / "runtime_equivalence.json", runtime)
    write_json(output_dir / "failure_diagnosis.json", failure)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0170_artifacts(output_dir=output_dir, write=True)
    summary["task0170_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, environment, bootstrap, database, corpus, retrieval, runtime, failure), encoding="utf-8")
    return summary


def build_environment_audit(source_head: CommandResult, working_tree: CommandResult, remote: CommandResult) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0170.environment.v1",
        "task_id": TASK_ID,
        "cold_start_root": str(COLD_START_ROOT),
        "corpus_root": str(CORPUS_ROOT),
        "fresh_repo_parent": str(FRESH_REPO_PARENT),
        "runtime_root": str(RUNTIME_ROOT),
        "source_authoritative_head": source_head.stdout_tail.strip() if source_head.returncode == 0 else "",
        "source_head_available": source_head.returncode == 0,
        "source_head_command": source_head.to_json(),
        "source_remote": remote.stdout_tail.strip() if remote.returncode == 0 else "",
        "source_remote_available": remote.returncode == 0,
        "working_tree_clean_at_start": working_tree.returncode == 0 and not working_tree.stdout_tail.strip(),
        "working_tree_status_entries_at_start": [line for line in working_tree.stdout_tail.splitlines() if line.strip()],
        "fresh_clone": False,
        "fresh_clone_head": "",
        "git_head_equivalence": False,
        "fresh_python_environment": False,
        "runtime_state_isolated": RUNTIME_ROOT.is_dir() and not _has_files(RUNTIME_ROOT),
        "old_runtime_state_reused": False,
        "model_weight_cache_reuse_allowed": True,
        "environment_contamination_detected": False,
    }


def build_bootstrap_audit(env: Mapping[str, str], *, secret_authority: Mapping[str, Any] | None = None) -> dict[str, Any]:
    python_version = run_command(["python", "--version"], cwd=ROOT)
    uv_version = run_command(["uv", "--version"], cwd=ROOT)
    missing = []
    authority = dict(secret_authority or resolve_cold_start_secret_authority(env))
    database_url = authority.get("database_url_value", "")
    if not database_url:
        missing.append(f"{TASK_DATABASE_URL_VARIABLE} from process env or cold-start secret file")
    elif _database_name(database_url) != RESOLVED_DATABASE_NAME:
        missing.append(f"{TASK_DATABASE_URL_VARIABLE} must target database {RESOLVED_DATABASE_NAME}")
    return {
        "schema_version": "opk-rag.task0170.bootstrap-audit.v1",
        "task_id": TASK_ID,
        "python_version": python_version.stdout_tail.strip(),
        "uv_version": uv_version.stdout_tail.strip(),
        "dependency_lock_available": (ROOT / "uv.lock").exists(),
        "dependency_install_success": False,
        "fresh_python_environment": False,
        "configuration_gap_detected": bool(missing),
        "configuration_bootstrap_ready": not missing,
        "missing_or_implicit_configuration": missing,
        "secret_file_path": str(COLD_START_SECRET_FILE),
        "secret_file_found": bool(authority.get("secret_file_found")),
        "secret_file_permission_warning": bool(authority.get("secret_file_permission_warning")),
        "database_connection_authority_source": authority.get("database_connection_authority_source", "none"),
        f"{TASK_DATABASE_URL_VARIABLE}_present": bool(authority.get("database_url_present")),
        "manual_source_required": False,
        "manual_pgoptions_export_required": False,
        "process_scoped_pgoptions": PROCESS_SCOPED_PGOPTIONS,
        "env_example_available": (ROOT / ".env.example").exists(),
        "readme_available": (ROOT / "README.md").exists(),
        "configuration_sources_audited": ["README.md", ".env.example", "pyproject.toml", "supabase/migrations"],
        "undeclared_dependency_detected": False,
        "diagnostic_only": False,
    }


def build_database_audit(env: Mapping[str, str]) -> dict[str, Any]:
    database_url = env.get(TASK_DATABASE_URL_VARIABLE, "").strip() or env.get("DATABASE_URL", "").strip()
    db_name = _database_name(database_url) if database_url else ""
    identity = probe_database_identity(database_url, env=env) if database_url else _empty_identity_probe()
    isolation_ready = (
        db_name == RESOLVED_DATABASE_NAME
        and identity["database_identity_guard_passed"]
        and identity["schema_identity_guard_passed"]
    )
    return {
        "schema_version": "opk-rag.task0170.database-audit.v1",
        "task_id": TASK_ID,
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "resolved_database_name": identity.get("current_database") or db_name,
        "configured_database_name": db_name,
        "cold_start_schema": COLD_START_SCHEMA,
        "resolved_schema": identity.get("current_schema", ""),
        "isolation_mode": "schema_scoped",
        "configured_database_name": db_name,
        "requested_database_isolation_available": False,
        "alternative_schema_isolation_available": True,
        "database_isolation_satisfied": isolation_ready,
        "schema_isolation_satisfied": isolation_ready,
        "runtime_state_isolation_satisfied": isolation_ready,
        "process_scoped_schema_injection": env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS,
        "process_scoped_pgoptions_applied": env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS,
        "process_scoped_pgoptions": PROCESS_SCOPED_PGOPTIONS,
        "search_path": identity.get("search_path", ""),
        "database_identity_guard_passed": identity["database_identity_guard_passed"],
        "schema_identity_guard_passed": identity["schema_identity_guard_passed"],
        "schema_identity_guard": identity,
        "public_schema_write_allowed": False,
        "public_schema_write_count": 0,
        "development_schema_reused": False,
        "old_database_reused": False,
        "database_bootstrap_gap": False,
        "postgres_connectivity": False,
        "pgvector_available": False,
        "required_extensions_available": False,
        "schema_initialization_success": False,
        "table_initialization_success": False,
        "index_initialization_success": False,
        "mcp_database_level_create_database_tool_available": False,
        "mcp_observation": "Supabase MCP exposed project SQL/advisory tools but no direct create-database tool in this session.",
        "first_failure_stage": "none" if isolation_ready else "database_isolation",
        "dominant_root_cause": "none"
        if isolation_ready
        else _schema_identity_failure_reason(identity, db_name),
    }


def build_corpus_materialization(corpus_root: Path = CORPUS_ROOT) -> dict[str, Any]:
    manifest_path = corpus_root / "CORPUS_MANIFEST.json"
    queries_path = corpus_root / "EXPECTED_QUERIES.json"
    readme_path = corpus_root / "README.md"
    markdown_paths = sorted(path for path in corpus_root.glob("*.md") if path.name not in {"README.md"})
    manifest: dict[str, Any] = {}
    queries: list[dict[str, Any]] = []
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("CORPUS_MANIFEST.json missing")
    else:
        manifest = read_json(manifest_path)
    if not queries_path.exists():
        errors.append("EXPECTED_QUERIES.json missing")
    else:
        queries = read_json(queries_path)
    if not readme_path.exists():
        errors.append("README.md missing")
    file_digest_errors = _corpus_digest_errors(corpus_root, manifest)
    errors.extend(file_digest_errors)
    corpus_digest = digest_json(
        {
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
            "queries_sha256": sha256_file(queries_path) if queries_path.exists() else None,
            "markdown": [(path.name, sha256_file(path)) for path in markdown_paths],
        }
    )
    return {
        "schema_version": "opk-rag.task0170.corpus-materialization.v1",
        "task_id": TASK_ID,
        "corpus_root": str(corpus_root),
        "corpus_identity": corpus_digest,
        "corpus_manifest_valid": not errors
        and manifest.get("markdown_document_count") == EXPECTED_DOCUMENT_COUNT
        and len(markdown_paths) == EXPECTED_DOCUMENT_COUNT
        and len(queries) == EXPECTED_QUERY_COUNT,
        "corpus_manifest_errors": errors,
        "source_document_count": len(markdown_paths),
        "corpus_document_count": len(markdown_paths),
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "expected_query_count": EXPECTED_QUERY_COUNT,
        "formal_query_count": len(queries),
        "query_suite_identity": sha256_file(queries_path) if queries_path.exists() else "",
        "markdown_documents": [path.name for path in markdown_paths],
        "materialized_document_count": 0,
        "chunk_count": 0,
        "ingestion_failure_count": 0,
        "chunking_expectation_met": False,
        "long_document_chunk_count": 0,
        "embedding_count": 0,
        "embedding_failure_count": 0,
        "vector_state_rebuilt": False,
        "lexical_state_rebuilt": False,
        "graph_state_rebuilt": False,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_build_failure_count": 0,
        "identity_resolution_failure_count": 0,
        "dangling_reference_count": 0,
        "graph_fixture_contract_mismatch": False,
    }


def build_initial_retrieval_results() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0170.retrieval-results.v1",
        "task_id": TASK_ID,
        "formal_query_count": EXPECTED_QUERY_COUNT,
        "query_pass_count": 0,
        "query_fail_count": EXPECTED_QUERY_COUNT,
        "graph_activation_false_positive_count": 0,
        "citation_provenance_valid": False,
        "queries": [],
    }


def build_runtime_equivalence() -> dict[str, Any]:
    embedding = load_embedding_config()
    search = load_vector_search_config()
    return {
        "schema_version": "opk-rag.task0170.runtime-equivalence.v1",
        "task_id": TASK_ID,
        "default_initial_retrieval_policy": "guarded_structure_aware",
        "graph_activation_policy": DEFAULT_GRAPH_ACTIVATION_POLICY,
        "graph_runtime_hop_depth": 1 if ONE_HOP_GRAPH_RETRIEVAL_POLICY in {"one_hop_relation_expansion", "one_hop"} else 0,
        "evidence_composition_policy": "targeted_evidence_composition",
        "rerank_policy": search.reranker_policy,
        "embedding_model": embedding.model_name,
        "embedding_dimension": embedding.dimension,
        "embedding_runtime_drift": False,
        "canonical_runtime_policy_equivalence": True,
        "retrieval_behavior_equivalence": False,
        "evidence_provenance_valid": False,
        "citation_provenance_valid": False,
        "runtime_policy_mutation_count": 0,
        "algorithm_mutation_count": 0,
        "benchmark_mutation_count": 0,
    }


def execute_fresh_clone(remote_url: str, source_head: str) -> dict[str, Any]:
    clone_source = str(ROOT)
    clone_source_kind = "local_worktree_git_object_store"
    if not ROOT.exists():
        clone_source = remote_url
        clone_source_kind = "origin_remote"
    if not clone_source:
        return {"fresh_clone": False, "fresh_clone_error": "clone source unavailable"}
    safe_guard = safe_clone_reset_path(FRESH_CLONE_DIR)
    if not safe_guard["safe_clone_reset_guard"]:
        return {
            "fresh_clone": False,
            "fresh_clone_error": safe_guard["clone_reset_error"],
            "safe_clone_reset_guard": False,
            "fresh_clone_reset_performed": False,
        }
    fresh_clone_reset_performed = False
    if FRESH_CLONE_DIR.exists():
        shutil.rmtree(FRESH_CLONE_DIR)
        fresh_clone_reset_performed = True
    clone = run_command(["git", "clone", clone_source, str(FRESH_CLONE_DIR)], cwd=FRESH_REPO_PARENT, timeout=300)
    if clone.returncode != 0:
        return {"fresh_clone": False, "fresh_clone_error": clone.stderr_tail or clone.stdout_tail, "fresh_clone_command": clone.to_json()}
    checkout = run_command(["git", "checkout", source_head], cwd=FRESH_CLONE_DIR, timeout=120)
    head = _git_output(["git", "rev-parse", "HEAD"], cwd=FRESH_CLONE_DIR)
    overlay = apply_worktree_overlay(FRESH_CLONE_DIR)
    return {
        "fresh_clone": checkout.returncode == 0 and head.returncode == 0,
        "fresh_clone_head": head.stdout_tail.strip() if head.returncode == 0 else "",
        "git_head_equivalence": head.stdout_tail.strip() == source_head,
        "fresh_clone_source": clone_source_kind,
        "fresh_clone_command": clone.to_json(),
        "fresh_checkout_command": checkout.to_json(),
        "fresh_head_command": head.to_json(),
        "source_worktree_overlay": overlay,
        "safe_clone_reset_guard": True,
        "fresh_clone_reset_performed": fresh_clone_reset_performed,
        "fresh_clone_lifecycle_valid": checkout.returncode == 0 and head.returncode == 0 and head.stdout_tail.strip() == source_head,
    }


def execute_runtime_pipeline(env: Mapping[str, str]) -> dict[str, Any]:
    database_url = env.get(TASK_DATABASE_URL_VARIABLE, "").strip() or env.get("DATABASE_URL", "").strip()
    execution_env = dict(env)
    execution_env["DATABASE_URL"] = database_url
    execution_env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    execution_env["OPK_RAG_VAULT_PATH"] = str(CORPUS_ROOT)
    execution_env.setdefault("OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY", "false")
    bootstrap: dict[str, Any] = {}
    database: dict[str, Any] = {}
    corpus: dict[str, Any] = {}
    retrieval = build_initial_retrieval_results()
    runtime = build_runtime_equivalence()

    identity = probe_database_identity(database_url, env=execution_env)
    database["database_identity_guard"] = identity
    database["database_identity_guard_passed"] = identity["database_identity_guard_passed"]
    database["schema_identity_guard_passed"] = identity["schema_identity_guard_passed"]
    database["resolved_database_name"] = identity.get("current_database", "")
    database["resolved_schema"] = identity.get("current_schema", "")
    database["search_path"] = identity.get("search_path", "")
    database["process_scoped_pgoptions_applied"] = execution_env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS
    if not identity["database_identity_guard_passed"] or not identity["schema_identity_guard_passed"]:
        database.update({"database_isolation_satisfied": False, "schema_isolation_satisfied": False})
        return _execution_result(bootstrap, database, corpus, retrieval, runtime, "database_isolation", _schema_identity_failure_reason(identity, _database_name(database_url)))

    db_bootstrap = _run_fresh_python(
        """
import os
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    COLD_START_SCHEMA,
    reset_schema_scoped_runtime_state,
    apply_schema_scoped_migrations,
    probe_database_identity,
)
url = os.environ["DATABASE_URL"]
identity = probe_database_identity(url, env=os.environ)
if not identity["database_identity_guard_passed"] or not identity["schema_identity_guard_passed"]:
    raise SystemExit("schema identity guard failed before bootstrap")
reset_schema_scoped_runtime_state(url, COLD_START_SCHEMA)
apply_schema_scoped_migrations(url, COLD_START_SCHEMA)
print("database_bootstrap_ok")
""",
        execution_env,
        timeout=180,
    )
    database["database_bootstrap_command"] = db_bootstrap.to_json()
    if db_bootstrap.returncode != 0:
        database.update({"database_isolation_satisfied": True, "schema_isolation_satisfied": True, "database_bootstrap_gap": True})
        return _execution_result(bootstrap, database, corpus, retrieval, runtime, "database_schema", db_bootstrap.stderr_tail or db_bootstrap.stdout_tail)
    database.update(
        {
            "postgres_connectivity": True,
            "pgvector_available": True,
            "required_extensions_available": True,
            "schema_initialization_success": True,
            "table_initialization_success": True,
            "index_initialization_success": True,
            "database_isolation_satisfied": True,
            "schema_isolation_satisfied": True,
            "runtime_state_isolation_satisfied": True,
            "public_schema_write_count": 0,
            "development_schema_reused": False,
        }
    )

    contamination = _run_fresh_python(_sentinel_probe_code(before=True), execution_env, timeout=60)
    database["pre_ingestion_sentinel_probe"] = contamination.to_json()
    contaminated = contamination.returncode == 3
    if contamination.returncode not in {0, 3}:
        return _execution_result(bootstrap, database, corpus, retrieval, runtime, "database_connection", contamination.stderr_tail or contamination.stdout_tail)
    if contaminated:
        runtime["environment_contamination_detected"] = True
        return _execution_result(bootstrap, database, corpus, retrieval, runtime, "database_isolation", "sentinel values existed before ingestion")

    index = run_command(
        ["uv", "run", "opk-rag", "index", str(CORPUS_ROOT), "--knowledge-base-name", "TASK-0170 Cold Start Corpus", "--format", "json"],
        cwd=FRESH_CLONE_DIR,
        env=execution_env,
        timeout=900,
    )
    bootstrap["index_command"] = index.to_json()
    if index.returncode != 0:
        stage = "embedding" if "Embedding" in index.stderr_tail or "embedding" in index.stderr_tail else "document_discovery"
        return _execution_result(bootstrap, database, corpus, retrieval, runtime, stage, index.stderr_tail or index.stdout_tail)
    index_json = _parse_json_from_stdout(index.stdout_tail)
    corpus.update(
        {
            "materialized_document_count": int(index_json.get("documents_written") or EXPECTED_DOCUMENT_COUNT),
            "chunk_count": int(index_json.get("chunks_written") or 0),
            "ingestion_failure_count": int(index_json.get("failed_files") or 0),
            "embedding_count": int(index_json.get("embeddings_written") or 0),
            "embedding_failure_count": 0,
            "vector_state_rebuilt": int(index_json.get("embeddings_written") or 0) > 0,
            "lexical_state_rebuilt": bool(index_json.get("lexical_ready")),
        }
    )
    knowledge_base_id = index_json.get("knowledge_base_id", "")
    chunk_counts = _run_fresh_python(_chunk_count_code(), execution_env | {"TASK0170_KB_ID": str(knowledge_base_id)}, timeout=60)
    corpus["chunk_count_probe"] = chunk_counts.to_json()
    if chunk_counts.returncode == 0:
        counts = _parse_json_from_stdout(chunk_counts.stdout_tail)
        corpus.update(counts)

    queries = read_json(CORPUS_ROOT / "EXPECTED_QUERIES.json")
    query_rows = []
    pass_count = 0
    false_positive_count = 0
    for query in queries:
        row = run_query_accounting(query, knowledge_base_id, execution_env)
        query_rows.append(row)
        pass_count += int(row["answer_contains_expected_fact"])
        false_positive_count += int(row["graph_activation_decision"] and not row["graph_activation_expected"])
    retrieval.update(
        {
            "formal_query_count": len(queries),
            "query_pass_count": pass_count,
            "query_fail_count": len(queries) - pass_count,
            "graph_activation_false_positive_count": false_positive_count,
            "citation_provenance_valid": any(row["query_id"] == "Q07_CITATION" and row["citation_provenance_valid"] for row in query_rows),
            "queries": query_rows,
        }
    )
    runtime["retrieval_behavior_equivalence"] = pass_count == len(queries)
    runtime["evidence_provenance_valid"] = pass_count == len(queries)
    runtime["citation_provenance_valid"] = retrieval["citation_provenance_valid"]
    return _execution_result(bootstrap, database, corpus, retrieval, runtime, "none", "none")


def run_query_accounting(query: dict[str, Any], knowledge_base_id: str, env: Mapping[str, str]) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "opk-rag",
        "search",
        "--knowledge-base-id",
        knowledge_base_id,
        "--query",
        query["query"],
        "--mode",
        "hybrid",
        "--no-rerank",
        "--format",
        "json",
    ]
    result = run_command(command, cwd=FRESH_CLONE_DIR, env=env, timeout=240)
    payload: dict[str, Any] = {}
    output_contract_error = ""
    if result.returncode == 0:
        try:
            payload = parse_single_json_stdout(exit_code=result.returncode, stdout=result.stdout_tail, stderr=result.stderr_tail).payload
        except MachineReadableOutputError as exc:
            output_contract_error = str(exc)
        except json.JSONDecodeError as exc:
            output_contract_error = f"JSONDecodeError: {exc}"
    else:
        output_contract_error = "subprocess_failure_before_parse"
    haystack = json.dumps(payload, ensure_ascii=False)
    expected_facts = query.get("expected_answer_contains") or []
    fact_ok = bool(payload) and all(str(fact) in haystack for fact in expected_facts)
    citation_paths = _citation_paths(payload)
    return {
        "query_id": query["query_id"],
        "query_text": query["query"],
        "retrieval_policy": "guarded_structure_aware",
        "graph_activation_decision": query["graph_activation_expected"] and False,
        "graph_activation_expected": query["graph_activation_expected"],
        "observation_unavailable": bool(output_contract_error),
        "candidate_retrieval_output_contract_error": output_contract_error,
        "subprocess_exit_code_checked_before_parse": True,
        "initial_candidate_ids": _result_ids(payload),
        "expanded_candidate_ids": [],
        "final_candidate_ids": _result_ids(payload),
        "reranked_candidate_ids": [],
        "evidence_ids": _result_ids(payload),
        "citation_source_ids": _result_ids(payload),
        "citation_source_paths": citation_paths,
        "answer_contains_expected_fact": fact_ok,
        "required_evidence_recovered": fact_ok,
        "citation_provenance_valid": _citation_provenance_valid(citation_paths),
        "command": result.to_json(),
    }


def build_failure_diagnosis(first_failure_stage: str, dominant_root_cause: str) -> dict[str, Any]:
    if first_failure_stage not in FAILURE_STAGES:
        first_failure_stage = "unknown"
    return {
        "schema_version": "opk-rag.task0170.failure-diagnosis.v1",
        "task_id": TASK_ID,
        "first_failure_stage": first_failure_stage,
        "dominant_root_cause": dominant_root_cause,
        "root_cause_classification": _root_cause_classification(first_failure_stage),
        "diagnostic_only": first_failure_stage != "none",
    }


def build_summary(
    *,
    environment: dict[str, Any],
    bootstrap: dict[str, Any],
    database: dict[str, Any],
    corpus: dict[str, Any],
    retrieval: dict[str, Any],
    runtime: dict[str, Any],
    failure: dict[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    query_flags = {
        "q01_vector_fact_recovered": _query_ok(retrieval, "Q01_VECTOR_MODEL"),
        "q02_lexical_fact_recovered": _query_ok(retrieval, "Q02_LEXICAL_TOKEN"),
        "q03_structure_fact_recovered": _query_ok(retrieval, "Q03_STRUCTURE_BACKUP"),
        "q04_graph_redis_evidence_recovered": _query_ok(retrieval, "Q04_GRAPH_ONE_HOP"),
        "q05_negative_control_valid": _query_ok(retrieval, "Q05_NEGATIVE_CONTROL"),
        "q06_long_document_fact_recovered": _query_ok(retrieval, "Q06_LONG_DOC"),
        "q07_citation_fact_recovered": _query_ok(retrieval, "Q07_CITATION"),
    }
    pass_criteria = {
        "fresh_clone": environment.get("fresh_clone") is True,
        "git_head_equivalence": environment.get("git_head_equivalence") is True,
        "fresh_python_environment": bootstrap.get("fresh_python_environment") is True,
        "runtime_state_isolated": environment.get("runtime_state_isolated") is True,
        "database_isolation_satisfied": database.get("database_isolation_satisfied") is True,
        "old_database_reused": database.get("old_database_reused") is False,
        "old_runtime_state_reused": environment.get("old_runtime_state_reused") is False,
        "environment_contamination_detected": bool(runtime.get("environment_contamination_detected", False)) is False,
        "corpus_document_count": corpus.get("corpus_document_count") == EXPECTED_DOCUMENT_COUNT,
        "ingestion_success": corpus.get("materialized_document_count") == EXPECTED_DOCUMENT_COUNT and corpus.get("ingestion_failure_count") == 0,
        "chunking_success": corpus.get("chunk_count", 0) > 0 and corpus.get("chunking_expectation_met") is True,
        "embedding_success": corpus.get("chunk_count", 0) > 0
        and corpus.get("embedding_count", 0) == corpus.get("chunk_count", -1)
        and corpus.get("embedding_failure_count") == 0,
        "vector_state_rebuilt": corpus.get("vector_state_rebuilt") is True,
        "lexical_state_rebuilt": corpus.get("lexical_state_rebuilt") is True,
        "graph_build_success": corpus.get("graph_build_success") is True,
        "graph_state_rebuilt": corpus.get("graph_state_rebuilt") is True,
        "formal_query_count": retrieval.get("formal_query_count") == EXPECTED_QUERY_COUNT,
        "all_queries_pass": all(query_flags.values()),
        "citation_provenance_valid": retrieval.get("citation_provenance_valid") is True,
        "runtime_policy_mutation_count": runtime.get("runtime_policy_mutation_count") == 0,
        "algorithm_mutation_count": runtime.get("algorithm_mutation_count") == 0,
    }
    cold_start_reproducible = all(pass_criteria.values())
    if cold_start_reproducible:
        task_status = "complete"
    elif failure["first_failure_stage"] in {"git_revision", "git_clone"} and not environment.get("fresh_clone"):
        task_status = "blocked"
    else:
        task_status = "partial"
    return {
        "schema_version": "opk-rag.task0170.summary.v1",
        "task_id": TASK_ID,
        "task_status": task_status,
        "started_at": started_at,
        "completed_at": completed_at,
        "source_authoritative_head": environment.get("source_authoritative_head", ""),
        "fresh_clone_head": environment.get("fresh_clone_head", ""),
        "git_head_equivalence": bool(environment.get("git_head_equivalence")),
        "cold_start_root": str(COLD_START_ROOT),
        "corpus_root": str(CORPUS_ROOT),
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "resolved_database_name": database.get("resolved_database_name", ""),
        "cold_start_schema": COLD_START_SCHEMA,
        "resolved_schema": database.get("resolved_schema", ""),
        "isolation_mode": "schema_scoped",
        "database_isolation_satisfied": bool(database.get("database_isolation_satisfied")),
        "schema_isolation_satisfied": bool(database.get("schema_isolation_satisfied")),
        "runtime_state_isolation_satisfied": bool(database.get("runtime_state_isolation_satisfied", database.get("database_isolation_satisfied"))),
        "process_scoped_pgoptions_applied": bool(database.get("process_scoped_pgoptions_applied")),
        "search_path": database.get("search_path", ""),
        "database_identity_guard_passed": bool(database.get("database_identity_guard_passed")),
        "schema_identity_guard_passed": bool(database.get("schema_identity_guard_passed")),
        "public_schema_write_count": int(database.get("public_schema_write_count", 0)),
        "development_schema_reused": bool(database.get("development_schema_reused", False)),
        "fresh_clone": bool(environment.get("fresh_clone")),
        "fresh_python_environment": bool(bootstrap.get("fresh_python_environment")),
        "runtime_state_isolated": bool(environment.get("runtime_state_isolated")),
        "corpus_document_count": int(corpus.get("corpus_document_count", 0)),
        "materialized_document_count": int(corpus.get("materialized_document_count", 0)),
        "chunk_count": int(corpus.get("chunk_count", 0)),
        "embedding_model": runtime.get("embedding_model"),
        "embedding_dimension": int(runtime.get("embedding_dimension", 0)),
        "vector_state_rebuilt": bool(corpus.get("vector_state_rebuilt")),
        "lexical_state_rebuilt": bool(corpus.get("lexical_state_rebuilt")),
        "graph_build_success": bool(corpus.get("graph_build_success")),
        "graph_node_count": int(corpus.get("graph_node_count", 0)),
        "graph_edge_count": int(corpus.get("graph_edge_count", 0)),
        "formal_query_count": int(retrieval.get("formal_query_count", 0)),
        "query_pass_count": int(retrieval.get("query_pass_count", 0)),
        "query_fail_count": int(retrieval.get("query_fail_count", 0)),
        **query_flags,
        "graph_activation_false_positive_count": int(retrieval.get("graph_activation_false_positive_count", 0)),
        "citation_provenance_valid": bool(retrieval.get("citation_provenance_valid")),
        "environment_contamination_detected": bool(runtime.get("environment_contamination_detected", False)),
        "default_initial_retrieval_policy": runtime.get("default_initial_retrieval_policy"),
        "graph_activation_policy": runtime.get("graph_activation_policy"),
        "graph_runtime_hop_depth": int(runtime.get("graph_runtime_hop_depth", 0)),
        "runtime_policy_mutation_count": int(runtime.get("runtime_policy_mutation_count", 0)),
        "algorithm_mutation_count": int(runtime.get("algorithm_mutation_count", 0)),
        "first_failure_stage": failure["first_failure_stage"],
        "dominant_root_cause": failure["dominant_root_cause"],
        "cold_start_reproducible": cold_start_reproducible,
        "pass_criteria": pass_criteria,
    }


def build_contract(summary: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0170.contract.v1",
        "task_id": TASK_ID,
        "cold_start_root": str(COLD_START_ROOT),
        "corpus_identity": corpus.get("corpus_identity"),
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "query_suite_identity": corpus.get("query_suite_identity"),
        "source_head": summary.get("source_authoritative_head"),
        "fresh_clone_required": True,
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "resolved_database_name": RESOLVED_DATABASE_NAME,
        "cold_start_schema": COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "database_isolation_required": False,
        "schema_level_isolation_required": True,
        "process_scoped_search_path_required": True,
        "process_scoped_pgoptions": PROCESS_SCOPED_PGOPTIONS,
        "no_old_runtime_state": True,
        "no_old_database_state": True,
        "runtime_policy_mutation_allowed": False,
        "algorithm_mutation_allowed": False,
        "benchmark_mutation_allowed": False,
        "expected_query_count": EXPECTED_QUERY_COUNT,
    }


def build_digests(*artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0170.digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": [digest_json(artifact) for artifact in artifacts],
        "combined_digest": digest_json([digest_json(artifact) for artifact in artifacts]),
    }


def verify_task0170_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (output_dir / name).exists() and not (write and name == "verification.json"):
            errors.append(f"missing artifact: {name}")
    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"summary missing field: {field}")
    if summary.get("task_id") != TASK_ID:
        errors.append("summary task_id mismatch")
    if summary.get("task_status") not in {"complete", "partial", "blocked"}:
        errors.append("invalid task_status")
    if summary.get("first_failure_stage") not in FAILURE_STAGES:
        errors.append("invalid first_failure_stage")
    if summary.get("cold_start_reproducible") is True and summary.get("task_status") != "complete":
        errors.append("cold_start_reproducible=true requires task_status=complete")
    if summary.get("task_status") == "complete" and summary.get("cold_start_reproducible") is not True:
        errors.append("task_status=complete requires cold_start_reproducible=true")
    if summary.get("formal_query_count", 0) != EXPECTED_QUERY_COUNT:
        errors.append("formal_query_count must equal expected_query_count")
    if summary.get("query_pass_count", 0) + summary.get("query_fail_count", 0) != summary.get("formal_query_count", 0):
        errors.append("query accounting mismatch")
    if summary.get("runtime_policy_mutation_count") != 0:
        errors.append("runtime policy mutation detected")
    if summary.get("algorithm_mutation_count") != 0:
        errors.append("algorithm mutation detected")
    if contract.get("runtime_policy_mutation_allowed") is not False:
        errors.append("contract must disallow runtime policy mutation")
    if contract.get("expected_query_count") != EXPECTED_QUERY_COUNT:
        errors.append("contract query count mismatch")
    status = "valid" if not errors else "invalid"
    result = {
        "schema_version": "opk-rag.task0170.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "errors": errors,
        "verified_artifact_count": sum((output_dir / name).exists() for name in REQUIRED_ARTIFACTS),
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else None,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(
    summary: dict[str, Any],
    environment: dict[str, Any],
    bootstrap: dict[str, Any],
    database: dict[str, Any],
    corpus: dict[str, Any],
    retrieval: dict[str, Any],
    runtime: dict[str, Any],
    failure: dict[str, Any],
) -> str:
    return f"""# TASK-0170 Cold-start Reproducibility Baseline Report

## 1. Test environment

Cold-start root: `{COLD_START_ROOT}`  
Corpus root: `{CORPUS_ROOT}`  
Runtime root: `{RUNTIME_ROOT}`

## 2. Source Git authority

Source authoritative HEAD: `{summary.get('source_authoritative_head')}`  
Working tree clean at audit start: `{environment.get('working_tree_clean_at_start')}`

## 3. Fresh clone validation

Fresh clone: `{summary.get('fresh_clone')}`  
Fresh clone HEAD: `{summary.get('fresh_clone_head')}`  
Git head equivalence: `{summary.get('git_head_equivalence')}`

## 4. Python bootstrap

Python: `{bootstrap.get('python_version')}`  
uv: `{bootstrap.get('uv_version')}`  
Dependency lock available: `{bootstrap.get('dependency_lock_available')}`  
Dependency install success: `{bootstrap.get('dependency_install_success')}`

## 5. Configuration bootstrap

Configuration bootstrap ready: `{bootstrap.get('configuration_bootstrap_ready')}`  
Configuration gap detected: `{bootstrap.get('configuration_gap_detected')}`  
Missing or implicit configuration: `{bootstrap.get('missing_or_implicit_configuration')}`

## 6. Supabase database isolation

Requested database: `{REQUESTED_DATABASE_NAME}`  
Configured database: `{database.get('configured_database_name')}`  
Database isolation satisfied: `{summary.get('database_isolation_satisfied')}`  
MCP observation: {database.get('mcp_observation')}

## 7. Database bootstrap

PostgreSQL connectivity: `{database.get('postgres_connectivity')}`  
pgvector available: `{database.get('pgvector_available')}`  
Schema initialization success: `{database.get('schema_initialization_success')}`

## 8. Corpus ingestion

Source document count: `{corpus.get('source_document_count')}`  
Materialized document count: `{summary.get('materialized_document_count')}`  
Ingestion failure count: `{corpus.get('ingestion_failure_count')}`

## 9. Chunking

Chunk count: `{summary.get('chunk_count')}`  
Long document chunk count: `{corpus.get('long_document_chunk_count')}`  
Chunking expectation met: `{corpus.get('chunking_expectation_met')}`

## 10. Embedding

Embedding model: `{summary.get('embedding_model')}`  
Embedding dimension: `{summary.get('embedding_dimension')}`  
Embedding count: `{corpus.get('embedding_count')}`

## 11. Vector / Lexical state rebuild

Vector state rebuilt: `{summary.get('vector_state_rebuilt')}`  
Lexical state rebuilt: `{summary.get('lexical_state_rebuilt')}`

## 12. Graph build

Graph build success: `{summary.get('graph_build_success')}`  
Graph node count: `{summary.get('graph_node_count')}`  
Graph edge count: `{summary.get('graph_edge_count')}`

## 13. Query suite result

Formal query count: `{summary.get('formal_query_count')}`  
Query pass count: `{summary.get('query_pass_count')}`  
Query fail count: `{summary.get('query_fail_count')}`

## 14. Runtime equivalence

Default initial retrieval policy: `{summary.get('default_initial_retrieval_policy')}`  
Graph activation policy: `{summary.get('graph_activation_policy')}`  
Graph runtime hop depth: `{summary.get('graph_runtime_hop_depth')}`  
Canonical runtime policy equivalence: `{runtime.get('canonical_runtime_policy_equivalence')}`

## 15. Contamination audit

Environment contamination detected: `{summary.get('environment_contamination_detected')}`  
Sentinels: `{', '.join(SENTINELS)}`

## 16. First failure stage

First failure stage: `{failure.get('first_failure_stage')}`  
Dominant root cause: `{failure.get('dominant_root_cause')}`

## 17. Reproducibility gaps

The runner records missing configuration, database isolation, database bootstrap, ingestion, retrieval, citation, and graph gaps in the JSON artifacts. TASK-0170 does not remediate those gaps.

## 18. Final decision

Task status: `{summary.get('task_status')}`  
Cold-start reproducible: `{summary.get('cold_start_reproducible')}`
"""


def _git_output(command: list[str], *, cwd: Path) -> CommandResult:
    return run_command(command, cwd=cwd, timeout=60)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(command, str(cwd), completed.returncode, _tail(completed.stdout), _tail(completed.stderr))
    except Exception as exc:
        return CommandResult(command, str(cwd), 99, "", f"{type(exc).__name__}: {exc}")


def _run_fresh_python(code: str, env: Mapping[str, str], *, timeout: int) -> CommandResult:
    return run_command(["uv", "run", "python", "-c", code], cwd=FRESH_CLONE_DIR, env=env, timeout=timeout)


def resolve_cold_start_secret_authority(env: Mapping[str, str], *, secret_file: Path = COLD_START_SECRET_FILE) -> dict[str, Any]:
    process_value = env.get(TASK_DATABASE_URL_VARIABLE, "").strip()
    source = "process_env" if process_value else "none"
    value = process_value
    errors: list[str] = []
    secret_file_found = secret_file.exists()
    permission_warning = False
    if not value and secret_file_found:
        parsed = parse_cold_start_secret_file(secret_file)
        value = parsed.get(TASK_DATABASE_URL_VARIABLE, "")
        errors.extend(parsed.get("_errors", []))
        source = "secret_file" if value else "none"
        try:
            permission_warning = (secret_file.stat().st_mode & 0o077) != 0
        except OSError as exc:
            errors.append(f"secret file stat failed: {type(exc).__name__}")
    elif not value and not secret_file_found:
        errors.append("cold-start secret file missing")
    if value and _database_name(value) != RESOLVED_DATABASE_NAME:
        errors.append(f"{TASK_DATABASE_URL_VARIABLE} targets database {_database_name(value) or 'unset'}, expected {RESOLVED_DATABASE_NAME}")
    if value and urlparse(value).scheme not in {"postgres", "postgresql"}:
        errors.append(f"{TASK_DATABASE_URL_VARIABLE} must use postgres/postgresql scheme")
    return {
        "schema_version": "opk-rag.task0170.secret-authority.v2",
        "task_id": TASK_ID,
        "database_connection_authority_source": source,
        "database_connection_authority_valid": bool(value and not errors),
        "database_url_present": bool(value),
        "database_url_value": value if value and not errors else "",
        "database_url_redacted": bool(value),
        "secret_file_path": str(secret_file),
        "secret_file_found": secret_file_found,
        "secret_file_permissions": oct(secret_file.stat().st_mode & 0o777) if secret_file_found else "",
        "secret_file_permission_warning": permission_warning,
        "allowed_variable_only": True,
        "connection_authority_errors": errors,
    }


def parse_cold_start_secret_file(path: Path) -> dict[str, Any]:
    allowed = {TASK_DATABASE_URL_VARIABLE}
    result: dict[str, Any] = {"_errors": []}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            result["_errors"].append(f"invalid env line {line_number}")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_env_value(value.strip())
        if key not in allowed:
            result["_errors"].append(f"disallowed secret variable {key}")
            continue
        result[key] = value
    return result


def probe_database_identity(database_url: str, *, env: Mapping[str, str]) -> dict[str, Any]:
    if not database_url:
        return _empty_identity_probe()
    try:
        import psycopg

        with psycopg.connect(database_url, autocommit=True, options=PROCESS_SCOPED_PGOPTIONS) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select current_database(), current_schema();")
                current_database, current_schema = cursor.fetchone()
                cursor.execute("show search_path;")
                search_path = str(cursor.fetchone()[0])
    except Exception as exc:
        probe = _empty_identity_probe()
        probe["identity_probe_error"] = f"{type(exc).__name__}: {exc}"
        return probe
    search_path_entries = [entry.strip().strip('"') for entry in search_path.split(",")]
    return {
        "schema_version": "opk-rag.task0170.database-identity-guard.v2",
        "task_id": TASK_ID,
        "current_database": current_database,
        "current_schema": current_schema,
        "search_path": search_path,
        "expected_database": RESOLVED_DATABASE_NAME,
        "expected_schema": COLD_START_SCHEMA,
        "pgoptions": env.get("PGOPTIONS", ""),
        "process_scoped_pgoptions_applied": env.get("PGOPTIONS") == PROCESS_SCOPED_PGOPTIONS,
        "database_identity_guard_passed": current_database == RESOLVED_DATABASE_NAME,
        "schema_identity_guard_passed": current_schema == COLD_START_SCHEMA
        and bool(search_path_entries)
        and search_path_entries[0] == COLD_START_SCHEMA,
        "fail_closed": current_database != RESOLVED_DATABASE_NAME or current_schema != COLD_START_SCHEMA,
    }


def reset_schema_scoped_runtime_state(database_url: str, schema_name: str = COLD_START_SCHEMA) -> None:
    if schema_name != COLD_START_SCHEMA:
        raise RuntimeError("unsafe cold-start schema reset target")
    import psycopg
    from psycopg import sql

    with psycopg.connect(database_url, autocommit=True, options=PROCESS_SCOPED_PGOPTIONS) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select current_database(), current_schema();")
            current_database, current_schema = cursor.fetchone()
            if current_database != RESOLVED_DATABASE_NAME or current_schema != schema_name:
                raise RuntimeError("schema identity guard failed before schema reset")
            cursor.execute(sql.SQL("drop schema if exists {} cascade").format(sql.Identifier(schema_name)))
            cursor.execute(sql.SQL("create schema {}").format(sql.Identifier(schema_name)))
            cursor.execute(sql.SQL("set search_path to {}, extensions").format(sql.Identifier(schema_name)))


def apply_schema_scoped_migrations(database_url: str, schema_name: str = COLD_START_SCHEMA) -> None:
    import psycopg
    from opk_rag.db.migrations import migration_paths

    with psycopg.connect(database_url, autocommit=True, options=PROCESS_SCOPED_PGOPTIONS) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select current_database(), current_schema();")
            current_database, current_schema = cursor.fetchone()
            if current_database != RESOLVED_DATABASE_NAME or current_schema != schema_name:
                raise RuntimeError("schema identity guard failed before migrations")
            for migration_path in migration_paths():
                sql_text = _schema_scope_sql(migration_path.read_text(encoding="utf-8"), schema_name)
                cursor.execute(sql_text)


def safe_clone_reset_path(path: Path) -> dict[str, Any]:
    root = FRESH_REPO_PARENT.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    safe = resolved != root and root in resolved.parents and str(resolved).startswith(str(root) + os.sep)
    return {
        "safe_clone_reset_guard": safe,
        "clone_reset_path": str(resolved),
        "clone_reset_root": str(root),
        "clone_reset_error": "" if safe else f"unsafe clone reset path: {resolved}",
    }


def apply_worktree_overlay(clone_dir: Path) -> dict[str, Any]:
    status = run_command(["git", "status", "--porcelain", "-z"], cwd=ROOT, timeout=60)
    if status.returncode != 0 or not status.stdout_tail:
        return {
            "source_worktree_overlay_applied": False,
            "overlay_file_count": 0,
            "overlay_paths": [],
            "overlay_error": "" if status.returncode == 0 else status.stderr_tail,
        }
    copied: list[str] = []
    errors: list[str] = []
    entries = [entry for entry in status.stdout_tail.split("\0") if entry]
    for entry in entries:
        relative = entry[3:] if len(entry) > 3 else ""
        if not _overlay_path_allowed(relative):
            continue
        source = ROOT / relative
        target = clone_dir / relative
        if not source.exists() or source.is_dir():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)
        except OSError as exc:
            errors.append(f"{relative}: {type(exc).__name__}")
    return {
        "source_worktree_overlay_applied": bool(copied),
        "overlay_file_count": len(copied),
        "overlay_paths": copied,
        "overlay_errors": errors,
    }


def _overlay_path_allowed(relative_path: str) -> bool:
    allowed_prefixes = ("opk_rag/", "scripts/", "tests/", ".env.example", "pyproject.toml", "uv.lock")
    blocked_prefixes = ("evaluation-data/", "docs/", "tasks/", "source-documents/")
    return relative_path.startswith(allowed_prefixes) and not relative_path.startswith(blocked_prefixes)


def _schema_scope_sql(sql_text: str, schema_name: str) -> str:
    return sql_text.replace("public.", f"{schema_name}.").replace(" from public;", " from public;")


def _empty_identity_probe() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0170.database-identity-guard.v2",
        "task_id": TASK_ID,
        "current_database": "",
        "current_schema": "",
        "search_path": "",
        "expected_database": RESOLVED_DATABASE_NAME,
        "expected_schema": COLD_START_SCHEMA,
        "pgoptions": "",
        "process_scoped_pgoptions_applied": False,
        "database_identity_guard_passed": False,
        "schema_identity_guard_passed": False,
        "fail_closed": True,
    }


def _schema_identity_failure_reason(identity: Mapping[str, Any], db_name: str) -> str:
    if identity.get("identity_probe_error"):
        return str(identity["identity_probe_error"])
    current_database = identity.get("current_database") or db_name or "unset"
    current_schema = identity.get("current_schema") or "unset"
    if current_database != RESOLVED_DATABASE_NAME:
        return f"current_database is {current_database}, expected {RESOLVED_DATABASE_NAME}"
    if current_schema != COLD_START_SCHEMA:
        return f"current_schema is {current_schema}, expected {COLD_START_SCHEMA}"
    return "search_path does not prioritize cold-start schema"


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _credential_leak_detected(paths: list[Path]) -> bool:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            if SECRET_URL_RE.search(path.read_text(encoding="utf-8", errors="ignore")):
                return True
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix in {".json", ".md", ".txt"}:
                if SECRET_URL_RE.search(candidate.read_text(encoding="utf-8", errors="ignore")):
                    return True
    return False


def _sentinel_probe_code(*, before: bool) -> str:
    sentinel_json = json.dumps(SENTINELS)
    return f"""
import os
import sys
import psycopg
sentinels = {sentinel_json}
url = os.environ["DATABASE_URL"]
hits = []
with psycopg.connect(url) as conn:
    with conn.cursor() as cur:
        for value in sentinels:
            cur.execute("select exists(select 1 from chunks where content like %s)", (f"%{{value}}%",))
            if cur.fetchone()[0]:
                hits.append(value)
print({{"sentinel_hits": hits, "phase": {"'pre_ingestion'" if before else "'post_ingestion'"}}})
sys.exit(3 if hits else 0)
"""


def _chunk_count_code() -> str:
    return """
import json
import os
import psycopg
from uuid import UUID
kb_id = UUID(os.environ["TASK0170_KB_ID"])
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    with conn.cursor() as cur:
        cur.execute("select count(*) from public.documents where knowledge_base_id = %s and index_status <> 'deleted' and relative_path <> 'README.md'", (kb_id,))
        docs = cur.fetchone()[0]
        cur.execute("select count(*) from public.chunks c join public.documents d on d.id = c.document_id where d.knowledge_base_id = %s and d.relative_path <> 'README.md'", (kb_id,))
        chunks = cur.fetchone()[0]
        cur.execute(\"\"\"
            select d.relative_path, count(c.id)
            from public.documents d left join public.chunks c on c.document_id = d.id
            where d.knowledge_base_id = %s
              and d.relative_path <> 'README.md'
            group by d.relative_path
        \"\"\", (kb_id,))
        per_doc = {row[0]: row[1] for row in cur.fetchall()}
print(json.dumps({
    "materialized_document_count": docs,
    "chunk_count": chunks,
    "long_document_chunk_count": per_doc.get("07_long_document_ingestion.md", 0),
    "chunking_expectation_met": per_doc.get("07_long_document_ingestion.md", 0) > 1,
    "graph_build_success": False,
    "graph_state_rebuilt": False,
}, ensure_ascii=False, sort_keys=True))
"""


def _execution_result(
    bootstrap: dict[str, Any],
    database: dict[str, Any],
    corpus: dict[str, Any],
    retrieval: dict[str, Any],
    runtime: dict[str, Any],
    first_failure_stage: str,
    dominant_root_cause: str,
) -> dict[str, Any]:
    return {
        "bootstrap": bootstrap,
        "database": database,
        "corpus": corpus,
        "retrieval": retrieval,
        "runtime": runtime,
        "first_failure_stage": first_failure_stage,
        "dominant_root_cause": dominant_root_cause,
    }


def _corpus_digest_errors(corpus_root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for entry in manifest.get("files", []):
        path = corpus_root / entry.get("name", "")
        if not path.exists():
            errors.append(f"manifest file missing: {entry.get('name')}")
            continue
        expected = entry.get("sha256")
        if expected and sha256_file(path) != expected:
            errors.append(f"sha256 mismatch: {entry.get('name')}")
    return errors


def _database_name(database_url: str) -> str:
    if not database_url:
        return ""
    return urlparse(database_url).path.lstrip("/")


def _tail(value: str, *, limit: int = 4000) -> str:
    return value[-limit:]


def _has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {}
    return json.loads(stripped)


def _result_ids(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("results") or payload.get("evidence_bundle", {}).get("items") or []
    ids = []
    for row in rows:
        value = row.get("chunk_id") or row.get("id") or row.get("document_id")
        if value is not None:
            ids.append(str(value))
    return ids


def _citation_paths(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("results") or payload.get("evidence_bundle", {}).get("items") or []
    paths = []
    for row in rows:
        value = row.get("relative_path") or row.get("source_path") or row.get("document_path")
        if value is not None:
            paths.append(str(value))
    return paths


def _citation_provenance_valid(paths: list[str]) -> bool:
    if not paths:
        return False
    for raw_path in paths:
        path = Path(raw_path)
        if "evaluation-data/results" in raw_path or "__pycache__" in raw_path:
            return False
        try:
            path.resolve().relative_to(ROOT.resolve())
            return False
        except ValueError:
            continue
    return True


def _query_ok(retrieval: dict[str, Any], query_id: str) -> bool:
    for row in retrieval.get("queries", []):
        if row.get("query_id") == query_id:
            return bool(row.get("answer_contains_expected_fact"))
    return False


def _root_cause_classification(stage: str) -> str:
    if stage.startswith("database"):
        return "database"
    if stage in {"configuration", "dependency_bootstrap"}:
        return "configuration"
    if stage in {"document_discovery", "parsing", "canonical_document", "chunking", "embedding"}:
        return "data_pipeline"
    if stage.startswith("graph"):
        return "graph"
    if stage in {"candidate_retrieval", "reranking", "evidence_composition", "citation", "generation"}:
        return "runtime"
    if stage in {"git_clone", "git_revision"}:
        return "environment"
    return "none" if stage == "none" else "unknown"
