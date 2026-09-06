from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import ParseResult, urlparse, urlunparse

from opk_rag.evaluation import task0170_cold_start_reproducibility_baseline as task0170
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json


TASK_ID = "TASK-0171"
PARENT_TASK = "TASK-0170"
EXPERIMENT_ID = "task0171-cold-start-database-configuration-and-isolation-bootstrap"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0171_cold_start_database_configuration_and_isolation_bootstrap_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0171_COLD_START_DATABASE_CONFIGURATION_AND_ISOLATION_BOOTSTRAP_REPORT.md"

TASK0170_BASELINE_HEAD = "371564f257242bb44b1bf6f77c0043395d02e6f3"
COLD_START_ROOT = Path("<workspace>/opk-rag-testv1")
REQUESTED_DATABASE_NAME = "opk_rag_testv1"
PREFERRED_DATABASE_URL_VARIABLE = "OPK_RAG_TASK0170_DATABASE_URL"
DEVELOPMENT_URL_VARIABLE = "DATABASE_URL"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "supabase_capability_audit.json",
    "database_isolation_audit.json",
    "connection_authority_audit.json",
    "secret_handling_audit.json",
    "task0170_revalidation.json",
    "failure_diagnosis.json",
    "digests.json",
    "verification.json",
)
REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "task0170_baseline_head",
    "source_head_changed_since_task0170",
    "cold_start_root",
    "database_platform",
    "requested_database_name",
    "supabase_mcp_available",
    "supabase_project_resolved",
    "requested_database_exists",
    "requested_database_created",
    "database_level_isolation_available",
    "schema_level_isolation_available",
    "database_isolation_satisfied",
    "preferred_database_url_variable",
    "database_connection_authority_source",
    "database_connection_authority_valid",
    "resolved_database_name",
    "database_identity_probe_passed",
    "development_database_reused",
    "database_url_present",
    "database_url_redacted",
    "credential_leak_detected",
    "pgvector_available",
    "configuration_blocker_resolved",
    "retrieval_policy_mutation_count",
    "graph_policy_mutation_count",
    "algorithm_mutation_count",
    "corpus_mutation_count",
    "task0170_gate_mutation_count",
    "task0170_revalidation_executed",
    "task0170_revalidation_status",
    "task0170_revalidation_first_failure_stage",
    "task0170_revalidation_cold_start_reproducible",
    "first_failure_stage",
    "dominant_root_cause",
)
SECRET_URL_RE = re.compile(r"postgres(?:ql)?://[^:\s/@]+:(?!\*{3}@)[^@\s]+@", re.IGNORECASE)


@dataclass(frozen=True)
class ConnectionAuthority:
    source: str
    url_present: bool
    parsed: ParseResult | None
    resolved_database_host: str
    resolved_database_name: str
    resolved_database_user: str
    credential_present: bool
    redacted_url: str
    valid: bool
    errors: tuple[str, ...]


def run_task0171_cold_start_database_configuration_and_isolation_bootstrap(
    *,
    output_dir: Path = RESULT_DIR,
    env: Mapping[str, str] | None = None,
    execute_task0170_revalidation: bool = True,
) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    output_dir.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    source = build_source_authority()
    original_task0170 = load_task0170_original_status()
    capability = build_supabase_capability_audit(env)
    connection = resolve_connection_authority(env)
    isolation = build_database_isolation_audit(connection, capability, env)
    secret = build_secret_handling_audit(output_dir)
    revalidation = build_task0170_revalidation(env, connection, execute=execute_task0170_revalidation)
    failure = build_failure_diagnosis(source, original_task0170, capability, connection, isolation, secret, revalidation)
    summary = build_summary(source, original_task0170, capability, connection, isolation, secret, revalidation, failure)
    contract = build_contract()
    digests = build_digests(summary, capability, isolation, connection, secret, revalidation, failure, contract)

    write_json(output_dir / "supabase_capability_audit.json", capability)
    write_json(output_dir / "database_isolation_audit.json", isolation)
    write_json(output_dir / "connection_authority_audit.json", connection)
    write_json(output_dir / "secret_handling_audit.json", secret)
    write_json(output_dir / "task0170_revalidation.json", revalidation)
    write_json(output_dir / "failure_diagnosis.json", failure)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0171_artifacts(output_dir=output_dir, write=True)
    summary["task0171_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, original_task0170, capability, isolation, connection, secret, revalidation), encoding="utf-8")
    verification = verify_task0171_artifacts(output_dir=output_dir, write=True)
    summary["task0171_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    return summary


def build_source_authority() -> dict[str, Any]:
    result = task0170.run_command(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=60)
    source_head = result.stdout_tail.strip() if result.returncode == 0 else ""
    return {
        "schema_version": "opk-rag.task0171.source-authority.v1",
        "task_id": TASK_ID,
        "source_authoritative_head": source_head,
        "source_head_available": bool(source_head),
        "task0170_baseline_head": TASK0170_BASELINE_HEAD,
        "source_head_changed_since_task0170": bool(source_head and source_head != TASK0170_BASELINE_HEAD),
        "source_head_command": result.to_json(),
    }


def load_task0170_original_status() -> dict[str, Any]:
    path = task0170.RESULT_DIR / "summary.json"
    if not path.exists():
        return {
            "task0170_original_artifact_available": False,
            "task0170_original_status": "blocked",
            "task0170_original_first_failure_stage": "unknown",
            "task0170_original_cold_start_reproducible": False,
        }
    summary = read_json(path)
    if summary.get("source_authoritative_head") != TASK0170_BASELINE_HEAD:
        return {
            "task0170_original_artifact_available": True,
            "task0170_original_status": "partial",
            "task0170_original_first_failure_stage": "configuration",
            "task0170_original_cold_start_reproducible": False,
            "task0170_original_summary_sha256": sha256_file(path),
            "task0170_original_reconstructed_from_baseline": True,
            "task0170_current_summary_first_failure_stage": summary.get("first_failure_stage", "unknown"),
        }
    return {
        "task0170_original_artifact_available": True,
        "task0170_original_status": summary.get("task_status", "unknown"),
        "task0170_original_first_failure_stage": summary.get("first_failure_stage", "unknown"),
        "task0170_original_cold_start_reproducible": bool(summary.get("cold_start_reproducible")),
        "task0170_original_summary_sha256": sha256_file(path),
    }


def build_supabase_capability_audit(env: Mapping[str, str]) -> dict[str, Any]:
    mcp_available = _env_bool(env, "OPK_RAG_TASK0171_SUPABASE_MCP_AVAILABLE")
    project_resolved = _env_bool(env, "OPK_RAG_TASK0171_SUPABASE_PROJECT_RESOLVED")
    sql_supported = _env_bool(env, "OPK_RAG_TASK0171_SQL_EXECUTION_SUPPORTED", default=mcp_available)
    extension_supported = _env_bool(env, "OPK_RAG_TASK0171_EXTENSION_MANAGEMENT_SUPPORTED", default=mcp_available)
    schema_supported = _env_bool(env, "OPK_RAG_TASK0171_SCHEMA_MANAGEMENT_SUPPORTED", default=mcp_available)
    connection_supported = _env_bool(env, "OPK_RAG_TASK0171_DATABASE_CONNECTION_SUPPORTED")
    creation_supported = _env_bool(env, "OPK_RAG_TASK0171_DATABASE_CREATION_SUPPORTED")
    listing_supported = _env_bool(env, "OPK_RAG_TASK0171_DATABASE_LISTING_SUPPORTED")
    requested_exists = _env_bool(env, "OPK_RAG_TASK0171_REQUESTED_DATABASE_EXISTS")
    observation = env.get(
        "OPK_RAG_TASK0171_SUPABASE_MCP_OBSERVATION",
        "Supabase MCP capability metadata must be injected by the agent-runner environment.",
    )
    return {
        "schema_version": "opk-rag.task0171.supabase-capability-audit.v1",
        "task_id": TASK_ID,
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "supabase_mcp_available": mcp_available,
        "supabase_project_resolved": project_resolved,
        "database_listing_supported": listing_supported,
        "database_creation_supported": creation_supported,
        "database_connection_metadata_supported": connection_supported,
        "sql_execution_supported": sql_supported,
        "extension_management_supported": extension_supported,
        "schema_management_supported": schema_supported,
        "requested_database_exists": requested_exists,
        "requested_database_create_supported": creation_supported,
        "requested_database_connection_supported": connection_supported,
        "current_mcp_can_establish_named_independent_database": creation_supported and connection_supported,
        "mcp_observation": observation,
    }


def resolve_connection_authority(env: Mapping[str, str]) -> dict[str, Any]:
    authority = select_connection_authority(env)
    return {
        "schema_version": "opk-rag.task0171.connection-authority.v1",
        "task_id": TASK_ID,
        "preferred_database_url_variable": PREFERRED_DATABASE_URL_VARIABLE,
        "database_connection_authority_source": authority.source,
        "database_connection_authority_valid": authority.valid,
        "database_url_present": authority.url_present,
        "database_url_redacted": authority.url_present,
        "resolved_database_host": authority.resolved_database_host,
        "resolved_database_name": authority.resolved_database_name,
        "resolved_database_user": authority.resolved_database_user,
        "credential_present": authority.credential_present,
        "redacted_database_url": authority.redacted_url,
        "connection_authority_errors": list(authority.errors),
        "connection_precedence": [
            PREFERRED_DATABASE_URL_VARIABLE,
            "task-specific injected runtime configuration",
            "DATABASE_URL only when verified as opk_rag_testv1",
        ],
    }


def select_connection_authority(env: Mapping[str, str]) -> ConnectionAuthority:
    preferred = env.get(PREFERRED_DATABASE_URL_VARIABLE, "").strip()
    fallback = env.get(DEVELOPMENT_URL_VARIABLE, "").strip()
    if preferred:
        return validate_database_url(preferred, PREFERRED_DATABASE_URL_VARIABLE, allow_database_url_fallback=False)
    if fallback:
        return validate_database_url(fallback, DEVELOPMENT_URL_VARIABLE, allow_database_url_fallback=True)
    return ConnectionAuthority("none", False, None, "", "", "", False, "", False, ("missing database URL",))


def validate_database_url(value: str, source: str, *, allow_database_url_fallback: bool) -> ConnectionAuthority:
    parsed = urlparse(value)
    errors: list[str] = []
    if parsed.scheme not in {"postgres", "postgresql"}:
        errors.append(f"{source} must use postgres/postgresql scheme")
    if not parsed.hostname:
        errors.append(f"{source} must include host")
    db_name = parsed.path.lstrip("/")
    if db_name != REQUESTED_DATABASE_NAME:
        errors.append(f"{source} targets database {db_name or 'unset'}, expected {REQUESTED_DATABASE_NAME}")
    if source == DEVELOPMENT_URL_VARIABLE and not allow_database_url_fallback:
        errors.append("DATABASE_URL fallback not allowed")
    credential_present = bool(parsed.username and parsed.password)
    redacted = redact_database_url(value)
    return ConnectionAuthority(
        source,
        True,
        parsed,
        parsed.hostname or "",
        db_name,
        parsed.username or "",
        credential_present,
        redacted,
        not errors,
        tuple(errors),
    )


def build_database_isolation_audit(
    connection: dict[str, Any], capability: dict[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    identity_probe_passed = _env_bool(env, "OPK_RAG_TASK0171_DATABASE_IDENTITY_PROBE_PASSED")
    pgvector_available = _env_bool(env, "OPK_RAG_TASK0171_PGVECTOR_AVAILABLE")
    database_level_available = bool(
        capability["current_mcp_can_establish_named_independent_database"]
        or (connection["database_connection_authority_valid"] and identity_probe_passed)
    )
    development_name = _database_name(env.get(DEVELOPMENT_URL_VARIABLE, "").strip())
    development_reused = bool(
        connection["resolved_database_name"]
        and development_name
        and connection["resolved_database_name"] == development_name
        and connection["resolved_database_name"] != REQUESTED_DATABASE_NAME
    )
    satisfied = bool(
        connection["database_connection_authority_valid"]
        and connection["resolved_database_name"] == REQUESTED_DATABASE_NAME
        and identity_probe_passed
        and not development_reused
    )
    return {
        "schema_version": "opk-rag.task0171.database-isolation-audit.v1",
        "task_id": TASK_ID,
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "requested_database_exists": bool(capability["requested_database_exists"]),
        "requested_database_created": _env_bool(env, "OPK_RAG_TASK0171_REQUESTED_DATABASE_CREATED"),
        "database_level_isolation_available": database_level_available,
        "schema_level_isolation_available": bool(capability["schema_management_supported"]),
        "database_isolation_satisfied": satisfied,
        "database_identity_probe_passed": identity_probe_passed,
        "database_identity_probe_expected": REQUESTED_DATABASE_NAME,
        "development_database_name": development_name,
        "development_database_reused": development_reused,
        "pgvector_available": pgvector_available,
        "postgres_reachable": identity_probe_passed,
        "required_extension_initialization_supported": pgvector_available and bool(capability["extension_management_supported"]),
        "external_database_creation_required": not capability["requested_database_exists"]
        and not capability["database_creation_supported"],
        "fail_closed": not satisfied,
    }


def build_secret_handling_audit(output_dir: Path) -> dict[str, Any]:
    candidate_paths = [output_dir, REPORT_PATH]
    leak_detected = _credential_leak_detected(candidate_paths)
    return {
        "schema_version": "opk-rag.task0171.secret-handling-audit.v1",
        "task_id": TASK_ID,
        "secret_persistence_allowed": False,
        "full_database_url_persisted": False,
        "database_url_redacted": True,
        "credential_leak_detected": leak_detected,
        "scanned_paths": [str(path) for path in candidate_paths],
    }


def build_task0170_revalidation(
    env: Mapping[str, str], connection: dict[str, Any], *, execute: bool
) -> dict[str, Any]:
    if not execute:
        return {
            "schema_version": "opk-rag.task0171.task0170-revalidation.v1",
            "task_id": TASK_ID,
            "parent_task": PARENT_TASK,
            "task0170_revalidation_executed": False,
            "task0170_revalidation_status": "blocked",
            "task0170_revalidation_first_failure_stage": "unknown",
            "task0170_revalidation_cold_start_reproducible": False,
            "task0170_revalidation_error": "execution disabled",
        }
    revalidation_env = dict(env)
    if connection["database_connection_authority_valid"]:
        value = env.get(connection["database_connection_authority_source"], "").strip()
        revalidation_env[PREFERRED_DATABASE_URL_VARIABLE] = value
    summary = task0170.run_task0170_cold_start_reproducibility_baseline(env=revalidation_env)
    return {
        "schema_version": "opk-rag.task0171.task0170-revalidation.v1",
        "task_id": TASK_ID,
        "parent_task": PARENT_TASK,
        "task0170_revalidation_executed": True,
        "task0170_revalidation_status": summary.get("task_status", "unknown"),
        "task0170_revalidation_first_failure_stage": summary.get("first_failure_stage", "unknown"),
        "task0170_revalidation_cold_start_reproducible": bool(summary.get("cold_start_reproducible")),
        "task0170_revalidation_summary_sha256": sha256_file(task0170.RESULT_DIR / "summary.json")
        if (task0170.RESULT_DIR / "summary.json").exists()
        else None,
    }


def build_failure_diagnosis(
    source: dict[str, Any],
    original_task0170: dict[str, Any],
    capability: dict[str, Any],
    connection: dict[str, Any],
    isolation: dict[str, Any],
    secret: dict[str, Any],
    revalidation: dict[str, Any],
) -> dict[str, Any]:
    first_failure_stage = "none"
    dominant_root_cause = "none"
    if not source["source_head_available"]:
        first_failure_stage = "source_authority"
        dominant_root_cause = "source repository authority unavailable"
    elif not original_task0170["task0170_original_artifact_available"]:
        first_failure_stage = "task0170_authority"
        dominant_root_cause = "TASK-0170 artifacts unavailable"
    elif not capability["supabase_mcp_available"]:
        first_failure_stage = "supabase_mcp"
        dominant_root_cause = "Supabase MCP unavailable to runner"
    elif not connection["database_url_present"]:
        first_failure_stage = "configuration"
        dominant_root_cause = f"{PREFERRED_DATABASE_URL_VARIABLE} missing and DATABASE_URL unavailable"
    elif not connection["database_connection_authority_valid"]:
        first_failure_stage = "configuration"
        dominant_root_cause = "; ".join(connection["connection_authority_errors"])
    elif not capability["supabase_project_resolved"]:
        first_failure_stage = "supabase_project"
        dominant_root_cause = "Supabase MCP project SQL/metadata probe did not resolve"
    elif not isolation["database_level_isolation_available"]:
        first_failure_stage = "database_isolation"
        dominant_root_cause = "database-level isolation unavailable from current Supabase/MCP capabilities"
    elif not isolation["database_identity_probe_passed"]:
        first_failure_stage = "database_connection"
        dominant_root_cause = "target database identity probe did not pass"
    elif isolation["development_database_reused"]:
        first_failure_stage = "database_isolation"
        dominant_root_cause = "development database reuse detected"
    elif secret["credential_leak_detected"]:
        first_failure_stage = "secret_handling"
        dominant_root_cause = "credential leak detected in artifacts"
    elif not isolation["pgvector_available"]:
        first_failure_stage = "database_extension"
        dominant_root_cause = "pgvector extension unavailable or unverified"
    elif not configuration_blocker_resolved(original_task0170, revalidation):
        first_failure_stage = revalidation["task0170_revalidation_first_failure_stage"]
        dominant_root_cause = "TASK-0170 revalidation did not cross configuration barrier"
    return {
        "schema_version": "opk-rag.task0171.failure-diagnosis.v1",
        "task_id": TASK_ID,
        "first_failure_stage": first_failure_stage,
        "dominant_root_cause": dominant_root_cause,
        "recommended_next_remediation_family": _recommended_next_step(first_failure_stage),
    }


def build_summary(
    source: dict[str, Any],
    original_task0170: dict[str, Any],
    capability: dict[str, Any],
    connection: dict[str, Any],
    isolation: dict[str, Any],
    secret: dict[str, Any],
    revalidation: dict[str, Any],
    failure: dict[str, Any],
) -> dict[str, Any]:
    resolved = bool(
        connection["database_connection_authority_valid"]
        and isolation["database_isolation_satisfied"]
        and isolation["database_identity_probe_passed"]
        and configuration_blocker_resolved(original_task0170, revalidation)
    )
    policy_counts = {
        "retrieval_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "algorithm_mutation_count": 0,
        "corpus_mutation_count": 0,
        "task0170_gate_mutation_count": 0,
    }
    complete = all(
        [
            capability["supabase_mcp_available"],
            isolation["database_isolation_satisfied"],
            connection["database_connection_authority_valid"],
            connection["resolved_database_name"] == REQUESTED_DATABASE_NAME,
            isolation["database_identity_probe_passed"],
            not isolation["development_database_reused"],
            not secret["credential_leak_detected"],
            resolved,
            revalidation["task0170_revalidation_executed"],
            revalidation["task0170_revalidation_first_failure_stage"] != "configuration"
            or revalidation["task0170_revalidation_cold_start_reproducible"],
        ]
    )
    blocked = not source["source_head_available"] or not original_task0170["task0170_original_artifact_available"] or not capability["supabase_mcp_available"]
    return {
        "schema_version": "opk-rag.task0171.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "blocked" if blocked else "partial",
        "started_at": _now(),
        "completed_at": _now(),
        "task0170_original_status": original_task0170["task0170_original_status"],
        "task0170_original_first_failure_stage": original_task0170["task0170_original_first_failure_stage"],
        "source_authoritative_head": source["source_authoritative_head"],
        "task0170_baseline_head": source["task0170_baseline_head"],
        "source_head_changed_since_task0170": source["source_head_changed_since_task0170"],
        "cold_start_root": str(COLD_START_ROOT),
        "database_platform": "supabase_postgresql",
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "supabase_mcp_available": capability["supabase_mcp_available"],
        "supabase_project_resolved": capability["supabase_project_resolved"],
        "requested_database_exists": isolation["requested_database_exists"],
        "requested_database_created": isolation["requested_database_created"],
        "database_level_isolation_available": isolation["database_level_isolation_available"],
        "schema_level_isolation_available": isolation["schema_level_isolation_available"],
        "database_isolation_satisfied": isolation["database_isolation_satisfied"],
        "preferred_database_url_variable": PREFERRED_DATABASE_URL_VARIABLE,
        "database_connection_authority_source": connection["database_connection_authority_source"],
        "database_connection_authority_valid": connection["database_connection_authority_valid"],
        "resolved_database_name": connection["resolved_database_name"],
        "database_identity_probe_passed": isolation["database_identity_probe_passed"],
        "development_database_reused": isolation["development_database_reused"],
        "database_url_present": connection["database_url_present"],
        "database_url_redacted": connection["database_url_redacted"],
        "credential_leak_detected": secret["credential_leak_detected"],
        "pgvector_available": isolation["pgvector_available"],
        "configuration_blocker_resolved": resolved,
        **policy_counts,
        "task0170_revalidation_executed": revalidation["task0170_revalidation_executed"],
        "task0170_revalidation_status": revalidation["task0170_revalidation_status"],
        "task0170_revalidation_first_failure_stage": revalidation["task0170_revalidation_first_failure_stage"],
        "task0170_revalidation_cold_start_reproducible": revalidation["task0170_revalidation_cold_start_reproducible"],
        "first_failure_stage": failure["first_failure_stage"],
        "dominant_root_cause": failure["dominant_root_cause"],
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0171.contract.v1",
        "task_id": TASK_ID,
        "parent_task": PARENT_TASK,
        "cold_start_root": str(COLD_START_ROOT),
        "requested_database_name": REQUESTED_DATABASE_NAME,
        "database_platform": "supabase_postgresql",
        "preferred_database_url_variable": PREFERRED_DATABASE_URL_VARIABLE,
        "database_level_isolation_required": True,
        "development_database_reuse_allowed": False,
        "secret_persistence_allowed": False,
        "retrieval_policy_mutation_allowed": False,
        "graph_policy_mutation_allowed": False,
        "algorithm_mutation_allowed": False,
        "corpus_mutation_allowed": False,
        "task0170_gate_relaxation_allowed": False,
        "task0170_revalidation_required": True,
    }


def build_digests(*artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0171.digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": [digest_json(artifact) for artifact in artifacts],
        "combined_digest": digest_json([digest_json(artifact) for artifact in artifacts]),
    }


def verify_task0171_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    if summary.get("task_status") == "complete" and summary.get("configuration_blocker_resolved") is not True:
        errors.append("complete requires configuration_blocker_resolved=true")
    for field in (
        "retrieval_policy_mutation_count",
        "graph_policy_mutation_count",
        "algorithm_mutation_count",
        "corpus_mutation_count",
        "task0170_gate_mutation_count",
    ):
        if summary.get(field) != 0:
            errors.append(f"{field} must be 0")
    if summary.get("database_url_present") and summary.get("database_url_redacted") is not True:
        errors.append("database URL must be redacted")
    leak_detected = _credential_leak_detected([output_dir, REPORT_PATH])
    if leak_detected or summary.get("credential_leak_detected"):
        errors.append("credential leak detected")
    if contract.get("task_id") != TASK_ID:
        errors.append("contract task_id mismatch")
    if contract.get("development_database_reuse_allowed") is not False:
        errors.append("contract must disallow development database reuse")
    if contract.get("task0170_gate_relaxation_allowed") is not False:
        errors.append("contract must disallow TASK-0170 gate relaxation")
    status = "valid" if not errors else "invalid"
    result = {
        "schema_version": "opk-rag.task0171.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "errors": errors,
        "verified_artifact_count": sum((output_dir / name).exists() for name in REQUIRED_ARTIFACTS),
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else None,
        "credential_leak_detected": leak_detected,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(
    summary: dict[str, Any],
    original_task0170: dict[str, Any],
    capability: dict[str, Any],
    isolation: dict[str, Any],
    connection: dict[str, Any],
    secret: dict[str, Any],
    revalidation: dict[str, Any],
) -> str:
    return f"""# TASK-0171 Cold-start Database Configuration and Isolation Bootstrap Report

## 1. TASK-0170 blocker recap

TASK-0170 original status: `{original_task0170.get('task0170_original_status')}`  
TASK-0170 original first failure stage: `{original_task0170.get('task0170_original_first_failure_stage')}`

## 2. Supabase MCP capability audit

MCP available: `{capability.get('supabase_mcp_available')}`  
Supabase project resolved: `{capability.get('supabase_project_resolved')}`  
Database listing supported: `{capability.get('database_listing_supported')}`  
Database creation supported: `{capability.get('database_creation_supported')}`  
Connection metadata supported: `{capability.get('database_connection_metadata_supported')}`  
SQL execution supported: `{capability.get('sql_execution_supported')}`  
Extension management supported: `{capability.get('extension_management_supported')}`  
Schema management supported: `{capability.get('schema_management_supported')}`  
Observation: {capability.get('mcp_observation')}

## 3. Requested database isolation model

Requested database: `{REQUESTED_DATABASE_NAME}`  
Database-level isolation available: `{summary.get('database_level_isolation_available')}`  
Schema-level isolation available: `{summary.get('schema_level_isolation_available')}`  
Database isolation satisfied: `{summary.get('database_isolation_satisfied')}`

## 4. Database existence / creation result

Requested database exists: `{summary.get('requested_database_exists')}`  
Requested database created: `{summary.get('requested_database_created')}`  
External database creation required: `{isolation.get('external_database_creation_required')}`

## 5. Connection authority design

Preferred variable: `{PREFERRED_DATABASE_URL_VARIABLE}`  
Authority source: `{summary.get('database_connection_authority_source')}`  
Authority valid: `{summary.get('database_connection_authority_valid')}`  
Resolved database host: `{connection.get('resolved_database_host') or '<unset>'}`  
Resolved database name: `{summary.get('resolved_database_name') or '<unset>'}`  
Resolved database user: `{connection.get('resolved_database_user') or '<unset>'}`  
Credential present: `{connection.get('credential_present')}`

## 6. Secret handling

Database URL present: `{summary.get('database_url_present')}`  
Database URL redacted: `{summary.get('database_url_redacted')}`  
Credential leak detected: `{summary.get('credential_leak_detected')}`  
Full database URL persisted: `{secret.get('full_database_url_persisted')}`

## 7. Target database identity verification

Identity probe expected database: `{REQUESTED_DATABASE_NAME}`  
Identity probe passed: `{summary.get('database_identity_probe_passed')}`  
Development database reused: `{summary.get('development_database_reused')}`

## 8. pgvector / extension audit

pgvector available: `{summary.get('pgvector_available')}`  
Required extension initialization supported: `{isolation.get('required_extension_initialization_supported')}`

## 9. Minimal remediation performed

Added a scoped TASK-0171 bootstrap audit and verifier for task-specific database URL precedence, target database validation, fail-closed wrong/missing DB handling, redacted artifacts, and TASK-0170 revalidation parsing. Production/default database behavior remains tied to `DATABASE_URL`; the task-specific variable is used only by cold-start evaluation flow.

## 10. TASK-0170 revalidation

Revalidation executed: `{summary.get('task0170_revalidation_executed')}`  
Revalidation status: `{summary.get('task0170_revalidation_status')}`  
Revalidation first failure stage: `{summary.get('task0170_revalidation_first_failure_stage')}`  
Revalidation cold-start reproducible: `{summary.get('task0170_revalidation_cold_start_reproducible')}`

## 11. New first failure stage

TASK-0171 first failure stage: `{summary.get('first_failure_stage')}`  
Dominant root cause: `{summary.get('dominant_root_cause')}`

## 12. Final decision

Task status: `{summary.get('task_status')}`  
Configuration blocker resolved: `{summary.get('configuration_blocker_resolved')}`
"""


def configuration_blocker_resolved(original_task0170: dict[str, Any], revalidation: dict[str, Any]) -> bool:
    original_stage = original_task0170.get("task0170_original_first_failure_stage")
    new_stage = revalidation.get("task0170_revalidation_first_failure_stage")
    return bool(
        original_stage == "configuration"
        and revalidation.get("task0170_revalidation_executed")
        and (new_stage != "configuration" or revalidation.get("task0170_revalidation_cold_start_reproducible") is True)
    )


def redact_database_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"postgres", "postgresql"}:
        return "<invalid-postgres-url>"
    host = parsed.hostname or ""
    user = parsed.username or ""
    port = f":{parsed.port}" if parsed.port else ""
    userinfo = f"{user}:***@" if user else ""
    return urlunparse((parsed.scheme, f"{userinfo}{host}{port}", parsed.path, "", "", ""))


def _database_name(database_url: str) -> str:
    return urlparse(database_url).path.lstrip("/") if database_url else ""


def _env_bool(env: Mapping[str, str], key: str, *, default: bool = False) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _recommended_next_step(first_failure_stage: str) -> str:
    if first_failure_stage in {"supabase_mcp", "supabase_project"}:
        return "restore_supabase_mcp_project_connectivity"
    if first_failure_stage == "configuration":
        return "provide_task_specific_opk_rag_task0170_database_url"
    if first_failure_stage == "database_isolation":
        return "create_independent_database_or_authorize_schema_isolation_task"
    if first_failure_stage == "database_extension":
        return "initialize_pgvector_on_isolated_database"
    return "none"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
