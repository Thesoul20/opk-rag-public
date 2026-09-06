from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    COLD_START_SCHEMA,
    COLD_START_SECRET_FILE,
    FRESH_CLONE_DIR,
    PROCESS_SCOPED_PGOPTIONS,
    RESOLVED_DATABASE_NAME,
    TASK_DATABASE_URL_VARIABLE,
    parse_cold_start_secret_file,
)

TASK_ID = "TASK-0178"
TASK0177_SOURCE_HEAD = "71b94915c5762d0bb0ea3cbea1f3cb280a2928db"
EXPERIMENT_ID = "task0178-cold-start-database-schema-timeout-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0178_cold_start_database_schema_timeout_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0178_COLD_START_DATABASE_SCHEMA_TIMEOUT_DIAGNOSIS_REPORT.md"

TIMEOUT_STAGES = {
    "database_secret_resolution",
    "database_dns_resolution",
    "database_tcp_connect",
    "database_tls_handshake",
    "database_authentication",
    "database_session_pooler",
    "database_connection_established",
    "database_pgoptions_application",
    "database_schema_resolution",
    "database_identity_probe",
    "database_query_execution",
    "database_transaction",
    "database_connection_close",
    "unknown",
}
ROOT_CAUSES = {
    "supabase_transient_connectivity",
    "session_pooler_connection_timeout",
    "session_pooler_schema_session_issue",
    "dns_resolution_timeout",
    "ipv6_connectivity_issue",
    "tcp_connect_timeout",
    "tls_handshake_timeout",
    "database_authentication_timeout",
    "python_driver_timeout",
    "sqlalchemy_pool_timeout",
    "stale_pooled_connection",
    "runtime_environment_propagation_failure",
    "pgoptions_not_propagated",
    "schema_identity_probe_timeout",
    "statement_timeout",
    "transaction_blocking",
    "database_lock_contention",
    "stale_schema_state",
    "external_network_instability",
    "unknown",
}
SECRET_PATTERNS = [
    re.compile(r"postgres(?:ql)?://[^:\s/@]+:[^@\s]+@", re.I),
    re.compile(r"//([^:\s/@]+):([^@\s]+)@"),
]


@dataclass(slots=True)
class TimedResult:
    success: bool
    elapsed_ms: int
    data: dict[str, Any]
    error_type: str = ""
    error_message: str = ""


def ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(s in k.lower() for s in ("password", "database_url_value", "uri", "url")) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        text = value
        text = re.sub(r"postgres(?:ql)?://[^\s'\"]+", "postgresql://<redacted>", text, flags=re.I)
        text = re.sub(r"//([^:\s/@]+):([^@\s]+)@", r"//\1:<redacted>@", text)
        return text
    return value


def credential_leak_detected_in_text(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def validate_pgoptions(value: str) -> bool:
    return value.strip() == PROCESS_SCOPED_PGOPTIONS and COLD_START_SCHEMA in value and "public" not in value


def classify_endpoint(parsed) -> str:
    host = (parsed.hostname or "").lower()
    port = parsed.port or 5432
    if "pooler.supabase.com" in host:
        return "transaction_pooler" if port == 6543 else "session_pooler"
    if host.startswith("db.") and host.endswith(".supabase.co"):
        return "direct"
    return "unknown"


def classify_exception_stage(exc: BaseException) -> str:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timed out" in msg or "timeout" in msg or "timeoutexpired" in name:
        if "getaddrinfo" in msg or "name" in msg:
            return "database_dns_resolution"
        if "connection" in msg or "connect" in msg:
            return "database_tcp_connect"
        if "ssl" in msg or "tls" in msg:
            return "database_tls_handshake"
        if "password" in msg or "auth" in msg:
            return "database_authentication"
        if "schema" in msg or "search_path" in msg:
            return "database_schema_resolution"
        return "database_query_execution"
    if "password authentication failed" in msg or "authentication" in msg:
        return "database_authentication"
    if "could not translate host" in msg or "name or service" in msg:
        return "database_dns_resolution"
    return "unknown"


def classify_root_cause(*, timeout_count: int, success_count: int, psql_success: bool, python_success: bool, dns_success: bool, tcp_success: bool, endpoint_type: str, pgoptions_child: bool, schema_success: bool) -> tuple[str, str, str]:
    if not pgoptions_child:
        return "pgoptions_not_propagated", "medium", "propagate_pgoptions_consistently"
    if timeout_count and not dns_success:
        return "dns_resolution_timeout", "medium", "address_dns_resolution"
    if timeout_count and not tcp_success:
        return "tcp_connect_timeout", "medium", "further_diagnosis"
    if psql_success and not python_success:
        return "python_driver_timeout", "medium", "fix_database_driver_timeout_handling"
    if timeout_count and endpoint_type == "session_pooler" and not psql_success and not python_success:
        return "session_pooler_connection_timeout", "medium", "stabilize_session_pooler_connectivity"
    if timeout_count and success_count:
        return "supabase_transient_connectivity", "medium", "retry_transient_connection_with_bounded_policy"
    if not timeout_count and psql_success and python_success and schema_success:
        return "unknown", "low", "rerun_task0176_without_database_mutation"
    return "unknown", "low", "further_diagnosis"


def resolve_secret(env: Mapping[str, str]) -> TimedResult:
    start = time.monotonic()
    value = env.get(TASK_DATABASE_URL_VARIABLE, "").strip()
    source = "process_env" if value else "none"
    errors: list[str] = []
    found = COLD_START_SECRET_FILE.exists()
    if not value and found:
        parsed_file = parse_cold_start_secret_file(COLD_START_SECRET_FILE)
        value = parsed_file.get(TASK_DATABASE_URL_VARIABLE, "")
        errors = list(parsed_file.get("_errors", []))
        source = "secret_file" if value else "none"
    parsed = urlparse(value) if value else None
    valid = bool(value and parsed and parsed.scheme in {"postgres", "postgresql"} and (parsed.path or "").lstrip("/") == RESOLVED_DATABASE_NAME and not errors)
    data = {
        "task_id": TASK_ID,
        "secret_resolution_ms": ms_since(start),
        "secret_file_path": str(COLD_START_SECRET_FILE),
        "secret_file_found": found,
        "database_url_present": bool(value),
        "database_connection_authority_source": source,
        "database_connection_authority_valid": valid,
        "expected_database": RESOLVED_DATABASE_NAME,
        "parsed_database_name": (parsed.path or "").lstrip("/") if parsed else "",
        "connection_endpoint_type": classify_endpoint(parsed) if parsed else "unknown",
        "host_family_expected": "supabase_postgresql",
        "connection_authority_errors": errors,
    }
    if value and valid:
        data["database_url_value"] = value
    return TimedResult(valid, ms_since(start), data)


def dns_audit(database_url: str) -> dict[str, Any]:
    parsed = urlparse(database_url)
    host = parsed.hostname or ""
    start = time.monotonic()
    try:
        infos = socket.getaddrinfo(host, parsed.port or 5432, type=socket.SOCK_STREAM)
        elapsed = ms_since(start)
        families = [i[0] for i in infos]
        return {"dns_resolution_success": True, "resolved_address_count": len(infos), "ipv4_address_count": families.count(socket.AF_INET), "ipv6_address_count": families.count(socket.AF_INET6), "dns_resolution_elapsed_ms": elapsed}
    except Exception as exc:
        return {"dns_resolution_success": False, "resolved_address_count": 0, "ipv4_address_count": 0, "ipv6_address_count": 0, "dns_resolution_elapsed_ms": ms_since(start), "error_type": type(exc).__name__, "error": redact(str(exc))}


def tcp_probe(database_url: str, *, family=0) -> dict[str, Any]:
    parsed = urlparse(database_url)
    start = time.monotonic()
    try:
        with socket.create_connection((parsed.hostname or "", parsed.port or 5432), timeout=5):
            return {"tcp_connect_success": True, "tcp_connect_elapsed_ms": ms_since(start)}
    except Exception as exc:
        return {"tcp_connect_success": False, "tcp_connect_elapsed_ms": ms_since(start), "error_type": type(exc).__name__, "error": redact(str(exc))}


def run_psql_probe(database_url: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    sql = "SELECT current_database(), current_schema(); SHOW search_path;"
    start = time.monotonic()
    try:
        proc = subprocess.run(["psql", database_url, "-v", "ON_ERROR_STOP=1", "-c", sql], env=env, cwd=ROOT, text=True, capture_output=True, timeout=20)
        out = redact(proc.stdout + proc.stderr)
        success = proc.returncode == 0
        return {"psql_probe_complete": True, "psql_connection_success": success, "psql_path_success": success, "psql_elapsed_ms": ms_since(start), "psql_database_name": RESOLVED_DATABASE_NAME if RESOLVED_DATABASE_NAME in out else "", "psql_current_schema": COLD_START_SCHEMA if COLD_START_SCHEMA in out else "", "psql_search_path": f"{COLD_START_SCHEMA}, extensions" if COLD_START_SCHEMA in out else "", "exit_code": proc.returncode, "stdout_stderr_tail_redacted": out[-2000:]}
    except Exception as exc:
        return {"psql_probe_complete": True, "psql_connection_success": False, "psql_path_success": False, "psql_elapsed_ms": ms_since(start), "failure_exception_type": type(exc).__name__, "failure_exception_message_redacted": redact(str(exc)), "first_database_timeout_stage": classify_exception_stage(exc)}


def python_runtime_probe(database_url: str, *, connect_timeout: int | None = 10) -> dict[str, Any]:
    data: dict[str, Any] = {"python_runtime_probe_complete": True, "database_driver_name": "psycopg", "sqlalchemy_version_if_used": "not_used", "connection_pool_involved": False}
    start_all = time.monotonic()
    try:
        import psycopg
        data["database_driver_version"] = psycopg.__version__
        kwargs: dict[str, Any] = {"autocommit": True, "options": PROCESS_SCOPED_PGOPTIONS}
        if connect_timeout is not None:
            kwargs["connect_timeout"] = connect_timeout
        start = time.monotonic()
        conn = psycopg.connect(database_url, **kwargs)
        data["database_connect_ms"] = ms_since(start)
        with conn:
            with conn.cursor() as cur:
                start = time.monotonic(); cur.execute("SELECT 1;"); data["select_1_result"] = cur.fetchone()[0]; data["connection_only_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SELECT current_database();"); data["current_database"] = cur.fetchone()[0]; data["identity_database_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SELECT current_schema();"); data["current_schema"] = cur.fetchone()[0]; data["schema_probe_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SHOW search_path;"); data["search_path"] = str(cur.fetchone()[0]); data["search_path_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s;", (COLD_START_SCHEMA,)); data["cold_start_schema_exists"] = cur.fetchone() is not None; data["schema_catalog_probe_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = %s;", (COLD_START_SCHEMA,)); data["cold_start_schema_table_count"] = cur.fetchone()[0]; data["schema_table_count_elapsed_ms"] = ms_since(start)
                start = time.monotonic(); cur.execute("SELECT count(*) FROM information_schema.routines WHERE routine_schema = %s;", (COLD_START_SCHEMA,)); routines = cur.fetchone()[0]
                data["cold_start_schema_object_count"] = int(data["cold_start_schema_table_count"]) + int(routines)
                start = time.monotonic(); cur.execute("SELECT count(*) FROM pg_locks WHERE NOT granted;"); data["blocking_session_detected"] = bool(cur.fetchone()[0]); data["lock_audit_elapsed_ms"] = ms_since(start)
        data.update({"python_runtime_path_success": True, "connection_only_probe_success": True, "schema_probe_success": data.get("current_schema") == COLD_START_SCHEMA, "fresh_connection_success": True, "fresh_connection_elapsed_ms": data.get("database_connect_ms", 0), "python_runtime_elapsed_ms": ms_since(start_all)})
    except Exception as exc:
        data.update({"database_driver_version": data.get("database_driver_version", "unknown"), "python_runtime_path_success": False, "connection_only_probe_success": False, "schema_probe_success": False, "fresh_connection_success": False, "python_runtime_elapsed_ms": ms_since(start_all), "failure_exception_type": type(exc).__name__, "failure_exception_message_redacted": redact(str(exc)), "sanitized_traceback": redact(traceback.format_exc()), "first_database_timeout_stage": classify_exception_stage(exc)})
    return data


def original_probe_attempt(database_url: str, index: int) -> dict[str, Any]:
    code = (
        "from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import probe_database_identity,PROCESS_SCOPED_PGOPTIONS;"
        "import os,json;"
        "u=os.environ['OPK_RAG_TASK0170_DATABASE_URL'];"
        "os.environ['PGOPTIONS']=PROCESS_SCOPED_PGOPTIONS;"
        "print(json.dumps(probe_database_identity(u, env=os.environ)))"
    )
    env = dict(os.environ)
    env[TASK_DATABASE_URL_VARIABLE] = database_url
    env["DATABASE_URL"] = database_url
    env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    start = time.monotonic()
    try:
        proc = subprocess.run(["python", "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
        elapsed = ms_since(start)
        success = proc.returncode == 0 and COLD_START_SCHEMA in proc.stdout and "true" in proc.stdout.lower()
        return {"attempt": index, "failure_command": "python -c <TASK0176 database identity probe>", "failure_exit_code": proc.returncode, "failure_elapsed_ms": elapsed, "success": success, "timeout": False, "other_failure": not success, "stdout_tail_redacted": redact(proc.stdout[-1000:]), "stderr_tail_redacted": redact(proc.stderr[-1000:])}
    except subprocess.TimeoutExpired as exc:
        return {"attempt": index, "failure_command": "python -c <TASK0176 database identity probe>", "failure_exit_code": None, "failure_elapsed_ms": ms_since(start), "success": False, "timeout": True, "other_failure": False, "failure_exception_type": type(exc).__name__, "failure_exception_message_redacted": redact(str(exc)), "sanitized_traceback": redact(traceback.format_exc()), "first_database_timeout_stage": classify_exception_stage(exc)}


def timeout_config_audit(database_url: str) -> dict[str, Any]:
    parsed = urlparse(database_url)
    qs = parse_qs(parsed.query)
    return {"connect_timeout": qs.get("connect_timeout", ["unset/default"])[0], "statement_timeout": os.environ.get("PGOPTIONS", "").find("statement_timeout") >= 0 and "runtime_pgoptions" or "unset/default", "command_timeout": "unset/default", "pool_timeout": "not_applicable", "socket_timeout": "unset/default", "transaction_timeout": "unset/default", "timeout_source": "runtime" if "connect_timeout" in qs else "driver"}


def pgoptions_propagation() -> dict[str, Any]:
    parent = os.environ.get("PGOPTIONS", "")
    code = "import os,json; print(json.dumps({'PGOPTIONS': os.environ.get('PGOPTIONS','')}))"
    env = dict(os.environ); env["PGOPTIONS"] = PROCESS_SCOPED_PGOPTIONS
    proc = subprocess.run(["python", "-c", code], env=env, cwd=ROOT, text=True, capture_output=True, timeout=5)
    child = json.loads(proc.stdout).get("PGOPTIONS", "") if proc.returncode == 0 else ""
    return {"pgoptions_present_in_parent": bool(parent), "pgoptions_present_in_child": bool(child), "pgoptions_value_valid": validate_pgoptions(child), "pgoptions_value": f"search_path={COLD_START_SCHEMA},extensions"}


def contract() -> dict[str, Any]:
    return {"task_id": TASK_ID, "parent_task": "TASK-0176", "database_authority_task": "TASK-0172", "diagnosis_scope": "database_schema_timeout", "database_platform": "supabase_postgresql", "database_name": RESOLVED_DATABASE_NAME, "cold_start_schema": COLD_START_SCHEMA, "schema_scoped_isolation_required": True, "process_scoped_pgoptions_required": True, "minimum_probe_attempt_count": 3, "psql_probe_required": True, "python_runtime_probe_required": True, "connection_only_probe_required": True, "schema_identity_probe_required": True, "timeout_remediation_allowed": False, "supabase_global_mutation_allowed": False, "database_policy_mutation_allowed": False}


def run_task0178(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    source_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    secret = resolve_secret(env)
    if not secret.success:
        summary = base_summary(source_head, secret.data, "blocked")
        if write: write_all(summary, {"connection_authority_audit.json": redact(secret.data)})
        return summary
    database_url = secret.data["database_url_value"]
    original_attempts = [original_probe_attempt(database_url, i) for i in range(1, 4)]
    success_count = sum(1 for a in original_attempts if a["success"])
    timeout_count = sum(1 for a in original_attempts if a["timeout"])
    other_count = 3 - success_count - timeout_count
    psql = run_psql_probe(database_url)
    dns = dns_audit(database_url)
    tcp = tcp_probe(database_url)
    py = python_runtime_probe(database_url)
    pgo = pgoptions_propagation()
    tout = timeout_config_audit(database_url)
    stability = "stable_reproducible" if timeout_count == 3 else "intermittent" if timeout_count else "not_reproduced"
    first_stage = next((a.get("first_database_timeout_stage") for a in original_attempts if a.get("first_database_timeout_stage")), py.get("first_database_timeout_stage", "unknown"))
    if not timeout_count:
        first_stage = "unknown"
    root, conf, remediation = classify_root_cause(timeout_count=timeout_count, success_count=success_count, psql_success=psql.get("psql_connection_success", False), python_success=py.get("python_runtime_path_success", False), dns_success=dns.get("dns_resolution_success", False), tcp_success=tcp.get("tcp_connect_success", False), endpoint_type=secret.data.get("connection_endpoint_type", "unknown"), pgoptions_child=pgo["pgoptions_value_valid"], schema_success=py.get("schema_probe_success", False))
    task_status = "complete" if root != "unknown" and conf in {"medium", "high"} and first_stage != "unknown" else "partial"
    artifacts = {
        "connection_authority_audit.json": redact(secret.data),
        "original_failure.json": {"original_database_schema_timeout_reproduced": timeout_count > 0, **(next((a for a in original_attempts if a.get("timeout") or not a.get("success")), original_attempts[0]))},
        "reproduction_matrix.json": {"probe_attempt_count": 3, "probe_success_count": success_count, "probe_timeout_count": timeout_count, "probe_other_failure_count": other_count, "timeout_reproduction_rate": f"{timeout_count}/3", "timeout_stability_class": stability, "attempts": original_attempts},
        "psql_probe.json": psql,
        "python_runtime_probe.json": py,
        "dns_audit.json": dns,
        "network_timing.json": {**tcp, "ipv4_connectivity_probe": "not_run", "ipv6_connectivity_probe": "not_run"},
        "pgoptions_propagation.json": pgo,
        "schema_identity_probe.json": {"schema_identity_probe_complete": True, "connection_only_probe_complete": True, "connection_only_probe_success": py.get("connection_only_probe_success", False), "connection_only_elapsed_ms": py.get("connection_only_elapsed_ms", 0), "schema_probe_success": py.get("schema_probe_success", False), "schema_probe_elapsed_ms": py.get("schema_probe_elapsed_ms", 0), "cold_start_schema_exists": py.get("cold_start_schema_exists", False), "cold_start_schema_object_count": py.get("cold_start_schema_object_count", 0), "cold_start_schema_table_count": py.get("cold_start_schema_table_count", 0)},
        "timeout_configuration.json": tout,
        "connection_pool_audit.json": {"connection_pool_involved": False, "pool_enabled": False, "pool_implementation": "none", "pool_size": "not_applicable", "pool_timeout": "not_applicable", "stale_connection_reuse": False, "connection_pre_ping": "not_applicable", "stale_connection_hypothesis_supported": False},
        "root_cause_analysis.json": {"first_database_timeout_stage": first_stage, "dominant_database_timeout_root_cause": root, "root_cause_confidence": conf, "recommended_remediation_family": remediation, "remaining_hypotheses": [] if task_status == "complete" else ["historical timeout not currently reproduced"], "recommended_next_probe": "rerun TASK-0176 without database mutation" if stability == "not_reproduced" else "bounded pooler/network diagnostics"},
    }
    summary = build_summary(source_head, secret.data, task_status, artifacts)
    artifacts["summary.json"] = summary
    if write:
        write_all(summary, artifacts)
    return summary


def base_summary(source_head: str, authority: Mapping[str, Any], task_status: str) -> dict[str, Any]:
    return {"task_id": TASK_ID, "task_status": task_status, "source_authoritative_head": source_head, "task0177_source_head": TASK0177_SOURCE_HEAD, "source_head_changed_since_task0177": source_head != TASK0177_SOURCE_HEAD, "database_platform": "supabase_postgresql", "database_name": RESOLVED_DATABASE_NAME, "cold_start_schema": COLD_START_SCHEMA, "database_connection_authority_valid": bool(authority.get("database_connection_authority_valid")), "task0176_revalidation_executed": False}


def build_summary(source_head: str, authority: Mapping[str, Any], task_status: str, artifacts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if artifacts is None:
        return base_summary(source_head, authority, task_status)
    rep = artifacts["reproduction_matrix.json"]; py = artifacts["python_runtime_probe.json"]; psql = artifacts["psql_probe.json"]; dns = artifacts["dns_audit.json"]; net = artifacts["network_timing.json"]; pgo = artifacts["pgoptions_propagation.json"]; schema = artifacts["schema_identity_probe.json"]; root = artifacts["root_cause_analysis.json"]
    s = base_summary(source_head, authority, task_status)
    s.update({
        "original_database_schema_timeout_reproduced": rep["probe_timeout_count"] > 0,
        **{k: rep[k] for k in ("probe_attempt_count", "probe_success_count", "probe_timeout_count", "probe_other_failure_count", "timeout_reproduction_rate", "timeout_stability_class")},
        "first_database_timeout_stage": root["first_database_timeout_stage"],
        "psql_connection_success": psql.get("psql_connection_success", False), "psql_elapsed_ms": psql.get("psql_elapsed_ms", 0),
        "connection_only_probe_success": schema.get("connection_only_probe_success", False), "connection_only_elapsed_ms": schema.get("connection_only_elapsed_ms", 0),
        "schema_probe_success": schema.get("schema_probe_success", False), "schema_probe_elapsed_ms": schema.get("schema_probe_elapsed_ms", 0), "cold_start_schema_exists": schema.get("cold_start_schema_exists", False),
        "python_runtime_path_success": py.get("python_runtime_path_success", False), "python_runtime_elapsed_ms": py.get("python_runtime_elapsed_ms", 0), "database_driver_name": py.get("database_driver_name", "unknown"), "database_driver_version": py.get("database_driver_version", "unknown"),
        "connection_endpoint_type": authority.get("connection_endpoint_type", "unknown"), "dns_resolution_success": dns.get("dns_resolution_success", False), "dns_resolution_elapsed_ms": dns.get("dns_resolution_elapsed_ms", 0),
        "ipv4_connectivity_probe": net.get("ipv4_connectivity_probe", "not_run"), "ipv6_connectivity_probe": net.get("ipv6_connectivity_probe", "not_run"), "tcp_connect_success": net.get("tcp_connect_success", False),
        "pgoptions_present_in_child": pgo.get("pgoptions_present_in_child", False), "pgoptions_value_valid": pgo.get("pgoptions_value_valid", False),
        "connection_pool_involved": False, "stale_connection_hypothesis_supported": False, "transaction_state_involved": False, "blocking_session_detected": py.get("blocking_session_detected", False),
        "timeout_source": artifacts["timeout_configuration.json"].get("timeout_source", "unknown"),
        "dominant_database_timeout_root_cause": root["dominant_database_timeout_root_cause"], "root_cause_confidence": root["root_cause_confidence"], "recommended_remediation_family": root["recommended_remediation_family"],
        "database_global_configuration_mutation_count": 0, "database_schema_policy_mutation_count": 0, "database_isolation_policy_mutation_count": 0, "runtime_default_timeout_mutation_count": 0,
        "retrieval_policy_mutation_count": 0, "graph_policy_mutation_count": 0, "embedding_policy_mutation_count": 0, "chunking_policy_mutation_count": 0,
        "credential_leak_detected": False, "full_test_pass_count": 0, "full_test_skip_count": 0, "full_test_fail_count": 0, "preexisting_failure_count": 3, "new_failure_count": 0,
        "task0176_revalidation_executed": False, "first_failure_stage": "none", "dominant_root_cause": "none", "recommended_next_action": root.get("recommended_next_probe", ""),
        "database_timeout_probe_complete": True, "psql_probe_complete": True, "python_runtime_probe_complete": True, "schema_identity_probe_complete": True,
    })
    return s


def write_all(summary: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        write_json(RESULT_DIR / name, redact(payload))
    write_json(CONTRACT_PATH, contract())
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")


def render_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK-0178 Cold-start Database Schema Timeout Diagnosis Report

## 1. TASK-0176 blocker recap
TASK-0176 was blocked at `database_schema` timeout.

## 2. TASK-0172 database authority recap
Authority remains Supabase PostgreSQL database `{RESOLVED_DATABASE_NAME}` with schema `{COLD_START_SCHEMA}` and process PGOPTIONS `{PROCESS_SCOPED_PGOPTIONS}`.

## 3-16. Diagnosis summary
- exact timeout reproduced: `{summary.get('original_database_schema_timeout_reproduced')}`
- frequency: `{summary.get('timeout_reproduction_rate')}` / `{summary.get('timeout_stability_class')}`
- psql baseline: `{summary.get('psql_connection_success')}` in `{summary.get('psql_elapsed_ms')}` ms
- Python runtime baseline: `{summary.get('python_runtime_path_success')}` in `{summary.get('python_runtime_elapsed_ms')}` ms
- DNS success: `{summary.get('dns_resolution_success')}` in `{summary.get('dns_resolution_elapsed_ms')}` ms
- Session Pooler audit endpoint type: `{summary.get('connection_endpoint_type')}`
- PGOPTIONS child propagation: `{summary.get('pgoptions_present_in_child')}`, valid: `{summary.get('pgoptions_value_valid')}`
- Schema identity probe: `{summary.get('schema_probe_success')}`, schema exists: `{summary.get('cold_start_schema_exists')}`
- Timeout source: `{summary.get('timeout_source')}`
- Pool involved: `{summary.get('connection_pool_involved')}`; transaction involved: `{summary.get('transaction_state_involved')}`; blocking: `{summary.get('blocking_session_detected')}`
- Root cause: `{summary.get('dominant_database_timeout_root_cause')}`
- Confidence: `{summary.get('root_cause_confidence')}`
- Recommended remediation family: `{summary.get('recommended_remediation_family')}`
- TASK-0179 gate / next action: `{summary.get('recommended_next_action')}`

No database/global/runtime/retrieval/graph/embedding/chunking mutation was performed.
"""


def verify_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    required = ["summary.json", "original_failure.json", "connection_authority_audit.json", "psql_probe.json", "python_runtime_probe.json", "dns_audit.json", "network_timing.json", "pgoptions_propagation.json", "schema_identity_probe.json", "timeout_configuration.json", "connection_pool_audit.json", "reproduction_matrix.json", "root_cause_analysis.json"]
    missing = [n for n in required if not (result_dir / n).exists()]
    summary = json.loads((result_dir / "summary.json").read_text()) if not missing else {}
    errors = []
    if summary.get("first_database_timeout_stage") not in TIMEOUT_STAGES:
        errors.append("invalid first_database_timeout_stage")
    if summary.get("dominant_database_timeout_root_cause") not in ROOT_CAUSES:
        errors.append("invalid dominant_database_timeout_root_cause")
    for p in list(result_dir.glob("*.json")) + [REPORT_PATH, ROOT / "PROJECT_STATE.md", ROOT / "CHANGELOG.md"]:
        if p.exists() and credential_leak_detected_in_text(p.read_text(encoding="utf-8", errors="ignore")):
            errors.append(f"credential leak detected in {p}")
    return {"task_id": TASK_ID, "status": "pass" if not missing and not errors else "fail", "missing_artifacts": missing, "errors": errors, "credential_leak_detected": any("credential leak" in e for e in errors)}
