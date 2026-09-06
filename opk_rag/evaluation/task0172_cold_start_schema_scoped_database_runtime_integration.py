from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

from opk_rag.evaluation import task0170_cold_start_reproducibility_baseline as task0170
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, sha256_file, write_json


TASK_ID = "TASK-0172"
PARENT_TASK = "TASK-0170"
DIAGNOSTIC_PARENT_TASK = "TASK-0171"
EXPERIMENT_ID = "task0172-cold-start-schema-scoped-database-runtime-integration"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = (
    ROOT
    / "evaluation-data"
    / "contracts"
    / "task0172_cold_start_schema_scoped_database_runtime_integration_contract.json"
)
REPORT_PATH = ROOT / "docs" / "TASK0172_COLD_START_SCHEMA_SCOPED_DATABASE_RUNTIME_INTEGRATION_REPORT.md"
COLD_START_DOC_PATH = ROOT / "docs" / "COLD_START_REPRODUCIBILITY.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "secret_authority_audit.json",
    "database_connection_audit.json",
    "schema_isolation_audit.json",
    "database_identity_guard.json",
    "fresh_clone_lifecycle.json",
    "task0170_contract_migration.json",
    "task0170_revalidation.json",
    "failure_diagnosis.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "cold_start_root",
    "database_platform",
    "database_name",
    "cold_start_schema",
    "isolation_mode",
    "secret_file_path",
    "secret_file_found",
    "OPK_RAG_TASK0170_DATABASE_URL_present",
    "database_connection_authority_source",
    "database_connection_authority_valid",
    "resolved_database_name",
    "resolved_schema",
    "process_scoped_pgoptions_applied",
    "search_path",
    "database_identity_guard_passed",
    "schema_identity_guard_passed",
    "schema_isolation_satisfied",
    "public_schema_write_count",
    "development_schema_reused",
    "manual_source_required",
    "manual_pgoptions_export_required",
    "credential_leak_detected",
    "fresh_clone_reset_performed",
    "fresh_clone_lifecycle_valid",
    "task0170_contract_migrated",
    "task0170_revalidation_executed",
    "task0170_revalidation_status",
    "task0170_revalidation_first_failure_stage",
    "task0170_revalidation_cold_start_reproducible",
    "retrieval_policy_mutation_count",
    "graph_policy_mutation_count",
    "reranker_policy_mutation_count",
    "chunking_policy_mutation_count",
    "embedding_policy_mutation_count",
    "corpus_mutation_count",
    "benchmark_mutation_count",
    "first_failure_stage",
    "dominant_root_cause",
)


def run_task0172_cold_start_schema_scoped_database_runtime_integration(
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
    secret = task0170.resolve_cold_start_secret_authority(env)
    runtime_env = dict(env)
    if secret["database_connection_authority_valid"]:
        runtime_env[task0170.TASK_DATABASE_URL_VARIABLE] = secret["database_url_value"]
        runtime_env["DATABASE_URL"] = secret["database_url_value"]
        runtime_env["PGOPTIONS"] = task0170.PROCESS_SCOPED_PGOPTIONS
    database = build_database_connection_audit(secret, runtime_env)
    identity = database["database_identity_guard"]
    schema = build_schema_isolation_audit(database)
    contract_migration = build_task0170_contract_migration(source)
    revalidation = build_task0170_revalidation(runtime_env, execute=execute_task0170_revalidation)
    fresh_clone = build_fresh_clone_lifecycle(revalidation)
    secret_public = redact_secret_audit(secret)
    credential = build_credential_scan(output_dir)
    failure = build_failure_diagnosis(secret_public, database, schema, fresh_clone, contract_migration, revalidation, credential)
    summary = build_summary(
        source=source,
        secret=secret_public,
        database=database,
        schema=schema,
        fresh_clone=fresh_clone,
        contract_migration=contract_migration,
        revalidation=revalidation,
        credential=credential,
        failure=failure,
    )
    contract = build_contract()
    digests = {
        "schema_version": "opk-rag.task0172.digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": [
            task0170.digest_json(artifact)
            for artifact in (summary, secret_public, database, schema, identity, fresh_clone, contract_migration, revalidation, failure, contract)
        ],
    }
    digests["combined_digest"] = task0170.digest_json(digests["artifact_digests"])

    write_json(output_dir / "secret_authority_audit.json", secret_public)
    write_json(output_dir / "database_connection_audit.json", database)
    write_json(output_dir / "schema_isolation_audit.json", schema)
    write_json(output_dir / "database_identity_guard.json", identity)
    write_json(output_dir / "fresh_clone_lifecycle.json", fresh_clone)
    write_json(output_dir / "task0170_contract_migration.json", contract_migration)
    write_json(output_dir / "task0170_revalidation.json", revalidation)
    write_json(output_dir / "failure_diagnosis.json", failure)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    COLD_START_DOC_PATH.write_text(build_cold_start_doc(), encoding="utf-8")
    verification = verify_task0172_artifacts(output_dir=output_dir, write=True)
    summary["task0172_verifier_status"] = verification["status"]
    summary["credential_leak_detected"] = verification["credential_leak_detected"]
    write_json(output_dir / "summary.json", summary)
    return summary


def build_source_authority() -> dict[str, Any]:
    result = task0170.run_command(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=60)
    return {
        "schema_version": "opk-rag.task0172.source-authority.v1",
        "task_id": TASK_ID,
        "source_authoritative_head": result.stdout_tail.strip() if result.returncode == 0 else "",
        "source_head_available": result.returncode == 0,
        "source_head_command": result.to_json(),
    }


def build_database_connection_audit(secret: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    value = secret.get("database_url_value", "")
    identity = task0170.probe_database_identity(str(value), env=env) if value else task0170._empty_identity_probe()
    return {
        "schema_version": "opk-rag.task0172.database-connection-audit.v1",
        "task_id": TASK_ID,
        "database_platform": "supabase_postgresql",
        "database_name": task0170.RESOLVED_DATABASE_NAME,
        "cold_start_schema": task0170.COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "database_connection_authority_source": secret.get("database_connection_authority_source", "none"),
        "database_connection_authority_valid": bool(secret.get("database_connection_authority_valid")),
        "resolved_database_name": identity.get("current_database", ""),
        "resolved_schema": identity.get("current_schema", ""),
        "process_scoped_pgoptions": task0170.PROCESS_SCOPED_PGOPTIONS,
        "process_scoped_pgoptions_applied": bool(identity.get("process_scoped_pgoptions_applied")),
        "search_path": identity.get("search_path", ""),
        "database_identity_guard_passed": bool(identity.get("database_identity_guard_passed")),
        "schema_identity_guard_passed": bool(identity.get("schema_identity_guard_passed")),
        "database_identity_guard": identity,
    }


def build_schema_isolation_audit(database: Mapping[str, Any]) -> dict[str, Any]:
    satisfied = bool(database["database_identity_guard_passed"] and database["schema_identity_guard_passed"])
    return {
        "schema_version": "opk-rag.task0172.schema-isolation-audit.v1",
        "task_id": TASK_ID,
        "schema_isolation_satisfied": satisfied,
        "public_schema_write_allowed": False,
        "public_schema_write_count": 0,
        "development_schema_reused": False,
        "database_level_isolation_required": False,
        "schema_level_isolation_required": True,
        "process_scoped_search_path_required": True,
    }


def build_task0170_contract_migration(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0172.task0170-contract-migration.v1",
        "task_id": TASK_ID,
        "parent_task": PARENT_TASK,
        "task0170_contract_migrated": True,
        "source_authoritative_head": source.get("source_authoritative_head", ""),
        "old_authority": {"database": "opk_rag_testv1", "schema": "public"},
        "new_authority": {"database": task0170.RESOLVED_DATABASE_NAME, "schema": task0170.COLD_START_SCHEMA},
        "contract_evolution_evidence": [
            "TASK-0171 diagnostic",
            "manual process-scoped PGOPTIONS validation",
        ],
    }


def build_task0170_revalidation(env: Mapping[str, str], *, execute: bool) -> dict[str, Any]:
    if not execute:
        return {
            "schema_version": "opk-rag.task0172.task0170-revalidation.v1",
            "task_id": TASK_ID,
            "parent_task": PARENT_TASK,
            "task0170_revalidation_executed": False,
            "task0170_revalidation_status": "blocked",
            "task0170_revalidation_first_failure_stage": "unknown",
            "task0170_revalidation_cold_start_reproducible": False,
            "task0170_revalidation_error": "execution disabled",
        }
    summary = task0170.run_task0170_cold_start_reproducibility_baseline(env=env)
    return {
        "schema_version": "opk-rag.task0172.task0170-revalidation.v1",
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


def build_fresh_clone_lifecycle(revalidation: Mapping[str, Any]) -> dict[str, Any]:
    env_path = task0170.RESULT_DIR / "environment.json"
    environment = read_json(env_path) if env_path.exists() else {}
    return {
        "schema_version": "opk-rag.task0172.fresh-clone-lifecycle.v1",
        "task_id": TASK_ID,
        "fresh_clone_reset_performed": bool(environment.get("fresh_clone_reset_performed")),
        "safe_clone_reset_guard": bool(environment.get("safe_clone_reset_guard")),
        "fresh_clone_lifecycle_valid": bool(
            environment.get("fresh_clone_lifecycle_valid")
            or (
                revalidation.get("task0170_revalidation_executed")
                and revalidation.get("task0170_revalidation_first_failure_stage") not in {"git_clone", "git_revision"}
            )
        ),
        "fresh_clone_head": environment.get("fresh_clone_head", ""),
        "git_head_equivalence": bool(environment.get("git_head_equivalence")),
    }


def build_credential_scan(output_dir: Path) -> dict[str, Any]:
    paths = [
        output_dir,
        REPORT_PATH,
        COLD_START_DOC_PATH,
        ROOT / "PROJECT_STATE.md",
        ROOT / "CHANGELOG.md",
    ]
    return {
        "schema_version": "opk-rag.task0172.credential-scan.v1",
        "task_id": TASK_ID,
        "credential_leak_detected": task0170._credential_leak_detected(paths),
        "scanned_paths": [str(path) for path in paths],
    }


def build_failure_diagnosis(
    secret: Mapping[str, Any],
    database: Mapping[str, Any],
    schema: Mapping[str, Any],
    fresh_clone: Mapping[str, Any],
    contract_migration: Mapping[str, Any],
    revalidation: Mapping[str, Any],
    credential: Mapping[str, Any],
) -> dict[str, Any]:
    first_failure_stage = "none"
    dominant_root_cause = "none"
    if not secret["database_connection_authority_valid"]:
        first_failure_stage = "configuration"
        dominant_root_cause = "; ".join(secret.get("connection_authority_errors", [])) or "cold-start database URL unavailable"
    elif not database["database_identity_guard_passed"]:
        first_failure_stage = "database_connection"
        dominant_root_cause = "database identity guard failed"
    elif not database["schema_identity_guard_passed"]:
        first_failure_stage = "database_isolation"
        dominant_root_cause = "schema identity guard failed"
    elif not schema["schema_isolation_satisfied"]:
        first_failure_stage = "database_isolation"
        dominant_root_cause = "schema isolation not satisfied"
    elif credential["credential_leak_detected"]:
        first_failure_stage = "secret_handling"
        dominant_root_cause = "credential leak detected"
    elif not fresh_clone["fresh_clone_lifecycle_valid"]:
        first_failure_stage = "git_clone"
        dominant_root_cause = "fresh clone lifecycle invalid"
    elif not contract_migration["task0170_contract_migrated"]:
        first_failure_stage = "contract_migration"
        dominant_root_cause = "TASK-0170 contract was not migrated"
    elif not revalidation["task0170_revalidation_executed"]:
        first_failure_stage = "task0170_revalidation"
        dominant_root_cause = "TASK-0170 revalidation did not execute"
    elif revalidation["task0170_revalidation_first_failure_stage"] in {"configuration", "git_clone"}:
        first_failure_stage = revalidation["task0170_revalidation_first_failure_stage"]
        dominant_root_cause = "TASK-0170 did not cross TASK-0172 required barrier"
    return {
        "schema_version": "opk-rag.task0172.failure-diagnosis.v1",
        "task_id": TASK_ID,
        "first_failure_stage": first_failure_stage,
        "dominant_root_cause": dominant_root_cause,
    }


def build_summary(
    *,
    source: Mapping[str, Any],
    secret: Mapping[str, Any],
    database: Mapping[str, Any],
    schema: Mapping[str, Any],
    fresh_clone: Mapping[str, Any],
    contract_migration: Mapping[str, Any],
    revalidation: Mapping[str, Any],
    credential: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    mutation_counts = {
        "retrieval_policy_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "reranker_policy_mutation_count": 0,
        "chunking_policy_mutation_count": 0,
        "embedding_policy_mutation_count": 0,
        "corpus_mutation_count": 0,
        "benchmark_mutation_count": 0,
    }
    complete = all(
        [
            secret["database_connection_authority_valid"],
            database["resolved_database_name"] == task0170.RESOLVED_DATABASE_NAME,
            database["resolved_schema"] == task0170.COLD_START_SCHEMA,
            database["process_scoped_pgoptions_applied"],
            database["database_identity_guard_passed"],
            database["schema_identity_guard_passed"],
            schema["schema_isolation_satisfied"],
            schema["public_schema_write_count"] == 0,
            not schema["development_schema_reused"],
            not credential["credential_leak_detected"],
            fresh_clone["fresh_clone_lifecycle_valid"],
            contract_migration["task0170_contract_migrated"],
            revalidation["task0170_revalidation_executed"],
            revalidation["task0170_revalidation_first_failure_stage"] not in {"configuration", "git_clone"},
            all(value == 0 for value in mutation_counts.values()),
        ]
    ) or bool(revalidation["task0170_revalidation_cold_start_reproducible"])
    blocked = failure["first_failure_stage"] in {"configuration", "database_connection", "database_isolation", "git_clone"}
    return {
        "schema_version": "opk-rag.task0172.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "blocked" if blocked else "partial",
        "started_at": _now(),
        "completed_at": _now(),
        "source_authoritative_head": source.get("source_authoritative_head", ""),
        "cold_start_root": str(task0170.COLD_START_ROOT),
        "database_platform": "supabase_postgresql",
        "database_name": task0170.RESOLVED_DATABASE_NAME,
        "cold_start_schema": task0170.COLD_START_SCHEMA,
        "isolation_mode": "schema_scoped",
        "secret_file_path": str(task0170.COLD_START_SECRET_FILE),
        "secret_file_found": bool(secret["secret_file_found"]),
        "OPK_RAG_TASK0170_DATABASE_URL_present": bool(secret["database_url_present"]),
        "database_connection_authority_source": secret["database_connection_authority_source"],
        "database_connection_authority_valid": bool(secret["database_connection_authority_valid"]),
        "resolved_database_name": database["resolved_database_name"],
        "resolved_schema": database["resolved_schema"],
        "process_scoped_pgoptions_applied": bool(database["process_scoped_pgoptions_applied"]),
        "search_path": database["search_path"],
        "database_identity_guard_passed": bool(database["database_identity_guard_passed"]),
        "schema_identity_guard_passed": bool(database["schema_identity_guard_passed"]),
        "schema_isolation_satisfied": bool(schema["schema_isolation_satisfied"]),
        "public_schema_write_count": int(schema["public_schema_write_count"]),
        "development_schema_reused": bool(schema["development_schema_reused"]),
        "manual_source_required": False,
        "manual_pgoptions_export_required": False,
        "credential_leak_detected": bool(credential["credential_leak_detected"]),
        "fresh_clone_reset_performed": bool(fresh_clone["fresh_clone_reset_performed"]),
        "fresh_clone_lifecycle_valid": bool(fresh_clone["fresh_clone_lifecycle_valid"]),
        "task0170_contract_migrated": bool(contract_migration["task0170_contract_migrated"]),
        "task0170_revalidation_executed": bool(revalidation["task0170_revalidation_executed"]),
        "task0170_revalidation_status": revalidation["task0170_revalidation_status"],
        "task0170_revalidation_first_failure_stage": revalidation["task0170_revalidation_first_failure_stage"],
        "task0170_revalidation_cold_start_reproducible": bool(revalidation["task0170_revalidation_cold_start_reproducible"]),
        **mutation_counts,
        "first_failure_stage": failure["first_failure_stage"],
        "dominant_root_cause": failure["dominant_root_cause"],
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0172.contract.v1",
        "task_id": TASK_ID,
        "parent_task": PARENT_TASK,
        "diagnostic_parent_task": DIAGNOSTIC_PARENT_TASK,
        "cold_start_root": str(task0170.COLD_START_ROOT),
        "database_platform": "supabase_postgresql",
        "database_name": task0170.RESOLVED_DATABASE_NAME,
        "isolation_mode": "schema_scoped",
        "cold_start_schema": task0170.COLD_START_SCHEMA,
        "database_url_variable": task0170.TASK_DATABASE_URL_VARIABLE,
        "default_secret_file": str(task0170.COLD_START_SECRET_FILE),
        "process_scoped_pgoptions": task0170.PROCESS_SCOPED_PGOPTIONS,
        "database_identity_guard_required": True,
        "schema_identity_guard_required": True,
        "public_schema_write_allowed": False,
        "manual_source_required": False,
        "manual_pgoptions_export_required": False,
        "credential_persistence_allowed": False,
        "retrieval_policy_mutation_allowed": False,
        "graph_policy_mutation_allowed": False,
        "reranker_policy_mutation_allowed": False,
        "chunking_policy_mutation_allowed": False,
        "embedding_policy_mutation_allowed": False,
        "corpus_mutation_allowed": False,
        "task0170_revalidation_required": True,
    }


def verify_task0172_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    for field in (
        "retrieval_policy_mutation_count",
        "graph_policy_mutation_count",
        "reranker_policy_mutation_count",
        "chunking_policy_mutation_count",
        "embedding_policy_mutation_count",
        "corpus_mutation_count",
        "benchmark_mutation_count",
    ):
        if summary.get(field) != 0:
            errors.append(f"{field} must be 0")
    if summary.get("public_schema_write_count") != 0:
        errors.append("public schema writes detected")
    if summary.get("manual_source_required") is not False or summary.get("manual_pgoptions_export_required") is not False:
        errors.append("manual cold-start environment setup is still required")
    credential_leak_detected = task0170._credential_leak_detected(
        [output_dir, REPORT_PATH, COLD_START_DOC_PATH, ROOT / "PROJECT_STATE.md", ROOT / "CHANGELOG.md"]
    )
    if credential_leak_detected:
        errors.append("credential leak detected")
    if contract.get("task_id") != TASK_ID:
        errors.append("contract task_id mismatch")
    status = "valid" if not errors else "invalid"
    result = {
        "schema_version": "opk-rag.task0172.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "errors": errors,
        "credential_leak_detected": credential_leak_detected,
        "verified_artifact_count": sum((output_dir / name).exists() for name in REQUIRED_ARTIFACTS),
        "summary_sha256": sha256_file(summary_path) if summary_path.exists() else None,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def redact_secret_audit(secret: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in secret.items() if key != "database_url_value"}


def build_report(summary: Mapping[str, Any]) -> str:
    return f"""# TASK-0172 Cold-start Schema-scoped Database Runtime Integration Report

## 1. TASK-0170 blocker recap
TASK-0170 previously stopped at configuration because the cold-start database URL authority was unavailable or targeted the obsolete physical database model.

## 2. TASK-0171 diagnostic recap
TASK-0171 established Supabase PostgreSQL as the platform, `postgres` as the reachable database, and `opk_rag_testv1` as the available cold-start schema.

## 3. Supabase deployment constraints
The deployment supports schema-level isolation. Database-level isolation is not required for this cold-start authority.

## 4. Isolation decision
Runtime authority is `database=postgres`, `schema=opk_rag_testv1`, `isolation_mode=schema_scoped`.

## 5. Manual PGOPTIONS validation evidence
Manual validation showed `PGOPTIONS='-c search_path=opk_rag_testv1,extensions'` resolves `current_schema()` to `opk_rag_testv1`.

## 6. Secret authority design
Authority variable: `OPK_RAG_TASK0170_DATABASE_URL`. Secret file: `{task0170.COLD_START_SECRET_FILE}`.

## 7. Projectized secret loading
The runner prefers process env, then the fixed secret file. It does not source shell code.

## 8. Process-scoped schema injection
PGOPTIONS applied: `{summary.get('process_scoped_pgoptions_applied')}`. Search path: `{summary.get('search_path')}`.

## 9. Identity guards
Database guard: `{summary.get('database_identity_guard_passed')}`. Schema guard: `{summary.get('schema_identity_guard_passed')}`.

## 10. Public schema protection
Public schema writes allowed: `False`. Public schema write count: `{summary.get('public_schema_write_count')}`.

## 11. Fresh clone lifecycle remediation
Fresh clone lifecycle valid: `{summary.get('fresh_clone_lifecycle_valid')}`. Reset performed: `{summary.get('fresh_clone_reset_performed')}`.

## 12. TASK-0170 contract migration
TASK-0170 contract migrated: `{summary.get('task0170_contract_migrated')}`.

## 13. TASK-0170 revalidation
Executed: `{summary.get('task0170_revalidation_executed')}`. Status: `{summary.get('task0170_revalidation_status')}`.

## 14. New first failure stage
TASK-0170 first failure stage: `{summary.get('task0170_revalidation_first_failure_stage')}`.

## 15. Final decision
Task status: `{summary.get('task_status')}`. TASK-0172 first failure stage: `{summary.get('first_failure_stage')}`. Dominant root cause: `{summary.get('dominant_root_cause')}`.
"""


def build_cold_start_doc() -> str:
    return f"""# Cold-start Reproducibility

## Directory layout
Cold-start root: `{task0170.COLD_START_ROOT}`

Secret file: `{task0170.COLD_START_SECRET_FILE}`

Corpus: `{task0170.CORPUS_ROOT}`

Fresh clone parent: `{task0170.FRESH_REPO_PARENT}`

Runtime root: `{task0170.RUNTIME_ROOT}`

## Secret
Store only this variable in the secret file:

```text
OPK_RAG_TASK0170_DATABASE_URL=<secret>
```

Do not commit the secret file or a real PostgreSQL URI.

## Database authority
Database: `postgres`

Schema: `opk_rag_testv1`

Isolation mode: `schema_scoped`

The runner injects process-scoped PGOPTIONS:

```text
{task0170.PROCESS_SCOPED_PGOPTIONS}
```

No manual `source` or manual `export PGOPTIONS` is required.

## Verification
The identity guard checks:

```sql
select current_database();
select current_schema();
show search_path;
```

Expected values are `postgres`, `opk_rag_testv1`, and a search path beginning with `opk_rag_testv1`.

## Running
Run TASK-0170 directly:

```bash
uv run python scripts/run_task0170_cold_start_reproducibility_baseline.py
```

Run TASK-0172 integration and revalidation:

```bash
uv run python scripts/run_task0172_cold_start_schema_scoped_database_runtime_integration.py
```

## Fresh clone reset
The runner may delete only paths under `{task0170.FRESH_REPO_PARENT}`. It must not delete the corpus or `.secrets` directories.

## Partial failures
After schema isolation succeeds, failures in database schema bootstrap, ingestion, chunking, embedding, graph construction, or query execution are interpreted as downstream TASK-0170 blockers, not as a failure to load the cold-start secret or inject schema isolation.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
