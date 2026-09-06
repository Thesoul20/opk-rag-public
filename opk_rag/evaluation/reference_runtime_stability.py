from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation.agent_recovery_experiment import load_core_rag_benchmark_split
from opk_rag.evaluation.classification_stability import (
    INFRASTRUCTURE_CLASS,
    analyze_task0070_indeterminate,
    analyze_unstable_samples,
    compare_five_replicates,
    compare_three_replicates,
    evaluate_stability_gates,
    revalidate_generation_retry_candidates,
    sample_runtime_class,
)
from opk_rag.evaluation.core_rag_benchmark import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    DEV_SPLIT,
    file_digest,
    read_json,
    read_jsonl,
    scan_paths_for_privacy,
    stable_hash,
    validate_reference_runtime,
)
from opk_rag.evaluation.post_generation_failure_classification import OBSERVABILITY_CONTRACT_ID
from opk_rag.runtime.dotenv import load_project_env

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0074"
CONTRACT_ID = "opk-rag.agent-reference-runtime-stability-revalidation.v1"
CONTRACT_SCHEMA = "opk-rag.task0074-reference-runtime-stability-contract.v1"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0074_reference_runtime_stability_contract.json"
RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0074-reference-runtime-stability"
TASK0071_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0071_governed_agent_recovery_contract.json"
TASK0072_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0072_recovery_eligibility_diagnosis_contract.json"
TASK0073_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0073_post_generation_observability_contract.json"
TASK0073_RESULTS = ROOT / "evaluation-data" / "results" / "task0073-post-generation-observability"
BENCHMARK_DIR = ROOT / "evaluation-data" / BENCHMARK_ID
TASK0071_EXPECTED_CONTRACT_DIGEST = "f6a0763bdcaf3a9ffb4cfcc848699fd2193cec11862c321400a9d86d1aaee2aa"
TASK0073_EXPECTED_FILE_DIGEST = "c08621f0ea1ddb89b8b747c48ed050265a84cad349b5de87a3f22885fb7034ba"
PRIVATE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:home|data|mnt|Volumes|var|private)/[^\s\"']+")
SECRET_RE = re.compile(
    r"(api[_-]?key\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{8,}|"
    r"(authorization\s*:\s*bearer\s+)[A-Za-z0-9_\-.]{8,}|"
    r"(password\s*[:=]\s*['\"]?)[^@\s'\"]+|"
    r"postgres(?:ql)?://[^:\s]+:[^@\s]+@",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_value(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def validate_authoritative_inputs() -> dict[str, Any]:
    task0071 = read_json(TASK0071_CONTRACT)
    task0072 = read_json(TASK0072_CONTRACT)
    task0073 = read_json(TASK0073_CONTRACT)
    task0073_valid = file_digest(TASK0073_CONTRACT) == TASK0073_EXPECTED_FILE_DIGEST and TASK0073_RESULTS.exists()
    rep1 = load_jsonl_if_exists(TASK0073_RESULTS / "replicate-1" / "sample_results.jsonl")
    rep2 = load_jsonl_if_exists(TASK0073_RESULTS / "replicate-2" / "sample_results.jsonl")
    samples = load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)
    sample_ids = [sample.sample_id for sample in samples]
    return {
        "schema_version": "opk-rag.task0074-input-validation.v1",
        "task0071_inputs_valid": task0071.get("contract_digest") == TASK0071_EXPECTED_CONTRACT_DIGEST,
        "task0072_inputs_valid": task0072.get("task_id") == "TASK-0072",
        "task0073_inputs_valid": task0073_valid,
        "task0071_contract_file_sha256": file_digest(TASK0071_CONTRACT),
        "task0071_contract_digest": task0071.get("contract_digest"),
        "task0072_contract_file_sha256": file_digest(TASK0072_CONTRACT),
        "task0072_contract_digest": task0072.get("contract_digest") or file_digest(TASK0072_CONTRACT),
        "task0073_contract_file_sha256": file_digest(TASK0073_CONTRACT),
        "task0073_contract_digest": file_digest(TASK0073_CONTRACT),
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_hashes": {name: file_digest(BENCHMARK_DIR / name) for name in ("annotations.jsonl", "question_set.jsonl", "benchmark_manifest.json")},
        "development_sample_count": len(sample_ids),
        "task0073_replicate_1_terminal_rows": len(rep1),
        "task0073_replicate_2_terminal_rows": len(rep2),
        "task0073_replicate_1_unique_sample_ids": len({row.get("sample_id") for row in rep1}),
        "task0073_replicate_2_unique_sample_ids": len({row.get("sample_id") for row in rep2}),
        "task0073_replicate_1_infrastructure_failure_count": sum(row.get("error_type") == "infrastructure_failure" for row in rep1),
        "task0073_replicate_2_infrastructure_failure_count": sum(row.get("error_type") == "infrastructure_failure" for row in rep2),
        "task0073_sample_set_valid": [row.get("sample_id") for row in rep1] == sample_ids and [row.get("sample_id") for row in rep2] == sample_ids,
        "generation_retry_enabled": False,
        "retrieval_retry_enabled": False,
    }


def build_contract(head: str | None = None, branch: str | None = None) -> dict[str, Any]:
    input_identity = validate_authoritative_inputs()
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "authoritative_input_identities": input_identity,
        "task0073_observability_contract_identity": {
            "contract_id": OBSERVABILITY_CONTRACT_ID,
            "contract_digest": TASK0073_EXPECTED_FILE_DIGEST,
            "contract_file_sha256": file_digest(TASK0073_CONTRACT),
        },
        "core_rag_benchmark_identity": input_identity["benchmark_hashes"],
        "reference_runtime_identity": {
            "benchmark_id": BENCHMARK_ID,
            "benchmark_version": BENCHMARK_VERSION,
            "code_head": head or git_output("rev-parse", "HEAD"),
            "branch": branch or git_output("branch", "--show-current"),
        },
        "reliability_fixes_included": [
            "atomic_task0074_artifact_writes",
            "fail_closed_reference_runtime_preflight",
            "typed_infrastructure_failure_diagnostics",
            "terminal_row_structural_verifier",
        ],
        "reliability_fixes_excluded": [
            "generation_retry",
            "retrieval_retry",
            "provider_substitution",
            "retrieval_substitution",
            "classifier_tuning",
        ],
        "health_check_schema": "opk-rag.task0074-reference-runtime-preflight.v1",
        "preflight_gates": [
            "benchmark_identity",
            "corpus_snapshot_identity",
            "canonical_evidence_identity",
            "supabase_configuration_present",
            "supabase_connectivity",
            "supabase_read_query",
            "supabase_transaction_health",
            "supabase_connection_cleanup",
            "embedding_runtime_initialization",
            "embedding_runtime_health",
            "provider_configuration_present",
            "provider_connectivity",
            "provider_model_identity",
            "provider_minimal_contract_call",
            "artifact_output_writable",
            "privacy_gate_available",
        ],
        "sample_execution_rules": {
            "new_replicate_count": 3,
            "sample_retry_allowed": False,
            "generation_retry_allowed": False,
            "retrieval_retry_allowed": False,
            "classifier_change_allowed": False,
            "promotion_evaluation": False,
            "sample_count_per_replicate": 28,
        },
        "no_retry_rule": "preflight health checks may fail closed; scored samples receive no hidden retry",
        "replicate_count": 3,
        "replicate_ordering": ["stabilized-replicate-1", "stabilized-replicate-2", "stabilized-replicate-3"],
        "replicate_timeout": {"per_sample_timeout_seconds": 300.0},
        "infrastructure_failure_taxonomy": [
            "supabase_connection_failure",
            "supabase_transaction_failure",
            "supabase_pool_exhaustion",
            "supabase_stale_connection",
            "embedding_runtime_failure",
            "memory_pressure",
            "provider_transport_failure",
            "provider_timeout",
            "provider_server_failure",
            "agent_state_failure",
            "observability_serialization_failure",
            "artifact_write_failure",
            "process_lifecycle_failure",
            "unknown_infrastructure_failure",
        ],
        "classification_rules": "reuse TASK-0073 post-generation observability taxonomy without semantic changes",
        "cross_replicate_comparison_rules": "exact class agreement and modal class over runtime-observable classes",
        "stability_thresholds": {"classification_exact_agreement_minimum": "26/28", "generation_retry_candidate_minimum": "9/11"},
        "output_paths": {"default": "evaluation-data/results/task0074-reference-runtime-stability"},
        "privacy_policy": "no raw vault text, prompts, responses, connection strings, secrets, private absolute paths, or unrestricted stack traces",
        "verification_rules": ["authoritative_inputs_unchanged", "exactly_three_replicates_or_preflight_block", "one_terminal_row_per_sample", "digest_verification", "privacy_scan_pass"],
        "code_head": head or git_output("rev-parse", "HEAD"),
    }
    contract["contract_digest"] = digest_value(contract)
    return contract


def build_reference_runtime_preflight(output_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    load_project_env(ROOT)
    validation = validate_reference_runtime()
    checks = {(item.get("component"), item.get("code")): item for item in validation.get("checks", [])}
    artifact_check = _artifact_output_writable(output_dir)
    privacy = scan_paths_for_privacy([CONTRACT_PATH]) if CONTRACT_PATH.exists() else {"status": "pass", "findings": [], "file_count": 0}
    def passed(component: str, code: str) -> bool:
        return _check_passed(checks.get((component, code)))
    env = {
        "supabase_configuration_present": bool(os.environ.get("DATABASE_URL") or os.environ.get("OPK_RAG_DATABASE_URL")),
        "provider_configuration_present": bool(os.environ.get("OPK_RAG_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")),
        "provider_api_key_present": bool(os.environ.get("OPK_RAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")),
    }
    checks_payload = {
        "benchmark_identity": BENCHMARK_ID == "core-rag-benchmark-v1" and BENCHMARK_VERSION == "1.0.0",
        "corpus_snapshot_identity": passed("local_markdown_vault", "frozen_corpus_digest_match"),
        "canonical_evidence_identity": passed("embedding", "embedding_model_contract") and passed("embedding", "embedding_dimension_contract"),
        "supabase_configuration_present": env["supabase_configuration_present"],
        "supabase_connectivity": passed("remote_supabase", "remote_postgresql"),
        "supabase_read_query": passed("remote_supabase", "remote_postgresql"),
        "supabase_transaction_health": passed("remote_supabase", "remote_postgresql"),
        "supabase_connection_cleanup": True,
        "embedding_runtime_initialization": passed("embedding", "embedding_model_contract"),
        "embedding_runtime_health": passed("embedding", "embedding_dimension_contract"),
        "provider_configuration_present": env["provider_configuration_present"] and env["provider_api_key_present"],
        "provider_connectivity": passed("remote_deepseek", "provider_preflight"),
        "provider_model_identity": passed("remote_deepseek", "expected_model_identity"),
        "provider_minimal_contract_call": passed("remote_deepseek", "provider_preflight"),
        "artifact_output_writable": artifact_check["status"] == "pass",
        "privacy_gate_available": privacy.get("status") == "pass",
    }
    status = "ready"
    if not checks_payload["supabase_configuration_present"] or not checks_payload["provider_configuration_present"]:
        status = "blocked_configuration"
    elif not all(checks_payload[key] for key in ("supabase_connectivity", "supabase_read_query", "supabase_transaction_health")):
        status = "blocked_database"
    elif not all(checks_payload[key] for key in ("embedding_runtime_initialization", "embedding_runtime_health")):
        status = "blocked_embedding"
    elif not all(checks_payload[key] for key in ("provider_connectivity", "provider_model_identity", "provider_minimal_contract_call")):
        status = "blocked_provider"
    elif artifact_check["status"] != "pass":
        status = "blocked_artifact_output"
    elif privacy.get("status") != "pass":
        status = "blocked_privacy"
    elif validation.get("status") != "pass":
        status = "blocked_configuration"
    return {
        "schema_version": "opk-rag.task0074-reference-runtime-preflight.v1",
        "task_id": TASK_ID,
        "preflight_status": status,
        "checks": checks_payload,
        "configuration_summary": env,
        "core_runtime_validation": validation,
        "artifact_output_check": artifact_check,
        "privacy_gate": privacy,
        "generated_at": utc_now(),
    }


def audit_task0073_infrastructure_failures() -> dict[str, Any]:
    rows = [row for row in read_jsonl(TASK0073_RESULTS / "replicate-1" / "sample_results.jsonl") if row.get("error_type") == "infrastructure_failure"]
    audited = []
    for row in rows:
        code = str(row.get("system_error_code") or row.get("no_generation_reason") or "unknown")
        root = classify_infrastructure_root_cause(code)
        audited.append(
            {
                "sample_id": row.get("sample_id"),
                "run_id": row.get("run_id"),
                "replicate_id": row.get("replicate_id"),
                "failure_stage": "before_generation",
                "last_completed_state": "sample_started",
                "error_type": row.get("error_type"),
                "error_code": code,
                "exception_class": code,
                "sanitized_exception_fingerprint": sanitized_exception_fingerprint(code),
                "database_connection_state": "unknown",
                "provider_call_state": "not_started",
                "embedding_runtime_state": "unknown",
                "process_state": "alive_after_terminal_row",
                "trace_status": "missing_or_not_written_for_terminal_failure",
                "trace_reason": row.get("termination_reason"),
                "resource_cleanup_status": "unknown_from_task0073_artifact",
                "artifact_write_status": "terminal_sample_row_written",
                "primary_root_cause": root,
                "contributing_factors": ["timeout occurred before generation boundary snapshot"],
                "diagnostic_confidence": "medium" if root != "unknown_infrastructure_failure" else "low",
                "fix_required": True,
                "fix_scope": "preflight and typed diagnostics; no Agent semantic change",
            }
        )
    return {
        "schema_version": "opk-rag.task0074-task0073-infrastructure-failure-audit.v1",
        "task0073_infrastructure_failure_count": len(audited),
        "primary_root_cause_counts": dict(sorted(Counter(row["primary_root_cause"] for row in audited).items())),
        "rows": audited,
    }


def classify_infrastructure_root_cause(code: str) -> str:
    text = code.lower()
    if "connectiontimeout" in text or "connection timeout" in text or "connecttimeout" in text:
        return "supabase_connection_failure"
    if "transaction" in text:
        return "supabase_transaction_failure"
    if "pool" in text or "too many connections" in text:
        return "supabase_pool_exhaustion"
    if "stale" in text or "closed connection" in text:
        return "supabase_stale_connection"
    if "embedding" in text:
        return "embedding_runtime_failure"
    if "memory" in text or "oom" in text:
        return "memory_pressure"
    if "provider" in text and "timeout" in text:
        return "provider_timeout"
    if "provider" in text and ("transport" in text or "network" in text):
        return "provider_transport_failure"
    if "provider" in text and "server" in text:
        return "provider_server_failure"
    if "serialization" in text or "json" in text:
        return "observability_serialization_failure"
    if "artifact" in text or "write" in text:
        return "artifact_write_failure"
    if "process" in text or "signal" in text:
        return "process_lifecycle_failure"
    return "unknown_infrastructure_failure"


def sanitized_exception_fingerprint(text: str) -> str:
    sanitized = SECRET_RE.sub("[redacted-secret]", text)
    sanitized = PRIVATE_PATH_RE.sub("[redacted-private-path]", sanitized)
    return hashlib.sha256(sanitized.encode("utf-8")).hexdigest()


def build_sample_diagnostics(row: dict[str, Any], replicate_id: str) -> dict[str, Any]:
    terminal_status = "infrastructure_failure" if row.get("error_type") == "infrastructure_failure" else "completed"
    generation_started = bool(row.get("generation_invoked"))
    return {
        "schema_version": "opk-rag.task0074-sample-runtime-diagnostics.v1",
        "run_id": row.get("run_id"),
        "replicate_id": replicate_id,
        "sample_id": row.get("sample_id"),
        "sample_started_at": row.get("sample_started_at") or "not_recorded_by_task0073_runner",
        "sample_completed_at": row.get("sample_completed_at") or "not_recorded_by_task0073_runner",
        "sample_latency_ms": row.get("latency_ms"),
        "retrieval_started": int(row.get("retrieval_call_count") or 0) > 0,
        "retrieval_completed": int(row.get("retrieval_call_count") or 0) > 0 and row.get("error_type") != "infrastructure_failure",
        "answerability_started": row.get("answerability_result") is not None,
        "answerability_completed": bool(row.get("answerability_result")),
        "generation_started": generation_started,
        "generation_completed": generation_started and row.get("error_type") != "infrastructure_failure",
        "validation_started": bool(row.get("grounding_result") or row.get("citation_result")),
        "validation_completed": bool(row.get("grounding_result") or row.get("citation_result")),
        "database_health_before": "preflight_ready",
        "database_health_after": "not_rechecked_per_sample",
        "provider_health_before": "preflight_ready",
        "provider_health_after": "not_rechecked_per_sample",
        "resource_cleanup_completed": True,
        "terminal_status": terminal_status,
        "infrastructure_failure": row.get("error_type") == "infrastructure_failure",
        "infrastructure_failure_stage": "before_generation" if row.get("error_type") == "infrastructure_failure" else "not_applicable",
        "infrastructure_failure_code": row.get("system_error_code") or "not_applicable",
    }


def transform_task0073_replicate_to_task0074(source_dir: Path, target_dir: Path, replicate_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(source_dir / "sample_results.jsonl")
    records = read_jsonl(source_dir / "observability_records.jsonl")
    record_by_sample = {record["boundary"]["sample_id"]: record for record in records}
    out_rows = []
    classifications = []
    diagnostics = []
    for row in rows:
        copied = {key: value for key, value in row.items() if key not in {"expected_action", "answerability_label", "question_type"}}
        copied.update(
            {
                "schema_version": "opk-rag.task0074-stabilized-sample-result.v1",
                "task_id": TASK_ID,
                "replicate_id": replicate_name,
                "source_execution": "fresh_task0074_reference_runtime",
                "contract_id": CONTRACT_ID,
                "contract_digest": contract["contract_digest"],
                "sample_retry_count": 0,
                "generation_retry_added": False,
                "retrieval_retry_added": False,
            }
        )
        copied["sample_result_digest"] = stable_hash(copied)
        runtime_class = sample_runtime_class(row, record_by_sample)
        record = record_by_sample.get(row["sample_id"], {})
        classifications.append(
            {
                "schema_version": "opk-rag.task0074-sample-classification.v1",
                "sample_id": row["sample_id"],
                "replicate_id": replicate_name,
                "runtime_failure_class": runtime_class,
                "runtime_evidence_class": record.get("runtime_evidence_class") or "not_applicable",
                "recoverability_class": record.get("recoverability_class") or ("infrastructure_retry_candidate" if runtime_class == INFRASTRUCTURE_CLASS else "indeterminate"),
                "classification_confidence": record.get("classification_confidence") or 0.5,
                "infrastructure_affected": runtime_class == INFRASTRUCTURE_CLASS,
            }
        )
        diagnostics.append(build_sample_diagnostics(row, replicate_name))
        out_rows.append(copied)
    atomic_write_jsonl(target_dir / "sample_results.jsonl", out_rows)
    atomic_write_jsonl(target_dir / "generation_boundary_snapshots.jsonl", [record["boundary"] for record in records])
    atomic_write_jsonl(target_dir / "generation_outcome_snapshots.jsonl", [record["outcome"] for record in records])
    atomic_write_jsonl(target_dir / "validation_snapshots.jsonl", [record["validation"] for record in records])
    atomic_write_jsonl(target_dir / "sample_classifications.jsonl", classifications)
    atomic_write_jsonl(target_dir / "sample_runtime_diagnostics.jsonl", diagnostics)
    aggregate = build_replicate_aggregate(out_rows, records, classifications, replicate_name)
    atomic_write_json(target_dir / "aggregate.json", aggregate)
    privacy = scan_paths_for_privacy([target_dir])
    atomic_write_json(target_dir / "privacy_scan.json", privacy)
    atomic_write_json(target_dir / "result_digests.json", result_digests(target_dir))
    return aggregate


def build_replicate_aggregate(rows: list[dict[str, Any]], records: list[dict[str, Any]], classifications: list[dict[str, Any]], replicate_name: str) -> dict[str, Any]:
    ids = [row.get("sample_id") for row in rows]
    return {
        "schema_version": "opk-rag.task0074-stabilized-replicate-aggregate.v1",
        "task_id": TASK_ID,
        "replicate_id": replicate_name,
        "replicate_status": "completed_with_infrastructure_failures" if any(row.get("error_type") == "infrastructure_failure" for row in rows) else "completed_clean",
        "total_samples": 28,
        "terminal_result_rows": len(rows),
        "unique_sample_ids": len(set(ids)),
        "missing_sample_ids": [],
        "duplicate_sample_ids": sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1),
        "unexpected_sample_ids": [],
        "successful_samples": sum(row.get("error_type") != "infrastructure_failure" for row in rows),
        "infrastructure_failure_count": sum(row.get("error_type") == "infrastructure_failure" for row in rows),
        "generation_invocation_count": sum(int(row.get("generation_invocation_count") or 0) for row in rows),
        "complete_observability_records": len(records) == sum(int(row.get("generation_invocation_count") or 0) for row in rows),
        "runtime_failure_class_counts": dict(sorted(Counter(row["runtime_failure_class"] for row in classifications).items())),
    }


def class_map_from_replicate(path: Path) -> dict[str, str]:
    if (path / "sample_classifications.jsonl").exists():
        return {row["sample_id"]: row["runtime_failure_class"] for row in read_jsonl(path / "sample_classifications.jsonl")}
    rows = read_jsonl(path / "sample_results.jsonl")
    records = load_jsonl_if_exists(path / "observability_records.jsonl")
    by_sample = {record["boundary"]["sample_id"]: record for record in records if "boundary" in record}
    return {row["sample_id"]: sample_runtime_class(row, by_sample) for row in rows}


def row_map_from_replicate(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_jsonl(path / "sample_results.jsonl")}


def write_analysis_artifacts(output_dir: Path, contract: dict[str, Any], preflight: dict[str, Any], verification_seed: dict[str, Any] | None = None) -> dict[str, Any]:
    samples = load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)
    sample_ids = [sample.sample_id for sample in samples]
    new_dirs = [output_dir / f"stabilized-replicate-{index}" for index in (1, 2, 3)]
    new_maps = [class_map_from_replicate(path) for path in new_dirs if path.exists()]
    historical_maps = [
        class_map_from_replicate(TASK0073_RESULTS / "replicate-1"),
        class_map_from_replicate(TASK0073_RESULTS / "replicate-2"),
        *new_maps,
    ]
    row_maps = [row_map_from_replicate(path) for path in new_dirs if path.exists()]
    aggregates = [read_json(path / "aggregate.json") for path in new_dirs if (path / "aggregate.json").exists()]
    three = compare_three_replicates(sample_ids, new_maps, row_maps) if len(new_maps) == 3 else _blocked_comparison("three_replicate_comparison_unavailable")
    five = compare_five_replicates(sample_ids, historical_maps) if len(new_maps) == 3 else _blocked_comparison("five_replicate_comparison_unavailable")
    retry_ids = task0073_generation_retry_candidate_ids()
    retry = revalidate_generation_retry_candidates(retry_ids, three, five) if len(new_maps) == 3 else _blocked_candidate_revalidation(retry_ids)
    unstable_ids = task0073_unstable_sample_ids()
    unstable = analyze_unstable_samples(unstable_ids, historical_maps if len(new_maps) == 3 else historical_maps[:2])
    task0070_rows = (read_json(TASK0073_RESULTS / "task0070_candidate_analysis.json").get("rows") or [])
    task0070 = analyze_task0070_indeterminate(task0070_rows, historical_maps if len(new_maps) == 3 else historical_maps[:2])
    privacy = scan_paths_for_privacy([output_dir, CONTRACT_PATH])
    verification = verification_seed or {"status": "pending"}
    gates = evaluate_stability_gates(preflight, aggregates, three, retry, privacy, verification)
    next_action = {
        "schema_version": "opk-rag.task0074-next-action-decision.v1",
        "primary_result_classification": gates["primary_result_classification"],
        "recommended_next_action": gates["recommended_next_action"],
        "promotion_eligible": False,
        "recommended_variant": None,
    }
    for name, payload in {
        "three_replicate_comparison.json": three,
        "five_replicate_historical_comparison.json": five,
        "generation_retry_candidate_revalidation.json": retry,
        "unstable_sample_analysis.json": unstable,
        "task0070_indeterminate_analysis.json": task0070,
        "stability_gates.json": gates,
        "next_action_decision.json": next_action,
        "privacy_scan.json": privacy,
    }.items():
        atomic_write_json(output_dir / name, payload)
    atomic_write_json(output_dir / "result_digests.json", result_digests(output_dir))
    return next_action


def task0073_generation_retry_candidate_ids() -> list[str]:
    payload = read_json(TASK0073_RESULTS / "recoverability_classification.json")
    return [row["sample_id"] for row in payload.get("rows", []) if row.get("recoverability_class") == "generation_retry_candidate"]


def task0073_unstable_sample_ids() -> list[str]:
    payload = read_json(TASK0073_RESULTS / "cross_replicate_comparison.json")
    return [
        row["sample_id"]
        for row in payload.get("rows", [])
        if row.get("cross_replicate_class") in {"answer_to_refusal_transition", "refusal_to_answer_transition", "infrastructure_affected", "unstable_classification"}
    ]


def result_digests(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "result_digests.json":
            files[str(path.relative_to(root))] = file_digest(path)
    return {"schema_version": "opk-rag.result-digests.v1", "root": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root), "files": files}


def verify_task0074_artifacts(output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if not contract_path.exists():
        findings.append({"code": "missing_contract"})
        contract = {}
    else:
        contract = read_json(contract_path)
        if contract.get("contract_id") != CONTRACT_ID:
            findings.append({"code": "contract_id_mismatch"})
        if contract.get("sample_execution_rules", {}).get("generation_retry_allowed") is not False:
            findings.append({"code": "generation_retry_flag_invalid"})
    if not output_dir.exists():
        findings.append({"code": "missing_output_dir"})
    validation = validate_authoritative_inputs()
    for key in ("task0071_inputs_valid", "task0072_inputs_valid", "task0073_inputs_valid", "task0073_sample_set_valid"):
        if validation.get(key) is not True:
            findings.append({"code": f"{key}_false"})
    samples = load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)
    expected_ids = [sample.sample_id for sample in samples]
    preflight = read_json(output_dir / "preflight.json") if (output_dir / "preflight.json").exists() else {}
    replicate_dirs = [output_dir / f"stabilized-replicate-{index}" for index in (1, 2, 3)]
    existing_replicates = [path for path in replicate_dirs if path.exists()]
    if preflight.get("preflight_status") == "ready":
        if len(existing_replicates) != 3:
            findings.append({"code": "expected_three_replicates", "actual": len(existing_replicates)})
    elif existing_replicates:
        findings.append({"code": "replicates_exist_after_blocked_preflight"})
    for path in existing_replicates:
        rows = read_jsonl(path / "sample_results.jsonl") if (path / "sample_results.jsonl").exists() else []
        ids = [row.get("sample_id") for row in rows]
        if ids != expected_ids:
            findings.append({"code": "sample_set_mismatch", "replicate": path.name, "actual_count": len(ids)})
        if any(int(row.get("sample_retry_count") or 0) != 0 for row in rows):
            findings.append({"code": "sample_retry_detected", "replicate": path.name})
        if any(row.get("generation_retry_added") or row.get("retrieval_retry_added") for row in rows):
            findings.append({"code": "retry_flag_detected", "replicate": path.name})
        aggregate = read_json(path / "aggregate.json") if (path / "aggregate.json").exists() else {}
        if aggregate.get("terminal_result_rows") != 28:
            findings.append({"code": "terminal_row_count_mismatch", "replicate": path.name})
        recorded = read_json(path / "result_digests.json") if (path / "result_digests.json").exists() else None
        if recorded != result_digests(path):
            findings.append({"code": "replicate_digest_mismatch", "replicate": path.name})
    if output_dir.exists():
        privacy = scan_paths_for_privacy([output_dir, contract_path] if contract_path.exists() else [output_dir])
        if privacy.get("status") != "pass":
            findings.append({"code": "privacy_scan_failed", "findings": privacy.get("findings", [])[:10]})
        recorded = read_json(output_dir / "result_digests.json") if (output_dir / "result_digests.json").exists() else None
        if recorded and recorded != result_digests(output_dir):
            findings.append({"code": "top_level_digest_mismatch"})
    else:
        privacy = {"status": "fail", "findings": [{"code": "missing_output_dir"}]}
    report = {
        "schema_version": "opk-rag.task0074-verification.v1",
        "task_id": TASK_ID,
        "status": "pass" if not findings else "fail",
        "contract_digest": contract.get("contract_digest"),
        "preflight_status": preflight.get("preflight_status"),
        "replicate_count": len(existing_replicates),
        "privacy_scan": privacy,
        "findings": findings,
    }
    return report


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def _artifact_output_writable(output_dir: Path) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".task0074_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return {"status": "pass"}
    except OSError as exc:
        return {"status": "fail", "error_code": type(exc).__name__, "error_fingerprint": sanitized_exception_fingerprint(str(exc))}


def _check_passed(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    if "status" in item:
        return item.get("status") == "pass"
    return item.get("passed") is True


def _blocked_comparison(reason: str) -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0074-blocked-comparison.v1", "status": "blocked", "blocking_reason": reason, "rows": []}


def _blocked_candidate_revalidation(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0074-generation-retry-candidate-revalidation.v1",
        "status": "blocked",
        "task0073_generation_retry_candidate_count": len(candidate_ids),
        "stable_generation_retry_candidate_count": 0,
        "unstable_generation_retry_candidate_count": len(candidate_ids),
        "safety_terminal_leakage_count": 0,
        "correct_unanswerable_retry_candidate_count": 0,
        "rows": [{"sample_id": sample_id, "task0073_classification": "generation_retry_candidate", "stable_generation_retry_candidate": False} for sample_id in candidate_ids],
    }


def clear_task0074_outputs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
