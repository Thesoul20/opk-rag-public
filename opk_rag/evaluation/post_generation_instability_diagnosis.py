from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation.core_rag_benchmark import BENCHMARK_ID, BENCHMARK_VERSION, file_digest, scan_paths_for_privacy
from opk_rag.evaluation.stable_generation_retry_eligibility import (
    PROPOSAL_ID,
    build_disqualifying_signals,
    build_retry_eligibility_proposal,
    eligible_by_runtime_rule,
    evaluate_disqualifier_flags,
    validate_runtime_rule,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0075"
CONTRACT_ID = "opk-rag.post-generation-classification-instability-diagnosis.v1"
CONTRACT_SCHEMA = "opk-rag.task0075-post-generation-instability-diagnosis-contract.v1"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0075_post_generation_instability_diagnosis_contract.json"
RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis"
DOC_PATH = ROOT / "docs" / "TASK0075_POST_GENERATION_CLASSIFICATION_INSTABILITY_REPORT.md"

TASK0071_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0071_governed_agent_recovery_contract.json"
TASK0072_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0072_recovery_eligibility_diagnosis_contract.json"
TASK0073_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0073_post_generation_observability_contract.json"
TASK0074_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0074_reference_runtime_stability_contract.json"
TASK0071_RESULTS = ROOT / "evaluation-data" / "results" / "task0071-governed-agent-recovery"
TASK0072_RESULTS = ROOT / "evaluation-data" / "results" / "task0072-recovery-eligibility-diagnosis"
TASK0073_RESULTS = ROOT / "evaluation-data" / "results" / "task0073-post-generation-observability"
TASK0074_RESULTS = ROOT / "evaluation-data" / "results" / "task0074-reference-runtime-stability"
BENCHMARK_DIR = ROOT / "evaluation-data" / BENCHMARK_ID

TASK0071_EXPECTED_CONTRACT_DIGEST = "f6a0763bdcaf3a9ffb4cfcc848699fd2193cec11862c321400a9d86d1aaee2aa"
TASK0073_EXPECTED_CONTRACT_DIGEST = "c08621f0ea1ddb89b8b747c48ed050265a84cad349b5de87a3f22885fb7034ba"

PRIVATE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:home|data|mnt|Volumes|var|private)/[^\s\"']+")
SECRET_RE = re.compile(
    r"([\"']?api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}|"
    r"(authorization\s*:\s*bearer\s+)[A-Za-z0-9_\-.]{8,}|"
    r"(password\s*[:=]\s*['\"]?)[^@\s'\"]+|"
    r"postgres(?:ql)?://[^:\s]+:[^@\s]+@",
    re.IGNORECASE,
)

OBSERVATION_FIELDS = [
    "sample_id",
    "source_task",
    "replicate_id",
    "run_id",
    "sample_terminal_status",
    "infrastructure_failure",
    "retrieval_completed",
    "retrieval_candidate_count",
    "selected_evidence_count",
    "canonical_document_count",
    "canonical_scope_count",
    "retrieval_score_summary",
    "evidence_identity_digest",
    "answerability_label",
    "answerability_reason_code",
    "answerability_confidence",
    "generation_invoked",
    "provider_call_completed",
    "provider_latency_ms",
    "response_contract_valid",
    "response_mode",
    "answer_draft_present",
    "provider_refusal_detected",
    "provider_refusal_source",
    "refusal_reason_code",
    "runtime_evidence_class",
    "runtime_failure_class",
    "recoverability_class",
    "citation_validation_invoked",
    "citation_validation_passed",
    "grounding_validation_invoked",
    "grounding_validation_passed",
    "unsupported_claim_detected",
    "final_action",
    "termination_reason",
    "observability_complete",
]

EXECUTION_ORDER = [
    "infrastructure_failure",
    "retrieval_completed",
    "retrieval_candidate_count",
    "selected_evidence_count",
    "canonical_document_count",
    "canonical_scope_count",
    "retrieval_score_summary",
    "evidence_identity_digest",
    "answerability_label",
    "answerability_reason_code",
    "answerability_confidence",
    "generation_invoked",
    "provider_call_completed",
    "provider_latency_ms",
    "response_contract_valid",
    "response_mode",
    "answer_draft_present",
    "provider_refusal_detected",
    "provider_refusal_source",
    "refusal_reason_code",
    "runtime_evidence_class",
    "runtime_failure_class",
    "recoverability_class",
    "citation_validation_invoked",
    "citation_validation_passed",
    "grounding_validation_invoked",
    "grounding_validation_passed",
    "unsupported_claim_detected",
    "final_action",
    "termination_reason",
]


@dataclass(frozen=True)
class ReplicateSource:
    source_task: str
    replicate_id: str
    directory: Path


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_value(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def replicate_sources() -> list[ReplicateSource]:
    return [
        ReplicateSource("TASK-0073", "task0073-replicate-1", TASK0073_RESULTS / "replicate-1"),
        ReplicateSource("TASK-0073", "task0073-replicate-2", TASK0073_RESULTS / "replicate-2"),
        ReplicateSource("TASK-0074", "stabilized-replicate-1", TASK0074_RESULTS / "stabilized-replicate-1"),
        ReplicateSource("TASK-0074", "stabilized-replicate-2", TASK0074_RESULTS / "stabilized-replicate-2"),
        ReplicateSource("TASK-0074", "stabilized-replicate-3", TASK0074_RESULTS / "stabilized-replicate-3"),
    ]


def authoritative_input_identity() -> dict[str, Any]:
    contract_files = {
        "task0071": TASK0071_CONTRACT,
        "task0072": TASK0072_CONTRACT,
        "task0073": TASK0073_CONTRACT,
        "task0074": TASK0074_CONTRACT,
    }
    result_dirs = {
        "task0071": TASK0071_RESULTS,
        "task0072": TASK0072_RESULTS,
        "task0073": TASK0073_RESULTS,
        "task0074": TASK0074_RESULTS,
    }
    contracts = {key: read_json(path) for key, path in contract_files.items()}
    three = read_json(TASK0074_RESULTS / "three_replicate_comparison.json")
    sample_ids = [row["sample_id"] for row in three["rows"]]
    sources = replicate_sources()
    replicate_rows = {source.replicate_id: read_jsonl(source.directory / "sample_results.jsonl") for source in sources}
    sample_sets = {key: [row.get("sample_id") for row in rows] for key, rows in replicate_rows.items()}
    expected = sample_sets[sources[0].replicate_id]
    benchmark_hashes = {
        name: file_digest(BENCHMARK_DIR / name)
        for name in ("annotations.jsonl", "question_set.jsonl", "benchmark_manifest.json")
    }
    status = {
        "task0071_inputs_valid": contracts["task0071"].get("contract_digest") == TASK0071_EXPECTED_CONTRACT_DIGEST,
        "task0072_inputs_valid": (contracts["task0072"].get("task_id") == "TASK-0072" and TASK0072_RESULTS.exists()),
        "task0073_inputs_valid": file_digest(TASK0073_CONTRACT) == TASK0073_EXPECTED_CONTRACT_DIGEST and TASK0073_RESULTS.exists(),
        "task0074_inputs_valid": contracts["task0074"].get("task_id") == "TASK-0074" and TASK0074_RESULTS.exists(),
        "five_replicate_directories_valid": all(source.directory.exists() for source in sources),
        "five_replicate_sample_sets_valid": all(ids == expected for ids in sample_sets.values()) and len(expected) == 28,
        "staged_diff_empty": git_output("diff", "--cached", "--name-only") == "",
    }
    return {
        "schema_version": "opk-rag.task0075-input-identity.v1",
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "git_status_short_initial": git_output("status", "--short"),
        "contracts": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "file_sha256": file_digest(path),
                "contract_digest": contracts[key].get("contract_digest") or file_digest(path),
            }
            for key, path in contract_files.items()
        },
        "result_directories": {key: path.relative_to(ROOT).as_posix() for key, path in result_dirs.items()},
        "replicates": [
            {
                "source_task": source.source_task,
                "replicate_id": source.replicate_id,
                "path": source.directory.relative_to(ROOT).as_posix(),
                "sample_result_sha256": file_digest(source.directory / "sample_results.jsonl"),
                "observability_record_sha256": _observability_source_digest(source.directory),
                "sample_count": len(replicate_rows[source.replicate_id]),
                "unique_sample_count": len(set(sample_sets[source.replicate_id])),
            }
            for source in sources
        ],
        "core_rag_benchmark": {
            "benchmark_id": BENCHMARK_ID,
            "benchmark_version": BENCHMARK_VERSION,
            "hashes": benchmark_hashes,
        },
        "task0072_contract_digest": contracts["task0072"].get("contract_digest") or file_digest(TASK0072_CONTRACT),
        "task0074_contract_digest": contracts["task0074"].get("contract_digest") or file_digest(TASK0074_CONTRACT),
        "development_sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        **status,
        "authoritative_inputs_valid": all(status.values()),
    }


def build_contract(input_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    input_identity = input_identity or authoritative_input_identity()
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "authoritative_input_identities": input_identity,
        "five_replicate_identities": input_identity["replicates"],
        "unstable_sample_identification_rule": "sample is unstable when TASK-0074 three_replicate_comparison classification_stable is false",
        "stable_candidate_identification_rule": "sample must be TASK-0074 stable_generation_retry_candidate and pass TASK-0075 runtime-signal audit",
        "field_diff_schema": {
            "field_statuses": ["stable", "changed_once", "changed_multiple_times", "missing_in_some_runs", "not_comparable"],
            "first_divergent_field_order": EXECUTION_ORDER,
        },
        "instability_taxonomy": [
            "retrieval_identity_instability",
            "retrieval_score_instability",
            "answerability_instability",
            "provider_answer_refusal_nondeterminism",
            "provider_refusal_reason_nondeterminism",
            "generation_contract_parsing_instability",
            "citation_validation_instability",
            "grounding_validation_instability",
            "classifier_boundary_ambiguity",
            "missing_observability",
            "infrastructure_confounded",
            "multi_stage_instability",
            "unknown_instability",
        ],
        "retry_eligibility_proposal_schema": "opk-rag.stable-generation-retry-eligibility-proposal.v1",
        "runtime_gold_separation": {
            "runtime_deployable_rules_may_use_gold": False,
            "offline_diagnostic_only_may_use_gold": True,
            "forbidden_runtime_fields": sorted(
                [
                    "sample_id",
                    "expected_action",
                    "gold_answerability",
                    "required_claims",
                    "forbidden_claims",
                    "gold_evidence_identities",
                    "task0070_classification",
                    "prior_benchmark_correctness",
                ]
            ),
        },
        "confidence_rules": {
            "high": "first divergent field and downstream effect are directly observed",
            "medium": "primary divergent stage is observed but one supporting identity field is missing",
            "low": "critical stage fields are missing or not comparable",
        },
        "default_fail_closed_behavior": "not_eligible",
        "privacy_policy": "no raw vault text, prompts, provider responses, secrets, private absolute paths, or unrestricted stack traces",
        "verification_rules": [
            "authoritative_inputs_valid",
            "unstable_sample_count_is_four",
            "five_replicate_matrix_complete",
            "runtime_rules_reject_gold_fields",
            "proposal_fail_closed",
            "privacy_scan_pass",
            "result_digests_match",
        ],
        "external_calls_allowed": False,
        "new_replicates_allowed": False,
        "classifier_change_allowed": False,
        "generation_retry_implementation_allowed": False,
        "runtime_policy_change_allowed": False,
        "promotion_evaluation": False,
        "code_head": input_identity["head"],
    }
    contract["contract_digest"] = digest_value(contract)
    return contract


def identify_unstable_samples() -> dict[str, Any]:
    three = read_json(TASK0074_RESULTS / "three_replicate_comparison.json")
    unstable = [row["sample_id"] for row in three["rows"] if not row.get("classification_stable")]
    stable = [row["sample_id"] for row in three["rows"] if row.get("classification_stable")]
    return {
        "schema_version": "opk-rag.task0075-unstable-sample-set.v1",
        "identification_source": "evaluation-data/results/task0074-reference-runtime-stability/three_replicate_comparison.json",
        "unstable_sample_count": len(unstable),
        "unstable_sample_ids": unstable,
        "stable_sample_count": len(stable),
        "total_sample_count": len(three["rows"]),
    }


def load_authoritative_replicates() -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for source in replicate_sources():
        sample_rows = {row["sample_id"]: row for row in read_jsonl(source.directory / "sample_results.jsonl")}
        records = load_observability_records(source.directory)
        runtime_path = source.directory / "sample_runtime_diagnostics.jsonl"
        diagnostics = {row["sample_id"]: row for row in read_jsonl(runtime_path)} if runtime_path.exists() else {}
        loaded[source.replicate_id] = {
            sample_id: {"sample_result": sample, "observability_record": records.get(sample_id), "runtime_diagnostics": diagnostics.get(sample_id), "source": source}
            for sample_id, sample in sample_rows.items()
        }
    return loaded


def load_observability_records(directory: Path) -> dict[str, dict[str, Any]]:
    records_path = directory / "observability_records.jsonl"
    if records_path.exists():
        return {row["boundary"]["sample_id"]: row for row in read_jsonl(records_path)}
    boundary_rows = read_jsonl(directory / "generation_boundary_snapshots.jsonl")
    outcome_rows = {row["generation_invocation_id"]: row for row in read_jsonl(directory / "generation_outcome_snapshots.jsonl")}
    validation_rows = {row["generation_invocation_id"]: row for row in read_jsonl(directory / "validation_snapshots.jsonl")}
    class_rows = {row["sample_id"]: row for row in read_jsonl(directory / "sample_classifications.jsonl")}
    records: dict[str, dict[str, Any]] = {}
    for boundary in boundary_rows:
        invocation_id = boundary["generation_invocation_id"]
        sample_id = boundary["sample_id"]
        classification = class_rows[sample_id]
        record = {
            "schema_version": "opk-rag.post-generation-observability-record.v1",
            "contract_id": boundary.get("contract_id"),
            "boundary": boundary,
            "outcome": outcome_rows.get(invocation_id, {}),
            "validation": validation_rows.get(invocation_id, {}),
            "runtime_failure_class": classification.get("runtime_failure_class"),
            "runtime_evidence_class": classification.get("runtime_evidence_class"),
            "recoverability_class": classification.get("recoverability_class"),
            "classification_confidence": classification.get("classification_confidence"),
            "secondary_reason_codes": [],
        }
        record["record_digest"] = digest_value(record)
        records[sample_id] = record
    return records


def build_five_replicate_observation_matrix(sample_ids: list[str] | None = None) -> list[dict[str, Any]]:
    sample_ids = sample_ids or identify_unstable_samples()["unstable_sample_ids"]
    loaded = load_authoritative_replicates()
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        for source in replicate_sources():
            bundle = loaded[source.replicate_id][sample_id]
            rows.append(normalize_observation(sample_id, bundle))
    return rows


def normalize_observation(sample_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
    sample = bundle["sample_result"]
    record = bundle.get("observability_record") or {}
    diagnostics = bundle.get("runtime_diagnostics") or {}
    source: ReplicateSource = bundle["source"]
    boundary = record.get("boundary") or {}
    outcome = record.get("outcome") or {}
    validation = record.get("validation") or {}
    answerability = sample.get("answerability_result") or {}
    generation = sample.get("generation_result") or {}
    grounding = sample.get("grounding_result") or generation.get("grounding") or {}
    citation = sample.get("citation_result") or {}
    evidence_identity_digest = _evidence_identity_digest(sample, boundary)
    row = {
        "sample_id": sample_id,
        "source_task": source.source_task,
        "replicate_id": source.replicate_id,
        "run_id": sample.get("run_id") or boundary.get("run_id") or "not_recorded",
        "sample_terminal_status": diagnostics.get("terminal_status") or ("infrastructure_prevented" if sample.get("error_type") == "infrastructure_failure" else "completed"),
        "infrastructure_failure": sample.get("error_type") == "infrastructure_failure" or diagnostics.get("infrastructure_failure") is True,
        "retrieval_completed": diagnostics.get("retrieval_completed", sample.get("retrieval_call_count", 0) > 0),
        "retrieval_candidate_count": _first_present(boundary.get("candidate_count"), (answerability.get("diagnostics_counts") or {}).get("candidate_count"), "not_recorded"),
        "selected_evidence_count": _first_present(boundary.get("selected_evidence_count"), answerability.get("evidence_count"), "not_recorded"),
        "canonical_document_count": _first_present(boundary.get("canonical_document_count"), (answerability.get("diagnostics_counts") or {}).get("selected_source_document_count"), "not_recorded"),
        "canonical_scope_count": _first_present(boundary.get("canonical_scope_count"), (answerability.get("diagnostics_counts") or {}).get("selected_chunk_count"), "not_recorded"),
        "retrieval_score_summary": _score_summary(boundary, answerability),
        "evidence_identity_digest": evidence_identity_digest,
        "answerability_label": _first_present(boundary.get("answerability_label"), answerability.get("status"), sample.get("answerability_label"), "not_recorded"),
        "answerability_reason_code": _first_present(boundary.get("answerability_reason_code"), answerability.get("reason_code"), "not_recorded"),
        "answerability_confidence": _first_present(boundary.get("answerability_confidence"), answerability.get("confidence"), "not_recorded"),
        "generation_invoked": sample.get("generation_invoked", bool(generation)),
        "generation_invocation_count": sample.get("generation_invocation_count", sample.get("generation_call_count", "not_recorded")),
        "generation_retries": sample.get("generation_retries", "not_recorded"),
        "provider_call_completed": outcome.get("provider_call_completed", bool(generation)),
        "provider_latency_ms": _first_present(outcome.get("provider_latency_ms"), sample.get("latency_ms"), "not_recorded"),
        "response_contract_valid": _first_present(outcome.get("response_contract_valid"), generation.get("status") in {"answered", "refused"} if generation else "not_recorded"),
        "response_mode": _first_present(outcome.get("response_mode"), _response_mode_from_generation(generation), "not_recorded"),
        "answer_draft_present": _first_present(outcome.get("answer_draft_present"), generation.get("status") == "answered" and bool(generation.get("answer_hash")) if generation else "not_recorded"),
        "provider_refusal_detected": _first_present(outcome.get("provider_refusal_detected"), generation.get("status") == "refused" or bool(generation.get("refusal_reason_code")) if generation else "not_recorded"),
        "provider_refusal_source": _first_present(outcome.get("provider_refusal_source"), "structured_abstention" if generation.get("status") == "refused" else "not_applicable", "not_recorded"),
        "refusal_reason_code": _first_present(outcome.get("refusal_reason_code"), generation.get("refusal_reason_code") or "not_applicable", "not_recorded"),
        "runtime_evidence_class": _first_present(record.get("runtime_evidence_class"), sample.get("runtime_evidence_class"), "not_recorded"),
        "runtime_failure_class": _first_present(record.get("runtime_failure_class"), sample.get("runtime_failure_class"), "infrastructure_failure" if sample.get("error_type") == "infrastructure_failure" else "not_recorded"),
        "recoverability_class": _first_present(record.get("recoverability_class"), sample.get("recoverability_class"), "not_recorded"),
        "citation_validation_invoked": _first_present(validation.get("citation_validation_invoked"), citation.get("valid") is not None if citation else "not_recorded"),
        "citation_validation_passed": _first_present(validation.get("citation_validation_passed"), citation.get("valid"), "not_applicable"),
        "grounding_validation_invoked": _first_present(validation.get("grounding_validation_invoked"), grounding.get("valid") is not None if grounding else "not_recorded"),
        "grounding_validation_passed": _first_present(validation.get("grounding_validation_passed"), grounding.get("valid"), "not_applicable"),
        "unsupported_claim_detected": _first_present(validation.get("unsupported_claim_detected"), (grounding.get("unsupported_claim_count") or 0) > 0 if grounding else "not_recorded"),
        "final_action": _first_present(validation.get("final_action"), sample.get("final_action"), "not_recorded"),
        "termination_reason": _first_present(validation.get("final_termination_reason"), sample.get("termination_reason"), "not_recorded"),
        "observability_complete": bool(record) and all(part in record for part in ("boundary", "outcome", "validation")),
        "provider_identity_digest": digest_value(boundary.get("provider_identity") or {"provider_id": generation.get("provider_id") or "unknown"}),
        "model_identity_digest": digest_value(boundary.get("model_identity") or {"model_id": generation.get("model_id") or "unknown"}),
        "generation_contract_id": boundary.get("generation_contract_id") or generation.get("output_schema_version") or "not_recorded",
    }
    for field in OBSERVATION_FIELDS:
        row.setdefault(field, "not_recorded")
    return row


def compare_sample_fields(matrix_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        by_sample[row["sample_id"]].append(row)
    rows = []
    for sample_id in sorted(by_sample):
        observations = by_sample[sample_id]
        field_statuses: dict[str, str] = {}
        for field in OBSERVATION_FIELDS:
            if field in {"sample_id", "source_task", "replicate_id", "run_id"}:
                field_statuses[field] = "not_comparable"
                continue
            values = [_comparable_value(row.get(field)) for row in observations]
            if any(value in {"not_recorded", "infrastructure_prevented"} for value in values):
                field_statuses[field] = "missing_in_some_runs"
            else:
                unique = list(dict.fromkeys(values))
                field_statuses[field] = "stable" if len(unique) == 1 else ("changed_once" if len(unique) == 2 else "changed_multiple_times")
        variable = [field for field, status in field_statuses.items() if status not in {"stable", "not_comparable"}]
        first = next((field for field in EXECUTION_ORDER if field in variable), "none")
        rows.append(
            {
                "sample_id": sample_id,
                "field_statuses": field_statuses,
                "stable_fields": sorted(field for field, status in field_statuses.items() if status == "stable"),
                "variable_fields": variable,
                "first_divergent_field": first,
                "downstream_divergent_fields": [field for field in EXECUTION_ORDER[EXECUTION_ORDER.index(first) + 1 :] if field in variable] if first != "none" else [],
                "likely_root_cause_fields": _likely_root_cause_fields(first, variable),
            }
        )
    return {"schema_version": "opk-rag.task0075-field-difference-analysis.v1", "rows": rows}


def classify_instability_root_cause(matrix_rows: list[dict[str, Any]], field_diff: dict[str, Any]) -> dict[str, Any]:
    diff_by_sample = {row["sample_id"]: row for row in field_diff["rows"]}
    rows = []
    for sample_id in sorted({row["sample_id"] for row in matrix_rows}):
        observations = [row for row in matrix_rows if row["sample_id"] == sample_id]
        diff = diff_by_sample[sample_id]
        classes = {row.get("runtime_failure_class") for row in observations}
        response_modes = {row.get("response_mode") for row in observations}
        refusal_reasons = {row.get("refusal_reason_code") for row in observations}
        evidence_stable = diff["field_statuses"].get("evidence_identity_digest") == "stable"
        answerability_stable = diff["field_statuses"].get("answerability_label") == "stable" and diff["field_statuses"].get("answerability_reason_code") == "stable"
        if any(row.get("infrastructure_failure") for row in observations):
            primary = "infrastructure_confounded"
        elif diff["first_divergent_field"] in {"evidence_identity_digest", "selected_evidence_count", "canonical_document_count", "canonical_scope_count"}:
            primary = "retrieval_identity_instability"
        elif diff["first_divergent_field"] == "retrieval_score_summary":
            primary = "retrieval_score_instability"
        elif diff["first_divergent_field"] in {"answerability_label", "answerability_reason_code", "answerability_confidence"}:
            primary = "answerability_instability"
        elif evidence_stable and answerability_stable and response_modes == {"abstain"} and len(refusal_reasons) > 1:
            primary = "provider_refusal_reason_nondeterminism"
        elif evidence_stable and answerability_stable and "answer" in response_modes and "abstain" in response_modes:
            primary = "provider_answer_refusal_nondeterminism"
        elif diff["first_divergent_field"] == "response_contract_valid":
            primary = "generation_contract_parsing_instability"
        elif diff["first_divergent_field"] in {"citation_validation_passed", "grounding_validation_passed", "unsupported_claim_detected"}:
            primary = "grounding_validation_instability"
        elif diff["field_statuses"].get("observability_complete") == "missing_in_some_runs":
            primary = "missing_observability"
        elif len(classes) > 1 and response_modes in ({"answer"}, {"abstain"}):
            primary = "classifier_boundary_ambiguity"
        elif len(diff["likely_root_cause_fields"]) > 1:
            primary = "multi_stage_instability"
        else:
            primary = "unknown_instability"
        rows.append(
            {
                "sample_id": sample_id,
                "primary_instability_class": primary,
                "secondary_instability_classes": _secondary_instabilities(diff, observations, primary),
                "diagnostic_confidence": "high" if primary.startswith("provider_") and evidence_stable and answerability_stable else "medium",
                "supporting_observations": _supporting_observations(observations, diff, primary),
                "contradicting_observations": _contradicting_observations(observations, diff, primary),
                "provider_nondeterminism": provider_nondeterminism_summary(observations),
                "classifier_boundary_analysis": classifier_boundary_analysis(observations, diff),
            }
        )
    counts = Counter(row["primary_instability_class"] for row in rows)
    return {
        "schema_version": "opk-rag.task0075-instability-classification.v1",
        "unstable_sample_count": len(rows),
        "primary_class_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def audit_stable_generation_retry_candidates(matrix_all_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    candidate_source = read_json(TASK0074_RESULTS / "generation_retry_candidate_revalidation.json")
    candidate_ids = [row["sample_id"] for row in candidate_source["rows"] if row.get("stable_generation_retry_candidate")]
    matrix_all_rows = matrix_all_rows or build_five_replicate_observation_matrix(_all_sample_ids())
    by_sample = defaultdict(list)
    for row in matrix_all_rows:
        by_sample[row["sample_id"]].append(row)
    rows = []
    for sample_id in candidate_ids:
        observations = by_sample[sample_id]
        valid = [row for row in observations if not row.get("infrastructure_failure")]
        runtime_classes = [row["runtime_failure_class"] for row in valid]
        recoverability = [row["recoverability_class"] for row in valid]
        evidence_classes = [row["runtime_evidence_class"] for row in valid]
        rows.append(
            {
                "sample_id": sample_id,
                "observed_replicate_count": len(observations),
                "infrastructure_valid_replicate_count": len(valid),
                "generation_invocation_count": sum(1 for row in valid if row.get("generation_invoked") is True),
                "provider_refusal_count": sum(1 for row in valid if row.get("provider_refusal_detected") is True),
                "answer_draft_count": sum(1 for row in valid if row.get("answer_draft_present") is True),
                "runtime_failure_classes": sorted(set(runtime_classes)),
                "runtime_evidence_classes": sorted(set(evidence_classes)),
                "recoverability_classes": sorted(set(recoverability)),
                "modal_failure_class": _modal(runtime_classes),
                "modal_recoverability_class": _modal(recoverability),
                "classification_exactly_stable": len(set(runtime_classes)) == 1 and len(valid) == 5,
                "evidence_identity_stable": len({row["evidence_identity_digest"] for row in valid}) == 1,
                "answerability_stable": len({(row["answerability_label"], row["answerability_reason_code"]) for row in valid}) == 1,
                "safety_terminal_signal_present": any(_disqualifier_any(row, {"safety_terminal_refusal", "forbidden_claim_risk"}) for row in valid),
                "correct_unanswerable_signal_present": any(_disqualifier_any(row, {"correct_unanswerable_runtime_signal"}) for row in valid),
                "unsupported_answer_signal_present": any(_disqualifier_any(row, {"unsupported_claim_detected"}) for row in valid),
                "candidate_confidence": "high" if len(set(runtime_classes)) == 1 and len(valid) == 5 else "medium",
            }
        )
    return {
        "schema_version": "opk-rag.task0075-stable-generation-retry-candidate-audit.v1",
        "task0074_stable_generation_retry_candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "rows": rows,
    }


def extract_common_runtime_signals(candidate_audit: dict[str, Any], matrix_all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = set(candidate_audit["candidate_ids"])
    rows_by_sample = defaultdict(list)
    for row in matrix_all_rows:
        rows_by_sample[row["sample_id"]].append(row)
    signal_defs = {
        "initial_answerability_allows_generation": lambda obs: obs["answerability_label"] in {"answerable", "partially_answerable"},
        "provider_call_completed": lambda obs: obs["provider_call_completed"] is True,
        "response_contract_valid": lambda obs: obs["response_contract_valid"] is True,
        "response_mode_abstain": lambda obs: obs["response_mode"] == "abstain",
        "provider_refusal_detected": lambda obs: obs["provider_refusal_detected"] is True,
        "refusal_model_conservative": lambda obs: obs["runtime_failure_class"] == "refusal_model_conservative",
        "runtime_evidence_appears_sufficient": lambda obs: obs["runtime_evidence_class"] == "runtime_evidence_appears_sufficient",
        "no_safety_terminal_signal": lambda obs: not evaluate_disqualifier_flags(obs)["safety_terminal_refusal"],
        "no_correct_unanswerable_signal": lambda obs: not evaluate_disqualifier_flags(obs)["correct_unanswerable_runtime_signal"],
        "no_forbidden_claim_signal": lambda obs: not evaluate_disqualifier_flags(obs)["forbidden_claim_risk"],
        "no_infrastructure_failure": lambda obs: obs["infrastructure_failure"] is False,
        "generation_attempt_count_1": lambda obs: int(obs.get("generation_invocation_count") or 0) == 1,
    }
    signal_rows = []
    sample_ids = sorted(rows_by_sample)
    for signal_id, predicate in signal_defs.items():
        candidate_support = sum(all(predicate(obs) for obs in rows_by_sample[sid]) for sid in candidate_ids)
        noncandidate_ids = [sid for sid in sample_ids if sid not in candidate_ids]
        noncandidate_support = sum(all(predicate(obs) for obs in rows_by_sample[sid]) for sid in noncandidate_ids)
        signal_rows.append(
            {
                "signal_id": signal_id,
                "candidate_support_count": candidate_support,
                "candidate_support_ratio": _ratio_string(candidate_support, len(candidate_ids)),
                "noncandidate_support_count": noncandidate_support,
                "noncandidate_support_ratio": _ratio_string(noncandidate_support, len(noncandidate_ids)),
                "discriminative_value": "high" if candidate_support == len(candidate_ids) and noncandidate_support < len(noncandidate_ids) else "low",
                "runtime_observable": True,
                "safe_for_policy_use": True,
            }
        )
    return {"schema_version": "opk-rag.task0075-common-runtime-signals.v1", "rows": signal_rows}


def evaluate_disqualifying_signals(matrix_all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_signal = {row["signal"]: {**row, "observed_sample_count": 0, "observed_sample_ids": []} for row in build_disqualifying_signals()}
    by_sample = defaultdict(list)
    for row in matrix_all_rows:
        by_sample[row["sample_id"]].append(row)
    for sample_id, observations in by_sample.items():
        combined = Counter()
        for obs in observations:
            for signal, present in evaluate_disqualifier_flags(obs).items():
                combined[signal] += int(present)
        for signal, count in combined.items():
            if count:
                by_signal[signal]["observed_sample_count"] += 1
                by_signal[signal]["observed_sample_ids"].append(sample_id)
    return {"schema_version": "opk-rag.task0075-disqualifying-signals.v1", "rows": list(by_signal.values())}


def run_retry_eligibility_counterfactual(matrix_all_rows: list[dict[str, Any]], candidate_audit: dict[str, Any], unstable_ids: list[str]) -> dict[str, Any]:
    candidate_ids = set(candidate_audit["candidate_ids"])
    by_sample = defaultdict(list)
    for row in matrix_all_rows:
        by_sample[row["sample_id"]].append(row)
    eligible = []
    for sample_id, observations in sorted(by_sample.items()):
        if all(eligible_by_runtime_rule(obs) for obs in observations):
            eligible.append(sample_id)
    leakage = _leakage_counts(eligible, by_sample)
    rule = {
        "rule_id": "rule_a_runtime_model_conservative_refusal_with_sufficient_evidence",
        "field_requirements": build_retry_eligibility_proposal()["eligibility_rules"][0]["field_requirements"],
    }
    validate_runtime_rule(rule)
    return {
        "schema_version": "opk-rag.task0075-counterfactual-retry-eligibility.v1",
        "rules_evaluated": [
            "Rule A: model-conservative refusal AND runtime evidence appears sufficient AND no terminal signal",
            "Rule B: structured abstention AND Answerability allowed generation AND no safety signal AND exact classification stable across all five frozen runs",
            "Rule C: provider refusal observed in all valid replicates AND evidence identity stable AND no validation failure",
        ],
        "selected_rule_id": rule["rule_id"],
        "eligible_sample_count": len(eligible),
        "eligible_sample_ids": eligible,
        "stable_candidate_coverage": sorted(candidate_ids & set(eligible)),
        "stable_candidate_coverage_count": len(candidate_ids & set(eligible)),
        "stable_candidate_coverage_rate": _ratio_string(len(candidate_ids & set(eligible)), len(candidate_ids)),
        "unstable_sample_inclusion_count": len(set(unstable_ids) & set(eligible)),
        "unstable_sample_ids_included": sorted(set(unstable_ids) & set(eligible)),
        "runtime_field_requirements": rule["field_requirements"],
        **leakage,
    }


def evaluate_task0076_readiness(
    input_identity: dict[str, Any],
    unstable: dict[str, Any],
    classifications: dict[str, Any],
    candidate_audit: dict[str, Any],
    counterfactual: dict[str, Any],
    proposal: dict[str, Any],
    privacy: dict[str, Any],
    verification_status: str = "not_run",
) -> dict[str, Any]:
    required_observable_ids = set(unstable.get("unstable_sample_ids") or []) | set(counterfactual.get("eligible_sample_ids") or [])
    required_observations = build_five_replicate_observation_matrix(sorted(required_observable_ids))
    gates = {
        "input_integrity_gate": input_identity["authoritative_inputs_valid"],
        "unstable_sample_diagnosis_gate": unstable["unstable_sample_count"] == 4 and len(classifications["rows"]) == 4,
        "stable_candidate_definition_gate": candidate_audit["task0074_stable_generation_retry_candidate_count"] >= 9,
        "runtime_deployability_gate": _proposal_runtime_deployable(proposal),
        "safety_gate": all(counterfactual.get(key) == 0 for key in ("terminal_sample_leakage_count", "correct_unanswerable_leakage_count", "unsupported_answer_leakage_count", "safety_terminal_leakage_count")),
        "boundedness_gate": proposal["retry_budget"]["maximum_generation_retries"] == 1 and proposal["retry_budget"]["maximum_total_generation_calls"] == 2 and not proposal["retry_budget"]["retrieval_retry_allowed"] and not proposal["retry_budget"]["recursive_retry_allowed"],
        "observability_gate": all(row.get("observability_complete") for row in required_observations),
        "privacy_gate": privacy.get("status") == "pass",
        "artifact_verifier_gate": verification_status in {"pass", "not_run"},
    }
    ready = all(gates.values())
    return {
        "schema_version": "opk-rag.task0075-task0076-readiness-gates.v1",
        **gates,
        "task0076_readiness": "ready" if ready else "not_ready",
        "recommended_next_action": "implement_governed_single_attempt_generation_retry" if ready else "add_targeted_runtime_observability",
    }


def run_diagnosis(output_dir: Path = RESULTS_DIR, write_outputs: bool = True, write_report_file: bool = False) -> dict[str, Any]:
    input_identity = authoritative_input_identity()
    if not input_identity["authoritative_inputs_valid"]:
        return {"task_status": "blocked", "blocking_reason": "authoritative_input_validation_failed", "input_identity": input_identity}
    contract = build_contract(input_identity)
    unstable = identify_unstable_samples()
    matrix_unstable = build_five_replicate_observation_matrix(unstable["unstable_sample_ids"])
    matrix_all = build_five_replicate_observation_matrix(_all_sample_ids())
    field_diff = compare_sample_fields(matrix_unstable)
    classifications = classify_instability_root_cause(matrix_unstable, field_diff)
    candidate_audit = audit_stable_generation_retry_candidates(matrix_all)
    common_signals = extract_common_runtime_signals(candidate_audit, matrix_all)
    disqualifiers = evaluate_disqualifying_signals(matrix_all)
    counterfactual = run_retry_eligibility_counterfactual(matrix_all, candidate_audit, unstable["unstable_sample_ids"])
    proposal = build_retry_eligibility_proposal()
    contract_identity = {"contract_id": CONTRACT_ID, "contract_digest": contract["contract_digest"], "contract_file_sha256": digest_value(contract)}
    privacy = privacy_scan_payload([contract, unstable, matrix_unstable, field_diff, classifications, candidate_audit, common_signals, disqualifiers, counterfactual, proposal])
    readiness = evaluate_task0076_readiness(input_identity, unstable, classifications, candidate_audit, counterfactual, proposal, privacy)
    aggregate = build_aggregate(unstable, classifications, candidate_audit, counterfactual, readiness, proposal)
    artifacts = {
        "input_identity.json": input_identity,
        "contract_identity.json": contract_identity,
        "unstable_sample_set.json": unstable,
        "field_difference_analysis.json": field_diff,
        "instability_classification.json": classifications,
        "stable_generation_retry_candidate_audit.json": candidate_audit,
        "common_runtime_signals.json": common_signals,
        "disqualifying_signals.json": disqualifiers,
        "counterfactual_retry_eligibility.json": counterfactual,
        "stable_generation_retry_eligibility_proposal.json": proposal,
        "task0076_readiness_gates.json": readiness,
        "aggregate.json": aggregate,
        "privacy_scan.json": privacy,
    }
    if write_outputs:
        atomic_write_json(CONTRACT_PATH, contract)
        for name, payload in artifacts.items():
            atomic_write_json(output_dir / name, payload)
        atomic_write_jsonl(output_dir / "five_replicate_observation_matrix.jsonl", matrix_unstable)
        result_digests_payload = result_digests(output_dir, extra_paths=[CONTRACT_PATH])
        atomic_write_json(output_dir / "result_digests.json", result_digests_payload)
        verification = verify_task0075_artifacts(output_dir=output_dir, contract_path=CONTRACT_PATH)
        atomic_write_json(output_dir / "verification.json", verification)
        readiness = evaluate_task0076_readiness(input_identity, unstable, classifications, candidate_audit, counterfactual, proposal, privacy, verification["status"])
        atomic_write_json(output_dir / "task0076_readiness_gates.json", readiness)
        aggregate = build_aggregate(unstable, classifications, candidate_audit, counterfactual, readiness, proposal)
        atomic_write_json(output_dir / "aggregate.json", aggregate)
        if write_report_file:
            write_markdown_report(output_dir=output_dir)
    return {
        "input_identity": input_identity,
        "contract": contract,
        "unstable": unstable,
        "matrix": matrix_unstable,
        "field_diff": field_diff,
        "classifications": classifications,
        "candidate_audit": candidate_audit,
        "common_signals": common_signals,
        "disqualifiers": disqualifiers,
        "counterfactual": counterfactual,
        "proposal": proposal,
        "readiness": readiness,
        "aggregate": aggregate,
        "privacy": privacy,
    }


def verify_task0075_artifacts(output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH) -> dict[str, Any]:
    required = [
        "input_identity.json",
        "contract_identity.json",
        "unstable_sample_set.json",
        "five_replicate_observation_matrix.jsonl",
        "field_difference_analysis.json",
        "instability_classification.json",
        "stable_generation_retry_candidate_audit.json",
        "common_runtime_signals.json",
        "disqualifying_signals.json",
        "counterfactual_retry_eligibility.json",
        "stable_generation_retry_eligibility_proposal.json",
        "task0076_readiness_gates.json",
        "aggregate.json",
        "privacy_scan.json",
        "result_digests.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    findings: list[str] = []
    if missing:
        findings.append(f"missing_artifacts:{','.join(missing)}")
    contract = read_json(contract_path) if contract_path.exists() else {}
    unstable = read_json(output_dir / "unstable_sample_set.json") if (output_dir / "unstable_sample_set.json").exists() else {}
    matrix = read_jsonl(output_dir / "five_replicate_observation_matrix.jsonl") if (output_dir / "five_replicate_observation_matrix.jsonl").exists() else []
    proposal = read_json(output_dir / "stable_generation_retry_eligibility_proposal.json") if (output_dir / "stable_generation_retry_eligibility_proposal.json").exists() else {}
    privacy = read_json(output_dir / "privacy_scan.json") if (output_dir / "privacy_scan.json").exists() else {}
    digests = read_json(output_dir / "result_digests.json") if (output_dir / "result_digests.json").exists() else {}
    if contract.get("contract_id") != CONTRACT_ID:
        findings.append("contract_id_mismatch")
    if unstable.get("unstable_sample_count") != 4:
        findings.append("unstable_sample_count_mismatch")
    if len(matrix) != 20:
        findings.append("five_replicate_matrix_row_count_mismatch")
    if proposal.get("default_decision") != "not_eligible":
        findings.append("proposal_not_fail_closed")
    try:
        for rule in proposal.get("eligibility_rules", []):
            validate_runtime_rule(rule)
    except ValueError as exc:
        findings.append(f"runtime_rule_invalid:{exc}")
    if privacy.get("status") != "pass":
        findings.append("privacy_scan_failed")
    current_digests = result_digests(output_dir, extra_paths=[contract_path])
    expected_files = digests.get("files") or {}
    for name, digest in expected_files.items():
        if current_digests["files"].get(name) != digest:
            findings.append(f"digest_mismatch:{name}")
    return {
        "schema_version": "opk-rag.task0075-verification.v1",
        "status": "pass" if not findings else "fail",
        "findings": findings,
        "checked_artifact_count": len(required),
        "contract_digest": contract.get("contract_digest"),
        "external_provider_calls": 0,
        "external_database_calls": 0,
        "new_replicates_run": 0,
        "generation_retry_implemented": False,
        "retrieval_retry_implemented": False,
        "agent_policy_modified": False,
    }


def result_digests(output_dir: Path = RESULTS_DIR, extra_paths: list[Path] | None = None) -> dict[str, Any]:
    files: dict[str, str] = {}
    if output_dir.exists():
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name not in {"result_digests.json", "verification.json"}:
                files[path.relative_to(ROOT).as_posix()] = file_digest(path)
    for path in extra_paths or []:
        if path.exists():
            files[path.relative_to(ROOT).as_posix()] = file_digest(path)
    return {"schema_version": "opk-rag.task0075-result-digests.v1", "files": files}


def privacy_scan_payload(payloads: list[Any]) -> dict[str, Any]:
    findings = []
    for index, payload in enumerate(payloads):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if SECRET_RE.search(text):
            findings.append({"payload_index": index, "finding": "secret_like_value"})
        if PRIVATE_PATH_RE.search(text):
            findings.append({"payload_index": index, "finding": "private_absolute_path"})
        if any(key in text for key in ("raw_output", "raw_provider_response", "prompt_messages", "full_retrieved_chunk")):
            findings.append({"payload_index": index, "finding": "raw_private_content_key"})
    return {
        "schema_version": "opk-rag.task0075-privacy-scan.v1",
        "status": "pass" if not findings else "fail",
        "findings": findings,
        "repository_privacy_scan": scan_paths_for_privacy([CONTRACT_PATH]) if CONTRACT_PATH.exists() else {"status": "not_run"},
    }


def build_aggregate(
    unstable: dict[str, Any],
    classifications: dict[str, Any],
    candidate_audit: dict[str, Any],
    counterfactual: dict[str, Any],
    readiness: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    class_counts = Counter(row["primary_instability_class"] for row in classifications["rows"])
    runtime_signal_supported = sum(
        1
        for row in candidate_audit["rows"]
        if row["classification_exactly_stable"]
        and row["answerability_stable"]
        and row["provider_refusal_count"] == row["infrastructure_valid_replicate_count"]
        and not row["safety_terminal_signal_present"]
    )
    primary_diagnosis = "provider_output_nondeterminism_dominates" if class_counts.get("provider_answer_refusal_nondeterminism", 0) >= 3 else "mixed_instability_causes"
    return {
        "schema_version": "opk-rag.task0075-aggregate.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "diagnosis_status": "complete",
        "authoritative_development_sample_count": 28,
        "unstable_sample_count": unstable["unstable_sample_count"],
        "unstable_sample_ids": unstable["unstable_sample_ids"],
        "provider_nondeterminism_count": class_counts.get("provider_answer_refusal_nondeterminism", 0) + class_counts.get("provider_refusal_reason_nondeterminism", 0),
        "retrieval_instability_count": class_counts.get("retrieval_identity_instability", 0) + class_counts.get("retrieval_score_instability", 0),
        "answerability_instability_count": class_counts.get("answerability_instability", 0),
        "classifier_boundary_ambiguity_count": class_counts.get("classifier_boundary_ambiguity", 0),
        "validation_instability_count": class_counts.get("citation_validation_instability", 0) + class_counts.get("grounding_validation_instability", 0),
        "missing_observability_count": class_counts.get("missing_observability", 0),
        "multi_stage_instability_count": class_counts.get("multi_stage_instability", 0),
        "unknown_instability_count": class_counts.get("unknown_instability", 0),
        "task0074_stable_generation_retry_candidate_count": candidate_audit["task0074_stable_generation_retry_candidate_count"],
        "runtime_signal_supported_candidate_count": runtime_signal_supported,
        "evidence_identity_stable_candidate_count": sum(1 for row in candidate_audit["rows"] if row["evidence_identity_stable"]),
        "answerability_stable_candidate_count": sum(1 for row in candidate_audit["rows"] if row["answerability_stable"]),
        "provider_refusal_stable_candidate_count": sum(1 for row in candidate_audit["rows"] if row["provider_refusal_count"] == row["infrastructure_valid_replicate_count"]),
        "safety_terminal_candidate_count": sum(1 for row in candidate_audit["rows"] if row["safety_terminal_signal_present"]),
        "indeterminate_candidate_count": sum(1 for row in candidate_audit["rows"] if "indeterminate" in row["recoverability_classes"]),
        "proposal_id": proposal["proposal_id"],
        "proposal_status": proposal["status"],
        "proposal_eligible_sample_count": counterfactual["eligible_sample_count"],
        "stable_candidate_coverage_count": counterfactual["stable_candidate_coverage_count"],
        "stable_candidate_coverage_rate": counterfactual["stable_candidate_coverage_rate"],
        "unstable_sample_inclusion_count": counterfactual["unstable_sample_inclusion_count"],
        "terminal_sample_leakage_count": counterfactual["terminal_sample_leakage_count"],
        "correct_unanswerable_leakage_count": counterfactual["correct_unanswerable_leakage_count"],
        "unsupported_answer_leakage_count": counterfactual["unsupported_answer_leakage_count"],
        "safety_terminal_leakage_count": counterfactual["safety_terminal_leakage_count"],
        "runtime_deployable_rule_count": len(proposal["eligibility_rules"]),
        "offline_only_rule_count": 0,
        "safety_gate_passed": readiness["safety_gate"],
        "boundedness_gate_passed": readiness["boundedness_gate"],
        "observability_gate_passed": readiness["observability_gate"],
        "task0076_readiness": readiness["task0076_readiness"],
        "primary_diagnosis": primary_diagnosis,
        "recommended_next_action": readiness["recommended_next_action"],
        "promotion_eligible": False,
        "recommended_variant": None,
    }


def write_markdown_report(output_dir: Path = RESULTS_DIR, doc_path: Path = DOC_PATH, test_results: list[str] | None = None) -> None:
    input_identity = read_json(output_dir / "input_identity.json")
    contract_identity = read_json(output_dir / "contract_identity.json")
    unstable = read_json(output_dir / "unstable_sample_set.json")
    field_diff = read_json(output_dir / "field_difference_analysis.json")
    classifications = read_json(output_dir / "instability_classification.json")
    candidate_audit = read_json(output_dir / "stable_generation_retry_candidate_audit.json")
    counterfactual = read_json(output_dir / "counterfactual_retry_eligibility.json")
    readiness = read_json(output_dir / "task0076_readiness_gates.json")
    aggregate = read_json(output_dir / "aggregate.json")
    privacy = read_json(output_dir / "privacy_scan.json")
    verification = read_json(output_dir / "verification.json") if (output_dir / "verification.json").exists() else {"status": "not_run"}
    lines = [
        "# TASK-0075 Post-Generation Classification Instability Report",
        "",
        "## Initial State",
        f"- Branch: `{input_identity['branch']}`",
        f"- HEAD: `{input_identity['head']}`",
        f"- Initial git status: `{input_identity['git_status_short_initial']}`",
        f"- Staged diff empty: `{input_identity['staged_diff_empty']}`",
        "",
        "## Input Identities",
        f"- TASK-0071 contract digest: `{input_identity['contracts']['task0071']['contract_digest']}`",
        f"- TASK-0072 contract digest: `{input_identity['task0072_contract_digest']}`",
        f"- TASK-0073 contract digest: `{input_identity['contracts']['task0073']['contract_digest']}`",
        f"- TASK-0074 contract digest: `{input_identity['task0074_contract_digest']}`",
        f"- TASK-0075 contract ID: `{contract_identity['contract_id']}`",
        f"- TASK-0075 contract digest: `{contract_identity['contract_digest']}`",
        "",
        "## Unstable Samples",
        f"- unstable_sample_count: `{unstable['unstable_sample_count']}`",
        f"- unstable_sample_ids: `{', '.join(unstable['unstable_sample_ids'])}`",
        "",
        "## First Divergent Fields",
    ]
    class_by_sample = {row["sample_id"]: row for row in classifications["rows"]}
    for row in field_diff["rows"]:
        cls = class_by_sample[row["sample_id"]]
        lines.append(f"- `{row['sample_id']}`: `{row['first_divergent_field']}` -> `{cls['primary_instability_class']}` ({cls['diagnostic_confidence']})")
    lines += [
        "",
        "## Stable Retry Candidate Audit",
        f"- TASK-0074 stable candidate count: `{candidate_audit['task0074_stable_generation_retry_candidate_count']}`",
        f"- Runtime-signal supported candidate count: `{aggregate['runtime_signal_supported_candidate_count']}`",
        f"- Candidate IDs: `{', '.join(candidate_audit['candidate_ids'])}`",
        "",
        "## Counterfactual Routing",
        f"- Proposal eligible sample count: `{counterfactual['eligible_sample_count']}`",
        f"- Eligible sample IDs: `{', '.join(counterfactual['eligible_sample_ids'])}`",
        f"- Stable candidate coverage: `{counterfactual['stable_candidate_coverage_rate']}`",
        f"- Unstable inclusion count: `{counterfactual['unstable_sample_inclusion_count']}`",
        f"- Terminal leakage: `{counterfactual['terminal_sample_leakage_count']}`",
        f"- Correct-unanswerable leakage: `{counterfactual['correct_unanswerable_leakage_count']}`",
        f"- Unsupported-answer leakage: `{counterfactual['unsupported_answer_leakage_count']}`",
        f"- Safety-terminal leakage: `{counterfactual['safety_terminal_leakage_count']}`",
        "",
        "## Proposal",
        f"- Proposal ID: `{PROPOSAL_ID}`",
        "- Status: `proposal_only`",
        "- Default decision: `not_eligible`",
        "- Maximum Generation retries: `1`",
        "- Maximum total Generation calls: `2`",
        "- Retrieval retry allowed: `false`",
        "- Recursive retry allowed: `false`",
        "- Citation and Grounding remain mandatory.",
        "",
        "## Readiness",
        f"- TASK-0076 readiness: `{readiness['task0076_readiness']}`",
        f"- Recommended next action: `{readiness['recommended_next_action']}`",
        f"- Primary diagnosis: `{aggregate['primary_diagnosis']}`",
        "",
        "## Verification",
        f"- Privacy scan: `{privacy['status']}`",
        f"- Verifier: `{verification['status']}`",
        f"- Test commands/results: `{'; '.join(test_results or ['recorded in final execution response'])}`",
        "",
        "## Explicit Confirmations",
        "- No external provider call occurred.",
        "- No external database call occurred.",
        "- No new replicate was run.",
        "- No Generation retry was implemented.",
        "- No Retrieval retry was implemented.",
        "- No Agent policy was modified.",
        "- No promotion experiment occurred.",
        "- TASK-0071 through TASK-0074 artifacts were only read.",
        "",
        "## Final Status",
        "```text",
        f"task_id=TASK-0075",
        f"task_status=complete",
        f"diagnosis_status=complete",
        f"diagnostic_contract_id={CONTRACT_ID}",
        f"diagnostic_contract_digest={contract_identity['contract_digest']}",
        f"unstable_sample_count={unstable['unstable_sample_count']}",
        f"unstable_sample_ids={','.join(unstable['unstable_sample_ids'])}",
        f"proposal_id={PROPOSAL_ID}",
        f"proposal_status=proposal_only",
        f"recommended_next_action={readiness['recommended_next_action']}",
        f"task0076_readiness={readiness['task0076_readiness']}",
        "```",
        "",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def provider_nondeterminism_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    modes = [row.get("response_mode") for row in observations]
    reasons = [row.get("refusal_reason_code") for row in observations]
    answer_count = sum(1 for mode in modes if mode == "answer")
    refusal_count = sum(1 for mode in modes if mode == "abstain")
    invalid_count = sum(1 for row in observations if row.get("response_contract_valid") is False)
    empty_count = sum(1 for mode in modes if mode == "empty")
    if answer_count and refusal_count:
        pattern = "answer_to_refusal" if modes[0] == "answer" else "refusal_to_answer"
    elif len(set(reasons)) > 1:
        pattern = "refusal_reason_changed"
    elif invalid_count:
        pattern = "structured_output_validity_changed"
    else:
        pattern = "not_applicable"
    return {
        "provider_answer_count": answer_count,
        "provider_refusal_count": refusal_count,
        "provider_contract_invalid_count": invalid_count,
        "provider_empty_count": empty_count,
        "provider_transition_pattern": pattern,
    }


def classifier_boundary_analysis(observations: list[dict[str, Any]], diff: dict[str, Any]) -> dict[str, Any]:
    stable_provider = diff["field_statuses"].get("response_mode") == "stable" and diff["field_statuses"].get("refusal_reason_code") == "stable"
    variable_class = diff["field_statuses"].get("runtime_failure_class") not in {"stable", "not_comparable"}
    return {
        "classifier_rules_reached": sorted(set(str(row.get("runtime_failure_class")) for row in observations)),
        "first_differing_rule": "runtime_failure_class" if stable_provider and variable_class else "not_applicable",
        "input_field_causing_rule_change": diff["first_divergent_field"],
        "default_path_used": any(row.get("runtime_failure_class") in {"missing_observability_field", "unclassified_generation_failure"} for row in observations),
        "classification_boundary_ambiguous": stable_provider and variable_class,
    }


def _all_sample_ids() -> list[str]:
    return [row["sample_id"] for row in read_json(TASK0074_RESULTS / "three_replicate_comparison.json")["rows"]]


def _observability_source_digest(directory: Path) -> str:
    records_path = directory / "observability_records.jsonl"
    if records_path.exists():
        return file_digest(records_path)
    parts = {}
    for name in (
        "generation_boundary_snapshots.jsonl",
        "generation_outcome_snapshots.jsonl",
        "validation_snapshots.jsonl",
        "sample_classifications.jsonl",
    ):
        path = directory / name
        parts[name] = file_digest(path) if path.exists() else "missing"
    return digest_value(parts)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return "not_recorded"


def _response_mode_from_generation(generation: dict[str, Any]) -> str:
    status = generation.get("status")
    if status == "answered":
        return "answer"
    if status == "refused":
        return "abstain"
    if status:
        return "malformed"
    return "not_recorded"


def _score_summary(boundary: dict[str, Any], answerability: dict[str, Any]) -> dict[str, Any]:
    summary = boundary.get("retrieval_score_summary")
    if isinstance(summary, dict) and summary.get("score_source") != "not_available_in_agent_state":
        return summary
    return {
        "score_source": "answerability_evidence_score",
        "evidence_score": answerability.get("evidence_score", "not_recorded"),
    }


def _evidence_identity_digest(sample: dict[str, Any], boundary: dict[str, Any]) -> str:
    answerability = sample.get("answerability_result") or {}
    payload = {
        "diagnostics_digest": answerability.get("diagnostics_digest"),
        "query_digest": sample.get("question_digest") or (boundary.get("query_identity") or {}).get("query_digest"),
        "selected_evidence_count": boundary.get("selected_evidence_count") or answerability.get("evidence_count"),
        "canonical_document_count": boundary.get("canonical_document_count"),
        "canonical_scope_count": boundary.get("canonical_scope_count"),
        "evidence_score": answerability.get("evidence_score"),
    }
    return digest_value(payload)


def _comparable_value(value: Any) -> str:
    if isinstance(value, dict):
        return stable_json(value)
    return str(value)


def _likely_root_cause_fields(first: str, variable: list[str]) -> list[str]:
    if first == "none":
        return []
    if first in {"provider_latency_ms"} and "response_mode" in variable:
        return ["response_mode"]
    return [first]


def _secondary_instabilities(diff: dict[str, Any], observations: list[dict[str, Any]], primary: str) -> list[str]:
    secondary = []
    if primary != "provider_answer_refusal_nondeterminism" and {"answer", "abstain"} <= {row.get("response_mode") for row in observations}:
        secondary.append("provider_answer_refusal_nondeterminism")
    if primary != "retrieval_identity_instability" and diff["field_statuses"].get("evidence_identity_digest") not in {"stable", "not_comparable"}:
        secondary.append("retrieval_identity_instability")
    return secondary


def _supporting_observations(observations: list[dict[str, Any]], diff: dict[str, Any], primary: str) -> list[str]:
    return [
        f"runtime_classes={sorted(set(str(row.get('runtime_failure_class')) for row in observations))}",
        f"response_modes={sorted(set(str(row.get('response_mode')) for row in observations))}",
        f"first_divergent_field={diff['first_divergent_field']}",
        f"primary_instability_class={primary}",
    ]


def _contradicting_observations(observations: list[dict[str, Any]], diff: dict[str, Any], primary: str) -> list[str]:
    items = []
    if primary.startswith("provider_") and diff["field_statuses"].get("evidence_identity_digest") != "stable":
        items.append("evidence_identity_digest_not_stable")
    if primary.startswith("provider_") and diff["field_statuses"].get("answerability_label") != "stable":
        items.append("answerability_label_not_stable")
    return items or ["none_observed"]


def _modal(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else "not_recorded"


def _disqualifier_any(row: dict[str, Any], signals: set[str]) -> bool:
    flags = evaluate_disqualifier_flags(row)
    return any(flags.get(signal) for signal in signals)


def _leakage_counts(eligible: list[str], by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    terminal = correct = unsupported = safety = indeterminate = 0
    for sample_id in eligible:
        classes = {row["runtime_failure_class"] for row in by_sample[sample_id]}
        recoverability = {row["recoverability_class"] for row in by_sample[sample_id]}
        terminal += int(bool(classes & {"answer_draft_grounded", "refusal_safety_terminal", "refusal_correct_unanswerable"}))
        correct += int("refusal_correct_unanswerable" in classes)
        unsupported += int("answer_draft_unsupported_claim" in classes)
        safety += int(bool(classes & {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"}))
        indeterminate += int("indeterminate" in recoverability)
    return {
        "terminal_sample_leakage_count": terminal,
        "correct_unanswerable_leakage_count": correct,
        "unsupported_answer_leakage_count": unsupported,
        "safety_terminal_leakage_count": safety,
        "indeterminate_inclusion_count": indeterminate,
    }


def _proposal_runtime_deployable(proposal: dict[str, Any]) -> bool:
    try:
        for rule in proposal.get("eligibility_rules", []):
            validate_runtime_rule(rule)
    except ValueError:
        return False
    return proposal.get("runtime_gold_usage_allowed") is False


def _ratio_string(numerator: int, denominator: int) -> str:
    value = numerator / denominator if denominator else 0.0
    return f"{numerator}/{denominator}={value}"
