from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation import phase2_governance as gov


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "phase2_sealed_baseline_v1_summary.json"
DECISION_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json"
REPORT_PATH = ROOT / "docs" / "PHASE2_DOGFOODING_STAGE_ACCEPTANCE_REPORT.md"
BASELINE_CONTRACT_PATH = gov.BASELINE_CONTRACT_PATH
REGISTRY_PATH = gov.REGISTRY_PATH

SCHEMA_VERSION = "opk-rag.phase2-stage-acceptance.v1"
DECISION_ID = "phase2-stage-acceptance-v1"
PHASE = "phase2-dogfooding"
BASELINE_ID = "phase2-sealed-baseline-v1"
RUN_ID = "phase2-sealed-baseline-v1-20260728T041333Z"
HOLDOUT_ID = "phase2-dogfooding-holdout-v1"
SCORING_CONTRACT_ID = "phase2-dogfooding-scoring-v1"
CORPUS_SNAPSHOT_ID = "phase2-corpus-v1"
NEXT_PHASE = "Phase 2B Dogfooding Quality Stabilization"
REQUIRED_EVIDENCE_DIAGNOSTIC_ID = "phase2-required-evidence-diagnostic-v1"

PRIVATE_PATH_PATTERNS = (
    ".private/",
    "phase2_holdout_v1.jsonl",
    "sample_results.jsonl",
    "generation_outputs.jsonl",
    "retrieval_traces.jsonl",
)
PUBLIC_FORBIDDEN_KEYS = {
    "question",
    "answer_text",
    "model_answer",
    "gold_claims",
    "required_claims",
    "optional_claims",
    "forbidden_claims",
    "required_evidence",
    "acceptable_evidence",
    "evidence_items",
    "evidence_bundle",
    "source_path",
    "relative_path",
    "snippet",
}
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|data|mnt|Volumes|var|private)/[^\s\"')]+")


class Phase2StageAcceptanceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> Any:
    assert_public_input_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_public_input_path(path: Path | str) -> None:
    raw = str(path).replace("\\", "/")
    if any(pattern in raw for pattern in PRIVATE_PATH_PATTERNS):
        raise Phase2StageAcceptanceError(f"Refusing to read private or per-sample sealed artifact: {raw}")
    resolved = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
        raise Phase2StageAcceptanceError(f"Path escapes repository root: {raw}")


def contains_forbidden_public_content(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in PUBLIC_FORBIDDEN_KEYS:
                return True
            if contains_forbidden_public_content(item):
                return True
        return False
    if isinstance(value, list):
        return any(contains_forbidden_public_content(item) for item in value)
    if isinstance(value, str):
        return bool(ABSOLUTE_PATH_RE.search(value))
    return False


def load_public_baseline_summary(path: Path = SUMMARY_PATH) -> dict[str, Any]:
    summary = read_json(path)
    if contains_forbidden_public_content(summary):
        raise Phase2StageAcceptanceError("Public baseline summary contains forbidden private-content fields.")
    return summary


def _metric(summary: dict[str, Any], name: str) -> dict[str, Any]:
    metric = (summary.get("aggregate_metrics") or {}).get(name)
    if not isinstance(metric, dict):
        return {"value": None, "numerator": None, "denominator": None}
    return metric


def _metric_value(summary: dict[str, Any], name: str) -> Any:
    return _metric(summary, name).get("value")


def _question_metric(summary: dict[str, Any], qtype: str, name: str) -> Any:
    return (((summary.get("question_type_metrics") or {}).get(qtype) or {}).get("metrics") or {}).get(name, {}).get("value")


def _infrastructure_failure_count(summary: dict[str, Any]) -> int | None:
    infra = summary.get("infrastructure") or {}
    if isinstance(infra.get("failure_count"), int):
        return infra["failure_count"]
    metric = _metric(summary, "infrastructure_failure_rate")
    if metric.get("numerator") is not None:
        return int(metric["numerator"])
    return None


def evaluate_evaluation_validity(summary: dict[str, Any], *, artifact_verify_status: str = "valid", baseline_verify_status: str = "valid") -> dict[str, Any]:
    checks = [
        {"check_id": "artifact_verify", "observed": artifact_verify_status, "required": "valid", "status": "pass" if artifact_verify_status == "valid" else "fail"},
        {"check_id": "baseline_verifier", "observed": baseline_verify_status, "required": "valid", "status": "pass" if baseline_verify_status == "valid" else "fail"},
        {"check_id": "run_id", "observed": summary.get("run_id"), "required": RUN_ID, "status": "pass" if summary.get("run_id") == RUN_ID else "fail"},
        {"check_id": "baseline_id", "observed": summary.get("baseline_id"), "required": BASELINE_ID, "status": "pass" if summary.get("baseline_id") == BASELINE_ID else "fail"},
        {"check_id": "sample_count", "observed": summary.get("sample_count"), "required": 40, "status": "pass" if summary.get("sample_count") == 40 else "fail"},
        {"check_id": "replicate_count", "observed": summary.get("replicate_count"), "required": 2, "status": "pass" if summary.get("replicate_count") == 2 else "fail"},
        {"check_id": "sample_execution_count", "observed": summary.get("sample_execution_count"), "required": 80, "status": "pass" if summary.get("sample_execution_count") == 80 else "fail"},
        {"check_id": "infrastructure_failure_count", "observed": _infrastructure_failure_count(summary), "required": 0, "status": "pass" if _infrastructure_failure_count(summary) == 0 else "fail"},
        {"check_id": "baseline_status", "observed": summary.get("status"), "required": "completed_hard_gates_failed", "status": "pass" if summary.get("status") == "completed_hard_gates_failed" else "fail"},
        {"check_id": "public_privacy_redaction", "observed": False, "required": False, "status": "pass" if not contains_forbidden_public_content(summary) else "fail"},
    ]
    status = "valid" if all(item["status"] == "pass" for item in checks) else "invalid"
    return {"status": status, "checks": checks}


def evaluate_hard_gate_acceptance(summary: dict[str, Any]) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    for gate_id, row in sorted((summary.get("hard_gate_results") or {}).items()):
        status = row.get("status")
        severity = "critical" if gate_id in {"unsupported_material_claim_count", "hidden_conflict_count"} else "blocking" if status == "fail" else "none"
        gates.append(
            {
                "gate_id": gate_id,
                "observed_value": row.get("observed_value"),
                "required_value": row.get("required_value"),
                "status": status,
                "severity": severity,
                "stage_impact": "blocks_product_acceptance" if status == "fail" else "does_not_block",
            }
        )
    hard_gate_status = "passed" if gates and all(row["status"] == "pass" for row in gates) else "failed"
    return {"status": hard_gate_status, "gates": gates}


def review_product_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rules = {
        "end_to_end_success_rate": ("0.1500 shows the full quality contract succeeds on a small minority of executions.", True, False),
        "answerability_accuracy": ("0.5500 is too weak for broad daily routing between answer, partial answer, correction, conflict, and refusal.", True, True),
        "safe_action_rate": ("0.8750 is useful context but cannot offset failed material-claim and conflict gates.", False, True),
        "over_abstention_rate": ("0.0417 is low overall, but answerable type success is still zero.", False, True),
        "unsafe_answer_rate": ("0.4545 indicates high risk among cases where unsafe answering can be measured.", True, True),
        "required_evidence_coverage": ("0.0056 is retained as the historical observed adapter output, but is invalid as a quality conclusion because the required evidence identity mapping is not comparable.", True, True),
        "citation_validity_rate": ("1.0000 confirms citation identifiers are valid, not that every claim is supported.", False, True),
        "citation_claim_coverage": ("0.5144 means cited support covers only about half of claim expectations.", True, True),
        "unsupported_claim_rate": ("0.0612 plus a nonzero hard gate count is a safety blocker.", True, True),
        "grounded_answer_rate": ("0.3125 is insufficient for dependable knowledge-base answering.", True, True),
    }
    rows = []
    for metric_id, (interpretation, blocks, diagnostic) in rules.items():
        row = {
            "metric_id": metric_id,
            "observation": _metric(summary, metric_id),
            "interpretation": interpretation,
            "product_impact": "blocks_dogfooding_exit" if blocks else "diagnostic_context",
            "blocks_dogfooding_exit": blocks,
            "diagnostic_only": diagnostic and not blocks,
            "requires_non_holdout_reproduction": diagnostic,
        }
        if metric_id == "required_evidence_coverage":
            row.update(
                {
                    "diagnostic_artifact_id": REQUIRED_EVIDENCE_DIAGNOSTIC_ID,
                    "observation_validity": "invalid_due_to_identity_mapping",
                    "quality_conclusion_validity": "invalid",
                    "candidate_retrieval_failure": "confirmed",
                    "evidence_bundle_loss": "present_secondary",
                }
            )
        rows.append(row)
    return rows


def review_question_types(summary: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = {
        "fully_answerable": ("stable_full_answer_not_established", True, "Complete answerable queries have E2E 0.0000, blocking daily knowledge-base querying."),
        "partially_answerable": ("safe_partial_answer_not_established", True, "Partial answer success is 0.0000, so support/unknown splitting is not reliable."),
        "no_evidence": ("safe_no_evidence_abstention_established", False, "No-evidence E2E is 1.0000 and is an accepted narrow safety capability."),
        "false_premise": ("false_premise_correction_not_established", True, "False-premise acceptance gate passes, but E2E is 0.0000; correction-contract root cause remains a hypothesis."),
        "conflicting_evidence": ("conflict_disclosure_not_established", True, "Conflict E2E is 0.0000 and hidden conflict count is nonzero, a safety blocker."),
    }
    rows = []
    for qtype, (status, blocks, interpretation) in reviews.items():
        rows.append(
            {
                "question_type": qtype,
                "sample_count": ((summary.get("question_type_metrics") or {}).get(qtype) or {}).get("sample_count"),
                "end_to_end_success_rate": _question_metric(summary, qtype, "end_to_end_success_rate"),
                "status": status,
                "interpretation": interpretation,
                "blocks_dogfooding_exit": blocks,
            }
        )
    return rows


def classify_capability_status(summary: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    accepted = [
        {"capability_id": "stable_infrastructure_execution", "status": "established", "basis": "80/80 executions completed with 0 infrastructure failures."},
        {"capability_id": "frozen_corpus_and_index_reproducibility", "status": "established", "basis": "Corpus snapshot and indexed corpus gates pass."},
        {"capability_id": "valid_citation_identifiers", "status": "established", "basis": "Citation validity rate is 1.0000 and unknown citation count is 0."},
        {"capability_id": "scope_contract_compliance", "status": "established", "basis": "Forbidden scope answer count is 0."},
        {"capability_id": "no_evidence_abstention", "status": "established", "basis": "No-evidence E2E success is 1.0000."},
        {"capability_id": "sealed_evaluation_repeatability", "status": "established", "basis": "Two replicates completed against frozen public contracts."},
    ]
    unaccepted = [
        {"capability_id": "reliable_fully_answerable_responses", "status": "not_established", "basis": "Fully answerable E2E success is 0.0000."},
        {"capability_id": "reliable_partial_answers", "status": "not_established", "basis": "Partial answer success is 0.0000."},
        {"capability_id": "conflict_disclosure", "status": "not_established", "basis": "Hidden conflict count is 8."},
        {"capability_id": "high_required_evidence_coverage", "status": "not_established", "basis": "Required evidence coverage observation 0.0056 is invalid due to identity mapping and cannot verify retrieval capability."},
        {"capability_id": "low_unsupported_claim_risk", "status": "not_established", "basis": "Unsupported material claim count is 12."},
        {"capability_id": "high_grounded_answer_coverage", "status": "not_established", "basis": "Grounded answer rate is 0.3125."},
        {"capability_id": "safe_handling_across_answerability_types", "status": "not_established", "basis": "Four of five question types have E2E 0.0000."},
    ]
    return {"accepted_capabilities": accepted, "unaccepted_capabilities": unaccepted}


def dogfooding_exit_criteria(summary: dict[str, Any], hard_gates: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = [
        ("valid_formal_baseline", True, "Formal baseline artifact is valid."),
        ("controlled_infrastructure_failures", _infrastructure_failure_count(summary) == 0, "Infrastructure failure count is 0."),
        ("stable_corpus_and_index_contracts", True, "Corpus and index gates pass."),
        ("unsupported_material_claim_gate_passes", _gate_passes(hard_gates, "unsupported_material_claim_count"), "Unsupported material claim hard gate must pass."),
        ("hidden_conflict_gate_passes", _gate_passes(hard_gates, "hidden_conflict_count"), "Hidden conflict hard gate must pass."),
        ("unknown_citation_gate_passes", _gate_passes(hard_gates, "unknown_citation_count"), "Unknown citation hard gate must pass."),
        ("forbidden_scope_gate_passes", _gate_passes(hard_gates, "forbidden_scope_answer_count"), "Forbidden scope hard gate must pass."),
        ("false_premise_acceptance_gate_passes", _gate_passes(hard_gates, "false_premise_acceptance_count"), "False premise acceptance hard gate must pass."),
        ("usable_fully_answerable_success", (_question_metric(summary, "fully_answerable", "end_to_end_success_rate") or 0) > 0, "Fully answerable E2E must be nonzero and usable."),
        ("safe_partial_answer_success", (_question_metric(summary, "partially_answerable", "end_to_end_success_rate") or 0) > 0, "Partial answer E2E must be nonzero and safe."),
        ("explicit_conflict_disclosure", _gate_passes(hard_gates, "hidden_conflict_count"), "Conflict cases must disclose conflict rather than hide it."),
        ("explainable_required_evidence_coverage", (_metric_value(summary, "required_evidence_coverage") or 0) >= 0.5, "Required evidence coverage must be explainable and materially higher."),
        ("usable_grounded_answer_rate", (_metric_value(summary, "grounded_answer_rate") or 0) >= 0.7, "Grounded answer rate must meet real-use expectations."),
        ("not_only_no_evidence_success", (_question_metric(summary, "no_evidence", "end_to_end_success_rate") == 1.0 and (_metric_value(summary, "end_to_end_success_rate") or 0) > _question_metric(summary, "no_evidence", "end_to_end_success_rate") * 0.3), "Success cannot be concentrated in no-evidence refusal."),
        ("daily_use_under_limits", False, "Current answerable and conflict behavior is not sufficient for daily use."),
    ]
    return [{"criterion_id": item[0], "status": "met" if item[1] else "not_met", "rationale": item[2]} for item in criteria]


def _gate_passes(hard_gates: dict[str, Any], gate_id: str) -> bool:
    return any(row["gate_id"] == gate_id and row["status"] == "pass" for row in hard_gates["gates"])


def build_stage_acceptance_decision(
    summary: dict[str, Any] | None = None,
    *,
    decided_at: str | None = None,
    artifact_verify_status: str = "valid",
    baseline_verify_status: str = "valid",
) -> dict[str, Any]:
    summary = summary or load_public_baseline_summary()
    evaluation_validity = evaluate_evaluation_validity(summary, artifact_verify_status=artifact_verify_status, baseline_verify_status=baseline_verify_status)
    hard_gates = evaluate_hard_gate_acceptance(summary)
    soft_status = "pending" if all((row.get("target_status") == "pending") for row in (summary.get("soft_target_observations") or {}).values()) else "frozen_or_mixed"
    if evaluation_validity["status"] != "valid":
        stage_decision = "invalid_evaluation"
        dogfooding_exit_status = "not_evaluated"
    elif hard_gates["status"] == "failed":
        stage_decision = "not_accepted"
        dogfooding_exit_status = "not_met"
    else:
        stage_decision = "accepted_with_conditions" if soft_status == "pending" else "accepted"
        dogfooding_exit_status = "met"

    capability = classify_capability_status(summary)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": DECISION_ID,
        "phase": PHASE,
        "baseline_id": BASELINE_ID,
        "run_id": RUN_ID,
        "holdout_id": HOLDOUT_ID,
        "corpus_snapshot_id": CORPUS_SNAPSHOT_ID,
        "scoring_contract_id": SCORING_CONTRACT_ID,
        "public_summary_path": relative_path(SUMMARY_PATH),
        "public_summary_digest": file_digest(SUMMARY_PATH),
        "evaluation_validity": evaluation_validity["status"],
        "evaluation_validity_checks": evaluation_validity["checks"],
        "stage_decision": stage_decision,
        "hard_gate_status": hard_gates["status"],
        "hard_gates": hard_gates["gates"],
        "soft_target_status": soft_status,
        "soft_target_decision": "keep_pending",
        "required_evidence_diagnostic": {
            "diagnostic_artifact_id": REQUIRED_EVIDENCE_DIAGNOSTIC_ID,
            "root_cause_classification": "measurement_valid_retrieval_failure_confirmed",
            "primary_root_cause": "measurement_valid_retrieval_failure_confirmed",
            "secondary_classification": "mapping_bug_resolved_for_future_runs",
            "canonical_identity_status": "frozen",
            "runtime_identity_adapter_status": "fixed",
            "gold_identity_migration_status": "verified",
            "required_evidence_measurement_status": "measurable_on_public_datasets",
            "mapping_bug_status": "resolved_for_future_runs",
            "required_evidence_metric_validity": "invalid_due_to_identity_mapping",
            "candidate_retrieval_failure": "confirmed",
            "evidence_bundle_loss": "present_secondary",
            "reranking_or_cutoff_loss": "present_secondary",
            "generation_context_loss": "not_observed",
            "serialization_identity_loss": "not_observed_after_fix",
            "deduplication_loss": "not_separately_measurable",
            "bundle_budget_loss": "not_separately_measurable",
            "gold_annotation_compatibility": "verified_on_public_datasets",
            "sealed_baseline_rerun_required": "pending_governance_decision",
            "sealed_baseline_rerun_recommendation": "pending_governance_decision",
            "sealed_baseline_v1_metric_validity": "required_evidence_metric_invalid",
            "historical_required_evidence_metric_validity": "required_evidence_metric_invalid",
            "adapter_fix_applies_to": "future_runs_and_public_diagnostics",
            "affected_metric": "required_evidence_coverage",
            "historical_observation": _metric(summary, "required_evidence_coverage"),
            "historical_observation_preserved": True,
            "quality_conclusion_validity": "invalid",
        },
        "dogfooding_exit_status": dogfooding_exit_status,
        "dogfooding_exit_criteria": dogfooding_exit_criteria(summary, hard_gates),
        "holdout_status": "exposed",
        "holdout_v1_exposure_confirmed": True,
        "holdout_v1_used_for_implementation": False,
        "holdout_v1_used_for_parameter_selection": False,
        "holdout_v1_used_for_metric_calculation": False,
        "holdout_v1_future_acceptance_eligible": False,
        "historical_sealed_baseline_validity": "retained_as_historical_run",
        "future_stage_acceptance_eligibility": "revoked_after_exposure",
        "sealed_holdout_v1_available_for_final_acceptance": False,
        "replacement_holdout_required": True,
        "replacement_holdout_id": "pending",
        "replacement_holdout_creation_status": "pending_independent_process",
        "holdout_policy": "Holdout v1 was accidentally read during TASK-0059, was not used for implementation, parameter selection, or metrics, and is no longer eligible for future stage acceptance. Historical pre-exposure baseline results are retained as a historical run.",
        "accepted_capabilities": capability["accepted_capabilities"],
        "unaccepted_capabilities": capability["unaccepted_capabilities"],
        "safety_blockers": [
            {"blocker_id": "unsupported_material_claims", "severity": "critical", "evidence": "unsupported_material_claim_count = 12"},
            {"blocker_id": "hidden_conflicts", "severity": "critical", "evidence": "hidden_conflict_count = 8"},
            {"blocker_id": "unsafe_answer_rate", "severity": "high", "evidence": "unsafe_answer_rate = 0.4545"},
        ],
        "usability_blockers": [
            {"blocker_id": "fully_answerable_e2e_zero", "severity": "high", "evidence": "fully_answerable E2E = 0.0000"},
            {"blocker_id": "partial_answer_e2e_zero", "severity": "high", "evidence": "partially_answerable E2E = 0.0000"},
            {"blocker_id": "required_evidence_metric_invalid", "severity": "high", "evidence": "required_evidence_coverage observed 0.0056, invalid due to identity mapping"},
            {"blocker_id": "grounded_answer_rate_low", "severity": "medium", "evidence": "grounded_answer_rate = 0.3125"},
        ],
        "metric_reviews": review_product_metrics(summary),
        "question_type_reviews": review_question_types(summary),
        "root_cause_hypotheses": [
            "Required evidence identity or scoring adapter mismatch may contribute to low required evidence coverage.",
            "Retrieval candidate recall or evidence bundle assembly may be missing required sources.",
            "Answerability and partial-answer execution may be collapsing partial or correction cases into unsafe answer/refusal behavior.",
            "Generation safety and claim-level validation may still permit material unsupported claims and hidden conflicts.",
        ],
        "next_phase": NEXT_PHASE,
        "next_phase_workstreams": [
            "Evaluation Mapping Validation",
            "Retrieval and Evidence Assembly",
            "Answerability and Partial Answer",
            "Generation Safety",
        ],
        "next_task_recommendation": "TASK-0054: Retrieval recall and evidence bundle loss remediation planning; do not start until TASK-0053 governance closure is accepted.",
        "privacy_attestation": {
            "read_private_holdout": True,
            "read_per_sample_results": False,
            "ran_model": False,
            "modified_system_behavior": False,
            "modified_database": False,
            "holdout_content_persisted_to_public_artifacts": False,
            "holdout_content_persisted_to_git_worktree": False,
            "holdout_content_persisted_to_test_fixtures": False,
        },
        "decision_reasons": [
            "Formal baseline execution is valid.",
            "Holdout v1 is exposed after an accidental TASK-0059 read and cannot be used for future acceptance.",
            "Product quality hard gates failed.",
            "Unsupported material claims are nonzero.",
            "Hidden conflicts are nonzero.",
            "Answerable question types have zero end-to-end success.",
            "Required evidence coverage observed 0.0056 is preserved but invalid as a quality conclusion due to identity mapping.",
            "Grounded answer rate is insufficient for daily knowledge-base answering.",
            "The strongest accepted behavior is no-evidence abstention, not reliable answering.",
        ],
        "decided_at": decided_at or utc_now(),
    }
    decision["decision_digest"] = stage_decision_digest(decision)
    return decision


def stage_decision_digest(decision: dict[str, Any]) -> str:
    payload = {key: value for key, value in decision.items() if key not in {"decision_digest", "decided_at"}}
    return stable_hash(payload)


def validate_stage_acceptance_decision(decision: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    required = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": DECISION_ID,
        "baseline_id": BASELINE_ID,
        "run_id": RUN_ID,
        "evaluation_validity": "valid",
        "stage_decision": "not_accepted",
        "dogfooding_exit_status": "not_met",
        "holdout_status": "exposed",
        "soft_target_decision": "keep_pending",
        "holdout_v1_exposure_confirmed": True,
        "holdout_v1_used_for_implementation": False,
        "holdout_v1_used_for_parameter_selection": False,
        "holdout_v1_used_for_metric_calculation": False,
        "holdout_v1_future_acceptance_eligible": False,
        "historical_sealed_baseline_validity": "retained_as_historical_run",
        "future_stage_acceptance_eligibility": "revoked_after_exposure",
        "sealed_holdout_v1_available_for_final_acceptance": False,
        "replacement_holdout_required": True,
        "replacement_holdout_id": "pending",
        "replacement_holdout_creation_status": "pending_independent_process",
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            issues.append({"severity": "error", "code": f"{key}_mismatch", "message": f"{key} must be {expected!r}"})
    if decision.get("decision_digest") != stage_decision_digest(decision):
        issues.append({"severity": "error", "code": "decision_digest_mismatch", "message": "Decision digest does not match stable payload."})
    if contains_forbidden_public_content(decision):
        issues.append({"severity": "error", "code": "private_content_leak", "message": "Decision contains forbidden private-content fields or paths."})
    gate_failures = [gate for gate in decision.get("hard_gates", []) if gate.get("status") == "fail"]
    if gate_failures and decision.get("stage_decision") != "not_accepted":
        issues.append({"severity": "error", "code": "hard_gate_overridden", "message": "Hard gate failures must force not_accepted."})
    if decision.get("soft_target_status") == "pending" and decision.get("evaluation_validity") != "valid":
        issues.append({"severity": "error", "code": "soft_target_invalidity_confusion", "message": "Soft target pending must not be treated as evaluation invalidity."})
    privacy = decision.get("privacy_attestation") or {}
    if privacy.get("read_private_holdout") is not True:
        issues.append({"severity": "error", "code": "holdout_exposure_attestation_missing", "message": "Exposure governance must attest that private holdout was read."})
    for key in (
        "holdout_content_persisted_to_public_artifacts",
        "holdout_content_persisted_to_git_worktree",
        "holdout_content_persisted_to_test_fixtures",
    ):
        if privacy.get(key) is not False:
            issues.append({"severity": "error", "code": f"{key}_not_false", "message": f"privacy_attestation.{key} must be false."})
    diagnostic = decision.get("required_evidence_diagnostic") or {}
    expected_diagnostic = {
        "diagnostic_artifact_id": REQUIRED_EVIDENCE_DIAGNOSTIC_ID,
        "root_cause_classification": "measurement_valid_retrieval_failure_confirmed",
        "required_evidence_metric_validity": "invalid_due_to_identity_mapping",
        "candidate_retrieval_failure": "confirmed",
        "evidence_bundle_loss": "present_secondary",
        "gold_annotation_compatibility": "verified_on_public_datasets",
        "sealed_baseline_rerun_required": "pending_governance_decision",
    }
    for key, expected in expected_diagnostic.items():
        if diagnostic.get(key) != expected:
            issues.append({"severity": "error", "code": f"required_evidence_diagnostic_{key}_mismatch", "message": f"required_evidence_diagnostic.{key} must be {expected!r}"})
    return issues


def verify_stage_acceptance_artifacts() -> dict[str, Any]:
    decision = read_json(DECISION_PATH)
    issues = validate_stage_acceptance_decision(decision)
    baseline = read_json(BASELINE_CONTRACT_PATH)
    stage = baseline.get("stage_acceptance") or {}
    if stage.get("decision_artifact") != relative_path(DECISION_PATH):
        issues.append({"severity": "error", "code": "baseline_contract_stage_binding_missing", "message": "Baseline contract does not bind the stage decision artifact."})
    registry = read_json(REGISTRY_PATH)
    registry_item = next((item for item in registry.get("benchmarks", []) if item.get("benchmark_id") == DECISION_ID), None)
    if registry_item is None:
        issues.append({"severity": "error", "code": "registry_governance_artifact_missing", "message": "Benchmark registry does not register the governance decision artifact."})
    elif registry_item.get("role") != "stage_acceptance_decision":
        issues.append({"severity": "error", "code": "registry_role_mismatch", "message": "Registry artifact role must be stage_acceptance_decision."})
    return {"status": "valid" if not issues else "invalid", "issues": issues, "decision_digest": decision.get("decision_digest")}


def status_payload() -> dict[str, Any]:
    summary = load_public_baseline_summary()
    decision = build_stage_acceptance_decision(summary)
    return {
        "status": "reviewed",
        "run_id": decision["run_id"],
        "evaluation_validity": decision["evaluation_validity"],
        "stage_decision": decision["stage_decision"],
        "dogfooding_exit_status": decision["dogfooding_exit_status"],
        "hard_gate_status": decision["hard_gate_status"],
        "soft_target_status": decision["soft_target_status"],
        "holdout_status": decision["holdout_status"],
        "decision_digest": decision["decision_digest"],
    }


def write_decision_artifact(decision: dict[str, Any]) -> None:
    write_json(DECISION_PATH, decision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review Phase 2 dogfooding stage acceptance from public artifacts.")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("review", "verify", "status"):
        command = sub.add_parser(name)
        command.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    emit_json = bool(args.json)

    try:
        if args.command == "review":
            decision = build_stage_acceptance_decision()
            write_decision_artifact(decision)
            payload = {"status": "written", "decision_path": relative_path(DECISION_PATH), "decision_digest": decision["decision_digest"], "stage_decision": decision["stage_decision"]}
        elif args.command == "verify":
            payload = verify_stage_acceptance_artifacts()
        elif args.command == "status":
            payload = status_payload()
        else:
            raise Phase2StageAcceptanceError(f"Unknown command: {args.command}")
    except Phase2StageAcceptanceError as exc:
        payload = {"status": "blocked", "error": str(exc)}
        if emit_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    if emit_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"{args.command}: {payload.get('status', 'ok')}")
    return 0 if payload.get("status") not in {"blocked", "invalid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
