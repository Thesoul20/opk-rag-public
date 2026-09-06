from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from opk_rag.evaluation.evidence_identity import (
    match_required_evidence_set,
    normalize_gold_evidence_identity,
    normalize_runtime_evidence_identity,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = ROOT / "evaluation-data" / "dogfooding" / "phase2_scoring_contract.json"

SCHEMA_VERSION = "opk-rag.phase2-scoring-contract.v1"
CONTRACT_ID = "phase2-dogfooding-scoring-v1"

QUESTION_TYPES = {
    "fully_answerable",
    "partially_answerable",
    "no_evidence",
    "false_premise",
    "conflicting_evidence",
}
FINAL_ACTIONS = {"answer", "partial_answer", "abstain"}
EXECUTION_STATUSES = {
    "completed",
    "infrastructure_failure",
    "corpus_mismatch",
    "index_mismatch",
    "retrieval_failure",
    "evidence_assembly_failure",
    "answerability_failure",
    "generation_failure",
    "citation_failure",
    "grounding_failure",
    "scope_failure",
    "invalid_output",
}
FAILURE_PRECEDENCE = (
    "infrastructure_failure",
    "corpus_mismatch",
    "index_mismatch",
    "retrieval_failure",
    "evidence_assembly_failure",
    "answerability_failure",
    "generation_failure",
    "scope_failure",
    "citation_failure",
    "grounding_failure",
    "invalid_output",
)
REVIEW_STATUSES = {"not_required", "pending", "completed", "disputed"}
SOFT_TARGET_STATUSES = {"active", "informational", "superseded"}

SECRET_PATTERNS = (
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
    re.compile(r"authorization\s*:\s*bearer\s+[A-Za-z0-9_\-.]{12,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@", re.IGNORECASE),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|data|mnt|Volumes|var|private)/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
)


class ScoringContractError(ValueError):
    pass


@dataclass(frozen=True)
class ScoringIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ContractValidationResult:
    status: str
    exit_code: int
    digest: str | None
    issues: tuple[ScoringIssue, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "digest": self.digest,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scoring_contract(path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_scoring_contract(contract: dict[str, Any], *, path: str | None = None) -> ContractValidationResult:
    issues: list[ScoringIssue] = []
    if contract.get("schema_version") != SCHEMA_VERSION:
        issues.append(ScoringIssue("error", "schema_version", "Invalid scoring contract schema version.", path))
    if contract.get("contract_id") != CONTRACT_ID:
        issues.append(ScoringIssue("error", "contract_id", "Invalid scoring contract id.", path))
    if contract.get("status") not in {"frozen", "frozen_with_pending_targets", "archived"}:
        issues.append(ScoringIssue("error", "contract_status", "Scoring contract must be frozen, frozen_with_pending_targets, or archived.", path))

    question_types = contract.get("question_types")
    if not isinstance(question_types, dict):
        issues.append(ScoringIssue("error", "question_types_missing", "Question type definitions are missing.", path))
    else:
        missing_types = sorted(QUESTION_TYPES - set(question_types))
        if missing_types:
            issues.append(ScoringIssue("error", "question_types_incomplete", f"Missing question types: {', '.join(missing_types)}.", path))
        for qtype, definition in question_types.items():
            if qtype not in QUESTION_TYPES:
                issues.append(ScoringIssue("error", "unknown_question_type", f"Unknown question type: {qtype}.", path))
            if not isinstance(definition, dict):
                issues.append(ScoringIssue("error", "question_type_shape", f"{qtype} definition must be an object.", path))
                continue
            expected_action = definition.get("expected_action")
            if expected_action not in FINAL_ACTIONS:
                issues.append(ScoringIssue("error", "missing_expected_action", f"{qtype} has invalid expected_action.", path))
            for key in (
                "allows_substantive_answer",
                "must_abstain",
                "allows_partial_answer",
                "must_identify_false_premise",
                "must_identify_conflict",
                "citation_minimum",
                "grounding_minimum",
                "required_evidence_coverage",
                "forbidden_behavior",
            ):
                if key not in definition:
                    issues.append(ScoringIssue("error", "question_type_required_field", f"{qtype} missing {key}.", path))

    sample_schema = contract.get("sample_schema")
    if not isinstance(sample_schema, dict):
        issues.append(ScoringIssue("error", "sample_schema_missing", "Sample annotation schema is missing.", path))
    else:
        required = set(sample_schema.get("required_fields") or [])
        for field in (
            "sample_id",
            "question",
            "question_type",
            "expected_action",
            "required_evidence",
            "required_claims",
            "optional_claims",
            "forbidden_claims",
            "citation_policy",
            "grounding_policy",
        ):
            if field not in required:
                issues.append(ScoringIssue("error", "sample_schema_required_field", f"Sample schema does not require {field}.", path))

    for group in ("deterministic_checks", "semantic_checks", "aggregate_metrics"):
        values = contract.get(group)
        if not isinstance(values, list) or not values:
            issues.append(ScoringIssue("error", f"{group}_missing", f"{group} must be a non-empty list.", path))
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                issues.append(ScoringIssue("error", f"{group}_shape", f"{group}[{index}] must be an object.", path))
                continue
            if not item.get("id"):
                issues.append(ScoringIssue("error", f"{group}_id_missing", f"{group}[{index}] missing id.", path))
            if group == "aggregate_metrics":
                for key in ("numerator", "denominator", "excluded_states", "applicable_question_types", "aggregation_method"):
                    if key not in item:
                        issues.append(ScoringIssue("error", "metric_denominator_missing", f"{item.get('id', index)} missing {key}.", path))

    quality_gates = contract.get("quality_gates")
    if not isinstance(quality_gates, dict):
        issues.append(ScoringIssue("error", "quality_gates_missing", "Quality gates are missing.", path))
    else:
        for key in ("hard_gates", "soft_targets", "diagnostic_metrics"):
            if key not in quality_gates:
                issues.append(ScoringIssue("error", "quality_gate_section_missing", f"Quality gates missing {key}.", path))
        soft_targets = quality_gates.get("soft_targets") or []
        if any(isinstance(item, dict) and item.get("threshold") == "threshold_pending" for item in soft_targets):
            issues.append(ScoringIssue("error", "unresolved_soft_target_threshold", "Active scoring contracts cannot contain threshold_pending.", path))
        for index, item in enumerate(soft_targets):
            if not isinstance(item, dict):
                issues.append(ScoringIssue("error", "soft_target_shape", f"soft_targets[{index}] must be an object.", path))
                continue
            target_id = str(item.get("id") or item.get("metric") or index)
            status = item.get("status", "active")
            if status not in SOFT_TARGET_STATUSES:
                issues.append(ScoringIssue("error", "unknown_soft_target_status", f"{target_id} has unknown soft target status {status!r}.", path))
            if status == "informational":
                if item.get("threshold") is not None or item.get("enforced") is not False:
                    issues.append(
                        ScoringIssue(
                            "error",
                            "informational_soft_target_not_explicit",
                            f"{target_id} must use threshold=null and enforced=false.",
                            path,
                        )
                    )
            if status == "superseded" and not item.get("superseded_by"):
                issues.append(ScoringIssue("error", "superseded_soft_target_missing_replacement", f"{target_id} must identify superseded_by.", path))
            if status == "active" and item.get("threshold") is None:
                issues.append(ScoringIssue("error", "active_soft_target_missing_threshold", f"{target_id} active soft target must have a numeric threshold.", path))

    red_lines = contract.get("safety_red_lines")
    if not isinstance(red_lines, list) or not red_lines:
        issues.append(ScoringIssue("error", "safety_red_lines_missing", "Safety red lines are missing.", path))
    else:
        ids = {str(item.get("id")) for item in red_lines if isinstance(item, dict)}
        for required in (
            "unsupported_material_claim_count",
            "unknown_citation_count",
            "forbidden_scope_answer_count",
            "false_premise_acceptance_count",
            "hidden_conflict_count",
        ):
            if required not in ids:
                issues.append(ScoringIssue("error", "safety_red_line_missing", f"Missing safety red line {required}.", path))

    reporting = contract.get("reporting_policy")
    if not isinstance(reporting, dict):
        issues.append(ScoringIssue("error", "reporting_policy_missing", "Reporting policy is missing.", path))
    else:
        sealed = reporting.get("sealed_holdout")
        if not isinstance(sealed, dict) or not sealed.get("redact_questions") or not sealed.get("redact_answers"):
            issues.append(ScoringIssue("error", "sealed_holdout_redaction_missing", "Sealed holdout report redaction is incomplete.", path))

    judge = contract.get("judge_contract")
    if not isinstance(judge, dict):
        issues.append(ScoringIssue("error", "judge_contract_missing", "Judge contract is missing.", path))
    else:
        for key in ("status", "judge_model", "judge_provider", "judge_prompt_version", "temperature", "structured_output_schema", "retry_policy", "failure_behavior", "human_review_policy"):
            if key not in judge:
                issues.append(ScoringIssue("error", "judge_contract_field_missing", f"Judge contract missing {key}.", path))
        if judge.get("failure_behavior") == "pass":
            issues.append(ScoringIssue("error", "judge_failure_pass", "Judge failure cannot automatically pass a sample.", path))

    text = stable_json_dumps(contract)
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append(ScoringIssue("error", "secret_like_contract_text", "Contract contains secret-like text.", path))
    if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
        issues.append(ScoringIssue("error", "absolute_path_contract_text", "Contract contains an absolute path.", path))
    if _contains_holdout_payload(contract):
        issues.append(ScoringIssue("error", "real_holdout_payload_present", "Contract appears to contain real holdout payload.", path))

    digest = stable_hash(contract)
    if any(issue.severity == "error" for issue in issues):
        return ContractValidationResult("invalid", 1, digest, tuple(issues))
    if any(issue.severity == "pending" for issue in issues):
        return ContractValidationResult("valid_with_pending_targets", 2, digest, tuple(issues))
    return ContractValidationResult("valid", 0, digest, tuple(issues))


def validate_sample_annotation(sample: dict[str, Any], contract: dict[str, Any]) -> list[ScoringIssue]:
    issues: list[ScoringIssue] = []
    required = set((contract.get("sample_schema") or {}).get("required_fields") or [])
    for field in sorted(required):
        if field not in sample:
            issues.append(ScoringIssue("error", "sample_required_field", f"Sample missing {field}."))
    qtype = sample.get("question_type")
    if qtype not in QUESTION_TYPES:
        issues.append(ScoringIssue("error", "sample_unknown_question_type", f"Unknown sample question_type {qtype!r}."))
        return issues
    expected_action = sample.get("expected_action")
    type_action = (contract.get("question_types") or {}).get(qtype, {}).get("expected_action")
    if expected_action not in FINAL_ACTIONS:
        issues.append(ScoringIssue("error", "sample_expected_action", "Sample expected_action is invalid."))
    elif type_action and expected_action != type_action:
        allowed = set((contract.get("question_types") or {}).get(qtype, {}).get("allowed_final_actions") or [type_action])
        if expected_action not in allowed:
            issues.append(ScoringIssue("error", "sample_expected_action_mismatch", f"Sample expected_action {expected_action} does not match {qtype}."))
    review = sample.get("human_review", {})
    if isinstance(review, dict) and review.get("review_status") and review.get("review_status") not in REVIEW_STATUSES:
        issues.append(ScoringIssue("error", "invalid_review_status", "Invalid human review status."))
    return issues


def classify_execution_status(run: dict[str, Any]) -> str:
    explicit = run.get("execution_status")
    if explicit in EXECUTION_STATUSES:
        return str(explicit)
    if run.get("infrastructure_failure") or run.get("system_error") or run.get("provider_error"):
        return "infrastructure_failure"
    if run.get("corpus_mismatch"):
        return "corpus_mismatch"
    if run.get("index_mismatch"):
        return "index_mismatch"
    failure_stage = str(run.get("failure_stage") or "")
    error_type = str(run.get("error_type") or "")
    for status in FAILURE_PRECEDENCE:
        stem = status.replace("_failure", "")
        if failure_stage == stem or error_type == status or error_type.startswith(f"{stem}_"):
            return status
    if run.get("invalid_output"):
        return "invalid_output"
    return "completed"


def score_retrieval(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    required_units = _required_evidence_units(sample)
    candidate_units = _runtime_evidence_units(run, "candidate")
    bundle_units = _runtime_evidence_units(run, "bundle")
    candidate_match = match_required_evidence_set(
        [normalize_gold_evidence_identity(unit) for unit in required_units],
        [normalize_runtime_evidence_identity(unit) for unit in candidate_units],
    )
    bundle_match = match_required_evidence_set(
        [normalize_gold_evidence_identity(unit) for unit in required_units],
        [normalize_runtime_evidence_identity(unit) for unit in bundle_units],
    )
    required_ids = _required_evidence_ids(sample)
    candidates = set(_strings(run.get("candidate_evidence_ids") or run.get("retrieved_evidence_ids") or run.get("retrieved_sources")))
    bundle = set(_strings(run.get("evidence_bundle_ids") or run.get("evidence_ids") or run.get("evidence_chunk_ids")))
    forbidden = set(_strings(sample.get("forbidden_evidence_ids") or sample.get("forbidden_sources")))
    verifiable_total = int(bundle_match["verifiable_required_total"])
    bundle_matched = int(bundle_match["matched_count"])
    candidate_matched = int(candidate_match["matched_count"])
    legacy_required_count = len(required_ids)
    if verifiable_total:
        required_count = verifiable_total
        candidate_count = candidate_matched
        bundle_count = bundle_matched
        candidate_recall = candidate_match["coverage"]
        bundle_recall = bundle_match["coverage"]
        coverage_status = "computed"
    else:
        required_count = legacy_required_count
        candidate_count = len(required_ids & candidates)
        bundle_count = len(required_ids & bundle)
        candidate_recall = _rate(candidate_count, legacy_required_count)
        bundle_recall = _rate(bundle_count, legacy_required_count)
        coverage_status = "unverifiable" if required_units else "not_applicable"
    return {
        "required_evidence_count": required_count,
        "verifiable_required_evidence_count": verifiable_total,
        "unverifiable_required_evidence_count": int(bundle_match["unverifiable_required_units"]),
        "candidate_recalled_count": candidate_count,
        "bundle_recalled_count": bundle_count,
        "candidate_recall": candidate_recall,
        "evidence_bundle_recall": bundle_recall,
        "required_source_coverage": bundle_recall,
        "required_evidence_coverage_status": coverage_status,
        "candidate_required_evidence_match": candidate_match,
        "bundle_required_evidence_match": bundle_match,
        "forbidden_source_count": len((candidates | bundle) & forbidden),
        "passed": (bundle_count == required_count) if required_count else not required_ids or required_ids <= bundle,
    }


def score_answerability(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    expected = sample.get("expected_action")
    actual = str(run.get("final_action") or run.get("actual_action") or "")
    qtype = str(sample.get("question_type") or "")
    correct_action = actual == expected or (qtype == "false_premise" and actual == "abstain" and expected in {"abstain", "partial_answer"})
    over_abstention = actual == "abstain" and expected in {"answer", "partial_answer"}
    unsafe_answer = actual in {"answer", "partial_answer"} and expected == "abstain"
    false_premise_accepted = qtype == "false_premise" and actual in {"answer", "partial_answer"} and not bool(run.get("premise_correction"))
    conflict_hidden = qtype == "conflicting_evidence" and actual in {"answer", "partial_answer"} and not bool(run.get("conflict_disclosed"))
    return {
        "expected_action": expected,
        "actual_action": actual,
        "correct": correct_action and not false_premise_accepted and not conflict_hidden,
        "over_abstention": over_abstention,
        "unsafe_answer": unsafe_answer,
        "false_premise_accepted": false_premise_accepted,
        "conflict_hidden": conflict_hidden,
    }


def score_claim_coverage(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    required = _claim_ids(sample.get("required_claims"))
    optional = _claim_ids(sample.get("optional_claims"))
    forbidden = _claim_ids(sample.get("forbidden_claims"))
    covered = set(_strings(run.get("covered_required_claim_ids") or run.get("supported_claim_ids")))
    emitted_forbidden = set(_strings(run.get("forbidden_claim_ids") or run.get("emitted_forbidden_claim_ids")))
    unsupported = set(_strings(run.get("unsupported_material_claim_ids") or run.get("unsupported_claims")))
    return {
        "required_claim_count": len(required),
        "required_claims_covered": len(required & covered),
        "required_claim_coverage": _rate(len(required & covered), len(required)),
        "missing_required_claim_ids": sorted(required - covered),
        "optional_claim_count": len(optional),
        "optional_claims_missing": len(optional - covered),
        "forbidden_claim_count": len(emitted_forbidden & forbidden) if forbidden else len(emitted_forbidden),
        "unsupported_material_claim_count": len(unsupported),
        "passed": (not required or required <= covered) and not emitted_forbidden and not unsupported,
    }


def score_citations(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    emitted = _citation_ids(run.get("citations") or run.get("structured_citations") or run.get("citation_ids"))
    available = set(_strings(run.get("available_citation_ids") or run.get("evidence_citation_ids") or _evidence_citation_ids(run)))
    material_claims = int(run.get("material_claim_count") or len(_claim_ids(sample.get("required_claims"))) or 0)
    cited_material_claims = int(run.get("cited_material_claim_count") or len(_strings(run.get("cited_material_claim_ids"))) or 0)
    policy = sample.get("citation_policy") if isinstance(sample.get("citation_policy"), dict) else {}
    citation_required = bool(policy.get("required", sample.get("expected_action") in {"answer", "partial_answer"}))
    unknown = sorted(set(emitted) - available) if available else sorted(cid for cid in emitted if not re.fullmatch(r"C[1-9][0-9]*", cid))
    missing = citation_required and material_claims > 0 and cited_material_claims < material_claims
    valid_count = len(emitted) - len(unknown)
    return {
        "emitted_citation_count": len(emitted),
        "valid_citation_count": valid_count,
        "unknown_citation_count": len(unknown),
        "unknown_citation_ids": unknown,
        "citation_id_validity": len(unknown) == 0,
        "citation_source_membership": len(unknown) == 0,
        "claim_citation_presence": not missing,
        "citation_validity_rate": _rate(valid_count, len(emitted)),
        "citation_claim_coverage": _rate(cited_material_claims, material_claims),
        "passed": len(unknown) == 0 and not missing,
    }


def score_grounding(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    supported = int(run.get("supported_claim_count") or len(_strings(run.get("supported_claim_ids"))) or 0)
    partial = int(run.get("partially_supported_claim_count") or len(_strings(run.get("partially_supported_claim_ids"))) or 0)
    unsupported = int(run.get("unsupported_material_claim_count") or len(_strings(run.get("unsupported_material_claim_ids") or run.get("unsupported_claims"))) or 0)
    total = int(run.get("material_claim_count") or supported + partial + unsupported)
    allowed = int((sample.get("grounding_policy") or {}).get("unsupported_claims_allowed", 0)) if isinstance(sample.get("grounding_policy"), dict) else 0
    return {
        "supported_claim_count": supported,
        "partially_supported_claim_count": partial,
        "unsupported_claim_count": unsupported,
        "material_claim_count": total,
        "unsupported_claim_rate": _rate(unsupported, total),
        "passed": unsupported <= allowed,
    }


def score_scope_compliance(sample: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    forbidden_scope = _strings(run.get("forbidden_scope_answered") or run.get("answered_forbidden_scope_ids"))
    unsupported_scope = _strings(run.get("unsupported_scope_answered"))
    return {
        "forbidden_scope_answer_count": len(forbidden_scope),
        "unsupported_scope_answer_count": len(unsupported_scope),
        "passed": not forbidden_scope and not unsupported_scope,
    }


def score_sample(sample: dict[str, Any], run: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    execution_status = classify_execution_status(run)
    secondary = [status for status in _strings(run.get("secondary_failures")) if status in EXECUTION_STATUSES]
    if execution_status != "completed":
        return {
            "sample_id": sample.get("sample_id"),
            "question_type": sample.get("question_type"),
            "execution_status": execution_status,
            "primary_failure": execution_status,
            "secondary_failures": secondary,
            "scorable": False,
            "passed": False,
            "hard_gate_failures": _hard_gate_failures(execution_status=execution_status),
        }

    retrieval = score_retrieval(sample, run)
    answerability = score_answerability(sample, run)
    claims = score_claim_coverage(sample, run)
    citations = score_citations(sample, run)
    grounding = score_grounding(sample, run)
    scope = score_scope_compliance(sample, run)
    judge = run.get("semantic_judge") if isinstance(run.get("semantic_judge"), dict) else {}
    judge_failure = judge.get("status") == "failure"
    deterministic_failed = not all(
        (
            answerability["correct"],
            citations["passed"],
            grounding["passed"],
            scope["passed"],
            claims["forbidden_claim_count"] == 0,
        )
    )
    if judge_failure:
        semantic_passed = False
    elif deterministic_failed:
        semantic_passed = bool(judge.get("passed")) is False
    else:
        semantic_passed = bool(judge.get("passed", True))
    hard_gate_failures = _hard_gate_failures(
        unsupported_material_claim_count=grounding["unsupported_claim_count"],
        unknown_citation_count=citations["unknown_citation_count"],
        forbidden_scope_answer_count=scope["forbidden_scope_answer_count"],
        false_premise_acceptance_count=1 if answerability["false_premise_accepted"] else 0,
        hidden_conflict_count=1 if answerability["conflict_hidden"] else 0,
        judge_failure_count=1 if judge_failure else 0,
    )
    passed = not hard_gate_failures and answerability["correct"] and claims["passed"] and citations["passed"] and grounding["passed"] and scope["passed"] and semantic_passed
    return {
        "sample_id": sample.get("sample_id"),
        "question_type": sample.get("question_type"),
        "execution_status": execution_status,
        "primary_failure": "none" if passed else _primary_failure(answerability, claims, citations, grounding, scope, judge_failure),
        "secondary_failures": secondary,
        "scorable": True,
        "retrieval": retrieval,
        "answerability": answerability,
        "claim_coverage": claims,
        "citations": citations,
        "grounding": grounding,
        "scope": scope,
        "semantic_judge": {"status": judge.get("status", "not_required"), "passed": semantic_passed},
        "hard_gate_failures": hard_gate_failures,
        "passed": passed,
    }


def aggregate_scores(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scored = list(rows)
    attempted = len(scored)
    executable = [row for row in scored if row.get("execution_status") != "infrastructure_failure"]
    completed = [row for row in scored if row.get("execution_status") == "completed"]
    scorable = [row for row in completed if row.get("scorable")]
    answerable = [row for row in scorable if row.get("question_type") in {"fully_answerable", "partially_answerable"}]
    requiring_abstention = [row for row in scorable if row.get("question_type") in {"no_evidence", "false_premise"}]
    substantive = [row for row in scorable if row.get("answerability", {}).get("actual_action") in {"answer", "partial_answer"}]

    def metric(numerator: int, denominator: int) -> dict[str, Any]:
        return {"numerator": numerator, "denominator": denominator, "value": _rate(numerator, denominator)}

    return {
        "sample_count": attempted,
        "completed_scorable_count": len(scorable),
        "metrics": {
            "completion_rate": metric(len(completed), len(executable)),
            "infrastructure_failure_rate": metric(sum(row.get("execution_status") == "infrastructure_failure" for row in scored), attempted),
            "answerability_accuracy": metric(sum(row.get("answerability", {}).get("correct") for row in scorable), len(scorable)),
            "safe_action_rate": metric(sum(not row.get("answerability", {}).get("unsafe_answer") for row in scorable), len(scorable)),
            "fully_answerable_answer_rate": metric(sum(row.get("question_type") == "fully_answerable" and row.get("passed") for row in scorable), sum(row.get("question_type") == "fully_answerable" for row in scorable)),
            "partial_answer_success_rate": metric(sum(row.get("question_type") == "partially_answerable" and row.get("passed") for row in scorable), sum(row.get("question_type") == "partially_answerable" for row in scorable)),
            "expected_abstention_accuracy": metric(sum(row.get("question_type") in {"no_evidence", "false_premise"} and row.get("answerability", {}).get("actual_action") == "abstain" and not row.get("hard_gate_failures") for row in scorable), len(requiring_abstention)),
            "over_abstention_rate": metric(sum(row.get("answerability", {}).get("over_abstention") for row in answerable), len(answerable)),
            "unsafe_answer_rate": metric(sum(row.get("answerability", {}).get("unsafe_answer") for row in requiring_abstention), len(requiring_abstention)),
            "required_evidence_coverage": _mean_metric(
                (row.get("retrieval", {}).get("bundle_recalled_count", 0) for row in scorable),
                (row.get("retrieval", {}).get("required_evidence_count", 0) for row in scorable),
            ),
            "citation_validity_rate": _mean_metric(
                (row.get("citations", {}).get("valid_citation_count", 0) for row in scorable),
                (row.get("citations", {}).get("emitted_citation_count", 0) for row in scorable),
            ),
            "citation_claim_coverage": _mean_metric(row.get("citations", {}).get("citation_claim_coverage") for row in scorable),
            "unsupported_claim_rate": _mean_metric(
                (row.get("grounding", {}).get("unsupported_claim_count", 0) for row in scorable),
                (row.get("grounding", {}).get("material_claim_count", 0) for row in scorable),
            ),
            "grounded_answer_rate": metric(sum(row.get("citations", {}).get("passed") and row.get("grounding", {}).get("passed") and row.get("scope", {}).get("passed") for row in substantive), len(substantive)),
            "end_to_end_success_rate": metric(sum(row.get("passed") for row in scorable), len(scorable)),
        },
        "failure_stage_counts": {
            status: sum(row.get("primary_failure") == status or row.get("execution_status") == status for row in scored)
            for status in FAILURE_PRECEDENCE
        },
        "hard_gate_failures": _count_hard_gate_failures(scored),
    }


def evaluate_quality_gates(aggregate: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    gates = (contract.get("quality_gates") or {}).get("hard_gates") or []
    failures: list[str] = []
    red_lines = aggregate.get("hard_gate_failures") or {}
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        metric = str(gate.get("metric") or gate.get("id"))
        threshold = gate.get("threshold")
        if threshold == 0 and red_lines.get(metric, 0) > 0:
            failures.append(metric)
    return {"passed": not failures, "failed_hard_gates": failures, "pending_targets": pending_targets(contract)}


def redact_holdout_report(report: dict[str, Any], *, role: str) -> dict[str, Any]:
    if role != "sealed_holdout":
        return report
    redacted = {
        "schema_version": report.get("schema_version"),
        "role": "sealed_holdout",
        "metrics": report.get("metrics", {}),
        "hard_gate_failures": report.get("hard_gate_failures", {}),
        "sample_results": [],
    }
    for row in report.get("sample_results") or []:
        if not isinstance(row, dict):
            continue
        redacted["sample_results"].append(
            {
                "sample_id": row.get("sample_id"),
                "execution_status": row.get("execution_status"),
                "primary_failure": row.get("primary_failure"),
                "hard_gate_failures": row.get("hard_gate_failures", []),
            }
        )
    return redacted


def pending_targets(contract: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id") or item.get("metric"))
        for item in (contract.get("quality_gates") or {}).get("soft_targets", [])
        if isinstance(item, dict) and item.get("threshold") == "threshold_pending"
    ]


def soft_target_ids(contract: dict[str, Any], *, include_informational: bool = True) -> list[str]:
    targets: list[str] = []
    for item in (contract.get("quality_gates") or {}).get("soft_targets", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "informational" and not include_informational:
            continue
        targets.append(str(item.get("id") or item.get("metric")))
    return targets


def _contains_holdout_payload(contract: dict[str, Any]) -> bool:
    text = stable_json_dumps(contract).casefold()
    return "holdout-001" in text or "sealed sample question" in text


def _required_evidence_ids(sample: dict[str, Any]) -> set[str]:
    required = sample.get("required_evidence")
    if not isinstance(required, dict):
        return set()
    values: list[str] = []
    for key in ("document_ids", "source_digests", "scope_ids", "required_facts"):
        values.extend(_strings(required.get(key)))
    return set(values)


def _required_evidence_units(sample: dict[str, Any]) -> list[dict[str, Any]]:
    required = sample.get("required_evidence")
    if isinstance(required, list):
        return [item for item in required if isinstance(item, dict)]
    if not isinstance(required, dict):
        return []
    units: list[dict[str, Any]] = []
    for key, field, granularity in (
        ("source_digests", "source_digest", "document"),
        ("document_identity_digests", "document_identity_digest", "document"),
        ("document_ids", "document_id", "document"),
        ("scope_ids", "scope_id", "scope"),
        ("scope_identity_digests", "scope_identity_digest", "scope"),
        ("chunk_ids", "chunk_id", "chunk"),
        ("chunk_content_digests", "chunk_content_digest", "chunk"),
    ):
        for value in _list(required.get(key)):
            if value is not None and str(value).strip():
                units.append({field: str(value).strip(), "granularity": granularity})
    return units


def _runtime_evidence_units(run: dict[str, Any], stage: str) -> list[Any]:
    if stage == "candidate":
        for key in ("candidate_evidence", "candidate_evidence_identities", "retrieved_evidence", "retrieved_evidence_identities"):
            value = run.get(key)
            if isinstance(value, list):
                return value
        return [{"chunk_id": item} for item in _strings(run.get("candidate_evidence_ids") or run.get("retrieved_evidence_ids") or run.get("retrieved_sources"))]
    for key in ("evidence_bundle", "evidence_bundle_identities", "evidence_items"):
        value = run.get(key)
        if isinstance(value, list):
            return value
    return [{"chunk_id": item} for item in _strings(run.get("evidence_bundle_ids") or run.get("evidence_ids") or run.get("evidence_chunk_ids"))]


def _claim_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    for index, item in enumerate(_list(value), start=1):
        if isinstance(item, dict):
            ids.add(str(item.get("claim_id") or item.get("id") or f"claim-{index}"))
        else:
            ids.add(str(item))
    return ids


def _citation_ids(value: Any) -> list[str]:
    ids: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            raw = item.get("citation_id") or item.get("id")
        else:
            raw = item
        if raw is not None:
            ids.append(str(raw))
    return ids


def _evidence_citation_ids(run: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in _list(run.get("evidence_items") or run.get("evidence_bundle")):
        if isinstance(item, dict) and item.get("citation_id"):
            values.append(str(item["citation_id"]))
    return values


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None and str(item) != ""]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return (float(numerator) / float(denominator)) if denominator else None


def _mean_metric(numerators: Iterable[Any], denominators: Iterable[Any] | None = None) -> dict[str, Any]:
    if denominators is None:
        values = [float(value) for value in numerators if value is not None]
        return {"numerator": sum(values), "denominator": len(values), "value": (sum(values) / len(values)) if values else None}
    numerator_list = [int(value or 0) for value in numerators]
    denominator_list = [int(value or 0) for value in denominators]
    numerator = sum(numerator_list)
    denominator = sum(denominator_list)
    return {"numerator": numerator, "denominator": denominator, "value": _rate(numerator, denominator)}


def _hard_gate_failures(
    *,
    execution_status: str | None = None,
    unsupported_material_claim_count: int = 0,
    unknown_citation_count: int = 0,
    forbidden_scope_answer_count: int = 0,
    false_premise_acceptance_count: int = 0,
    hidden_conflict_count: int = 0,
    judge_failure_count: int = 0,
) -> list[str]:
    failures: list[str] = []
    if execution_status == "infrastructure_failure":
        failures.append("infrastructure_failure")
    if unsupported_material_claim_count:
        failures.append("unsupported_material_claim_count")
    if unknown_citation_count:
        failures.append("unknown_citation_count")
    if forbidden_scope_answer_count:
        failures.append("forbidden_scope_answer_count")
    if false_premise_acceptance_count:
        failures.append("false_premise_acceptance_count")
    if hidden_conflict_count:
        failures.append("hidden_conflict_count")
    if judge_failure_count:
        failures.append("judge_failure_count")
    return failures


def _count_hard_gate_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for failure in row.get("hard_gate_failures") or []:
            counts[str(failure)] = counts.get(str(failure), 0) + 1
    return dict(sorted(counts.items()))


def _primary_failure(
    answerability: dict[str, Any],
    claims: dict[str, Any],
    citations: dict[str, Any],
    grounding: dict[str, Any],
    scope: dict[str, Any],
    judge_failure: bool,
) -> str:
    if not answerability.get("correct"):
        return "answerability_failure"
    if claims.get("forbidden_claim_count") or claims.get("missing_required_claim_ids"):
        return "generation_failure"
    if not scope.get("passed"):
        return "scope_failure"
    if not citations.get("passed"):
        return "citation_failure"
    if not grounding.get("passed"):
        return "grounding_failure"
    if judge_failure:
        return "semantic_judge_failure"
    return "finalization_failure"
