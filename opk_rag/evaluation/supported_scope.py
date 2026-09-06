from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

SUPPORTED_SCOPE_DIAGNOSIS_SCHEMA_VERSION = "supported-scope-diagnosis.v1"

PRIMARY_SCOPE_FAILURES = {
    "supported_scope_ignored",
    "unsupported_scope_answered",
    "supported_scope_underanswered",
    "whole-question-abstention",
    "partial_instruction_ignored",
    "subquestion_boundary_lost",
    "evidence-to-scope-mapping_unclear",
    "premise_correction_ignored",
    "unsupported_inference_added",
    "provider_variance",
    "grounding_true_rejection",
    "other",
}


@dataclass(frozen=True)
class SupportedScopeTargetSet:
    target_sample_ids: tuple[str, ...]
    control_sample_ids: tuple[str, ...]


def build_target_set(*, baseline_results: list[dict[str, Any]], task0041_results: list[dict[str, Any]] | None = None) -> SupportedScopeTargetSet:
    targets = {
        str(row["sample_id"])
        for row in baseline_results
        if row.get("error_type") in {"generation_abstention", "partial_answer_missed"}
    }
    controls = {
        str(row["sample_id"])
        for row in baseline_results
        if _primary_outcome(row) in {"correct_full_answer", "correct_partial_answer", "correct_abstention"}
        or row.get("error_type") in {"answerability_unsafe_allow", "retrieval_wrong_source"}
    }
    for row in task0041_results or []:
        if row.get("error_type") in {"generation_abstention", "partial_answer_missed"}:
            targets.add(str(row["sample_id"]))
    return SupportedScopeTargetSet(tuple(sorted(targets)), tuple(sorted(controls)))


def diagnose_supported_scope_sample(row: dict[str, Any]) -> dict[str, Any]:
    model_decision = row.get("model_decision") if isinstance(row.get("model_decision"), dict) else {}
    answerability = row.get("answerability") if isinstance(row.get("answerability"), dict) else {}
    diagnostics = answerability.get("diagnostics") if isinstance(answerability.get("diagnostics"), dict) else {}
    supported_scope = _list(row.get("supported_scope") or model_decision.get("supported_scope") or diagnostics.get("covered_query_terms"))
    unsupported_scope = _list(row.get("unsupported_scope") or model_decision.get("unsupported_scope") or diagnostics.get("missing_exact_requirements"))
    final_action = str(row.get("final_action") or "")
    error_type = str(row.get("error_type") or "")
    grounding_status = str(row.get("grounding_status") or "")
    citation_status = str(row.get("citation_status") or "")
    unsupported_claims = bool(row.get("unsupported_claims") or model_decision.get("unsupported_claims"))
    decision = str(model_decision.get("decision") or "")
    explicit_abstention = final_action == "abstain" or decision == "abstain"
    primary = _primary_scope_failure(
        expected_final_action=str(row.get("expected_final_action") or ""),
        error_type=error_type,
        reason=str(row.get("refusal_reason_code") or row.get("generation_reason") or row.get("decision_reason") or ""),
        final_action=final_action,
        supported_scope=supported_scope,
        unsupported_scope=unsupported_scope,
        unsupported_claims=unsupported_claims,
        grounding_status=grounding_status,
    )
    return {
        "schema_version": SUPPORTED_SCOPE_DIAGNOSIS_SCHEMA_VERSION,
        "sample_id": row.get("sample_id"),
        "expected_answerability": row.get("expected_answerability"),
        "expected_final_action": row.get("expected_final_action"),
        "baseline_final_action": row.get("baseline_final_action") or final_action,
        "scope_input": {
            "supported_scope_present": bool(supported_scope),
            "supported_scope_item_count": len(supported_scope),
            "unsupported_scope_present": bool(unsupported_scope),
            "unsupported_scope_item_count": len(unsupported_scope),
            "premise_correction_present": bool(row.get("premise_correction") or model_decision.get("premise_correction")),
            "subquestion_count": max(1, len(supported_scope) + len(unsupported_scope)),
        },
        "provider_output": {
            "answer_present": bool(row.get("answer") or model_decision.get("answer")),
            "partial_answer_present": row.get("final_action") == "partial_answer" or answerability.get("status") == "partially_answerable",
            "explicit_abstention": explicit_abstention,
            "citation_count": len(_list(row.get("citations") or model_decision.get("citations"))),
            "unsupported_claim_detected": unsupported_claims,
            "supported_scope_covered": bool(supported_scope) and not explicit_abstention,
            "unsupported_scope_answered": primary == "unsupported_scope_answered",
            "contract_following_status": "failed" if primary not in {"other", "grounding_true_rejection"} else "unknown",
        },
        "validation": {
            "citation_status": citation_status or "unknown",
            "grounding_status": grounding_status or "unknown",
            "failure_reason": row.get("refusal_reason_code") or row.get("generation_reason") or row.get("decision_reason"),
        },
        "diagnosis": {
            "primary_scope_failure": primary,
            "contributing_factors": _contributing_factors(primary, unsupported_claims=unsupported_claims, explicit_abstention=explicit_abstention),
            "confidence": "medium" if primary != "other" else "low",
        },
    }


def summarize_supported_scope_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [diagnose_supported_scope_sample(row) for row in rows]
    by_failure = Counter(item["diagnosis"]["primary_scope_failure"] for item in diagnostics)
    return {
        "schema_version": "supported-scope-diagnosis-summary.v1",
        "sample_count": len(diagnostics),
        "primary_scope_failure_counts": dict(sorted(by_failure.items())),
        "sample_ids_by_primary_scope_failure": {
            key: [str(item["sample_id"]) for item in diagnostics if item["diagnosis"]["primary_scope_failure"] == key]
            for key in sorted(by_failure)
        },
        "diagnostics": diagnostics,
        "privacy": {
            "contains_questions": False,
            "contains_paths": False,
            "contains_evidence_text": False,
            "contains_raw_provider_output": False,
            "contains_secrets": False,
        },
    }


def _primary_scope_failure(
    *,
    expected_final_action: str,
    error_type: str,
    reason: str,
    final_action: str,
    supported_scope: list[str],
    unsupported_scope: list[str],
    unsupported_claims: bool,
    grounding_status: str,
) -> str:
    if "variance" in reason:
        return "provider_variance"
    if "false_premise" in reason:
        return "premise_correction_ignored"
    if "unsupported_inference" in reason:
        return "unsupported_inference_added"
    if error_type == "generation_abstention" or (expected_final_action in {"answer", "partial_answer"} and final_action == "abstain"):
        return "whole-question-abstention" if supported_scope else "supported_scope_ignored"
    if error_type == "partial_answer_missed":
        return "partial_instruction_ignored"
    if "partial_supported_scope_missing" in reason:
        return "supported_scope_underanswered"
    if "partial_unsupported_scope_fabricated" in reason:
        return "unsupported_scope_answered"
    if unsupported_claims:
        return "unsupported_scope_answered" if unsupported_scope else "unsupported_inference_added"
    if grounding_status not in {"", "grounded", "disabled", "not_applicable", "not_evaluated"}:
        return "grounding_true_rejection"
    return "other"


def _contributing_factors(primary: str, *, unsupported_claims: bool, explicit_abstention: bool) -> list[str]:
    factors = [primary]
    if unsupported_claims:
        factors.append("unsupported_claim_detected")
    if explicit_abstention:
        factors.append("explicit_abstention")
    return list(dict.fromkeys(factors))


def _primary_outcome(row: dict[str, Any]) -> str:
    if row.get("primary_outcome"):
        return str(row["primary_outcome"])
    expected = row.get("expected_final_action")
    error = row.get("error_type")
    final = row.get("final_action")
    if expected == "answer" and final == "answer" and error in {"no_failure", "no_error"}:
        return "correct_full_answer"
    if expected == "partial_answer" and final == "partial_answer" and error in {"no_failure", "no_error"}:
        return "correct_partial_answer"
    if expected == "abstain" and final == "abstain":
        return "correct_abstention"
    return str(error or "other")


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
