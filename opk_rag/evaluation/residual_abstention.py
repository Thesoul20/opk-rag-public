from __future__ import annotations

from collections import Counter
from typing import Any

from opk_rag.answerability import AnswerabilityDecision
from opk_rag.evaluation.generation_abstention import classify_abstention_row

RESIDUAL_ABSTENTION_SCHEMA_VERSION = "residual-abstention-attribution.v1"

PRIMARY_ATTRIBUTIONS = (
    "correct_no_evidence_abstention",
    "correct_insufficient_evidence_abstention",
    "correct_conflicting_evidence_abstention",
    "correct_unsupported_scope_abstention",
    "correct_ambiguous_question_abstention",
    "retrieval_gold_miss",
    "retrieval_gold_below_cutoff",
    "evidence_bundle_dropped_gold",
    "evidence_bundle_partial_coverage",
    "evidence_bundle_duplicate_dominance",
    "evidence_bundle_context_truncation",
    "cross_document_evidence_missing",
    "answerability_false_abstention",
    "answerability_supported_scope_too_narrow",
    "answerability_unsupported_scope_too_broad",
    "answerability_partial_answer_not_exposed",
    "answerability_decision_evidence_mismatch",
    "generation_ignored_supported_evidence",
    "generation_overweighted_abstention_instruction",
    "generation_failed_partial_answer",
    "generation_citation_avoidance",
    "generation_contract_overconstraint",
    "generation_unknown_abstention",
    "gold_evidence_ambiguous",
    "expected_answerability_questionable",
    "dataset_annotation_insufficient",
    "unclassified_residual_abstention",
)

CORRECT_ATTRIBUTIONS = {
    "correct_no_evidence_abstention",
    "correct_insufficient_evidence_abstention",
    "correct_conflicting_evidence_abstention",
    "correct_unsupported_scope_abstention",
    "correct_ambiguous_question_abstention",
}

FALSE_ABSTENTION_ATTRIBUTIONS = set(PRIMARY_ATTRIBUTIONS) - CORRECT_ATTRIBUTIONS - {
    "gold_evidence_ambiguous",
    "expected_answerability_questionable",
    "dataset_annotation_insufficient",
    "unclassified_residual_abstention",
}


def residual_abstention_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_residual_abstention(row)]


def audit_residual_abstention(
    *,
    baseline_row: dict[str, Any],
    dataset_record: dict[str, Any],
    fixture_row: dict[str, Any],
    answerability: AnswerabilityDecision | None,
    prompt_hash: str | None = None,
    request_hash: str | None = None,
) -> dict[str, Any]:
    gold_match = fixture_row.get("gold_match_result") if isinstance(fixture_row.get("gold_match_result"), dict) else {}
    gold_evidence = fixture_row.get("gold_evidence") if isinstance(fixture_row.get("gold_evidence"), list) else []
    candidate_matches = gold_match.get("candidate_gold_matches") if isinstance(gold_match.get("candidate_gold_matches"), list) else []
    bundle_matches = gold_match.get("evidence_bundle_gold_matches") if isinstance(gold_match.get("evidence_bundle_gold_matches"), list) else []
    candidates = fixture_row.get("candidates") if isinstance(fixture_row.get("candidates"), list) else []
    evidence_bundle = fixture_row.get("evidence_bundle") if isinstance(fixture_row.get("evidence_bundle"), list) else []
    attribution, flags = _primary_attribution(
        dataset_record=dataset_record,
        fixture_row=fixture_row,
        candidate_matches=candidate_matches,
        bundle_matches=bundle_matches,
        answerability=answerability,
    )
    model_decision = baseline_row.get("model_decision") if isinstance(baseline_row.get("model_decision"), dict) else {}
    classified = classify_abstention_row(baseline_row)
    return {
        "schema_version": RESIDUAL_ABSTENTION_SCHEMA_VERSION,
        "sample_id": baseline_row.get("sample_id") or dataset_record.get("id"),
        "question": dataset_record.get("question") or fixture_row.get("question"),
        "split": dataset_record.get("split") or baseline_row.get("split") or fixture_row.get("split"),
        "dataset_label": dataset_record.get("answerability_type"),
        "expected_answerability": dataset_record.get("expected_answerability"),
        "expected_action": dataset_record.get("expected_action"),
        "expected_evidence": gold_evidence,
        "candidate_count": len(candidates),
        "candidate_gold_hit": bool(gold_match.get("candidate_hit")),
        "gold_rank": gold_match.get("first_relevant_rank"),
        "candidate_gold_matches": candidate_matches,
        "evidence_bundle_count": len(evidence_bundle),
        "evidence_bundle_gold_hit": bool(gold_match.get("evidence_bundle_hit")),
        "evidence_bundle_gold_matches": bundle_matches,
        "evidence_bundle_duplicate_count": _duplicate_chunk_count(evidence_bundle),
        "evidence_bundle_context_token_count": _retrieval_config(fixture_row).get("context_token_count"),
        "evidence_bundle_context_token_budget": _retrieval_config(fixture_row).get("context_token_budget"),
        "answerability_decision": _answerability_payload(answerability),
        "supported_scope": _scope_from_model_or_answerability(model_decision, answerability, "supported_scope"),
        "unsupported_scope": _scope_from_model_or_answerability(model_decision, answerability, "unsupported_scope"),
        "recommended_action": _recommended_action(attribution),
        "rendered_prompt_hash": prompt_hash or baseline_row.get("prompt_hash") or baseline_row.get("contract_fingerprint"),
        "request_hash": request_hash or baseline_row.get("request_hash"),
        "raw_model_output": baseline_row.get("raw_output"),
        "parsed_action": model_decision.get("decision") or baseline_row.get("action"),
        "generation_reason": model_decision.get("reason") or baseline_row.get("generation_reason"),
        "citation_status": _citation_status(baseline_row),
        "grounding_status": baseline_row.get("grounding_status"),
        "final_action": baseline_row.get("final_action"),
        "abstention_category": classified["abstention_category"],
        "primary_attribution": attribution,
        "secondary_diagnostic_flags": flags,
        "is_correct_abstention": attribution in CORRECT_ATTRIBUTIONS,
        "is_false_abstention_candidate": attribution in FALSE_ABSTENTION_ATTRIBUTIONS,
        "partial_answer_candidate": _partial_answer_candidate(dataset_record, answerability, attribution),
    }


def summarize_residual_abstention(audits: list[dict[str, Any]], *, baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter(flag for row in audits for flag in row.get("secondary_diagnostic_flags", ()))
    no_evidence_ids = {row["sample_id"] for row in audits if row.get("dataset_label") == "no_evidence"}
    no_evidence_rows = [row for row in baseline_rows if row.get("sample_id") in no_evidence_ids]
    return {
        "schema_version": "residual-abstention-summary.v1",
        "sample_count": len({row.get("sample_id") for row in baseline_rows}),
        "baseline_metrics": {
            "grounded_answer": sum(row.get("final_action") == "answer" and row.get("grounding_status") == "grounded" for row in baseline_rows),
            "explicit_abstain": len(audits),
            "correct_abstain": sum(row.get("is_correct_abstention") is True for row in audits),
            "suspected_false_abstain": sum(row.get("is_false_abstention_candidate") is True for row in audits),
            "partial_answer": 0,
            "unsupported_answer": sum(row.get("unsupported_answer") is True for row in baseline_rows),
            "citation_failure": sum(row.get("citation_status") == "invalid" for row in baseline_rows),
            "grounding_failure": sum(row.get("unsupported_answer") is True for row in baseline_rows),
            "provider_failure": sum(row.get("provider_success") is False for row in baseline_rows),
            "infrastructure_failure": sum(row.get("provider_success") is False for row in baseline_rows),
            "no_evidence_answer_count": sum(row.get("final_action") == "answer" for row in no_evidence_rows),
        },
        "attribution_metrics": {
            "primary_attribution_distribution": dict(Counter(row["primary_attribution"] for row in audits)),
            "secondary_diagnostic_distribution": dict(flags),
            "correct_abstain_count": sum(row.get("is_correct_abstention") is True for row in audits),
            "false_abstain_candidate_count": sum(row.get("is_false_abstention_candidate") is True for row in audits),
            "evidence_bundle_failure_count": sum(str(row["primary_attribution"]).startswith("evidence_bundle_") for row in audits),
            "answerability_failure_count": sum(str(row["primary_attribution"]).startswith("answerability_") for row in audits),
            "generation_failure_count": sum(str(row["primary_attribution"]).startswith("generation_") for row in audits),
            "dataset_issue_count": sum(row["primary_attribution"] in {"gold_evidence_ambiguous", "expected_answerability_questionable", "dataset_annotation_insufficient"} for row in audits),
            "partial_answer_candidate_count": sum(row.get("partial_answer_candidate") is True for row in audits),
        },
        "audited_sample_ids": [row["sample_id"] for row in audits],
    }


def _primary_attribution(
    *,
    dataset_record: dict[str, Any],
    fixture_row: dict[str, Any],
    candidate_matches: list[dict[str, Any]],
    bundle_matches: list[dict[str, Any]],
    answerability: AnswerabilityDecision | None,
) -> tuple[str, list[str]]:
    expected_action = str(dataset_record.get("expected_action") or "")
    label = str(dataset_record.get("answerability_type") or "")
    expected_answerability = str(dataset_record.get("expected_answerability") or "")
    gold_match = fixture_row.get("gold_match_result") if isinstance(fixture_row.get("gold_match_result"), dict) else {}
    candidate_hit = bool(gold_match.get("candidate_hit"))
    bundle_hit = bool(gold_match.get("evidence_bundle_hit"))
    flags = _secondary_flags(dataset_record=dataset_record, fixture_row=fixture_row, answerability=answerability)

    if label == "no_evidence" or (expected_action == "abstain" and expected_answerability == "unanswerable" and not candidate_hit):
        return "correct_no_evidence_abstention", flags
    if label in {"related_topic_missing_fact", "unsupported_inference"} and expected_action == "abstain":
        return "correct_unsupported_scope_abstention", flags
    if label == "ambiguous_question" and expected_action == "abstain":
        return "correct_ambiguous_question_abstention", flags
    if not candidate_hit:
        return "retrieval_gold_miss", flags
    if candidate_hit and not bundle_hit:
        return "evidence_bundle_dropped_gold", flags
    if _bundle_partial_coverage(dataset_record, candidate_matches, bundle_matches):
        return "evidence_bundle_partial_coverage", flags
    if _duplicate_chunk_count(fixture_row.get("evidence_bundle") or []) > 0:
        return "evidence_bundle_duplicate_dominance", flags
    if answerability is not None and answerability.reason_code in {"false_premise", "partial_evidence"}:
        return "answerability_decision_evidence_mismatch", flags
    if expected_action in {"answer", "correct_premise"} and bundle_hit:
        return "generation_ignored_supported_evidence", flags
    if label == "conflicting_evidence":
        return "correct_conflicting_evidence_abstention", flags
    if expected_action == "abstain":
        return "correct_insufficient_evidence_abstention", flags
    return "generation_unknown_abstention", flags


def _secondary_flags(*, dataset_record: dict[str, Any], fixture_row: dict[str, Any], answerability: AnswerabilityDecision | None) -> list[str]:
    gold_match = fixture_row.get("gold_match_result") if isinstance(fixture_row.get("gold_match_result"), dict) else {}
    flags = [
        f"dataset_label:{dataset_record.get('answerability_type')}",
        f"expected_action:{dataset_record.get('expected_action')}",
        f"candidate_hit:{bool(gold_match.get('candidate_hit'))}",
        f"evidence_bundle_hit:{bool(gold_match.get('evidence_bundle_hit'))}",
    ]
    if answerability is not None:
        flags.extend(
            [
                f"answerability_status:{answerability.status}",
                f"answerability_reason:{answerability.reason_code}",
            ]
        )
        if answerability.status == "partially_answerable" or answerability.reason_code == "false_premise":
            flags.append("answerability_should_be_exposed_to_generation")
    if _duplicate_chunk_count(fixture_row.get("evidence_bundle") or []) > 0:
        flags.append("duplicate_chunk_in_bundle")
    if _retrieval_config(fixture_row).get("context_token_count") == _retrieval_config(fixture_row).get("context_token_budget"):
        flags.append("context_budget_exactly_filled")
    return flags


def _bundle_partial_coverage(dataset_record: dict[str, Any], candidate_matches: list[dict[str, Any]], bundle_matches: list[dict[str, Any]]) -> bool:
    if dataset_record.get("expected_action") not in {"answer", "correct_premise"}:
        return False
    candidate_gold = {row.get("gold_key") for row in candidate_matches}
    bundle_gold = {row.get("gold_key") for row in bundle_matches}
    return bool(bundle_gold) and bool(candidate_gold - bundle_gold)


def _partial_answer_candidate(dataset_record: dict[str, Any], answerability: AnswerabilityDecision | None, attribution: str) -> bool:
    if dataset_record.get("expected_action") == "abstain":
        return False
    if attribution in {"retrieval_gold_miss", "correct_no_evidence_abstention"}:
        return False
    if answerability is not None and (answerability.status == "partially_answerable" or answerability.reason_code == "false_premise"):
        return True
    return bool(dataset_record.get("supported_subquestions") or dataset_record.get("unsupported_subquestions"))


def _answerability_payload(decision: AnswerabilityDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "status": decision.status,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "evidence_chunk_ids": list(decision.evidence_chunk_ids),
        "evidence_score": decision.evidence_score,
        "evidence_count": decision.evidence_count,
        "considered_evidence_count": decision.considered_evidence_count,
        "diagnostics": decision.diagnostics,
    }


def _scope_from_model_or_answerability(model_decision: dict[str, Any], answerability: AnswerabilityDecision | None, key: str) -> list[str]:
    value = model_decision.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    if answerability is None:
        return []
    diagnostics_key = "covered_query_terms" if key == "supported_scope" else "missing_exact_requirements"
    raw = answerability.diagnostics.get(diagnostics_key)
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _recommended_action(attribution: str) -> str:
    if attribution.startswith("correct_"):
        return "retain_abstention"
    if attribution.startswith("retrieval_"):
        return "record_retrieval_followup"
    if attribution.startswith("evidence_bundle_"):
        return "inspect_evidence_bundle_selection"
    if attribution.startswith("answerability_"):
        return "expose_answerability_contract_or_calibrate_policy"
    if attribution.startswith("generation_"):
        return "tighten_generation_contract_without_relaxing_safety"
    return "manual_review"


def _citation_status(row: dict[str, Any]) -> str:
    if row.get("citation_presence") is True:
        return "present"
    if row.get("final_action") == "abstain":
        return "not_applicable"
    return "missing"


def _duplicate_chunk_count(items: list[dict[str, Any]]) -> int:
    chunk_ids = [str(item.get("chunk_id")) for item in items if item.get("chunk_id")]
    return len(chunk_ids) - len(set(chunk_ids))


def _retrieval_config(fixture_row: dict[str, Any]) -> dict[str, Any]:
    return fixture_row.get("retrieval_configuration") if isinstance(fixture_row.get("retrieval_configuration"), dict) else {}


def _is_residual_abstention(row: dict[str, Any]) -> bool:
    return row.get("final_action") == "abstain" or row.get("action") == "abstain"
