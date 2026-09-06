from __future__ import annotations

from collections import Counter
from typing import Any

PRIMARY_FAILURE_ATTRIBUTIONS = (
    "infrastructure_failure",
    "retrieval_no_gold_evidence",
    "retrieval_gold_ranked_below_cutoff",
    "evidence_bundle_dropped_gold",
    "answerability_correct_refusal",
    "answerability_false_refusal_candidate",
    "generation_contract_failure",
    "partial_answer_failure",
    "generation_abstention",
    "citation_failure",
    "grounding_failure",
    "successful_grounded_answer",
    "unclassified_failure",
)


def attribute_failure(*, sample: dict[str, Any], result: dict[str, Any] | None = None, retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
    result = result or {}
    retrieval = retrieval or {}
    expected_action = sample.get("expected_action")
    expected_behavior = "answer" if expected_action in {"answer", "partial_answer", "correct_premise"} else "abstain"
    final_action = result.get("final_action") or result.get("actual_action")
    expected_positive = expected_behavior == "answer"
    retrieval_evaluable = retrieval.get("retrieval_evaluable")
    candidate_hit = retrieval.get("candidate_hit")
    bundle_hit = retrieval.get("evidence_bundle_hit")
    first_rank = retrieval.get("first_relevant_rank")
    cutoff = _effective_cutoff(retrieval)
    flags: list[str] = []

    if not result or final_action is None:
        if expected_positive and retrieval_evaluable is True and not candidate_hit:
            primary = "retrieval_no_gold_evidence"
        elif expected_positive and retrieval_evaluable is True and first_rank is not None and cutoff is not None and first_rank > cutoff:
            primary = "retrieval_gold_ranked_below_cutoff"
        elif expected_positive and retrieval_evaluable is True and candidate_hit and not bundle_hit:
            primary = "evidence_bundle_dropped_gold"
        else:
            primary = "unclassified_failure"
    elif result.get("error_type") == "infrastructure_failure" or result.get("failure_stage") == "infrastructure" or final_action == "system_error":
        primary = "infrastructure_failure"
    elif expected_positive and retrieval_evaluable is True and not candidate_hit:
        primary = "retrieval_no_gold_evidence"
    elif expected_positive and retrieval_evaluable is True and first_rank is not None and cutoff is not None and first_rank > cutoff:
        primary = "retrieval_gold_ranked_below_cutoff"
    elif expected_positive and retrieval_evaluable is True and candidate_hit and not bundle_hit:
        primary = "evidence_bundle_dropped_gold"
    elif not expected_positive and final_action == "abstain":
        primary = "answerability_correct_refusal"
    elif expected_positive and _decision_refused(result):
        primary = "answerability_false_refusal_candidate" if candidate_hit or retrieval_evaluable is not True else "answerability_false_refusal_candidate"
    elif _contract_failure(result):
        primary = "generation_contract_failure"
    elif expected_action == "partial_answer" and final_action != "answer":
        primary = "partial_answer_failure"
    elif result.get("generation_status") == "abstained" or result.get("error_type") == "generation_abstention":
        primary = "generation_abstention"
    elif result.get("citation_status") in {"invalid", "source_mismatch"} or result.get("error_type") == "citation_failure":
        primary = "citation_failure"
    elif result.get("grounding_status") not in {None, "", "grounded", "not_applicable"} or result.get("error_type") == "grounding_failure":
        primary = "grounding_failure"
    elif expected_positive and final_action == "answer" and result.get("grounding_status") == "grounded":
        primary = "successful_grounded_answer"
    else:
        primary = "unclassified_failure"

    if retrieval_evaluable is False:
        flags.append("retrieval_not_evaluable")
    if candidate_hit and not bundle_hit:
        flags.append("candidate_hit_bundle_miss")
    if candidate_hit and final_action == "abstain":
        flags.append("retrieval_success_final_abstain")
    if result.get("generation_status") == "generated" and final_action == "abstain":
        flags.append("generated_then_abstained")
    if result.get("citation_status") == "missing":
        flags.append("missing_citations")
    return {
        "primary_failure_attribution": primary,
        "secondary_diagnostic_flags": sorted(set(flags)),
    }


def attribution_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("primary_failure_attribution", "unclassified_failure") for row in rows).items()))


def _effective_cutoff(retrieval: dict[str, Any]) -> int | None:
    config = retrieval.get("retrieval_config") if isinstance(retrieval.get("retrieval_config"), dict) else {}
    value = config.get("final_top_k") or config.get("requested_top_k")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _decision_refused(result: dict[str, Any]) -> bool:
    return result.get("decision_status") == "refused" or result.get("failure_stage") == "decision"


def _contract_failure(result: dict[str, Any]) -> bool:
    return result.get("error_type") in {"partial_answer_generation_failure", "false_premise_failure", "unsupported_inference_failure"} or result.get("schema_parse_status") == "failed"
