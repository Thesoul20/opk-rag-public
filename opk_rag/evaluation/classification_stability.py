from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


INFRASTRUCTURE_CLASS = "infrastructure_failure"
GENERATION_RETRY_CLASS = "generation_retry_candidate"
TERMINAL_CLASSES = {"answer_draft_grounded", "refusal_safety_terminal", "refusal_correct_unanswerable"}
SAFETY_CLASSES = {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"}


def sample_runtime_class(row: dict[str, Any], records_by_sample: dict[str, dict[str, Any]] | None = None) -> str:
    if row.get("error_type") == "infrastructure_failure" or row.get("infrastructure_failure"):
        return INFRASTRUCTURE_CLASS
    sample_id = row.get("sample_id")
    if records_by_sample and sample_id in records_by_sample:
        return str(records_by_sample[sample_id].get("runtime_failure_class") or "missing_observability_field")
    generation = row.get("generation_result") or {}
    if generation.get("status") == "answered" or row.get("final_action") == "answer":
        return "answer_draft_grounded"
    if generation.get("status") == "refused":
        return "refusal_model_conservative"
    return str(row.get("runtime_failure_class") or row.get("no_generation_reason") or "insufficient_runtime_signal")


def recoverability_class(runtime_class: str, evidence_class: str | None = None) -> str:
    evidence_class = evidence_class or "runtime_evidence_appears_sufficient"
    if runtime_class in {"provider_transport_failure", "provider_timeout", "provider_server_failure", INFRASTRUCTURE_CLASS}:
        return "infrastructure_retry_candidate"
    if runtime_class in SAFETY_CLASSES:
        return "terminal_safety_failure"
    if runtime_class in {"answer_draft_grounded", "refusal_correct_unanswerable"}:
        return "not_recoverable_by_agent"
    if runtime_class in {"refusal_model_conservative", "generation_contract_abstention", "generation_empty_output", "generation_schema_repair_exhausted"} and evidence_class == "runtime_evidence_appears_sufficient":
        return GENERATION_RETRY_CLASS
    if runtime_class in {"refusal_claimed_insufficient_evidence", "refusal_claimed_missing_scope"}:
        return "retrieval_recovery_candidate"
    if runtime_class in {"answer_draft_grounding_failure", "answer_draft_unsupported_claim"}:
        return "retrieval_then_generation_candidate"
    return "indeterminate"


def transition_pattern(classes: list[str]) -> str:
    observed = [value for value in classes if value]
    if not observed:
        return "mixed"
    unique = set(observed)
    if INFRASTRUCTURE_CLASS in unique:
        return "infrastructure_affected" if len(unique) == 1 else ("terminal_to_nonterminal" if len(unique - {INFRASTRUCTURE_CLASS}) else "infrastructure_affected")
    if len(unique) == 1:
        only = next(iter(unique))
        if only == "answer_draft_grounded":
            return "stable_answer"
        if recoverability_class(only) == GENERATION_RETRY_CLASS:
            return "stable_generation_retry_candidate"
        if recoverability_class(only) in {"terminal_safety_failure", "correct_terminal_abstention", "not_recoverable_by_agent"}:
            return "stable_terminal"
        if recoverability_class(only) == "indeterminate":
            return "stable_indeterminate"
        return "stable_terminal"
    has_answer = "answer_draft_grounded" in unique
    has_refusal = any(value.startswith("refusal_") or value == "generation_contract_abstention" for value in unique)
    if has_answer and has_refusal:
        first = observed[0]
        last = observed[-1]
        if first == "answer_draft_grounded" and last != "answer_draft_grounded":
            return "answer_to_refusal"
        if first != "answer_draft_grounded" and last == "answer_draft_grounded":
            return "refusal_to_answer"
        return "refusal_to_answer"
    terminal = {value for value in unique if value in TERMINAL_CLASSES}
    if terminal and len(unique - terminal) > 0:
        return "terminal_to_nonterminal"
    return "mixed"


def instability_score(classes: Iterable[str]) -> float:
    values = [value for value in classes if value]
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return 0.0 if len(counts) == 1 else entropy / max_entropy


def compare_three_replicates(sample_ids: list[str], replicate_classes: list[dict[str, str]], replicate_rows: list[dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    rows = []
    exact = 0
    modal_agreement = 0
    infrastructure_affected = 0
    for sample_id in sample_ids:
        classes = [replicate.get(sample_id) for replicate in replicate_classes]
        observed = [value for value in classes if value]
        counts = Counter(observed)
        modal_class, modal_count = counts.most_common(1)[0] if counts else (None, 0)
        stable = len(set(observed)) == 1 and len(observed) == 3
        if stable:
            exact += 1
        if modal_count >= 2:
            modal_agreement += 1
        if INFRASTRUCTURE_CLASS in observed:
            infrastructure_affected += 1
        rows.append(
            {
                "sample_id": sample_id,
                "replicate_1_class": classes[0],
                "replicate_2_class": classes[1],
                "replicate_3_class": classes[2],
                "modal_class": modal_class,
                "modal_class_count": modal_count,
                "classification_stable": stable,
                "classification_transition_pattern": transition_pattern([value for value in classes if value]),
                "infrastructure_affected": INFRASTRUCTURE_CLASS in observed,
                "generation_invocation_count": _sum_row_value(sample_id, replicate_rows, "generation_invocation_count"),
                "refusal_count": sum(1 for value in observed if value.startswith("refusal_") or value == "generation_contract_abstention"),
                "answer_count": sum(1 for value in observed if value == "answer_draft_grounded"),
                "terminal_count": sum(1 for value in observed if value in TERMINAL_CLASSES),
                "indeterminate_count": sum(1 for value in observed if recoverability_class(value) == "indeterminate"),
            }
        )
    total = len(sample_ids)
    return {
        "schema_version": "opk-rag.task0074-three-replicate-comparison.v1",
        "sample_count": total,
        "three_replicate_exact_agreement_count": exact,
        "three_replicate_exact_agreement_rate": exact / total if total else None,
        "three_replicate_modal_agreement_count": modal_agreement,
        "three_replicate_modal_agreement_rate": modal_agreement / total if total else None,
        "unstable_sample_count": total - exact,
        "infrastructure_affected_sample_count": infrastructure_affected,
        "rows": rows,
    }


def compare_five_replicates(sample_ids: list[str], replicate_classes: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    counts_by_status = Counter()
    for sample_id in sample_ids:
        observed = [replicate.get(sample_id) for replicate in replicate_classes if replicate.get(sample_id)]
        counts = Counter(observed)
        modal_class, modal_count = counts.most_common(1)[0] if counts else (None, 0)
        infra = counts.get(INFRASTRUCTURE_CLASS, 0)
        valid_count = len(observed) - infra
        ratio = modal_count / len(observed) if observed else None
        if not observed or valid_count < 3:
            status = "insufficient_observations"
        elif infra and valid_count < len(observed):
            status = "infrastructure_confounded"
        elif ratio == 1:
            status = "stable"
        elif ratio is not None and ratio >= 0.8:
            status = "mostly_stable"
        else:
            status = "unstable"
        counts_by_status[status] += 1
        rows.append(
            {
                "sample_id": sample_id,
                "observed_replicate_count": len(observed),
                "infrastructure_valid_replicate_count": valid_count,
                "class_counts": dict(sorted(counts.items())),
                "modal_class": modal_class,
                "modal_class_ratio": ratio,
                "classification_entropy_score": instability_score(observed),
                "historical_stability": status,
            }
        )
    return {
        "schema_version": "opk-rag.task0074-five-replicate-historical-comparison.v1",
        "sample_count": len(sample_ids),
        "five_replicate_stable_count": counts_by_status["stable"],
        "five_replicate_mostly_stable_count": counts_by_status["mostly_stable"],
        "five_replicate_unstable_count": counts_by_status["unstable"],
        "five_replicate_infrastructure_confounded_count": counts_by_status["infrastructure_confounded"],
        "five_replicate_insufficient_observations_count": counts_by_status["insufficient_observations"],
        "instability_score_definition": "normalized Shannon entropy over observed runtime classes; 0.0 means exact agreement",
        "rows": rows,
    }


def revalidate_generation_retry_candidates(candidate_ids: list[str], three: dict[str, Any], five: dict[str, Any], record_maps: list[dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    three_rows = {row["sample_id"]: row for row in three.get("rows", [])}
    five_rows = {row["sample_id"]: row for row in five.get("rows", [])}
    rows = []
    stable_count = 0
    safety_count = 0
    correct_unanswerable_count = 0
    for sample_id in candidate_ids:
        row = three_rows.get(sample_id, {})
        new_classes = [row.get("replicate_1_class"), row.get("replicate_2_class"), row.get("replicate_3_class")]
        safety = any(value in SAFETY_CLASSES for value in new_classes if value)
        correct_unanswerable = any(value == "refusal_correct_unanswerable" for value in new_classes if value)
        infra = row.get("infrastructure_affected") is True
        stable = (
            not safety
            and not correct_unanswerable
            and not infra
            and all(recoverability_class(value or "") == GENERATION_RETRY_CLASS for value in new_classes)
        )
        stable_count += int(stable)
        safety_count += int(safety)
        correct_unanswerable_count += int(correct_unanswerable)
        rows.append(
            {
                "sample_id": sample_id,
                "task0073_classification": GENERATION_RETRY_CLASS,
                "new_replicate_1_classification": new_classes[0],
                "new_replicate_2_classification": new_classes[1],
                "new_replicate_3_classification": new_classes[2],
                "five_replicate_modal_class": (five_rows.get(sample_id) or {}).get("modal_class"),
                "stable_generation_retry_candidate": stable,
                "runtime_signal_consistency": "consistent" if len(set(new_classes)) == 1 else "variable",
                "provider_refusal_consistency": "consistent" if all(str(value).startswith("refusal_") or value == "generation_contract_abstention" for value in new_classes if value) else "variable",
                "evidence_class_consistency": "not_recomputed_by_candidate_analysis",
                "safety_terminal_signal_present": safety,
                "classification_confidence": 0.9 if stable else 0.4,
            }
        )
    total = len(candidate_ids)
    return {
        "schema_version": "opk-rag.task0074-generation-retry-candidate-revalidation.v1",
        "task0073_generation_retry_candidate_count": total,
        "stable_generation_retry_candidate_count": stable_count,
        "unstable_generation_retry_candidate_count": total - stable_count,
        "reclassified_terminal_count": sum(1 for row in rows if recoverability_class(str(row.get("five_replicate_modal_class"))) == "not_recoverable_by_agent"),
        "reclassified_indeterminate_count": sum(1 for row in rows if recoverability_class(str(row.get("five_replicate_modal_class"))) == "indeterminate"),
        "safety_terminal_leakage_count": safety_count,
        "correct_unanswerable_retry_candidate_count": correct_unanswerable_count,
        "rows": rows,
    }


def analyze_unstable_samples(unstable_ids: list[str], replicate_classes: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for sample_id in unstable_ids:
        classes = [replicate.get(sample_id) for replicate in replicate_classes]
        observed = [value for value in classes if value]
        if INFRASTRUCTURE_CLASS in observed:
            cause = "infrastructure_failure"
        elif "answer_draft_grounded" in observed and any(str(value).startswith("refusal_") for value in observed):
            cause = "provider_nondeterminism"
        elif len(set(observed)) > 1:
            cause = "classification_boundary_ambiguity"
        else:
            cause = "unknown"
        rows.append(
            {
                "sample_id": sample_id,
                "classes": classes,
                "primary_instability_association": cause,
                "observable_differences": {"runtime_classes": sorted(set(observed)), "transition_pattern": transition_pattern(observed)},
            }
        )
    return {"schema_version": "opk-rag.task0074-unstable-sample-analysis.v1", "unstable_sample_count": len(rows), "rows": rows}


def analyze_task0070_indeterminate(candidate_rows: list[dict[str, Any]], replicate_classes: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for candidate in candidate_rows:
        sample_id = candidate["sample_id"]
        observed = [replicate.get(sample_id) for replicate in replicate_classes if replicate.get(sample_id)]
        counts = Counter(observed)
        modal_class, _ = counts.most_common(1)[0] if counts else (None, 0)
        remaining = len(observed) == 0 or recoverability_class(modal_class or "") == "indeterminate"
        rows.append(
            {
                "sample_id": sample_id,
                "generation_invocation_count": len(observed),
                "answer_count": counts["answer_draft_grounded"],
                "refusal_count": sum(count for value, count in counts.items() if value.startswith("refusal_") or value == "generation_contract_abstention"),
                "infrastructure_failure_count": counts[INFRASTRUCTURE_CLASS],
                "runtime_evidence_classes": [],
                "runtime_failure_classes": sorted(counts),
                "modal_class": modal_class,
                "remaining_indeterminate": remaining,
                "indeterminate_reason": "no_development_runtime_observation" if not observed else ("modal_class_indeterminate" if remaining else "resolved_by_repeated_observation"),
                "additional_signal_required": remaining,
            }
        )
    newly_resolved = sum(1 for row in rows if not row["remaining_indeterminate"])
    return {
        "schema_version": "opk-rag.task0074-task0070-indeterminate-analysis.v1",
        "task0070_candidate_count": len(rows),
        "previously_indeterminate_count": sum(1 for candidate in candidate_rows if candidate.get("remaining_indeterminate") is True),
        "newly_resolved_count": newly_resolved,
        "remaining_indeterminate_count": sum(1 for row in rows if row["remaining_indeterminate"]),
        "stable_generation_retry_candidate_count": sum(1 for row in rows if recoverability_class(row.get("modal_class") or "") == GENERATION_RETRY_CLASS),
        "stable_terminal_count": sum(1 for row in rows if row.get("modal_class") in TERMINAL_CLASSES),
        "infrastructure_confounded_count": sum(1 for row in rows if row["infrastructure_failure_count"] > 0),
        "rows": rows,
    }


def evaluate_stability_gates(preflight: dict[str, Any], replicate_aggregates: list[dict[str, Any]], three: dict[str, Any], candidate: dict[str, Any], privacy: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    total_infra = sum(int(aggregate.get("infrastructure_failure_count") or 0) for aggregate in replicate_aggregates)
    structural = len(replicate_aggregates) == 3 and all(int(aggregate.get("terminal_result_rows") or 0) == 28 for aggregate in replicate_aggregates)
    gates = {
        "preflight_gate": preflight.get("preflight_status") == "ready",
        "structural_completion_gate": structural,
        "infrastructure_stability_gate": total_infra == 0,
        "classification_stability_gate": int(three.get("three_replicate_exact_agreement_count") or 0) >= 26,
        "generation_retry_candidate_stability_gate": int(candidate.get("stable_generation_retry_candidate_count") or 0) >= 9,
        "safety_gate": int(candidate.get("safety_terminal_leakage_count") or 0) == 0 and int(candidate.get("correct_unanswerable_retry_candidate_count") or 0) == 0,
        "observability_gate": all(aggregate.get("complete_observability_records") is True for aggregate in replicate_aggregates),
        "privacy_gate": privacy.get("status") == "pass",
        "artifact_verifier_gate": verification.get("status") == "pass",
    }
    all_pass = all(gates.values())
    if all_pass:
        primary = "reference_runtime_stable_classification_stable"
        next_action = "implement_governed_single_attempt_generation_retry"
    elif not gates["infrastructure_stability_gate"]:
        primary = "reference_runtime_infrastructure_still_unstable"
        next_action = "fix_remaining_runtime_instability"
    elif not gates["classification_stability_gate"]:
        primary = "reference_runtime_stable_classification_unstable"
        next_action = "collect_additional_observability"
    elif not gates["generation_retry_candidate_stability_gate"]:
        primary = "generation_retry_candidates_not_reproducible"
        next_action = "retain_current_terminal_behavior"
    elif not gates["safety_gate"]:
        primary = "safety_gate_prevents_generation_retry"
        next_action = "retain_current_terminal_behavior"
    else:
        primary = "diagnosis_inconclusive"
        next_action = "collect_additional_observability"
    return {
        "schema_version": "opk-rag.task0074-stability-gates.v1",
        **gates,
        "all_gates_pass": all_pass,
        "primary_result_classification": primary,
        "recommended_next_action": next_action,
        "promotion_eligible": False,
        "recommended_variant": None,
    }


def _sum_row_value(sample_id: str, replicate_rows: list[dict[str, dict[str, Any]]] | None, key: str) -> int:
    if not replicate_rows:
        return 0
    return sum(int((rows.get(sample_id) or {}).get(key) or 0) for rows in replicate_rows)
