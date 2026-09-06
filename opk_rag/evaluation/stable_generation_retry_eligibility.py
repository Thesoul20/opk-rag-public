from __future__ import annotations

from collections import Counter
from typing import Any


PROPOSAL_ID = "opk-rag.stable-generation-retry-eligibility.v1-proposal"
FORBIDDEN_RUNTIME_RULE_FIELDS = {
    "benchmark_label",
    "expected_action",
    "forbidden_claims",
    "gold_answerability",
    "gold_evidence",
    "gold_evidence_identities",
    "prior_benchmark_correctness",
    "required_claims",
    "sample_id",
    "task0070_classification",
}

RUNTIME_DEPLOYABLE_FIELDS = {
    "answer_draft_present",
    "correct_unanswerable_signal",
    "final_action",
    "forbidden_claim_signal",
    "generation_attempt_count",
    "generation_contract_id",
    "generation_invoked",
    "grounding_validation_passed",
    "initial_answerability_label",
    "initial_answerability_reason_code",
    "malformed_generation_contract",
    "observability_complete",
    "original_generation_outcome_class",
    "original_refusal_reason_code",
    "original_response_contract_valid",
    "prior_generation_retry_already_attempted",
    "provider_call_completed",
    "provider_identity",
    "provider_refusal_detected",
    "response_mode",
    "retry_budget_remaining",
    "runtime_evidence_class",
    "safety_terminal_signal",
    "selected_evidence_count",
    "selected_evidence_identity_digest",
    "unsupported_claim_signal",
}


def validate_runtime_rule(rule: dict[str, Any]) -> None:
    present = sorted(FORBIDDEN_RUNTIME_RULE_FIELDS & _collect_keys(rule))
    if present:
        raise ValueError(f"runtime rule contains forbidden fields: {present}")
    fields = set(rule.get("field_requirements") or [])
    unknown = sorted(fields - RUNTIME_DEPLOYABLE_FIELDS)
    if unknown:
        raise ValueError(f"runtime rule contains unknown runtime fields: {unknown}")


def build_disqualifying_signals() -> list[dict[str, Any]]:
    rows = [
        ("safety_terminal_refusal", "final_abstention", False, False, False, "generation_retry_ineligible_safety_terminal"),
        ("correct_unanswerable_runtime_signal", "final_abstention", False, False, False, "generation_retry_ineligible_correct_unanswerable"),
        ("forbidden_claim_risk", "final_abstention", False, False, False, "generation_retry_ineligible_forbidden_claim"),
        ("unsupported_claim_detected", "final_abstention", False, True, False, "generation_retry_ineligible_unsupported_claim"),
        ("citation_failure_after_answer", "final_abstention", False, True, False, "generation_retry_ineligible_citation_failure"),
        ("grounding_failure_after_answer", "final_abstention", False, True, False, "generation_retry_ineligible_grounding_failure"),
        ("provider_authentication_failure", "infrastructure_failure", False, False, True, "generation_retry_ineligible_provider_authentication_failure"),
        ("provider_transport_failure", "infrastructure_failure", False, False, True, "generation_retry_ineligible_provider_transport_failure"),
        ("provider_timeout", "infrastructure_failure", False, False, True, "generation_retry_ineligible_provider_timeout"),
        ("malformed_generation_contract", "final_abstention", False, False, False, "generation_retry_ineligible_malformed_contract"),
        ("empty_output_without_refusal_semantics", "final_abstention", False, False, False, "generation_retry_ineligible_empty_without_refusal"),
        ("missing_observability", "final_abstention", False, False, False, "generation_retry_ineligible_missing_observability"),
        ("retrieval_evidence_appears_insufficient", "final_abstention", False, True, False, "generation_retry_ineligible_insufficient_evidence"),
        ("retry_budget_exhausted", "final_abstention", False, False, False, "generation_retry_ineligible_budget_exhausted"),
        ("prior_generation_retry_already_attempted", "final_abstention", False, False, False, "generation_retry_ineligible_recursive_retry"),
    ]
    return [
        {
            "signal": signal,
            "terminal_action": terminal_action,
            "generation_retry_permitted": generation_retry,
            "retrieval_recovery_permitted": retrieval_recovery,
            "infrastructure_retry_permitted": infrastructure_retry,
            "required_trace_reason": trace_reason,
        }
        for signal, terminal_action, generation_retry, retrieval_recovery, infrastructure_retry, trace_reason in rows
    ]


def build_retry_eligibility_proposal() -> dict[str, Any]:
    eligibility_rules = [
        {
            "rule_id": "runtime_model_conservative_refusal_with_sufficient_evidence",
            "rule_type": "runtime_deployable",
            "field_requirements": [
                "generation_invoked",
                "provider_call_completed",
                "original_response_contract_valid",
                "response_mode",
                "provider_refusal_detected",
                "original_refusal_reason_code",
                "runtime_evidence_class",
                "initial_answerability_label",
                "generation_attempt_count",
                "retry_budget_remaining",
                "observability_complete",
                "safety_terminal_signal",
                "correct_unanswerable_signal",
                "unsupported_claim_signal",
                "forbidden_claim_signal",
            ],
            "conditions": {
                "generation_invoked": True,
                "provider_call_completed": True,
                "original_response_contract_valid": True,
                "response_mode": "abstain",
                "provider_refusal_detected": True,
                "original_refusal_reason_code": ["unsupported_claims", "ungrounded_answer", "invalid_citations", "model_abstained", "answerable_generation_abstained"],
                "runtime_evidence_class": "runtime_evidence_appears_sufficient",
                "initial_answerability_label": ["answerable", "partially_answerable"],
                "generation_attempt_count": 1,
                "retry_budget_remaining": 1,
                "observability_complete": True,
                "safety_terminal_signal": False,
                "correct_unanswerable_signal": False,
                "unsupported_claim_signal": False,
                "forbidden_claim_signal": False,
            },
            "decision": "eligible_for_single_generation_retry",
        }
    ]
    for rule in eligibility_rules:
        validate_runtime_rule(rule)
    return {
        "proposal_id": PROPOSAL_ID,
        "schema_version": "opk-rag.stable-generation-retry-eligibility-proposal.v1",
        "status": "proposal_only",
        "default_decision": "not_eligible",
        "deterministic_rule_order": [
            "disqualifying_rules",
            "eligibility_rules",
            "indeterminate_rules",
            "default_not_eligible",
        ],
        "required_inputs": [
            "original_generation_invocation_id",
            "original_generation_outcome_class",
            "original_refusal_reason_code",
            "original_response_contract_valid",
            "runtime_evidence_class",
            "initial_answerability_label",
            "initial_answerability_reason_code",
            "selected_evidence_identity_digest",
            "selected_evidence_count",
            "provider_identity",
            "model_identity",
            "generation_contract_id",
            "generation_attempt_count",
            "retry_budget_remaining",
            "safety_terminal_signal",
            "correct_unanswerable_signal",
            "forbidden_claim_signal",
            "unsupported_claim_signal",
            "observability_complete",
        ],
        "eligibility_rules": eligibility_rules,
        "disqualifying_rules": build_disqualifying_signals(),
        "indeterminate_rules": [
            {
                "rule_id": "missing_or_conflicting_runtime_signal",
                "rule_type": "runtime_deployable",
                "decision": "not_eligible",
                "trace_reason": "generation_retry_ineligible_indeterminate_runtime_signal",
            }
        ],
        "retry_budget": {
            "maximum_generation_retries": 1,
            "maximum_total_generation_calls": 2,
            "retrieval_retry_allowed": False,
            "query_reformulation_allowed": False,
            "recursive_retry_allowed": False,
        },
        "same_invocation_constraints": {
            "same_evidence_bundle_required": True,
            "prompt_change_allowed": False,
            "provider_change_allowed": False,
            "model_change_allowed": False,
            "safety_rule_override_allowed": False,
        },
        "terminal_actions": {
            "retry_answer_grounded": "final_answer",
            "retry_answer_citation_failure": "final_abstention",
            "retry_answer_grounding_failure": "final_abstention",
            "retry_answer_unsupported_claim": "final_abstention",
            "retry_refusal": "final_abstention",
            "retry_contract_failure": "final_abstention",
            "retry_empty_output": "final_abstention",
            "retry_infrastructure_failure": "infrastructure_failure",
            "retry_budget_exhausted": "final_abstention",
            "retry_not_eligible": "final_abstention",
        },
        "trace_requirements": [
            "generation_retry_eligibility_decision",
            "generation_retry_eligibility_reason",
            "generation_attempt_count",
            "retry_budget_remaining",
            "selected_evidence_identity_digest",
            "original_generation_invocation_id",
        ],
        "validation_requirements": {
            "citation_validation_required": True,
            "grounding_validation_required": True,
            "retry_refusal_terminates": True,
            "retry_validation_failure_terminates": True,
            "unsupported_answer_fallback_allowed": False,
        },
        "runtime_gold_usage_allowed": False,
        "promotion_evaluation": False,
    }


def evaluate_disqualifier_flags(observation: dict[str, Any]) -> dict[str, bool]:
    runtime_class = str(observation.get("runtime_failure_class") or "")
    return {
        "safety_terminal_refusal": runtime_class in {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"},
        "correct_unanswerable_runtime_signal": runtime_class == "refusal_correct_unanswerable",
        "forbidden_claim_risk": runtime_class == "refusal_forbidden_claim_risk",
        "unsupported_claim_detected": observation.get("unsupported_claim_detected") is True or runtime_class == "answer_draft_unsupported_claim",
        "citation_failure_after_answer": observation.get("citation_validation_passed") is False,
        "grounding_failure_after_answer": observation.get("grounding_validation_passed") is False and observation.get("answer_draft_present") is True,
        "provider_authentication_failure": runtime_class == "provider_authentication_failure",
        "provider_transport_failure": runtime_class == "provider_transport_failure",
        "provider_timeout": runtime_class == "provider_timeout",
        "malformed_generation_contract": observation.get("response_contract_valid") is False,
        "empty_output_without_refusal_semantics": observation.get("response_mode") == "empty" and observation.get("provider_refusal_detected") is not True,
        "missing_observability": observation.get("observability_complete") is not True,
        "retrieval_evidence_appears_insufficient": observation.get("runtime_evidence_class") == "runtime_evidence_appears_insufficient",
        "retry_budget_exhausted": observation.get("retry_budget_remaining") == 0,
        "prior_generation_retry_already_attempted": _safe_int(observation.get("generation_retries")) > 0,
    }


def eligible_by_runtime_rule(observation: dict[str, Any]) -> bool:
    flags = evaluate_disqualifier_flags(observation)
    if any(flags.values()):
        return False
    return (
        observation.get("generation_invoked") is True
        and observation.get("provider_call_completed") is True
        and observation.get("response_contract_valid") is True
        and observation.get("response_mode") == "abstain"
        and observation.get("provider_refusal_detected") is True
        and observation.get("runtime_failure_class") == "refusal_model_conservative"
        and observation.get("runtime_evidence_class") == "runtime_evidence_appears_sufficient"
        and observation.get("answerability_label") in {"answerable", "partially_answerable"}
        and int(observation.get("generation_invocation_count") or 0) == 1
    )


def summarize_classes(observations: list[dict[str, Any]], field: str) -> list[str]:
    return sorted(str(value) for value in Counter(obs.get(field) for obs in observations) if value not in {"None", ""})


def _collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_collect_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_collect_keys(child))
        return keys
    return set()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
