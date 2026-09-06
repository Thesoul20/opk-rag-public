from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.errors import AgentRuntimeError
from opk_rag.evaluation.post_generation_failure_classification import build_observability_record
from opk_rag.evaluation.stable_generation_retry_eligibility import (
    FORBIDDEN_RUNTIME_RULE_FIELDS,
    build_retry_eligibility_proposal,
    evaluate_disqualifier_flags,
)

GENERATION_RETRY_POLICY_ID = "opk-rag.governed-single-attempt-generation-retry.v1"
ELIGIBILITY_RULE_ID = "runtime_model_conservative_refusal_with_sufficient_evidence"

RETRY_OUTCOMES = {
    "retry_not_eligible",
    "retry_answer_draft_grounded",
    "retry_answer_draft_citation_failure",
    "retry_answer_draft_grounding_failure",
    "retry_answer_draft_unsupported_claim",
    "retry_refusal",
    "retry_contract_failure",
    "retry_empty_output",
    "retry_provider_failure",
    "retry_budget_exhausted",
    "retry_request_identity_mismatch",
}

DISQUALIFIER_REASON_CODES = {
    "safety_terminal_refusal": "generation_retry_ineligible_safety_terminal",
    "correct_unanswerable_runtime_signal": "generation_retry_ineligible_correct_unanswerable",
    "forbidden_claim_risk": "generation_retry_ineligible_forbidden_claim",
    "unsupported_claim_detected": "generation_retry_ineligible_unsupported_claim",
    "citation_failure_after_answer": "generation_retry_ineligible_citation_failure",
    "grounding_failure_after_answer": "generation_retry_ineligible_grounding_failure",
    "provider_authentication_failure": "generation_retry_ineligible_provider_authentication_failure",
    "provider_transport_failure": "generation_retry_ineligible_provider_transport_failure",
    "provider_timeout": "generation_retry_ineligible_provider_timeout",
    "provider_server_failure": "generation_retry_ineligible_provider_server_failure",
    "malformed_generation_contract": "generation_retry_ineligible_malformed_contract",
    "empty_output_without_refusal_semantics": "generation_retry_ineligible_empty_without_refusal",
    "missing_observability": "generation_retry_ineligible_missing_observability",
    "retrieval_evidence_appears_insufficient": "generation_retry_ineligible_insufficient_evidence",
    "retrieval_failure": "generation_retry_ineligible_retrieval_failure",
    "database_failure": "generation_retry_ineligible_database_failure",
    "retry_budget_exhausted": "generation_retry_ineligible_budget_exhausted",
    "prior_generation_retry_already_attempted": "generation_retry_ineligible_recursive_retry",
    "answer_draft_present": "generation_retry_ineligible_answer_draft_present",
    "valid_grounded_answer_already_exists": "generation_retry_ineligible_answer_already_grounded",
}

REQUIRED_SIGNAL_FIELDS = (
    "generation_invoked",
    "provider_call_completed",
    "response_contract_valid",
    "response_mode",
    "provider_refusal_detected",
    "runtime_failure_class",
    "runtime_evidence_class",
    "answerability_label",
    "generation_invocation_count",
    "generation_retries",
    "retry_budget_remaining",
    "observability_complete",
)


@dataclass(frozen=True)
class GenerationRetryEligibilityDecision:
    eligible: bool = False
    decision_reason: str = "retry_not_eligible_default"
    matched_rule_id: str | None = None
    disqualifying_rule_ids: tuple[str, ...] = ()
    required_signal_status: dict[str, str] = field(default_factory=dict)
    retry_budget_remaining: int = 0
    observability_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "opk-rag.generation-retry-eligibility-decision.v1",
            "eligible": self.eligible,
            "decision_reason": self.decision_reason,
            "matched_rule_id": self.matched_rule_id,
            "disqualifying_rule_ids": list(self.disqualifying_rule_ids),
            "required_signal_status": self.required_signal_status,
            "retry_budget_remaining": self.retry_budget_remaining,
            "observability_complete": self.observability_complete,
        }


def build_generation_retry_eligibility_input(
    *,
    observation: dict[str, Any],
    generation_invocation_count: int = 1,
    generation_retries: int = 0,
    retry_budget_remaining: int = 1,
) -> dict[str, Any]:
    _reject_forbidden_runtime_fields(observation)
    boundary = observation.get("boundary") or {}
    outcome = observation.get("outcome") or {}
    validation = observation.get("validation") or {}
    return {
        "schema_version": "opk-rag.generation-retry-eligibility-input.v1",
        "generation_invoked": bool(outcome),
        "provider_call_completed": outcome.get("provider_call_completed") is True,
        "response_contract_valid": outcome.get("response_contract_valid") is True,
        "response_mode": outcome.get("response_mode"),
        "provider_refusal_detected": outcome.get("provider_refusal_detected") is True,
        "original_refusal_reason_code": outcome.get("refusal_reason_code"),
        "runtime_failure_class": observation.get("runtime_failure_class"),
        "runtime_evidence_class": observation.get("runtime_evidence_class"),
        "answerability_label": boundary.get("answerability_label"),
        "initial_answerability_reason_code": boundary.get("answerability_reason_code"),
        "generation_invocation_count": generation_invocation_count,
        "generation_retries": generation_retries,
        "retry_budget_remaining": retry_budget_remaining,
        "observability_complete": _observability_complete(observation),
        "safety_terminal_signal": observation.get("runtime_failure_class") in {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"},
        "correct_unanswerable_signal": observation.get("runtime_failure_class") == "refusal_correct_unanswerable",
        "forbidden_claim_signal": observation.get("runtime_failure_class") == "refusal_forbidden_claim_risk",
        "unsupported_claim_signal": validation.get("unsupported_claim_detected") is True or observation.get("runtime_failure_class") == "answer_draft_unsupported_claim",
        "answer_draft_present": outcome.get("answer_draft_present") is True,
        "citation_validation_passed": validation.get("citation_validation_passed"),
        "grounding_validation_passed": validation.get("grounding_validation_passed"),
        "generation_error_code": outcome.get("generation_error_code"),
        "generation_error_type": outcome.get("generation_error_type"),
        "selected_evidence_identity_digest": _selected_evidence_digest(boundary),
        "selected_evidence_count": boundary.get("selected_evidence_count"),
        "generation_contract_id": boundary.get("generation_contract_id"),
        "provider_identity": boundary.get("provider_identity"),
        "model_identity": boundary.get("model_identity"),
    }


def decide_generation_retry_eligibility(observation: dict[str, Any], *, proposal: dict[str, Any] | None = None) -> GenerationRetryEligibilityDecision:
    _reject_forbidden_runtime_fields(observation)
    proposal = proposal or build_retry_eligibility_proposal()
    required = _required_signal_status(observation)
    retry_budget = _safe_int(observation.get("retry_budget_remaining"))
    complete = observation.get("observability_complete") is True

    flags = _disqualifier_flags(observation)
    disqualified = tuple(rule_id for rule_id, present in flags.items() if present)
    if disqualified:
        first = disqualified[0]
        return GenerationRetryEligibilityDecision(
            eligible=False,
            decision_reason=DISQUALIFIER_REASON_CODES.get(first, f"generation_retry_ineligible_{first}"),
            matched_rule_id=first,
            disqualifying_rule_ids=disqualified,
            required_signal_status=required,
            retry_budget_remaining=retry_budget,
            observability_complete=complete,
        )

    rule = (proposal.get("eligibility_rules") or [{}])[0]
    conditions = rule.get("conditions") or {}
    mismatches = [_condition_mismatch(field, expected, observation.get(_runtime_field_name(field))) for field, expected in conditions.items()]
    mismatches = [value for value in mismatches if value is not None]
    if mismatches:
        return GenerationRetryEligibilityDecision(
            eligible=False,
            decision_reason="retry_not_eligible_default",
            matched_rule_id=None,
            disqualifying_rule_ids=(),
            required_signal_status=required | {field: "mismatch" for field in mismatches},
            retry_budget_remaining=retry_budget,
            observability_complete=complete,
        )
    if observation.get("runtime_failure_class") != "refusal_model_conservative":
        return GenerationRetryEligibilityDecision(
            eligible=False,
            decision_reason="retry_not_eligible_default",
            required_signal_status=required | {"runtime_failure_class": "mismatch"},
            retry_budget_remaining=retry_budget,
            observability_complete=complete,
        )
    return GenerationRetryEligibilityDecision(
        eligible=True,
        decision_reason="eligible_for_single_generation_retry",
        matched_rule_id=ELIGIBILITY_RULE_ID,
        required_signal_status=required,
        retry_budget_remaining=retry_budget,
        observability_complete=complete,
    )


def generation_request_identity_digest(request: dict[str, Any]) -> str:
    _reject_forbidden_runtime_fields(request)
    sanitized = dict(request)
    for key in ("generation_invocation_id", "generation_attempt_index", "retry_of_invocation_id", "started_at", "completed_at", "latency_ms", "provider_latency_ms"):
        sanitized.pop(key, None)
    return stable_digest(sanitized)


def validate_retry_request_identity(attempt_1_request: dict[str, Any], attempt_2_request: dict[str, Any]) -> dict[str, Any]:
    digest_1 = generation_request_identity_digest(attempt_1_request)
    digest_2 = generation_request_identity_digest(attempt_2_request)
    return {
        "schema_version": "opk-rag.generation-retry-request-identity-validation.v1",
        "valid": digest_1 == digest_2,
        "generation_request_identity_digest_attempt_1": digest_1,
        "generation_request_identity_digest_attempt_2": digest_2,
        "reason_code": "request_identity_match" if digest_1 == digest_2 else "request_identity_mismatch",
    }


def classify_retry_outcome(observation: dict[str, Any] | None = None, *, provider_failure: bool = False, request_identity_mismatch: bool = False, budget_exhausted: bool = False) -> str:
    if request_identity_mismatch:
        return "retry_request_identity_mismatch"
    if budget_exhausted:
        return "retry_budget_exhausted"
    if provider_failure:
        return "retry_provider_failure"
    if not observation:
        return "retry_contract_failure"
    outcome = observation.get("outcome") or {}
    validation = observation.get("validation") or {}
    runtime_class = observation.get("runtime_failure_class")
    if outcome.get("response_mode") == "empty":
        return "retry_empty_output"
    if outcome.get("response_contract_valid") is False or runtime_class == "generation_contract_invalid":
        return "retry_contract_failure"
    if runtime_class == "answer_draft_grounded" and validation.get("grounding_validation_passed") is True:
        return "retry_answer_draft_grounded"
    if runtime_class == "answer_draft_unsupported_claim" or validation.get("unsupported_claim_detected") is True:
        return "retry_answer_draft_unsupported_claim"
    if runtime_class == "answer_draft_citation_failure" or validation.get("citation_validation_passed") is False:
        return "retry_answer_draft_citation_failure"
    if runtime_class == "answer_draft_grounding_failure" or validation.get("grounding_validation_passed") is False:
        return "retry_answer_draft_grounding_failure"
    if outcome.get("provider_refusal_detected") is True or str(runtime_class or "").startswith("refusal_"):
        return "retry_refusal"
    return "retry_contract_failure"


def execute_single_generation_retry(
    *,
    eligibility_decision: GenerationRetryEligibilityDecision,
    attempt_1_request: dict[str, Any],
    attempt_2_request: dict[str, Any],
    invoke_generation: Callable[[], dict[str, Any]],
    build_retry_observation: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if not eligibility_decision.eligible:
        return {"retry_attempted": False, "retry_outcome": "retry_not_eligible", "eligibility": eligibility_decision.to_dict()}
    identity = validate_retry_request_identity(attempt_1_request, attempt_2_request)
    if not identity["valid"]:
        return {"retry_attempted": False, "retry_outcome": "retry_request_identity_mismatch", "request_identity": identity, "eligibility": eligibility_decision.to_dict()}
    try:
        generation_result = invoke_generation()
    except AgentRuntimeError:
        raise
    except Exception:
        return {"retry_attempted": True, "retry_outcome": "retry_provider_failure", "request_identity": identity, "eligibility": eligibility_decision.to_dict()}
    retry_observation = build_retry_observation(generation_result)
    return {
        "retry_attempted": True,
        "retry_outcome": classify_retry_outcome(retry_observation),
        "request_identity": identity,
        "eligibility": eligibility_decision.to_dict(),
        "retry_observation": retry_observation,
    }


def build_retry_observability_record(boundary: dict[str, Any], outcome: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return build_observability_record(boundary, outcome, validation).to_dict()


def _disqualifier_flags(observation: dict[str, Any]) -> dict[str, bool]:
    flags = evaluate_disqualifier_flags(observation)
    runtime_class = str(observation.get("runtime_failure_class") or "")
    flags["provider_server_failure"] = runtime_class == "provider_server_failure"
    flags["retrieval_failure"] = runtime_class == "database_failure_before_generation"
    flags["database_failure"] = runtime_class == "database_failure_before_generation" or str(observation.get("generation_error_type") or "") == "database"
    flags["answer_draft_present"] = observation.get("answer_draft_present") is True
    flags["valid_grounded_answer_already_exists"] = runtime_class == "answer_draft_grounded" and observation.get("grounding_validation_passed") is True
    return flags


def _required_signal_status(observation: dict[str, Any]) -> dict[str, str]:
    return {field: ("present" if field in observation and observation.get(field) is not None else "missing") for field in REQUIRED_SIGNAL_FIELDS}


def _condition_mismatch(field: str, expected: Any, actual: Any) -> str | None:
    if isinstance(expected, list):
        return None if actual in set(expected) else field
    return None if actual == expected else field


def _runtime_field_name(proposal_field: str) -> str:
    return {
        "original_response_contract_valid": "response_contract_valid",
        "initial_answerability_label": "answerability_label",
        "generation_attempt_count": "generation_invocation_count",
    }.get(proposal_field, proposal_field)


def _observability_complete(observation: dict[str, Any]) -> bool:
    return all(isinstance(observation.get(name), dict) for name in ("boundary", "outcome", "validation"))


def _selected_evidence_digest(boundary: dict[str, Any]) -> str:
    return stable_digest(
        {
            "selected_evidence_count": boundary.get("selected_evidence_count"),
            "canonical_document_count": boundary.get("canonical_document_count"),
            "canonical_scope_count": boundary.get("canonical_scope_count"),
            "generation_invocation_id": boundary.get("generation_invocation_id"),
        }
    )


def _reject_forbidden_runtime_fields(value: dict[str, Any]) -> None:
    present = sorted(FORBIDDEN_RUNTIME_RULE_FIELDS & _collect_keys(value))
    if present:
        raise ValueError(f"runtime generation retry input contains forbidden fields: {present}")


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
