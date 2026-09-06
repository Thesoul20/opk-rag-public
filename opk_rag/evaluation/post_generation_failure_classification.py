from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

OBSERVABILITY_CONTRACT_ID = "opk-rag.post-generation-failure-observability.v1"
BOUNDARY_SNAPSHOT_SCHEMA = "opk-rag.generation-boundary-snapshot.v1"
OUTCOME_SNAPSHOT_SCHEMA = "opk-rag.generation-outcome-snapshot.v1"
VALIDATION_SNAPSHOT_SCHEMA = "opk-rag.post-generation-validation-snapshot.v1"
OBSERVABILITY_RECORD_SCHEMA = "opk-rag.post-generation-observability-record.v1"

RuntimeFailureClass = Literal[
    "answer_draft_grounded",
    "answer_draft_citation_failure",
    "answer_draft_grounding_failure",
    "answer_draft_unsupported_claim",
    "answer_draft_other_validation_failure",
    "refusal_claimed_insufficient_evidence",
    "refusal_claimed_missing_scope",
    "refusal_claimed_ambiguous_evidence",
    "refusal_claimed_conflicting_evidence",
    "refusal_safety_terminal",
    "refusal_forbidden_claim_risk",
    "refusal_correct_unanswerable",
    "refusal_policy_terminal",
    "refusal_model_conservative",
    "refusal_without_specific_reason",
    "generation_contract_abstention",
    "generation_contract_invalid",
    "generation_empty_output",
    "generation_schema_repair_exhausted",
    "provider_transport_failure",
    "provider_authentication_failure",
    "provider_timeout",
    "provider_server_failure",
    "database_failure_before_generation",
    "agent_loop_failure_before_snapshot",
    "insufficient_runtime_signal",
    "conflicting_runtime_signals",
    "missing_observability_field",
    "unclassified_generation_failure",
]

RecoverabilityClass = Literal[
    "retrieval_recovery_candidate",
    "generation_retry_candidate",
    "retrieval_then_generation_candidate",
    "terminal_safety_failure",
    "correct_terminal_abstention",
    "infrastructure_retry_candidate",
    "not_recoverable_by_agent",
    "indeterminate",
]


@dataclass(frozen=True)
class GenerationObservabilityRecord:
    boundary: dict[str, Any]
    outcome: dict[str, Any]
    validation: dict[str, Any]
    runtime_failure_class: RuntimeFailureClass
    runtime_evidence_class: str
    recoverability_class: RecoverabilityClass
    classification_confidence: float
    secondary_reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": OBSERVABILITY_RECORD_SCHEMA,
            "contract_id": OBSERVABILITY_CONTRACT_ID,
            "boundary": self.boundary,
            "outcome": self.outcome,
            "validation": self.validation,
            "runtime_failure_class": self.runtime_failure_class,
            "runtime_evidence_class": self.runtime_evidence_class,
            "recoverability_class": self.recoverability_class,
            "classification_confidence": self.classification_confidence,
            "secondary_reason_codes": list(self.secondary_reason_codes),
        }
        payload["record_digest"] = stable_digest(payload)
        return payload


def build_generation_boundary_snapshot(
    *,
    run_id: str,
    sample_id: str | None,
    split: str | None,
    replicate_id: str | int | None,
    variant: str,
    generation_invocation_id: str,
    question: str,
    evidence_state: dict[str, Any],
    answerability_state: dict[str, Any],
    provider_identity: dict[str, Any] | None = None,
    model_identity: dict[str, Any] | None = None,
    generation_contract_id: str = "answer-response-v3",
    generation_prompt_identity: str = "answer-prompt-v4",
    started_at: str | None = None,
) -> dict[str, Any]:
    citation_ids = list(evidence_state.get("citation_ids") or [])
    chunk_ids = list(evidence_state.get("chunk_ids") or [])
    document_ids = list(evidence_state.get("document_ids") or [])
    evidence_count = int(answerability_state.get("evidence_count") or len(chunk_ids) or len(citation_ids))
    return _complete(
        {
            "schema_version": BOUNDARY_SNAPSHOT_SCHEMA,
            "contract_id": OBSERVABILITY_CONTRACT_ID,
            "run_id": run_id,
            "replicate_id": str(replicate_id) if replicate_id is not None else "not_applicable",
            "split": split or "unknown",
            "sample_id": sample_id,
            "variant": variant,
            "generation_invocation_id": generation_invocation_id,
            "query_identity": {"query_digest": stable_digest(question), "query_length": len(question)},
            "answerability_label": answerability_state.get("status") or "unknown",
            "answerability_reason_code": answerability_state.get("reason_code") or "unknown",
            "answerability_confidence": answerability_state.get("confidence"),
            "partial_answer_mode": answerability_state.get("status") == "partially_answerable",
            "retrieval_attempt_count": int(evidence_state.get("search_round_count") or 0),
            "candidate_count": evidence_state.get("result_count"),
            "selected_evidence_count": evidence_count,
            "canonical_document_count": len(set(str(value) for value in document_ids)),
            "canonical_scope_count": len(set(str(value) for value in chunk_ids)),
            "document_diversity": _ratio(len(set(document_ids)), evidence_count),
            "scope_diversity": _ratio(len(set(chunk_ids)), evidence_count),
            "retrieval_score_summary": {
                "score_source": "not_available_in_agent_state",
                "min": None,
                "max": None,
                "margin": None,
            },
            "evidence_token_estimate": evidence_state.get("context_token_count"),
            "evidence_budget_utilization": _ratio(evidence_state.get("context_token_count"), evidence_state.get("context_token_budget")),
            "required_generation_mode": "grounded_answer_json_contract",
            "generation_contract_id": generation_contract_id,
            "generation_prompt_identity": generation_prompt_identity,
            "provider_identity": provider_identity or {"provider_id": "unknown"},
            "model_identity": model_identity or {"model_id": "unknown"},
            "started_at": started_at or utc_now(),
        }
    )


def build_generation_outcome_snapshot(
    *,
    generation_invocation_id: str,
    tool_summary: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    payload = (tool_summary or {}).get("answer_response") or tool_summary or {}
    status = payload.get("status")
    citations = [citation for citation in payload.get("citations", []) if isinstance(citation, dict)]
    refusal_reason = payload.get("refusal_reason_code")
    error_code = (failure or {}).get("code")
    error_origin = (failure or {}).get("origin")
    response_mode = _response_mode(payload, failure)
    provider_refusal = status == "refused" or bool(refusal_reason)
    return _complete(
        {
            "schema_version": OUTCOME_SNAPSHOT_SCHEMA,
            "contract_id": OBSERVABILITY_CONTRACT_ID,
            "generation_invocation_id": generation_invocation_id,
            "provider_call_attempted": failure is None or error_origin in {"provider", "infrastructure", "core_rag"},
            "provider_call_completed": failure is None and bool(payload),
            "provider_transport_status": _transport_status(error_code, error_origin),
            "provider_http_status_class": _http_status_class(failure),
            "provider_latency_ms": latency_ms,
            "response_received": bool(payload),
            "response_contract_valid": status in {"answered", "refused"},
            "response_schema_version": payload.get("output_schema_version") or "unknown",
            "response_mode": response_mode,
            "answer_draft_present": status == "answered" and bool(payload.get("answer_hash")),
            "answer_draft_length_bucket": _answer_length_bucket(payload),
            "citation_reference_count": len(citations),
            "citation_reference_identity_count": len({citation.get("citation_id") for citation in citations}),
            "abstention_present": status == "refused",
            "provider_refusal_detected": provider_refusal,
            "provider_refusal_source": "structured_abstention" if status == "refused" else "not_applicable",
            "refusal_reason_code": refusal_reason or "not_applicable",
            "refusal_reason_confidence": 0.9 if refusal_reason else None,
            "empty_output": bool(payload.get("empty_output")) or response_mode == "empty",
            "malformed_output": response_mode == "malformed",
            "repair_attempted": False,
            "repair_succeeded": False,
            "generation_error_type": error_origin or "not_applicable",
            "generation_error_code": error_code or "not_applicable",
            "completed_at": completed_at or utc_now(),
        }
    )


def build_validation_snapshot(
    *,
    generation_invocation_id: str,
    generation_state: dict[str, Any],
    verification_state: dict[str, Any],
    final_action: str | None,
    final_termination_reason: str | None,
) -> dict[str, Any]:
    answer_present = generation_state.get("status") == "answered"
    verification_attempted = bool(verification_state.get("attempted"))
    grounding_passed = verification_state.get("valid") is True
    return _complete(
        {
            "schema_version": VALIDATION_SNAPSHOT_SCHEMA,
            "contract_id": OBSERVABILITY_CONTRACT_ID,
            "generation_invocation_id": generation_invocation_id,
            "citation_validation_invoked": answer_present,
            "citation_validation_passed": True if answer_present else None,
            "citation_failure_reason": "not_applicable" if answer_present else "validation_not_invoked_for_abstention",
            "grounding_validation_invoked": verification_attempted,
            "grounding_validation_passed": grounding_passed if verification_attempted else None,
            "grounding_failure_reason": verification_state.get("reason_code") or ("not_applicable" if verification_attempted else "validation_not_invoked"),
            "unsupported_claim_check_invoked": verification_attempted,
            "unsupported_claim_detected": verification_state.get("reason_code") == "unsupported_claims" if verification_attempted else None,
            "final_action": final_action or "unknown",
            "final_termination_reason": final_termination_reason or "unknown",
        }
    )


def build_observability_record(boundary: dict[str, Any], outcome: dict[str, Any], validation: dict[str, Any]) -> GenerationObservabilityRecord:
    runtime_evidence_class = classify_runtime_evidence(boundary)
    runtime_class, confidence, reasons = classify_runtime_generation_outcome(boundary, outcome, validation)
    recoverability = classify_recoverability(runtime_class, runtime_evidence_class)
    return GenerationObservabilityRecord(
        boundary=boundary,
        outcome=outcome,
        validation=validation,
        runtime_failure_class=runtime_class,
        runtime_evidence_class=runtime_evidence_class,
        recoverability_class=recoverability,
        classification_confidence=confidence,
        secondary_reason_codes=tuple(reasons),
    )


def classify_runtime_evidence(boundary: dict[str, Any]) -> str:
    label = boundary.get("answerability_label")
    selected = int(boundary.get("selected_evidence_count") or 0)
    docs = int(boundary.get("canonical_document_count") or 0)
    if label in {"answerable", "partially_answerable"} and selected > 0 and docs > 0:
        return "runtime_evidence_appears_sufficient"
    if label in {"unanswerable", "insufficient_evidence"} or selected == 0:
        return "runtime_evidence_appears_insufficient"
    if boundary.get("answerability_reason_code") in {"conflicting_evidence"}:
        return "runtime_evidence_conflicting"
    return "runtime_evidence_indeterminate"


def classify_runtime_generation_outcome(boundary: dict[str, Any], outcome: dict[str, Any], validation: dict[str, Any]) -> tuple[RuntimeFailureClass, float, list[str]]:
    if outcome.get("provider_call_completed") is False and outcome.get("generation_error_code") != "not_applicable":
        code = str(outcome.get("generation_error_code"))
        if "timeout" in code:
            return "provider_timeout", 0.9, [code]
        if "auth" in code:
            return "provider_authentication_failure", 0.8, [code]
        if "server" in code or outcome.get("provider_http_status_class") == "5xx":
            return "provider_server_failure", 0.8, [code]
        return "provider_transport_failure", 0.75, [code]
    if outcome.get("empty_output"):
        return "generation_empty_output", 0.9, []
    if outcome.get("malformed_output"):
        return "generation_contract_invalid", 0.85, []
    if outcome.get("response_contract_valid") is False:
        return "generation_contract_invalid", 0.75, []
    if outcome.get("answer_draft_present"):
        if validation.get("grounding_validation_passed") is True:
            return "answer_draft_grounded", 0.95, []
        if validation.get("grounding_failure_reason") == "unsupported_claims":
            return "answer_draft_unsupported_claim", 0.9, ["unsupported_claims"]
        if validation.get("grounding_validation_passed") is False:
            return "answer_draft_grounding_failure", 0.85, [str(validation.get("grounding_failure_reason"))]
        if validation.get("citation_validation_passed") is False:
            return "answer_draft_citation_failure", 0.85, [str(validation.get("citation_failure_reason"))]
        return "answer_draft_other_validation_failure", 0.55, []
    reason = str(outcome.get("refusal_reason_code") or "")
    if outcome.get("abstention_present") or outcome.get("provider_refusal_detected"):
        if reason in {"no_evidence", "insufficient_evidence", "no_relevant_context", "partial_evidence"}:
            return "refusal_claimed_insufficient_evidence", 0.85, [reason]
        if reason in {"partial_supported_scope_missing", "missing_scope"}:
            return "refusal_claimed_missing_scope", 0.8, [reason]
        if reason == "conflicting_evidence":
            return "refusal_claimed_conflicting_evidence", 0.85, [reason]
        if reason in {"prompt_injection_detected", "false_premise", "forbidden", "disallowed"}:
            return "refusal_safety_terminal", 0.9, [reason]
        if reason in {"unsupported_claims", "ungrounded_answer", "invalid_citations"}:
            if classify_runtime_evidence(boundary) == "runtime_evidence_appears_sufficient":
                return "refusal_model_conservative", 0.7, [reason]
            return "generation_contract_abstention", 0.6, [reason]
        if reason in {"model_abstained", "answerable_generation_abstained"}:
            if classify_runtime_evidence(boundary) == "runtime_evidence_appears_sufficient":
                return "refusal_model_conservative", 0.75, [reason]
            return "generation_contract_abstention", 0.65, [reason]
        return "refusal_without_specific_reason", 0.55, [reason or "missing_reason_code"]
    if _missing_required(outcome):
        return "missing_observability_field", 0.8, []
    return "unclassified_generation_failure", 0.4, []


def classify_recoverability(runtime_failure_class: RuntimeFailureClass, runtime_evidence_class: str) -> RecoverabilityClass:
    if runtime_failure_class in {"provider_transport_failure", "provider_timeout", "provider_server_failure"}:
        return "infrastructure_retry_candidate"
    if runtime_failure_class in {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"}:
        return "terminal_safety_failure"
    if runtime_failure_class in {"answer_draft_grounded", "refusal_correct_unanswerable"}:
        return "not_recoverable_by_agent"
    if runtime_failure_class in {"refusal_model_conservative", "generation_contract_abstention", "generation_empty_output", "generation_schema_repair_exhausted"} and runtime_evidence_class == "runtime_evidence_appears_sufficient":
        return "generation_retry_candidate"
    if runtime_failure_class in {"refusal_claimed_insufficient_evidence", "refusal_claimed_missing_scope"}:
        return "retrieval_recovery_candidate"
    if runtime_failure_class in {"answer_draft_grounding_failure", "answer_draft_unsupported_claim"}:
        return "retrieval_then_generation_candidate"
    return "indeterminate"


def classify_offline_failure_interpretation(runtime_record: dict[str, Any], gold: dict[str, Any] | None) -> dict[str, Any]:
    gold = gold or {}
    expected = gold.get("expected_action")
    runtime_class = runtime_record.get("runtime_failure_class")
    interpretation = "offline_indeterminate"
    if expected == "answer" and runtime_class in {"refusal_model_conservative", "generation_contract_abstention"}:
        interpretation = "suspected_false_generation_refusal"
    elif expected == "abstain" and runtime_class in {"refusal_claimed_insufficient_evidence", "refusal_safety_terminal"}:
        interpretation = "likely_correct_terminal_abstention"
    elif runtime_class == "answer_draft_grounded":
        interpretation = "grounded_answer_observed"
    return {
        "schema_version": "opk-rag.offline-post-generation-failure-interpretation.v1",
        "uses_gold_fields": bool(gold),
        "offline_failure_interpretation": interpretation,
        "gold_field_names": sorted(gold),
    }


def reject_runtime_rule(rule: dict[str, Any]) -> None:
    forbidden = {"sample_id", "gold_evidence", "gold_document", "expected_action", "required_claims", "forbidden_claims", "benchmark_label"}
    present = sorted(forbidden & set(rule))
    if present:
        raise ValueError(f"runtime classification rule contains forbidden gold/sample fields: {present}")


def compare_replicate_classifications(records_by_replicate: list[list[dict[str, Any]]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for records in records_by_replicate:
        for record in records:
            sample = str(record["boundary"].get("sample_id") or "unknown")
            by_sample.setdefault(sample, []).append(record)
    rows = []
    stable = 0
    unstable = 0
    for sample_id, records in sorted(by_sample.items()):
        classes = {record.get("runtime_failure_class") for record in records}
        final_actions = {record["validation"].get("final_action") for record in records}
        status = "stable_generation_refusal" if len(classes) == 1 and next(iter(classes)).startswith("refusal_") else "unstable_generation_outcome"
        if len(classes) == 1 and final_actions == {"answer"}:
            status = "stable_grounded_answer"
        elif any(str(record.get("runtime_failure_class", "")).startswith("provider_") for record in records):
            status = "infrastructure_affected"
        elif len(classes) == 1 and next(iter(classes)) in {"unclassified_generation_failure", "missing_observability_field"}:
            status = "stable_indeterminate"
        if len(classes) == 1:
            stable += 1
        else:
            unstable += 1
        rows.append({"sample_id": sample_id, "replicate_count": len(records), "classes": sorted(classes), "final_actions": sorted(final_actions), "cross_replicate_classification": status})
    total = stable + unstable
    return {
        "schema_version": "opk-rag.post-generation-replicate-classification-comparison.v1",
        "sample_count": total,
        "stable_classification_count": stable,
        "unstable_classification_count": unstable,
        "classification_agreement_rate": _ratio(stable, total),
        "rows": rows,
    }


def build_post_generation_decision_proposal() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.post-generation-recovery-decision.v1-proposal",
        "status": "proposal_only",
        "state_name": "POST_GENERATION_OUTCOME_INSPECTION",
        "default_action": "final_abstention",
        "inputs": ["generation_boundary_snapshot", "generation_outcome_snapshot", "post_generation_validation_snapshot"],
        "terminal_classes": ["answer_draft_grounded", "refusal_safety_terminal", "refusal_correct_unanswerable"],
        "retrieval_recovery_classes": ["refusal_claimed_insufficient_evidence", "refusal_claimed_missing_scope", "answer_draft_grounding_failure"],
        "generation_retry_classes": ["refusal_model_conservative", "generation_contract_abstention", "generation_empty_output"],
        "indeterminate_classes": ["insufficient_runtime_signal", "conflicting_runtime_signals", "missing_observability_field", "unclassified_generation_failure"],
        "rules": [
            {"class": "refusal_model_conservative", "allowed_future_action": "bounded_generation_retry", "budget_required": True},
            {"class": "refusal_claimed_missing_scope", "allowed_future_action": "bounded_retrieval_recovery", "budget_required": True},
        ],
        "runtime_gold_usage_allowed": False,
        "bypass_citation_or_grounding_allowed": False,
    }


def observability_contract_manifest(**identities: Any) -> dict[str, Any]:
    return {
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "schema_version": "opk-rag.post-generation-failure-observability-contract-manifest.v1",
        "behavior_modification_allowed": False,
        "generation_retry_allowed": False,
        "retrieval_retry_allowed": False,
        "promotion_evaluation": False,
        "instrumentation_points": ["before_generate_grounded_answer", "after_generate_grounded_answer", "after_validation_or_terminal"],
        "snapshot_schemas": [BOUNDARY_SNAPSHOT_SCHEMA, OUTCOME_SNAPSHOT_SCHEMA, VALIDATION_SNAPSHOT_SCHEMA],
        "sanitization_rules": ["no_raw_query_text", "no_raw_evidence_text", "no_full_prompt", "no_api_keys", "no_private_absolute_paths"],
        "outcome_taxonomy_version": "opk-rag.post-generation-outcome-taxonomy.v1",
        "offline_classification_rules": "gold_fields_allowed_only_in_offline_failure_interpretation",
        "diagnostic_run_mode": "diagnostic_shadow",
        "replicate_count": 2,
        "timeout_rules": {"behavior": "observe_existing_runtime_timeout_only", "new_timeout_added": False},
        "output_paths": {"default": "evaluation-data/results/task0073-post-generation-observability"},
        "privacy_policy": "privacy_safe_digests_counts_and_bounded_metadata_only",
        "verification_rules": ["complete_snapshots", "one_primary_runtime_failure_class", "no_runtime_gold_fields"],
        "identities": identities,
    }


def stable_record_json(record: dict[str, Any]) -> str:
    return stable_json(record)


def stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _response_mode(payload: dict[str, Any], failure: dict[str, Any] | None) -> str:
    if failure:
        return "error"
    if not payload:
        return "empty"
    status = payload.get("status")
    if status == "answered":
        return "answer"
    if status == "refused":
        return "abstain"
    return "malformed" if status is not None else "unknown"


def _transport_status(error_code: str | None, error_origin: str | None) -> str:
    if error_code is None:
        return "completed"
    if error_origin == "provider":
        return "provider_error"
    if error_origin == "infrastructure":
        return "infrastructure_error"
    return "error"


def _http_status_class(failure: dict[str, Any] | None) -> str:
    detail = (failure or {}).get("detail") or {}
    http_status = detail.get("http_status")
    if isinstance(http_status, int):
        return f"{http_status // 100}xx"
    return "not_applicable"


def _answer_length_bucket(payload: dict[str, Any]) -> str:
    if payload.get("status") != "answered":
        return "not_applicable"
    if not payload.get("answer_hash"):
        return "empty"
    return "present_redacted"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        denominator = int(denominator)
        numerator = int(numerator)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return numerator / denominator


def _complete(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: ("not_applicable" if value is None else value) for key, value in payload.items()}


def _missing_required(outcome: dict[str, Any]) -> bool:
    return any(outcome.get(key) in {None, "unknown"} for key in ("provider_call_attempted", "response_mode", "response_contract_valid"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in sorted(value.__dataclass_fields__)}
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
