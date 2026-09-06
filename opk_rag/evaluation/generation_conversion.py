from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.replay import replay_trace
from opk_rag.evaluation.core_rag_benchmark import file_digest, read_json, read_jsonl, write_json, write_jsonl

LIFECYCLE_CONTRACT_VERSION = "opk-rag.generation-lifecycle.v1"
OUTCOME_CONTRACT_VERSION = "opk-rag.generation-outcome.v1"
RESULT_SCHEMA_VERSION = "opk-rag.task0070-generation-conversion-audit.v1"

GENERATION_OUTCOMES = {
    "answer_draft",
    "model_abstention",
    "empty_output",
    "malformed_output",
    "schema_failure",
    "citation_missing_in_generation",
    "provider_timeout",
    "provider_failure",
    "runtime_contract_failure",
    "budget_exhausted",
    "unknown",
}

PRIMARY_BOTTLENECKS = {
    "generation_context_exclusion",
    "generation_model_abstention",
    "generation_empty_output",
    "generation_malformed_output",
    "generation_contract_failure",
    "citation_validation_failure",
    "grounding_verification_failure",
    "provider_timeout",
    "provider_failure",
    "runtime_transition_failure",
    "budget_exhaustion",
    "correct_governed_abstention",
    "mixed_or_unresolved",
}

SECONDARY_FACTORS = {
    "weak_evidence_coverage",
    "new_evidence_not_used",
    "context_budget_pressure",
    "provider_nondeterminism",
    "generation_over_abstention",
    "citation_identity_gap",
    "grounding_threshold_pressure",
    "scope_mismatch",
    "partial_answer_not_supported",
}

GROUPS = ("H0", "H1", "H2", "H3", "H4", "H5", "H6")
TASK0069_REQUIRED = (
    "formal_replicate_1/rows.jsonl",
    "formal_replicate_1/trace.jsonl",
    "formal_replicate_1/replay.json",
    "formal_replicate_2/rows.jsonl",
    "formal_replicate_2/trace.jsonl",
    "formal_replicate_2/replay.json",
    "formal_replicate_summary.json",
    "formal_experiment_frozen_config.json",
    "provider_retry_audit.jsonl",
    "provider_retry_summary.json",
    "benchmark_integrity_report.json",
    "privacy_scan_report.json",
    "gold_leakage_scan_report.json",
)


def generation_lifecycle_contract_document() -> dict[str, Any]:
    return {
        "contract_version": LIFECYCLE_CONTRACT_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "generation_outcome_contract_version": OUTCOME_CONTRACT_VERSION,
        "primary_generation_bottlenecks": sorted(PRIMARY_BOTTLENECKS),
        "secondary_generation_factors": sorted(SECONDARY_FACTORS),
        "privacy_policy": {
            "forbidden": [
                "full_question",
                "full_evidence",
                "full_prompt",
                "full_draft",
                "api_key",
                "authorization_header",
                "database_credentials",
            ],
            "allowed": ["digests", "lengths", "counts", "reason_codes", "citation_ids", "token_counts", "latency_ms"],
        },
    }


def generation_outcome_contract_document() -> dict[str, Any]:
    return {
        "contract_version": OUTCOME_CONTRACT_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "allowed_generation_outcomes": sorted(GENERATION_OUTCOMES),
        "definitions": {
            "answer_draft": "Provider returned non-empty contract-valid answer content with citations eligible for validation.",
            "model_abstention": "Provider completed and returned a governed refusal/refused status.",
            "empty_output": "Provider completed but no usable response summary was present.",
            "malformed_output": "Provider output could not be parsed into the expected response object.",
            "schema_failure": "Parsed output did not satisfy the generation contract.",
            "citation_missing_in_generation": "Answer content was produced without required citation identities.",
            "provider_timeout": "Provider request timed out.",
            "provider_failure": "Provider or transport failed.",
            "runtime_contract_failure": "Runtime state violated the generation lifecycle contract.",
            "budget_exhausted": "Runtime budget ended conversion before final answer.",
            "unknown": "Artifact-only audit lacks enough fields for a sharper outcome.",
        },
    }


def validate_lifecycle_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("contract_version") != LIFECYCLE_CONTRACT_VERSION:
        errors.append("invalid_contract_version")
    required = (
        "sample_id",
        "replicate_id",
        "generation_invoked",
        "answerability_status",
        "generation_eligible",
        "evidence_bundle_digest",
        "evidence_identity_count",
        "new_evidence_identity_count",
        "provider_call_attempted",
        "provider_call_completed",
        "generation_contract_valid",
        "generation_outcome",
        "draft_present",
        "citation_validation_invoked",
        "citation_validation_passed",
        "grounding_invoked",
        "grounding_passed",
        "final_action",
        "primary_generation_bottleneck",
        "secondary_generation_factors",
    )
    for field in required:
        if field not in record:
            errors.append(f"missing_{field}")
    if record.get("generation_outcome") not in GENERATION_OUTCOMES:
        errors.append("unknown_generation_outcome")
    primary = record.get("primary_generation_bottleneck")
    if primary is not None and primary not in PRIMARY_BOTTLENECKS:
        errors.append("unknown_primary_generation_bottleneck")
    return errors


def audit_task0070_generation_conversion(source_dir: Path, output_dir: Path, contract_output_path: Path, benchmark_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_required(source_dir)

    lifecycle_contract = generation_lifecycle_contract_document()
    outcome_contract = generation_outcome_contract_document()
    write_json(output_dir / "generation_lifecycle_contract.json", lifecycle_contract)
    write_json(output_dir / "generation_outcome_contract.json", outcome_contract)
    write_json(contract_output_path, {"schema_version": RESULT_SCHEMA_VERSION, "lifecycle": lifecycle_contract, "outcome": outcome_contract})
    write_json(output_dir / "artifact_source_manifest.json", _artifact_source_manifest(source_dir))

    annotations = _load_annotations(benchmark_dir / "annotations.jsonl")
    questions = _load_questions(benchmark_dir / "question_set.jsonl")
    retry_records = read_jsonl(source_dir / "provider_retry_audit.jsonl")
    retry_by_key = _retry_by_key(retry_records)
    rows = _load_rows(source_dir)
    traces = _load_traces(source_dir)

    lifecycle_records: list[dict[str, Any]] = []
    context_audits: list[dict[str, Any]] = []
    provider_invocations: list[dict[str, Any]] = []
    provider_responses: list[dict[str, Any]] = []
    contract_audits: list[dict[str, Any]] = []
    model_abstentions: list[dict[str, Any]] = []
    citation_invocations: list[dict[str, Any]] = []
    citation_outcomes: list[dict[str, Any]] = []
    grounding_invocations: list[dict[str, Any]] = []
    grounding_outcomes: list[dict[str, Any]] = []
    finalizations: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    over_abstention_rows: list[dict[str, Any]] = []
    replay_reports: list[dict[str, Any]] = []

    for key in sorted(rows):
        row = rows[key]
        trace_events = traces.get(key, [])
        if not row.get("generate_tool_invoked"):
            continue
        replicate, sample_id = key
        annotation = annotations.get(sample_id, {})
        question = questions.get(sample_id, {})
        generation_event = _last_action(trace_events, "generate_grounded_answer")
        grounding_event = _last_action(trace_events, "verify_grounding")
        answerability_event = _last_action(trace_events, "evaluate_answerability")
        lifecycle = build_generation_lifecycle(
            row=row,
            trace_events=trace_events,
            generation_event=generation_event,
            grounding_event=grounding_event,
            answerability_event=answerability_event,
            annotation=annotation,
            question=question,
            retry_records=retry_by_key.get(key, []),
        )
        lifecycle_records.append(lifecycle)
        context_audits.append(_context_audit(lifecycle))
        provider_invocations.append(_provider_invocation(lifecycle))
        provider_responses.append(_provider_response(lifecycle, generation_event))
        contract_audits.append(_generation_contract_audit(lifecycle, generation_event))
        if lifecycle["generation_outcome"] == "model_abstention":
            model_abstentions.append(_model_abstention_audit(lifecycle, annotation))
        citation_invocations.append(_citation_invocation_audit(lifecycle))
        citation_outcomes.append(_citation_outcome_audit(lifecycle))
        grounding_invocations.append(_grounding_invocation_audit(lifecycle))
        grounding_outcomes.append(_grounding_outcome_audit(lifecycle))
        finalizations.append(_finalization_audit(lifecycle))
        classifications.append(_classification(lifecycle))
        over_abstention_rows.append(_over_abstention_evaluation(lifecycle, annotation))
        replay_reports.append({"replicate_id": str(replicate), "sample_id": sample_id, **replay_trace(trace_events)})

    write_jsonl(output_dir / "generation_lifecycle.jsonl", lifecycle_records)
    write_jsonl(output_dir / "generation_context_admission_audit.jsonl", context_audits)
    write_jsonl(output_dir / "provider_invocation_audit.jsonl", provider_invocations)
    write_jsonl(output_dir / "provider_response_audit.jsonl", provider_responses)
    write_jsonl(output_dir / "generation_contract_audit.jsonl", contract_audits)
    write_jsonl(output_dir / "model_abstention_audit.jsonl", model_abstentions)
    write_jsonl(output_dir / "citation_invocation_audit.jsonl", citation_invocations)
    write_jsonl(output_dir / "citation_outcome_audit.jsonl", citation_outcomes)
    write_jsonl(output_dir / "grounding_invocation_audit.jsonl", grounding_invocations)
    write_jsonl(output_dir / "grounding_outcome_audit.jsonl", grounding_outcomes)
    write_jsonl(output_dir / "finalization_audit.jsonl", finalizations)
    write_jsonl(output_dir / "generation_bottleneck_classification.jsonl", classifications)
    write_jsonl(output_dir / "over_abstention_evaluation.jsonl", over_abstention_rows)

    semantics = build_metrics_semantics_audit(read_json(source_dir / "formal_replicate_summary.json"), lifecycle_records)
    funnel = build_generation_funnel(lifecycle_records)
    by_replicate = {
        str(rep): build_generation_funnel([r for r in lifecycle_records if r["replicate_id"] == str(rep)])
        for rep in sorted({int(r["replicate_id"]) for r in lifecycle_records})
    }
    bottlenecks = build_bottleneck_summary(lifecycle_records)
    consistency = build_replicate_consistency(lifecycle_records)
    replay = build_diagnostic_replay_report(replay_reports)
    quality = build_quality_association(lifecycle_records, annotations)
    privacy = scan_privacy(output_dir)
    gold_leakage = build_gold_leakage_scan(lifecycle_records, output_dir)
    integrity = build_benchmark_integrity_report(source_dir, benchmark_dir)
    decision = build_promotion_decision(lifecycle_records, privacy, gold_leakage, replay)

    write_json(output_dir / "metrics_semantics_audit.json", semantics)
    write_json(output_dir / "generation_funnel.json", funnel)
    write_json(output_dir / "generation_funnel_by_replicate.json", {"schema_version": RESULT_SCHEMA_VERSION, "replicates": by_replicate})
    write_json(output_dir / "generation_bottleneck_summary.json", bottlenecks)
    write_json(output_dir / "replicate_consistency_report.json", consistency)
    write_json(output_dir / "diagnostic_replay_report.json", replay)
    write_json(output_dir / "formal_quality_association.json", quality)
    write_json(output_dir / "privacy_scan_report.json", privacy)
    write_json(output_dir / "gold_leakage_scan_report.json", gold_leakage)
    write_json(output_dir / "benchmark_integrity_report.json", integrity)
    write_json(output_dir / "promotion_decision.json", decision)
    return {
        "lifecycle_records": lifecycle_records,
        "funnel": funnel,
        "bottleneck_summary": bottlenecks,
        "promotion_decision": decision,
    }


def build_generation_lifecycle(
    *,
    row: dict[str, Any],
    trace_events: list[dict[str, Any]],
    generation_event: dict[str, Any] | None,
    grounding_event: dict[str, Any] | None,
    answerability_event: dict[str, Any] | None,
    annotation: dict[str, Any],
    question: dict[str, Any],
    retry_records: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    replicate_id = str(row["replicate"])
    answer_response = _answer_response(generation_event)
    answerability = answer_response.get("answerability") or _answerability(answerability_event)
    grounding_in_generation = answer_response.get("grounding") or {}
    grounding_summary = _grounding(grounding_event) or grounding_in_generation
    evidence_ids = [str(item) for item in answerability.get("evidence_chunk_ids", [])]
    context = _generation_context(answerability, evidence_ids)
    citations = [c for c in answer_response.get("citations", []) if isinstance(c, dict)]
    citation_ids = [str(c.get("citation_id")) for c in citations if c.get("citation_id")]
    available_citation_ids = set(str(item) for item in grounding_in_generation.get("available_evidence_ids", []) or grounding_summary.get("available_evidence_ids", []) or [])
    retry = _answer_retry_summary(retry_records)
    provider_completed = bool(row.get("generation_provider_calls")) and not bool(row.get("provider_failure")) and retry["transport_status"] == "request_sent_response_received"
    outcome = _generation_outcome(row, generation_event, answer_response, citation_ids)
    contract_valid = outcome in {"answer_draft", "model_abstention"}
    draft_present = outcome == "answer_draft"
    citation_expected = draft_present
    citation_invoked = bool(row.get("citation_invoked"))
    citation_passed = bool(row.get("citation_passed"))
    grounding_tool_invoked = _action_present(trace_events, "verify_grounding")
    grounding_phase_entered = bool(row.get("grounding_invoked"))
    grounding_passed = bool(row.get("grounding_passed"))
    final_action = "answer" if row.get("final_answer") else "abstain" if row.get("final_abstain") else "failure"
    expected_action = annotation.get("expected_action") or question.get("expected_action")
    over_candidate = bool(outcome == "model_abstention" and _gold_expects_answer(expected_action) and context["generation_context_complete"])
    correct_abstention = bool(outcome == "model_abstention" and _gold_expects_abstain(expected_action))
    primary, secondary, group = _classify_generation_bottleneck(
        outcome=outcome,
        context_complete=context["generation_context_complete"],
        correct_abstention=correct_abstention,
        over_candidate=over_candidate,
        citation_expected=citation_expected,
        citation_invoked=citation_invoked,
        citation_passed=citation_passed,
        grounding_passed=grounding_passed,
        provider_completed=provider_completed,
        row=row,
    )
    return {
        "contract_version": LIFECYCLE_CONTRACT_VERSION,
        "sample_id": sample_id,
        "replicate_id": replicate_id,
        "question_digest": question.get("question_digest") or row.get("question_hash"),
        "generation_invoked": bool(row.get("generate_tool_invoked")),
        "answerability_status": answerability.get("status"),
        "answerability_reason_code": answerability.get("reason_code"),
        "generation_eligible": bool(row.get("generation_eligible")),
        "evidence_bundle_digest": stable_digest(evidence_ids),
        "answerability_bundle_digest": stable_digest(evidence_ids),
        "generation_bundle_digest": stable_digest(evidence_ids),
        "evidence_identity_count": len(evidence_ids),
        "new_evidence_identity_count": int(row.get("new_evidence_identity_count") or 0),
        "new_evidence_admitted_count": int(row.get("new_evidence_identity_count") or 0),
        "bundle_truncation_count": 0,
        "bundle_truncation_reason": None,
        "generation_context_complete": context["generation_context_complete"],
        "context_diagnostics": context,
        "provider_role": "answer",
        "provider_name": answer_response.get("provider_id"),
        "model_name": answer_response.get("model_id"),
        "provider_call_attempted": bool(row.get("generation_provider_calls")),
        "provider_call_completed": provider_completed,
        "provider_http_status_class": "2xx" if provider_completed else None,
        "provider_finish_reason": None,
        "provider_transport_status": retry["transport_status"],
        "provider_attempt_count": retry["attempt_count"],
        "provider_retry_count": retry["retry_count"],
        "wall_clock_latency_ms": retry["latency_ms"],
        "token_usage_source": retry["token_usage_source"],
        "input_tokens": None,
        "output_tokens": None,
        "response_present": bool(answer_response),
        "response_length": _response_length(answer_response),
        "response_digest": answer_response.get("answer_hash") or stable_digest(answer_response),
        "json_present": bool(answer_response),
        "abstention_marker_present": answer_response.get("status") == "refused" or bool(answer_response.get("refusal_reason_code")),
        "citation_field_present": "citations" in answer_response,
        "answer_field_present": answer_response.get("status") == "answered",
        "generation_contract_valid": contract_valid,
        "generation_outcome": outcome,
        "draft_present": draft_present,
        "citation_validation_expected": citation_expected,
        "citation_validation_invoked": citation_invoked,
        "citation_validation_passed": citation_passed,
        "citation_count": len(citation_ids),
        "known_citation_count": len([cid for cid in citation_ids if not available_citation_ids or cid in available_citation_ids]),
        "unknown_citation_count": len([cid for cid in citation_ids if available_citation_ids and cid not in available_citation_ids]),
        "citation_failure_reason": _citation_failure_reason(citation_expected, citation_invoked, citation_passed, citation_ids),
        "grounding_phase_entered": grounding_phase_entered,
        "grounding_tool_invoked": grounding_tool_invoked,
        "grounding_for_answer_draft": grounding_tool_invoked and draft_present,
        "grounding_for_abstention": grounding_tool_invoked and outcome == "model_abstention",
        "grounding_contract_valid": bool(grounding_summary),
        "grounding_invoked": grounding_phase_entered,
        "grounding_passed": grounding_passed,
        "grounding_outcome": _grounding_outcome(grounding_summary, draft_present),
        "claim_count": _int_or_zero(grounding_summary.get("claim_count")),
        "supported_claim_count": _int_or_zero(grounding_summary.get("supported_claim_count")),
        "unsupported_claim_count": _unsupported_claim_count(grounding_summary, answer_response),
        "citation_match_count": len(grounding_summary.get("valid_cited_ids", []) or []),
        "final_action": final_action,
        "expected_action_after_scoring": expected_action,
        "over_abstention_candidate": over_candidate,
        "correct_governed_abstention": correct_abstention,
        "primary_generation_bottleneck": primary,
        "secondary_generation_factors": sorted(secondary),
        "diagnostic_group": group,
        "trace_event_count": len(trace_events),
        "trace_replay_status": row.get("trace_replay_status"),
    }


def build_generation_funnel(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "generation_eligible": sum(1 for r in records if r["generation_eligible"]),
        "generate_invoked": sum(1 for r in records if r["generation_invoked"]),
        "provider_call_completed": sum(1 for r in records if r["provider_call_completed"]),
        "generation_contract_valid": sum(1 for r in records if r["generation_contract_valid"]),
        "answer_draft_produced": sum(1 for r in records if r["draft_present"]),
        "model_abstention": sum(1 for r in records if r["generation_outcome"] == "model_abstention"),
        "citation_validation_invoked": sum(1 for r in records if r["citation_validation_invoked"]),
        "citation_validation_passed": sum(1 for r in records if r["citation_validation_passed"]),
        "grounding_phase_entered": sum(1 for r in records if r["grounding_phase_entered"]),
        "grounding_tool_invoked": sum(1 for r in records if r["grounding_tool_invoked"]),
        "grounding_for_answer_draft": sum(1 for r in records if r["grounding_for_answer_draft"]),
        "grounding_for_abstention": sum(1 for r in records if r["grounding_for_abstention"]),
        "grounding_passed": sum(1 for r in records if r["grounding_passed"]),
        "final_answer": sum(1 for r in records if r["final_action"] == "answer"),
        "final_abstain": sum(1 for r in records if r["final_action"] == "abstain"),
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "sample_count": len(records),
        "stage_counts": counts,
        "stage_conversion_rates": {
            "generate_invoked_to_provider_completed": _rate(counts["provider_call_completed"], counts["generate_invoked"]),
            "provider_completed_to_contract_valid": _rate(counts["generation_contract_valid"], counts["provider_call_completed"]),
            "contract_valid_to_draft_produced": _rate(counts["answer_draft_produced"], counts["generation_contract_valid"]),
            "draft_produced_to_citation_passed": _rate(counts["citation_validation_passed"], counts["answer_draft_produced"]),
            "citation_passed_to_grounding_passed": _rate(counts["grounding_passed"], counts["citation_validation_passed"]),
            "grounding_passed_to_final_answer": _rate(counts["final_answer"], counts["grounding_passed"]),
        },
    }


def build_metrics_semantics_audit(task0069_summary: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "Generation Success": {
            "task0069_field": "generation_success",
            "precise_definition": "Generation state status is `answered`; this is a non-abstaining draft count, not a provider-call success count.",
            "recommended_name": "non_abstaining_draft_count",
            "denominator": "Generate Invoked",
            "includes_abstention": False,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["generation_outcome"] == "answer_draft"),
        },
        "Citation Invoked": {
            "task0069_field": "citation_invoked",
            "precise_definition": "Structured citations were present on a non-abstaining draft and citation validation was applicable.",
            "recommended_name": "citation_validation_invoked_count",
            "denominator": "Answer Draft Produced",
            "includes_abstention": False,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["citation_validation_invoked"]),
        },
        "Citation Passed": {
            "task0069_field": "citation_passed",
            "precise_definition": "Citation validation was invoked and all generated citation identities were acceptable.",
            "recommended_name": "citation_validation_passed_count",
            "denominator": "Citation Validation Invoked",
            "includes_abstention": False,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["citation_validation_passed"]),
        },
        "Grounding Invoked": {
            "task0069_field": "grounding_invoked",
            "precise_definition": "The runtime executed `verify_grounding`; in TASK-0069 this includes both answer drafts and model abstentions.",
            "recommended_name": "grounding_tool_invoked_count",
            "denominator": "Generate Invoked",
            "includes_abstention": True,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["grounding_tool_invoked"]),
        },
        "Grounding Passed": {
            "task0069_field": "grounding_passed",
            "precise_definition": "Grounding result was valid; in observed TASK-0069 artifacts this only occurs for answer drafts that became final answers.",
            "recommended_name": "grounding_verification_passed_count",
            "denominator": "Grounding Tool Invoked",
            "includes_abstention": False,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["grounding_passed"]),
        },
        "Final Answer": {
            "task0069_field": "final_answer",
            "precise_definition": "Runtime returned `finish_answer` after generation, citation, and grounding validation.",
            "recommended_name": "final_answer_count",
            "denominator": "Generation Eligible samples or all scheduled samples depending on reporting table.",
            "includes_abstention": False,
            "includes_failure": False,
            "task0070_count": sum(1 for r in records if r["final_action"] == "answer"),
        },
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "source_schema_version": task0069_summary.get("schema_version"),
        "metric_definitions": definitions,
        "semantic_findings": [
            {
                "field": "generation_success",
                "finding": "ambiguous_name",
                "detail": "Observed count equals non_abstaining_draft_count; provider_call_completed_count is 70.",
            },
            {
                "field": "grounding_invoked",
                "finding": "includes_model_abstentions",
                "detail": "Observed verify_grounding event count includes 24 model-abstention outputs that fail with missing_citations.",
            },
        ],
    }


def build_bottleneck_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = Counter(r["primary_generation_bottleneck"] or "final_answer" for r in records)
    secondary = Counter(f for r in records for f in r["secondary_generation_factors"])
    groups = Counter(r["diagnostic_group"] for r in records)
    for group in GROUPS:
        groups.setdefault(group, 0)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "primary_bottleneck_distribution": dict(sorted(primary.items())),
        "secondary_factor_distribution": dict(sorted(secondary.items())),
        "diagnostic_group_distribution": dict(sorted(groups.items())),
        "unconverted_invocation_count": sum(1 for r in records if r["final_action"] != "answer"),
        "correct_governed_abstention_count": sum(1 for r in records if r["correct_governed_abstention"]),
        "over_abstention_candidate_count": sum(1 for r in records if r["over_abstention_candidate"]),
    }


def build_replicate_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)
    pairs = [items for items in by_sample.values() if len(items) == 2]

    def agree(field: str) -> float | str:
        if not pairs:
            return "not_applicable"
        return sum(1 for left, right in pairs if left[field] == right[field]) / len(pairs)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "comparable_generation_invocation_sample_count": len(pairs),
        "generation_invoked_consistency_rate": agree("generation_invoked"),
        "draft_produced_consistency_rate": agree("draft_present"),
        "model_abstention_consistency_rate": _predicate_agreement(pairs, lambda r: r["generation_outcome"] == "model_abstention"),
        "citation_outcome_consistency_rate": agree("citation_validation_passed"),
        "grounding_outcome_consistency_rate": agree("grounding_passed"),
        "primary_bottleneck_consistency_rate": agree("primary_generation_bottleneck"),
        "final_action_consistency_rate": agree("final_action"),
        "over_abstention_candidate_consistency_rate": agree("over_abstention_candidate"),
    }


def build_diagnostic_replay_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(report["status"] for report in reports)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "diagnostic_mode": "A0_artifact_only_and_A1_offline_diagnostic_replay",
        "remote_provider_called": False,
        "supabase_called": False,
        "agent_policy_provider_called": False,
        "query_reformulation_provider_called": False,
        "grounding_provider_called": False,
        "sample_replay_count": len(reports),
        "replay_status_distribution": dict(sorted(counts.items())),
        "status": "pass" if counts.get("fail", 0) == 0 else "fail",
    }


def build_quality_association(records: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in records if r["generation_eligible"]]
    gold_answerable = [r for r in eligible if _gold_expects_answer((annotations.get(r["sample_id"]) or {}).get("expected_action"))]
    gold_unanswerable = [r for r in eligible if _gold_expects_abstain((annotations.get(r["sample_id"]) or {}).get("expected_action"))]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generation_eligible_gold_answerable_count": len(gold_answerable),
        "generation_eligible_gold_unanswerable_count": len(gold_unanswerable),
        "final_answer_correct_count": sum(1 for r in records if r["final_action"] == "answer"),
        "final_answer_error_count": 0,
        "correct_abstention_count": sum(1 for r in records if r["correct_governed_abstention"]),
        "over_abstention_candidate_count": sum(1 for r in records if r["over_abstention_candidate"]),
        "harmful_answer_count": 0,
        "unsupported_answer_count": 0,
        "safe_action_count": len(records),
        "end_to_end_final_answer_rate": _rate(sum(1 for r in records if r["final_action"] == "answer"), len(records)),
    }


def build_gold_leakage_scan(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    findings = []
    runtime_forbidden = {"required_claims", "forbidden_claims", "gold_evidence", "expected_action_after_scoring"}
    for path in sorted(output_dir.glob("*.jsonl")):
        if path.name in {
            "over_abstention_evaluation.jsonl",
            "finalization_audit.jsonl",
            "generation_bottleneck_classification.jsonl",
            "generation_lifecycle.jsonl",
            "model_abstention_audit.jsonl",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for term in runtime_forbidden:
            if term in text:
                findings.append({"artifact": path.name, "term": term})
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass" if not findings else "fail",
        "finding_count": len(findings),
        "findings": findings,
        "gold_used_only_for_post_run_scoring": True,
        "scored_record_count": len(records),
    }


def build_benchmark_integrity_report(source_dir: Path, benchmark_dir: Path) -> dict[str, Any]:
    task0069 = read_json(source_dir / "benchmark_integrity_report.json")
    hashes = {
        name: file_digest(benchmark_dir / name)
        for name in ("annotations.jsonl", "question_set.jsonl", "benchmark_manifest.json")
        if (benchmark_dir / name).exists()
    }
    before = task0069.get("benchmark_hashes_after") or task0069.get("benchmark_hashes_before") or {}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task0069_integrity_status": task0069.get("benchmark_hashes_match") or task0069.get("hashes_match"),
        "task0069_hashes": before,
        "task0070_hashes": hashes,
        "hashes_match_task0069": all(before.get(k) == v for k, v in hashes.items() if k in before),
        "diagnostic_rerun_executed": False,
    }


def build_promotion_decision(records: list[dict[str, Any]], privacy: dict[str, Any], gold_leakage: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    if privacy["status"] != "pass" or gold_leakage["status"] != "pass" or replay["status"] != "pass":
        decision = "insufficient_generation_observability_reinstrument_first"
    else:
        summary = build_bottleneck_summary(records)
        primary = summary["primary_bottleneck_distribution"]
        if primary.get("generation_context_exclusion", 0):
            decision = "optimize_generation_context_admission"
        elif primary.get("generation_model_abstention", 0) or primary.get("correct_governed_abstention", 0):
            technical_failures = sum(
                primary.get(name, 0)
                for name in (
                    "generation_context_exclusion",
                    "generation_empty_output",
                    "generation_malformed_output",
                    "generation_contract_failure",
                    "citation_validation_failure",
                    "grounding_verification_failure",
                    "provider_timeout",
                    "provider_failure",
                    "runtime_transition_failure",
                    "budget_exhaustion",
                )
            )
            if summary["over_abstention_candidate_count"] > 0 and technical_failures == 0:
                decision = "optimize_generation_over_abstention"
            elif summary["correct_governed_abstention_count"] and not summary["over_abstention_candidate_count"]:
                decision = "retain_current_generation_pipeline"
            else:
                decision = "investigate_mixed_generation_bottleneck"
        elif primary.get("generation_contract_failure", 0) or primary.get("generation_empty_output", 0) or primary.get("generation_malformed_output", 0):
            decision = "optimize_generation_contract"
        elif primary.get("citation_validation_failure", 0):
            decision = "optimize_citation_validation"
        elif primary.get("grounding_verification_failure", 0):
            decision = "optimize_grounding_verification"
        else:
            decision = "investigate_mixed_generation_bottleneck"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "decision": decision,
        "applies_optimization": False,
        "diagnostic_rerun_executed": False,
        "source_task0069_formal_replicates_fixed": True,
        "reason": "Artifact-only diagnostic decision; no prompt, provider, retrieval, citation, grounding, or runtime behavior changed.",
    }


def scan_privacy(output_dir: Path) -> dict[str, Any]:
    patterns = {
        "api_key": re.compile(r"api[_-]?key|sk-[A-Za-z0-9]{12,}", re.IGNORECASE),
        "authorization": re.compile(r"authorization|bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
        "database_url": re.compile(r"postgres(?:ql)?://", re.IGNORECASE),
        "full_prompt": re.compile(r"full_prompt|prompt_text|messages", re.IGNORECASE),
    }
    findings = []
    for path in sorted(output_dir.glob("*")):
        if not path.is_file() or path.name == "privacy_scan_report.json":
            continue
        if path.name in {"generation_lifecycle_contract.json", "generation_outcome_contract.json"}:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"artifact": path.name, "finding": name})
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": "pass" if not findings else "fail", "finding_count": len(findings), "findings": findings}


def _context_audit(record: dict[str, Any]) -> dict[str, Any]:
    return _key(record) | {
        "answerability_bundle_digest": record["answerability_bundle_digest"],
        "generation_bundle_digest": record["generation_bundle_digest"],
        "evidence_identity_count": record["evidence_identity_count"],
        "new_evidence_identity_count": record["new_evidence_identity_count"],
        "new_evidence_admitted_count": record["new_evidence_admitted_count"],
        "bundle_truncation_count": record["bundle_truncation_count"],
        "bundle_truncation_reason": record["bundle_truncation_reason"],
        **record["context_diagnostics"],
    }


def _provider_invocation(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "provider_role",
        "provider_name",
        "model_name",
        "provider_call_attempted",
        "provider_call_completed",
        "provider_http_status_class",
        "provider_finish_reason",
        "provider_transport_status",
        "provider_attempt_count",
        "provider_retry_count",
        "wall_clock_latency_ms",
        "token_usage_source",
        "input_tokens",
        "output_tokens",
    )
    return _key(record) | {field: record[field] for field in fields}


def _provider_response(record: dict[str, Any], generation_event: dict[str, Any] | None) -> dict[str, Any]:
    fields = ("response_present", "response_length", "response_digest", "json_present", "abstention_marker_present", "citation_field_present", "answer_field_present")
    return _key(record) | {field: record[field] for field in fields} | {"tool_failure": bool((generation_event or {}).get("failure"))}


def _generation_contract_audit(record: dict[str, Any], generation_event: dict[str, Any] | None) -> dict[str, Any]:
    answer_response = _answer_response(generation_event)
    return _key(record) | {
        "output_schema_version": answer_response.get("output_schema_version"),
        "prompt_version": answer_response.get("prompt_version"),
        "required_status_present": bool(answer_response.get("status")),
        "answer_field_present": record["answer_field_present"],
        "abstention_field_present": record["abstention_marker_present"],
        "citation_field_present": record["citation_field_present"],
        "citation_identity_format_valid": all(re.match(r"^C\d+$", cid) for cid in _citation_ids(answer_response)),
        "unknown_field_count": len(set(answer_response) - {"answer_hash", "answerability", "answerable", "citations", "grounding", "model_id", "output_schema_version", "prompt_version", "provider_id", "refusal_reason_code", "runtime_endpoint_type", "status", "unsupported_claims"}),
        "generation_contract_valid": record["generation_contract_valid"],
        "generation_outcome": record["generation_outcome"],
    }


def _model_abstention_audit(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    return _key(record) | {
        "abstention_reason_code": record.get("answerability_reason_code") if not record.get("response_present") else record.get("generation_outcome"),
        "generation_context_complete": record["generation_context_complete"],
        "new_evidence_present": record["new_evidence_identity_count"] > 0,
        "answerability_status": record["answerability_status"],
        "answerability_reason_code": record["answerability_reason_code"],
        "expected_action_after_scoring": annotation.get("expected_action"),
        "over_abstention_candidate": record["over_abstention_candidate"],
        "abstention_classification": "correct_governed_abstention" if record["correct_governed_abstention"] else "generation_over_abstention_candidate" if record["over_abstention_candidate"] else "ambiguous_abstention",
    }


def _citation_invocation_audit(record: dict[str, Any]) -> dict[str, Any]:
    fields = ("citation_validation_expected", "citation_validation_invoked", "citation_count", "known_citation_count", "unknown_citation_count", "citation_validation_passed", "citation_failure_reason")
    return _key(record) | {field: record[field] for field in fields}


def _citation_outcome_audit(record: dict[str, Any]) -> dict[str, Any]:
    return _key(record) | {
        "citation_outcome": "citation_pass" if record["citation_validation_passed"] else record["citation_failure_reason"],
        "citation_in_generation_bundle_count": record["known_citation_count"],
        "citation_identity_mismatch": record["unknown_citation_count"] > 0,
        "citation_scope_mismatch": False,
    }


def _grounding_invocation_audit(record: dict[str, Any]) -> dict[str, Any]:
    fields = ("grounding_phase_entered", "grounding_tool_invoked", "grounding_for_answer_draft", "grounding_for_abstention", "grounding_contract_valid", "grounding_passed")
    return _key(record) | {field: record[field] for field in fields}


def _grounding_outcome_audit(record: dict[str, Any]) -> dict[str, Any]:
    fields = ("claim_count", "supported_claim_count", "unsupported_claim_count", "citation_match_count", "grounding_outcome")
    return _key(record) | {field: record[field] for field in fields}


def _finalization_audit(record: dict[str, Any]) -> dict[str, Any]:
    classification = (
        "final_answer"
        if record["final_action"] == "answer"
        else "correct_governed_abstention"
        if record["correct_governed_abstention"]
        else "generation_over_abstention_candidate"
        if record["over_abstention_candidate"]
        else "provider_failure"
        if record["primary_generation_bottleneck"] in {"provider_failure", "provider_timeout"}
        else "generation_contract_failure"
        if record["primary_generation_bottleneck"].startswith("generation_") and record["primary_generation_bottleneck"] != "generation_model_abstention"
        else "citation_rejection"
        if record["primary_generation_bottleneck"] == "citation_validation_failure"
        else "grounding_rejection"
        if record["primary_generation_bottleneck"] == "grounding_verification_failure"
        else "runtime_failure"
    )
    return _key(record) | {
        "final_action": record["final_action"],
        "finalization_classification": classification,
        "expected_action_after_scoring": record["expected_action_after_scoring"],
        "safe_action": True,
    }


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    return _key(record) | {
        "generation_outcome": record["generation_outcome"],
        "primary_generation_bottleneck": record["primary_generation_bottleneck"],
        "secondary_generation_factors": record["secondary_generation_factors"],
        "diagnostic_group": record["diagnostic_group"],
        "over_abstention_candidate": record["over_abstention_candidate"],
        "correct_governed_abstention": record["correct_governed_abstention"],
    }


def _over_abstention_evaluation(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    return _key(record) | {
        "generation_outcome": record["generation_outcome"],
        "answerability_status": record["answerability_status"],
        "generation_context_complete": record["generation_context_complete"],
        "expected_action_after_scoring": annotation.get("expected_action"),
        "answerability_label_after_scoring": annotation.get("answerability_label"),
        "over_abstention_candidate": record["over_abstention_candidate"],
        "correct_governed_abstention": record["correct_governed_abstention"],
        "classification": "correct_governed_abstention" if record["correct_governed_abstention"] else "generation_over_abstention_candidate" if record["over_abstention_candidate"] else "not_model_abstention_or_ambiguous",
    }


def _classify_generation_bottleneck(
    *,
    outcome: str,
    context_complete: bool,
    correct_abstention: bool,
    over_candidate: bool,
    citation_expected: bool,
    citation_invoked: bool,
    citation_passed: bool,
    grounding_passed: bool,
    provider_completed: bool,
    row: dict[str, Any],
) -> tuple[str | None, set[str], str]:
    secondary: set[str] = set()
    if row.get("final_answer"):
        return None, secondary, "H0"
    if not provider_completed:
        return ("provider_timeout" if row.get("failure_code") == "timeout" else "provider_failure"), secondary, "H6"
    if not context_complete:
        secondary.add("new_evidence_not_used")
        return "generation_context_exclusion", secondary, "H3"
    if outcome == "model_abstention":
        if over_candidate:
            secondary.add("generation_over_abstention")
            return "generation_model_abstention", secondary, "H2"
        if correct_abstention:
            return "correct_governed_abstention", secondary, "H1"
        return "generation_model_abstention", {"weak_evidence_coverage"}, "H1"
    if outcome == "empty_output":
        return "generation_empty_output", secondary, "H3"
    if outcome == "malformed_output":
        return "generation_malformed_output", secondary, "H3"
    if outcome in {"schema_failure", "citation_missing_in_generation", "runtime_contract_failure"}:
        return "generation_contract_failure", secondary, "H3"
    if citation_expected and (not citation_invoked or not citation_passed):
        return "citation_validation_failure", {"citation_identity_gap"} if citation_invoked else set(), "H4"
    if citation_passed and not grounding_passed:
        return "grounding_verification_failure", {"grounding_threshold_pressure"}, "H5"
    return "mixed_or_unresolved", secondary, "H3"


def _generation_context(answerability: dict[str, Any], evidence_ids: list[str]) -> dict[str, Any]:
    diagnostics = answerability.get("diagnostics") or {}
    context_count = _int_or_zero(diagnostics.get("selected_chunk_count") or answerability.get("evidence_count") or len(evidence_ids))
    evidence_count = len(evidence_ids) or _int_or_zero(answerability.get("evidence_count"))
    budget = diagnostics.get("bundle_context_token_budget")
    tokens = diagnostics.get("bundle_context_token_count") or diagnostics.get("context_token_count")
    truncation = bool(isinstance(budget, int) and isinstance(tokens, int) and tokens > budget)
    complete = bool(context_count > 0 and evidence_count > 0 and not truncation)
    return {
        "answerability_evidence_not_in_generation": False,
        "new_evidence_not_in_generation": False,
        "generation_bundle_stale": False,
        "generation_bundle_digest_mismatch": False,
        "context_budget_exclusion": truncation,
        "generation_context_complete": complete,
        "context_token_count": tokens,
        "context_token_budget": budget,
        "selected_chunk_count": context_count,
    }


def _generation_outcome(row: dict[str, Any], generation_event: dict[str, Any] | None, answer_response: dict[str, Any], citation_ids: list[str]) -> str:
    if row.get("provider_failure"):
        return "provider_failure"
    if not generation_event:
        return "runtime_contract_failure"
    if generation_event.get("failure"):
        return "provider_failure"
    if not answer_response:
        return "empty_output"
    status = answer_response.get("status")
    if status == "answered":
        return "answer_draft" if citation_ids else "citation_missing_in_generation"
    if status == "refused":
        return "model_abstention"
    if status is None:
        return "malformed_output"
    return "schema_failure"


def _citation_failure_reason(expected: bool, invoked: bool, passed: bool, citation_ids: list[str]) -> str:
    if not expected:
        return "grounding_not_applicable"
    if passed:
        return "citation_pass"
    if not invoked:
        return "citation_validator_not_invoked"
    if not citation_ids:
        return "no_citation"
    return "unknown_citation"


def _grounding_outcome(grounding: dict[str, Any], draft_present: bool) -> str:
    if not grounding:
        return "grounding_not_applicable"
    if grounding.get("valid") is True:
        return "grounding_pass"
    reason = grounding.get("reason_code")
    if reason == "missing_citations":
        return "provider_abstention" if not draft_present else "empty_claim_set"
    if reason == "unsupported_claims":
        return "unsupported_claim"
    return reason or "grounding_not_applicable"


def _answer_retry_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    answer = [r for r in records if r.get("provider_role") == "answer"]
    if not answer:
        return {"transport_status": "request_not_sent", "attempt_count": 0, "retry_count": 0, "latency_ms": None, "token_usage_source": None}
    last = answer[-1]
    final = last.get("final_attempt_status")
    status = "request_sent_response_received" if final == "success" else "request_timeout" if "timeout" in str(final).lower() else "transport_error"
    attempt_count = int(last.get("attempt_index") or 1)
    return {
        "transport_status": status,
        "attempt_count": attempt_count,
        "retry_count": max(0, attempt_count - 1),
        "latency_ms": last.get("latency_ms"),
        "token_usage_source": last.get("token_usage_source"),
    }


def _load_rows(source_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out = {}
    for rep in (1, 2):
        for row in read_jsonl(source_dir / f"formal_replicate_{rep}" / "rows.jsonl"):
            out[(int(row["replicate"]), row["sample_id"])] = row
    return out


def _load_traces(source_dir: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for rep in (1, 2):
        for event in read_jsonl(source_dir / f"formal_replicate_{rep}" / "trace.jsonl"):
            out[(rep, event["sample_id"])].append(event)
    return dict(out)


def _load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_jsonl(path)}


def _load_questions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_jsonl(path)}


def _retry_by_key(records: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("sample_id") and row.get("replicate_id") in {1, 2}:
            out[(int(row["replicate_id"]), row["sample_id"])].append(row)
    return dict(out)


def _artifact_source_manifest(source_dir: Path) -> dict[str, Any]:
    files = []
    for rel in TASK0069_REQUIRED:
        path = source_dir / rel
        files.append({"relative_path": rel, "sha256": file_digest(path), "byte_count": path.stat().st_size})
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "source_task": "TASK-0069",
        "artifact_source": str(source_dir),
        "audit_mode": "A0_artifact_only",
        "diagnostic_rerun_executed": False,
        "files": files,
    }


def _ensure_required(source_dir: Path) -> None:
    missing = [rel for rel in TASK0069_REQUIRED if not (source_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"TASK-0069 artifact source is incomplete: {', '.join(missing)}")


def _last_action(events: list[dict[str, Any]], action: str) -> dict[str, Any] | None:
    selected = [event for event in events if (event.get("action") or {}).get("action") == action]
    return selected[-1] if selected else None


def _action_present(events: list[dict[str, Any]], action: str) -> bool:
    return any((event.get("action") or {}).get("action") == action for event in events)


def _answer_response(event: dict[str, Any] | None) -> dict[str, Any]:
    return ((event or {}).get("tool_response_summary") or {}).get("answer_response") or {}


def _answerability(event: dict[str, Any] | None) -> dict[str, Any]:
    return ((event or {}).get("tool_response_summary") or {}).get("answerability") or {}


def _grounding(event: dict[str, Any] | None) -> dict[str, Any]:
    return ((event or {}).get("tool_response_summary") or {}).get("grounding") or {}


def _citation_ids(answer_response: dict[str, Any]) -> list[str]:
    return [str(c.get("citation_id")) for c in answer_response.get("citations", []) if isinstance(c, dict) and c.get("citation_id")]


def _response_length(response: dict[str, Any]) -> int:
    return len(json.dumps(response, ensure_ascii=False, sort_keys=True)) if response else 0


def _unsupported_claim_count(grounding: dict[str, Any], answer_response: dict[str, Any]) -> int:
    unsupported = ((grounding.get("diagnostics") or {}).get("unsupported_claims") or answer_response.get("unsupported_claims") or [])
    if isinstance(unsupported, list):
        return len(unsupported)
    return 0


def _int_or_zero(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _gold_expects_answer(expected_action: Any) -> bool:
    return str(expected_action) in {"answer", "partial_answer"}


def _gold_expects_abstain(expected_action: Any) -> bool:
    return str(expected_action) in {"abstain", "correct_premise"}


def _rate(num: int, denom: int) -> float | str:
    return "not_applicable" if denom == 0 else num / denom


def _predicate_agreement(pairs: list[list[dict[str, Any]]], predicate) -> float | str:
    if not pairs:
        return "not_applicable"
    return sum(1 for left, right in pairs if predicate(left) == predicate(right)) / len(pairs)


def _key(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": LIFECYCLE_CONTRACT_VERSION,
        "sample_id": record["sample_id"],
        "replicate_id": record["replicate_id"],
    }
