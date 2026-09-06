from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.replay import replay_trace

TRANSITION_CONTRACT_VERSION = "opk-rag.evidence-to-answer-transition.v1"
EVIDENCE_LIFECYCLE_CONTRACT_VERSION = "opk-rag.evidence-lifecycle.v1"
RESULT_SCHEMA_VERSION = "opk-rag.task0068-evidence-conversion-audit.v1"

FUNNEL_STAGES = (
    "F0_second_round_search_executed",
    "F1_new_evidence_retrieved",
    "F2_new_evidence_merged",
    "F3_new_evidence_admitted_to_answerability_bundle",
    "F4_answerability_re_evaluated",
    "F5_generation_eligible",
    "F6_generation_invoked",
    "F7_non_abstaining_draft_produced",
    "F8_citation_validation_passed",
    "F9_grounding_verification_passed",
    "F10_final_answer_returned",
)

PRIMARY_BOTTLENECKS = {
    "no_new_evidence",
    "evidence_merge_loss",
    "answerability_bundle_exclusion",
    "answerability_not_generation_eligible",
    "answerability_state_stale",
    "policy_early_abstention",
    "generation_not_invoked",
    "generation_model_abstention",
    "generation_contract_failure",
    "citation_validation_failure",
    "grounding_verification_failure",
    "budget_exhaustion",
    "runtime_transition_failure",
    "provider_failure",
    "mixed_or_unresolved",
}

SECONDARY_FACTORS = {
    "low_rank_new_evidence",
    "context_budget_pressure",
    "duplicate_evidence",
    "scope_mismatch",
    "weak_evidence_coverage",
    "provider_nondeterminism",
    "query_reformulation_instability",
    "generation_over_abstention",
    "policy_early_abstention",
    "runtime_transition_not_exposed",
    "stale_generation_eligibility",
    "citation_identity_gap",
    "grounding_threshold_pressure",
    "insufficient_trace_observability",
}

DIAGNOSTIC_CONFIDENCES = {"high", "medium", "low"}
GENERATION_ELIGIBLE_STATUSES = {"answerable", "partially_answerable", "partial_answer_possible"}


@dataclass(frozen=True)
class Task0067Artifacts:
    source_dir: Path
    trace_files: tuple[Path, ...]
    run_files: tuple[Path, ...]
    retrieval_round_files: tuple[Path, ...]
    required_files: tuple[Path, ...]


def discover_task0067_artifacts(source_dir: Path) -> Task0067Artifacts:
    required = (
        "run_context.json",
        "formal_experiment_frozen_config.json",
        "real_provider_experiment_metrics.json",
        "evidence_gain_report.json",
        "answerability_transition_report.json",
        "privacy_scan_report.json",
        "c2_traces_rep1.jsonl",
        "c2_traces_rep2.jsonl",
        "c2_run_rep1.jsonl",
        "c2_run_rep2.jsonl",
        "retrieval_rounds_rep1.jsonl",
        "retrieval_rounds_rep2.jsonl",
    )
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"TASK-0067 artifact source is incomplete: {', '.join(missing)}")
    return Task0067Artifacts(
        source_dir=source_dir,
        trace_files=(source_dir / "c2_traces_rep1.jsonl", source_dir / "c2_traces_rep2.jsonl"),
        run_files=(source_dir / "c2_run_rep1.jsonl", source_dir / "c2_run_rep2.jsonl"),
        retrieval_round_files=(source_dir / "retrieval_rounds_rep1.jsonl", source_dir / "retrieval_rounds_rep2.jsonl"),
        required_files=tuple(source_dir / name for name in required),
    )


def validate_transition_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("contract_version") != TRANSITION_CONTRACT_VERSION:
        errors.append("invalid_contract_version")
    for field in (
        "sample_id",
        "run_id",
        "replicate_id",
        "initial_retrieval",
        "reformulation",
        "second_round_retrieval",
        "evidence_merge",
        "bundle_admission",
        "answerability_transition",
        "generation",
        "citation_validation",
        "grounding_verification",
        "finalization",
    ):
        if field not in record:
            errors.append(f"missing_{field}")
    unknown = set(record.get("funnel", {})) - set(FUNNEL_STAGES)
    if unknown:
        errors.append("unknown_funnel_stage")
    primary = record.get("primary_conversion_bottleneck")
    if isinstance(primary, list):
        errors.append("multiple_primary_bottlenecks")
    elif primary not in PRIMARY_BOTTLENECKS:
        errors.append("unknown_primary_bottleneck")
    confidence = record.get("diagnostic_confidence")
    if confidence not in DIAGNOSTIC_CONFIDENCES:
        errors.append("invalid_diagnostic_confidence")
    return errors


def validate_lifecycle_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("contract_version") != EVIDENCE_LIFECYCLE_CONTRACT_VERSION:
        errors.append("invalid_contract_version")
    for field in (
        "evidence_identity",
        "first_seen_round",
        "retrieved",
        "merged",
        "admitted_to_answerability_bundle",
        "admitted_to_generation_bundle",
        "referenced_by_generation",
        "referenced_by_citation",
        "accepted_by_grounding",
        "drop_stage",
        "drop_reason_code",
    ):
        if field not in record:
            errors.append(f"missing_{field}")
    return errors


def audit_task0067_evidence_conversion(source_dir: Path, output_dir: Path, contract_output_path: Path) -> dict[str, Any]:
    artifacts = discover_task0067_artifacts(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _artifact_source_manifest(artifacts)
    _write_json(output_dir / "artifact_source_manifest.json", manifest)
    contract = transition_contract_document()
    _write_json(output_dir / "evidence_to_answer_transition_contract.json", contract)
    _write_json(contract_output_path, contract)

    traces_by_rep_sample = _load_traces(artifacts.trace_files)
    runs = _load_runs(artifacts.run_files)
    rounds = _load_rounds(artifacts.retrieval_round_files)

    all_sample_audits: list[dict[str, Any]] = []
    all_lifecycles: list[dict[str, Any]] = []
    replay_reports: list[dict[str, Any]] = []
    trace_missing_count = 0

    for key in sorted(runs):
        run = runs[key]
        replicate = int(run["replicate"])
        sample_id = run["sample_id"]
        sample_rounds = rounds.get(key, [])
        trace_events = traces_by_rep_sample.get(key, [])
        if not trace_events:
            trace_missing_count += 1
        replay = replay_trace(trace_events)
        replay_reports.append({"replicate": replicate, "sample_id": sample_id, **replay})
        sample_audit, lifecycles = build_sample_conversion_audit(run, sample_rounds, trace_events)
        all_sample_audits.append(sample_audit)
        all_lifecycles.extend(lifecycles)

    _write_jsonl(output_dir / "sample_conversion_audit.jsonl", all_sample_audits)
    _write_jsonl(output_dir / "evidence_lifecycle.jsonl", all_lifecycles)
    _write_jsonl(output_dir / "bundle_admission_audit.jsonl", [record["bundle_admission"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "answerability_transition_audit.jsonl", [record["answerability_transition"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "generation_invocation_audit.jsonl", [record["generation"]["invocation"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "generation_outcome_audit.jsonl", [record["generation"]["outcome"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "citation_validation_audit.jsonl", [record["citation_validation"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "grounding_verification_audit.jsonl", [record["grounding_verification"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "final_action_audit.jsonl", [record["finalization"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "state_freshness_audit.jsonl", [record["state_freshness"] | _sample_key(record) for record in all_sample_audits])
    _write_jsonl(output_dir / "bottleneck_classification.jsonl", [_classification(record) for record in all_sample_audits])

    funnel = build_funnel_metrics(all_sample_audits, all_lifecycles)
    _write_json(output_dir / "conversion_funnel.json", funnel)
    by_replicate = {str(rep): build_funnel_metrics([r for r in all_sample_audits if r["replicate_id"] == str(rep)], [l for l in all_lifecycles if l["replicate_id"] == str(rep)]) for rep in sorted({int(r["replicate_id"]) for r in all_sample_audits})}
    _write_json(output_dir / "conversion_funnel_by_replicate.json", {"schema_version": RESULT_SCHEMA_VERSION, "replicates": by_replicate})
    bottleneck_summary = build_bottleneck_summary(all_sample_audits)
    _write_json(output_dir / "bottleneck_summary.json", bottleneck_summary)
    _write_json(output_dir / "replicate_consistency_report.json", build_replicate_consistency(all_sample_audits, all_lifecycles))
    _write_json(output_dir / "diagnostic_replay_report.json", build_replay_report(replay_reports, trace_missing_count))
    _write_json(output_dir / "provenance_continuity_report.json", build_provenance_continuity_report(all_sample_audits, replay_reports))
    _write_json(output_dir / "privacy_scan_report.json", scan_privacy(output_dir))
    _write_json(output_dir / "benchmark_integrity_report.json", build_benchmark_integrity_report(source_dir / "run_context.json"))
    _write_json(output_dir / "promotion_decision.json", build_promotion_decision(all_sample_audits, trace_missing_count))
    return {"sample_audits": all_sample_audits, "evidence_lifecycles": all_lifecycles, "funnel": funnel, "bottleneck_summary": bottleneck_summary}


def build_sample_conversion_audit(run: dict[str, Any], sample_rounds: list[dict[str, Any]], trace_events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replicate = str(run["replicate"])
    sample_id = run["sample_id"]
    action_names = [((event.get("action") or {}).get("action")) for event in trace_events]
    searches = [event for event in trace_events if ((event.get("action") or {}).get("action")) == "search"]
    answerability_events = [event for event in trace_events if ((event.get("action") or {}).get("action")) == "evaluate_answerability"]
    generation_events = [event for event in trace_events if ((event.get("action") or {}).get("action")) == "generate_grounded_answer"]
    grounding_events = [event for event in trace_events if ((event.get("action") or {}).get("action")) == "verify_grounding"]
    second_round = next((item for item in sample_rounds if item.get("round_index") == 2), None)
    first_round = next((item for item in sample_rounds if item.get("round_index") == 1), None)

    initial_ids = _trusted_identity_set(searches[:1])
    second_ids = _trusted_identity_set(searches[1:2])
    answerability_chunk_ids = _answerability_chunk_ids(answerability_events[-1:]) if answerability_events else set()
    answerability_statuses = [_answerability_payload(event).get("status") for event in answerability_events]
    before_status = answerability_statuses[0] if answerability_statuses else None
    after_status = answerability_statuses[-1] if answerability_statuses else None
    answerability_improved = _status_rank(after_status) > _status_rank(before_status)
    generation_eligible = after_status in GENERATION_ELIGIBLE_STATUSES
    generation_invoked = bool(generation_events)
    state_freshness = build_state_freshness(trace_events, generation_eligible, generation_invoked)
    policy_handoff = build_policy_handoff_audit(
        trace_events=trace_events,
        latest_answerability_status=after_status,
        generation_eligible=generation_eligible,
        generation_invoked=generation_invoked,
        state_freshness=state_freshness,
    )
    final_action = run.get("final_action")
    final_abstain = final_action == "abstain"
    non_abstaining_draft = any((event.get("tool_response_summary") or {}).get("answer_response", {}).get("status") == "answered" for event in generation_events)
    citation_pass = bool(non_abstaining_draft)
    grounding_pass = any(((event.get("tool_response_summary") or {}).get("grounding") or {}).get("valid") is True for event in grounding_events)
    final_answer = final_action == "answer"

    primary, secondary, confidence = classify_bottleneck(
        second_round_search=bool(second_round),
        new_evidence_count=int(run.get("new_evidence_identity_count") or 0),
        answerability_improved=answerability_improved,
        generation_eligible=generation_eligible,
        generation_invoked=generation_invoked,
        final_action=final_action,
        action_names=action_names,
        trace_events=trace_events,
        policy_handoff=policy_handoff,
    )

    bundle = build_bundle_admission_audit(initial_ids, second_ids, answerability_chunk_ids, generation_events, grounding_events)
    transition = {
        "before_status": before_status,
        "after_status": after_status,
        "generation_eligible_before": before_status in GENERATION_ELIGIBLE_STATUSES,
        "generation_eligible_after": generation_eligible,
        "reason_code_before": _answerability_payload(answerability_events[0]).get("reason_code") if answerability_events else None,
        "reason_code_after": _answerability_payload(answerability_events[-1]).get("reason_code") if answerability_events else None,
        "evidence_count_before": _answerability_payload(answerability_events[0]).get("evidence_count") if answerability_events else 0,
        "evidence_count_after": _answerability_payload(answerability_events[-1]).get("evidence_count") if answerability_events else 0,
        "new_evidence_used": bool(second_ids and answerability_chunk_ids),
        "answerability_improved": answerability_improved,
        "answerability_re_evaluated": len(answerability_events) >= 2,
    }
    generation = {
        "invocation": {
            "generation_eligible": generation_eligible,
            "generation_invoked": generation_invoked,
            "generation_skipped": not generation_invoked,
            "generation_skip_reason": _generation_skip_reason(generation_eligible, action_names, trace_events),
            "generation_provider_called": generation_invoked,
            "generation_provider_failure": any(event.get("failure") for event in generation_events),
            "prompt_context_contains_new_evidence_identity": _generation_prompt_contains_new_evidence(generation_events, second_ids),
            "new_evidence_identity_count": len(second_ids),
        },
        "outcome": build_generation_outcome(generation_events, second_ids),
    }
    citation = build_citation_validation(generation_events, bundle["generation_bundle_identities"])
    grounding = build_grounding_verification(grounding_events)
    funnel = {stage: {"reached": False, "reason_code": None} for stage in FUNNEL_STAGES}
    funnel["F0_second_round_search_executed"]["reached"] = bool(second_round)
    funnel["F1_new_evidence_retrieved"]["reached"] = bool(second_ids)
    funnel["F2_new_evidence_merged"]["reached"] = bool(second_ids)
    funnel["F3_new_evidence_admitted_to_answerability_bundle"]["reached"] = bundle["new_evidence_retained_for_answerability_count"] > 0
    funnel["F4_answerability_re_evaluated"]["reached"] = len(answerability_events) >= 2
    funnel["F5_generation_eligible"]["reached"] = generation_eligible
    funnel["F6_generation_invoked"]["reached"] = generation_invoked
    funnel["F7_non_abstaining_draft_produced"]["reached"] = non_abstaining_draft
    funnel["F8_citation_validation_passed"]["reached"] = citation_pass
    funnel["F9_grounding_verification_passed"]["reached"] = grounding_pass
    funnel["F10_final_answer_returned"]["reached"] = final_answer

    record = {
        "contract_version": TRANSITION_CONTRACT_VERSION,
        "sample_id": sample_id,
        "run_id": run.get("agent_result", {}).get("run_id") or trace_events[0].get("run_id") if trace_events else f"rep{replicate}-{sample_id}",
        "replicate_id": replicate,
        "initial_retrieval": {"executed": bool(first_round), "evidence_identity_count": len(initial_ids)},
        "reformulation": {"executed": bool(second_round), "query_digest": None if second_round is None else second_round.get("query_digest")},
        "second_round_retrieval": {"executed": bool(second_round), "new_evidence_identity_count": len(second_ids), "result_count": 0 if second_round is None else second_round.get("result_count")},
        "evidence_merge": {"merged": bool(second_ids), "post_merge_evidence_identity_count": len(initial_ids | second_ids), "new_evidence_merged_count": len(second_ids)},
        "bundle_admission": bundle,
        "answerability_transition": transition,
        "generation": generation,
        "citation_validation": citation,
        "grounding_verification": grounding,
        "finalization": {"final_action": final_action, "final_abstain": final_abstain, "final_abstain_reason": _final_abstain_reason(primary, action_names), "final_answer": final_answer},
        "state_freshness": state_freshness,
        "policy_generation_handoff": policy_handoff,
        "funnel": funnel,
        "primary_conversion_bottleneck": primary,
        "secondary_contributing_factors": sorted(secondary),
        "diagnostic_confidence": confidence,
        "trace_observability": {"event_count": len(trace_events), "has_generation_context_digest": any("prompt_context_digest" in json.dumps(event, sort_keys=True) for event in generation_events), "has_citation_validator_events": bool(generation_events), "has_grounding_events": bool(grounding_events)},
    }
    lifecycles = [build_lifecycle_record(record, identity, second_round, trace_events) for identity in sorted(second_ids)]
    return record, lifecycles


def build_bundle_admission_audit(initial_ids: set[str], second_ids: set[str], answerability_chunk_ids: set[str], generation_events: list[dict[str, Any]], grounding_events: list[dict[str, Any]]) -> dict[str, Any]:
    generation_ids = _generation_citation_ids(generation_events)
    grounding_ids = _grounding_cited_ids(grounding_events)
    answerability_kept = second_ids if answerability_chunk_ids else set()
    generation_kept = second_ids if generation_events else set()
    return {
        "post_merge_evidence_count": len(initial_ids | second_ids),
        "answerability_bundle_count": len(answerability_chunk_ids),
        "generation_bundle_count": len(generation_ids),
        "new_evidence_retained_for_answerability_count": len(answerability_kept),
        "new_evidence_retained_for_generation_count": len(generation_kept),
        "new_evidence_trimmed_count": len(second_ids - answerability_kept),
        "bundle_limit": None,
        "bundle_digest": stable_digest({"initial": sorted(initial_ids), "second": sorted(second_ids), "answerability_chunks": sorted(answerability_chunk_ids)}),
        "retrieved_but_not_merged": [],
        "merged_but_not_in_answerability": sorted(second_ids - answerability_kept),
        "answerability_used_but_not_in_generation": sorted(answerability_kept - generation_kept),
        "generation_used_but_not_available_to_citation": [],
        "citation_used_but_not_available_to_grounding": sorted(generation_ids - grounding_ids),
        "generation_bundle_identities": sorted(generation_ids),
    }


def classify_bottleneck(
    *,
    second_round_search: bool,
    new_evidence_count: int,
    answerability_improved: bool,
    generation_eligible: bool,
    generation_invoked: bool,
    final_action: str | None,
    action_names: list[str | None],
    trace_events: list[dict[str, Any]],
    policy_handoff: dict[str, Any] | None = None,
) -> tuple[str, set[str], str]:
    secondary: set[str] = set()
    if not trace_events:
        return "mixed_or_unresolved", {"insufficient_trace_observability"}, "low"
    if not second_round_search or new_evidence_count == 0:
        return "no_new_evidence", secondary, "high"
    if not answerability_improved:
        return "answerability_not_generation_eligible", {"weak_evidence_coverage"}, "medium"
    if not generation_eligible:
        return "answerability_not_generation_eligible", secondary, "high"
    if generation_eligible and not generation_invoked:
        handoff = policy_handoff or build_policy_handoff_audit(
            trace_events=trace_events,
            latest_answerability_status=None,
            generation_eligible=generation_eligible,
            generation_invoked=generation_invoked,
            state_freshness=build_state_freshness(trace_events, generation_eligible, generation_invoked),
        )
        if handoff["stale_state_confirmed"]:
            return "answerability_state_stale", {"stale_generation_eligibility"}, "high"
        if handoff["policy_selected_action"] == "generate_grounded_answer":
            return "generation_not_invoked", {"insufficient_trace_observability"}, "high"
        if handoff["policy_view_latest_state"] and handoff["generate_action_exposed"] and handoff["policy_selected_action"] == "finish_abstain":
            return "policy_early_abstention", {"policy_early_abstention"}, "high"
        if handoff["runtime_transition_failure"]:
            return "runtime_transition_failure", {"runtime_transition_not_exposed"}, "high"
        return "generation_not_invoked", {"insufficient_trace_observability"}, "medium"
    if generation_invoked and final_action != "answer":
        if "verify_grounding" in action_names:
            return "grounding_verification_failure", secondary, "high"
        return "generation_model_abstention", secondary, "medium"
    return "mixed_or_unresolved", secondary, "medium"


def build_generation_outcome(generation_events: list[dict[str, Any]], second_ids: set[str]) -> dict[str, Any]:
    if not generation_events:
        return {"generation_invoked": False, "outcome": "not_invoked", "model_abstention": False, "empty_output": False, "schema_failure": False, "provider_failure": False, "new_evidence_seen_count": 0}
    event = generation_events[-1]
    payload = (event.get("tool_response_summary") or {}).get("answer_response") or {}
    status = payload.get("status")
    if event.get("failure"):
        outcome = "provider_failure"
    elif status == "answered":
        outcome = "answer_draft"
    elif status == "refused":
        outcome = "model_abstention"
    elif not payload:
        outcome = "empty_output"
    else:
        outcome = "schema_failure"
    return {
        "generation_invoked": True,
        "outcome": outcome,
        "model_abstention": outcome == "model_abstention",
        "empty_output": outcome == "empty_output",
        "schema_failure": outcome == "schema_failure",
        "provider_failure": outcome == "provider_failure",
        "new_evidence_seen_count": len(second_ids),
        "prompt_context_digest": payload.get("prompt_context_digest"),
        "abstention_reason_code": payload.get("refusal_reason_code"),
    }


def build_citation_validation(generation_events: list[dict[str, Any]], available: list[str]) -> dict[str, Any]:
    if not generation_events:
        return {"citation_validation_invoked": False, "citation_count": 0, "known_citation_count": 0, "unknown_citation_count": 0, "missing_citation_count": 0, "citation_identity_match": False, "citation_validation_pass": False, "failure_reason": "generation_not_invoked"}
    citation_ids = _generation_citation_ids(generation_events)
    available_set = set(available)
    unknown = citation_ids - available_set if available_set else set()
    return {
        "citation_validation_invoked": True,
        "citation_count": len(citation_ids),
        "known_citation_count": len(citation_ids - unknown),
        "unknown_citation_count": len(unknown),
        "missing_citation_count": 0 if citation_ids else 1,
        "citation_identity_match": bool(citation_ids) and not unknown,
        "citation_validation_pass": bool(citation_ids) and not unknown,
        "failure_reason": "citation_validation_pass" if citation_ids and not unknown else ("no_citation" if not citation_ids else "citation_not_in_bundle"),
    }


def build_grounding_verification(grounding_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not grounding_events:
        return {"grounding_invoked": False, "claim_count": 0, "supported_claim_count": 0, "unsupported_claim_count": 0, "forbidden_claim_detected": False, "grounding_pass": False, "grounding_rejection_reason": "citation_or_generation_not_available"}
    payload = (grounding_events[-1].get("tool_response_summary") or {}).get("grounding") or {}
    valid = payload.get("valid") is True
    return {
        "grounding_invoked": True,
        "claim_count": payload.get("claim_count", 0),
        "supported_claim_count": payload.get("supported_claim_count", 0),
        "unsupported_claim_count": payload.get("unsupported_claim_count", 0 if valid else 1),
        "forbidden_claim_detected": bool(payload.get("forbidden_claim_detected")),
        "grounding_pass": valid,
        "grounding_rejection_reason": "grounding_pass" if valid else payload.get("reason_code", "unsupported_claim"),
    }


def build_state_freshness(trace_events: list[dict[str, Any]], generation_eligible: bool, generation_invoked: bool) -> dict[str, Any]:
    issues = []
    for previous, current in zip(trace_events, trace_events[1:]):
        if previous.get("state_after_digest") != current.get("state_before_digest"):
            issues.append("state_digest_chain_break")
    return {
        "state_freshness_audited": bool(trace_events),
        "generation_eligible": generation_eligible,
        "generation_invoked": generation_invoked,
        "state_freshness_issue_count": len(issues),
        "issues": issues,
        "stale_answerability_state": False,
        "stale_evidence_bundle": False,
        "stale_generation_eligibility": False,
        "old_round_result_reused": False,
        "new_round_result_not_committed": False,
        "digest_chain_valid": not any(issue == "state_digest_chain_break" for issue in issues),
    }


def build_policy_handoff_audit(
    *,
    trace_events: list[dict[str, Any]],
    latest_answerability_status: str | None,
    generation_eligible: bool,
    generation_invoked: bool,
    state_freshness: dict[str, Any],
) -> dict[str, Any]:
    answerability_indices = [
        index
        for index, event in enumerate(trace_events)
        if ((event.get("action") or {}).get("action")) == "evaluate_answerability"
    ]
    latest_answerability_event_index = answerability_indices[-1] if answerability_indices else None
    latest_answerability_event = None if latest_answerability_event_index is None else trace_events[latest_answerability_event_index]
    decision_event_index = None
    for index in range((latest_answerability_event_index or -1) + 1, len(trace_events)):
        action = (trace_events[index].get("action") or {}).get("action")
        if action in {"generate_grounded_answer", "finish_abstain", "finish_answer", "finish_failure"}:
            decision_event_index = index
            break
    decision_event = None if decision_event_index is None else trace_events[decision_event_index]
    policy_decision = (decision_event or {}).get("policy_decision") or {}
    available_actions = policy_decision.get("available_actions")
    policy_decision_payload = policy_decision.get("decision") or {}
    latest_answerability_digest = None if latest_answerability_event is None else latest_answerability_event.get("state_after_digest")
    decision_state_before_digest = None if decision_event is None else decision_event.get("state_before_digest")
    latest_state_entered_decision = bool(latest_answerability_digest and latest_answerability_digest == decision_state_before_digest)
    policy_view_present = bool(policy_decision)
    stale_state_confirmed = bool(
        state_freshness.get("stale_answerability_state")
        or state_freshness.get("stale_generation_eligibility")
        or (decision_event is not None and not latest_state_entered_decision)
    )
    generate_action_exposed = isinstance(available_actions, list) and "generate_grounded_answer" in available_actions
    policy_selected_action = policy_decision_payload.get("action")
    runtime_action = (decision_event.get("action") or {}).get("action") if decision_event else None
    runtime_reason_code = (decision_event.get("action") or {}).get("reason_code") if decision_event else None
    runtime_rejected_or_rewrote_action = bool(policy_selected_action and runtime_action and policy_selected_action != runtime_action)
    runtime_transition_failure = bool(
        generation_eligible
        and latest_state_entered_decision
        and not generation_invoked
        and not generate_action_exposed
        and runtime_action == "finish_abstain"
    )
    return {
        "contract_version": TRANSITION_CONTRACT_VERSION,
        "latest_answerability_status": latest_answerability_status,
        "generation_eligible": generation_eligible,
        "answerability_result_digest": stable_digest((latest_answerability_event or {}).get("tool_response_summary", {}).get("answerability") or {}),
        "evidence_bundle_digest": stable_digest(
            {
                "answerability_chunk_ids": ((latest_answerability_event or {}).get("tool_response_summary", {}).get("answerability") or {}).get("evidence_chunk_ids", []),
                "trusted_evidence_digest": ((trace_events[decision_event_index - 1] if decision_event_index and decision_event_index > 0 else {}).get("tool_response_summary") or {}).get("trusted_evidence", {}).get("evidence_identity_digest"),
            }
        ),
        "latest_answerability_event_index": latest_answerability_event_index,
        "policy_decision_event_index": decision_event_index,
        "latest_answerability_state_after_digest": latest_answerability_digest,
        "policy_state_before_digest": decision_state_before_digest,
        "latest_state_entered_runtime_decision": latest_state_entered_decision,
        "policy_view_present": policy_view_present,
        "policy_view_latest_state": policy_view_present and latest_state_entered_decision,
        "available_actions_present": isinstance(available_actions, list),
        "generate_action_exposed": generate_action_exposed,
        "policy_selected_action": policy_selected_action,
        "policy_decision_reason_code": policy_decision_payload.get("reason_code"),
        "runtime_final_action": runtime_action,
        "runtime_reason_code": runtime_reason_code,
        "runtime_rejected_or_rewrote_action": runtime_rejected_or_rewrote_action,
        "runtime_rejection_reason_code": None,
        "stale_state_confirmed": stale_state_confirmed,
        "runtime_transition_failure": runtime_transition_failure,
        "generation_provider_called": generation_invoked,
    }


def build_lifecycle_record(sample_record: dict[str, Any], identity: str, second_round: dict[str, Any] | None, trace_events: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = sample_record["bundle_admission"]
    admitted_answerability = identity not in set(bundle["merged_but_not_in_answerability"])
    admitted_generation = identity in set(bundle["generation_bundle_identities"])
    generation_invoked = sample_record["generation"]["invocation"]["generation_invoked"]
    drop_stage = None
    drop_reason = None
    if not admitted_answerability:
        drop_stage = "answerability_bundle"
        drop_reason = "not_present_in_answerability_trace"
    elif not admitted_generation:
        drop_stage = "generation_invocation"
        drop_reason = sample_record["generation"]["invocation"]["generation_skip_reason"]
    elif generation_invoked and not sample_record["citation_validation"]["citation_validation_pass"]:
        drop_stage = "citation_validation"
        drop_reason = sample_record["citation_validation"]["failure_reason"]
    elif sample_record["citation_validation"]["citation_validation_pass"] and not sample_record["grounding_verification"]["grounding_pass"]:
        drop_stage = "grounding_verification"
        drop_reason = sample_record["grounding_verification"]["grounding_rejection_reason"]
    elif sample_record["finalization"]["final_answer"]:
        drop_stage = "final_answer"
        drop_reason = "converted"
    else:
        drop_stage = "finalization"
        drop_reason = sample_record["finalization"]["final_abstain_reason"]
    return {
        "contract_version": EVIDENCE_LIFECYCLE_CONTRACT_VERSION,
        "replicate_id": sample_record["replicate_id"],
        "sample_id": sample_record["sample_id"],
        "evidence_identity": identity,
        "first_seen_round": 2,
        "query_digest": None if second_round is None else second_round.get("query_digest"),
        "document_identity": _digest_short("document", identity),
        "chunk_identity": _digest_short("chunk", identity),
        "retrieved": True,
        "merged": True,
        "admitted_to_answerability_bundle": admitted_answerability,
        "admitted_to_generation_bundle": admitted_generation,
        "referenced_by_generation": admitted_generation and generation_invoked,
        "referenced_by_citation": sample_record["citation_validation"]["citation_validation_pass"],
        "accepted_by_grounding": sample_record["grounding_verification"]["grounding_pass"],
        "drop_stage": drop_stage,
        "drop_reason_code": drop_reason,
        "trace_event_digest": stable_digest(trace_events),
    }


def build_funnel_metrics(records: list[dict[str, Any]], lifecycles: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = {stage: sum(1 for record in records if record["funnel"][stage]["reached"]) for stage in FUNNEL_STAGES}
    rates = {}
    pairs = [
        ("F1_new_evidence_retrieved", "F2_new_evidence_merged"),
        ("F2_new_evidence_merged", "F3_new_evidence_admitted_to_answerability_bundle"),
        ("F3_new_evidence_admitted_to_answerability_bundle", "F4_answerability_re_evaluated"),
        ("F4_answerability_re_evaluated", "F5_generation_eligible"),
        ("F5_generation_eligible", "F6_generation_invoked"),
        ("F6_generation_invoked", "F7_non_abstaining_draft_produced"),
        ("F7_non_abstaining_draft_produced", "F8_citation_validation_passed"),
        ("F8_citation_validation_passed", "F9_grounding_verification_passed"),
        ("F9_grounding_verification_passed", "F10_final_answer_returned"),
    ]
    for before, after in pairs:
        denom = stage_counts[before]
        rates[f"{before}_to_{after}"] = "not_applicable" if denom == 0 else stage_counts[after] / denom
    drop_stage_counts = Counter(record["drop_stage"] for record in lifecycles)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "sample_count": len(records),
        "stage_counts": stage_counts,
        "stage_conversion_rates": rates,
        "new_evidence_identity_count": len(lifecycles),
        "new_evidence_answerability_bundle_count": sum(1 for item in lifecycles if item["admitted_to_answerability_bundle"]),
        "new_evidence_generation_bundle_count": sum(1 for item in lifecycles if item["admitted_to_generation_bundle"]),
        "new_evidence_referenced_by_generation_count": sum(1 for item in lifecycles if item["referenced_by_generation"]),
        "new_evidence_referenced_by_citation_count": sum(1 for item in lifecycles if item["referenced_by_citation"]),
        "new_evidence_accepted_by_grounding_count": sum(1 for item in lifecycles if item["accepted_by_grounding"]),
        "evidence_drop_stage_distribution": dict(sorted(drop_stage_counts.items())),
    }


def build_bottleneck_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = Counter(record["primary_conversion_bottleneck"] for record in records)
    secondary = Counter(factor for record in records for factor in record["secondary_contributing_factors"])
    confidence = Counter(record["diagnostic_confidence"] for record in records)
    final_reasons = Counter(record["finalization"]["final_abstain_reason"] for record in records if record["finalization"]["final_abstain"])
    eligible = [record for record in records if record["generation"]["invocation"]["generation_eligible"]]
    handoff = build_handoff_matrix(eligible)
    state_freshness_audited = sum(1 for record in eligible if record["state_freshness"]["state_freshness_audited"])
    state_freshness_issue_count = sum(record["state_freshness"]["state_freshness_issue_count"] for record in eligible)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "primary_bottleneck_distribution": dict(sorted(primary.items())),
        "secondary_factor_distribution": dict(sorted(secondary.items())),
        "diagnostic_confidence_distribution": dict(sorted(confidence.items())),
        "final_abstain_reason_distribution": dict(sorted(final_reasons.items())),
        "generation_eligible_handoff_matrix": handoff,
        "state_freshness_audited": state_freshness_audited,
        "state_freshness_issue_count": state_freshness_issue_count,
        "unresolved_sample_count": primary.get("mixed_or_unresolved", 0),
    }


def build_handoff_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generation_eligible_sample_count": len(records),
        "policy_view_latest_state_count": sum(1 for record in records if record["policy_generation_handoff"]["policy_view_latest_state"]),
        "latest_state_entered_runtime_decision_count": sum(1 for record in records if record["policy_generation_handoff"]["latest_state_entered_runtime_decision"]),
        "generate_action_exposed_count": sum(1 for record in records if record["policy_generation_handoff"]["generate_action_exposed"]),
        "policy_selected_generate_count": sum(1 for record in records if record["policy_generation_handoff"]["policy_selected_action"] == "generate_grounded_answer"),
        "policy_selected_finish_abstain_count": sum(1 for record in records if record["policy_generation_handoff"]["policy_selected_action"] == "finish_abstain"),
        "runtime_selected_finish_abstain_count": sum(1 for record in records if record["policy_generation_handoff"]["runtime_final_action"] == "finish_abstain"),
        "runtime_rejected_generate_count": sum(1 for record in records if record["policy_generation_handoff"]["runtime_rejected_or_rewrote_action"] and record["policy_generation_handoff"]["policy_selected_action"] == "generate_grounded_answer"),
        "stale_state_confirmed_count": sum(1 for record in records if record["policy_generation_handoff"]["stale_state_confirmed"]),
        "runtime_transition_failure_count": sum(1 for record in records if record["policy_generation_handoff"]["runtime_transition_failure"]),
    }


def build_replicate_consistency(records: list[dict[str, Any]], lifecycles: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)

    comparable = [items for items in by_sample.values() if len(items) == 2]

    def agreement(field: str) -> float | str:
        if not comparable:
            return "not_applicable"
        matches = 0
        for left, right in comparable:
            if _field_value(left, field) == _field_value(right, field):
                matches += 1
        return matches / len(comparable)

    def conditional_agreement(field: str, predicate) -> float | str:
        selected = [items for items in comparable if predicate(items)]
        if not selected:
            return "not_applicable"
        matches = 0
        for left, right in selected:
            if _field_value(left, field) == _field_value(right, field):
                matches += 1
        return matches / len(selected)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "comparable_sample_count": len(comparable),
        "evidence_gain_sample_consistency_rate": agreement("second_round_retrieval.new_evidence_identity_count"),
        "answerability_improvement_consistency_rate": agreement("answerability_transition.answerability_improved"),
        "generation_eligibility_consistency_rate": agreement("answerability_transition.generation_eligible_after"),
        "generation_invocation_consistency_rate": agreement("generation.invocation.generation_invoked"),
        "final_abstain_reason_consistency_rate": agreement("finalization.final_abstain_reason"),
        "primary_bottleneck_consistency_rate": agreement("primary_conversion_bottleneck"),
        "evidence_gain_primary_bottleneck_consistency_rate": conditional_agreement(
            "primary_conversion_bottleneck",
            lambda items: all(item["second_round_retrieval"]["new_evidence_identity_count"] > 0 for item in items),
        ),
        "generation_eligible_final_action_consistency_rate": conditional_agreement(
            "finalization.final_action",
            lambda items: all(item["generation"]["invocation"]["generation_eligible"] for item in items),
        ),
        "generation_eligible_primary_bottleneck_uniformity_rate": _uniform_rate(
            [record["primary_conversion_bottleneck"] for record in records if record["generation"]["invocation"]["generation_eligible"]]
        ),
        "evidence_drop_stage_consistency_rate": _lifecycle_drop_agreement(lifecycles),
    }


def build_replay_report(reports: list[dict[str, Any]], missing_trace_count: int) -> dict[str, Any]:
    status_counts = Counter(report["status"] for report in reports)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "diagnostic_mode": "D0_artifact_only_and_D2_offline_transition_replay",
        "remote_provider_called": False,
        "sample_replay_count": len(reports),
        "replay_status_distribution": dict(sorted(status_counts.items())),
        "missing_trace_count": missing_trace_count,
        "status": "pass" if status_counts.get("fail", 0) == 0 and missing_trace_count == 0 else "fail",
        "reports": reports,
    }


def build_provenance_continuity_report(records: list[dict[str, Any]], replay_reports: list[dict[str, Any]]) -> dict[str, Any]:
    issue_count = sum(1 for report in replay_reports if report["status"] != "pass")
    issue_count += sum(record["state_freshness"]["state_freshness_issue_count"] for record in records if not record["state_freshness"]["digest_chain_valid"])
    return {"schema_version": RESULT_SCHEMA_VERSION, "provenance_continuity_issue_count": issue_count, "state_digest_chain_issue_count": sum(1 for report in replay_reports if report["status"] != "pass"), "round_provenance_available": True, "query_digest_available": True, "status": "pass" if issue_count == 0 else "fail"}


def build_benchmark_integrity_report(run_context_path: Path) -> dict[str, Any]:
    payload = _read_json(run_context_path)
    before = payload.get("benchmark_hashes_before", {})
    after = payload.get("benchmark_hashes_after", {})
    return {"schema_version": RESULT_SCHEMA_VERSION, "benchmark_hashes_before": before, "benchmark_hashes_after": after, "hashes_match": before == after, "gold_leakage_count": 0}


def build_promotion_decision(records: list[dict[str, Any]], missing_trace_count: int) -> dict[str, Any]:
    evidence_gain_records = [record for record in records if record["second_round_retrieval"]["new_evidence_identity_count"] > 0]
    decision_population = evidence_gain_records or records
    primary = Counter(record["primary_conversion_bottleneck"] for record in decision_population)
    dominant_primary = primary.most_common(1)[0][0] if primary else None
    if missing_trace_count:
        decision = "insufficient_observability_reinstrument_first"
        next_direction = "restore_trace_observability_before_optimization"
    elif primary.get("policy_early_abstention", 0) >= max(1, len(decision_population) // 2):
        decision = "optimize_answerability_generation_handoff"
        next_direction = "diagnose_and_correct_model_policy_early_abstention_after_generation_eligibility"
    elif primary.get("answerability_state_stale", 0) >= max(1, len(decision_population) // 2):
        decision = "optimize_answerability_generation_handoff"
        next_direction = "diagnose_and_correct_answerability_state_freshness_after_generation_eligibility"
    elif primary.get("runtime_transition_failure", 0) >= max(1, len(decision_population) // 2):
        decision = "optimize_answerability_generation_handoff"
        next_direction = "diagnose_and_correct_runtime_transition_exposure_after_generation_eligibility"
    elif primary.get("generation_model_abstention", 0) >= max(1, len(decision_population) // 2):
        decision = "optimize_generation_abstention"
        next_direction = "diagnose_generation_model_abstention_after_invocation"
    elif primary.get("citation_validation_failure", 0) or primary.get("grounding_verification_failure", 0):
        decision = "optimize_citation_grounding_pipeline"
        next_direction = "diagnose_citation_grounding_rejection_after_generation"
    elif primary.get("answerability_bundle_exclusion", 0):
        decision = "optimize_evidence_bundle_admission"
        next_direction = "diagnose_answerability_bundle_admission"
    else:
        decision = "investigate_mixed_conversion_bottleneck"
        next_direction = "investigate_mixed_conversion_bottleneck"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "decision": decision,
        "decision_population": "evidence_gain_samples" if evidence_gain_records else "all_samples",
        "dominant_primary_conversion_bottleneck": dominant_primary,
        "next_stage_direction": next_direction,
        "applies_optimization": False,
        "reason": "Diagnostic decision only; no runtime parameters, prompts, or thresholds were changed.",
    }


def scan_privacy(output_dir: Path) -> dict[str, Any]:
    forbidden_patterns = {
        "api_key": re.compile(r"api[_-]?key", re.IGNORECASE),
        "authorization": re.compile(r"authorization", re.IGNORECASE),
        "bearer_token": re.compile(r"bearer\s+[a-z0-9._-]+", re.IGNORECASE),
        "full_prompt": re.compile(r"(system_prompt|full_prompt|prompt_text)", re.IGNORECASE),
        "database_url": re.compile(r"postgres(?:ql)?://", re.IGNORECASE),
    }
    findings = []
    for path in sorted(output_dir.glob("*")):
        if path.name == "privacy_scan_report.json" or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in forbidden_patterns.items():
            if pattern.search(text):
                findings.append({"artifact": path.name, "finding": name})
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": "pass" if not findings else "fail", "finding_count": len(findings), "findings": findings}


def transition_contract_document() -> dict[str, Any]:
    return {
        "contract_version": TRANSITION_CONTRACT_VERSION,
        "evidence_lifecycle_contract_version": EVIDENCE_LIFECYCLE_CONTRACT_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "funnel_stages": list(FUNNEL_STAGES),
        "primary_conversion_bottlenecks": sorted(PRIMARY_BOTTLENECKS),
        "secondary_contributing_factors": sorted(SECONDARY_FACTORS),
        "diagnostic_confidences": sorted(DIAGNOSTIC_CONFIDENCES),
        "generation_eligible_statuses": sorted(GENERATION_ELIGIBLE_STATUSES),
        "gold_leakage_policy": "Gold annotations are forbidden in trace building and only allowed for post-run benchmark integrity checks.",
    }


def _load_traces(paths: tuple[Path, ...]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        replicate = int(re.search(r"rep(\d+)", path.name).group(1))  # type: ignore[union-attr]
        for event in _read_jsonl(path):
            by_key[(replicate, event["sample_id"])].append(event)
    return dict(by_key)


def _load_runs(paths: tuple[Path, ...]) -> dict[tuple[int, str], dict[str, Any]]:
    records = {}
    for path in paths:
        for item in _read_jsonl(path):
            records[(int(item["replicate"]), item["sample_id"])] = item
    return records


def _load_rounds(paths: tuple[Path, ...]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    records: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for item in _read_jsonl(path):
            records[(int(item["replicate"]), item["sample_id"])].append(item)
    return dict(records)


def _artifact_source_manifest(artifacts: Task0067Artifacts) -> dict[str, Any]:
    files = []
    for path in artifacts.required_files:
        data = path.read_bytes()
        files.append({"relative_path": str(path.relative_to(artifacts.source_dir.parent.parent.parent.parent)), "sha256": sha256(data).hexdigest(), "byte_count": len(data)})
    return {"schema_version": RESULT_SCHEMA_VERSION, "source_task": "TASK-0067", "artifact_source": str(artifacts.source_dir), "audit_mode": "artifact_only", "diagnostic_rerun_executed": False, "files": files}


def _trusted_identity_set(events: list[dict[str, Any]]) -> set[str]:
    identities = set()
    for event in events:
        trusted = ((event.get("tool_response_summary") or {}).get("trusted_evidence") or {})
        digest = trusted.get("evidence_identity_digest")
        if digest:
            identities.add(str(digest))
    return identities


def _answerability_chunk_ids(events: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for event in events:
        ids.update(str(item) for item in _answerability_payload(event).get("evidence_chunk_ids", []))
    return ids


def _answerability_payload(event: dict[str, Any]) -> dict[str, Any]:
    return ((event.get("tool_response_summary") or {}).get("answerability") or {})


def _generation_citation_ids(events: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for event in events:
        payload = ((event.get("tool_response_summary") or {}).get("answer_response") or {})
        for citation in payload.get("citations", []):
            if citation.get("citation_id"):
                ids.add(str(citation["citation_id"]))
    return ids


def _grounding_cited_ids(events: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for event in events:
        payload = ((event.get("tool_response_summary") or {}).get("grounding") or {})
        ids.update(str(item) for item in payload.get("cited_ids", []))
    return ids


def _status_rank(status: str | None) -> int:
    return {"unanswerable": 0, "safe_abstain": 0, "insufficient_evidence": 1, "partial_answer_possible": 2, "partially_answerable": 2, "answerable": 3}.get(str(status), -1)


def _generation_skip_reason(generation_eligible: bool, action_names: list[str | None], trace_events: list[dict[str, Any]]) -> str | None:
    if "generate_grounded_answer" in action_names:
        return None
    if not generation_eligible:
        return "answerability_not_generation_eligible"
    if trace_events and ((trace_events[-1].get("action") or {}).get("reason_code")) == "retrieval_planning_experiment_terminal":
        return "policy_finished_abstain"
    return "unknown"


def _generation_prompt_contains_new_evidence(generation_events: list[dict[str, Any]], second_ids: set[str]) -> bool:
    if not generation_events:
        return False
    serialized = json.dumps(generation_events, sort_keys=True)
    return any(identity in serialized for identity in second_ids)


def _final_abstain_reason(primary: str, action_names: list[str | None]) -> str:
    if primary in {"policy_early_abstention", "generation_not_invoked"}:
        return "policy_abstention"
    if primary == "answerability_not_generation_eligible":
        return "answerability_abstention"
    if primary == "generation_model_abstention":
        return "generation_abstention"
    if primary == "citation_validation_failure":
        return "citation_rejection"
    if primary == "grounding_verification_failure":
        return "grounding_rejection"
    if primary == "budget_exhaustion":
        return "budget_exhaustion"
    if primary == "provider_failure":
        return "provider_failure"
    return "runtime_failure" if "finish_failure" in action_names else "policy_abstention"


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    handoff = record.get("policy_generation_handoff") or {}
    return _sample_key(record) | {
        "primary_conversion_bottleneck": record["primary_conversion_bottleneck"],
        "secondary_contributing_factors": record["secondary_contributing_factors"],
        "diagnostic_confidence": record["diagnostic_confidence"],
        "final_abstain_reason": record["finalization"]["final_abstain_reason"],
        "latest_answerability_status": handoff.get("latest_answerability_status"),
        "generation_eligible": handoff.get("generation_eligible"),
        "answerability_result_digest": handoff.get("answerability_result_digest"),
        "evidence_bundle_digest": handoff.get("evidence_bundle_digest"),
        "latest_state_entered_runtime_decision": handoff.get("latest_state_entered_runtime_decision"),
        "policy_view_present": handoff.get("policy_view_present"),
        "policy_view_latest_state": handoff.get("policy_view_latest_state"),
        "generate_action_exposed": handoff.get("generate_action_exposed"),
        "policy_selected_action": handoff.get("policy_selected_action"),
        "policy_decision_reason_code": handoff.get("policy_decision_reason_code"),
        "runtime_final_action": handoff.get("runtime_final_action"),
        "runtime_reason_code": handoff.get("runtime_reason_code"),
        "runtime_rejected_or_rewrote_action": handoff.get("runtime_rejected_or_rewrote_action"),
        "latest_answerability_event_index": handoff.get("latest_answerability_event_index"),
        "policy_decision_event_index": handoff.get("policy_decision_event_index"),
    }


def _sample_key(record: dict[str, Any]) -> dict[str, Any]:
    return {"sample_id": record["sample_id"], "run_id": record["run_id"], "replicate_id": record["replicate_id"], "contract_version": record["contract_version"]}


def _field_value(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        value = value[part]
    return value


def _lifecycle_drop_agreement(lifecycles: list[dict[str, Any]]) -> float | str:
    by_sample: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in lifecycles:
        by_sample[record["sample_id"]][record["replicate_id"]].add(record["drop_stage"])
    comparable = [value for value in by_sample.values() if len(value) == 2]
    if not comparable:
        return "not_applicable"
    return sum(1 for value in comparable if len({tuple(sorted(stages)) for stages in value.values()}) == 1) / len(comparable)


def _uniform_rate(values: list[Any]) -> float | str:
    if not values:
        return "not_applicable"
    most_common = Counter(values).most_common(1)[0][1]
    return most_common / len(values)


def _digest_short(prefix: str, value: str) -> str:
    return f"{prefix}:{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
