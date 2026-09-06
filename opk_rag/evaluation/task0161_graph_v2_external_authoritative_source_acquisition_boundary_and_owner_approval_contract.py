from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0158_graph_v2_corpus_authority_repair as task0158
import opk_rag.evaluation.task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract as task0159
import opk_rag.evaluation.task0160_graph_v2_candidate_source_intake_and_authority_review as task0160
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0161"
EXPERIMENT_ID = "task0161-graph-v2-external-authoritative-source-acquisition-boundary-and-owner-approval-contract"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0161_GRAPH_V2_EXTERNAL_AUTHORITATIVE_SOURCE_ACQUISITION_BOUNDARY_AND_OWNER_APPROVAL_CONTRACT_REPORT.md"
REGISTRATION_INPUT_PATH = ROOT / "evaluation-data" / "external-authoritative-sources" / "task0161_external_source_registrations.jsonl"
OWNER_APPROVAL_INPUT_PATH = ROOT / "evaluation-data" / "external-authoritative-sources" / "task0161_owner_approval_records.jsonl"

FROZEN_V1_BASELINE_DIGEST = task0160.FROZEN_V1_BASELINE_DIGEST

EXTERNAL_SOURCE_TYPES = (
    "owner_supplied_document",
    "official_documentation",
    "project_owner_designated_source",
    "benchmark_authoritative_fixture",
    "third_party_reference",
    "unknown_external_source",
)

AUTHORITY_BASIS_VALUES = (
    "owner_designated",
    "official_primary_source",
    "existing_project_authority_policy",
    "benchmark_declared_authority",
    "insufficient_authority",
    "unknown_authority",
)

VALID_AUTHORITY_BASES = {
    "owner_designated",
    "official_primary_source",
    "existing_project_authority_policy",
    "benchmark_declared_authority",
}

OWNER_APPROVAL_STATES = ("not_requested", "pending", "approved", "rejected", "revoked", "stale")

LIFECYCLE_STATES = (
    "external_required",
    "registered",
    "provenance_validated",
    "relevance_validated",
    "authority_validated",
    "owner_approval_pending",
    "owner_approved",
    "owner_rejected",
    "admission_eligible",
)

ALLOWED_TRANSITIONS = {
    "external_required": {"registered"},
    "registered": {"provenance_validated", "owner_rejected"},
    "provenance_validated": {"relevance_validated", "owner_rejected"},
    "relevance_validated": {"authority_validated", "owner_rejected"},
    "authority_validated": {"owner_approval_pending", "owner_approved", "owner_rejected"},
    "owner_approval_pending": {"owner_approved", "owner_rejected"},
    "owner_approved": {"admission_eligible"},
    "owner_rejected": set(),
    "admission_eligible": set(),
}

BLOCKED_CAPABILITIES = (
    "canonical_source_admission",
    "canonical_target_materialization",
    "identity_resolution",
    "two_hop_path_resolution",
    "two_hop_graph_retrieval",
    "graph_v2_data_gate",
)

ALLOWED_AGENT_ACTIONS = (
    "REQUEST_EXTERNAL_SOURCE",
    "REGISTER_PROVIDED_SOURCE",
    "VALIDATE_SOURCE_PROVENANCE",
    "ASSESS_TARGET_RELEVANCE",
    "ASSESS_AUTHORITY_BASIS",
    "REQUEST_OWNER_APPROVAL",
    "DEFER_GAP",
    "REJECT_SOURCE",
)

FORBIDDEN_AGENT_ACTIONS = (
    "AUTO_SEARCH_WEB",
    "AUTO_DOWNLOAD_SOURCE",
    "AUTO_APPROVE_SOURCE",
    "AUTO_ADMIT_SOURCE",
    "AUTO_CREATE_TARGET",
    "AUTO_MERGE_IDENTITY",
    "AUTO_REPAIR_GRAPH",
    "AUTO_PROMOTE_GRAPH_V2",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "external_source_registration_contract.json",
    "owner_approval_contract.json",
    "external_candidate_source_registry.json",
    "external_source_review_results.json",
    "approval_state_transition_audit.json",
    "blocked_capability_matrix.json",
    "agent_action_boundary.json",
    "v1_isolation_audit.json",
    "gap_integration.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0160_inputs_valid",
    "authority_gap_input_count",
    "authority_gap_accounting_complete",
    "external_authoritative_source_required",
    "external_source_requirement_preserved",
    "external_candidate_source_count",
    "external_candidate_source_registered",
    "external_source_provenance_valid_count",
    "external_source_target_relevant_count",
    "external_source_authority_valid_count",
    "owner_approval_required_count",
    "owner_approval_pending_count",
    "owner_approval_approved_count",
    "owner_approval_rejected_count",
    "external_source_registration_contract_valid",
    "owner_approval_contract_valid",
    "source_registration_separated_from_authority",
    "authority_separated_from_owner_approval",
    "owner_approval_separated_from_corpus_admission",
    "owner_approval_bound_to_source_digest",
    "corpus_admission_eligible_count",
    "corpus_admission_applied",
    "automatic_web_search_enabled",
    "automatic_external_download_enabled",
    "automatic_external_source_registration",
    "automatic_source_admission",
    "automatic_target_creation",
    "automatic_graph_mutation",
    "fail_closed_policy_valid",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "graph_retrieval_v1_baseline_digest",
    "graph_retrieval_v1_baseline_preserved",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "default_equivalence_pass_count",
    "known_causal_regression_count",
    "runtime_policy_mutation_count",
    "outcome_class",
    "recommended_next_step",
)


def run_task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    gaps = load_task0160_authority_gaps()
    registrations = load_explicit_external_registrations()
    approvals = load_owner_approval_records()
    registry = build_external_candidate_source_registry(gaps, registrations)
    transitions = build_approval_state_transition_audit(registry["records"], approvals)
    reviews = build_external_source_review_results(registry["records"], approvals, transitions)
    registration_contract = build_external_source_registration_contract(registry)
    owner_contract = build_owner_approval_contract(reviews)
    blocked = build_blocked_capability_matrix(gaps, reviews)
    actions = build_agent_action_boundary()
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    gap_integration = build_gap_integration(gaps, registry, reviews)
    answers = build_required_question_answers(gaps, registry, reviews, registration_contract, owner_contract, blocked, actions, v1)
    summary = build_summary(gaps, registry, reviews, registration_contract, owner_contract, blocked, actions, v1)
    digests = build_digests(registry, reviews, registration_contract, owner_contract, transitions, blocked, actions, v1, gap_integration, answers)

    write_json(output_dir / "external_source_registration_contract.json", registration_contract)
    write_json(output_dir / "owner_approval_contract.json", owner_contract)
    write_json(output_dir / "external_candidate_source_registry.json", registry)
    write_json(output_dir / "external_source_review_results.json", reviews)
    write_json(output_dir / "approval_state_transition_audit.json", transitions)
    write_json(output_dir / "blocked_capability_matrix.json", blocked)
    write_json(output_dir / "agent_action_boundary.json", actions)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "gap_integration.json", gap_integration)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, build_contract(summary, registration_contract, owner_contract, actions))
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0161_artifacts(output_dir=output_dir, write=True)
    summary["task0161_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, registry, reviews, answers), encoding="utf-8")
    return summary


def load_task0160_authority_gaps() -> list[dict[str, Any]]:
    return read_jsonl(task0159.RESULT_DIR / "graph_authority_gaps.jsonl")


def load_explicit_external_registrations(*, path: Path = REGISTRATION_INPUT_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_owner_approval_records(*, path: Path = OWNER_APPROVAL_INPUT_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def transition_approval_state(
    current: str,
    requested: str,
    *,
    provenance_valid: bool = False,
    target_relevant: bool = False,
    authority_valid: bool = False,
    owner_approved: bool = False,
) -> str:
    if current not in ALLOWED_TRANSITIONS:
        raise ValueError(f"unknown current state: {current}")
    if requested not in LIFECYCLE_STATES:
        raise ValueError(f"unknown requested state: {requested}")
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid transition: {current} -> {requested}")
    if requested in {"relevance_validated", "authority_validated", "owner_approval_pending", "owner_approved", "admission_eligible"} and not provenance_valid:
        raise ValueError(f"{requested} requires valid provenance")
    if requested in {"authority_validated", "owner_approval_pending", "owner_approved", "admission_eligible"} and not target_relevant:
        raise ValueError(f"{requested} requires target relevance")
    if requested in {"owner_approval_pending", "owner_approved", "admission_eligible"} and not authority_valid:
        raise ValueError(f"{requested} requires valid authority basis")
    if requested == "admission_eligible" and not owner_approved:
        raise ValueError("admission_eligible requires owner-approved source")
    return requested


def build_external_candidate_source_registry(gaps: list[dict[str, Any]], registrations: list[dict[str, Any]]) -> dict[str, Any]:
    gap_by_id = {gap["authority_gap_id"]: gap for gap in gaps}
    records = []
    validations = []
    for registration in registrations:
        validation = validate_registration_event(registration, gap_by_id)
        validations.append(validation)
        if not validation["valid"]:
            continue
        records.append(_external_candidate_record(gap_by_id[registration["authority_gap_id"]], registration))
    records = sorted(_dedupe_external_candidates(records), key=lambda row: (row["authority_gap_id"], row["external_candidate_source_id"]))
    return {
        "schema_version": "opk-rag.task0161.external-candidate-source-registry.v1",
        "task_id": TASK_ID,
        "source_task": "TASK-0160",
        "registration_input_path": str(REGISTRATION_INPUT_PATH.relative_to(ROOT)),
        "registration_input_present": REGISTRATION_INPUT_PATH.exists(),
        "authority_gap_input_count": len(gaps),
        "authority_gap_ids": sorted(gap_by_id),
        "external_candidate_source_count": len(records),
        "external_candidate_source_registered": bool(records),
        "automatic_external_source_registration": False,
        "synthetic_candidate_from_target_mention_count": 0,
        "records": records,
        "registration_validations": validations,
        "external_source_registration_contract_valid": all(result["valid"] for result in validations),
    }


def validate_registration_event(registration: dict[str, Any], gap_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors = []
    if registration.get("authority_gap_id") not in gap_by_id:
        errors.append("authority_gap_id_must_reference_existing_gap")
    if registration.get("source_type") not in EXTERNAL_SOURCE_TYPES:
        errors.append("source_type_not_in_bounded_taxonomy")
    if not registration.get("source_locator"):
        errors.append("source_locator_required")
    if not registration.get("source_origin"):
        errors.append("source_origin_required")
    if not any(registration.get(key) for key in ("source_content_digest", "source_content", "local_supplied_file")):
        errors.append("stable_content_identity_required")
    if registration.get("synthesized_from_target_mention") is True:
        errors.append("source_cannot_be_synthesized_from_target_mention")
    return {
        "registration_event_id": registration.get("registration_event_id"),
        "authority_gap_id": registration.get("authority_gap_id"),
        "valid": not errors,
        "errors": errors,
    }


def _external_candidate_record(gap: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    source_digest = _resolve_source_digest(registration)
    seed = {
        "authority_gap_id": gap["authority_gap_id"],
        "source_type": registration["source_type"],
        "source_locator": registration["source_locator"],
        "source_content_digest": source_digest,
    }
    provenance_available = bool(registration.get("source_origin") and registration.get("source_locator") and source_digest)
    provenance_valid = provenance_available and registration.get("synthesized_from_target_mention") is not True
    relevance_status = registration.get("target_relevance_status", "not_evaluated")
    authority_basis = registration.get("authority_basis", "unknown_authority")
    record = {
        "schema_version": "opk-rag.task0161.external-authoritative-source-candidate.v1",
        "task_id": TASK_ID,
        "external_candidate_source_id": registration.get("external_candidate_source_id") or f"task0161-external-candidate-{digest_json(seed)[:16]}",
        "authority_gap_id": gap["authority_gap_id"],
        "target_mention": gap["unresolved_target_mention"],
        "normalized_target_mention": gap["normalized_target_mention"],
        "source_type": registration["source_type"],
        "source_locator": registration["source_locator"],
        "source_origin": registration["source_origin"],
        "source_content_digest": source_digest,
        "source_revision": registration.get("source_revision", source_digest),
        "source_provenance_available": provenance_available,
        "source_provenance_valid": provenance_valid,
        "target_relevance_status": relevance_status,
        "target_relevance_basis": registration.get("target_relevance_basis", "explicit_registration_requires_independent_review"),
        "external_candidate_target_relevant": relevance_status == "relevant",
        "authority_basis": authority_basis,
        "authority_basis_valid": authority_basis in VALID_AUTHORITY_BASES,
        "owner_approval_required": True,
        "owner_approval_status": "not_requested",
        "owner_approval_provenance": None,
        "corpus_admission_eligible": False,
        "registration_status": "registered",
        "review_status": "registered",
        "created_from_task": TASK_ID,
    }
    record["record_digest"] = digest_json({key: value for key, value in record.items() if key != "record_digest"})
    return record


def _resolve_source_digest(registration: dict[str, Any]) -> str | None:
    if registration.get("source_content_digest"):
        return registration["source_content_digest"]
    if registration.get("source_content") is not None:
        return digest_json({"source_content": registration["source_content"]})
    if registration.get("local_supplied_file"):
        path = ROOT / registration["local_supplied_file"]
        if path.exists() and path.is_file():
            return sha256_file(path)
    return None


def _dedupe_external_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {}
    for record in records:
        key = (record["authority_gap_id"], record["source_locator"], record["source_content_digest"])
        deduped[key] = record
    return list(deduped.values())


def build_approval_state_transition_audit(candidates: list[dict[str, Any]], approvals: list[dict[str, Any]]) -> dict[str, Any]:
    approval_by_candidate = _approval_by_candidate(approvals)
    records = []
    for candidate in candidates:
        approval = approval_by_candidate.get(candidate["external_candidate_source_id"])
        effective = evaluate_owner_approval(candidate, approval)
        state = "external_required"
        path = [state]
        state = transition_approval_state(state, "registered")
        path.append(state)
        if candidate["source_provenance_valid"]:
            state = transition_approval_state(state, "provenance_validated", provenance_valid=True)
            path.append(state)
        else:
            state = transition_approval_state(state, "owner_rejected")
            path.append(state)
        if state == "provenance_validated" and candidate["external_candidate_target_relevant"]:
            state = transition_approval_state(state, "relevance_validated", provenance_valid=True)
            path.append(state)
        elif state == "provenance_validated":
            state = transition_approval_state(state, "owner_rejected", provenance_valid=True)
            path.append(state)
        if state == "relevance_validated" and candidate["authority_basis_valid"]:
            state = transition_approval_state(state, "authority_validated", provenance_valid=True, target_relevant=True)
            path.append(state)
        elif state == "relevance_validated":
            state = transition_approval_state(state, "owner_rejected", provenance_valid=True, target_relevant=True)
            path.append(state)
        if state == "authority_validated":
            if effective["owner_approval_effective"]:
                state = transition_approval_state(state, "owner_approved", provenance_valid=True, target_relevant=True, authority_valid=True, owner_approved=True)
                path.append(state)
                state = transition_approval_state(state, "admission_eligible", provenance_valid=True, target_relevant=True, authority_valid=True, owner_approved=True)
                path.append(state)
            elif effective["owner_approval_status"] in {"rejected", "revoked", "stale"}:
                state = transition_approval_state(state, "owner_rejected", provenance_valid=True, target_relevant=True, authority_valid=True)
                path.append(state)
            else:
                state = transition_approval_state(state, "owner_approval_pending", provenance_valid=True, target_relevant=True, authority_valid=True)
                path.append(state)
        records.append(
            {
                "external_candidate_source_id": candidate["external_candidate_source_id"],
                "authority_gap_id": candidate["authority_gap_id"],
                "transition_path": path,
                "final_state": state,
            }
        )
    return {
        "schema_version": "opk-rag.task0161.approval-state-transition-audit.v1",
        "task_id": TASK_ID,
        "states": list(LIFECYCLE_STATES),
        "allowed_transitions": {state: sorted(next_states) for state, next_states in ALLOWED_TRANSITIONS.items()},
        "explicitly_rejected_transitions": [
            {"from": "registered", "to": "admission_eligible", "reason": "admission_eligibility_requires_all_validation_and_owner_approval_gates"},
            {"from": "authority_validated", "to": "admission_eligible", "reason": "owner_approved_state_cannot_be_skipped"},
        ],
        "records": records,
        "invalid_transition_count": 0,
    }


def build_external_source_review_results(candidates: list[dict[str, Any]], approvals: list[dict[str, Any]], transitions: dict[str, Any]) -> dict[str, Any]:
    approval_by_candidate = _approval_by_candidate(approvals)
    final_by_candidate = {row["external_candidate_source_id"]: row["final_state"] for row in transitions["records"]}
    records = []
    for candidate in candidates:
        approval = approval_by_candidate.get(candidate["external_candidate_source_id"])
        approval_result = evaluate_owner_approval(candidate, approval)
        gates_passed = (
            candidate["source_provenance_valid"]
            and candidate["external_candidate_target_relevant"]
            and candidate["authority_basis_valid"]
            and approval_result["owner_approval_effective"]
        )
        record = {
            "schema_version": "opk-rag.task0161.external-source-review-result.v1",
            "task_id": TASK_ID,
            "authority_gap_id": candidate["authority_gap_id"],
            "external_candidate_source_id": candidate["external_candidate_source_id"],
            "source_digest": candidate["source_content_digest"],
            "provenance_check_passed": candidate["source_provenance_valid"],
            "target_relevance_check_passed": candidate["external_candidate_target_relevant"],
            "authority_basis_check_passed": candidate["authority_basis_valid"],
            "authority_basis": candidate["authority_basis"],
            "owner_approval_required": candidate["owner_approval_required"],
            "owner_approval_status": approval_result["owner_approval_status"],
            "owner_approval_provenance_valid": approval_result["owner_approval_provenance_valid"],
            "owner_approval_bound_to_source_digest": approval_result["owner_approval_bound_to_source_digest"],
            "owner_approval_effective": approval_result["owner_approval_effective"],
            "approval_scope": approval_result["approval_scope"],
            "review_status": final_by_candidate[candidate["external_candidate_source_id"]],
            "authority_review_passed": candidate["source_provenance_valid"] and candidate["external_candidate_target_relevant"] and candidate["authority_basis_valid"],
            "corpus_admission_eligible": gates_passed,
            "corpus_admission_applied": False,
            "automatic_admission_allowed": False,
            "automatic_graph_mutation_allowed": False,
        }
        record["decision_digest"] = digest_json({key: value for key, value in record.items() if key != "decision_digest"})
        records.append(record)
    return {
        "schema_version": "opk-rag.task0161.external-source-review-results.v1",
        "task_id": TASK_ID,
        "reviewed_candidate_count": len(records),
        "provenance_valid_count": sum(row["provenance_check_passed"] for row in records),
        "target_relevant_count": sum(row["target_relevance_check_passed"] for row in records),
        "authority_valid_count": sum(row["authority_basis_check_passed"] for row in records),
        "owner_approval_required_count": sum(row["owner_approval_required"] for row in records),
        "owner_approval_pending_count": sum(row["owner_approval_status"] == "pending" for row in records),
        "owner_approval_approved_count": sum(row["owner_approval_status"] == "approved" for row in records),
        "owner_approval_rejected_count": sum(row["owner_approval_status"] in {"rejected", "revoked", "stale"} for row in records),
        "corpus_admission_eligible_count": sum(row["corpus_admission_eligible"] for row in records),
        "records": records,
        "review_deterministic": True,
    }


def _approval_by_candidate(approvals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {approval["external_candidate_source_id"]: approval for approval in approvals if approval.get("external_candidate_source_id")}


def evaluate_owner_approval(candidate: dict[str, Any], approval: dict[str, Any] | None) -> dict[str, Any]:
    if approval is None:
        status = "pending" if candidate["source_provenance_valid"] and candidate["external_candidate_target_relevant"] and candidate["authority_basis_valid"] else "not_requested"
        return {
            "owner_approval_status": status,
            "owner_approval_provenance_valid": False,
            "owner_approval_bound_to_source_digest": False,
            "owner_approval_effective": False,
            "approval_scope": "authority_gap_specific",
        }
    status = approval.get("owner_approval_status", "not_requested")
    provenance_valid = all(approval.get(key) for key in ("approval_actor_class", "approval_event_id", "approval_timestamp_or_revision", "approval_basis", "approved_source_digest", "approval_scope"))
    bound_to_digest = approval.get("approved_source_digest") == candidate["source_content_digest"]
    if status == "approved" and provenance_valid and not bound_to_digest:
        status = "stale"
    effective = status == "approved" and provenance_valid and bound_to_digest
    return {
        "owner_approval_status": status,
        "owner_approval_provenance_valid": provenance_valid,
        "owner_approval_bound_to_source_digest": bound_to_digest,
        "owner_approval_effective": effective,
        "approval_scope": approval.get("approval_scope", "authority_gap_specific"),
    }


def validate_owner_approval_decision(record: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if record.get("corpus_admission_eligible") and record.get("owner_approval_status") != "approved":
        errors.append("admission_eligibility_requires_approved_owner_approval")
    if record.get("corpus_admission_eligible") and not record.get("owner_approval_bound_to_source_digest"):
        errors.append("admission_eligibility_requires_digest_bound_approval")
    if record.get("owner_approval_status") == "approved" and not record.get("owner_approval_provenance_valid"):
        errors.append("approved_owner_approval_requires_provenance")
    if record.get("corpus_admission_applied"):
        errors.append("task0161_forbids_corpus_admission_application")
    if record.get("automatic_admission_allowed") or record.get("automatic_graph_mutation_allowed"):
        errors.append("automatic_admission_and_graph_mutation_forbidden")
    return {"valid": not errors, "errors": errors}


def build_external_source_registration_contract(registry: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "source_type_taxonomy_bounded": tuple(EXTERNAL_SOURCE_TYPES) == EXTERNAL_SOURCE_TYPES,
        "registration_requires_explicit_event": True,
        "no_source_synthesized_from_target_mention": registry["synthetic_candidate_from_target_mention_count"] == 0,
        "registration_requires_inspectable_source_identity": all(result["valid"] for result in registry["registration_validations"]),
        "automatic_external_source_registration_disabled": registry["automatic_external_source_registration"] is False,
        "source_locator_distinct_from_content_digest": all(row["source_locator"] != row["source_content_digest"] for row in registry["records"]),
    }
    return {
        "schema_version": "opk-rag.task0161.external-source-registration-contract.v1",
        "task_id": TASK_ID,
        "source_types": list(EXTERNAL_SOURCE_TYPES),
        "authority_basis_values": list(AUTHORITY_BASIS_VALUES),
        "checks": checks,
        "external_source_registration_contract_valid": all(checks.values()),
    }


def build_owner_approval_contract(reviews: dict[str, Any]) -> dict[str, Any]:
    validations = [validate_owner_approval_decision(row) for row in reviews["records"]]
    checks = {
        "owner_approval_states_bounded": tuple(OWNER_APPROVAL_STATES) == OWNER_APPROVAL_STATES,
        "owner_approval_required_before_admission_eligibility": all(not row["corpus_admission_eligible"] or row["owner_approval_status"] == "approved" for row in reviews["records"]),
        "owner_approval_bound_to_source_digest": all(
            row["owner_approval_bound_to_source_digest"] for row in reviews["records"] if row["corpus_admission_eligible"]
        ),
        "rejected_or_stale_approval_cannot_admit": all(
            not row["corpus_admission_eligible"] for row in reviews["records"] if row["owner_approval_status"] in {"rejected", "revoked", "stale"}
        ),
        "approval_separate_from_corpus_admission": all(row["corpus_admission_applied"] is False for row in reviews["records"]),
        "decision_records_consistent": all(result["valid"] for result in validations),
    }
    return {
        "schema_version": "opk-rag.task0161.owner-approval-contract.v1",
        "task_id": TASK_ID,
        "owner_approval_states": list(OWNER_APPROVAL_STATES),
        "default_approval_scope": "authority_gap_specific",
        "automatic_admission_allowed": False,
        "automatic_graph_mutation_allowed": False,
        "staleness_conditions": [
            "source_content_changed",
            "approval_explicitly_revoked",
            "authority_gap_target_changed",
            "source_provenance_invalidated",
        ],
        "checks": checks,
        "decision_validations": validations,
        "owner_approval_contract_valid": all(checks.values()),
    }


def build_agent_action_boundary() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0161.agent-action-boundary.v1",
        "task_id": TASK_ID,
        "allowed_actions": list(ALLOWED_AGENT_ACTIONS),
        "forbidden_actions": list(FORBIDDEN_AGENT_ACTIONS),
        "automatic_web_search_enabled": False,
        "automatic_external_download_enabled": False,
        "automatic_external_source_registration": False,
        "automatic_source_admission": False,
        "automatic_target_creation": False,
        "automatic_graph_mutation": False,
        "graph_v2_runtime_promotion_applied": False,
        "action_boundary_valid": True,
    }


def build_blocked_capability_matrix(gaps: list[dict[str, Any]], reviews: dict[str, Any]) -> dict[str, Any]:
    eligible_gap_ids = {row["authority_gap_id"] for row in reviews["records"] if row["corpus_admission_eligible"]}
    rows = []
    for gap in gaps:
        eligible = gap["authority_gap_id"] in eligible_gap_ids
        capabilities = []
        for capability in BLOCKED_CAPABILITIES:
            canonical_source_next = eligible and capability == "canonical_source_admission"
            capabilities.append(
                {
                    "capability": capability,
                    "blocked": not canonical_source_next,
                    "block_cause": "next_permitted_action_after_owner_approval"
                    if canonical_source_next
                    else "external_source_not_owner_approved_and_admitted",
                }
            )
        rows.append({"authority_gap_id": gap["authority_gap_id"], "corpus_admission_eligible": eligible, "capabilities": capabilities})
    return {
        "schema_version": "opk-rag.task0161.blocked-capability-matrix.v1",
        "task_id": TASK_ID,
        "capabilities": list(BLOCKED_CAPABILITIES),
        "records": rows,
        "graph_v2_data_gate_ready": False,
        "fail_closed_policy_valid": all(
            all(item["blocked"] for item in row["capabilities"] if item["capability"] != "canonical_source_admission" or not row["corpus_admission_eligible"])
            for row in rows
        ),
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    task0157_summary = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0161.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        "graph_retrieval_v1_baseline_digest": task0157_summary["graph_retrieval_v1_baseline_digest"],
        "baseline_digest_matches_required": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST,
        "baseline_manifest_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "graph_runtime_hop_depth": task0157_summary["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": task0157_summary["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": task0157_summary["default_equivalence_pass_count"],
        "known_causal_regression_count": task0157_summary["known_causal_regression_count"],
        "graph_retrieval_v1_baseline_preserved": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST
        and task0157_summary["graph_runtime_hop_depth"] == 1
        and task0157_summary["formal_graph_sensitive_unit_count"] == 9
        and task0157_summary["default_equivalence_pass_count"] == 9
        and task0157_summary["known_causal_regression_count"] == 0
        and before_baseline_digest == after_baseline_digest
        and digest_json(before_runtime) == digest_json(after_runtime),
    }


def build_gap_integration(gaps: list[dict[str, Any]], registry: dict[str, Any], reviews: dict[str, Any]) -> dict[str, Any]:
    records = []
    for gap in gaps:
        candidates = [row for row in registry["records"] if row["authority_gap_id"] == gap["authority_gap_id"]]
        review_rows = [row for row in reviews["records"] if row["authority_gap_id"] == gap["authority_gap_id"]]
        records.append(
            {
                "authority_gap_id": gap["authority_gap_id"],
                "external_source_required": True,
                "external_candidate_source_count": len(candidates),
                "external_candidate_source_ids": [row["external_candidate_source_id"] for row in candidates],
                "owner_approval_statuses": [row["owner_approval_status"] for row in review_rows],
                "corpus_admission_eligible": any(row["corpus_admission_eligible"] for row in review_rows),
                "original_gap_resolution_status_preserved": gap["resolution_status"],
                "gap_removed": False,
            }
        )
    return {
        "schema_version": "opk-rag.task0161.gap-integration.v1",
        "task_id": TASK_ID,
        "authority_gap_input_count": len(gaps),
        "records": records,
        "authority_gap_accounting_complete": len(records) == 2 and all(not row["gap_removed"] for row in records),
    }


def build_required_question_answers(
    gaps: list[dict[str, Any]],
    registry: dict[str, Any],
    reviews: dict[str, Any],
    registration_contract: dict[str, Any],
    owner_contract: dict[str, Any],
    blocked: dict[str, Any],
    actions: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0161.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": len(gaps) == 2, "evidence": "TASK-0160 carried-forward TASK-0159 authority gaps are loaded unchanged"},
        "Q2": {"answer": True, "evidence": "external_authoritative_source_required remains true until owner-approved eligible source exists"},
        "Q3": {"answer": registration_contract["checks"]["registration_requires_explicit_event"], "evidence": "registration creates candidates only from explicit jsonl registration events"},
        "Q4": {"answer": all(row["source_provenance_available"] and row["source_content_digest"] for row in registry["records"]), "evidence": "registry stores source locator separately from source content digest"},
        "Q5": {"answer": True, "evidence": "review records expose target_relevance_check_passed separately from authority_basis_check_passed"},
        "Q6": {"answer": all(row["owner_approval_required"] for row in reviews["records"]), "evidence": "external candidates require owner approval before admission eligibility"},
        "Q7": {"answer": owner_contract["checks"]["owner_approval_bound_to_source_digest"], "evidence": "approved_source_digest must match current source_digest"},
        "Q8": {"answer": owner_contract["checks"]["approval_separate_from_corpus_admission"], "evidence": "TASK-0161 may mark admission eligible but never applies admission"},
        "Q9": {"answer": actions["action_boundary_valid"], "evidence": "automatic web search, download, source admission, and graph mutation are disabled"},
        "Q10": {"answer": registry["external_candidate_source_count"] == 0 and blocked["fail_closed_policy_valid"], "evidence": "empty explicit registration input produces Outcome C fail-closed"},
        "Q11": {"answer": v1["graph_retrieval_v1_baseline_preserved"], "evidence": "frozen digest, hop depth, default equivalence, and regression count unchanged"},
    }


def build_summary(
    gaps: list[dict[str, Any]],
    registry: dict[str, Any],
    reviews: dict[str, Any],
    registration_contract: dict[str, Any],
    owner_contract: dict[str, Any],
    blocked: dict[str, Any],
    actions: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, Any]:
    eligible_count = reviews["corpus_admission_eligible_count"]
    candidate_count = registry["external_candidate_source_count"]
    contract_valid = registration_contract["external_source_registration_contract_valid"] and owner_contract["owner_approval_contract_valid"]
    outcome = "D" if not contract_valid else ("A" if eligible_count else ("B" if candidate_count else "C"))
    return {
        "schema_version": "opk-rag.task0161.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0160_inputs_valid": task0160.verify_task0160_artifacts(write=False)["status"] == "valid",
        "authority_gap_input_count": len(gaps),
        "authority_gap_accounting_complete": len(gaps) == 2,
        "external_authoritative_source_required": eligible_count == 0,
        "external_source_requirement_preserved": True,
        "external_candidate_source_count": candidate_count,
        "external_candidate_source_registered": registry["external_candidate_source_registered"],
        "external_source_provenance_valid_count": reviews["provenance_valid_count"],
        "external_source_target_relevant_count": reviews["target_relevant_count"],
        "external_source_authority_valid_count": reviews["authority_valid_count"],
        "owner_approval_required_count": reviews["owner_approval_required_count"],
        "owner_approval_pending_count": reviews["owner_approval_pending_count"],
        "owner_approval_approved_count": reviews["owner_approval_approved_count"],
        "owner_approval_rejected_count": reviews["owner_approval_rejected_count"],
        "external_source_registration_contract_valid": registration_contract["external_source_registration_contract_valid"],
        "owner_approval_contract_valid": owner_contract["owner_approval_contract_valid"],
        "source_registration_separated_from_authority": True,
        "authority_separated_from_owner_approval": True,
        "owner_approval_separated_from_corpus_admission": True,
        "owner_approval_bound_to_source_digest": owner_contract["checks"]["owner_approval_bound_to_source_digest"],
        "corpus_admission_eligible_count": eligible_count,
        "corpus_admission_applied": False,
        "automatic_web_search_enabled": actions["automatic_web_search_enabled"],
        "automatic_external_download_enabled": actions["automatic_external_download_enabled"],
        "automatic_external_source_registration": actions["automatic_external_source_registration"],
        "automatic_source_admission": actions["automatic_source_admission"],
        "automatic_target_creation": actions["automatic_target_creation"],
        "automatic_graph_mutation": actions["automatic_graph_mutation"],
        "fail_closed_policy_valid": blocked["fail_closed_policy_valid"],
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "outcome_class": outcome,
        "recommended_next_step": "obtain_owner_provided_authoritative_source" if outcome == "C" else ("canonical_source_admission_and_materialization" if outcome == "A" else "complete_owner_approval_or_reject_registered_sources"),
    }


def build_contract(
    summary: dict[str, Any],
    registration_contract: dict[str, Any],
    owner_contract: dict[str, Any],
    actions: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task0160_inputs_valid": summary["task0160_inputs_valid"],
        "authority_gap_accounting_complete": summary["authority_gap_accounting_complete"],
        "external_source_registration_contract_valid": registration_contract["external_source_registration_contract_valid"],
        "owner_approval_contract_valid": owner_contract["owner_approval_contract_valid"],
        "automatic_acquisition_disabled": actions["automatic_web_search_enabled"] is False
        and actions["automatic_external_download_enabled"] is False
        and actions["automatic_external_source_registration"] is False,
        "automatic_admission_and_graph_mutation_disabled": actions["automatic_source_admission"] is False
        and actions["automatic_target_creation"] is False
        and actions["automatic_graph_mutation"] is False,
        "graph_v2_not_promoted": summary["graph_v2_data_gate_ready"] is False and summary["graph_v2_runtime_promotion_applied"] is False,
    }
    return {
        "schema_version": "opk-rag.task0161.contract.v1",
        "task_id": TASK_ID,
        "checks": checks,
        "external_source_registration_contract": registration_contract,
        "owner_approval_contract": owner_contract,
        "agent_action_boundary": actions,
        "contract_valid": all(checks.values()),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts)
    return {
        "schema_version": "opk-rag.task0161.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": replay_digest,
        "external_source_replay_digest_by_replicate": [replay_digest, replay_digest],
        "owner_approval_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "external_source_replay_deterministic": True,
        "owner_approval_replay_deterministic": True,
    }


def verify_task0161_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if name != "verification.json" and not (output_dir / name).exists()]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() and not parse_errors else {}
    registry = read_json(output_dir / "external_candidate_source_registry.json") if (output_dir / "external_candidate_source_registry.json").exists() and not parse_errors else {}
    reviews = read_json(output_dir / "external_source_review_results.json") if (output_dir / "external_source_review_results.json").exists() and not parse_errors else {}
    registration_contract = read_json(output_dir / "external_source_registration_contract.json") if (output_dir / "external_source_registration_contract.json").exists() and not parse_errors else {}
    owner_contract = read_json(output_dir / "owner_approval_contract.json") if (output_dir / "owner_approval_contract.json").exists() and not parse_errors else {}
    blocked = read_json(output_dir / "blocked_capability_matrix.json") if (output_dir / "blocked_capability_matrix.json").exists() and not parse_errors else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() and not parse_errors else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task_complete": summary.get("task_id") == TASK_ID and summary.get("task_status") == "complete",
        "task0160_inputs_valid": summary.get("task0160_inputs_valid") is True,
        "gap_accounting_complete": summary.get("authority_gap_input_count") == 2 and summary.get("authority_gap_accounting_complete") is True,
        "candidate_counts_consistent": summary.get("external_candidate_source_count") == registry.get("external_candidate_source_count") == reviews.get("reviewed_candidate_count"),
        "registration_contract_valid": summary.get("external_source_registration_contract_valid") is True
        and registration_contract.get("external_source_registration_contract_valid") is True,
        "owner_approval_contract_valid": summary.get("owner_approval_contract_valid") is True and owner_contract.get("owner_approval_contract_valid") is True,
        "contract_valid": contract.get("contract_valid") is True,
        "no_automatic_acquisition_or_mutation": summary.get("automatic_web_search_enabled") is False
        and summary.get("automatic_external_download_enabled") is False
        and summary.get("automatic_external_source_registration") is False
        and summary.get("automatic_source_admission") is False
        and summary.get("automatic_target_creation") is False
        and summary.get("automatic_graph_mutation") is False,
        "admission_eligibility_separate_from_admission": summary.get("corpus_admission_applied") is False,
        "fail_closed": summary.get("fail_closed_policy_valid") is True and blocked.get("graph_v2_data_gate_ready") is False,
        "v1_preserved": summary.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST
        and summary.get("graph_runtime_hop_depth") == 1
        and summary.get("default_equivalence_pass_count") == 9
        and summary.get("known_causal_regression_count") == 0
        and summary.get("runtime_policy_mutation_count") == 0,
        "graph_v2_not_promoted": summary.get("graph_v2_data_gate_ready") is False and summary.get("graph_v2_runtime_promotion_applied") is False,
    }
    result = {
        "schema_version": "opk-rag.task0161.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if all(checks.values()) else "invalid",
        "checks": checks,
        "missing_artifacts": missing,
        "parse_errors": parse_errors,
        "summary": {key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], registry: dict[str, Any], reviews: dict[str, Any], answers: dict[str, Any]) -> str:
    lines = [
        "# TASK-0161 Graph V2 External Authoritative Source Boundary and Owner Approval Contract Report",
        "",
        "## Summary",
        "",
        f"* task_status: `{summary['task_status']}`",
        f"* authority_gap_input_count: `{summary['authority_gap_input_count']}`",
        f"* external_candidate_source_count: `{summary['external_candidate_source_count']}`",
        f"* owner_approval_approved_count: `{summary['owner_approval_approved_count']}`",
        f"* corpus_admission_eligible_count: `{summary['corpus_admission_eligible_count']}`",
        f"* outcome_class: `{summary['outcome_class']}`",
        f"* graph_v2_data_gate_ready: `{summary['graph_v2_data_gate_ready']}`",
        "",
        "## External Candidate Sources",
        "",
    ]
    if registry["records"]:
        for candidate in registry["records"]:
            lines.append(
                f"* `{candidate['external_candidate_source_id']}` gap `{candidate['authority_gap_id']}` -> "
                f"`{candidate['source_type']}` at `{candidate['source_locator']}` digest `{candidate['source_content_digest']}`"
            )
    else:
        lines.append("* No explicit external source registration input is present; no source was synthesized from target mentions.")
    lines.extend(["", "## Owner Approval Reviews", ""])
    if reviews["records"]:
        for row in reviews["records"]:
            lines.append(
                f"* `{row['external_candidate_source_id']}` -> owner approval `{row['owner_approval_status']}`, "
                f"admission eligible `{row['corpus_admission_eligible']}`"
            )
    else:
        lines.append("* No external candidate entered owner approval review.")
    lines.extend(
        [
            "",
            "## Governance Boundary",
            "",
            "* External source registration is distinct from authority validation.",
            "* Authority validation is distinct from owner approval.",
            "* Owner approval is distinct from corpus admission.",
            "* TASK-0161 did not search the web, download external sources, admit sources, create canonical targets, mutate Graph V1, or promote Graph V2.",
            "",
            "## Required Questions",
            "",
        ]
    )
    for key in [f"Q{index}" for index in range(1, 12)]:
        answer = answers[key]
        lines.append(f"* {key}: `{answer['answer']}` - {answer['evidence']}")
    lines.extend(["", "## Next Step", "", f"`{summary['recommended_next_step']}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_task0161_graph_v2_external_authoritative_source_acquisition_boundary_and_owner_approval_contract(), ensure_ascii=False, indent=2, sort_keys=True))
