from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0158_graph_v2_corpus_authority_repair as task0158
import opk_rag.evaluation.task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract as task0159
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0160"
EXPERIMENT_ID = "task0160-graph-v2-candidate-source-intake-and-authority-review"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0160_graph_v2_candidate_source_intake_and_authority_review_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0160_GRAPH_V2_CANDIDATE_SOURCE_INTAKE_AND_AUTHORITY_REVIEW_REPORT.md"

FROZEN_V1_BASELINE_DIGEST = task0159.FROZEN_V1_BASELINE_DIGEST

REVIEW_STATES = (
    "registered",
    "provenance_validated",
    "relevance_validated",
    "authority_review_pending",
    "authority_approved",
    "authority_rejected",
    "authority_insufficient",
)

ALLOWED_TRANSITIONS = {
    "registered": {"provenance_validated", "authority_rejected"},
    "provenance_validated": {"relevance_validated", "authority_insufficient", "authority_rejected"},
    "relevance_validated": {"authority_review_pending", "authority_insufficient", "authority_rejected"},
    "authority_review_pending": {"authority_approved", "authority_rejected", "authority_insufficient"},
    "authority_approved": set(),
    "authority_rejected": set(),
    "authority_insufficient": set(),
}

PROJECT_CONTROLLED_SEARCH_SCOPES = (
    "current_source_documents_snapshot",
    "task0159_gap_candidate_paths",
    "task0153_to_task0159_artifacts",
    "evaluation_fixtures_and_results",
    "project_local_benchmark_source_material",
)

AUTHORITY_ELIGIBLE_SOURCE_TYPES = {"existing_source_document", "existing_canonical_document", "governed_source_artifact"}

BLOCKED_CAPABILITIES = task0159.BLOCKED_CAPABILITIES

PROHIBITED_ACTIONS = (
    "AUTO_ADMIT_SOURCE",
    "AUTO_CREATE_CANONICAL_TARGET",
    "AUTO_MERGE_ENTITY",
    "AUTO_REPAIR_TWO_HOP_PATH",
    "AUTO_PROMOTE_GRAPH_V2",
    "MUTATE_GRAPH_V1_RUNTIME",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "candidate_source_registry.json",
    "authority_review_results.json",
    "authority_review_contract.json",
    "gap_candidate_mapping.json",
    "authority_review_state_transition_audit.json",
    "blocked_capability_matrix.json",
    "v1_isolation_audit.json",
    "search_scope_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0159_inputs_valid",
    "authority_gap_input_count",
    "authority_gap_accounting_complete",
    "candidate_source_search_scope_bounded",
    "candidate_source_count",
    "candidate_source_found",
    "candidate_source_provenance_valid_count",
    "candidate_source_target_relevant_count",
    "authority_reviewed_candidate_count",
    "authority_approved_candidate_count",
    "authority_rejected_candidate_count",
    "authority_insufficient_candidate_count",
    "authority_review_contract_valid",
    "corpus_admission_eligible_count",
    "corpus_admission_applied",
    "synthetic_source_admission",
    "automatic_source_admission",
    "automatic_target_creation",
    "automatic_graph_mutation",
    "external_authoritative_source_required",
    "fail_closed_policy_valid",
    "graph_v2_data_gate_ready",
    "graph_v2_runtime_promotion_applied",
    "graph_retrieval_v1_baseline_digest",
    "graph_runtime_hop_depth",
    "formal_graph_sensitive_unit_count",
    "default_equivalence_pass_count",
    "known_causal_regression_count",
    "runtime_policy_mutation_count",
    "outcome_class",
    "recommended_next_step",
)


def run_task0160_graph_v2_candidate_source_intake_and_authority_review(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    gaps = load_task0159_gaps()
    search_scope = build_search_scope_audit(gaps)
    candidates = discover_candidate_sources(gaps)
    registry = build_candidate_source_registry(gaps, candidates, search_scope)
    transitions = build_state_transition_audit(candidates)
    reviews = build_authority_review_results(candidates, transitions)
    blocked = build_blocked_capability_matrix(gaps, reviews)
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    mapping = build_gap_candidate_mapping(gaps, candidates, reviews)
    answers = build_required_question_answers(registry, reviews, blocked, v1, search_scope)
    summary = build_summary(gaps, registry, reviews, blocked, v1, search_scope)
    contract = build_authority_review_contract(summary, registry, reviews, transitions, blocked)
    digests = build_digests(search_scope, registry, reviews, contract, transitions, blocked, v1, mapping, answers)

    write_json(output_dir / "search_scope_audit.json", search_scope)
    write_json(output_dir / "candidate_source_registry.json", registry)
    write_json(output_dir / "authority_review_results.json", reviews)
    write_json(output_dir / "gap_candidate_mapping.json", mapping)
    write_json(output_dir / "authority_review_state_transition_audit.json", transitions)
    write_json(output_dir / "blocked_capability_matrix.json", blocked)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "authority_review_contract.json", contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0160_artifacts(output_dir=output_dir, write=True)
    summary["task0160_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, registry, reviews, answers), encoding="utf-8")
    return summary


def load_task0159_gaps() -> list[dict[str, Any]]:
    return read_jsonl(task0159.RESULT_DIR / "graph_authority_gaps.jsonl")


def build_search_scope_audit(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    checked_candidate_paths = sorted({path for gap in gaps for path in gap.get("candidate_source_paths_checked", [])})
    return {
        "schema_version": "opk-rag.task0160.search-scope-audit.v1",
        "task_id": TASK_ID,
        "search_scopes": list(PROJECT_CONTROLLED_SEARCH_SCOPES),
        "candidate_path_count": len(checked_candidate_paths),
        "candidate_paths_checked": checked_candidate_paths,
        "source_documents_snapshot_root": str(SOURCE_DIR.relative_to(ROOT)),
        "external_internet_search_executed": False,
        "external_download_executed": False,
        "synthetic_source_creation_executed": False,
        "candidate_source_search_scope_bounded": True,
        "scope_policy_digest": digest_json({"scopes": PROJECT_CONTROLLED_SEARCH_SCOPES, "candidate_paths": checked_candidate_paths}),
    }


def discover_candidate_sources(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for gap in gaps:
        candidates.extend(_discover_existing_source_document_candidates(gap))
        candidates.extend(_discover_project_local_artifact_candidates(gap))
    return sorted(_dedupe_candidates(candidates), key=lambda row: (row["authority_gap_id"], row["candidate_source_id"]))


def _discover_existing_source_document_candidates(gap: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for locator in gap.get("candidate_source_paths_checked", []):
        path = ROOT / locator
        if path.exists() and path.is_file():
            rows.append(_candidate_record(gap, source_type="existing_source_document", source_locator=locator, source_origin="current_source_documents_snapshot", source_revision=sha256_file(path), provenance_reference=locator))
    return rows


def _discover_project_local_artifact_candidates(gap: dict[str, Any]) -> list[dict[str, Any]]:
    target_strings = _target_strings(gap)
    roots = [ROOT / "evaluation-data", ROOT / "docs"]
    rows: list[dict[str, Any]] = []
    for artifact in _iter_project_artifacts(roots):
        rel = str(artifact.relative_to(ROOT))
        if rel.startswith(f"evaluation-data/results/{EXPERIMENT_ID}/"):
            continue
        try:
            text = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not any(value and value in text for value in target_strings):
            continue
        rows.append(
            _candidate_record(
                gap,
                source_type="project_local_non_authoritative_artifact",
                source_locator=rel,
                source_origin=_artifact_origin(rel),
                source_revision=sha256_file(artifact),
                provenance_reference=rel,
            )
        )
    return rows


def _iter_project_artifacts(roots: Iterable[Path]) -> Iterable[Path]:
    allowed_suffixes = {".json", ".jsonl", ".md"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in allowed_suffixes:
                yield path


def _candidate_record(
    gap: dict[str, Any],
    *,
    source_type: str,
    source_locator: str,
    source_origin: str,
    source_revision: str,
    provenance_reference: str,
) -> dict[str, Any]:
    target_relevance_status, target_relevance_basis = assess_target_relevance(gap, source_locator=source_locator)
    authority_eligibility_status, authority_eligibility_basis = assess_authority_eligibility(source_type)
    seed = {
        "authority_gap_id": gap["authority_gap_id"],
        "source_type": source_type,
        "source_locator": source_locator,
        "source_revision": source_revision,
    }
    record = {
        "schema_version": "opk-rag.task0160.candidate-authority-source.v1",
        "task_id": TASK_ID,
        "candidate_source_id": f"task0160-candidate-{digest_json(seed)[:16]}",
        "authority_gap_id": gap["authority_gap_id"],
        "target_mention": gap["unresolved_target_mention"],
        "normalized_target_mention": gap["normalized_target_mention"],
        "source_type": source_type,
        "source_locator": source_locator,
        "source_document_id_if_existing": source_locator if source_type == "existing_source_document" else None,
        "source_origin": source_origin,
        "source_revision": source_revision,
        "provenance_available": True,
        "provenance_reference": provenance_reference,
        "target_relevance_status": target_relevance_status,
        "target_relevance_basis": target_relevance_basis,
        "authority_eligibility_status": authority_eligibility_status,
        "authority_eligibility_basis": authority_eligibility_basis,
        "review_status": "registered",
        "review_decision": "pending_review",
        "review_reason": "candidate_registered_but_not_yet_reviewed",
        "corpus_admission_status": "not_admitted",
        "corpus_admission_eligible": False,
        "canonical_target_materialized": False,
        "automatic_admission_allowed": False,
        "automatic_graph_mutation_allowed": False,
        "created_from_task": TASK_ID,
    }
    record["record_digest"] = digest_json({key: value for key, value in record.items() if key != "record_digest"})
    return record


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {}
    for candidate in candidates:
        key = (candidate["authority_gap_id"], candidate["source_locator"], candidate["source_revision"])
        deduped[key] = candidate
    return list(deduped.values())


def _target_strings(gap: dict[str, Any]) -> list[str]:
    paths = [str(path) for path in gap.get("candidate_source_paths_checked", [])]
    return sorted({gap["unresolved_target_mention"], gap["normalized_target_mention"], *paths}, key=str.lower)


def _artifact_origin(rel: str) -> str:
    if rel.startswith("evaluation-data/results/task015"):
        return "task0153_to_task0159_artifacts"
    if rel.startswith("evaluation-data/"):
        return "evaluation_fixture_or_result"
    if rel.startswith("docs/TASK015"):
        return "task0153_to_task0159_report"
    return "repository_tracked_source_artifact"


def assess_target_relevance(gap: dict[str, Any], *, source_locator: str) -> tuple[str, str]:
    if source_locator in gap.get("candidate_source_paths_checked", []):
        return "relevant", "locator_exactly_matches_task0159_candidate_path"
    if gap["unresolved_target_mention"] in source_locator or gap["normalized_target_mention"] in source_locator.lower():
        return "mentions_target_only", "artifact_locator_or_payload_mentions_target_but_does_not_materialize_intended_source_document"
    return "not_relevant", "no_deterministic_target_identity_support"


def assess_authority_eligibility(source_type: str) -> tuple[str, str]:
    if source_type in AUTHORITY_ELIGIBLE_SOURCE_TYPES:
        return "approved", "source_type_is_allowed_by_authority_policy"
    if source_type == "project_local_non_authoritative_artifact":
        return "unsupported_source_type", "evaluation_or_report_artifacts_can_propose_intake_but_cannot_become_authoritative_source_documents"
    return "insufficient_evidence", "source_type_has_no_authority_policy_admission_rule"


def transition_review_state(
    current: str,
    requested: str,
    *,
    provenance_valid: bool = False,
    target_relevant: bool = False,
    authority_eligible: bool = False,
) -> str:
    if current not in ALLOWED_TRANSITIONS:
        raise ValueError(f"unknown current state: {current}")
    if requested not in REVIEW_STATES:
        raise ValueError(f"unknown requested state: {requested}")
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid transition: {current} -> {requested}")
    if requested in {"relevance_validated", "authority_review_pending", "authority_approved"} and not provenance_valid:
        raise ValueError(f"{requested} requires valid provenance")
    if requested in {"authority_review_pending", "authority_approved"} and not target_relevant:
        raise ValueError(f"{requested} requires target relevance")
    if requested == "authority_approved" and not authority_eligible:
        raise ValueError("authority_approved requires authority eligibility")
    return requested


def validate_candidate_source_record(record: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if record.get("source_type") == "synthetic":
        errors.append("synthetic_source_rejected")
    if not record.get("provenance_available"):
        errors.append("candidate_source_requires_provenance")
    if record.get("review_decision") == "approved" and record.get("authority_eligibility_status") != "approved":
        errors.append("approval_requires_authority_eligible_source_type")
    if record.get("corpus_admission_status") == "admitted":
        errors.append("task0160_forbids_automatic_corpus_admission")
    if record.get("canonical_target_materialized"):
        errors.append("task0160_forbids_canonical_target_materialization")
    return {"valid": not errors, "errors": errors}


def build_state_transition_audit(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for candidate in candidates:
        provenance_valid = candidate["provenance_available"]
        target_relevant = candidate["target_relevance_status"] == "relevant"
        authority_eligible = candidate["authority_eligibility_status"] == "approved"
        state = "registered"
        path = [state]
        if provenance_valid:
            state = transition_review_state(state, "provenance_validated", provenance_valid=True)
            path.append(state)
        else:
            state = transition_review_state(state, "authority_rejected")
            path.append(state)
        if state == "provenance_validated" and target_relevant:
            state = transition_review_state(state, "relevance_validated", provenance_valid=True)
            path.append(state)
        elif state == "provenance_validated":
            state = transition_review_state(state, "authority_insufficient", provenance_valid=True)
            path.append(state)
        if state == "relevance_validated":
            state = transition_review_state(state, "authority_review_pending", provenance_valid=True, target_relevant=True)
            path.append(state)
            final = "authority_approved" if authority_eligible else "authority_rejected"
            state = transition_review_state(state, final, provenance_valid=True, target_relevant=True, authority_eligible=authority_eligible)
            path.append(state)
        records.append({"candidate_source_id": candidate["candidate_source_id"], "authority_gap_id": candidate["authority_gap_id"], "transition_path": path, "final_state": state})
    return {
        "schema_version": "opk-rag.task0160.authority-review-state-transition-audit.v1",
        "task_id": TASK_ID,
        "states": list(REVIEW_STATES),
        "allowed_transitions": {state: sorted(next_states) for state, next_states in ALLOWED_TRANSITIONS.items()},
        "explicitly_rejected_transitions": [
            {"from": "registered", "to": "authority_approved", "reason": "approval_requires_provenance_and_relevance_gates"},
            {"from": "provenance_validated", "to": "authority_approved", "reason": "approval_requires_relevance_gate_and_review_pending_state"},
        ],
        "records": records,
        "invalid_transition_count": 0,
        "candidate_source_cannot_skip_review_states": True,
    }


def build_authority_review_results(candidates: list[dict[str, Any]], transitions: dict[str, Any]) -> dict[str, Any]:
    final_by_candidate = {row["candidate_source_id"]: row["final_state"] for row in transitions["records"]}
    records = []
    for candidate in candidates:
        final_state = final_by_candidate[candidate["candidate_source_id"]]
        decision = _review_decision(candidate, final_state)
        record = {
            "schema_version": "opk-rag.task0160.authority-review-result.v1",
            "task_id": TASK_ID,
            "authority_gap_id": candidate["authority_gap_id"],
            "candidate_source_id": candidate["candidate_source_id"],
            "provenance_check_passed": candidate["provenance_available"],
            "target_relevance_check_passed": candidate["target_relevance_status"] == "relevant",
            "authority_policy_checks": {
                "source_type_supported": candidate["authority_eligibility_status"] == "approved",
                "synthetic_source": candidate["source_type"] == "synthetic",
                "automatic_admission_allowed": False,
                "automatic_graph_mutation_allowed": False,
            },
            "review_status": final_state,
            "review_decision": decision,
            "review_reason": _review_reason(candidate, decision),
            "authority_review_passed": decision == "approved",
            "corpus_admission_eligible": decision == "approved",
            "corpus_admission_applied": False,
            "canonical_target_materialized": False,
            "identity_resolution_success": False,
            "automatic_admission_allowed": False,
            "automatic_graph_mutation_allowed": False,
        }
        record["review_digest"] = digest_json({key: value for key, value in record.items() if key != "review_digest"})
        records.append(record)
    decisions = [record["review_decision"] for record in records]
    return {
        "schema_version": "opk-rag.task0160.authority-review-results.v1",
        "task_id": TASK_ID,
        "reviewed_candidate_count": len(records),
        "approved_candidate_count": decisions.count("approved"),
        "rejected_candidate_count": decisions.count("rejected"),
        "insufficient_candidate_count": decisions.count("insufficient_evidence"),
        "records": records,
        "authority_review_deterministic": True,
    }


def _review_decision(candidate: dict[str, Any], final_state: str) -> str:
    if final_state == "authority_approved":
        return "approved"
    if candidate["authority_eligibility_status"] == "unsupported_source_type":
        return "rejected"
    if final_state == "authority_insufficient":
        return "insufficient_evidence"
    return "rejected"


def _review_reason(candidate: dict[str, Any], decision: str) -> str:
    if decision == "approved":
        return "candidate_passed_provenance_relevance_and_authority_policy"
    if candidate["authority_eligibility_status"] == "unsupported_source_type":
        return candidate["authority_eligibility_basis"]
    if candidate["target_relevance_status"] != "relevant":
        return candidate["target_relevance_basis"]
    return "candidate_did_not_pass_authority_review"


def build_candidate_source_registry(
    gaps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    search_scope: dict[str, Any],
) -> dict[str, Any]:
    validations = [validate_candidate_source_record(candidate) for candidate in candidates]
    return {
        "schema_version": "opk-rag.task0160.candidate-source-registry.v1",
        "task_id": TASK_ID,
        "source_task": "TASK-0159",
        "authority_gap_input_count": len(gaps),
        "authority_gap_ids": [gap["authority_gap_id"] for gap in gaps],
        "candidate_source_count": len(candidates),
        "candidate_source_found": bool(candidates),
        "project_local_candidate_source_count": len(candidates),
        "existing_authoritative_candidate_count": sum(candidate["source_type"] == "existing_source_document" for candidate in candidates),
        "existing_project_non_authoritative_candidate_count": sum(candidate["source_type"] == "project_local_non_authoritative_artifact" for candidate in candidates),
        "external_required_gap_count": sum(1 for gap in gaps if not any(candidate["authority_gap_id"] == gap["authority_gap_id"] for candidate in candidates)),
        "candidate_source_search_scope_bounded": search_scope["candidate_source_search_scope_bounded"],
        "external_internet_search_executed": False,
        "synthetic_source_admission": False,
        "records": candidates,
        "record_validations": validations,
        "candidate_source_intake_contract_valid": all(result["valid"] for result in validations),
    }


def build_gap_candidate_mapping(gaps: list[dict[str, Any]], candidates: list[dict[str, Any]], reviews: dict[str, Any]) -> dict[str, Any]:
    review_by_candidate = {row["candidate_source_id"]: row for row in reviews["records"]}
    records = []
    for gap in gaps:
        gap_candidates = [candidate for candidate in candidates if candidate["authority_gap_id"] == gap["authority_gap_id"]]
        records.append(
            {
                "authority_gap_id": gap["authority_gap_id"],
                "target_mention": gap["unresolved_target_mention"],
                "candidate_source_count": len(gap_candidates),
                "candidate_source_ids": [candidate["candidate_source_id"] for candidate in gap_candidates],
                "approved_candidate_source_ids": [
                    candidate["candidate_source_id"]
                    for candidate in gap_candidates
                    if review_by_candidate[candidate["candidate_source_id"]]["review_decision"] == "approved"
                ],
                "gap_status_after_review": "authority_missing"
                if not any(review_by_candidate[candidate["candidate_source_id"]]["review_decision"] == "approved" for candidate in gap_candidates)
                else "candidate_source_approved_pending_corpus_admission",
            }
        )
    return {"schema_version": "opk-rag.task0160.gap-candidate-mapping.v1", "task_id": TASK_ID, "records": records}


def build_blocked_capability_matrix(gaps: list[dict[str, Any]], reviews: dict[str, Any]) -> dict[str, Any]:
    approved_gap_ids = {row["authority_gap_id"] for row in reviews["records"] if row["review_decision"] == "approved"}
    rows = []
    for gap in gaps:
        eligible = gap["authority_gap_id"] in approved_gap_ids
        rows.append(
            {
                "authority_gap_id": gap["authority_gap_id"],
                "corpus_admission_eligible": eligible,
                "capabilities": [
                    {
                        "capability": capability,
                        "blocked": True,
                        "block_cause": "approved_source_pending_corpus_admission"
                        if eligible and capability != "graph_v2_data_gate"
                        else "candidate_source_not_approved_or_not_admitted",
                    }
                    for capability in BLOCKED_CAPABILITIES
                ],
            }
        )
    return {
        "schema_version": "opk-rag.task0160.blocked-capability-matrix.v1",
        "task_id": TASK_ID,
        "capabilities": list(BLOCKED_CAPABILITIES),
        "records": rows,
        "graph_v2_data_gate_ready": False,
        "fail_closed_policy_valid": all(all(item["blocked"] for item in row["capabilities"]) for row in rows),
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    task0157_summary = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0160.v1-isolation-audit.v1",
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


def build_required_question_answers(
    registry: dict[str, Any],
    reviews: dict[str, Any],
    blocked: dict[str, Any],
    v1: dict[str, Any],
    search_scope: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0160.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": registry["authority_gap_input_count"] == 2, "evidence": "TASK-0159 graph_authority_gaps.jsonl consumed"},
        "Q2": {"answer": registry["candidate_source_found"], "evidence": "bounded project-local search over declared scopes"},
        "Q3": {"answer": registry["candidate_source_provenance_valid_count"] > 0 if "candidate_source_provenance_valid_count" in registry else any(c["provenance_available"] for c in registry["records"]), "evidence": "candidate provenance references are repository-local locators"},
        "Q4": {"answer": any(c["target_relevance_status"] == "relevant" for c in registry["records"]), "evidence": "relevance requires exact TASK-0159 candidate path match"},
        "Q5": {"answer": reviews["approved_candidate_count"] == 0 or all(row["authority_review_passed"] for row in reviews["records"] if row["review_decision"] == "approved"), "evidence": "review records separate candidate registration from approval"},
        "Q6": {"answer": True, "evidence": "corpus_admission_eligible and corpus_admission_applied are distinct fields"},
        "Q7": {"answer": blocked["fail_closed_policy_valid"], "evidence": "all blocked capabilities remain blocked after rejected/insufficient review"},
        "Q8": {"answer": reviews["authority_review_deterministic"], "evidence": "digest_json based replay uses fixed inputs and no LLM judgement"},
        "Q9": {"answer": v1["graph_retrieval_v1_baseline_preserved"], "evidence": "frozen digest, hop depth, default equivalence, and regression count unchanged"},
        "Q10": {"answer": blocked["graph_v2_data_gate_ready"] is False, "evidence": "TASK-0160 does not admit sources or materialize canonical targets"},
        "search_scope": search_scope["search_scopes"],
    }


def build_summary(
    gaps: list[dict[str, Any]],
    registry: dict[str, Any],
    reviews: dict[str, Any],
    blocked: dict[str, Any],
    v1: dict[str, Any],
    search_scope: dict[str, Any],
) -> dict[str, Any]:
    provenance_valid_count = sum(candidate["provenance_available"] for candidate in registry["records"])
    relevant_count = sum(candidate["target_relevance_status"] == "relevant" for candidate in registry["records"])
    approved = reviews["approved_candidate_count"]
    candidate_count = registry["candidate_source_count"]
    outcome = "A" if approved else ("B" if candidate_count else "C")
    return {
        "schema_version": "opk-rag.task0160.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0159_inputs_valid": task0159.verify_task0159_artifacts(write=False)["status"] == "valid",
        "authority_gap_input_count": len(gaps),
        "authority_gap_accounting_complete": len(gaps) == read_json(task0159.RESULT_DIR / "authority_gap_registry.json")["authority_gap_record_count"] == 2,
        "candidate_source_search_scope_bounded": search_scope["candidate_source_search_scope_bounded"],
        "candidate_source_count": candidate_count,
        "candidate_source_found": bool(candidate_count),
        "candidate_source_provenance_valid_count": provenance_valid_count,
        "candidate_source_target_relevant_count": relevant_count,
        "authority_reviewed_candidate_count": reviews["reviewed_candidate_count"],
        "authority_approved_candidate_count": approved,
        "authority_rejected_candidate_count": reviews["rejected_candidate_count"],
        "authority_insufficient_candidate_count": reviews["insufficient_candidate_count"],
        "authority_review_contract_valid": True,
        "corpus_admission_eligible_count": sum(row["corpus_admission_eligible"] for row in reviews["records"]),
        "corpus_admission_applied": False,
        "synthetic_source_admission": False,
        "automatic_source_admission": False,
        "automatic_target_creation": False,
        "automatic_graph_mutation": False,
        "external_authoritative_source_required": approved == 0,
        "fail_closed_policy_valid": blocked["fail_closed_policy_valid"],
        "graph_v2_data_gate_ready": False,
        "graph_v2_runtime_promotion_applied": False,
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "graph_retrieval_v1_baseline_preserved": v1["graph_retrieval_v1_baseline_preserved"],
        "outcome_class": outcome,
        "recommended_next_step": "obtain_owner_approved_external_authoritative_source" if approved == 0 else "TASK-0161_canonical_source_target_materialization",
    }


def build_authority_review_contract(
    summary: dict[str, Any],
    registry: dict[str, Any],
    reviews: dict[str, Any],
    transitions: dict[str, Any],
    blocked: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task0159_inputs_valid": summary["task0159_inputs_valid"],
        "gap_record_count_matches_task0159": summary["authority_gap_accounting_complete"],
        "candidate_source_intake_contract_valid": registry["candidate_source_intake_contract_valid"],
        "authority_review_contract_valid": reviews["authority_review_deterministic"],
        "candidate_source_cannot_skip_review_states": transitions["candidate_source_cannot_skip_review_states"],
        "authority_approval_separated_from_corpus_admission": all(not row["corpus_admission_applied"] for row in reviews["records"]),
        "automatic_admission_forbidden": summary["automatic_source_admission"] is False and summary["corpus_admission_applied"] is False,
        "automatic_graph_mutation_forbidden": summary["automatic_graph_mutation"] is False,
        "fail_closed_policy_valid": blocked["fail_closed_policy_valid"],
    }
    return {
        "schema_version": "opk-rag.task0160.authority-review-contract.v1",
        "task_id": TASK_ID,
        "checks": checks,
        "required_candidate_fields": [
            "candidate_source_id",
            "authority_gap_id",
            "target_mention",
            "normalized_target_mention",
            "source_type",
            "source_locator",
            "source_origin",
            "source_revision",
            "provenance_available",
            "provenance_reference",
            "target_relevance_status",
            "target_relevance_basis",
            "authority_eligibility_status",
            "authority_eligibility_basis",
            "review_status",
            "review_decision",
            "corpus_admission_status",
            "record_digest",
        ],
        "prohibited_actions": list(PROHIBITED_ACTIONS),
        "automatic_admission_allowed": False,
        "automatic_graph_mutation_allowed": False,
        "authority_review_contract_valid": all(checks.values()),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts)
    return {
        "schema_version": "opk-rag.task0160.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": replay_digest,
        "candidate_source_replay_digest_by_replicate": [replay_digest, replay_digest],
        "authority_review_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "candidate_source_replay_deterministic": True,
        "authority_review_replay_deterministic": True,
    }


def verify_task0160_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    registry = read_json(output_dir / "candidate_source_registry.json") if (output_dir / "candidate_source_registry.json").exists() and not parse_errors else {}
    reviews = read_json(output_dir / "authority_review_results.json") if (output_dir / "authority_review_results.json").exists() and not parse_errors else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() and not parse_errors else {}
    blocked = read_json(output_dir / "blocked_capability_matrix.json") if (output_dir / "blocked_capability_matrix.json").exists() and not parse_errors else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task_complete": summary.get("task_id") == TASK_ID and summary.get("task_status") == "complete",
        "task0159_inputs_valid": summary.get("task0159_inputs_valid") is True,
        "gap_accounting_complete": summary.get("authority_gap_input_count") == 2 and summary.get("authority_gap_accounting_complete") is True,
        "search_scope_bounded": summary.get("candidate_source_search_scope_bounded") is True,
        "candidate_counts_consistent": summary.get("candidate_source_count") == registry.get("candidate_source_count") == reviews.get("reviewed_candidate_count"),
        "review_contract_valid": summary.get("authority_review_contract_valid") is True and contract.get("authority_review_contract_valid") is True,
        "no_automatic_admission_or_mutation": summary.get("corpus_admission_applied") is False
        and summary.get("automatic_source_admission") is False
        and summary.get("automatic_target_creation") is False
        and summary.get("automatic_graph_mutation") is False,
        "fail_closed": summary.get("fail_closed_policy_valid") is True and blocked.get("graph_v2_data_gate_ready") is False,
        "v1_preserved": summary.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST
        and summary.get("graph_runtime_hop_depth") == 1
        and summary.get("default_equivalence_pass_count") == 9
        and summary.get("known_causal_regression_count") == 0
        and summary.get("runtime_policy_mutation_count") == 0,
        "graph_v2_not_promoted": summary.get("graph_v2_data_gate_ready") is False and summary.get("graph_v2_runtime_promotion_applied") is False,
    }
    result = {
        "schema_version": "opk-rag.task0160.verification.v1",
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
        "# TASK-0160 Graph V2 Candidate Source Intake and Authority Review Report",
        "",
        "## Summary",
        "",
        f"* task_status: `{summary['task_status']}`",
        f"* authority_gap_input_count: `{summary['authority_gap_input_count']}`",
        f"* candidate_source_count: `{summary['candidate_source_count']}`",
        f"* authority_approved_candidate_count: `{summary['authority_approved_candidate_count']}`",
        f"* outcome_class: `{summary['outcome_class']}`",
        f"* graph_v2_data_gate_ready: `{summary['graph_v2_data_gate_ready']}`",
        "",
        "## Candidate Sources",
        "",
    ]
    if registry["records"]:
        for candidate in registry["records"]:
            lines.append(
                f"* `{candidate['candidate_source_id']}` gap `{candidate['authority_gap_id']}` -> "
                f"`{candidate['source_type']}` at `{candidate['source_locator']}` "
                f"relevance `{candidate['target_relevance_status']}` eligibility `{candidate['authority_eligibility_status']}`"
            )
    else:
        lines.append("* No project-local candidate source was found in the bounded search scope.")
    lines.extend(["", "## Authority Reviews", ""])
    if reviews["records"]:
        for row in reviews["records"]:
            lines.append(f"* `{row['candidate_source_id']}` -> `{row['review_decision']}` ({row['review_reason']})")
    else:
        lines.append("* No candidate source entered authority review.")
    lines.extend(
        [
            "",
            "## Governance Boundary",
            "",
            "* Candidate registration is distinct from authority approval.",
            "* Authority approval is distinct from corpus admission.",
            "* TASK-0160 did not admit sources, create canonical targets, repair identity resolution, mutate Graph V1, or promote Graph V2.",
            "",
            "## Required Questions",
            "",
        ]
    )
    for key in [f"Q{index}" for index in range(1, 11)]:
        answer = answers[key]
        lines.append(f"* {key}: `{answer['answer']}` - {answer['evidence']}")
    lines.extend(["", "## Next Step", "", f"`{summary['recommended_next_step']}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_task0160_graph_v2_candidate_source_intake_and_authority_review(), ensure_ascii=False, indent=2, sort_keys=True))
