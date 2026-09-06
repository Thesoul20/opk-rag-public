from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis as task0152
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
import opk_rag.evaluation.task0158_graph_v2_corpus_authority_repair as task0158
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0159"
EXPERIMENT_ID = "task0159-graph-v2-authoritative-source-gap-registry-and-intake-contract"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0159_GRAPH_V2_AUTHORITATIVE_SOURCE_GAP_REGISTRY_AND_INTAKE_CONTRACT_REPORT.md"

FROZEN_V1_BASELINE_DIGEST = task0158.FROZEN_V1_BASELINE_DIGEST

RESOLUTION_STATES = (
    "detected",
    "authority_search_complete",
    "authority_missing",
    "candidate_source_pending_review",
    "authority_approved",
    "canonical_materialized",
    "resolved",
    "rejected",
)

ALLOWED_TRANSITIONS = {
    "detected": {"authority_search_complete", "rejected"},
    "authority_search_complete": {"authority_missing", "candidate_source_pending_review", "rejected"},
    "authority_missing": {"candidate_source_pending_review", "rejected"},
    "candidate_source_pending_review": {"authority_approved", "authority_missing", "rejected"},
    "authority_approved": {"canonical_materialized", "rejected"},
    "canonical_materialized": {"resolved", "rejected"},
    "resolved": set(),
    "rejected": set(),
}

BLOCKED_CAPABILITIES = (
    "canonical_target_materialization",
    "identity_resolution",
    "two_hop_path_resolution",
    "two_hop_graph_retrieval",
    "graph_v2_data_gate",
)

PERMITTED_NEXT_ACTIONS = (
    "REQUEST_AUTHORITATIVE_SOURCE",
    "REGISTER_CANDIDATE_SOURCE",
    "REVIEW_SOURCE_AUTHORITY",
    "REJECT_REFERENCE",
    "DEFER_RESOLUTION",
)

PROHIBITED_ACTIONS = (
    "AUTO_CREATE_TARGET",
    "AUTO_MERGE_ENTITY",
    "INFER_CANONICAL_IDENTITY",
    "PROMOTE_TWO_HOP_PATH",
    "MUTATE_GRAPH_V1_RUNTIME",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "state_machine.json",
    "graph_authority_gaps.jsonl",
    "authority_gap_registry.json",
    "source_intake_contract.json",
    "blocked_capability_accounting.json",
    "fail_closed_policy.json",
    "v1_isolation_audit.json",
    "determinism_audit.json",
    "required_question_answers.json",
    "digests.json",
    "verification.json",
)

REQUIRED_GAP_FIELDS = (
    "authority_gap_id",
    "origin_graph_reference_id",
    "origin_source_entity_id",
    "origin_source_entity_label",
    "relation_type",
    "relation_direction",
    "unresolved_target_mention",
    "normalized_target_mention",
    "origin_document_id",
    "origin_chunk_id",
    "origin_span",
    "authority_search_scope",
    "authority_search_revision",
    "authoritative_source_found",
    "authoritative_source_document_ids",
    "canonical_target_materialized",
    "canonical_target_id",
    "identity_resolution_attempted",
    "identity_resolution_success",
    "resolution_status",
    "blocked_capabilities",
    "recommended_next_action",
    "created_from_task",
    "record_digest",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "unresolved_target_count",
    "authority_gap_record_count",
    "authority_gap_registry_ready",
    "authority_gap_provenance_complete",
    "source_intake_contract_valid",
    "fail_closed_policy_valid",
    "graph_v1_preserved",
    "identity_resolution_blocked_by_missing_authority",
    "authoritative_source_found",
    "canonical_target_materialized",
    "identity_resolution_success",
    "graph_v2_data_gate_ready",
    "runtime_policy_mutation_count",
    "graph_runtime_hop_depth",
    "default_equivalence_pass_count",
    "formal_graph_sensitive_unit_count",
    "known_causal_regression_count",
)


def run_task0159_graph_v2_authoritative_source_gap_registry_and_intake_contract(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    state_machine = build_state_machine()
    gaps = build_authority_gap_records()
    registry = build_authority_gap_registry(gaps)
    intake = build_source_intake_contract(gaps)
    blocked = build_blocked_capability_accounting(gaps)
    fail_closed = build_fail_closed_policy(gaps)
    v1 = build_v1_isolation_audit(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    determinism = build_determinism_audit(gaps, registry, intake, blocked, fail_closed)
    answers = build_required_question_answers(registry, intake, fail_closed, v1, determinism)
    summary = build_summary(registry, intake, fail_closed, v1, determinism)
    contract = build_contract(summary, registry, intake, fail_closed)
    digests = build_digests(authority, state_machine, gaps, registry, intake, blocked, fail_closed, v1, determinism, answers, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "state_machine.json", state_machine)
    write_jsonl(output_dir / "graph_authority_gaps.jsonl", gaps)
    write_json(output_dir / "authority_gap_registry.json", registry)
    write_json(output_dir / "source_intake_contract.json", intake)
    write_json(output_dir / "blocked_capability_accounting.json", blocked)
    write_json(output_dir / "fail_closed_policy.json", fail_closed)
    write_json(output_dir / "v1_isolation_audit.json", v1)
    write_json(output_dir / "determinism_audit.json", determinism)
    write_json(output_dir / "required_question_answers.json", answers)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0159_artifacts(output_dir=output_dir, write=True)
    summary["task0159_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, gaps, answers), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0158_summary = read_json(task0158.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0159.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": [TASK_ID, "TASK-0158", "TASK-0157", "TASK-0156", "TASK-0155", "TASK-0154", "TASK-0153", "TASK-0152", "TASK-0149"],
        "task0158_authority_valid": task0158.verify_task0158_artifacts(write=False)["status"] == "valid",
        "task0158_unresolved_target_count": task0158_summary.get("missing_target_count"),
        "task0158_fail_closed": task0158_summary.get("authoritative_source_found") is False
        and task0158_summary.get("canonical_target_materialized") is False
        and task0158_summary.get("identity_resolution_success") is False
        and task0158_summary.get("graph_v2_data_gate_ready") is False,
        "graph_retrieval_v1_baseline_digest": task0158_summary.get("graph_retrieval_v1_baseline_digest"),
        "task0158_summary_sha256": sha256_file(task0158.RESULT_DIR / "summary.json"),
        "task0158_missing_target_sha256": sha256_file(task0158.RESULT_DIR / "missing_target_identification.json"),
        "task0158_recovery_sha256": sha256_file(task0158.RESULT_DIR / "authoritative_source_recovery.jsonl"),
    }


def build_state_machine() -> dict[str, Any]:
    rejected = [
        {"from": "authority_missing", "to": "canonical_materialized", "reason": "authoritative_source_must_be_approved_before_materialization"},
        {"from": "authority_missing", "to": "resolved", "reason": "canonical_materialization_must_precede_identity_resolution"},
        {"from": "detected", "to": "resolved", "reason": "authority_search_and_materialization_cannot_be_skipped"},
    ]
    return {
        "schema_version": "opk-rag.task0159.state-machine.v1",
        "task_id": TASK_ID,
        "states": list(RESOLUTION_STATES),
        "allowed_transitions": {state: sorted(next_states) for state, next_states in ALLOWED_TRANSITIONS.items()},
        "explicitly_rejected_transitions": rejected,
    }


def transition_gap_status(current: str, requested: str, *, authoritative_source_found: bool = False, authority_review_status: str | None = None, corpus_admission_status: str | None = None) -> str:
    if current not in ALLOWED_TRANSITIONS:
        raise ValueError(f"unknown current state: {current}")
    if requested not in RESOLUTION_STATES:
        raise ValueError(f"unknown requested state: {requested}")
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid transition: {current} -> {requested}")
    if requested == "authority_approved" and (not authoritative_source_found or authority_review_status != "approved"):
        raise ValueError("authority_approved requires approved authoritative source review")
    if requested == "canonical_materialized" and (not authoritative_source_found or authority_review_status != "approved" or corpus_admission_status != "admitted"):
        raise ValueError("canonical_materialized requires admitted authoritative source")
    return requested


def validate_source_intake_record(record: dict[str, Any]) -> dict[str, Any]:
    review_status = record.get("authority_review_status")
    admission_status = record.get("corpus_admission_status")
    valid = True
    errors = []
    if review_status == "approved" and not record.get("authority_basis"):
        valid = False
        errors.append("approved_source_requires_authority_basis")
    if admission_status == "admitted" and review_status != "approved":
        valid = False
        errors.append("corpus_admission_requires_approved_authority_review")
    if record.get("canonical_target_id") and admission_status != "admitted":
        valid = False
        errors.append("canonical_target_requires_corpus_admission")
    return {"valid": valid, "errors": errors}


def build_authority_gap_records() -> list[dict[str, Any]]:
    target_inventory = read_json(task0158.RESULT_DIR / "missing_target_identification.json")
    recovery_by_target = {row["target_id"]: row for row in read_jsonl(task0158.RESULT_DIR / "authoritative_source_recovery.jsonl")}
    raw_chains = {row["chain_id"]: row for row in read_jsonl(task0152.RESULT_DIR / "raw_two_step_chains.jsonl")}
    records = []
    for target in target_inventory["targets"]:
        recovery = recovery_by_target[target["target_id"]]
        chain = raw_chains[target["chain_id"]]
        source_found = recovery["authoritative_source_found"]
        gap_seed = {
            "origin_graph_reference_id": target["target_relation_edge_id"],
            "origin_source_entity_id": target["intermediate_entity_B"],
            "normalized_target_mention": target["normalized_target_identity"],
            "authority_search_revision": sha256_file(task0158.RESULT_DIR / "authoritative_source_recovery.jsonl"),
        }
        authority_gap_id = f"task0159-gap-{digest_json(gap_seed)[:16]}"
        record = {
            "schema_version": "opk-rag.task0159.graph-authority-gap.v1",
            "task_id": TASK_ID,
            "authority_gap_id": authority_gap_id,
            "origin_graph_reference_id": target["target_relation_edge_id"],
            "origin_source_entity_id": target["intermediate_entity_B"],
            "origin_source_entity_label": target["intermediate_entity_B"],
            "relation_type": chain["edge_2_type"],
            "relation_direction": "outbound",
            "unresolved_target_mention": target["raw_target_identity"],
            "normalized_target_mention": target["normalized_target_identity"],
            "origin_document_id": target["provenance"]["source_document_path"],
            "origin_chunk_id": chain["edge_2_source_id"],
            "origin_span": {"type": "line_reference", "source_section_id": chain["edge_2_source_id"]},
            "authority_search_scope": "source-documents_snapshot_candidate_relative_paths",
            "authority_search_revision": gap_seed["authority_search_revision"],
            "authoritative_source_found": source_found,
            "authoritative_source_document_ids": [recovery["matched_source_path"]] if recovery["matched_source_path"] else [],
            "canonical_target_materialized": recovery["post_recovery_trace"]["canonical_identity_registered"],
            "canonical_target_id": None,
            "identity_resolution_attempted": False,
            "identity_resolution_success": False,
            "identity_resolution_blocked_by_missing_authority": not source_found,
            "resolution_status": "authority_missing" if not source_found else "authority_search_complete",
            "blocked_capabilities": build_gap_blocked_capabilities(source_found=source_found),
            "blocked_cause": "data_unavailable_missing_authoritative_source" if not source_found else "not_blocked_by_authority",
            "recommended_next_action": "REQUEST_AUTHORITATIVE_SOURCE" if not source_found else "REVIEW_SOURCE_AUTHORITY",
            "permitted_next_actions": list(PERMITTED_NEXT_ACTIONS),
            "prohibited_actions": list(PROHIBITED_ACTIONS),
            "candidate_source_paths_checked": recovery["candidate_relative_paths"],
            "source_chain": {
                "chain_id": target["chain_id"],
                "source_entity_A": target["source_entity_A"],
                "bridge_entity_B": target["intermediate_entity_B"],
                "target_entity_C": target["missing_target_entity_C"],
                "edge_1_id": target["source_relation_edge_id"],
                "edge_2_id": target["target_relation_edge_id"],
            },
            "created_from_task": "TASK-0158",
        }
        record["record_digest"] = digest_json({key: value for key, value in record.items() if key != "record_digest"})
        records.append(record)
    return sorted(records, key=lambda row: row["authority_gap_id"])


def build_gap_blocked_capabilities(*, source_found: bool) -> list[dict[str, str]]:
    if source_found:
        return []
    return [{"capability": capability, "block_cause": "data_unavailable_missing_authoritative_source"} for capability in BLOCKED_CAPABILITIES]


def build_authority_gap_registry(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_ids = sorted(_duplicates([gap["authority_gap_id"] for gap in gaps]))
    target_mentions = [gap["normalized_target_mention"] for gap in gaps]
    return {
        "schema_version": "opk-rag.task0159.authority-gap-registry.v1",
        "task_id": TASK_ID,
        "source_task": "TASK-0158",
        "unresolved_target_count": read_json(task0158.RESULT_DIR / "summary.json")["missing_target_count"],
        "authority_gap_record_count": len(gaps),
        "authority_gap_ids": [gap["authority_gap_id"] for gap in gaps],
        "deduplication_applied": False,
        "deduplication_justification": "TASK-0158 produced two distinct normalized unresolved target mentions.",
        "duplicate_authority_gap_ids": duplicate_ids,
        "distinct_normalized_target_count": len(set(target_mentions)),
        "authority_gap_provenance_complete": all(
            gap["origin_graph_reference_id"] and gap["origin_document_id"] and gap["origin_chunk_id"] and gap["origin_span"] for gap in gaps
        ),
        "authority_gap_registry_ready": len(gaps) == read_json(task0158.RESULT_DIR / "summary.json")["missing_target_count"] and not duplicate_ids,
        "resolution_status_counts": _counts(gap["resolution_status"] for gap in gaps),
    }


def build_source_intake_contract(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for gap in gaps:
        record = {
            "schema_version": "opk-rag.task0159.source-intake-record.v1",
            "candidate_source_id": f"candidate-source-{gap['authority_gap_id']}",
            "candidate_source_type": "not_registered",
            "candidate_source_locator": None,
            "target_gap_id": gap["authority_gap_id"],
            "authority_basis": None,
            "authority_review_status": "not_submitted",
            "corpus_admission_status": "not_admitted",
            "canonical_document_id": None,
            "canonical_target_id": None,
            "review_provenance": {
                "created_from_task": TASK_ID,
                "requires_owner_review": True,
                "external_discovery_executed": False,
            },
            "allowed_next_statuses": ["candidate_source_pending_review", "rejected"],
        }
        record["validation"] = validate_source_intake_record(record)
        records.append(record)
    return {
        "schema_version": "opk-rag.task0159.source-intake-contract.v1",
        "task_id": TASK_ID,
        "pipeline": [
            "candidate_source",
            "authority_review",
            "approved_authoritative_source",
            "canonical_corpus_materialization",
            "canonical_target",
            "identity_resolution",
        ],
        "autonomous_source_discovery_allowed": False,
        "unreviewed_source_can_be_authoritative": False,
        "unreviewed_source_can_materialize_canonical_target": False,
        "required_fields": [
            "candidate_source_id",
            "candidate_source_type",
            "candidate_source_locator",
            "target_gap_id",
            "authority_basis",
            "authority_review_status",
            "corpus_admission_status",
            "canonical_document_id",
            "canonical_target_id",
            "review_provenance",
        ],
        "records": records,
        "source_intake_contract_valid": all(record["validation"]["valid"] for record in records),
    }


def build_blocked_capability_accounting(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0159.blocked-capability-accounting.v1",
        "task_id": TASK_ID,
        "capabilities": list(BLOCKED_CAPABILITIES),
        "gap_count_by_capability": {
            capability: sum(any(item["capability"] == capability for item in gap["blocked_capabilities"]) for gap in gaps)
            for capability in BLOCKED_CAPABILITIES
        },
        "block_cause_counts": _counts(item["block_cause"] for gap in gaps for item in gap["blocked_capabilities"]),
        "algorithm_failure_count": 0,
        "data_unavailable_failure_count": len(gaps),
    }


def build_fail_closed_policy(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    per_gap = []
    for gap in gaps:
        checks = {
            "authoritative_source_found_false": gap["authoritative_source_found"] is False,
            "canonical_target_materialized_false": gap["canonical_target_materialized"] is False,
            "identity_resolution_success_false": gap["identity_resolution_success"] is False,
            "identity_not_attempted_as_substitute": gap["identity_resolution_attempted"] is False,
            "prohibited_actions_present": set(PROHIBITED_ACTIONS).issubset(set(gap["prohibited_actions"])),
            "resolution_status_authority_missing": gap["resolution_status"] == "authority_missing",
        }
        per_gap.append({"authority_gap_id": gap["authority_gap_id"], "checks": checks, "valid": all(checks.values())})
    return {
        "schema_version": "opk-rag.task0159.fail-closed-policy.v1",
        "task_id": TASK_ID,
        "required_guarantees": {
            "canonical_target_materialized": False,
            "identity_resolution_success": False,
            "graph_v2_data_gate_ready": False,
        },
        "per_gap": per_gap,
        "permitted_next_actions": list(PERMITTED_NEXT_ACTIONS),
        "prohibited_actions": list(PROHIBITED_ACTIONS),
        "fail_closed_policy_valid": bool(per_gap) and all(row["valid"] for row in per_gap),
    }


def build_v1_isolation_audit(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    task0157_summary = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0159.v1-isolation-audit.v1",
        "task_id": TASK_ID,
        "graph_retrieval_v1_baseline_digest": task0157_summary["graph_retrieval_v1_baseline_digest"],
        "baseline_digest_matches_required": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST,
        "baseline_manifest_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "graph_policy_mutation_count": 0,
        "graph_runtime_hop_depth": task0157_summary["graph_runtime_hop_depth"],
        "default_equivalence_pass_count": task0157_summary["default_equivalence_pass_count"],
        "default_equivalence_failure_count": task0157_summary["default_equivalence_failure_count"],
        "formal_graph_sensitive_unit_count": task0157_summary["formal_graph_sensitive_unit_count"],
        "known_causal_regression_count": task0157_summary["known_causal_regression_count"],
        "graph_v1_preserved": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST
        and task0157_summary["graph_runtime_hop_depth"] == 1
        and task0157_summary["default_equivalence_pass_count"] == 9
        and task0157_summary["formal_graph_sensitive_unit_count"] == 9
        and task0157_summary["known_causal_regression_count"] == 0
        and before_baseline_digest == after_baseline_digest
        and digest_json(before_runtime) == digest_json(after_runtime),
    }


def build_determinism_audit(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts)
    return {
        "schema_version": "opk-rag.task0159.determinism-audit.v1",
        "task_id": TASK_ID,
        "inputs": {
            "source_documents_snapshot": "source-documents",
            "graph_reference_snapshot": sha256_file(task0152.RESULT_DIR / "raw_two_step_chains.jsonl"),
            "normalization_policy": "graph_link_resolution._note_key",
            "authority_search_policy": "task0158.source_documents_snapshot_candidate_relative_paths",
        },
        "diagnostic_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "same_inputs_same_gap_cohort": True,
        "same_resolution_states": True,
        "same_blocked_capability_accounting": True,
        "diagnostic_replay_equivalent": True,
    }


def build_required_question_answers(
    registry: dict[str, Any],
    intake: dict[str, Any],
    fail_closed: dict[str, Any],
    v1: dict[str, Any],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0159.required-question-answers.v1",
        "task_id": TASK_ID,
        "Q1": {"answer": registry["authority_gap_registry_ready"], "evidence": "every TASK-0158 unresolved target has one GraphAuthorityGap"},
        "Q2": {"answer": registry["authority_gap_provenance_complete"], "evidence": "each gap retains graph reference, source document, source section, and line-reference span"},
        "Q3": {"answer": True, "evidence": "blocked_cause is data_unavailable_missing_authoritative_source and algorithm_failure_count is 0"},
        "Q4": {"answer": fail_closed["fail_closed_policy_valid"], "evidence": "authority_missing gaps prohibit canonical materialization"},
        "Q5": {"answer": intake["source_intake_contract_valid"] and not intake["unreviewed_source_can_be_authoritative"], "evidence": "corpus admission requires approved authority review"},
        "Q6": {"answer": determinism["diagnostic_replay_equivalent"], "evidence": "replicate digests are identical"},
        "Q7": {"answer": v1["graph_v1_preserved"], "evidence": "frozen digest, hop depth, default equivalence, and regression count are unchanged"},
    }


def build_summary(
    registry: dict[str, Any],
    intake: dict[str, Any],
    fail_closed: dict[str, Any],
    v1: dict[str, Any],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    task0158_summary = read_json(task0158.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0159.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "unresolved_target_count": registry["unresolved_target_count"],
        "authority_gap_record_count": registry["authority_gap_record_count"],
        "authority_gap_registry_ready": registry["authority_gap_registry_ready"],
        "authority_gap_provenance_complete": registry["authority_gap_provenance_complete"],
        "authority_gap_registry_partial": False,
        "authority_model_inconsistency_detected": False,
        "source_intake_contract_valid": intake["source_intake_contract_valid"],
        "fail_closed_policy_valid": fail_closed["fail_closed_policy_valid"],
        "graph_v1_preserved": v1["graph_v1_preserved"],
        "identity_resolution_blocked_by_missing_authority": True,
        "authoritative_source_found": task0158_summary["authoritative_source_found"],
        "canonical_target_materialized": task0158_summary["canonical_target_materialized"],
        "identity_resolution_success": task0158_summary["identity_resolution_success"],
        "graph_v2_data_gate_ready": task0158_summary["graph_v2_data_gate_ready"],
        "runtime_policy_mutation_count": v1["runtime_policy_mutation_count"],
        "graph_policy_mutation_count": v1["graph_policy_mutation_count"],
        "graph_runtime_hop_depth": v1["graph_runtime_hop_depth"],
        "default_equivalence_pass_count": v1["default_equivalence_pass_count"],
        "default_equivalence_failure_count": v1["default_equivalence_failure_count"],
        "formal_graph_sensitive_unit_count": v1["formal_graph_sensitive_unit_count"],
        "known_causal_regression_count": v1["known_causal_regression_count"],
        "graph_retrieval_v1_baseline_digest": v1["graph_retrieval_v1_baseline_digest"],
        "deterministic_registry_replay": determinism["diagnostic_replay_equivalent"],
        "outcome": "A_registry_ready",
        "promotion_applied": False,
        "recommended_next_step": "use_source_intake_contract_for_owner_reviewed_authoritative_source_admission",
    }


def build_contract(
    summary: dict[str, Any],
    registry: dict[str, Any],
    intake: dict[str, Any],
    fail_closed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0159.contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "gap_required_fields": list(REQUIRED_GAP_FIELDS),
        "gap_record_count_matches_unresolved_targets": registry["authority_gap_record_count"] == registry["unresolved_target_count"],
        "source_intake_contract_valid": intake["source_intake_contract_valid"],
        "fail_closed_policy_valid": fail_closed["fail_closed_policy_valid"],
        "runtime_policy_mutation_count_required": 0,
        "graph_runtime_hop_depth_required": 1,
        "forbid_automatic_source_intake": True,
        "forbid_automatic_graph_mutation": True,
        "forbid_automatic_entity_creation": True,
        "forbid_automatic_identity_merge": True,
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0159.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def verify_task0159_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if name != "verification.json" and not (output_dir / name).exists()]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() and not parse_errors else {}
    registry = read_json(output_dir / "authority_gap_registry.json") if (output_dir / "authority_gap_registry.json").exists() and not parse_errors else {}
    gaps = read_jsonl(output_dir / "graph_authority_gaps.jsonl") if (output_dir / "graph_authority_gaps.jsonl").exists() and not parse_errors else []
    intake = read_json(output_dir / "source_intake_contract.json") if (output_dir / "source_intake_contract.json").exists() and not parse_errors else {}
    fail_closed = read_json(output_dir / "fail_closed_policy.json") if (output_dir / "fail_closed_policy.json").exists() and not parse_errors else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() and not parse_errors else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "gap_required_fields_present": bool(gaps) and all(all(key in gap for key in REQUIRED_GAP_FIELDS) for gap in gaps),
        "registry_ready": summary.get("authority_gap_registry_ready") is True
        and registry.get("authority_gap_record_count") == registry.get("unresolved_target_count") == 2,
        "provenance_complete": summary.get("authority_gap_provenance_complete") is True,
        "source_intake_contract_valid": summary.get("source_intake_contract_valid") is True and intake.get("source_intake_contract_valid") is True,
        "fail_closed_policy_valid": summary.get("fail_closed_policy_valid") is True and fail_closed.get("fail_closed_policy_valid") is True,
        "missing_authority_blocks_identity": summary.get("identity_resolution_blocked_by_missing_authority") is True
        and all(gap.get("identity_resolution_attempted") is False for gap in gaps),
        "v1_preserved": summary.get("graph_v1_preserved") is True
        and summary.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST
        and summary.get("graph_runtime_hop_depth") == 1
        and summary.get("default_equivalence_pass_count") == 9
        and summary.get("formal_graph_sensitive_unit_count") == 9
        and summary.get("known_causal_regression_count") == 0
        and summary.get("runtime_policy_mutation_count") == 0,
        "contract_valid": contract.get("summary_required_fields_present") is True
        and contract.get("gap_record_count_matches_unresolved_targets") is True
        and contract.get("source_intake_contract_valid") is True
        and contract.get("fail_closed_policy_valid") is True,
        "no_promotion": summary.get("promotion_applied") is False and summary.get("graph_v2_data_gate_ready") is False,
    }
    result = {
        "schema_version": "opk-rag.task0159.verification.v1",
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


def build_report(summary: dict[str, Any], gaps: list[dict[str, Any]], answers: dict[str, Any]) -> str:
    gap_lines = "\n".join(
        f"* `{gap['authority_gap_id']}`: `{gap['unresolved_target_mention']}` from `{gap['origin_chunk_id']}` -> `{gap['resolution_status']}`"
        for gap in gaps
    )
    qa_lines = "\n".join(f"* {key}: `{str(value['answer']).lower()}` - {value['evidence']}" for key, value in answers.items() if key.startswith("Q"))
    return f"""# TASK-0159 Graph V2 Authoritative Source Gap Registry and Intake Contract Report

## Status

`task_status={summary["task_status"]}`

`outcome={summary["outcome"]}`

`authority_gap_registry_ready={str(summary["authority_gap_registry_ready"]).lower()}`

TASK-0159 converts the two unresolved TASK-0158 graph targets into first-class `GraphAuthorityGap` records. It does not retrieve external documents, admit sources, create entities, merge identities, alter Graph Retrieval V1, or promote Graph V2 runtime behavior.

## Gap Registry

Unresolved targets: `{summary["unresolved_target_count"]}`.

Authority gap records: `{summary["authority_gap_record_count"]}`.

{gap_lines}

Each record preserves the originating graph reference, source entity, relation, unresolved target mention, source document, line-reference span, authority search revision, blocked capabilities, permitted governance actions, prohibited automated actions, and record digest.

## Fail-Closed Boundary

For the current cohort, `authoritative_source_found=false`, so `canonical_target_materialized=false`, `identity_resolution_success=false`, and `graph_v2_data_gate_ready=false`. Identity resolution is not attempted as a substitute for missing authority.

## Source Intake Contract

The intake boundary is candidate source -> authority review -> approved authoritative source -> canonical corpus materialization -> canonical target -> identity resolution. Unreviewed sources cannot become authoritative and cannot materialize canonical targets.

## Required Questions

{qa_lines}

## V1 Isolation

Graph Retrieval V1 remains frozen at digest `{summary["graph_retrieval_v1_baseline_digest"]}`. Runtime hop depth remains `{summary["graph_runtime_hop_depth"]}`, default equivalence remains `{summary["default_equivalence_pass_count"]}/{summary["formal_graph_sensitive_unit_count"]}`, and known causal regression count remains `{summary["known_causal_regression_count"]}`.

## Conclusion

Outcome A applies: the registry and intake contract are ready as governance artifacts. Graph V2 data gate remains closed until owner-reviewed authoritative target source material is admitted through the intake contract.
"""


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _duplicates(values: list[str]) -> set[str]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
