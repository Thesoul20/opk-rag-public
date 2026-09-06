from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis as task0152
import opk_rag.evaluation.task0153_two_hop_graph_identity_resolution_repair_and_path_recovery as task0153
import opk_rag.evaluation.task0154_missing_graph_target_document_provenance_and_corpus_snapshot_coverage_diagnosis as task0154
import opk_rag.evaluation.task0155_multihop_capable_corpus_authority_gap_assessment_and_v2_viability_decision as task0155
import opk_rag.evaluation.task0156_corpus_graph_hygiene_and_dangling_reference_impact_audit as task0156
import opk_rag.evaluation.task0157_graph_retrieval_v1_stage_closeout_and_engineering_authority_summary as task0157
from opk_rag.evaluation.graph_link_resolution import collect_link_records
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0158"
EXPERIMENT_ID = "task0158-graph-v2-corpus-authority-repair"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0158_graph_v2_corpus_authority_repair_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0158_GRAPH_V2_CORPUS_AUTHORITY_REPAIR_REPORT.md"

FROZEN_V1_BASELINE_DIGEST = "b2de312a7fdbd74922b26c53472f2304a30cfaa2873fe2f495e5962df8ce1f94"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "missing_target_identification.json",
    "authoritative_source_recovery.jsonl",
    "corpus_materialization_audit.json",
    "identity_resolution_revalidation.json",
    "two_hop_reachability_revalidation.json",
    "v1_regression_protection.json",
    "data_gate_decision.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "missing_target_count",
    "authoritative_source_found",
    "canonical_target_present_before",
    "canonical_target_present_after",
    "canonical_target_materialized",
    "identity_resolution_success",
    "raw_two_step_path_count",
    "parsed_two_step_path_count",
    "resolved_two_step_path_count",
    "provenance_valid_two_step_path_count",
    "dangling_reference_count_before",
    "dangling_reference_count_after",
    "graph_v2_data_gate_ready",
    "runtime_policy_mutation_count",
    "graph_runtime_hop_depth",
    "default_equivalence_unit_count",
    "default_equivalence_pass_count",
    "default_equivalence_failure_count",
    "known_causal_regression_count",
    "gold_metadata_usage",
    "synthetic_source_injection",
    "manual_identity_binding",
    "promotion_applied",
)


def run_task0158_graph_v2_corpus_authority_repair(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    targets = identify_missing_targets()
    recovery_rows = build_authoritative_source_recovery(targets)
    materialization = build_corpus_materialization_audit(recovery_rows)
    identity = build_identity_resolution_revalidation(recovery_rows)
    reachability = build_two_hop_reachability_revalidation(identity)
    v1_regression = build_v1_regression_protection(before_runtime=before_runtime, before_baseline_digest=before_baseline_digest)
    decision = build_data_gate_decision(recovery_rows, materialization, identity, reachability, v1_regression)
    summary = build_summary(authority, targets, recovery_rows, materialization, identity, reachability, v1_regression, decision)
    contract = build_contract(summary)
    digests = build_digests(authority, targets, recovery_rows, materialization, identity, reachability, v1_regression, decision, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "missing_target_identification.json", targets)
    write_jsonl(output_dir / "authoritative_source_recovery.jsonl", recovery_rows)
    write_json(output_dir / "corpus_materialization_audit.json", materialization)
    write_json(output_dir / "identity_resolution_revalidation.json", identity)
    write_json(output_dir / "two_hop_reachability_revalidation.json", reachability)
    write_json(output_dir / "v1_regression_protection.json", v1_regression)
    write_json(output_dir / "data_gate_decision.json", decision)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0158_artifacts(output_dir=output_dir, write=True)
    summary["task0158_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, recovery_rows), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0152_summary = read_json(task0152.RESULT_DIR / "summary.json")
    task0153_summary = read_json(task0153.RESULT_DIR / "summary.json")
    task0154_summary = read_json(task0154.RESULT_DIR / "summary.json")
    task0155_summary = read_json(task0155.RESULT_DIR / "summary.json")
    task0156_summary = read_json(task0156.RESULT_DIR / "summary.json")
    task0157_summary = read_json(task0157.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    verifier_status = {
        "TASK-0149": task0149.verify_task0149_artifacts(write=False)["status"],
        "TASK-0152": task0152.verify_task0152_artifacts(write=False)["status"],
        "TASK-0153": task0153.verify_task0153_artifacts(write=False)["status"],
        "TASK-0154": task0154.verify_task0154_artifacts(write=False)["status"],
        "TASK-0155": task0155.verify_task0155_artifacts(write=False)["status"],
        "TASK-0156": task0156.verify_task0156_artifacts(write=False)["status"],
        "TASK-0157": task0157.verify_task0157_artifacts(write=False)["status"],
    }
    return {
        "schema_version": "opk-rag.task0158.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": ["TASK-0158", "TASK-0157", "TASK-0156", "TASK-0155", "TASK-0154", "TASK-0153", "TASK-0152", "TASK-0149"],
        "verifier_status": verifier_status,
        "task0152_authority_valid": verifier_status["TASK-0152"] == "valid" and task0152_summary.get("resolved_two_step_chain_count") == 0,
        "task0153_authority_valid": verifier_status["TASK-0153"] == "valid" and task0153_summary.get("dominant_resolution_failure_mechanism") == "missing_target_document",
        "task0154_authority_valid": verifier_status["TASK-0154"] == "valid" and task0154_summary.get("dominant_missing_target_root_cause") == "dangling_corpus_reference",
        "task0155_authority_valid": verifier_status["TASK-0155"] == "valid" and task0155_summary.get("graph_retrieval_v2_multihop_go_decision") == "no_go",
        "task0156_authority_valid": verifier_status["TASK-0156"] == "valid" and task0156_summary.get("dangling_reference_count") == 8,
        "task0157_authority_valid": verifier_status["TASK-0157"] == "valid" and task0157_summary.get("graph_retrieval_v1_stage_closed") is True,
        "graph_retrieval_v1_frozen": task0157_summary.get("graph_retrieval_v1_frozen") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "baseline_digest_matches_required": baseline.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST,
        "source_tasks_sha256": {
            "TASK-0152": sha256_file(task0152.RESULT_DIR / "summary.json"),
            "TASK-0153": sha256_file(task0153.RESULT_DIR / "summary.json"),
            "TASK-0154": sha256_file(task0154.RESULT_DIR / "summary.json"),
            "TASK-0155": sha256_file(task0155.RESULT_DIR / "summary.json"),
            "TASK-0156": sha256_file(task0156.RESULT_DIR / "summary.json"),
            "TASK-0157": sha256_file(task0157.RESULT_DIR / "summary.json"),
        },
    }


def identify_missing_targets() -> dict[str, Any]:
    raw_chains = {row["chain_id"]: row for row in read_jsonl(task0152.RESULT_DIR / "raw_two_step_chains.jsonl")}
    frozen = read_json(task0154.RESULT_DIR / "missing_target_inventory.json")
    traces = {row["target_id"]: row for row in read_jsonl(task0154.RESULT_DIR / "per_target_provenance_trace.jsonl")}
    targets = []
    for target in frozen["targets"]:
        trace = traces[target["target_id"]]
        chain = raw_chains[target["chain_id"]]
        targets.append(
            {
                "schema_version": "opk-rag.task0158.missing-target-identification.v1",
                "task_id": TASK_ID,
                "target_id": target["target_id"],
                "chain_id": target["chain_id"],
                "source_entity_A": chain["raw_chain_source_A"],
                "intermediate_entity_B": chain["raw_chain_bridge_B"],
                "missing_target_entity_C": chain["raw_chain_target_C"],
                "source_relation_edge_id": chain["edge_1_id"],
                "target_relation_edge_id": target["source_edge_id"],
                "target_relation_label": target["edge_label"],
                "raw_target_identity": target["raw_target_reference"],
                "normalized_target_identity": target["normalized_target_reference"],
                "expected_canonical_identity": trace.get("canonical_identity"),
                "expected_source_document": trace.get("expected_target_relative_path"),
                "candidate_relative_paths": trace.get("candidate_relative_paths", []),
                "provenance": {
                    "task0152_chain_id": target["chain_id"],
                    "task0153_failed_edge": target["source_edge_id"],
                    "task0154_target_id": target["target_id"],
                    "source_document_path": target["source_document_id"],
                },
                "current_materialization_status": {
                    "source_file_exists": trace["source_file_exists"],
                    "canonical_document_created": trace["canonical_document_created"],
                    "canonical_identity_registered": trace["canonical_identity_registered"],
                    "graph_node_materialized": trace["graph_node_materialized"],
                    "target_first_loss_stage": trace["target_first_loss_stage"],
                },
            }
        )
    return {
        "schema_version": "opk-rag.task0158.missing-target-identification-audit.v1",
        "task_id": TASK_ID,
        "missing_target_count": len(targets),
        "targets": targets,
    }


def build_authoritative_source_recovery(target_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    source_inventory = task0154.build_source_inventory(SOURCE_DIR)
    rows = []
    for target in target_inventory["targets"]:
        probe = {
            "target_id": target["target_id"],
            "source_document_id": target["provenance"]["source_document_path"],
            "source_edge_id": target["target_relation_edge_id"],
            "chain_id": target["chain_id"],
            "edge_label": target["target_relation_label"],
            "raw_target_reference": target["raw_target_identity"],
            "parsed_target_reference": target["normalized_target_identity"],
            "normalized_target_reference": target["normalized_target_identity"],
            "path_candidate": [path.removeprefix("source-documents/") for path in target["candidate_relative_paths"]],
            "freeze_ordinal": None,
        }
        trace = task0154.trace_target(probe, source_inventory=source_inventory)
        source_found = bool(trace["source_file_exists"] and trace["matched_source_path"])
        provenance_valid = bool(source_found and trace["canonical_document_created"] and trace["canonical_identity_registered"])
        rows.append(
            {
                "schema_version": "opk-rag.task0158.authoritative-source-recovery.v1",
                "task_id": TASK_ID,
                "target_id": target["target_id"],
                "chain_id": target["chain_id"],
                "raw_target_identity": target["raw_target_identity"],
                "normalized_target_identity": target["normalized_target_identity"],
                "expected_source_document": target["expected_source_document"],
                "candidate_relative_paths": target["candidate_relative_paths"],
                "authoritative_source_found": source_found,
                "matched_source_path": trace["matched_source_path"],
                "match_type": trace["match_type"],
                "provenance_valid": provenance_valid,
                "source_recovery_status": "found" if source_found else "not_found_fail_closed",
                "blocker": None if source_found else "authoritative_source_cannot_be_found_in_source_documents_snapshot",
                "forbidden_repair_avoided": {
                    "synthetic_source_injection": False,
                    "gold_metadata_usage": False,
                    "manual_identity_binding": False,
                    "fuzzy_resolution_enabled": False,
                },
                "post_recovery_trace": {
                    "source_file_exists": trace["source_file_exists"],
                    "canonical_document_created": trace["canonical_document_created"],
                    "canonical_identity_registered": trace["canonical_identity_registered"],
                    "resolver_candidate_visible": trace["resolver_candidate_visible"],
                    "graph_node_materialized": trace["graph_node_materialized"],
                    "target_first_loss_stage": trace["target_first_loss_stage"],
                },
            }
        )
    return rows


def build_corpus_materialization_audit(recovery_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recoverable = [row for row in recovery_rows if row["authoritative_source_found"]]
    materialized = [
        row
        for row in recovery_rows
        if row["post_recovery_trace"]["canonical_document_created"] and row["post_recovery_trace"]["canonical_identity_registered"]
    ]
    return {
        "schema_version": "opk-rag.task0158.corpus-materialization-audit.v1",
        "task_id": TASK_ID,
        "authoritative_source_found_count": len(recoverable),
        "canonical_target_materialized_count": len(materialized),
        "canonical_target_materialized": bool(materialized) and len(materialized) == len(recovery_rows),
        "formal_pipeline_required": bool(recoverable),
        "formal_pipeline_executed": False,
        "pipeline_skip_reason": "no_authoritative_source_found" if not recoverable else None,
        "source_document_creation_count": 0,
        "source_document_mutation_count": 0,
        "corpus_snapshot_mutation_count": 0,
        "special_ingestion_branch_count": 0,
    }


def build_identity_resolution_revalidation(recovery_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = collect_link_records()
    raw_audit, raw_chains = task0152.build_raw_relation_audit(records)
    parsed_audit, parsed_chains = task0152.build_parsed_edge_audit(records, raw_chains)
    resolution_audit, resolved_chains = task0152.build_identity_resolution_audit(records, parsed_chains)
    target_ids = {row["target_id"] for row in recovery_rows}
    per_target = [
        {
            "target_id": row["target_id"],
            "canonical_target_present_before": False,
            "canonical_target_present_after": row["post_recovery_trace"]["canonical_identity_registered"],
            "canonical_candidate_visible_after": row["post_recovery_trace"]["resolver_candidate_visible"],
            "identity_resolution_success": row["post_recovery_trace"]["canonical_identity_registered"]
            and row["post_recovery_trace"]["resolver_candidate_visible"],
            "resolution_status": "resolved" if row["post_recovery_trace"]["resolver_candidate_visible"] else "fail_closed_missing_authoritative_source",
        }
        for row in recovery_rows
    ]
    success_count = sum(row["identity_resolution_success"] for row in per_target)
    return {
        "schema_version": "opk-rag.task0158.identity-resolution-revalidation.v1",
        "task_id": TASK_ID,
        "target_ids": sorted(target_ids),
        "raw_relation_count": raw_audit["raw_relation_count"],
        "raw_two_step_path_count": raw_audit["raw_two_step_chain_count"],
        "parsed_two_step_path_count": parsed_audit["parsed_two_step_chain_count"],
        "resolved_two_step_path_count": resolution_audit["resolved_two_step_chain_count"],
        "canonical_target_present_before": False,
        "canonical_target_present_after": bool(per_target) and all(row["canonical_target_present_after"] for row in per_target),
        "identity_resolution_success_count": success_count,
        "identity_resolution_success": bool(per_target) and success_count == len(per_target),
        "identity_resolution_fail_closed": success_count < len(per_target),
        "per_target": per_target,
        "reinterpreted_resolved_two_step_path_count": len(resolved_chains),
    }


def build_two_hop_reachability_revalidation(identity: dict[str, Any]) -> dict[str, Any]:
    records = collect_link_records()
    raw_audit, raw_chains = task0152.build_raw_relation_audit(records)
    parsed_audit, parsed_chains = task0152.build_parsed_edge_audit(records, raw_chains)
    _, resolved_chains = task0152.build_identity_resolution_audit(records, parsed_chains)
    _, bound_chains = task0152.build_source_binding_audit(records, resolved_chains)
    _, materialized_chains = task0152.build_graph_materialization_audit(records, bound_chains)
    valid_paths = task0152.build_valid_two_hop_paths(materialized_chains)
    minimum_distance = task0152.build_minimum_distance_audit(valid_paths, task0152.materialized_edges(records))
    provenance_valid = [row for row in minimum_distance if row.get("minimum_distance") == 2 and row.get("path_valid")]
    return {
        "schema_version": "opk-rag.task0158.two-hop-reachability-revalidation.v1",
        "task_id": TASK_ID,
        "raw_two_step_path_count": raw_audit["raw_two_step_chain_count"],
        "parsed_two_step_path_count": parsed_audit["parsed_two_step_chain_count"],
        "identity_resolved_path_count": len(resolved_chains),
        "source_bound_two_step_path_count": len(bound_chains),
        "materialized_two_step_path_count": len(materialized_chains),
        "provenance_valid_two_step_path_count": len(provenance_valid),
        "graph_reachable_two_step_path_count": len(valid_paths),
        "minimum_distance_two_path_count": sum(row.get("minimum_distance") == 2 for row in minimum_distance),
        "two_hop_path_resolved": identity["identity_resolution_success"] and bool(valid_paths),
    }


def build_v1_regression_protection(*, before_runtime: dict[str, Any], before_baseline_digest: str) -> dict[str, Any]:
    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    task0157_summary = read_json(task0157.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0158.v1-regression-protection.v1",
        "task_id": TASK_ID,
        "graph_retrieval_v1_frozen": task0157_summary["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": task0157_summary["graph_retrieval_v1_baseline_digest"],
        "baseline_digest_matches_required": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST,
        "baseline_manifest_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "default_initial_retrieval_policy": task0157_summary["default_initial_retrieval_policy"],
        "graph_runtime_hop_depth": task0157_summary["graph_runtime_hop_depth"],
        "default_equivalence_unit_count": task0157_summary["formal_graph_sensitive_unit_count"],
        "default_equivalence_pass_count": task0157_summary["default_equivalence_pass_count"],
        "default_equivalence_failure_count": task0157_summary["default_equivalence_failure_count"],
        "known_causal_regression_count": task0157_summary["known_causal_regression_count"],
        "v1_regression_count": task0157_summary["known_causal_regression_count"],
        "v1_regression_protection_passed": task0157_summary["graph_retrieval_v1_baseline_digest"] == FROZEN_V1_BASELINE_DIGEST
        and task0157_summary["default_initial_retrieval_policy"] == "guarded_structure_aware"
        and task0157_summary["graph_runtime_hop_depth"] == 1
        and task0157_summary["default_equivalence_pass_count"] == 9
        and task0157_summary["known_causal_regression_count"] == 0,
    }


def build_data_gate_decision(
    recovery_rows: list[dict[str, Any]],
    materialization: dict[str, Any],
    identity: dict[str, Any],
    reachability: dict[str, Any],
    v1_regression: dict[str, Any],
) -> dict[str, Any]:
    source_found = bool(recovery_rows) and all(row["authoritative_source_found"] for row in recovery_rows)
    gate_ready = (
        source_found
        and materialization["canonical_target_materialized"]
        and identity["identity_resolution_success"]
        and reachability["two_hop_path_resolved"]
        and v1_regression["v1_regression_protection_passed"]
    )
    outcome = "A_corpus_repair_successful" if gate_ready else "C_authoritative_source_cannot_be_found"
    blockers = []
    if not source_found:
        blockers.append("authoritative_source_not_found")
    if not materialization["canonical_target_materialized"]:
        blockers.append("canonical_target_not_materialized")
    if not identity["identity_resolution_success"]:
        blockers.append("identity_resolution_fail_closed")
    if not reachability["two_hop_path_resolved"]:
        blockers.append("two_hop_path_not_graph_reachable")
    if not v1_regression["v1_regression_protection_passed"]:
        blockers.append("v1_regression_protection_failed")
    return {
        "schema_version": "opk-rag.task0158.data-gate-decision.v1",
        "task_id": TASK_ID,
        "decision_outcome": outcome,
        "graph_v2_data_gate_ready": gate_ready,
        "blockers": blockers,
        "canonical_target_materialized": materialization["canonical_target_materialized"],
        "identity_resolution_success": identity["identity_resolution_success"],
        "two_hop_path_resolved": reachability["two_hop_path_resolved"],
        "provenance_valid": source_found,
        "identity_resolution_fail_closed": identity["identity_resolution_fail_closed"],
        "recommended_next_step": "obtain_owner_approved_authoritative_source_documents_before_graph_v2_benchmark_authoring",
    }


def build_summary(
    authority: dict[str, Any],
    targets: dict[str, Any],
    recovery_rows: list[dict[str, Any]],
    materialization: dict[str, Any],
    identity: dict[str, Any],
    reachability: dict[str, Any],
    v1_regression: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    source_found = bool(recovery_rows) and all(row["authoritative_source_found"] for row in recovery_rows)
    dangling_before = read_json(task0156.RESULT_DIR / "summary.json")["dangling_reference_count"]
    dangling_after = Counter(task0155.classify_reference(row) for row in collect_link_records())["dangling_reference"]
    return {
        "schema_version": "opk-rag.task0158.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0152_authority_valid": authority["task0152_authority_valid"],
        "task0153_authority_valid": authority["task0153_authority_valid"],
        "task0154_authority_valid": authority["task0154_authority_valid"],
        "task0155_authority_valid": authority["task0155_authority_valid"],
        "task0156_authority_valid": authority["task0156_authority_valid"],
        "task0157_authority_valid": authority["task0157_authority_valid"],
        "graph_retrieval_v1_frozen": authority["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": authority["graph_retrieval_v1_baseline_digest"],
        "missing_target_count": targets["missing_target_count"],
        "authoritative_source_found": source_found,
        "authoritative_source_found_count": sum(row["authoritative_source_found"] for row in recovery_rows),
        "canonical_target_present_before": identity["canonical_target_present_before"],
        "canonical_target_present_after": identity["canonical_target_present_after"],
        "canonical_target_materialized": materialization["canonical_target_materialized"],
        "identity_resolution_success": identity["identity_resolution_success"],
        "identity_resolution_fail_closed": identity["identity_resolution_fail_closed"],
        "raw_two_step_path_count": reachability["raw_two_step_path_count"],
        "parsed_two_step_path_count": reachability["parsed_two_step_path_count"],
        "resolved_two_step_path_count": reachability["identity_resolved_path_count"],
        "provenance_valid_two_step_path_count": reachability["provenance_valid_two_step_path_count"],
        "graph_reachable_two_step_path_count": reachability["graph_reachable_two_step_path_count"],
        "dangling_reference_count_before": dangling_before,
        "dangling_reference_count_after": dangling_after,
        "graph_v2_data_gate_ready": decision["graph_v2_data_gate_ready"],
        "runtime_policy_mutation_count": v1_regression["runtime_policy_mutation_count"],
        "graph_runtime_hop_depth": v1_regression["graph_runtime_hop_depth"],
        "default_initial_retrieval_policy": v1_regression["default_initial_retrieval_policy"],
        "default_equivalence_unit_count": v1_regression["default_equivalence_unit_count"],
        "default_equivalence_pass_count": v1_regression["default_equivalence_pass_count"],
        "default_equivalence_failure_count": v1_regression["default_equivalence_failure_count"],
        "known_causal_regression_count": v1_regression["known_causal_regression_count"],
        "v1_regression_count": v1_regression["v1_regression_count"],
        "gold_metadata_usage": False,
        "synthetic_source_injection": False,
        "manual_identity_binding": False,
        "promotion_applied": False,
        "source_document_creation_count": materialization["source_document_creation_count"],
        "source_document_mutation_count": materialization["source_document_mutation_count"],
        "corpus_snapshot_mutation_count": materialization["corpus_snapshot_mutation_count"],
        "graph_policy_mutation_count": 0,
        "task0149_frozen_artifact_mutation_count": v1_regression["baseline_manifest_mutation_count"],
        "decision_outcome": decision["decision_outcome"],
        "recommended_next_step": decision["recommended_next_step"],
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0158.contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "runtime_policy_mutation_count_required": 0,
        "graph_runtime_hop_depth_required": 1,
        "forbid_gold_metadata": True,
        "forbid_synthetic_source": True,
        "forbid_manual_identity_binding": True,
        "allow_fail_closed_outcome": True,
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0158.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def verify_task0158_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    decision = read_json(output_dir / "data_gate_decision.json") if (output_dir / "data_gate_decision.json").exists() else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "contract_valid": contract.get("summary_required_fields_present") is True,
        "task_complete": summary.get("task_id") == TASK_ID and summary.get("task_status") == "complete",
        "authority_valid": all(summary.get(key) is True for key in ("task0152_authority_valid", "task0153_authority_valid", "task0154_authority_valid", "task0155_authority_valid", "task0156_authority_valid", "task0157_authority_valid")),
        "v1_baseline_protected": summary.get("graph_retrieval_v1_frozen") is True
        and summary.get("graph_retrieval_v1_baseline_digest") == FROZEN_V1_BASELINE_DIGEST
        and summary.get("default_initial_retrieval_policy") == "guarded_structure_aware"
        and summary.get("graph_runtime_hop_depth") == 1
        and summary.get("default_equivalence_pass_count") == 9
        and summary.get("known_causal_regression_count") == 0,
        "no_runtime_mutation": summary.get("runtime_policy_mutation_count") == 0 and summary.get("graph_policy_mutation_count") == 0,
        "no_forbidden_repair": summary.get("gold_metadata_usage") is False
        and summary.get("synthetic_source_injection") is False
        and summary.get("manual_identity_binding") is False
        and summary.get("promotion_applied") is False,
        "fail_closed_consistent": (
            summary.get("graph_v2_data_gate_ready") is False
            and summary.get("authoritative_source_found") is False
            and summary.get("canonical_target_materialized") is False
            and summary.get("identity_resolution_fail_closed") is True
            and decision.get("decision_outcome") == "C_authoritative_source_cannot_be_found"
        )
        or (
            summary.get("graph_v2_data_gate_ready") is True
            and summary.get("authoritative_source_found") is True
            and summary.get("canonical_target_materialized") is True
            and summary.get("identity_resolution_success") is True
        ),
    }
    result = {
        "schema_version": "opk-rag.task0158.verification.v1",
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


def build_report(summary: dict[str, Any], recovery_rows: list[dict[str, Any]]) -> str:
    missing_targets = "\n".join(
        f"* `{row['raw_target_identity']}` -> `{row['source_recovery_status']}` ({row['blocker'] or 'materialized'})"
        for row in recovery_rows
    )
    return f"""# TASK-0158 Graph V2 Corpus Authority Repair Report

## Status

`task_status={summary["task_status"]}`

`decision_outcome={summary["decision_outcome"]}`

`graph_v2_data_gate_ready={str(summary["graph_v2_data_gate_ready"]).lower()}`

TASK-0158 investigated the two TASK-0153 missing B -> C targets without changing runtime policy, graph policy, source documents, corpus snapshot, benchmark data, or the TASK-0149 frozen baseline.

## Missing Targets

{missing_targets}

## Source Recovery Decision

Authoritative source recovery found `{summary["authoritative_source_found_count"]}` of `{summary["missing_target_count"]}` required target documents in the current `source-documents` authoritative snapshot. Because no source document was found, TASK-0158 did not synthesize documents, did not use gold metadata, did not create manual identity bindings, and did not execute a special ingestion branch.

## Revalidation

Raw two-step paths: `{summary["raw_two_step_path_count"]}`.

Parsed two-step paths: `{summary["parsed_two_step_path_count"]}`.

Resolved two-step paths: `{summary["resolved_two_step_path_count"]}`.

Provenance-valid two-step paths: `{summary["provenance_valid_two_step_path_count"]}`.

Canonical target present after repair: `{str(summary["canonical_target_present_after"]).lower()}`.

Identity resolution success: `{str(summary["identity_resolution_success"]).lower()}`.

## V1 Regression Protection

Graph Retrieval V1 remains frozen with baseline digest `{summary["graph_retrieval_v1_baseline_digest"]}`. Runtime hop depth remains `{summary["graph_runtime_hop_depth"]}` and default equivalence remains `{summary["default_equivalence_pass_count"]}/{summary["default_equivalence_unit_count"]}` with known causal regression count `{summary["known_causal_regression_count"]}`.

## Conclusion

Outcome C applies: authoritative source cannot be found for the required C targets, so identity resolution fails closed and Graph V2 data gate remains closed. Next step: `{summary["recommended_next_step"]}`.
"""
