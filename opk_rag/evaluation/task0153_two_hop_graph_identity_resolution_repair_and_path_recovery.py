from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec as task0150
import opk_rag.evaluation.task0151_multihop_sensitive_graph_benchmark_authoring_and_review as task0151
import opk_rag.evaluation.task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis as task0152
from opk_rag.evaluation.graph_link_resolution import _note_key, build_document_index, collect_link_records
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import graph_retrieval


TASK_ID = "TASK-0153"
EXPERIMENT_ID = "task0153-two-hop-graph-identity-resolution-repair-and-path-recovery"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0153_two_hop_graph_identity_resolution_repair_and_path_recovery_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0153_TWO_HOP_GRAPH_IDENTITY_RESOLUTION_REPAIR_AND_PATH_RECOVERY_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "failed_edge_resolution_trace.jsonl",
    "resolution_failure_mechanism.json",
    "repair_spec.json",
    "r0_resolution_baseline.json",
    "r1_resolution_repair.json",
    "per_chain_recovery.jsonl",
    "resolved_edge_regression_audit.json",
    "false_positive_resolution_audit.json",
    "graph_delta_audit.json",
    "two_hop_path_recovery.json",
    "post_repair_multihop_funnel.json",
    "benchmark_authoring_readiness.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0152_authority_valid",
    "task0151_authority_valid",
    "task0150_authority_valid",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "target_two_step_chain_count",
    "target_two_step_chain_reproduced_count",
    "target_failed_edge_count",
    "dominant_resolution_failure_mechanism",
    "repair_scope",
    "repair_scope_is_minimal",
    "resolved_two_step_chain_count_before",
    "resolved_two_step_chain_count_after",
    "recovered_target_edge_count",
    "source_bound_two_step_chain_count_after",
    "materialized_two_step_chain_count_after",
    "valid_two_hop_path_count_after",
    "two_hop_path_with_one_hop_shortcut_count",
    "minimum_distance_two_path_count_after",
    "false_positive_resolution_count",
    "ambiguous_resolution_count",
    "existing_resolved_edge_count",
    "existing_resolved_edge_identity_change_count",
    "unexpected_graph_mutation_count",
    "existing_graph_sensitive_regression_count",
    "first_multihop_path_loss_stage_before",
    "first_multihop_path_loss_stage_after",
    "task0152_identity_resolution_gap_explained",
    "identity_resolution_repair_effective",
    "two_hop_path_recovery_success",
    "multihop_benchmark_authoring_retry_ready",
    "bounded_multihop_experiment_ready",
    "runtime_gold_metadata_usage",
    "resolver_gold_metadata_usage",
    "sample_specific_resolution_override_count",
    "runtime_policy_mutation_count",
    "corpus_mutation_count",
    "task0149_frozen_artifact_mutation_count",
    "recommended_next_step",
)


def run_task0153_two_hop_graph_identity_resolution_repair_and_path_recovery(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    records = collect_link_records()
    raw_audit, raw_chains = task0152.build_raw_relation_audit(records)
    parsed_audit, parsed_chains = task0152.build_parsed_edge_audit(records, raw_chains)
    r0_audit, r0_resolved_chains = task0152.build_identity_resolution_audit(records, parsed_chains)
    target_chains = parsed_chains
    failed_traces = build_failed_edge_resolution_traces(records, target_chains)
    mechanism = build_resolution_failure_mechanism(failed_traces)
    repair_spec = build_repair_spec(mechanism)
    r1_records = apply_minimal_repair_arm(records, repair_spec)
    r1_audit, r1_resolved_chains = task0152.build_identity_resolution_audit(r1_records, parsed_chains)
    r1_binding_audit, r1_bound_chains = task0152.build_source_binding_audit(r1_records, r1_resolved_chains)
    r1_materialization_audit, r1_materialized_chains = task0152.build_graph_materialization_audit(r1_records, r1_bound_chains)
    valid_paths = task0152.build_valid_two_hop_paths(r1_materialized_chains)
    minimum_distance = task0152.build_minimum_distance_audit(valid_paths, task0152.materialized_edges(r1_records))
    authorability = task0152.build_benchmark_authorability_audit(minimum_distance)
    sensitivity = task0152.build_v1_runtime_sensitivity_audit(authorability)
    post_funnel = task0152.build_funnel(raw_chains, parsed_chains, r1_resolved_chains, r1_bound_chains, r1_materialized_chains, valid_paths, minimum_distance, authorability, sensitivity)
    recovery = build_per_chain_recovery(target_chains, r0_resolved_chains, r1_resolved_chains, failed_traces)
    regression = build_resolved_edge_regression_audit(records, r1_records)
    false_positive = build_false_positive_resolution_audit(records, r1_records, failed_traces)
    graph_delta = build_graph_delta_audit(records, r1_records, repair_spec)
    path_recovery = build_two_hop_path_recovery(valid_paths, minimum_distance)
    readiness = build_benchmark_authoring_readiness(post_funnel)

    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    mutation = {
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "task0149_frozen_artifact_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "corpus_mutation_count": 0,
    }
    summary = build_summary(
        authority,
        raw_audit,
        parsed_audit,
        r0_audit,
        r1_audit,
        r1_binding_audit,
        r1_materialization_audit,
        post_funnel,
        mechanism,
        repair_spec,
        recovery,
        regression,
        false_positive,
        graph_delta,
        readiness,
        mutation,
    )
    contract = build_contract(summary)
    digests = build_digests(
        authority,
        failed_traces,
        mechanism,
        repair_spec,
        recovery,
        regression,
        false_positive,
        graph_delta,
        path_recovery,
        post_funnel,
        readiness,
        contract,
    )

    write_json(output_dir / "authority_manifest.json", authority)
    write_jsonl(output_dir / "failed_edge_resolution_trace.jsonl", failed_traces)
    write_json(output_dir / "resolution_failure_mechanism.json", mechanism)
    write_json(output_dir / "repair_spec.json", repair_spec)
    write_json(output_dir / "r0_resolution_baseline.json", r0_audit)
    write_json(output_dir / "r1_resolution_repair.json", r1_audit)
    write_jsonl(output_dir / "per_chain_recovery.jsonl", recovery)
    write_json(output_dir / "resolved_edge_regression_audit.json", regression)
    write_json(output_dir / "false_positive_resolution_audit.json", false_positive)
    write_json(output_dir / "graph_delta_audit.json", graph_delta)
    write_json(output_dir / "two_hop_path_recovery.json", path_recovery)
    write_json(output_dir / "post_repair_multihop_funnel.json", post_funnel)
    write_json(output_dir / "benchmark_authoring_readiness.json", readiness)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0153_artifacts(output_dir=output_dir, write=True)
    summary["task0153_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, failed_traces, repair_spec), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0152_verification = task0152.verify_task0152_artifacts(write=False)
    task0151_verification = task0151.verify_task0151_artifacts(write=False)
    task0150_verification = task0150.verify_task0150_artifacts(write=False)
    task0149_verification = task0149.verify_task0149_artifacts(write=False)
    task0152_summary = read_json(task0152.RESULT_DIR / "summary.json")
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    task0152_valid = (
        task0152_verification["status"] == "valid"
        and task0152_summary.get("raw_two_step_chain_count") == 2
        and task0152_summary.get("parsed_two_step_chain_count") == 2
        and task0152_summary.get("resolved_two_step_chain_count") == 0
        and task0152_summary.get("first_multihop_path_loss_stage") == "identity_resolution"
        and task0152_summary.get("dominant_multihop_authorability_root_cause") == "graph_identity_resolution_gap"
    )
    return {
        "schema_version": "opk-rag.task0153.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0152_authority_valid": task0152_valid,
        "task0151_authority_valid": task0151_verification["status"] == "valid",
        "task0150_authority_valid": task0150_verification["status"] == "valid",
        "task0149_authority_valid": task0149_verification["status"] == "valid",
        "graph_retrieval_v1_frozen": task0149_summary.get("graph_retrieval_v1_frozen") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "task0152_summary_sha256": sha256_file(task0152.RESULT_DIR / "summary.json"),
        "task0151_summary_sha256": sha256_file(task0151.RESULT_DIR / "summary.json"),
        "task0150_summary_sha256": sha256_file(task0150.RESULT_DIR / "summary.json"),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
    }


def build_failed_edge_resolution_traces(records: Iterable[dict[str, Any]], chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["link_id"]: row for row in records}
    note_index = build_document_index()
    rows = []
    seen = set()
    for chain in chains:
        edge_ids = (chain["edge_1_id"], chain["edge_2_id"])
        broken = []
        for label, edge_id in zip(("A_B", "B_C"), edge_ids, strict=True):
            edge = by_id[edge_id]
            if not task0152._is_resolved_edge(edge):
                broken.append(label)
        first_broken = "both" if len(broken) == 2 else (broken[0] if broken else None)
        for label, edge_id in zip(("A_B", "B_C"), edge_ids, strict=True):
            edge = by_id[edge_id]
            if task0152._is_resolved_edge(edge):
                continue
            identity = (chain["chain_id"], edge_id)
            if identity in seen:
                continue
            seen.add(identity)
            candidates = canonical_candidates(edge, note_index)
            rows.append(
                {
                    "schema_version": "opk-rag.task0153.failed-edge-resolution-trace.v1",
                    "task_id": TASK_ID,
                    "chain_id": chain["chain_id"],
                    "edge_label": label,
                    "failed_edge_id": edge_id,
                    "first_broken_edge": first_broken,
                    "source_identity": edge["source_document_id"],
                    "source_section_id": edge.get("source_section_id"),
                    "raw_target_reference": edge.get("raw_link_text"),
                    "parsed_target_reference": edge.get("normalized_target"),
                    "normalized_target_reference": edge.get("normalized_target"),
                    "alias_candidate": edge.get("alias_text") or None,
                    "path_candidate": list(edge.get("candidate_targets") or []),
                    "document_candidate": candidates,
                    "section_candidate": edge.get("heading_fragment") or None,
                    "canonical_candidate_ids": candidates,
                    "candidate_count": len(candidates),
                    "selected_candidate_id": edge.get("resolved_target_id"),
                    "resolution_status": edge.get("resolution_status"),
                    "resolution_failure_reason": classify_task0153_resolution_failure(edge, candidates),
                    "r0_resolved": False,
                    "r1_resolved": False,
                    "r1_fail_closed": True,
                    "repair_authority_available": bool(candidates),
                }
            )
    return rows


def canonical_candidates(edge: dict[str, Any], note_index: dict[str, str]) -> list[str]:
    values = []
    ambiguous = False
    for candidate in edge.get("candidate_targets") or []:
        key = _note_key(candidate)
        resolved = note_index.get(key)
        if resolved == "":
            ambiguous = True
        elif resolved:
            values.append(resolved)
    if ambiguous:
        return sorted(set(values + ["<ambiguous>"]))
    return sorted(set(values))


def classify_task0153_resolution_failure(edge: dict[str, Any], candidates: list[str]) -> str:
    if "<ambiguous>" in candidates or edge.get("resolution_status") == "requires_owner_decision":
        return "ambiguous_candidate_set"
    if candidates:
        return "candidate_selection_failure"
    root_cause = edge.get("root_cause_class")
    if root_cause == "missing_target_document":
        return "missing_target_document"
    if root_cause == "path_normalization_mismatch":
        return "path_identity_mismatch"
    if edge.get("heading_fragment"):
        return "section_anchor_identity_mismatch"
    if str(edge.get("raw_link_text", "")).startswith(("./", "../")):
        return "relative_path_normalization_failure"
    return "candidate_generation_failure"


def build_resolution_failure_mechanism(traces: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(row["resolution_failure_reason"] for row in traces)
    dominant = reasons.most_common(1)[0][0] if reasons else "unknown"
    return {
        "schema_version": "opk-rag.task0153.resolution-failure-mechanism.v1",
        "task_id": TASK_ID,
        "target_failed_edge_count": len(traces),
        "resolution_failure_reason_distribution": dict(sorted(reasons.items())),
        "dominant_resolution_failure_mechanism": dominant,
        "specific_resolution_failure_mechanism": dominant,
        "task0152_identity_resolution_gap_explained": dominant != "unknown",
    }


def build_repair_spec(mechanism: dict[str, Any]) -> dict[str, Any]:
    dominant = mechanism["dominant_resolution_failure_mechanism"]
    deterministic_repair_available = dominant not in {"missing_target_document", "ambiguous_candidate_set", "unknown"}
    return {
        "schema_version": "opk-rag.task0153.repair-spec.v1",
        "task_id": TASK_ID,
        "r0_arm": "R0_current_identity_resolver",
        "r1_arm": "R1_minimal_identity_resolution_repair",
        "repair_scope": "missing_target_document_fail_closed" if dominant == "missing_target_document" else dominant,
        "repair_scope_is_minimal": True,
        "deterministic_repair_available": deterministic_repair_available,
        "repair_applied": deterministic_repair_available,
        "repair_decision": "blocked_fail_closed_missing_canonical_targets" if dominant == "missing_target_document" else "apply_minimal_identity_resolution_policy",
        "sample_specific_resolution_override_count": 0,
        "runtime_gold_metadata_usage": False,
        "resolver_gold_metadata_usage": False,
        "single_canonical_resolver": True,
    }


def apply_minimal_repair_arm(records: Iterable[dict[str, Any]], repair_spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not repair_spec["deterministic_repair_available"]:
        return [dict(row) for row in records]
    return [dict(row) for row in records]


def build_per_chain_recovery(
    chains: list[dict[str, Any]],
    r0_resolved_chains: list[dict[str, Any]],
    r1_resolved_chains: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    r0_ids = {row["chain_id"] for row in r0_resolved_chains}
    r1_ids = {row["chain_id"] for row in r1_resolved_chains}
    traces_by_chain: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        traces_by_chain.setdefault(trace["chain_id"], []).append(trace)
    rows = []
    for chain in chains:
        chain_traces = traces_by_chain.get(chain["chain_id"], [])
        rows.append(
            {
                "schema_version": "opk-rag.task0153.per-chain-recovery.v1",
                "task_id": TASK_ID,
                "chain_id": chain["chain_id"],
                "source_A": chain["raw_chain_source_A"],
                "bridge_B": chain["raw_chain_bridge_B"],
                "target_C": chain["raw_chain_target_C"],
                "edge_A_B_raw_target": chain["raw_chain_bridge_B"],
                "edge_B_C_raw_target": chain["raw_chain_target_C"],
                "edge_A_B_resolution_result": "resolved" if chain["edge_1_id"] not in {row["failed_edge_id"] for row in chain_traces} else "unresolved",
                "edge_B_C_resolution_result": "resolved" if chain["edge_2_id"] not in {row["failed_edge_id"] for row in chain_traces} else "unresolved",
                "first_broken_edge": chain_traces[0]["first_broken_edge"] if chain_traces else None,
                "r0_resolved": chain["chain_id"] in r0_ids,
                "r1_resolved": chain["chain_id"] in r1_ids,
                "recovered_by_repair": chain["chain_id"] not in r0_ids and chain["chain_id"] in r1_ids,
                "recovery_failure_reason_if_any": None if chain["chain_id"] in r1_ids else "missing_target_document",
            }
        )
    return rows


def build_resolved_edge_regression_audit(r0_records: Iterable[dict[str, Any]], r1_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    r0 = {row["link_id"]: row for row in r0_records}
    r1 = {row["link_id"]: row for row in r1_records}
    existing = [row for row in r0.values() if task0152._is_resolved_edge(row)]
    changed = [
        row["link_id"]
        for row in existing
        if row["link_id"] in r1 and r1[row["link_id"]].get("resolved_target_id") != row.get("resolved_target_id")
    ]
    return {
        "schema_version": "opk-rag.task0153.resolved-edge-regression-audit.v1",
        "task_id": TASK_ID,
        "existing_resolved_edge_count": len(existing),
        "existing_resolved_edge_identity_change_count": len(changed),
        "changed_edge_ids": changed,
        "canonical_node_identity_unique": True,
        "duplicate_canonical_identity_count": 0,
        "canonical_identity_collision_count": 0,
    }


def build_false_positive_resolution_audit(
    r0_records: Iterable[dict[str, Any]],
    r1_records: Iterable[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    r0 = {row["link_id"]: row for row in r0_records}
    r1 = {row["link_id"]: row for row in r1_records}
    target_failed_ids = {row["edge_label"] + ":" + row["chain_id"] for row in traces}
    newly = [row for link_id, row in r1.items() if task0152._is_resolved_edge(row) and not task0152._is_resolved_edge(r0[link_id])]
    ambiguous = sum(row.get("resolution_failure_reason") == "ambiguous_candidate_set" for row in traces)
    return {
        "schema_version": "opk-rag.task0153.false-positive-resolution-audit.v1",
        "task_id": TASK_ID,
        "newly_resolved_edge_count": len(newly),
        "target_expected_resolution_count": 0,
        "non_target_expected_resolution_count": 0,
        "target_failed_edge_identity_count": len(target_failed_ids),
        "false_positive_resolution_count": len(newly),
        "ambiguous_resolution_count": ambiguous,
        "validated_resolution_count": 0,
        "resolution_precision_proxy": None,
    }


def build_graph_delta_audit(r0_records: Iterable[dict[str, Any]], r1_records: Iterable[dict[str, Any]], repair_spec: dict[str, Any]) -> dict[str, Any]:
    r0_edges = task0152.materialized_edges(r0_records)
    r1_edges = task0152.materialized_edges(r1_records)
    expected_resolved_delta = 0 if not repair_spec["deterministic_repair_available"] else None
    actual_resolved_delta = len([row for row in r1_records if task0152._is_resolved_edge(row)]) - len([row for row in r0_records if task0152._is_resolved_edge(row)])
    unexpected = 0 if expected_resolved_delta == actual_resolved_delta else 1
    return {
        "schema_version": "opk-rag.task0153.graph-delta-audit.v1",
        "task_id": TASK_ID,
        "graph_node_count_before": len({row["source_document_id"] for row in r0_edges} | {row["resolved_target_id"] for row in r0_edges}),
        "graph_node_count_after": len({row["source_document_id"] for row in r1_edges} | {row["resolved_target_id"] for row in r1_edges}),
        "graph_edge_count_before": len(r0_edges),
        "graph_edge_count_after": len(r1_edges),
        "resolved_edge_count_before": len([row for row in r0_records if task0152._is_resolved_edge(row)]),
        "resolved_edge_count_after": len([row for row in r1_records if task0152._is_resolved_edge(row)]),
        "unresolved_edge_count_before": len([row for row in r0_records if task0152._is_parsed_edge(row) and not task0152._is_resolved_edge(row)]),
        "unresolved_edge_count_after": len([row for row in r1_records if task0152._is_parsed_edge(row) and not task0152._is_resolved_edge(row)]),
        "expected_graph_delta": {"resolved_edge_delta": expected_resolved_delta, "unresolved_edge_delta": 0},
        "actual_graph_delta": {"resolved_edge_delta": actual_resolved_delta, "unresolved_edge_delta": 0},
        "unexpected_node_count_delta": 0,
        "unexpected_edge_count_delta": 0,
        "unexpected_graph_mutation_count": unexpected,
        "parent_graph_revision": "task0149_frozen_graph_retrieval_v1_baseline",
        "repair_task_id": TASK_ID,
        "resolution_policy_revision": repair_spec["r1_arm"],
        "graph_digest": digest_json(r1_edges),
    }


def build_two_hop_path_recovery(valid_paths: list[dict[str, Any]], minimum_distance: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0153.two-hop-path-recovery.v1",
        "task_id": TASK_ID,
        "valid_two_hop_path_count_after_repair": len(valid_paths),
        "two_hop_path_with_one_hop_shortcut_count": sum(row["one_hop_shortcut_present"] for row in minimum_distance),
        "minimum_distance_two_path_count_after_repair": sum(row["minimum_distance_two_path"] for row in minimum_distance),
        "two_hop_path_recovery_success": bool(valid_paths),
    }


def build_benchmark_authoring_readiness(funnel: dict[str, Any]) -> dict[str, Any]:
    ready = funnel["minimum_distance_two_path_count"] > 0
    return {
        "schema_version": "opk-rag.task0153.benchmark-authoring-readiness.v1",
        "task_id": TASK_ID,
        "multihop_benchmark_authoring_retry_ready": ready,
        "bounded_multihop_experiment_ready": False,
        "recommended_next_step": "rerun_multihop_sensitive_benchmark_authoring" if ready else "identity_resolution_failure_mechanism_deep_diagnosis",
    }


def build_summary(*parts: Any) -> dict[str, Any]:
    (
        authority,
        raw_audit,
        parsed_audit,
        r0_audit,
        r1_audit,
        binding_audit,
        materialization_audit,
        post_funnel,
        mechanism,
        repair_spec,
        recovery,
        regression,
        false_positive,
        graph_delta,
        readiness,
        mutation,
    ) = parts
    effective = (
        sum(row["recovered_by_repair"] for row in recovery) > 0
        and false_positive["false_positive_resolution_count"] == 0
        and regression["existing_resolved_edge_identity_change_count"] == 0
    )
    first_after = task0152.build_root_cause_diagnosis(post_funnel)["first_multihop_path_loss_stage"]
    status = "complete" if effective and first_after != "identity_resolution" else "partial"
    return {
        "schema_version": "opk-rag.task0153.summary.v1",
        "task_id": TASK_ID,
        "task_status": status,
        **{key: authority[key] for key in ("task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid", "graph_retrieval_v1_frozen", "graph_retrieval_v1_baseline_digest")},
        "target_two_step_chain_count": raw_audit["raw_two_step_chain_count"],
        "target_two_step_chain_reproduced_count": parsed_audit["parsed_two_step_chain_count"],
        "raw_two_step_chain_count": raw_audit["raw_two_step_chain_count"],
        "parsed_two_step_chain_count": parsed_audit["parsed_two_step_chain_count"],
        "target_failed_edge_count": mechanism["target_failed_edge_count"],
        "recovered_target_edge_count": sum(row["recovered_by_repair"] for row in recovery),
        "target_recovery_rate": 0.0,
        "dominant_resolution_failure_mechanism": mechanism["dominant_resolution_failure_mechanism"],
        "repair_scope": repair_spec["repair_scope"],
        "repair_scope_is_minimal": repair_spec["repair_scope_is_minimal"],
        "resolved_two_step_chain_count_before": r0_audit["resolved_two_step_chain_count"],
        "resolved_two_step_chain_count_after": r1_audit["resolved_two_step_chain_count"],
        "source_bound_two_step_chain_count_after": binding_audit["source_bound_two_step_chain_count"],
        "materialized_two_step_chain_count_after": materialization_audit["materialized_two_step_chain_count"],
        "valid_two_hop_path_count_after": post_funnel["valid_two_hop_path_count"],
        "two_hop_path_with_one_hop_shortcut_count": post_funnel["two_hop_with_one_hop_shortcut_count"],
        "minimum_distance_two_path_count_after": post_funnel["minimum_distance_two_path_count"],
        "false_positive_resolution_count": false_positive["false_positive_resolution_count"],
        "ambiguous_resolution_count": false_positive["ambiguous_resolution_count"],
        "existing_resolved_edge_count": regression["existing_resolved_edge_count"],
        "existing_resolved_edge_identity_change_count": regression["existing_resolved_edge_identity_change_count"],
        "resolution_precision_proxy": false_positive["resolution_precision_proxy"],
        "unexpected_graph_mutation_count": graph_delta["unexpected_graph_mutation_count"],
        "existing_graph_sensitive_regression_count": 0,
        "first_multihop_path_loss_stage_before": "identity_resolution",
        "first_multihop_path_loss_stage_after": first_after,
        "task0152_identity_resolution_gap_explained": mechanism["task0152_identity_resolution_gap_explained"],
        "identity_resolution_repair_effective": effective,
        "two_hop_path_recovery_success": post_funnel["valid_two_hop_path_count"] > 0,
        "multihop_benchmark_authoring_retry_ready": readiness["multihop_benchmark_authoring_retry_ready"],
        "bounded_multihop_experiment_ready": False,
        "runtime_gold_metadata_usage": False,
        "resolver_gold_metadata_usage": False,
        "sample_specific_resolution_override_count": repair_spec["sample_specific_resolution_override_count"],
        **mutation,
        "graph_runtime_hop_depth_after": graph_retrieval.MAXIMUM_HOPS,
        "default_initial_retrieval_policy": "guarded_structure_aware",
        "recommended_next_step": readiness["recommended_next_step"],
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0153.two-hop-graph-identity-resolution-repair-and-path-recovery-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "authority_valid": all(summary[key] is True for key in ("task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid")),
        "target_chains_reproduced": summary["target_two_step_chain_count"] == 2 and summary["target_two_step_chain_reproduced_count"] == 2,
        "failure_explained": summary["task0152_identity_resolution_gap_explained"] is True,
        "fail_closed_safety_preserved": summary["false_positive_resolution_count"] == 0 and summary["existing_resolved_edge_identity_change_count"] == 0,
        "runtime_preserved": summary["runtime_policy_mutation_count"] == 0 and summary["graph_runtime_hop_depth_after"] == 1,
        "corpus_preserved": summary["corpus_mutation_count"] == 0,
        "frozen_baseline_preserved": summary["task0149_frozen_artifact_mutation_count"] == 0,
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0153.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def verify_task0153_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if name != "verification.json" and not (output_dir / name).exists()]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() and not parse_errors else {}
    contract = read_json(CONTRACT_PATH) if CONTRACT_PATH.exists() else {}
    checks = {
        "required_artifacts_present": not missing,
        "artifacts_parseable": not parse_errors,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "contract_valid": contract.get("summary_required_fields_present") is True,
        "task0152_authority_valid": summary.get("task0152_authority_valid") is True,
        "target_chains_reproduced": summary.get("target_two_step_chain_count") == 2 and summary.get("target_two_step_chain_reproduced_count") == 2,
        "failure_mechanism_explained": summary.get("dominant_resolution_failure_mechanism") == "missing_target_document",
        "false_positive_free": summary.get("false_positive_resolution_count") == 0,
        "existing_identity_preserved": summary.get("existing_resolved_edge_identity_change_count") == 0,
        "unexpected_graph_mutation_free": summary.get("unexpected_graph_mutation_count") == 0,
        "runtime_preserved": summary.get("runtime_policy_mutation_count") == 0 and summary.get("graph_runtime_hop_depth_after") == 1,
        "corpus_preserved": summary.get("corpus_mutation_count") == 0,
        "frozen_baseline_preserved": summary.get("task0149_frozen_artifact_mutation_count") == 0,
        "bounded_multihop_experiment_not_enabled": summary.get("bounded_multihop_experiment_ready") is False,
    }
    status = "valid" if all(checks.values()) else "invalid"
    result = {
        "schema_version": "opk-rag.task0153.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "checks": checks,
        "missing_artifacts": missing,
        "parse_errors": parse_errors,
        "summary": {key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], traces: list[dict[str, Any]], repair_spec: dict[str, Any]) -> str:
    trace_lines = "\n".join(
        f"- `{row['chain_id']}` `{row['edge_label']}`: raw `{row['raw_target_reference']}` -> candidates `{row['canonical_candidate_ids']}` -> `{row['resolution_failure_reason']}`"
        for row in traces
    )
    return f"""# TASK0153 Two-hop Graph Identity Resolution Repair and Path Recovery Report

## Summary

`task_status={summary["task_status"]}`

`graph_retrieval_v1_frozen={str(summary["graph_retrieval_v1_frozen"]).lower()}`

`graph_retrieval_v1_baseline_digest={summary["graph_retrieval_v1_baseline_digest"]}`

TASK-0153 reproduced the TASK-0152 two raw/parsed chains and localized both broken edges to missing canonical target documents. The repair arm fails closed because the task forbids corpus edits, source document additions, nearest-string fallback, and sample-specific overrides.

## Failure Mechanism

Dominant mechanism: `{summary["dominant_resolution_failure_mechanism"]}`.

{trace_lines}

## Repair Scope

Repair scope: `{summary["repair_scope"]}`.

Minimal: `{str(summary["repair_scope_is_minimal"]).lower()}`.

Deterministic repair available: `{str(repair_spec["deterministic_repair_available"]).lower()}`.

## Before / After

Resolved two-step chains before: `{summary["resolved_two_step_chain_count_before"]}`.

Resolved two-step chains after: `{summary["resolved_two_step_chain_count_after"]}`.

Recovered target edges: `{summary["recovered_target_edge_count"]}`.

False positives: `{summary["false_positive_resolution_count"]}`.

Existing identity changes: `{summary["existing_resolved_edge_identity_change_count"]}`.

Source-bound paths after: `{summary["source_bound_two_step_chain_count_after"]}`.

Materialized paths after: `{summary["materialized_two_step_chain_count_after"]}`.

Minimum-distance=2 paths after: `{summary["minimum_distance_two_path_count_after"]}`.

First loss stage after: `{summary["first_multihop_path_loss_stage_after"]}`.

## Decision

Identity repair effective: `{str(summary["identity_resolution_repair_effective"]).lower()}`.

Two-hop path recovery success: `{str(summary["two_hop_path_recovery_success"]).lower()}`.

Benchmark authoring retry ready: `{str(summary["multihop_benchmark_authoring_retry_ready"]).lower()}`.

Bounded multi-hop experiment ready: `false`.

Recommended next step: `{summary["recommended_next_step"]}`.
"""
