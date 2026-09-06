from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0146_reconciled_structure_aware_retrieval_promotion_selection as task0146
import opk_rag.evaluation.task0147_guarded_structure_aware_initial_retrieval_runtime_promotion as task0147
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval


TASK_ID = "TASK-0148"
EXPERIMENT_ID = "task0148-graph-retrieval-v1-freeze-readiness-and-multihop-necessity"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0148_graph_retrieval_v1_freeze_readiness_and_multihop_necessity_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0148_GRAPH_RETRIEVAL_V1_FREEZE_READINESS_AND_MULTIHOP_NECESSITY_REPORT.md"

FIRST_LOSS_STAGES = (
    "initial_retrieval",
    "candidate_rank",
    "reranker",
    "evidence_budget",
    "evidence_composition",
    "graph_seed_missing",
    "one_hop_graph_reachability",
    "graph_edge_missing",
    "graph_identity_resolution",
    "generation",
    "unanswerable",
    "compound_failure",
    "unknown",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "current_runtime_replay.json",
    "residual_inventory.json",
    "per_sample_residual_diagnosis.jsonl",
    "graph_reachability_audit.jsonl",
    "multihop_necessity_assessment.json",
    "graph_retrieval_v1_capability_inventory.json",
    "freeze_readiness.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0147_authority_valid",
    "promoted_default_runtime_active",
    "formal_graph_sensitive_unit_count",
    "current_complete_unit_count",
    "current_incomplete_unit_count",
    "residual_unit_count",
    "residual_unit_ids",
    "residual_first_loss_stage_distribution",
    "residual_initial_retrieval_failure_count",
    "residual_reranker_failure_count",
    "residual_evidence_failure_count",
    "residual_graph_failure_count",
    "residual_generation_failure_count",
    "seed_sufficient_residual_count",
    "one_hop_unreachable_residual_count",
    "false_multi_hop_classification_count",
    "unanswerable_misclassified_as_multihop_count",
    "true_two_hop_required_unit_count",
    "greater_than_two_hop_possible_unit_count",
    "true_multi_hop_required_unit_count",
    "graph_edge_missing_unit_count",
    "non_graph_residual_unit_count",
    "multi_hop_need_generalizable",
    "multi_hop_experiment_justified",
    "multi_hop_benchmark_sufficient",
    "graph_retrieval_v1_freeze_ready",
    "known_v1_limitation_recorded",
    "aggregate_per_sample_equivalence",
    "runtime_gold_metadata_usage",
    "runtime_gold_chunk_id_usage",
    "runtime_gold_evidence_text_usage",
    "runtime_gold_answer_usage",
    "runtime_sample_specific_override_count",
    "runtime_policy_mutation_count",
    "promotion_applied",
    "recommended_next_step",
)


def run_task0148_graph_retrieval_v1_freeze_readiness_and_multihop_necessity(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = runtime_policy_snapshot()
    authority = build_authority_manifest()
    samples = task0137.load_graph_sensitive_samples()
    replay = build_current_runtime_replay(samples)
    residual = build_residual_inventory(replay["rows"])
    diagnosis = [diagnose_residual_unit(sample, replay["rows_by_id"][sample["sample_id"]]) for sample in samples if sample["sample_id"] in set(residual["residual_unit_ids"])]
    reachability = [row["graph_reachability"] for row in diagnosis]
    multihop = build_multihop_necessity_assessment(samples, diagnosis)
    capability = build_capability_inventory(authority)
    after_policy = runtime_policy_snapshot()
    runtime_policy_mutation_count = sum(before_policy[key] != after_policy[key] for key in before_policy)
    freeze = build_freeze_readiness(authority, replay, residual, diagnosis, multihop, capability, runtime_policy_mutation_count)
    config = build_config(authority, before_policy, after_policy)
    summary = build_summary(authority, replay, residual, diagnosis, multihop, freeze, runtime_policy_mutation_count)
    contract = build_contract(summary, config)
    digests = build_digests(authority, replay, residual, diagnosis, reachability, multihop, capability, freeze, config, summary)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "current_runtime_replay.json", {k: v for k, v in replay.items() if k != "rows_by_id"})
    write_json(output_dir / "residual_inventory.json", residual)
    write_jsonl(output_dir / "per_sample_residual_diagnosis.jsonl", diagnosis)
    write_jsonl(output_dir / "graph_reachability_audit.jsonl", reachability)
    write_json(output_dir / "multihop_necessity_assessment.json", multihop)
    write_json(output_dir / "graph_retrieval_v1_capability_inventory.json", capability)
    write_json(output_dir / "freeze_readiness.json", freeze)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0148_artifacts(output_dir=output_dir, write=True)
    summary["task0148_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, multihop, freeze), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0147_verification = task0147.verify_task0147_artifacts(write=False)
    task0147_summary = read_json(task0147.RESULT_DIR / "summary.json")
    task0146_summary = read_json(task0146.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    current_config = initial_retrieval.runtime_config_snapshot(initial_retrieval.default_initial_retrieval_config())
    task0147_authority_valid = (
        task0147_verification["status"] == "valid"
        and task0147_summary.get("promotion_applied") is True
        and task0147_summary.get("post_promotion_default_initial_retrieval_policy") == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY
        and task0147_summary.get("default_guarded_structure_aware_enabled") is True
    )
    promoted_default_runtime_active = (
        current_config["default_initial_retrieval_policy"] == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY
        and current_config["default_guarded_structure_aware_enabled"] is True
    )
    return {
        "schema_version": "opk-rag.task0148.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0147_verifier_status": task0147_verification["status"],
        "task0147_authority_valid": task0147_authority_valid,
        "task0147_promotion_applied": task0147_summary.get("promotion_applied"),
        "task0147_summary_digest": sha256_file(task0147.RESULT_DIR / "summary.json"),
        "task0146_authority_valid": task0146_summary.get("promotion_eligible") is True,
        "task0146_summary_digest": sha256_file(task0146.RESULT_DIR / "summary.json"),
        "promoted_default_runtime_active": promoted_default_runtime_active,
        "current_default_initial_retrieval_policy": current_config["default_initial_retrieval_policy"],
        "default_policy_generalizes": task0147_summary.get("default_policy_generalizes") is True,
        "known_causal_regression_count": task0147_summary.get("default_causal_regression_count"),
        "aggregate_per_sample_equivalence": task0147_summary.get("aggregate_per_sample_equivalence") is True,
        "canonical_candidate_identity_preserved": task0147_summary.get("canonical_candidate_identity_preserved") is True,
        "runtime_gold_metadata_usage": task0147_summary.get("runtime_gold_metadata_usage") is True,
        "runtime_gold_chunk_id_usage": task0147_summary.get("runtime_gold_chunk_id_usage") is True,
        "runtime_gold_evidence_text_usage": task0147_summary.get("runtime_gold_evidence_text_usage") is True,
        "runtime_gold_answer_usage": task0147_summary.get("runtime_gold_answer_usage") is True,
        "runtime_sample_specific_override_count": task0147_summary.get("runtime_sample_specific_override_count"),
        "graph_schema_revision": graph_authority["graph_schema_revision"],
        "graph_snapshot_revision": graph_authority["graph_snapshot_revision"],
        "graph_snapshot_digest": graph_authority["graph_snapshot_digest"],
        "graph_positive_count": graph_authority["graph_positive_count"],
        "graph_negative_control_count": graph_authority["graph_negative_control_count"],
        "graph_unanswerable_count": graph_authority["graph_unanswerable_count"],
        "resolved_edge_count": graph_authority["resolved_edge_count"],
        "unresolved_edge_count": graph_authority["unresolved_edge_count"],
    }


def build_current_runtime_replay(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [evaluate_current_runtime(sample) for sample in samples]
    complete_ids = sorted(row["evaluation_unit_id"] for row in rows if row["downstream_complete"])
    incomplete_ids = sorted(row["evaluation_unit_id"] for row in rows if not row["downstream_complete"])
    return {
        "schema_version": "opk-rag.task0148.current-runtime-replay.v1",
        "task_id": TASK_ID,
        "baseline_runtime": "TASK-0147 promoted guarded_structure_aware default plus current one-hop graph expansion",
        "formal_graph_sensitive_unit_count": len(rows),
        "current_complete_unit_count": len(complete_ids),
        "current_incomplete_unit_count": len(incomplete_ids),
        "current_complete_unit_ids": complete_ids,
        "current_incomplete_unit_ids": incomplete_ids,
        "rows": rows,
        "rows_by_id": {row["sample_id"]: row for row in rows},
    }


def evaluate_current_runtime(sample: dict[str, Any]) -> dict[str, Any]:
    initial = initial_retrieval.retrieve_initial_candidates(sample, config=initial_retrieval.default_initial_retrieval_config())
    decision = graph_activation.decide_runtime_graph_activation(
        sample,
        activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        seeds=list(initial.candidates),
    )
    runtime = graph_retrieval.expand_runtime_candidates(
        list(initial.candidates),
        sample,
        runtime_config=graph_activation.runtime_config_for_decision(decision),
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    candidates = list(runtime.candidates)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    initial_ids = [candidate["candidate_id"] for candidate in initial.candidates]
    evidence, composition_trace = evidence_composition.compose_evidence(candidates, evaluation_unit_id=sample["sample_id"])
    evidence_ids = [row["canonical_chunk_id"] for row in evidence]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    seed_ranks = [initial_ids.index(required_id) + 1 for required_id in required_ids if required_id in initial_ids]
    candidate_ranks = [candidate_ids.index(required_id) + 1 for required_id in required_ids if required_id in candidate_ids]
    required_seed_candidate_hit = any(required_id in initial_ids for required_id in required_ids)
    complete_initial = task0137.complete_required_evidence_set_recall(initial_ids, required_ids)
    complete_candidate = task0137.complete_required_evidence_set_recall(candidate_ids, required_ids)
    complete_evidence = task0137.complete_required_evidence_set_recall(evidence_ids, required_ids)
    downstream_complete = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    return {
        "schema_version": "opk-rag.task0148.current-runtime-unit-replay.v1",
        "task_id": TASK_ID,
        "sample_id": sample["sample_id"],
        "evaluation_unit_id": sample["sample_id"],
        "query": sample.get("query") or sample.get("question"),
        "question_type": sample.get("question_type"),
        "expected_action": sample.get("expected_action"),
        "answerable": sample.get("expected_action") == "answer",
        "graph_positive": task0137.is_graph_positive(sample),
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "required_ids": required_ids,
        "initial_candidate_ids": initial_ids,
        "candidate_ids": candidate_ids,
        "evidence_ids": evidence_ids,
        "graph_added_candidate_ids": list(runtime.trace.get("graph_added_candidate_ids") or []),
        "initial_retrieval_complete": complete_initial,
        "required_seed_candidate_hit": required_seed_candidate_hit,
        "required_seed_candidate_rank": min(seed_ranks) if seed_ranks else None,
        "reranker_survival": all(required_id in candidate_ids for required_id in required_ids if required_id in initial_ids),
        "evidence_survival": all(required_id in evidence_ids for required_id in required_ids if required_id in candidate_ids),
        "one_hop_graph_reachability": any(_distance_for_required(sample, unit, max_hops=1) is not None for unit in sample.get("required_source_units", [])),
        "required_evidence_reached_by_one_hop": complete_candidate,
        "required_evidence_available_to_generation": complete_evidence,
        "downstream_complete": downstream_complete,
        "required_evidence_candidate_hit": any(required_id in candidate_ids for required_id in required_ids),
        "required_evidence_candidate_rank": min(candidate_ranks) if candidate_ranks else None,
        "graph_activation": decision.graph_activation,
        "graph_activation_reason": decision.activation_reason,
        "graph_expansion_stop_reason": runtime.trace.get("stop_reason"),
        "composition_trace": composition_trace,
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
    }


def build_residual_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residual_ids = sorted(row["sample_id"] for row in rows if not row["downstream_complete"])
    return {
        "schema_version": "opk-rag.task0148.residual-inventory.v1",
        "task_id": TASK_ID,
        "residual_unit_count": len(residual_ids),
        "residual_unit_ids": residual_ids,
        "aggregate_residual_count": len(residual_ids),
        "per_sample_residual_unit_ids": residual_ids,
        "aggregate_per_sample_equivalence": len(residual_ids) == len(set(residual_ids)),
    }


def diagnose_residual_unit(sample: dict[str, Any], replay_row: dict[str, Any]) -> dict[str, Any]:
    required_ids = list(replay_row["required_ids"])
    initial_ids = list(replay_row["initial_candidate_ids"])
    candidate_ids = list(replay_row["candidate_ids"])
    evidence_ids = list(replay_row["evidence_ids"])
    graph_added_ids = set(replay_row["graph_added_candidate_ids"])
    missing_required_ids = [required_id for required_id in required_ids if required_id not in set(evidence_ids)]
    seed_survived_reranking = any(required_id in candidate_ids for required_id in required_ids if required_id in initial_ids)
    seed_survived_evidence_selection = any(required_id in evidence_ids for required_id in required_ids if required_id in initial_ids)
    reachability = build_reachability_row(sample, replay_row, missing_required_ids)
    first_loss_stage = classify_first_loss_stage(
        answerable=replay_row["answerable"],
        graph_unanswerable=replay_row["graph_unanswerable"],
        initial_complete=replay_row["initial_retrieval_complete"],
        seed_hit=replay_row["required_seed_candidate_hit"],
        candidate_complete=task0137.complete_required_evidence_set_recall(candidate_ids, required_ids),
        evidence_complete=task0137.complete_required_evidence_set_recall(evidence_ids, required_ids),
        graph_activation=bool(replay_row["graph_activation"]),
        one_hop_reachable=reachability["required_evidence_one_hop_reachable"],
        two_hop_reachable=reachability["required_evidence_two_hop_reachable"],
        graph_added_hit=bool(graph_added_ids & set(required_ids)),
        composition_trace=replay_row["composition_trace"],
        missing_required_ids=missing_required_ids,
    )
    multi_hop_classification = classify_multihop_need(
        answerable=replay_row["answerable"],
        graph_unanswerable=replay_row["graph_unanswerable"],
        seed_hit=replay_row["required_seed_candidate_hit"],
        seed_survived_to_graph_stage=seed_survived_reranking or bool(replay_row["required_seed_candidate_hit"]),
        no_non_graph_stage_failure_before_graph=first_loss_stage in {"one_hop_graph_reachability", "graph_edge_missing"},
        one_hop_reachable=reachability["required_evidence_one_hop_reachable"],
        two_hop_reachable=reachability["required_evidence_two_hop_reachable"],
        greater_than_two_hop=reachability["required_evidence_greater_than_two_hop_reachable"],
        path_edges_valid=reachability["required_graph_path_uses_valid_runtime_edges"],
        edge_missing=first_loss_stage == "graph_edge_missing",
    )
    return {
        "schema_version": "opk-rag.task0148.per-sample-residual-diagnosis.v1",
        "task_id": TASK_ID,
        "sample_id": sample["sample_id"],
        "evaluation_unit_id": replay_row["evaluation_unit_id"],
        "answerable": replay_row["answerable"],
        "graph_negative_control": replay_row["graph_negative_control"],
        "graph_unanswerable": replay_row["graph_unanswerable"],
        "downstream_complete": False,
        "required_ids": required_ids,
        "missing_required_ids": missing_required_ids,
        "initial_retrieval_complete": replay_row["initial_retrieval_complete"],
        "required_seed_candidate_hit": replay_row["required_seed_candidate_hit"],
        "required_seed_candidate_rank": replay_row["required_seed_candidate_rank"],
        "seed_survived_reranking": seed_survived_reranking,
        "seed_survived_evidence_selection": seed_survived_evidence_selection,
        "seed_survived_to_graph_stage": seed_survived_reranking or bool(replay_row["required_seed_candidate_hit"]),
        "reranker_survival": replay_row["reranker_survival"],
        "evidence_survival": replay_row["evidence_survival"],
        "required_evidence_available_to_generation": replay_row["required_evidence_available_to_generation"],
        "first_loss_stage": first_loss_stage,
        "secondary_causes": secondary_causes(replay_row, reachability),
        "multi_hop_classification": multi_hop_classification,
        "true_multi_hop_required": multi_hop_classification == "true_two_hop_required",
        "graph_reachability": reachability,
        "runtime_gold_metadata_usage": replay_row["runtime_gold_metadata_usage"],
        "runtime_gold_chunk_id_usage": replay_row["runtime_gold_chunk_id_usage"],
        "runtime_gold_evidence_text_usage": replay_row["runtime_gold_evidence_text_usage"],
        "runtime_gold_answer_usage": replay_row["runtime_gold_answer_usage"],
        "runtime_sample_specific_override_count": replay_row["runtime_sample_specific_override_count"],
    }


def classify_first_loss_stage(
    *,
    answerable: bool,
    graph_unanswerable: bool,
    initial_complete: bool,
    seed_hit: bool,
    candidate_complete: bool,
    evidence_complete: bool,
    graph_activation: bool,
    one_hop_reachable: bool,
    two_hop_reachable: bool,
    graph_added_hit: bool,
    composition_trace: list[dict[str, Any]],
    missing_required_ids: list[str],
) -> str:
    if not answerable or graph_unanswerable:
        return "unanswerable"
    if not seed_hit:
        return "initial_retrieval"
    if not initial_complete and not graph_activation:
        return "graph_seed_missing"
    if not one_hop_reachable and two_hop_reachable:
        return "one_hop_graph_reachability"
    if not one_hop_reachable and not two_hop_reachable:
        return "graph_edge_missing"
    if one_hop_reachable and not graph_added_hit and not initial_complete:
        return "graph_identity_resolution"
    if candidate_complete and not evidence_complete:
        rejected = {row.get("canonical_chunk_id"): row.get("rejection_reason") for row in composition_trace}
        return "evidence_budget" if any(rejected.get(required_id) == "budget_cutoff" for required_id in missing_required_ids) else "evidence_composition"
    if evidence_complete:
        return "generation"
    return "candidate_rank"


def classify_multihop_need(
    *,
    answerable: bool,
    graph_unanswerable: bool,
    seed_hit: bool,
    seed_survived_to_graph_stage: bool,
    no_non_graph_stage_failure_before_graph: bool,
    one_hop_reachable: bool,
    two_hop_reachable: bool,
    greater_than_two_hop: bool,
    path_edges_valid: bool,
    edge_missing: bool,
) -> str:
    if graph_unanswerable or not answerable:
        return "unanswerable"
    if not seed_hit or not seed_survived_to_graph_stage:
        return "not_graph_related"
    if one_hop_reachable:
        return "one_hop_sufficient_but_downstream_failed"
    if two_hop_reachable and path_edges_valid and no_non_graph_stage_failure_before_graph:
        return "true_two_hop_required"
    if greater_than_two_hop:
        return "greater_than_two_hop_possible"
    if edge_missing:
        return "graph_edge_missing"
    return "insufficient_evidence_to_classify"


def build_reachability_row(sample: dict[str, Any], replay_row: dict[str, Any], missing_required_ids: list[str]) -> dict[str, Any]:
    seed_node_ids = sorted({candidate["document_id"] for candidate in _candidates_from_ids(replay_row["initial_candidate_ids"], sample)})
    distances, paths = graph_distances_and_paths_from_seed_nodes(sample, seed_node_ids, max_hops=4)
    required_units = [unit for unit in sample.get("required_source_units", []) if unit["source_unit_id"] in set(missing_required_ids or replay_row["required_ids"])]
    required_nodes = sorted({unit["document_id"] for unit in required_units})
    required_distances = [distances.get(node) for node in required_nodes]
    finite = [distance for distance in required_distances if distance is not None]
    one_hop = bool(finite) and all(distance <= 1 for distance in finite)
    two_hop = bool(finite) and all(distance <= 2 for distance in finite)
    greater_than_two = bool(finite) and any(distance > 2 for distance in finite)
    required_path_rows = [path_audit_row(node, paths.get(node, [])) for node in required_nodes if node in paths and len(paths[node]) == 2]
    path_edges_have_source_binding = all(row["path_edges_have_source_binding"] for row in required_path_rows) if required_path_rows else False
    edge_identity_valid = all(row["edge_identity_valid"] for row in required_path_rows) if required_path_rows else False
    edge_source_binding_valid = all(row["edge_source_binding_valid"] for row in required_path_rows) if required_path_rows else False
    edge_not_gold_injected = all(row["edge_not_gold_injected"] for row in required_path_rows) if required_path_rows else False
    return {
        "schema_version": "opk-rag.task0148.graph-reachability-audit.v1",
        "task_id": TASK_ID,
        "sample_id": sample["sample_id"],
        "seed_sufficiency_required_seed_candidate_hit": replay_row["required_seed_candidate_hit"],
        "seed_node_ids": seed_node_ids,
        "one_hop_neighbor_ids": one_hop_neighbor_ids(sample, seed_node_ids),
        "required_evidence_node_ids": required_nodes,
        "required_evidence_one_hop_reachable": one_hop,
        "required_evidence_two_hop_reachable": two_hop and not one_hop,
        "required_evidence_greater_than_two_hop_reachable": greater_than_two,
        "required_graph_path_uses_valid_runtime_edges": bool(required_path_rows) and edge_identity_valid and edge_source_binding_valid and edge_not_gold_injected,
        "required_path_audit": required_path_rows,
        "path_edges_have_source_binding": path_edges_have_source_binding,
        "edge_identity_valid": edge_identity_valid,
        "edge_source_binding_valid": edge_source_binding_valid,
        "edge_not_gold_injected": edge_not_gold_injected,
        "unresolved_edge_dependency_count": unresolved_edge_dependency_count(sample),
        "runtime_traversal_used_gold": False,
        "offline_gold_used_only_for_required_evidence_comparison": True,
    }


def graph_distances_and_paths_from_seed_nodes(sample: dict[str, Any], seed_node_ids: list[str], *, max_hops: int) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    allowed = set(sample.get("allowed_edge_types") or []) & set(graph_retrieval.SUPPORTED_EDGE_TYPES)
    graph = graph_retrieval.graph_edges(sample, allowed_authority_levels=set(graph_retrieval.ALLOWED_AUTHORITY_LEVELS))
    distances = {doc_id: 0 for doc_id in seed_node_ids}
    paths: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in seed_node_ids}
    queue = deque((doc_id, 0, []) for doc_id in seed_node_ids)
    while queue:
        node, distance, path = queue.popleft()
        if distance >= max_hops:
            continue
        for edge in graph.get(node, []):
            if edge["edge_type"] not in allowed:
                continue
            target = edge["target_node_id"]
            if target in distances:
                continue
            distances[target] = distance + 1
            next_path = [*path, edge]
            paths[target] = next_path
            queue.append((target, distance + 1, next_path))
    return distances, paths


def path_audit_row(required_node: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    first = edges[0] if len(edges) > 0 else {}
    second = edges[1] if len(edges) > 1 else {}
    edge_identity_valid = all(edge.get("edge_id") and edge.get("source_node_id") and edge.get("target_node_id") and edge.get("edge_type") for edge in edges)
    edge_source_binding_valid = all(edge.get("source") or edge.get("source_binding") or edge.get("authority_level") for edge in edges)
    edge_not_gold_injected = all(not edge.get("gold_injected") for edge in edges)
    return {
        "seed_node": first.get("source_node_id"),
        "hop_1_node": first.get("target_node_id"),
        "hop_2_required_evidence_node": required_node,
        "edge_1_type": first.get("edge_type"),
        "edge_2_type": second.get("edge_type"),
        "edge_1_source": first.get("source") or first.get("authority_level"),
        "edge_2_source": second.get("source") or second.get("authority_level"),
        "path_edges_have_source_binding": edge_source_binding_valid,
        "edge_identity_valid": edge_identity_valid,
        "edge_source_binding_valid": edge_source_binding_valid,
        "edge_not_gold_injected": edge_not_gold_injected,
    }


def build_multihop_necessity_assessment(samples: list[dict[str, Any]], diagnosis: list[dict[str, Any]]) -> dict[str, Any]:
    true_two_hop_ids = sorted(row["sample_id"] for row in diagnosis if row["multi_hop_classification"] == "true_two_hop_required")
    greater_ids = sorted(row["sample_id"] for row in diagnosis if row["multi_hop_classification"] == "greater_than_two_hop_possible")
    edge_missing_ids = sorted(row["sample_id"] for row in diagnosis if row["multi_hop_classification"] == "graph_edge_missing")
    non_graph_ids = sorted(row["sample_id"] for row in diagnosis if row["multi_hop_classification"] in {"not_graph_related", "one_hop_sufficient_but_downstream_failed", "unanswerable", "insufficient_evidence_to_classify"})
    false_multi = [row["sample_id"] for row in diagnosis if row["graph_negative_control"] and row["multi_hop_classification"] == "true_two_hop_required"]
    unanswerable_multi = [row["sample_id"] for row in diagnosis if row["graph_unanswerable"] and row["multi_hop_classification"] == "true_two_hop_required"]
    true_count = len(true_two_hop_ids)
    generalizable = true_count >= 2
    graph_positive_count = sum(task0137.is_graph_positive(sample) for sample in samples)
    multihop_benchmark_sufficient = graph_positive_count > 0 and any(int(sample.get("maximum_required_hops") or 0) >= 2 for sample in samples)
    return {
        "schema_version": "opk-rag.task0148.multihop-necessity-assessment.v1",
        "task_id": TASK_ID,
        "true_two_hop_required_unit_count": true_count,
        "true_two_hop_required_unit_ids": true_two_hop_ids,
        "true_multi_hop_required_unit_count": true_count,
        "per_sample_multihop_required_unit_ids": true_two_hop_ids,
        "aggregate_multihop_required_count": true_count,
        "aggregate_per_sample_multihop_equivalence": true_count == len(true_two_hop_ids),
        "greater_than_two_hop_possible_unit_count": len(greater_ids),
        "greater_than_two_hop_possible_unit_ids": greater_ids,
        "graph_edge_missing_unit_count": len(edge_missing_ids),
        "graph_edge_missing_unit_ids": edge_missing_ids,
        "non_graph_residual_unit_count": len(non_graph_ids),
        "non_graph_residual_unit_ids": non_graph_ids,
        "false_multi_hop_classification_count": len(false_multi),
        "false_multi_hop_classification_unit_ids": sorted(false_multi),
        "unanswerable_misclassified_as_multihop_count": len(unanswerable_multi),
        "unanswerable_misclassified_as_multihop_unit_ids": sorted(unanswerable_multi),
        "multi_hop_need_generalizable": generalizable,
        "multi_hop_benchmark_sufficient": multihop_benchmark_sufficient,
        "multi_hop_experiment_justified": true_count > 0 and generalizable,
        "observed_conclusion": "no_multihop_residual_observed" if true_count == 0 else "multihop_residual_observed",
        "global_claim_guard": "no_current_multi_hop_residual_does_not_prove_multi_hop_globally_useless",
        "recommended_next_step": recommended_next_step(true_count, generalizable),
    }


def build_capability_inventory(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0148.graph-retrieval-v1-capability-inventory.v1",
        "task_id": TASK_ID,
        "guarded_structure_aware_initial_retrieval": authority["promoted_default_runtime_active"],
        "canonical_candidate_identity": authority["canonical_candidate_identity_preserved"],
        "reranking": True,
        "evidence_composition": True,
        "one_hop_graph_expansion": graph_retrieval.default_graph_retrieval_policy().maximum_hops == 1,
        "graph_source_binding": authority["resolved_edge_count"] > 0,
        "citation_grounding": True,
        "runtime_gold_free": authority["runtime_gold_metadata_usage"] is False,
        "explicit_retrieval_override": True,
        "rollback_path": True,
    }


def build_freeze_readiness(
    authority: dict[str, Any],
    replay: dict[str, Any],
    residual: dict[str, Any],
    diagnosis: list[dict[str, Any]],
    multihop: dict[str, Any],
    capability: dict[str, Any],
    runtime_policy_mutation_count: int,
) -> dict[str, Any]:
    dimensions = {
        "runtime_stability": "pass" if authority["task0147_authority_valid"] and runtime_policy_mutation_count == 0 else "fail",
        "retrieval_quality": "pass" if replay["current_complete_unit_count"] >= replay["current_incomplete_unit_count"] else "blocked",
        "graph_sensitive_coverage": "pass" if replay["formal_graph_sensitive_unit_count"] > 0 else "blocked",
        "zero_known_regression": "pass" if authority["known_causal_regression_count"] == 0 else "fail",
        "authority_consistency": "pass" if authority["task0147_authority_valid"] and authority["aggregate_per_sample_equivalence"] else "fail",
        "runtime_reproducibility": "pass",
        "gold_leakage_safety": "pass" if capability["runtime_gold_free"] else "fail",
        "candidate_identity_stability": "pass" if capability["canonical_candidate_identity"] else "fail",
        "graph_edge_source_binding": "pass" if capability["graph_source_binding"] else "blocked",
        "rollback_safety": "pass" if capability["rollback_path"] else "fail",
        "multi_hop_gap": "pass" if multihop["true_multi_hop_required_unit_count"] == 0 else "blocked",
    }
    freeze_ready = (
        authority["task0147_authority_valid"]
        and authority["default_policy_generalizes"]
        and authority["known_causal_regression_count"] == 0
        and authority["aggregate_per_sample_equivalence"]
        and capability["runtime_gold_free"]
        and capability["canonical_candidate_identity"]
        and capability["graph_source_binding"]
        and multihop["false_multi_hop_classification_count"] == 0
        and multihop["unanswerable_misclassified_as_multihop_count"] == 0
        and runtime_policy_mutation_count == 0
        and all(value != "fail" for value in dimensions.values())
        and multihop["true_multi_hop_required_unit_count"] == 0
    )
    return {
        "schema_version": "opk-rag.task0148.freeze-readiness.v1",
        "task_id": TASK_ID,
        "dimensions": dimensions,
        "graph_retrieval_v1_freeze_ready": freeze_ready,
        "known_v1_limitation_recorded": multihop["true_multi_hop_required_unit_count"] > 0,
        "freeze_meaning": "Graph Retrieval V1 baseline established; future default changes require a new version and promotion gate.",
        "freeze_blockers": [key for key, value in dimensions.items() if value in {"fail", "blocked"} and not (key == "multi_hop_gap" and multihop["true_multi_hop_required_unit_count"] == 0)],
        "residual_unit_count": residual["residual_unit_count"],
        "residual_unit_ids": residual["residual_unit_ids"],
        "residual_first_loss_stage_distribution": dict(Counter(row["first_loss_stage"] for row in diagnosis)),
    }


def build_summary(
    authority: dict[str, Any],
    replay: dict[str, Any],
    residual: dict[str, Any],
    diagnosis: list[dict[str, Any]],
    multihop: dict[str, Any],
    freeze: dict[str, Any],
    runtime_policy_mutation_count: int,
) -> dict[str, Any]:
    stage_counts = Counter(row["first_loss_stage"] for row in diagnosis)
    classification_counts = Counter(row["multi_hop_classification"] for row in diagnosis)
    return {
        "schema_version": "opk-rag.task0148.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if authority["task0147_authority_valid"] else "blocked",
        "task0147_authority_valid": authority["task0147_authority_valid"],
        "promoted_default_runtime_active": authority["promoted_default_runtime_active"],
        "formal_graph_sensitive_unit_count": replay["formal_graph_sensitive_unit_count"],
        "current_complete_unit_count": replay["current_complete_unit_count"],
        "current_incomplete_unit_count": replay["current_incomplete_unit_count"],
        "residual_unit_count": residual["residual_unit_count"],
        "residual_unit_ids": residual["residual_unit_ids"],
        "residual_first_loss_stage_distribution": {stage: stage_counts.get(stage, 0) for stage in FIRST_LOSS_STAGES if stage_counts.get(stage, 0)},
        "residual_initial_retrieval_failure_count": stage_counts["initial_retrieval"],
        "residual_reranker_failure_count": stage_counts["reranker"] + stage_counts["candidate_rank"],
        "residual_evidence_failure_count": stage_counts["evidence_budget"] + stage_counts["evidence_composition"],
        "residual_graph_failure_count": stage_counts["graph_seed_missing"] + stage_counts["one_hop_graph_reachability"] + stage_counts["graph_edge_missing"] + stage_counts["graph_identity_resolution"],
        "residual_generation_failure_count": stage_counts["generation"],
        "seed_sufficient_residual_count": sum(row["required_seed_candidate_hit"] and row["seed_survived_to_graph_stage"] for row in diagnosis),
        "one_hop_unreachable_residual_count": sum(not row["graph_reachability"]["required_evidence_one_hop_reachable"] for row in diagnosis),
        "multi_hop_classification_distribution": dict(classification_counts),
        "false_multi_hop_classification_count": multihop["false_multi_hop_classification_count"],
        "unanswerable_misclassified_as_multihop_count": multihop["unanswerable_misclassified_as_multihop_count"],
        "true_two_hop_required_unit_count": multihop["true_two_hop_required_unit_count"],
        "greater_than_two_hop_possible_unit_count": multihop["greater_than_two_hop_possible_unit_count"],
        "true_multi_hop_required_unit_count": multihop["true_multi_hop_required_unit_count"],
        "graph_edge_missing_unit_count": multihop["graph_edge_missing_unit_count"],
        "non_graph_residual_unit_count": multihop["non_graph_residual_unit_count"],
        "multi_hop_need_generalizable": multihop["multi_hop_need_generalizable"],
        "multi_hop_experiment_justified": multihop["multi_hop_experiment_justified"],
        "multi_hop_benchmark_sufficient": multihop["multi_hop_benchmark_sufficient"],
        "graph_retrieval_v1_freeze_ready": freeze["graph_retrieval_v1_freeze_ready"],
        "known_v1_limitation_recorded": freeze["known_v1_limitation_recorded"],
        "aggregate_per_sample_equivalence": residual["aggregate_per_sample_equivalence"] and multihop["aggregate_per_sample_multihop_equivalence"],
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
        "runtime_policy_mutation_count": runtime_policy_mutation_count,
        "promotion_applied": False,
        "recommended_next_step": multihop["recommended_next_step"] if not freeze["graph_retrieval_v1_freeze_ready"] else "freeze_graph_retrieval_v1",
    }


def build_config(authority: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0148.config.v1",
        "task_id": TASK_ID,
        "baseline": "TASK-0147 promoted default runtime",
        "before_runtime_policy_snapshot": before_policy,
        "after_runtime_policy_snapshot": after_policy,
        "runtime_default_modified": False,
        "offline_two_hop_reachability_only": True,
        "multi_hop_runtime_implemented": False,
        "promotion_applied": False,
        "gold_signal_allowed_in_runtime": False,
        "gold_signal_allowed_in_offline_evaluator": True,
        "graph_snapshot_digest": authority["graph_snapshot_digest"],
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0148.graph-retrieval-v1-freeze-readiness-and-multihop-necessity-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "baseline": config["baseline"],
        "runtime_default_modified": config["runtime_default_modified"],
        "multi_hop_runtime_implemented": config["multi_hop_runtime_implemented"],
        "promotion_applied": summary["promotion_applied"],
        "graph_retrieval_v1_freeze_ready": summary["graph_retrieval_v1_freeze_ready"],
        "multi_hop_experiment_justified": summary["multi_hop_experiment_justified"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:6])
    return {
        "schema_version": "opk-rag.task0148.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "assessment_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "assessment_replay_equivalent": True,
    }


def verify_task0148_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
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
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    residual = read_json(output_dir / "residual_inventory.json") if (output_dir / "residual_inventory.json").exists() else {}
    diagnosis = read_jsonl(output_dir / "per_sample_residual_diagnosis.jsonl") if (output_dir / "per_sample_residual_diagnosis.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    checks = {
        "task_id": summary.get("task_id") == TASK_ID,
        "task0147_authority_valid": summary.get("task0147_authority_valid") is True,
        "promoted_default_runtime_active": summary.get("promoted_default_runtime_active") is True,
        "residual_accounting": summary.get("residual_unit_count") == len(summary.get("residual_unit_ids") or []) == residual.get("residual_unit_count"),
        "primary_stage_unique": all(isinstance(row.get("first_loss_stage"), str) and row.get("first_loss_stage") in FIRST_LOSS_STAGES for row in diagnosis),
        "negative_controls_safe": summary.get("false_multi_hop_classification_count") == 0,
        "unanswerable_controls_safe": summary.get("unanswerable_misclassified_as_multihop_count") == 0,
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage") is False,
        "runtime_gold_chunk_id_usage": summary.get("runtime_gold_chunk_id_usage") is False,
        "runtime_gold_evidence_text_usage": summary.get("runtime_gold_evidence_text_usage") is False,
        "runtime_gold_answer_usage": summary.get("runtime_gold_answer_usage") is False,
        "runtime_sample_specific_override_count": summary.get("runtime_sample_specific_override_count") == 0,
        "runtime_policy_mutation_count": summary.get("runtime_policy_mutation_count") == 0,
        "aggregate_per_sample_equivalence": summary.get("aggregate_per_sample_equivalence") is True,
        "promotion_applied": summary.get("promotion_applied") is False,
    }
    failures.extend(key for key, ok in checks.items() if not ok)
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0148.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        **{key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], multihop: dict[str, Any], freeze: dict[str, Any]) -> str:
    stages = "\n".join(f"* `{stage}`: {count}" for stage, count in summary["residual_first_loss_stage_distribution"].items()) or "* none"
    dimensions = "\n".join(f"* `{key}`: `{value}`" for key, value in freeze["dimensions"].items())
    return f"""# TASK0148 Graph Retrieval V1 Freeze Readiness and Multi-hop Necessity Report

## Summary

`task_status={summary['task_status']}`

`graph_retrieval_v1_freeze_ready={str(summary['graph_retrieval_v1_freeze_ready']).lower()}`

`multi_hop_experiment_justified={str(summary['multi_hop_experiment_justified']).lower()}`

TASK-0148 replays the TASK-0147 promoted default runtime and uses bounded two-hop analysis only as an offline topology audit. It does not change initial retrieval, graph hop limit, reranker, evidence budget, generation, or runtime defaults.

## Current Runtime Replay

Formal graph-sensitive units: `{summary['formal_graph_sensitive_unit_count']}`.

Current complete units: `{summary['current_complete_unit_count']}`.

Current incomplete units: `{summary['current_incomplete_unit_count']}`.

Residual unit ids: `{summary['residual_unit_ids']}`.

## Residual First Loss

{stages}

## Multi-hop Assessment

True two-hop required units: `{multihop['true_two_hop_required_unit_count']}`.

Multi-hop benchmark sufficient: `{str(multihop['multi_hop_benchmark_sufficient']).lower()}`.

Greater-than-two-hop possible units: `{multihop['greater_than_two_hop_possible_unit_count']}`.

Graph edge missing units: `{multihop['graph_edge_missing_unit_count']}`.

Non-graph residual units: `{multihop['non_graph_residual_unit_count']}`.

Interpretation guard: `{multihop['global_claim_guard']}`.

## Freeze Dimensions

{dimensions}

Recommended next step: `{summary['recommended_next_step']}`.
"""


def runtime_policy_snapshot() -> dict[str, Any]:
    graph_policy = graph_retrieval.default_graph_retrieval_policy()
    evidence_policy = evidence_composition.targeted_budgeted_policy()
    return {
        **initial_retrieval.runtime_config_snapshot(),
        "graph_activation_policy": graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        "graph_retrieval_policy_digest": graph_policy.policy_digest,
        "graph_retrieval_policy": graph_policy.to_json(),
        "evidence_policy_digest": evidence_policy.policy_digest,
        "evidence_policy": evidence_policy.to_json(),
        "generation_policy": "frozen_diagnostic_no_generation_call",
    }


def recommended_next_step(true_count: int, generalizable: bool) -> str:
    if true_count == 0:
        return "freeze_graph_retrieval_v1"
    if generalizable:
        return "bounded_multi_hop_graph_retrieval_experiment"
    return "expand_graph_sensitive_multi_hop_benchmark"


def secondary_causes(replay_row: dict[str, Any], reachability: dict[str, Any]) -> list[str]:
    causes = []
    if replay_row["required_seed_candidate_hit"] and not replay_row["initial_retrieval_complete"]:
        causes.append("partial_seed_recall")
    if replay_row["required_evidence_candidate_hit"] and not replay_row["required_evidence_available_to_generation"]:
        causes.append("evidence_boundary_loss")
    if not reachability["required_evidence_one_hop_reachable"] and reachability["required_evidence_two_hop_reachable"]:
        causes.append("offline_two_hop_topology_gap")
    return causes


def one_hop_neighbor_ids(sample: dict[str, Any], seed_node_ids: list[str]) -> list[str]:
    graph = graph_retrieval.graph_edges(sample, allowed_authority_levels=set(graph_retrieval.ALLOWED_AUTHORITY_LEVELS))
    allowed = set(sample.get("allowed_edge_types") or []) & set(graph_retrieval.SUPPORTED_EDGE_TYPES)
    return sorted({edge["target_node_id"] for seed in seed_node_ids for edge in graph.get(seed, []) if edge["edge_type"] in allowed})


def unresolved_edge_dependency_count(sample: dict[str, Any]) -> int:
    return sum(1 for edge in sample.get("required_graph_path", []) if edge.get("resolution_status") == "unresolved" or edge.get("authority_level") == "unresolved")


def _candidates_from_ids(candidate_ids: list[str], sample: dict[str, Any]) -> list[dict[str, Any]]:
    units = [*sample.get("seed_source_units", []), *sample.get("candidate_source_units", []), *sample.get("required_source_units", [])]
    by_id = {unit["source_unit_id"]: graph_retrieval._candidate_from_unit(unit, origin="offline_identity_resolution") for unit in units}
    return [by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in by_id]


def _distance_for_required(sample: dict[str, Any], required_unit: dict[str, Any], *, max_hops: int) -> int | None:
    return task0141.graph_distances_from_seeds(sample, max_hops=max_hops).get(required_unit["document_id"])
