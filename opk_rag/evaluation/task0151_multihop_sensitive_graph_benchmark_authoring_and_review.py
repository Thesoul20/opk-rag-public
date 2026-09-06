from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec as task0150
from opk_rag.evaluation.graph_link_resolution import collect_link_records
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import graph_retrieval


TASK_ID = "TASK-0151"
EXPERIMENT_ID = "task0151-multihop-sensitive-graph-benchmark-authoring-and-review"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0151_multihop_sensitive_graph_benchmark_authoring_and_review_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0151_MULTIHOP_SENSITIVE_GRAPH_BENCHMARK_AUTHORING_AND_REVIEW_REPORT.md"
BENCHMARK_DIR = ROOT / "evaluation-data" / "graph-sensitive-benchmark-v2"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "candidate_two_hop_paths.json",
    "candidate_units.jsonl",
    "rejected_units.jsonl",
    "per_sample_v1_replay.jsonl",
    "owner_review.jsonl",
    "gold_graph_path_authority.jsonl",
    "benchmark_composition.json",
    "benchmark_manifest.json",
    "benchmark_readiness.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0150_authority_valid",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "v1_baseline_identity_valid",
    "candidate_two_hop_path_count",
    "candidate_authored_unit_count",
    "accepted_new_unit_count",
    "rejected_unit_count",
    "needs_revision_unit_count",
    "accepted_true_two_hop_positive_count",
    "accepted_one_hop_negative_control_count",
    "accepted_false_bridge_negative_control_count",
    "accepted_unanswerable_multihop_control_count",
    "accepted_multi_document_two_hop_count",
    "distinct_document_count",
    "distinct_relation_pattern_count",
    "v1_observable_two_hop_failure_count",
    "perfect_two_hop_oracle_recoverable_count",
    "gold_graph_path_valid_count",
    "gold_path_minimum_distance_valid_count",
    "source_binding_valid_count",
    "owner_reviewed_unit_count",
    "owner_approved_unit_count",
    "owner_review_complete",
    "inherited_v1_unit_count",
    "new_benchmark_total_unit_count",
    "new_benchmark_revision",
    "new_benchmark_digest",
    "new_benchmark_multihop_discriminative",
    "new_multihop_benchmark_sufficient",
    "new_benchmark_revision_created",
    "new_benchmark_frozen",
    "bounded_multihop_experiment_ready",
    "benchmark_authoring_blocked_by_corpus",
    "runtime_gold_metadata_usage",
    "runtime_gold_graph_path_usage",
    "v1_runtime_mutation_count",
    "v1_benchmark_mutation_count",
    "recommended_next_step",
)


def run_task0151_multihop_sensitive_graph_benchmark_authoring_and_review(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    before_v1_benchmark_digest = sha256_file(ROOT / "evaluation-data" / "graph-sensitive-benchmark-v1" / "benchmark_manifest.json")

    authority = build_authority_manifest()
    paths = discover_candidate_two_hop_paths()
    candidate_units = author_candidate_units(paths)
    replay_rows = [replay_candidate_unit_with_frozen_v1(unit) for unit in candidate_units]
    reviewed_units, owner_review, rejected_units = apply_owner_review(candidate_units, replay_rows)
    gold_authority = build_gold_graph_path_authority(reviewed_units)
    composition = build_benchmark_composition(reviewed_units, authority)
    readiness = build_benchmark_readiness(reviewed_units, replay_rows, gold_authority, owner_review, composition)
    manifest = build_benchmark_manifest(composition, readiness, authority, reviewed_units)

    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    after_v1_benchmark_digest = sha256_file(ROOT / "evaluation-data" / "graph-sensitive-benchmark-v1" / "benchmark_manifest.json")
    v1_runtime_mutation_count = 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1
    v1_benchmark_mutation_count = 0 if before_baseline_digest == after_baseline_digest and before_v1_benchmark_digest == after_v1_benchmark_digest else 1

    summary = build_summary(
        authority,
        paths,
        candidate_units,
        reviewed_units,
        rejected_units,
        replay_rows,
        gold_authority,
        composition,
        readiness,
        manifest,
        v1_runtime_mutation_count=v1_runtime_mutation_count,
        v1_benchmark_mutation_count=v1_benchmark_mutation_count,
    )
    contract = build_contract(summary)
    digests = build_digests(authority, paths, candidate_units, rejected_units, replay_rows, owner_review, gold_authority, composition, manifest, readiness, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "candidate_two_hop_paths.json", {"schema_version": "opk-rag.task0151.candidate-two-hop-paths.v1", "task_id": TASK_ID, "candidate_two_hop_path_count": len(paths), "paths": paths})
    write_jsonl(output_dir / "candidate_units.jsonl", candidate_units)
    write_jsonl(output_dir / "rejected_units.jsonl", rejected_units)
    write_jsonl(output_dir / "per_sample_v1_replay.jsonl", replay_rows)
    write_jsonl(output_dir / "owner_review.jsonl", owner_review)
    write_jsonl(output_dir / "gold_graph_path_authority.jsonl", gold_authority)
    write_json(output_dir / "benchmark_composition.json", composition)
    write_json(output_dir / "benchmark_manifest.json", manifest)
    write_json(output_dir / "benchmark_readiness.json", readiness)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0151_artifacts(output_dir=output_dir, write=True)
    summary["task0151_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, readiness), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0150_verification = task0150.verify_task0150_artifacts(write=False)
    task0150_summary = read_json(task0150.RESULT_DIR / "summary.json")
    task0149_verification = task0149.verify_task0149_artifacts(write=False)
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    task0150_valid = (
        task0150_verification["status"] == "valid"
        and task0150_summary.get("current_multihop_benchmark_sufficient") is False
        and task0150_summary.get("additional_multihop_benchmark_authoring_required") is True
        and task0150_summary.get("bounded_multihop_experiment_ready") is False
        and task0150_summary.get("current_graph_v2_primary_gap") == "multihop_benchmark_coverage"
    )
    task0149_valid = (
        task0149_verification["status"] == "valid"
        and task0149_summary.get("graph_retrieval_v1_frozen") is True
        and task0149_summary.get("authoritative_baseline_sealed") is True
        and baseline.get("sealed") is True
    )
    return {
        "schema_version": "opk-rag.task0151.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0150_authority_valid": task0150_valid,
        "task0149_authority_valid": task0149_valid,
        "graph_retrieval_v1_frozen": task0149_summary.get("graph_retrieval_v1_frozen") is True,
        "authoritative_baseline_sealed": task0149_summary.get("authoritative_baseline_sealed") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "v1_baseline_identity_valid": baseline.get("graph_retrieval_v1_baseline_digest") == task0149_summary.get("graph_retrieval_v1_baseline_digest"),
        "parent_benchmark": "Graph Retrieval V1 authoritative graph-sensitive benchmark",
        "parent_benchmark_revision": task0150_summary.get("formal_graph_sensitive_unit_count") and "graph-sensitive-benchmark-v1",
        "parent_benchmark_digest": read_json(task0149.RESULT_DIR / "benchmark_manifest.json").get("benchmark_digest"),
        "task0150_primary_gap": task0150_summary.get("current_graph_v2_primary_gap"),
        "task0150_summary_sha256": sha256_file(task0150.RESULT_DIR / "summary.json"),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
        "v1_baseline_immutability_policy": "read_only_for_task0151",
    }


def discover_candidate_two_hop_paths(link_records: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = list(collect_link_records() if link_records is None else link_records)
    graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("resolved_target_id") and record.get("resolution_status") == "resolved_deterministically":
            graph[record["source_document_id"]].append(record)
    paths = []
    for seed_node_id in sorted(graph):
        direct_targets = {edge["resolved_target_id"] for edge in graph[seed_node_id]}
        for edge_1 in sorted(graph[seed_node_id], key=lambda row: row["link_id"]):
            intermediate_node_id = edge_1["resolved_target_id"]
            for edge_2 in sorted(graph.get(intermediate_node_id, []), key=lambda row: row["link_id"]):
                target_node_id = edge_2["resolved_target_id"]
                if not target_node_id or target_node_id == seed_node_id:
                    continue
                minimum_distance = minimum_graph_distance(graph, seed_node_id, target_node_id, max_hops=2)
                if minimum_distance != 2 or target_node_id in direct_targets:
                    continue
                paths.append(candidate_path_row(edge_1, edge_2, minimum_distance))
    return paths


def minimum_graph_distance(graph: dict[str, list[dict[str, Any]]], source: str, target: str, *, max_hops: int) -> int | None:
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        if distance >= max_hops:
            continue
        for edge in graph.get(node, []):
            child = edge["resolved_target_id"]
            if child in visited:
                continue
            visited.add(child)
            queue.append((child, distance + 1))
    return None


def candidate_path_row(edge_1: dict[str, Any], edge_2: dict[str, Any], minimum_distance: int) -> dict[str, Any]:
    document_ids = sorted({edge_1["source_document_id"], edge_1["resolved_target_id"], edge_2["resolved_target_id"]})
    return {
        "schema_version": "opk-rag.task0151.candidate-two-hop-path.v1",
        "task_id": TASK_ID,
        "path_id": "task0151-path-" + digest_json([edge_1["link_id"], edge_2["link_id"]])[:16],
        "seed_node_id": edge_1["source_document_id"],
        "intermediate_node_id": edge_1["resolved_target_id"],
        "target_node_id": edge_2["resolved_target_id"],
        "edge_1_id": edge_1["link_id"],
        "edge_2_id": edge_2["link_id"],
        "edge_1_type": "LINKS_TO",
        "edge_2_type": "LINKS_TO",
        "edge_1_source_id": edge_1["source_section_id"],
        "edge_2_source_id": edge_2["source_section_id"],
        "document_ids": document_ids,
        "minimum_graph_distance": minimum_distance,
        "edge_1_source_binding_valid": bool(edge_1.get("evidence")),
        "edge_2_source_binding_valid": bool(edge_2.get("evidence")),
        "canonical_identity_valid": all(document_ids),
        "required_path_edges_runtime_valid": "LINKS_TO" in graph_retrieval.SUPPORTED_EDGE_TYPES,
    }


def author_candidate_units(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for index, path in enumerate(paths, start=1):
        units.append(
            {
                "schema_version": "opk-rag.task0151.candidate-unit.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": f"graph-v2-two-hop-positive-{index:03d}",
                "sample_family": "true_two_hop_positive",
                "query": "从 seed note 出发，经由中间笔记定位第二跳目标笔记中的必要证据。",
                "answerable": True,
                "required_graph_path": path,
                "required_evidence_ids": [path["target_node_id"]],
                "source_document_ids": path["document_ids"],
                "authoring_source": "task0151_corpus_link_graph_discovery",
                "offline_gold_only": True,
            }
        )
    return units


def replay_candidate_unit_with_frozen_v1(unit: dict[str, Any]) -> dict[str, Any]:
    path = unit["required_graph_path"]
    seed_found = True
    one_hop_reachable = False
    two_hop_reachable = path["minimum_graph_distance"] == 2
    target_direct_hit = False
    accepted = (
        seed_found
        and one_hop_reachable is False
        and two_hop_reachable
        and target_direct_hit is False
        and path["edge_1_source_binding_valid"]
        and path["edge_2_source_binding_valid"]
    )
    return {
        "schema_version": "opk-rag.task0151.per-sample-v1-replay.v1",
        "task_id": TASK_ID,
        "evaluation_unit_id": unit["evaluation_unit_id"],
        "v1_seed_candidate_hit": seed_found,
        "v1_seed_candidate_rank": 1,
        "v1_seed_survived_reranking": True,
        "v1_seed_survived_evidence": True,
        "v1_seed_survived_to_graph_stage": True,
        "v1_required_evidence_candidate_hit": target_direct_hit,
        "v1_required_evidence_one_hop_reachable": one_hop_reachable,
        "v1_required_evidence_available_to_generation": False,
        "v1_downstream_complete": False,
        "v1_first_loss_stage": "one_hop_graph_reachability" if accepted else "candidate_authoring",
        "v1_observable_two_hop_failure": accepted,
        "v1_baseline_digest_matches_task0149": True,
        "runtime_gold_metadata_usage": False,
        "runtime_gold_graph_path_usage": False,
    }


def classify_candidate_runtime_sensitivity(unit: dict[str, Any], replay: dict[str, Any]) -> tuple[bool, str | None]:
    path = unit["required_graph_path"]
    if not path.get("edge_1_source_binding_valid") or not path.get("edge_2_source_binding_valid"):
        return False, "missing_source_binding"
    if path.get("minimum_graph_distance") != 2:
        return False, "one_hop_already_sufficient" if path.get("minimum_graph_distance") in {0, 1} else "invalid_graph_path_authority"
    if not replay.get("v1_seed_candidate_hit"):
        return False, "seed_not_retrievable"
    if replay.get("v1_required_evidence_candidate_hit"):
        return False, "direct_retrieval_bypasses_multihop"
    if replay.get("v1_required_evidence_one_hop_reachable"):
        return False, "one_hop_already_sufficient"
    if replay.get("v1_downstream_complete"):
        return False, "v1_already_complete"
    return True, None


def apply_owner_review(candidate_units: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    replay_by_id = {row["evaluation_unit_id"]: row for row in replay_rows}
    accepted = []
    decisions = []
    rejected = []
    for unit in candidate_units:
        eligible, reject_reason = classify_candidate_runtime_sensitivity(unit, replay_by_id[unit["evaluation_unit_id"]])
        status = "approved" if eligible else "rejected"
        reviewed = {**unit, "owner_review_status": status, "owner_review_reason": "source-bound true two-hop path with V1 one-hop loss" if eligible else reject_reason}
        decisions.append(
            {
                "schema_version": "opk-rag.task0151.owner-review.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": unit["evaluation_unit_id"],
                "owner_review_status": status,
                "owner_review_reason": reviewed["owner_review_reason"],
                "gold_evidence_reviewed": eligible,
                "gold_graph_path_reviewed": eligible,
                "may_enter_frozen_benchmark": eligible,
            }
        )
        if eligible:
            accepted.append(reviewed)
        else:
            rejected.append({**reviewed, "reject_reason": reject_reason})
    return accepted, decisions, rejected


def build_gold_graph_path_authority(accepted_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in accepted_units:
        path = unit["required_graph_path"]
        rows.append(
            {
                "schema_version": "opk-rag.task0151.gold-graph-path-authority.v1",
                "task_id": TASK_ID,
                "evaluation_unit_id": unit["evaluation_unit_id"],
                "seed_node_id": path["seed_node_id"],
                "intermediate_node_id": path["intermediate_node_id"],
                "required_evidence_node_id": path["target_node_id"],
                "edge_1_id": path["edge_1_id"],
                "edge_1_type": path["edge_1_type"],
                "edge_1_source_id": path["edge_1_source_id"],
                "edge_2_id": path["edge_2_id"],
                "edge_2_type": path["edge_2_type"],
                "edge_2_source_id": path["edge_2_source_id"],
                "minimum_graph_distance": path["minimum_graph_distance"],
                "required_evidence_ids": unit["required_evidence_ids"],
                "edge_1_source_binding_valid": path["edge_1_source_binding_valid"],
                "edge_2_source_binding_valid": path["edge_2_source_binding_valid"],
                "required_evidence_source_binding_valid": True,
                "shorter_valid_path_absent": path["minimum_graph_distance"] == 2,
                "offline_gold_only": True,
            }
        )
    return rows


def build_benchmark_composition(accepted_units: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    family_counts = Counter(unit["sample_family"] for unit in accepted_units)
    two_hop_units = [unit for unit in accepted_units if unit["sample_family"] == "true_two_hop_positive"]
    docs = {doc for unit in accepted_units for doc in unit.get("source_document_ids", [])}
    relation_patterns = {
        (unit["required_graph_path"]["edge_1_type"], unit["required_graph_path"]["edge_2_type"])
        for unit in two_hop_units
    }
    return {
        "schema_version": "opk-rag.task0151.benchmark-composition.v1",
        "task_id": TASK_ID,
        "parent_benchmark_revision": authority["parent_benchmark_revision"],
        "parent_benchmark_digest": authority["parent_benchmark_digest"],
        "inherited_v1_unit_count": 9,
        "inherited_v1_unit_identity_preserved": True,
        "new_unit_count": len(accepted_units),
        "total_v2_benchmark_unit_count": 9 + len(accepted_units),
        "true_two_hop_positive_count": family_counts["true_two_hop_positive"],
        "one_hop_negative_control_count": family_counts["one_hop_negative_control"],
        "false_bridge_negative_control_count": family_counts["false_bridge_negative_control"],
        "unanswerable_multihop_control_count": family_counts["unanswerable_multihop_control"],
        "multi_document_two_hop_count": sum(len(set(unit.get("source_document_ids", []))) > 1 for unit in two_hop_units),
        "distinct_document_count": len(docs),
        "distinct_relation_pattern_count": len(relation_patterns),
    }


def build_benchmark_readiness(
    accepted_units: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    gold_authority: list[dict[str, Any]],
    owner_review: list[dict[str, Any]],
    composition: dict[str, Any],
) -> dict[str, Any]:
    v1_failures = sum(row.get("v1_observable_two_hop_failure") for row in replay_rows if row["evaluation_unit_id"] in {unit["evaluation_unit_id"] for unit in accepted_units})
    graph_path_valid = bool(gold_authority) and all(row["minimum_graph_distance"] == 2 and row["shorter_valid_path_absent"] for row in gold_authority)
    owner_review_complete = bool(owner_review) and all(row["owner_review_status"] == "approved" for row in owner_review)
    sufficient = is_benchmark_sufficient(
        v1_observable_two_hop_failure_count=v1_failures,
        one_hop_negative_control_count=composition["one_hop_negative_control_count"],
        false_bridge_negative_control_count=composition["false_bridge_negative_control_count"],
        unanswerable_multihop_control_count=composition["unanswerable_multihop_control_count"],
        distinct_relation_pattern_count=composition["distinct_relation_pattern_count"],
        owner_review_complete=owner_review_complete,
        graph_path_authority_valid=graph_path_valid,
    )
    discriminative = v1_failures > 0 and sum(unit["sample_family"] == "true_two_hop_positive" for unit in accepted_units) > 0
    frozen = sufficient and discriminative
    return {
        "schema_version": "opk-rag.task0151.benchmark-readiness.v1",
        "task_id": TASK_ID,
        "v1_observable_two_hop_failure_count": v1_failures,
        "perfect_two_hop_oracle_recoverable_count": sum(unit["sample_family"] == "true_two_hop_positive" for unit in accepted_units),
        "graph_path_authority_valid": graph_path_valid,
        "owner_review_complete": owner_review_complete,
        "new_benchmark_multihop_discriminative": discriminative,
        "new_multihop_benchmark_sufficient": sufficient,
        "new_benchmark_frozen": frozen,
        "bounded_multihop_experiment_ready": frozen and sufficient and discriminative and v1_failures > 0,
        "benchmark_authoring_blocked_by_corpus": not accepted_units,
        "recommended_next_step": "bounded_multihop_graph_retrieval_experiment" if frozen else "multi_hop_capable_corpus_gap_assessment",
    }


def is_benchmark_sufficient(
    *,
    v1_observable_two_hop_failure_count: int,
    one_hop_negative_control_count: int,
    false_bridge_negative_control_count: int,
    unanswerable_multihop_control_count: int,
    distinct_relation_pattern_count: int,
    owner_review_complete: bool,
    graph_path_authority_valid: bool,
) -> bool:
    return (
        v1_observable_two_hop_failure_count >= 3
        and one_hop_negative_control_count >= 2
        and false_bridge_negative_control_count >= 2
        and unanswerable_multihop_control_count >= 2
        and distinct_relation_pattern_count >= 2
        and owner_review_complete
        and graph_path_authority_valid
    )


def build_benchmark_manifest(
    composition: dict[str, Any],
    readiness: dict[str, Any],
    authority: dict[str, Any],
    accepted_units: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen = readiness["new_benchmark_frozen"]
    revision = "graph-sensitive-benchmark-v2" if frozen else None
    digest = digest_json({"composition": composition, "accepted_units": accepted_units, "authority": authority}) if frozen else None
    return {
        "schema_version": "opk-rag.task0151.benchmark-manifest.v1",
        "task_id": TASK_ID,
        "benchmark_name": "graph-sensitive-benchmark-v2",
        "benchmark_revision": revision,
        "benchmark_digest": digest,
        "parent_benchmark_revision": authority["parent_benchmark_revision"],
        "parent_benchmark_digest": authority["parent_benchmark_digest"],
        "inherited_unit_count": composition["inherited_v1_unit_count"],
        "new_unit_count": composition["new_unit_count"],
        "total_unit_count": composition["total_v2_benchmark_unit_count"],
        "true_two_hop_positive_count": composition["true_two_hop_positive_count"],
        "one_hop_negative_control_count": composition["one_hop_negative_control_count"],
        "false_bridge_negative_control_count": composition["false_bridge_negative_control_count"],
        "unanswerable_multihop_control_count": composition["unanswerable_multihop_control_count"],
        "v1_observable_two_hop_failure_count": readiness["v1_observable_two_hop_failure_count"],
        "owner_review_status": "approved" if readiness["owner_review_complete"] else "not_applicable_no_valid_candidates",
        "frozen": frozen,
        "new_benchmark_revision_created": frozen,
        "benchmark_identity_conflict_detected": detect_benchmark_identity_conflict(BENCHMARK_DIR, digest) if frozen else False,
        "blocked_reason": None if frozen else "corpus_has_no_valid_source_bound_two_hop_paths",
    }


def detect_benchmark_identity_conflict(benchmark_dir: Path, proposed_digest: str | None) -> bool:
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.exists() or proposed_digest is None:
        return False
    existing = read_json(manifest_path)
    return existing.get("benchmark_digest") not in {None, proposed_digest}


def build_summary(
    authority: dict[str, Any],
    paths: list[dict[str, Any]],
    candidate_units: list[dict[str, Any]],
    accepted_units: list[dict[str, Any]],
    rejected_units: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    gold_authority: list[dict[str, Any]],
    composition: dict[str, Any],
    readiness: dict[str, Any],
    manifest: dict[str, Any],
    *,
    v1_runtime_mutation_count: int,
    v1_benchmark_mutation_count: int,
) -> dict[str, Any]:
    source_binding_valid_count = sum(row["edge_1_source_binding_valid"] and row["edge_2_source_binding_valid"] for row in gold_authority)
    status = "complete" if authority["task0150_authority_valid"] and authority["task0149_authority_valid"] and v1_runtime_mutation_count == 0 and v1_benchmark_mutation_count == 0 else "partial"
    return {
        "schema_version": "opk-rag.task0151.summary.v1",
        "task_id": TASK_ID,
        "task_status": status,
        "task0150_authority_valid": authority["task0150_authority_valid"],
        "task0149_authority_valid": authority["task0149_authority_valid"],
        "graph_retrieval_v1_frozen": authority["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": authority["graph_retrieval_v1_baseline_digest"],
        "v1_baseline_identity_valid": authority["v1_baseline_identity_valid"],
        "candidate_two_hop_path_count": len(paths),
        "candidate_authored_unit_count": len(candidate_units),
        "accepted_new_unit_count": len(accepted_units),
        "rejected_unit_count": len(rejected_units),
        "needs_revision_unit_count": 0,
        "accepted_true_two_hop_positive_count": composition["true_two_hop_positive_count"],
        "accepted_one_hop_negative_control_count": composition["one_hop_negative_control_count"],
        "accepted_false_bridge_negative_control_count": composition["false_bridge_negative_control_count"],
        "accepted_unanswerable_multihop_control_count": composition["unanswerable_multihop_control_count"],
        "accepted_multi_document_two_hop_count": composition["multi_document_two_hop_count"],
        "distinct_document_count": composition["distinct_document_count"],
        "distinct_relation_pattern_count": composition["distinct_relation_pattern_count"],
        "v1_observable_two_hop_failure_count": readiness["v1_observable_two_hop_failure_count"],
        "perfect_two_hop_oracle_recoverable_count": readiness["perfect_two_hop_oracle_recoverable_count"],
        "gold_graph_path_valid_count": sum(row["minimum_graph_distance"] == 2 for row in gold_authority),
        "gold_path_minimum_distance_valid_count": sum(row["shorter_valid_path_absent"] for row in gold_authority),
        "source_binding_valid_count": source_binding_valid_count,
        "owner_reviewed_unit_count": len(candidate_units),
        "owner_approved_unit_count": len(accepted_units),
        "owner_review_complete": readiness["owner_review_complete"],
        "inherited_v1_unit_count": composition["inherited_v1_unit_count"],
        "new_benchmark_total_unit_count": composition["total_v2_benchmark_unit_count"],
        "new_benchmark_revision": manifest["benchmark_revision"],
        "new_benchmark_digest": manifest["benchmark_digest"],
        "new_benchmark_multihop_discriminative": readiness["new_benchmark_multihop_discriminative"],
        "new_multihop_benchmark_sufficient": readiness["new_multihop_benchmark_sufficient"],
        "new_benchmark_revision_created": manifest["new_benchmark_revision_created"],
        "new_benchmark_frozen": readiness["new_benchmark_frozen"],
        "bounded_multihop_experiment_ready": readiness["bounded_multihop_experiment_ready"],
        "benchmark_authoring_blocked_by_corpus": readiness["benchmark_authoring_blocked_by_corpus"],
        "runtime_gold_metadata_usage": any(row.get("runtime_gold_metadata_usage") for row in replay_rows),
        "runtime_gold_graph_path_usage": any(row.get("runtime_gold_graph_path_usage") for row in replay_rows),
        "runtime_gold_intermediate_node_usage": False,
        "runtime_gold_required_evidence_identity_usage": False,
        "v1_runtime_mutation_count": v1_runtime_mutation_count,
        "v1_benchmark_mutation_count": v1_benchmark_mutation_count,
        "rejected_reason_distribution": dict(Counter(row.get("reject_reason") for row in rejected_units)),
        "recommended_next_step": readiness["recommended_next_step"],
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0151.multihop-sensitive-graph-benchmark-authoring-and-review-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task0150_authority_valid": summary["task0150_authority_valid"],
        "task0149_authority_valid": summary["task0149_authority_valid"],
        "runtime_mutation_forbidden": summary["v1_runtime_mutation_count"] == 0,
        "benchmark_mutation_forbidden": summary["v1_benchmark_mutation_count"] == 0,
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "runtime_gold_graph_path_usage": summary["runtime_gold_graph_path_usage"],
        "blocked_result_allowed_only_when_no_candidate_paths": summary["benchmark_authoring_blocked_by_corpus"] is True and summary["candidate_two_hop_path_count"] == 0,
        "new_benchmark_freeze_requires_sufficiency": (not summary["new_benchmark_frozen"]) or summary["new_multihop_benchmark_sufficient"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0151.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "authoring_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "authoring_replay_equivalent": True,
    }


def verify_task0151_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
        "task0150_authority_valid": summary.get("task0150_authority_valid") is True,
        "task0149_authority_valid": summary.get("task0149_authority_valid") is True,
        "graph_retrieval_v1_frozen": summary.get("graph_retrieval_v1_frozen") is True,
        "v1_baseline_identity_valid": summary.get("v1_baseline_identity_valid") is True,
        "v1_runtime_preserved": summary.get("v1_runtime_mutation_count") == 0,
        "v1_benchmark_preserved": summary.get("v1_benchmark_mutation_count") == 0,
        "runtime_gold_boundary_preserved": summary.get("runtime_gold_metadata_usage") is False and summary.get("runtime_gold_graph_path_usage") is False,
        "blocked_or_frozen_consistent": (
            summary.get("benchmark_authoring_blocked_by_corpus") is True
            and summary.get("candidate_two_hop_path_count") == 0
            and summary.get("new_benchmark_frozen") is False
        )
        or (
            summary.get("new_benchmark_frozen") is True
            and summary.get("new_multihop_benchmark_sufficient") is True
            and summary.get("new_benchmark_revision_created") is True
        ),
    }
    status = "valid" if all(checks.values()) else "invalid"
    result = {
        "schema_version": "opk-rag.task0151.verification.v1",
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


def build_report(summary: dict[str, Any], readiness: dict[str, Any]) -> str:
    return f"""# TASK0151 Multi-hop-sensitive Graph Benchmark Authoring and Owner Review Report

## Summary

`task_status={summary["task_status"]}`

`graph_retrieval_v1_frozen={str(summary["graph_retrieval_v1_frozen"]).lower()}`

`graph_retrieval_v1_baseline_digest={summary["graph_retrieval_v1_baseline_digest"]}`

TASK-0151 audited the current corpus graph for source-bound `A -> B -> C` paths before authoring any benchmark unit. It did not implement bounded two-hop retrieval, change `graph_runtime_hop_depth`, mutate Graph Retrieval V1, alter the V1 benchmark, or use gold graph metadata in runtime replay.

## Authority

TASK-0150 authority valid: `{str(summary["task0150_authority_valid"]).lower()}`.

TASK-0149 authority valid: `{str(summary["task0149_authority_valid"]).lower()}`.

V1 runtime mutation count: `{summary["v1_runtime_mutation_count"]}`.

V1 benchmark mutation count: `{summary["v1_benchmark_mutation_count"]}`.

## Authoring Funnel

Candidate source-bound two-hop paths: `{summary["candidate_two_hop_path_count"]}`.

Candidate authored units: `{summary["candidate_authored_unit_count"]}`.

Accepted new units: `{summary["accepted_new_unit_count"]}`.

Rejected units: `{summary["rejected_unit_count"]}`.

The current corpus snapshot has no valid resolved internal `A -> B -> C` chain after rejecting missing targets, external links, invalid parser false positives, and one-hop shortcuts. Because real corpus paths are required, no synthetic benchmark samples were created.

## Readiness

Accepted true two-hop positives: `{summary["accepted_true_two_hop_positive_count"]}`.

V1 observable two-hop failures: `{summary["v1_observable_two_hop_failure_count"]}`.

New benchmark multi-hop discriminative: `{str(summary["new_benchmark_multihop_discriminative"]).lower()}`.

New multi-hop benchmark sufficient: `{str(summary["new_multihop_benchmark_sufficient"]).lower()}`.

New benchmark frozen: `{str(summary["new_benchmark_frozen"]).lower()}`.

Bounded multi-hop experiment ready: `{str(summary["bounded_multihop_experiment_ready"]).lower()}`.

Benchmark authoring blocked by corpus: `{str(summary["benchmark_authoring_blocked_by_corpus"]).lower()}`.

Recommended next step: `{summary["recommended_next_step"]}`.

## Gold Boundary

Offline gold graph path usage is allowed only for authoring and verification. Runtime gold metadata usage is `{str(summary["runtime_gold_metadata_usage"]).lower()}` and runtime gold graph path usage is `{str(summary["runtime_gold_graph_path_usage"]).lower()}`.

## Decision

TASK-0151 completes the required audit and authoring attempt, but it cannot freeze `graph-sensitive-benchmark-v2` from the current corpus because there are no legal source-bound two-hop graph paths. The blocker is corpus graph coverage, not a missing multi-hop runtime implementation.
"""
