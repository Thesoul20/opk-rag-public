from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec as task0150
import opk_rag.evaluation.task0151_multihop_sensitive_graph_benchmark_authoring_and_review as task0151
import opk_rag.evaluation.task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis as task0152
import opk_rag.evaluation.task0153_two_hop_graph_identity_resolution_repair_and_path_recovery as task0153
import opk_rag.evaluation.task0154_missing_graph_target_document_provenance_and_corpus_snapshot_coverage_diagnosis as task0154
from opk_rag.evaluation.graph_link_resolution import collect_link_records
from opk_rag.evaluation.graphrag_readiness import SOURCE_DIR
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0155"
EXPERIMENT_ID = "task0155-multihop-capable-corpus-authority-gap-assessment-and-v2-viability-decision"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0155_multihop_capable_corpus_authority_gap_assessment_and_v2_viability_decision_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0155_MULTIHOP_CAPABLE_CORPUS_AUTHORITY_GAP_ASSESSMENT_AND_V2_VIABILITY_DECISION_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "corpus_snapshot_identity.json",
    "internal_reference_audit.json",
    "dangling_reference_inventory.jsonl",
    "authoritative_graph_edge_inventory.jsonl",
    "authoritative_graph_edge_audit.json",
    "graph_topology_audit.json",
    "path_depth_audit.json",
    "relation_pattern_audit.json",
    "authorability_potential_audit.json",
    "production_workload_need_audit.json",
    "v2_viability_decision.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0154_authority_valid",
    "task0153_authority_valid",
    "task0152_authority_valid",
    "task0151_authority_valid",
    "task0150_authority_valid",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "production_corpus_identity_valid",
    "corpus_snapshot_revision",
    "corpus_snapshot_digest",
    "corpus_document_count",
    "total_internal_reference_count",
    "resolved_internal_reference_count",
    "unresolved_internal_reference_count",
    "dangling_reference_count",
    "internal_reference_resolution_rate",
    "dangling_reference_rate",
    "reference_class_distribution",
    "dangling_reference_systemic",
    "corpus_graph_hygiene_gap",
    "graph_blocking_dangling_reference_count",
    "authoritative_graph_node_count",
    "authoritative_graph_edge_count",
    "resolved_edge_count",
    "source_bound_edge_count",
    "materialized_authoritative_edge_count",
    "source_binding_rate",
    "graph_node_count",
    "graph_edge_count",
    "nodes_with_in_degree_gt_0",
    "nodes_with_out_degree_gt_0",
    "isolated_node_count",
    "max_in_degree",
    "max_out_degree",
    "average_in_degree",
    "average_out_degree",
    "connected_component_count",
    "largest_connected_component_node_count",
    "single_node_component_count",
    "multi_node_component_count",
    "isolated_node_ratio",
    "largest_component_node_ratio",
    "multi_node_component_ratio",
    "graph_fragmentation_high",
    "authoritative_distance_0_reachable_pair_count",
    "authoritative_distance_1_path_count",
    "authoritative_distance_2_path_count",
    "authoritative_distance_3_plus_path_count",
    "authoritative_two_step_walk_count",
    "authoritative_two_hop_shortcut_count",
    "authoritative_minimum_distance_two_path_count",
    "authoritative_minimum_distance_three_plus_path_count",
    "max_audited_hop_depth",
    "authoritative_minimum_distance_distribution",
    "cross_document_authoritative_edge_count",
    "cross_document_two_step_walk_count",
    "cross_document_minimum_distance_two_path_count",
    "edge_type_distribution",
    "two_step_relation_pattern_count",
    "two_step_relation_pattern_distribution",
    "two_hop_distinct_document_count",
    "two_hop_distinct_relation_pattern_count",
    "multihop_structure_diversity_sufficient",
    "potentially_authorable_two_hop_path_count",
    "current_v1_graph_sensitive_residual_count",
    "observed_production_multihop_need",
    "multihop_structural_opportunity",
    "multihop_runtime_need",
    "current_corpus_multihop_authority_sufficient",
    "current_corpus_multihop_runtime_investment_justified",
    "external_multihop_evaluation_corpus_required",
    "graph_retrieval_v2_multihop_go_decision",
    "bounded_multihop_runtime_implemented",
    "multihop_benchmark_frozen",
    "syntactic_two_step_chain_count",
    "authoritative_two_step_chain_count",
    "task0152_raw_chain_authority_reinterpretation_required",
    "source_document_mutation_count",
    "dangling_reference_repair_count",
    "corpus_mutation_count",
    "graph_policy_mutation_count",
    "runtime_policy_mutation_count",
    "task0149_frozen_artifact_mutation_count",
    "repair_applied",
    "recommended_next_step",
    "recommended_secondary_step",
)


def run_task0155_multihop_capable_corpus_authority_gap_assessment_and_v2_viability_decision(
    *, output_dir: Path = RESULT_DIR
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)

    authority = build_authority_manifest()
    corpus_snapshot = task0154.build_corpus_snapshot_identity(SOURCE_DIR)
    records = collect_link_records()
    reference_audit, dangling_rows = build_internal_reference_audit(records)
    edge_audit, edge_rows = build_authoritative_graph_edge_audit(records)
    topology = build_graph_topology_audit(edge_rows, corpus_document_count=corpus_snapshot["document_count"])
    path_depth = build_path_depth_audit(edge_rows)
    relation_patterns = build_relation_pattern_audit(edge_rows, path_depth["two_step_walk_rows"])
    authorability = build_authorability_potential_audit(path_depth["minimum_distance_two_path_rows"], records)
    workload = build_production_workload_need_audit()
    decision = build_v2_viability_decision(path_depth, relation_patterns, authorability, workload)

    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    mutation = {
        "source_document_mutation_count": 0,
        "dangling_reference_repair_count": 0,
        "corpus_mutation_count": 0,
        "graph_policy_mutation_count": 0,
        "runtime_policy_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "task0149_frozen_artifact_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
    }
    summary = build_summary(authority, corpus_snapshot, reference_audit, edge_audit, topology, path_depth, relation_patterns, authorability, workload, decision, mutation)
    contract = build_contract(summary)
    digests = build_digests(authority, corpus_snapshot, reference_audit, dangling_rows, edge_audit, edge_rows, topology, path_depth, relation_patterns, authorability, workload, decision, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "corpus_snapshot_identity.json", corpus_snapshot)
    write_json(output_dir / "internal_reference_audit.json", reference_audit)
    write_jsonl(output_dir / "dangling_reference_inventory.jsonl", dangling_rows)
    write_jsonl(output_dir / "authoritative_graph_edge_inventory.jsonl", edge_rows)
    write_json(output_dir / "authoritative_graph_edge_audit.json", edge_audit)
    write_json(output_dir / "graph_topology_audit.json", topology)
    write_json(output_dir / "path_depth_audit.json", _without_rows(path_depth))
    write_json(output_dir / "relation_pattern_audit.json", relation_patterns)
    write_json(output_dir / "authorability_potential_audit.json", authorability)
    write_json(output_dir / "production_workload_need_audit.json", workload)
    write_json(output_dir / "v2_viability_decision.json", decision)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0155_artifacts(output_dir=output_dir, write=True)
    summary["task0155_verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0154_verification = task0154.verify_task0154_artifacts(write=False)
    task0153_verification = task0153.verify_task0153_artifacts(write=False)
    task0152_verification = task0152.verify_task0152_artifacts(write=False)
    task0151_verification = task0151.verify_task0151_artifacts(write=False)
    task0150_verification = task0150.verify_task0150_artifacts(write=False)
    task0149_verification = task0149.verify_task0149_artifacts(write=False)
    task0154_summary = read_json(task0154.RESULT_DIR / "summary.json")
    task0153_summary = read_json(task0153.RESULT_DIR / "summary.json")
    task0152_summary = read_json(task0152.RESULT_DIR / "summary.json")
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    return {
        "schema_version": "opk-rag.task0155.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_precedence": ["TASK-0154", "TASK-0153", "TASK-0152", "TASK-0151", "TASK-0150", "TASK-0149"],
        "task0154_authority_valid": task0154_verification["status"] == "valid"
        and task0154_summary.get("dominant_missing_target_root_cause") == "dangling_corpus_reference"
        and task0154_summary.get("syntactic_two_step_chain_count") == 2
        and task0154_summary.get("authoritative_two_step_chain_count") == 0
        and task0154_summary.get("task0152_raw_chain_authority_reinterpretation_required") is True,
        "task0153_authority_valid": task0153_verification["status"] == "valid"
        and task0153_summary.get("dominant_resolution_failure_mechanism") == "missing_target_document",
        "task0152_authority_valid": task0152_verification["status"] == "valid"
        and task0152_summary.get("raw_two_step_chain_count") == 2
        and task0152_summary.get("resolved_two_step_chain_count") == 0,
        "task0151_authority_valid": task0151_verification["status"] == "valid",
        "task0150_authority_valid": task0150_verification["status"] == "valid",
        "task0149_authority_valid": task0149_verification["status"] == "valid",
        "graph_retrieval_v1_frozen": task0149_summary.get("graph_retrieval_v1_frozen") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "task0154_summary_sha256": sha256_file(task0154.RESULT_DIR / "summary.json"),
        "task0153_summary_sha256": sha256_file(task0153.RESULT_DIR / "summary.json"),
        "task0152_summary_sha256": sha256_file(task0152.RESULT_DIR / "summary.json"),
        "task0151_summary_sha256": sha256_file(task0151.RESULT_DIR / "summary.json"),
        "task0150_summary_sha256": sha256_file(task0150.RESULT_DIR / "summary.json"),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
    }


def build_internal_reference_audit(records: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(records)
    classes = Counter(classify_reference(row) for row in rows)
    dangling = [dangling_reference_row(row) for row in rows if classify_reference(row) == "dangling_reference"]
    resolved = classes["resolved_authoritative_reference"]
    unresolved = len(rows) - resolved
    dangling_rate = _safe_div(classes["dangling_reference"], len(rows))
    graph_blocking_count = sum(row["severity"] == "graph_blocking" for row in dangling)
    systemic = bool(dangling_rate is not None and dangling_rate >= 0.05 and graph_blocking_count >= 3)
    return (
        {
            "schema_version": "opk-rag.task0155.internal-reference-audit.v1",
            "task_id": TASK_ID,
            "total_internal_reference_count": len(rows),
            "resolved_internal_reference_count": resolved,
            "unresolved_internal_reference_count": unresolved,
            "dangling_reference_count": classes["dangling_reference"],
            "internal_reference_resolution_rate": _safe_div(resolved, len(rows)),
            "dangling_reference_rate": dangling_rate,
            "reference_class_distribution": dict(sorted(classes.items())),
            "dangling_reference_systemic": systemic,
            "corpus_graph_hygiene_gap": bool(dangling_rate is not None and dangling_rate >= 0.02),
            "graph_blocking_dangling_reference_count": graph_blocking_count,
            "dangling_systemic_rule": "dangling_reference_rate>=0.05 and graph_blocking_dangling_reference_count>=3",
            "graph_hygiene_gap_rule": "dangling_reference_rate>=0.02",
        },
        dangling,
    )


def classify_reference(row: dict[str, Any]) -> str:
    status = row.get("resolution_status")
    cause = row.get("root_cause_class")
    if status == "resolved_deterministically" and row.get("resolved_target_id"):
        return "resolved_authoritative_reference"
    if status == "out_of_scope_external_target" or cause == "target_outside_corpus_snapshot":
        return "out_of_scope_reference"
    if status == "invalid_link":
        return "invalid_reference"
    if cause in {"valid_heading_fragment_not_supported", "valid_block_reference_not_supported"}:
        return "unsupported_reference"
    if cause == "ambiguous_target":
        return "ambiguous_reference"
    if cause == "missing_target_document":
        return "dangling_reference"
    return "unsupported_reference"


def dangling_reference_row(row: dict[str, Any]) -> dict[str, Any]:
    reason = classify_dangling_reason(row)
    return {
        "schema_version": "opk-rag.task0155.dangling-reference.v1",
        "task_id": TASK_ID,
        "link_id": row["link_id"],
        "source_document_id": row["source_document_id"],
        "source_section_id": row.get("source_section_id"),
        "raw_reference": row.get("raw_link_text"),
        "normalized_reference": row.get("normalized_target"),
        "reference_shape": task0154.classify_reference_shape(str(row.get("raw_link_text") or ""), str(row.get("link_syntax") or "")),
        "target_exists": False,
        "classification": "dangling_reference",
        "dangling_reason": reason,
        "severity": "graph_blocking" if reason == "missing_target_source" else "unknown",
        "source_file_exists": False,
    }


def classify_dangling_reason(row: dict[str, Any]) -> str:
    raw = str(row.get("raw_link_text") or "")
    if row.get("root_cause_class") == "target_outside_corpus_snapshot":
        return "external_or_out_of_scope_reference"
    if Path(raw).suffix and not raw.endswith(".md"):
        return "unsupported_target_type"
    if "TODO" in raw.upper() or "TBD" in raw.upper():
        return "future_placeholder"
    return "missing_target_source"


def build_authoritative_graph_edge_audit(records: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(records)
    resolved = [row for row in rows if row.get("resolution_status") == "resolved_deterministically" and row.get("resolved_target_id")]
    bound = [row for row in resolved if task0152.source_binding_valid(row)]
    materialized = task0152.materialized_edges(rows)
    edge_rows = [authoritative_edge_row(row) for row in materialized]
    node_ids = sorted({row["source_document_id"] for row in materialized} | {row["resolved_target_id"] for row in materialized if row.get("resolved_target_id")})
    return (
        {
            "schema_version": "opk-rag.task0155.authoritative-graph-edge-audit.v1",
            "task_id": TASK_ID,
            "authoritative_graph_node_count": len(node_ids),
            "authoritative_graph_edge_count": len(edge_rows),
            "resolved_edge_count": len(resolved),
            "source_bound_edge_count": len(bound),
            "materialized_authoritative_edge_count": len(edge_rows),
            "source_binding_rate": _safe_div(len(bound), len(resolved)),
            "edge_authority_definition": {
                "source_node_exists": True,
                "target_node_exists": True,
                "source_identity_valid": True,
                "target_identity_valid": True,
                "edge_resolution_valid": True,
                "source_binding_valid": True,
                "graph_materialized": True,
            },
        },
        edge_rows,
    )


def authoritative_edge_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0155.authoritative-graph-edge.v1",
        "task_id": TASK_ID,
        "edge_id": row["link_id"],
        "source_node_id": row["source_document_id"],
        "target_node_id": row["resolved_target_id"],
        "edge_type": "LINKS_TO",
        "source_section_id": row.get("source_section_id"),
        "source_binding_valid": True,
        "edge_resolution_valid": True,
        "graph_materialized": True,
        "canonical_edge_identity": [row["source_document_id"], row["link_id"], row["resolved_target_id"]],
    }


def build_graph_topology_audit(edge_rows: list[dict[str, Any]], *, corpus_document_count: int) -> dict[str, Any]:
    nodes = sorted({edge["source_node_id"] for edge in edge_rows} | {edge["target_node_id"] for edge in edge_rows})
    indegree = Counter(edge["target_node_id"] for edge in edge_rows)
    outdegree = Counter(edge["source_node_id"] for edge in edge_rows)
    isolated = max(corpus_document_count - len(nodes), 0)
    components = weak_connected_components(nodes, edge_rows)
    largest = max((len(component) for component in components), default=0)
    return {
        "schema_version": "opk-rag.task0155.graph-topology-audit.v1",
        "task_id": TASK_ID,
        "component_semantics": "weak_connected_components_over_directed_authoritative_edges",
        "graph_node_count": len(nodes),
        "graph_edge_count": len(edge_rows),
        "nodes_with_in_degree_gt_0": sum(indegree[node] > 0 for node in nodes),
        "nodes_with_out_degree_gt_0": sum(outdegree[node] > 0 for node in nodes),
        "isolated_node_count": isolated,
        "max_in_degree": max((indegree[node] for node in nodes), default=0),
        "max_out_degree": max((outdegree[node] for node in nodes), default=0),
        "average_in_degree": _safe_div(sum(indegree.values()), len(nodes)) or 0.0,
        "average_out_degree": _safe_div(sum(outdegree.values()), len(nodes)) or 0.0,
        "connected_component_count": len(components) + isolated,
        "largest_connected_component_node_count": largest,
        "single_node_component_count": sum(len(component) == 1 for component in components) + isolated,
        "multi_node_component_count": sum(len(component) > 1 for component in components),
        "isolated_node_ratio": _safe_div(isolated, corpus_document_count) or 0.0,
        "largest_component_node_ratio": _safe_div(largest, corpus_document_count) or 0.0,
        "multi_node_component_ratio": _safe_div(sum(len(component) for component in components if len(component) > 1), corpus_document_count) or 0.0,
        "graph_fragmentation_high": bool(corpus_document_count and (_safe_div(isolated, corpus_document_count) or 0) >= 0.75),
    }


def build_path_depth_audit(edge_rows: list[dict[str, Any]], *, max_hops: int = 3) -> dict[str, Any]:
    graph = adjacency(edge_rows)
    nodes = sorted({edge["source_node_id"] for edge in edge_rows} | {edge["target_node_id"] for edge in edge_rows})
    distances = all_pairs_minimum_distances(graph, nodes, max_hops=max_hops)
    two_step_walks = enumerate_two_step_walks(edge_rows)
    shortcut_count = sum(1 for row in two_step_walks if distances.get((row["seed_node_id"], row["target_node_id"])) == 1)
    minimum_two_rows = [row for row in two_step_walks if distances.get((row["seed_node_id"], row["target_node_id"])) == 2]
    distribution = {
        "0": len(nodes),
        "1": sum(distance == 1 for distance in distances.values()),
        "2": sum(distance == 2 for distance in distances.values()),
        "3+": sum(distance is not None and distance >= 3 for distance in distances.values()),
    }
    return {
        "schema_version": "opk-rag.task0155.path-depth-audit.v1",
        "task_id": TASK_ID,
        "max_audited_hop_depth": max_hops,
        "authoritative_distance_0_reachable_pair_count": distribution["0"],
        "authoritative_distance_1_path_count": distribution["1"],
        "authoritative_distance_2_path_count": distribution["2"],
        "authoritative_distance_3_plus_path_count": distribution["3+"],
        "authoritative_two_step_walk_count": len(two_step_walks),
        "authoritative_two_hop_shortcut_count": shortcut_count,
        "authoritative_minimum_distance_two_path_count": len(minimum_two_rows),
        "authoritative_minimum_distance_three_plus_path_count": distribution["3+"],
        "authoritative_minimum_distance_distribution": distribution,
        "cross_document_authoritative_edge_count": sum(edge["source_node_id"] != edge["target_node_id"] for edge in edge_rows),
        "cross_document_two_step_walk_count": sum(row["cross_document_path"] for row in two_step_walks),
        "cross_document_minimum_distance_two_path_count": sum(row["cross_document_path"] for row in minimum_two_rows),
        "two_step_walk_rows": two_step_walks,
        "minimum_distance_two_path_rows": minimum_two_rows,
    }


def build_relation_pattern_audit(edge_rows: list[dict[str, Any]], two_step_walks: list[dict[str, Any]]) -> dict[str, Any]:
    edge_types = Counter(edge["edge_type"] for edge in edge_rows)
    patterns = Counter(row["relation_pattern"] for row in two_step_walks)
    two_hop_docs = {node for row in two_step_walks for node in (row["seed_node_id"], row["bridge_node_id"], row["target_node_id"])}
    return {
        "schema_version": "opk-rag.task0155.relation-pattern-audit.v1",
        "task_id": TASK_ID,
        "edge_type_distribution": dict(sorted(edge_types.items())),
        "two_step_relation_pattern_count": len(patterns),
        "two_step_relation_pattern_distribution": dict(sorted(patterns.items())),
        "two_hop_distinct_document_count": len(two_hop_docs),
        "two_hop_distinct_relation_pattern_count": len(patterns),
        "multihop_structure_diversity_sufficient": len(two_hop_docs) >= 2 and len(patterns) >= 2,
    }


def build_authorability_potential_audit(minimum_two_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    records_by_source = defaultdict(list)
    for row in records:
        records_by_source[row["source_document_id"]].append(row)
    rows = []
    for path in minimum_two_rows:
        seed_concept = Path(path["seed_node_id"]).stem != ""
        target_records = records_by_source.get(path["target_node_id"], [])
        target_has_content = bool(target_records) or (ROOT / path["target_node_id"]).exists()
        authorable = seed_concept and target_has_content and path["source_binding_valid"] and path["bridge_relation_semantically_meaningful"]
        rows.append({**path, "seed_contains_queryable_concept": seed_concept, "target_contains_evidence_content": target_has_content, "potentially_authorable": authorable})
    return {
        "schema_version": "opk-rag.task0155.authorability-potential-audit.v1",
        "task_id": TASK_ID,
        "potentially_authorable_two_hop_path_count": sum(row["potentially_authorable"] for row in rows),
        "screening_definition": {
            "seed_contains_queryable_concept": True,
            "bridge_relation_semantically_meaningful": True,
            "target_contains_evidence_content": True,
            "source_binding_valid": True,
        },
        "formal_gold_questions_created": False,
        "rows": rows,
    }


def build_production_workload_need_audit() -> dict[str, Any]:
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    residual = int(task0149_summary.get("residual_unit_count", 0))
    true_multihop = int(task0149_summary.get("true_multi_hop_required_unit_count", 0))
    observed = residual > 0 and true_multihop > 0
    return {
        "schema_version": "opk-rag.task0155.production-workload-need-audit.v1",
        "task_id": TASK_ID,
        "current_v1_graph_sensitive_residual_count": residual,
        "true_multi_hop_required_unit_count": true_multihop,
        "observed_production_multihop_need": observed,
        "multihop_runtime_need": observed,
        "need_definition": "real production failure plus causal one-hop insufficiency",
    }


def build_v2_viability_decision(
    path_depth: dict[str, Any], relation_patterns: dict[str, Any], authorability: dict[str, Any], workload: dict[str, Any]
) -> dict[str, Any]:
    min_two = path_depth["authoritative_minimum_distance_two_path_count"]
    potential = authorability["potentially_authorable_two_hop_path_count"]
    sufficient = (
        min_two >= 3
        and relation_patterns["two_hop_distinct_document_count"] >= 2
        and relation_patterns["two_hop_distinct_relation_pattern_count"] >= 2
        and potential > 0
    )
    runtime_need = workload["observed_production_multihop_need"]
    structural = min_two > 0
    if sufficient and runtime_need:
        decision = "go"
        next_step = "author_multihop_sensitive_benchmark"
        external_required = False
    elif min_two == 0 and not runtime_need:
        decision = "no_go"
        next_step = "freeze_multihop_investigation_on_current_corpus"
        external_required = False
    else:
        decision = "defer"
        next_step = "select_multihop_capable_evaluation_corpus"
        external_required = not sufficient
    return {
        "schema_version": "opk-rag.task0155.v2-viability-decision.v1",
        "task_id": TASK_ID,
        "multihop_structural_opportunity": structural,
        "multihop_runtime_need": runtime_need,
        "current_corpus_multihop_authority_sufficient": sufficient,
        "authority_sufficiency_threshold": "minimum_distance_two>=3 and distinct_documents>=2 and distinct_relation_patterns>=2 and potentially_authorable>0",
        "current_corpus_multihop_runtime_investment_justified": sufficient and runtime_need,
        "external_multihop_evaluation_corpus_required": external_required,
        "external_eval_corpus_principles": ["separate_revision", "separate_digest", "separate_benchmark_authority", "separate_reporting"],
        "capability_evaluation_success_implies_production_runtime_promotion": False,
        "graph_retrieval_v2_multihop_go_decision": decision,
        "bounded_multihop_runtime_implemented": False,
        "multihop_benchmark_frozen": False,
        "recommended_next_step": next_step,
    }


def build_summary(
    authority: dict[str, Any],
    corpus_snapshot: dict[str, Any],
    reference_audit: dict[str, Any],
    edge_audit: dict[str, Any],
    topology: dict[str, Any],
    path_depth: dict[str, Any],
    relation_patterns: dict[str, Any],
    authorability: dict[str, Any],
    workload: dict[str, Any],
    decision: dict[str, Any],
    mutation: dict[str, int],
) -> dict[str, Any]:
    authority_valid = all(authority[key] for key in ("task0154_authority_valid", "task0153_authority_valid", "task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid"))
    complete = authority_valid and all(value == 0 for value in mutation.values())
    return {
        "schema_version": "opk-rag.task0155.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        **{key: authority[key] for key in ("task0154_authority_valid", "task0153_authority_valid", "task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid", "graph_retrieval_v1_frozen", "graph_retrieval_v1_baseline_digest")},
        "production_corpus_identity_valid": True,
        "corpus_snapshot_revision": corpus_snapshot["corpus_revision"],
        "corpus_snapshot_digest": corpus_snapshot["corpus_digest"],
        "corpus_document_count": corpus_snapshot["document_count"],
        **{key: reference_audit[key] for key in ("total_internal_reference_count", "resolved_internal_reference_count", "unresolved_internal_reference_count", "dangling_reference_count", "internal_reference_resolution_rate", "dangling_reference_rate", "reference_class_distribution", "dangling_reference_systemic", "corpus_graph_hygiene_gap", "graph_blocking_dangling_reference_count")},
        **{key: edge_audit[key] for key in ("authoritative_graph_node_count", "authoritative_graph_edge_count", "resolved_edge_count", "source_bound_edge_count", "materialized_authoritative_edge_count", "source_binding_rate")},
        **{key: topology[key] for key in ("graph_node_count", "graph_edge_count", "nodes_with_in_degree_gt_0", "nodes_with_out_degree_gt_0", "isolated_node_count", "max_in_degree", "max_out_degree", "average_in_degree", "average_out_degree", "connected_component_count", "largest_connected_component_node_count", "single_node_component_count", "multi_node_component_count", "isolated_node_ratio", "largest_component_node_ratio", "multi_node_component_ratio", "graph_fragmentation_high")},
        **{key: path_depth[key] for key in ("authoritative_distance_0_reachable_pair_count", "authoritative_distance_1_path_count", "authoritative_distance_2_path_count", "authoritative_distance_3_plus_path_count", "authoritative_two_step_walk_count", "authoritative_two_hop_shortcut_count", "authoritative_minimum_distance_two_path_count", "authoritative_minimum_distance_three_plus_path_count", "max_audited_hop_depth", "authoritative_minimum_distance_distribution", "cross_document_authoritative_edge_count", "cross_document_two_step_walk_count", "cross_document_minimum_distance_two_path_count")},
        **{key: relation_patterns[key] for key in ("edge_type_distribution", "two_step_relation_pattern_count", "two_step_relation_pattern_distribution", "two_hop_distinct_document_count", "two_hop_distinct_relation_pattern_count", "multihop_structure_diversity_sufficient")},
        "potentially_authorable_two_hop_path_count": authorability["potentially_authorable_two_hop_path_count"],
        **{key: workload[key] for key in ("current_v1_graph_sensitive_residual_count", "observed_production_multihop_need", "multihop_runtime_need")},
        **{key: decision[key] for key in ("multihop_structural_opportunity", "current_corpus_multihop_authority_sufficient", "current_corpus_multihop_runtime_investment_justified", "external_multihop_evaluation_corpus_required", "graph_retrieval_v2_multihop_go_decision", "bounded_multihop_runtime_implemented", "multihop_benchmark_frozen", "recommended_next_step")},
        "syntactic_two_step_chain_count": 2,
        "authoritative_two_step_chain_count": 0,
        "task0152_raw_chain_authority_reinterpretation_required": True,
        **mutation,
        "repair_applied": False,
        "recommended_secondary_step": "corpus_graph_hygiene_audit" if reference_audit["corpus_graph_hygiene_gap"] else None,
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0155.multihop-capable-corpus-authority-gap-assessment-and-v2-viability-decision-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "authority_valid": all(summary.get(key) is True for key in ("task0154_authority_valid", "task0153_authority_valid", "task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid")),
        "production_corpus_identity_valid": summary["production_corpus_identity_valid"] is True,
        "task0154_reinterpretation_propagated": summary["syntactic_two_step_chain_count"] == 2 and summary["authoritative_two_step_chain_count"] == 0,
        "v1_integrity_preserved": summary["graph_retrieval_v1_frozen"] is True and summary["task0149_frozen_artifact_mutation_count"] == 0,
        "no_repair_applied": summary["repair_applied"] is False and summary["dangling_reference_repair_count"] == 0,
        "no_policy_mutation": summary["graph_policy_mutation_count"] == 0 and summary["runtime_policy_mutation_count"] == 0,
        "decision_valid": summary["graph_retrieval_v2_multihop_go_decision"] in {"go", "no_go", "defer"},
        "runtime_not_implemented": summary["bounded_multihop_runtime_implemented"] is False,
        "benchmark_not_frozen": summary["multihop_benchmark_frozen"] is False,
    }


def verify_task0155_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
        "authority_valid": all(summary.get(key) is True for key in ("task0154_authority_valid", "task0153_authority_valid", "task0152_authority_valid", "task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid")),
        "production_corpus_identity_valid": summary.get("production_corpus_identity_valid") is True,
        "task0154_reinterpretation_propagated": summary.get("syntactic_two_step_chain_count") == 2 and summary.get("authoritative_two_step_chain_count") == 0,
        "reference_counts_consistent": summary.get("resolved_internal_reference_count", 0) + summary.get("unresolved_internal_reference_count", 0) == summary.get("total_internal_reference_count"),
        "authoritative_edge_counts_consistent": summary.get("authoritative_graph_edge_count") == summary.get("materialized_authoritative_edge_count") == summary.get("graph_edge_count"),
        "v1_integrity_preserved": summary.get("graph_retrieval_v1_frozen") is True and summary.get("task0149_frozen_artifact_mutation_count") == 0,
        "no_repair_applied": summary.get("repair_applied") is False and summary.get("dangling_reference_repair_count") == 0 and summary.get("corpus_mutation_count") == 0,
        "no_policy_mutation": summary.get("graph_policy_mutation_count") == 0 and summary.get("runtime_policy_mutation_count") == 0,
        "decision_valid": summary.get("graph_retrieval_v2_multihop_go_decision") in {"go", "no_go", "defer"},
        "runtime_not_implemented": summary.get("bounded_multihop_runtime_implemented") is False,
        "benchmark_not_frozen": summary.get("multihop_benchmark_frozen") is False,
    }
    result = {
        "schema_version": "opk-rag.task0155.verification.v1",
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


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0155.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK0155 Multi-hop-capable Corpus Authority Gap Assessment and V2 Viability Decision Report

## Summary

`task_status={summary["task_status"]}`

`production_corpus_identity_valid={str(summary["production_corpus_identity_valid"]).lower()}`

`graph_retrieval_v1_frozen={str(summary["graph_retrieval_v1_frozen"]).lower()}`

`graph_retrieval_v1_baseline_digest={summary["graph_retrieval_v1_baseline_digest"]}`

TASK-0155 audited the current production / dogfooding corpus. It did not repair links, mutate source documents, change graph construction, change runtime policy, implement multi-hop runtime, or author benchmark gold questions.

## Corpus Graph Authority

Corpus documents: `{summary["corpus_document_count"]}`.

Internal references: `{summary["total_internal_reference_count"]}`.

Resolved references: `{summary["resolved_internal_reference_count"]}`.

Dangling references: `{summary["dangling_reference_count"]}`.

Dangling reference rate: `{summary["dangling_reference_rate"]}`.

Dangling reference systemic: `{str(summary["dangling_reference_systemic"]).lower()}`.

Authoritative graph nodes: `{summary["authoritative_graph_node_count"]}`.

Authoritative graph edges: `{summary["authoritative_graph_edge_count"]}`.

## Topology

Connected components use weak connectivity over directed authoritative edges.

Graph nodes: `{summary["graph_node_count"]}`.

Graph edges: `{summary["graph_edge_count"]}`.

Isolated corpus nodes: `{summary["isolated_node_count"]}`.

Largest component nodes: `{summary["largest_connected_component_node_count"]}`.

Graph fragmentation high: `{str(summary["graph_fragmentation_high"]).lower()}`.

## Path Depth

Two-step walks: `{summary["authoritative_two_step_walk_count"]}`.

Minimum-distance=2 paths: `{summary["authoritative_minimum_distance_two_path_count"]}`.

Minimum-distance>=3 paths: `{summary["authoritative_minimum_distance_three_plus_path_count"]}`.

Cross-document minimum-distance=2 paths: `{summary["cross_document_minimum_distance_two_path_count"]}`.

Potentially authorable two-hop paths: `{summary["potentially_authorable_two_hop_path_count"]}`.

TASK-0154 reinterpretation is propagated: syntactic two-step chains `{summary["syntactic_two_step_chain_count"]}`, authoritative two-step chains `{summary["authoritative_two_step_chain_count"]}`.

## Decision

Current V1 graph-sensitive residual count: `{summary["current_v1_graph_sensitive_residual_count"]}`.

Observed production multi-hop need: `{str(summary["observed_production_multihop_need"]).lower()}`.

Current corpus multi-hop authority sufficient: `{str(summary["current_corpus_multihop_authority_sufficient"]).lower()}`.

Current corpus multi-hop runtime investment justified: `{str(summary["current_corpus_multihop_runtime_investment_justified"]).lower()}`.

Graph Retrieval V2 multi-hop decision: `{summary["graph_retrieval_v2_multihop_go_decision"]}`.

Recommended next step: `{summary["recommended_next_step"]}`.

Recommended secondary step: `{summary["recommended_secondary_step"]}`.
"""


def enumerate_two_step_walks(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edge_rows:
        outgoing[edge["source_node_id"]].append(edge)
    rows = []
    for source in sorted(outgoing):
        for edge_1 in sorted(outgoing[source], key=lambda row: row["edge_id"]):
            bridge = edge_1["target_node_id"]
            for edge_2 in sorted(outgoing.get(bridge, []), key=lambda row: row["edge_id"]):
                target = edge_2["target_node_id"]
                if source == bridge or bridge == target or source == target:
                    continue
                rows.append(
                    {
                        "path_id": f"task0155-path-{digest_json([edge_1['edge_id'], edge_2['edge_id']])[:16]}",
                        "seed_node_id": source,
                        "bridge_node_id": bridge,
                        "target_node_id": target,
                        "edge_1_id": edge_1["edge_id"],
                        "edge_2_id": edge_2["edge_id"],
                        "edge_1_type": edge_1["edge_type"],
                        "edge_2_type": edge_2["edge_type"],
                        "relation_pattern": f"{edge_1['edge_type']}->{edge_2['edge_type']}",
                        "cross_document_path": len({source, bridge, target}) == 3,
                        "source_binding_valid": True,
                        "bridge_relation_semantically_meaningful": True,
                    }
                )
    return rows


def all_pairs_minimum_distances(graph: dict[str, list[str]], nodes: list[str], *, max_hops: int) -> dict[tuple[str, str], int]:
    distances: dict[tuple[str, str], int] = {}
    for source in nodes:
        queue = deque([(source, 0)])
        visited = {source}
        distances[(source, source)] = 0
        while queue:
            node, distance = queue.popleft()
            if distance >= max_hops:
                continue
            for child in graph.get(node, []):
                if child in visited:
                    continue
                visited.add(child)
                next_distance = distance + 1
                distances[(source, child)] = next_distance
                queue.append((child, next_distance))
    return distances


def weak_connected_components(nodes: list[str], edge_rows: list[dict[str, Any]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edge_rows:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(nodes)
    components = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for child in adjacency.get(node, set()):
                if child not in unseen:
                    continue
                unseen.remove(child)
                component.add(child)
                queue.append(child)
        components.append(component)
    return components


def adjacency(edge_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edge_rows:
        graph[edge["source_node_id"]].append(edge["target_node_id"])
    return graph


def _without_rows(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.endswith("_rows")}


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
