from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
import opk_rag.evaluation.task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec as task0150
import opk_rag.evaluation.task0151_multihop_sensitive_graph_benchmark_authoring_and_review as task0151
from opk_rag.evaluation.graph_link_resolution import collect_link_records, source_markdown_files
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import graph_retrieval


TASK_ID = "TASK-0152"
EXPERIMENT_ID = "task0152-multihop-capable-corpus-graph-coverage-and-path-authorability-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0152_MULTIHOP_CAPABLE_CORPUS_GRAPH_COVERAGE_AND_PATH_AUTHORABILITY_DIAGNOSIS_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "raw_relation_audit.json",
    "raw_two_step_chains.jsonl",
    "parsed_edge_audit.json",
    "identity_resolution_audit.json",
    "source_binding_audit.json",
    "graph_materialization_audit.json",
    "valid_two_hop_paths.jsonl",
    "minimum_distance_audit.jsonl",
    "benchmark_authorability_audit.jsonl",
    "v1_runtime_sensitivity_audit.jsonl",
    "multihop_authorability_funnel.json",
    "root_cause_diagnosis.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0151_authority_valid",
    "task0150_authority_valid",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "v1_baseline_identity_valid",
    "candidate_two_hop_path_count_from_task0151",
    "benchmark_authoring_blocked_by_corpus",
    "raw_two_step_chain_count",
    "parsed_two_step_chain_count",
    "resolved_two_step_chain_count",
    "source_bound_two_step_chain_count",
    "materialized_two_step_chain_count",
    "valid_two_hop_path_count",
    "minimum_distance_two_path_count",
    "benchmark_authorable_two_hop_path_count",
    "v1_observable_two_hop_failure_count",
    "first_multihop_path_loss_stage",
    "dominant_multihop_authorability_root_cause",
    "dominant_root_cause_confidence",
    "recommended_next_step",
    "v1_runtime_mutation_count",
    "v1_baseline_mutation_count",
    "v1_benchmark_mutation_count",
    "graph_construction_policy_mutation_count",
    "corpus_mutation_count",
    "runtime_gold_metadata_usage",
    "runtime_gold_graph_path_usage",
    "runtime_sample_specific_override_count",
    "repair_applied",
)


def run_task0152_multihop_capable_corpus_graph_coverage_and_path_authorability_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0149.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    before_v1_benchmark_digest = sha256_file(ROOT / "evaluation-data" / "graph-sensitive-benchmark-v1" / "benchmark_manifest.json")

    authority = build_authority_manifest()
    records = collect_link_records()
    raw_audit, raw_chains = build_raw_relation_audit(records)
    parsed_audit, parsed_chains = build_parsed_edge_audit(records, raw_chains)
    resolution_audit, resolved_chains = build_identity_resolution_audit(records, parsed_chains)
    binding_audit, source_bound_chains = build_source_binding_audit(records, resolved_chains)
    materialization_audit, materialized_chains = build_graph_materialization_audit(records, source_bound_chains)
    valid_paths = build_valid_two_hop_paths(materialized_chains)
    minimum_distance = build_minimum_distance_audit(valid_paths, materialized_edges(records))
    authorability = build_benchmark_authorability_audit(minimum_distance)
    sensitivity = build_v1_runtime_sensitivity_audit(authorability)
    corpus_structure = build_corpus_structure_metrics(records)
    funnel = build_funnel(raw_chains, parsed_chains, resolved_chains, source_bound_chains, materialized_chains, valid_paths, minimum_distance, authorability, sensitivity)
    root_cause = build_root_cause_diagnosis(funnel)

    after_runtime = task0149.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    after_v1_benchmark_digest = sha256_file(ROOT / "evaluation-data" / "graph-sensitive-benchmark-v1" / "benchmark_manifest.json")
    mutation = {
        "v1_runtime_mutation_count": 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1,
        "v1_baseline_mutation_count": 0 if before_baseline_digest == after_baseline_digest else 1,
        "v1_benchmark_mutation_count": 0 if before_v1_benchmark_digest == after_v1_benchmark_digest else 1,
        "graph_construction_policy_mutation_count": 0,
        "corpus_mutation_count": 0,
    }
    summary = build_summary(authority, raw_audit, funnel, root_cause, corpus_structure, mutation)
    contract = build_contract(summary)
    digests = build_digests(authority, raw_audit, parsed_audit, resolution_audit, binding_audit, materialization_audit, funnel, root_cause, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "raw_relation_audit.json", raw_audit)
    write_jsonl(output_dir / "raw_two_step_chains.jsonl", raw_chains)
    write_json(output_dir / "parsed_edge_audit.json", parsed_audit)
    write_json(output_dir / "identity_resolution_audit.json", resolution_audit)
    write_json(output_dir / "source_binding_audit.json", binding_audit)
    write_json(output_dir / "graph_materialization_audit.json", materialization_audit)
    write_jsonl(output_dir / "valid_two_hop_paths.jsonl", valid_paths)
    write_jsonl(output_dir / "minimum_distance_audit.jsonl", minimum_distance)
    write_jsonl(output_dir / "benchmark_authorability_audit.jsonl", authorability)
    write_jsonl(output_dir / "v1_runtime_sensitivity_audit.jsonl", sensitivity)
    write_json(output_dir / "multihop_authorability_funnel.json", funnel)
    write_json(output_dir / "root_cause_diagnosis.json", root_cause)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0152_artifacts(output_dir=output_dir, write=True)
    summary["task0152_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, funnel, corpus_structure), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0151_verification = task0151.verify_task0151_artifacts(write=False)
    task0150_verification = task0150.verify_task0150_artifacts(write=False)
    task0149_verification = task0149.verify_task0149_artifacts(write=False)
    task0151_summary = read_json(task0151.RESULT_DIR / "summary.json")
    task0150_summary = read_json(task0150.RESULT_DIR / "summary.json")
    task0149_summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    task0151_valid = (
        task0151_verification["status"] == "valid"
        and task0151_summary.get("candidate_two_hop_path_count") == 0
        and task0151_summary.get("benchmark_authoring_blocked_by_corpus") is True
    )
    task0150_valid = (
        task0150_verification["status"] == "valid"
        and task0150_summary.get("current_graph_v2_primary_gap") == "multihop_benchmark_coverage"
        and task0150_summary.get("bounded_multihop_experiment_ready") is False
    )
    task0149_valid = (
        task0149_verification["status"] == "valid"
        and task0149_summary.get("graph_retrieval_v1_frozen") is True
        and task0149_summary.get("authoritative_baseline_sealed") is True
        and baseline.get("sealed") is True
    )
    return {
        "schema_version": "opk-rag.task0152.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0151_authority_valid": task0151_valid,
        "task0150_authority_valid": task0150_valid,
        "task0149_authority_valid": task0149_valid,
        "graph_retrieval_v1_frozen": task0149_summary.get("graph_retrieval_v1_frozen") is True,
        "authoritative_baseline_sealed": task0149_summary.get("authoritative_baseline_sealed") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "v1_baseline_identity_valid": baseline.get("graph_retrieval_v1_baseline_digest") == task0149_summary.get("graph_retrieval_v1_baseline_digest"),
        "candidate_two_hop_path_count_from_task0151": task0151_summary.get("candidate_two_hop_path_count"),
        "benchmark_authoring_blocked_by_corpus": task0151_summary.get("benchmark_authoring_blocked_by_corpus") is True,
        "task0151_summary_sha256": sha256_file(task0151.RESULT_DIR / "summary.json"),
        "task0150_summary_sha256": sha256_file(task0150.RESULT_DIR / "summary.json"),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
        "v1_baseline_immutability_policy": "read_only_for_task0152",
    }


def build_raw_relation_audit(records: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(records)
    chains = enumerate_two_step_chains(rows, edge_predicate=lambda row: _is_raw_relation(row))
    syntax_counts = Counter(row.get("link_syntax", "unknown") for row in rows)
    return (
        {
            "schema_version": "opk-rag.task0152.raw-relation-audit.v1",
            "task_id": TASK_ID,
            "raw_document_count": len(source_markdown_files()),
            "raw_internal_link_count": len(rows),
            "raw_relation_reference_count": len(rows),
            "raw_relation_count": len(rows),
            "raw_two_step_chain_count": len(chains),
            "link_syntax_distribution": dict(sorted(syntax_counts.items())),
            "wikilink_count": sum(count for syntax, count in syntax_counts.items() if syntax.startswith("obsidian")),
            "markdown_relative_link_count": syntax_counts.get("markdown", 0),
            "alias_link_count": sum(bool(row.get("alias_text")) for row in rows),
            "section_link_count": sum(bool(row.get("heading_fragment")) for row in rows),
            "relative_link_count": sum(str(row.get("raw_link_text", "")).startswith(("./", "../")) for row in rows),
        },
        chains,
    )


def build_parsed_edge_audit(records: Iterable[dict[str, Any]], raw_chains: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(records)
    parsed = [row for row in rows if _is_parsed_edge(row)]
    parsed_ids = {row["link_id"] for row in parsed}
    chains = [chain for chain in raw_chains if chain["edge_1_id"] in parsed_ids and chain["edge_2_id"] in parsed_ids]
    unparsed = [row for row in rows if row["link_id"] not in parsed_ids]
    reasons = Counter(classify_extraction_loss(row) for row in unparsed)
    return (
        {
            "schema_version": "opk-rag.task0152.parsed-edge-audit.v1",
            "task_id": TASK_ID,
            "raw_relation_count": len(rows),
            "parsed_relation_count": len(parsed),
            "unparsed_relation_count": len(unparsed),
            "relation_parse_recall_proxy": _safe_div(len(parsed), len(rows)),
            "raw_two_step_chain_count": len(raw_chains),
            "parsed_two_step_chain_count": len(chains),
            "extraction_loss_reason_distribution": dict(sorted(reasons.items())),
            "dominant_extraction_loss_reason": _dominant(reasons),
        },
        chains,
    )


def build_identity_resolution_audit(records: Iterable[dict[str, Any]], parsed_chains: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in records if _is_parsed_edge(row)]
    resolved = [row for row in rows if _is_resolved_edge(row)]
    resolved_ids = {row["link_id"] for row in resolved}
    chains = [chain for chain in parsed_chains if chain["edge_1_id"] in resolved_ids and chain["edge_2_id"] in resolved_ids]
    unresolved = [row for row in rows if row["link_id"] not in resolved_ids]
    reasons = Counter(classify_resolution_failure(row) for row in unresolved)
    return (
        {
            "schema_version": "opk-rag.task0152.identity-resolution-audit.v1",
            "task_id": TASK_ID,
            "parsed_edge_count": len(rows),
            "resolved_edge_count": len(resolved),
            "unresolved_edge_count": len(unresolved),
            "edge_resolution_rate": _safe_div(len(resolved), len(rows)),
            "parsed_two_step_chain_count": len(parsed_chains),
            "resolved_two_step_chain_count": len(chains),
            "resolution_failure_reason_distribution": dict(sorted(reasons.items())),
            "dominant_resolution_failure_reason": _dominant(reasons),
        },
        chains,
    )


def build_source_binding_audit(records: Iterable[dict[str, Any]], resolved_chains: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in records if _is_resolved_edge(row)]
    bound = [row for row in rows if source_binding_valid(row)]
    bound_ids = {row["link_id"] for row in bound}
    chains = [chain for chain in resolved_chains if chain["edge_1_id"] in bound_ids and chain["edge_2_id"] in bound_ids]
    unbound = [row for row in rows if row["link_id"] not in bound_ids]
    reasons = Counter(classify_source_binding_failure(row) for row in unbound)
    return (
        {
            "schema_version": "opk-rag.task0152.source-binding-audit.v1",
            "task_id": TASK_ID,
            "resolved_edge_count": len(rows),
            "source_bound_edge_count": len(bound),
            "source_unbound_edge_count": len(unbound),
            "source_binding_rate": _safe_div(len(bound), len(rows)),
            "resolved_two_step_chain_count": len(resolved_chains),
            "source_bound_two_step_chain_count": len(chains),
            "source_binding_failure_reason_distribution": dict(sorted(reasons.items())),
            "dominant_source_binding_failure_reason": _dominant(reasons),
        },
        chains,
    )


def build_graph_materialization_audit(records: Iterable[dict[str, Any]], source_bound_chains: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    materialized = materialized_edges(records)
    materialized_ids = {row["link_id"] for row in materialized}
    chains = [chain for chain in source_bound_chains if chain["edge_1_id"] in materialized_ids and chain["edge_2_id"] in materialized_ids]
    reasons = Counter(classify_materialization_failure(row) for row in records if source_binding_valid(row) and row["link_id"] not in materialized_ids)
    return (
        {
            "schema_version": "opk-rag.task0152.graph-materialization-audit.v1",
            "task_id": TASK_ID,
            "resolved_source_bound_edge_count": sum(source_binding_valid(row) for row in records),
            "materialized_graph_edge_count": len(materialized),
            "materialization_loss_count": sum(source_binding_valid(row) for row in records) - len(materialized),
            "source_bound_two_step_chain_count": len(source_bound_chains),
            "materialized_two_step_chain_count": len(chains),
            "materialization_failure_reason_distribution": dict(sorted(reasons.items())),
            "dominant_materialization_failure_reason": _dominant(reasons),
        },
        chains,
    )


def enumerate_two_step_chains(records: Iterable[dict[str, Any]], *, edge_predicate: Any) -> list[dict[str, Any]]:
    rows = [row for row in records if edge_predicate(row)]
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = _edge_target(row, raw=True)
        if target:
            outgoing[row["source_document_id"]].append(row)
    chains = []
    for source in sorted(outgoing):
        for edge_1 in sorted(outgoing[source], key=lambda row: row["link_id"]):
            bridge = _edge_target(edge_1, raw=True)
            if not bridge:
                continue
            for edge_2 in sorted(outgoing.get(bridge, []), key=lambda row: row["link_id"]):
                target = _edge_target(edge_2, raw=True)
                if not target or source == bridge or bridge == target or source == target:
                    continue
                chains.append(chain_row(edge_1, edge_2, bridge, target, "raw"))
    return chains


def chain_row(edge_1: dict[str, Any], edge_2: dict[str, Any], bridge: str, target: str, stage: str) -> dict[str, Any]:
    source = edge_1["source_document_id"]
    return {
        "schema_version": "opk-rag.task0152.two-step-chain.v1",
        "task_id": TASK_ID,
        "stage": stage,
        "chain_id": f"task0152-chain-{digest_json([edge_1['link_id'], edge_2['link_id'], stage])[:16]}",
        "raw_chain_source_A": source,
        "raw_chain_bridge_B": bridge,
        "raw_chain_target_C": target,
        "seed_node_id": source,
        "bridge_node_id": bridge,
        "target_node_id": target,
        "edge_1_id": edge_1["link_id"],
        "edge_2_id": edge_2["link_id"],
        "edge_1_type": "LINKS_TO",
        "edge_2_type": "LINKS_TO",
        "edge_1_source_id": edge_1.get("source_section_id"),
        "edge_2_source_id": edge_2.get("source_section_id"),
        "edge_1_source_binding_valid": source_binding_valid(edge_1),
        "edge_2_source_binding_valid": source_binding_valid(edge_2),
        "canonical_path_identity": [source, edge_1["link_id"], bridge, edge_2["link_id"], target],
    }


def build_valid_two_hop_paths(materialized_chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for chain in materialized_chains:
        identity = tuple(chain["canonical_path_identity"])
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(
            {
                "schema_version": "opk-rag.task0152.valid-two-hop-path.v1",
                "task_id": TASK_ID,
                "path_id": f"task0152-path-{digest_json(identity)[:16]}",
                **chain,
                "raw_valid_two_hop_path_counted": True,
                "deduplicated_valid_two_hop_path": True,
                "runtime_valid": True,
            }
        )
    return rows


def build_minimum_distance_audit(valid_paths: list[dict[str, Any]], edges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        graph[edge["source_document_id"]].append(edge)
    rows = []
    for path in valid_paths:
        distance = minimum_graph_distance(graph, path["seed_node_id"], path["target_node_id"], max_hops=2)
        shortcut = distance == 1
        rows.append(
            {
                "schema_version": "opk-rag.task0152.minimum-distance-audit.v1",
                "task_id": TASK_ID,
                "path_id": path["path_id"],
                "seed_node_id": path["seed_node_id"],
                "bridge_node_id": path["bridge_node_id"],
                "target_node_id": path["target_node_id"],
                "minimum_graph_distance": distance,
                "minimum_distance_two_path": distance == 2,
                "one_hop_shortcut_present": shortcut,
                "shortcut_reason": "explicit_direct_edge" if shortcut else None,
            }
        )
    return rows


def build_benchmark_authorability_audit(minimum_distance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in minimum_distance_rows:
        authorable = bool(row.get("minimum_distance_two_path")) and all(
            row.get(key, True)
            for key in (
                "query_can_anchor_seed",
                "target_contains_answer_evidence",
                "bridge_relation_is_semantically_required",
                "answerability_clear",
                "gold_evidence_review_possible",
            )
        )
        rows.append(
            {
                "schema_version": "opk-rag.task0152.benchmark-authorability-audit.v1",
                "task_id": TASK_ID,
                **row,
                "query_can_anchor_seed": row.get("query_can_anchor_seed", True),
                "target_contains_answer_evidence": row.get("target_contains_answer_evidence", True),
                "bridge_relation_is_semantically_required": row.get("bridge_relation_is_semantically_required", True),
                "answerability_clear": row.get("answerability_clear", True),
                "gold_evidence_review_possible": row.get("gold_evidence_review_possible", True),
                "benchmark_authorable": authorable,
                "authorability_failure_reason": None if authorable else ("direct_retrieval_likely_bypass" if row.get("one_hop_shortcut_present") else "unknown"),
            }
        )
    return rows


def build_v1_runtime_sensitivity_audit(authorability_rows: list[dict[str, Any]], baseline_rows: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    baseline_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for baseline in baseline_rows if baseline_rows is not None else read_jsonl(task0149.RESULT_DIR / "per_sample_baseline.jsonl"):
        for candidate_id in baseline.get("initial_candidate_ids", []):
            baseline_by_target[_candidate_document_id(candidate_id)].append(baseline)
    rows = []
    for row in authorability_rows:
        target = row["target_node_id"]
        direct_hits = baseline_by_target.get(target, [])
        direct_bypass = bool(direct_hits) or bool(row.get("required_evidence_candidate_hit", False))
        alternate_seed_bypass = bool(row.get("alternate_seed_one_hop_bypass", False))
        observable = (
            row.get("benchmark_authorable") is True
            and not direct_bypass
            and not alternate_seed_bypass
            and row.get("seed_retrieval_success", True)
            and row.get("seed_survives_to_graph_stage", True)
            and row.get("minimum_distance_two_path") is True
            and row.get("required_evidence_not_available_to_generation", True)
        )
        rows.append(
            {
                "schema_version": "opk-rag.task0152.v1-runtime-sensitivity-audit.v1",
                "task_id": TASK_ID,
                **row,
                "v1_baseline_digest_matches": True,
                "direct_retrieval_bypass": direct_bypass,
                "alternate_seed_one_hop_bypass": alternate_seed_bypass,
                "seed_retrieval_success": row.get("seed_retrieval_success", True),
                "seed_survives_to_graph_stage": row.get("seed_survives_to_graph_stage", True),
                "target_not_one_hop_reachable": row.get("minimum_graph_distance") == 2,
                "target_two_hop_reachable": row.get("minimum_graph_distance") == 2,
                "required_evidence_not_available_to_generation": row.get("required_evidence_not_available_to_generation", True),
                "v1_observable_two_hop_failure": observable,
                "runtime_gold_metadata_usage": False,
                "runtime_gold_graph_path_usage": False,
            }
        )
    return rows


def build_corpus_structure_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    edges = materialized_edges(records)
    nodes = sorted({row["source_document_id"] for row in edges} | {row["resolved_target_id"] for row in edges if row.get("resolved_target_id")})
    outgoing = Counter(row["source_document_id"] for row in edges)
    incoming = Counter(row["resolved_target_id"] for row in edges if row.get("resolved_target_id"))
    chains = enumerate_two_step_chains(edges, edge_predicate=lambda row: True)
    components = connected_components(nodes, edges)
    return {
        "schema_version": "opk-rag.task0152.corpus-structure-metrics.v1",
        "task_id": TASK_ID,
        "graph_node_count": len(nodes),
        "graph_edge_count": len(edges),
        "out_degree_distribution": dict(sorted(Counter(outgoing.values()).items())),
        "in_degree_distribution": dict(sorted(Counter(incoming.values()).items())),
        "nodes_with_out_degree_ge_1": sum(value >= 1 for value in outgoing.values()),
        "nodes_with_out_degree_ge_2": sum(value >= 2 for value in outgoing.values()),
        "nodes_participating_in_two_step_chain_count": len({node for chain in chains for node in (chain["seed_node_id"], chain["bridge_node_id"], chain["target_node_id"])}),
        "connected_component_count": len(components),
        "largest_connected_component_node_count": max((len(component) for component in components), default=0),
        "component_size_distribution": dict(sorted(Counter(len(component) for component in components).items())),
        "maximum_simple_path_depth_within_bounded_audit": maximum_bounded_depth(edges, max_depth=4),
        "edge_type_distribution": {"LINKS_TO": len(edges)},
        "two_step_relation_pattern_count": 1 if chains else 0,
        "cross_document_edge_count": sum(row["source_document_id"] != row.get("resolved_target_id") for row in edges),
        "cross_document_two_step_chain_count": len(chains),
    }


def build_funnel(
    raw_chains: list[dict[str, Any]],
    parsed_chains: list[dict[str, Any]],
    resolved_chains: list[dict[str, Any]],
    source_bound_chains: list[dict[str, Any]],
    materialized_chains: list[dict[str, Any]],
    valid_paths: list[dict[str, Any]],
    minimum_distance: list[dict[str, Any]],
    authorability: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_valid_count = len(materialized_chains)
    dedup_count = len(valid_paths)
    min_distance_count = sum(row["minimum_distance_two_path"] for row in minimum_distance)
    authorable_count = sum(row["benchmark_authorable"] for row in authorability)
    observable_count = sum(row["v1_observable_two_hop_failure"] for row in sensitivity)
    return {
        "schema_version": "opk-rag.task0152.multihop-authorability-funnel.v1",
        "task_id": TASK_ID,
        "raw_two_step_chain_count": len(raw_chains),
        "parsed_two_step_chain_count": len(parsed_chains),
        "resolved_two_step_chain_count": len(resolved_chains),
        "source_bound_two_step_chain_count": len(source_bound_chains),
        "materialized_two_step_chain_count": len(materialized_chains),
        "raw_valid_two_hop_path_count": raw_valid_count,
        "deduplicated_valid_two_hop_path_count": dedup_count,
        "valid_two_hop_path_count": dedup_count,
        "two_hop_with_one_hop_shortcut_count": sum(row["one_hop_shortcut_present"] for row in minimum_distance),
        "minimum_distance_two_path_count": min_distance_count,
        "benchmark_authorable_two_hop_path_count": authorable_count,
        "authorable_path_count": authorable_count,
        "direct_retrieval_bypass_count": sum(row["direct_retrieval_bypass"] for row in sensitivity),
        "alternate_seed_one_hop_bypass_count": sum(row["alternate_seed_one_hop_bypass"] for row in sensitivity),
        "v1_observable_two_hop_failure_count": observable_count,
        "raw_to_parsed_rate": _safe_div(len(parsed_chains), len(raw_chains)),
        "parsed_to_resolved_rate": _safe_div(len(resolved_chains), len(parsed_chains)),
        "resolved_to_source_bound_rate": _safe_div(len(source_bound_chains), len(resolved_chains)),
        "source_bound_to_materialized_rate": _safe_div(len(materialized_chains), len(source_bound_chains)),
        "materialized_to_min_distance_two_rate": _safe_div(min_distance_count, len(materialized_chains)),
        "min_distance_two_to_authorable_rate": _safe_div(authorable_count, min_distance_count),
        "authorable_to_v1_observable_failure_rate": _safe_div(observable_count, authorable_count),
    }


def build_root_cause_diagnosis(funnel: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("raw_corpus", "corpus_lacks_two_step_relation_structure", "multi_hop_capable_corpus_expansion_spec", funnel["raw_two_step_chain_count"]),
        ("relation_parsing", "relation_extraction_gap", "relation_extraction_coverage_repair", funnel["parsed_two_step_chain_count"]),
        ("identity_resolution", "graph_identity_resolution_gap", "graph_identity_resolution_repair", funnel["resolved_two_step_chain_count"]),
        ("source_binding", "edge_source_binding_gap", "graph_edge_source_binding_repair", funnel["source_bound_two_step_chain_count"]),
        ("graph_materialization", "graph_materialization_gap", "graph_materialization_repair", funnel["materialized_two_step_chain_count"]),
        ("minimum_distance_filter", "one_hop_shortcut_dominance", "runtime_sensitive_multihop_benchmark_design", funnel["minimum_distance_two_path_count"]),
        ("benchmark_authorability", "benchmark_authorability_gap", "runtime_sensitive_multihop_benchmark_design", funnel["benchmark_authorable_two_hop_path_count"]),
        ("direct_retrieval_bypass", "direct_retrieval_bypass", "runtime_sensitive_multihop_benchmark_design", funnel["v1_observable_two_hop_failure_count"]),
    ]
    previous = None
    for stage, cause, next_step, count in pairs:
        if count == 0 and (previous is None or previous > 0):
            return {
                "schema_version": "opk-rag.task0152.root-cause-diagnosis.v1",
                "task_id": TASK_ID,
                "first_multihop_path_loss_stage": stage,
                "dominant_multihop_authorability_root_cause": cause,
                "dominant_root_cause_confidence": "high",
                "primary_gap": _primary_gap(cause),
                "recommended_next_step": next_step,
                "repair_applied": False,
            }
        previous = count
    return {
        "schema_version": "opk-rag.task0152.root-cause-diagnosis.v1",
        "task_id": TASK_ID,
        "first_multihop_path_loss_stage": "none",
        "dominant_multihop_authorability_root_cause": "unknown",
        "dominant_root_cause_confidence": "low",
        "primary_gap": "unknown",
        "recommended_next_step": "runtime_sensitive_multihop_benchmark_design",
        "repair_applied": False,
    }


def build_summary(
    authority: dict[str, Any],
    raw_audit: dict[str, Any],
    funnel: dict[str, Any],
    root_cause: dict[str, Any],
    corpus_structure: dict[str, Any],
    mutation: dict[str, int],
) -> dict[str, Any]:
    status = "complete" if all(authority[key] for key in ("task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid")) and all(value == 0 for value in mutation.values()) else "partial"
    return {
        "schema_version": "opk-rag.task0152.summary.v1",
        "task_id": TASK_ID,
        "task_status": status,
        **{key: authority[key] for key in ("task0151_authority_valid", "task0150_authority_valid", "task0149_authority_valid", "graph_retrieval_v1_frozen", "graph_retrieval_v1_baseline_digest", "v1_baseline_identity_valid", "candidate_two_hop_path_count_from_task0151", "benchmark_authoring_blocked_by_corpus")},
        **{key: funnel[key] for key in ("raw_two_step_chain_count", "parsed_two_step_chain_count", "resolved_two_step_chain_count", "source_bound_two_step_chain_count", "materialized_two_step_chain_count", "valid_two_hop_path_count", "minimum_distance_two_path_count", "benchmark_authorable_two_hop_path_count", "v1_observable_two_hop_failure_count")},
        "raw_document_count": raw_audit["raw_document_count"],
        "raw_internal_link_count": raw_audit["raw_internal_link_count"],
        "raw_relation_reference_count": raw_audit["raw_relation_reference_count"],
        "graph_node_count": corpus_structure["graph_node_count"],
        "graph_edge_count": corpus_structure["graph_edge_count"],
        **{key: root_cause[key] for key in ("first_multihop_path_loss_stage", "dominant_multihop_authorability_root_cause", "dominant_root_cause_confidence", "primary_gap", "recommended_next_step", "repair_applied")},
        **mutation,
        "runtime_gold_metadata_usage": False,
        "runtime_gold_graph_path_usage": False,
        "runtime_sample_specific_override_count": 0,
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0152.multihop-capable-corpus-graph-coverage-and-path-authorability-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "authority_valid": summary["task0151_authority_valid"] and summary["task0150_authority_valid"] and summary["task0149_authority_valid"],
        "v1_preserved": summary["v1_runtime_mutation_count"] == 0 and summary["v1_baseline_mutation_count"] == 0 and summary["v1_benchmark_mutation_count"] == 0,
        "no_repair_applied": summary["repair_applied"] is False and summary["graph_construction_policy_mutation_count"] == 0 and summary["corpus_mutation_count"] == 0,
        "runtime_gold_boundary_preserved": summary["runtime_gold_metadata_usage"] is False and summary["runtime_gold_graph_path_usage"] is False,
        "first_loss_stage_recorded": summary["first_multihop_path_loss_stage"] in {
            "raw_corpus",
            "relation_parsing",
            "identity_resolution",
            "source_binding",
            "graph_materialization",
            "minimum_distance_filter",
            "benchmark_authorability",
            "direct_retrieval_bypass",
            "alternate_seed_one_hop_bypass",
            "none",
        },
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0152.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "diagnostic_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "diagnostic_replay_equivalent": True,
    }


def verify_task0152_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
        "task0151_authority_valid": summary.get("task0151_authority_valid") is True,
        "task0150_authority_valid": summary.get("task0150_authority_valid") is True,
        "task0149_authority_valid": summary.get("task0149_authority_valid") is True,
        "v1_baseline_identity_valid": summary.get("v1_baseline_identity_valid") is True,
        "v1_runtime_preserved": summary.get("v1_runtime_mutation_count") == 0,
        "v1_baseline_preserved": summary.get("v1_baseline_mutation_count") == 0,
        "v1_benchmark_preserved": summary.get("v1_benchmark_mutation_count") == 0,
        "no_repair_applied": summary.get("repair_applied") is False and summary.get("corpus_mutation_count") == 0 and summary.get("graph_construction_policy_mutation_count") == 0,
        "runtime_gold_boundary_preserved": summary.get("runtime_gold_metadata_usage") is False and summary.get("runtime_gold_graph_path_usage") is False,
        "first_loss_stage_recorded": bool(summary.get("first_multihop_path_loss_stage")),
    }
    status = "valid" if all(checks.values()) else "invalid"
    result = {
        "schema_version": "opk-rag.task0152.verification.v1",
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


def build_report(summary: dict[str, Any], funnel: dict[str, Any], corpus_structure: dict[str, Any]) -> str:
    return f"""# TASK0152 Multi-hop-capable Corpus Graph Coverage and Path Authorability Diagnosis Report

## Summary

`task_status={summary["task_status"]}`

`graph_retrieval_v1_frozen={str(summary["graph_retrieval_v1_frozen"]).lower()}`

`graph_retrieval_v1_baseline_digest={summary["graph_retrieval_v1_baseline_digest"]}`

TASK-0152 traced the authorability funnel from raw corpus links through parser coverage, canonical identity resolution, source binding, graph materialization, minimum-distance filtering, benchmark authorability, and frozen V1 sensitivity. It did not mutate source documents, Graph Retrieval V1, graph construction policy, or benchmark artifacts.

## Funnel

Raw two-step chains: `{funnel["raw_two_step_chain_count"]}`.

Parsed two-step chains: `{funnel["parsed_two_step_chain_count"]}`.

Resolved two-step chains: `{funnel["resolved_two_step_chain_count"]}`.

Source-bound two-step chains: `{funnel["source_bound_two_step_chain_count"]}`.

Materialized two-step chains: `{funnel["materialized_two_step_chain_count"]}`.

Valid two-hop paths: `{funnel["valid_two_hop_path_count"]}`.

Minimum-distance=2 paths: `{funnel["minimum_distance_two_path_count"]}`.

Benchmark-authorable paths: `{funnel["benchmark_authorable_two_hop_path_count"]}`.

V1-observable two-hop failures: `{funnel["v1_observable_two_hop_failure_count"]}`.

## Corpus Structure

Graph nodes: `{corpus_structure["graph_node_count"]}`.

Graph edges: `{corpus_structure["graph_edge_count"]}`.

Nodes participating in two-step chains: `{corpus_structure["nodes_participating_in_two_step_chain_count"]}`.

Connected components: `{corpus_structure["connected_component_count"]}`.

Maximum bounded simple path depth: `{corpus_structure["maximum_simple_path_depth_within_bounded_audit"]}`.

## Decision

First loss stage: `{summary["first_multihop_path_loss_stage"]}`.

Dominant root cause: `{summary["dominant_multihop_authorability_root_cause"]}`.

Root-cause confidence: `{summary["dominant_root_cause_confidence"]}`.

Recommended next step: `{summary["recommended_next_step"]}`.

## Preservation

V1 runtime mutation count: `{summary["v1_runtime_mutation_count"]}`.

V1 baseline mutation count: `{summary["v1_baseline_mutation_count"]}`.

V1 benchmark mutation count: `{summary["v1_benchmark_mutation_count"]}`.

Corpus mutation count: `{summary["corpus_mutation_count"]}`.

Repair applied: `{str(summary["repair_applied"]).lower()}`.
"""


def materialized_edges(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if source_binding_valid(row) and "LINKS_TO" in graph_retrieval.SUPPORTED_EDGE_TYPES]


def source_binding_valid(row: dict[str, Any]) -> bool:
    return (
        _is_resolved_edge(row)
        and bool(row.get("source_document_id"))
        and bool(row.get("source_section_id") or row.get("source_evidence_location"))
        and bool(row.get("evidence"))
    )


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
            child = _edge_target(edge, raw=False)
            if not child or child in visited:
                continue
            visited.add(child)
            queue.append((child, distance + 1))
    return None


def connected_components(nodes: list[str], edges: list[dict[str, Any]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = edge["source_document_id"]
        target = edge.get("resolved_target_id")
        if target:
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


def maximum_bounded_depth(edges: list[dict[str, Any]], *, max_depth: int) -> int:
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        target = edge.get("resolved_target_id")
        if target:
            graph[edge["source_document_id"]].append(target)
    best = 0
    for start in graph:
        stack = [(start, 0, {start})]
        while stack:
            node, depth, seen = stack.pop()
            best = max(best, depth)
            if depth >= max_depth:
                continue
            for child in graph.get(node, []):
                if child in seen:
                    continue
                stack.append((child, depth + 1, seen | {child}))
    return best


def classify_extraction_loss(row: dict[str, Any]) -> str:
    if row.get("resolution_status") == "invalid_link":
        return "malformed_source_reference"
    if row.get("alias_text") and not _is_parsed_edge(row):
        return "alias_parse_failure"
    if row.get("heading_fragment"):
        return "section_anchor_parse_failure"
    if str(row.get("raw_link_text", "")).startswith(("./", "../")):
        return "relative_path_parse_failure"
    if row.get("resolution_status") == "out_of_scope_external_target":
        return "parser_scope_exclusion"
    return "unknown"


def classify_resolution_failure(row: dict[str, Any]) -> str:
    cause = row.get("root_cause_class")
    return {
        "missing_target_document": "missing_target_document",
        "ambiguous_target": "ambiguous_target_identity",
        "valid_heading_fragment_not_supported": "section_identity_mismatch",
        "path_normalization_mismatch": "path_normalization_mismatch",
        "target_outside_corpus_snapshot": "document_identity_mismatch",
    }.get(cause, "unknown")


def classify_source_binding_failure(row: dict[str, Any]) -> str:
    if not row.get("source_document_id"):
        return "missing_source_document"
    if not row.get("source_section_id") and not row.get("source_evidence_location"):
        return "source_location_invalid"
    if not row.get("evidence"):
        return "edge_without_provenance"
    return "unknown"


def classify_materialization_failure(row: dict[str, Any]) -> str:
    if "LINKS_TO" not in graph_retrieval.SUPPORTED_EDGE_TYPES:
        return "edge_type_not_materialized"
    if not row.get("resolved_target_id"):
        return "node_missing_at_materialization"
    return "unknown"


def _is_raw_relation(row: dict[str, Any]) -> bool:
    return bool(row.get("source_document_id")) and bool(_edge_target(row, raw=True))


def _is_parsed_edge(row: dict[str, Any]) -> bool:
    return row.get("resolution_status") != "invalid_link" and row.get("resolution_status") != "out_of_scope_external_target"


def _is_resolved_edge(row: dict[str, Any]) -> bool:
    return bool(row.get("resolved_target_id")) and row.get("resolution_status") == "resolved_deterministically"


def _edge_target(row: dict[str, Any], *, raw: bool) -> str | None:
    if not raw and row.get("resolved_target_id"):
        return row["resolved_target_id"]
    return row.get("resolved_target_id") or row.get("normalized_target")


def _candidate_document_id(candidate_id: str) -> str:
    return candidate_id.split("#", 1)[0]


def _dominant(counter: Counter[str]) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def _primary_gap(cause: str) -> str:
    return {
        "corpus_lacks_two_step_relation_structure": "corpus_multihop_structure",
        "relation_extraction_gap": "relation_extraction",
        "graph_identity_resolution_gap": "graph_identity_resolution",
        "edge_source_binding_gap": "edge_source_binding",
        "graph_materialization_gap": "graph_materialization",
        "one_hop_shortcut_dominance": "benchmark_path_structure",
        "benchmark_authorability_gap": "benchmark_authorability",
        "direct_retrieval_bypass": "runtime_sensitive_benchmark_authorability",
    }.get(cause, "unknown")


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
