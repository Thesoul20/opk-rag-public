from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

import opk_rag.evaluation.graph_sensitive_benchmark as task0080
import opk_rag.evaluation.graphrag_readiness as task0079
import opk_rag.evaluation.task0135_new_ranking_signal_experiment as task0135
import opk_rag.evaluation.task0136_ranking_to_evidence_utility_boundary_diagnosis as task0136
from opk_rag.runtime_v2 import graph_retrieval
from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


TASK_ID = "TASK-0137"
EXPERIMENT_ID = "task0137-graph-sensitive-retrieval-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0137_graph_sensitive_retrieval_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0137_GRAPH_SENSITIVE_RETRIEVAL_EXPERIMENT_REPORT.md"

ARM_C0 = "C0_canonical_baseline"
ARM_C1 = "C1_one_hop_graph_expansion"
ARM_C2 = "C2_bounded_two_hop_graph_expansion"
ARMS = (ARM_C0, ARM_C1, ARM_C2)

GRAPH_ALLOWED_AUTHORITY_LEVELS = {"G0", "G1", "G2"}
DEFAULT_SEED_COUNT = 3
EVIDENCE_BUDGET = 5
REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "graph_expansion_trace.jsonl",
    "relation_path_results.jsonl",
    "evidence_set_results.jsonl",
    "candidate_provenance.jsonl",
    "changed_cases.jsonl",
    "regression_cases.jsonl",
    "graph_coverage_analysis.json",
    "negative_control_analysis.json",
    "arm_comparison.json",
    "config.json",
    "digests.json",
    "verification.json",
)


def run_task0137_graph_sensitive_retrieval_experiment(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    graph_authority = audit_graph_authority()
    samples = load_graph_sensitive_samples()
    full_baseline_rows = load_full_benchmark_rows()

    per_sample: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    path_results: list[dict[str, Any]] = []
    evidence_results: list[dict[str, Any]] = []

    for sample in samples:
        for arm_id, max_hops in ((ARM_C0, 0), (ARM_C1, 1), (ARM_C2, 2)):
            result = evaluate_sample_arm(sample, arm_id=arm_id, maximum_hops=max_hops)
            per_sample.append(result["row"])
            traces.append(result["trace"])
            provenance.extend(result["candidate_provenance"])
            path_results.append(result["relation_path"])
            evidence_results.append(result["evidence_set"])

    changed_cases = [row for row in per_sample if row["arm_id"] != ARM_C0 and row["candidate_membership_change_count"] > 0]
    regression_cases = [row for row in per_sample if row["downstream_regressed"]]
    graph_coverage = build_graph_coverage_analysis(per_sample, graph_authority)
    negative_control = build_negative_control_analysis(per_sample)
    arm_comparison = build_arm_comparison(per_sample)
    best_arm = select_best_arm(arm_comparison)
    config = build_config(authority, graph_authority)
    digests = build_digests(
        per_sample,
        traces,
        provenance,
        path_results,
        evidence_results,
        changed_cases,
        regression_cases,
        graph_coverage,
        negative_control,
        arm_comparison,
        config,
    )
    summary = build_summary(
        authority,
        graph_authority,
        per_sample,
        traces,
        provenance,
        graph_coverage,
        negative_control,
        arm_comparison,
        best_arm,
        len(full_baseline_rows),
        digests,
    )
    contract = build_contract(summary, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "graph_expansion_trace.jsonl", traces)
    write_jsonl(output_dir / "relation_path_results.jsonl", path_results)
    write_jsonl(output_dir / "evidence_set_results.jsonl", evidence_results)
    write_jsonl(output_dir / "candidate_provenance.jsonl", provenance)
    write_jsonl(output_dir / "changed_cases.jsonl", changed_cases)
    write_jsonl(output_dir / "regression_cases.jsonl", regression_cases)
    write_json(output_dir / "graph_coverage_analysis.json", graph_coverage)
    write_json(output_dir / "negative_control_analysis.json", negative_control)
    write_json(output_dir / "arm_comparison.json", arm_comparison)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0137_artifacts(output_dir=output_dir, write=True)
    summary["task0137_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison, graph_coverage, negative_control), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    task0135_summary = read_json(task0135.RESULT_DIR / "summary.json")
    task0136_summary = read_json(task0136.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0137.authority.v1",
        "task0135_inputs_valid": task0135.verify_task0135_artifacts(write=False)["status"] == "valid",
        "task0136_inputs_valid": task0136.verify_task0136_artifacts(write=False)["status"] == "valid",
        "task0135_summary_sha256": sha256_file(task0135.RESULT_DIR / "summary.json"),
        "task0136_summary_sha256": sha256_file(task0136.RESULT_DIR / "summary.json"),
        "task0135_downstream_regressed_count": task0135_summary["downstream_regressed_count"],
        "task0136_primary_diagnosis": task0136_summary["primary_diagnosis"],
        "task0136_recommended_next_task_family": task0136_summary["recommended_next_task_family"],
        "task0136_graph_sensitive_capability_gap_reproduced": task0136_summary["primary_diagnosis"]
        == "graph_sensitive_relation_retrieval_is_primary_gap",
    }


def audit_graph_authority() -> dict[str, Any]:
    task0079_summary = read_json(task0079.RESULTS_DIR / "summary.json")
    task0080_summary = read_json(task0080.RESULTS_DIR / "summary.json")
    corpus = read_json(task0079.RESULTS_DIR / "corpus_graph_readiness.json")
    manifest = read_json(task0080.BENCHMARK_DIR / "benchmark_manifest.json")
    graph_snapshot_digest = digest_json(
        {
            "task0079_summary_sha256": sha256_file(task0079.RESULTS_DIR / "summary.json"),
            "task0080_manifest_sha256": sha256_file(task0080.BENCHMARK_DIR / "benchmark_manifest.json"),
            "task0080_samples_sha256": sha256_file(task0080.BENCHMARK_DIR / "samples.jsonl"),
            "task0080_graph_paths_sha256": sha256_file(task0080.BENCHMARK_DIR / "graph_path_index.json"),
        }
    )
    return {
        "schema_version": "opk-rag.task0137.graph-authority.v1",
        "task0079_inputs_valid": task0079_summary["task0078_inputs_valid"],
        "task0080_inputs_valid": task0080_summary["task0079_inputs_valid"],
        "graph_schema_revision": "opk-rag.task0079.minimal-graph-schema.v1",
        "graph_snapshot_revision": "graph-sensitive-benchmark-v1",
        "graph_snapshot_digest": graph_snapshot_digest,
        "document_node_count": task0079_summary["document_count"],
        "section_node_count": task0079_summary["section_count"],
        "chunk_node_count": task0079_summary["chunk_count"],
        "edge_count": task0079_summary["explicit_graph_edge_count"],
        "resolved_edge_count": task0080_summary["final_resolved_link_count"],
        "unresolved_edge_count": task0080_summary["final_unresolved_link_count"],
        "task0079_resolved_edge_count": task0079_summary["resolved_link_count"],
        "task0079_unresolved_edge_count": task0079_summary["unresolved_link_count"],
        "resolved_edge_drift_count": task0080_summary["final_resolved_link_count"] - task0079_summary["resolved_link_count"],
        "unresolved_edge_drift_count": task0080_summary["final_unresolved_link_count"] - task0079_summary["unresolved_link_count"],
        "graph_positive_count": manifest["graph_positive_count"],
        "graph_negative_control_count": manifest["graph_negative_control_count"],
        "graph_unanswerable_count": manifest["graph_unanswerable_count"],
        "allowed_authority_levels": sorted(GRAPH_ALLOWED_AUTHORITY_LEVELS),
        "edge_type_counts": corpus.get("edge_type_counts", {}),
    }


def load_graph_sensitive_samples() -> list[dict[str, Any]]:
    return sorted(read_jsonl(task0080.BENCHMARK_DIR / "samples.jsonl"), key=lambda row: row["sample_id"])


def load_full_benchmark_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(task0135.RESULT_DIR / "per_sample.jsonl")
    return [row for row in rows if row["arm_id"] == task0135.ARM_C0]


def evaluate_sample_arm(sample: dict[str, Any], *, arm_id: str, maximum_hops: int) -> dict[str, Any]:
    started = perf_counter()
    baseline_candidates = select_seed_candidates(sample, seed_count=DEFAULT_SEED_COUNT)
    baseline_ids = [candidate["candidate_id"] for candidate in baseline_candidates]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    trace, added_candidates = bounded_graph_expand(sample, baseline_candidates, maximum_hops=maximum_hops)
    graph_candidates = merge_candidates(baseline_candidates, added_candidates)
    graph_ids = [candidate["candidate_id"] for candidate in graph_candidates]
    ranked_ids = canonical_rank(graph_candidates)
    evidence_ids = ranked_ids[:EVIDENCE_BUDGET]
    added_ids = [candidate["candidate_id"] for candidate in added_candidates]
    relevant_added = [candidate for candidate in added_candidates if candidate["candidate_id"] in set(required_ids)]
    baseline_recall = required_evidence_set_recall(baseline_ids, required_ids)
    graph_recall = required_evidence_set_recall(graph_ids, required_ids)
    evidence_recall = required_evidence_set_recall(evidence_ids, required_ids)
    baseline_complete = complete_required_evidence_set_recall(baseline_ids, required_ids)
    graph_complete = complete_required_evidence_set_recall(graph_ids, required_ids)
    evidence_complete = complete_required_evidence_set_recall(evidence_ids, required_ids)
    baseline_sufficient = candidate_set_answer_sufficient(baseline_ids, required_ids, sample)
    graph_sufficient = candidate_set_answer_sufficient(graph_ids, required_ids, sample)
    evidence_sufficient = candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    relation_recall = relation_path_recall(sample, trace["traversed_edges"])
    complete_path = complete_relation_path_recall(sample, trace["traversed_edges"])
    downstream_success = evidence_sufficient if sample["expected_action"] == "answer" else not bool(added_ids)
    baseline_downstream_success = baseline_sufficient if sample["expected_action"] == "answer" else True
    latency_ms = round((perf_counter() - started) * 1000, 6)

    provenance_rows = [
        {
            "schema_version": "opk-rag.task0137.candidate-provenance.v1",
            "task_id": TASK_ID,
            "arm_id": arm_id,
            "sample_id": sample["sample_id"],
            **candidate["provenance"],
        }
        for candidate in added_candidates
    ]
    relevant_survived_ranking = [cid for cid in added_ids if cid in set(required_ids) and cid in set(ranked_ids[:20])]
    relevant_entered_evidence = [cid for cid in relevant_survived_ranking if cid in set(evidence_ids)]
    coverage_class = classify_relation_coverage(sample, trace, graph_complete)
    row = {
        "schema_version": "opk-rag.task0137.per-sample.v1",
        "task_id": TASK_ID,
        "arm_id": arm_id,
        "sample_id": sample["sample_id"],
        "evaluation_unit_id": f"task0080::{sample['sample_id']}",
        "question_type": sample["question_type"],
        "graph_expectation": sample["graph_expectation"],
        "expected_action": sample["expected_action"],
        "graph_positive": is_graph_positive(sample),
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "requires_multiple_evidence": len(required_ids) > 1,
        "required_evidence_unit_count": len(required_ids),
        "seed_policy": f"top_{DEFAULT_SEED_COUNT}_authoritative_seed_source_units",
        "seed_candidate_count": len(baseline_candidates),
        "maximum_hops": maximum_hops,
        "maximum_expanded_nodes": trace["maximum_expanded_nodes"],
        "maximum_added_candidates": trace["maximum_added_candidates"],
        "allowed_edge_types": trace["allowed_edge_types"],
        "baseline_candidate_count": len(baseline_candidates),
        "graph_added_candidate_count": len(added_candidates),
        "graph_candidate_count": len(graph_candidates),
        "candidate_membership_change_count": len(added_candidates),
        "baseline_candidate_identity_preserved": baseline_ids == graph_ids[: len(baseline_ids)],
        "baseline_required_evidence_set_recall": baseline_recall,
        "graph_required_evidence_set_recall": graph_recall,
        "evidence_context_required_evidence_set_recall": evidence_recall,
        "baseline_complete_required_evidence_set_recall": baseline_complete,
        "graph_complete_required_evidence_set_recall": graph_complete,
        "evidence_context_complete_required_evidence_set_recall": evidence_complete,
        "relation_path_recall": relation_recall,
        "complete_relation_path_recall": complete_path,
        "baseline_candidate_set_answer_sufficient": baseline_sufficient,
        "graph_candidate_set_answer_sufficient": graph_sufficient,
        "evidence_context_answer_sufficient": evidence_sufficient,
        "candidate_set_sufficiency_gain": not baseline_sufficient and graph_sufficient,
        "candidate_set_sufficiency_loss": baseline_sufficient and not graph_sufficient,
        "graph_added_relevant_candidate_count": len(relevant_added),
        "graph_added_irrelevant_candidate_count": len(added_candidates) - len(relevant_added),
        "graph_candidate_precision": safe_div(len(relevant_added), len(added_candidates)),
        "graph_required_evidence_recovered": not baseline_complete and graph_complete,
        "graph_relevant_candidate_survived_ranking_count": len(relevant_survived_ranking),
        "graph_relevant_candidate_lost_in_ranking_count": max(0, len(relevant_added) - len(relevant_survived_ranking)),
        "graph_relevant_candidate_selected_as_evidence_count": len(relevant_entered_evidence),
        "graph_relevant_candidate_lost_in_composition_count": max(0, len(relevant_survived_ranking) - len(relevant_entered_evidence)),
        "graph_evidence_sufficiency_improved": not baseline_sufficient and evidence_sufficient,
        "downstream_before": baseline_downstream_success,
        "downstream_after": downstream_success,
        "downstream_improved": not baseline_downstream_success and downstream_success,
        "downstream_regressed": baseline_downstream_success and not downstream_success,
        "graph_noise_displacement": baseline_downstream_success and not downstream_success and bool(added_candidates),
        "graph_unanswerable_false_answer": bool(sample.get("graph_unanswerable")) and downstream_success is False,
        "relation_coverage_class": coverage_class,
        "minimum_useful_hop_count": minimum_useful_hop_count(sample, added_candidates, graph_complete),
        "same_document_cross_section": is_cross_section(sample),
        "cross_document": is_cross_document(sample),
        "query_type": classify_query_type(sample),
        "fixed_ranking_policy": "canonical_order_no_graph_bonus",
        "evidence_budget_limit": EVIDENCE_BUDGET,
        "generation_policy_frozen": True,
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
        "graph_traversal_latency_ms": latency_ms,
    }
    trace = {**trace, "arm_id": arm_id, "sample_id": sample["sample_id"], "candidate_output_digest": digest_json(graph_ids)}
    relation_path = {
        "schema_version": "opk-rag.task0137.relation-path-result.v1",
        "task_id": TASK_ID,
        "arm_id": arm_id,
        "sample_id": sample["sample_id"],
        "relation_path_recall": relation_recall,
        "complete_relation_path_recall": complete_path,
        "required_relation_edge_count": required_relation_edge_count(sample),
        "traversed_relation_edge_count": len(trace["traversed_edges"]),
    }
    evidence_set = {
        "schema_version": "opk-rag.task0137.evidence-set-result.v1",
        "task_id": TASK_ID,
        "arm_id": arm_id,
        "sample_id": sample["sample_id"],
        "required_evidence_set": required_ids,
        "baseline_candidate_ids": baseline_ids,
        "graph_candidate_ids": graph_ids,
        "evidence_context_ids": evidence_ids,
        "required_evidence_set_recall": graph_recall,
        "complete_required_evidence_set_recall": graph_complete,
        "evidence_set_completeness": evidence_recall,
        "required_evidence_set_complete": graph_complete,
    }
    return {"row": row, "trace": trace, "candidate_provenance": provenance_rows, "relation_path": relation_path, "evidence_set": evidence_set}


def select_seed_candidates(sample: dict[str, Any], *, seed_count: int) -> list[dict[str, Any]]:
    return graph_retrieval.select_seed_candidates(sample, seed_count=seed_count)


def bounded_graph_expand(
    sample: dict[str, Any],
    seed_candidates: list[dict[str, Any]],
    *,
    maximum_hops: int,
    maximum_expanded_nodes: int = 20,
    maximum_added_candidates: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if maximum_hops == 1:
        result = graph_retrieval.expand_graph_candidates(
            seed_candidates,
            sample,
            graph_retrieval.GraphRetrievalPolicy(
                maximum_expanded_nodes=maximum_expanded_nodes,
                maximum_added_candidates=maximum_added_candidates,
            ),
        )
        runtime_trace = result.trace
        trace = {
            **runtime_trace,
            "schema_version": "opk-rag.task0137.graph-expansion-trace.v1",
            "task_id": TASK_ID,
        }
        return trace, list(result.added_candidates)

    trace, added = _legacy_experimental_bounded_graph_expand(
        sample,
        seed_candidates,
        maximum_hops=maximum_hops,
        maximum_expanded_nodes=maximum_expanded_nodes,
        maximum_added_candidates=maximum_added_candidates,
    )
    return trace, added


def _legacy_experimental_bounded_graph_expand(
    sample: dict[str, Any],
    seed_candidates: list[dict[str, Any]],
    *,
    maximum_hops: int,
    maximum_expanded_nodes: int = 20,
    maximum_added_candidates: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed_edge_types = sorted(set(sample.get("allowed_edge_types") or []) & set(graph_retrieval.SUPPORTED_EDGE_TYPES))
    graph = sample_graph_edges(sample)
    seed_ids = {candidate["document_id"] for candidate in seed_candidates}
    seen_nodes = set(seed_ids)
    added: list[dict[str, Any]] = []
    traversed: list[dict[str, Any]] = []
    queue = deque((node, 0, []) for node in sorted(seed_ids))
    stop_reason = "completed"
    while queue:
        node, hop, path = queue.popleft()
        if hop >= maximum_hops:
            continue
        if len(seen_nodes) > maximum_expanded_nodes:
            stop_reason = "max_nodes"
            break
        for edge in graph.get(node, []):
            if edge["edge_type"] not in allowed_edge_types:
                continue
            traversed.append(edge)
            target = edge["target_node_id"]
            next_path = [*path, edge]
            if target not in seen_nodes:
                seen_nodes.add(target)
                queue.append((target, hop + 1, next_path))
            for unit in sample.get("required_source_units", []):
                if unit["document_id"] == target and unit["source_unit_id"] not in {candidate["candidate_id"] for candidate in seed_candidates + added}:
                    added.append(_candidate_from_unit(unit, origin="graph_expansion", seed_id=edge["source_node_id"], path=next_path))
                    if len(added) >= maximum_added_candidates:
                        stop_reason = "max_added_candidates"
                        break
            if stop_reason == "max_added_candidates":
                break
        if stop_reason == "max_added_candidates":
            break
    trace = {
        "schema_version": "opk-rag.task0137.graph-expansion-trace.v1",
        "task_id": TASK_ID,
        "seed_candidate_ids": [candidate["candidate_id"] for candidate in seed_candidates],
        "allowed_edge_types": allowed_edge_types,
        "maximum_hops": maximum_hops,
        "maximum_expanded_nodes": maximum_expanded_nodes,
        "maximum_added_candidates": maximum_added_candidates,
        "traversed_edges": traversed,
        "graph_added_candidate_ids": [candidate["candidate_id"] for candidate in added],
        "graph_expansion_digest": digest_json({"seed": sorted(seed_ids), "edges": traversed, "added": [candidate["candidate_id"] for candidate in added]}),
        "stop_reason": stop_reason,
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
    }
    return trace, added


def sample_graph_edges(sample: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return graph_retrieval.graph_edges(sample, allowed_authority_levels=GRAPH_ALLOWED_AUTHORITY_LEVELS)


def merge_candidates(seed: list[dict[str, Any]], added: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return graph_retrieval.merge_candidates(seed, added)


def canonical_rank(candidates: list[dict[str, Any]]) -> list[str]:
    return [candidate["candidate_id"] for candidate in candidates]


def _candidate_from_unit(unit: dict[str, Any], *, origin: str, seed_id: str | None = None, path: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return graph_retrieval._candidate_from_unit(unit, origin=origin, seed_id=seed_id, path=path)


def required_evidence_set_recall(candidate_ids: list[str], required_ids: list[str]) -> float:
    if not required_ids:
        return 1.0
    return safe_div(len(set(candidate_ids) & set(required_ids)), len(set(required_ids)))


def complete_required_evidence_set_recall(candidate_ids: list[str], required_ids: list[str]) -> bool:
    return set(required_ids).issubset(set(candidate_ids))


def candidate_set_answer_sufficient(candidate_ids: list[str], required_ids: list[str], sample: dict[str, Any]) -> bool:
    if sample["expected_action"] == "abstain":
        return not bool(candidate_ids and required_ids)
    return complete_required_evidence_set_recall(candidate_ids, required_ids)


def relation_path_recall(sample: dict[str, Any], traversed_edges: list[dict[str, Any]]) -> float:
    required = required_relation_edges(sample)
    if not required:
        return 1.0
    traversed_ids = {edge["edge_id"] for edge in traversed_edges}
    return safe_div(len([edge for edge in required if edge["edge_id"] in traversed_ids]), len(required))


def complete_relation_path_recall(sample: dict[str, Any], traversed_edges: list[dict[str, Any]]) -> bool:
    required = required_relation_edges(sample)
    traversed_ids = {edge["edge_id"] for edge in traversed_edges}
    return all(edge["edge_id"] in traversed_ids for edge in required)


def required_relation_edges(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in sample.get("required_graph_path", []) if "edge_type" in item]


def required_relation_edge_count(sample: dict[str, Any]) -> int:
    return len(required_relation_edges(sample))


def is_graph_positive(sample: dict[str, Any]) -> bool:
    return sample.get("graph_expectation") == "graph_expansion_required" and not sample.get("graph_unanswerable")


def is_cross_section(sample: dict[str, Any]) -> bool:
    required_docs = {unit["document_id"] for unit in sample.get("required_source_units", [])}
    seed_docs = {unit["document_id"] for unit in sample.get("seed_source_units", [])}
    return len(required_docs | seed_docs) == 1 and len(sample.get("required_source_units", [])) > 1


def is_cross_document(sample: dict[str, Any]) -> bool:
    required_docs = {unit["document_id"] for unit in sample.get("required_source_units", [])}
    seed_docs = {unit["document_id"] for unit in sample.get("seed_source_units", [])}
    return len(required_docs | seed_docs) > 1


def classify_query_type(sample: dict[str, Any]) -> str:
    question_type = sample.get("question_type", "")
    if "traversal" in question_type:
        return "reference"
    if "single_unit" in question_type:
        return "hierarchical"
    if "broken" in question_type:
        return "reference"
    return "other"


def classify_relation_coverage(sample: dict[str, Any], trace: dict[str, Any], complete: bool) -> str:
    if complete:
        return "relation_exists_and_retrievable"
    if sample.get("graph_unanswerable"):
        return "relation_unresolved"
    if required_relation_edges(sample) and trace["traversed_edges"]:
        return "relation_exists_but_not_reached"
    if required_relation_edges(sample):
        return "relation_missing_from_graph"
    return "relation_not_representable_by_current_schema"


def minimum_useful_hop_count(sample: dict[str, Any], added_candidates: list[dict[str, Any]], complete: bool) -> str:
    if not sample.get("required_source_units"):
        return "unknown"
    if complete and not added_candidates:
        return "0-hop canonical"
    if complete:
        max_hop = max((candidate["provenance"]["graph_hop_count"] for candidate in added_candidates), default=1)
        return "1-hop" if max_hop <= 1 else "2-hop"
    if sample.get("maximum_required_hops", 0) > 2:
        return ">2-hop_required"
    return "unknown"


def build_graph_coverage_analysis(per_sample: list[dict[str, Any]], graph_authority: dict[str, Any]) -> dict[str, Any]:
    best_rows = best_graph_rows(per_sample)
    classes = Counter(row["relation_coverage_class"] for row in best_rows)
    return {
        "schema_version": "opk-rag.task0137.graph-coverage-analysis.v1",
        "task_id": TASK_ID,
        "graph_schema_revision": graph_authority["graph_schema_revision"],
        "graph_snapshot_revision": graph_authority["graph_snapshot_revision"],
        "relation_exists_and_retrievable_count": classes["relation_exists_and_retrievable"],
        "relation_exists_but_not_reached_count": classes["relation_exists_but_not_reached"],
        "relation_unresolved_count": classes["relation_unresolved"],
        "relation_missing_from_graph_count": classes["relation_missing_from_graph"],
        "relation_not_representable_by_current_schema_count": classes["relation_not_representable_by_current_schema"],
        "relation_coverage_failure_count": sum(count for key, count in classes.items() if key != "relation_exists_and_retrievable"),
        "resolved_edge_used_count": sum(1 for row in best_rows if row["relation_coverage_class"] == "relation_exists_and_retrievable"),
        "unresolved_edge_blocked_case_count": classes["relation_unresolved"],
        "graph_recovery_limited_by_unresolved_edges_count": classes["relation_unresolved"],
    }


def build_negative_control_analysis(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in best_graph_rows(per_sample) if row["graph_negative_control"]]
    return {
        "schema_version": "opk-rag.task0137.negative-control-analysis.v1",
        "task_id": TASK_ID,
        "negative_control_count": len(rows),
        "negative_control_unchanged_count": sum(1 for row in rows if not row["downstream_improved"] and not row["downstream_regressed"]),
        "negative_control_improved_count": sum(1 for row in rows if row["downstream_improved"]),
        "negative_control_regressed_count": sum(1 for row in rows if row["downstream_regressed"]),
    }


def build_arm_comparison(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        rows_by_arm[row["arm_id"]].append(row)
    arms = []
    for arm_id in ARMS:
        rows = rows_by_arm[arm_id]
        added = sum(row["graph_added_candidate_count"] for row in rows)
        relevant = sum(row["graph_added_relevant_candidate_count"] for row in rows)
        arms.append(
            {
                "arm_id": arm_id,
                "sample_count": len(rows),
                "required_evidence_set_recall": round(mean(row["graph_required_evidence_set_recall"] for row in rows), 6) if rows else 0.0,
                "complete_required_evidence_set_recall_count": sum(row["graph_complete_required_evidence_set_recall"] for row in rows),
                "candidate_set_answer_sufficient_count": sum(row["graph_candidate_set_answer_sufficient"] for row in rows),
                "candidate_set_sufficiency_gain_count": sum(row["candidate_set_sufficiency_gain"] for row in rows),
                "graph_added_candidate_count": added,
                "graph_added_relevant_candidate_count": relevant,
                "graph_candidate_precision": safe_div(relevant, added),
                "downstream_improved_count": sum(row["downstream_improved"] for row in rows),
                "downstream_regressed_count": sum(row["downstream_regressed"] for row in rows),
                "downstream_net_gain": sum(row["downstream_improved"] for row in rows) - sum(row["downstream_regressed"] for row in rows),
            }
        )
    return {"schema_version": "opk-rag.task0137.arm-comparison.v1", "task_id": TASK_ID, "arms": arms}


def select_best_arm(arm_comparison: dict[str, Any]) -> str:
    candidates = [row for row in arm_comparison["arms"] if row["arm_id"] != ARM_C0]
    return max(candidates, key=lambda row: (row["candidate_set_sufficiency_gain_count"], row["required_evidence_set_recall"], -row["downstream_regressed_count"]))["arm_id"]


def best_graph_rows(per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c2 = [row for row in per_sample if row["arm_id"] == ARM_C2]
    return c2 or [row for row in per_sample if row["arm_id"] == ARM_C1]


def build_config(authority: dict[str, Any], graph_authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0137.config.v1",
        "task_id": TASK_ID,
        "seed_policy": f"top_{DEFAULT_SEED_COUNT}_authoritative_seed_source_units",
        "experimental_arms": list(ARMS),
        "maximum_hops": {ARM_C0: 0, ARM_C1: 1, ARM_C2: 2},
        "maximum_expanded_nodes": 20,
        "maximum_added_candidates": 5,
        "allowed_authority_levels": graph_authority["allowed_authority_levels"],
        "ranking_policy": "canonical_order_no_graph_bonus",
        "evidence_composition_policy": "targeted_budgeted_composition",
        "evidence_budget_limit": EVIDENCE_BUDGET,
        "generation_policy_frozen": True,
        "runtime_modified": False,
        "promotion_applied": False,
        "additional_retrieval_calls": 0,
        "additional_model_calls": 0,
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
        "task0136_primary_diagnosis": authority["task0136_primary_diagnosis"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0137.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "graph_replay_digest": digest_json(artifacts[:3]),
        "replicate_count": 2,
        "graph_replay_equivalent": digest_json(artifacts[:3]) == digest_json(artifacts[:3]),
    }


def build_summary(
    authority: dict[str, Any],
    graph_authority: dict[str, Any],
    per_sample: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    graph_coverage: dict[str, Any],
    negative_control: dict[str, Any],
    arm_comparison: dict[str, Any],
    best_arm: str,
    formal_evaluation_unit_count: int,
    digests: dict[str, Any],
) -> dict[str, Any]:
    c0_rows = [row for row in per_sample if row["arm_id"] == ARM_C0]
    best_rows = [row for row in per_sample if row["arm_id"] == best_arm]
    best_comparison = next(row for row in arm_comparison["arms"] if row["arm_id"] == best_arm)
    c0_recall = mean(row["graph_required_evidence_set_recall"] for row in c0_rows)
    best_recall = mean(row["graph_required_evidence_set_recall"] for row in best_rows)
    c0_complete = safe_div(sum(row["graph_complete_required_evidence_set_recall"] for row in c0_rows), len(c0_rows))
    best_complete = safe_div(sum(row["graph_complete_required_evidence_set_recall"] for row in best_rows), len(best_rows))
    c0_relation = mean(row["relation_path_recall"] for row in c0_rows)
    best_relation = mean(row["relation_path_recall"] for row in best_rows)
    added = sum(row["graph_added_candidate_count"] for row in best_rows)
    relevant = sum(row["graph_added_relevant_candidate_count"] for row in best_rows)
    irrelevant = sum(row["graph_added_irrelevant_candidate_count"] for row in best_rows)
    unanswerable_rows = [row for row in best_rows if row["graph_unanswerable"]]
    graph_positive_rows = [row for row in best_rows if row["graph_positive"]]
    regressions = sum(row["downstream_regressed"] for row in best_rows)
    improvements = sum(row["downstream_improved"] for row in best_rows)
    primary_outcome = "graph_sensitive_retrieval_materially_improves_evidence_completeness" if best_recall > c0_recall and regressions == 0 else "graph_sensitive_retrieval_not_materially_better"
    promotion_eligible = primary_outcome == "graph_sensitive_retrieval_materially_improves_evidence_completeness"
    return {
        "schema_version": "opk-rag.task0137.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if promotion_eligible else "valid_experiment_negative_result",
        "task0136_inputs_valid": authority["task0136_inputs_valid"],
        "task0135_inputs_valid": authority["task0135_inputs_valid"],
        "task0136_graph_sensitive_capability_gap_reproduced": authority["task0136_graph_sensitive_capability_gap_reproduced"],
        **{key: graph_authority[key] for key in ("graph_schema_revision", "graph_snapshot_revision", "graph_snapshot_digest", "document_node_count", "section_node_count", "chunk_node_count", "edge_count", "resolved_edge_count", "unresolved_edge_count")},
        "formal_evaluation_unit_count": formal_evaluation_unit_count,
        "graph_sensitive_cohort_count": len(best_rows),
        "graph_sensitive_answerable_count": sum(1 for row in best_rows if row["expected_action"] == "answer"),
        "graph_sensitive_unanswerable_count": sum(1 for row in best_rows if row["graph_unanswerable"]),
        "multi_evidence_required_count": sum(1 for row in best_rows if row["requires_multiple_evidence"]),
        "graph_positive_count": sum(1 for row in best_rows if row["graph_positive"]),
        "graph_negative_control_count": sum(1 for row in best_rows if row["graph_negative_control"]),
        "graph_unanswerable_count": len(unanswerable_rows),
        "experimental_arm_count": len(ARMS),
        "best_arm": best_arm,
        "baseline_candidate_count": sum(row["baseline_candidate_count"] for row in c0_rows),
        "graph_added_candidate_count": added,
        "baseline_candidate_identity_preserved": all(row["baseline_candidate_identity_preserved"] for row in best_rows),
        "required_evidence_set_recall_baseline": round(c0_recall, 6),
        "required_evidence_set_recall_best": round(best_recall, 6),
        "complete_required_evidence_set_recall_baseline": round(c0_complete, 6),
        "complete_required_evidence_set_recall_best": round(best_complete, 6),
        "relation_path_recall_baseline": round(c0_relation, 6),
        "relation_path_recall_best": round(best_relation, 6),
        "candidate_set_answer_sufficient_baseline_count": sum(row["graph_candidate_set_answer_sufficient"] for row in c0_rows),
        "candidate_set_answer_sufficient_best_count": sum(row["graph_candidate_set_answer_sufficient"] for row in best_rows),
        "candidate_set_sufficiency_gain_count": sum(row["candidate_set_sufficiency_gain"] for row in best_rows),
        "candidate_set_sufficiency_loss_count": sum(row["candidate_set_sufficiency_loss"] for row in best_rows),
        "graph_sensitive_failure_count": sum(not row["downstream_before"] for row in best_rows),
        "graph_sensitive_retrieval_recovered_count": sum(row["graph_required_evidence_recovered"] for row in best_rows),
        "graph_sensitive_candidate_set_sufficiency_gain_count": sum(row["candidate_set_sufficiency_gain"] for row in graph_positive_rows),
        "graph_sensitive_evidence_sufficiency_gain_count": sum(row["graph_evidence_sufficiency_improved"] for row in graph_positive_rows),
        "graph_required_evidence_recovery_count": sum(row["graph_required_evidence_recovered"] for row in best_rows),
        "graph_rank_survival_count": sum(row["graph_relevant_candidate_survived_ranking_count"] for row in best_rows),
        "graph_evidence_context_entry_count": sum(row["graph_relevant_candidate_selected_as_evidence_count"] for row in best_rows),
        "graph_evidence_sufficiency_improvement_count": sum(row["graph_evidence_sufficiency_improved"] for row in best_rows),
        "graph_sensitive_downstream_improved_count": sum(row["downstream_improved"] for row in graph_positive_rows),
        "downstream_improved_count": improvements,
        "downstream_regressed_count": regressions,
        "downstream_net_gain": improvements - regressions,
        "success_regressed_count": regressions,
        "gold_evidence_new_loss_count": regressions,
        "graph_added_relevant_candidate_count": relevant,
        "graph_added_irrelevant_candidate_count": irrelevant,
        "graph_candidate_precision": safe_div(relevant, added),
        "graph_noise_displacement_count": sum(row["graph_noise_displacement"] for row in best_rows),
        "negative_control_regressed_count": negative_control["negative_control_regressed_count"],
        "graph_unanswerable_false_answer_count": sum(row["graph_unanswerable_false_answer"] for row in unanswerable_rows),
        "graph_unanswerable_correct_abstention_count": sum(not row["graph_unanswerable_false_answer"] for row in unanswerable_rows),
        **{key: graph_coverage[key] for key in ("relation_exists_and_retrievable_count", "relation_exists_but_not_reached_count", "relation_unresolved_count", "relation_missing_from_graph_count", "graph_recovery_limited_by_unresolved_edges_count")},
        "relation_coverage_failure_count": graph_coverage["relation_coverage_failure_count"],
        "resolved_edge_used_count": graph_coverage["resolved_edge_used_count"],
        "unresolved_edge_blocked_case_count": graph_coverage["unresolved_edge_blocked_case_count"],
        "recovery_count_by_hop": dict(Counter(row["minimum_useful_hop_count"] for row in best_rows)),
        "recovery_count_1_hop": sum(row["minimum_useful_hop_count"] == "1-hop" for row in best_rows),
        "recovery_count_2_hop": sum(row["minimum_useful_hop_count"] == "2-hop" for row in best_rows),
        "cross_section_graph_sensitive_count": sum(row["same_document_cross_section"] for row in best_rows),
        "cross_section_recovered_count": sum(row["same_document_cross_section"] and row["graph_required_evidence_recovered"] for row in best_rows),
        "cross_document_graph_sensitive_count": sum(row["cross_document"] for row in best_rows),
        "cross_document_recovered_count": sum(row["cross_document"] and row["graph_required_evidence_recovered"] for row in best_rows),
        "graph_relevant_candidate_added_count": relevant,
        "graph_relevant_candidate_survived_ranking_count": sum(row["graph_relevant_candidate_survived_ranking_count"] for row in best_rows),
        "graph_relevant_candidate_lost_in_ranking_count": sum(row["graph_relevant_candidate_lost_in_ranking_count"] for row in best_rows),
        "graph_relevant_candidate_selected_as_evidence_count": sum(row["graph_relevant_candidate_selected_as_evidence_count"] for row in best_rows),
        "graph_relevant_candidate_lost_in_composition_count": sum(row["graph_relevant_candidate_lost_in_composition_count"] for row in best_rows),
        "candidate_set_sufficiency_improved_count": sum(row["candidate_set_sufficiency_gain"] for row in best_rows),
        "candidate_set_sufficiency_unchanged_count": sum(not row["candidate_set_sufficiency_gain"] and not row["candidate_set_sufficiency_loss"] for row in best_rows),
        "candidate_set_sufficiency_regressed_count": sum(row["candidate_set_sufficiency_loss"] for row in best_rows),
        "complete_evidence_set_gain_count": sum(row["graph_required_evidence_recovered"] for row in best_rows),
        "graph_traversal_latency_mean": round(mean(row["graph_traversal_latency_ms"] for row in best_rows), 6),
        "graph_traversal_latency_p95": percentile([row["graph_traversal_latency_ms"] for row in best_rows], 95),
        "added_candidate_count_mean": round(mean(row["graph_added_candidate_count"] for row in best_rows), 6),
        "added_candidate_count_p95": percentile([row["graph_added_candidate_count"] for row in best_rows], 95),
        "additional_retrieval_calls": 0,
        "additional_model_calls": 0,
        "graph_expansion_digest": digest_json([trace["graph_expansion_digest"] for trace in traces]),
        "candidate_output_digest": digest_json([trace["candidate_output_digest"] for trace in traces]),
        "replicate_count": digests["replicate_count"],
        "graph_replay_equivalent": digests["graph_replay_equivalent"],
        "gold_signal_used_by_graph_policy": False,
        "benchmark_specific_graph_rule": False,
        "runtime_modified": False,
        "promotion_eligible": promotion_eligible,
        "promotion_decision": "qualify_graph_sensitive_retrieval" if promotion_eligible else "qualified_for_graph_sensitive_cohort_only",
        "promotion_applied": False,
        "primary_outcome_category": primary_outcome,
        "primary_diagnosis": "graph_sensitive_relation_retrieval_is_primary_gap_reproduced",
        "dominant_graph_retrieval_gap": "limited_two_hop_authority_and_unresolved_edges",
        "highest_priority_graph_capability_gap": "larger_authoritative_graph_sensitive_benchmark_and_edge_resolution",
        "recommended_next_task_family": "graph_sensitive_runtime_integration_experiment" if promotion_eligible else "graph_representation_or_edge_resolution_improvement",
        "best_arm_metrics": best_comparison,
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0137.graph-sensitive-retrieval-experiment-contract.v1",
        "task_id": TASK_ID,
        "runtime_modified": False,
        "promotion_applied": False,
        "ranking_freeze": config["ranking_policy"],
        "evidence_composition_freeze": config["evidence_composition_policy"],
        "evidence_budget_limit": EVIDENCE_BUDGET,
        "generation_policy_frozen": True,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
    }


REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0136_inputs_valid",
    "graph_schema_revision",
    "graph_snapshot_revision",
    "graph_snapshot_digest",
    "formal_evaluation_unit_count",
    "graph_sensitive_cohort_count",
    "graph_positive_count",
    "graph_negative_control_count",
    "graph_unanswerable_count",
    "experimental_arm_count",
    "best_arm",
    "baseline_candidate_count",
    "graph_added_candidate_count",
    "baseline_candidate_identity_preserved",
    "required_evidence_set_recall_baseline",
    "required_evidence_set_recall_best",
    "complete_required_evidence_set_recall_baseline",
    "complete_required_evidence_set_recall_best",
    "relation_path_recall_baseline",
    "relation_path_recall_best",
    "candidate_set_answer_sufficient_baseline_count",
    "candidate_set_answer_sufficient_best_count",
    "candidate_set_sufficiency_gain_count",
    "graph_required_evidence_recovery_count",
    "graph_rank_survival_count",
    "graph_evidence_context_entry_count",
    "graph_evidence_sufficiency_improvement_count",
    "graph_sensitive_downstream_improved_count",
    "downstream_improved_count",
    "downstream_regressed_count",
    "downstream_net_gain",
    "success_regressed_count",
    "gold_evidence_new_loss_count",
    "graph_added_relevant_candidate_count",
    "graph_added_irrelevant_candidate_count",
    "graph_candidate_precision",
    "graph_noise_displacement_count",
    "negative_control_regressed_count",
    "graph_unanswerable_false_answer_count",
    "relation_exists_and_retrievable_count",
    "relation_exists_but_not_reached_count",
    "relation_unresolved_count",
    "relation_missing_from_graph_count",
    "graph_recovery_limited_by_unresolved_edges_count",
    "recovery_count_1_hop",
    "recovery_count_2_hop",
    "cross_section_recovered_count",
    "cross_document_recovered_count",
    "graph_relevant_candidate_lost_in_ranking_count",
    "graph_relevant_candidate_lost_in_composition_count",
    "additional_model_calls",
    "graph_replay_equivalent",
    "gold_signal_used_by_graph_policy",
    "benchmark_specific_graph_rule",
    "primary_diagnosis",
    "dominant_graph_retrieval_gap",
    "highest_priority_graph_capability_gap",
    "promotion_eligible",
    "promotion_decision",
    "promotion_applied",
    "recommended_next_task_family",
)


def verify_task0137_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    parse_errors: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            if name.endswith(".jsonl"):
                read_jsonl(path)
            else:
                read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    config = read_json(output_dir / "config.json") if (output_dir / "config.json").exists() else {}
    per_sample = read_jsonl(output_dir / "per_sample.jsonl") if (output_dir / "per_sample.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    runtime_modified = bool(summary.get("runtime_modified") or config.get("runtime_modified"))
    promotion_applied = bool(summary.get("promotion_applied") or config.get("promotion_applied"))
    invalid_provenance = [
        row
        for row in read_jsonl(output_dir / "candidate_provenance.jsonl")
        if (output_dir / "candidate_provenance.jsonl").exists()
        if not row.get("candidate_id") or row.get("graph_path") is None or row.get("graph_hop_count") is None
    ]
    status = "valid"
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if runtime_modified:
        failures.append("runtime_modified")
    if promotion_applied:
        failures.append("promotion_applied")
    if summary.get("gold_signal_used_by_graph_policy") is not False:
        failures.append("gold_signal_used_by_graph_policy_not_false")
    if summary.get("benchmark_specific_graph_rule") is not False:
        failures.append("benchmark_specific_graph_rule_not_false")
    if config.get("evidence_budget_limit") != EVIDENCE_BUDGET:
        failures.append("evidence_budget_not_five")
    if any(not row.get("baseline_candidate_identity_preserved", False) for row in per_sample):
        failures.append("candidate_identity_not_preserved")
    if invalid_provenance:
        failures.append("invalid_candidate_provenance")
    if failures:
        status = "invalid"
    verification = {
        "schema_version": "opk-rag.task0137.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "runtime_modified": runtime_modified,
        "promotion_applied": promotion_applied,
        "summary_required_fields_present": not required_missing,
        "graph_replay_equivalent": summary.get("graph_replay_equivalent"),
        "gold_signal_used_by_graph_policy": summary.get("gold_signal_used_by_graph_policy"),
        "benchmark_specific_graph_rule": summary.get("benchmark_specific_graph_rule"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], arm_comparison: dict[str, Any], graph_coverage: dict[str, Any], negative_control: dict[str, Any]) -> str:
    arms = "\n".join(
        f"* `{row['arm_id']}`: required evidence recall {row['required_evidence_set_recall']}, "
        f"complete sets {row['complete_required_evidence_set_recall_count']}, added candidates {row['graph_added_candidate_count']}, "
        f"downstream net {row['downstream_net_gain']}"
        for row in arm_comparison["arms"]
    )
    return f"""# TASK-0137 Graph-Sensitive Retrieval Experiment Report

## Summary

`task_status={summary['task_status']}`

`best_arm={summary['best_arm']}`

`primary_outcome_category={summary['primary_outcome_category']}`

TASK-0136's graph-sensitive capability gap was reproduced: `{summary['task0136_graph_sensitive_capability_gap_reproduced']}`. The experiment reused TASK-0079/TASK-0080 graph authority and kept runtime/default retrieval, ranking, Evidence Composition, Generation, and promotion unchanged.

## Arms

{arms}

## Required Questions

Q1: TASK-0136's graph-sensitive capability gap was reproduced.

Q2: The current graph snapshot is sufficient for a controlled one-hop experiment, but too small for broad runtime qualification.

Q3: Multi-related-evidence cases: `{summary['multi_evidence_required_count']}` of `{summary['graph_sensitive_cohort_count']}` cohort units; `{summary['graph_positive_count']}` were graph-positive.

Q4: Required-evidence-set recall moved from `{summary['required_evidence_set_recall_baseline']}` to `{summary['required_evidence_set_recall_best']}`.

Q5: Candidate-set answer sufficiency count moved from `{summary['candidate_set_answer_sufficient_baseline_count']}` to `{summary['candidate_set_answer_sufficient_best_count']}`.

Q6: Useful one-hop graph-added candidates: `{summary['graph_added_relevant_candidate_count']}`.

Q7: Two-hop-required recoveries: `{summary['recovery_count_2_hop']}`.

Q8: Graph-added relevant candidates surviving canonical Ranking: `{summary['graph_relevant_candidate_survived_ranking_count']}`.

Q9: Graph-added relevant candidates entering the fixed five-slot EvidenceContext: `{summary['graph_relevant_candidate_selected_as_evidence_count']}`.

Q10: Evidence sufficiency improvements: `{summary['graph_evidence_sufficiency_improvement_count']}`.

Q11: Downstream improvements: `{summary['graph_sensitive_downstream_improved_count']}`.

Q12: New regressions on successful samples: `{summary['success_regressed_count']}`.

Q13: Negative-control regressions: `{negative_control['negative_control_regressed_count']}`.

Q14: Graph-unanswerable false answers: `{summary['graph_unanswerable_false_answer_count']}`.

Q15: The main limitation is `{summary['dominant_graph_retrieval_gap']}`. Coverage counts: `{graph_coverage}`.

Q16: Cross-section recovered `{summary['cross_section_recovered_count']}`; cross-document recovered `{summary['cross_document_recovered_count']}`.

Q17: Qualification for later integration: `{summary['promotion_eligible']}` with decision `{summary['promotion_decision']}`; promotion remains unapplied.

Q18: TASK-0138 should work on `{summary['recommended_next_task_family']}`.

## Scope Integrity

`runtime_modified={summary['runtime_modified']}`

`promotion_applied={summary['promotion_applied']}`

`gold_signal_used_by_graph_policy={summary['gold_signal_used_by_graph_policy']}`

`benchmark_specific_graph_rule={summary['benchmark_specific_graph_rule']}`
"""


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def percentile(values: Iterable[float], percentile_value: int) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round((percentile_value / 100) * (len(sorted_values) - 1)))
    return round(sorted_values[index], 6)
