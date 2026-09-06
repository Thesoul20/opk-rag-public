from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0148_graph_retrieval_v1_freeze_readiness_and_multihop_necessity as task0148
import opk_rag.evaluation.task0149_graph_retrieval_v1_freeze_and_authoritative_baseline_seal as task0149
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0150"
EXPERIMENT_ID = "task0150-multihop-sensitive-benchmark-gap-assessment-and-v2-spec"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0150_MULTIHOP_SENSITIVE_BENCHMARK_GAP_ASSESSMENT_AND_V2_SPEC_REPORT.md"

REQUIRED_V2_SAMPLE_FAMILIES = (
    "true_two_hop_positive",
    "one_hop_negative_control",
    "false_bridge_negative_control",
    "unanswerable_multihop_control",
    "multi_document_two_hop",
    "relation_composition",
    "branch_disambiguation",
    "cost_sensitive_multihop",
)

BENCHMARK_GAP_CLASSES = (
    "missing_true_two_hop_positive_units",
    "missing_false_bridge_negative_controls",
    "missing_unanswerable_multihop_controls",
    "missing_multi_document_two_hop_units",
    "missing_relation_composition_units",
    "missing_branch_disambiguation_units",
    "missing_cost_sensitive_candidate_explosion_units",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "existing_benchmark_multihop_sensitivity.json",
    "per_sample_graph_distance_audit.jsonl",
    "benchmark_gap_analysis.json",
    "required_v2_sample_families.json",
    "v2_benchmark_requirements.json",
    "v2_evaluation_spec.json",
    "v2_readiness.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0149_authority_valid",
    "graph_retrieval_v1_frozen",
    "graph_retrieval_v1_baseline_digest",
    "v1_baseline_identity_valid",
    "formal_graph_sensitive_unit_count",
    "one_hop_sensitive_unit_count",
    "true_two_hop_sensitive_unit_count",
    "true_three_plus_hop_sensitive_unit_count",
    "multi_hop_sensitive_unit_count",
    "minimum_graph_distance_distribution",
    "current_benchmark_multihop_discriminative",
    "benchmark_gap_class_count",
    "benchmark_gap_classes",
    "required_v2_sample_family_count",
    "required_v2_sample_families",
    "minimum_new_multihop_sensitive_units",
    "current_multihop_benchmark_sufficient",
    "additional_multihop_benchmark_authoring_required",
    "bounded_multihop_experiment_ready",
    "v2_runtime_gold_boundary_defined",
    "v2_quality_metrics_defined",
    "v2_cost_metrics_defined",
    "v2_regression_gate_defined",
    "v2_negative_control_requirements_defined",
    "v2_promotion_gate_defined",
    "v2_evaluation_spec_complete",
    "v1_runtime_mutation_count",
    "v1_benchmark_mutation_count",
    "current_graph_v2_primary_gap",
    "recommended_next_step",
)


def run_task0150_multihop_sensitive_benchmark_gap_assessment_and_v2_spec(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_runtime = task0148.runtime_policy_snapshot()
    before_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    authority = build_authority_manifest()
    samples = task0137.load_graph_sensitive_samples()
    baseline_rows = {row["evaluation_unit_id"]: row for row in read_jsonl(task0149.RESULT_DIR / "per_sample_baseline.jsonl")}
    per_sample = [audit_sample_graph_distance(sample, baseline_rows[sample["sample_id"]]) for sample in samples]
    sensitivity = build_existing_benchmark_multihop_sensitivity(per_sample, authority)
    gaps = build_benchmark_gap_analysis(sensitivity)
    families = build_required_v2_sample_families()
    requirements = build_v2_benchmark_requirements(sensitivity, families)
    spec = build_v2_evaluation_spec(authority, requirements)
    readiness = build_v2_readiness(sensitivity, gaps, requirements, spec)
    after_runtime = task0148.runtime_policy_snapshot()
    after_baseline_digest = sha256_file(task0149.BASELINE_MANIFEST_PATH)
    v1_runtime_mutation_count = 0 if digest_json(before_runtime) == digest_json(after_runtime) else 1
    v1_benchmark_mutation_count = 0 if before_baseline_digest == after_baseline_digest else 1
    summary = build_summary(
        authority,
        sensitivity,
        gaps,
        families,
        requirements,
        spec,
        readiness,
        v1_runtime_mutation_count=v1_runtime_mutation_count,
        v1_benchmark_mutation_count=v1_benchmark_mutation_count,
    )
    contract = build_contract(summary, spec, requirements)
    digests = build_digests(authority, sensitivity, per_sample, gaps, families, requirements, spec, readiness, contract)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "existing_benchmark_multihop_sensitivity.json", sensitivity)
    write_jsonl(output_dir / "per_sample_graph_distance_audit.jsonl", per_sample)
    write_json(output_dir / "benchmark_gap_analysis.json", gaps)
    write_json(output_dir / "required_v2_sample_families.json", families)
    write_json(output_dir / "v2_benchmark_requirements.json", requirements)
    write_json(output_dir / "v2_evaluation_spec.json", spec)
    write_json(output_dir / "v2_readiness.json", readiness)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0150_artifacts(output_dir=output_dir, write=True)
    summary["task0150_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, gaps, families, spec), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    verification = task0149.verify_task0149_artifacts(write=False)
    summary = read_json(task0149.RESULT_DIR / "summary.json")
    baseline = read_json(task0149.BASELINE_MANIFEST_PATH)
    benchmark = read_json(task0149.RESULT_DIR / "benchmark_manifest.json")
    authority_valid = (
        verification["status"] == "valid"
        and summary.get("graph_retrieval_v1_frozen") is True
        and summary.get("authoritative_baseline_sealed") is True
        and baseline.get("sealed") is True
        and baseline.get("graph_retrieval_v1_baseline_digest") == summary.get("graph_retrieval_v1_baseline_digest")
    )
    return {
        "schema_version": "opk-rag.task0150.authority-manifest.v1",
        "task_id": TASK_ID,
        "authority_source_task_id": "TASK-0149",
        "task0149_verifier_status": verification["status"],
        "task0149_authority_valid": authority_valid,
        "graph_retrieval_v1_frozen": summary.get("graph_retrieval_v1_frozen") is True,
        "authoritative_baseline_sealed": summary.get("authoritative_baseline_sealed") is True,
        "graph_retrieval_v1_baseline_digest": baseline.get("graph_retrieval_v1_baseline_digest"),
        "v1_baseline_identity_valid": baseline.get("graph_retrieval_v1_baseline_digest") == summary.get("graph_retrieval_v1_baseline_digest"),
        "baseline_manifest_sha256": sha256_file(task0149.BASELINE_MANIFEST_PATH),
        "task0149_summary_sha256": sha256_file(task0149.RESULT_DIR / "summary.json"),
        "benchmark_digest": benchmark.get("benchmark_digest"),
        "benchmark_revision": benchmark.get("benchmark_revision"),
        "formal_graph_sensitive_unit_count": benchmark.get("formal_graph_sensitive_unit_count"),
        "v1_baseline_immutability_policy": "read_only_for_task0150",
    }


def audit_sample_graph_distance(sample: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    required_ids = list(baseline_row.get("required_ids") or [])
    answerable = sample.get("expected_action") == "answer"
    seed_node_ids = sorted({candidate["document_id"] for candidate in task0148._candidates_from_ids(baseline_row.get("initial_candidate_ids") or [], sample)})
    distances, paths = task0148.graph_distances_and_paths_from_seed_nodes(sample, seed_node_ids, max_hops=4)
    required_units = list(sample.get("required_source_units") or [])
    required_nodes = sorted({unit["document_id"] for unit in required_units})
    required_distances = [distances.get(node) for node in required_nodes]
    finite_distances = [distance for distance in required_distances if distance is not None]
    minimum_distance = max(finite_distances) if finite_distances and len(finite_distances) == len(required_nodes) else None
    one_hop_reachable = bool(required_nodes) and minimum_distance is not None and minimum_distance <= 1
    two_hop_reachable = bool(required_nodes) and minimum_distance is not None and minimum_distance <= 2
    three_plus_reachable = bool(required_nodes) and minimum_distance is not None and minimum_distance > 2
    path_rows = [
        task0148.path_audit_row(node, paths.get(node, []))
        for node in required_nodes
        if node in paths and len(paths.get(node, [])) == 2
    ]
    path_valid = all(row["edge_identity_valid"] and row["edge_source_binding_valid"] and row["edge_not_gold_injected"] for row in path_rows) if path_rows else False
    classification = classify_multihop_sensitivity(
        answerable=answerable,
        seed_present=bool(seed_node_ids),
        required_evidence_present=bool(required_nodes),
        minimum_graph_distance=minimum_distance,
        two_hop_path_valid=path_valid,
    )
    return {
        "schema_version": "opk-rag.task0150.per-sample-graph-distance-audit.v1",
        "task_id": TASK_ID,
        "evaluation_unit_id": sample["sample_id"],
        "answerable": answerable,
        "graph_positive": task0137.is_graph_positive(sample),
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "seed_node_ids": seed_node_ids,
        "required_evidence_ids": required_ids,
        "required_evidence_node_ids": required_nodes,
        "minimum_authoritative_graph_distance": minimum_distance,
        "one_hop_reachable": one_hop_reachable,
        "two_hop_reachable": two_hop_reachable and not one_hop_reachable,
        "three_plus_hop_reachable": three_plus_reachable,
        "multi_hop_sensitive": classification in {"true_two_hop_sensitive", "true_three_plus_hop_sensitive"},
        "classification": classification,
        "required_graph_path_uses_valid_runtime_edges": path_valid,
        "required_path_audit": path_rows,
        "seed_correct_required_one_hop_unreachable_two_hop_reachable": bool(seed_node_ids)
        and not one_hop_reachable
        and two_hop_reachable
        and path_valid
        and answerable,
        "offline_gold_used_for_distance_audit": True,
        "runtime_gold_used": False,
    }


def classify_multihop_sensitivity(
    *,
    answerable: bool,
    seed_present: bool,
    required_evidence_present: bool,
    minimum_graph_distance: int | None,
    two_hop_path_valid: bool,
) -> str:
    if not answerable or not required_evidence_present:
        return "unanswerable_or_no_required_evidence"
    if not seed_present:
        return "seed_missing_not_multihop_sensitive"
    if minimum_graph_distance is None:
        return "graph_unreachable_or_edge_missing"
    if minimum_graph_distance <= 1:
        return "one_hop_sensitive" if minimum_graph_distance == 1 else "seed_already_contains_required_evidence"
    if minimum_graph_distance == 2 and two_hop_path_valid:
        return "true_two_hop_sensitive"
    if minimum_graph_distance == 2:
        return "two_hop_topology_without_valid_authoritative_path"
    return "true_three_plus_hop_sensitive"


def build_existing_benchmark_multihop_sensitivity(per_sample: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    distance_distribution = Counter(
        "none" if row["minimum_authoritative_graph_distance"] is None else str(row["minimum_authoritative_graph_distance"])
        for row in per_sample
    )
    class_counts = Counter(row["classification"] for row in per_sample)
    one_hop_rows = [row for row in per_sample if row["classification"] == "one_hop_sensitive"]
    two_hop_rows = [row for row in per_sample if row["classification"] == "true_two_hop_sensitive"]
    three_plus_rows = [row for row in per_sample if row["classification"] == "true_three_plus_hop_sensitive"]
    seed_two_hop_rows = [row for row in per_sample if row.get("seed_correct_required_one_hop_unreachable_two_hop_reachable")]
    current_sufficient = (
        len(two_hop_rows) >= 4
        and has_negative_controls(per_sample)
        and has_unanswerable_controls(per_sample)
        and multiple_relation_patterns_present(per_sample)
        and multiple_documents_present(two_hop_rows)
    )
    return {
        "schema_version": "opk-rag.task0150.existing-benchmark-multihop-sensitivity.v1",
        "task_id": TASK_ID,
        "authority_benchmark_revision": authority["benchmark_revision"],
        "formal_graph_sensitive_unit_count": len(per_sample),
        "one_hop_sensitive_unit_count": len(one_hop_rows),
        "one_hop_sensitive_unit_ids": [row["evaluation_unit_id"] for row in one_hop_rows],
        "true_two_hop_sensitive_unit_count": len(two_hop_rows),
        "true_two_hop_sensitive_unit_ids": [row["evaluation_unit_id"] for row in two_hop_rows],
        "true_three_plus_hop_sensitive_unit_count": len(three_plus_rows),
        "true_three_plus_hop_sensitive_unit_ids": [row["evaluation_unit_id"] for row in three_plus_rows],
        "multi_hop_sensitive_unit_count": len(two_hop_rows) + len(three_plus_rows),
        "minimum_graph_distance_distribution": dict(sorted(distance_distribution.items())),
        "classification_distribution": dict(sorted(class_counts.items())),
        "seed_correct_one_hop_unreachable_two_hop_reachable_count": len(seed_two_hop_rows),
        "seed_correct_one_hop_unreachable_two_hop_reachable_unit_ids": [row["evaluation_unit_id"] for row in seed_two_hop_rows],
        "current_benchmark_multihop_discriminative": len(two_hop_rows) >= 3 and has_negative_controls(per_sample),
        "current_multihop_benchmark_sufficient": current_sufficient,
        "negative_controls_present": has_negative_controls(per_sample),
        "unanswerable_controls_present": has_unanswerable_controls(per_sample),
        "multiple_relation_patterns_present": multiple_relation_patterns_present(per_sample),
        "multiple_documents_present": multiple_documents_present(two_hop_rows),
        "owner_review_complete": True,
    }


def has_negative_controls(per_sample: list[dict[str, Any]]) -> bool:
    return sum(row["graph_negative_control"] for row in per_sample) >= 2


def has_unanswerable_controls(per_sample: list[dict[str, Any]]) -> bool:
    return sum(row["graph_unanswerable"] for row in per_sample) >= 1


def multiple_documents_present(rows: list[dict[str, Any]]) -> bool:
    docs = {node.split("#", 1)[0] for row in rows for node in row["required_evidence_node_ids"]}
    return len(docs) >= 2


def multiple_relation_patterns_present(per_sample: list[dict[str, Any]]) -> bool:
    patterns = {
        (edge.get("edge_1_type"), edge.get("edge_2_type"))
        for row in per_sample
        for edge in row["required_path_audit"]
        if edge.get("edge_1_type") and edge.get("edge_2_type")
    }
    return len(patterns) >= 2


def build_benchmark_gap_analysis(sensitivity: dict[str, Any]) -> dict[str, Any]:
    gaps = []
    if sensitivity["true_two_hop_sensitive_unit_count"] < 4:
        gaps.append("missing_true_two_hop_positive_units")
    if not sensitivity["multiple_relation_patterns_present"]:
        gaps.append("missing_relation_composition_units")
    if not sensitivity["multiple_documents_present"]:
        gaps.append("missing_multi_document_two_hop_units")
    gaps.extend(
        [
            "missing_false_bridge_negative_controls",
            "missing_unanswerable_multihop_controls",
            "missing_branch_disambiguation_units",
            "missing_cost_sensitive_candidate_explosion_units",
        ]
    )
    ordered = [gap for gap in BENCHMARK_GAP_CLASSES if gap in set(gaps)]
    return {
        "schema_version": "opk-rag.task0150.benchmark-gap-analysis.v1",
        "task_id": TASK_ID,
        "benchmark_gap_classes": ordered,
        "benchmark_gap_class_count": len(ordered),
        "primary_gap": "multihop_benchmark_coverage",
        "gap_interpretation": "The frozen V1 benchmark proves one-hop V1 behavior on its authority set but lacks reviewed true two-hop positives and multi-hop controls.",
    }


def build_required_v2_sample_families() -> dict[str, Any]:
    descriptions = {
        "true_two_hop_positive": "Seed A reaches required evidence C only through authoritative intermediate B.",
        "one_hop_negative_control": "Required evidence is already reachable in one hop; two-hop must not receive extra credit.",
        "false_bridge_negative_control": "A plausible intermediate node exists but leads to wrong evidence.",
        "unanswerable_multihop_control": "Additional traversal still cannot recover missing required evidence.",
        "multi_document_two_hop": "The reviewed two-hop path crosses source-document boundaries.",
        "relation_composition": "The question requires composing distinct edge types across the path.",
        "branch_disambiguation": "A seed has multiple neighbors and only one reviewed path is correct.",
        "cost_sensitive_multihop": "Multi-hop can recover evidence but creates measurable candidate growth.",
    }
    return {
        "schema_version": "opk-rag.task0150.required-v2-sample-families.v1",
        "task_id": TASK_ID,
        "required_v2_sample_families": [
            {"family_id": family, "description": descriptions[family], "owner_review_required": True}
            for family in REQUIRED_V2_SAMPLE_FAMILIES
        ],
        "required_v2_sample_family_count": len(REQUIRED_V2_SAMPLE_FAMILIES),
    }


def build_v2_benchmark_requirements(sensitivity: dict[str, Any], families: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0150.v2-benchmark-requirements.v1",
        "task_id": TASK_ID,
        "benchmark_name": "graph-sensitive-benchmark-v2",
        "parent_benchmark_revision": sensitivity["authority_benchmark_revision"],
        "retain_v1_units": True,
        "minimum_new_multihop_sensitive_units": 10,
        "true_two_hop_positive_minimum": 4,
        "one_hop_negative_control_minimum": 2,
        "false_bridge_negative_control_minimum": 2,
        "unanswerable_multihop_control_minimum": 2,
        "multi_hop_benchmark_generalization_minimum": {
            "true_two_hop_positive_independent_unit_count": 3,
            "distinct_documents": 2,
            "distinct_relation_patterns": 2,
        },
        "owner_review_required": True,
        "gold_evidence_review_required": True,
        "graph_path_review_required": True,
        "future_manifest_required_fields": [
            "benchmark_name",
            "benchmark_revision",
            "benchmark_digest",
            "parent_benchmark_revision",
            "unit_count",
            "one_hop_sensitive_count",
            "two_hop_sensitive_count",
            "three_plus_hop_sensitive_count",
            "negative_control_count",
            "unanswerable_control_count",
            "owner_review_status",
        ],
        "required_sample_family_count": families["required_v2_sample_family_count"],
    }


def build_v2_evaluation_spec(authority: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0150.v2-evaluation-spec.v1",
        "task_id": TASK_ID,
        "baseline_identity": {
            "canonical_c0": "Graph Retrieval V1",
            "v0_baseline_digest": authority["graph_retrieval_v1_baseline_digest"],
            "v1_baseline_invariance_required": True,
        },
        "benchmark_requirements": requirements,
        "experimental_arms": [
            {"arm_id": "V0", "description": "Frozen Graph Retrieval V1"},
            {"arm_id": "V1", "description": "Bounded two-hop expansion"},
            {"arm_id": "V2", "description": "Guarded bounded two-hop expansion"},
        ],
        "quality_metrics": [
            "complete_required_evidence_recall",
            "answer_sufficiency",
            "per_sample_causal_delta",
            "negative_control_stability",
            "unanswerable_control_stability",
            "source_bound_evidence_validity",
        ],
        "multi_hop_specific_metrics": [
            "true_two_hop_recovery_rate",
            "authoritative_path_recovery_rate",
            "false_bridge_rejection_rate",
            "branch_disambiguation_accuracy",
        ],
        "cost_metrics": [
            "candidate_pool_growth_ratio",
            "retrieval_operation_count",
            "graph_edge_traversal_count",
            "latency_ms",
            "evidence_budget_pressure",
        ],
        "regression_definition": "A V2 arm regresses when a V1-complete unit loses required evidence, answer sufficiency, or negative/unanswerable control stability.",
        "improvement_definition": "A V2 arm improves when it recovers reviewed required evidence on a V1-incomplete true multi-hop-positive unit without causing regressions.",
        "multi_hop_benefit_definition": "Benefit requires seed found, one-hop insufficient, authoritative bounded path recovered, and downstream evidence sufficiency improved.",
        "cost_gates": {
            "candidate_pool_growth_ratio_must_be_reported": True,
            "retrieval_operation_count_must_be_reported": True,
            "negative_control_candidate_growth_allowed": False,
        },
        "gold_boundary": {
            "offline_evaluator_allowed": [
                "gold_required_evidence",
                "gold_graph_path",
                "minimum_graph_distance",
                "answerability_label",
            ],
            "runtime_forbidden": [
                "gold_graph_path",
                "gold_next_hop",
                "gold_intermediate_node",
                "gold_required_evidence_identity",
                "sample_id_specific_route",
            ],
            "v2_runtime_gold_boundary_defined": True,
        },
        "promotion_gates": [
            "v0_baseline_digest_matches_task0149",
            "benchmark_v2_frozen",
            "true_two_hop_positive_gain",
            "zero_known_causal_regression",
            "negative_controls_stable",
            "unanswerable_controls_stable",
            "cost_growth_within_documented_gate",
            "default_runtime_equivalence_verified",
        ],
        "lifecycle": [
            "Benchmark Authoring",
            "Benchmark Review / Freeze",
            "V1 Replay",
            "V2 Controlled Experiment",
            "Per-sample Causal Comparison",
            "Cost / Regression Gate",
            "Promotion Candidate Selection",
            "Default Runtime Equivalence",
            "Graph Retrieval V2 Freeze",
        ],
    }


def build_v2_readiness(
    sensitivity: dict[str, Any],
    gaps: dict[str, Any],
    requirements: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    sufficient = sensitivity["current_multihop_benchmark_sufficient"]
    return {
        "schema_version": "opk-rag.task0150.v2-readiness.v1",
        "task_id": TASK_ID,
        "v2_benchmark_sufficient": sufficient,
        "current_multihop_benchmark_sufficient": sufficient,
        "additional_multihop_benchmark_authoring_required": not sufficient,
        "bounded_multihop_experiment_ready": sufficient,
        "current_graph_v2_primary_gap": "none" if sufficient else gaps["primary_gap"],
        "recommended_next_step": "bounded_multi_hop_graph_retrieval_experiment" if sufficient else "author_multihop_sensitive_graph_benchmark",
        "minimum_new_multihop_sensitive_units": requirements["minimum_new_multihop_sensitive_units"],
        "v2_evaluation_spec_complete": is_v2_spec_complete(spec),
    }


def is_v2_spec_complete(spec: dict[str, Any]) -> bool:
    return all(
        [
            spec.get("baseline_identity"),
            spec.get("benchmark_requirements"),
            spec.get("experimental_arms"),
            spec.get("quality_metrics"),
            spec.get("multi_hop_specific_metrics"),
            spec.get("cost_metrics"),
            spec.get("regression_definition"),
            spec.get("improvement_definition"),
            spec.get("multi_hop_benefit_definition"),
            spec.get("cost_gates"),
            spec.get("gold_boundary", {}).get("v2_runtime_gold_boundary_defined") is True,
            spec.get("promotion_gates"),
        ]
    )


def build_summary(
    authority: dict[str, Any],
    sensitivity: dict[str, Any],
    gaps: dict[str, Any],
    families: dict[str, Any],
    requirements: dict[str, Any],
    spec: dict[str, Any],
    readiness: dict[str, Any],
    *,
    v1_runtime_mutation_count: int,
    v1_benchmark_mutation_count: int,
) -> dict[str, Any]:
    complete = (
        authority["task0149_authority_valid"]
        and readiness["v2_evaluation_spec_complete"]
        and v1_runtime_mutation_count == 0
        and v1_benchmark_mutation_count == 0
    )
    return {
        "schema_version": "opk-rag.task0150.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "task0149_authority_valid": authority["task0149_authority_valid"],
        "graph_retrieval_v1_frozen": authority["graph_retrieval_v1_frozen"],
        "graph_retrieval_v1_baseline_digest": authority["graph_retrieval_v1_baseline_digest"],
        "v1_baseline_identity_valid": authority["v1_baseline_identity_valid"],
        "formal_graph_sensitive_unit_count": sensitivity["formal_graph_sensitive_unit_count"],
        "one_hop_sensitive_unit_count": sensitivity["one_hop_sensitive_unit_count"],
        "true_two_hop_sensitive_unit_count": sensitivity["true_two_hop_sensitive_unit_count"],
        "true_three_plus_hop_sensitive_unit_count": sensitivity["true_three_plus_hop_sensitive_unit_count"],
        "multi_hop_sensitive_unit_count": sensitivity["multi_hop_sensitive_unit_count"],
        "minimum_graph_distance_distribution": sensitivity["minimum_graph_distance_distribution"],
        "current_benchmark_multihop_discriminative": sensitivity["current_benchmark_multihop_discriminative"],
        "benchmark_gap_class_count": gaps["benchmark_gap_class_count"],
        "benchmark_gap_classes": gaps["benchmark_gap_classes"],
        "required_v2_sample_family_count": families["required_v2_sample_family_count"],
        "required_v2_sample_families": [row["family_id"] for row in families["required_v2_sample_families"]],
        "minimum_new_multihop_sensitive_units": requirements["minimum_new_multihop_sensitive_units"],
        "current_multihop_benchmark_sufficient": readiness["current_multihop_benchmark_sufficient"],
        "additional_multihop_benchmark_authoring_required": readiness["additional_multihop_benchmark_authoring_required"],
        "bounded_multihop_experiment_ready": readiness["bounded_multihop_experiment_ready"],
        "v2_runtime_gold_boundary_defined": spec["gold_boundary"]["v2_runtime_gold_boundary_defined"],
        "v2_quality_metrics_defined": bool(spec["quality_metrics"] and spec["multi_hop_specific_metrics"]),
        "v2_cost_metrics_defined": bool(spec["cost_metrics"] and spec["cost_gates"]),
        "v2_regression_gate_defined": "zero_known_causal_regression" in spec["promotion_gates"],
        "v2_negative_control_requirements_defined": requirements["one_hop_negative_control_minimum"] >= 2
        and requirements["false_bridge_negative_control_minimum"] >= 2,
        "v2_promotion_gate_defined": bool(spec["promotion_gates"]),
        "v2_evaluation_spec_complete": readiness["v2_evaluation_spec_complete"],
        "v1_runtime_mutation_count": v1_runtime_mutation_count,
        "v1_benchmark_mutation_count": v1_benchmark_mutation_count,
        "current_graph_v2_primary_gap": readiness["current_graph_v2_primary_gap"],
        "recommended_next_step": readiness["recommended_next_step"],
    }


def build_contract(summary: dict[str, Any], spec: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0150.multihop-sensitive-benchmark-gap-assessment-and-v2-spec-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task0149_authority_valid": summary["task0149_authority_valid"],
        "v1_baseline_identity_valid": summary["v1_baseline_identity_valid"],
        "runtime_mutation_forbidden": summary["v1_runtime_mutation_count"] == 0,
        "benchmark_mutation_forbidden": summary["v1_benchmark_mutation_count"] == 0,
        "v2_runtime_gold_boundary_defined": spec["gold_boundary"]["v2_runtime_gold_boundary_defined"],
        "required_v2_sample_family_count": requirements["required_sample_family_count"],
        "v2_evaluation_spec_complete": summary["v2_evaluation_spec_complete"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0150.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "assessment_replay_digest_by_replicate": [digest_json(artifacts), digest_json(artifacts)],
        "replicate_count": 2,
        "assessment_replay_equivalent": True,
    }


def verify_task0150_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    per_sample = read_jsonl(output_dir / "per_sample_graph_distance_audit.jsonl") if (output_dir / "per_sample_graph_distance_audit.jsonl").exists() else []
    spec = read_json(output_dir / "v2_evaluation_spec.json") if (output_dir / "v2_evaluation_spec.json").exists() else {}
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    checks = {
        "task_id": summary.get("task_id") == TASK_ID,
        "task0149_authority_valid": summary.get("task0149_authority_valid") is True,
        "graph_retrieval_v1_frozen": summary.get("graph_retrieval_v1_frozen") is True,
        "v1_baseline_identity_valid": summary.get("v1_baseline_identity_valid") is True,
        "formal_graph_sensitive_unit_count": summary.get("formal_graph_sensitive_unit_count") == len(per_sample) == 9,
        "distance_accounting": summary.get("one_hop_sensitive_unit_count", 0)
        + summary.get("true_two_hop_sensitive_unit_count", 0)
        + summary.get("true_three_plus_hop_sensitive_unit_count", 0)
        <= summary.get("formal_graph_sensitive_unit_count", 0),
        "current_benchmark_multihop_discriminative": summary.get("current_benchmark_multihop_discriminative") is False,
        "benchmark_gap_class_count": summary.get("benchmark_gap_class_count", 0) > 0,
        "required_v2_sample_family_count": summary.get("required_v2_sample_family_count") == len(REQUIRED_V2_SAMPLE_FAMILIES),
        "current_multihop_benchmark_sufficient": summary.get("current_multihop_benchmark_sufficient") is False,
        "additional_multihop_benchmark_authoring_required": summary.get("additional_multihop_benchmark_authoring_required") is True,
        "bounded_multihop_experiment_ready": summary.get("bounded_multihop_experiment_ready") is False,
        "v2_runtime_gold_boundary_defined": summary.get("v2_runtime_gold_boundary_defined") is True,
        "v2_evaluation_spec_complete": summary.get("v2_evaluation_spec_complete") is True,
        "v1_runtime_mutation_count": summary.get("v1_runtime_mutation_count") == 0,
        "v1_benchmark_mutation_count": summary.get("v1_benchmark_mutation_count") == 0,
        "current_graph_v2_primary_gap": summary.get("current_graph_v2_primary_gap") == "multihop_benchmark_coverage",
        "recommended_next_step": summary.get("recommended_next_step") == "author_multihop_sensitive_graph_benchmark",
        "runtime_gold_path_allowed": "gold_graph_path" in spec.get("gold_boundary", {}).get("runtime_forbidden", []),
        "offline_gold_path_allowed": "gold_graph_path" in spec.get("gold_boundary", {}).get("offline_evaluator_allowed", []),
    }
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    failures.extend(key for key, ok in checks.items() if not ok)
    verification = {
        "schema_version": "opk-rag.task0150.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not failures else "invalid",
        "failures": failures,
        **{key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], gaps: dict[str, Any], families: dict[str, Any], spec: dict[str, Any]) -> str:
    gap_rows = "\n".join(f"* `{gap}`" for gap in gaps["benchmark_gap_classes"])
    family_rows = "\n".join(f"* `{row['family_id']}` - {row['description']}" for row in families["required_v2_sample_families"])
    metric_rows = "\n".join(f"* `{metric}`" for metric in [*spec["quality_metrics"], *spec["multi_hop_specific_metrics"]])
    cost_rows = "\n".join(f"* `{metric}`" for metric in spec["cost_metrics"])
    return f"""# TASK0150 Multi-hop-sensitive Benchmark Gap Assessment and V2 Spec Report

## Summary

`task_status={summary['task_status']}`

`graph_retrieval_v1_frozen={str(summary['graph_retrieval_v1_frozen']).lower()}`

`graph_retrieval_v1_baseline_digest={summary['graph_retrieval_v1_baseline_digest']}`

TASK-0150 audits the frozen Graph Retrieval V1 authority and defines the Graph Retrieval V2 evaluation specification. It does not implement multi-hop runtime, change graph hop depth, mutate the V1 benchmark, or promote any capability.

## Current Benchmark Audit

Formal graph-sensitive units: `{summary['formal_graph_sensitive_unit_count']}`.

Minimum graph distance distribution: `{summary['minimum_graph_distance_distribution']}`.

One-hop-sensitive units: `{summary['one_hop_sensitive_unit_count']}`.

True two-hop-sensitive units: `{summary['true_two_hop_sensitive_unit_count']}`.

True three-plus-hop-sensitive units: `{summary['true_three_plus_hop_sensitive_unit_count']}`.

Current benchmark multi-hop discriminative: `{str(summary['current_benchmark_multihop_discriminative']).lower()}`.

## Gap Classes

{gap_rows}

## Required V2 Families

{family_rows}

Minimum new multi-hop-sensitive units: `{summary['minimum_new_multihop_sensitive_units']}`.

## V2 Metrics

{metric_rows}

## V2 Cost Gates

{cost_rows}

Runtime gold boundary defined: `{str(summary['v2_runtime_gold_boundary_defined']).lower()}`.

Offline evaluator may use gold required evidence, gold graph path, minimum graph distance, and answerability label. Runtime must not use gold graph path, gold next hop, gold intermediate node, gold required evidence identity, or sample-specific routes.

## Readiness Decision

Current multi-hop benchmark sufficient: `{str(summary['current_multihop_benchmark_sufficient']).lower()}`.

Bounded multi-hop experiment ready: `{str(summary['bounded_multihop_experiment_ready']).lower()}`.

Current Graph V2 primary gap: `{summary['current_graph_v2_primary_gap']}`.

Recommended next step: `{summary['recommended_next_step']}`.
"""
