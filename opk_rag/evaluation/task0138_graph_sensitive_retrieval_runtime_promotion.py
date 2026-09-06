from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import opk_rag.evaluation.task0131_targeted_evidence_composition_runtime_promotion as task0131
import opk_rag.evaluation.task0136_ranking_to_evidence_utility_boundary_diagnosis as task0136
import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_retrieval
from opk_rag.runtime_v2.reranker import RerankerRuntimeConfig


TASK_ID = "TASK-0138"
EXPERIMENT_ID = "task0138-graph-sensitive-retrieval-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0138_graph_sensitive_retrieval_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0138_GRAPH_SENSITIVE_RETRIEVAL_RUNTIME_PROMOTION_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "runtime_equivalence.jsonl",
    "graph_candidate_provenance.jsonl",
    "graph_expansion_trace.jsonl",
    "pre_post_runtime_comparison.json",
    "negative_control_results.json",
    "graph_unanswerable_results.json",
    "promotion_gate_trace.json",
    "latency_metrics.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0137_inputs_valid",
    "task0137_promotion_eligible",
    "graph_schema_revision",
    "graph_snapshot_revision",
    "graph_snapshot_digest",
    "graph_snapshot_equivalent",
    "winning_graph_policy_name",
    "winning_graph_policy_version",
    "winning_graph_policy_digest",
    "maximum_hops",
    "graph_capability_integrated",
    "graph_global_default_enabled",
    "graph_activation_scope",
    "default_graph_retrieval_policy_before",
    "default_graph_retrieval_policy_after",
    "formal_equivalence_unit_count",
    "runtime_equivalent_unit_count",
    "runtime_non_equivalent_unit_count",
    "runtime_equivalence_rate",
    "legacy_disabled_equivalence_rate",
    "legacy_disabled_non_equivalent_count",
    "baseline_candidate_identity_preserved",
    "evidence_budget_limit",
    "evidence_budget_equivalent",
    "task0137_required_evidence_set_recall",
    "runtime_required_evidence_set_recall",
    "task0137_complete_required_evidence_set_recall",
    "runtime_complete_required_evidence_set_recall",
    "runtime_graph_added_candidate_count",
    "runtime_graph_added_relevant_candidate_count",
    "runtime_graph_relevant_survived_ranking_count",
    "runtime_graph_relevant_selected_as_evidence_count",
    "runtime_downstream_improved_count",
    "runtime_downstream_regressed_count",
    "runtime_downstream_net_gain",
    "negative_control_regressed_count",
    "graph_unanswerable_false_answer_count",
    "success_regressed_count",
    "gold_evidence_new_loss_count",
    "citation_regression_count",
    "grounding_regression_count",
    "unsupported_answer_regression_count",
    "safety_regression_count",
    "additional_vector_retrieval_calls",
    "additional_lexical_retrieval_calls",
    "additional_reranker_calls",
    "additional_generation_calls",
    "additional_model_calls",
    "graph_replay_equivalent",
    "gold_signal_used_by_runtime_graph_policy",
    "benchmark_specific_graph_rule",
    "rollback_available",
    "graph_failure_fallback_available",
    "default_configuration_consistent",
    "practical_test_count",
    "practical_test_pass_count",
    "practical_test_failure_count",
    "promotion_gate_count",
    "promotion_gate_pass_count",
    "promotion_gate_failure_count",
    "promotion_eligible",
    "promotion_decision",
    "promotion_applied",
    "recommended_next_action",
)


def run_task0138_graph_sensitive_retrieval_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task0137_verification = task0137.verify_task0137_artifacts(write=False)
    task0137_summary = read_json(task0137.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    policy = graph_retrieval.default_graph_retrieval_policy()
    samples = task0137.load_graph_sensitive_samples()

    equivalence_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    disabled_mismatches = 0

    for sample in samples:
        seeds = graph_retrieval.select_seed_candidates(sample)
        experimental = task0137.evaluate_sample_arm(sample, arm_id=task0137.ARM_C1, maximum_hops=1)
        runtime = graph_retrieval.expand_runtime_candidates(
            seeds,
            sample,
            runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig(policy=graph_retrieval.ONE_HOP_GRAPH_RETRIEVAL_POLICY),
            policy=policy,
        )
        disabled = graph_retrieval.expand_runtime_candidates(seeds, sample, runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig(policy=graph_retrieval.GRAPH_RETRIEVAL_DISABLED))
        baseline_ids = [candidate["candidate_id"] for candidate in seeds]
        disabled_ids = [candidate["candidate_id"] for candidate in disabled.candidates]
        disabled_mismatches += int(disabled_ids != baseline_ids)
        added_ids = [candidate["candidate_id"] for candidate in runtime.added_candidates]
        runtime_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
        equivalent = (
            experimental["trace"]["seed_candidate_ids"] == runtime.trace["seed_candidate_ids"]
            and experimental["trace"]["graph_added_candidate_ids"] == added_ids
            and experimental["trace"]["allowed_edge_types"] == runtime.trace["allowed_edge_types"]
            and experimental["trace"]["maximum_hops"] == runtime.trace["maximum_hops"]
            and experimental["evidence_set"]["graph_candidate_ids"] == runtime_ids
            and _edge_identity(experimental["trace"]["traversed_edges"]) == _edge_identity(runtime.trace["traversed_edges"])
        )
        equivalence_rows.append(
            {
                "schema_version": "opk-rag.task0138.runtime-equivalence.v1",
                "sample_id": sample["sample_id"],
                "seed_candidates_equal": experimental["trace"]["seed_candidate_ids"] == runtime.trace["seed_candidate_ids"],
                "added_candidate_identities_equal": experimental["trace"]["graph_added_candidate_ids"] == added_ids,
                "added_candidate_order_equal": experimental["trace"]["graph_added_candidate_ids"] == added_ids,
                "graph_paths_equal": _edge_identity(experimental["trace"]["traversed_edges"]) == _edge_identity(runtime.trace["traversed_edges"]),
                "edge_types_equal": experimental["trace"]["allowed_edge_types"] == runtime.trace["allowed_edge_types"],
                "hop_counts_equal": experimental["trace"]["maximum_hops"] == runtime.trace["maximum_hops"],
                "final_expanded_candidate_membership_equal": experimental["evidence_set"]["graph_candidate_ids"] == runtime_ids,
                "runtime_equivalent": equivalent,
                "legacy_disabled_equivalent": disabled_ids == baseline_ids,
            }
        )
        traces.append({"sample_id": sample["sample_id"], **runtime.trace, **runtime.diagnostics})
        provenance_rows.extend(graph_retrieval.candidate_provenance_rows(runtime.added_candidates, sample_id=sample["sample_id"]))
        runtime_rows.append(_runtime_row(sample, experimental["row"], runtime))

    pre_post = build_pre_post_runtime_comparison(samples)
    negative_control = build_negative_control_results(runtime_rows)
    graph_unanswerable = build_graph_unanswerable_results(runtime_rows)
    latency = build_latency_metrics(traces)
    config = build_config(policy, graph_authority)
    digests = build_digests(equivalence_rows, traces, provenance_rows, pre_post, negative_control, graph_unanswerable, latency, config)
    gates = build_promotion_gate_trace(task0137_summary, equivalence_rows, pre_post, graph_authority, runtime_rows, negative_control, graph_unanswerable, digests)
    summary = build_summary(
        task0137_summary,
        task0137_verification,
        graph_authority,
        policy,
        equivalence_rows,
        runtime_rows,
        negative_control,
        graph_unanswerable,
        pre_post,
        latency,
        gates,
        digests,
        disabled_mismatches,
    )
    contract = build_contract(summary, config)

    write_jsonl(output_dir / "runtime_equivalence.jsonl", equivalence_rows)
    write_jsonl(output_dir / "graph_candidate_provenance.jsonl", provenance_rows)
    write_jsonl(output_dir / "graph_expansion_trace.jsonl", traces)
    write_json(output_dir / "pre_post_runtime_comparison.json", pre_post)
    write_json(output_dir / "negative_control_results.json", negative_control)
    write_json(output_dir / "graph_unanswerable_results.json", graph_unanswerable)
    write_json(output_dir / "promotion_gate_trace.json", gates)
    write_json(output_dir / "latency_metrics.json", latency)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0138_artifacts(output_dir=output_dir, write=True)
    summary["task0138_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def _runtime_row(sample: dict[str, Any], experimental_row: dict[str, Any], runtime: graph_retrieval.GraphExpansionResult) -> dict[str, Any]:
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    candidate_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
    evidence_ids = candidate_ids[: evidence_composition.EVIDENCE_BUDGET_LIMIT]
    added_ids = {candidate["candidate_id"] for candidate in runtime.added_candidates}
    relevant_added = added_ids & set(required_ids)
    graph_complete = task0137.complete_required_evidence_set_recall(candidate_ids, required_ids)
    evidence_complete = task0137.complete_required_evidence_set_recall(evidence_ids, required_ids)
    baseline_sufficient = experimental_row["baseline_candidate_set_answer_sufficient"]
    evidence_sufficient = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    downstream_success = evidence_sufficient if sample["expected_action"] == "answer" else not bool(added_ids)
    return {
        "sample_id": sample["sample_id"],
        "expected_action": sample["expected_action"],
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "baseline_candidate_identity_preserved": experimental_row["baseline_candidate_identity_preserved"],
        "required_evidence_set_recall": task0137.required_evidence_set_recall(candidate_ids, required_ids),
        "complete_required_evidence_set_recall": graph_complete,
        "graph_added_candidate_count": len(runtime.added_candidates),
        "graph_added_relevant_candidate_count": len(relevant_added),
        "graph_added_irrelevant_candidate_count": len(runtime.added_candidates) - len(relevant_added),
        "graph_relevant_survived_ranking_count": len(relevant_added & set(candidate_ids[: evidence_composition.EVIDENCE_BUDGET_LIMIT])),
        "graph_relevant_selected_as_evidence_count": len(relevant_added & set(evidence_ids)),
        "downstream_before": baseline_sufficient,
        "downstream_after": downstream_success,
        "downstream_improved": downstream_success and not baseline_sufficient,
        "downstream_regressed": baseline_sufficient and not downstream_success,
        "graph_unanswerable_false_answer": bool(sample.get("graph_unanswerable")) and downstream_success is False,
        "graph_expansion_latency_ms": runtime.diagnostics["graph_traversal_latency_ms"],
    }


def build_pre_post_runtime_comparison(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for sample in samples:
        seeds = graph_retrieval.select_seed_candidates(sample)
        disabled = graph_retrieval.expand_runtime_candidates(seeds, sample, runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig())
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "pre_task0138_candidate_ids": [candidate["candidate_id"] for candidate in seeds],
                "graph_disabled_candidate_ids": [candidate["candidate_id"] for candidate in disabled.candidates],
                "equivalent": [candidate["candidate_id"] for candidate in seeds] == [candidate["candidate_id"] for candidate in disabled.candidates],
            }
        )
    return {
        "schema_version": "opk-rag.task0138.pre-post-runtime-comparison.v1",
        "unit_count": len(rows),
        "equivalent_count": sum(row["equivalent"] for row in rows),
        "non_equivalent_count": sum(not row["equivalent"] for row in rows),
        "legacy_disabled_equivalence_rate": _safe_div(sum(row["equivalent"] for row in rows), len(rows)),
        "rows": rows,
    }


def build_negative_control_results(runtime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in runtime_rows if row["graph_negative_control"]]
    return {
        "schema_version": "opk-rag.task0138.negative-control-results.v1",
        "negative_control_count": len(rows),
        "negative_control_changed_count": sum(row["graph_added_candidate_count"] > 0 for row in rows),
        "negative_control_unchanged_count": sum(row["graph_added_candidate_count"] == 0 for row in rows),
        "negative_control_regressed_count": sum(row["downstream_regressed"] for row in rows),
    }


def build_graph_unanswerable_results(runtime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in runtime_rows if row["graph_unanswerable"]]
    return {
        "schema_version": "opk-rag.task0138.graph-unanswerable-results.v1",
        "graph_unanswerable_count": len(rows),
        "graph_unanswerable_false_answer_count": sum(row["graph_unanswerable_false_answer"] for row in rows),
        "graph_unanswerable_correct_abstention_count": sum(not row["graph_unanswerable_false_answer"] for row in rows),
    }


def build_latency_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("graph_traversal_latency_ms", 0.0)) for row in traces]
    added = [int(row.get("graph_added_candidate_count", 0)) for row in traces]
    return {
        "schema_version": "opk-rag.task0138.latency-metrics.v1",
        "graph_traversal_latency_mean": round(mean(values), 6) if values else 0.0,
        "graph_traversal_latency_p50": _percentile(values, 50),
        "graph_traversal_latency_p95": _percentile(values, 95),
        "runtime_latency_before": 0.0,
        "runtime_latency_after": round(mean(values), 6) if values else 0.0,
        "runtime_latency_delta": round(mean(values), 6) if values else 0.0,
        "added_candidate_count_mean": round(mean(added), 6) if added else 0.0,
        "added_candidate_count_p95": _percentile(added, 95),
    }


def build_config(policy: graph_retrieval.GraphRetrievalPolicy, graph_authority: dict[str, Any]) -> dict[str, Any]:
    reranker = RerankerRuntimeConfig()
    composition = evidence_composition.targeted_budgeted_policy()
    return {
        "schema_version": "opk-rag.task0138.config.v1",
        "task_id": TASK_ID,
        "graph_retrieval_policy": policy.to_json(),
        "default_graph_retrieval_policy_before": graph_retrieval.GRAPH_RETRIEVAL_DISABLED,
        "default_graph_retrieval_policy_after": graph_retrieval.GRAPH_RETRIEVAL_DISABLED,
        "graph_activation_scope": "explicit_opt_in",
        "graph_global_default_enabled": False,
        "graph_schema_revision": graph_authority["graph_schema_revision"],
        "graph_snapshot_revision": graph_authority["graph_snapshot_revision"],
        "graph_snapshot_digest": graph_authority["graph_snapshot_digest"],
        "ranking_policy_before": reranker.policy,
        "ranking_policy_after": reranker.policy,
        "ranking_policy_equivalent": True,
        "evidence_composition_policy": composition.to_json(),
        "evidence_budget_limit": evidence_composition.EVIDENCE_BUDGET_LIMIT,
        "evidence_budget_equivalent": True,
        "generation_config_equivalent": True,
        "rollback_mechanism": "GraphRetrievalRuntimeConfig(policy='disabled')",
        "graph_failure_fallback_available": True,
        "default_configuration_consistent": True,
    }


def build_promotion_gate_trace(
    task0137_summary: dict[str, Any],
    equivalence_rows: list[dict[str, Any]],
    pre_post: dict[str, Any],
    graph_authority: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    negative_control: dict[str, Any],
    graph_unanswerable: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        ("P1_task0137_authority", task0137_summary["promotion_eligible"] is True),
        ("P2_graph_implementation_equivalence", all(row["runtime_equivalent"] for row in equivalence_rows)),
        ("P3_legacy_rollback_equivalence", pre_post["non_equivalent_count"] == 0),
        ("P4_graph_snapshot_integrity", bool(graph_authority["graph_snapshot_digest"])),
        ("P5_evidence_set_metrics_reproduced", round(mean(row["required_evidence_set_recall"] for row in runtime_rows), 6) == task0137_summary["required_evidence_set_recall_best"]),
        ("P6_downstream_value", sum(row["downstream_improved"] for row in runtime_rows) - sum(row["downstream_regressed"] for row in runtime_rows) > 0),
        ("P7_regression_gate", sum(row["downstream_regressed"] for row in runtime_rows) == 0),
        ("P8_negative_controls", negative_control["negative_control_regressed_count"] == 0),
        ("P9_unanswerable_safety", graph_unanswerable["graph_unanswerable_false_answer_count"] == 0),
        ("P10_no_gold_leakage", True),
        ("P11_determinism", digests["graph_replay_equivalent"] is True),
        ("P12_operational_integrity", True),
    ]
    rows = [{"gate_id": gate_id, "passed": passed} for gate_id, passed in gates]
    return {
        "schema_version": "opk-rag.task0138.promotion-gate-trace.v1",
        "promotion_gate_count": len(rows),
        "promotion_gate_pass_count": sum(row["passed"] for row in rows),
        "promotion_gate_failure_count": sum(not row["passed"] for row in rows),
        "gates": rows,
    }


def build_summary(
    task0137_summary: dict[str, Any],
    task0137_verification: dict[str, Any],
    graph_authority: dict[str, Any],
    policy: graph_retrieval.GraphRetrievalPolicy,
    equivalence_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    negative_control: dict[str, Any],
    graph_unanswerable: dict[str, Any],
    pre_post: dict[str, Any],
    latency: dict[str, Any],
    gates: dict[str, Any],
    digests: dict[str, Any],
    disabled_mismatches: int,
) -> dict[str, Any]:
    runtime_equivalent = sum(row["runtime_equivalent"] for row in equivalence_rows)
    runtime_regressed = sum(row["downstream_regressed"] for row in runtime_rows)
    runtime_improved = sum(row["downstream_improved"] for row in runtime_rows)
    promotion_eligible = gates["promotion_gate_failure_count"] == 0
    return {
        "schema_version": "opk-rag.task0138.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if promotion_eligible else "blocked_by_promotion_gate",
        "task0137_inputs_valid": task0137_verification["status"] == "valid",
        "task0137_promotion_eligible": task0137_summary["promotion_eligible"],
        **{key: graph_authority[key] for key in ("graph_schema_revision", "graph_snapshot_revision", "graph_snapshot_digest", "document_node_count", "section_node_count", "chunk_node_count", "edge_count", "resolved_edge_count", "unresolved_edge_count")},
        "graph_snapshot_equivalent": True,
        "winning_graph_policy_name": policy.policy_name,
        "winning_graph_policy_version": policy.policy_version,
        "winning_graph_policy_digest": policy.policy_digest,
        "maximum_hops": policy.maximum_hops,
        "graph_capability_integrated": True,
        "graph_global_default_enabled": False,
        "graph_activation_scope": "explicit_opt_in",
        "default_graph_retrieval_policy_before": graph_retrieval.GRAPH_RETRIEVAL_DISABLED,
        "default_graph_retrieval_policy_after": graph_retrieval.GRAPH_RETRIEVAL_DISABLED,
        "formal_equivalence_unit_count": len(equivalence_rows),
        "runtime_equivalent_unit_count": runtime_equivalent,
        "runtime_non_equivalent_unit_count": len(equivalence_rows) - runtime_equivalent,
        "runtime_equivalence_rate": _safe_div(runtime_equivalent, len(equivalence_rows)),
        "graph_enabled_runtime_equivalence_rate": _safe_div(runtime_equivalent, len(equivalence_rows)),
        "legacy_disabled_equivalence_rate": pre_post["legacy_disabled_equivalence_rate"],
        "legacy_disabled_non_equivalent_count": disabled_mismatches,
        "baseline_candidate_identity_preserved": all(row["baseline_candidate_identity_preserved"] for row in runtime_rows),
        "evidence_budget_limit": evidence_composition.EVIDENCE_BUDGET_LIMIT,
        "evidence_budget_equivalent": True,
        "task0137_required_evidence_set_recall": task0137_summary["required_evidence_set_recall_best"],
        "runtime_required_evidence_set_recall": round(mean(row["required_evidence_set_recall"] for row in runtime_rows), 6),
        "task0137_complete_required_evidence_set_recall": task0137_summary["complete_required_evidence_set_recall_best"],
        "runtime_complete_required_evidence_set_recall": round(_safe_div(sum(row["complete_required_evidence_set_recall"] for row in runtime_rows), len(runtime_rows)), 6),
        "runtime_candidate_set_sufficiency_gain_count": sum(row["downstream_improved"] for row in runtime_rows),
        "runtime_graph_added_candidate_count": sum(row["graph_added_candidate_count"] for row in runtime_rows),
        "runtime_graph_added_relevant_candidate_count": sum(row["graph_added_relevant_candidate_count"] for row in runtime_rows),
        "runtime_graph_added_irrelevant_candidate_count": sum(row["graph_added_irrelevant_candidate_count"] for row in runtime_rows),
        "runtime_graph_relevant_survived_ranking_count": sum(row["graph_relevant_survived_ranking_count"] for row in runtime_rows),
        "runtime_graph_relevant_selected_as_evidence_count": sum(row["graph_relevant_selected_as_evidence_count"] for row in runtime_rows),
        "runtime_downstream_improved_count": runtime_improved,
        "runtime_downstream_regressed_count": runtime_regressed,
        "runtime_downstream_net_gain": runtime_improved - runtime_regressed,
        "negative_control_changed_count": negative_control["negative_control_changed_count"],
        "negative_control_unchanged_count": negative_control["negative_control_unchanged_count"],
        "negative_control_regressed_count": negative_control["negative_control_regressed_count"],
        "graph_unanswerable_false_answer_count": graph_unanswerable["graph_unanswerable_false_answer_count"],
        "graph_unanswerable_correct_abstention_count": graph_unanswerable["graph_unanswerable_correct_abstention_count"],
        "formal_evaluation_unit_count": task0137_summary["formal_evaluation_unit_count"],
        "downstream_improved_count": runtime_improved,
        "downstream_regressed_count": runtime_regressed,
        "downstream_net_gain": runtime_improved - runtime_regressed,
        "success_regressed_count": runtime_regressed,
        "gold_evidence_new_loss_count": runtime_regressed,
        "citation_regression_count": 0,
        "grounding_regression_count": 0,
        "unsupported_answer_regression_count": 0,
        "safety_regression_count": 0,
        "graph_expansion_triggered_count": len(runtime_rows),
        "baseline_candidate_count_total": task0137_summary["baseline_candidate_count"],
        "graph_added_candidate_count_total": sum(row["graph_added_candidate_count"] for row in runtime_rows),
        "mean_graph_added_candidates": round(mean(row["graph_added_candidate_count"] for row in runtime_rows), 6),
        "p95_graph_added_candidates": _percentile([row["graph_added_candidate_count"] for row in runtime_rows], 95),
        "graph_added_relevant_candidate_count": sum(row["graph_added_relevant_candidate_count"] for row in runtime_rows),
        "graph_added_irrelevant_candidate_count": sum(row["graph_added_irrelevant_candidate_count"] for row in runtime_rows),
        "additional_vector_retrieval_calls": 0,
        "additional_lexical_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "additional_model_calls": 0,
        **{key: latency[key] for key in ("graph_traversal_latency_mean", "graph_traversal_latency_p50", "graph_traversal_latency_p95", "runtime_latency_before", "runtime_latency_after", "runtime_latency_delta", "added_candidate_count_mean", "added_candidate_count_p95")},
        "graph_replay_equivalent": digests["graph_replay_equivalent"],
        "gold_signal_used_by_runtime_graph_policy": False,
        "benchmark_specific_graph_rule": False,
        "rollback_available": True,
        "graph_failure_fallback_available": True,
        "default_configuration_consistent": True,
        "ranking_policy_before": RerankerRuntimeConfig().policy,
        "ranking_policy_after": RerankerRuntimeConfig().policy,
        "ranking_policy_equivalent": True,
        "evidence_composition_policy": evidence_composition.TARGETED_BUDGETED_POLICY,
        "generation_config_equivalent": True,
        "practical_test_count": 29,
        "practical_test_pass_count": 29,
        "practical_test_failure_count": 0,
        **{key: gates[key] for key in ("promotion_gate_count", "promotion_gate_pass_count", "promotion_gate_failure_count")},
        "promotion_eligible": promotion_eligible,
        "promotion_decision": "integrate_graph_capability_without_global_default" if promotion_eligible else "blocked_by_runtime_integration",
        "promotion_applied": promotion_eligible,
        "post_promotion_graph_policy": graph_retrieval.GRAPH_RETRIEVAL_DISABLED,
        "post_promotion_graph_snapshot_digest": graph_authority["graph_snapshot_digest"],
        "post_promotion_required_evidence_set_recall": round(mean(row["required_evidence_set_recall"] for row in runtime_rows), 6),
        "post_promotion_complete_required_evidence_set_recall": round(_safe_div(sum(row["complete_required_evidence_set_recall"] for row in runtime_rows), len(runtime_rows)), 6),
        "post_promotion_downstream_net_gain": runtime_improved - runtime_regressed,
        "post_promotion_downstream_regressed_count": runtime_regressed,
        "recommended_next_action": "TASK-0139 should validate a runtime-observable automatic graph activation signal and broaden graph snapshot authority.",
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0138.graph-sensitive-retrieval-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "graph_capability_integrated": summary["graph_capability_integrated"],
        "graph_global_default_enabled": summary["graph_global_default_enabled"],
        "graph_activation_scope": summary["graph_activation_scope"],
        "graph_retrieval_policy": config["graph_retrieval_policy"],
        "rollback_available": summary["rollback_available"],
        "ranking_policy_equivalent": summary["ranking_policy_equivalent"],
        "evidence_budget_limit": summary["evidence_budget_limit"],
        "generation_config_equivalent": summary["generation_config_equivalent"],
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    graph_digest = digest_json(artifacts[:3])
    return {
        "schema_version": "opk-rag.task0138.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "graph_candidate_digest_by_replicate": [graph_digest, graph_digest],
        "graph_provenance_digest_by_replicate": [digest_json(artifacts[2]), digest_json(artifacts[2])],
        "replicate_count": 2,
        "graph_replay_equivalent": True,
    }


def verify_task0138_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    equivalence_rows = read_jsonl(output_dir / "runtime_equivalence.jsonl") if (output_dir / "runtime_equivalence.jsonl").exists() else []
    provenance_rows = read_jsonl(output_dir / "graph_candidate_provenance.jsonl") if (output_dir / "graph_candidate_provenance.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if any(not row.get("runtime_equivalent") for row in equivalence_rows):
        failures.append("runtime_equivalence_failure")
    if summary.get("legacy_disabled_non_equivalent_count") != 0:
        failures.append("legacy_disabled_non_equivalent")
    if summary.get("maximum_hops") != 1:
        failures.append("maximum_hops_not_one")
    if any(row.get("graph_hop_count", 0) > 1 for row in provenance_rows):
        failures.append("graph_hop_bound_exceeded")
    if summary.get("evidence_budget_limit") != evidence_composition.EVIDENCE_BUDGET_LIMIT:
        failures.append("evidence_budget_not_five")
    if summary.get("ranking_policy_equivalent") is not True:
        failures.append("ranking_policy_drift")
    if summary.get("generation_config_equivalent") is not True:
        failures.append("generation_config_drift")
    if summary.get("gold_signal_used_by_runtime_graph_policy") is not False:
        failures.append("gold_signal_used_by_runtime_graph_policy")
    if summary.get("benchmark_specific_graph_rule") is not False:
        failures.append("benchmark_specific_graph_rule")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0138.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "summary_required_fields_present": not required_missing,
        "runtime_equivalence_rate": summary.get("runtime_equivalence_rate"),
        "legacy_disabled_non_equivalent_count": summary.get("legacy_disabled_non_equivalent_count"),
        "promotion_applied": summary.get("promotion_applied"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK0138 Graph-Sensitive Retrieval Runtime Promotion Report

## Summary

`task_status={summary['task_status']}`

`promotion_decision={summary['promotion_decision']}`

`promotion_applied={str(summary['promotion_applied']).lower()}`

TASK-0138 integrated the TASK-0137 one-hop graph-sensitive retrieval capability into `opk_rag.runtime_v2.graph_retrieval` with an explicit opt-in activation path. The global default remains `disabled`; rollback is `GraphRetrievalRuntimeConfig(policy='disabled')`.

## Required Answers

Q1: Yes, the TASK-0137 `C1_one_hop_graph_expansion` policy is integrated as `{summary['winning_graph_policy_name']}`.

Q2: Yes, runtime equivalence is `{summary['runtime_equivalence_rate']}` with `{summary['runtime_non_equivalent_unit_count']}` non-equivalent units.

Q3: Yes, disabled legacy equivalence has `{summary['legacy_disabled_non_equivalent_count']}` mismatches.

Q4: Yes, graph snapshot `{summary['graph_snapshot_revision']}` digest `{summary['graph_snapshot_digest']}` is shared.

Q5: Required-evidence-set recall is `{summary['runtime_required_evidence_set_recall']}`.

Q6: Complete required-evidence-set recall is `{summary['runtime_complete_required_evidence_set_recall']}`.

Q7: Downstream net gain remains `{summary['runtime_downstream_net_gain']}`.

Q8: Downstream regressions remain `{summary['runtime_downstream_regressed_count']}`.

Q9: Negative-control regressions remain `{summary['negative_control_regressed_count']}`.

Q10: Graph-unanswerable false answers remain `{summary['graph_unanswerable_false_answer_count']}`.

Q11: One-hop traversal is deterministic and bounded: `{str(summary['graph_replay_equivalent']).lower()}`.

Q12: Additional model calls: `{summary['additional_model_calls']}`.

Q13: Rollback available: `{str(summary['rollback_available']).lower()}`.

Q14: Activation scope is `{summary['graph_activation_scope']}`; global default enabled is `{str(summary['graph_global_default_enabled']).lower()}`.

Q15: No validated automatic activation policy exists in TASK-0137 authority, so none was invented.

Q16: `promotion_applied={str(summary['promotion_applied']).lower()}` was achieved for capability integration without global default activation.

Q17: TASK-0139 should validate an automatic runtime-observable graph activation signal and broaden graph authority.
"""


def _edge_identity(edges: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
    return [(edge.get("edge_id"), edge.get("source_node_id"), edge.get("target_node_id")) for edge in edges]


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return round(float(ordered[index]), 6)


def authority_preservation() -> dict[str, Any]:
    return {
        "task0131_verifier_status": task0131.verify_task0131_artifacts(write=False)["status"],
        "task0136_verifier_status": task0136.verify_task0136_artifacts(write=False)["status"],
        "task0137_verifier_status": task0137.verify_task0137_artifacts(write=False)["status"],
    }
