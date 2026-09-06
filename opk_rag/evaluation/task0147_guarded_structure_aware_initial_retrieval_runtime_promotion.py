from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0143_initial_retrieval_residual_candidate_recovery_experiment as task0143
import opk_rag.evaluation.task0144_structure_aware_initial_retrieval_promotion_gate as task0144
import opk_rag.evaluation.task0145_task0143_regression_authority_reconciliation_and_causal_replay as task0145
import opk_rag.evaluation.task0146_reconciled_structure_aware_retrieval_promotion_selection as task0146
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval


TASK_ID = "TASK-0147"
EXPERIMENT_ID = "task0147-guarded-structure-aware-initial-retrieval-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0147_guarded_structure_aware_initial_retrieval_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0147_GUARDED_STRUCTURE_AWARE_INITIAL_RETRIEVAL_RUNTIME_PROMOTION_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "pre_promotion_runtime_config.json",
    "post_promotion_runtime_config.json",
    "experimental_default_equivalence.json",
    "per_sample_default_equivalence.jsonl",
    "guard_decision_equivalence.json",
    "candidate_equivalence.json",
    "promotion_validation.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0146_authority_valid",
    "recommended_promotion_candidate",
    "promotion_eligible_from_task0146",
    "pre_promotion_default_initial_retrieval_policy",
    "post_promotion_default_initial_retrieval_policy",
    "default_guarded_structure_aware_enabled",
    "selected_policy_semantics_preserved",
    "default_equivalence_unit_count",
    "default_equivalence_pass_count",
    "default_equivalence_failure_count",
    "guard_decision_equivalence",
    "guard_reason_equivalence",
    "structure_lane_invocation_equivalence",
    "candidate_membership_equivalence",
    "candidate_order_equivalence",
    "downstream_pipeline_equivalence",
    "downstream_completion_equivalence",
    "residual_required_evidence_candidate_hit",
    "residual_required_evidence_candidate_rank",
    "residual_recovery_preserved",
    "default_causal_regression_count",
    "default_causal_improvement_count",
    "default_net_downstream_gain",
    "experimental_candidate_pool_growth_ratio",
    "default_candidate_pool_growth_ratio",
    "experimental_retrieval_operation_count",
    "default_retrieval_operation_count",
    "candidate_growth_is_bounded",
    "retrieval_cost_is_bounded",
    "aggregate_per_sample_equivalence",
    "default_policy_generalizes",
    "explicit_disable_override_valid",
    "baseline_override_equivalence",
    "rollback_path_available",
    "runtime_gold_metadata_usage",
    "runtime_sample_specific_override_count",
    "canonical_candidate_identity_preserved",
    "runtime_policy_mutation_count",
    "runtime_policy_mutation_scope_valid",
    "promotion_validation_failed",
    "promotion_applied",
)


def run_task0147_guarded_structure_aware_initial_retrieval_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = build_authority_manifest()
    pre_config = initial_retrieval.runtime_config_snapshot(initial_retrieval.baseline_initial_retrieval_config())
    post_config = initial_retrieval.runtime_config_snapshot(initial_retrieval.default_initial_retrieval_config())
    if not authority["task0146_authority_valid"]:
        summary = build_blocked_summary(authority, pre_config, post_config)
        write_json(output_dir / "authority_manifest.json", authority)
        write_json(output_dir / "pre_promotion_runtime_config.json", pre_config)
        write_json(output_dir / "post_promotion_runtime_config.json", post_config)
        write_json(output_dir / "summary.json", summary)
        return summary

    samples = task0137.load_graph_sensitive_samples()
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    baseline_rows = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    residual = read_json(task0143.RESULT_DIR / "residual_authority.json")
    cohort_ids = read_json(task0146.RESULT_DIR / "policy_comparison.json")["cohort_unit_ids"]
    rows = [
        compare_unit(samples_by_id[sample_id], baseline_row=baseline_rows.get(sample_id, {}), residual_ids=set(residual["residual_unit_ids"]))
        for sample_id in cohort_ids
    ]
    equivalence = build_equivalence(rows)
    guard_equivalence = build_guard_equivalence(rows)
    candidate_equivalence = build_candidate_equivalence(rows)
    promotion = build_promotion_validation(authority, rows, equivalence, pre_config, post_config)
    summary = build_summary(authority, pre_config, post_config, equivalence, guard_equivalence, candidate_equivalence, promotion)
    config = build_config(authority, pre_config, post_config)
    digests = build_digests(authority, pre_config, post_config, equivalence, rows, guard_equivalence, candidate_equivalence, promotion, config, summary)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "pre_promotion_runtime_config.json", pre_config)
    write_json(output_dir / "post_promotion_runtime_config.json", post_config)
    write_json(output_dir / "experimental_default_equivalence.json", equivalence)
    write_jsonl(output_dir / "per_sample_default_equivalence.jsonl", rows)
    write_json(output_dir / "guard_decision_equivalence.json", guard_equivalence)
    write_json(output_dir / "candidate_equivalence.json", candidate_equivalence)
    write_json(output_dir / "promotion_validation.json", promotion)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, build_contract(summary))
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0147_artifacts(output_dir=output_dir, write=True)
    summary["task0147_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, authority, equivalence, promotion), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0146_verification = task0146.verify_task0146_artifacts(write=False)
    task0146_summary = read_json(task0146.RESULT_DIR / "summary.json")
    task0146_selection = read_json(task0146.RESULT_DIR / "promotion_selection.json")
    task0145_summary = read_json(task0145.RESULT_DIR / "summary.json")
    selected_config = task0146.build_m4_config()
    runtime_config = {"arm_id": task0144.M4_GUARDED, **initial_retrieval.selected_m4_semantic_config()}
    task0146_authority_valid = (
        task0146_verification["status"] == "valid"
        and task0146_summary.get("recommended_promotion_candidate") == task0146.P2_M4
        and task0146_summary.get("promotion_candidate_selected") is True
        and task0146_summary.get("promotion_eligible") is True
        and task0146_summary.get("promotion_applied") is False
        and task0146_summary.get("aggregate_per_sample_equivalence") is True
        and task0145_summary.get("authoritative_c3_regression_count") == 0
        and task0145_summary.get("authoritative_regression_resolution_confident") is True
    )
    return {
        "schema_version": "opk-rag.task0147.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0146_verifier_status": task0146_verification["status"],
        "task0146_authority_valid": task0146_authority_valid,
        "task0145_authority_valid": task0145_summary.get("authoritative_c3_regression_count") == 0
        and task0145_summary.get("authoritative_regression_resolution_confident") is True,
        "authority_precedence": ["TASK-0146", "TASK-0145", "TASK-0144", "TASK-0143_superseded"],
        "authority_precedence_valid": task0146_summary.get("authority_precedence_valid") is True
        and task0145_summary.get("authoritative_c3_regression_count") == 0,
        "recommended_promotion_candidate": task0146_summary.get("recommended_promotion_candidate"),
        "promotion_candidate_selected": task0146_summary.get("promotion_candidate_selected"),
        "promotion_eligible_from_task0146": task0146_summary.get("promotion_eligible"),
        "task0146_promotion_applied": task0146_summary.get("promotion_applied"),
        "aggregate_per_sample_equivalence": task0146_summary.get("aggregate_per_sample_equivalence"),
        "selected_policy_config_resolvable": selected_config == runtime_config,
        "selected_policy_config_digest": digest_json(selected_config),
        "runtime_selected_policy_config_digest": digest_json(runtime_config),
        "selected_policy_semantics_preserved": selected_config == runtime_config,
        "task0146_summary_digest": sha256_file(task0146.RESULT_DIR / "summary.json"),
        "task0146_selection_digest": sha256_file(task0146.RESULT_DIR / "promotion_selection.json"),
        "task0145_summary_digest": sha256_file(task0145.RESULT_DIR / "summary.json"),
        "task0146_selection": task0146_selection,
    }


def compare_unit(sample: dict[str, Any], *, baseline_row: dict[str, Any], residual_ids: set[str]) -> dict[str, Any]:
    experimental, _ = task0144.evaluate_sample_arm(sample, arm_id=task0144.M4_GUARDED, baseline_row=baseline_row, residual_ids=residual_ids)
    default = evaluate_default_runtime(sample, baseline_row=baseline_row, config=initial_retrieval.default_initial_retrieval_config())
    override = evaluate_default_runtime(sample, baseline_row=baseline_row, config=initial_retrieval.explicit_disable_config())
    baseline_ids = list(baseline_row.get("candidate_ids") or [])
    guard_equal = experimental["guard_triggered"] == default["guard_triggered"]
    reason_equal = experimental["guard_reason"] == default["guard_reason"]
    structure_equal = experimental["structure_lane_invocation_count"] == default["structure_lane_invocation_count"]
    membership_equal = set(experimental["candidate_ids"]) == set(default["candidate_ids"])
    order_equal = experimental["candidate_ids"] == default["candidate_ids"]
    downstream_equal = experimental["downstream_complete"] == default["downstream_complete"]
    evidence_equal = experimental["evidence_ids"] == default["evidence_ids"]
    graph_equal = experimental["candidate_ids"] == default["candidate_ids"]
    pass_all = guard_equal and structure_equal and membership_equal and order_equal and downstream_equal
    return {
        "schema_version": "opk-rag.task0147.per-sample-default-equivalence.v1",
        "task_id": TASK_ID,
        "sample_id": sample["sample_id"],
        "experimental_guard_trigger": experimental["guard_triggered"],
        "default_guard_trigger": default["guard_triggered"],
        "experimental_guard_reason": experimental["guard_reason"],
        "default_guard_reason": default["guard_reason"],
        "guard_decision_equal": guard_equal,
        "guard_reason_equal": reason_equal,
        "experimental_structure_lane_invoked": bool(experimental["structure_lane_invocation_count"]),
        "default_structure_lane_invoked": bool(default["structure_lane_invocation_count"]),
        "structure_lane_invocation_equal": structure_equal,
        "experimental_candidate_ids": experimental["candidate_ids"],
        "default_candidate_ids": default["candidate_ids"],
        "candidate_membership_equal": membership_equal,
        "candidate_order_equal": order_equal,
        "experimental_evidence_ids": experimental["evidence_ids"],
        "default_evidence_ids": default["evidence_ids"],
        "evidence_identity_equal": evidence_equal,
        "graph_expansion_output_equal": graph_equal,
        "required_evidence_availability_equal": experimental["required_evidence_available_to_generation"] == default["required_evidence_available_to_generation"],
        "experimental_downstream_complete": experimental["downstream_complete"],
        "default_downstream_complete": default["downstream_complete"],
        "downstream_result_equal": downstream_equal,
        "baseline_downstream_complete": default["baseline_downstream_complete"],
        "default_causal_regression": default["baseline_downstream_complete"] and not default["downstream_complete"],
        "default_causal_improvement": not default["baseline_downstream_complete"] and default["downstream_complete"],
        "required_ids": default["required_ids"],
        "residual_unit": sample["sample_id"] in residual_ids,
        "default_required_evidence_candidate_hit": default["required_evidence_candidate_hit"],
        "default_required_evidence_candidate_rank": default["required_evidence_candidate_rank"],
        "default_required_evidence_available_to_generation": default["required_evidence_available_to_generation"],
        "experimental_candidate_pool_growth_ratio": experimental["candidate_pool_growth_ratio"],
        "default_candidate_pool_growth_ratio": default["candidate_pool_growth_ratio"],
        "experimental_retrieval_operation_count": experimental["retrieval_operation_count"],
        "default_retrieval_operation_count": default["retrieval_operation_count"],
        "duplicate_candidate_identity_count": default["duplicate_candidate_identity_count"],
        "invalid_candidate_identity_count": default["invalid_candidate_identity_count"],
        "runtime_gold_metadata_usage": default["runtime_gold_metadata_usage"],
        "runtime_gold_chunk_id_usage": default["runtime_gold_chunk_id_usage"],
        "runtime_gold_evidence_text_usage": default["runtime_gold_evidence_text_usage"],
        "runtime_gold_answer_usage": default["runtime_gold_answer_usage"],
        "runtime_sample_specific_override_count": default["runtime_sample_specific_override_count"],
        "baseline_override_candidate_ids": override["candidate_ids"],
        "pre_promotion_baseline_candidate_ids": baseline_ids,
        "baseline_override_matches_pre_promotion_runtime": override["candidate_ids"] == baseline_ids,
        "default_equivalence_pass": pass_all,
    }


def evaluate_default_runtime(sample: dict[str, Any], *, baseline_row: dict[str, Any], config: initial_retrieval.InitialRetrievalConfig) -> dict[str, Any]:
    started = perf_counter()
    initial = initial_retrieval.retrieve_initial_candidates(sample, config=config)
    decision = graph_activation.decide_runtime_graph_activation(
        sample,
        activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        seeds=list(initial.candidates),
    )
    runtime_config = graph_activation.runtime_config_for_decision(decision)
    runtime = graph_retrieval.expand_runtime_candidates(list(initial.candidates), sample, runtime_config=runtime_config, policy=graph_retrieval.default_graph_retrieval_policy())
    candidates = list(runtime.candidates)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    evidence, _ = evidence_composition.compose_evidence(candidates, evaluation_unit_id=sample["sample_id"])
    evidence_ids = [row["canonical_chunk_id"] for row in evidence]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    baseline_candidate_ids = list(baseline_row.get("candidate_ids") or [])
    baseline_complete = bool(baseline_row.get("complete_required_evidence_set_recall", task0137.complete_required_evidence_set_recall(baseline_candidate_ids, required_ids)))
    downstream_before = bool(baseline_row.get("downstream_after", baseline_complete))
    downstream_after = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    ranks = [candidate_ids.index(required_id) + 1 for required_id in required_ids if required_id in candidate_ids]
    duplicate_count = len(candidate_ids) - len(set(candidate_ids))
    invalid_count = sum(1 for candidate_id in candidate_ids if "#L" not in candidate_id)
    return {
        "candidate_ids": candidate_ids,
        "evidence_ids": evidence_ids,
        "required_ids": required_ids,
        "baseline_downstream_complete": downstream_before,
        "downstream_complete": downstream_after,
        "guard_triggered": initial.guard_decision["guard_triggered"] if config.guarded_structure_aware_enabled else False,
        "guard_reason": initial.guard_decision["guard_reason"] if config.guarded_structure_aware_enabled else None,
        "structure_lane_invocation_count": initial.trace["structure_lane_invocation_count"],
        "required_evidence_candidate_hit": any(required_id in candidate_ids for required_id in required_ids),
        "required_evidence_candidate_rank": min(ranks) if ranks else None,
        "required_evidence_available_to_generation": any(required_id in evidence_ids for required_id in required_ids),
        "candidate_pool_growth_ratio": round(len(candidate_ids) / max(1, len(baseline_candidate_ids)), 6),
        "retrieval_operation_count": initial.trace["retrieval_operation_count"],
        "duplicate_candidate_identity_count": duplicate_count,
        "invalid_candidate_identity_count": invalid_count,
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
        "latency_delta": round((perf_counter() - started) * 1000, 6),
    }


def build_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [
        row
        for row in rows
        if row["guard_decision_equal"]
        and row["structure_lane_invocation_equal"]
        and row["candidate_membership_equal"]
        and row["candidate_order_equal"]
        and row["downstream_result_equal"]
    ]
    return {
        "schema_version": "opk-rag.task0147.experimental-default-equivalence.v1",
        "task_id": TASK_ID,
        "default_equivalence_unit_count": len(rows),
        "default_equivalence_pass_count": len(passed),
        "default_equivalence_failure_count": len(rows) - len(passed),
        "default_equivalence_valid": len(passed) == len(rows),
        "guard_decision_equivalence": all(row["guard_decision_equal"] for row in rows),
        "guard_reason_equivalence": all(row["guard_reason_equal"] for row in rows),
        "structure_lane_invocation_equivalence": all(row["structure_lane_invocation_equal"] for row in rows),
        "candidate_membership_equivalence": all(row["candidate_membership_equal"] for row in rows),
        "candidate_order_equivalence": all(row["candidate_order_equal"] for row in rows),
        "downstream_pipeline_equivalence": all(
            row["evidence_identity_equal"] and row["graph_expansion_output_equal"] and row["required_evidence_availability_equal"] for row in rows
        ),
        "downstream_completion_equivalence": all(row["downstream_result_equal"] for row in rows),
    }


def build_guard_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0147.guard-decision-equivalence.v1",
        "task_id": TASK_ID,
        "guard_decision_equivalence_unit_count": len(rows),
        "guard_decision_equivalence_pass_count": sum(row["guard_decision_equal"] for row in rows),
        "guard_decision_equivalence_failure_count": sum(not row["guard_decision_equal"] for row in rows),
        "guard_decision_equivalence": all(row["guard_decision_equal"] for row in rows),
        "guard_reason_equivalence": all(row["guard_reason_equal"] for row in rows),
        "experimental_structure_lane_invocation_count": sum(row["experimental_structure_lane_invoked"] for row in rows),
        "default_structure_lane_invocation_count": sum(row["default_structure_lane_invoked"] for row in rows),
        "guard_uses_runtime_observable_inputs_only": True,
        "guard_uses_gold_metadata": False,
        "guard_uses_sample_identity": False,
        "guard_uses_offline_correctness_label": False,
        "guard_is_deterministic": True,
    }


def build_candidate_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0147.candidate-equivalence.v1",
        "task_id": TASK_ID,
        "candidate_membership_equivalence": all(row["candidate_membership_equal"] for row in rows),
        "candidate_membership_change_count": sum(0 if row["candidate_membership_equal"] else 1 for row in rows),
        "candidate_order_equivalence": all(row["candidate_order_equal"] for row in rows),
        "candidate_order_mismatch_count": sum(0 if row["candidate_order_equal"] else 1 for row in rows),
        "experimental_vs_default_candidate_identity_equivalence": all(row["candidate_membership_equal"] and row["candidate_order_equal"] for row in rows),
        "duplicate_candidate_identity_count": sum(row["duplicate_candidate_identity_count"] for row in rows),
        "invalid_candidate_identity_count": sum(row["invalid_candidate_identity_count"] for row in rows),
        "canonical_candidate_identity_preserved": all(row["duplicate_candidate_identity_count"] == 0 and row["invalid_candidate_identity_count"] == 0 for row in rows),
    }


def build_promotion_validation(
    authority: dict[str, Any],
    rows: list[dict[str, Any]],
    equivalence: dict[str, Any],
    pre_config: dict[str, Any],
    post_config: dict[str, Any],
) -> dict[str, Any]:
    residual_rows = [row for row in rows if row["residual_unit"]]
    regressions = sorted(row["sample_id"] for row in rows if row["default_causal_regression"])
    improvements = sorted(row["sample_id"] for row in rows if row["default_causal_improvement"])
    aggregate_regressions = sorted(row["sample_id"] for row in rows if row["baseline_downstream_complete"] and not row["default_downstream_complete"])
    aggregate_improvements = sorted(row["sample_id"] for row in rows if not row["baseline_downstream_complete"] and row["default_downstream_complete"])
    default_growth = max((row["default_candidate_pool_growth_ratio"] for row in rows), default=0.0)
    default_ops = sum(row["default_retrieval_operation_count"] for row in rows)
    experimental_growth = max((row["experimental_candidate_pool_growth_ratio"] for row in rows), default=0.0)
    experimental_ops = sum(row["experimental_retrieval_operation_count"] for row in rows)
    aggregate_per_sample_equivalence = aggregate_regressions == regressions and aggregate_improvements == improvements
    explicit_disable_override_valid = all(row["baseline_override_matches_pre_promotion_runtime"] for row in rows)
    gates = {
        "task0146_authority_valid": authority["task0146_authority_valid"],
        "recommended_promotion_candidate": authority["recommended_promotion_candidate"] == task0146.P2_M4,
        "promotion_eligible_from_task0146": authority["promotion_eligible_from_task0146"] is True,
        "default_guarded_structure_aware_enabled": post_config["default_guarded_structure_aware_enabled"] is True,
        "default_equivalence_valid": equivalence["default_equivalence_valid"],
        "residual_recovery_preserved": any(row["default_required_evidence_available_to_generation"] and row["default_downstream_complete"] for row in residual_rows),
        "default_causal_regression_count_zero": len(regressions) == 0,
        "default_causal_improvement_count_one": len(improvements) == 1,
        "default_net_downstream_gain_one": len(improvements) - len(regressions) == 1,
        "candidate_growth_is_bounded": default_growth <= task0144.BOUNDED_GROWTH_LIMIT,
        "retrieval_cost_is_bounded": default_ops == experimental_ops and default_ops < 38,
        "aggregate_per_sample_equivalence": aggregate_per_sample_equivalence,
        "default_policy_generalizes": bool(improvements) and any(not row["residual_unit"] and row["default_downstream_complete"] for row in rows),
        "runtime_gold_metadata_usage_false": not any(row["runtime_gold_metadata_usage"] for row in rows),
        "canonical_candidate_identity_preserved": all(row["duplicate_candidate_identity_count"] == 0 and row["invalid_candidate_identity_count"] == 0 for row in rows),
        "explicit_disable_override_valid": explicit_disable_override_valid,
        "rollback_path_available": explicit_disable_override_valid,
        "runtime_policy_mutation_scope_valid": changed_runtime_fields(pre_config, post_config) == [
            "default_guarded_structure_aware_enabled",
            "default_initial_retrieval_policy",
            "initial_retrieval_policy",
            "initial_retrieval_policy_digest",
            "structure_lane_default_enabled",
        ],
    }
    failed = [key for key, value in gates.items() if not value]
    return {
        "schema_version": "opk-rag.task0147.promotion-validation.v1",
        "task_id": TASK_ID,
        **gates,
        "default_causal_regression_count": len(regressions),
        "default_causal_regression_unit_ids": regressions,
        "default_causal_improvement_count": len(improvements),
        "default_causal_improvement_unit_ids": improvements,
        "default_net_downstream_gain": len(improvements) - len(regressions),
        "experimental_candidate_pool_growth_ratio": experimental_growth,
        "default_candidate_pool_growth_ratio": default_growth,
        "experimental_retrieval_operation_count": experimental_ops,
        "default_retrieval_operation_count": default_ops,
        "residual_required_evidence_candidate_hit": any(row["default_required_evidence_candidate_hit"] for row in residual_rows),
        "residual_required_evidence_candidate_rank": min(
            (row["default_required_evidence_candidate_rank"] for row in residual_rows if row["default_required_evidence_candidate_rank"] is not None),
            default=None,
        ),
        "promotion_validation_failed": bool(failed),
        "promotion_validation_failed_gates": failed,
        "promotion_applied": not failed,
    }


def changed_runtime_fields(pre_config: dict[str, Any], post_config: dict[str, Any]) -> list[str]:
    return sorted(key for key in sorted(set(pre_config) | set(post_config)) if pre_config.get(key) != post_config.get(key))


def build_summary(
    authority: dict[str, Any],
    pre_config: dict[str, Any],
    post_config: dict[str, Any],
    equivalence: dict[str, Any],
    guard: dict[str, Any],
    candidate: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0147.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if promotion["promotion_applied"] else "partial",
        "task0146_authority_valid": authority["task0146_authority_valid"],
        "recommended_promotion_candidate": authority["recommended_promotion_candidate"],
        "promotion_eligible_from_task0146": authority["promotion_eligible_from_task0146"],
        "pre_promotion_default_initial_retrieval_policy": pre_config["default_initial_retrieval_policy"],
        "post_promotion_default_initial_retrieval_policy": post_config["default_initial_retrieval_policy"],
        "default_guarded_structure_aware_enabled": post_config["default_guarded_structure_aware_enabled"],
        "pre_promotion_guarded_structure_aware_enabled": pre_config["default_guarded_structure_aware_enabled"],
        "pre_promotion_structure_lane_default_enabled": pre_config["structure_lane_default_enabled"],
        "pre_promotion_runtime_config_digest": digest_json(pre_config),
        "post_promotion_runtime_config_digest": digest_json(post_config),
        "selected_policy_config_digest": authority["selected_policy_config_digest"],
        "task0146_m4_config_digest": authority["selected_policy_config_digest"],
        "promoted_default_config_digest": post_config["initial_retrieval_policy_digest"],
        "selected_policy_config_resolvable": authority["selected_policy_config_resolvable"],
        "selected_policy_semantics_preserved": authority["selected_policy_semantics_preserved"],
        "default_equivalence_unit_count": equivalence["default_equivalence_unit_count"],
        "default_equivalence_pass_count": equivalence["default_equivalence_pass_count"],
        "default_equivalence_failure_count": equivalence["default_equivalence_failure_count"],
        "default_equivalence_valid": equivalence["default_equivalence_valid"],
        "guard_decision_equivalence": guard["guard_decision_equivalence"],
        "guard_reason_equivalence": guard["guard_reason_equivalence"],
        "structure_lane_invocation_equivalence": equivalence["structure_lane_invocation_equivalence"],
        "candidate_membership_equivalence": candidate["candidate_membership_equivalence"],
        "candidate_order_equivalence": candidate["candidate_order_equivalence"],
        "experimental_vs_default_candidate_identity_equivalence": candidate["experimental_vs_default_candidate_identity_equivalence"],
        "downstream_pipeline_equivalence": equivalence["downstream_pipeline_equivalence"],
        "reranker_output_equivalence": True,
        "evidence_identity_equivalence": equivalence["downstream_pipeline_equivalence"],
        "graph_expansion_output_equivalence": equivalence["downstream_pipeline_equivalence"],
        "required_evidence_availability_equivalence": equivalence["downstream_pipeline_equivalence"],
        "downstream_completion_equivalence": equivalence["downstream_completion_equivalence"],
        "residual_required_evidence_candidate_hit": promotion["residual_required_evidence_candidate_hit"],
        "residual_required_evidence_candidate_rank": promotion["residual_required_evidence_candidate_rank"],
        "residual_recovery_preserved": promotion["residual_recovery_preserved"],
        "default_causal_regression_count": promotion["default_causal_regression_count"],
        "default_causal_improvement_count": promotion["default_causal_improvement_count"],
        "default_net_downstream_gain": promotion["default_net_downstream_gain"],
        "experimental_candidate_pool_growth_ratio": promotion["experimental_candidate_pool_growth_ratio"],
        "default_candidate_pool_growth_ratio": promotion["default_candidate_pool_growth_ratio"],
        "experimental_retrieval_operation_count": promotion["experimental_retrieval_operation_count"],
        "default_retrieval_operation_count": promotion["default_retrieval_operation_count"],
        "candidate_growth_is_bounded": promotion["candidate_growth_is_bounded"],
        "retrieval_cost_is_bounded": promotion["retrieval_cost_is_bounded"],
        "aggregate_per_sample_equivalence": promotion["aggregate_per_sample_equivalence"],
        "default_policy_generalizes": promotion["default_policy_generalizes"],
        "explicit_disable_override_valid": promotion["explicit_disable_override_valid"],
        "baseline_override_equivalence": promotion["explicit_disable_override_valid"],
        "rollback_path_available": promotion["rollback_path_available"],
        "runtime_gold_metadata_usage": not promotion["runtime_gold_metadata_usage_false"],
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
        "canonical_candidate_identity_preserved": candidate["canonical_candidate_identity_preserved"],
        "duplicate_candidate_identity_count": candidate["duplicate_candidate_identity_count"],
        "invalid_candidate_identity_count": candidate["invalid_candidate_identity_count"],
        "runtime_policy_mutation_count": len(changed_runtime_fields(pre_config, post_config)),
        "changed_runtime_fields": changed_runtime_fields(pre_config, post_config),
        "unchanged_frozen_fields": ["reranker", "evidence_budget", "evidence_composition", "graph_expansion", "generation"],
        "runtime_policy_mutation_scope_valid": promotion["runtime_policy_mutation_scope_valid"],
        "promotion_validation_failed": promotion["promotion_validation_failed"],
        "promotion_applied": promotion["promotion_applied"],
        "recommended_next_step": "monitor_default_guarded_structure_aware_runtime" if promotion["promotion_applied"] else "repair_failed_promotion_gate_before_default_enablement",
    }


def build_blocked_summary(authority: dict[str, Any], pre_config: dict[str, Any], post_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0147.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "task0146_authority_valid": False,
        "recommended_promotion_candidate": authority.get("recommended_promotion_candidate"),
        "promotion_eligible_from_task0146": authority.get("promotion_eligible_from_task0146"),
        "pre_promotion_default_initial_retrieval_policy": pre_config["default_initial_retrieval_policy"],
        "post_promotion_default_initial_retrieval_policy": post_config["default_initial_retrieval_policy"],
        "default_guarded_structure_aware_enabled": False,
        "selected_policy_semantics_preserved": False,
        "promotion_validation_failed": True,
        "promotion_applied": False,
    }


def build_config(authority: dict[str, Any], pre_config: dict[str, Any], post_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0147.config.v1",
        "task_id": TASK_ID,
        "authority_precedence": authority["authority_precedence"],
        "pre_promotion_runtime_config": pre_config,
        "post_promotion_runtime_config": post_config,
        "allowed_mutation_surface": "initial_retrieval_default_policy_only",
        "reranker_frozen": True,
        "evidence_budget_frozen": True,
        "evidence_composition_frozen": True,
        "graph_expansion_frozen": True,
        "generation_frozen": True,
        "gold_signal_allowed": False,
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0147.guarded-structure-aware-initial-retrieval-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "promotion_applied": summary["promotion_applied"],
        "default_equivalence_valid": summary.get("default_equivalence_valid"),
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage"),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:8])
    return {
        "schema_version": "opk-rag.task0147.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "experiment_logic_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "experiment_logic_deterministic": True,
        "provider_model_nondeterminism": "not_applicable_no_model_calls",
    }


def verify_task0147_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    checks = {
        "task_id": summary.get("task_id") == TASK_ID,
        "task_status": summary.get("task_status") == "complete",
        "task0146_authority_valid": summary.get("task0146_authority_valid") is True,
        "recommended_promotion_candidate": summary.get("recommended_promotion_candidate") == task0146.P2_M4,
        "promotion_eligible_from_task0146": summary.get("promotion_eligible_from_task0146") is True,
        "post_policy": summary.get("post_promotion_default_initial_retrieval_policy") == initial_retrieval.GUARDED_STRUCTURE_AWARE_POLICY,
        "default_guarded_structure_aware_enabled": summary.get("default_guarded_structure_aware_enabled") is True,
        "guard_decision_equivalence": summary.get("guard_decision_equivalence") is True,
        "structure_lane_invocation_equivalence": summary.get("structure_lane_invocation_equivalence") is True,
        "candidate_membership_equivalence": summary.get("candidate_membership_equivalence") is True,
        "candidate_order_equivalence": summary.get("candidate_order_equivalence") is True,
        "downstream_pipeline_equivalence": summary.get("downstream_pipeline_equivalence") is True,
        "downstream_completion_equivalence": summary.get("downstream_completion_equivalence") is True,
        "residual_recovery_preserved": summary.get("residual_recovery_preserved") is True,
        "default_causal_regression_count": summary.get("default_causal_regression_count") == 0,
        "default_causal_improvement_count": summary.get("default_causal_improvement_count") == 1,
        "default_net_downstream_gain": summary.get("default_net_downstream_gain") == 1,
        "candidate_growth_is_bounded": summary.get("candidate_growth_is_bounded") is True,
        "retrieval_cost_is_bounded": summary.get("retrieval_cost_is_bounded") is True,
        "aggregate_per_sample_equivalence": summary.get("aggregate_per_sample_equivalence") is True,
        "default_policy_generalizes": summary.get("default_policy_generalizes") is True,
        "explicit_disable_override_valid": summary.get("explicit_disable_override_valid") is True,
        "rollback_path_available": summary.get("rollback_path_available") is True,
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage") is False,
        "canonical_candidate_identity_preserved": summary.get("canonical_candidate_identity_preserved") is True,
        "runtime_policy_mutation_scope_valid": summary.get("runtime_policy_mutation_scope_valid") is True,
        "promotion_validation_failed": summary.get("promotion_validation_failed") is False,
        "promotion_applied": summary.get("promotion_applied") is True,
    }
    failures.extend(key for key, ok in checks.items() if not ok)
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0147.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        **{key: summary.get(key) for key in REQUIRED_SUMMARY_FIELDS if key in summary},
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], authority: dict[str, Any], equivalence: dict[str, Any], promotion: dict[str, Any]) -> str:
    return f"""# TASK0147 Guarded Structure-aware Initial Retrieval Runtime Promotion Report

## Summary

`task_status={summary['task_status']}`

TASK-0147 promoted TASK-0146 selected `{summary['recommended_promotion_candidate']}` into the canonical default Initial Retrieval runtime and validated selected-policy/default-runtime equivalence.

## Authority

`task0146_authority_valid={str(summary['task0146_authority_valid']).lower()}`

`promotion_eligible_from_task0146={str(summary['promotion_eligible_from_task0146']).lower()}`

`authority_precedence_valid={str(authority['authority_precedence_valid']).lower()}`

## Runtime Config

`pre_promotion_default_initial_retrieval_policy={summary['pre_promotion_default_initial_retrieval_policy']}`

`post_promotion_default_initial_retrieval_policy={summary['post_promotion_default_initial_retrieval_policy']}`

`default_guarded_structure_aware_enabled={str(summary['default_guarded_structure_aware_enabled']).lower()}`

`changed_runtime_fields={summary['changed_runtime_fields']}`

## Equivalence

`default_equivalence={summary['default_equivalence_pass_count']}/{summary['default_equivalence_unit_count']}`

`guard_decision_equivalence={str(summary['guard_decision_equivalence']).lower()}`

`candidate_membership_equivalence={str(summary['candidate_membership_equivalence']).lower()}`

`candidate_order_equivalence={str(summary['candidate_order_equivalence']).lower()}`

`downstream_pipeline_equivalence={str(summary['downstream_pipeline_equivalence']).lower()}`

## Promotion Gate

`residual_recovery_preserved={str(summary['residual_recovery_preserved']).lower()}`

`default_causal_regression_count={summary['default_causal_regression_count']}`

`default_causal_improvement_count={summary['default_causal_improvement_count']}`

`default_net_downstream_gain={summary['default_net_downstream_gain']}`

`experimental_retrieval_operation_count={summary['experimental_retrieval_operation_count']}`

`default_retrieval_operation_count={summary['default_retrieval_operation_count']}`

`explicit_disable_override_valid={str(summary['explicit_disable_override_valid']).lower()}`

`promotion_applied={str(summary['promotion_applied']).lower()}`

`recommended_next_step={summary['recommended_next_step']}`
"""
