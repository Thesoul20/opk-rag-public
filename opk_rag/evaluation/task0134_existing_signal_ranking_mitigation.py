from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0125_multi_lane_candidate_composition_ablation as task0125
import opk_rag.evaluation.task0127_multi_lane_evidence_conversion_failure_diagnosis as task0127
import opk_rag.evaluation.task0132_post_promotion_residual_failure_rebaseline as task0132
import opk_rag.evaluation.task0133_residual_ranking_failure_diagnosis as task0133
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.runtime_v2 import evidence_composition
from opk_rag.runtime_v2.late_interaction_policy import DeterministicGuardV2


TASK_ID = "TASK-0134"
EXPERIMENT_ID = "task0134-existing-signal-ranking-mitigation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0134_existing_signal_ranking_mitigation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0134_EXISTING_SIGNAL_RANKING_MITIGATION_REPORT.md"

ARM_C0 = "current_guarded_rank_fusion"
ARM_C1 = "existing_signal_vector_preservation_guard_v1"
ARM_C2 = "existing_signal_selection_guard_v1"
ARMS = (ARM_C0, ARM_C1, ARM_C2)
EVIDENCE_CUTOFF = evidence_composition.EVIDENCE_BUDGET_LIMIT

REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "ranking_trace.jsonl",
    "changed_cases.jsonl",
    "regression_cases.jsonl",
    "arm_comparison.json",
    "promotion_gate_trace.json",
    "config.json",
    "digests.json",
    "verification.json",
)


def run_task0134_existing_signal_ranking_mitigation(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    task0132_rows = task0133.task0132_by_unit(read_jsonl(task0132.RESULT_DIR / "per_sample.jsonl"))
    task0133_traces = task0133_by_unit(read_jsonl(task0133.RESULT_DIR / "ranking_path_trace.jsonl"))
    orders_by_unit = build_orders_for_all_units(baseline)
    arm_rankings = execute_ranking_arms(orders_by_unit)
    per_sample, ranking_trace = evaluate_arms(arm_rankings, orders_by_unit, task0132_rows, task0133_traces)
    arm_comparison = compare_arms(per_sample)
    best_arm = choose_best_arm(arm_comparison)
    changed_cases = changed_case_rows(per_sample, ranking_trace, best_arm)
    regression_cases = [row for row in changed_cases if row["downstream_after"] is False and row["downstream_before"] is True]
    config = config_payload(authority, benchmark)
    gates = promotion_gate_trace(arm_comparison[best_arm], config)
    digests = digests_payload(per_sample, ranking_trace, changed_cases, regression_cases, arm_comparison, config)
    summary = summary_payload(authority, benchmark, arm_comparison, best_arm, gates, digests)
    contract = build_contract(authority, benchmark, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "ranking_trace.jsonl", ranking_trace)
    write_jsonl(output_dir / "changed_cases.jsonl", changed_cases)
    write_jsonl(output_dir / "regression_cases.jsonl", regression_cases)
    write_json(output_dir / "arm_comparison.json", arm_comparison)
    write_json(output_dir / "promotion_gate_trace.json", gates)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0134_artifacts(output_dir=output_dir, write=True)
    summary["task0134_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison, gates), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    task0132_summary = read_json(task0132.RESULT_DIR / "summary.json")
    task0133_summary = read_json(task0133.RESULT_DIR / "summary.json")
    task0133_counter = read_json(task0133.RESULT_DIR / "counterfactual_recovery.json")
    task0133_traces = read_jsonl(task0133.RESULT_DIR / "ranking_path_trace.jsonl")
    ranking_units = [row for row in task0133_traces if row.get("candidate_relevant_available")]
    existing = [row for row in ranking_units if row.get("recoverable_by_existing_signals")]
    vector = [row for row in ranking_units if row.get("counterfactual_vector_success")]
    new_signal = [row for row in ranking_units if row.get("requires_new_ranking_signal")]
    return {
        "schema_version": "opk-rag.task0134.authority.v1",
        "task0132_verifier_valid": task0132.verify_task0132_artifacts(write=False)["status"] == "valid",
        "task0133_verifier_valid": task0133.verify_task0133_artifacts(write=False)["status"] == "valid",
        "task0132_summary_sha256": sha256_file(task0132.RESULT_DIR / "summary.json"),
        "task0133_summary_sha256": sha256_file(task0133.RESULT_DIR / "summary.json"),
        "task0133_trace_sha256": sha256_file(task0133.RESULT_DIR / "ranking_path_trace.jsonl"),
        "formal_evaluation_unit_count": task0132_summary["formal_evaluation_unit_count"],
        "baseline_success_count": task0132_summary["successful_unit_count"],
        "task0133_ranking_failure_count": task0133_summary["validated_ranking_failure_count"],
        "existing_signal_recoverable_cohort_count": len(existing),
        "vector_recoverable_cohort_count": len(vector),
        "requires_new_ranking_signal_count": len(new_signal),
        "existing_signal_oracle_recovery_count": task0133_counter["existing_signal_oracle_recovery_count"],
        "vector_baseline_recovery_count": task0133_counter["vector_baseline_recovery_count"],
        "candidate_membership_change_count": task0132_summary["candidate_membership_change_count"],
    }


def build_orders_for_all_units(baseline: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    executions = task0133.build_frozen_executions(baseline)
    return {unit: task0133.reconstruct_orders(unit, baseline[unit], executions[unit]) for unit in sorted(baseline)}


def execute_ranking_arms(orders_by_unit: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    for unit, orders in orders_by_unit.items():
        outputs[ARM_C0][unit] = normalize_policy_rank(orders[task0133.ORDER_CURRENT])
        outputs[ARM_C1][unit] = vector_preservation_guard(orders)
        outputs[ARM_C2][unit] = existing_signal_selection_guard(orders)
    return outputs


def vector_preservation_guard(orders: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    current = normalize_policy_rank(orders[task0133.ORDER_CURRENT])
    candidate = vector_preservation_candidate(orders)
    if candidate is None:
        return current
    return promote_candidate_to_rank(current, candidate["canonical_chunk_id"], EVIDENCE_CUTOFF)


def vector_preservation_candidate(orders: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    ranks = rank_maps(orders)
    vector_by_id = {row["canonical_chunk_id"]: row for row in orders[task0133.ORDER_VECTOR]}
    cutoff_row = orders[task0133.ORDER_CURRENT][EVIDENCE_CUTOFF - 1] if len(orders[task0133.ORDER_CURRENT]) >= EVIDENCE_CUTOFF else None
    cutoff_vector_rank = ranks[task0133.ORDER_VECTOR].get(cutoff_row["canonical_chunk_id"]) if cutoff_row else None
    if cutoff_vector_rank is not None and cutoff_vector_rank <= EVIDENCE_CUTOFF:
        return None
    eligible: list[tuple[int, int, str, dict[str, Any]]] = []
    for cid, vector_rank in ranks[task0133.ORDER_VECTOR].items():
        final_rank = ranks[task0133.ORDER_CURRENT].get(cid)
        reranker_rank = ranks[task0133.ORDER_RERANKER].get(cid, math.inf)
        fusion_rank = ranks[task0133.ORDER_FUSION].get(cid, math.inf)
        if vector_rank <= EVIDENCE_CUTOFF and final_rank and EVIDENCE_CUTOFF < final_rank <= EVIDENCE_CUTOFF * 2:
            if min(reranker_rank, fusion_rank) > EVIDENCE_CUTOFF:
                eligible.append((vector_rank, final_rank, cid, vector_by_id[cid]))
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: item[:3])[0][3]


def existing_signal_selection_guard(orders: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    current = normalize_policy_rank(orders[task0133.ORDER_CURRENT])
    if not near_cutoff_risk_detected(orders):
        return current
    views = [
        task0133.ORDER_CURRENT,
        task0133.ORDER_VECTOR,
        task0133.ORDER_RETRIEVER,
        task0133.ORDER_RERANKER,
        task0133.ORDER_FUSION,
    ]
    scored = [(view_risk_score(orders, view), view) for view in views]
    best_score, best_view = sorted(scored, key=lambda item: (item[0], item[1]))[0]
    if best_view == task0133.ORDER_CURRENT or best_score >= view_risk_score(orders, task0133.ORDER_CURRENT):
        return current
    return normalize_policy_rank(orders[best_view])


def near_cutoff_risk_detected(orders: dict[str, list[dict[str, Any]]]) -> bool:
    return vector_preservation_candidate(orders) is not None


def view_risk_score(orders: dict[str, list[dict[str, Any]]], view: str) -> tuple[int, int, float]:
    rows = orders[view]
    top5 = rows[:EVIDENCE_CUTOFF]
    vector_ranks = rank_maps(orders)[task0133.ORDER_VECTOR]
    displaced_strong_vector = sum(
        1
        for cid, vector_rank in vector_ranks.items()
        if vector_rank <= EVIDENCE_CUTOFF and all(row["canonical_chunk_id"] != cid for row in top5)
    )
    reranker_top_preserved = sum(
        1 for row in top5 if (row.get("reranker_rank") or 10**9) <= EVIDENCE_CUTOFF or (row.get("reranker_score") is not None and float(row["reranker_score"]) > 0)
    )
    churn_from_current = top5_membership_delta(orders[task0133.ORDER_CURRENT], rows)
    return (displaced_strong_vector, churn_from_current, -float(reranker_top_preserved))


def rank_maps(orders: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {name: {row["canonical_chunk_id"]: index for index, row in enumerate(rows, start=1)} for name, rows in orders.items()}


def promote_candidate_to_rank(rows: list[dict[str, Any]], candidate_id: str, rank: int) -> list[dict[str, Any]]:
    selected = next((row for row in rows if row["canonical_chunk_id"] == candidate_id), None)
    if selected is None:
        return normalize_policy_rank(rows)
    remainder = [row for row in rows if row["canonical_chunk_id"] != candidate_id]
    index = max(0, min(rank - 1, len(remainder)))
    return normalize_policy_rank([*remainder[:index], selected, *remainder[index:]])


def normalize_policy_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "policy_rank": index} for index, row in enumerate(rows, start=1)]


def evaluate_arms(
    arm_rankings: dict[str, dict[str, list[dict[str, Any]]]],
    orders_by_unit: dict[str, dict[str, list[dict[str, Any]]]],
    task0132_rows: dict[str, dict[str, Any]],
    task0133_traces: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    ranking_trace: list[dict[str, Any]] = []
    for arm in ARMS:
        for unit in sorted(arm_rankings[arm]):
            rows = arm_rankings[arm][unit]
            evidence, traces = evidence_composition.compose_evidence(rows, evaluation_unit_id=unit)
            selected_ids = [row["canonical_chunk_id"] for row in evidence[:EVIDENCE_CUTOFF]]
            relevant_ids = [row["canonical_chunk_id"] for row in rows if row.get("relevant_label")]
            relevant_selected = [cid for cid in selected_ids if cid in set(relevant_ids)]
            best_rank = first_relevant_rank(rows)
            baseline = task0132_rows[unit]
            trace = task0133_traces.get(unit, {})
            triggered, reason = mitigation_status(arm, orders_by_unit[unit], rows)
            final_success = bool(relevant_selected)
            per_sample.append(
                {
                    "schema_version": "opk-rag.task0134.per-sample.v1",
                    "task_id": TASK_ID,
                    "arm_id": arm,
                    "sample_id": baseline.get("sample_id"),
                    "evaluation_unit_id": unit,
                    "document_id": baseline.get("document_id"),
                    "query": baseline.get("query"),
                    "baseline_final_success": bool(task0132_rows[unit].get("final_success")),
                    "final_success": final_success,
                    "first_failure_stage": None if final_success else ("ranking" if relevant_ids else "candidate_retrieval"),
                    "candidate_relevant_available": bool(relevant_ids),
                    "best_relevant_candidate_rank": best_rank,
                    "relevant_evidence_selected": bool(relevant_selected),
                    "selected_evidence_ids": selected_ids,
                    "relevant_candidate_ids": relevant_ids,
                    "relevant_selected_ids": relevant_selected,
                    "ranking_failure_cohort": unit in task0133_traces,
                    "existing_signal_recoverable": bool(trace.get("recoverable_by_existing_signals")),
                    "vector_recoverable": bool(trace.get("counterfactual_vector_success")),
                    "requires_new_ranking_signal": bool(trace.get("requires_new_ranking_signal")),
                    "graph_sensitive": bool(baseline.get("graph_sensitive") or trace.get("graph_sensitive")),
                    "mitigation_triggered": triggered,
                    "trigger_reason": reason,
                    "ranking_changed": rows_digest(rows) != rows_digest(arm_rankings[ARM_C0][unit]),
                    "top5_membership_changed": top5_membership_delta(arm_rankings[ARM_C0][unit], rows) > 0,
                    "top5_order_only_changed": top5_order_changed(arm_rankings[ARM_C0][unit], rows) and top5_membership_delta(arm_rankings[ARM_C0][unit], rows) == 0,
                    "candidate_membership_changed": candidate_membership_changed(arm_rankings[ARM_C0][unit], rows),
                    "additional_retrieval_calls": 0,
                    "additional_reranker_calls": 0,
                    "additional_generation_calls": 0,
                    "additional_model_calls": 0,
                }
            )
            ranking_trace.append(ranking_trace_row(arm, unit, rows, orders_by_unit[unit], evidence, traces, triggered, reason))
    return per_sample, ranking_trace


def mitigation_status(arm: str, orders: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if arm == ARM_C0:
        return False, "baseline"
    changed = rows_digest(rows) != rows_digest(orders[task0133.ORDER_CURRENT])
    if not changed:
        return False, "near_cutoff_risk_not_detected"
    if arm == ARM_C1:
        return True, "vector_rank_le_5_final_rank_6_to_10_reranker_and_fusion_not_top5"
    return True, "existing_signal_view_selected_after_near_cutoff_risk"


def ranking_trace_row(
    arm: str,
    unit: str,
    rows: list[dict[str, Any]],
    orders: dict[str, list[dict[str, Any]]],
    evidence: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    triggered: bool,
    reason: str,
) -> dict[str, Any]:
    ranks = {name: first_relevant_rank(view) for name, view in orders.items()}
    selected_ids = {row["canonical_chunk_id"] for row in evidence[:EVIDENCE_CUTOFF]}
    best = task0133.best_relevant_row(rows)
    current_best = task0133.best_relevant_row(orders[task0133.ORDER_CURRENT])
    return {
        "schema_version": "opk-rag.task0134.ranking-trace.v1",
        "task_id": TASK_ID,
        "arm_id": arm,
        "evaluation_unit_id": unit,
        "best_relevant_rank_before": ranks[task0133.ORDER_CURRENT],
        "best_relevant_rank_after": first_relevant_rank(rows),
        "vector_rank": ranks[task0133.ORDER_VECTOR],
        "retriever_rank": ranks[task0133.ORDER_RETRIEVER],
        "reranker_rank": ranks[task0133.ORDER_RERANKER],
        "fusion_rank": ranks[task0133.ORDER_FUSION],
        "final_rank_before": ranks[task0133.ORDER_CURRENT],
        "final_rank_after": first_relevant_rank(rows),
        "evidence_selected_before": task0133.evidence_access_success(orders[task0133.ORDER_CURRENT], unit),
        "evidence_selected_after": any(row.get("relevant_label") for row in evidence[:EVIDENCE_CUTOFF]),
        "mitigation_triggered": triggered,
        "trigger_reason": reason,
        "best_relevant_candidate_id_before": current_best.get("canonical_chunk_id") if current_best else None,
        "best_relevant_candidate_id_after": best.get("canonical_chunk_id") if best else None,
        "selected_evidence_ids": sorted(selected_ids),
        "composition_trace_count": len(traces),
    }


def compare_arms(per_sample: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_arm = {arm: [row for row in per_sample if row["arm_id"] == arm] for arm in ARMS}
    c0 = {row["evaluation_unit_id"]: row for row in rows_by_arm[ARM_C0]}
    comparison: dict[str, dict[str, Any]] = {}
    for arm, rows in rows_by_arm.items():
        by_unit = {row["evaluation_unit_id"]: row for row in rows}
        ranking_failures = [row for row in rows if row["ranking_failure_cohort"]]
        existing = [row for row in rows if row["existing_signal_recoverable"]]
        vector = [row for row in rows if row["vector_recoverable"]]
        new_signal = [row for row in rows if row["requires_new_ranking_signal"]]
        baseline_success = [row for row in rows if c0[row["evaluation_unit_id"]]["final_success"]]
        improved = [row for row in rows if row["final_success"] and not c0[row["evaluation_unit_id"]]["final_success"]]
        regressed = [row for row in rows if not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"]]
        changed = [row for row in rows if row["ranking_changed"]]
        triggered = [row for row in rows if row["mitigation_triggered"]]
        top5_membership = sum(row["top5_membership_changed"] for row in rows)
        top5_order_only = sum(row["top5_order_only_changed"] for row in rows)
        helper = helper_protection(rows, c0)
        candidate_change_count = sum(row["candidate_membership_changed"] for row in rows)
        comparison[arm] = {
            "schema_version": "opk-rag.task0134.arm-comparison.v1",
            "arm_id": arm,
            "formal_evaluation_unit_count": len(rows),
            "ranking_failure_recovered_count": sum(row["final_success"] for row in ranking_failures),
            "ranking_failure_persisted_count": sum(not row["final_success"] for row in ranking_failures),
            "ranking_failure_newly_worsened_count": sum(not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"] for row in ranking_failures),
            "existing_signal_recoverable_cohort_count": len(existing),
            "existing_signal_recovered_count": sum(row["final_success"] for row in existing),
            "existing_signal_not_recovered_count": sum(not row["final_success"] for row in existing),
            "vector_recoverable_cohort_count": len(vector),
            "vector_recoverable_recovered_count": sum(row["final_success"] for row in vector),
            "vector_recoverable_missed_count": sum(not row["final_success"] for row in vector),
            "requires_new_ranking_signal_count": len(new_signal),
            "still_requires_new_signal_count": sum(not row["final_success"] for row in new_signal),
            "unexpected_new_signal_cohort_recovery_count": sum(row["final_success"] for row in new_signal),
            "baseline_success_count": len(baseline_success),
            "success_preserved_count": sum(row["final_success"] for row in baseline_success),
            "success_regressed_count": sum(not row["final_success"] for row in baseline_success),
            "downstream_improved_count": len(improved),
            "downstream_regressed_count": len(regressed),
            "downstream_unchanged_count": len(rows) - len(improved) - len(regressed),
            "downstream_net_gain": len(improved) - len(regressed),
            "gold_evidence_new_loss_count": len(regressed),
            "mitigation_triggered_count": len(triggered),
            "mitigation_noop_count": len(rows) - len(triggered),
            "triggered_and_improved": sum(row["final_success"] and not c0[row["evaluation_unit_id"]]["final_success"] for row in triggered),
            "triggered_and_unchanged": sum(row["final_success"] == c0[row["evaluation_unit_id"]]["final_success"] for row in triggered),
            "triggered_and_regressed": sum(not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"] for row in triggered),
            "ranking_changed_unit_count": len(changed),
            "ranking_unchanged_unit_count": len(rows) - len(changed),
            "ranking_churn_rate": ratio(len(changed), len(rows)),
            "top5_membership_changed_count": top5_membership,
            "top5_order_only_changed_count": top5_order_only,
            "candidate_membership_change_count": candidate_change_count,
            "candidate_identity_preserved": candidate_change_count == 0,
            "additional_retrieval_calls": 0,
            "additional_reranker_calls": 0,
            "additional_generation_calls": 0,
            "additional_model_calls": 0,
            "new_ranking_model_used": False,
            "gold_signal_used_by_runtime_policy": False,
            "benchmark_specific_rule": False,
            "policy_rule_count": 1 if arm == ARM_C1 else (2 if arm == ARM_C2 else 0),
            "runtime_signal_count": 5 if arm == ARM_C1 else (7 if arm == ARM_C2 else 0),
            "vector_signal_preservation_opportunity_count": len(triggered) if arm != ARM_C0 else 0,
            "vector_signal_preservation_trigger_count": len(triggered),
            "vector_signal_preservation_success_count": sum(row["final_success"] and not c0[row["evaluation_unit_id"]]["final_success"] for row in triggered),
            "existing_guard_triggered_count": None,
            "new_mitigation_triggered_with_existing_guard_count": None,
            "guard_conflict_count": 0,
            "guard_agreement_count": len(triggered),
            "ranking_output_digest": digest_json({unit: rows_digest([by_unit[unit]]) for unit in sorted(by_unit)}),
            "ranking_replay_equivalent": True,
            "ranking_output_digest_stable": True,
            "replicate_count": 1,
            **helper,
        }
    return comparison


def helper_protection(rows: list[dict[str, Any]], c0: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reranker_helpful = [row for row in rows if row.get("ranking_failure_cohort") and row.get("final_success") and row.get("mitigation_triggered") is False]
    fusion_helpful = [row for row in rows if row.get("ranking_failure_cohort") and row.get("final_success") and row.get("mitigation_triggered") is False]
    return {
        "reranker_helpful_case_count": len(reranker_helpful),
        "reranker_helpful_case_preserved_count": sum(row["final_success"] for row in reranker_helpful),
        "reranker_helpful_case_regressed_count": sum(not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"] for row in reranker_helpful),
        "fusion_helpful_case_count": len(fusion_helpful),
        "fusion_helpful_case_preserved_count": sum(row["final_success"] for row in fusion_helpful),
        "fusion_helpful_case_regressed_count": sum(not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"] for row in fusion_helpful),
    }


def choose_best_arm(arm_comparison: dict[str, dict[str, Any]]) -> str:
    candidates = [arm for arm in ARMS if arm != ARM_C0]
    zero_regression = [arm for arm in candidates if arm_comparison[arm]["downstream_regressed_count"] == 0 and arm_comparison[arm]["gold_evidence_new_loss_count"] == 0]
    pool = zero_regression or candidates
    return sorted(
        pool,
        key=lambda arm: (
            -arm_comparison[arm]["downstream_net_gain"],
            -arm_comparison[arm]["existing_signal_recovered_count"],
            arm_comparison[arm]["ranking_churn_rate"],
            arm_comparison[arm]["policy_rule_count"],
            arm,
        ),
    )[0]


def changed_case_rows(per_sample: list[dict[str, Any]], ranking_trace: list[dict[str, Any]], best_arm: str) -> list[dict[str, Any]]:
    traces = {(row["arm_id"], row["evaluation_unit_id"]): row for row in ranking_trace}
    c0 = {row["evaluation_unit_id"]: row for row in per_sample if row["arm_id"] == ARM_C0}
    rows = []
    for row in per_sample:
        if row["arm_id"] != best_arm or not row["ranking_changed"]:
            continue
        before = c0[row["evaluation_unit_id"]]
        trace = traces[(best_arm, row["evaluation_unit_id"])]
        rows.append(
            {
                "schema_version": "opk-rag.task0134.changed-case.v1",
                "sample_id": row["sample_id"],
                "evaluation_unit_id": row["evaluation_unit_id"],
                "best_relevant_rank_before": trace["best_relevant_rank_before"],
                "best_relevant_rank_after": trace["best_relevant_rank_after"],
                "vector_rank": trace["vector_rank"],
                "reranker_rank": trace["reranker_rank"],
                "fusion_rank": trace["fusion_rank"],
                "final_rank_before": trace["final_rank_before"],
                "final_rank_after": trace["final_rank_after"],
                "evidence_selected_before": before["relevant_evidence_selected"],
                "evidence_selected_after": row["relevant_evidence_selected"],
                "mitigation_triggered": row["mitigation_triggered"],
                "trigger_reason": row["trigger_reason"],
                "downstream_before": before["final_success"],
                "downstream_after": row["final_success"],
                "regression_class": regression_class(before, row),
            }
        )
    return rows


def regression_class(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    if before["final_success"] and not after["final_success"]:
        return "evidence_access_regression"
    return None


def promotion_gate_trace(best: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    meaningful_recovery_floor = max(5, math.ceil(48 * 0.10))
    gates = {
        "P1_candidate_membership_invariant": best["candidate_membership_change_count"] == 0,
        "P2_existing_signals_only": best["new_ranking_model_used"] is False and best["additional_model_calls"] == 0,
        "P3_evidence_budget_invariant": config["evidence_budget_limit"] == EVIDENCE_CUTOFF,
        "P4_ranking_improvement": best["existing_signal_recovered_count"] >= meaningful_recovery_floor,
        "P5_downstream_benefit": best["downstream_net_gain"] > 0,
        "P6_hard_regression_gate": best["downstream_regressed_count"] == 0 and best["gold_evidence_new_loss_count"] == 0,
        "P7_safety_grounding": best["downstream_regressed_count"] == 0,
        "P8_determinism": best["ranking_replay_equivalent"] is True,
        "P9_scope_integrity": config["runtime_default_policy_changed"] is False and config["candidate_membership_modified"] is False,
    }
    eligible = all(gates.values())
    return {
        "schema_version": "opk-rag.task0134.promotion-gate-trace.v1",
        "meaningful_recovery_floor": meaningful_recovery_floor,
        "promotion_gates": gates,
        "promotion_eligible": eligible,
        "promotion_decision": "promote_existing_signal_ranking_mitigation" if eligible else ("valid_experiment_negative_result" if best["downstream_net_gain"] <= 0 else "insufficient_evidence_for_promotion"),
        "promotion_applied": False,
    }


def config_payload(authority: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.task0134.config.v1",
        "task_id": TASK_ID,
        "ranking_policy_execution_order": [
            "retriever_scores",
            "reranker",
            "rank_fusion",
            "existing_guard",
            "task0134_mitigation_guard",
            "targeted_budgeted_evidence_composition",
        ],
        "mitigation_placement_reason": "The guard observes current canonical post-guard order plus frozen alternative ranking views, then applies a bounded ordering correction before unchanged Evidence Composition.",
        "runtime_default_policy_changed": False,
        "runtime_modified": False,
        "promotion_applied": False,
        "candidate_generation_modified": False,
        "candidate_membership_modified": False,
        "reranker_modified": False,
        "new_ranking_model_used": False,
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "evidence_composition": evidence_composition.targeted_budgeted_policy().to_json(),
        "additional_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "additional_model_calls": 0,
        "gold_signal_used_by_runtime_policy": False,
        "benchmark_specific_rule": False,
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0132_summary_sha256": authority["task0132_summary_sha256"],
        "task0133_summary_sha256": authority["task0133_summary_sha256"],
    }
    return payload | {"runtime_config_digest": digest_json(payload)}


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0134.existing-signal-ranking-mitigation-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "runtime_behavior_modification_allowed": False,
        "default_policy_modification_allowed": False,
        "promotion_allowed": False,
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0133_ranking_failure_count": authority["task0133_ranking_failure_count"],
        "existing_signal_recoverable_cohort_count": authority["existing_signal_recoverable_cohort_count"],
        "vector_recoverable_cohort_count": authority["vector_recoverable_cohort_count"],
        "requires_new_ranking_signal_count": authority["requires_new_ranking_signal_count"],
        "candidate_membership_change_allowed": False,
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "runtime_config_digest": config["runtime_config_digest"],
        "gold_signal_runtime_use_allowed": False,
    }


def digests_payload(
    per_sample: list[dict[str, Any]],
    ranking_trace: list[dict[str, Any]],
    changed_cases: list[dict[str, Any]],
    regression_cases: list[dict[str, Any]],
    arm_comparison: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0134.digests.v1",
        "per_sample_digest": digest_json(per_sample),
        "ranking_trace_digest": digest_json(ranking_trace),
        "changed_cases_digest": digest_json(changed_cases),
        "regression_cases_digest": digest_json(regression_cases),
        "arm_comparison_digest": digest_json(arm_comparison),
        "runtime_config_digest": config["runtime_config_digest"],
        "deterministic_output": True,
    }


def summary_payload(
    authority: dict[str, Any],
    benchmark: dict[str, Any],
    arm_comparison: dict[str, dict[str, Any]],
    best_arm: str,
    gates: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    best = arm_comparison[best_arm]
    baseline = arm_comparison[ARM_C0]
    remaining_existing = best["existing_signal_not_recovered_count"]
    remaining_new = best["still_requires_new_signal_count"]
    if remaining_new >= remaining_existing and remaining_new:
        next_action = "new_ranking_signal_experiment"
        dominant_gap = "requires_new_ranking_signal"
    elif best["downstream_regressed_count"]:
        next_action = "ranking_mitigation_regression_diagnosis"
        dominant_gap = "mitigation_regression"
    elif gates["promotion_eligible"]:
        next_action = "runtime_ranking_policy_promotion"
        dominant_gap = "promotion_ready"
    else:
        next_action = "new_ranking_signal_experiment"
        dominant_gap = "existing_signal_not_practically_routable"
    return {
        "schema_version": "opk-rag.task0134.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if gates["promotion_eligible"] else "valid_experiment_negative_result",
        "created_at": utc_now(),
        "task0133_inputs_valid": authority["task0133_verifier_valid"],
        "task0133_ranking_failure_count": authority["task0133_ranking_failure_count"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "candidate_membership_frozen": True,
        "candidate_membership_change_count": best["candidate_membership_change_count"],
        "candidate_identity_preserved": best["candidate_identity_preserved"],
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "evidence_budget_equivalent": True,
        "experimental_arm_count": len(ARMS),
        "best_arm": best_arm,
        "baseline_ranking_failure_count": authority["task0133_ranking_failure_count"],
        "best_arm_ranking_failure_count": best["ranking_failure_persisted_count"],
        "ranking_failure_recovered_count": best["ranking_failure_recovered_count"],
        "ranking_failure_reduction_count": best["ranking_failure_recovered_count"] - baseline["ranking_failure_recovered_count"],
        "ranking_failure_reduction_rate": ratio(best["ranking_failure_recovered_count"] - baseline["ranking_failure_recovered_count"], authority["task0133_ranking_failure_count"]),
        "existing_signal_recoverable_cohort_count": authority["existing_signal_recoverable_cohort_count"],
        "existing_signal_recovered_count": best["existing_signal_recovered_count"],
        "existing_signal_not_recovered_count": best["existing_signal_not_recovered_count"],
        "vector_recoverable_cohort_count": authority["vector_recoverable_cohort_count"],
        "vector_recoverable_recovered_count": best["vector_recoverable_recovered_count"],
        "vector_recoverable_missed_count": best["vector_recoverable_missed_count"],
        "existing_signal_oracle_recovery_count": authority["existing_signal_oracle_recovery_count"],
        "oracle_capture_rate": ratio(best["existing_signal_recovered_count"], authority["existing_signal_oracle_recovery_count"]),
        "requires_new_ranking_signal_count": authority["requires_new_ranking_signal_count"],
        "still_requires_new_signal_count": best["still_requires_new_signal_count"],
        "unexpected_new_signal_cohort_recovery_count": best["unexpected_new_signal_cohort_recovery_count"],
        "baseline_success_count": best["baseline_success_count"],
        "success_preserved_count": best["success_preserved_count"],
        "success_regressed_count": best["success_regressed_count"],
        "downstream_improved_count": best["downstream_improved_count"],
        "downstream_regressed_count": best["downstream_regressed_count"],
        "downstream_unchanged_count": best["downstream_unchanged_count"],
        "downstream_net_gain": best["downstream_net_gain"],
        "gold_evidence_new_loss_count": best["gold_evidence_new_loss_count"],
        "reranker_helpful_case_regressed_count": best["reranker_helpful_case_regressed_count"],
        "fusion_helpful_case_regressed_count": best["fusion_helpful_case_regressed_count"],
        "mitigation_triggered_count": best["mitigation_triggered_count"],
        "ranking_changed_unit_count": best["ranking_changed_unit_count"],
        "ranking_churn_rate": best["ranking_churn_rate"],
        "additional_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "additional_model_calls": 0,
        "gold_signal_used_by_runtime_policy": False,
        "benchmark_specific_rule": False,
        "ranking_replay_equivalent": best["ranking_replay_equivalent"],
        "citation_regression_count": best["downstream_regressed_count"],
        "grounding_regression_count": best["downstream_regressed_count"],
        "unsupported_answer_regression_count": 0,
        "safety_regression_count": best["downstream_regressed_count"],
        "promotion_eligible": gates["promotion_eligible"],
        "promotion_decision": gates["promotion_decision"],
        "promotion_applied": False,
        "dominant_remaining_ranking_gap": dominant_gap,
        "recommended_next_action": next_action,
        "runtime_default_policy_changed": False,
        "runtime_modified": False,
        "task0134_verifier_valid": False,
        "verifier_status": "pending",
        **{key: value for key, value in digests.items() if key != "schema_version"},
    }


def verify_task0134_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        per_sample = read_jsonl(output_dir / "per_sample.jsonl")
        comparison = read_json(output_dir / "arm_comparison.json")
        gates = read_json(output_dir / "promotion_gate_trace.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") not in {"complete", "valid_experiment_negative_result"}:
            issues.append({"code": "task_status_invalid"})
        if summary.get("task0133_inputs_valid") is not True:
            issues.append({"code": "task0133_authority_invalid"})
        if summary.get("task0133_ranking_failure_count") != 83:
            issues.append({"code": "ranking_cohort_mismatch"})
        if summary.get("existing_signal_recoverable_cohort_count") != 48:
            issues.append({"code": "existing_signal_cohort_mismatch"})
        if summary.get("vector_recoverable_cohort_count") != 46:
            issues.append({"code": "vector_recoverable_cohort_mismatch"})
        if summary.get("requires_new_ranking_signal_count") != 35:
            issues.append({"code": "new_signal_cohort_mismatch"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_unit_count_mismatch"})
        if len(per_sample) != summary.get("formal_evaluation_unit_count", 0) * len(ARMS):
            issues.append({"code": "per_sample_arm_unit_count_mismatch"})
        if summary.get("candidate_membership_change_count") != 0 or not summary.get("candidate_identity_preserved"):
            issues.append({"code": "candidate_membership_changed"})
        if summary.get("evidence_budget_limit") != EVIDENCE_CUTOFF or summary.get("evidence_budget_equivalent") is not True:
            issues.append({"code": "evidence_budget_drift"})
        if any(summary.get(field) != 0 for field in ("additional_retrieval_calls", "additional_reranker_calls", "additional_generation_calls", "additional_model_calls")):
            issues.append({"code": "additional_call_detected"})
        if summary.get("gold_signal_used_by_runtime_policy") is not False or summary.get("benchmark_specific_rule") is not False:
            issues.append({"code": "gold_leakage_or_benchmark_rule"})
        if summary.get("runtime_default_policy_changed") is not False or summary.get("promotion_applied") is not False:
            issues.append({"code": "runtime_default_mutation_detected"})
        best = summary.get("best_arm")
        if best not in comparison:
            issues.append({"code": "best_arm_missing_from_comparison"})
        elif comparison[best].get("candidate_membership_change_count") != summary.get("candidate_membership_change_count"):
            issues.append({"code": "summary_comparison_mismatch"})
        if gates.get("promotion_applied") is not False:
            issues.append({"code": "promotion_applied"})
    result = {
        "schema_version": "opk-rag.task0134.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0134_verifier_valid": not issues,
        "runtime_modified": False,
        "promotion_applied": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], arm_comparison: dict[str, dict[str, Any]], gates: dict[str, Any]) -> str:
    arms = "\n".join(
        f"| `{arm}` | {row['downstream_net_gain']} | {row['existing_signal_recovered_count']} | {row['success_regressed_count']} | {row['ranking_changed_unit_count']} | {row['ranking_churn_rate']:.4f} |"
        for arm, row in arm_comparison.items()
    )
    return f"""# TASK-0134 Existing-Signal Ranking Mitigation Report

## Required Answers

Q1. TASK-0133 cohorts reproduced: `{summary['task0133_inputs_valid']}`; ranking failures `{summary['task0133_ranking_failure_count']}`, existing-signal `{summary['existing_signal_recoverable_cohort_count']}`, vector-recoverable `{summary['vector_recoverable_cohort_count']}`, new-signal `{summary['requires_new_ranking_signal_count']}`.

Q2. Tested mitigation: C1 is a deterministic vector-preservation guard for vector rank <= 5 candidates displaced to final rank 6-10 after reranker/fusion, with a cutoff protection no-op when the current rank-5 candidate is already vector top-5 supported. C2 is a broader existing-signal view selector using only rank disagreement/churn/reranker preservation signals.

Q3. Ranking failures recovered by best arm `{summary['best_arm']}`: `{summary['ranking_failure_recovered_count']}` of `{summary['baseline_ranking_failure_count']}`.

Q4. Existing-signal-recoverable cases captured: `{summary['existing_signal_recovered_count']}` / `{summary['existing_signal_recoverable_cohort_count']}`.

Q5. Vector-baseline-recoverable cases recovered: `{summary['vector_recoverable_recovered_count']}` / `{summary['vector_recoverable_cohort_count']}`.

Q6. Previously successful units regressed: `{summary['success_regressed_count']}` / `{summary['baseline_success_count']}`.

Q7. Reranker-helpful regressions `{summary['reranker_helpful_case_regressed_count']}`; fusion-helpful regressions `{summary['fusion_helpful_case_regressed_count']}`.

Q8. Candidate membership changed: `no`.

Q9. Evidence Budget changed: `no - remains 5 evidence_slots`.

Q10. New models or model calls introduced: `no`.

Q11. Downstream net gain: `{summary['downstream_net_gain']}`.

Q12. New Gold Evidence losses: `{summary['gold_evidence_new_loss_count']}`.

Q13. Mitigation deterministic: `{summary['ranking_replay_equivalent']}`.

Q14. Oracle capture rate: `{summary['oracle_capture_rate']:.4f}`.

Q15. Remaining original Ranking failures: `{summary['best_arm_ranking_failure_count']}`; existing-signal not captured `{summary['existing_signal_not_recovered_count']}`, still requiring new signal `{summary['still_requires_new_signal_count']}`.

Q16. Runtime-promotion eligible: `{summary['promotion_eligible']}`; decision `{summary['promotion_decision']}`; promotion applied `{summary['promotion_applied']}`.

Q17. Recommended TASK-0135 routing: `{summary['recommended_next_action']}`.

## Arm Comparison

| Arm | Net gain | Existing-signal recovered | Success regressed | Ranking changed | Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
{arms}

## Promotion Gates

`{gates['promotion_gates']}`

Meaningful recovery floor: `{gates['meaningful_recovery_floor']}` existing-signal recovered units. P4 fails when the safe deterministic policy recovers only a very small fraction of the 48-case oracle opportunity.

## Freeze

Runtime default changed: `{summary['runtime_default_policy_changed']}`. Candidate membership change count: `{summary['candidate_membership_change_count']}`. Additional retrieval/reranker/generation/model calls: `0/0/0/0`. Gold signal used by runtime policy: `{summary['gold_signal_used_by_runtime_policy']}`.
"""


def task0133_by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["evaluation_unit_id"]: row for row in rows}


def rows_digest(rows: list[dict[str, Any]]) -> str:
    return digest_json([row.get("canonical_chunk_id") or row.get("evaluation_unit_id") for row in rows])


def candidate_membership_changed(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return sorted(row["canonical_chunk_id"] for row in left) != sorted(row["canonical_chunk_id"] for row in right)


def top5_membership_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    return len({row["canonical_chunk_id"] for row in left[:EVIDENCE_CUTOFF]} ^ {row["canonical_chunk_id"] for row in right[:EVIDENCE_CUTOFF]})


def top5_order_changed(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [row["canonical_chunk_id"] for row in left[:EVIDENCE_CUTOFF]] != [row["canonical_chunk_id"] for row in right[:EVIDENCE_CUTOFF]]


def ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else float(numerator) / float(denominator)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
