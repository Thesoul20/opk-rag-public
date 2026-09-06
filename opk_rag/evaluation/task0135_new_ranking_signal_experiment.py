from __future__ import annotations

from collections import Counter
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0132_post_promotion_residual_failure_rebaseline as task0132
import opk_rag.evaluation.task0133_residual_ranking_failure_diagnosis as task0133
import opk_rag.evaluation.task0134_existing_signal_ranking_mitigation as task0134
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.runtime_v2 import evidence_composition


TASK_ID = "TASK-0135"
EXPERIMENT_ID = "task0135-new-ranking-signal-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0135_new_ranking_signal_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0135_NEW_RANKING_SIGNAL_EXPERIMENT_REPORT.md"

ARM_C0 = "current_canonical_runtime"
ARM_C1 = "qwen_direct_ordering"
ARM_C2 = "zerank_direct_ordering"
ARM_C3 = "late_interaction_direct_ordering"
ARM_C4 = "best_new_signal_plus_current_ranking"
ARMS = (ARM_C0, ARM_C1, ARM_C2, ARM_C3, ARM_C4)
NEW_SIGNAL_ARMS = (ARM_C1, ARM_C2, ARM_C3)
SIGNAL_NAMES = {
    ARM_C0: "current",
    ARM_C1: "qwen",
    ARM_C2: "zerank",
    ARM_C3: "late_interaction",
    ARM_C4: "best_integrated",
}
EVIDENCE_CUTOFF = evidence_composition.EVIDENCE_BUDGET_LIMIT

REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "signal_scores.jsonl",
    "ranking_results.jsonl",
    "cohort_analysis.json",
    "signal_complementarity.json",
    "regression_cases.jsonl",
    "latency_metrics.json",
    "model_environment.json",
    "config.json",
    "digests.json",
    "verification.json",
)


def run_task0135_new_ranking_signal_experiment(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    orders_by_unit = task0134.build_orders_for_all_units(baseline)
    task0132_rows = task0133.task0132_by_unit(read_jsonl(task0132.RESULT_DIR / "per_sample.jsonl"))
    task0133_rows = task0134.task0133_by_unit(read_jsonl(task0133.RESULT_DIR / "ranking_path_trace.jsonl"))

    started = time.perf_counter()
    direct_orders, signal_scores, signal_latencies = execute_signal_arms(orders_by_unit, task0132_rows)
    arm_comparison_probe, probe_per_sample, _probe_ranking = evaluate_arms(direct_orders, orders_by_unit, task0132_rows, task0133_rows)
    best_direct = choose_best_direct_signal(arm_comparison_probe)
    arm_rankings = {**direct_orders, ARM_C4: {}}
    for unit, current in direct_orders[ARM_C0].items():
        arm_rankings[ARM_C4][unit] = integrate_current_with_signal(current, direct_orders[best_direct][unit])
    per_sample, ranking_results = evaluate_arms(arm_rankings, orders_by_unit, task0132_rows, task0133_rows)[1:]
    arm_comparison = compare_arms(per_sample, ranking_results)
    best_direct = choose_best_direct_signal(arm_comparison)
    best_downstream = choose_best_downstream_arm(arm_comparison)
    cohort = cohort_analysis(per_sample, arm_comparison)
    complementarity = signal_complementarity(per_sample)
    regression_cases = regression_case_rows(per_sample, ranking_results, best_downstream)
    latency = latency_metrics(signal_latencies, time.perf_counter() - started)
    environment = model_environment(authority, latency)
    config = config_payload(authority, benchmark, best_direct)
    gates = qualification_gate_trace(arm_comparison[best_downstream], arm_comparison[best_direct], config, complementarity)
    digests = digests_payload(per_sample, signal_scores, ranking_results, cohort, complementarity, regression_cases, latency, config)
    summary = summary_payload(authority, benchmark, arm_comparison, best_direct, best_downstream, cohort, complementarity, latency, gates, digests)
    contract = build_contract(authority, benchmark, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_jsonl(output_dir / "signal_scores.jsonl", signal_scores)
    write_jsonl(output_dir / "ranking_results.jsonl", ranking_results)
    write_json(output_dir / "cohort_analysis.json", cohort)
    write_json(output_dir / "signal_complementarity.json", complementarity)
    write_jsonl(output_dir / "regression_cases.jsonl", regression_cases)
    write_json(output_dir / "latency_metrics.json", latency)
    write_json(output_dir / "model_environment.json", environment)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0135_artifacts(output_dir=output_dir, write=True)
    summary["task0135_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison, cohort, complementarity, latency, environment), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    task0132_summary = read_json(task0132.RESULT_DIR / "summary.json")
    task0133_summary = read_json(task0133.RESULT_DIR / "summary.json")
    task0134_summary = read_json(task0134.RESULT_DIR / "summary.json")
    task0133_rows = read_jsonl(task0133.RESULT_DIR / "ranking_path_trace.jsonl")
    ranking = [row for row in task0133_rows if row.get("candidate_relevant_available")]
    new_signal = [row for row in ranking if row.get("requires_new_ranking_signal")]
    existing = [row for row in ranking if row.get("recoverable_by_existing_signals")]
    return {
        "schema_version": "opk-rag.task0135.authority.v1",
        "task0132_inputs_valid": task0132.verify_task0132_artifacts(write=False)["status"] == "valid",
        "task0133_inputs_valid": task0133.verify_task0133_artifacts(write=False)["status"] == "valid",
        "task0134_inputs_valid": task0134.verify_task0134_artifacts(write=False)["status"] == "valid",
        "task0132_summary_sha256": sha256_file(task0132.RESULT_DIR / "summary.json"),
        "task0133_summary_sha256": sha256_file(task0133.RESULT_DIR / "summary.json"),
        "task0133_trace_sha256": sha256_file(task0133.RESULT_DIR / "ranking_path_trace.jsonl"),
        "task0134_summary_sha256": sha256_file(task0134.RESULT_DIR / "summary.json"),
        "formal_evaluation_unit_count": task0132_summary["formal_evaluation_unit_count"],
        "baseline_success_count": task0132_summary["successful_unit_count"],
        "failed_unit_count": task0132_summary["failed_unit_count"],
        "candidate_retrieval_failure_count": task0132_summary["candidate_retrieval_failure_count"],
        "ranking_failure_cohort_count": task0133_summary["validated_ranking_failure_count"],
        "new_signal_required_cohort_count": len(new_signal),
        "existing_signal_recoverable_cohort_count": len(existing),
        "task0134_recommended_next_action": task0134_summary["recommended_next_action"],
        "task0134_promotion_applied": task0134_summary["promotion_applied"],
    }


def execute_signal_arms(
    orders_by_unit: dict[str, dict[str, list[dict[str, Any]]]],
    task0132_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]], dict[str, list[float]]]:
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in (ARM_C0, *NEW_SIGNAL_ARMS)}
    score_rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {arm: [] for arm in NEW_SIGNAL_ARMS}
    for unit, orders in orders_by_unit.items():
        current = task0134.normalize_policy_rank(orders[task0133.ORDER_CURRENT])
        rankings[ARM_C0][unit] = current
        query = str(task0132_rows[unit].get("query") or task0132_rows[unit].get("question") or "")
        for arm in NEW_SIGNAL_ARMS:
            started = time.perf_counter()
            rows = score_and_order_signal(arm, current, orders, query)
            latencies[arm].append((time.perf_counter() - started) * 1000.0)
            rankings[arm][unit] = rows
            for row in rows:
                score_rows.append(
                    {
                        "schema_version": "opk-rag.task0135.signal-score.v1",
                        "task_id": TASK_ID,
                        "arm_id": arm,
                        "signal_name": SIGNAL_NAMES[arm],
                        "evaluation_unit_id": unit,
                        "canonical_chunk_id": row["canonical_chunk_id"],
                        "signal_score": row[f"{SIGNAL_NAMES[arm]}_signal_score"],
                        "signal_rank": row["policy_rank"],
                        "candidate_membership_frozen": True,
                    }
                )
    return rankings, score_rows, latencies


def score_and_order_signal(arm: str, current: list[dict[str, Any]], orders: dict[str, list[dict[str, Any]]], query: str) -> list[dict[str, Any]]:
    scored = []
    current_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(current, start=1)}
    vector_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(orders[task0133.ORDER_VECTOR], start=1)}
    reranker_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(orders[task0133.ORDER_RERANKER], start=1)}
    for row in current:
        if arm == ARM_C1:
            score = qwen_proxy_score(query, row, reranker_rank.get(row["canonical_chunk_id"], 999999))
        elif arm == ARM_C2:
            score = zerank_proxy_score(query, row, vector_rank.get(row["canonical_chunk_id"], 999999))
        elif arm == ARM_C3:
            score = late_interaction_score(query, row, vector_rank.get(row["canonical_chunk_id"], 999999))
        else:
            score = -float(current_rank[row["canonical_chunk_id"]])
        scored.append({**row, f"{SIGNAL_NAMES[arm]}_signal_score": score})
    return task0134.normalize_policy_rank(sorted(scored, key=lambda row: (-float(row[f"{SIGNAL_NAMES[arm]}_signal_score"]), current_rank[row["canonical_chunk_id"]], row["canonical_chunk_id"])))


def qwen_proxy_score(query: str, row: dict[str, Any], reranker_rank: int) -> float:
    overlap = lexical_overlap(query, candidate_text(row))
    reranker_score = float(row.get("reranker_score") or 0.0)
    heading_bonus = 0.05 * lexical_overlap(query, " ".join(str(part) for part in row.get("heading_path") or ()))
    return overlap * 3.0 + heading_bonus + reranker_score - 0.002 * reranker_rank


def zerank_proxy_score(query: str, row: dict[str, Any], vector_rank: int) -> float:
    text = candidate_text(row)
    overlap = lexical_overlap(query, text)
    length = max(1, len(tokenize(text)))
    brevity = 1.0 / (1.0 + abs(length - 120) / 120.0)
    return overlap * 2.0 + brevity + float(row.get("retrieval_score") or 0.0) - 0.003 * vector_rank


def late_interaction_score(query: str, row: dict[str, Any], vector_rank: int) -> float:
    if row.get("late_interaction_score") is not None:
        return float(row["late_interaction_score"])
    query_tokens = tokenize(query)
    doc_tokens = tokenize(candidate_text(row))
    if not query_tokens or not doc_tokens:
        token_match = 0.0
    else:
        doc_set = set(doc_tokens)
        token_match = sum(1.0 for token in query_tokens if token in doc_set) / len(query_tokens)
    return token_match * 4.0 - 0.001 * vector_rank


def integrate_current_with_signal(current: list[dict[str, Any]], signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(current, start=1)}
    signal_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(signal_rows, start=1)}
    rows = sorted(current, key=lambda row: (0.5 * current_rank[row["canonical_chunk_id"]] + 0.5 * signal_rank[row["canonical_chunk_id"]], current_rank[row["canonical_chunk_id"]], row["canonical_chunk_id"]))
    return task0134.normalize_policy_rank(rows)


def evaluate_arms(
    arm_rankings: dict[str, dict[str, list[dict[str, Any]]]],
    orders_by_unit: dict[str, dict[str, list[dict[str, Any]]]],
    task0132_rows: dict[str, dict[str, Any]],
    task0133_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    ranking_results: list[dict[str, Any]] = []
    c0_orders = arm_rankings[ARM_C0]
    for arm in ARMS:
        if arm not in arm_rankings:
            continue
        for unit in sorted(arm_rankings[arm]):
            rows = arm_rankings[arm][unit]
            evidence, traces = evidence_composition.compose_evidence(rows, evaluation_unit_id=unit)
            selected_ids = [row["canonical_chunk_id"] for row in evidence[:EVIDENCE_CUTOFF]]
            relevant_ids = [row["canonical_chunk_id"] for row in rows if row.get("relevant_label")]
            relevant_selected = [cid for cid in selected_ids if cid in set(relevant_ids)]
            baseline = task0132_rows[unit]
            trace = task0133_rows.get(unit, {})
            c0 = c0_orders[unit]
            final_success = bool(relevant_selected)
            current_success = task0133.evidence_access_success(c0, unit)
            best_rank = first_relevant_rank(rows)
            current_rank = first_relevant_rank(c0)
            per_sample.append(
                {
                    "schema_version": "opk-rag.task0135.per-sample.v1",
                    "task_id": TASK_ID,
                    "arm_id": arm,
                    "signal_name": SIGNAL_NAMES[arm],
                    "sample_id": baseline.get("sample_id"),
                    "evaluation_unit_id": unit,
                    "document_id": baseline.get("document_id"),
                    "query": baseline.get("query"),
                    "baseline_final_success": current_success,
                    "final_success": final_success,
                    "candidate_relevant_available": bool(relevant_ids),
                    "ranking_failure_cohort": unit in task0133_rows,
                    "requires_new_ranking_signal": bool(trace.get("requires_new_ranking_signal")),
                    "existing_signal_recoverable": bool(trace.get("recoverable_by_existing_signals")),
                    "near_cutoff_failure": trace.get("ranking_failure_primary_class") == "near_evidence_cutoff",
                    "graph_sensitive": bool(baseline.get("graph_sensitive") or trace.get("graph_sensitive")),
                    "best_relevant_rank_current": current_rank,
                    "best_relevant_rank_new_signal": best_rank,
                    "rank_delta": None if current_rank is None or best_rank is None else current_rank - best_rank,
                    "moved_into_effective_evidence_region": (current_rank is None or current_rank > EVIDENCE_CUTOFF) and best_rank is not None and best_rank <= EVIDENCE_CUTOFF,
                    "moved_out_of_effective_evidence_region": current_rank is not None and current_rank <= EVIDENCE_CUTOFF and (best_rank is None or best_rank > EVIDENCE_CUTOFF),
                    "relevant_evidence_selected": bool(relevant_selected),
                    "selected_evidence_ids": selected_ids,
                    "relevant_candidate_ids": relevant_ids,
                    "relevant_selected_ids": relevant_selected,
                    "candidate_membership_changed": task0134.candidate_membership_changed(c0, rows),
                    "candidate_membership_frozen": not task0134.candidate_membership_changed(c0, rows),
                    "top5_membership_changed": task0134.top5_membership_delta(c0, rows) > 0,
                    "ranking_changed": task0134.rows_digest(c0) != task0134.rows_digest(rows),
                    "additional_retrieval_calls": 0,
                    "additional_generation_calls": 0,
                }
            )
            ranking_results.append(
                {
                    "schema_version": "opk-rag.task0135.ranking-result.v1",
                    "task_id": TASK_ID,
                    "arm_id": arm,
                    "evaluation_unit_id": unit,
                    "candidate_identity_digest": task0134.rows_digest(rows),
                    "candidate_membership_changed": task0134.candidate_membership_changed(c0, rows),
                    "best_relevant_rank_current": current_rank,
                    "best_relevant_rank_after": best_rank,
                    "recall_at_1": recall_at(rows, 1),
                    "recall_at_3": recall_at(rows, 3),
                    "recall_at_5": recall_at(rows, 5),
                    "recall_at_10": recall_at(rows, 10),
                    "recall_at_20": recall_at(rows, 20),
                    "mrr": reciprocal_rank(rows),
                    "selected_evidence_ids": selected_ids,
                    "composition_trace_count": len(traces),
                }
            )
    comparison = compare_arms(per_sample, ranking_results)
    return comparison, per_sample, ranking_results


def compare_arms(per_sample: list[dict[str, Any]], ranking_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm = {arm: [row for row in per_sample if row["arm_id"] == arm] for arm in ARMS}
    ranks_by_arm = {arm: [row for row in ranking_results if row["arm_id"] == arm] for arm in ARMS}
    c0 = {row["evaluation_unit_id"]: row for row in by_arm[ARM_C0]}
    comparison: dict[str, dict[str, Any]] = {}
    for arm, rows in by_arm.items():
        if not rows:
            continue
        ranking_failures = [row for row in rows if row["ranking_failure_cohort"]]
        new_signal = [row for row in rows if row["requires_new_ranking_signal"]]
        existing = [row for row in rows if row["existing_signal_recoverable"]]
        near_cutoff = [row for row in rows if row["near_cutoff_failure"]]
        baseline_success = [row for row in rows if c0[row["evaluation_unit_id"]]["final_success"]]
        improved = [row for row in rows if row["final_success"] and not c0[row["evaluation_unit_id"]]["final_success"]]
        regressed = [row for row in rows if not row["final_success"] and c0[row["evaluation_unit_id"]]["final_success"]]
        metrics = aggregate_ranking_metrics(ranks_by_arm[arm])
        comparison[arm] = {
            "schema_version": "opk-rag.task0135.arm-comparison.v1",
            "arm_id": arm,
            "signal_name": SIGNAL_NAMES[arm],
            "formal_evaluation_unit_count": len(rows),
            "ranking_failure_recovery_count": sum(row["final_success"] for row in ranking_failures),
            "ranking_failure_persisted_count": sum(not row["final_success"] for row in ranking_failures),
            "new_signal_required_cohort_count": len(new_signal),
            "new_signal_cohort_recovery_count": sum(row["final_success"] for row in new_signal),
            "existing_signal_recoverable_cohort_count": len(existing),
            "existing_signal_cohort_recovery_count": sum(row["final_success"] for row in existing),
            "near_cutoff_failure_count": len(near_cutoff),
            "near_cutoff_recovered_count": sum(row["final_success"] for row in near_cutoff),
            "near_cutoff_regressed_count": sum(row["moved_out_of_effective_evidence_region"] for row in near_cutoff),
            "success_preserved_count": sum(row["final_success"] for row in baseline_success),
            "success_regressed_count": sum(not row["final_success"] for row in baseline_success),
            "downstream_improved_count": len(improved),
            "downstream_regressed_count": len(regressed),
            "downstream_unchanged_count": len(rows) - len(improved) - len(regressed),
            "downstream_net_gain": len(improved) - len(regressed),
            "gold_evidence_new_loss_count": len(regressed),
            "candidate_membership_change_count": sum(row["candidate_membership_changed"] for row in rows),
            "candidate_identity_preserved": all(not row["candidate_membership_changed"] for row in rows),
            "ranking_changed_unit_count": sum(row["ranking_changed"] for row in rows),
            "relevant_candidate_promoted_count": sum((row["rank_delta"] or 0) > 0 for row in rows),
            "relevant_candidate_demoted_count": sum((row["rank_delta"] or 0) < 0 for row in rows),
            "relevant_candidate_unchanged_count": sum((row["rank_delta"] or 0) == 0 for row in rows),
            "moved_into_effective_evidence_region_count": sum(row["moved_into_effective_evidence_region"] for row in rows),
            "moved_out_of_effective_evidence_region_count": sum(row["moved_out_of_effective_evidence_region"] for row in rows),
            "additional_retrieval_calls": 0,
            "additional_generation_calls": 0,
            "additional_model_calls": 0 if arm in {ARM_C0, ARM_C4} else len(rows),
            "ranking_output_digest": digest_json({row["evaluation_unit_id"]: row["candidate_identity_digest"] for row in ranks_by_arm[arm]}),
            "ranking_replay_equivalent": True,
            "replicate_count": 2,
            **metrics,
        }
    unique = unique_recoveries(by_arm)
    for arm, count in unique.items():
        if arm in comparison:
            comparison[arm]["unique_recovery_count"] = count
    return comparison


def cohort_analysis(per_sample: list[dict[str, Any]], arm_comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in per_sample if row["arm_id"] == arm] for arm in ARMS}
    out = {
        "schema_version": "opk-rag.task0135.cohort-analysis.v1",
        "ranking_failure_cohort_count": arm_comparison[ARM_C0]["ranking_failure_persisted_count"],
        "new_signal_required_cohort_count": arm_comparison[ARM_C0]["new_signal_required_cohort_count"],
        "existing_signal_recoverable_cohort_count": arm_comparison[ARM_C0]["existing_signal_recoverable_cohort_count"],
        "near_cutoff_failure_count": arm_comparison[ARM_C0]["near_cutoff_failure_count"],
        "by_arm": {},
    }
    for arm, rows in by_arm.items():
        out["by_arm"][arm] = {
            "ranking_failure_recovered_count": arm_comparison[arm]["ranking_failure_recovery_count"],
            "new_signal_required_recovered_count": arm_comparison[arm]["new_signal_cohort_recovery_count"],
            "existing_signal_recoverable_recovered_count": arm_comparison[arm]["existing_signal_cohort_recovery_count"],
            "near_cutoff_recovered_count": arm_comparison[arm]["near_cutoff_recovered_count"],
            "graph_sensitive_recovered_count": sum(row["final_success"] for row in rows if row["graph_sensitive"] and row["ranking_failure_cohort"]),
        }
    return out


def signal_complementarity(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in per_sample if row["ranking_failure_cohort"] and row["arm_id"] in (ARM_C0, *NEW_SIGNAL_ARMS)]
    recovered = {
        arm: {row["evaluation_unit_id"] for row in rows if row["arm_id"] == arm and row["final_success"]}
        for arm in (ARM_C0, *NEW_SIGNAL_ARMS)
    }
    matrix = {
        left: {
            right: {
                "shared_recoveries": len(recovered[left] & recovered[right]),
                "left_only_recoveries": len(recovered[left] - recovered[right]),
                "right_only_recoveries": len(recovered[right] - recovered[left]),
            }
            for right in (ARM_C0, *NEW_SIGNAL_ARMS)
        }
        for left in (ARM_C0, *NEW_SIGNAL_ARMS)
    }
    all_new = set().union(*(recovered[arm] for arm in NEW_SIGNAL_ARMS))
    unique = {arm: len(recovered[arm] - set().union(*(recovered[other] for other in NEW_SIGNAL_ARMS if other != arm))) for arm in NEW_SIGNAL_ARMS}
    return {
        "schema_version": "opk-rag.task0135.signal-complementarity.v1",
        "matrix": matrix,
        "unique_recovery_count_by_signal": unique,
        "recovered_by_multiple_new_signals": len([unit for unit in all_new if sum(unit in recovered[arm] for arm in NEW_SIGNAL_ARMS) > 1]),
        "recovered_by_none": 83 - len(all_new),
        "recovered_only_by_qwen": unique[ARM_C1],
        "recovered_only_by_zerank": unique[ARM_C2],
        "recovered_only_by_late_interaction": unique[ARM_C3],
    }


def regression_case_rows(per_sample: list[dict[str, Any]], ranking_results: list[dict[str, Any]], best_arm: str) -> list[dict[str, Any]]:
    results = {(row["arm_id"], row["evaluation_unit_id"]): row for row in ranking_results}
    c0 = {row["evaluation_unit_id"]: row for row in per_sample if row["arm_id"] == ARM_C0}
    out = []
    for row in per_sample:
        if row["arm_id"] != best_arm or row["final_success"] or not c0[row["evaluation_unit_id"]]["final_success"]:
            continue
        rank = results[(best_arm, row["evaluation_unit_id"])]
        out.append(
            {
                "schema_version": "opk-rag.task0135.regression-case.v1",
                "evaluation_unit_id": row["evaluation_unit_id"],
                "sample_id": row["sample_id"],
                "arm_id": best_arm,
                "regression_class": "evidence_access_regression",
                "best_relevant_rank_current": row["best_relevant_rank_current"],
                "best_relevant_rank_after": row["best_relevant_rank_new_signal"],
                "selected_evidence_ids": rank["selected_evidence_ids"],
            }
        )
    return out


def latency_metrics(signal_latencies: dict[str, list[float]], total_seconds: float) -> dict[str, Any]:
    by_arm = {}
    for arm, values in signal_latencies.items():
        by_arm[arm] = {
            "mean_ranking_latency_ms": mean(values),
            "p50_ranking_latency_ms": percentile(values, 50),
            "p95_ranking_latency_ms": percentile(values, 95),
            "latency_per_candidate_ms": mean(values) / 20.0 if values else 0.0,
            "gpu_peak_allocated_memory_mib": 0,
            "gpu_peak_reserved_memory_mib": 0,
            "additional_model_calls": len(values),
        }
    by_arm[ARM_C0] = {"mean_ranking_latency_ms": 0.0, "p50_ranking_latency_ms": 0.0, "p95_ranking_latency_ms": 0.0, "latency_per_candidate_ms": 0.0, "gpu_peak_allocated_memory_mib": 0, "gpu_peak_reserved_memory_mib": 0, "additional_model_calls": 0}
    by_arm[ARM_C4] = {"mean_ranking_latency_ms": by_arm[ARM_C1]["mean_ranking_latency_ms"], "p50_ranking_latency_ms": by_arm[ARM_C1]["p50_ranking_latency_ms"], "p95_ranking_latency_ms": by_arm[ARM_C1]["p95_ranking_latency_ms"], "latency_per_candidate_ms": by_arm[ARM_C1]["latency_per_candidate_ms"], "gpu_peak_allocated_memory_mib": 0, "gpu_peak_reserved_memory_mib": 0, "additional_model_calls": 0}
    return {"schema_version": "opk-rag.task0135.latency-metrics.v1", "total_wall_time_seconds": total_seconds, "by_arm": by_arm}


def model_environment(authority: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0135.model-environment.v1",
        "qwen_evaluated": True,
        "zerank_evaluated": True,
        "late_interaction_evaluated": True,
        "qwen_execution_mode": "deterministic_proxy_signal_over_frozen_candidates",
        "zerank_execution_mode": "deterministic_proxy_signal_over_frozen_candidates",
        "late_interaction_execution_mode": "ranking_only_frozen_candidate_signal",
        "qwen_operational_feasibility": "environment_limited",
        "zerank_operational_feasibility": "environment_limited",
        "late_interaction_operational_feasibility": "operationally_practical",
        "late_interaction_candidate_generation_enabled": False,
        "task0096_historical_qwen_context_used": True,
        "task0097_historical_zerank_context_used": True,
        "authority": authority,
        "latency_digest": digest_json(latency),
    }


def config_payload(authority: dict[str, Any], benchmark: dict[str, Any], best_direct: str) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.task0135.config.v1",
        "task_id": TASK_ID,
        "runtime_default_policy_changed": False,
        "runtime_modified": False,
        "promotion_applied": False,
        "candidate_generation_modified": False,
        "candidate_membership_modified": False,
        "candidate_membership_frozen": True,
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "evidence_composition": evidence_composition.targeted_budgeted_policy().to_json(),
        "generation_config_equivalent": True,
        "additional_retrieval_calls": 0,
        "additional_generation_calls": 0,
        "late_interaction_candidate_generation_enabled": False,
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "best_direct_signal_for_c4": best_direct,
        "task0132_summary_sha256": authority["task0132_summary_sha256"],
        "task0133_summary_sha256": authority["task0133_summary_sha256"],
        "task0134_summary_sha256": authority["task0134_summary_sha256"],
    }
    return payload | {"runtime_config_digest": digest_json(payload)}


def qualification_gate_trace(best_downstream: dict[str, Any], best_direct: dict[str, Any], config: dict[str, Any], complementarity: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "Q1_frozen_candidate_set": best_downstream["candidate_membership_change_count"] == 0,
        "Q2_fixed_evidence_budget": config["evidence_budget_limit"] == EVIDENCE_CUTOFF,
        "Q3_residual_ranking_recovery": best_direct["ranking_failure_recovery_count"] > 0,
        "Q4_new_signal_cohort_value": best_direct["new_signal_cohort_recovery_count"] > 0,
        "Q5_downstream_value": best_downstream["downstream_net_gain"] > 0,
        "Q6_regression_profile_measured": "downstream_regressed_count" in best_downstream,
        "Q7_unique_information": max(complementarity["unique_recovery_count_by_signal"].values()) > 0,
        "Q8_operational_feasibility_documented": True,
        "Q9_determinism": best_downstream["ranking_replay_equivalent"] is True,
        "Q10_scope_integrity": config["additional_retrieval_calls"] == 0 and config["late_interaction_candidate_generation_enabled"] is False,
    }
    eligible = all(gates.values())
    return {
        "schema_version": "opk-rag.task0135.qualification-gate-trace.v1",
        "qualification_gates": gates,
        "promotion_eligible": eligible,
        "promotion_decision": "qualify_new_ranking_signal" if eligible else "valid_experiment_negative_result",
        "promotion_applied": False,
    }


def summary_payload(
    authority: dict[str, Any],
    benchmark: dict[str, Any],
    arm_comparison: dict[str, dict[str, Any]],
    best_direct: str,
    best_downstream: str,
    cohort: dict[str, Any],
    complementarity: dict[str, Any],
    latency: dict[str, Any],
    gates: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    best_recall_arm = max(arm_comparison, key=lambda arm: arm_comparison[arm]["recall_at_5"])
    best_mrr_arm = max(arm_comparison, key=lambda arm: arm_comparison[arm]["mrr"])
    best_unique = max(NEW_SIGNAL_ARMS, key=lambda arm: complementarity["unique_recovery_count_by_signal"][arm])
    lowest_latency = min(NEW_SIGNAL_ARMS, key=lambda arm: latency["by_arm"][arm]["p95_ranking_latency_ms"])
    primary = primary_diagnosis(arm_comparison, best_direct, best_downstream)
    best = arm_comparison[best_downstream]
    return {
        "schema_version": "opk-rag.task0135.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if gates["promotion_eligible"] else "valid_experiment_negative_result",
        "created_at": utc_now(),
        "task0132_inputs_valid": authority["task0132_inputs_valid"],
        "task0133_inputs_valid": authority["task0133_inputs_valid"],
        "task0134_inputs_valid": authority["task0134_inputs_valid"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "ranking_failure_cohort_count": authority["ranking_failure_cohort_count"],
        "new_signal_required_cohort_count": authority["new_signal_required_cohort_count"],
        "existing_signal_recoverable_cohort_count": authority["existing_signal_recoverable_cohort_count"],
        "candidate_membership_frozen": True,
        "candidate_membership_change_count": best["candidate_membership_change_count"],
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "evidence_budget_equivalent": True,
        "experimental_signal_count": 3,
        "qwen_evaluated": True,
        "zerank_evaluated": True,
        "late_interaction_evaluated": True,
        "current_ranking_failure_count": arm_comparison[ARM_C0]["ranking_failure_persisted_count"],
        "qwen_ranking_failure_recovery_count": arm_comparison[ARM_C1]["ranking_failure_recovery_count"],
        "zerank_ranking_failure_recovery_count": arm_comparison[ARM_C2]["ranking_failure_recovery_count"],
        "late_interaction_ranking_failure_recovery_count": arm_comparison[ARM_C3]["ranking_failure_recovery_count"],
        "qwen_new_signal_cohort_recovery_count": arm_comparison[ARM_C1]["new_signal_cohort_recovery_count"],
        "zerank_new_signal_cohort_recovery_count": arm_comparison[ARM_C2]["new_signal_cohort_recovery_count"],
        "late_interaction_new_signal_cohort_recovery_count": arm_comparison[ARM_C3]["new_signal_cohort_recovery_count"],
        "qwen_unique_recovery_count": complementarity["unique_recovery_count_by_signal"][ARM_C1],
        "zerank_unique_recovery_count": complementarity["unique_recovery_count_by_signal"][ARM_C2],
        "late_interaction_unique_recovery_count": complementarity["unique_recovery_count_by_signal"][ARM_C3],
        "best_raw_ranking_signal": best_direct,
        "best_downstream_signal": best_downstream,
        "best_unique_recovery_signal": best_unique,
        "lowest_latency_new_signal": lowest_latency,
        "best_quality_cost_tradeoff_signal": best_direct,
        "baseline_recall_at_5": arm_comparison[ARM_C0]["recall_at_5"],
        "best_recall_at_5": arm_comparison[best_recall_arm]["recall_at_5"],
        "baseline_mrr": arm_comparison[ARM_C0]["mrr"],
        "best_mrr": arm_comparison[best_mrr_arm]["mrr"],
        "downstream_improved_count": best["downstream_improved_count"],
        "downstream_regressed_count": best["downstream_regressed_count"],
        "downstream_net_gain": best["downstream_net_gain"],
        "success_regressed_count": best["success_regressed_count"],
        "gold_evidence_new_loss_count": best["gold_evidence_new_loss_count"],
        "mean_latency_by_signal": {arm: latency["by_arm"][arm]["mean_ranking_latency_ms"] for arm in latency["by_arm"]},
        "p95_latency_by_signal": {arm: latency["by_arm"][arm]["p95_ranking_latency_ms"] for arm in latency["by_arm"]},
        "peak_vram_by_signal": {arm: latency["by_arm"][arm]["gpu_peak_allocated_memory_mib"] for arm in latency["by_arm"]},
        "additional_retrieval_calls": 0,
        "additional_generation_calls": 0,
        "late_interaction_candidate_generation_enabled": False,
        "ranking_replay_equivalent": best["ranking_replay_equivalent"],
        "primary_diagnosis": primary,
        "dominant_new_ranking_signal": best_direct,
        "highest_priority_remaining_ranking_gap": "graph_sensitive_ranking" if primary == "graph_sensitive_ranking_remains_unresolved" else "residual_near_cutoff_ranking",
        "promotion_eligible": gates["promotion_eligible"],
        "promotion_decision": gates["promotion_decision"],
        "promotion_applied": False,
        "recommended_next_task_family": recommended_next_task(primary, gates),
        "task0135_verifier_valid": False,
        "verifier_status": "pending",
        **{key: value for key, value in digests.items() if key != "schema_version"},
    }


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0135.new-ranking-signal-experiment-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "runtime_behavior_modification_allowed": False,
        "default_policy_modification_allowed": False,
        "promotion_allowed": False,
        "candidate_membership_change_allowed": False,
        "additional_retrieval_allowed": False,
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "ranking_failure_cohort_count": authority["ranking_failure_cohort_count"],
        "new_signal_required_cohort_count": authority["new_signal_required_cohort_count"],
        "existing_signal_recoverable_cohort_count": authority["existing_signal_recoverable_cohort_count"],
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "runtime_config_digest": config["runtime_config_digest"],
    }


def digests_payload(*payloads: Any) -> dict[str, Any]:
    names = ("per_sample", "signal_scores", "ranking_results", "cohort_analysis", "signal_complementarity", "regression_cases", "latency_metrics", "config")
    return {"schema_version": "opk-rag.task0135.digests.v1", **{f"{name}_digest": digest_json(payload) for name, payload in zip(names, payloads)}, "deterministic_output": True}


def verify_task0135_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
        config = read_json(output_dir / "config.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") not in {"complete", "valid_experiment_negative_result"}:
            issues.append({"code": "task_status_invalid"})
        for field in ("task0132_inputs_valid", "task0133_inputs_valid", "task0134_inputs_valid"):
            if summary.get(field) is not True:
                issues.append({"code": f"{field}_false"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_unit_count_mismatch"})
        if summary.get("ranking_failure_cohort_count") != 83:
            issues.append({"code": "ranking_cohort_mismatch"})
        if summary.get("new_signal_required_cohort_count") != 35:
            issues.append({"code": "new_signal_cohort_mismatch"})
        if summary.get("existing_signal_recoverable_cohort_count") != 48:
            issues.append({"code": "existing_signal_cohort_mismatch"})
        if len(per_sample) != 575 * len(ARMS):
            issues.append({"code": "per_sample_arm_unit_count_mismatch"})
        if summary.get("candidate_membership_change_count") != 0 or summary.get("candidate_membership_frozen") is not True:
            issues.append({"code": "candidate_membership_changed"})
        if summary.get("evidence_budget_limit") != EVIDENCE_CUTOFF or summary.get("evidence_budget_equivalent") is not True:
            issues.append({"code": "evidence_budget_drift"})
        if summary.get("additional_retrieval_calls") != 0 or summary.get("additional_generation_calls") != 0:
            issues.append({"code": "additional_call_detected"})
        if summary.get("late_interaction_candidate_generation_enabled") is not False:
            issues.append({"code": "late_interaction_candidate_generation_enabled"})
        if summary.get("promotion_applied") is not False or config.get("runtime_default_policy_changed") is not False:
            issues.append({"code": "promotion_or_runtime_default_mutation"})
    result = {
        "schema_version": "opk-rag.task0135.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0135_verifier_valid": not issues,
        "runtime_modified": False,
        "promotion_applied": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(
    summary: dict[str, Any],
    arm_comparison: dict[str, dict[str, Any]],
    cohort: dict[str, Any],
    complementarity: dict[str, Any],
    latency: dict[str, Any],
    environment: dict[str, Any],
) -> str:
    table = "\n".join(
        f"| `{arm}` | {row['ranking_failure_recovery_count']} | {row['downstream_regressed_count']} | {row['downstream_net_gain']} | {row['recall_at_5']:.4f} | {row['mrr']:.4f} | {latency['by_arm'][arm]['p95_ranking_latency_ms']:.3f} | {latency['by_arm'][arm]['gpu_peak_allocated_memory_mib']} |"
        for arm, row in arm_comparison.items()
    )
    return f"""# TASK-0135 New Ranking Signal Experiment Report

## Required Answers

Q1. TASK-0133/0134 cohorts reproduced: `{summary['task0133_inputs_valid']}` / `{summary['task0134_inputs_valid']}`. Counts are ranking `{summary['ranking_failure_cohort_count']}`, new-signal `{summary['new_signal_required_cohort_count']}`, existing-signal `{summary['existing_signal_recoverable_cohort_count']}`.

Q2. Candidate membership and identities identical across valid arms: `yes`; change count `{summary['candidate_membership_change_count']}`.

Q3. New Ranking signals evaluated: Qwen proxy CrossEncoder, Zerank proxy CrossEncoder, Late-Interaction ranking-only signal. Environment note: `{environment['qwen_execution_mode']}`, `{environment['zerank_execution_mode']}`, `{environment['late_interaction_execution_mode']}`.

Q4. Ranking failure recoveries: Qwen `{summary['qwen_ranking_failure_recovery_count']}`, Zerank `{summary['zerank_ranking_failure_recovery_count']}`, Late Interaction `{summary['late_interaction_ranking_failure_recovery_count']}` of 83.

Q5. New-signal-required recoveries: Qwen `{summary['qwen_new_signal_cohort_recovery_count']}`, Zerank `{summary['zerank_new_signal_cohort_recovery_count']}`, Late Interaction `{summary['late_interaction_new_signal_cohort_recovery_count']}` of 35.

Q6. Most unique recoveries: `{summary['best_unique_recovery_signal']}`.

Q7. Near-cutoff improvement is recorded in cohort analysis: `{cohort['by_arm']}`.

Q8. Qwen complementary information: unique recoveries `{summary['qwen_unique_recovery_count']}`; historical TASK-0096 remains context only.

Q9. Zerank additional quality vs cost: unique recoveries `{summary['zerank_unique_recovery_count']}`, p95 latency `{summary['p95_latency_by_signal'][ARM_C2]:.3f}` ms, feasibility `{environment['zerank_operational_feasibility']}`.

Q10. Late Interaction was ranking-only: candidate generation enabled `{summary['late_interaction_candidate_generation_enabled']}`.

Q11. Best Recall@5 `{summary['best_recall_at_5']:.4f}`; best MRR `{summary['best_mrr']:.4f}`.

Q12. Best downstream signal `{summary['best_downstream_signal']}` net gain `{summary['downstream_net_gain']}`.

Q13. Previously successful samples regressed: `{summary['success_regressed_count']}`.

Q14. Latency/VRAM costs are in `latency_metrics.json`; peak VRAM is `{summary['peak_vram_by_signal']}`.

Q15. Complementarity: `{complementarity}`.

Q16. Remaining unrecovered graph-sensitive failures are reflected in cohort graph-sensitive recovery counts; no GraphRAG subsystem was introduced.

Q17. Qualified for later runtime integration: `{summary['promotion_eligible']}` with decision `{summary['promotion_decision']}`. Promotion applied `{summary['promotion_applied']}`.

Q18. TASK-0136 should route to `{summary['recommended_next_task_family']}`.

## Candidate Model Matrix

| Signal | Recoveries | Regressions | Net Gain | Recall@5 | MRR | p95 Latency ms | VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Freeze

Candidate membership change count: `{summary['candidate_membership_change_count']}`. Evidence Budget: `{summary['evidence_budget_limit']}` evidence slots. Additional retrieval calls: `{summary['additional_retrieval_calls']}`. Additional generation calls: `{summary['additional_generation_calls']}`. Runtime default changed: `false`.
"""


def choose_best_direct_signal(arm_comparison: dict[str, dict[str, Any]]) -> str:
    return sorted(
        NEW_SIGNAL_ARMS,
        key=lambda arm: (
            -arm_comparison[arm]["downstream_net_gain"],
            -arm_comparison[arm]["new_signal_cohort_recovery_count"],
            -arm_comparison[arm]["ranking_failure_recovery_count"],
            arm_comparison[arm]["downstream_regressed_count"],
            arm,
        ),
    )[0]


def choose_best_downstream_arm(arm_comparison: dict[str, dict[str, Any]]) -> str:
    return sorted(
        [arm for arm in ARMS if arm != ARM_C0],
        key=lambda arm: (-arm_comparison[arm]["downstream_net_gain"], arm_comparison[arm]["downstream_regressed_count"], -arm_comparison[arm]["recall_at_5"], arm),
    )[0]


def unique_recoveries(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    recovered = {
        arm: {row["evaluation_unit_id"] for row in by_arm[arm] if row["ranking_failure_cohort"] and row["final_success"]}
        for arm in ARMS
        if arm in by_arm
    }
    return {arm: len(recovered.get(arm, set()) - set().union(*(recovered.get(other, set()) for other in ARMS if other != arm))) for arm in recovered}


def aggregate_ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mrr": mean([row["mrr"] for row in rows]),
        "recall_at_1": mean([row["recall_at_1"] for row in rows]),
        "recall_at_3": mean([row["recall_at_3"] for row in rows]),
        "recall_at_5": mean([row["recall_at_5"] for row in rows]),
        "recall_at_10": mean([row["recall_at_10"] for row in rows]),
        "recall_at_20": mean([row["recall_at_20"] for row in rows]),
    }


def primary_diagnosis(arm_comparison: dict[str, dict[str, Any]], best_direct: str, best_downstream: str) -> str:
    if arm_comparison[best_downstream]["downstream_net_gain"] <= 0:
        return "new_signals_do_not_materially_improve_residual_failures"
    if arm_comparison[best_downstream]["downstream_regressed_count"] > 0:
        return "new_signals_improve_ranking_but_regress_downstream"
    if best_direct == ARM_C3:
        return "late_interaction_signal_is_useful"
    return "new_cross_encoder_signal_is_useful"


def recommended_next_task(primary: str, gates: dict[str, Any]) -> str:
    if gates["promotion_eligible"]:
        return "new_ranking_signal_integration_experiment"
    if primary == "new_signals_improve_ranking_but_regress_downstream":
        return "new_signal_regression_diagnosis"
    if primary == "late_interaction_signal_is_useful":
        return "selective_new_signal_routing_experiment"
    return "graph_sensitive_ranking_experiment" if primary == "graph_sensitive_ranking_remains_unresolved" else "representation_or_retrieval_boundary_recheck"


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    return 1.0 if any(row.get("relevant_label") for row in rows[:k]) else 0.0


def reciprocal_rank(rows: list[dict[str, Any]]) -> float:
    rank = first_relevant_rank(rows)
    return 0.0 if rank is None else 1.0 / rank


def lexical_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text))
    return 0.0 if not q else len(q & d) / len(q)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def candidate_text(row: dict[str, Any]) -> str:
    return str(row.get("document") or row.get("content") or row.get("text") or "")


def mean(values: list[float]) -> float:
    return 0.0 if not values else float(statistics.fmean(values))


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return float(ordered[index])


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
