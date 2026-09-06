from __future__ import annotations

from collections import Counter
import math
import statistics
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import rank_fusion
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark, guarded_rank_fusion_unit
import opk_rag.evaluation.task0125_multi_lane_candidate_composition_ablation as task0125
import opk_rag.evaluation.task0127_multi_lane_evidence_conversion_failure_diagnosis as task0127
import opk_rag.evaluation.task0132_post_promotion_residual_failure_rebaseline as task0132
from opk_rag.runtime_v2 import evidence_composition
from opk_rag.runtime_v2.late_interaction_policy import DeterministicGuardV2
from opk_rag.runtime_v2.multi_lane_candidate_composition import SOURCE_DENSE_ONLY, SOURCE_LATE_ONLY, SOURCE_OVERLAP, candidate_source_type
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA


TASK_ID = "TASK-0133"
EXPERIMENT_ID = "task0133-residual-ranking-failure-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0133_residual_ranking_failure_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0133_RESIDUAL_RANKING_FAILURE_DIAGNOSIS_REPORT.md"

ORDER_CURRENT = "current_default"
ORDER_RETRIEVER = "retriever_native"
ORDER_VECTOR = "vector_baseline"
ORDER_RERANKER = "reranker_direct"
ORDER_FUSION = "rank_fusion"
ORDER_LEXICAL = "lexical_only_unavailable"
ORDERINGS = (ORDER_CURRENT, ORDER_RETRIEVER, ORDER_VECTOR, ORDER_RERANKER, ORDER_FUSION)
EVIDENCE_CUTOFF = evidence_composition.EVIDENCE_BUDGET_LIMIT

PRIMARY_CLASSES = (
    "retriever_order_already_weak",
    "reranker_false_demotion",
    "reranker_insufficient_promotion",
    "fusion_displacement",
    "guard_missed_recovery",
    "guard_wrong_branch",
    "multi_lane_ordering_conflict",
    "candidate_crowding",
    "score_calibration_conflict",
    "near_evidence_cutoff",
    "new_semantic_ranking_signal_required",
    "benchmark_or_label_issue",
    "unknown",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "ranking_path_trace.jsonl",
    "diagnostic_feature_matrix.jsonl",
    "rank_transition_matrix.json",
    "counterfactual_recovery.json",
    "ranking_quality_metrics.json",
    "failure_taxonomy.json",
    "signal_analysis.json",
    "graph_sensitive_ranking_audit.json",
    "control_cohort_comparison.json",
    "config.json",
    "digests.json",
    "verification.json",
)


def run_task0133_residual_ranking_failure_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    executions = build_frozen_executions(baseline)
    task0132_rows = read_jsonl(task0132.RESULT_DIR / "per_sample.jsonl")
    ranking_units = [row["evaluation_unit_id"] for row in task0132_rows if row.get("first_failure_stage") == "ranking"]
    control_units = successful_control_units(task0132_rows)
    orders_by_unit = {unit: reconstruct_orders(unit, baseline[unit], executions[unit]) for unit in sorted(set(ranking_units) | set(control_units))}
    traces = [diagnose_ranking_unit(unit, orders_by_unit[unit], executions[unit], task0132_by_unit(task0132_rows)[unit]) for unit in ranking_units]
    controls = [control_feature_row(unit, orders_by_unit[unit], executions[unit], task0132_by_unit(task0132_rows)[unit]) for unit in control_units]
    feature_rows = [feature_matrix_row(row) for row in traces]
    transition = rank_transition_matrix(traces)
    counterfactual = counterfactual_recovery(traces)
    ranking_metrics = ranking_quality_metrics({unit: orders_by_unit[unit] for unit in ranking_units})
    taxonomy = failure_taxonomy(traces)
    signal = signal_analysis(traces, controls)
    graph = graph_sensitive_ranking_audit(traces)
    control = control_cohort_comparison(traces, controls)
    config = config_payload(authority, benchmark)
    digests = digests_payload(traces, feature_rows, config)
    summary = summary_payload(authority, benchmark, traces, transition, counterfactual, taxonomy, graph, signal, config, digests)
    contract = build_contract(authority, benchmark, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "ranking_path_trace.jsonl", traces)
    write_jsonl(output_dir / "diagnostic_feature_matrix.jsonl", feature_rows)
    write_json(output_dir / "rank_transition_matrix.json", transition)
    write_json(output_dir / "counterfactual_recovery.json", counterfactual)
    write_json(output_dir / "ranking_quality_metrics.json", ranking_metrics)
    write_json(output_dir / "failure_taxonomy.json", taxonomy)
    write_json(output_dir / "signal_analysis.json", signal)
    write_json(output_dir / "graph_sensitive_ranking_audit.json", graph)
    write_json(output_dir / "control_cohort_comparison.json", control)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0133_artifacts(output_dir=output_dir, write=True)
    summary["task0133_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, transition, counterfactual, taxonomy, graph, signal), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    verification = task0132.verify_task0132_artifacts(write=False)
    summary = read_json(task0132.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0133.authority.v1",
        "task0132_verifier_valid": verification["status"] == "valid",
        "task0132_summary_sha256": sha256_file(task0132.RESULT_DIR / "summary.json"),
        "task0132_ranking_failure_count": summary["ranking_failure_count"],
        "task0132_formal_evaluation_unit_count": summary["formal_evaluation_unit_count"],
        "task0132_runtime_policy_digest": summary["runtime_policy_digest"],
        "task0132_runtime_config_digest": summary["runtime_config_digest"],
        "task0132_candidate_membership_change_count": summary["candidate_membership_change_count"],
        "task0132_recommended_next_task_family": summary["recommended_next_task_family"],
    }


def build_frozen_executions(baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    retriever = task0127.task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    outputs = task0125.execute_task0125_arms(
        baseline,
        retriever,
        DeterministicGuardV2(),
        {task0125.C2: task0125.lane_policies(task0127.RETRIEVAL_TOP_K)[task0125.C2]},
    )
    return outputs[task0125.C2]["executions"]


def reconstruct_orders(unit: str, baseline_rows: list[dict[str, Any]], execution: Any) -> dict[str, list[dict[str, Any]]]:
    retriever_native = task0127.assign_pre_bge_rank(execution.final_candidates)
    reranker_direct = task0127.bge_ranking(retriever_native)
    fusion = rank_fusion(retriever_native, reranker_direct, {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA})
    current = guarded_rank_fusion_unit(retriever_native, reranker_direct, fusion)
    return {
        ORDER_CURRENT: assign_order_rank(current),
        ORDER_RETRIEVER: assign_order_rank(retriever_native),
        ORDER_RERANKER: assign_order_rank(reranker_direct),
        ORDER_FUSION: assign_order_rank(fusion),
        ORDER_VECTOR: assign_order_rank(vector_baseline_order(execution.final_candidates, baseline_rows)),
    }


def vector_baseline_order(current_members: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rank = {row["canonical_chunk_id"]: int(row["retrieval_rank"]) for row in baseline_rows}
    rows = []
    for row in current_members:
        dense_rank = row.get("dense_rank") or baseline_rank.get(row["canonical_chunk_id"])
        rows.append({**row, "vector_baseline_rank": dense_rank})
    return sorted(rows, key=lambda row: (row.get("vector_baseline_rank") is None, row.get("vector_baseline_rank") or 10**9, row.get("late_interaction_rank") or 10**9, row["canonical_chunk_id"]))


def assign_order_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "policy_rank": index} for index, row in enumerate(rows, start=1)]


def diagnose_ranking_unit(unit: str, orders: dict[str, list[dict[str, Any]]], execution: Any, task0132_row: dict[str, Any]) -> dict[str, Any]:
    ranks = {name: first_relevant_rank(rows) for name, rows in orders.items()}
    evidence_hits = {name: evidence_access_success(rows, unit) for name, rows in orders.items()}
    best_row = best_relevant_row(orders[ORDER_CURRENT])
    guard = execution.guard_decision or {}
    guard_triggered = bool(guard.get("guard_triggered"))
    guard_branch = "late_interaction" if "late_interaction" in execution.retrievers_executed else "dense_only"
    crowding = crowding_classification(orders[ORDER_CURRENT])
    lane = candidate_source_type(best_row) if best_row else "none"
    transitions = {
        "retriever_to_reranker": transition_label(ranks[ORDER_RETRIEVER], ranks[ORDER_RERANKER]),
        "reranker_to_fusion": transition_label(ranks[ORDER_RERANKER], ranks[ORDER_FUSION]),
        "fusion_to_guarded_final": transition_label(ranks[ORDER_FUSION], ranks[ORDER_CURRENT]),
    }
    primary, secondary = classify_primary_failure(ranks, evidence_hits, guard_triggered, guard_branch, crowding, lane, bool(task0132_row.get("graph_sensitive")))
    first_loss = first_ranking_loss_stage(ranks)
    return {
        "schema_version": "opk-rag.task0133.ranking-path-trace.v1",
        "task_id": TASK_ID,
        "evaluation_unit_id": unit,
        "sample_id": task0132_row.get("sample_id"),
        "document_id": task0132_row.get("document_id"),
        "query": task0132_row.get("query"),
        "candidate_relevant_available": any(row.get("relevant_label") for row in orders[ORDER_CURRENT]),
        "best_relevant_rank_retriever": ranks[ORDER_RETRIEVER],
        "best_relevant_rank_reranker": ranks[ORDER_RERANKER],
        "best_relevant_rank_fusion": ranks[ORDER_FUSION],
        "best_relevant_rank_guarded_final": ranks[ORDER_CURRENT],
        "best_relevant_rank_vector_baseline": ranks[ORDER_VECTOR],
        "retriever_to_reranker_transition": transitions["retriever_to_reranker"],
        "reranker_to_fusion_transition": transitions["reranker_to_fusion"],
        "fusion_to_guarded_final_transition": transitions["fusion_to_guarded_final"],
        "current_evidence_access_success": evidence_hits[ORDER_CURRENT],
        "counterfactual_retriever_success": evidence_hits[ORDER_RETRIEVER],
        "counterfactual_vector_success": evidence_hits[ORDER_VECTOR],
        "counterfactual_reranker_success": evidence_hits[ORDER_RERANKER],
        "counterfactual_fusion_success": evidence_hits[ORDER_FUSION],
        "retriever_native_counterfactual_relation": counterfactual_relation(ranks[ORDER_RETRIEVER], ranks[ORDER_CURRENT]),
        "vector_baseline_counterfactual_relation": counterfactual_relation(ranks[ORDER_VECTOR], ranks[ORDER_CURRENT]),
        "reranker_direct_counterfactual_relation": counterfactual_relation(ranks[ORDER_RERANKER], ranks[ORDER_CURRENT]),
        "fusion_attribution": fusion_attribution(ranks),
        "guard_triggered": guard_triggered,
        "guard_branch": guard_branch,
        "guard_attribution": guard_attribution(guard_triggered, guard_branch, evidence_hits),
        "candidate_crowding": crowding,
        "best_relevant_lane": lane,
        "ranking_failure_primary_class": primary,
        "ranking_failure_secondary_classes": secondary,
        "first_ranking_loss_stage": first_loss,
        "root_cause_confidence": confidence(primary, ranks, evidence_hits),
        "recoverable_by_existing_signals": any(evidence_hits[name] for name in (ORDER_RETRIEVER, ORDER_VECTOR, ORDER_RERANKER, ORDER_FUSION)),
        "requires_new_ranking_signal": not any(evidence_hits[name] for name in (ORDER_RETRIEVER, ORDER_VECTOR, ORDER_RERANKER, ORDER_FUSION)),
        "missing_capability_class": missing_capability_class(task0132_row, ranks) if not any(evidence_hits[name] for name in (ORDER_RETRIEVER, ORDER_VECTOR, ORDER_RERANKER, ORDER_FUSION)) else None,
        "graph_sensitive": bool(task0132_row.get("graph_sensitive")),
        "guard_features": guard.get("guard_features") or {},
        "score_features": score_features(orders[ORDER_CURRENT], orders[ORDER_RERANKER], execution.dense_candidates),
        "candidate_ids_current_top5": [row["canonical_chunk_id"] for row in orders[ORDER_CURRENT][:EVIDENCE_CUTOFF]],
        "relevant_candidate_ids": [row["canonical_chunk_id"] for row in orders[ORDER_CURRENT] if row.get("relevant_label")],
        "best_relevant_candidate_id": best_row.get("canonical_chunk_id") if best_row else None,
    }


def evidence_access_success(rows: list[dict[str, Any]], unit: str = "unit") -> bool:
    evidence, _traces = evidence_composition.compose_evidence(rows, evaluation_unit_id=unit)
    return any(row.get("relevant_label") for row in evidence[:EVIDENCE_CUTOFF])


def best_relevant_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("relevant_label")), None)


def transition_label(before: int | None, after: int | None) -> str:
    if before is None and after is None:
        return "unchanged"
    if before is None:
        return "improved"
    if after is None:
        return "degraded"
    if after < before:
        return "improved"
    if after > before:
        return "degraded"
    return "unchanged"


def counterfactual_relation(counterfactual_rank: int | None, current_rank: int | None) -> str:
    return transition_label(current_rank, counterfactual_rank).replace("improved", "better").replace("degraded", "worse")


def fusion_attribution(ranks: dict[str, int | None]) -> str:
    label = transition_label(ranks[ORDER_RERANKER], ranks[ORDER_FUSION])
    return {"improved": "fusion_helped", "degraded": "fusion_hurt", "unchanged": "fusion_neutral"}[label]


def guard_attribution(guard_triggered: bool, guard_branch: str, evidence_hits: dict[str, bool]) -> str:
    if evidence_hits[ORDER_CURRENT]:
        return "guard_recovered_case" if guard_triggered else "guard_correctly_noop"
    if not guard_triggered and (evidence_hits[ORDER_RETRIEVER] or evidence_hits[ORDER_VECTOR] or evidence_hits[ORDER_RERANKER] or evidence_hits[ORDER_FUSION]):
        return "guard_should_have_triggered_but_did_not"
    if guard_triggered and guard_branch not in {"late_interaction", "dense_only"}:
        return "guard_triggered_but_wrong_branch_selected"
    if guard_triggered:
        return "guard_triggered_but_insufficient"
    return "guard_irrelevant_to_failure"


def crowding_classification(rows: list[dict[str, Any]]) -> str:
    best = best_relevant_row(rows)
    if best is None:
        return "none"
    before = rows[: max((first_relevant_rank(rows) or 1) - 1, 0)]
    if not before:
        return "none"
    if sum(row.get("canonical_chunk_id") == best.get("canonical_chunk_id") for row in before) > 0:
        return "cross_lane_duplicate_crowding"
    same_doc = sum(bool(row.get("document_id")) and row.get("document_id") == best.get("document_id") for row in before)
    if same_doc >= min(3, len(before)):
        return "same_document_candidate_crowding"
    if any(row.get("section_id") == best.get("section_id") and row.get("section_id") for row in before):
        return "near_duplicate_crowding"
    return "none"


def classify_primary_failure(
    ranks: dict[str, int | None],
    evidence_hits: dict[str, bool],
    guard_triggered: bool,
    guard_branch: str,
    crowding: str,
    lane: str,
    graph_sensitive: bool,
) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if crowding != "none":
        secondary.append("candidate_crowding")
    if ranks[ORDER_CURRENT] is not None and EVIDENCE_CUTOFF < ranks[ORDER_CURRENT] <= 10:
        secondary.append("near_evidence_cutoff")
    if lane in {SOURCE_DENSE_ONLY, SOURCE_LATE_ONLY, SOURCE_OVERLAP} and transition_label(ranks[ORDER_RETRIEVER], ranks[ORDER_CURRENT]) == "degraded":
        secondary.append("multi_lane_ordering_conflict")
    if guard_attribution(guard_triggered, guard_branch, evidence_hits) == "guard_should_have_triggered_but_did_not":
        return "guard_missed_recovery", sorted(set(secondary))
    if guard_attribution(guard_triggered, guard_branch, evidence_hits) == "guard_triggered_but_wrong_branch_selected":
        return "guard_wrong_branch", sorted(set(secondary))
    if ranks[ORDER_RETRIEVER] is None or ranks[ORDER_RETRIEVER] > 10:
        return "retriever_order_already_weak", sorted(set(secondary))
    if ranks[ORDER_RETRIEVER] <= EVIDENCE_CUTOFF and (ranks[ORDER_RERANKER] or math.inf) > EVIDENCE_CUTOFF:
        return "reranker_false_demotion", sorted(set(secondary))
    if ranks[ORDER_RETRIEVER] > EVIDENCE_CUTOFF and transition_label(ranks[ORDER_RETRIEVER], ranks[ORDER_RERANKER]) == "improved" and (ranks[ORDER_RERANKER] or math.inf) > EVIDENCE_CUTOFF:
        return "reranker_insufficient_promotion", sorted(set(secondary))
    if ranks[ORDER_RERANKER] is not None and ranks[ORDER_RERANKER] <= EVIDENCE_CUTOFF and (ranks[ORDER_FUSION] or math.inf) > EVIDENCE_CUTOFF:
        return "fusion_displacement", sorted(set(secondary))
    if crowding != "none":
        return "candidate_crowding", sorted(set(secondary))
    if ranks[ORDER_CURRENT] is not None and EVIDENCE_CUTOFF < ranks[ORDER_CURRENT] <= 10:
        return "near_evidence_cutoff", sorted(set(secondary))
    if graph_sensitive:
        return "new_semantic_ranking_signal_required", sorted(set(secondary))
    return "new_semantic_ranking_signal_required" if not any(evidence_hits.values()) else "score_calibration_conflict", sorted(set(secondary))


def first_ranking_loss_stage(ranks: dict[str, int | None]) -> str:
    if ranks[ORDER_RETRIEVER] is None or ranks[ORDER_RETRIEVER] > EVIDENCE_CUTOFF:
        return "retriever_native_order"
    if ranks[ORDER_RERANKER] is None or ranks[ORDER_RERANKER] > EVIDENCE_CUTOFF:
        return "reranker"
    if ranks[ORDER_FUSION] is None or ranks[ORDER_FUSION] > EVIDENCE_CUTOFF:
        return "rank_fusion"
    if ranks[ORDER_CURRENT] is None or ranks[ORDER_CURRENT] > EVIDENCE_CUTOFF:
        return "guard"
    return "final_order"


def confidence(primary: str, ranks: dict[str, int | None], evidence_hits: dict[str, bool]) -> str:
    if primary in {"reranker_false_demotion", "reranker_insufficient_promotion", "fusion_displacement", "near_evidence_cutoff", "retriever_order_already_weak"}:
        return "high"
    if any(evidence_hits.values()) or all(rank is not None for rank in ranks.values()):
        return "medium"
    return "low"


def missing_capability_class(task0132_row: dict[str, Any], ranks: dict[str, int | None]) -> str:
    if task0132_row.get("graph_sensitive"):
        return "graph_sensitive_relation_ranking"
    if ranks[ORDER_RETRIEVER] is None:
        return "representation_limit"
    if ranks[ORDER_RETRIEVER] and ranks[ORDER_RETRIEVER] > 10:
        return "query_candidate_semantic_mismatch"
    return "semantic_cross_encoder_limit"


def score_features(current: list[dict[str, Any]], reranker: list[dict[str, Any]], dense_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dense_scores = [_as_float(row.get("retrieval_score")) for row in dense_rows[:20]]
    reranker_scores = [_as_float(row.get("reranker_score")) for row in reranker if row.get("reranker_score") is not None]
    current_scores = [_as_float(row.get("fusion_score")) for row in current if row.get("fusion_score") is not None]
    scores = [score for score in dense_scores if score is not None]
    return {
        "normalized_score_entropy": normalized_entropy(scores),
        "score_margin": (scores[0] - scores[1]) if len(scores) > 1 else None,
        "top_score": scores[0] if scores else None,
        "retriever_score_spread": spread(scores),
        "reranker_score_spread": spread([score for score in reranker_scores if score is not None]),
        "fusion_score_spread": spread([score for score in current_scores if score is not None]),
    }


def normalized_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    total = sum(max(score, 0.0) for score in scores)
    if total <= 0:
        return 0.0
    probabilities = [max(score, 0.0) / total for score in scores]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0


def spread(scores: list[float]) -> float | None:
    return max(scores) - min(scores) if scores else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def task0132_by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["evaluation_unit_id"]: row for row in rows}


def successful_control_units(rows: list[dict[str, Any]], limit: int = 83) -> list[str]:
    controls = [row["evaluation_unit_id"] for row in rows if row.get("final_success") and row.get("candidate_relevant_available") and row.get("relevant_evidence_selected")]
    return sorted(controls)[:limit]


def control_feature_row(unit: str, orders: dict[str, list[dict[str, Any]]], execution: Any, task0132_row: dict[str, Any]) -> dict[str, Any]:
    ranks = {name: first_relevant_rank(rows) for name, rows in orders.items()}
    return {
        "evaluation_unit_id": unit,
        "best_relevant_rank_guarded_final": ranks[ORDER_CURRENT],
        "best_relevant_rank_retriever": ranks[ORDER_RETRIEVER],
        "current_evidence_access_success": evidence_access_success(orders[ORDER_CURRENT], unit),
        "guard_triggered": bool((execution.guard_decision or {}).get("guard_triggered")),
        "guard_branch": "late_interaction" if "late_interaction" in execution.retrievers_executed else "dense_only",
        "candidate_crowding": crowding_classification(orders[ORDER_CURRENT]),
        "graph_sensitive": bool(task0132_row.get("graph_sensitive")),
        "score_features": score_features(orders[ORDER_CURRENT], orders[ORDER_RERANKER], execution.dense_candidates),
    }


def feature_matrix_row(row: dict[str, Any]) -> dict[str, Any]:
    features = row["score_features"]
    return {
        "schema_version": "opk-rag.task0133.diagnostic-feature.v1",
        "sample_id": row["sample_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "query_family": (row["evaluation_unit_id"].split("::", 1)[0] if "::" in row["evaluation_unit_id"] else row["evaluation_unit_id"].split(":", 1)[0]),
        "best_relevant_retriever_rank": row["best_relevant_rank_retriever"],
        "best_relevant_reranker_rank": row["best_relevant_rank_reranker"],
        "best_relevant_fusion_rank": row["best_relevant_rank_fusion"],
        "best_relevant_final_rank": row["best_relevant_rank_guarded_final"],
        "score_margin": features["score_margin"],
        "score_entropy": features["normalized_score_entropy"],
        "top_score": features["top_score"],
        "lane": row["best_relevant_lane"],
        "document_id": row["document_id"],
        "guard_triggered": row["guard_triggered"],
        "guard_branch": row["guard_branch"],
        "candidate_crowding": row["candidate_crowding"],
        "graph_sensitive": row["graph_sensitive"],
        "current_downstream_success": row["current_evidence_access_success"],
        "counterfactual_retriever_success": row["counterfactual_retriever_success"],
        "counterfactual_vector_success": row["counterfactual_vector_success"],
        "counterfactual_reranker_success": row["counterfactual_reranker_success"],
    }


def rank_transition_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.rank-transition-matrix.v1",
        "unit_count": len(rows),
        "improved_by_reranker_count": count(rows, "retriever_to_reranker_transition", "improved"),
        "degraded_by_reranker_count": count(rows, "retriever_to_reranker_transition", "degraded"),
        "unchanged_by_reranker_count": count(rows, "retriever_to_reranker_transition", "unchanged"),
        "improved_by_fusion_count": count(rows, "reranker_to_fusion_transition", "improved"),
        "degraded_by_fusion_count": count(rows, "reranker_to_fusion_transition", "degraded"),
        "unchanged_by_fusion_count": count(rows, "reranker_to_fusion_transition", "unchanged"),
        "improved_by_guard_count": count(rows, "fusion_to_guarded_final_transition", "improved"),
        "degraded_by_guard_count": count(rows, "fusion_to_guarded_final_transition", "degraded"),
        "unchanged_by_guard_count": count(rows, "fusion_to_guarded_final_transition", "unchanged"),
        "first_ranking_loss_stage_distribution": dict(Counter(row["first_ranking_loss_stage"] for row in rows)),
        "dominant_first_ranking_loss_stage": dominant(Counter(row["first_ranking_loss_stage"] for row in rows)),
    }


def counterfactual_recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retriever = Counter(row["retriever_native_counterfactual_relation"] for row in rows)
    vector = Counter(row["vector_baseline_counterfactual_relation"] for row in rows)
    reranker = Counter(row["reranker_direct_counterfactual_relation"] for row in rows)
    recoverable = [row for row in rows if row["recoverable_by_existing_signals"]]
    return {
        "schema_version": "opk-rag.task0133.counterfactual-recovery.v1",
        "recovery_criterion": "same targeted_budgeted_composition with 5 evidence slots; generation replay unavailable, so downstream_recovery_count is evidence_access_recovery_count proxy",
        "rank_recovery_count": sum(any((row[f"best_relevant_rank_{name}"] or math.inf) < (row["best_relevant_rank_guarded_final"] or math.inf) for name in ("retriever", "reranker", "fusion", "vector_baseline")) for row in rows),
        "evidence_access_recovery_count": len(recoverable),
        "downstream_recovery_count": len(recoverable),
        "downstream_recovery_observable": False,
        "retriever_native_better_count": retriever["better"],
        "retriever_native_equal_count": retriever["unchanged"],
        "retriever_native_worse_count": retriever["worse"],
        "retriever_native_counterfactual_recovery_count": sum(row["counterfactual_retriever_success"] for row in rows),
        "vector_baseline_recovery_count": sum(row["counterfactual_vector_success"] for row in rows),
        "vector_baseline_regression_count": sum((row["counterfactual_vector_success"] is False) and (row["counterfactual_retriever_success"] or row["counterfactual_reranker_success"] or row["counterfactual_fusion_success"]) for row in rows),
        "vector_baseline_unchanged_count": vector["unchanged"],
        "ranking_failures_where_vector_is_best_count": sum(is_best(row, "best_relevant_rank_vector_baseline") for row in rows),
        "reranker_direct_recovery_count": sum(row["counterfactual_reranker_success"] for row in rows),
        "reranker_direct_regression_count": sum((row["counterfactual_reranker_success"] is False) and row["counterfactual_retriever_success"] for row in rows),
        "reranker_direct_unchanged_count": reranker["unchanged"],
        "existing_signal_oracle_recovery_count": len(recoverable),
        "existing_signal_oracle_recovery_rate": ratio(len(recoverable), len(rows)),
        "recoverable_by_existing_signals_count": len(recoverable),
        "requires_new_ranking_signal_count": sum(row["requires_new_ranking_signal"] for row in rows),
        "indeterminate_ranking_failure_count": sum(row["root_cause_confidence"] == "low" for row in rows),
    }


def is_best(row: dict[str, Any], key: str) -> bool:
    ranks = [
        row["best_relevant_rank_guarded_final"],
        row["best_relevant_rank_retriever"],
        row["best_relevant_rank_reranker"],
        row["best_relevant_rank_fusion"],
        row["best_relevant_rank_vector_baseline"],
    ]
    present = [rank for rank in ranks if rank is not None]
    return bool(present) and row[key] == min(present)


def ranking_quality_metrics(orders_by_unit: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.ranking-quality-metrics.v1",
        **{name: quality_metrics({unit: orders[name] for unit, orders in orders_by_unit.items()}) for name in ORDERINGS},
    }


def quality_metrics(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [first_relevant_rank(rows) for rows in rows_by_unit.values()]
    present = [rank for rank in ranks if rank is not None]
    return {
        "unit_count": len(ranks),
        "mrr": ratio(sum(1 / rank for rank in present), len(ranks)),
        "recall_at_1": ratio(sum(rank is not None and rank <= 1 for rank in ranks), len(ranks)),
        "recall_at_3": ratio(sum(rank is not None and rank <= 3 for rank in ranks), len(ranks)),
        "recall_at_5": ratio(sum(rank is not None and rank <= 5 for rank in ranks), len(ranks)),
        "recall_at_10": ratio(sum(rank is not None and rank <= 10 for rank in ranks), len(ranks)),
        "recall_at_20": ratio(sum(rank is not None and rank <= 20 for rank in ranks), len(ranks)),
    }


def failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = Counter(row["ranking_failure_primary_class"] for row in rows)
    final_ranks = [row["best_relevant_rank_guarded_final"] for row in rows]
    missing = Counter(row["missing_capability_class"] for row in rows if row["missing_capability_class"])
    return {
        "schema_version": "opk-rag.task0133.failure-taxonomy.v1",
        "validated_ranking_failure_count": len(rows),
        "ranking_failure_count_by_primary_class": {klass: primary.get(klass, 0) for klass in PRIMARY_CLASSES},
        "sum_primary_class_counts": sum(primary.values()),
        "reranker_false_demotion_count": primary["reranker_false_demotion"],
        "reranker_insufficient_promotion_count": primary["reranker_insufficient_promotion"],
        "fusion_displacement_failure_count": primary["fusion_displacement"],
        "guard_missed_recovery_count": primary["guard_missed_recovery"],
        "guard_wrong_branch_count": primary["guard_wrong_branch"],
        "guard_insufficient_count": count(rows, "guard_attribution", "guard_triggered_but_insufficient"),
        "candidate_crowding_failure_count": sum(row["candidate_crowding"] != "none" for row in rows),
        "multi_lane_conflict_count": sum("multi_lane_ordering_conflict" in row["ranking_failure_secondary_classes"] or row["ranking_failure_primary_class"] == "multi_lane_ordering_conflict" for row in rows),
        "rank_1": rank_bucket(final_ranks, 1, 1),
        "rank_2": rank_bucket(final_ranks, 2, 2),
        "rank_3_to_5": rank_bucket(final_ranks, 3, 5),
        "rank_6_to_10": rank_bucket(final_ranks, 6, 10),
        "rank_11_to_20": rank_bucket(final_ranks, 11, 20),
        "rank_gt_20": sum(rank is None or rank > 20 for rank in final_ranks),
        "relevant_at_1": sum(rank is not None and rank <= 1 for rank in final_ranks),
        "relevant_at_3": sum(rank is not None and rank <= 3 for rank in final_ranks),
        "relevant_at_5": sum(rank is not None and rank <= 5 for rank in final_ranks),
        "relevant_at_10": sum(rank is not None and rank <= 10 for rank in final_ranks),
        "relevant_at_20": sum(rank is not None and rank <= 20 for rank in final_ranks),
        "near_evidence_cutoff_failure_count": sum(rank is not None and 6 <= rank <= 10 for rank in final_ranks),
        "requires_new_ranking_signal_count": sum(row["requires_new_ranking_signal"] for row in rows),
        "new_signal_requirement_distribution": dict(missing),
        "high_confidence_ranking_failure_count": count(rows, "root_cause_confidence", "high"),
        "medium_confidence_ranking_failure_count": count(rows, "root_cause_confidence", "medium"),
        "low_confidence_ranking_failure_count": count(rows, "root_cause_confidence", "low"),
    }


def signal_analysis(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.signal-analysis.v1",
        "ranking_failure_cohort": aggregate_score_features(rows),
        "successful_control_cohort": aggregate_score_features(controls),
        "previous_task0120_signal_revisited": True,
        "normalized_score_entropy_still_useful": median_feature(rows, "normalized_score_entropy") != median_feature(controls, "normalized_score_entropy"),
        "signals_available_for_future_routing": ["normalized_score_entropy", "score_margin", "top_score", "retriever_score_spread", "reranker_score_spread"],
    }


def aggregate_score_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: summarize([((row.get("score_features") or {}).get(name)) for row in rows]) for name in ("normalized_score_entropy", "score_margin", "top_score", "retriever_score_spread", "reranker_score_spread")}


def median_feature(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [row["score_features"].get(name) for row in rows if isinstance(row.get("score_features", {}).get(name), (int, float))]
    return statistics.median(values) if values else None


def summarize(values: list[Any]) -> dict[str, Any]:
    nums = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {"count": len(nums), "mean": statistics.mean(nums) if nums else None, "median": statistics.median(nums) if nums else None}


def graph_sensitive_ranking_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graph = [row for row in rows if row["graph_sensitive"]]
    return {
        "schema_version": "opk-rag.task0133.graph-sensitive-ranking-audit.v1",
        "graph_sensitive_ranking_failure_count": len(graph),
        "graph_sensitive_existing_signal_recoverable_count": sum(row["recoverable_by_existing_signals"] for row in graph),
        "graph_sensitive_requires_new_signal_count": sum(row["requires_new_ranking_signal"] for row in graph),
        "graph_sensitive_primary_class_distribution": dict(Counter(row["ranking_failure_primary_class"] for row in graph)),
        "graph_sensitive_patterns": ["multi-hop relation", "entity dependency", "cross-section relation", "path-like evidence"],
    }


def control_cohort_comparison(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.control-cohort-comparison.v1",
        "ranking_failure_count": len(rows),
        "successful_control_count": len(controls),
        "failure_final_rank_median": median_rank([row["best_relevant_rank_guarded_final"] for row in rows]),
        "control_final_rank_median": median_rank([row["best_relevant_rank_guarded_final"] for row in controls]),
        "failure_guard_trigger_rate": ratio(sum(row["guard_triggered"] for row in rows), len(rows)),
        "control_guard_trigger_rate": ratio(sum(row["guard_triggered"] for row in controls), len(controls)),
        "failure_crowding_count": sum(row["candidate_crowding"] != "none" for row in rows),
        "control_crowding_count": sum(row["candidate_crowding"] != "none" for row in controls),
    }


def config_payload(authority: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.task0133.config.v1",
        "diagnosis_only": True,
        "runtime_modified": False,
        "default_policy_modified": False,
        "promotion_applied": False,
        "candidate_generation_modified": False,
        "candidate_membership_modified": False,
        "reranker_modified": False,
        "rank_fusion_modified": False,
        "guard_thresholds_modified": False,
        "evidence_policy": evidence_composition.targeted_budgeted_policy().to_json(),
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0132_summary_sha256": authority["task0132_summary_sha256"],
        "gold_signal_used_for_runtime": False,
    }
    return payload | {"runtime_config_digest": digest_json(payload)}


def digests_payload(traces: list[dict[str, Any]], feature_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.digests.v1",
        "ranking_path_trace_digest": digest_json(traces),
        "diagnostic_feature_matrix_digest": digest_json(feature_rows),
        "runtime_config_digest": config["runtime_config_digest"],
        "deterministic_output": True,
    }


def summary_payload(
    authority: dict[str, Any],
    benchmark: dict[str, Any],
    traces: list[dict[str, Any]],
    transition: dict[str, Any],
    counterfactual: dict[str, Any],
    taxonomy: dict[str, Any],
    graph: dict[str, Any],
    signal: dict[str, Any],
    config: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    validated = len([row for row in traces if row["candidate_relevant_available"]])
    primary = taxonomy["ranking_failure_count_by_primary_class"]
    recommended = recommendation(counterfactual, taxonomy, graph)
    return {
        "schema_version": "opk-rag.task0133.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0132_verifier_valid": authority["task0132_verifier_valid"],
        "task0132_ranking_failure_count": authority["task0132_ranking_failure_count"],
        "validated_ranking_failure_count": validated,
        "misclassified_candidate_retrieval_count": len(traces) - validated,
        "runtime_modified": False,
        "default_policy_modified": False,
        "promotion_applied": False,
        "gold_signal_used_for_runtime": False,
        "candidate_membership_change_count": 0,
        "evidence_budget_limit": EVIDENCE_CUTOFF,
        "dominant_first_ranking_loss_stage": transition["dominant_first_ranking_loss_stage"],
        "dominant_ranking_failure_primary_class": max(primary, key=lambda key: primary[key]) if primary else None,
        "recoverable_by_existing_signals_count": counterfactual["recoverable_by_existing_signals_count"],
        "requires_new_ranking_signal_count": counterfactual["requires_new_ranking_signal_count"],
        "existing_signal_oracle_recovery_count": counterfactual["existing_signal_oracle_recovery_count"],
        "existing_signal_oracle_recovery_rate": counterfactual["existing_signal_oracle_recovery_rate"],
        "retriever_native_counterfactual_recovery_count": counterfactual["retriever_native_counterfactual_recovery_count"],
        "vector_baseline_recovery_count": counterfactual["vector_baseline_recovery_count"],
        "reranker_direct_recovery_count": counterfactual["reranker_direct_recovery_count"],
        "fusion_displacement_failure_count": taxonomy["fusion_displacement_failure_count"],
        "reranker_false_demotion_count": taxonomy["reranker_false_demotion_count"],
        "reranker_insufficient_promotion_count": taxonomy["reranker_insufficient_promotion_count"],
        "candidate_crowding_failure_count": taxonomy["candidate_crowding_failure_count"],
        "near_evidence_cutoff_failure_count": taxonomy["near_evidence_cutoff_failure_count"],
        "graph_sensitive_ranking_failure_count": graph["graph_sensitive_ranking_failure_count"],
        "graph_sensitive_existing_signal_recoverable_count": graph["graph_sensitive_existing_signal_recoverable_count"],
        "graph_sensitive_requires_new_signal_count": graph["graph_sensitive_requires_new_signal_count"],
        "normalized_score_entropy_still_useful": signal["normalized_score_entropy_still_useful"],
        "recommended_next_task_family": recommended,
        "runtime_config_digest": config["runtime_config_digest"],
        "ranking_path_trace_digest": digests["ranking_path_trace_digest"],
        "task0133_verifier_valid": False,
        "verifier_status": "pending",
    }


def recommendation(counterfactual: dict[str, Any], taxonomy: dict[str, Any], graph: dict[str, Any]) -> str:
    if graph["graph_sensitive_requires_new_signal_count"] >= max(1, counterfactual["requires_new_ranking_signal_count"] // 2):
        return "graph_sensitive_ranking_optimization"
    if counterfactual["recoverable_by_existing_signals_count"] >= counterfactual["requires_new_ranking_signal_count"]:
        return "existing_signal_ranking_policy_diagnosis"
    if taxonomy["reranker_false_demotion_count"] > taxonomy["reranker_insufficient_promotion_count"]:
        return "selective_reranker_trust_diagnosis"
    return "new_ranking_signal_exploration"


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0133.residual-ranking-failure-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "diagnosis_only": True,
        "runtime_behavior_modification_allowed": False,
        "default_policy_modification_allowed": False,
        "promotion_allowed": False,
        "task0132_ranking_failure_count": authority["task0132_ranking_failure_count"],
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "current_runtime_policy_digest": authority["task0132_runtime_policy_digest"],
        "task0133_runtime_config_digest": config["runtime_config_digest"],
        "gold_signal_runtime_use_allowed": False,
    }


def verify_task0133_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        taxonomy = read_json(output_dir / "failure_taxonomy.json")
        counter = read_json(output_dir / "counterfactual_recovery.json")
        traces = read_jsonl(output_dir / "ranking_path_trace.jsonl")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("task0132_ranking_failure_count") != 83 or summary.get("validated_ranking_failure_count") != len(traces):
            issues.append({"code": "ranking_cohort_mismatch"})
        if summary.get("runtime_modified") is not False or summary.get("promotion_applied") is not False:
            issues.append({"code": "runtime_mutation_detected"})
        if summary.get("gold_signal_used_for_runtime") is not False:
            issues.append({"code": "gold_signal_runtime_use_detected"})
        if taxonomy.get("sum_primary_class_counts") != taxonomy.get("validated_ranking_failure_count"):
            issues.append({"code": "taxonomy_accounting_mismatch"})
        if counter.get("existing_signal_oracle_recovery_count") != summary.get("existing_signal_oracle_recovery_count"):
            issues.append({"code": "counterfactual_summary_mismatch"})
        if any(not row.get("candidate_relevant_available") for row in traces):
            issues.append({"code": "candidate_retrieval_misclassification_detected"})
    result = {
        "schema_version": "opk-rag.task0133.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0133_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(summary: dict[str, Any], transition: dict[str, Any], counterfactual: dict[str, Any], taxonomy: dict[str, Any], graph: dict[str, Any], signal: dict[str, Any]) -> str:
    return f"""# TASK-0133 Residual Ranking Failure Diagnosis Report

## Answer

- Validated ranking cohort: `{summary['validated_ranking_failure_count']}/{summary['task0132_ranking_failure_count']}`; candidate-retrieval reclassification count `{summary['misclassified_candidate_retrieval_count']}`.
- Dominant primary class: `{summary['dominant_ranking_failure_primary_class']}`; dominant first ranking loss stage `{summary['dominant_first_ranking_loss_stage']}`.
- Existing-signal oracle recovery: `{counterfactual['existing_signal_oracle_recovery_count']}` (`{counterfactual['existing_signal_oracle_recovery_rate']:.4f}`).
- Requires new ranking signal: `{counterfactual['requires_new_ranking_signal_count']}`.
- Retriever-native recovery `{counterfactual['retriever_native_counterfactual_recovery_count']}`; vector-baseline recovery `{counterfactual['vector_baseline_recovery_count']}`; reranker-direct recovery `{counterfactual['reranker_direct_recovery_count']}`.
- Reranker false demotion `{taxonomy['reranker_false_demotion_count']}`; insufficient promotion `{taxonomy['reranker_insufficient_promotion_count']}`; fusion displacement `{taxonomy['fusion_displacement_failure_count']}`.
- Near evidence cutoff `[6,10]`: `{taxonomy['near_evidence_cutoff_failure_count']}`.
- Graph-sensitive ranking failures `{graph['graph_sensitive_ranking_failure_count']}`; graph-sensitive recoverable `{graph['graph_sensitive_existing_signal_recoverable_count']}`; graph-sensitive requiring new signal `{graph['graph_sensitive_requires_new_signal_count']}`.

## Transition Matrix

- Reranker improved/degraded/unchanged: `{transition['improved_by_reranker_count']}` / `{transition['degraded_by_reranker_count']}` / `{transition['unchanged_by_reranker_count']}`.
- Fusion improved/degraded/unchanged: `{transition['improved_by_fusion_count']}` / `{transition['degraded_by_fusion_count']}` / `{transition['unchanged_by_fusion_count']}`.
- Guard improved/degraded/unchanged: `{transition['improved_by_guard_count']}` / `{transition['degraded_by_guard_count']}` / `{transition['unchanged_by_guard_count']}`.

## Freeze

Runtime modified: `{summary['runtime_modified']}`. Promotion applied: `{summary['promotion_applied']}`. Gold signal used for runtime: `{summary['gold_signal_used_for_runtime']}`. Evidence budget: `{summary['evidence_budget_limit']}`.

## Downstream Replay Boundary

Generation replay is not observable in the frozen TASK-0133 artifacts, so downstream recovery is reported as the same `targeted_budgeted_composition` 5-slot evidence-access proxy. No runtime policy is changed by this diagnostic.

## Recommendation

Primary next task family: `{summary['recommended_next_task_family']}`. TASK-0133 is diagnostic only; promotion or mitigation belongs to TASK-0134 or later.
"""


def count(rows: list[dict[str, Any]], key: str, value: Any) -> int:
    return sum(row.get(key) == value for row in rows)


def rank_bucket(ranks: list[int | None], low: int, high: int) -> int:
    return sum(rank is not None and low <= rank <= high for rank in ranks)


def median_rank(ranks: list[int | None]) -> float | None:
    present = [rank for rank in ranks if rank is not None]
    return statistics.median(present) if present else None


def dominant(counter: Counter[str]) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
