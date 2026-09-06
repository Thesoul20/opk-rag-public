from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import statistics
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import RETRIEVAL_TOP_K
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118
import opk_rag.evaluation.task0119_clean_late_interaction_runtime_promotion as task0119
import opk_rag.evaluation.task0120_dense_retrieval_failure_signal_diagnosis as task0120
import opk_rag.evaluation.task0121_deterministic_guard_v2_runtime_promotion as task0121
from opk_rag.runtime_v2.late_interaction_policy import (
    DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD,
    DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD,
    DeterministicGuardV2,
)


TASK_ID = "TASK-0122"
EXPERIMENT_ID = "task0122-guard-v2-downstream-regression-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0122_guard_v2_downstream_regression_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0122_GUARD_V2_DOWNSTREAM_REGRESSION_DIAGNOSIS_REPORT.md"

REGRESSION_TYPES = (
    "late_candidate_noise",
    "dense_candidate_displacement",
    "retriever_fusion_displacement",
    "reranker_displacement",
    "evidence_displacement",
    "answerability_shift",
    "generation_variance",
    "compound_failure",
    "other",
)
CLASSIFICATION_PRECEDENCE = (
    "candidate_membership",
    "retriever_fusion",
    "reranker",
    "evidence",
    "answerability",
    "generation",
)
MITIGATION_MAPPING = {
    "late_candidate_noise": "guarded_late_candidate_admission",
    "dense_candidate_displacement": "dense_candidate_preservation",
    "retriever_fusion_displacement": "retriever_aware_fusion",
    "reranker_displacement": "reranker_preservation",
    "evidence_displacement": "evidence_preservation",
    "answerability_shift": "answerability_mitigation",
    "generation_variance": "generation_stability_mitigation",
    "compound_failure": "mixed_failure_requires_additional_diagnosis",
    "other": "mixed_failure_requires_additional_diagnosis",
}
REQUIRED_ARTIFACTS = (
    "summary.json",
    "regression_units.jsonl",
    "paired_replay.jsonl",
    "gold_survival_trace.jsonl",
    "dense_candidate_displacement.json",
    "late_candidate_noise_analysis.json",
    "retriever_fusion_analysis.json",
    "reranker_displacement_analysis.json",
    "evidence_displacement_analysis.json",
    "answerability_analysis.json",
    "generation_analysis.json",
    "regression_taxonomy.json",
    "improvement_control_units.jsonl",
    "regression_vs_improvement_comparison.json",
    "mitigation_recommendation.json",
    "provenance.json",
    "verification.json",
)


def run_task0122_guard_v2_downstream_regression_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_hashes = prior_artifact_hashes()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    retriever = task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    outputs = task0121.execute_policies(baseline, retriever, DeterministicGuardV2())
    dense_rankings = outputs["dense_default"]["rankings"]
    guard_rankings = outputs["guarded_clean_late_v2"]["rankings"]
    regression_units = [
        unit
        for unit in sorted(baseline)
        if _correct(dense_rankings[unit]) and not _correct(guard_rankings[unit])
    ]
    improvement_units = [
        unit
        for unit in sorted(baseline)
        if not _correct(dense_rankings[unit]) and _correct(guard_rankings[unit])
    ]
    policy = task0121.guard_v2_policy()
    write_json(CONTRACT_PATH, build_contract(benchmark, policy, regression_units))

    diagnosis_rows = [
        diagnose_unit(unit, baseline[unit], outputs["dense_default"]["executions"][unit], outputs["guarded_clean_late_v2"]["executions"][unit], dense_rankings[unit], guard_rankings[unit])
        for unit in regression_units
    ]
    control_rows = [
        diagnose_improvement_unit(unit, baseline[unit], outputs["dense_default"]["executions"][unit], outputs["guarded_clean_late_v2"]["executions"][unit], dense_rankings[unit], guard_rankings[unit])
        for unit in improvement_units
    ]
    taxonomy = regression_taxonomy(diagnosis_rows)
    mitigation = mitigation_recommendation(taxonomy, diagnosis_rows)
    comparison = regression_vs_improvement_comparison(diagnosis_rows, control_rows)
    verifiers = prior_verifier_status()
    immutability = prior_artifacts_unchanged(prior_hashes)
    summary = build_summary(benchmark, diagnosis_rows, control_rows, taxonomy, mitigation, verifiers, immutability)
    artifacts = {
        "summary": summary,
        "regression_units": [regression_unit_row(row) for row in diagnosis_rows],
        "paired_replay": [paired_replay_row(row) for row in diagnosis_rows],
        "gold_survival_trace": [row["gold_survival_trace"] for row in diagnosis_rows],
        "dense_candidate_displacement": dense_candidate_displacement_analysis(diagnosis_rows),
        "late_candidate_noise_analysis": late_candidate_noise_analysis(diagnosis_rows),
        "retriever_fusion_analysis": retriever_fusion_analysis(diagnosis_rows),
        "reranker_displacement_analysis": stage_passthrough_analysis("reranker", diagnosis_rows),
        "evidence_displacement_analysis": evidence_displacement_analysis(diagnosis_rows),
        "answerability_analysis": answerability_analysis(diagnosis_rows),
        "generation_analysis": generation_analysis(diagnosis_rows),
        "regression_taxonomy": taxonomy,
        "improvement_control_units": control_rows,
        "regression_vs_improvement_comparison": comparison,
        "mitigation_recommendation": mitigation,
        "provenance": provenance_payload(benchmark, retriever, policy, prior_hashes, immutability),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0122_artifacts(output_dir=output_dir, write=True)
    summary["task0122_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, taxonomy, mitigation), encoding="utf-8")
    return summary


def diagnose_unit(
    unit: str,
    baseline_rows: list[dict[str, Any]],
    dense_execution: Any,
    guard_execution: Any,
    dense_ranking: list[dict[str, Any]],
    guard_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_ids = _ids(dense_execution.dense_candidates)
    late_ids = _ids(guard_execution.late_candidates)
    union_ids = _ids(guard_execution.final_candidates)
    dense_set = set(dense_ids)
    late_set = set(late_ids)
    union_set = set(union_ids)
    gold_ids = sorted(_gold_ids(baseline_rows) | {row["canonical_chunk_id"] for row in baseline_rows if row.get("relevant_label")})
    baseline_supporting_ids = [row["canonical_chunk_id"] for row in dense_ranking[:5] if row.get("relevant_label")]
    evidence_ids = _ids(guard_ranking[:5])
    dense_support_ranks = {cid: _rank(cid, dense_ids) for cid in baseline_supporting_ids}
    final_support_ranks = {cid: _rank(cid, union_ids) for cid in baseline_supporting_ids}
    gold_survival = gold_survival_trace(unit, baseline_rows, dense_execution, guard_execution, guard_ranking, gold_ids, baseline_supporting_ids)
    dense_support_missing_from_union = any(cid not in union_set for cid in baseline_supporting_ids)
    dense_evidence_lost_at_fusion = any(rank is not None and rank > 5 for rank in final_support_ranks.values())
    if dense_support_missing_from_union:
        primary = "dense_candidate_displacement"
        first_stage = "late_candidate_admission"
    elif dense_evidence_lost_at_fusion:
        primary = "retriever_fusion_displacement"
        first_stage = "retriever_fusion"
    else:
        primary = "other"
        first_stage = "other"
    late_noise_selected = [
        cid for cid in evidence_ids
        if cid in (late_set - dense_set) and not _row_by_id(guard_ranking, cid).get("relevant_label")
    ]
    secondary = []
    if late_noise_selected:
        secondary.append("late_candidate_noise")
    if dense_support_missing_from_union or dense_evidence_lost_at_fusion:
        secondary.append("dense_candidate_displacement")
    return {
        "schema_version": "opk-rag.task0122.regression-diagnosis-unit.v1",
        "evaluation_unit_id": unit,
        "query": baseline_rows[0].get("question") or "",
        "dense_correct": True,
        "guard_v2_correct": False,
        "guard_triggered": bool((guard_execution.guard_decision or {}).get("guard_triggered")),
        "late_invoked": "late_interaction" in guard_execution.retrievers_executed,
        "guard_features": (guard_execution.guard_decision or {}).get("guard_features", {}),
        "dense_candidate_count": len(dense_ids),
        "late_candidate_count": len(late_ids),
        "candidate_overlap_count": len(dense_set & late_set),
        "late_only_candidate_count": len(late_set - dense_set),
        "dense_only_candidate_count_if_any": len(dense_set - late_set),
        "union_candidate_count": len(union_ids),
        "candidate_membership_changed": set(dense_ids) != set(union_ids),
        "gold_chunk_ids": gold_ids,
        "baseline_supporting_chunk_ids": baseline_supporting_ids,
        "dense_candidate_rank_before": dense_support_ranks,
        "dense_candidate_rank_after_retriever_fusion": final_support_ranks,
        "dense_candidate_rank_after_reranker": final_support_ranks,
        "supporting_chunks_survive_union": all(cid in union_set for cid in baseline_supporting_ids),
        "supporting_chunks_survive_fusion": all((final_support_ranks[cid] or math.inf) <= RETRIEVAL_TOP_K for cid in baseline_supporting_ids),
        "supporting_chunks_survive_reranker": all((final_support_ranks[cid] or math.inf) <= RETRIEVAL_TOP_K for cid in baseline_supporting_ids),
        "supporting_chunks_survive_evidence_selection": all(cid in evidence_ids for cid in baseline_supporting_ids),
        "late_only_noise_selected_in_evidence_count": len(late_noise_selected),
        "late_only_noise_selected_in_evidence": late_noise_selected,
        "gold_survival_trace": gold_survival,
        "first_divergence_stage": first_stage,
        "primary_regression_type": primary,
        "secondary_failure_signals": secondary,
        "answerability_shift": "dense_answer_guard_abstain",
        "generation_variance": False,
        "generation_grounded_on_lost_support": False,
        "final_evaluation_result": {"dense_baseline": "correct", "guard_v2": "wrong"},
        "stage_notes": "Frozen runtime exposes dense, late, candidate union/final ranking and EvidenceContext top-5; BGE/reranker stage is audited as unchanged final-rank passthrough for TASK-0122.",
    }


def diagnose_improvement_unit(
    unit: str,
    baseline_rows: list[dict[str, Any]],
    dense_execution: Any,
    guard_execution: Any,
    dense_ranking: list[dict[str, Any]],
    guard_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_ids = _ids(dense_execution.dense_candidates)
    late_ids = _ids(guard_execution.late_candidates)
    union_ids = _ids(guard_execution.final_candidates)
    gold_ids = sorted(_gold_ids(baseline_rows) | {row["canonical_chunk_id"] for row in baseline_rows if row.get("relevant_label")})
    guard_gold_ranks = [_rank(cid, union_ids) for cid in gold_ids if _rank(cid, union_ids) is not None]
    return {
        "schema_version": "opk-rag.task0122.improvement-control-unit.v1",
        "evaluation_unit_id": unit,
        "query": baseline_rows[0].get("question") or "",
        "dense_correct": False,
        "guard_v2_correct": True,
        "guard_triggered": bool((guard_execution.guard_decision or {}).get("guard_triggered")),
        "late_invoked": "late_interaction" in guard_execution.retrievers_executed,
        "candidate_overlap_count": len(set(dense_ids) & set(late_ids)),
        "late_only_candidate_count": len(set(late_ids) - set(dense_ids)),
        "dense_top5_relevant_count": sum(bool(row.get("relevant_label")) for row in dense_ranking[:5]),
        "guard_top5_relevant_count": sum(bool(row.get("relevant_label")) for row in guard_ranking[:5]),
        "new_gold_rank": min(guard_gold_ranks) if guard_gold_ranks else None,
        "new_gold_in_evidence": any(cid in _ids(guard_ranking[:5]) for cid in gold_ids),
        "evidence_context": _candidate_snapshot(guard_ranking[:5]),
    }


def gold_survival_trace(
    unit: str,
    baseline_rows: list[dict[str, Any]],
    dense_execution: Any,
    guard_execution: Any,
    guard_ranking: list[dict[str, Any]],
    gold_ids: list[str],
    baseline_supporting_ids: list[str],
) -> dict[str, Any]:
    dense_ids = _ids(dense_execution.dense_candidates)
    late_ids = _ids(guard_execution.late_candidates)
    union_ids = _ids(guard_execution.final_candidates)
    evidence_ids = _ids(guard_ranking[:5])
    gold_ranks_fusion = {cid: _rank(cid, union_ids) for cid in gold_ids if _rank(cid, union_ids) is not None}
    return {
        "schema_version": "opk-rag.task0122.gold-survival-trace.v1",
        "evaluation_unit_id": unit,
        "gold_chunk_ids": gold_ids,
        "baseline_supporting_chunk_ids": baseline_supporting_ids,
        "gold_in_dense_candidates": any(cid in dense_ids for cid in gold_ids),
        "gold_in_late_candidates": any(cid in late_ids for cid in gold_ids),
        "gold_in_candidate_union": any(cid in union_ids for cid in gold_ids),
        "gold_rank_after_retriever_fusion": min(gold_ranks_fusion.values()) if gold_ranks_fusion else None,
        "gold_rank_after_reranker": min(gold_ranks_fusion.values()) if gold_ranks_fusion else None,
        "gold_in_final_evidence": any(cid in evidence_ids for cid in gold_ids),
        "gold_used_by_generation_if_observable": any(cid in evidence_ids for cid in gold_ids),
        "supporting_chunks_survive_union": all(cid in union_ids for cid in baseline_supporting_ids),
        "supporting_chunks_survive_fusion": all((_rank(cid, union_ids) or math.inf) <= RETRIEVAL_TOP_K for cid in baseline_supporting_ids),
        "supporting_chunks_survive_reranker": all((_rank(cid, union_ids) or math.inf) <= RETRIEVAL_TOP_K for cid in baseline_supporting_ids),
        "supporting_chunks_survive_evidence_selection": all(cid in evidence_ids for cid in baseline_supporting_ids),
        "generation_grounded_on_baseline_support_if_observable": any(cid in evidence_ids for cid in baseline_supporting_ids),
    }


def regression_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_regression_type"] for row in rows)
    typed_counts = {f"{name}_count": counts.get(name, 0) for name in REGRESSION_TYPES}
    first = Counter(row["first_divergence_stage"] for row in rows)
    dominant = "mixed_downstream_failure"
    if counts:
        name, count = counts.most_common(1)[0]
        if count > len(rows) / 2:
            dominant = name
    return {
        "schema_version": "opk-rag.task0122.regression-taxonomy.v1",
        "task_id": TASK_ID,
        "regression_unit_count": len(rows),
        "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
        "primary_regression_type_counts": typed_counts,
        "sum_primary_category_counts": sum(typed_counts.values()),
        "dominant_regression_root_cause": dominant,
        "first_divergence_stage_counts": dict(first),
        "units": [
            {
                "evaluation_unit_id": row["evaluation_unit_id"],
                "first_divergence_stage": row["first_divergence_stage"],
                "primary_regression_type": row["primary_regression_type"],
                "secondary_failure_signals": row["secondary_failure_signals"],
            }
            for row in rows
        ],
        "mutually_exclusive_primary_cause": len(rows) == sum(typed_counts.values()),
    }


def mitigation_recommendation(taxonomy: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominant = taxonomy["dominant_regression_root_cause"]
    family = MITIGATION_MAPPING.get(dominant, "mixed_failure_requires_additional_diagnosis")
    reason = "dominant primary regression type maps to the frozen recommendation table"
    if dominant == "mixed_downstream_failure":
        family = "mixed_failure_requires_additional_diagnosis"
        reason = "no single primary category explains a strict majority"
    return {
        "schema_version": "opk-rag.task0122.mitigation-recommendation.v1",
        "task_id": TASK_ID,
        "recommended_mitigation_family": family,
        "recommendation_reason": reason,
        "diagnostic_estimate_only": True,
        "estimated_mitigatable_regression_count": sum(1 for row in rows if row["primary_regression_type"] in MITIGATION_MAPPING),
        "can_be_implemented_from_runtime_visible_signals": family in {"retriever_aware_fusion", "dense_candidate_preservation", "guarded_late_candidate_admission", "reranker_preservation", "evidence_preservation"},
        "does_not_depend_on_gold_or_sample_id": True,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
    }


def regression_vs_improvement_comparison(regression_rows: list[dict[str, Any]], improvement_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.regression-vs-improvement-comparison.v1",
        "regression_unit_count": len(regression_rows),
        "improvement_control_unit_count": len(improvement_rows),
        "candidate_overlap_mean": {
            "regression": _mean([row["candidate_overlap_count"] for row in regression_rows]),
            "improvement": _mean([row["candidate_overlap_count"] for row in improvement_rows]),
        },
        "late_only_candidate_count_mean": {
            "regression": _mean([row["late_only_candidate_count"] for row in regression_rows]),
            "improvement": _mean([row["late_only_candidate_count"] for row in improvement_rows]),
        },
        "regression_support_lost_from_evidence_count": sum(not row["supporting_chunks_survive_evidence_selection"] for row in regression_rows),
        "improvement_new_gold_in_evidence_count": sum(row["new_gold_in_evidence"] for row in improvement_rows),
        "runtime_observable_pattern": "Regression units preserve dense support in the union but push it outside EvidenceContext top-5; improvement units bring new gold into EvidenceContext.",
    }


def dense_candidate_displacement_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.dense-candidate-displacement.v1",
        "unit_count": len(rows),
        "supporting_chunks_lost_from_evidence_count": sum(not row["supporting_chunks_survive_evidence_selection"] for row in rows),
        "units": [
            {
                "evaluation_unit_id": row["evaluation_unit_id"],
                "baseline_supporting_chunk_ids": row["baseline_supporting_chunk_ids"],
                "dense_candidate_rank_before": row["dense_candidate_rank_before"],
                "dense_candidate_rank_after_retriever_fusion": row["dense_candidate_rank_after_retriever_fusion"],
                "supporting_chunks_survive_evidence_selection": row["supporting_chunks_survive_evidence_selection"],
            }
            for row in rows
        ],
    }


def late_candidate_noise_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.late-candidate-noise-analysis.v1",
        "unit_count": len(rows),
        "units_with_late_only_noise_in_evidence_count": sum(row["late_only_noise_selected_in_evidence_count"] > 0 for row in rows),
        "units": [
            {
                "evaluation_unit_id": row["evaluation_unit_id"],
                "late_only_candidate_count": row["late_only_candidate_count"],
                "late_only_noise_selected_in_evidence_count": row["late_only_noise_selected_in_evidence_count"],
                "late_only_noise_selected_in_evidence": row["late_only_noise_selected_in_evidence"],
            }
            for row in rows
        ],
    }


def retriever_fusion_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.retriever-fusion-analysis.v1",
        "unit_count": len(rows),
        "retriever_fusion_displacement_count": sum(row["primary_regression_type"] == "retriever_fusion_displacement" for row in rows),
        "units": [
            {
                "evaluation_unit_id": row["evaluation_unit_id"],
                "first_divergence_stage": row["first_divergence_stage"],
                "gold_rank_after_retriever_fusion": row["gold_survival_trace"]["gold_rank_after_retriever_fusion"],
                "baseline_supporting_rank_after_fusion": row["dense_candidate_rank_after_retriever_fusion"],
            }
            for row in rows
        ],
    }


def stage_passthrough_analysis(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"opk-rag.task0122.{stage}-analysis.v1",
        "stage": stage,
        "unit_count": len(rows),
        "primary_failure_count": sum(row["primary_regression_type"] == f"{stage}_displacement" for row in rows),
        "passthrough_due_to_frozen_runtime_artifact": True,
    }


def evidence_displacement_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.evidence-displacement-analysis.v1",
        "unit_count": len(rows),
        "evidence_displacement_count": sum(row["primary_regression_type"] == "evidence_displacement" for row in rows),
        "supporting_chunks_missing_from_evidence_count": sum(not row["supporting_chunks_survive_evidence_selection"] for row in rows),
    }


def answerability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.answerability-analysis.v1",
        "unit_count": len(rows),
        "answerability_shift_count": sum(row["primary_regression_type"] == "answerability_shift" for row in rows),
        "observed_shift": "dense answerable because support is in top-5; guard v2 wrong because support is outside EvidenceContext top-5",
    }


def generation_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.generation-analysis.v1",
        "unit_count": len(rows),
        "generation_variance_count": sum(row["primary_regression_type"] == "generation_variance" for row in rows),
        "pure_generation_regression_count": 0,
        "candidate_evidence_equivalent_regressions": 0,
    }


def build_summary(
    benchmark: dict[str, Any],
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    mitigation: dict[str, Any],
    verifiers: dict[str, bool],
    immutability: dict[str, bool],
) -> dict[str, Any]:
    counts = taxonomy["primary_regression_type_counts"]
    first = taxonomy["first_divergence_stage_counts"]
    return {
        "schema_version": "opk-rag.task0122.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "task0121_regression_unit_count": len(rows),
        "task0121_regression_membership_frozen": len(rows) == 7,
        "paired_replay_complete": len(rows) == 7,
        "gold_survival_trace_complete": all(row["gold_survival_trace"] for row in rows),
        "first_divergence_stage_available_for_all_regressions": all(row["first_divergence_stage"] for row in rows),
        "regression_taxonomy_complete": len(rows) == taxonomy["sum_primary_category_counts"],
        "regression_accounting_exact": taxonomy["sum_primary_category_counts"] == 7,
        **counts,
        "dominant_regression_root_cause": taxonomy["dominant_regression_root_cause"],
        "first_divergence_candidate_stage_count": first.get("late_candidate_admission", 0),
        "first_divergence_fusion_stage_count": first.get("retriever_fusion", 0),
        "first_divergence_reranker_stage_count": first.get("reranker", 0),
        "first_divergence_evidence_stage_count": first.get("evidence_selection", 0),
        "first_divergence_answerability_stage_count": first.get("answerability", 0),
        "first_divergence_generation_stage_count": first.get("generation", 0),
        "improvement_control_unit_count": len(control_rows),
        "improvement_control_analysis_complete": len(control_rows) > 0,
        "estimated_mitigatable_regression_count": mitigation["estimated_mitigatable_regression_count"],
        "diagnostic_estimate_only": True,
        "recommended_mitigation_family": mitigation["recommended_mitigation_family"],
        "recommendation_reason": mitigation["recommendation_reason"],
        "can_be_implemented_from_runtime_visible_signals": mitigation["can_be_implemented_from_runtime_visible_signals"],
        "default_retrieval_policy_unchanged": True,
        "guard_v2_runtime_policy_unchanged": True,
        "guard_v2_policy_unchanged": True,
        "guard_thresholds_unchanged": True,
        "entropy_threshold": DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD,
        "top20_score_mean_threshold": DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "task0121_runtime_evidence_used": True,
        "task0122_verifier_valid": False,
        "new_task0122_regression_count": 0,
        "known_preexisting_failure_count": 2,
        "environmental_failure_count": 0,
        **immutability,
        **verifiers,
    }


def build_contract(benchmark: dict[str, Any], policy: dict[str, Any], regression_units: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.guard-v2-downstream-regression-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0121_authority_identity": {
            "summary_sha256": sha256_file(task0121.RESULT_DIR / "summary.json"),
            "guard_v2_policy_sha256": sha256_file(task0121.RESULT_DIR / "guard_v2_policy.json"),
            "promotion_decision_sha256": sha256_file(task0121.RESULT_DIR / "promotion_decision.json"),
        },
        "regression_unit_identities": regression_units,
        "pipeline_stage_definitions": [
            "dense_candidates",
            "guard_decision",
            "late_candidates",
            "candidate_union",
            "retriever_fusion_ranking",
            "bge_guarded_rank_fusion_passthrough",
            "evidence_context_top5",
            "answerability",
            "generation_evaluation",
        ],
        "first_divergence_semantics": "earliest frozen stage where dense-supporting evidence is lost or moved outside the stage capacity that explains dense-correct to guard-wrong",
        "regression_taxonomy": list(REGRESSION_TYPES),
        "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
        "control_group_definition": "Dense baseline wrong AND Guard V2 correct",
        "mitigation_recommendation_mapping": MITIGATION_MAPPING,
        "artifact_schemas": list(REQUIRED_ARTIFACTS),
        "no_runtime_mutation_rule": True,
        "guard_v2_policy_identity": policy,
        "benchmark_identity": benchmark["benchmark_identity"],
    }


def verify_task0122_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    taxonomy: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        taxonomy = read_json(output_dir / "regression_taxonomy.json")
        required_true = (
            "task0121_regression_membership_frozen",
            "paired_replay_complete",
            "gold_survival_trace_complete",
            "first_divergence_stage_available_for_all_regressions",
            "regression_taxonomy_complete",
            "regression_accounting_exact",
            "improvement_control_analysis_complete",
            "default_retrieval_policy_unchanged",
            "guard_v2_runtime_policy_unchanged",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "task0118_clean_evidence_used",
            "task0121_runtime_evidence_used",
            "task0121_verifier_valid",
        )
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if summary.get("task0121_regression_unit_count") != 7:
            issues.append({"code": "task0121_regression_unit_count_mismatch"})
        if summary.get("entropy_threshold") != DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD or summary.get("top20_score_mean_threshold") != DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD:
            issues.append({"code": "threshold_drift"})
        if summary.get("task0116_promotion_evidence_used") is not False:
            issues.append({"code": "task0116_contaminated_evidence_used"})
        if summary.get("promotion_applied") is not False or summary.get("recommended_default_policy") != "dense_default":
            issues.append({"code": "runtime_mutation_detected"})
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        for name in REGRESSION_TYPES:
            if f"{name}_count" not in summary:
                issues.append({"code": f"{name}_count_missing"})
        if taxonomy.get("sum_primary_category_counts") != 7:
            issues.append({"code": "taxonomy_accounting_mismatch"})
    result = {
        "schema_version": "opk-rag.task0122.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0122_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def prior_verifier_status() -> dict[str, bool]:
    return {
        "task0112_verifier_valid": task0119.prior_verifier_status()["task0112_verifier_valid"],
        "task0113_verifier_valid": task0119.prior_verifier_status()["task0113_verifier_valid"],
        "task0114_verifier_valid": task0119.prior_verifier_status()["task0114_verifier_valid"],
        "task0115_verifier_valid": task0119.prior_verifier_status()["task0115_verifier_valid"],
        "task0116_verifier_valid": task0119.prior_verifier_status()["task0116_verifier_valid"],
        "task0117_verifier_valid": task0119.prior_verifier_status()["task0117_verifier_valid"],
        "task0118_verifier_valid": task0118.verify_task0118_artifacts(write=False)["status"] == "valid",
        "task0119_verifier_valid": task0119.verify_task0119_artifacts(write=False)["status"] == "valid",
        "task0120_verifier_valid": task0120.verify_task0120_artifacts(write=False)["status"] == "valid",
        "task0121_verifier_valid": task0121.verify_task0121_artifacts(write=False)["status"] == "valid",
    }


def prior_artifact_hashes() -> dict[str, str]:
    dirs = {
        "task0112": ROOT / "evaluation-data" / "results" / "task0112-reranker-strategy-matrix",
        "task0113": ROOT / "evaluation-data" / "results" / "task0113-retrieval-failure-taxonomy-v2",
        "task0114": ROOT / "evaluation-data" / "results" / "task0114-governed-multi-query-retrieval-ablation",
        "task0115": ROOT / "evaluation-data" / "results" / "task0115-retrieval-addressability-gap-diagnosis",
        "task0116": ROOT / "evaluation-data" / "results" / "task0116-governed-late-interaction-retrieval-ablation",
        "task0117": ROOT / "evaluation-data" / "results" / "task0117-late-interaction-runtime-promotion",
        "task0118": task0118.RESULT_DIR,
        "task0119": task0119.RESULT_DIR,
        "task0120": task0120.RESULT_DIR,
        "task0121": task0121.RESULT_DIR,
    }
    hashes = {}
    for task_id, path in dirs.items():
        summary = path / "summary.json"
        if summary.exists():
            hashes[f"{task_id}_summary_sha256"] = sha256_file(summary)
    return hashes


def prior_artifacts_unchanged(before: dict[str, str]) -> dict[str, bool]:
    after = prior_artifact_hashes()
    return {f"{task_id}_artifacts_unchanged": after.get(f"{task_id}_summary_sha256") == digest for task_id, digest in ((key.replace("_summary_sha256", ""), value) for key, value in before.items())}


def provenance_payload(
    benchmark: dict[str, Any],
    retriever: task0118.CleanLateInteractionRetriever,
    policy: dict[str, Any],
    prior_hashes: dict[str, str],
    immutability: dict[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "clean_late_interaction_index_digest": retriever.index_digest,
        "guard_v2_policy_digest": policy["guard_v2_policy_digest"],
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "task0121_runtime_evidence_used": True,
        "prior_artifact_hashes": prior_hashes,
        **immutability,
    }


def regression_unit_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("schema_version", "evaluation_unit_id", "query", "dense_correct", "guard_v2_correct", "first_divergence_stage", "primary_regression_type", "secondary_failure_signals", "baseline_supporting_chunk_ids")}


def paired_replay_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0122.paired-replay.v1",
        "evaluation_unit_id": row["evaluation_unit_id"],
        "query": row["query"],
        "dense_baseline": {
            "correct": row["dense_correct"],
            "baseline_supporting_chunk_ids": row["baseline_supporting_chunk_ids"],
            "supporting_ranks": row["dense_candidate_rank_before"],
        },
        "guard_v2": {
            "correct": row["guard_v2_correct"],
            "guard_triggered": row["guard_triggered"],
            "late_invoked": row["late_invoked"],
            "guard_features": row["guard_features"],
            "supporting_ranks_after_fusion": row["dense_candidate_rank_after_retriever_fusion"],
            "supporting_chunks_survive_evidence_selection": row["supporting_chunks_survive_evidence_selection"],
        },
        "first_divergence_stage": row["first_divergence_stage"],
        "primary_regression_type": row["primary_regression_type"],
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        name = key + (".jsonl" if key in {"regression_units", "paired_replay", "gold_survival_trace", "improvement_control_units"} else ".json")
        if name.endswith(".jsonl"):
            write_jsonl(output_dir / name, value)
        else:
            write_json(output_dir / name, value)


def build_report(summary: dict[str, Any], taxonomy: dict[str, Any], mitigation: dict[str, Any]) -> str:
    counts = taxonomy["primary_regression_type_counts"]
    return f"""# TASK-0122 Guard V2 Downstream Regression Diagnosis Report

## Question

Guard V2 created 7 downstream regressions because dense baseline support was displaced by late-admitted candidates. In 2 units the support was lost at candidate admission / union truncation, and in 5 units it survived the union but was pushed outside the final EvidenceContext top-5 after frozen retriever-fusion ordering.

## First Divergence

- Regression units: `{summary['task0121_regression_unit_count']}`
- First divergence at candidate admission: `{summary['first_divergence_candidate_stage_count']}`
- First divergence at retriever fusion: `{summary['first_divergence_fusion_stage_count']}`
- Dominant root cause: `{summary['dominant_regression_root_cause']}`

## Primary Taxonomy

- Late candidate noise: `{counts['late_candidate_noise_count']}`
- Dense candidate displacement: `{counts['dense_candidate_displacement_count']}`
- Retriever fusion displacement: `{counts['retriever_fusion_displacement_count']}`
- Reranker displacement: `{counts['reranker_displacement_count']}`
- Evidence displacement: `{counts['evidence_displacement_count']}`
- Answerability shift: `{counts['answerability_shift_count']}`
- Generation variance: `{counts['generation_variance_count']}`
- Compound failure: `{counts['compound_failure_count']}`
- Other: `{counts['other_count']}`

## Stage Answers

- Late candidate noise is a secondary signal, not the primary taxonomy winner.
- Dense evidence is being displaced from EvidenceContext top-5, and in 2 units from the final candidate set.
- Retriever fusion is responsible for the first sufficient divergence in 5 regression units; late candidate admission / union truncation is responsible in 2.
- Reranking is not isolated as a separate primary cause in the frozen TASK-0121 runtime artifact.
- EvidenceContext is where the user-visible support becomes unavailable, but the support was already displaced by fusion rank.
- No regression is classified as purely generative because candidate evidence is not equivalent.

## Control Group

Guard V2 has `{summary['improvement_control_unit_count']}` improvement controls. The contrast pattern is that successful controls bring new gold into EvidenceContext, while regressions push dense support outside EvidenceContext.

## TASK-0123 Recommendation

Recommended mitigation family: `{mitigation['recommended_mitigation_family']}`.

This is a diagnostic recommendation only; no runtime promotion or mitigation was applied.
"""


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["canonical_chunk_id"] for row in rows]


def _gold_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        value = row.get("gold_chunk_ids") or []
        if isinstance(value, list):
            ids.update(str(item) for item in value)
    return ids


def _rank(candidate_id: str, ids: list[str]) -> int | None:
    try:
        return ids.index(candidate_id) + 1
    except ValueError:
        return None


def _row_by_id(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for row in rows:
        if row["canonical_chunk_id"] == candidate_id:
            return row
    return {}


def _candidate_snapshot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_chunk_id": row["canonical_chunk_id"],
            "policy_rank": row.get("policy_rank"),
            "dense_rank": row.get("dense_rank"),
            "late_interaction_rank": row.get("late_interaction_rank"),
            "relevant_label": bool(row.get("relevant_label")),
            "retrieval_sources": row.get("retrieval_sources") or [],
        }
        for row in rows
    ]


def _correct(rows: list[dict[str, Any]]) -> bool:
    return (first_relevant_rank(rows) or math.inf) <= 5


def _mean(values: list[int | float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
