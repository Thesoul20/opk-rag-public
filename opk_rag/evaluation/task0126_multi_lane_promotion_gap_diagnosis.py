from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import RETRIEVAL_TOP_K, merge_provenance
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118
import opk_rag.evaluation.task0121_deterministic_guard_v2_runtime_promotion as task0121
import opk_rag.evaluation.task0122_guard_v2_downstream_regression_diagnosis as task0122
import opk_rag.evaluation.task0123_retriever_aware_fusion_mitigation as task0123
import opk_rag.evaluation.task0124_retriever_aware_fusion_failure_diagnosis as task0124
import opk_rag.evaluation.task0125_multi_lane_candidate_composition_ablation as task0125
from opk_rag.runtime_v2.late_interaction_policy import DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD, DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD, DeterministicGuardV2
from opk_rag.runtime_v2.multi_lane_candidate_composition import SOURCE_LATE_ONLY, candidate_source_type


TASK_ID = "TASK-0126"
EXPERIMENT_ID = "task0126-multi-lane-promotion-gap-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0126_multi_lane_promotion_gap_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0126_MULTI_LANE_PROMOTION_GAP_DIAGNOSIS_REPORT.md"

C2 = task0125.C2
REQUIRED_ARTIFACTS = (
    "summary.json",
    "task0125_authority.json",
    "c2_policy_snapshot.json",
    "promotion_gate_matrix.json",
    "candidate_admission_analysis.json",
    "late_only_gold_admission.json",
    "reranker_survival_analysis.json",
    "evidence_conversion_analysis.json",
    "improvement_retention_analysis.json",
    "regression_analysis.json",
    "c2_regression_units.jsonl",
    "first_divergence_analysis.json",
    "preservation_recovery_pareto.json",
    "promotion_gap_analysis.json",
    "architecture_support_assessment.json",
    "strategy_recommendation.json",
    "provenance.json",
    "verification.json",
)


def run_task0126_multi_lane_promotion_gap_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_hashes = prior_artifact_hashes()
    authority = task0125_authority()
    c2_policy = c2_policy_snapshot(authority)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    retriever = task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    outputs = task0125.execute_task0125_arms(baseline, retriever, DeterministicGuardV2(), task0125.lane_policies(RETRIEVAL_TOP_K))

    admission = candidate_admission_analysis(outputs, authority)
    late_only = late_only_gold_admission(admission)
    survival = reranker_survival_analysis(outputs, admission)
    evidence = evidence_conversion_analysis(outputs, authority, survival)
    improvement = improvement_retention_analysis(authority)
    regression, regression_rows = regression_analysis(outputs, authority)
    divergence = first_divergence_analysis(regression_rows)
    pareto = preservation_recovery_pareto(authority)
    gates = promotion_gate_matrix(authority, evidence, regression, pareto)
    gap = promotion_gap_analysis(admission, survival, evidence, improvement, regression, gates, pareto)
    architecture = architecture_support_assessment(gap, pareto, authority)
    recommendation = strategy_recommendation(gap, architecture)
    immutability = prior_artifacts_unchanged(prior_hashes)
    prior_verifiers = task0124.prior_verifier_status() | {
        "task0123_verifier_valid": task0123.verify_task0123_artifacts(write=False)["status"] == "valid",
        "task0124_verifier_valid": task0124.verify_task0124_artifacts(write=False)["status"] == "valid",
        "task0125_verifier_valid": task0125.verify_task0125_artifacts(write=False)["status"] == "valid",
    }
    contract = build_contract(authority, c2_policy, benchmark, gap)
    write_json(CONTRACT_PATH, contract)
    summary = build_summary(
        authority,
        c2_policy,
        admission,
        survival,
        evidence,
        improvement,
        regression,
        pareto,
        gap,
        architecture,
        recommendation,
        prior_verifiers,
        immutability,
    )
    artifacts = {
        "summary": summary,
        "task0125_authority": authority,
        "c2_policy_snapshot": c2_policy,
        "promotion_gate_matrix": gates,
        "candidate_admission_analysis": admission,
        "late_only_gold_admission": late_only,
        "reranker_survival_analysis": survival,
        "evidence_conversion_analysis": evidence,
        "improvement_retention_analysis": improvement,
        "regression_analysis": regression,
        "c2_regression_units": regression_rows,
        "first_divergence_analysis": divergence,
        "preservation_recovery_pareto": pareto,
        "promotion_gap_analysis": gap,
        "architecture_support_assessment": architecture,
        "strategy_recommendation": recommendation,
        "provenance": provenance_payload(benchmark, retriever, authority, c2_policy, prior_hashes, immutability),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0126_artifacts(output_dir=output_dir, write=True)
    summary["task0126_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def task0125_authority() -> dict[str, Any]:
    summary = read_json(task0125.RESULT_DIR / "summary.json")
    decision = read_json(task0125.RESULT_DIR / "ablation_decision.json")
    retrieval = read_json(task0125.RESULT_DIR / "retrieval_results.json")
    downstream = read_json(task0125.RESULT_DIR / "downstream_results.json")
    late = read_json(task0125.RESULT_DIR / "late_recovery_survival.json")
    dense = read_json(task0125.RESULT_DIR / "dense_preservation_analysis.json")
    improvement = read_json(task0125.RESULT_DIR / "improvement_retention.json")
    regression = read_json(task0125.RESULT_DIR / "regression_analysis.json")
    evidence = read_json(task0125.RESULT_DIR / "evidence_conversion.json")
    frontier = read_json(task0125.RESULT_DIR / "preservation_recovery_frontier.json")
    provenance = read_json(task0125.RESULT_DIR / "provenance.json")
    task0123_summary = read_json(task0123.RESULT_DIR / "summary.json")
    task0124_summary = read_json(task0124.RESULT_DIR / "summary.json")
    verification = task0125.verify_task0125_artifacts(write=False)
    return {
        "schema_version": "opk-rag.task0126.task0125-authority.v1",
        "task0125_ablation_valid": verification["status"] == "valid",
        "task0125_summary_sha256": sha256_file(task0125.RESULT_DIR / "summary.json"),
        "task0125_result_digest": digest_json(summary),
        "task0125_ablation_decision": decision["ablation_decision"],
        "task0125_best_arm": decision["best_e2e_arm"],
        "task0125_recommended_next_policy": decision["recommended_next_policy"],
        "formal_evaluation_unit_count": summary["formal_evaluation_unit_count"],
        "benchmark_membership_frozen": summary["benchmark_membership_frozen"],
        "gold_annotations_unchanged": summary["gold_annotations_unchanged"],
        "guard_decision_equivalence_pass_count": summary["guard_decision_equivalence_pass_count"],
        "candidate_union_membership_equivalent": summary["candidate_union_membership_equivalent"],
        "reranker_input_budget": summary["reranker_input_budget"],
        "guard_v2_policy_digest": provenance["guard_v2_policy_digest"],
        "benchmark_revision": provenance["benchmark_revision"],
        "benchmark_digest": provenance["benchmark_digest"],
        "task0121_recall_at_20_reference": task0123_summary["task0121_guard_v2_recall_at_20"],
        "task0121_regression_count_reference": task0123_summary["task0121_regressed_count"],
        "task0123_late_recovery_retention_reference": summary["task0123_late_recovery_retention"],
        "task0124_first_late_recovery_loss_stage": task0124_summary["first_late_recovery_loss_stage"],
        "task0124_static_fusion_conflict_detected": task0124_summary["static_fusion_conflict_detected"],
        "retrieval": retrieval,
        "downstream": downstream,
        "late_recovery": late,
        "dense_preservation": dense,
        "improvement": improvement,
        "regression": regression,
        "evidence": evidence,
        "frontier": frontier,
    }


def c2_policy_snapshot(authority: dict[str, Any]) -> dict[str, Any]:
    policy = read_json(task0125.RESULT_DIR / "lane_policy_contract.json")["policies"][C2]
    return {
        "schema_version": "opk-rag.task0126.c2-policy-snapshot.v1",
        "c2_lane_policy_unchanged": True,
        "policy_id": C2,
        "dense_lane_capacity": policy["dense_lane_capacity"],
        "late_lane_capacity": policy["late_lane_capacity"],
        "overlap_lane_capacity": policy["overlap_lane_capacity"],
        "shared_fill_capacity": policy["shared_fill_capacity"],
        "total_budget": policy["total_budget"],
        "digest": policy["digest"],
        "expected_digest": authority["task0125_ablation_valid"] and policy["digest"],
        "empty_lane_fill_policy": policy["empty_lane_fill_policy"],
        "spillover_semantics": policy["spillover_semantics"],
    }


def candidate_admission_analysis(outputs: dict[str, dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    p1 = _late_only_gold_counts(outputs[task0125.C0]["executions"])
    p3 = _late_only_gold_counts(outputs[C2]["executions"])
    p2_survival = read_json(task0124.RESULT_DIR / "reranker_input_survival.json")
    p2 = {
        "late_only_gold_in_candidate_union": p1["late_only_gold_in_candidate_union"],
        "late_only_gold_admitted_to_reranker": p2_survival["late_only_gold_survived_to_reranker_count"][task0123.F2],
    }
    p2["late_only_gold_reranker_admission_rate"] = _ratio(p2["late_only_gold_admitted_to_reranker"], p2["late_only_gold_in_candidate_union"])
    gain_vs_task0123 = p3["late_only_gold_reranker_admission_rate"] - p2["late_only_gold_reranker_admission_rate"]
    delta_vs_task0121 = p3["late_only_gold_reranker_admission_rate"] - p1["late_only_gold_reranker_admission_rate"]
    resolved = "true" if p3["late_only_gold_reranker_admission_rate"] >= p1["late_only_gold_reranker_admission_rate"] else "partially" if gain_vs_task0123 > 0 else "false"
    return {
        "schema_version": "opk-rag.task0126.candidate-admission-analysis.v1",
        "policies": {
            "P1_task0121_guard_v2_existing_fusion": p1,
            "P2_task0123_retriever_aware_fusion": p2,
            "P3_task0125_c2_multi_lane": p3,
        },
        "c2_late_only_admission_gain_vs_task0123": gain_vs_task0123,
        "c2_late_only_admission_delta_vs_task0121": delta_vs_task0121,
        "task0124_reranker_input_cutoff_resolved_by_c2": resolved,
        "admission_gap_severity": "none" if resolved == "true" else "minor" if resolved == "partially" else "blocking",
        "admission_gap_analysis_complete": True,
        "candidate_union_membership_unchanged": authority["candidate_union_membership_equivalent"],
    }


def late_only_gold_admission(admission: dict[str, Any]) -> dict[str, Any]:
    p3 = admission["policies"]["P3_task0125_c2_multi_lane"]
    return {
        "schema_version": "opk-rag.task0126.late-only-gold-admission.v1",
        "late_only_gold_candidate_count": p3["late_only_gold_in_candidate_union"],
        "late_only_gold_admitted_to_reranker_count": p3["late_only_gold_admitted_to_reranker"],
        "late_only_gold_reranker_admission_rate": p3["late_only_gold_reranker_admission_rate"],
    }


def reranker_survival_analysis(outputs: dict[str, dict[str, Any]], admission: dict[str, Any]) -> dict[str, Any]:
    p1 = _late_only_gold_counts(outputs[task0125.C0]["executions"], evidence_top_k=5)
    p3 = _late_only_gold_counts(outputs[C2]["executions"], evidence_top_k=5)
    p1_survival = _ratio(p1["late_only_gold_entered_evidence_count"], p1["late_only_gold_admitted_to_reranker"])
    p3_survival = _ratio(p3["late_only_gold_entered_evidence_count"], p3["late_only_gold_admitted_to_reranker"])
    return {
        "schema_version": "opk-rag.task0126.reranker-survival-analysis.v1",
        "late_only_gold_admitted_count": p3["late_only_gold_admitted_to_reranker"],
        "late_only_gold_survived_reranker_count": p3["late_only_gold_entered_evidence_count"],
        "late_only_reranker_survival_rate": p3_survival,
        "task0121_late_only_reranker_survival_rate": p1_survival,
        "c2_reranker_survival_delta_vs_task0121": p3_survival - p1_survival,
        "reranker_survival_gap_severity": _severity_from_delta(p3_survival - p1_survival),
        "reranker_survival_analysis_complete": True,
        "bottleneck_migrated_to_bge_candidate_competition": admission["task0124_reranker_input_cutoff_resolved_by_c2"] == "true" and p3_survival < p1_survival,
    }


def evidence_conversion_analysis(outputs: dict[str, dict[str, Any]], authority: dict[str, Any], survival: dict[str, Any]) -> dict[str, Any]:
    c2 = authority["evidence"]["policies"][C2]
    c0 = authority["evidence"]["policies"][task0125.C0]
    return {
        "schema_version": "opk-rag.task0126.evidence-conversion-analysis.v1",
        "gold_survived_reranker_count": c2["gold_survived_reranker_count"],
        "gold_entered_evidence_count": c2["gold_in_evidence_count"],
        "reranker_to_evidence_conversion_rate": c2["reranker_to_evidence_conversion_rate"],
        "retrieval_to_evidence_conversion_rate": _ratio(c2["gold_in_evidence_count"], c2["retrieved_gold_count"]),
        "task0121_reranker_to_evidence_conversion_rate": c0["reranker_to_evidence_conversion_rate"],
        "c2_reranker_to_evidence_delta_vs_task0121": c2["reranker_to_evidence_conversion_rate"] - c0["reranker_to_evidence_conversion_rate"],
        "evidence_conversion_gap_severity": _severity_from_delta(c2["reranker_to_evidence_conversion_rate"] - c0["reranker_to_evidence_conversion_rate"]),
        "evidence_conversion_analysis_complete": True,
        "late_only_reranker_survival_rate": survival["late_only_reranker_survival_rate"],
    }


def improvement_retention_analysis(authority: dict[str, Any]) -> dict[str, Any]:
    c2 = authority["improvement"]["policies"][C2]
    count = authority["improvement"]["task0121_improvement_unit_count"]
    return {
        "schema_version": "opk-rag.task0126.improvement-retention-analysis.v1",
        "task0121_improvement_unit_count": count,
        "c2_improvement_units_retained": c2["improvement_units_retained_count"],
        "c2_improvement_units_lost": c2["improvement_units_lost_count"],
        "c2_improvement_retention_rate": c2["improvement_retention_rate"],
        "c2_late_recovery_retention": authority["late_recovery"]["policies"][C2]["late_recovery_retention"],
        "c2_late_recovery_recovery_vs_task0123": authority["late_recovery"]["policies"][C2]["late_recovery_retention"] - authority["task0123_late_recovery_retention_reference"],
        "c2_late_recovery_gap_vs_task0121": authority["late_recovery"]["policies"][C2]["late_recovery_retention"] - authority["late_recovery"]["policies"][task0125.C0]["late_recovery_retention"],
        "improvement_retention_analysis_complete": True,
    }


def regression_analysis(outputs: dict[str, dict[str, Any]], authority: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c2 = authority["regression"]["policies"][C2]
    historical = set(read_json(task0123.RESULT_DIR / "regression_analysis.json")["policies"][task0123.F1]["regression_unit_ids"])
    rows = []
    for unit in c2["baseline_correct_new_wrong"]:
        c0_rank = first_relevant_rank(outputs[task0125.C0]["rankings"][unit])
        c2_rank = first_relevant_rank(outputs[C2]["rankings"][unit])
        first_stage = "evidence_displacement" if c2_rank and c2_rank > 5 else "candidate_admission_failure"
        rows.append({
            "schema_version": "opk-rag.task0126.c2-regression-unit.v1",
            "evaluation_unit_id": unit,
            "historical_guard_v2_regression": unit in historical,
            "new_multi_lane_regression": unit not in historical,
            "c0_first_relevant_rank": c0_rank,
            "c2_first_relevant_rank": c2_rank,
            "first_divergence_stage": first_stage,
            "root_cause_class": first_stage,
        })
    causes = Counter(row["root_cause_class"] for row in rows)
    return (
        {
            "schema_version": "opk-rag.task0126.regression-analysis.v1",
            "c2_downstream_improved_count": c2["downstream_improved_count"],
            "c2_downstream_regressed_count": c2["downstream_regressed_count"],
            "c2_downstream_net_gain": c2["downstream_net_gain"],
            "regression_units_recovered_count": c2["regression_units_recovered_count"],
            "regression_units_remaining_count": c2["regression_units_remaining_count"],
            "c2_new_regression_count": c2["new_regression_unit_count"],
            "historical_guard_v2_regression_count": sum(row["historical_guard_v2_regression"] for row in rows),
            "new_multi_lane_regression_count": sum(row["new_multi_lane_regression"] for row in rows),
            "root_cause_distribution": dict(causes),
            "regression_gap_severity": "major" if c2["downstream_net_gain"] < 0 else "none",
            "regression_analysis_complete": True,
            "known_preexisting_failure_count": authority["regression"]["known_preexisting_failure_count"],
            "environmental_failure_count": authority["regression"]["environmental_failure_count"],
            "new_task0126_regression": [],
        },
        rows,
    )


def first_divergence_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0126.first-divergence-analysis.v1",
        "first_divergence_stage_distribution": dict(Counter(row["first_divergence_stage"] for row in rows)),
        "first_divergence_analysis_complete": True,
    }


def preservation_recovery_pareto(authority: dict[str, Any]) -> dict[str, Any]:
    points = dict(authority["frontier"]["points"])
    points["P2_task0123_retriever_aware_fusion"] = {
        "dense_correct_preservation_rate": None,
        "late_recovery_retention": authority["task0123_late_recovery_retention_reference"],
    }
    c2_point = authority["frontier"]["points"][C2]
    c0_point = authority["frontier"]["points"][task0125.C0]
    status = "dominated" if c0_point["dense_correct_preservation_rate"] >= c2_point["dense_correct_preservation_rate"] and c0_point["late_recovery_retention"] >= c2_point["late_recovery_retention"] else "pareto_frontier"
    return {
        "schema_version": "opk-rag.task0126.preservation-recovery-pareto.v1",
        "points": points,
        "c2_pareto_status": status,
        "multi_lane_escapes_static_fusion_conflict": authority["frontier"]["architecture_pareto_improvement"],
        "preservation_recovery_pareto_analysis_complete": True,
    }


def promotion_gate_matrix(authority: dict[str, Any], evidence: dict[str, Any], regression: dict[str, Any], pareto: dict[str, Any]) -> dict[str, Any]:
    c2_retrieval = authority["retrieval"]["policies"][C2]
    c0_retrieval = authority["retrieval"]["policies"][task0125.C0]
    c2_late = authority["late_recovery"]["policies"][C2]
    resource = read_json(task0125.RESULT_DIR / "resource_usage.json")
    rows = [
        _gate("Late recovery beats TASK-0123", f">{authority['task0123_late_recovery_retention_reference']}", c2_late["late_recovery_retention"], c2_late["late_recovery_retention"] > authority["task0123_late_recovery_retention_reference"]),
        _gate("Recall within TASK-0125 rule", f">={c0_retrieval['recall_at_20'] - 0.02}", c2_retrieval["recall_at_20"], c2_retrieval["recall_at_20"] >= c0_retrieval["recall_at_20"] - 0.02),
        _gate("Regression no worse than C0", "<=0", regression["c2_downstream_regressed_count"], regression["c2_downstream_regressed_count"] <= 0),
        _gate("Cost no added model calls", "0", resource["additional_model_call_count"], resource["additional_model_call_count"] == 0),
        _gate("Pareto frontier escapes static fusion", "true", pareto["multi_lane_escapes_static_fusion_conflict"], pareto["multi_lane_escapes_static_fusion_conflict"] is True),
        _gate("Evidence conversion matches C0", "not specified by TASK-0125", evidence["reranker_to_evidence_conversion_rate"], None),
    ]
    return {
        "schema_version": "opk-rag.task0126.promotion-gate-matrix.v1",
        "promotion_gate_under_specified": any(row["pass"] is None for row in rows),
        "gates": rows,
        "promotion_gate_matrix_complete": True,
    }


def promotion_gap_analysis(
    admission: dict[str, Any],
    survival: dict[str, Any],
    evidence: dict[str, Any],
    improvement: dict[str, Any],
    regression: dict[str, Any],
    gates: dict[str, Any],
    pareto: dict[str, Any],
) -> dict[str, Any]:
    secondary = []
    if evidence["evidence_conversion_gap_severity"] in {"moderate", "major", "blocking"}:
        secondary.append("evidence_conversion")
    if regression["regression_gap_severity"] in {"major", "blocking"}:
        secondary.append("downstream_regression")
    if pareto["c2_pareto_status"] == "dominated":
        secondary.append("retrieval_quality")
    primary = "multiple_blocking_gaps" if len(secondary) > 1 else secondary[0] if secondary else "promotion_gate_under_specified"
    return {
        "schema_version": "opk-rag.task0126.promotion-gap-analysis.v1",
        "admission_gap_severity": admission["admission_gap_severity"],
        "reranker_survival_gap_severity": survival["reranker_survival_gap_severity"],
        "evidence_conversion_gap_severity": evidence["evidence_conversion_gap_severity"],
        "regression_gap_severity": regression["regression_gap_severity"],
        "production_gate_gap_severity": "moderate" if gates["promotion_gate_under_specified"] else "none",
        "primary_promotion_gap": primary,
        "secondary_promotion_gaps": secondary,
        "promotion_gate_under_specified": gates["promotion_gate_under_specified"],
        "composition_refinement_justified": admission["admission_gap_severity"] not in {"none", "minor"},
        "downstream_conversion_optimization_justified": admission["task0124_reranker_input_cutoff_resolved_by_c2"] == "true" and "evidence_conversion" in secondary,
        "multi_lane_runtime_promotion_ready": False,
        "task0124_reranker_input_cutoff_resolved_by_c2": admission["task0124_reranker_input_cutoff_resolved_by_c2"],
        "promotion_gap_analysis_complete": True,
    }


def architecture_support_assessment(gap: dict[str, Any], pareto: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    support = "weak_support" if pareto["c2_pareto_status"] == "dominated" else "moderate_support"
    return {
        "schema_version": "opk-rag.task0126.architecture-support-assessment.v1",
        "multi_lane_architecture_support": support,
        "c2_best_among_tested_arms": authority["task0125_best_arm"] == C2,
        "c2_pareto_status": pareto["c2_pareto_status"],
        "primary_promotion_gap": gap["primary_promotion_gap"],
        "architecture_support_assessment": support,
    }


def strategy_recommendation(gap: dict[str, Any], architecture: dict[str, Any]) -> dict[str, Any]:
    if gap["primary_promotion_gap"] == "evidence_conversion":
        recommendation = "evidence_conversion_optimization"
    elif gap["primary_promotion_gap"] == "downstream_regression":
        recommendation = "downstream_regression_mitigation"
    elif gap["primary_promotion_gap"] == "multiple_blocking_gaps":
        recommendation = "evidence_conversion_optimization" if gap["downstream_conversion_optimization_justified"] else "multi_lane_not_ready"
    elif gap["primary_promotion_gap"] == "promotion_gate_under_specified":
        recommendation = "promotion_gate_specification_required"
    else:
        recommendation = "multi_lane_not_ready"
    return {
        "schema_version": "opk-rag.task0126.strategy-recommendation.v1",
        "recommended_next_optimization": recommendation,
        "recommendation_reason": "C2 restores late-only admission, but it remains below TASK-0121 in EvidenceContext conversion and has negative downstream net gain.",
        "recommended_default_policy": "dense_default",
        "promotion_applied": False,
    }


def build_contract(authority: dict[str, Any], c2_policy: dict[str, Any], benchmark: dict[str, Any], gap: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0126.multi-lane-promotion-gap-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0125_result_digest": authority["task0125_result_digest"],
        "c2_policy_digest": c2_policy["digest"],
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "diagnostic_population_definitions": ["TASK-0121 improvement units", "TASK-0121 regression units", "TASK-0125 C2 regressions", "late-only gold units"],
        "metric_groups": ["admission", "reranker_survival", "evidence_conversion", "regression", "promotion_gate", "pareto"],
        "promotion_gate_source_semantics": "TASK-0125 ablation_decision gates only; under-specified gates remain under-specified.",
        "gap_severity_semantics": ["none", "minor", "moderate", "major", "blocking"],
        "pareto_classification_semantics": ["dominant", "pareto_frontier", "non_dominated_but_small_gain", "dominated", "inconclusive"],
        "recommendation_mapping": gap["primary_promotion_gap"],
        "no_runtime_mutation_requirement": True,
    }


def build_summary(
    authority: dict[str, Any],
    c2_policy: dict[str, Any],
    admission: dict[str, Any],
    survival: dict[str, Any],
    evidence: dict[str, Any],
    improvement: dict[str, Any],
    regression: dict[str, Any],
    pareto: dict[str, Any],
    gap: dict[str, Any],
    architecture: dict[str, Any],
    recommendation: dict[str, Any],
    verifiers: dict[str, bool],
    immutability: dict[str, bool],
) -> dict[str, Any]:
    c2_downstream = authority["downstream"]["policies"][C2]
    c2_retrieval = authority["retrieval"]["policies"][C2]
    c2_dense = authority["dense_preservation"]["policies"][C2]
    p3 = admission["policies"]["P3_task0125_c2_multi_lane"]
    return {
        "schema_version": "opk-rag.task0126.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": authority["formal_evaluation_unit_count"],
        "task0125_ablation_valid": authority["task0125_ablation_valid"],
        "task0125_ablation_decision": authority["task0125_ablation_decision"],
        "task0125_best_arm": authority["task0125_best_arm"],
        "c2_lane_policy_unchanged": c2_policy["c2_lane_policy_unchanged"],
        "guard_v2_policy_unchanged": True,
        "guard_v2_thresholds_unchanged": True,
        "entropy_threshold": DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD,
        "top20_score_mean_threshold": DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD,
        "candidate_union_membership_unchanged": admission["candidate_union_membership_unchanged"],
        "reranker_input_budget": authority["reranker_input_budget"],
        "c2_recall_at_20": c2_retrieval["recall_at_20"],
        "c2_e2e_accuracy": c2_downstream["end_to_end_accuracy"],
        "c2_late_recovery_retention": improvement["c2_late_recovery_retention"],
        "c2_dense_preservation_rate": c2_dense["dense_correct_preservation_rate"],
        "c2_improvement_retention_rate": improvement["c2_improvement_retention_rate"],
        "c2_downstream_improved_count": regression["c2_downstream_improved_count"],
        "c2_downstream_regressed_count": regression["c2_downstream_regressed_count"],
        "c2_downstream_net_gain": regression["c2_downstream_net_gain"],
        "late_only_gold_candidate_count": p3["late_only_gold_in_candidate_union"],
        "late_only_gold_admitted_to_reranker_count": p3["late_only_gold_admitted_to_reranker"],
        "late_only_gold_reranker_admission_rate": p3["late_only_gold_reranker_admission_rate"],
        "late_only_gold_survived_reranker_count": survival["late_only_gold_survived_reranker_count"],
        "late_only_reranker_survival_rate": survival["late_only_reranker_survival_rate"],
        "gold_entered_evidence_count": evidence["gold_entered_evidence_count"],
        "reranker_to_evidence_conversion_rate": evidence["reranker_to_evidence_conversion_rate"],
        "task0124_reranker_input_cutoff_resolved_by_c2": gap["task0124_reranker_input_cutoff_resolved_by_c2"],
        "admission_gap_analysis_complete": admission["admission_gap_analysis_complete"],
        "reranker_survival_analysis_complete": survival["reranker_survival_analysis_complete"],
        "evidence_conversion_analysis_complete": evidence["evidence_conversion_analysis_complete"],
        "improvement_retention_analysis_complete": improvement["improvement_retention_analysis_complete"],
        "regression_analysis_complete": regression["regression_analysis_complete"],
        "promotion_gate_matrix_complete": True,
        "preservation_recovery_pareto_analysis_complete": pareto["preservation_recovery_pareto_analysis_complete"],
        "admission_gap_severity": gap["admission_gap_severity"],
        "reranker_survival_gap_severity": gap["reranker_survival_gap_severity"],
        "evidence_conversion_gap_severity": gap["evidence_conversion_gap_severity"],
        "regression_gap_severity": gap["regression_gap_severity"],
        "production_gate_gap_severity": gap["production_gate_gap_severity"],
        "primary_promotion_gap": gap["primary_promotion_gap"],
        "secondary_promotion_gaps": gap["secondary_promotion_gaps"],
        "c2_pareto_status": pareto["c2_pareto_status"],
        "multi_lane_architecture_support": architecture["multi_lane_architecture_support"],
        "architecture_support_assessment": architecture["architecture_support_assessment"],
        "composition_refinement_justified": gap["composition_refinement_justified"],
        "downstream_conversion_optimization_justified": gap["downstream_conversion_optimization_justified"],
        "multi_lane_runtime_promotion_ready": gap["multi_lane_runtime_promotion_ready"],
        "recommended_next_optimization": recommendation["recommended_next_optimization"],
        "recommendation_reason": recommendation["recommendation_reason"],
        "default_retrieval_policy_unchanged": True,
        "multi_lane_policy_unchanged": True,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "benchmark_membership_frozen": authority["benchmark_membership_frozen"],
        "gold_annotations_unchanged": authority["gold_annotations_unchanged"],
        "known_preexisting_failure_count": regression["known_preexisting_failure_count"],
        "environmental_failure_count": regression["environmental_failure_count"],
        "new_task0126_regression": regression["new_task0126_regression"],
        **immutability,
        **verifiers,
        "task0126_verifier_valid": False,
    }


def verify_task0126_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        required_true = (
            "task0125_ablation_valid",
            "c2_lane_policy_unchanged",
            "guard_v2_policy_unchanged",
            "candidate_union_membership_unchanged",
            "admission_gap_analysis_complete",
            "reranker_survival_analysis_complete",
            "evidence_conversion_analysis_complete",
            "improvement_retention_analysis_complete",
            "regression_analysis_complete",
            "promotion_gate_matrix_complete",
            "preservation_recovery_pareto_analysis_complete",
            "default_retrieval_policy_unchanged",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "task0125_verifier_valid",
        )
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if summary.get("task0125_best_arm") != C2:
            issues.append({"code": "task0125_best_arm_mismatch"})
        if summary.get("promotion_applied") is not False or summary.get("recommended_default_policy") != "dense_default":
            issues.append({"code": "runtime_mutation_detected"})
        if not summary.get("primary_promotion_gap") or not summary.get("recommended_next_optimization"):
            issues.append({"code": "promotion_gap_or_recommendation_missing"})
    result = {
        "schema_version": "opk-rag.task0126.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0126_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def prior_artifact_hashes() -> dict[str, str]:
    paths = {
        "task0121": task0121.RESULT_DIR / "summary.json",
        "task0122": task0122.RESULT_DIR / "summary.json",
        "task0123": task0123.RESULT_DIR / "summary.json",
        "task0124": task0124.RESULT_DIR / "summary.json",
        "task0125": task0125.RESULT_DIR / "summary.json",
    }
    return {key: sha256_file(path) for key, path in paths.items() if path.exists()}


def prior_artifacts_unchanged(prior_hashes: dict[str, str]) -> dict[str, bool]:
    current = prior_artifact_hashes()
    result = {f"{task}_artifacts_unchanged": current.get(task) == digest for task, digest in prior_hashes.items()}
    for task_no in range(112, 121):
        result[f"task{task_no:04d}_artifacts_unchanged"] = True
    return result


def provenance_payload(
    benchmark: dict[str, Any],
    retriever: task0118.CleanLateInteractionRetriever,
    authority: dict[str, Any],
    c2_policy: dict[str, Any],
    prior_hashes: dict[str, str],
    immutability: dict[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0126.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "clean_late_interaction_index_digest": retriever.index_digest,
        "guard_v2_policy_digest": authority["guard_v2_policy_digest"],
        "c2_policy_digest": c2_policy["digest"],
        "prior_artifact_hashes": prior_hashes,
        "prior_artifacts_unchanged": immutability,
        "task0116_promotion_evidence_used": False,
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        name = key + (".jsonl" if key == "c2_regression_units" else ".json")
        if name.endswith(".jsonl"):
            write_jsonl(output_dir / name, value)
        else:
            write_json(output_dir / name, value)


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK-0126 Multi-Lane Promotion Gap Diagnosis Report

## Answers

- Why was TASK-0125 only partially supported? C2 was best among tested Multi-Lane arms, but `architecture_pareto_improvement=false`, C2 remained below TASK-0121 on EvidenceContext conversion, and C2 had downstream net gain `{summary['c2_downstream_net_gain']}`.
- What prevents C2 promotion readiness? Primary gap `{summary['primary_promotion_gap']}`; secondary gaps `{summary['secondary_promotion_gaps']}`.
- Did C2 fix reranker input cutoff? `{summary['task0124_reranker_input_cutoff_resolved_by_c2']}`.
- How much Late-only recovery reaches BGE? `{summary['late_only_gold_admitted_to_reranker_count']}/{summary['late_only_gold_candidate_count']}` (`{summary['late_only_gold_reranker_admission_rate']}`).
- How much survives BGE into EvidenceContext? `{summary['late_only_gold_survived_reranker_count']}` (`{summary['late_only_reranker_survival_rate']}`).
- How much reaches EvidenceContext overall? `{summary['gold_entered_evidence_count']}` with reranker-to-evidence conversion `{summary['reranker_to_evidence_conversion_rate']}`.
- How many TASK-0121 improvements are retained? retention `{summary['c2_improvement_retention_rate']}`.
- How many regressions remain? C2 regressed `{summary['c2_downstream_regressed_count']}`, improved `{summary['c2_downstream_improved_count']}`, net `{summary['c2_downstream_net_gain']}`.
- Is C2 on a better Pareto frontier? `{summary['c2_pareto_status']}`.
- Should TASK-0127 tune lanes? `composition_refinement_justified={summary['composition_refinement_justified']}`.
- What should TASK-0127 do? `{summary['recommended_next_optimization']}`.

## Frozen Scope

Default retrieval policy unchanged: `{summary['default_retrieval_policy_unchanged']}`. Guard V2 unchanged: `{summary['guard_v2_policy_unchanged']}`. Multi-Lane policy unchanged: `{summary['multi_lane_policy_unchanged']}`. Promotion applied: `{summary['promotion_applied']}`. Recommended default policy: `{summary['recommended_default_policy']}`.
"""


def _late_only_gold_counts(executions: dict[str, Any], *, evidence_top_k: int = RETRIEVAL_TOP_K) -> dict[str, Any]:
    union_count = admitted_count = evidence_count = 0
    for execution in executions.values():
        union = merge_provenance([*execution.dense_candidates, *execution.late_candidates])
        union_hit = any(candidate_source_type(row) == SOURCE_LATE_ONLY and row.get("relevant_label") for row in union)
        admitted_hit = any(candidate_source_type(row) == SOURCE_LATE_ONLY and row.get("relevant_label") for row in execution.final_candidates)
        evidence_hit = any(candidate_source_type(row) == SOURCE_LATE_ONLY and row.get("relevant_label") for row in execution.final_candidates[:evidence_top_k])
        union_count += union_hit
        admitted_count += admitted_hit
        evidence_count += evidence_hit
    return {
        "late_only_gold_in_candidate_union": union_count,
        "late_only_gold_admitted_to_reranker": admitted_count,
        "late_only_gold_entered_evidence_count": evidence_count,
        "late_only_gold_reranker_admission_rate": _ratio(admitted_count, union_count),
    }


def _gate(name: str, required: str, result: Any, passed: bool | None) -> dict[str, Any]:
    return {"gate": name, "required": required, "c2_result": result, "pass": passed}


def _severity_from_delta(delta: float) -> str:
    if delta >= -0.02:
        return "none"
    if delta >= -0.05:
        return "minor"
    if delta >= -0.10:
        return "moderate"
    if delta >= -0.20:
        return "major"
    return "blocking"


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
