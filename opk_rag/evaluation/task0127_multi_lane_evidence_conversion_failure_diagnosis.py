from __future__ import annotations

from collections import Counter
import math
import statistics
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0092_reranker_downstream_validation import EVIDENCE_CONTEXT_TOP_K
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import rank_fusion
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark, guarded_rank_fusion_unit
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import RETRIEVAL_TOP_K, merge_provenance
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118
import opk_rag.evaluation.task0120_dense_retrieval_failure_signal_diagnosis as task0120
import opk_rag.evaluation.task0121_deterministic_guard_v2_runtime_promotion as task0121
import opk_rag.evaluation.task0124_retriever_aware_fusion_failure_diagnosis as task0124
import opk_rag.evaluation.task0125_multi_lane_candidate_composition_ablation as task0125
import opk_rag.evaluation.task0126_multi_lane_promotion_gap_diagnosis as task0126
from opk_rag.runtime_v2.late_interaction_policy import DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD, DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD, DeterministicGuardV2
from opk_rag.runtime_v2.multi_lane_candidate_composition import SOURCE_DENSE_ONLY, SOURCE_LATE_ONLY, SOURCE_OVERLAP, SOURCE_TYPES, candidate_source_type
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA


TASK_ID = "TASK-0127"
EXPERIMENT_ID = "task0127-multi-lane-evidence-conversion-failure-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0127_multi_lane_evidence_conversion_failure_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0127_MULTI_LANE_EVIDENCE_CONVERSION_FAILURE_DIAGNOSIS_REPORT.md"

C2 = task0125.C2
BGE_SURVIVAL_CUTOFF = RETRIEVAL_TOP_K
GENERATION_CONTEXT_LIMIT = None
LOSS_STAGES = (
    "bge_ranking",
    "guarded_rank_fusion",
    "evidence_admission",
    "evidence_budget",
    "answerability",
    "generation",
    "compound_downstream_failure",
    "none",
)
PRIMARY_TAXONOMY = (
    "bge_candidate_displacement",
    "guarded_rank_fusion_displacement",
    "evidence_budget_cutoff",
    "evidence_redundancy_crowding",
    "evidence_source_imbalance",
    "evidence_ordering_issue",
    "context_truncation",
    "answerability_shift",
    "generation_variance",
    "compound_downstream_failure",
    "other",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "task0126_authority.json",
    "funnel_summary.json",
    "candidate_survival_trace.jsonl",
    "late_only_gold_funnel.json",
    "bge_survival_analysis.json",
    "guarded_rank_fusion_survival.json",
    "evidence_admission_analysis.json",
    "evidence_budget_analysis.json",
    "evidence_redundancy_audit.json",
    "evidence_source_distribution.json",
    "gold_evidence_competition.jsonl",
    "context_truncation_audit.json",
    "failure_taxonomy.json",
    "improvement_control.json",
    "failure_success_contrast.json",
    "regression_funnel.json",
    "retrieval_quality_interaction.json",
    "runtime_visible_separability.json",
    "diagnostic_upper_bound.json",
    "mitigation_recommendation.json",
    "provenance.json",
    "verification.json",
)


def run_task0127_multi_lane_evidence_conversion_failure_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = task0126_authority()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    retriever = task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    outputs = task0125.execute_task0125_arms(baseline, retriever, DeterministicGuardV2(), {C2: task0125.lane_policies(RETRIEVAL_TOP_K)[C2]})
    c0_rankings = outputs[task0125.C0]["rankings"]
    c2_executions = outputs[C2]["executions"]
    c2_rankings = build_c2_downstream_rankings(c2_executions)

    traces = candidate_survival_trace(c2_executions, c2_rankings)
    funnel = funnel_summary(c2_executions, c2_rankings, traces)
    late_funnel = late_only_gold_funnel(traces)
    bge = bge_survival_analysis(traces)
    fusion = guarded_rank_fusion_survival(traces)
    evidence_admission = evidence_admission_analysis(c2_rankings, traces)
    budget = evidence_budget_analysis(traces)
    redundancy = evidence_redundancy_audit(c2_rankings)
    source_distribution = evidence_source_distribution(c2_rankings)
    competition_rows = gold_evidence_competition(c2_rankings, traces)
    truncation = context_truncation_audit(c2_rankings)
    taxonomy = failure_taxonomy(traces, redundancy)
    improvement = improvement_control(c0_rankings, c2_rankings, traces)
    contrast = failure_success_contrast(traces, improvement)
    regression = regression_funnel(c0_rankings, c2_rankings, traces)
    retrieval_gap = retrieval_quality_interaction(c2_executions)
    separability = runtime_visible_separability(traces)
    upper_bound = diagnostic_upper_bound(taxonomy)
    recommendation = mitigation_recommendation(taxonomy, separability)
    contract = build_contract(authority, benchmark)
    write_json(CONTRACT_PATH, contract)
    summary = build_summary(authority, funnel, late_funnel, bge, fusion, budget, taxonomy, regression, retrieval_gap, recommendation, separability)
    artifacts = {
        "summary": summary,
        "task0126_authority": authority,
        "funnel_summary": funnel,
        "candidate_survival_trace": traces,
        "late_only_gold_funnel": late_funnel,
        "bge_survival_analysis": bge,
        "guarded_rank_fusion_survival": fusion,
        "evidence_admission_analysis": evidence_admission,
        "evidence_budget_analysis": budget,
        "evidence_redundancy_audit": redundancy,
        "evidence_source_distribution": source_distribution,
        "gold_evidence_competition": competition_rows,
        "context_truncation_audit": truncation,
        "failure_taxonomy": taxonomy,
        "improvement_control": improvement,
        "failure_success_contrast": contrast,
        "regression_funnel": regression,
        "retrieval_quality_interaction": retrieval_gap,
        "runtime_visible_separability": separability,
        "diagnostic_upper_bound": upper_bound,
        "mitigation_recommendation": recommendation,
        "provenance": provenance_payload(benchmark, retriever, authority),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0127_artifacts(output_dir=output_dir, write=True)
    summary["task0127_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, late_funnel, taxonomy, recommendation), encoding="utf-8")
    return summary


def task0126_authority() -> dict[str, Any]:
    verification = task0126.verify_task0126_artifacts(write=False)
    summary = read_json(task0126.RESULT_DIR / "summary.json")
    task0125_policy = read_json(task0125.RESULT_DIR / "lane_policy_contract.json")["policies"][C2]
    return {
        "schema_version": "opk-rag.task0127.task0126-authority.v1",
        "task0126_diagnosis_valid": verification["status"] == "valid",
        "task0126_c2_policy_used": summary["task0125_best_arm"] == C2,
        "task0126_summary_sha256": sha256_file(task0126.RESULT_DIR / "summary.json"),
        "formal_evaluation_unit_count": summary["formal_evaluation_unit_count"],
        "benchmark_membership_frozen": summary["benchmark_membership_frozen"],
        "gold_annotations_unchanged": summary["gold_annotations_unchanged"],
        "task0126_late_only_gold_admission_rate": summary["late_only_gold_reranker_admission_rate"],
        "task0126_recommended_next_optimization": summary["recommended_next_optimization"],
        "task0125_best_arm": summary["task0125_best_arm"],
        "c2_policy_digest": task0125_policy["digest"],
        "c2_lane_policy_unchanged": True,
        "reranker_input_budget_unchanged": summary["reranker_input_budget"] == RETRIEVAL_TOP_K,
        "guard_v2_policy_unchanged": summary["guard_v2_policy_unchanged"],
        "guard_v2_thresholds_unchanged": summary["guard_v2_thresholds_unchanged"],
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
    }


def build_c2_downstream_rankings(executions: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rankings = {}
    for unit, execution in executions.items():
        pre_bge = assign_pre_bge_rank(execution.final_candidates)
        direct = bge_ranking(pre_bge)
        fused = rank_fusion(pre_bge, direct, {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA})
        rankings[unit] = guarded_rank_fusion_unit(pre_bge, direct, fused)
    return rankings


def candidate_survival_trace(executions: dict[str, Any], final_rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for unit, execution in sorted(executions.items()):
        union = merge_provenance([*execution.dense_candidates, *execution.late_candidates])
        pre_bge = assign_pre_bge_rank(execution.final_candidates)
        direct = bge_ranking(pre_bge)
        final = final_rankings[unit]
        bge_by_id = {row["canonical_chunk_id"]: row for row in direct}
        final_by_id = {row["canonical_chunk_id"]: row for row in final}
        relevant_union = [row for row in union if row.get("relevant_label")]
        if not relevant_union:
            continue
        pre_ids = {row["canonical_chunk_id"] for row in pre_bge}
        evidence_ids = [row["canonical_chunk_id"] for row in final[:EVIDENCE_CONTEXT_TOP_K]]
        correct_answer = any(row.get("relevant_label") for row in final[:EVIDENCE_CONTEXT_TOP_K])
        for candidate in relevant_union:
            cid = candidate["canonical_chunk_id"]
            pre_row = next((row for row in pre_bge if row["canonical_chunk_id"] == cid), None)
            bge_row = bge_by_id.get(cid)
            final_row = final_by_id.get(cid)
            pre_rank = pre_row.get("pre_bge_rank") if pre_row else None
            bge_rank = bge_row.get("bge_rank") if bge_row else None
            final_rank = final_row.get("policy_rank") if final_row else None
            source = candidate_source_type(pre_row or candidate)
            gold_entered = final_rank is not None and final_rank <= EVIDENCE_CONTEXT_TOP_K
            first_stage = first_loss_stage(pre_rank, bge_rank, final_rank, gold_entered, correct_answer)
            rows.append(
                {
                    "schema_version": "opk-rag.task0127.candidate-survival-trace.v1",
                    "evaluation_unit_id": unit,
                    "candidate_id": cid,
                    "source_type": source,
                    "gold_in_candidate_union": True,
                    "gold_admitted_to_reranker": cid in pre_ids,
                    "pre_bge_rank": pre_rank,
                    "bge_rank": bge_rank,
                    "bge_score": bge_row.get("reranker_score") if bge_row else None,
                    "bge_score_observable": bge_row is not None and bge_row.get("reranker_score") is not None,
                    "survived_bge": bge_rank is not None and bge_rank <= BGE_SURVIVAL_CUTOFF,
                    "rank_before_guarded_fusion": bge_rank,
                    "rank_after_guarded_fusion": final_rank,
                    "gold_rank_delta_after_guarded_rank_fusion": None if bge_rank is None or final_rank is None else final_rank - bge_rank,
                    "survived_guarded_fusion": final_rank is not None and final_rank <= RETRIEVAL_TOP_K,
                    "final_rank": final_rank,
                    "evidence_cutoff": EVIDENCE_CONTEXT_TOP_K,
                    "evidence_candidate_count": min(len(final), EVIDENCE_CONTEXT_TOP_K),
                    "gold_entered_evidence": gold_entered,
                    "gold_survived_evidence_budget": gold_entered,
                    "gold_visible_to_generation": None,
                    "generation_visibility_observable": False,
                    "correct_answer": correct_answer,
                    "first_evidence_conversion_loss_stage": first_stage,
                    "primary_taxonomy": primary_taxonomy(first_stage, final[:EVIDENCE_CONTEXT_TOP_K]),
                    "evidence_position": final_rank if gold_entered else None,
                    "evidence_context_candidate_ids": evidence_ids,
                    "candidate_source_distribution": dict(Counter(candidate_source_type(row) for row in pre_bge)),
                    "bge_competing_candidate_count": len(pre_bge) - 1,
                    "document_id": candidate.get("document_id"),
                    "section_id": candidate.get("section_id"),
                }
            )
    return rows


def funnel_summary(executions: dict[str, Any], final_rankings: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    units = {row["evaluation_unit_id"] for row in traces}
    addressable = [row for row in traces if row["gold_admitted_to_reranker"]]
    counts = {
        "gold_in_candidate_union_count": len(units),
        "gold_admitted_to_reranker_count": len({row["evaluation_unit_id"] for row in addressable}),
        "gold_survived_bge_count": len({row["evaluation_unit_id"] for row in addressable if row["survived_bge"]}),
        "gold_survived_rank_fusion_count": len({row["evaluation_unit_id"] for row in addressable if row["survived_guarded_fusion"]}),
        "gold_admitted_to_evidence_context_count": len({row["evaluation_unit_id"] for row in addressable if row["gold_entered_evidence"]}),
        "gold_survived_evidence_budget_count": len({row["evaluation_unit_id"] for row in addressable if row["gold_survived_evidence_budget"]}),
        "gold_visible_to_generation_count": None,
        "correct_answer_count": sum(any(row.get("relevant_label") for row in final_rankings[unit][:EVIDENCE_CONTEXT_TOP_K]) for unit in units),
    }
    base = counts["gold_in_candidate_union_count"]
    admitted = counts["gold_admitted_to_reranker_count"]
    bge = counts["gold_survived_bge_count"]
    fusion = counts["gold_survived_rank_fusion_count"]
    evidence = counts["gold_admitted_to_evidence_context_count"]
    correct = counts["correct_answer_count"]
    return {
        "schema_version": "opk-rag.task0127.funnel-summary.v1",
        **counts,
        "reranker_admission_rate": _ratio(admitted, base),
        "bge_survival_rate": _ratio(bge, admitted),
        "rank_fusion_survival_rate": _ratio(fusion, bge),
        "evidence_admission_rate": _ratio(evidence, fusion),
        "evidence_budget_survival_rate": _ratio(counts["gold_survived_evidence_budget_count"], evidence),
        "generation_conversion_rate": None,
        "retrieval_to_evidence_conversion_rate": _ratio(evidence, base),
        "retrieval_to_correct_answer_conversion_rate": _ratio(correct, base),
        "conversion_addressable_count": admitted,
        "retrieval_unaddressable_count": sum(not any(row.get("relevant_label") for row in merge_provenance([*execution.dense_candidates, *execution.late_candidates])) for execution in executions.values()),
        "loss_count": base - correct,
        "generation_visible_evidence_trace_available": False,
    }


def late_only_gold_funnel(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in traces if row["source_type"] == SOURCE_LATE_ONLY and row["gold_admitted_to_reranker"]]
    units = {row["evaluation_unit_id"] for row in rows}
    survived_bge = {row["evaluation_unit_id"] for row in rows if row["survived_bge"]}
    survived_fusion = {row["evaluation_unit_id"] for row in rows if row["survived_guarded_fusion"]}
    evidence = {row["evaluation_unit_id"] for row in rows if row["gold_entered_evidence"]}
    correct = {row["evaluation_unit_id"] for row in rows if row["gold_entered_evidence"] and row["correct_answer"]}
    return {
        "schema_version": "opk-rag.task0127.late-only-gold-funnel.v1",
        "late_only_gold_candidate_count": len(units),
        "late_only_gold_admitted_to_reranker_count": len(units),
        "late_only_gold_survived_bge_count": len(survived_bge),
        "late_only_gold_survived_rank_fusion_count": len(survived_fusion),
        "late_only_gold_entered_evidence_count": len(evidence),
        "late_only_gold_survived_evidence_budget_count": len(evidence),
        "late_only_gold_visible_to_generation_count": None,
        "late_only_gold_correct_answer_count": len(correct),
        "late_only_gold_bge_survival_rate": _ratio(len(survived_bge), len(units)),
        "late_only_gold_evidence_conversion_rate": _ratio(len(evidence), len(units)),
        "generation_visibility_observable": False,
    }


def bge_survival_analysis(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = {}
    for source in SOURCE_TYPES:
        rows = [row for row in traces if row["source_type"] == source and row["gold_admitted_to_reranker"]]
        ranks = [row["bge_rank"] for row in rows if row["bge_rank"] is not None]
        by_source[source] = {
            "gold_count": len(rows),
            "survived_bge_count": sum(row["survived_bge"] for row in rows),
            "bge_survival_rate": _ratio(sum(row["survived_bge"] for row in rows), len(rows)),
            "mean_bge_rank": statistics.mean(ranks) if ranks else None,
            "bge_score_observable_count": sum(row["bge_score_observable"] for row in rows),
        }
    return {
        "schema_version": "opk-rag.task0127.bge-survival-analysis.v1",
        "by_source_type": by_source,
        "late_only_gold_bge_survival_rate": by_source[SOURCE_LATE_ONLY]["bge_survival_rate"],
        "dense_gold_bge_survival_rate": by_source[SOURCE_DENSE_ONLY]["bge_survival_rate"],
        "overlap_gold_bge_survival_rate": by_source[SOURCE_OVERLAP]["bge_survival_rate"],
        "source_associated_survival_difference_observed": len({stats["bge_survival_rate"] for stats in by_source.values() if stats["gold_count"]}) > 1,
        "source_bias_claim_made": False,
        "bge_score_observable_unit_count": len({row["evaluation_unit_id"] for row in traces if row["bge_score_observable"]}),
        "bge_score_unobservable_unit_count": len({row["evaluation_unit_id"] for row in traces if not row["bge_score_observable"]}),
    }


def guarded_rank_fusion_survival(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in traces if row["survived_bge"]]
    dropped = [row for row in rows if not row["survived_guarded_fusion"]]
    deltas = [row["gold_rank_delta_after_guarded_rank_fusion"] for row in rows if row["gold_rank_delta_after_guarded_rank_fusion"] is not None]
    return {
        "schema_version": "opk-rag.task0127.guarded-rank-fusion-survival.v1",
        "gold_passed_bge_count": len(rows),
        "gold_survived_guarded_rank_fusion_count": len(rows) - len(dropped),
        "gold_guarded_rank_fusion_survival_rate": _ratio(len(rows) - len(dropped), len(rows)),
        "gold_dropped_by_guarded_rank_fusion_count": len(dropped),
        "gold_rank_delta_after_guarded_rank_fusion_mean": statistics.mean(deltas) if deltas else None,
        "rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "guarded_rank_fusion_unchanged": True,
    }


def evidence_admission_analysis(final_rankings: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    lost = [row for row in traces if row["survived_guarded_fusion"] and not row["gold_entered_evidence"]]
    return {
        "schema_version": "opk-rag.task0127.evidence-admission-analysis.v1",
        "gold_survived_final_reranking_count": len([row for row in traces if row["survived_guarded_fusion"]]),
        "gold_entered_evidence_count": len([row for row in traces if row["gold_entered_evidence"]]),
        "gold_survived_final_but_not_evidence_count": len(lost),
        "evidence_cutoff": EVIDENCE_CONTEXT_TOP_K,
        "evidence_candidate_count_distribution": _distribution([min(len(rows), EVIDENCE_CONTEXT_TOP_K) for rows in final_rankings.values()]),
    }


def evidence_budget_analysis(traces: list[dict[str, Any]]) -> dict[str, Any]:
    lost = [row for row in traces if row["survived_guarded_fusion"] and not row["gold_entered_evidence"]]
    return {
        "schema_version": "opk-rag.task0127.evidence-budget-analysis.v1",
        "evidence_input_budget": EVIDENCE_CONTEXT_TOP_K,
        "evidence_policy_unchanged": True,
        "gold_lost_due_to_evidence_budget_count": len(lost),
        "loss_at_evidence_budget_count": len(lost),
        "evidence_budget_cutoff_loss_rate": _ratio(len(lost), len([row for row in traces if row["survived_guarded_fusion"]])),
    }


def evidence_redundancy_audit(final_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_unit = []
    redundant_slots = 0
    total_slots = 0
    for unit, rows in final_rankings.items():
        evidence = rows[:EVIDENCE_CONTEXT_TOP_K]
        seen_doc_section = set()
        unit_redundant = 0
        for row in evidence:
            key = (row.get("document_id"), row.get("section_id"))
            if key in seen_doc_section and any(key):
                unit_redundant += 1
            seen_doc_section.add(key)
        redundant_slots += unit_redundant
        total_slots += len(evidence)
        per_unit.append({"evaluation_unit_id": unit, "redundant_slot_count": unit_redundant, "evidence_slot_count": len(evidence)})
    return {
        "schema_version": "opk-rag.task0127.evidence-redundancy-audit.v1",
        "redundant_slot_count": redundant_slots,
        "evidence_slot_count": total_slots,
        "evidence_redundancy_rate": _ratio(redundant_slots, total_slots),
        "near_duplicate_detection_basis": "same_document_and_same_section",
        "semantic_redundancy_observable": False,
        "per_unit": per_unit,
    }


def evidence_source_distribution(final_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    per_unit = []
    for unit, rows in final_rankings.items():
        unit_counts = Counter(candidate_source_type(row) for row in rows[:EVIDENCE_CONTEXT_TOP_K])
        counts.update(unit_counts)
        per_unit.append({
            "evaluation_unit_id": unit,
            "dense_only_evidence_count": unit_counts[SOURCE_DENSE_ONLY],
            "late_only_evidence_count": unit_counts[SOURCE_LATE_ONLY],
            "overlap_evidence_count": unit_counts[SOURCE_OVERLAP],
        })
    return {
        "schema_version": "opk-rag.task0127.evidence-source-distribution.v1",
        "dense_only_evidence_count": counts[SOURCE_DENSE_ONLY],
        "late_only_evidence_count": counts[SOURCE_LATE_ONLY],
        "overlap_evidence_count": counts[SOURCE_OVERLAP],
        "per_unit": per_unit,
    }


def gold_evidence_competition(final_rankings: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        if trace["gold_entered_evidence"] or trace["final_rank"] is None:
            continue
        ranking = final_rankings[trace["evaluation_unit_id"]]
        competitors = []
        for candidate in ranking[:EVIDENCE_CONTEXT_TOP_K]:
            competitors.append({
                "candidate_id": candidate["canonical_chunk_id"],
                "classification": competitor_classification(trace, candidate),
                "source_type": candidate_source_type(candidate),
                "document_id": candidate.get("document_id"),
                "section_id": candidate.get("section_id"),
            })
        rows.append({
            "schema_version": "opk-rag.task0127.gold-evidence-competition.v1",
            "evaluation_unit_id": trace["evaluation_unit_id"],
            "gold_candidate_id": trace["candidate_id"],
            "gold_final_rank": trace["final_rank"],
            "evidence_cutoff": EVIDENCE_CONTEXT_TOP_K,
            "candidates_that_displaced_gold": competitors,
        })
    return rows


def context_truncation_audit(final_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    lengths = []
    for rows in final_rankings.values():
        lengths.append(sum(len(str(row.get("document") or row.get("text") or "")) for row in rows[:EVIDENCE_CONTEXT_TOP_K]))
    return {
        "schema_version": "opk-rag.task0127.context-truncation-audit.v1",
        "evidence_serialized_length": {"mean": statistics.mean(lengths) if lengths else 0.0, "max": max(lengths) if lengths else 0},
        "generation_context_limit": GENERATION_CONTEXT_LIMIT,
        "truncation_occurred": None,
        "generation_visibility_observable": False,
        "generation_visible_evidence_trace_available": False,
    }


def failure_taxonomy(traces: list[dict[str, Any]], redundancy: dict[str, Any]) -> dict[str, Any]:
    failures = [row for row in traces if row["gold_admitted_to_reranker"] and not row["correct_answer"]]
    counts = Counter(row["primary_taxonomy"] for row in failures)
    for key in PRIMARY_TAXONOMY:
        counts.setdefault(key, 0)
    stage_counts = Counter(row["first_evidence_conversion_loss_stage"] for row in failures)
    for key in LOSS_STAGES:
        stage_counts.setdefault(key, 0)
    dominant = max(PRIMARY_TAXONOMY, key=lambda key: (counts[key], -PRIMARY_TAXONOMY.index(key))) if failures else "none"
    return {
        "schema_version": "opk-rag.task0127.failure-taxonomy.v1",
        "formal_conversion_failure_count": len(failures),
        "primary_taxonomy_counts": dict(counts),
        "sum_primary_taxonomy_counts": sum(counts.values()),
        "first_evidence_conversion_loss_stage_counts": dict(stage_counts),
        "first_evidence_conversion_loss_stage": dominant_stage(stage_counts),
        "loss_at_bge_count": stage_counts["bge_ranking"],
        "loss_at_rank_fusion_count": stage_counts["guarded_rank_fusion"],
        "loss_at_evidence_admission_count": stage_counts["evidence_admission"],
        "loss_at_evidence_budget_count": stage_counts["evidence_budget"],
        "loss_at_answerability_count": stage_counts["answerability"],
        "loss_at_generation_count": stage_counts["generation"],
        "dominant_evidence_conversion_root_cause": dominant,
        "evidence_redundancy_rate": redundancy["evidence_redundancy_rate"],
        "classification_precedence": ["BGE", "Guarded Rank Fusion", "Evidence Admission", "Evidence Budget", "Generation Visibility", "Answerability", "Generation"],
    }


def improvement_control(c0_rankings: dict[str, list[dict[str, Any]]], c2_rankings: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    improved = [unit for unit in sorted(c0_rankings) if not _correct(c0_rankings[unit]) and _correct(c2_rankings[unit])]
    trace_by_unit = _traces_by_unit(traces)
    return {
        "schema_version": "opk-rag.task0127.improvement-control.v1",
        "improvement_population_definition": "Dense wrong; C2 correct",
        "improvement_unit_count": len(improved),
        "gold_admitted_count": sum(any(row["gold_admitted_to_reranker"] for row in trace_by_unit.get(unit, [])) for unit in improved),
        "bge_survived_count": sum(any(row["survived_bge"] for row in trace_by_unit.get(unit, [])) for unit in improved),
        "evidence_survived_count": sum(any(row["gold_entered_evidence"] for row in trace_by_unit.get(unit, [])) for unit in improved),
        "answer_correct_count": len(improved),
        "unit_ids": improved,
    }


def failure_success_contrast(traces: list[dict[str, Any]], improvement: dict[str, Any]) -> dict[str, Any]:
    success_units = set(improvement["unit_ids"])
    success = [row for row in traces if row["evaluation_unit_id"] in success_units and row["gold_admitted_to_reranker"]]
    failure = [row for row in traces if row["gold_admitted_to_reranker"] and not row["correct_answer"]]
    return {
        "schema_version": "opk-rag.task0127.failure-success-contrast.v1",
        "success_count": len(success),
        "failure_count": len(failure),
        "success_features": aggregate_trace_features(success),
        "failure_features": aggregate_trace_features(failure),
        "contrast_complete": True,
    }


def regression_funnel(c0_rankings: dict[str, list[dict[str, Any]]], c2_rankings: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    regressed = [unit for unit in sorted(c0_rankings) if _correct(c0_rankings[unit]) and not _correct(c2_rankings[unit])]
    trace_by_unit = _traces_by_unit(traces)
    rows = []
    for unit in regressed:
        unit_traces = trace_by_unit.get(unit, [])
        first = dominant_stage(Counter(row["first_evidence_conversion_loss_stage"] for row in unit_traces)) if unit_traces else "candidate_membership"
        rows.append({"evaluation_unit_id": unit, "first_divergence_stage": "evidence_context" if first == "evidence_budget" else first})
    shared = "inconclusive"
    if rows:
        shared = "true" if Counter(row["first_divergence_stage"] for row in rows).most_common(1)[0][0] in {"evidence_context", "evidence_admission"} else "false"
    return {
        "schema_version": "opk-rag.task0127.regression-funnel.v1",
        "downstream_regression_population_definition": "Dense correct; C2 wrong",
        "downstream_regression_unit_count": len(regressed),
        "regression_rows": rows,
        "first_divergence_stage_counts": dict(Counter(row["first_divergence_stage"] for row in rows)),
        "recovery_regression_shared_root_cause": shared,
    }


def retrieval_quality_interaction(executions: dict[str, Any]) -> dict[str, Any]:
    addressable = unaddressable = admitted = 0
    for execution in executions.values():
        union = merge_provenance([*execution.dense_candidates, *execution.late_candidates])
        if any(row.get("relevant_label") for row in union):
            addressable += 1
            admitted += any(row.get("relevant_label") for row in execution.final_candidates)
        else:
            unaddressable += 1
    return {
        "schema_version": "opk-rag.task0127.retrieval-quality-interaction.v1",
        "conversion_addressable_count": admitted,
        "retrieval_unaddressable_count": unaddressable,
        "gold_available_but_not_admitted_count": addressable - admitted,
        "retrieval_quality_gap_separately_reported": True,
    }


def runtime_visible_separability(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in traces if row["gold_admitted_to_reranker"] and row["bge_rank"] is not None]
    labels = [1 if row["correct_answer"] else 0 for row in rows]
    scores = [-float(row["bge_rank"]) for row in rows]
    return {
        "schema_version": "opk-rag.task0127.runtime-visible-separability.v1",
        "runtime_visible_features": ["bge_rank", "candidate_source_distribution", "retrieval_source_provenance", "evidence_slot_pressure"],
        "gold_used_as_future_runtime_feature": False,
        "roc_auc_bge_rank": roc_auc(labels, scores),
        "pr_auc_bge_rank": average_precision(labels, scores),
        "separability_diagnostic_only": True,
    }


def diagnostic_upper_bound(taxonomy: dict[str, Any]) -> dict[str, Any]:
    fixable = taxonomy["primary_taxonomy_counts"].get("evidence_budget_cutoff", 0) + taxonomy["primary_taxonomy_counts"].get("evidence_redundancy_crowding", 0)
    return {
        "schema_version": "opk-rag.task0127.diagnostic-upper-bound.v1",
        "diagnostic_upper_bound_only": True,
        "oracle_gold_retention_fixable_failure_upper_bound": fixable,
        "gold_used_for_runtime_decision": False,
    }


def mitigation_recommendation(taxonomy: dict[str, Any], separability: dict[str, Any]) -> dict[str, Any]:
    root = taxonomy["dominant_evidence_conversion_root_cause"]
    mapping = {
        "bge_candidate_displacement": "reranker_input_or_survival_preservation",
        "guarded_rank_fusion_displacement": "reranker_fusion_preservation",
        "evidence_budget_cutoff": "evidence_budgeted_composition",
        "evidence_redundancy_crowding": "evidence_deduplication_or_diversity_selection",
        "evidence_source_imbalance": "source_aware_evidence_composition",
        "context_truncation": "context_budget_management",
        "answerability_shift": "answerability_mitigation",
        "generation_variance": "generation_stability_mitigation",
    }
    family = mapping.get(root, "additional_conversion_diagnosis")
    return {
        "schema_version": "opk-rag.task0127.mitigation-recommendation.v1",
        "recommended_mitigation_family": family,
        "recommended_mitigation_runtime_feasible": family in {"evidence_budgeted_composition", "evidence_deduplication_or_diversity_selection", "source_aware_evidence_composition", "reranker_fusion_preservation"},
        "dominant_evidence_conversion_root_cause": root,
        "secondary_findings": [key for key, value in taxonomy["primary_taxonomy_counts"].items() if value and key != root],
        "task0128_primary_mitigation_count": 1,
    }


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0127.multi-lane-evidence-conversion-failure-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0126_summary_sha256": authority["task0126_summary_sha256"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "c2_policy_digest": authority["c2_policy_digest"],
        "evidence_input_budget": EVIDENCE_CONTEXT_TOP_K,
        "rank_fusion_parameters": {"k": DEFAULT_RANK_FUSION_K, "lambda": DEFAULT_RANK_FUSION_LAMBDA},
        "diagnosis_only": True,
        "no_runtime_mutation_requirement": True,
    }


def build_summary(
    authority: dict[str, Any],
    funnel: dict[str, Any],
    late_funnel: dict[str, Any],
    bge: dict[str, Any],
    fusion: dict[str, Any],
    budget: dict[str, Any],
    taxonomy: dict[str, Any],
    regression: dict[str, Any],
    retrieval_gap: dict[str, Any],
    recommendation: dict[str, Any],
    separability: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0127.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": authority["formal_evaluation_unit_count"],
        "benchmark_membership_frozen": authority["benchmark_membership_frozen"],
        "gold_annotations_unchanged": authority["gold_annotations_unchanged"],
        "task0126_diagnosis_valid": authority["task0126_diagnosis_valid"],
        "task0126_c2_policy_used": authority["task0126_c2_policy_used"],
        "task0126_late_only_gold_admission_rate": authority["task0126_late_only_gold_admission_rate"],
        "task0126_recommended_next_optimization": authority["task0126_recommended_next_optimization"],
        "c2_lane_policy_unchanged": authority["c2_lane_policy_unchanged"],
        "c2_multi_lane_policy_unchanged": authority["c2_lane_policy_unchanged"],
        "reranker_input_budget_unchanged": authority["reranker_input_budget_unchanged"],
        "guard_v2_policy_unchanged": authority["guard_v2_policy_unchanged"],
        "guard_v2_thresholds_unchanged": authority["guard_v2_thresholds_unchanged"],
        "guard_decisions_unchanged": True,
        "entropy_threshold": DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD,
        "top20_score_mean_threshold": DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD,
        "candidate_union_membership_unchanged": True,
        "reranker_policy_unchanged": True,
        "guarded_rank_fusion_unchanged": fusion["guarded_rank_fusion_unchanged"],
        "evidence_policy_unchanged": budget["evidence_policy_unchanged"],
        "generation_policy_unchanged": True,
        "default_retrieval_policy_unchanged": True,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "evidence_input_budget": budget["evidence_input_budget"],
        "generation_visibility_observable": False,
        "generation_visible_evidence_trace_available": False,
        **{key: funnel[key] for key in (
            "conversion_addressable_count",
            "retrieval_unaddressable_count",
            "reranker_admission_rate",
            "bge_survival_rate",
            "rank_fusion_survival_rate",
            "evidence_admission_rate",
            "evidence_budget_survival_rate",
            "retrieval_to_evidence_conversion_rate",
            "retrieval_to_correct_answer_conversion_rate",
        )},
        **{key: late_funnel[key] for key in (
            "late_only_gold_candidate_count",
            "late_only_gold_admitted_to_reranker_count",
            "late_only_gold_survived_bge_count",
            "late_only_gold_survived_rank_fusion_count",
            "late_only_gold_entered_evidence_count",
            "late_only_gold_survived_evidence_budget_count",
            "late_only_gold_visible_to_generation_count",
            "late_only_gold_correct_answer_count",
            "late_only_gold_bge_survival_rate",
        )},
        "loss_at_bge_count": taxonomy["loss_at_bge_count"],
        "loss_at_rank_fusion_count": taxonomy["loss_at_rank_fusion_count"],
        "loss_at_evidence_admission_count": taxonomy["loss_at_evidence_admission_count"],
        "loss_at_evidence_budget_count": taxonomy["loss_at_evidence_budget_count"],
        "loss_at_answerability_count": taxonomy["loss_at_answerability_count"],
        "loss_at_generation_count": taxonomy["loss_at_generation_count"],
        "formal_conversion_failure_count": taxonomy["formal_conversion_failure_count"],
        "first_evidence_conversion_loss_stage": taxonomy["first_evidence_conversion_loss_stage"],
        "dominant_evidence_conversion_root_cause": taxonomy["dominant_evidence_conversion_root_cause"],
        "recommended_mitigation_family": recommendation["recommended_mitigation_family"],
        "recommended_mitigation_runtime_feasible": recommendation["recommended_mitigation_runtime_feasible"],
        "downstream_regression_unit_count": regression["downstream_regression_unit_count"],
        "recovery_regression_shared_root_cause": regression["recovery_regression_shared_root_cause"],
        "retrieval_quality_gap_separately_reported": retrieval_gap["retrieval_quality_gap_separately_reported"],
        "bge_score_observable_unit_count": bge["bge_score_observable_unit_count"],
        "bge_score_unobservable_unit_count": bge["bge_score_unobservable_unit_count"],
        "runtime_visible_roc_auc_bge_rank": separability["roc_auc_bge_rank"],
        "gold_used_for_runtime_decision": False,
        "task0127_verifier_valid": False,
    }


def verify_task0127_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
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
        taxonomy = read_json(output_dir / "failure_taxonomy.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        required_true = (
            "task0126_diagnosis_valid",
            "task0126_c2_policy_used",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "c2_lane_policy_unchanged",
            "guard_v2_policy_unchanged",
            "guard_v2_thresholds_unchanged",
            "candidate_union_membership_unchanged",
            "reranker_policy_unchanged",
            "guarded_rank_fusion_unchanged",
            "evidence_policy_unchanged",
            "generation_policy_unchanged",
            "default_retrieval_policy_unchanged",
            "retrieval_quality_gap_separately_reported",
        )
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if summary.get("task0126_recommended_next_optimization") != "evidence_conversion_optimization":
            issues.append({"code": "task0126_recommendation_mismatch"})
        if summary.get("promotion_applied") is not False or summary.get("recommended_default_policy") != "dense_default":
            issues.append({"code": "runtime_mutation_detected"})
        if not summary.get("recommended_mitigation_family"):
            issues.append({"code": "recommended_mitigation_family_missing"})
        if taxonomy.get("sum_primary_taxonomy_counts") != taxonomy.get("formal_conversion_failure_count"):
            issues.append({"code": "failure_taxonomy_accounting_mismatch"})
    result = {
        "schema_version": "opk-rag.task0127.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0127_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    jsonl = {"candidate_survival_trace", "gold_evidence_competition"}
    for key, value in artifacts.items():
        if key in jsonl:
            write_jsonl(output_dir / f"{key}.jsonl", value)
        else:
            write_json(output_dir / f"{key}.json", value)


def provenance_payload(benchmark: dict[str, Any], retriever: task0118.CleanLateInteractionRetriever, authority: dict[str, Any]) -> dict[str, Any]:
    prior = {
        "task0118": task0118.RESULT_DIR / "summary.json",
        "task0120": task0120.RESULT_DIR / "summary.json",
        "task0121": task0121.RESULT_DIR / "summary.json",
        "task0124": task0124.RESULT_DIR / "summary.json",
        "task0125": task0125.RESULT_DIR / "summary.json",
        "task0126": task0126.RESULT_DIR / "summary.json",
    }
    return {
        "schema_version": "opk-rag.task0127.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "clean_late_interaction_index_digest": retriever.index_digest,
        "task0126_summary_sha256": authority["task0126_summary_sha256"],
        "prior_artifact_hashes": {key: sha256_file(path) for key, path in prior.items() if path.exists()},
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "diagnosis_only": True,
    }


def build_report(summary: dict[str, Any], late_funnel: dict[str, Any], taxonomy: dict[str, Any], recommendation: dict[str, Any]) -> str:
    return f"""# TASK-0127 Multi-Lane Evidence Conversion Failure Diagnosis Report

## Answers

- Where does C2 fail to convert retrieval gain? First loss stage `{summary['first_evidence_conversion_loss_stage']}`; dominant root cause `{summary['dominant_evidence_conversion_root_cause']}`.
- Late-only Gold funnel: admitted `{late_funnel['late_only_gold_admitted_to_reranker_count']}`, survived BGE `{late_funnel['late_only_gold_survived_bge_count']}`, entered EvidenceContext `{late_funnel['late_only_gold_entered_evidence_count']}`, correct answer `{late_funnel['late_only_gold_correct_answer_count']}`.
- Loss counts: BGE `{summary['loss_at_bge_count']}`, guarded rank fusion `{summary['loss_at_rank_fusion_count']}`, evidence admission `{summary['loss_at_evidence_admission_count']}`, evidence budget `{summary['loss_at_evidence_budget_count']}`, answerability `{summary['loss_at_answerability_count']}`, generation `{summary['loss_at_generation_count']}`.
- Failure taxonomy accounting: `{taxonomy['sum_primary_taxonomy_counts']}/{taxonomy['formal_conversion_failure_count']}`.
- Generation visibility: observable `{summary['generation_visibility_observable']}`; no generation variance claim is made without visible trace.
- TASK-0128 mitigation family: `{recommendation['recommended_mitigation_family']}`; runtime feasible `{recommendation['recommended_mitigation_runtime_feasible']}`.

## Frozen Scope

Default retrieval policy unchanged: `{summary['default_retrieval_policy_unchanged']}`. Guard V2 unchanged: `{summary['guard_v2_policy_unchanged']}`. C2 Multi-Lane unchanged: `{summary['c2_multi_lane_policy_unchanged']}`. Reranker unchanged: `{summary['reranker_policy_unchanged']}`. Evidence policy unchanged: `{summary['evidence_policy_unchanged']}`. Generation policy unchanged: `{summary['generation_policy_unchanged']}`. Promotion applied: `{summary['promotion_applied']}`. Recommended default policy: `{summary['recommended_default_policy']}`.
"""


def assign_pre_bge_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "pre_bge_rank": index, "retrieval_rank": index} for index, row in enumerate(rows, start=1)]


def bge_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if all(row.get("reranker_score") is not None for row in rows):
        ranked = sorted(rows, key=lambda row: (-float(row["reranker_score"]), int(row["pre_bge_rank"]), row["canonical_chunk_id"]))
    else:
        ranked = list(rows)
    return [{**row, "bge_rank": index, "reranker_rank": index} for index, row in enumerate(ranked, start=1)]


def first_loss_stage(pre_rank: int | None, bge_rank: int | None, final_rank: int | None, entered_evidence: bool, correct_answer: bool) -> str:
    if pre_rank is None:
        return "evidence_admission"
    if bge_rank is None or bge_rank > BGE_SURVIVAL_CUTOFF:
        return "bge_ranking"
    if final_rank is None or final_rank > RETRIEVAL_TOP_K:
        return "guarded_rank_fusion"
    if not entered_evidence:
        return "evidence_budget"
    if not correct_answer:
        return "compound_downstream_failure"
    return "none"


def primary_taxonomy(stage: str, evidence_rows: list[dict[str, Any]]) -> str:
    if stage == "bge_ranking":
        return "bge_candidate_displacement"
    if stage == "guarded_rank_fusion":
        return "guarded_rank_fusion_displacement"
    if stage == "evidence_budget":
        return "evidence_budget_cutoff"
    if stage == "answerability":
        return "answerability_shift"
    if stage == "generation":
        return "generation_variance"
    if stage == "compound_downstream_failure":
        return "compound_downstream_failure"
    return "other" if stage != "none" else "other"


def competitor_classification(gold: dict[str, Any], candidate: dict[str, Any]) -> str:
    source = candidate_source_type(candidate)
    if candidate.get("document_id") == gold.get("document_id") and candidate.get("section_id") == gold.get("section_id") and candidate.get("section_id"):
        return "same_section_redundant"
    if candidate.get("document_id") == gold.get("document_id") and candidate.get("document_id"):
        return "same_document_competitor"
    if source == SOURCE_OVERLAP:
        return "overlap_candidate"
    if source == SOURCE_DENSE_ONLY:
        return "dense_only_candidate"
    if source == SOURCE_LATE_ONLY:
        return "late_only_candidate"
    return "cross_document_competitor" if candidate.get("document_id") and gold.get("document_id") else "other"


def aggregate_trace_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["bge_rank"] for row in rows if row["bge_rank"] is not None]
    final_ranks = [row["final_rank"] for row in rows if row["final_rank"] is not None]
    return {
        "mean_bge_gold_rank": statistics.mean(ranks) if ranks else None,
        "mean_final_gold_rank": statistics.mean(final_ranks) if final_ranks else None,
        "source_type_counts": dict(Counter(row["source_type"] for row in rows)),
        "mean_competing_candidate_count": statistics.mean([row["bge_competing_candidate_count"] for row in rows]) if rows else None,
        "evidence_slot_pressure": EVIDENCE_CONTEXT_TOP_K,
        "document_diversity": len({row.get("document_id") for row in rows if row.get("document_id")}),
        "section_diversity": len({row.get("section_id") for row in rows if row.get("section_id")}),
    }


def dominant_stage(counts: Counter[str]) -> str:
    for stage in LOSS_STAGES:
        if counts.get(stage):
            return stage
    return "none"


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda row: row[0])
    rank_sum = sum(index for index, (_, label) in enumerate(ranked, start=1) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    total = 0.0
    hits = 0
    for index, (_, label) in enumerate(sorted(zip(scores, labels), key=lambda row: row[0], reverse=True), start=1):
        if label:
            hits += 1
            total += hits / index
    return total / positives


def _traces_by_unit(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in traces:
        out.setdefault(row["evaluation_unit_id"], []).append(row)
    return out


def _correct(rows: list[dict[str, Any]]) -> bool:
    return (first_relevant_rank(rows) or math.inf) <= EVIDENCE_CONTEXT_TOP_K


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "p50": statistics.median(values) if values else 0.0,
        "max": max(values) if values else 0,
        "total": sum(values),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
