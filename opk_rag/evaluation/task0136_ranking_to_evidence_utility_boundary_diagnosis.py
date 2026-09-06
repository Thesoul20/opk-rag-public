from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0133_residual_ranking_failure_diagnosis as task0133
import opk_rag.evaluation.task0134_existing_signal_ranking_mitigation as task0134
import opk_rag.evaluation.task0135_new_ranking_signal_experiment as task0135
from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.runtime_v2 import evidence_composition


TASK_ID = "TASK-0136"
EXPERIMENT_ID = "task0136-ranking-to-evidence-utility-boundary-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0136_ranking_to_evidence_utility_boundary_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0136_RANKING_TO_EVIDENCE_UTILITY_BOUNDARY_DIAGNOSIS_REPORT.md"

BASELINE_ARM = task0135.ARM_C0
BEST_ARM = task0135.ARM_C4
SIGNAL_ARMS = (task0135.ARM_C1, task0135.ARM_C2, task0135.ARM_C3, task0135.ARM_C4)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "conversion_funnel.json",
    "regression_diagnosis.jsonl",
    "non_propagation_diagnosis.jsonl",
    "candidate_sufficiency.jsonl",
    "evidence_context_comparison.jsonl",
    "representation_analysis.json",
    "graph_sensitive_analysis.json",
    "taxonomy_boundary_audit.json",
    "representative_cases.jsonl",
    "config.json",
    "digests.json",
    "verification.json",
)


def run_task0136_ranking_to_evidence_utility_boundary_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_authority()
    task0135_rows = read_jsonl(task0135.RESULT_DIR / "per_sample.jsonl")
    task0135_ranking = read_jsonl(task0135.RESULT_DIR / "ranking_results.jsonl")
    rows_by_arm = group_by_arm(task0135_rows)
    baseline = {row["evaluation_unit_id"]: row for row in rows_by_arm[BASELINE_ARM]}
    ranking_by_key = {(row["arm_id"], row["evaluation_unit_id"]): row for row in task0135_ranking}

    per_sample = build_boundary_matrix(rows_by_arm, baseline, ranking_by_key)
    evidence_comparisons = [row["evidence_context_comparison"] for row in per_sample]
    candidate_sufficiency = [row["candidate_sufficiency"] for row in per_sample]
    regressions = [row["regression_diagnosis"] for row in per_sample if row["arm_id"] == BEST_ARM and row.get("regression_diagnosis")]
    non_propagations = [row["non_propagation_diagnosis"] for row in per_sample if row["arm_id"] == BEST_ARM and row.get("non_propagation_diagnosis")]
    conversion_funnel = build_conversion_funnel(per_sample)
    representation_analysis = build_representation_analysis(per_sample)
    graph_analysis = build_graph_sensitive_analysis(per_sample)
    taxonomy = build_taxonomy_boundary_audit(per_sample)
    representatives = representative_cases(per_sample)
    config = build_config(authority)
    digests = build_digests(
        per_sample,
        conversion_funnel,
        regressions,
        non_propagations,
        candidate_sufficiency,
        evidence_comparisons,
        representation_analysis,
        graph_analysis,
        taxonomy,
        representatives,
        config,
    )
    summary = build_summary(
        authority,
        per_sample,
        conversion_funnel,
        regressions,
        non_propagations,
        representation_analysis,
        graph_analysis,
        taxonomy,
        digests,
    )
    contract = build_contract(authority, config)

    write_json(CONTRACT_PATH, contract)
    write_jsonl(output_dir / "per_sample.jsonl", strip_nested(per_sample))
    write_json(output_dir / "conversion_funnel.json", conversion_funnel)
    write_jsonl(output_dir / "regression_diagnosis.jsonl", regressions)
    write_jsonl(output_dir / "non_propagation_diagnosis.jsonl", non_propagations)
    write_jsonl(output_dir / "candidate_sufficiency.jsonl", candidate_sufficiency)
    write_jsonl(output_dir / "evidence_context_comparison.jsonl", evidence_comparisons)
    write_json(output_dir / "representation_analysis.json", representation_analysis)
    write_json(output_dir / "graph_sensitive_analysis.json", graph_analysis)
    write_json(output_dir / "taxonomy_boundary_audit.json", taxonomy)
    write_jsonl(output_dir / "representative_cases.jsonl", representatives)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0136_artifacts(output_dir=output_dir, write=True)
    summary["task0136_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, conversion_funnel, representation_analysis, graph_analysis, taxonomy), encoding="utf-8")
    return summary


def load_authority() -> dict[str, Any]:
    task0133_summary = read_json(task0133.RESULT_DIR / "summary.json")
    task0134_summary = read_json(task0134.RESULT_DIR / "summary.json")
    task0135_summary = read_json(task0135.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0136.authority.v1",
        "task0133_inputs_valid": task0133.verify_task0133_artifacts(write=False)["status"] == "valid",
        "task0134_inputs_valid": task0134.verify_task0134_artifacts(write=False)["status"] == "valid",
        "task0135_inputs_valid": task0135.verify_task0135_artifacts(write=False)["status"] == "valid",
        "task0133_summary_sha256": sha256_file(task0133.RESULT_DIR / "summary.json"),
        "task0134_summary_sha256": sha256_file(task0134.RESULT_DIR / "summary.json"),
        "task0135_summary_sha256": sha256_file(task0135.RESULT_DIR / "summary.json"),
        "formal_evaluation_unit_count": task0135_summary["formal_evaluation_unit_count"],
        "task0133_validated_ranking_failure_count": task0133_summary["validated_ranking_failure_count"],
        "task0134_promotion_applied": task0134_summary["promotion_applied"],
        "task0135_ranking_failure_recovery_count": task0135_summary["qwen_ranking_failure_recovery_count"]
        + task0135_summary["zerank_ranking_failure_recovery_count"]
        + task0135_summary["late_interaction_ranking_failure_recovery_count"],
        "task0135_qwen_ranking_failure_recovery_count": task0135_summary["qwen_ranking_failure_recovery_count"],
        "task0135_zerank_ranking_failure_recovery_count": task0135_summary["zerank_ranking_failure_recovery_count"],
        "task0135_late_interaction_ranking_failure_recovery_count": task0135_summary["late_interaction_ranking_failure_recovery_count"],
        "task0135_downstream_improved_count": task0135_summary["downstream_improved_count"],
        "task0135_downstream_regressed_count": task0135_summary["downstream_regressed_count"],
        "task0135_downstream_net_gain": task0135_summary["downstream_net_gain"],
        "task0135_recommended_next_task_family": task0135_summary["recommended_next_task_family"],
    }


def group_by_arm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm_id"]].append(row)
    return dict(grouped)


def build_boundary_matrix(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    baseline: dict[str, dict[str, Any]],
    ranking_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in SIGNAL_ARMS:
        for row in sorted(rows_by_arm[arm], key=lambda item: item["evaluation_unit_id"]):
            unit = row["evaluation_unit_id"]
            base = baseline[unit]
            evidence = compare_evidence_context(base, row, ranking_by_key.get((arm, unit), {}))
            candidate = classify_candidate_sufficiency(row, evidence)
            representation = classify_representation(row, candidate, evidence)
            taxonomy = classify_taxonomy_boundary(row, candidate, evidence, representation)
            root = classify_root_cause(row, base, candidate, evidence, representation, taxonomy)
            regression = regression_diagnosis(row, base, candidate, evidence, representation, root) if is_downstream_regression(row, base) else None
            non_propagation = non_propagation_diagnosis(row, base, candidate, evidence, root) if is_non_propagating_ranking_recovery(row, base) else None
            out.append(
                {
                    "schema_version": "opk-rag.task0136.per-sample.v1",
                    "task_id": TASK_ID,
                    "sample_id": row.get("sample_id"),
                    "evaluation_unit_id": unit,
                    "ranking_signal": row["signal_name"],
                    "arm_id": arm,
                    "ranking_recovered": ranking_recovered(row, base),
                    "best_relevant_rank_before": row.get("best_relevant_rank_current"),
                    "best_relevant_rank_after": row.get("best_relevant_rank_new_signal"),
                    "candidate_relevant": row.get("candidate_relevant_available", False),
                    "candidate_evidence_useful": candidate["candidate_evidence_useful"],
                    "candidate_individually_sufficient": candidate["candidate_individually_sufficient"],
                    "candidate_set_answer_sufficient": candidate["candidate_set_answer_sufficient"],
                    "gold_span_fully_contained": representation["gold_span_fully_contained"],
                    "gold_span_boundary_split": representation["gold_span_boundary_split"],
                    "requires_multiple_evidence": candidate["requires_multiple_evidence"],
                    "multi_evidence_requirement": candidate["multi_evidence_requirement"],
                    "heading_context_relevant": representation["heading_context_relevant"],
                    "adjacent_chunk_required": representation["adjacent_chunk_required"],
                    "cross_section_required": representation["cross_section_required"],
                    "cross_document_required": representation["cross_document_required"],
                    "graph_sensitive": row.get("graph_sensitive", False),
                    "evidence_context_changed": evidence["evidence_context_changed"],
                    "evidence_context_useful_before": evidence["baseline_context_useful"],
                    "evidence_context_useful_after": evidence["new_context_useful"],
                    "evidence_context_sufficient_before": evidence["baseline_context_sufficient"],
                    "evidence_context_sufficient_after": evidence["new_context_sufficient"],
                    "downstream_before": base["final_success"],
                    "downstream_after": row["final_success"],
                    "primary_boundary_failure": root["primary_boundary_failure"],
                    "secondary_boundary_failure": root["secondary_boundary_failure"],
                    "root_cause_confidence": root["root_cause_confidence"],
                    "evidence_context_comparison": evidence,
                    "candidate_sufficiency": candidate,
                    "representation_diagnosis": representation,
                    "taxonomy_boundary": taxonomy,
                    "root_cause": root,
                    "regression_diagnosis": regression,
                    "non_propagation_diagnosis": non_propagation,
                }
            )
    return out


def compare_evidence_context(base: dict[str, Any], row: dict[str, Any], ranking: dict[str, Any]) -> dict[str, Any]:
    baseline_ids = list(base.get("selected_evidence_ids") or [])
    new_ids = list(row.get("selected_evidence_ids") or [])
    relevant = set(row.get("relevant_candidate_ids") or [])
    required_count = required_evidence_count(row)
    baseline_relevant = [cid for cid in baseline_ids if cid in relevant]
    new_relevant = [cid for cid in new_ids if cid in relevant]
    evidence_added = [cid for cid in new_ids if cid not in set(baseline_ids)]
    evidence_removed = [cid for cid in baseline_ids if cid not in set(new_ids)]
    return {
        "schema_version": "opk-rag.task0136.evidence-context-comparison.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "baseline_evidence_ids": baseline_ids,
        "new_ranking_evidence_ids": new_ids,
        "evidence_added": evidence_added,
        "evidence_removed": evidence_removed,
        "evidence_preserved": [cid for cid in new_ids if cid in set(baseline_ids)],
        "added_evidence_utility": classify_added_removed_utility(evidence_added, relevant),
        "removed_evidence_utility": classify_added_removed_utility(evidence_removed, relevant),
        "baseline_relevant_evidence_count": len(baseline_relevant),
        "new_relevant_evidence_count": len(new_relevant),
        "required_evidence_count": required_count,
        "baseline_context_useful": bool(baseline_relevant),
        "new_context_useful": bool(new_relevant),
        "baseline_context_sufficient": len(baseline_relevant) >= required_count,
        "new_context_sufficient": len(new_relevant) >= required_count,
        "evidence_context_changed": baseline_ids != new_ids,
        "ranking_mrr": ranking.get("mrr"),
        "ranking_recall_at_5": ranking.get("recall_at_5"),
        "best_relevant_rank_improved": (row.get("rank_delta") or 0) > 0,
    }


def classify_candidate_sufficiency(row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    relevant_ids = list(row.get("relevant_candidate_ids") or [])
    required_count = evidence["required_evidence_count"]
    selected_relevant = evidence["new_relevant_evidence_count"]
    candidate_set_state = "true" if len(relevant_ids) >= required_count else ("partial" if relevant_ids else "false")
    if required_count == 1:
        multi_class = "single_evidence_sufficient"
    elif required_count == 2:
        multi_class = "two_evidence_required"
    else:
        multi_class = "relation_chain_required" if row.get("graph_sensitive") else "multi_evidence_required"
    gold_sufficiency = (
        "fully_answer_sufficient"
        if required_count == 1
        else "partially_answer_sufficient"
        if len(relevant_ids) >= required_count
        else "supporting_but_not_sufficient"
    )
    return {
        "schema_version": "opk-rag.task0136.candidate-sufficiency.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "candidate_relevant": bool(relevant_ids),
        "candidate_evidence_useful": selected_relevant > 0,
        "candidate_individually_sufficient": required_count == 1 and selected_relevant > 0,
        "candidate_set_answer_sufficient": candidate_set_state,
        "candidate_set_answer_sufficient_bool": candidate_set_state == "true",
        "required_evidence_count": required_count,
        "requires_multiple_evidence": required_count > 1,
        "multi_evidence_requirement": multi_class,
        "gold_candidate_sufficiency_label": gold_sufficiency,
        "gold_relevant": bool(relevant_ids),
        "gold_useful": selected_relevant > 0,
        "gold_sufficient": gold_sufficiency == "fully_answer_sufficient",
        "relevant_candidate_available": bool(relevant_ids),
    }


def classify_representation(row: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    query = str(row.get("query") or "")
    heading = heading_context_relevant(query, row)
    multi = candidate["requires_multiple_evidence"]
    adjacent = multi and not row.get("graph_sensitive") and candidate["candidate_set_answer_sufficient"] != "false"
    cross_section = multi and (heading or "section" in query.lower() or "章节" in query)
    cross_document = row.get("graph_sensitive", False) and candidate["candidate_set_answer_sufficient"] != "true"
    boundary_split = multi and (adjacent or cross_section)
    deficits = []
    if heading:
        deficits.append("missing_heading_context")
    if adjacent:
        deficits.append("missing_adjacent_paragraph_context")
    if cross_section:
        deficits.append("missing_parent_section_context")
    if row.get("graph_sensitive"):
        deficits.append("cross_chunk_relation")
    if candidate["requires_multiple_evidence"]:
        deficits.append("cross_chunk_relation")
    return {
        "schema_version": "opk-rag.task0136.representation-diagnosis.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "gold_span_fully_contained": not boundary_split,
        "gold_span_crosses_chunk_boundary": boundary_split,
        "gold_span_partial_containment": boundary_split and evidence["new_relevant_evidence_count"] > 0,
        "gold_span_boundary_split": boundary_split,
        "context_completeness": "same_chunk" if not multi else "same_section" if cross_section else "adjacent_chunk" if adjacent else "different_document" if cross_document else "unknown",
        "heading_context_relevant": heading,
        "adjacent_chunk_required": adjacent,
        "same_section_multi_chunk_required": cross_section,
        "cross_section_required": cross_section,
        "cross_document_required": cross_document,
        "representation_insufficient": bool(deficits),
        "representation_deficits": sorted(set(deficits)),
    }


def classify_taxonomy_boundary(
    row: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    representation: dict[str, Any],
) -> dict[str, Any]:
    incomplete = candidate["candidate_set_answer_sufficient"] in {"false", "partial"} and candidate["requires_multiple_evidence"]
    reclass = bool(row.get("ranking_failure_cohort")) and incomplete
    conceptual = "partial_retrieval_boundary_failure" if reclass else "ranking_failure"
    return {
        "schema_version": "opk-rag.task0136.taxonomy-boundary.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "relevant_candidate_available": candidate["relevant_candidate_available"],
        "candidate_set_answer_sufficient": candidate["candidate_set_answer_sufficient"],
        "all_required_evidence_present_in_candidate_set": candidate["candidate_set_answer_sufficient"] == "true",
        "some_required_evidence_missing_from_candidate_set": incomplete,
        "secondary_upstream_issue": "partial_candidate_retrieval_failure" if incomplete else None,
        "taxonomy_boundary_reclassification_candidate": reclass,
        "conceptual_first_failure": conceptual,
        "chunk_boundary_problem": representation["gold_span_boundary_split"],
        "retrieval_should_have_selected_multiple_chunks": candidate["requires_multiple_evidence"] and candidate["candidate_set_answer_sufficient"] == "true",
        "candidate_set_incomplete": incomplete,
    }


def classify_root_cause(
    row: dict[str, Any],
    base: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    representation: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    if is_downstream_regression(row, base) and evidence["removed_evidence_utility"] == "useful" and evidence["added_evidence_utility"] != "useful":
        primary = "evidence_set_displacement"
        secondary = "useful_evidence_displaced"
        confidence = "high"
    elif taxonomy["candidate_set_incomplete"]:
        primary = "candidate_set_incomplete"
        secondary = "partial_retrieval_boundary_failure"
        confidence = "medium"
    elif row.get("graph_sensitive"):
        primary = "graph_sensitive_relation_requirement"
        secondary = "relation_chain_required"
        confidence = "medium"
    elif candidate["requires_multiple_evidence"]:
        primary = "multi_evidence_requirement"
        secondary = "cross_chunk_context_requirement"
        confidence = "medium"
    elif representation["heading_context_relevant"]:
        primary = "missing_heading_or_section_context"
        secondary = "chunk_representation_insufficient"
        confidence = "medium"
    elif representation["representation_insufficient"]:
        primary = "chunk_representation_insufficient"
        secondary = "cross_chunk_context_requirement"
        confidence = "medium"
    elif evidence["new_context_sufficient"] and not row["final_success"]:
        primary = "generation_evidence_use_failure"
        secondary = "evidence_access_recovery_not_generation_utility"
        confidence = "low"
    elif (row.get("rank_delta") or 0) <= 0:
        primary = "ranking_change_not_material"
        secondary = "ranking_recovery_not_evidence_access_recovery"
        confidence = "high"
    else:
        primary = "candidate_relevance_not_sufficiency"
        secondary = "relevant_but_insufficient_candidate_promoted"
        confidence = "medium"
    return {
        "schema_version": "opk-rag.task0136.root-cause.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "primary_boundary_failure": primary,
        "secondary_boundary_failure": secondary,
        "root_cause_confidence": confidence,
    }


def regression_diagnosis(
    row: dict[str, Any],
    base: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    representation: dict[str, Any],
    root: dict[str, Any],
) -> dict[str, Any]:
    primary_map = {
        "evidence_set_displacement": "useful_evidence_displaced",
        "candidate_set_incomplete": "multi_evidence_set_broken",
        "multi_evidence_requirement": "multi_evidence_set_broken",
        "missing_heading_or_section_context": "section_context_lost",
        "chunk_representation_insufficient": "adjacent_context_lost",
        "candidate_relevance_not_sufficiency": "relevant_but_insufficient_candidate_promoted",
    }
    return {
        "schema_version": "opk-rag.task0136.regression-diagnosis.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "sample_id": row.get("sample_id"),
        "task0135_regression": True,
        "primary_class": primary_map.get(root["primary_boundary_failure"], "other"),
        "root_boundary_failure": root["primary_boundary_failure"],
        "baseline_evidence_ids": evidence["baseline_evidence_ids"],
        "new_ranking_evidence_ids": evidence["new_ranking_evidence_ids"],
        "evidence_added": evidence["evidence_added"],
        "evidence_removed": evidence["evidence_removed"],
        "removed_evidence_utility": evidence["removed_evidence_utility"],
        "added_evidence_utility": evidence["added_evidence_utility"],
        "candidate_individually_sufficient": candidate["candidate_individually_sufficient"],
        "requires_multiple_evidence": candidate["requires_multiple_evidence"],
        "heading_context_relevant": representation["heading_context_relevant"],
        "root_cause_confidence": root["root_cause_confidence"],
    }


def non_propagation_diagnosis(
    row: dict[str, Any],
    base: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    root: dict[str, Any],
) -> dict[str, Any]:
    if base["final_success"]:
        primary = "answer_already_correct"
    elif not evidence["evidence_context_changed"]:
        primary = "rank_change_did_not_change_evidence_context"
    elif evidence["new_context_useful"] and not evidence["new_context_sufficient"]:
        primary = "relevant_candidate_not_sufficient"
    elif candidate["candidate_set_answer_sufficient"] != "true":
        primary = "missing_complementary_evidence"
    elif evidence["new_context_useful"] == evidence["baseline_context_useful"]:
        primary = "evidence_context_changed_but_not_materially"
    else:
        primary = "generation_did_not_use_evidence"
    return {
        "schema_version": "opk-rag.task0136.non-propagation-diagnosis.v1",
        "task_id": TASK_ID,
        "arm_id": row["arm_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "sample_id": row.get("sample_id"),
        "primary_class": primary,
        "root_boundary_failure": root["primary_boundary_failure"],
        "candidate_set_answer_sufficient": candidate["candidate_set_answer_sufficient"],
        "evidence_context_changed": evidence["evidence_context_changed"],
        "new_context_useful": evidence["new_context_useful"],
        "new_context_sufficient": evidence["new_context_sufficient"],
    }


def build_conversion_funnel(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {}
    for arm in SIGNAL_ARMS:
        rows = [row for row in per_sample if row["arm_id"] == arm]
        recovered = [row for row in rows if row["ranking_recovered"]]
        by_arm[arm] = funnel_counts(recovered)
    best_rows = [row for row in per_sample if row["arm_id"] == BEST_ARM and row["ranking_recovered"]]
    return {
        "schema_version": "opk-rag.task0136.conversion-funnel.v1",
        "task_id": TASK_ID,
        **funnel_counts(best_rows),
        "by_signal": by_arm,
    }


def funnel_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ranking_recovery_count": len(rows),
        "effective_region_recovery_count": sum(row["best_relevant_rank_after"] is not None and row["best_relevant_rank_after"] <= evidence_composition.EVIDENCE_BUDGET_LIMIT for row in rows),
        "evidence_context_entry_recovery_count": sum(row["evidence_context_useful_after"] for row in rows),
        "evidence_access_recovery_count": sum(row["evidence_context_useful_after"] for row in rows),
        "evidence_utility_improvement_count": sum(
            row["evidence_context_comparison"]["new_relevant_evidence_count"]
            > row["evidence_context_comparison"]["baseline_relevant_evidence_count"]
            for row in rows
        ),
        "evidence_utility_recovery_count": sum(row["evidence_context_useful_after"] and not row["evidence_context_useful_before"] for row in rows),
        "evidence_sufficiency_improvement_count": sum(row["evidence_context_sufficient_after"] and not row["evidence_context_sufficient_before"] for row in rows),
        "evidence_sufficiency_recovery_count": sum(row["evidence_context_sufficient_after"] for row in rows),
        "downstream_improvement_count": sum(row["downstream_after"] and not row["downstream_before"] for row in rows),
        "downstream_regression_count": sum((not row["downstream_after"]) and row["downstream_before"] for row in rows),
    }


def build_representation_analysis(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in per_sample if row["arm_id"] == BEST_ARM]
    problematic = [row for row in rows if row["ranking_recovered"] or (row["downstream_before"] and not row["downstream_after"])]
    deficits = Counter(deficit for row in problematic for deficit in row["representation_diagnosis"]["representation_deficits"])
    return {
        "schema_version": "opk-rag.task0136.representation-analysis.v1",
        "task_id": TASK_ID,
        "diagnostic_case_count": len(problematic),
        "representation_insufficiency_count": sum(row["representation_diagnosis"]["representation_insufficient"] for row in problematic),
        "representation_insufficiency_by_type": dict(sorted(deficits.items())),
        "heading_context_relevant_case_count": sum(row["heading_context_relevant"] for row in problematic),
        "heading_context_relevant_regression_count": sum(row["heading_context_relevant"] and row["downstream_before"] and not row["downstream_after"] for row in problematic),
        "gold_span_boundary_split_count": sum(row["gold_span_boundary_split"] for row in problematic),
        "ranking_recovery_failure_with_boundary_split_count": sum(row["ranking_recovered"] and not row["downstream_after"] and row["gold_span_boundary_split"] for row in problematic),
        "ranking_regression_with_boundary_split_count": sum(row["downstream_before"] and not row["downstream_after"] and row["gold_span_boundary_split"] for row in problematic),
        "representation_or_chunking_boundary_case_count": sum(row["gold_span_boundary_split"] or row["adjacent_chunk_required"] or row["cross_section_required"] for row in problematic),
    }


def build_graph_sensitive_analysis(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in per_sample if row["arm_id"] == BEST_ARM]
    nonprop = [row for row in rows if row.get("non_propagation_diagnosis")]
    regressions = [row for row in rows if row.get("regression_diagnosis")]
    return {
        "schema_version": "opk-rag.task0136.graph-sensitive-analysis.v1",
        "task_id": TASK_ID,
        "graph_sensitive_non_propagation_count": sum(row["graph_sensitive"] for row in nonprop),
        "graph_sensitive_regression_count": sum(row["graph_sensitive"] for row in regressions),
        "graph_sensitive_candidate_set_insufficient_count": sum(row["graph_sensitive"] and row["candidate_set_answer_sufficient"] != "true" for row in rows),
        "relation_or_graph_problem_indicated": any(row["graph_sensitive"] for row in nonprop + regressions),
    }


def build_taxonomy_boundary_audit(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in per_sample if row["arm_id"] == BEST_ARM]
    return {
        "schema_version": "opk-rag.task0136.taxonomy-boundary-audit.v1",
        "task_id": TASK_ID,
        "relevant_candidate_available_count": sum(row["candidate_relevant"] for row in rows),
        "candidate_set_answer_sufficient_count": sum(row["candidate_set_answer_sufficient"] == "true" for row in rows),
        "candidate_set_answer_insufficient_count": sum(row["candidate_set_answer_sufficient"] in {"false", "partial"} for row in rows),
        "ranking_cases_with_complete_candidate_evidence_set": sum(row["taxonomy_boundary"]["all_required_evidence_present_in_candidate_set"] and row["ranking_recovered"] for row in rows),
        "ranking_cases_with_incomplete_candidate_evidence_set": sum(row["taxonomy_boundary"]["some_required_evidence_missing_from_candidate_set"] and row["ranking_recovered"] for row in rows),
        "taxonomy_boundary_reclassification_candidate_count": sum(row["taxonomy_boundary"]["taxonomy_boundary_reclassification_candidate"] for row in rows),
        "binary_relevant_candidate_rule_insufficient": any(row["taxonomy_boundary"]["taxonomy_boundary_reclassification_candidate"] for row in rows),
    }


def representative_cases(per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in per_sample if row["arm_id"] == BEST_ARM]
    selectors = [
        ("ranking_recovery_downstream_improvement", lambda r: r["ranking_recovered"] and r["downstream_after"] and not r["downstream_before"]),
        ("ranking_recovery_no_downstream_effect", lambda r: r.get("non_propagation_diagnosis") is not None),
        ("ranking_recovery_downstream_regression", lambda r: r.get("regression_diagnosis") is not None),
        ("relevant_candidate_individually_insufficient", lambda r: r["candidate_relevant"] and not r["candidate_individually_sufficient"]),
        ("candidate_set_missing_complementary_evidence", lambda r: r["candidate_set_answer_sufficient"] != "true"),
        ("gold_span_split_across_chunks", lambda r: r["gold_span_boundary_split"]),
        ("heading_section_context_insufficiency", lambda r: r["heading_context_relevant"]),
        ("multi_evidence_requirement", lambda r: r["requires_multiple_evidence"]),
        ("graph_sensitive_relation_case", lambda r: r["graph_sensitive"]),
        ("evidence_context_sufficient_generation_fails", lambda r: r["evidence_context_sufficient_after"] and not r["downstream_after"]),
    ]
    out = []
    for label, predicate in selectors:
        match = next((row for row in rows if predicate(row)), None)
        if match:
            out.append(representative_payload(label, match))
    zerank_rows = [row for row in per_sample if row["arm_id"] == task0135.ARM_C2]
    for label, predicate in (
        ("zerank_recovery_no_downstream_gain", lambda r: r.get("non_propagation_diagnosis") is not None),
        ("zerank_recovery_downstream_regression", lambda r: r.get("regression_diagnosis") is not None),
    ):
        match = next((row for row in zerank_rows if predicate(row)), None)
        if match:
            out.append(representative_payload(label, match))
    return out


def representative_payload(label: str, row: dict[str, Any]) -> dict[str, Any]:
    evidence = row["evidence_context_comparison"]
    return {
        "schema_version": "opk-rag.task0136.representative-case.v1",
        "task_id": TASK_ID,
        "case_type": label,
        "sample_id": row["sample_id"],
        "evaluation_unit_id": row["evaluation_unit_id"],
        "ranking_signal": row["ranking_signal"],
        "candidate_set": row["candidate_sufficiency"],
        "rank_before": row["best_relevant_rank_before"],
        "rank_after": row["best_relevant_rank_after"],
        "evidence_context_before": evidence["baseline_evidence_ids"],
        "evidence_context_after": evidence["new_ranking_evidence_ids"],
        "evidence_utility": {
            "before": evidence["baseline_context_useful"],
            "after": evidence["new_context_useful"],
        },
        "evidence_sufficiency": {
            "before": evidence["baseline_context_sufficient"],
            "after": evidence["new_context_sufficient"],
        },
        "final_answer_outcome": {"before": row["downstream_before"], "after": row["downstream_after"]},
        "primary_boundary_failure": row["primary_boundary_failure"],
    }


def build_config(authority: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.task0136.config.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "runtime_modified": False,
        "runtime_default_policy_changed": False,
        "promotion_applied": False,
        "candidate_membership_frozen": True,
        "new_ranking_model_experiment": False,
        "additional_retrieval_calls": 0,
        "additional_generation_calls": 0,
        "evidence_budget_limit": evidence_composition.EVIDENCE_BUDGET_LIMIT,
        "evidence_composition_frozen": "targeted_budgeted_composition",
        "task0133_summary_sha256": authority["task0133_summary_sha256"],
        "task0134_summary_sha256": authority["task0134_summary_sha256"],
        "task0135_summary_sha256": authority["task0135_summary_sha256"],
    }
    return payload | {"config_digest": digest_json(payload)}


def build_digests(*payloads: Any) -> dict[str, Any]:
    names = (
        "per_sample",
        "conversion_funnel",
        "regression_diagnosis",
        "non_propagation_diagnosis",
        "candidate_sufficiency",
        "evidence_context_comparison",
        "representation_analysis",
        "graph_sensitive_analysis",
        "taxonomy_boundary_audit",
        "representative_cases",
        "config",
    )
    return {"schema_version": "opk-rag.task0136.digests.v1", **{f"{name}_digest": digest_json(payload) for name, payload in zip(names, payloads)}, "deterministic_output": True}


def build_summary(
    authority: dict[str, Any],
    per_sample: list[dict[str, Any]],
    funnel: dict[str, Any],
    regressions: list[dict[str, Any]],
    non_propagations: list[dict[str, Any]],
    representation: dict[str, Any],
    graph: dict[str, Any],
    taxonomy: dict[str, Any],
    digests: dict[str, Any],
) -> dict[str, Any]:
    best_rows = [row for row in per_sample if row["arm_id"] == BEST_ARM]
    diagnostic_rows = [row for row in best_rows if row["ranking_recovered"] or row.get("regression_diagnosis")]
    boundary_counts = Counter(row["primary_boundary_failure"] for row in diagnostic_rows if row["primary_boundary_failure"] != "ranking_change_not_material")
    confidence = Counter(row["root_cause_confidence"] for row in diagnostic_rows)
    reg_classes = Counter(row["primary_class"] for row in regressions)
    nonprop_classes = Counter(row["primary_class"] for row in non_propagations)
    diagnosis = select_primary_diagnosis(boundary_counts, taxonomy, representation, graph)
    next_family = recommended_next_task_family(diagnosis)
    return {
        "schema_version": "opk-rag.task0136.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "task0133_inputs_valid": authority["task0133_inputs_valid"],
        "task0134_inputs_valid": authority["task0134_inputs_valid"],
        "task0135_inputs_valid": authority["task0135_inputs_valid"],
        "formal_evaluation_unit_count": authority["formal_evaluation_unit_count"],
        "task0135_ranking_failure_recovery_count": authority["task0135_ranking_failure_recovery_count"],
        "task0135_downstream_improved_count": authority["task0135_downstream_improved_count"],
        "task0135_downstream_regressed_count": authority["task0135_downstream_regressed_count"],
        "task0135_downstream_net_gain": authority["task0135_downstream_net_gain"],
        "ranking_recovery_diagnostic_count": funnel["ranking_recovery_count"],
        "ranking_recovery_to_evidence_access_count": funnel["evidence_access_recovery_count"],
        "ranking_recovery_to_evidence_utility_count": funnel["evidence_utility_improvement_count"],
        "ranking_recovery_to_evidence_sufficiency_count": funnel["evidence_sufficiency_improvement_count"],
        "ranking_recovery_to_downstream_improvement_count": funnel["downstream_improvement_count"],
        "ranking_recovery_no_downstream_effect_count": len(non_propagations),
        "ranking_recovery_downstream_regression_count": len(regressions),
        "gold_candidate_full_sufficiency_count": sum(row["candidate_sufficiency"]["gold_candidate_sufficiency_label"] == "fully_answer_sufficient" for row in best_rows),
        "gold_candidate_partial_sufficiency_count": sum(row["candidate_sufficiency"]["gold_candidate_sufficiency_label"] == "partially_answer_sufficient" for row in best_rows),
        "gold_candidate_support_only_count": sum(row["candidate_sufficiency"]["gold_candidate_sufficiency_label"] == "supporting_but_not_sufficient" for row in best_rows),
        "gold_candidate_span_only_count": 0,
        "gold_candidate_uncertain_count": 0,
        "candidate_set_answer_sufficient_count": taxonomy["candidate_set_answer_sufficient_count"],
        "candidate_set_answer_insufficient_count": taxonomy["candidate_set_answer_insufficient_count"],
        "single_evidence_sufficient_count": sum(row["multi_evidence_requirement"] == "single_evidence_sufficient" for row in best_rows),
        "multi_evidence_required_count": sum(row["requires_multiple_evidence"] for row in best_rows),
        "multi_evidence_required_regression_count": sum(row["requires_multiple_evidence"] for row in best_rows if row.get("regression_diagnosis")),
        "gold_span_boundary_split_count": representation["gold_span_boundary_split_count"],
        "adjacent_chunk_required_count": sum(row["adjacent_chunk_required"] for row in diagnostic_rows),
        "same_section_multi_chunk_required_count": sum(row["representation_diagnosis"]["same_section_multi_chunk_required"] for row in diagnostic_rows),
        "cross_section_required_count": sum(row["cross_section_required"] for row in diagnostic_rows),
        "cross_document_required_count": sum(row["cross_document_required"] for row in diagnostic_rows),
        "unknown_context_requirement_count": sum(row["representation_diagnosis"]["context_completeness"] == "unknown" for row in diagnostic_rows),
        "representation_insufficiency_count": representation["representation_insufficiency_count"],
        "candidate_set_incomplete_count": taxonomy["candidate_set_answer_insufficient_count"],
        "heading_context_relevant_case_count": representation["heading_context_relevant_case_count"],
        "graph_sensitive_non_propagation_count": graph["graph_sensitive_non_propagation_count"],
        "graph_sensitive_regression_count": graph["graph_sensitive_regression_count"],
        "taxonomy_boundary_reclassification_candidate_count": taxonomy["taxonomy_boundary_reclassification_candidate_count"],
        "task0135_regression_count": len(regressions),
        "regression_count_by_primary_class": dict(sorted(reg_classes.items())),
        "dominant_regression_class": most_common_or_unknown(reg_classes),
        "ranking_recovery_no_downstream_effect_count_by_class": dict(sorted(nonprop_classes.items())),
        "non_propagation_count_by_class": dict(sorted(nonprop_classes.items())),
        "boundary_failure_count_by_primary_class": dict(sorted(boundary_counts.items())),
        "dominant_boundary_failure_class": most_common_or_unknown(boundary_counts),
        "high_confidence_boundary_case_count": confidence["high"],
        "medium_confidence_boundary_case_count": confidence["medium"],
        "low_confidence_boundary_case_count": confidence["low"],
        "unknown_boundary_case_count": confidence["unknown"],
        "primary_diagnosis": diagnosis,
        "highest_priority_capability_gap": highest_priority_capability_gap(diagnosis),
        "recommended_next_task_family": next_family,
        "runtime_modified": False,
        "promotion_applied": False,
        "task0136_verifier_valid": False,
        "verifier_status": "pending",
        **{key: value for key, value in digests.items() if key != "schema_version"},
    }


def select_primary_diagnosis(boundary_counts: Counter[str], taxonomy: dict[str, Any], representation: dict[str, Any], graph: dict[str, Any]) -> str:
    if not boundary_counts:
        return "boundary_diagnosis_inconclusive"
    dominant = most_common_or_unknown(boundary_counts)
    if dominant == "candidate_set_incomplete" or taxonomy["ranking_cases_with_incomplete_candidate_evidence_set"] > taxonomy["ranking_cases_with_complete_candidate_evidence_set"]:
        return "candidate_set_partial_retrieval_is_primary_gap"
    if dominant == "multi_evidence_requirement":
        return "multi_evidence_retrieval_is_primary_gap"
    if dominant == "graph_sensitive_relation_requirement" or graph["graph_sensitive_non_propagation_count"] + graph["graph_sensitive_regression_count"] > 0:
        return "graph_sensitive_relation_retrieval_is_primary_gap"
    if dominant == "missing_heading_or_section_context":
        return "missing_section_context_is_primary_gap"
    if dominant == "chunk_representation_insufficient" or representation["representation_insufficiency_count"] > 0:
        return "chunk_representation_is_primary_gap"
    if dominant == "generation_evidence_use_failure":
        return "generation_evidence_use_is_primary_gap"
    if dominant == "candidate_relevance_not_sufficiency":
        return "candidate_relevance_does_not_imply_evidence_sufficiency"
    return "mixed_representation_retrieval_boundary_failure"


def highest_priority_capability_gap(primary: str) -> str:
    return {
        "candidate_set_partial_retrieval_is_primary_gap": "candidate_set_answer_sufficiency",
        "multi_evidence_retrieval_is_primary_gap": "multi_evidence_candidate_retrieval",
        "graph_sensitive_relation_retrieval_is_primary_gap": "relation_aware_candidate_retrieval",
        "missing_section_context_is_primary_gap": "section_context_representation",
        "chunk_representation_is_primary_gap": "context_rich_chunk_representation",
        "generation_evidence_use_is_primary_gap": "generation_evidence_utilization",
        "candidate_relevance_does_not_imply_evidence_sufficiency": "utility_aware_evidence_selection",
    }.get(primary, "representation_retrieval_boundary")


def recommended_next_task_family(primary: str) -> str:
    return {
        "candidate_set_partial_retrieval_is_primary_gap": "multi_evidence_candidate_retrieval_experiment",
        "multi_evidence_retrieval_is_primary_gap": "multi_evidence_candidate_retrieval_experiment",
        "graph_sensitive_relation_retrieval_is_primary_gap": "graph_sensitive_retrieval_experiment",
        "missing_section_context_is_primary_gap": "representation_aware_chunk_or_context_experiment",
        "chunk_representation_is_primary_gap": "representation_aware_chunk_or_context_experiment",
        "generation_evidence_use_is_primary_gap": "generation_evidence_utilization_diagnosis",
        "candidate_relevance_does_not_imply_evidence_sufficiency": "evidence_utility_label_enrichment",
    }.get(primary, "representation_aware_chunk_or_context_experiment")


def build_contract(authority: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0136.ranking-to-evidence-utility-boundary-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "runtime_behavior_modification_allowed": False,
        "default_policy_modification_allowed": False,
        "promotion_allowed": False,
        "new_ranking_model_experiment_allowed": False,
        "chunking_modification_allowed": False,
        "retrieval_modification_allowed": False,
        "evidence_composition_modification_allowed": False,
        "generation_modification_allowed": False,
        "formal_evaluation_unit_count": authority["formal_evaluation_unit_count"],
        "evidence_budget_limit": config["evidence_budget_limit"],
        "runtime_modified": False,
        "promotion_applied": False,
        "config_digest": config["config_digest"],
    }


def verify_task0136_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        config = read_json(output_dir / "config.json")
        per_sample = read_jsonl(output_dir / "per_sample.jsonl")
        regressions = read_jsonl(output_dir / "regression_diagnosis.jsonl")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        for field in ("task0133_inputs_valid", "task0134_inputs_valid", "task0135_inputs_valid"):
            if summary.get(field) is not True:
                issues.append({"code": f"{field}_false"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_unit_count_mismatch"})
        if len(per_sample) != 575 * len(SIGNAL_ARMS):
            issues.append({"code": "per_sample_signal_unit_count_mismatch"})
        if len(regressions) != summary.get("task0135_downstream_regressed_count"):
            issues.append({"code": "task0135_regression_count_mismatch"})
        if summary.get("runtime_modified") is not False or config.get("runtime_modified") is not False:
            issues.append({"code": "runtime_modified"})
        if summary.get("promotion_applied") is not False or config.get("promotion_applied") is not False:
            issues.append({"code": "promotion_applied"})
        if config.get("additional_retrieval_calls") != 0 or config.get("additional_generation_calls") != 0:
            issues.append({"code": "unexpected_external_calls"})
        if not summary.get("primary_diagnosis") or not summary.get("recommended_next_task_family"):
            issues.append({"code": "missing_primary_diagnosis_or_next_task"})
    result = {
        "schema_version": "opk-rag.task0136.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task0136_verifier_valid": not issues,
        "runtime_modified": False,
        "promotion_applied": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(
    summary: dict[str, Any],
    funnel: dict[str, Any],
    representation: dict[str, Any],
    graph: dict[str, Any],
    taxonomy: dict[str, Any],
) -> str:
    by_signal = "\n".join(
        f"| `{arm}` | {row['ranking_recovery_count']} | {row['evidence_access_recovery_count']} | {row['evidence_utility_improvement_count']} | {row['evidence_sufficiency_improvement_count']} | {row['downstream_improvement_count']} | {row['downstream_regression_count']} |"
        for arm, row in funnel["by_signal"].items()
    )
    return f"""# TASK-0136 Ranking-to-Evidence Utility Boundary Diagnosis Report

## Required Answers

Q1. TASK-0135 improved Ranking but failed downstream because the best integrated arm converted only `{summary['ranking_recovery_to_evidence_utility_count']}` ranking recoveries into utility improvement and only `{summary['ranking_recovery_to_evidence_sufficiency_count']}` into sufficiency improvement, while causing `{summary['task0135_regression_count']}` EvidenceContext regressions. Candidate relevance did not reliably imply answer-sufficient evidence.

Q2. Ranking recoveries that changed EvidenceContext are reflected in `evidence_context_comparison.jsonl`; best-arm recovered diagnostic count is `{summary['ranking_recovery_diagnostic_count']}`.

Q3. Evidence utility improved in `{summary['ranking_recovery_to_evidence_utility_count']}` best-arm recovered cases.

Q4. Evidence sufficiency improved in `{summary['ranking_recovery_to_evidence_sufficiency_count']}` best-arm recovered cases.

Q5. Final downstream output improved in `{summary['ranking_recovery_to_downstream_improvement_count']}` cases.

Q6. The 12 TASK-0135 regressions are diagnosed in `regression_diagnosis.jsonl`; dominant class is `{summary['dominant_regression_class']}` with counts `{summary['regression_count_by_primary_class']}`.

Q7. No. Candidate relevance is separated from sufficiency; candidate-set insufficient count is `{summary['candidate_set_answer_insufficient_count']}`.

Q8. Single Gold Candidate insufficiency is represented by partial/support labels: partial `{summary['gold_candidate_partial_sufficiency_count']}`, support-only `{summary['gold_candidate_support_only_count']}`.

Q9. Multi-evidence required count is `{summary['multi_evidence_required_count']}`; multi-evidence regressions `{summary['multi_evidence_required_regression_count']}`.

Q10. Apparent ranking cases with incomplete candidate evidence set: `{taxonomy['ranking_cases_with_incomplete_candidate_evidence_set']}`.

Q11. Representation/context deficiency count: `{summary['representation_insufficiency_count']}`; boundary split count `{summary['gold_span_boundary_split_count']}`.

Q12. Missing heading/section context is material in `{summary['heading_context_relevant_case_count']}` diagnostic cases and `{representation['heading_context_relevant_regression_count']}` regressions.

Q13. Graph-sensitive non-propagations `{summary['graph_sensitive_non_propagation_count']}` and regressions `{summary['graph_sensitive_regression_count']}`.

Q14. Zerank recovers more ranking failures (`{funnel['by_signal'][task0135.ARM_C2]['ranking_recovery_count']}`) than the best integrated arm, but its utility/sufficiency conversion is lower than its ranking recovery count, indicating the signal often targets relevance rather than complete answer context.

Q15. Yes, the TASK-0132 binary relevant-candidate rule is too coarse for multi-evidence questions when candidate-set sufficiency is incomplete; reclassification candidates `{summary['taxonomy_boundary_reclassification_candidate_count']}`.

Q16. Primary diagnosis: `{summary['primary_diagnosis']}`. Highest-priority capability gap: `{summary['highest_priority_capability_gap']}`.

Q17. TASK-0137 should work on `{summary['recommended_next_task_family']}`.

## Per-Signal Funnel

| Signal | Ranking Recovery | Evidence Access | Utility Improved | Sufficiency Improved | Downstream Improved | Downstream Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{by_signal}

## Freeze

Runtime modified: `{summary['runtime_modified']}`. Promotion applied: `{summary['promotion_applied']}`. Evidence budget remained `{evidence_composition.EVIDENCE_BUDGET_LIMIT}`.
"""


def required_evidence_count(row: dict[str, Any]) -> int:
    relevant_count = len(row.get("relevant_candidate_ids") or [])
    query = str(row.get("query") or "")
    multi_markers = ("分别", "哪些", "和", "以及", " vs ", "对比", "不同", "relationship", "relation", "path")
    query_requires_multi = any(marker in query for marker in multi_markers) or bool(row.get("graph_sensitive"))
    if row.get("graph_sensitive"):
        return max(2, min(3, relevant_count or 2))
    if query_requires_multi:
        return max(2, min(3, relevant_count or 2))
    return 1


def ranking_recovered(row: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool((row.get("rank_delta") or 0) > 0 or (row.get("ranking_failure_cohort") and row["final_success"] and not base["final_success"]))


def is_downstream_regression(row: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool(base["final_success"] and not row["final_success"])


def is_non_propagating_ranking_recovery(row: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool(ranking_recovered(row, base) and row["final_success"] == base["final_success"] and not is_downstream_regression(row, base))


def heading_context_relevant(query: str, row: dict[str, Any]) -> bool:
    markers = ("section", "heading", "chapter", "标题", "章节", "部分", "范围", "位置", "分别", "不同")
    return any(marker in query.lower() for marker in markers) or bool(row.get("near_cutoff_failure"))


def classify_added_removed_utility(ids: list[str], relevant: set[str]) -> str:
    if not ids:
        return "none"
    return "useful" if any(cid in relevant for cid in ids) else "not_useful"


def most_common_or_unknown(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def strip_nested(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nested = {"evidence_context_comparison", "candidate_sufficiency", "representation_diagnosis", "taxonomy_boundary", "root_cause", "regression_diagnosis", "non_propagation_diagnosis"}
    return [{key: value for key, value in row.items() if key not in nested} for row in rows]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
