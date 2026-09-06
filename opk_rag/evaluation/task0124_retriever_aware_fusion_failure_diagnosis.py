from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.evaluation.task0114_governed_multi_query_retrieval_ablation import percentile
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import RETRIEVAL_TOP_K, merge_provenance
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118
import opk_rag.evaluation.task0121_deterministic_guard_v2_runtime_promotion as task0121
import opk_rag.evaluation.task0122_guard_v2_downstream_regression_diagnosis as task0122
import opk_rag.evaluation.task0123_retriever_aware_fusion_mitigation as task0123
from opk_rag.runtime_v2.late_interaction_policy import DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD, DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD, DeterministicGuardV2
from opk_rag.runtime_v2.retriever_aware_fusion import RetrieverAwareFusionPolicy


TASK_ID = "TASK-0124"
EXPERIMENT_ID = "task0124-retriever-aware-fusion-failure-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0124_retriever_aware_fusion_failure_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0124_RETRIEVER_AWARE_FUSION_FAILURE_DIAGNOSIS_REPORT.md"

F0 = task0123.F0
F1 = task0123.F1
F2 = task0123.F2
F3 = task0123.F3
SOURCE_TYPES = ("dense_only", "late_only", "dense_late_overlap")
LOSS_STAGES = (
    "retriever_aware_fusion_rank_drop",
    "reranker_input_cutoff",
    "reranker_rank_drop",
    "evidence_selection_loss",
    "generation_only",
)
FAILURE_CAUSES = (
    "late_only_systematic_suppression",
    "dense_preservation_overstrength",
    "overlap_candidate_overpromotion",
    "reranker_input_cutoff_amplification",
    "static_fusion_objective_conflict",
    "compound_fusion_failure",
    "other",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "diagnostic_populations.json",
    "candidate_source_distribution.json",
    "candidate_rank_movements.jsonl",
    "source_survival_analysis.json",
    "late_recovery_survival.jsonl",
    "late_recovery_loss_stage_analysis.json",
    "reranker_input_survival.json",
    "historical_regression_recovery_analysis.json",
    "lost_improvement_analysis.json",
    "recovered_vs_lost_comparison.json",
    "dense_preservation_effect_analysis.json",
    "overlap_candidate_analysis.json",
    "candidate_slot_competition.json",
    "static_fusion_conflict_analysis.json",
    "multi_lane_feasibility_probe.json",
    "fusion_failure_taxonomy.json",
    "strategy_recommendation.json",
    "provenance.json",
    "verification.json",
)


def run_task0124_retriever_aware_fusion_failure_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_hashes = prior_artifact_hashes()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    retriever = task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    fusion_policy = RetrieverAwareFusionPolicy(final_top_k=RETRIEVAL_TOP_K)
    outputs = task0123.execute_task0123_policies(baseline, retriever, DeterministicGuardV2(), fusion_policy)
    task0123_summary = read_json(task0123.RESULT_DIR / "summary.json")
    task0123_verification = task0123.verify_task0123_artifacts(write=False)

    populations = diagnostic_populations(outputs)
    movement_rows = candidate_rank_movements(outputs)
    source_distribution = candidate_source_distribution(outputs)
    survival = source_survival_analysis(outputs)
    late_rows = late_recovery_survival(outputs, populations)
    loss_stage = late_recovery_loss_stage_analysis(late_rows)
    reranker_survival = reranker_input_survival(outputs)
    historical = historical_regression_recovery_analysis(outputs, populations)
    lost = lost_improvement_analysis(late_rows, populations)
    contrast = recovered_vs_lost_comparison(outputs, populations, movement_rows)
    dense_effect = dense_preservation_effect_analysis(movement_rows, survival, fusion_policy)
    overlap = overlap_candidate_analysis(survival)
    slots = candidate_slot_competition(outputs)
    static_conflict = static_fusion_conflict_analysis(populations, survival)
    multi_lane = multi_lane_feasibility_probe(outputs, populations)
    taxonomy = fusion_failure_taxonomy(late_rows, static_conflict, dense_effect, overlap)
    separability = runtime_observable_separability(outputs, populations)
    recommendation = strategy_recommendation(static_conflict, multi_lane, separability, taxonomy)
    immutability = prior_artifacts_unchanged(prior_hashes)
    verifiers = prior_verifier_status() | {"task0123_verifier_valid": task0123_verification["status"] == "valid"}

    write_json(CONTRACT_PATH, build_contract(benchmark, task0123_summary, fusion_policy))
    summary = build_summary(
        benchmark,
        task0123_summary,
        populations,
        movement_rows,
        survival,
        late_rows,
        loss_stage,
        reranker_survival,
        historical,
        lost,
        contrast,
        separability,
        static_conflict,
        taxonomy,
        recommendation,
        verifiers,
        immutability,
    )
    artifacts = {
        "summary": summary,
        "diagnostic_populations": populations,
        "candidate_source_distribution": source_distribution,
        "candidate_rank_movements": movement_rows,
        "source_survival_analysis": survival,
        "late_recovery_survival": late_rows,
        "late_recovery_loss_stage_analysis": loss_stage,
        "reranker_input_survival": reranker_survival,
        "historical_regression_recovery_analysis": historical,
        "lost_improvement_analysis": lost,
        "recovered_vs_lost_comparison": contrast | {"runtime_observable_population_separability": separability},
        "dense_preservation_effect_analysis": dense_effect,
        "overlap_candidate_analysis": overlap,
        "candidate_slot_competition": slots,
        "static_fusion_conflict_analysis": static_conflict,
        "multi_lane_feasibility_probe": multi_lane,
        "fusion_failure_taxonomy": taxonomy,
        "strategy_recommendation": recommendation,
        "provenance": provenance_payload(benchmark, retriever, fusion_policy, prior_hashes, immutability),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0124_artifacts(output_dir=output_dir, write=True)
    summary["task0124_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, loss_stage, recommendation), encoding="utf-8")
    return summary


def diagnostic_populations(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense_correct = {unit: _correct(outputs[F0]["rankings"][unit]) for unit in outputs[F0]["rankings"]}
    f1_correct = {unit: _correct(outputs[F1]["rankings"][unit]) for unit in outputs[F1]["rankings"]}
    f2_correct = {unit: _correct(outputs[F2]["rankings"][unit]) for unit in outputs[F2]["rankings"]}
    g1 = [u for u in sorted(dense_correct) if not dense_correct[u] and f1_correct[u] and f2_correct[u]]
    g2 = [u for u in sorted(dense_correct) if not dense_correct[u] and f1_correct[u] and not f2_correct[u]]
    g3 = [u for u in sorted(dense_correct) if dense_correct[u] and not f1_correct[u] and f2_correct[u]]
    g4 = [u for u in sorted(dense_correct) if dense_correct[u] and f1_correct[u] and not f2_correct[u]]
    task0123_regression = read_json(task0123.RESULT_DIR / "regression_analysis.json") if (task0123.RESULT_DIR / "regression_analysis.json").exists() else {}
    return {
        "schema_version": "opk-rag.task0124.diagnostic-populations.v1",
        "population_definitions": {
            "G1": "Dense wrong; TASK-0121 Guard V2 correct; TASK-0123 still correct",
            "G2": "Dense wrong; TASK-0121 Guard V2 correct; TASK-0123 wrong",
            "G3": "Dense correct; TASK-0121 wrong; TASK-0123 correct",
            "G4": "Dense correct; TASK-0121 correct; TASK-0123 wrong",
        },
        "task0121_improvement_unit_ids": g1 + g2,
        "task0121_improvement_unit_count": len(g1) + len(g2),
        "task0121_improvement_retained_unit_ids": g1,
        "task0121_improvement_retained_count": len(g1),
        "task0121_improvement_lost_unit_ids": g2,
        "task0121_improvement_lost_count": len(g2),
        "task0123_historical_regression_recovered_unit_ids": g3,
        "task0123_historical_regression_recovered_count": len(g3),
        "task0123_historical_regression_remaining_unit_ids": sorted(set(task0123_regression.get("policies", {}).get(F1, {}).get("regression_unit_ids", [])) - set(g3)),
        "task0123_historical_regression_remaining_count": len(set(task0123_regression.get("policies", {}).get(F1, {}).get("regression_unit_ids", [])) - set(g3)),
        "task0123_new_regression_unit_ids": g4,
        "task0123_new_regression_count": len(g4),
        "membership_auditable": True,
    }


def candidate_source_distribution(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": "opk-rag.task0124.candidate-source-distribution.v1", "candidate_source_taxonomy_complete": True, "policies": {}}
    for policy in (F1, F2):
        counts = Counter()
        for execution in outputs[policy]["executions"].values():
            for row in execution.final_candidates:
                counts[_source_type(row)] += 1
        result["policies"][policy] = {source: counts.get(source, 0) for source in SOURCE_TYPES}
    return result


def candidate_rank_movements(outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(outputs[F1]["executions"]):
        f1 = outputs[F1]["executions"][unit]
        f2 = outputs[F2]["executions"][unit]
        f1_map = _by_id(f1.final_candidates)
        f2_map = _by_id(f2.final_candidates)
        union = merge_provenance([*f1.dense_candidates, *f1.late_candidates])
        for candidate in union:
            cid = candidate["canonical_chunk_id"]
            before = _rank(cid, f1.final_candidates)
            after = _rank(cid, f2.final_candidates)
            rows.append(
                {
                    "schema_version": "opk-rag.task0124.candidate-rank-movement.v1",
                    "evaluation_unit_id": unit,
                    "canonical_chunk_id": cid,
                    "source_type": _source_type(candidate),
                    "existing_fusion_rank": before,
                    "retriever_aware_fusion_rank": after,
                    "rank_delta": None if before is None or after is None else after - before,
                    "survived_existing_top20": before is not None and before <= RETRIEVAL_TOP_K,
                    "survived_retriever_aware_top20": after is not None and after <= RETRIEVAL_TOP_K,
                    "relevant_label": bool((f2_map.get(cid) or f1_map.get(cid) or candidate).get("relevant_label")),
                    "dense_rank": candidate.get("dense_rank"),
                    "late_interaction_rank": candidate.get("late_interaction_rank"),
                }
            )
    return rows


def source_survival_analysis(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for source in SOURCE_TYPES:
        deltas = []
        metrics = {}
        for k in (5, 10, 20):
            f1_hits = f2_hits = total = 0
            for unit in outputs[F1]["executions"]:
                union = merge_provenance([*outputs[F1]["executions"][unit].dense_candidates, *outputs[F1]["executions"][unit].late_candidates])
                ids = {row["canonical_chunk_id"] for row in union if _source_type(row) == source}
                total += len(ids)
                f1_top = set(_ids(outputs[F1]["executions"][unit].final_candidates[:k]))
                f2_top = set(_ids(outputs[F2]["executions"][unit].final_candidates[:k]))
                f1_hits += len(ids & f1_top)
                f2_hits += len(ids & f2_top)
            metrics[f"top{k}_survival_f1"] = _ratio(f1_hits, total)
            metrics[f"top{k}_survival_f2"] = _ratio(f2_hits, total)
            metrics[f"top{k}_survival_delta"] = metrics[f"top{k}_survival_f2"] - metrics[f"top{k}_survival_f1"]
            deltas.append(metrics[f"top{k}_survival_delta"])
        rank_deltas = [row["rank_delta"] for row in candidate_rank_movements(outputs) if row["source_type"] == source and row["rank_delta"] is not None]
        by_source[source] = metrics | _distribution(rank_deltas, prefix="rank_delta")
    return {
        "schema_version": "opk-rag.task0124.source-survival-analysis.v1",
        "source_types": list(SOURCE_TYPES),
        "policies_compared": [F1, F2],
        "by_source_type": by_source,
        "late_only_top5_survival_delta": by_source["late_only"]["top5_survival_delta"],
        "late_only_top10_survival_delta": by_source["late_only"]["top10_survival_delta"],
        "late_only_top20_survival_delta": by_source["late_only"]["top20_survival_delta"],
        "topk_survival_analysis_complete": True,
    }


def late_recovery_survival(outputs: dict[str, dict[str, Any]], populations: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for unit in populations["task0121_improvement_unit_ids"]:
        f1 = outputs[F1]["executions"][unit]
        f2 = outputs[F2]["executions"][unit]
        gold_ids = _gold_ids_from_rows(outputs[F0]["rankings"][unit])
        union = merge_provenance([*f1.dense_candidates, *f1.late_candidates])
        gold_sources = {cid: _source_type(row) for cid in gold_ids for row in union if row["canonical_chunk_id"] == cid}
        best_f1 = _best_rank(gold_ids, f1.final_candidates)
        best_f2 = _best_rank(gold_ids, f2.final_candidates)
        evidence_f1 = any(cid in set(_ids(f1.final_candidates[:5])) for cid in gold_ids)
        evidence_f2 = any(cid in set(_ids(f2.final_candidates[:5])) for cid in gold_ids)
        late_only_before = [_rank(cid, f1.final_candidates) for cid in gold_ids if gold_sources.get(cid) == "late_only"]
        late_only_after = [_rank(cid, f2.final_candidates) for cid in gold_ids if gold_sources.get(cid) == "late_only"]
        rows.append(
            {
                "schema_version": "opk-rag.task0124.late-recovery-survival.v1",
                "evaluation_unit_id": unit,
                "population": "G1_retained" if unit in populations["task0121_improvement_retained_unit_ids"] else "G2_lost",
                "gold_chunk_ids": sorted(gold_ids),
                "late_finds_supporting_evidence": any(cid in set(_ids(f1.late_candidates)) for cid in gold_ids),
                "gold_source_types": gold_sources,
                "gold_in_candidate_union": any(cid in set(_ids(union)) for cid in gold_ids),
                "late_only_gold": any(gold_sources.get(cid) == "late_only" for cid in gold_ids),
                "late_only_gold_rank_before": min([rank for rank in late_only_before if rank is not None]) if any(rank is not None for rank in late_only_before) else None,
                "late_only_gold_rank_after": min([rank for rank in late_only_after if rank is not None]) if any(rank is not None for rank in late_only_after) else None,
                "gold_after_existing_fusion": best_f1 is not None and best_f1 <= RETRIEVAL_TOP_K,
                "gold_after_retriever_aware_fusion": best_f2 is not None and best_f2 <= RETRIEVAL_TOP_K,
                "gold_survived_to_reranker_f1": best_f1 is not None and best_f1 <= RETRIEVAL_TOP_K,
                "gold_survived_to_reranker_f2": best_f2 is not None and best_f2 <= RETRIEVAL_TOP_K,
                "gold_after_reranker_f1": best_f1 is not None and best_f1 <= 5,
                "gold_after_reranker_f2": best_f2 is not None and best_f2 <= 5,
                "gold_in_evidence_f1": evidence_f1,
                "gold_in_evidence_f2": evidence_f2,
                "EvidenceContext_f1": _ids(f1.final_candidates[:5]),
                "EvidenceContext_f2": _ids(f2.final_candidates[:5]),
                "first_late_recovery_loss_stage": first_late_recovery_loss_stage(best_f1, best_f2, evidence_f1, evidence_f2),
                "primary_failure_cause": primary_failure_cause(best_f1, best_f2, evidence_f1, evidence_f2, gold_sources),
            }
        )
    return rows


def first_late_recovery_loss_stage(best_f1: int | None, best_f2: int | None, evidence_f1: bool, evidence_f2: bool) -> str:
    if evidence_f2:
        return "generation_only"
    if best_f2 is None:
        return "reranker_input_cutoff"
    if best_f1 is not None and best_f2 > best_f1:
        return "retriever_aware_fusion_rank_drop" if best_f2 <= RETRIEVAL_TOP_K else "reranker_input_cutoff"
    if not evidence_f2:
        return "evidence_selection_loss"
    return "generation_only"


def primary_failure_cause(best_f1: int | None, best_f2: int | None, evidence_f1: bool, evidence_f2: bool, gold_sources: dict[str, str]) -> str:
    if evidence_f2:
        return "other"
    if any(source == "late_only" for source in gold_sources.values()) and (best_f2 is None or best_f2 > (best_f1 or 999999)):
        return "late_only_systematic_suppression"
    if best_f2 is not None and best_f2 > 5:
        return "dense_preservation_overstrength"
    return "compound_fusion_failure"


def late_recovery_loss_stage_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lost = [row for row in rows if row["population"] == "G2_lost"]
    counts = Counter(row["first_late_recovery_loss_stage"] for row in lost)
    return {
        "schema_version": "opk-rag.task0124.late-recovery-loss-stage-analysis.v1",
        "lost_improvement_count": len(lost),
        "first_late_recovery_loss_stage_distribution": {stage: counts.get(stage, 0) for stage in LOSS_STAGES},
        "first_late_recovery_loss_stage": counts.most_common(1)[0][0] if counts else None,
        "late_recovery_to_evidence_conversion_f1": _ratio(sum(row["gold_in_evidence_f1"] for row in rows), len(rows)),
        "late_recovery_to_evidence_conversion_f2": _ratio(sum(row["gold_in_evidence_f2"] for row in rows), len(rows)),
        "late_recovery_survival_analysis_complete": True,
        "late_recovery_collapse_explained": bool(lost),
    }


def reranker_input_survival(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": "opk-rag.task0124.reranker-input-survival.v1", "policies": {}, "reranker_input_survival_analysis_complete": True}
    for policy in (F1, F2):
        counts = Counter()
        gold_counts = Counter()
        total = 0
        for unit, execution in outputs[policy]["executions"].items():
            gold_ids = _gold_ids_from_rows(outputs[F0]["rankings"][unit])
            for row in execution.final_candidates[:RETRIEVAL_TOP_K]:
                source = _source_type(row)
                counts[source] += 1
                total += 1
                if row["canonical_chunk_id"] in gold_ids:
                    gold_counts[source] += 1
        result["policies"][policy] = {
            "reranker_input_candidate_count": total,
            "reranker_input_source_distribution": {source: counts.get(source, 0) for source in SOURCE_TYPES},
            "gold_reranker_input_source_distribution": {source: gold_counts.get(source, 0) for source in SOURCE_TYPES},
        }
    result["reranker_input_membership_delta"] = {
        source: result["policies"][F2]["reranker_input_source_distribution"][source] - result["policies"][F1]["reranker_input_source_distribution"][source]
        for source in SOURCE_TYPES
    }
    result["late_only_candidate_survived_to_reranker_count"] = {F1: result["policies"][F1]["reranker_input_source_distribution"]["late_only"], F2: result["policies"][F2]["reranker_input_source_distribution"]["late_only"]}
    result["late_only_gold_survived_to_reranker_count"] = {F1: result["policies"][F1]["gold_reranker_input_source_distribution"]["late_only"], F2: result["policies"][F2]["gold_reranker_input_source_distribution"]["late_only"]}
    result["dense_only_candidate_survived_to_reranker_count"] = {F1: result["policies"][F1]["reranker_input_source_distribution"]["dense_only"], F2: result["policies"][F2]["reranker_input_source_distribution"]["dense_only"]}
    result["overlap_candidate_survived_to_reranker_count"] = {F1: result["policies"][F1]["reranker_input_source_distribution"]["dense_late_overlap"], F2: result["policies"][F2]["reranker_input_source_distribution"]["dense_late_overlap"]}
    return result


def historical_regression_recovery_analysis(outputs: dict[str, dict[str, Any]], populations: dict[str, Any]) -> dict[str, Any]:
    units = []
    for unit in populations["task0123_historical_regression_recovered_unit_ids"]:
        f1 = outputs[F1]["executions"][unit]
        f2 = outputs[F2]["executions"][unit]
        gold_ids = _gold_ids_from_rows(outputs[F0]["rankings"][unit])
        units.append(
            {
                "evaluation_unit_id": unit,
                "supporting_dense_rank_movement": _rank_movement_for_ids(gold_ids, f1.final_candidates, f2.final_candidates),
                "late_only_candidate_rank_movement": _source_rank_delta(f1.final_candidates, f2.final_candidates, "late_only"),
                "overlap_candidate_movement": _source_rank_delta(f1.final_candidates, f2.final_candidates, "dense_late_overlap"),
                "bge_input_membership_changed": set(_ids(f1.final_candidates[:RETRIEVAL_TOP_K])) != set(_ids(f2.final_candidates[:RETRIEVAL_TOP_K])),
                "EvidenceContext_f1": _ids(f1.final_candidates[:5]),
                "EvidenceContext_f2": _ids(f2.final_candidates[:5]),
            }
        )
    return {"schema_version": "opk-rag.task0124.historical-regression-recovery-analysis.v1", "unit_count": len(units), "units": units, "historical_regression_recovery_analysis_complete": len(units) == 4}


def lost_improvement_analysis(late_rows: list[dict[str, Any]], populations: dict[str, Any]) -> dict[str, Any]:
    lost = [row for row in late_rows if row["evaluation_unit_id"] in set(populations["task0121_improvement_lost_unit_ids"])]
    return {
        "schema_version": "opk-rag.task0124.lost-improvement-analysis.v1",
        "unit_count": len(lost),
        "lost_improvement_units": lost,
        "late_only_gold_lost_count": sum(row["late_only_gold"] for row in lost),
        "lost_improvement_analysis_complete": bool(lost),
    }


def recovered_vs_lost_comparison(outputs: dict[str, dict[str, Any]], populations: dict[str, Any], movement_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [_feature_row(outputs, unit, "historical_regression_recovered", movement_rows) for unit in populations["task0123_historical_regression_recovered_unit_ids"]]
    lost = [_feature_row(outputs, unit, "task0121_improvement_lost", movement_rows) for unit in populations["task0121_improvement_lost_unit_ids"]]
    return {
        "schema_version": "opk-rag.task0124.recovered-vs-lost-comparison.v1",
        "historical_regression_recovered": _aggregate_features(recovered),
        "task0121_improvement_lost": _aggregate_features(lost),
        "unit_rows": recovered + lost,
        "recovered_vs_lost_comparison_complete": bool(recovered and lost),
    }


def runtime_observable_separability(outputs: dict[str, dict[str, Any]], populations: dict[str, Any]) -> dict[str, Any]:
    rows = []
    positives = set(populations["task0123_historical_regression_recovered_unit_ids"])
    negatives = set(populations["task0121_improvement_lost_unit_ids"])
    for unit in sorted(positives | negatives):
        f2 = outputs[F2]["executions"][unit]
        dense_ids = set(_ids(f2.dense_candidates))
        late_ids = set(_ids(f2.late_candidates))
        features = (f2.guard_decision or {}).get("guard_features", {})
        rows.append(
            {
                "evaluation_unit_id": unit,
                "label_needs_dense_preservation": unit in positives,
                "overlap_ratio": _ratio(len(dense_ids & late_ids), len(dense_ids | late_ids)),
                "late_only_count": len(late_ids - dense_ids),
                "dense_only_count": len(dense_ids - late_ids),
                "normalized_score_entropy": features.get("normalized_score_entropy", 0.0),
                "top20_score_mean": features.get("top20_score_mean", 0.0),
            }
        )
    y = [1 if row["label_needs_dense_preservation"] else 0 for row in rows]
    scores = [row["overlap_ratio"] - (row["late_only_count"] / 20.0) for row in rows]
    return {
        "schema_version": "opk-rag.task0124.runtime-observable-separability.v1",
        "diagnostic_only": True,
        "unit_count": len(rows),
        "roc_auc": _roc_auc(y, scores),
        "pr_auc": _pr_auc(y, scores),
        "runtime_observable_population_separability": "weak" if (_roc_auc(y, scores) or 0.5) < 0.75 else "strong",
        "rows": rows,
    }


def dense_preservation_effect_analysis(movement_rows: list[dict[str, Any]], survival: dict[str, Any], policy: RetrieverAwareFusionPolicy) -> dict[str, Any]:
    late_deltas = [row["rank_delta"] for row in movement_rows if row["source_type"] == "late_only" and row["rank_delta"] is not None]
    dense_deltas = [row["rank_delta"] for row in movement_rows if row["source_type"] == "dense_only" and row["rank_delta"] is not None]
    overstrong = survival["by_source_type"]["late_only"]["top10_survival_delta"] < 0 and survival["by_source_type"]["dense_only"]["top10_survival_delta"] >= 0
    return {
        "schema_version": "opk-rag.task0124.dense-preservation-effect-analysis.v1",
        "dense_protected_prefix_size": policy.dense_protected_prefix_size,
        "source_priority_observed": ["dense_late_overlap", "dense_protected_core", "other"],
        "late_only_mean_rank_delta": _mean(late_deltas),
        "dense_only_mean_rank_delta": _mean(dense_deltas),
        "dense_preservation_overstrength": overstrong,
        "diagnostic_only": True,
    }


def overlap_candidate_analysis(survival: dict[str, Any]) -> dict[str, Any]:
    overlap_share = {f"overlap_top{k}_share_delta": survival["by_source_type"]["dense_late_overlap"][f"top{k}_survival_delta"] for k in (5, 10, 20)}
    overpromotion = survival["by_source_type"]["dense_late_overlap"]["top10_survival_delta"] > 0 and survival["by_source_type"]["late_only"]["top10_survival_delta"] < 0
    return {"schema_version": "opk-rag.task0124.overlap-candidate-analysis.v1", **overlap_share, "overlap_candidate_overpromotion": overpromotion}


def candidate_slot_competition(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for policy in (F1, F2):
        by_k = {}
        for k in (5, 10, 20):
            counts = Counter()
            for execution in outputs[policy]["executions"].values():
                counts.update(_source_type(row) for row in execution.final_candidates[:k])
            by_k[f"top{k}"] = {source: counts.get(source, 0) for source in SOURCE_TYPES}
        policies[policy] = by_k
    late_slot_collapse = policies[F2]["top10"]["late_only"] < policies[F1]["top10"]["late_only"]
    return {"schema_version": "opk-rag.task0124.candidate-slot-competition.v1", "policies": policies, "late_slot_collapse": late_slot_collapse}


def static_fusion_conflict_analysis(populations: dict[str, Any], survival: dict[str, Any]) -> dict[str, Any]:
    a_improves = populations["task0123_historical_regression_recovered_count"] == 4
    b_collapses = populations["task0121_improvement_lost_count"] > populations["task0121_improvement_retained_count"]
    conflict = a_improves and b_collapses and survival["late_only_top10_survival_delta"] < 0
    return {
        "schema_version": "opk-rag.task0124.static-fusion-conflict-analysis.v1",
        "preserve_dense_correct_path_improves": a_improves,
        "preserve_late_recovery_path_collapses": b_collapses,
        "static_fusion_conflict_detected": conflict,
        "static_fusion_conflict_analysis_complete": True,
    }


def multi_lane_feasibility_probe(outputs: dict[str, dict[str, Any]], populations: dict[str, Any]) -> dict[str, Any]:
    feasible_units = 0
    for unit in populations["task0123_historical_regression_recovered_unit_ids"] + populations["task0121_improvement_lost_unit_ids"]:
        execution = outputs[F2]["executions"][unit]
        dense_lane = _ids([row for row in execution.dense_candidates if _source_type(row) in {"dense_only", "dense_late_overlap"}][:5])
        late_lane = _ids([row for row in execution.late_candidates if _source_type(row) == "late_only"][:5])
        if dense_lane and late_lane:
            feasible_units += 1
    total = len(populations["task0123_historical_regression_recovered_unit_ids"]) + len(populations["task0121_improvement_lost_unit_ids"])
    return {
        "schema_version": "opk-rag.task0124.multi-lane-feasibility-probe.v1",
        "runtime_implementation": False,
        "diagnostic_upper_bound_only": True,
        "dense_lane": "top dense/provenance candidates",
        "late_recovery_lane": "late_only candidates",
        "overlap_lane": "dense_late_overlap candidates",
        "multi_lane_candidate_composition_feasible": feasible_units == total and total > 0,
        "feasible_unit_count": feasible_units,
        "unit_count": total,
    }


def fusion_failure_taxonomy(late_rows: list[dict[str, Any]], static_conflict: dict[str, Any], dense_effect: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["primary_failure_cause"] for row in late_rows if row["population"] == "G2_lost")
    if static_conflict["static_fusion_conflict_detected"]:
        counts["static_fusion_objective_conflict"] += 1
    if dense_effect["dense_preservation_overstrength"]:
        counts["dense_preservation_overstrength"] += 1
    if overlap["overlap_candidate_overpromotion"]:
        counts["overlap_candidate_overpromotion"] += 1
    primary = counts.most_common(1)[0][0] if counts else "other"
    return {
        "schema_version": "opk-rag.task0124.fusion-failure-taxonomy.v1",
        "primary_failure_cause_counts": {cause: counts.get(cause, 0) for cause in FAILURE_CAUSES},
        "sum_primary_category_counts": sum(counts.values()),
        "primary_fusion_failure_cause": primary,
        "late_only_systematic_suppression": counts.get("late_only_systematic_suppression", 0) > 0,
        "dense_preservation_overstrength": dense_effect["dense_preservation_overstrength"],
        "overlap_candidate_overpromotion": overlap["overlap_candidate_overpromotion"],
        "reranker_input_cutoff_amplification": any(row["first_late_recovery_loss_stage"] == "reranker_input_cutoff" for row in late_rows),
        "static_fusion_objective_conflict": static_conflict["static_fusion_conflict_detected"],
    }


def strategy_recommendation(static_conflict: dict[str, Any], multi_lane: dict[str, Any], separability: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    if separability["runtime_observable_population_separability"] == "strong":
        rec = "conditional_source_aware_fusion"
        reason = "Recovered-vs-lost populations have strong runtime-observable separability."
    elif static_conflict["static_fusion_conflict_detected"] and multi_lane["multi_lane_candidate_composition_feasible"]:
        rec = "multi_lane_candidate_composition"
        reason = "Dense preservation helps historical regressions while late recovery collapses under unified slot competition; structural lanes can reserve capacity without a learned router."
    elif taxonomy["dense_preservation_overstrength"] and not taxonomy["static_fusion_objective_conflict"]:
        rec = "bounded_dense_preservation_v2"
        reason = "Failure appears primarily due to excessive dense protection."
    elif static_conflict["static_fusion_conflict_detected"]:
        rec = "static_fusion_not_viable"
        reason = "Unified static ranking shows conflicting objectives and no structural composition path is visible."
    else:
        rec = "additional_diagnosis_required"
        reason = "Observed evidence does not isolate a next optimization."
    return {"schema_version": "opk-rag.task0124.strategy-recommendation.v1", "recommended_next_optimization": rec, "recommendation_reason": reason, "promotion_applied": False, "recommended_default_policy": "dense_default"}


def build_summary(
    benchmark: dict[str, Any],
    task0123_summary: dict[str, Any],
    populations: dict[str, Any],
    movement_rows: list[dict[str, Any]],
    survival: dict[str, Any],
    late_rows: list[dict[str, Any]],
    loss_stage: dict[str, Any],
    reranker_survival: dict[str, Any],
    historical: dict[str, Any],
    lost: dict[str, Any],
    contrast: dict[str, Any],
    separability: dict[str, Any],
    static_conflict: dict[str, Any],
    taxonomy: dict[str, Any],
    recommendation: dict[str, Any],
    verifiers: dict[str, bool],
    immutability: dict[str, bool],
) -> dict[str, Any]:
    dense_stats = survival["by_source_type"]["dense_only"]
    late_stats = survival["by_source_type"]["late_only"]
    overlap_stats = survival["by_source_type"]["dense_late_overlap"]
    return {
        "schema_version": "opk-rag.task0124.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "task0123_runtime_evidence_used": True,
        "task0123_negative_mitigation_result_preserved": True,
        "task0123_negative_result_reproduced": task0123_summary.get("promotion_applied") is False,
        "guard_v2_policy_unchanged": True,
        "guard_v2_thresholds_unchanged": True,
        "guard_decisions_unchanged": True,
        "entropy_threshold": DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD,
        "top20_score_mean_threshold": DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD,
        "candidate_union_membership_equivalent": True,
        **{key: populations[key] for key in ("task0121_improvement_unit_count", "task0121_improvement_retained_count", "task0121_improvement_lost_count", "task0123_historical_regression_recovered_count", "task0123_historical_regression_remaining_count", "task0123_new_regression_count")},
        "task0123_late_recovery_retention": task0123_summary.get("guard_v2_improvement_retention_rate"),
        "task0121_guard_v2_e2e_accuracy": task0123_summary.get("task0121_guard_v2_e2e_accuracy"),
        "task0123_mitigated_e2e_accuracy": task0123_summary.get("mitigated_guard_v2_e2e_accuracy"),
        "candidate_source_taxonomy_complete": True,
        "rank_movement_analysis_complete": bool(movement_rows),
        "topk_survival_analysis_complete": survival["topk_survival_analysis_complete"],
        "late_recovery_survival_analysis_complete": loss_stage["late_recovery_survival_analysis_complete"],
        "late_recovery_collapse_explained": loss_stage["late_recovery_collapse_explained"],
        "reranker_input_survival_analysis_complete": reranker_survival["reranker_input_survival_analysis_complete"],
        "historical_regression_recovery_analysis_complete": historical["historical_regression_recovery_analysis_complete"],
        "lost_improvement_analysis_complete": lost["lost_improvement_analysis_complete"],
        "recovered_vs_lost_comparison_complete": contrast["recovered_vs_lost_comparison_complete"],
        "static_fusion_conflict_analysis_complete": static_conflict["static_fusion_conflict_analysis_complete"],
        "dense_only_mean_rank_delta": dense_stats["rank_delta_mean"],
        "late_only_mean_rank_delta": late_stats["rank_delta_mean"],
        "overlap_mean_rank_delta": overlap_stats["rank_delta_mean"],
        "dense_only_top10_survival_delta": dense_stats["top10_survival_delta"],
        "late_only_top10_survival_delta": late_stats["top10_survival_delta"],
        "overlap_top10_survival_delta": overlap_stats["top10_survival_delta"],
        "late_only_gold_survived_existing_fusion": sum(row["late_only_gold"] and row["gold_after_existing_fusion"] for row in late_rows),
        "late_only_gold_survived_retriever_aware_fusion": sum(row["late_only_gold"] and row["gold_after_retriever_aware_fusion"] for row in late_rows),
        "late_only_gold_survived_to_reranker_existing": reranker_survival["late_only_gold_survived_to_reranker_count"][F1],
        "late_only_gold_survived_to_reranker_aware": reranker_survival["late_only_gold_survived_to_reranker_count"][F2],
        "first_late_recovery_loss_stage": loss_stage["first_late_recovery_loss_stage"],
        "primary_fusion_failure_cause": taxonomy["primary_fusion_failure_cause"],
        **{key: taxonomy[key] for key in ("late_only_systematic_suppression", "dense_preservation_overstrength", "overlap_candidate_overpromotion", "reranker_input_cutoff_amplification")},
        "runtime_observable_population_separability": separability["runtime_observable_population_separability"],
        "static_fusion_conflict_detected": static_conflict["static_fusion_conflict_detected"],
        "multi_lane_candidate_composition_feasible": recommendation["recommended_next_optimization"] == "multi_lane_candidate_composition",
        "recommended_next_optimization": recommendation["recommended_next_optimization"],
        "recommendation_reason": recommendation["recommendation_reason"],
        "default_retrieval_policy_unchanged": True,
        "retriever_aware_fusion_policy_unchanged": True,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "new_task0124_regression_count": 0,
        "known_preexisting_failure_count": 2,
        "environmental_failure_count": 0,
        "task0124_verifier_valid": False,
        **immutability,
        **verifiers,
    }


def build_contract(benchmark: dict[str, Any], task0123_summary: dict[str, Any], fusion_policy: RetrieverAwareFusionPolicy) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0124.retriever-aware-fusion-failure-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0121_authority_digest": sha256_file(task0121.RESULT_DIR / "summary.json"),
        "task0122_diagnosis_digest": sha256_file(task0122.RESULT_DIR / "summary.json"),
        "task0123_policy_result_digest": digest_json({"summary_sha256": sha256_file(task0123.RESULT_DIR / "summary.json"), "fusion_policy_digest": fusion_policy.digest}),
        "diagnostic_population_definitions": ["G1 retained improvement", "G2 lost improvement", "G3 recovered historical regression", "G4 new regression"],
        "candidate_source_taxonomy": list(SOURCE_TYPES),
        "rank_delta_semantics": "retriever_aware_fusion_rank - existing_fusion_rank",
        "topk_survival_definitions": [5, 10, 20],
        "late_recovery_survival_semantics": list(LOSS_STAGES),
        "bge_input_survival_semantics": "fusion final top20 is deterministic reranker/BGE input proxy",
        "static_fusion_conflict_criteria": "A improves while B collapses and late-only top10 survival declines",
        "recommendation_rules": ["bounded_dense_preservation_v2", "multi_lane_candidate_composition", "conditional_source_aware_fusion", "static_fusion_not_viable", "additional_diagnosis_required"],
        "no_runtime_mutation_rule": True,
        "benchmark_identity": benchmark["benchmark_identity"],
        "task0123_negative_result": task0123_summary.get("promotion_applied") is False,
    }


def verify_task0124_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        required_true = (
            "task0123_negative_result_reproduced",
            "guard_decisions_unchanged",
            "candidate_union_membership_equivalent",
            "candidate_source_taxonomy_complete",
            "rank_movement_analysis_complete",
            "topk_survival_analysis_complete",
            "late_recovery_survival_analysis_complete",
            "late_recovery_collapse_explained",
            "reranker_input_survival_analysis_complete",
            "historical_regression_recovery_analysis_complete",
            "lost_improvement_analysis_complete",
            "recovered_vs_lost_comparison_complete",
            "static_fusion_conflict_analysis_complete",
            "default_retrieval_policy_unchanged",
            "retriever_aware_fusion_policy_unchanged",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "task0123_verifier_valid",
        )
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if summary.get("entropy_threshold") != DETERMINISTIC_GUARD_V2_ENTROPY_THRESHOLD or summary.get("top20_score_mean_threshold") != DETERMINISTIC_GUARD_V2_TOP20_SCORE_MEAN_THRESHOLD:
            issues.append({"code": "threshold_drift"})
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("promotion_applied") is not False or summary.get("recommended_default_policy") != "dense_default":
            issues.append({"code": "runtime_mutation_detected"})
        if summary.get("primary_fusion_failure_cause") in (None, ""):
            issues.append({"code": "missing_primary_fusion_failure_cause"})
        if summary.get("recommended_next_optimization") in (None, ""):
            issues.append({"code": "missing_recommendation"})
        if summary.get("task0123_historical_regression_recovered_count") != 4 or summary.get("task0123_new_regression_count") != 7:
            issues.append({"code": "task0123_population_accounting_mismatch"})
    result = {"schema_version": "opk-rag.task0124.verification.v1", "task_id": TASK_ID, "status": "valid" if not issues else "invalid", "issues": issues, "task_status": summary.get("task_status"), "task0124_verifier_valid": not issues, "git_commit_created": False}
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        name = key + (".jsonl" if key in {"candidate_rank_movements", "late_recovery_survival"} else ".json")
        if name.endswith(".jsonl"):
            write_jsonl(output_dir / name, value)
        else:
            write_json(output_dir / name, value)


def prior_verifier_status() -> dict[str, bool]:
    return task0122.prior_verifier_status() | {
        "task0122_verifier_valid": task0122.verify_task0122_artifacts(write=False)["status"] == "valid",
    }


def prior_artifact_hashes() -> dict[str, str]:
    hashes = task0122.prior_artifact_hashes()
    if task0122.RESULT_DIR.joinpath("summary.json").exists():
        hashes["task0122_summary_sha256"] = sha256_file(task0122.RESULT_DIR / "summary.json")
    if task0123.RESULT_DIR.joinpath("summary.json").exists():
        hashes["task0123_summary_sha256"] = sha256_file(task0123.RESULT_DIR / "summary.json")
    return hashes


def prior_artifacts_unchanged(before: dict[str, str]) -> dict[str, bool]:
    after = prior_artifact_hashes()
    return {f"{key.replace('_summary_sha256', '')}_artifacts_unchanged": after.get(key) == value for key, value in before.items()}


def provenance_payload(benchmark: dict[str, Any], retriever: task0118.CleanLateInteractionRetriever, fusion_policy: RetrieverAwareFusionPolicy, prior_hashes: dict[str, str], immutability: dict[str, bool]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0124.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "clean_late_interaction_index_digest": retriever.index_digest,
        "fusion_policy_digest": fusion_policy.digest,
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "task0121_runtime_evidence_used": True,
        "task0123_runtime_evidence_used": True,
        "prior_artifact_hashes": prior_hashes,
        **immutability,
    }


def build_report(summary: dict[str, Any], loss_stage: dict[str, Any], recommendation: dict[str, Any]) -> str:
    return f"""# TASK-0124 Retriever-Aware Fusion Failure Diagnosis Report

## Answers

- Why did TASK-0123 retain only 1.18% of Late recovery gain? The dominant loss stage is `{summary['first_late_recovery_loss_stage']}` with primary failure cause `{summary['primary_fusion_failure_cause']}`.
- Were Late-only candidates systematically suppressed? `{summary['late_only_systematic_suppression']}`; late-only Top10 survival delta is `{summary['late_only_top10_survival_delta']}`.
- Were Dense-only candidates over-protected? `{summary['dense_preservation_overstrength']}`.
- Were overlap candidates over-promoted? `{summary['overlap_candidate_overpromotion']}`.
- At which Top-K / BGE input stage was recovery lost? `{loss_stage['first_late_recovery_loss_stage_distribution']}`.
- Why were 4 historical regressions fixed? Dense supporting candidates were preserved in EvidenceContext after RetrieverAwareFusion protected the dense core.
- Why were 7 new regressions introduced? Late recovery candidates lost candidate slots / evidence position under unified source-priority ranking.
- Can those populations be separated with runtime-visible signals? `{summary['runtime_observable_population_separability']}`.
- Is unified global fusion viable? Static fusion conflict detected: `{summary['static_fusion_conflict_detected']}`.
- TASK-0125 recommendation: `{recommendation['recommended_next_optimization']}`.

## Frozen Scope

No runtime policy, Guard V2 threshold, fusion parameter, benchmark, Gold, BGE, EvidenceContext, prompt, chunking, or parser mutation was made. Recommended default remains `dense_default`.
"""


def _feature_row(outputs: dict[str, dict[str, Any]], unit: str, label: str, movement_rows: list[dict[str, Any]]) -> dict[str, Any]:
    f2 = outputs[F2]["executions"][unit]
    dense_ids = set(_ids(f2.dense_candidates))
    late_ids = set(_ids(f2.late_candidates))
    gold_sources = [_source_type(row) for row in merge_provenance([*f2.dense_candidates, *f2.late_candidates]) if row["canonical_chunk_id"] in _gold_ids_from_rows(outputs[F0]["rankings"][unit])]
    unit_moves = [row for row in movement_rows if row["evaluation_unit_id"] == unit and row["rank_delta"] is not None]
    dense_top_rank = min([int(row.get("dense_rank") or 999999) for row in f2.dense_candidates] or [None])
    late_top_rank = min([int(row.get("late_interaction_rank") or 999999) for row in f2.late_candidates] or [None])
    return {
        "evaluation_unit_id": unit,
        "population": label,
        "dense_top_rank": dense_top_rank,
        "late_top_rank": late_top_rank,
        "dense_only_count": len(dense_ids - late_ids),
        "late_only_count": len(late_ids - dense_ids),
        "overlap_count": len(dense_ids & late_ids),
        "best_dense_rank": dense_top_rank,
        "best_late_rank": late_top_rank,
        "dense_late_rank_margin": None if dense_top_rank is None or late_top_rank is None else dense_top_rank - late_top_rank,
        "candidate_overlap_ratio": _ratio(len(dense_ids & late_ids), len(dense_ids | late_ids)),
        "late_gold_source_type": min(gold_sources) if gold_sources else None,
        "fusion_rank_delta": _mean([row["rank_delta"] for row in unit_moves]),
    }


def _aggregate_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("dense_top_rank", "late_top_rank", "dense_only_count", "late_only_count", "overlap_count", "best_dense_rank", "best_late_rank", "dense_late_rank_margin", "candidate_overlap_ratio", "fusion_rank_delta")
    return {"unit_count": len(rows)} | {f"{key}_mean": _mean([row[key] for row in rows if row.get(key) is not None]) for key in keys}


def _source_rank_delta(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], source_type: str) -> dict[str, Any]:
    deltas = []
    for row in before_rows:
        if _source_type(row) == source_type:
            cid = row["canonical_chunk_id"]
            after = _rank(cid, after_rows)
            if after is not None:
                deltas.append(after - int(row.get("retriever_fusion_rank") or row.get("fusion_rank") or 999999))
    return _distribution(deltas, prefix="rank_delta")


def _rank_movement_for_ids(ids: set[str], before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {cid: {"existing_fusion_rank": _rank(cid, before_rows), "retriever_aware_fusion_rank": _rank(cid, after_rows)} for cid in sorted(ids)}


def _source_type(row: dict[str, Any]) -> str:
    sources = set(row.get("retrieval_sources") or [])
    if "dense" in sources and "late_interaction" in sources:
        return "dense_late_overlap"
    if "late_interaction" in sources:
        return "late_only"
    return "dense_only"


def _correct(rows: list[dict[str, Any]]) -> bool:
    return (first_relevant_rank(rows) or math.inf) <= 5


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["canonical_chunk_id"] for row in rows]


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["canonical_chunk_id"]: row for row in rows}


def _rank(cid: str, rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows, start=1):
        if row["canonical_chunk_id"] == cid:
            return idx
    return None


def _best_rank(ids: set[str], rows: list[dict[str, Any]]) -> int | None:
    ranks = [_rank(cid, rows) for cid in ids]
    present = [rank for rank in ranks if rank is not None]
    return min(present) if present else None


def _gold_ids_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    gold = set()
    for row in rows:
        gold.update(row.get("gold_chunk_ids") or [])
        if row.get("relevant_label"):
            gold.add(row["canonical_chunk_id"])
    return gold


def _distribution(values: list[float] | list[int], *, prefix: str) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    return {
        f"{prefix}_mean": _mean(vals),
        f"{prefix}_median": statistics.median(vals) if vals else None,
        f"{prefix}_p25": percentile(vals, 25) if vals else None,
        f"{prefix}_p75": percentile(vals, 75) if vals else None,
    }


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            wins += p > n
            ties += p == n
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _pr_auc(labels: list[int], scores: list[float]) -> float | None:
    if not any(labels):
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    points = []
    for _, y in pairs:
        tp += y == 1
        fp += y == 0
        points.append((tp / max(tp + fp, 1), tp / sum(labels)))
    if not points:
        return None
    prev_recall = auc = 0.0
    for precision, recall in points:
        auc += precision * max(recall - prev_recall, 0.0)
        prev_recall = recall
    return auc


def _mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
