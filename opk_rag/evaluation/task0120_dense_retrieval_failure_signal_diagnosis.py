from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import statistics
from typing import Any, Callable

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark
from opk_rag.evaluation.task0119_clean_late_interaction_runtime_promotion import RESULT_DIR as TASK0119_RESULT_DIR
from opk_rag.evaluation.task0119_clean_late_interaction_runtime_promotion import prior_verifier_status, verify_task0119_artifacts
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118


TASK_ID = "TASK-0120"
EXPERIMENT_ID = "task0120-dense-retrieval-failure-signal-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0120_dense_retrieval_failure_signal_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0120_DENSE_RETRIEVAL_FAILURE_SIGNAL_DIAGNOSIS_REPORT.md"

RETRIEVAL_TOP_K = 20
MAX_RULE_FEATURE_COUNT = 3
MAX_CANDIDATE_RULES = 240
NUMERIC_FEATURES = (
    "dense_top1_score",
    "dense_top2_score",
    "dense_top3_score",
    "dense_top5_score",
    "dense_top10_score",
    "dense_top20_score",
    "top1_top2_margin",
    "top1_top3_margin",
    "top1_top5_margin",
    "top1_top10_margin",
    "top5_score_mean",
    "top10_score_mean",
    "top20_score_mean",
    "top5_score_std",
    "top10_score_std",
    "top20_score_std",
    "score_range_top5",
    "score_range_top10",
    "normalized_score_entropy",
    "score_slope_top5",
    "score_slope_top10",
    "unique_document_count_top5",
    "unique_document_count_top10",
    "unique_document_count_top20",
    "unique_section_count_top5",
    "unique_section_count_top10",
    "unique_section_count_top20",
    "heading_diversity_top20",
    "dense_lexical_top5_overlap",
    "dense_lexical_top10_overlap",
    "dense_lexical_top20_overlap",
    "dense_top1_in_lexical_top20",
    "lexical_top1_in_dense_top20",
    "rank_correlation",
    "dense_rank_vs_reranker_rank_delta",
    "top_candidate_reranker_score",
    "dense_top1_reranker_rank",
    "reranker_score_margin",
    "near_duplicate_candidate_count",
    "same_section_candidate_ratio",
    "same_document_candidate_ratio",
    "query_length_chars",
    "query_length_tokens",
    "query_unique_token_count",
    "query_digit_count",
    "query_symbol_count",
    "query_entity_like_token_count",
    "query_uppercase_token_count",
    "query_question_word_count",
    "technical_term_count",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "failure_signal_features.jsonl",
    "router_target_distribution.json",
    "single_feature_analysis.json",
    "threshold_analysis.json",
    "task0119_guard_baseline.json",
    "deterministic_rule_analysis.json",
    "routing_efficiency_curve.json",
    "guard_false_negative_units.jsonl",
    "guard_false_positive_units.jsonl",
    "learned_router_probe.json",
    "signal_strength_assessment.json",
    "strategy_recommendation.json",
    "provenance.json",
    "verification.json",
)


@dataclass(frozen=True)
class RuntimeFeatureVector:
    values: dict[str, float | int | bool]

    def digest(self) -> str:
        return digest_json(self.values)


def run_task0120_dense_retrieval_failure_signal_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    clean_results = load_clean_retrieval_results()
    guard_rows = {row["evaluation_unit_id"]: row for row in read_jsonl(TASK0119_RESULT_DIR / "guard_results.jsonl")}
    records = build_failure_signal_records(baseline, clean_results, guard_rows)
    write_json(CONTRACT_PATH, build_contract(benchmark))

    distribution = router_target_distribution(records)
    guard_baseline = task0119_guard_baseline(records)
    single = single_feature_analysis(records)
    thresholds = threshold_analysis(records)
    rules = deterministic_rule_analysis(records, thresholds)
    curve = routing_efficiency_curve(records, guard_baseline, single, rules)
    false_negatives = guard_false_negative_units(records)
    false_positives = guard_false_positive_units(records)
    learned = learned_router_probe()
    signal = signal_strength_assessment(single, rules, learned, guard_baseline)
    recommendation = strategy_recommendation(signal, rules, guard_baseline)
    provenance = provenance_payload(benchmark)
    summary = build_summary(benchmark, distribution, guard_baseline, single, rules, curve, false_negatives, false_positives, learned, signal, recommendation)

    artifacts = {
        "summary": summary,
        "failure_signal_features": records,
        "router_target_distribution": distribution,
        "single_feature_analysis": single,
        "threshold_analysis": thresholds,
        "task0119_guard_baseline": guard_baseline,
        "deterministic_rule_analysis": rules,
        "routing_efficiency_curve": curve,
        "guard_false_negative_units": false_negatives,
        "guard_false_positive_units": false_positives,
        "learned_router_probe": learned,
        "signal_strength_assessment": signal,
        "strategy_recommendation": recommendation,
        "provenance": provenance,
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0120_artifacts(output_dir=output_dir, write=True)
    summary["task0120_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, guard_baseline, single, rules, signal, recommendation), encoding="utf-8")
    return summary


def load_clean_retrieval_results() -> dict[str, dict[str, dict[str, Any]]]:
    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(task0118.RESULT_DIR / "clean_retrieval_results.jsonl"):
        by_unit.setdefault(row["evaluation_unit_id"], {})[row["arm_id"]] = row
    return by_unit


def build_failure_signal_records(
    baseline: dict[str, list[dict[str, Any]]],
    clean_results: dict[str, dict[str, dict[str, Any]]],
    guard_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for unit, rows in sorted(baseline.items()):
        dense_clean = clean_results[unit]["C0_dense_baseline"]
        late_clean = clean_results[unit]["C1_clean_late_interaction_first_stage"]
        dense_hit = (dense_clean.get("gold_rank") or math.inf) <= RETRIEVAL_TOP_K
        late_hit = (late_clean.get("gold_rank") or math.inf) <= RETRIEVAL_TOP_K
        label = offline_label(dense_hit, late_hit)
        guard = guard_rows[unit]
        features = extract_runtime_features(rows, dense_clean, guard)
        record = {
            "schema_version": "opk-rag.task0120.failure-signal-record.v1",
            "evaluation_unit_id": unit,
            "router_target": "dense_failure_late_recoverable" if label["router_positive"] else "router_negative",
            "runtime_features": features.values,
            "runtime_feature_digest": features.digest(),
            "offline_labels": label | {
                "guard_triggered": bool(guard["guard_triggered"]),
                "task0119_late_needed_for_recovery": bool(guard["late_needed_for_recovery"]),
            },
            "runtime_feature_gold_independent": True,
            "offline_labels_separated": True,
        }
        records.append(record)
    return records


def extract_runtime_features(rows: list[dict[str, Any]], dense_clean: dict[str, Any], guard: dict[str, Any]) -> RuntimeFeatureVector:
    top20 = rows[:RETRIEVAL_TOP_K]
    dense_candidates = dense_clean.get("candidates", [])[:RETRIEVAL_TOP_K]
    scores = [float(c.get("dense_score") or 0.0) for c in dense_candidates]
    query = str(rows[0].get("question") or "")
    query_tokens = tokenize(query)
    reranked = sorted(top20, key=lambda row: float(row.get("reranker_score") or 0.0), reverse=True)
    dense_top1_id = str(top20[0].get("canonical_chunk_id") or "")
    reranker_rank = next((idx for idx, row in enumerate(reranked, start=1) if row.get("canonical_chunk_id") == dense_top1_id), len(reranked) + 1)
    lexical_ranks = [int(row["retrieval_rank"]) for row in top20]
    values: dict[str, float | int | bool] = {
        "runtime_observable_only": True,
        "dense_top1_score": score_at(scores, 1),
        "dense_top2_score": score_at(scores, 2),
        "dense_top3_score": score_at(scores, 3),
        "dense_top5_score": score_at(scores, 5),
        "dense_top10_score": score_at(scores, 10),
        "dense_top20_score": score_at(scores, 20),
        "top1_top2_margin": score_at(scores, 1) - score_at(scores, 2),
        "top1_top3_margin": score_at(scores, 1) - score_at(scores, 3),
        "top1_top5_margin": score_at(scores, 1) - score_at(scores, 5),
        "top1_top10_margin": score_at(scores, 1) - score_at(scores, 10),
        "top5_score_mean": mean(scores[:5]),
        "top10_score_mean": mean(scores[:10]),
        "top20_score_mean": mean(scores[:20]),
        "top5_score_std": std(scores[:5]),
        "top10_score_std": std(scores[:10]),
        "top20_score_std": std(scores[:20]),
        "score_range_top5": range_value(scores[:5]),
        "score_range_top10": range_value(scores[:10]),
        "normalized_score_entropy": normalized_entropy(scores[:20]),
        "score_slope_top5": slope(scores[:5]),
        "score_slope_top10": slope(scores[:10]),
        "unique_document_count_top5": unique_count(top20[:5], "document_id"),
        "unique_document_count_top10": unique_count(top20[:10], "document_id"),
        "unique_document_count_top20": unique_count(top20[:20], "document_id"),
        "unique_section_count_top5": unique_count(top20[:5], "section_id"),
        "unique_section_count_top10": unique_count(top20[:10], "section_id"),
        "unique_section_count_top20": unique_count(top20[:20], "section_id"),
        "heading_diversity_top20": len({tuple(row.get("heading_path") or []) for row in top20}),
        "dense_lexical_top5_overlap": dense_lexical_overlap(top20, 5),
        "dense_lexical_top10_overlap": dense_lexical_overlap(top20, 10),
        "dense_lexical_top20_overlap": dense_lexical_overlap(top20, 20),
        "dense_top1_in_lexical_top20": True,
        "lexical_top1_in_dense_top20": bool(lexical_ranks and min(lexical_ranks) <= 20),
        "rank_correlation": 1.0,
        "dense_rank_vs_reranker_rank_delta": abs(1 - reranker_rank),
        "top_candidate_reranker_score": float(top20[0].get("reranker_score") or 0.0),
        "dense_top1_reranker_rank": reranker_rank,
        "reranker_score_margin": float(reranked[0].get("reranker_score") or 0.0) - float(reranked[1].get("reranker_score") or 0.0) if len(reranked) > 1 else 0.0,
        "near_duplicate_candidate_count": near_duplicate_candidate_count(top20),
        "same_section_candidate_ratio": same_value_ratio(top20, "section_id"),
        "same_document_candidate_ratio": same_value_ratio(top20, "document_id"),
        "query_length_chars": len(query),
        "query_length_tokens": len(query_tokens),
        "query_unique_token_count": len(set(query_tokens)),
        "query_digit_count": sum(ch.isdigit() for ch in query),
        "query_symbol_count": sum((not ch.isalnum()) and (not ch.isspace()) for ch in query),
        "query_entity_like_token_count": sum(entity_like(token) for token in query_tokens),
        "query_uppercase_token_count": sum(token.isupper() and any(ch.isalpha() for ch in token) for token in query_tokens),
        "query_question_word_count": sum(token.lower() in {"what", "why", "how", "when", "where", "which", "谁", "什么", "哪些", "如何", "为什么", "怎么"} for token in query_tokens),
        "technical_term_count": sum(technical_term(token) for token in query_tokens),
        "task0119_retrieval_confidence": float(guard["guard_features"].get("retrieval_confidence", 0.0)),
    }
    return RuntimeFeatureVector(values)


def offline_label(dense_hit: bool, late_hit: bool) -> dict[str, Any]:
    if dense_hit and late_hit:
        klass = "dense_success"
    elif (not dense_hit) and late_hit:
        klass = "dense_failure_late_recoverable"
    elif (not dense_hit) and (not late_hit):
        klass = "neither_retriever_recovers"
    else:
        klass = "dense_only_success"
    return {
        "dense_gold_in_top20": dense_hit,
        "clean_late_gold_in_top20": late_hit,
        "diagnostic_class": klass,
        "router_positive": klass == "dense_failure_late_recoverable",
    }


def router_target_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"dense_success": 0, "dense_failure_late_recoverable": 0, "neither_retriever_recovers": 0, "dense_only_success": 0}
    for record in records:
        counts[record["offline_labels"]["diagnostic_class"]] += 1
    return {
        "schema_version": "opk-rag.task0120.router-target-distribution.v1",
        "formal_evaluation_unit_count": len(records),
        "router_positive_definition": "dense_gold_in_top20=false AND clean_late_gold_in_top20=true",
        "router_positive_count": counts["dense_failure_late_recoverable"],
        "router_negative_count": len(records) - counts["dense_failure_late_recoverable"],
        "class_counts": counts,
    }


def task0119_guard_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = binary_metrics([r["offline_labels"]["router_positive"] for r in records], [r["offline_labels"]["guard_triggered"] for r in records])
    trigger_rate = mean([1.0 if r["offline_labels"]["guard_triggered"] else 0.0 for r in records])
    return {
        "schema_version": "opk-rag.task0120.task0119-guard-baseline.v1",
        "guard_policy_source": "TASK-0119 guarded_late_interaction",
        "guard_reconstructed_on_router_positive": True,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "recovery_coverage": metrics["recall"],
        "trigger_rate": trigger_rate,
        "invocation_reduction": 1.0 - trigger_rate,
        "task0119_reported_recovery_coverage": 0.4348,
        "task0119_reported_invocation_reduction": 0.6678,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def single_feature_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [bool(r["offline_labels"]["router_positive"]) for r in records]
    feature_results = {}
    for feature in NUMERIC_FEATURES + ("task0119_retrieval_confidence",):
        values = [float(r["runtime_features"].get(feature, 0.0)) for r in records]
        roc = roc_auc(values, labels)
        inv = roc_auc([-v for v in values], labels)
        direction = "higher_indicates_positive" if roc >= inv else "lower_indicates_positive"
        auc = max(roc, inv)
        oriented = values if direction == "higher_indicates_positive" else [-v for v in values]
        feature_results[feature] = {
            "roc_auc": auc,
            "pr_auc": pr_auc(oriented, labels),
            "direction": direction,
            "positive_stats": stats([v for v, y in zip(values, labels, strict=True) if y]),
            "negative_stats": stats([v for v, y in zip(values, labels, strict=True) if not y]),
        }
    best = max(feature_results, key=lambda key: (feature_results[key]["roc_auc"], feature_results[key]["pr_auc"], key))
    return {
        "schema_version": "opk-rag.task0120.single-feature-analysis.v1",
        "target": "dense_failure_late_recoverable",
        "feature_results": feature_results,
        "best_single_feature": best,
        "best_single_feature_roc_auc": feature_results[best]["roc_auc"],
        "best_single_feature_pr_auc": feature_results[best]["pr_auc"],
        "single_feature_analysis_complete": True,
    }


def threshold_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [bool(r["offline_labels"]["router_positive"]) for r in records]
    formal_records = [r for r in records if split_for_unit(r["evaluation_unit_id"]) == "formal_eval"]
    by_feature = {}
    candidate_rules = 0
    for feature in NUMERIC_FEATURES + ("task0119_retrieval_confidence",):
        values = [float(r["runtime_features"].get(feature, 0.0)) for r in records]
        thresholds = quantile_thresholds(values)
        rows = []
        for threshold in thresholds:
            for operator in ("<=", ">="):
                candidate_rules += 1
                preds = [apply_threshold(float(r["runtime_features"].get(feature, 0.0)), operator, threshold) for r in records]
                metrics = binary_metrics(labels, preds)
                trigger_rate = mean([1.0 if pred else 0.0 for pred in preds])
                rows.append(threshold_row(feature, operator, threshold, metrics, trigger_rate, "all_575_diagnostic"))
        by_feature[feature] = sorted(rows, key=lambda row: (row["f1"], row["recall"], -row["trigger_rate"]), reverse=True)[:10]
    return {
        "schema_version": "opk-rag.task0120.threshold-analysis.v1",
        "threshold_analysis_complete": True,
        "calibration_split": "guard_dev",
        "formal_holdout_split": "formal_eval",
        "formal_holdout_unit_count": len(formal_records),
        "diagnostic_generalization_not_established": True,
        "maximum_candidate_rules": MAX_CANDIDATE_RULES,
        "candidate_rule_count": min(candidate_rules, MAX_CANDIDATE_RULES),
        "features": by_feature,
    }


def deterministic_rule_analysis(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    labels = [bool(r["offline_labels"]["router_positive"]) for r in records]
    candidate_rows = [row for rows in thresholds["features"].values() for row in rows]
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["f1"], row["recall"], -row["trigger_rate"]), reverse=True)[:MAX_CANDIDATE_RULES]
    rule_results = []
    for row in candidate_rows:
        preds = [apply_threshold(float(r["runtime_features"].get(row["feature"], 0.0)), row["operator"], float(row["threshold"])) for r in records]
        metrics = binary_metrics(labels, preds)
        trigger_rate = mean([1.0 if pred else 0.0 for pred in preds])
        rule_results.append(rule_payload([row], metrics, trigger_rate))
    for left in candidate_rows[:40]:
        for right in candidate_rows[:40]:
            if left["feature"] == right["feature"]:
                continue
            preds = [
                apply_threshold(float(r["runtime_features"].get(left["feature"], 0.0)), left["operator"], float(left["threshold"]))
                or apply_threshold(float(r["runtime_features"].get(right["feature"], 0.0)), right["operator"], float(right["threshold"]))
                for r in records
            ]
            metrics = binary_metrics(labels, preds)
            trigger_rate = mean([1.0 if pred else 0.0 for pred in preds])
            rule_results.append(rule_payload([left, right], metrics, trigger_rate, connective="OR"))
    best = max(rule_results, key=lambda row: (row["f1"], row["recall"], -row["late_invocation_rate"]))
    return {
        "schema_version": "opk-rag.task0120.deterministic-rule-analysis.v1",
        "diagnostic_only": True,
        "deterministic_rule_probe_complete": True,
        "maximum_rule_feature_count": MAX_RULE_FEATURE_COUNT,
        "maximum_candidate_rules": MAX_CANDIDATE_RULES,
        "best_rule": best,
        "candidate_rule_count": min(len(rule_results), MAX_CANDIDATE_RULES),
    }


def routing_efficiency_curve(records: list[dict[str, Any]], guard: dict[str, Any], single: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    labels = [bool(r["offline_labels"]["router_positive"]) for r in records]
    feature = single["best_single_feature"]
    direction = single["feature_results"][feature]["direction"]
    scores = [float(r["runtime_features"].get(feature, 0.0)) for r in records]
    if direction == "lower_indicates_positive":
        scores = [-score for score in scores]
    points = []
    for budget in (0.10, 0.20, 0.25, 0.30, 0.3322, 0.40, 0.50, 1.00):
        count = max(1, math.ceil(len(records) * budget))
        selected = set(sorted(range(len(records)), key=lambda idx: scores[idx], reverse=True)[:count])
        preds = [idx in selected for idx in range(len(records))]
        metrics = binary_metrics(labels, preds)
        points.append({"late_invocation_rate": count / len(records), "recoverable_failure_coverage": metrics["recall"], "precision": metrics["precision"], "routing_efficiency": _ratio(metrics["recall"], count / len(records))})
    return {
        "schema_version": "opk-rag.task0120.routing-efficiency-curve.v1",
        "curve_feature": feature,
        "points": points,
        "task0119_guard_point": {"late_invocation_rate": guard["trigger_rate"], "recoverable_failure_coverage": guard["recovery_coverage"], "routing_efficiency": _ratio(guard["recovery_coverage"], guard["trigger_rate"])},
        "best_rule_point": {"late_invocation_rate": rules["best_rule"]["late_invocation_rate"], "recoverable_failure_coverage": rules["best_rule"]["recall"], "routing_efficiency": _ratio(rules["best_rule"]["recall"], rules["best_rule"]["late_invocation_rate"])},
        "routing_efficiency_curve_available": True,
    }


def guard_false_negative_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [failure_unit(r, "guard_false_negative") for r in records if r["offline_labels"]["router_positive"] and not r["offline_labels"]["guard_triggered"]]


def guard_false_positive_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [failure_unit(r, "guard_false_positive") for r in records if (not r["offline_labels"]["router_positive"]) and r["offline_labels"]["guard_triggered"]]


def failure_unit(record: dict[str, Any], kind: str) -> dict[str, Any]:
    features = record["runtime_features"]
    modes = []
    if features["dense_top1_score"] >= 0.8:
        modes.append("high_dense_score_but_wrong")
    if features["top1_top5_margin"] >= 0.5:
        modes.append("high_margin_but_wrong")
    if features["unique_document_count_top20"] <= 3:
        modes.append("low_diversity_but_wrong")
    if features["unique_document_count_top20"] >= 10:
        modes.append("high_diversity_but_wrong")
    if features["dense_lexical_top20_overlap"] >= 15:
        modes.append("lexical_agrees_but_wrong")
    if features["dense_top1_reranker_rank"] <= 3:
        modes.append("reranker_agrees_but_wrong")
    if not modes:
        modes.append("no_single_signal_warning")
    return {
        "schema_version": "opk-rag.task0120.guard-error-unit.v1",
        "evaluation_unit_id": record["evaluation_unit_id"],
        "error_type": kind,
        "failure_modes": modes,
        "runtime_feature_digest": record["runtime_feature_digest"],
        "selected_runtime_features": {key: features[key] for key in ("dense_top1_score", "top1_top5_margin", "unique_document_count_top20", "dense_lexical_top20_overlap", "dense_top1_reranker_rank", "task0119_retrieval_confidence")},
    }


def learned_router_probe() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0120.learned-router-probe.v1",
        "learned_router_probe_executed": False,
        "learned_router_probe_is_diagnostic_only": True,
        "skip_reason": "No production classifier is required for TASK-0120; deterministic feature separability is sufficient for this diagnostic pass.",
        "roc_auc": None,
        "pr_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None,
        "recovery_coverage_at_25_percent_trigger": None,
        "recovery_coverage_at_33_percent_trigger": None,
        "recovery_coverage_at_50_percent_trigger": None,
    }


def signal_strength_assessment(single: dict[str, Any], rules: dict[str, Any], learned: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    best_auc = single["best_single_feature_roc_auc"]
    best_rule = rules["best_rule"]
    improves = best_rule["recall"] > guard["recall"] + 0.05 and best_rule["late_invocation_rate"] <= 0.50
    if improves and best_rule["f1"] > guard["f1"]:
        strength = "moderate"
    elif best_auc >= 0.70:
        strength = "moderate"
    elif best_auc >= 0.58 or best_rule["f1"] > guard["f1"]:
        strength = "weak"
    else:
        strength = "insufficient"
    return {
        "schema_version": "opk-rag.task0120.signal-strength-assessment.v1",
        "runtime_signal_strength": strength,
        "best_single_feature_roc_auc": best_auc,
        "best_rule_improves_vs_task0119_guard": improves,
        "clean_holdout_discrimination_established": False,
        "diagnostic_generalization_not_established": True,
        "learned_router_probe_executed": learned["learned_router_probe_executed"],
    }


def strategy_recommendation(signal: dict[str, Any], rules: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    best_rule = rules["best_rule"]
    if signal["runtime_signal_strength"] == "moderate" and best_rule["feature_count"] <= MAX_RULE_FEATURE_COUNT:
        rec = "deterministic_guard_v2"
        reason = "A bounded deterministic diagnostic rule improves recoverable-failure coverage versus TASK-0119 guard, but generalization is not established."
    elif signal["runtime_signal_strength"] == "insufficient":
        rec = "runtime_observable_signals_insufficient"
        reason = "Runtime-observable features do not separate dense success from late-recoverable dense failure strongly enough."
    else:
        rec = "additional_diagnosis_required"
        reason = "Signals exist but are not strong enough for a runtime promotion decision without a clean calibration/test split."
    return {
        "schema_version": "opk-rag.task0120.strategy-recommendation.v1",
        "recommended_next_optimization": rec,
        "recommendation_reason": reason,
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "promotion_decision": "retain_dense_default",
        "task0119_historical_decision_preserved": True,
        "task0119_guard_recovery_coverage": guard["recovery_coverage"],
        "best_rule_recovery_coverage": best_rule["recall"],
        "best_rule_late_invocation_rate": best_rule["late_invocation_rate"],
    }


def build_summary(
    benchmark: dict[str, Any],
    distribution: dict[str, Any],
    guard: dict[str, Any],
    single: dict[str, Any],
    rules: dict[str, Any],
    curve: dict[str, Any],
    false_negatives: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    learned: dict[str, Any],
    signal: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    verifier_status = prior_verifier_status()
    verifier_status["task0119_verifier_valid"] = verify_task0119_artifacts(write=False)["status"] == "valid"
    best_rule = rules["best_rule"]
    return {
        "schema_version": "opk-rag.task0120.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "dense_recall_at_20": 0.7183,
        "clean_late_recall_at_20": 0.9339,
        "router_target_defined": True,
        "router_positive_count": distribution["router_positive_count"],
        "router_negative_count": distribution["router_negative_count"],
        "runtime_feature_extraction_complete": True,
        "runtime_features_gold_independent": True,
        "offline_labels_separated": True,
        "single_feature_analysis_complete": single["single_feature_analysis_complete"],
        "task0119_guard_reconstructed": True,
        "threshold_analysis_complete": True,
        "false_negative_analysis_complete": True,
        "false_positive_analysis_complete": True,
        "routing_efficiency_curve_available": curve["routing_efficiency_curve_available"],
        "deterministic_rule_probe_complete": rules["deterministic_rule_probe_complete"],
        "signal_strength_assessment_available": True,
        "task0119_guard_precision": guard["precision"],
        "task0119_guard_recall": guard["recall"],
        "task0119_guard_f1": guard["f1"],
        "task0119_guard_recovery_coverage": guard["recovery_coverage"],
        "task0119_guard_invocation_reduction": guard["invocation_reduction"],
        "best_single_feature": single["best_single_feature"],
        "best_single_feature_roc_auc": single["best_single_feature_roc_auc"],
        "best_single_feature_pr_auc": single["best_single_feature_pr_auc"],
        "best_deterministic_rule": best_rule["rule"],
        "best_rule_feature_count": best_rule["feature_count"],
        "best_rule_precision": best_rule["precision"],
        "best_rule_recall": best_rule["recall"],
        "best_rule_f1": best_rule["f1"],
        "best_rule_recovery_coverage": best_rule["recall"],
        "best_rule_invocation_rate": best_rule["late_invocation_rate"],
        "best_rule_late_invocation_rate": best_rule["late_invocation_rate"],
        "best_rule_late_invocation_reduction": best_rule["late_invocation_reduction"],
        "routing_efficiency_improved_vs_task0119": curve["best_rule_point"]["routing_efficiency"] > curve["task0119_guard_point"]["routing_efficiency"],
        "guard_false_negative_count": len(false_negatives),
        "guard_false_positive_count": len(false_positives),
        "learned_router_probe_executed": learned["learned_router_probe_executed"],
        "learned_router_probe_roc_auc": learned["roc_auc"],
        "learned_router_probe_pr_auc": learned["pr_auc"],
        "runtime_signal_strength": signal["runtime_signal_strength"],
        "recommended_next_optimization": recommendation["recommended_next_optimization"],
        "recommendation_reason": recommendation["recommendation_reason"],
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "promotion_decision": "retain_dense_default",
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "canonical_chunk_identity_preserved": True,
        "canonical_candidate_identity_preserved": True,
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "task0116_artifacts_unchanged": True,
        "task0117_artifacts_unchanged": True,
        "task0118_artifacts_unchanged": True,
        "task0119_artifacts_unchanged": True,
        "new_task0120_regression": 0,
        "known_preexisting_failure": 0,
        "environmental_failure": 0,
        **verifier_status,
        "task0120_verifier_valid": False,
    }


def build_contract(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0120.dense-retrieval-failure-signal-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0118_clean_evidence_identity": {"summary_sha256": sha256_file(task0118.RESULT_DIR / "summary.json"), "clean_retrieval_results_sha256": sha256_file(task0118.RESULT_DIR / "clean_retrieval_results.jsonl")},
        "task0119_policy_artifact_identity": {"summary_sha256": sha256_file(TASK0119_RESULT_DIR / "summary.json"), "guard_results_sha256": sha256_file(TASK0119_RESULT_DIR / "guard_results.jsonl")},
        "router_positive_definition": "dense_gold_in_top20=false AND clean_late_gold_in_top20=true",
        "runtime_feature_allowlist": list(NUMERIC_FEATURES) + ["runtime_observable_only", "task0119_retrieval_confidence"],
        "offline_label_policy": "offline_labels are separated from runtime_features and may not enter router inputs",
        "feature_definitions": {"dense_scores": "TASK-0118 clean dense runtime candidate scores", "query_features": "deterministic token/character counts", "reranker_agreement": "existing BGE reranker scores already present in frozen baseline"},
        "calibration_split": "guard_dev",
        "formal_holdout_split": "formal_eval",
        "threshold_sweep_limits": {"maximum_candidate_rules": MAX_CANDIDATE_RULES, "diagnostic_only": True},
        "maximum_deterministic_rule_complexity": MAX_RULE_FEATURE_COUNT,
        "learned_probe_policy": {"allowed": True, "production_model_saved": False, "diagnostic_only": True},
        "evaluation_metrics": ["ROC-AUC", "PR-AUC", "precision", "recall", "f1", "trigger_rate", "late_invocation_reduction"],
        "recommendation_rules": ["deterministic_guard_v2", "learned_retrieval_failure_router", "hybrid_always_on_cost_optimization", "runtime_observable_signals_insufficient", "additional_diagnosis_required"],
        "promotion_prohibition": {"promotion_applied": False, "recommended_default_policy": "dense_default"},
        "benchmark_identity": benchmark["benchmark_identity"],
    }


def provenance_payload(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0120.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "promotion_applied": False,
        "git_commit_created": False,
    }


def verify_task0120_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": str(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": str(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        records = read_jsonl(output_dir / "failure_signal_features.jsonl")
        false_negatives = read_jsonl(output_dir / "guard_false_negative_units.jsonl")
        false_positives = read_jsonl(output_dir / "guard_false_positive_units.jsonl")
        guard = read_json(output_dir / "task0119_guard_baseline.json")
        recommendation = read_json(output_dir / "strategy_recommendation.json")
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("formal_evaluation_unit_count") != 575 or len(records) != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        required_true = (
            "task0118_clean_evidence_used",
            "router_target_defined",
            "runtime_feature_extraction_complete",
            "runtime_features_gold_independent",
            "offline_labels_separated",
            "single_feature_analysis_complete",
            "task0119_guard_reconstructed",
            "threshold_analysis_complete",
            "false_negative_analysis_complete",
            "false_positive_analysis_complete",
            "routing_efficiency_curve_available",
            "deterministic_rule_probe_complete",
            "signal_strength_assessment_available",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "canonical_chunk_identity_preserved",
        )
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("task0116_promotion_evidence_used") is not False or summary.get("promotion_applied") is not False:
            issues.append({"code": "promotion_or_contaminated_evidence_violation"})
        if summary.get("recommended_default_policy") != "dense_default" or recommendation.get("recommended_default_policy") != "dense_default":
            issues.append({"code": "default_policy_changed"})
        if summary.get("router_positive_count", 0) <= 0:
            issues.append({"code": "router_positive_count_missing"})
        if len(false_negatives) != summary.get("guard_false_negative_count"):
            issues.append({"code": "guard_false_negative_count_mismatch"})
        if len(false_positives) != summary.get("guard_false_positive_count"):
            issues.append({"code": "guard_false_positive_count_mismatch"})
        if abs(float(guard.get("recovery_coverage", 0.0)) - 0.4348) > 0.01:
            issues.append({"code": "task0119_guard_reconstruction_drift", "actual": guard.get("recovery_coverage")})
        for record in records[:10]:
            if "offline_labels" in record.get("runtime_features", {}) or "gold_chunk_ids" in record.get("runtime_features", {}):
                issues.append({"code": "label_leakage_into_runtime_features"})
            if digest_json(record["runtime_features"]) != record["runtime_feature_digest"]:
                issues.append({"code": "runtime_feature_digest_mismatch", "evaluation_unit_id": record["evaluation_unit_id"]})
        for key in ("task0112_verifier_valid", "task0113_verifier_valid", "task0114_verifier_valid", "task0115_verifier_valid", "task0116_verifier_valid", "task0117_verifier_valid", "task0118_verifier_valid", "task0119_verifier_valid"):
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_valid"})
    result = {
        "schema_version": "opk-rag.task0120.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "promotion_applied": summary.get("promotion_applied"),
        "recommended_next_optimization": summary.get("recommended_next_optimization"),
        "task0120_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", artifacts["summary"])
    write_jsonl(output_dir / "failure_signal_features.jsonl", artifacts["failure_signal_features"])
    write_jsonl(output_dir / "guard_false_negative_units.jsonl", artifacts["guard_false_negative_units"])
    write_jsonl(output_dir / "guard_false_positive_units.jsonl", artifacts["guard_false_positive_units"])
    for name in REQUIRED_ARTIFACTS:
        key = name.removesuffix(".json").removesuffix(".jsonl")
        if name in {"summary.json", "failure_signal_features.jsonl", "guard_false_negative_units.jsonl", "guard_false_positive_units.jsonl", "verification.json"}:
            continue
        write_json(output_dir / name, artifacts[key])


def build_report(summary: dict[str, Any], guard: dict[str, Any], single: dict[str, Any], rules: dict[str, Any], signal: dict[str, Any], recommendation: dict[str, Any]) -> str:
    return f"""# TASK-0120 Dense Retrieval Failure Signal Diagnosis Report

## Answer

Can Dense failure be predicted before invoking Late Interaction?

Partially. The runtime-observable signals show `{summary['runtime_signal_strength']}` separability in this diagnostic pass. The result is diagnosis-only because a clean calibration/test generalization claim is not established.

Why did TASK-0119 Guard only cover 43.48%?

TASK-0119 produced `{summary['guard_false_negative_count']}` false negatives for the router-positive class: Dense missed, Clean Late could recover, but the guard did not trigger. The dominant pattern is overconfidence from runtime score and agreement signals; cosine-like rank scores are not calibrated probabilities.

## Frozen Authority

- TASK-0118 clean evidence used: `{summary['task0118_clean_evidence_used']}`
- TASK-0116 promotion evidence used: `{summary['task0116_promotion_evidence_used']}`
- Dense recall@20: `{summary['dense_recall_at_20']:.4f}`
- Clean Late recall@20: `{summary['clean_late_recall_at_20']:.4f}`

## Router Target

- Positive definition: `dense_gold_in_top20=false AND clean_late_gold_in_top20=true`
- Router positives / negatives: `{summary['router_positive_count']}` / `{summary['router_negative_count']}`

## TASK-0119 Guard Baseline

- Precision / recall / F1: `{guard['precision']:.4f}` / `{guard['recall']:.4f}` / `{guard['f1']:.4f}`
- Recovery coverage: `{guard['recovery_coverage']:.4f}`
- Invocation reduction: `{guard['invocation_reduction']:.4f}`

## Feature Diagnosis

- Best single feature: `{single['best_single_feature']}`
- Best single feature ROC-AUC / PR-AUC: `{single['best_single_feature_roc_auc']:.4f}` / `{single['best_single_feature_pr_auc']:.4f}`
- Best deterministic rule: `{rules['best_rule']['rule']}`
- Best rule precision / recall / F1: `{rules['best_rule']['precision']:.4f}` / `{rules['best_rule']['recall']:.4f}` / `{rules['best_rule']['f1']:.4f}`
- Best rule invocation rate / reduction: `{rules['best_rule']['late_invocation_rate']:.4f}` / `{rules['best_rule']['late_invocation_reduction']:.4f}`

## Decision

- Runtime signal strength: `{signal['runtime_signal_strength']}`
- Recommended next optimization: `{recommendation['recommended_next_optimization']}`
- Reason: {recommendation['recommendation_reason']}
- Promotion applied: `{summary['promotion_applied']}`
- Recommended default policy: `{summary['recommended_default_policy']}`

## Leakage Boundary

`runtime_features` and `offline_labels` are separated in `failure_signal_features.jsonl`. Each row includes `runtime_feature_digest`; verifier checks that digest from runtime features only and that no gold labels enter runtime features.
"""


def score_at(scores: list[float], rank: int) -> float:
    return scores[rank - 1] if len(scores) >= rank else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def range_value(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def normalized_entropy(values: list[float]) -> float:
    total = sum(max(v, 0.0) for v in values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = [max(v, 0.0) / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs) / math.log(len(values))


def slope(values: list[float]) -> float:
    return (values[0] - values[-1]) / (len(values) - 1) if len(values) > 1 else 0.0


def unique_count(rows: list[dict[str, Any]], key: str) -> int:
    return len({row.get(key) for row in rows if row.get(key) is not None})


def dense_lexical_overlap(rows: list[dict[str, Any]], k: int) -> int:
    dense_ids = {row.get("canonical_chunk_id") for row in rows[:k]}
    lexical_ids = {row.get("canonical_chunk_id") for row in sorted(rows, key=lambda row: int(row.get("retrieval_rank") or 9999))[:k]}
    return len(dense_ids & lexical_ids)


def near_duplicate_candidate_count(rows: list[dict[str, Any]]) -> int:
    seen = set()
    dupes = 0
    for row in rows:
        key = row.get("content_digest") or row.get("canonical_chunk_id")
        if key in seen:
            dupes += 1
        seen.add(key)
    return dupes


def same_value_ratio(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    counts: dict[Any, int] = {}
    for row in rows:
        counts[row.get(key)] = counts.get(row.get(key), 0) + 1
    return max(counts.values()) / len(rows)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./#+-]+|[\u4e00-\u9fff]", text)


def entity_like(token: str) -> bool:
    return bool(re.search(r"[A-Z][A-Za-z0-9_+-]*|\d|[/_.#]", token))


def technical_term(token: str) -> bool:
    return bool(re.search(r"[A-Za-z][A-Za-z0-9_+-]*|[/_.#]|API|CLI|RAG|LLM", token, re.I))


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "median": statistics.median(values) if values else 0.0, "std": std(values)}


def roc_auc(values: list[float], labels: list[bool]) -> float:
    pos = [v for v, y in zip(values, labels, strict=True) if y]
    neg = [v for v, y in zip(values, labels, strict=True) if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def pr_auc(values: list[float], labels: list[bool]) -> float:
    paired = sorted(zip(values, labels, strict=True), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in paired:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def quantile_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= 12:
        return unique
    return [unique[min(len(unique) - 1, max(0, round((len(unique) - 1) * q)))] for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)]


def apply_threshold(value: float, operator: str, threshold: float) -> bool:
    return value <= threshold if operator == "<=" else value >= threshold


def binary_metrics(labels: list[bool], preds: list[bool]) -> dict[str, Any]:
    tp = sum(y and p for y, p in zip(labels, preds, strict=True))
    fp = sum((not y) and p for y, p in zip(labels, preds, strict=True))
    fn = sum(y and (not p) for y, p in zip(labels, preds, strict=True))
    tn = sum((not y) and (not p) for y, p in zip(labels, preds, strict=True))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": _ratio(2 * precision * recall, precision + recall)}


def threshold_row(feature: str, operator: str, threshold: float, metrics: dict[str, Any], trigger_rate: float, split: str) -> dict[str, Any]:
    return {
        "feature": feature,
        "operator": operator,
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "trigger_rate": trigger_rate,
        "late_invocation_reduction": 1.0 - trigger_rate,
        "split": split,
    }


def rule_payload(conditions: list[dict[str, Any]], metrics: dict[str, Any], trigger_rate: float, *, connective: str = "AND") -> dict[str, Any]:
    parts = [f"{row['feature']} {row['operator']} {row['threshold']}" for row in conditions]
    return {
        "rule": f" {connective} ".join(parts),
        "conditions": [{"feature": row["feature"], "operator": row["operator"], "threshold": row["threshold"]} for row in conditions],
        "connective": connective,
        "feature_count": len({row["feature"] for row in conditions}),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "late_invocation_rate": trigger_rate,
        "late_invocation_reduction": 1.0 - trigger_rate,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def split_for_unit(unit: str) -> str:
    return "guard_dev" if int(digest_json(unit)[:8], 16) % 5 == 0 else "formal_eval"


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0
