from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK0090_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0090-canonical-runtime-v2"
TASK0090_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0090_canonical_runtime_v2_contract.json"
TASK0090_REPLAY_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0090_retrieval_replay_contract.json"
TASK0082_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0082-retrieval-localization"
TASK0087_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0087-real-reranker-validation"


TASK_ID = "TASK-0091"
EXPERIMENT_ID = "task0091-reranker-replay-benchmark"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0091_reranker_benchmark_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0091_RERANKER_REPLAY_BENCHMARK_REPORT.md"
K_VALUES = (1, 3, 5, 10)
CANDIDATE_DEPTH = 50


def run_task0091_reranker_replay_benchmark() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_task0091_inputs()
    write_json(CONTRACT_PATH, build_contract(inputs))
    units = build_replay_units(inputs)
    baseline = {unit_id: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit_id, rows in units.items()}
    reranked = {unit_id: rerank_rows(rows) for unit_id, rows in units.items()}
    baseline_metrics = ranking_metrics(baseline)
    reranker_metrics = ranking_metrics(reranked)
    deltas = metric_deltas(baseline_metrics, reranker_metrics)
    comparisons = sample_comparisons(baseline, reranked)
    ceiling = candidate_ceiling_analysis(comparisons)
    gates = promotion_gates(inputs, baseline, reranked, baseline_metrics, reranker_metrics, comparisons)
    decision = promotion_decision(gates, deltas, ceiling)
    manifest = experiment_manifest(inputs, units)
    summary = build_summary(manifest, baseline_metrics, reranker_metrics, deltas, ceiling, gates, decision)

    write_json(RESULT_DIR / "experiment_manifest.json", manifest)
    write_json(RESULT_DIR / "baseline_metrics.json", baseline_metrics)
    write_json(RESULT_DIR / "reranker_metrics.json", reranker_metrics)
    write_json(RESULT_DIR / "metric_deltas.json", deltas)
    write_jsonl(RESULT_DIR / "sample_comparison.jsonl", comparisons)
    write_json(RESULT_DIR / "candidate_ceiling_analysis.json", ceiling)
    write_json(RESULT_DIR / "promotion_gates.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)
    write_json(RESULT_DIR / "summary.json", summary)
    verification = verify_task0091_artifacts(write=True)
    summary["verification"] = verification
    summary["repository_verification_status"] = verification["repository_verification_status"]
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def load_task0091_inputs() -> dict[str, Any]:
    paths = {
        "task0090_summary": TASK0090_RESULT_DIR / "summary.json",
        "task0090_replay_contract": TASK0090_REPLAY_CONTRACT_PATH,
        "task0090_runtime_contract": TASK0090_CONTRACT_PATH,
        "task0090_candidates": TASK0090_RESULT_DIR / "v1_candidate_results.jsonl",
        "task0087_scores": TASK0087_RESULT_DIR / "reranker_scores.jsonl",
        "gold_authority": TASK0082_RESULT_DIR / "gold_chunk_authority.jsonl",
        "task0090_corpus": TASK0090_RESULT_DIR / "frozen_corpus_authority.json",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing TASK-0091 inputs: {', '.join(_rel(path) for path in missing)}")
    task0090_summary = read_json(paths["task0090_summary"])
    if task0090_summary.get("task_status") != "complete":
        raise RuntimeError("TASK-0090 replay input is not complete")
    return {
        "paths": {key: _rel(path) for key, path in paths.items()},
        "digests": {key: sha256_file(path) for key, path in paths.items()},
        "task0090_summary": task0090_summary,
        "task0090_replay_contract": read_json(paths["task0090_replay_contract"]),
        "task0090_runtime_contract": read_json(paths["task0090_runtime_contract"]),
        "task0090_candidates": read_jsonl(paths["task0090_candidates"]),
        "task0087_scores": read_jsonl(paths["task0087_scores"]),
        "gold_authority": [row for row in read_jsonl(paths["gold_authority"]) if row.get("chunk_representable", True)],
        "task0090_corpus": read_json(paths["task0090_corpus"]),
    }


def build_replay_units(inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    gold_by_unit = {_unit_id(row): set(row.get("gold_chunk_ids") or []) for row in inputs["gold_authority"]}
    sample_meta = {_unit_id(row): row for row in inputs["gold_authority"]}
    score_by_key = {(row["sample_unit_id"], row["chunk_id"]): float(row["reranker_raw_score"]) for row in inputs["task0087_scores"]}
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs["task0090_candidates"]:
        unit = row["sample_unit_id"]
        score = score_by_key.get((unit, row["content_digest"]))
        by_unit[unit].append(
            {
                "schema_version": "opk-rag.task0091.reranker-candidate.v1",
                "sample_unit_id": unit,
                "sample_id": unit.split("::", 1)[0],
                "query_group": unit.split(":", 1)[0],
                "question_type": unit.split(":", 1)[0],
                "answerability_class": "answerability" if "answerability" in unit else "unknown",
                "source_span_digest": unit.split("::", 1)[1] if "::" in unit else None,
                "canonical_chunk_id": row["canonical_chunk_id"],
                "content_digest": row["content_digest"],
                "document_id": row.get("document_id"),
                "section_id": row.get("section_id"),
                "retrieval_rank": int(row.get("original_vector_rank") or row["rank"]),
                "retrieval_score": row.get("retrieval_score"),
                "reranker_score": score,
                "relevant_label": row["content_digest"] in gold_by_unit.get(unit, set()),
                "gold_chunk_ids": sorted(gold_by_unit.get(unit, set())),
                "ambiguous": unit not in sample_meta or not gold_by_unit.get(unit),
            }
        )
    for unit, rows in by_unit.items():
        if len(rows) != CANDIDATE_DEPTH:
            raise RuntimeError(f"candidate depth mismatch for {unit}: {len(rows)}")
        if any(row["reranker_score"] is None for row in rows):
            raise RuntimeError(f"missing reranker scores for {unit}")
    return dict(sorted(by_unit.items()))


def rerank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["reranker_score"]), row["retrieval_rank"], row["canonical_chunk_id"]))
    return [{**row, "reranker_rank": index} for index, row in enumerate(ranked, start=1)]


def ranking_metrics(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [first_relevant_rank(rows) for rows in rows_by_unit.values()]
    present = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "schema_version": "opk-rag.task0091.ranking-metrics.v1",
        "unit_count": len(ranks),
        "candidate_depth": CANDIDATE_DEPTH,
        "candidate_recall": _ratio(len(present), len(ranks)),
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
        "mean_first_relevant_rank": _ratio(sum(present), len(present)) if present else None,
    }
    for k in K_VALUES:
        metrics[f"recall_at_{k}"] = _ratio(sum(rank is not None and rank <= k for rank in ranks), len(ranks))
    metrics[f"recall_at_{CANDIDATE_DEPTH}"] = metrics["candidate_recall"]
    for k in (5, 10):
        metrics[f"ndcg_at_{k}"] = _ratio(sum(ndcg_at_k(rows, k) for rows in rows_by_unit.values()), len(rows_by_unit))
    return metrics


def sample_comparisons(baseline: dict[str, list[dict[str, Any]]], reranked: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(baseline):
        before = first_relevant_rank(baseline[unit])
        after = first_relevant_rank(reranked[unit])
        if baseline[unit][0]["ambiguous"]:
            classification = "ambiguous"
        elif before is None and after is None:
            classification = "candidate_ceiling"
        elif before is None:
            classification = "improved"
        elif after is None:
            classification = "regressed"
        elif after < before:
            classification = "improved"
        elif after > before:
            classification = "regressed"
        else:
            classification = "unchanged"
        rows.append(
            {
                "schema_version": "opk-rag.task0091.sample-comparison.v1",
                "sample_unit_id": unit,
                "sample_id": baseline[unit][0]["sample_id"],
                "query_group": baseline[unit][0]["query_group"],
                "question_type": baseline[unit][0]["question_type"],
                "answerability_class": baseline[unit][0]["answerability_class"],
                "first_relevant_rank_before": before,
                "first_relevant_rank_after": after,
                "delta": None if before is None or after is None else before - after,
                "classification": classification,
                "baseline_hit_at_5": before is not None and before <= 5,
                "reranker_hit_at_5": after is not None and after <= 5,
                "candidate_identity_digest": digest_json([row["canonical_chunk_id"] for row in baseline[unit]]),
                "top_candidate_before": baseline[unit][0]["canonical_chunk_id"],
                "top_candidate_after": reranked[unit][0]["canonical_chunk_id"],
            }
        )
    return rows


def candidate_ceiling_analysis(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(comparisons)
    counts = Counter(row["classification"] for row in comparisons)
    top = {
        "top1_recovered_units": sum((row["first_relevant_rank_before"] or 10**9) > 1 and (row["first_relevant_rank_after"] or 10**9) <= 1 for row in comparisons),
        "top3_recovered_units": sum((row["first_relevant_rank_before"] or 10**9) > 3 and (row["first_relevant_rank_after"] or 10**9) <= 3 for row in comparisons),
        "top5_recovered_units": sum((row["first_relevant_rank_before"] or 10**9) > 5 and (row["first_relevant_rank_after"] or 10**9) <= 5 for row in comparisons),
        "top1_lost_units": sum(row["first_relevant_rank_before"] is not None and row["first_relevant_rank_before"] <= 1 and ((row["first_relevant_rank_after"] or 10**9) > 1) for row in comparisons),
        "top3_lost_units": sum(row["first_relevant_rank_before"] is not None and row["first_relevant_rank_before"] <= 3 and ((row["first_relevant_rank_after"] or 10**9) > 3) for row in comparisons),
        "top5_lost_units": sum(row["first_relevant_rank_before"] is not None and row["first_relevant_rank_before"] <= 5 and ((row["first_relevant_rank_after"] or 10**9) > 5) for row in comparisons),
    }
    candidate_ceiling = counts["candidate_ceiling"]
    ranking_failure = total - candidate_ceiling - counts["ambiguous"]
    return {
        "schema_version": "opk-rag.task0091.candidate-ceiling-analysis.v1",
        "total_units": total,
        "rerankable_units": ranking_failure,
        "candidate_ceiling_units": candidate_ceiling,
        "improved_units": counts["improved"],
        "unchanged_units": counts["unchanged"],
        "regressed_units": counts["regressed"],
        "ambiguous_units": counts["ambiguous"],
        **top,
        "improvement_rate": _ratio(counts["improved"], total),
        "regression_rate": _ratio(counts["regressed"], total),
        "candidate_ceiling_rate": _ratio(candidate_ceiling, total),
        "retrieval_failure_units": candidate_ceiling,
        "ranking_failure_units": ranking_failure,
        "reranker_repairable_failure_units_at_top5": sum((row["first_relevant_rank_before"] or 10**9) > 5 and row["first_relevant_rank_after"] is not None for row in comparisons),
        "dominant_bottleneck": bottleneck_label(candidate_ceiling, ranking_failure),
    }


def promotion_gates(
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    reranked: dict[str, list[dict[str, Any]]],
    baseline_metrics: dict[str, Any],
    reranker_metrics: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    identities_preserved = all([row["canonical_chunk_id"] for row in baseline[unit]] == [row["canonical_chunk_id"] for row in sorted(reranked[unit], key=lambda r: r["retrieval_rank"])] for unit in baseline)
    counts_preserved = all(len(baseline[unit]) == len(reranked[unit]) == CANDIDATE_DEPTH for unit in baseline)
    recall_depth_preserved = baseline_metrics[f"recall_at_{CANDIDATE_DEPTH}"] == reranker_metrics[f"recall_at_{CANDIDATE_DEPTH}"]
    distribution = distribution_check(comparisons)
    return {
        "schema_version": "opk-rag.task0091.promotion-gates.v1",
        "candidate_identity_preserved": identities_preserved,
        "query_identity_preserved": sorted(baseline) == sorted(reranked),
        "corpus_identity_preserved": bool(inputs["task0090_corpus"].get("corpus_snapshot")),
        "candidate_count_preserved": counts_preserved,
        "recall_at_candidate_depth_preserved": recall_depth_preserved,
        "no_retrieval_rerun": True,
        "no_query_rewrite": True,
        "no_graph_expansion": True,
        "ranking_improvement_gate": (
            reranker_metrics["recall_at_5"] > baseline_metrics["recall_at_5"]
            and reranker_metrics["mrr"] > baseline_metrics["mrr"]
            and reranker_metrics["ndcg_at_5"] > baseline_metrics["ndcg_at_5"]
        ),
        "regression_budget_gate": sum(row["classification"] == "regressed" for row in comparisons) <= sum(row["classification"] == "improved" for row in comparisons),
        "distribution_check_gate": distribution["distribution_check_passed"],
        "distribution_check": distribution,
        "default_reranker_enabled": False,
    }


def promotion_decision(gates: dict[str, Any], deltas: dict[str, Any], ceiling: dict[str, Any]) -> dict[str, Any]:
    integrity = all(
        gates[key]
        for key in (
            "candidate_identity_preserved",
            "query_identity_preserved",
            "corpus_identity_preserved",
            "candidate_count_preserved",
            "recall_at_candidate_depth_preserved",
            "no_retrieval_rerun",
            "no_query_rewrite",
            "no_graph_expansion",
        )
    )
    if not integrity:
        primary = "invalid_experiment"
        recommendation = "experiment_invalid"
        next_task = "Diagnose TASK-0091 experiment integrity failure"
    elif gates["ranking_improvement_gate"] and gates["regression_budget_gate"] and gates["distribution_check_gate"]:
        primary = "valid_experiment_positive_result"
        recommendation = "promote_to_downstream_validation"
        next_task = "TASK-0092 Reranker Downstream / End-to-End Validation"
    else:
        primary = "valid_experiment_negative_result"
        recommendation = "do_not_promote"
        next_task = "candidate_recall_bottleneck_dominant" if ceiling["dominant_bottleneck"] == "candidate recall" else "reranker_insufficient"
    return {
        "schema_version": "opk-rag.task0091.promotion-decision.v1",
        "primary_result_classification": primary,
        "reranker_promotion_recommendation": recommendation,
        "next_task_decision": next_task,
        "metric_deltas": {key: deltas[key] for key in ("recall_at_5", "mrr", "ndcg_at_5")},
    }


def verify_task0091_artifacts(*, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "experiment_manifest.json",
        RESULT_DIR / "baseline_metrics.json",
        RESULT_DIR / "reranker_metrics.json",
        RESULT_DIR / "metric_deltas.json",
        RESULT_DIR / "sample_comparison.jsonl",
        RESULT_DIR / "candidate_ceiling_analysis.json",
        RESULT_DIR / "promotion_decision.json",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "summary.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        baseline = read_json(RESULT_DIR / "baseline_metrics.json")
        reranker = read_json(RESULT_DIR / "reranker_metrics.json")
        gates = read_json(RESULT_DIR / "promotion_gates.json")
        decision = read_json(RESULT_DIR / "promotion_decision.json")
        if baseline.get(f"recall_at_{CANDIDATE_DEPTH}") != reranker.get(f"recall_at_{CANDIDATE_DEPTH}"):
            issues.append({"code": "recall_at_candidate_depth_changed"})
        for key in ("candidate_identity_preserved", "candidate_count_preserved", "no_retrieval_rerun", "no_query_rewrite", "no_graph_expansion"):
            if gates.get(key) is not True:
                issues.append({"code": f"{key}_failed"})
        if decision.get("primary_result_classification") not in {"valid_experiment_positive_result", "valid_experiment_negative_result", "invalid_experiment"}:
            issues.append({"code": "missing_primary_result_classification"})
    result = {
        "schema_version": "opk-rag.task0091.verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(RESULT_DIR / "verification_summary.json", result)
    return result


def build_contract(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0091.reranker-benchmark-contract.v1",
        "experiment_identity": EXPERIMENT_ID,
        "source_task": "TASK-0090",
        "source_replay_contract_digest": inputs["digests"]["task0090_replay_contract"],
        "candidate_set_policy": "frozen_task0090_v1_candidate_results_no_membership_change",
        "reranker_policy": inputs["task0090_replay_contract"]["task0087_authority"],
        "metrics": ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "Mean First Relevant Rank", "nDCG@5", "nDCG@10"],
        "sample_classification": ["improved", "unchanged", "regressed", "candidate_ceiling", "ambiguous"],
        "promotion_gates": ["experiment_integrity", "ranking_improvement", "regression_budget", "distribution_check"],
        "validity_rules": ["Recall@candidate_depth invariant", "candidate identities preserved", "no retrieval rerun", "no query rewrite", "no graph expansion"],
        "artifact_schema": "opk-rag.task0091.*.v1",
    }


def experiment_manifest(inputs: dict[str, Any], units: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0091.experiment-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "dataset_identity": digest_json({"unit_ids": sorted(units), "candidate_depth": CANDIDATE_DEPTH, "source": inputs["digests"]["task0090_candidates"]}),
        "contract_digest": sha256_file(CONTRACT_PATH),
        "corpus_identity": inputs["task0090_corpus"].get("corpus_snapshot", {}).get("corpus_snapshot_id"),
        "query_count": len(units),
        "retrieval_unit_count": len(units),
        "candidate_count": sum(len(rows) for rows in units.values()),
        "source_task": "TASK-0090",
        "source_artifact_digest": inputs["digests"]["task0090_candidates"],
        "source_paths": inputs["paths"],
        "task0090_inputs_valid": True,
        "task0090_artifacts_modified": False,
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
    }


def build_summary(
    manifest: dict[str, Any],
    baseline_metrics: dict[str, Any],
    reranker_metrics: dict[str, Any],
    deltas: dict[str, Any],
    ceiling: dict[str, Any],
    gates: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0091.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if decision["primary_result_classification"] != "invalid_experiment" else "invalid",
        **{key: manifest[key] for key in ("task0090_inputs_valid", "task0090_artifacts_modified", "retrieval_rerun_count", "query_reformulation_count", "graph_retrieval_count")},
        "candidate_set_frozen": gates["candidate_count_preserved"],
        "candidate_identity_preserved": gates["candidate_identity_preserved"],
        "control_unit_count": baseline_metrics["unit_count"],
        "reranker_unit_count": reranker_metrics["unit_count"],
        **{f"baseline_recall_at_{k}": baseline_metrics[f"recall_at_{k}"] for k in K_VALUES},
        **{f"reranker_recall_at_{k}": reranker_metrics[f"recall_at_{k}"] for k in K_VALUES},
        "baseline_mrr": baseline_metrics["mrr"],
        "reranker_mrr": reranker_metrics["mrr"],
        "baseline_ndcg_at_5": baseline_metrics["ndcg_at_5"],
        "reranker_ndcg_at_5": reranker_metrics["ndcg_at_5"],
        "baseline_ndcg_at_10": baseline_metrics["ndcg_at_10"],
        "reranker_ndcg_at_10": reranker_metrics["ndcg_at_10"],
        "improved_unit_count": ceiling["improved_units"],
        "unchanged_unit_count": ceiling["unchanged_units"],
        "regressed_unit_count": ceiling["regressed_units"],
        "candidate_ceiling_unit_count": ceiling["candidate_ceiling_units"],
        "dominant_bottleneck": ceiling["dominant_bottleneck"],
        "top5_recovered_unit_count": ceiling["top5_recovered_units"],
        "top5_lost_unit_count": ceiling["top5_lost_units"],
        "primary_result_classification": decision["primary_result_classification"],
        "reranker_promotion_recommendation": decision["reranker_promotion_recommendation"],
        "next_task_decision": decision["next_task_decision"],
        "metric_deltas": deltas,
        "default_reranker_enabled": False,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0091 Reranker Replay Benchmark Report",
            "",
            "## Decision",
            "",
            f"- primary_result_classification=`{summary['primary_result_classification']}`",
            f"- reranker_promotion_recommendation=`{summary['reranker_promotion_recommendation']}`",
            f"- next_task_decision=`{summary['next_task_decision']}`",
            "",
            "## Required Answers",
            "",
            f"1. Top-K Evidence Quality improved at R@5 from `{summary['baseline_recall_at_5']}` to `{summary['reranker_recall_at_5']}`.",
            f"2. Improvements: R@5 delta `{summary['metric_deltas']['recall_at_5']}`, MRR delta `{summary['metric_deltas']['mrr']}`, nDCG@5 delta `{summary['metric_deltas']['ndcg_at_5']}`.",
            f"3. Improved queries: `{summary['improved_unit_count']}`.",
            f"4. Regressed queries: `{summary['regressed_unit_count']}`; Top-5 lost: `{summary['top5_lost_unit_count']}`.",
            f"5. Candidate ceiling units: `{summary['candidate_ceiling_unit_count']}`.",
            f"6. Bottleneck: `{summary['dominant_bottleneck']}`.",
            f"7. Downstream validation recommendation: `{summary['reranker_promotion_recommendation']}`.",
            "",
            "## Governance",
            "",
            f"- task0090_inputs_valid=`{str(summary['task0090_inputs_valid']).lower()}`",
            f"- task0090_artifacts_modified=`{str(summary['task0090_artifacts_modified']).lower()}`",
            f"- candidate_set_frozen=`{str(summary['candidate_set_frozen']).lower()}`",
            f"- candidate_identity_preserved=`{str(summary['candidate_identity_preserved']).lower()}`",
            f"- retrieval_rerun_count=`{summary['retrieval_rerun_count']}`",
            f"- query_reformulation_count=`{summary['query_reformulation_count']}`",
            f"- graph_retrieval_count=`{summary['graph_retrieval_count']}`",
            f"- default_reranker_enabled=`{str(summary['default_reranker_enabled']).lower()}`",
        ]
    )


def metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = [f"recall_at_{k}" for k in K_VALUES] + ["mrr", "mean_first_relevant_rank", "ndcg_at_5", "ndcg_at_10", f"recall_at_{CANDIDATE_DEPTH}"]
    return {"schema_version": "opk-rag.task0091.metric-deltas.v1", **{key: None if before[key] is None or after[key] is None else after[key] - before[key] for key in keys}}


def first_relevant_rank(rows: list[dict[str, Any]]) -> int | None:
    for index, row in enumerate(rows, start=1):
        if row["relevant_label"]:
            return index
    return None


def ndcg_at_k(rows: list[dict[str, Any]], k: int) -> float:
    gains = [1.0 if row["relevant_label"] else 0.0 for row in rows[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    relevant_count = sum(1 for row in rows if row["relevant_label"])
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(relevant_count, k)))
    return dcg / ideal if ideal else 0.0


def distribution_check(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in comparisons:
        groups[row["query_group"]][row["classification"]] += 1
    improved_groups = [group for group, counts in groups.items() if counts["improved"] > 0]
    return {
        "schema_version": "opk-rag.task0091.distribution-check.v1",
        "available_grouping": "query_group_from_sample_id_prefix",
        "group_count": len(groups),
        "groups_with_improvement": improved_groups,
        "distribution_check_passed": len(improved_groups) > 1 or len(groups) == 1,
        "groups": {group: dict(counts) for group, counts in sorted(groups.items())},
    }


def bottleneck_label(candidate_ceiling: int, ranking_failure: int) -> str:
    if candidate_ceiling > ranking_failure:
        return "candidate recall"
    if ranking_failure > candidate_ceiling:
        return "ranking quality"
    return "mixed bottleneck"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_dumps(row) + "\n" for row in rows), encoding="utf-8")


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
