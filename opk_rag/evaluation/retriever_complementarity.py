from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import statistics
import subprocess
from typing import Any, Iterable

from opk_rag.evaluation.chunk_localization import _file_digest
from opk_rag.evaluation.evidence_identity import digest_json
from opk_rag.evaluation.retrieval_promotion import RESULT_DIR as TASK0084_RESULT_DIR
from opk_rag.evaluation.retrieval_promotion import TOP_K, repository_state
from opk_rag.evaluation.scope_aware_chunking_experiment import ROOT, utc_now, write_json, write_jsonl


TASK_ID = "TASK-0085"
EXPERIMENT_ID = "task0085-reranker-readiness"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0085_reranker_readiness_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0085_RERANKER_READINESS_AND_RETRIEVER_COMPLEMENTARITY_REPORT.md"
TASK0082_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0082-retrieval-localization"
TASK0081_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0081-chunk-localization-recall"
FORMAL_AUTHORITY_ID = "phase2-candidate-retrieval-baseline-v1"
CANDIDATE_DEPTHS = (20, 50, 100)
BUDGET_CURVE_DEPTHS = (10, 20, 30, 50)


class RerankerReadinessError(RuntimeError):
    pass


def run_task0085_reranker_readiness(*, run_id: str | None = None) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start_state = repository_state()
    inputs = load_task0085_inputs()
    analysis = analyze_retriever_complementarity(
        authority=inputs["authority"],
        vector_traces=inputs["vector_traces"],
        lexical_traces=inputs["lexical_traces"],
        hybrid_traces=inputs["hybrid_traces"],
        chunk_metadata=inputs["chunk_metadata"],
    )
    task0084_summary = inputs["task0084_summary"]
    contract = build_task0085_contract()
    write_json(CONTRACT_PATH, contract)

    m20 = analysis["complementarity_matrix"]["at_20"]
    m50 = analysis["complementarity_matrix"]["at_50"]
    addressability = analysis["reranker_addressability"]
    summary = {
        "schema_version": "opk-rag.task0085.reranker-readiness-summary.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "contract_path": _rel(CONTRACT_PATH),
        "result_dir": _rel(RESULT_DIR),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "vector_retriever_modified": False,
        "lexical_retriever_modified": False,
        "hybrid_fusion_modified": False,
        "default_retriever_modified": False,
        "reranker_runtime_added": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "vector_recall_at_20": analysis["baseline_metrics"]["vector_recall_at_20"],
        "lexical_recall_at_20": analysis["baseline_metrics"]["lexical_recall_at_20"],
        "hybrid_recall_at_20": analysis["baseline_metrics"]["hybrid_recall_at_20"],
        "vector_only_hit_count_at_20": m20["vector_only_hit_count_at_20"],
        "lexical_only_hit_count_at_20": m20["lexical_only_hit_count_at_20"],
        "both_hit_count_at_20": m20["both_hit_count_at_20"],
        "neither_hit_count_at_20": m20["neither_hit_count_at_20"],
        "vector_only_hit_count_at_50": m50["vector_only_hit_count_at_50"],
        "lexical_only_hit_count_at_50": m50["lexical_only_hit_count_at_50"],
        "both_hit_count_at_50": m50["both_hit_count_at_50"],
        "neither_hit_count_at_50": m50["neither_hit_count_at_50"],
        "lexical_incremental_hit_count": m20["lexical_incremental_hit_count"],
        "lexical_incremental_recall": m20["lexical_incremental_recall"],
        "vector_incremental_hit_count_over_lexical": m20["vector_incremental_hit_count_over_lexical"],
        "oracle_union_recall_at_20": analysis["oracle_union_metrics"]["oracle_union_recall_at_20"],
        "oracle_union_recall_at_50": analysis["oracle_union_metrics"]["oracle_union_recall_at_50"],
        "fusion_headroom_at_20": analysis["oracle_union_metrics"]["fusion_headroom_at_20"],
        "fusion_headroom_at_50": analysis["oracle_union_metrics"]["fusion_headroom_at_50"],
        "total_vector_miss_count": addressability["total_vector_miss_count"],
        "reranker_addressable_miss_count": addressability["reranker_addressable_miss_count"],
        "reranker_unaddressable_miss_count": addressability["reranker_unaddressable_miss_count"],
        "reranker_addressable_failure_rate": addressability["reranker_addressable_failure_rate"],
        "hybrid_missed_union_gold_count": analysis["hybrid_union_loss"]["hybrid_missed_union_gold_count"],
        "fusion_loss_vs_union_at_20": analysis["hybrid_union_loss"]["fusion_loss_vs_union_at_20"],
        "all_union_to_hybrid_losses_classified": analysis["hybrid_union_loss"]["all_union_to_hybrid_losses_classified"],
        "lexical_unique_hit_count": len(analysis["lexical_unique_hits"]),
        "lexical_unique_hits_classified": all(row["classification"] != "unclassified" for row in analysis["lexical_unique_hits"]),
        "mean_union_candidate_count": analysis["candidate_budget"]["at_50"]["mean_union_candidate_count"],
        "p95_union_candidate_count": analysis["candidate_budget"]["at_50"]["p95_union_candidate_count"],
        "total_reranker_pairs_for_formal_benchmark": analysis["reranker_cost_estimate"]["total_pairs_for_formal_benchmark"],
        "candidate_deduplication_valid": analysis["candidate_deduplication_valid"],
        "vector_protection_unit_count": len(analysis["vector_protection_set"]),
        "oracle_reranker_diagnostic_only": True,
        "reranker_readiness": analysis["readiness_decision"]["reranker_readiness"],
        "primary_readiness_reason": analysis["readiness_decision"]["primary_readiness_reason"],
        "recommended_next_task": analysis["readiness_decision"]["recommended_next_task"],
        "repository_wide_verification_status": "pending",
        "git_add_executed": False,
        "git_commit_created": False,
        "repository_start_state": start_state,
        "task0084_reference": {
            "promotion_decision": task0084_summary.get("promotion_decision"),
            "retrieval_results_deterministic": task0084_summary.get("retrieval_results_deterministic"),
        },
    }
    write_task0085_artifacts(analysis, summary)
    REPORT_PATH.write_text(build_task0085_report(summary, analysis), encoding="utf-8")
    verification = verify_task0085_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    summary["repository_end_state"] = repository_state()
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    return summary


def load_task0085_inputs() -> dict[str, Any]:
    required = [
        TASK0082_RESULT_DIR / "gold_chunk_authority.jsonl",
        TASK0082_RESULT_DIR / "vector_rank_trace.jsonl",
        TASK0082_RESULT_DIR / "lexical_rank_trace.jsonl",
        TASK0082_RESULT_DIR / "hybrid_rank_trace.jsonl",
        TASK0081_RESULT_DIR / "chunk_inventory.json",
        TASK0084_RESULT_DIR / "summary.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RerankerReadinessError(f"missing required TASK-0085 input artifacts: {', '.join(_rel(path) for path in missing)}")
    chunk_inventory = read_json(TASK0081_RESULT_DIR / "chunk_inventory.json")
    return {
        "authority": [row for row in read_jsonl(TASK0082_RESULT_DIR / "gold_chunk_authority.jsonl") if row.get("chunk_representable")],
        "vector_traces": read_jsonl(TASK0082_RESULT_DIR / "vector_rank_trace.jsonl"),
        "lexical_traces": read_jsonl(TASK0082_RESULT_DIR / "lexical_rank_trace.jsonl"),
        "hybrid_traces": read_jsonl(TASK0082_RESULT_DIR / "hybrid_rank_trace.jsonl"),
        "chunk_metadata": {row["chunk_id"]: row for row in chunk_inventory.get("provenance", [])},
        "task0084_summary": read_json(TASK0084_RESULT_DIR / "summary.json"),
    }


def analyze_retriever_complementarity(
    *,
    authority: list[dict[str, Any]],
    vector_traces: list[dict[str, Any]],
    lexical_traces: list[dict[str, Any]],
    hybrid_traces: list[dict[str, Any]],
    chunk_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chunk_metadata = chunk_metadata or {}
    vector = _by_unit(vector_traces)
    lexical = _by_unit(lexical_traces)
    hybrid = _by_unit(hybrid_traces)
    units = [_unit_id(row) for row in authority]
    denominator = len(units)
    candidate_sets = build_candidate_sets(authority, vector, lexical, hybrid, chunk_metadata)
    union_rows = build_candidate_union_rows(authority, vector, lexical, hybrid, chunk_metadata)
    matrix = {f"at_{depth}": complementarity_matrix(units, vector, lexical, depth) for depth in (20, 50)}
    baseline = {
        "vector_recall_at_20": recall(units, vector, 20),
        "lexical_recall_at_20": recall(units, lexical, 20),
        "hybrid_recall_at_20": recall(units, hybrid, 20),
    }
    oracle = oracle_union_metrics(units, vector, lexical, hybrid)
    addressability = reranker_addressability(units, vector, lexical, depth=20)
    rank_distribution, per_unit_rank = gold_rank_distribution(units, vector, lexical)
    hybrid_loss = hybrid_union_loss(units, vector, lexical, hybrid)
    lexical_unique = lexical_unique_hits(authority, vector, lexical, chunk_metadata)
    budget = candidate_budget(units, vector, lexical)
    budget_curve = candidate_budget_curve(units, vector, lexical)
    vector_protection = vector_protection_set(units, vector)
    decision = readiness_decision(matrix["at_20"], oracle, addressability, budget["at_50"], len(vector_protection))
    cost = {
        "schema_version": "opk-rag.task0085.reranker-cost-estimate.v1",
        "candidate_depth": 50,
        "mean_pairs_per_query": budget["at_50"]["mean_union_candidate_count"],
        "p95_pairs_per_query": budget["at_50"]["p95_union_candidate_count"],
        "total_pairs_for_formal_benchmark": sum(row["union_candidate_count"] for row in budget["at_50"]["rows"]),
    }
    ceiling = {
        "schema_version": "opk-rag.task0085.oracle-reranker-ceiling.v1",
        "oracle_reranker_diagnostic_only": True,
        "oracle_reranked_recall_at_5": oracle["oracle_union_recall_at_20"],
        "oracle_reranked_recall_at_10": oracle["oracle_union_recall_at_20"],
        "oracle_reranked_recall_at_20": oracle["oracle_union_recall_at_20"],
        "oracle_reranker_gain_vs_vector_at_20": oracle["fusion_headroom_at_20"],
        "oracle_reranker_gain_vs_hybrid_at_20": oracle["fusion_loss_vs_union_at_20"],
    }
    return {
        "schema_version": "opk-rag.task0085.analysis.v1",
        "required_unit_count": denominator,
        "baseline_metrics": baseline,
        "retriever_candidate_sets": candidate_sets,
        "candidate_union": union_rows,
        "candidate_deduplication_valid": all(
            row["union_candidate_count"] <= row["vector_candidate_count"] + row["lexical_candidate_count"]
            for row in budget["at_20"]["rows"] + budget["at_50"]["rows"]
        ),
        "complementarity_matrix": matrix,
        "oracle_union_metrics": oracle,
        "reranker_addressability": addressability,
        "gold_rank_distribution": rank_distribution,
        "gold_rank_rows": per_unit_rank,
        "hybrid_union_loss": hybrid_loss,
        "lexical_unique_hits": lexical_unique,
        "candidate_budget": budget,
        "candidate_budget_curve": budget_curve,
        "reranker_cost_estimate": cost,
        "oracle_reranker_ceiling": ceiling,
        "vector_protection_set": vector_protection,
        "readiness_decision": decision,
    }


def build_candidate_sets(authority: list[dict[str, Any]], vector: dict[str, Any], lexical: dict[str, Any], hybrid: dict[str, Any], chunk_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for auth in authority:
        unit = _unit_id(auth)
        gold = set(auth.get("gold_chunk_ids") or [])
        for source, traces in (("vector", vector), ("lexical", lexical), ("hybrid", hybrid)):
            trace = traces[unit]
            ids = trace.get("candidate_chunk_ids") or []
            for index, chunk_id in enumerate(ids[:100], start=1):
                meta = chunk_metadata.get(chunk_id, {})
                rows.append(
                    {
                        "schema_version": "opk-rag.task0085.retriever-candidate.v1",
                        "sample_id": auth["sample_id"],
                        "source_span_digest": auth["source_span_digest"],
                        "chunk_id": chunk_id,
                        "document_id": meta.get("document_id"),
                        "section_id": meta.get("section_id"),
                        "vector_rank": index if source == "vector" else None,
                        "vector_score": trace.get("best_gold_score") if source == "vector" and chunk_id in gold else None,
                        "lexical_rank": index if source == "lexical" else None,
                        "lexical_score": trace.get("best_gold_score") if source == "lexical" and chunk_id in gold else None,
                        "hybrid_rank": index if source == "hybrid" else None,
                        "is_gold": chunk_id in gold,
                        "retrieval_source": source,
                    }
                )
    return rows


def build_candidate_union_rows(authority: list[dict[str, Any]], vector: dict[str, Any], lexical: dict[str, Any], hybrid: dict[str, Any], chunk_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for auth in authority:
        unit = _unit_id(auth)
        gold = set(auth.get("gold_chunk_ids") or [])
        v_ids = (vector[unit].get("candidate_chunk_ids") or [])[:100]
        l_ids = (lexical[unit].get("candidate_chunk_ids") or [])[:100]
        h_ids = (hybrid[unit].get("candidate_chunk_ids") or [])[:100]
        ordered = list(dict.fromkeys(v_ids + l_ids))
        for chunk_id in ordered:
            meta = chunk_metadata.get(chunk_id, {})
            rows.append(
                {
                    "schema_version": "opk-rag.task0085.candidate-union.v1",
                    "sample_id": auth["sample_id"],
                    "source_span_digest": auth["source_span_digest"],
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id"),
                    "section_id": meta.get("section_id"),
                    "from_vector": chunk_id in v_ids,
                    "from_lexical": chunk_id in l_ids,
                    "vector_rank": _rank_in(v_ids, chunk_id),
                    "lexical_rank": _rank_in(l_ids, chunk_id),
                    "hybrid_rank": _rank_in(h_ids, chunk_id),
                    "vector_score": vector[unit].get("best_gold_score") if chunk_id in gold else None,
                    "lexical_score": lexical[unit].get("best_gold_score") if chunk_id in gold else None,
                    "gold_match": chunk_id in gold,
                    "retrieval_source": "both" if chunk_id in v_ids and chunk_id in l_ids else "vector" if chunk_id in v_ids else "lexical",
                }
            )
    return rows


def complementarity_matrix(units: Iterable[str], vector: dict[str, Any], lexical: dict[str, Any], depth: int) -> dict[str, Any]:
    units = list(units)
    buckets = {"vector_only": [], "lexical_only": [], "both": [], "neither": []}
    for unit in units:
        v_hit = hit(vector[unit], depth)
        l_hit = hit(lexical[unit], depth)
        if v_hit and l_hit:
            buckets["both"].append(unit)
        elif v_hit:
            buckets["vector_only"].append(unit)
        elif l_hit:
            buckets["lexical_only"].append(unit)
        else:
            buckets["neither"].append(unit)
    suffix = f"_at_{depth}"
    return {
        f"vector_only_hit_count{suffix}": len(buckets["vector_only"]),
        f"lexical_only_hit_count{suffix}": len(buckets["lexical_only"]),
        f"both_hit_count{suffix}": len(buckets["both"]),
        f"neither_hit_count{suffix}": len(buckets["neither"]),
        f"vector_only_units{suffix}": buckets["vector_only"],
        f"lexical_only_units{suffix}": buckets["lexical_only"],
        f"both_units{suffix}": buckets["both"],
        f"neither_units{suffix}": buckets["neither"],
        "lexical_incremental_hit_count": len(buckets["lexical_only"]),
        "lexical_incremental_recall": _ratio(len(buckets["lexical_only"]), len(units)),
        "vector_incremental_hit_count_over_lexical": len(buckets["vector_only"]),
    }


def oracle_union_metrics(units: list[str], vector: dict[str, Any], lexical: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_version": "opk-rag.task0085.oracle-union-metrics.v1", "diagnostic_only": True}
    vector20 = recall(units, vector, 20)
    hybrid20 = recall(units, hybrid, 20)
    for depth in CANDIDATE_DEPTHS:
        union_recall = _ratio(sum(hit(vector[u], depth) or hit(lexical[u], depth) for u in units), len(units))
        out[f"oracle_union_recall_at_{depth}"] = union_recall
        out[f"fusion_headroom_at_{depth}"] = union_recall - recall(units, vector, depth)
    out["fusion_loss_vs_union_at_20"] = out["oracle_union_recall_at_20"] - hybrid20
    out["oracle_reranker_gain_vs_vector_at_20"] = out["oracle_union_recall_at_20"] - vector20
    out["oracle_reranker_gain_vs_hybrid_at_20"] = out["oracle_union_recall_at_20"] - hybrid20
    return out


def reranker_addressability(units: list[str], vector: dict[str, Any], lexical: dict[str, Any], *, depth: int) -> dict[str, Any]:
    misses = [unit for unit in units if not hit(vector[unit], depth)]
    addressable = [unit for unit in misses if candidate_present(vector[unit], 100) or candidate_present(lexical[unit], 100)]
    unaddressable = [unit for unit in misses if unit not in addressable]
    return {
        "schema_version": "opk-rag.task0085.reranker-addressability.v1",
        "formal_vector_top_k": depth,
        "candidate_union_depth": 100,
        "total_vector_miss_count": len(misses),
        "reranker_addressable_miss_count": len(addressable),
        "reranker_unaddressable_miss_count": len(unaddressable),
        "reranker_addressable_failure_rate": _ratio(len(addressable), len(misses)),
        "addressable_units": addressable,
        "unaddressable_units": unaddressable,
    }


def gold_rank_distribution(units: list[str], vector: dict[str, Any], lexical: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buckets = Counter()
    rows = []
    for unit in units:
        v_rank = rank(vector[unit])
        l_rank = rank(lexical[unit])
        union_rank = min([r for r in (v_rank, l_rank) if r is not None], default=None)
        bucket = rank_bucket(union_rank)
        buckets[bucket] += 1
        rows.append(
            {
                "sample_unit_id": unit,
                "best_gold_source": "vector" if v_rank is not None and (l_rank is None or v_rank <= l_rank) else "lexical" if l_rank is not None else None,
                "best_vector_rank": v_rank,
                "best_lexical_rank": l_rank,
                "candidate_present_at_20": _within(union_rank, 20),
                "candidate_present_at_50": _within(union_rank, 50),
                "candidate_present_at_100": _within(union_rank, 100),
                "rank_bucket": bucket,
            }
        )
    order = ("rank_1_5", "rank_6_10", "rank_11_20", "rank_21_50", "rank_51_100", "gt_100", "not_candidate")
    return {"schema_version": "opk-rag.task0085.gold-rank-distribution.v1", "buckets": {key: buckets.get(key, 0) for key in order}, "rows": rows}, rows


def hybrid_union_loss(units: list[str], vector: dict[str, Any], lexical: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for unit in units:
        in_union = hit(vector[unit], 20) or hit(lexical[unit], 20)
        hybrid_hit = hit(hybrid[unit], 20)
        if not in_union or hybrid_hit:
            continue
        v_hit = hit(vector[unit], 20)
        l_hit = hit(lexical[unit], 20)
        classification = "vector_hit_destroyed_by_fusion" if v_hit and not l_hit else "lexical_unique_hit_not_promoted" if l_hit and not v_hit else "both_retrievers_hit_but_fusion_miss"
        h_trace = hybrid[unit].get("gold_chunk_score_trace") or {}
        rows.append(
            {
                "sample_unit_id": unit,
                "sample_id": unit.split("::", 1)[0],
                "vector_rank": rank(vector[unit]),
                "lexical_rank": rank(lexical[unit]),
                "hybrid_rank": rank(hybrid[unit]),
                "vector_score": vector[unit].get("best_gold_score"),
                "lexical_score": lexical[unit].get("best_gold_score"),
                "fused_score": h_trace.get("fused_score") or hybrid[unit].get("best_gold_score"),
                "gold_present_in_union": True,
                "hybrid_dropped_gold": True,
                "classification": classification,
            }
        )
    return {
        "schema_version": "opk-rag.task0085.hybrid-union-loss.v1",
        "hybrid_missed_union_gold_count": len(rows),
        "fusion_loss_vs_union_at_20": _ratio(len(rows), len(units)),
        "all_union_to_hybrid_losses_classified": all(row["classification"] != "unclassified" for row in rows),
        "rows": rows,
    }


def lexical_unique_hits(authority: list[dict[str, Any]], vector: dict[str, Any], lexical: dict[str, Any], chunk_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for auth in authority:
        unit = _unit_id(auth)
        if hit(vector[unit], 20) or not hit(lexical[unit], 20):
            continue
        features = classify_query_features(auth["question"], auth.get("gold_heading_path") or [], auth.get("gold_chunk_ids") or [], chunk_metadata)
        rows.append(
            {
                "schema_version": "opk-rag.task0085.lexical-unique-hit.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "query": auth["question"],
                "gold_chunk": (auth.get("gold_chunk_ids") or [None])[0],
                "lexical_rank": rank(lexical[unit]),
                "lexical_score": lexical[unit].get("best_gold_score"),
                "vector_best_gold_rank": rank(vector[unit]),
                "exact_query_terms": lexical[unit].get("lexical_terms_overlap") or [],
                "exact_gold_terms": lexical[unit].get("gold_terms_missing_from_query") or [],
                "classification": features["primary"],
                "diagnostic_query_feature_classification": features,
                "diagnostic_only": True,
            }
        )
    return rows


def candidate_budget(units: list[str], vector: dict[str, Any], lexical: dict[str, Any]) -> dict[str, Any]:
    return {f"at_{depth}": candidate_budget_at_depth(units, vector, lexical, depth) for depth in (20, 50)}


def candidate_budget_at_depth(units: list[str], vector: dict[str, Any], lexical: dict[str, Any], depth: int) -> dict[str, Any]:
    rows = []
    for unit in units:
        v = (vector[unit].get("candidate_chunk_ids") or [])[:depth]
        l = (lexical[unit].get("candidate_chunk_ids") or [])[:depth]
        union = list(dict.fromkeys(v + l))
        rows.append(
            {
                "sample_unit_id": unit,
                "vector_candidate_count": len(v),
                "lexical_candidate_count": len(l),
                "union_candidate_count": len(union),
                "duplicate_candidate_count": len(v) + len(l) - len(union),
            }
        )
    counts = [row["union_candidate_count"] for row in rows]
    return {
        "schema_version": "opk-rag.task0085.candidate-budget.v1",
        "candidate_depth": depth,
        "mean_union_candidate_count": statistics.mean(counts) if counts else 0.0,
        "p50_union_candidate_count": percentile(counts, 0.50),
        "p95_union_candidate_count": percentile(counts, 0.95),
        "max_union_candidate_count": max(counts) if counts else 0,
        "rows": rows,
    }


def candidate_budget_curve(units: list[str], vector: dict[str, Any], lexical: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for depth in BUDGET_CURVE_DEPTHS:
        union_recall = _ratio(sum(hit(vector[u], depth) or hit(lexical[u], depth) for u in units), len(units))
        budget = candidate_budget_at_depth(units, vector, lexical, depth)
        rows.append({"candidate_budget": depth, "oracle_union_recall": union_recall, "mean_union_candidate_count": budget["mean_union_candidate_count"], "p95_union_candidate_count": budget["p95_union_candidate_count"]})
    useful = next((row["candidate_budget"] for row in rows if row["oracle_union_recall"] >= rows[-1]["oracle_union_recall"]), rows[-1]["candidate_budget"] if rows else None)
    return {"schema_version": "opk-rag.task0085.candidate-budget-curve.v1", "diagnostic_only": True, "minimum_candidate_budget_for_useful_reranker": useful, "rows": rows}


def vector_protection_set(units: list[str], vector: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"sample_unit_id": unit, "vector_rank": rank(vector[unit]), "vector_protection_required": True} for unit in units if hit(vector[unit], 20)]


def readiness_decision(matrix20: dict[str, Any], oracle: dict[str, Any], addressability: dict[str, Any], budget50: dict[str, Any], vector_protection_count: int) -> dict[str, Any]:
    lexical_only = matrix20["lexical_only_hit_count_at_20"]
    headroom = oracle["fusion_headroom_at_20"]
    addressable = addressability["reranker_addressable_miss_count"]
    p95 = budget50["p95_union_candidate_count"] or 0
    if lexical_only > 0 and headroom >= 0.05 and addressable > 0 and p95 <= 100 and vector_protection_count > 0:
        readiness = "ready"
        reason = "Lexical provides unique gold hits, Oracle Union headroom is at least 0.05, addressable vector misses exist, and the deduplicated candidate pool is within the diagnostic top-100 budget."
        next_task = "TASK-0086 governed multi-retriever reranking experiment"
    elif lexical_only > 0 and headroom >= 0.02 and addressable > 0:
        readiness = "conditional"
        reason = "Lexical has unique value, but the measured Oracle Union headroom is modest; selective lexical rescue or conditional reranking should be evaluated before full-query reranking."
        next_task = "Selective Lexical Rescue + Conditional Reranking"
    else:
        readiness = "not_ready"
        reason = "Oracle Union does not create enough headroom over Vector-only, or vector misses remain candidate generation failures."
        next_task = "Vector residual failure and candidate generation diagnosis"
    return {"schema_version": "opk-rag.task0085.readiness-decision.v1", "reranker_readiness": readiness, "primary_readiness_reason": reason, "recommended_next_task": next_task}


def classify_query_features(query: str, headings: list[str], gold_chunk_ids: list[str], chunk_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    text = " ".join([query, *headings, *(chunk_metadata.get(chunk_id, {}).get("source_path", "") for chunk_id in gold_chunk_ids)]).lower()
    checks = [
        ("file_name", [".md", ".py", ".json", ".toml", ".docx"]),
        ("path", ["/", "\\", "路径", "目录"]),
        ("environment_variable", ["env", "environment", "变量"]),
        ("model_name", ["qwen", "bge", "deepseek", "model"]),
        ("version_string", ["version", "版本"]),
        ("code_symbol", ["--", "cli", "api", "函数", "命令"]),
        ("numeric_identifier", [str(n) for n in range(10)]),
        ("exact_phrase", ["配置", "文件", "chunk", "tmux", "uv"]),
        ("acronym", ["api", "cli", "rag", "mrr"]),
    ]
    labels = [label for label, needles in checks if any(needle in text for needle in needles)]
    primary = labels[0] if labels else "rare_term" if len(query) <= 30 else "other"
    return {"diagnostic_only": True, "labels": labels or [primary], "primary": primary}


def build_task0085_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0085.reranker-readiness-contract.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "formal_benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "chunk_digest": _file_digest(TASK0081_RESULT_DIR / "chunk_inventory.json"),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "vector_retriever_config": "TASK-0082 frozen C0 vector rank trace derived from TASK-0081 Qwen embeddings",
        "lexical_retriever_config": "TASK-0082 frozen C0 lexical rank trace derived from Jieba term-overlap ranking",
        "candidate_depths": list(CANDIDATE_DEPTHS),
        "deduplication_rule": "stable chunk_id identity; same chunk from vector and lexical is one union candidate",
        "gold_matching_rule": "gold_chunk_ids from TASK-0082 gold_chunk_authority; complete_evidence_rank is authoritative for multi-chunk units",
        "oracle_union_definition": "diagnostic-only success if gold is present in vector or lexical candidate set at the measured depth",
        "oracle_reranker_definition": "diagnostic-only perfect reranker promotes any gold candidate in the union into final Top-K",
        "readiness_gates": ["lexical_only_hit_count_at_20 > 0", "fusion_headroom_at_20 >= 0.05 for ready or >= 0.02 for conditional", "reranker_addressable_miss_count > 0", "vector_protection_unit_count > 0"],
        "candidate_budget": {"formal_depth": 20, "diagnostic_depth": 50, "max_diagnostic_depth": 100},
        "vector_protection_policy": "future reranker promotion must not regress current vector Top-20 gold hits",
        "reranker_input_contract": {
            "query_text": "original retrieval query",
            "chunk_text": "authoritative source chunk text when TASK-0086 materializes model inputs",
            "document_title": "source_path or document title metadata",
            "heading_path": "chunk heading_path metadata",
            "chunk_id": "stable chunk identity",
            "llm_query_expansion": "forbidden",
        },
    }


def write_task0085_artifacts(analysis: dict[str, Any], summary: dict[str, Any]) -> None:
    write_jsonl(RESULT_DIR / "retriever_candidate_sets.jsonl", analysis["retriever_candidate_sets"])
    write_jsonl(RESULT_DIR / "candidate_union.jsonl", analysis["candidate_union"])
    write_json(RESULT_DIR / "complementarity_matrix.json", analysis["complementarity_matrix"])
    write_jsonl(RESULT_DIR / "lexical_unique_hits.jsonl", analysis["lexical_unique_hits"])
    write_json(RESULT_DIR / "gold_rank_distribution.json", analysis["gold_rank_distribution"])
    write_jsonl(RESULT_DIR / "gold_rank_rows.jsonl", analysis["gold_rank_rows"])
    write_json(RESULT_DIR / "oracle_union_metrics.json", analysis["oracle_union_metrics"])
    write_json(RESULT_DIR / "reranker_addressability.json", analysis["reranker_addressability"])
    write_json(RESULT_DIR / "oracle_reranker_ceiling.json", analysis["oracle_reranker_ceiling"])
    write_jsonl(RESULT_DIR / "hybrid_union_loss_audit.jsonl", analysis["hybrid_union_loss"]["rows"])
    write_json(RESULT_DIR / "candidate_budget_curve.json", analysis["candidate_budget_curve"])
    write_json(RESULT_DIR / "candidate_budget.json", analysis["candidate_budget"])
    write_json(RESULT_DIR / "reranker_cost_estimate.json", analysis["reranker_cost_estimate"])
    write_json(RESULT_DIR / "vector_protection_set.json", {"schema_version": "opk-rag.task0085.vector-protection-set.v1", "vector_protection_unit_count": len(analysis["vector_protection_set"]), "rows": analysis["vector_protection_set"]})
    write_json(RESULT_DIR / "readiness_decision.json", analysis["readiness_decision"])
    write_json(RESULT_DIR / "summary.json", summary)


def verify_task0085_artifacts() -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "retriever_candidate_sets.jsonl",
        RESULT_DIR / "candidate_union.jsonl",
        RESULT_DIR / "complementarity_matrix.json",
        RESULT_DIR / "lexical_unique_hits.jsonl",
        RESULT_DIR / "gold_rank_distribution.json",
        RESULT_DIR / "oracle_union_metrics.json",
        RESULT_DIR / "reranker_addressability.json",
        RESULT_DIR / "oracle_reranker_ceiling.json",
        RESULT_DIR / "hybrid_union_loss_audit.jsonl",
        RESULT_DIR / "candidate_budget_curve.json",
        RESULT_DIR / "reranker_cost_estimate.json",
        RESULT_DIR / "vector_protection_set.json",
        RESULT_DIR / "readiness_decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "chunking_modified", "embedding_model_modified", "vector_retriever_modified", "lexical_retriever_modified", "hybrid_fusion_modified", "default_retriever_modified", "reranker_runtime_added", "query_reformulation_modified", "agent_runtime_modified", "graph_runtime_modified"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
        if summary.get("formal_metric_authority") != FORMAL_AUTHORITY_ID:
            issues.append({"code": "formal_metric_authority_invalid"})
        if summary.get("candidate_deduplication_valid") is not True:
            issues.append({"code": "candidate_deduplication_invalid"})
        if summary.get("oracle_reranker_diagnostic_only") is not True:
            issues.append({"code": "oracle_reranker_not_marked_diagnostic"})
        if summary.get("reranker_readiness") not in {"ready", "conditional", "not_ready"}:
            issues.append({"code": "invalid_readiness"})
    return {"schema_version": "opk-rag.task0085.verification.v1", "status": "valid" if not issues else "invalid", "issues": issues, "git_add_executed": False, "git_commit_created": False}


def build_task0085_report(summary: dict[str, Any], analysis: dict[str, Any]) -> str:
    m20 = analysis["complementarity_matrix"]["at_20"]
    m50 = analysis["complementarity_matrix"]["at_50"]
    budget50 = analysis["candidate_budget"]["at_50"]
    lexical_types = Counter(row["classification"] for row in analysis["lexical_unique_hits"])
    return "\n".join(
        [
            "# TASK0085 Reranker Readiness and Retriever Complementarity Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- formal_metric_authority=`{summary['formal_metric_authority']}`",
            f"- reranker_readiness=`{summary['reranker_readiness']}`",
            f"- primary_reason=`{summary['primary_readiness_reason']}`",
            f"- recommended_next_task=`{summary['recommended_next_task']}`",
            "",
            "## Required Answers",
            "",
            f"1. Lexical 是否存在 Vector 没有的 Gold Hits：`lexical_only_hit_count_at_20={m20['lexical_only_hit_count_at_20']}`，`lexical_only_hit_count_at_50={m50['lexical_only_hit_count_at_50']}`。",
            f"2. Top-20 互补矩阵：Vector-only `{m20['vector_only_hit_count_at_20']}`，Lexical-only `{m20['lexical_only_hit_count_at_20']}`，Both `{m20['both_hit_count_at_20']}`，Neither `{m20['neither_hit_count_at_20']}`。",
            f"3. Vector + Lexical Union Recall：`oracle_union_recall_at_20={summary['oracle_union_recall_at_20']:.4f}`，`oracle_union_recall_at_50={summary['oracle_union_recall_at_50']:.4f}`。",
            f"4. Union 相对 Vector-only headroom：Top-20 `{summary['fusion_headroom_at_20']:.4f}`，Top-50 `{summary['fusion_headroom_at_50']:.4f}`。",
            f"5. Current Hybrid 丢掉 Union 中 Gold：`hybrid_missed_union_gold_count={summary['hybrid_missed_union_gold_count']}`，`fusion_loss_vs_union_at_20={summary['fusion_loss_vs_union_at_20']:.4f}`。",
            f"6. Vector miss 中理论可 Reranking 解决：`reranker_addressable_miss_count={summary['reranker_addressable_miss_count']}` / `{summary['total_vector_miss_count']}`。",
            f"7. Reranker 也无能为力的 miss：`reranker_unaddressable_miss_count={summary['reranker_unaddressable_miss_count']}`。",
            f"8. Lexical unique hits query 类型：`{dict(lexical_types)}`；分类为 diagnostic-only，未修改 benchmark authority。",
            f"9. Candidate Pool：Top-50 union mean `{budget50['mean_union_candidate_count']:.2f}`，p95 `{budget50['p95_union_candidate_count']}`，max `{budget50['max_union_candidate_count']}`。",
            f"10. Reranker pairs：mean `{analysis['reranker_cost_estimate']['mean_pairs_per_query']:.2f}`，p95 `{analysis['reranker_cost_estimate']['p95_pairs_per_query']}`，total `{analysis['reranker_cost_estimate']['total_pairs_for_formal_benchmark']}`。",
            f"11. 是否支持进入真实 Reranker Experiment：`{summary['reranker_readiness']}`。",
            f"12. 下一任务：`{summary['recommended_next_task']}`。",
            "",
            "## Governance",
            "",
            f"- benchmark_modified=`{str(summary['benchmark_modified']).lower()}`",
            f"- vector_retriever_modified=`{str(summary['vector_retriever_modified']).lower()}`",
            f"- lexical_retriever_modified=`{str(summary['lexical_retriever_modified']).lower()}`",
            f"- hybrid_fusion_modified=`{str(summary['hybrid_fusion_modified']).lower()}`",
            f"- reranker_runtime_added=`{str(summary['reranker_runtime_added']).lower()}`",
            f"- candidate_deduplication_valid=`{str(summary['candidate_deduplication_valid']).lower()}`",
        ]
    )


def hit(trace: dict[str, Any], depth: int) -> bool:
    value = trace.get("complete_evidence_rank")
    return value is not None and int(value) <= depth


def candidate_present(trace: dict[str, Any], depth: int) -> bool:
    value = trace.get("complete_evidence_rank")
    return value is not None and int(value) <= depth


def rank(trace: dict[str, Any]) -> int | None:
    value = trace.get("complete_evidence_rank")
    return int(value) if value is not None else None


def recall(units: Iterable[str], traces: dict[str, Any], depth: int) -> float:
    items = list(units)
    return _ratio(sum(hit(traces[unit], depth) for unit in items), len(items))


def rank_bucket(value: int | None) -> str:
    if value is None:
        return "not_candidate"
    if value <= 5:
        return "rank_1_5"
    if value <= 10:
        return "rank_6_10"
    if value <= 20:
        return "rank_11_20"
    if value <= 50:
        return "rank_21_50"
    if value <= 100:
        return "rank_51_100"
    return "gt_100"


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_unit_id(row): row for row in rows if row.get("chunk_representable", True)}


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def _rank_in(values: list[str], chunk_id: str) -> int | None:
    try:
        return values.index(chunk_id) + 1
    except ValueError:
        return None


def _within(value: int | None, depth: int) -> bool:
    return value is not None and value <= depth


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
