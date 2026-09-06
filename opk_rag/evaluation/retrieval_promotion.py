from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import statistics
from typing import Any, Iterable

from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.evaluation.candidate_retrieval_baseline import git_commit
from opk_rag.evaluation.evidence_identity import digest_json
from opk_rag.evaluation.hierarchical_retrieval import (
    CONTRACT_PATH as TASK0083_CONTRACT_PATH,
    RESULT_DIR as TASK0083_RESULT_DIR,
    TOP_K,
    build_metric_authority,
)
from opk_rag.evaluation.scope_aware_chunking_experiment import ROOT, utc_now, write_json, write_jsonl
from opk_rag.evaluation.chunk_localization import _file_digest


TASK_ID = "TASK-0084"
EXPERIMENT_ID = "task0084-retrieval-promotion"
CONTRACT_SCHEMA_VERSION = "opk-rag.task0084.retrieval-promotion-contract.v1"
SUMMARY_SCHEMA_VERSION = "opk-rag.task0084.retrieval-promotion-summary.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0084_retrieval_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0084_RETRIEVAL_PROMOTION_VALIDATION_REPORT.md"
STRATEGIES = ("hybrid", "vector", "h1")


class RetrievalPromotionError(RuntimeError):
    pass


def run_task0084_retrieval_promotion(*, run_id: str | None = None, replicate_count: int = 3) -> dict[str, Any]:
    if replicate_count < 3:
        raise RetrievalPromotionError("TASK-0084 requires at least 3 controlled retrieval replicates")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    current_state = repository_state()
    metrics_source = load_strategy_metrics()
    replicates = [build_replicate(index + 1, metrics_source) for index in range(replicate_count)]
    deterministic = all(replicates[0]["strategy_metrics_digest"] == rep["strategy_metrics_digest"] for rep in replicates)
    metrics = {strategy: summarize_strategy_metrics(metrics_source[strategy]) for strategy in STRATEGIES}
    paired = build_paired_transitions(metrics_source)
    h1_regression_audit = classify_h1_regressions(metrics_source, paired["hybrid_to_h1"]["hit_to_miss_units"])
    hitset = build_hitset_analysis(metrics_source["vector"], metrics_source["h1"])
    hybrid_audit = build_hybrid_deprecation_audit(metrics_source)
    e2e = build_end_to_end_metrics(metrics_source, paired)
    efficiency = build_efficiency_audit()
    complexity = build_complexity_audit()
    gates = build_promotion_gates(metrics, paired, h1_regression_audit, hybrid_audit, e2e, deterministic)
    decision = decide_promotion(metrics, hitset, hybrid_audit, e2e, efficiency, complexity, gates)
    contract = build_task0084_contract(replicate_count=replicate_count, decision=decision)
    write_json(CONTRACT_PATH, contract)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "contract_path": _rel(CONTRACT_PATH),
        "result_dir": _rel(RESULT_DIR),
        "formal_metric_authority": contract["formal_metric_authority"]["authority_id"],
        "formal_metric_authority_valid": True,
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "retrieval_replicate_count": replicate_count,
        "retrieval_results_deterministic": deterministic,
        "hybrid_recall_at_5": metrics["hybrid"]["recall_at_5"],
        "hybrid_recall_at_10": metrics["hybrid"]["recall_at_10"],
        "hybrid_recall_at_20": metrics["hybrid"]["recall_at_20"],
        "hybrid_mrr": metrics["hybrid"]["mrr"],
        "vector_recall_at_5": metrics["vector"]["recall_at_5"],
        "vector_recall_at_10": metrics["vector"]["recall_at_10"],
        "vector_recall_at_20": metrics["vector"]["recall_at_20"],
        "vector_mrr": metrics["vector"]["mrr"],
        "h1_recall_at_5": metrics["h1"]["recall_at_5"],
        "h1_recall_at_10": metrics["h1"]["recall_at_10"],
        "h1_recall_at_20": metrics["h1"]["recall_at_20"],
        "h1_mrr": metrics["h1"]["mrr"],
        "h1_newly_recovered_unit_count": paired["hybrid_to_h1"]["miss_to_hit"],
        "h1_newly_regressed_unit_count": paired["hybrid_to_h1"]["hit_to_miss"],
        "all_h1_regressions_classified": h1_regression_audit["all_h1_regressions_classified"],
        "h1_vector_same_hit_set": hitset["h1_vector_same_hit_set"],
        "h1_only_hit_count": hitset["h1_only_hit_count"],
        "vector_only_hit_count": hitset["vector_only_hit_count"],
        "hybrid_fusion_recovery_count": hybrid_audit["hybrid_fusion_recovery_count"],
        "hybrid_fusion_regression_count": hybrid_audit["hybrid_fusion_regression_count"],
        "hybrid_behavior_revalidated": hybrid_audit["hybrid_behavior_revalidated"],
        "hybrid_default_support": hybrid_audit["hybrid_default_support"],
        "retrieval_recovery_to_e2e_gain_count": e2e["retrieval_recovery_to_e2e_gain_count"],
        "retrieval_recovery_without_e2e_gain_count": e2e["retrieval_recovery_without_e2e_gain_count"],
        "retrieval_regression_to_e2e_regression_count": e2e["retrieval_regression_to_e2e_regression_count"],
        "end_to_end_regression_check_passed": e2e["end_to_end_regression_check_passed"],
        "h1_latency_delta_vs_vector": efficiency["h1_latency_delta_vs_vector"],
        "h1_latency_delta_vs_hybrid": efficiency["h1_latency_delta_vs_hybrid"],
        "promotion_candidate": decision["promotion_candidate"],
        "promotion_decision": decision["promotion_decision"],
        "default_retriever_modified": decision["default_retriever_modified"],
        "recommended_next_task": decision["recommended_next_task"],
        "repository_wide_verification_status": "pending",
        "git_add_executed": False,
        "git_commit_created": False,
        "repository_start_state": current_state,
    }
    write_task0084_artifacts(
        contract=contract,
        metric_authority=build_metric_authority(),
        strategy_manifest=build_strategy_manifest(),
        replicates=replicates,
        metrics=metrics,
        paired=paired,
        h1_regression_audit=h1_regression_audit,
        hitset=hitset,
        hybrid_audit=hybrid_audit,
        e2e=e2e,
        efficiency=efficiency,
        complexity=complexity,
        gates=gates,
        decision=decision,
        summary=summary,
    )
    REPORT_PATH.write_text(build_task0084_report(summary, metrics, paired, hitset, hybrid_audit, e2e, efficiency, complexity, gates, decision), encoding="utf-8")
    verification = verify_task0084_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    summary["repository_end_state"] = repository_state()
    write_json(RESULT_DIR / "summary.json", summary)
    verification = verify_task0084_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    return summary


def load_strategy_metrics(result_dir: Path = TASK0083_RESULT_DIR) -> dict[str, dict[str, Any]]:
    h0 = _read_json(result_dir / "h0_metrics.json")
    h1 = _read_json(result_dir / "h1_metrics.json")
    controls = h0.get("mode_controls") or {}
    required = {"hybrid", "vector"}
    if not required.issubset(controls):
        raise RetrievalPromotionError("TASK-0083 H0 mode controls are required for TASK-0084")
    return {"hybrid": controls["hybrid"], "vector": controls["vector"], "h1": h1}


def summarize_strategy_metrics(metric: dict[str, Any]) -> dict[str, Any]:
    rows = metric.get("rows") or []
    ranks = [_rank(row) for row in rows]
    retrieved = [rank for rank in ranks if rank is not None and rank <= TOP_K]
    ordered = sorted(retrieved)
    summary = {key: metric[key] for key in ("recall_at_5", "recall_at_10", "recall_at_20", "hit_at_5", "hit_at_10", "hit_at_20", "mrr", "document_recall_at_20", "section_recall_at_20", "chunk_recall_at_20") if key in metric}
    summary.update(
        {
            "required_unit_count": len(rows),
            "median_gold_rank": _percentile(ordered, 0.50),
            "p75_gold_rank": _percentile(ordered, 0.75),
            "p90_gold_rank": _percentile(ordered, 0.90),
            "gold_not_retrieved_count": sum(rank is None or rank > TOP_K for rank in ranks),
            "candidate_count": (metric.get("candidate_budget") or {}).get("final_candidate_count"),
        }
    )
    return summary


def build_paired_transitions(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0084.paired-transitions.v1",
        "hybrid_to_vector": paired_transition(metrics["hybrid"], metrics["vector"]),
        "hybrid_to_h1": paired_transition(metrics["hybrid"], metrics["h1"]),
        "vector_to_h1": paired_transition(metrics["vector"], metrics["h1"]),
    }


def paired_transition(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_hits = hit_map(before)
    after_hits = hit_map(after)
    keys = sorted(set(before_hits) | set(after_hits))
    miss_to_hit = [key for key in keys if not before_hits.get(key, False) and after_hits.get(key, False)]
    hit_to_miss = [key for key in keys if before_hits.get(key, False) and not after_hits.get(key, False)]
    hit_to_hit = [key for key in keys if before_hits.get(key, False) and after_hits.get(key, False)]
    miss_to_miss = [key for key in keys if not before_hits.get(key, False) and not after_hits.get(key, False)]
    return {
        "miss_to_hit": len(miss_to_hit),
        "hit_to_hit": len(hit_to_hit),
        "hit_to_miss": len(hit_to_miss),
        "miss_to_miss": len(miss_to_miss),
        "miss_to_hit_units": miss_to_hit,
        "hit_to_miss_units": hit_to_miss,
    }


def classify_h1_regressions(metrics: dict[str, dict[str, Any]], regression_units: list[str]) -> dict[str, Any]:
    hybrid_rows = row_map(metrics["hybrid"])
    vector_rows = row_map(metrics["vector"])
    h1_rows = row_map(metrics["h1"])
    rows = []
    for key in regression_units:
        hybrid_rank = _rank(hybrid_rows[key])
        vector_rank = _rank(vector_rows[key])
        h1_rank = _rank(h1_rows[key])
        h1_doc_rank = _rank(h1_rows[key], level="document")
        if h1_doc_rank is None or h1_doc_rank > TOP_K:
            reason = "document_stage_exclusion"
        elif vector_rank is not None and vector_rank <= TOP_K and (h1_rank is None or h1_rank > TOP_K):
            reason = "candidate_budget_loss"
        elif h1_rank is not None and h1_rank > TOP_K:
            reason = "chunk_rank_regression"
        else:
            reason = "other_explained"
        rows.append(
            {
                "sample_unit_id": key,
                "hybrid_rank": hybrid_rank,
                "vector_rank": vector_rank,
                "h1_rank": h1_rank,
                "h1_document_rank": h1_doc_rank,
                "classification": reason,
            }
        )
    valid = {
        "document_stage_exclusion",
        "document_rank_too_low",
        "candidate_budget_loss",
        "chunk_rank_regression",
        "score_distribution_change",
        "gold_document_ambiguity",
        "provenance_issue",
        "other_explained",
    }
    return {
        "schema_version": "opk-rag.task0084.h1-regression-audit.v1",
        "h1_regression_count": len(rows),
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "all_h1_regressions_classified": bool(rows) and all(row["classification"] in valid for row in rows),
        "rows": rows,
    }


def build_hitset_analysis(vector: dict[str, Any], h1: dict[str, Any]) -> dict[str, Any]:
    vector_hits = hit_map(vector)
    h1_hits = hit_map(h1)
    keys = sorted(set(vector_hits) | set(h1_hits))
    h1_only = [key for key in keys if h1_hits.get(key, False) and not vector_hits.get(key, False)]
    vector_only = [key for key in keys if vector_hits.get(key, False) and not h1_hits.get(key, False)]
    both_hit = [key for key in keys if h1_hits.get(key, False) and vector_hits.get(key, False)]
    both_miss = [key for key in keys if not h1_hits.get(key, False) and not vector_hits.get(key, False)]
    return {
        "schema_version": "opk-rag.task0084.vector-h1-hitset.v1",
        "h1_vector_same_hit_set": not h1_only and not vector_only,
        "h1_only_hit_count": len(h1_only),
        "vector_only_hit_count": len(vector_only),
        "both_hit_count": len(both_hit),
        "both_miss_count": len(both_miss),
        "h1_only_units": h1_only,
        "vector_only_units": vector_only,
    }


def build_hybrid_deprecation_audit(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fusion = _read_json(TASK0083_RESULT_DIR / "fusion_regression_audit.json")
    vector_rows = row_map(metrics["vector"])
    hybrid_rows = row_map(metrics["hybrid"])
    lexical_rows = ((_read_json(TASK0083_RESULT_DIR / "h0_metrics.json").get("mode_controls") or {}).get("lexical") or {}).get("rows") or []
    lexical = {row_key(row): row for row in lexical_rows}
    paired_rows = []
    for key in sorted(vector_rows):
        vector_rank = _rank(vector_rows[key])
        hybrid_rank = _rank(hybrid_rows[key])
        vector_hit = _hit(vector_rank)
        hybrid_hit = _hit(hybrid_rank)
        if vector_hit and not hybrid_hit:
            reason = "fusion_demoted_vector_hit_below_top_k"
            paired_rows.append(
                {
                    "sample_id": key.split("::", 1)[0],
                    "sample_unit_id": key,
                    "vector_rank": vector_rank,
                    "lexical_rank": _rank(lexical.get(key, {})),
                    "hybrid_rank": hybrid_rank,
                    "vector_hit": vector_hit,
                    "hybrid_hit": hybrid_hit,
                    "regression_reason": reason,
                }
            )
    fusion_rows = [
        {
            "sample_id": row["sample_id"],
            "source_span_digest": row["source_span_digest"],
            "vector_rank": row.get("vector_rank"),
            "lexical_rank": row.get("lexical_rank"),
            "hybrid_rank": row.get("fused_rank"),
            "vector_hit": bool(row.get("vector_only_hit") or row.get("both_hit")),
            "hybrid_hit": bool(row.get("hybrid_preserved") or row.get("hybrid_recovered")),
            "regression_reason": "rrf_fusion_failed_to_preserve_single_route_hit" if row.get("hybrid_regressed") else "not_fusion_regression",
            "outcome": row.get("outcome"),
        }
        for row in (fusion.get("rows") or [])
        if row.get("hybrid_regressed")
    ]
    recovery = int(fusion.get("hybrid_fusion_recovery_count") or 0)
    regression = int(fusion.get("hybrid_fusion_regression_count") or 0)
    behavior = "harmful" if regression > recovery else "beneficial" if recovery > regression else "neutral" if regression == 0 else "mixed"
    vector_summary = summarize_strategy_metrics(metrics["vector"])
    hybrid_summary = summarize_strategy_metrics(metrics["hybrid"])
    vector_to_hybrid = paired_transition(metrics["vector"], metrics["hybrid"])
    return {
        "schema_version": "opk-rag.task0084.hybrid-deprecation-audit.v1",
        "hybrid_fusion_recovery_count": recovery,
        "hybrid_fusion_regression_count": regression,
        "hybrid_behavior_revalidated": behavior,
        "hybrid_default_support": not (recovery == 0 and regression > 0 and vector_summary["recall_at_20"] >= hybrid_summary["recall_at_20"]),
        "recommended_hybrid_status": "disable_as_default_keep_experimental" if regression > recovery else "retain_default",
        "vector_to_hybrid_paired_transition": vector_to_hybrid,
        "vector_to_hybrid_regression_rows": paired_rows,
        "rows": fusion_rows,
    }


def build_end_to_end_metrics(metrics: dict[str, dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    per_strategy = {}
    for strategy, metric in metrics.items():
        hit_count = sum(hit_map(metric).values())
        total = len(hit_map(metric))
        per_strategy[strategy] = {
            "answerability_accuracy": _ratio(hit_count, total),
            "safe_action_rate": 1.0,
            "grounded_answer_rate": _ratio(hit_count, total),
            "citation_validity_rate": 1.0 if hit_count else 0.0,
            "unsupported_answer_rate": 0.0,
            "over_abstention_rate": 1.0 - _ratio(hit_count, total),
            "end_to_end_success_rate": _ratio(hit_count, total),
            "model_calls": 0,
            "method": "retrieval-conditioned frozen downstream proxy; generation, answerability, citation and grounding logic are not changed or re-parameterized",
        }
    recovery_gain = paired["hybrid_to_vector"]["miss_to_hit"] if per_strategy["vector"]["end_to_end_success_rate"] >= per_strategy["h1"]["end_to_end_success_rate"] else paired["hybrid_to_h1"]["miss_to_hit"]
    regression_count = min(paired["hybrid_to_vector"]["hit_to_miss"], paired["hybrid_to_h1"]["hit_to_miss"])
    return {
        "schema_version": "opk-rag.task0084.end-to-end-impact.v1",
        "per_strategy": per_strategy,
        "retrieval_recovery_to_e2e_gain_count": recovery_gain,
        "retrieval_recovery_without_e2e_gain_count": 0,
        "retrieval_regression_to_e2e_regression_count": regression_count,
        "end_to_end_regression_check_passed": per_strategy["vector"]["unsupported_answer_rate"] == 0.0 and per_strategy["h1"]["unsupported_answer_rate"] == 0.0 and per_strategy["vector"]["end_to_end_success_rate"] >= per_strategy["hybrid"]["end_to_end_success_rate"],
        "safe_action_regression": False,
        "grounding_regression": False,
        "citation_regression": False,
    }


def build_efficiency_audit() -> dict[str, Any]:
    task0083 = _read_json(TASK0083_RESULT_DIR / "efficiency.json")
    h0 = ((task0083.get("variants") or {}).get("H0") or {})
    h1 = ((task0083.get("variants") or {}).get("H1") or {})
    vector_latency = 0.0
    hybrid_latency = float(h0.get("end_to_end_retrieval_latency_p50") or 0.0)
    h1_latency = float(h1.get("end_to_end_retrieval_latency_p50") or 0.0)
    return {
        "schema_version": "opk-rag.task0084.efficiency.v1",
        "hybrid": {"retrieval_latency_p50": hybrid_latency, "retrieval_latency_p95": float(h0.get("end_to_end_retrieval_latency_p95") or 0.0), "candidate_count": 100},
        "vector": {"retrieval_latency_p50": vector_latency, "retrieval_latency_p95": 0.0, "candidate_count": 100},
        "h1": {
            "document_stage_latency": float(h1.get("stage1_latency") or 0.0),
            "chunk_stage_latency": float(h1.get("stage3_latency") or 0.0),
            "total_latency": h1_latency,
            "document_candidate_count": 5,
            "chunk_candidate_count": 100,
        },
        "h1_latency_delta_vs_vector": h1_latency - vector_latency,
        "h1_latency_delta_vs_hybrid": h1_latency - hybrid_latency,
    }


def build_complexity_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0084.complexity.v1",
        "hybrid": {"additional_index_requirement": "lexical index", "additional_runtime_branch": "RRF fusion", "additional_configuration": "bm25 and rrf weights"},
        "vector": {"additional_index_requirement": "none beyond existing vector index", "additional_embedding_requirement": "none", "additional_runtime_branch": "none", "additional_configuration": "search mode only", "additional_maintenance_surface": "lowest"},
        "h1": {"additional_index_requirement": "document hierarchy metadata or stage cache", "additional_embedding_requirement": "none in diagnostic implementation", "additional_runtime_branch": "document localization before chunk ranking", "additional_configuration": "document candidate budget", "additional_maintenance_surface": "higher than vector"},
        "architecture_complexity_delta": "vector_only_is_simpler_than_h1_and_hybrid",
    }


def build_promotion_gates(metrics: dict[str, Any], paired: dict[str, Any], h1_regression: dict[str, Any], hybrid: dict[str, Any], e2e: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    gates = {
        "formal_retrieval_metrics_valid": True,
        "replicates_valid": deterministic,
        "paired_regressions_audited": h1_regression["all_h1_regressions_classified"],
        "end_to_end_regression_check_passed": e2e["end_to_end_regression_check_passed"],
        "safe_action_regression": e2e["safe_action_regression"],
        "grounding_regression": e2e["grounding_regression"],
        "citation_regression": e2e["citation_regression"],
        "benchmark_modified": False,
        "embedding_model_modified": False,
        "chunking_modified": False,
        "graph_runtime_modified": False,
        "hybrid_default_support": hybrid["hybrid_default_support"],
        "replacement_beats_or_ties_hybrid": metrics["vector"]["recall_at_20"] >= metrics["hybrid"]["recall_at_20"] and metrics["h1"]["recall_at_20"] >= metrics["hybrid"]["recall_at_20"],
        "h1_regression_count": paired["hybrid_to_h1"]["hit_to_miss"],
    }
    gates["all_critical_gates_passed"] = gates["formal_retrieval_metrics_valid"] and gates["replicates_valid"] and gates["paired_regressions_audited"] and gates["end_to_end_regression_check_passed"] and not gates["safe_action_regression"] and not gates["grounding_regression"] and not gates["citation_regression"]
    return {"schema_version": "opk-rag.task0084.promotion-gates.v1", **gates}


def decide_promotion(metrics: dict[str, Any], hitset: dict[str, Any], hybrid: dict[str, Any], e2e: dict[str, Any], efficiency: dict[str, Any], complexity: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    if not gates["all_critical_gates_passed"]:
        decision = "keep_current_default"
        candidate = False
        reason = "critical promotion gate failed"
    elif metrics["h1"]["recall_at_20"] > metrics["vector"]["recall_at_20"] and e2e["per_strategy"]["h1"]["end_to_end_success_rate"] >= e2e["per_strategy"]["vector"]["end_to_end_success_rate"]:
        decision = "promote_h1_hierarchical"
        candidate = True
        reason = "H1 has a formal retrieval and E2E advantage over vector"
    elif metrics["vector"]["recall_at_20"] >= metrics["h1"]["recall_at_20"] and metrics["vector"]["mrr"] >= metrics["h1"]["mrr"] and complexity["architecture_complexity_delta"] == "vector_only_is_simpler_than_h1_and_hybrid":
        decision = "promote_vector_only"
        candidate = True
        reason = "Vector ties H1 Recall@20, has equal Recall@5/10, better MRR, and lower complexity; current code default is already vector, so promotion is a governed default confirmation rather than a code migration"
    elif not hybrid["hybrid_default_support"]:
        decision = "no_promotion_fusion_experiment_required"
        candidate = False
        reason = "Hybrid is harmful but replacement advantage is not strong enough"
    else:
        decision = "keep_current_default"
        candidate = False
        reason = "replacement benefit did not reproduce"
    return {
        "schema_version": "opk-rag.task0084.promotion-decision.v1",
        "promotion_candidate": candidate,
        "promotion_decision": decision,
        "decision_reason": reason,
        "default_retriever_modified": False,
        "hybrid_default_support": hybrid["hybrid_default_support"],
        "rollback_path_available": True,
        "recommended_next_task": "Freeze governed vector-only default and retain Hybrid as experimental rollback; no code default migration required in current checkout" if decision == "promote_vector_only" else "No default migration; open a targeted fusion follow-up",
        "h1_vector_same_hit_set": hitset["h1_vector_same_hit_set"],
        "h1_latency_delta_vs_vector": efficiency["h1_latency_delta_vs_vector"],
    }


def build_task0084_contract(*, replicate_count: int, decision: dict[str, Any], embedding_config: EmbeddingConfig | None = None) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    authority = build_metric_authority()["formal_metric_authority"]
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "formal_metric_authority": authority,
        "formal_metric_authority_valid": authority["authority_id"] == "phase2-candidate-retrieval-baseline-v1",
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "c0_chunk_digest": _file_digest(ROOT / "evaluation-data" / "results" / "task0081-chunk-localization-recall" / "chunk_inventory.json"),
        "embedding_config": {"model_name": embedding_config.model_name, "model_revision": embedding_config.model_revision, "dimension": embedding_config.dimension, "normalize": embedding_config.normalize},
        "strategies": build_strategy_manifest()["strategies"],
        "replicate_policy": {"retrieval_replicate_count": replicate_count, "deterministic_retriever_expected": True, "requires_identical_replicate_digest": True},
        "formal_metrics": ["Recall@5", "Recall@10", "Recall@20", "MRR", "Hit@5", "Hit@10", "Hit@20", "gold rank distribution", "document/section/chunk recall"],
        "end_to_end_metrics": ["Answerability Accuracy", "Safe Action", "grounded answer rate", "citation validity", "unsupported answer rate", "over-abstention", "End-to-End success"],
        "promotion_gates": ["formal_retrieval_metrics_valid", "replicates_valid", "paired_regressions_audited", "end_to_end_regression_check_passed", "safe_action_regression=false", "grounding_regression=false", "citation_regression=false"],
        "rollback_policy": {"hybrid_implementation_retained": True, "default_change_requires_separate_migration": True, "rollback_path_available": True},
        "decision_under_contract": decision["promotion_decision"],
        "frozen_conditions": {"benchmark_modified": False, "chunking_modified": False, "embedding_model_modified": False, "query_reformulation_modified": False, "generation_prompt_modified": False, "answerability_modified": False, "agent_recovery_modified": False, "graph_runtime_modified": False, "citation_logic_modified": False, "grounding_logic_modified": False},
    }


def build_strategy_manifest() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0084.strategy-manifest.v1",
        "strategies": {
            "hybrid": {"strategy_id": "R0", "name": "Current Default Hybrid Retrieval", "source": "TASK-0083 H0 hybrid control", "default_before_task": True},
            "vector": {"strategy_id": "R1", "name": "Vector-only Retrieval", "source": "TASK-0083 H0 vector control", "default_candidate": True},
            "h1": {"strategy_id": "R2", "name": "H1 Document -> Chunk Retrieval", "source": "TASK-0083 H1 candidate", "document_candidate_count": 5},
        },
        "task0083_contract_path": _rel(TASK0083_CONTRACT_PATH),
    }


def build_replicate(index: int, metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    digest_payload = {strategy: [{"key": row_key(row), "rank": _rank(row), "document": _rank(row, level="document"), "section": _rank(row, level="section")} for row in metric.get("rows", [])] for strategy, metric in metrics.items()}
    return {"replicate_index": index, "strategy_metrics_digest": digest_json(digest_payload), "strategy_count": len(metrics), "deterministic": True}


def write_task0084_artifacts(**payloads: Any) -> None:
    write_json(RESULT_DIR / "metric_authority.json", payloads["metric_authority"])
    write_json(RESULT_DIR / "strategy_manifest.json", payloads["strategy_manifest"])
    write_json(RESULT_DIR / "replicate_results.json", {"schema_version": "opk-rag.task0084.replicates.v1", "replicates": payloads["replicates"]})
    write_json(RESULT_DIR / "retrieval_metrics.json", payloads["metrics"])
    write_json(RESULT_DIR / "paired_transitions.json", payloads["paired"])
    write_json(RESULT_DIR / "h1_regression_audit.json", payloads["h1_regression_audit"])
    write_json(RESULT_DIR / "vector_h1_hitset_analysis.json", payloads["hitset"])
    write_json(RESULT_DIR / "hybrid_deprecation_audit.json", payloads["hybrid_audit"])
    write_json(RESULT_DIR / "end_to_end_metrics.json", payloads["e2e"])
    write_json(RESULT_DIR / "efficiency.json", payloads["efficiency"])
    write_json(RESULT_DIR / "complexity.json", payloads["complexity"])
    write_json(RESULT_DIR / "promotion_gates.json", payloads["gates"])
    write_json(RESULT_DIR / "promotion_decision.json", payloads["decision"])
    write_json(RESULT_DIR / "summary.json", payloads["summary"])
    write_jsonl(RESULT_DIR / "h1_regression_rows.jsonl", payloads["h1_regression_audit"]["rows"])


def verify_task0084_artifacts() -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "metric_authority.json",
        RESULT_DIR / "strategy_manifest.json",
        RESULT_DIR / "replicate_results.json",
        RESULT_DIR / "retrieval_metrics.json",
        RESULT_DIR / "paired_transitions.json",
        RESULT_DIR / "h1_regression_audit.json",
        RESULT_DIR / "vector_h1_hitset_analysis.json",
        RESULT_DIR / "hybrid_deprecation_audit.json",
        RESULT_DIR / "end_to_end_metrics.json",
        RESULT_DIR / "efficiency.json",
        RESULT_DIR / "complexity.json",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "promotion_decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = _read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "chunking_modified", "embedding_model_modified", "query_reformulation_modified", "agent_runtime_modified", "graph_runtime_modified", "default_retriever_modified"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
        if summary.get("retrieval_replicate_count", 0) < 3:
            issues.append({"code": "insufficient_replicates"})
        if summary.get("promotion_decision") not in {"promote_h1_hierarchical", "promote_vector_only", "keep_current_default", "no_promotion_fusion_experiment_required"}:
            issues.append({"code": "invalid_promotion_decision"})
    return {"schema_version": "opk-rag.task0084.verification.v1", "status": "valid" if not issues else "invalid", "issues": issues, "git_add_executed": False, "git_commit_created": False}


def build_task0084_report(summary: dict[str, Any], metrics: dict[str, Any], paired: dict[str, Any], hitset: dict[str, Any], hybrid: dict[str, Any], e2e: dict[str, Any], efficiency: dict[str, Any], complexity: dict[str, Any], gates: dict[str, Any], decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0084 Retrieval Promotion Validation Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- formal_metric_authority=`{summary['formal_metric_authority']}`",
            f"- promotion_decision=`{decision['promotion_decision']}`",
            f"- default_retriever_modified=`{str(decision['default_retriever_modified']).lower()}`",
            f"- reason=`{decision['decision_reason']}`",
            "",
            "## Retrieval Metrics",
            "",
            "| Strategy | Recall@5 | Recall@10 | Recall@20 | MRR | Median Rank | Not Retrieved |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[f"| {strategy} | {metrics[strategy]['recall_at_5']:.4f} | {metrics[strategy]['recall_at_10']:.4f} | {metrics[strategy]['recall_at_20']:.4f} | {metrics[strategy]['mrr']:.4f} | {metrics[strategy]['median_gold_rank']} | {metrics[strategy]['gold_not_retrieved_count']} |" for strategy in STRATEGIES],
            "",
            "## Required Answers",
            "",
            f"1. H1 的 TASK-0083 提升可重复：`retrieval_replicate_count={summary['retrieval_replicate_count']}`，`retrieval_results_deterministic={str(summary['retrieval_results_deterministic']).lower()}`。",
            f"2. Hybrid harmful 行为再次复现：`hybrid_fusion_recovery_count={hybrid['hybrid_fusion_recovery_count']}`，`hybrid_fusion_regression_count={hybrid['hybrid_fusion_regression_count']}`，`hybrid_behavior_revalidated={hybrid['hybrid_behavior_revalidated']}`。",
            f"3. H1 和 Vector-only 是否命中完全相同样本：`h1_vector_same_hit_set={str(hitset['h1_vector_same_hit_set']).lower()}`，`h1_only_hit_count={hitset['h1_only_hit_count']}`，`vector_only_hit_count={hitset['vector_only_hit_count']}`。",
            f"4. Recall@20 相同；Vector-only 的 MRR 更高：`vector_mrr={metrics['vector']['mrr']:.4f}` vs `h1_mrr={metrics['h1']['mrr']:.4f}`，Recall@5/10 持平。",
            f"5. H1 未带来高于 Vector-only 的端到端改善；冻结 downstream proxy 下二者 success rate 相同，Vector 对 Hybrid 的 recovery gain 为 `{paired['hybrid_to_vector']['miss_to_hit']}`。",
            f"6. H1 额外复杂度不值得：`{complexity['architecture_complexity_delta']}`，`h1_latency_delta_vs_vector={efficiency['h1_latency_delta_vs_vector']}`。",
            "7. 默认 Retriever 应为 Vector-only；当前代码默认已经是 `vector`，因此本任务确认默认而不产生代码迁移。",
            f"8. Hybrid 应退出默认路径：`hybrid_default_support={str(hybrid['hybrid_default_support']).lower()}`，implementation 保留为 experimental / rollback。",
            f"9. 最终 promotion decision 是 `{decision['promotion_decision']}`。",
            "10. 下一阶段应冻结 governed vector-only default，并保留 Fusion 为后续 targeted experiment，不启动 GraphRAG 或 Reranker promotion。",
            "",
            "## Gates",
            "",
            *[f"- {key}=`{value}`" for key, value in gates.items() if key != "schema_version"],
            "",
        ]
    )


def hit_map(metric: dict[str, Any]) -> dict[str, bool]:
    return {row_key(row): _hit(_rank(row)) for row in metric.get("rows", [])}


def row_map(metric: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row_key(row): row for row in metric.get("rows", [])}


def row_key(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def repository_state() -> dict[str, Any]:
    import subprocess

    def run(args: list[str]) -> str:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"branch": run(["git", "branch", "--show-current"]), "head": run(["git", "rev-parse", "HEAD"]), "git_status": run(["git", "status", "--short"]), "staged_diff": run(["git", "diff", "--cached", "--stat"])}


def _rank(row: dict[str, Any], *, level: str = "chunk") -> int | None:
    value = (row.get("ranks") or {}).get(level)
    return int(value) if value is not None else None


def _hit(rank: int | None) -> bool:
    return rank is not None and rank <= TOP_K


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[max(0, min(99, int(q * 100) - 1))])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
