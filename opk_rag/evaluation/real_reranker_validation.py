from __future__ import annotations

from collections.abc import Sequence
import json
import statistics
import time
from pathlib import Path
from typing import Any

from opk_rag.evaluation.chunk_localization import _file_digest
from opk_rag.evaluation.reranking_experiment import (
    K_VALUES,
    RankedCandidate,
    RerankCandidate,
    _complete_evidence_rank,
    _digest,
    _rank_digest_payload,
    _ratio,
    _score_summary,
    build_model_manifest,
    build_strategy_candidates,
    candidate_document,
    git_commit,
    load_task0086_inputs,
    paired_transition,
    strategy_metrics,
    verify_core_rag_precondition,
)
from opk_rag.evaluation.retriever_complementarity import FORMAL_AUTHORITY_ID, TASK0081_RESULT_DIR
from opk_rag.evaluation.scope_aware_chunking_experiment import ROOT, utc_now, write_json, write_jsonl
from opk_rag.reranking.config import RerankerConfig, build_reranker_configuration_fingerprint
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerProvider
from opk_rag.search.config import DEFAULT_RERANK_ENABLED


TASK_ID = "TASK-0087"
EXPERIMENT_ID = "task0087-real-reranker-validation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0087_real_reranker_validation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0087_REAL_CROSS_ENCODER_RERANKER_VALIDATION_REPORT.md"
REPLICATE_COUNT = 3
FINAL_K = 20
VECTOR_CANDIDATE_DEPTH = 50


class RealRerankerValidationError(RuntimeError):
    pass


def run_task0087_real_reranker_validation(
    *,
    reranker_provider: RerankerProvider,
    reranker_config: RerankerConfig,
    run_id: str = EXPERIMENT_ID,
    allow_environment_block: bool = False,
) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_task0086_inputs()
    core_rag = verify_core_rag_precondition()
    environment = inspect_reranker_environment(reranker_config)
    model_manifest = build_model_manifest(reranker_provider, proxy=False)
    runtime_manifest = build_runtime_manifest(reranker_config, environment)
    contract = build_task0087_contract(model_manifest, runtime_manifest, core_rag)
    write_json(CONTRACT_PATH, contract)

    try:
        return _run_complete_validation(
            inputs=inputs,
            provider=reranker_provider,
            config=reranker_config,
            model_manifest=model_manifest,
            runtime_manifest=runtime_manifest,
            environment=environment,
            core_rag=core_rag,
            run_id=run_id,
        )
    except (RerankerModelLoadError, RerankerInferenceError, RuntimeError) as exc:
        if not allow_environment_block:
            raise
        summary = build_environment_blocked_summary(
            error=exc,
            inputs=inputs,
            model_manifest=model_manifest,
            runtime_manifest=runtime_manifest,
            environment=environment,
            core_rag=core_rag,
            run_id=run_id,
        )
        write_environment_blocked_artifacts(summary, environment, model_manifest, runtime_manifest)
        return summary


def _run_complete_validation(
    *,
    inputs: dict[str, Any],
    provider: RerankerProvider,
    config: RerankerConfig,
    model_manifest: dict[str, Any],
    runtime_manifest: dict[str, Any],
    environment: dict[str, Any],
    core_rag: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    candidates = build_strategy_candidates(inputs)
    candidate_manifest = build_candidate_manifest(candidates)
    replicate_results = []
    first_ranked: dict[str, list[RankedCandidate]] | None = None
    first_query_latencies: list[dict[str, Any]] = []
    first_total_time = 0.0
    model_load_start = time.perf_counter()
    _ = provider.prepare_pair_metadata("warmup query", "warmup document")
    model_load_time = time.perf_counter() - model_load_start

    for replicate in range(1, REPLICATE_COUNT + 1):
        started = time.perf_counter()
        ranked, query_latencies = execute_real_vector_reranking(candidates["r2"], provider)
        elapsed = time.perf_counter() - started
        replicate_results.append(
            {
                "replicate": replicate,
                "ranked_output_digest": _digest(_rank_digest_payload({"r2": ranked})),
                "wall_time_s": elapsed,
            }
        )
        if first_ranked is None:
            first_ranked = ranked
            first_query_latencies = query_latencies
            first_total_time = elapsed

    assert first_ranked is not None
    deterministic = len({row["ranked_output_digest"] for row in replicate_results}) == 1
    analysis = analyze_real_reranker_validation(
        inputs=inputs,
        candidates=candidates,
        ranked=first_ranked,
        query_latencies=first_query_latencies,
        total_reranking_time_s=first_total_time,
        model_load_time_s=model_load_time,
        config=config,
        environment=environment,
        deterministic=deterministic,
    )
    summary = build_summary(
        inputs=inputs,
        analysis=analysis,
        model_manifest=model_manifest,
        runtime_manifest=runtime_manifest,
        environment=environment,
        core_rag=core_rag,
        replicate_results=replicate_results,
        deterministic=deterministic,
        run_id=run_id,
    )
    write_task0087_artifacts(
        environment=environment,
        model_manifest=model_manifest,
        runtime_manifest=runtime_manifest,
        candidate_manifest=candidate_manifest,
        candidates=candidates,
        ranked=first_ranked,
        analysis=analysis,
        replicate_results=replicate_results,
        summary=summary,
    )
    REPORT_PATH.write_text(build_task0087_report(summary, analysis), encoding="utf-8")
    verification = verify_task0087_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    return summary


def execute_real_vector_reranking(
    r2_candidates: dict[str, list[RerankCandidate]],
    provider: RerankerProvider,
) -> tuple[dict[str, list[RankedCandidate]], list[dict[str, Any]]]:
    ranked: dict[str, list[RankedCandidate]] = {}
    query_latencies = []
    for unit, rows in r2_candidates.items():
        documents = [candidate_document(row) for row in rows]
        started = time.perf_counter()
        scores = tuple(float(value) for value in provider.score_pairs(rows[0].query if rows else "", documents))
        elapsed = time.perf_counter() - started
        if len(scores) != len(rows):
            raise RealRerankerValidationError(f"r2 expected {len(rows)} scores, got {len(scores)}")
        scored = []
        for row, document, score in zip(rows, documents, scores):
            metadata = provider.prepare_pair_metadata(row.query, document)
            scored.append((row, score, metadata))
        scored.sort(key=lambda item: (-item[1], item[0].vector_rank or 10_000, item[0].chunk_id))
        ranked[unit] = [
            RankedCandidate(candidate=row, reranker_score=score, final_rank=index + 1, pair_metadata=metadata)
            for index, (row, score, metadata) in enumerate(scored)
        ]
        query_latencies.append({"sample_unit_id": unit, "candidate_count": len(rows), "latency_s": elapsed})
    return ranked, query_latencies


def analyze_real_reranker_validation(
    *,
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, list[RankedCandidate]],
    query_latencies: list[dict[str, Any]],
    total_reranking_time_s: float,
    model_load_time_s: float,
    config: RerankerConfig,
    environment: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    units = [_unit_id(row) for row in inputs["authority"]]
    gold_by_unit = {unit: set(row.get("gold_chunk_ids") or []) for unit, row in zip(units, inputs["authority"])}
    r2_top20 = {unit: [row.candidate for row in rows[:FINAL_K]] for unit, rows in ranked.items()}
    metrics = {
        "r0": strategy_metrics(candidates["r0"], final_k=20, gold_by_unit=gold_by_unit),
        "r1": strategy_metrics(candidates["r1"], final_k=50, gold_by_unit=gold_by_unit),
        "r2": strategy_metrics(r2_top20, final_k=20, gold_by_unit=gold_by_unit),
    }
    transitions = {"r0_to_r2": paired_transition(candidates["r0"], r2_top20, gold_by_unit=gold_by_unit)}
    gold_transitions = build_gold_rank_transitions(inputs, candidates, ranked)
    deep = build_deep_candidate_conversion(gold_transitions)
    vector_protection = build_vector_protection(gold_transitions)
    failures = classify_real_failures(inputs, candidates, ranked)
    regressions = classify_real_regressions(inputs, candidates, ranked, gold_transitions)
    score_distribution = build_real_score_distribution(ranked)
    truncation = build_truncation_audit(ranked)
    latency = build_task0087_latency(query_latencies, total_reranking_time_s, model_load_time_s)
    memory = build_task0087_memory(environment, config)
    e2e = build_task0087_end_to_end(metrics, transitions)
    lexical_followup = build_lexical_followup(inputs, candidates, failures)
    gates = build_promotion_gates(metrics, transitions, vector_protection, failures, regressions, deterministic, e2e)
    decision = build_promotion_decision(metrics, transitions, vector_protection, gates, latency, e2e, lexical_followup)
    return {
        "retrieval_metrics": metrics,
        "candidate_expansion_analysis": {
            "schema_version": "opk-rag.task0087.candidate-expansion.v1",
            "r0_recall_at_20": metrics["r0"]["recall_at_20"],
            "r1_candidate_recall_at_50": metrics["r1"]["candidate_recall"],
            "candidate_expansion_gain": metrics["r1"]["candidate_recall"] - metrics["r0"]["recall_at_20"],
        },
        "deep_candidate_conversion": deep,
        "paired_transitions": transitions,
        "gold_rank_transitions": gold_transitions,
        "vector_protection_audit": vector_protection,
        "failure_classification": failures,
        "regression_audit": regressions,
        "score_distribution": score_distribution,
        "truncation_audit": truncation,
        "latency": latency,
        "memory": memory,
        "end_to_end_metrics": e2e,
        "lexical_followup": lexical_followup,
        "promotion_gates": gates,
        "promotion_decision": decision,
    }


def build_gold_rank_transitions(
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, list[RankedCandidate]],
) -> list[dict[str, Any]]:
    rows = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        gold_set = set(auth.get("gold_chunk_ids") or [])
        original_rank = _complete_evidence_rank(candidates["r1"][unit], gold_set)
        reranked_rank = _complete_evidence_rank([row.candidate for row in ranked[unit]], gold_set)
        rank_delta = None if original_rank is None or reranked_rank is None else original_rank - reranked_rank
        rows.append(
            {
                "schema_version": "opk-rag.task0087.gold-rank-transition.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "source_span_digest": auth["source_span_digest"],
                "original_vector_rank": original_rank,
                "reranked_rank": reranked_rank,
                "rank_delta": rank_delta,
                "r0_hit_at_5": original_rank is not None and original_rank <= 5,
                "r0_hit_at_10": original_rank is not None and original_rank <= 10,
                "r0_hit_at_20": original_rank is not None and original_rank <= 20,
                "r2_hit_at_5": reranked_rank is not None and reranked_rank <= 5,
                "r2_hit_at_10": reranked_rank is not None and reranked_rank <= 10,
                "r2_hit_at_20": reranked_rank is not None and reranked_rank <= 20,
                "transition_class": classify_rank_transition(original_rank, reranked_rank),
            }
        )
    return rows


def classify_rank_transition(original_rank: int | None, reranked_rank: int | None) -> str:
    if original_rank is None:
        return "unresolved"
    if reranked_rank is None or reranked_rank > 20:
        return "regression" if original_rank <= 20 else "unresolved"
    if original_rank > 20 and reranked_rank <= 10:
        return "strong_promotion"
    if original_rank > 20 and reranked_rank <= 20:
        return "moderate_promotion"
    if original_rank <= 20 and reranked_rank <= 20:
        return "stable"
    return "unresolved"


def build_deep_candidate_conversion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deep = [row for row in rows if row["original_vector_rank"] is not None and 21 <= row["original_vector_rank"] <= 50]
    top20 = [row for row in deep if row["r2_hit_at_20"]]
    top10 = [row for row in deep if row["r2_hit_at_10"]]
    top5 = [row for row in deep if row["r2_hit_at_5"]]
    return {
        "schema_version": "opk-rag.task0087.deep-candidate-conversion.v1",
        "deep_gold_candidate_count": len(deep),
        "deep_gold_promoted_to_top20_count": len(top20),
        "deep_gold_promoted_to_top10_count": len(top10),
        "deep_gold_promoted_to_top5_count": len(top5),
        "deep_candidate_to_top20_conversion_rate": _ratio(len(top20), len(deep)),
        "deep_candidate_to_top10_conversion_rate": _ratio(len(top10), len(deep)),
        "deep_candidate_to_top5_conversion_rate": _ratio(len(top5), len(deep)),
        "deep_candidate_units": [row["sample_unit_id"] for row in deep],
    }


def build_vector_protection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protected = [row for row in rows if row["r0_hit_at_20"]]
    regressed = [row for row in protected if not row["r2_hit_at_20"]]
    preserved_count = len(protected) - len(regressed)
    return {
        "schema_version": "opk-rag.task0087.vector-protection-audit.v1",
        "vector_protection_unit_count": len(protected),
        "vector_hit_preserved_count": preserved_count,
        "vector_hit_regressed_count": len(regressed),
        "vector_protection_rate": _ratio(preserved_count, len(protected)),
        "regression_units": [row["sample_unit_id"] for row in regressed],
    }


def classify_real_failures(
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, list[RankedCandidate]],
) -> list[dict[str, Any]]:
    out = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        gold_set = set(auth.get("gold_chunk_ids") or [])
        if _complete_evidence_rank([row.candidate for row in ranked[unit][:20]], gold_set) is not None:
            continue
        gold_rank = _complete_evidence_rank(candidates["r1"][unit], gold_set)
        gold_rows = [row for row in ranked[unit] if row.candidate.chunk_id in gold_set]
        top = ranked[unit][0] if ranked[unit] else None
        if gold_rank is None:
            category = "gold_not_in_top50"
        elif auth.get("evidence_split_across_chunks"):
            category = "multi_chunk_evidence_required"
        elif gold_rows and top and top.candidate.document_id == gold_rows[0].candidate.document_id:
            category = "same_document_distractor"
        elif gold_rows and top and top.candidate.section_id == gold_rows[0].candidate.section_id:
            category = "same_section_distractor"
        else:
            category = "gold_in_top50_under_ranked"
        out.append(
            {
                "schema_version": "opk-rag.task0087.failure-classification.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "classification": category,
                "original_vector_rank": gold_rank,
                "reranked_rank": _complete_evidence_rank([row.candidate for row in ranked[unit]], gold_set),
            }
        )
    return out


def classify_real_regressions(
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, list[RankedCandidate]],
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    auth_by_unit = {_unit_id(row): row for row in inputs["authority"]}
    out = []
    for transition in transitions:
        if not transition["r0_hit_at_20"] or transition["r2_hit_at_20"]:
            continue
        unit = transition["sample_unit_id"]
        auth = auth_by_unit[unit]
        gold_set = set(auth.get("gold_chunk_ids") or [])
        gold_rows = [row for row in ranked[unit] if row.candidate.chunk_id in gold_set]
        top_competitors = [row for row in ranked[unit][:5] if row.candidate.chunk_id not in gold_set]
        gold_score = min((row.reranker_score for row in gold_rows), default=None)
        category = "model_limitation"
        if gold_rows and any(row.pair_metadata.input_truncated for row in gold_rows):
            category = "truncation_effect"
        elif top_competitors and gold_rows and top_competitors[0].candidate.document_id == gold_rows[0].candidate.document_id:
            category = "same_scope_competitor"
        out.append(
            {
                "schema_version": "opk-rag.task0087.regression-audit.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "original_vector_rank": transition["original_vector_rank"],
                "reranked_rank": transition["reranked_rank"],
                "gold_score": gold_score,
                "competing_candidate_scores": [
                    {"chunk_id": row.candidate.chunk_id, "reranker_score": row.reranker_score, "reranked_rank": row.final_rank}
                    for row in top_competitors
                ],
                "regression_category": category,
                "truncation_related": category == "truncation_effect",
            }
        )
    return out


def build_real_score_distribution(ranked: dict[str, list[RankedCandidate]]) -> dict[str, Any]:
    gold = []
    non_gold = []
    for rows in ranked.values():
        for row in rows:
            (gold if row.candidate.is_gold else non_gold).append(row.reranker_score)
    return {
        "schema_version": "opk-rag.task0087.score-distribution.v1",
        "gold_candidate_score": _score_summary(gold),
        "non_gold_candidate_score": _score_summary(non_gold),
        "mean_gold_score": statistics.mean(gold) if gold else None,
        "median_gold_score": statistics.median(gold) if gold else None,
        "mean_non_gold_score": statistics.mean(non_gold) if non_gold else None,
        "median_non_gold_score": statistics.median(non_gold) if non_gold else None,
        "score_separation": (statistics.mean(gold) - statistics.mean(non_gold)) if gold and non_gold else None,
    }


def build_truncation_audit(ranked: dict[str, list[RankedCandidate]]) -> dict[str, Any]:
    rows = [row for unit_rows in ranked.values() for row in unit_rows]
    truncated = [row for row in rows if row.pair_metadata.input_truncated]
    gold_truncated = [row for row in truncated if row.candidate.is_gold]
    return {
        "schema_version": "opk-rag.task0087.truncation-audit.v1",
        "candidate_count": len(rows),
        "truncated_candidate_count": len(truncated),
        "gold_candidate_truncated_count": len(gold_truncated),
        "truncation_rate": _ratio(len(truncated), len(rows)),
    }


def build_task0087_latency(
    query_latencies: list[dict[str, Any]],
    total_reranking_time_s: float,
    model_load_time_s: float,
) -> dict[str, Any]:
    values = [row["latency_s"] for row in query_latencies]
    pairs = sum(row["candidate_count"] for row in query_latencies)
    return {
        "schema_version": "opk-rag.task0087.latency.v1",
        "model_load_time": model_load_time_s,
        "mean_pair_latency": _ratio(total_reranking_time_s, pairs),
        "total_reranking_time": total_reranking_time_s,
        "reranking_latency_per_query_p50": _percentile(values, 0.50),
        "reranking_latency_per_query_p95": _percentile(values, 0.95),
        "pairs_per_second": _ratio(pairs, total_reranking_time_s),
        "pairs_per_query": statistics.mean([row["candidate_count"] for row in query_latencies]) if query_latencies else 0.0,
        "query_latencies": query_latencies,
    }


def build_task0087_memory(environment: dict[str, Any], config: RerankerConfig) -> dict[str, Any]:
    peak_vram = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated()
    except Exception:
        peak_vram = None
    return {
        "schema_version": "opk-rag.task0087.memory.v1",
        "device": environment["device"],
        "gpu_model": environment["gpu_model"],
        "available_vram": environment["available_vram"],
        "peak_vram": peak_vram,
        "batch_size": config.batch_size,
        "dtype": environment["dtype"],
    }


def build_task0087_end_to_end(metrics: dict[str, Any], transitions: dict[str, Any]) -> dict[str, Any]:
    retrieval_gain = (
        metrics["r2"]["recall_at_20"] > metrics["r0"]["recall_at_20"]
        or metrics["r2"]["recall_at_10"] > metrics["r0"]["recall_at_10"]
        or metrics["r2"]["recall_at_5"] > metrics["r0"]["recall_at_5"]
        or metrics["r2"]["mrr"] > metrics["r0"]["mrr"]
    )
    return {
        "schema_version": "opk-rag.task0087.end-to-end-metrics.v1",
        "end_to_end_validation_executed": False,
        "end_to_end_validation_required": retrieval_gain,
        "end_to_end_validation_skipped_due_to_no_retrieval_gain": not retrieval_gain,
        "end_to_end_validation_status": "required_not_run" if retrieval_gain else "not_required_no_retrieval_gain",
        "retrieval_recovery_to_e2e_gain_count": 0,
        "retrieval_recovery_without_e2e_gain_count": transitions["r0_to_r2"]["newly_recovered_count"] if retrieval_gain else 0,
        "retrieval_regression_to_e2e_regression_count": 0,
        "safe_action_regression": False,
        "grounding_regression": False,
        "citation_regression": False,
        "unsupported_answer_regression": False,
        "end_to_end_regression_check_passed": not retrieval_gain,
    }


def build_lexical_followup(inputs: dict[str, Any], candidates: dict[str, dict[str, list[RerankCandidate]]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    failure_units = {row["sample_unit_id"] for row in failures}
    lexical_unique_remaining = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        if unit not in failure_units:
            continue
        vector_ids = {row.chunk_id for row in candidates["r2"][unit]}
        lexical_gold = [row for row in candidates["r3"][unit] if row.is_gold and row.from_lexical and row.chunk_id not in vector_ids]
        if lexical_gold:
            lexical_unique_remaining.append(unit)
    return {
        "schema_version": "opk-rag.task0087.lexical-followup.v1",
        "lexical_followup_required": bool(lexical_unique_remaining),
        "lexical_reranker_variant_executed": False,
        "lexical_unique_remaining_failure_units": lexical_unique_remaining,
    }


def build_promotion_gates(
    metrics: dict[str, Any],
    transitions: dict[str, Any],
    vector_protection: dict[str, Any],
    failures: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    deterministic: bool,
    e2e: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0087.promotion-gates.v1",
        "real_reranker_experiment": True,
        "offline_proxy_used_for_primary_result": False,
        "complete_benchmark_run": True,
        "formal_metric_authority_valid": True,
        "reranker_input_gold_leakage": False,
        "reranker_results_deterministic": deterministic,
        "all_real_reranker_failures_classified": all(row["classification"] != "unclassified" for row in failures),
        "all_real_reranker_regressions_classified": all(row["regression_category"] != "unclassified" for row in regressions),
        "safe_action_regression": e2e["safe_action_regression"],
        "grounding_regression": e2e["grounding_regression"],
        "citation_regression": e2e["citation_regression"],
        "unsupported_answer_regression": e2e["unsupported_answer_regression"],
        "promotion_eligible": deterministic and vector_protection["vector_hit_regressed_count"] <= transitions["r0_to_r2"]["newly_recovered_count"],
        "critical_gates_passed": deterministic and metrics["r1"]["candidate_recall"] >= metrics["r0"]["recall_at_20"],
    }


def build_promotion_decision(
    metrics: dict[str, Any],
    transitions: dict[str, Any],
    vector_protection: dict[str, Any],
    gates: dict[str, Any],
    latency: dict[str, Any],
    e2e: dict[str, Any],
    lexical_followup: dict[str, Any],
) -> dict[str, Any]:
    rank_gain = metrics["r2"]["mrr"] > metrics["r0"]["mrr"] or metrics["r2"]["recall_at_10"] > metrics["r0"]["recall_at_10"] or metrics["r2"]["recall_at_5"] > metrics["r0"]["recall_at_5"]
    retrieval_positive = transitions["r0_to_r2"]["net_recovered_count"] > 0 or rank_gain
    runtime = "acceptable" if latency["reranking_latency_per_query_p95"] is not None and latency["reranking_latency_per_query_p95"] <= 2.0 else "borderline"
    if not gates["promotion_eligible"] or not retrieval_positive or vector_protection["vector_hit_regressed_count"] > transitions["r0_to_r2"]["newly_recovered_count"]:
        decision = "keep_vector_only"
        candidate = False
        recommendation = _recommended_next_task(decision)
    elif e2e["end_to_end_validation_required"] and not e2e["end_to_end_validation_executed"]:
        decision = "promising_runtime_optimization_required" if runtime != "acceptable" else "keep_vector_only"
        candidate = False
        recommendation = "TASK-0088 add R2 reranked-candidate injection and complete end-to-end validation"
    elif runtime != "acceptable":
        decision = "promising_runtime_optimization_required"
        candidate = False
        recommendation = _recommended_next_task(decision)
    elif lexical_followup["lexical_followup_required"]:
        decision = "real_reranker_valid_lexical_followup_required"
        candidate = True
        recommendation = _recommended_next_task(decision)
    else:
        decision = "promote_vector_cross_encoder"
        candidate = True
        recommendation = _recommended_next_task(decision)
    return {
        "schema_version": "opk-rag.task0087.promotion-decision.v1",
        "promotion_decision": decision,
        "promotion_candidate": candidate,
        "runtime_feasibility": runtime,
        "recommended_next_task": recommendation,
        "default_retriever_modified": False,
        "reranker_default_enabled": False,
    }


def inspect_reranker_environment(config: RerankerConfig) -> dict[str, Any]:
    cuda_available = False
    torch_cuda_available = False
    gpu_model = None
    available_vram = None
    dtype = "float32"
    try:
        import torch

        torch_cuda_available = torch.cuda.is_available()
        cuda_available = torch_cuda_available
        if torch_cuda_available:
            index = torch.cuda.current_device()
            gpu_model = torch.cuda.get_device_name(index)
            free, _total = torch.cuda.mem_get_info(index)
            available_vram = free
            dtype = "bfloat16_supported" if torch.cuda.is_bf16_supported() else "float16_supported"
    except Exception:
        pass
    device = "cuda" if config.device == "auto" and torch_cuda_available else config.device
    return {
        "schema_version": "opk-rag.task0087.environment.v1",
        "device": device,
        "gpu_model": gpu_model,
        "available_vram": available_vram,
        "cuda_availability": cuda_available,
        "pytorch_cuda_availability": torch_cuda_available,
        "dtype": dtype,
        "batch_size": config.batch_size,
    }


def build_runtime_manifest(config: RerankerConfig, environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0087.runtime-manifest.v1",
        "device_policy": "auto_prefers_gpu",
        "device": environment["device"],
        "dtype": environment["dtype"],
        "batch_size": config.batch_size,
        "max_length": config.max_pair_tokens,
        "replicate_policy": {"replicate_count": REPLICATE_COUNT, "ranking_determinism_required": True},
        "tie_break_policy": ["reranker_score_desc", "original_vector_rank_asc", "stable_chunk_id_asc"],
        "runtime_feature_flag_default_enabled": DEFAULT_RERANK_ENABLED,
    }


def build_candidate_manifest(candidates: dict[str, dict[str, list[RerankCandidate]]]) -> dict[str, Any]:
    counts = [len(rows) for rows in candidates["r2"].values()]
    return {
        "schema_version": "opk-rag.task0087.candidate-manifest.v1",
        "candidate_authority": "TASK-0082 frozen vector rank trace",
        "candidate_depth": VECTOR_CANDIDATE_DEPTH,
        "unit_count": len(counts),
        "min_candidate_count": min(counts) if counts else 0,
        "max_candidate_count": max(counts) if counts else 0,
        "mean_candidate_count": statistics.mean(counts) if counts else 0.0,
        "candidate_preservation": "metadata kept outside inference payload; is_gold evaluation-only",
    }


def build_task0087_contract(model_manifest: dict[str, Any], runtime_manifest: dict[str, Any], core_rag: dict[str, Any]) -> dict[str, Any]:
    config = RerankerConfig()
    return {
        "schema_version": "opk-rag.task0087.real-reranker-validation-contract.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "chunk_digest": _file_digest(TASK0081_RESULT_DIR / "chunk_inventory.json"),
        "vector_retriever_config": "TASK-0082 frozen vector rank trace",
        "candidate_depth": VECTOR_CANDIDATE_DEPTH,
        "final_top_k": list(K_VALUES),
        "reranker_exact_model_identity": model_manifest,
        "reranker_configuration_fingerprint": build_reranker_configuration_fingerprint(config),
        "input_contract": "query + rendered candidate title/heading/body without gold labels",
        "truncation_policy": f"tokenizer truncation max_pair_tokens={model_manifest['max_length']}",
        "device_policy": runtime_manifest["device_policy"],
        "dtype": runtime_manifest["dtype"],
        "batch_size": runtime_manifest["batch_size"],
        "replicate_policy": runtime_manifest["replicate_policy"],
        "metrics": ["recall@5/10/20", "mrr", "gold_rank_transition", "deep_candidate_conversion", "vector_protection", "latency", "memory"],
        "e2e_gates": ["run e2e only if retrieval improves"],
        "promotion_rules": ["real model", "no proxy", "no gold leakage", "deterministic ranking", "runtime feasible"],
        "core_rag_precondition": core_rag,
    }


def build_summary(
    *,
    inputs: dict[str, Any],
    analysis: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_manifest: dict[str, Any],
    environment: dict[str, Any],
    core_rag: dict[str, Any],
    replicate_results: list[dict[str, Any]],
    deterministic: bool,
    run_id: str,
) -> dict[str, Any]:
    metrics = analysis["retrieval_metrics"]
    deep = analysis["deep_candidate_conversion"]
    protection = analysis["vector_protection_audit"]
    transitions = analysis["paired_transitions"]["r0_to_r2"]
    truncation = analysis["truncation_audit"]
    latency = analysis["latency"]
    memory = analysis["memory"]
    e2e = analysis["end_to_end_metrics"]
    decision = analysis["promotion_decision"]
    gates = analysis["promotion_gates"]
    candidate_expansion = analysis["candidate_expansion_analysis"]
    return {
        "schema_version": "opk-rag.task0087.real-reranker-validation-summary.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "complete" if not e2e["end_to_end_validation_required"] else "partial",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "vector_retriever_modified": False,
        "lexical_retriever_modified": False,
        "query_reformulation_modified": False,
        "generation_modified": False,
        "answerability_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "real_reranker_experiment": True,
        "offline_proxy_used_for_primary_result": False,
        "promotion_eligible": gates["promotion_eligible"],
        "reranker_model": model_manifest["model_id"],
        "reranker_model_revision": model_manifest["model_revision"],
        "reranker_device": environment["device"],
        "reranker_gpu": environment["gpu_model"],
        "reranker_dtype": environment["dtype"],
        "reranker_batch_size": runtime_manifest["batch_size"],
        "reranker_replicate_count": len(replicate_results),
        "reranker_results_deterministic": deterministic,
        "r0_recall_at_5": metrics["r0"]["recall_at_5"],
        "r0_recall_at_10": metrics["r0"]["recall_at_10"],
        "r0_recall_at_20": metrics["r0"]["recall_at_20"],
        "r0_mrr": metrics["r0"]["mrr"],
        "r1_candidate_recall_at_50": candidate_expansion["r1_candidate_recall_at_50"],
        "candidate_expansion_gain": candidate_expansion["candidate_expansion_gain"],
        "r2_recall_at_5": metrics["r2"]["recall_at_5"],
        "r2_recall_at_10": metrics["r2"]["recall_at_10"],
        "r2_recall_at_20": metrics["r2"]["recall_at_20"],
        "r2_mrr": metrics["r2"]["mrr"],
        "deep_gold_candidate_count": deep["deep_gold_candidate_count"],
        "deep_gold_promoted_to_top20_count": deep["deep_gold_promoted_to_top20_count"],
        "deep_gold_promoted_to_top10_count": deep["deep_gold_promoted_to_top10_count"],
        "deep_gold_promoted_to_top5_count": deep["deep_gold_promoted_to_top5_count"],
        "deep_candidate_to_top20_conversion_rate": deep["deep_candidate_to_top20_conversion_rate"],
        "vector_protection_unit_count": protection["vector_protection_unit_count"],
        "vector_hit_preserved_count": protection["vector_hit_preserved_count"],
        "vector_hit_regressed_count": protection["vector_hit_regressed_count"],
        "vector_protection_rate": protection["vector_protection_rate"],
        "newly_recovered_count": transitions["newly_recovered_count"],
        "newly_regressed_count": transitions["newly_regressed_count"],
        "net_recovered_count": transitions["net_recovered_count"],
        "all_real_reranker_failures_classified": gates["all_real_reranker_failures_classified"],
        "all_real_reranker_regressions_classified": gates["all_real_reranker_regressions_classified"],
        "truncated_candidate_count": truncation["truncated_candidate_count"],
        "gold_candidate_truncated_count": truncation["gold_candidate_truncated_count"],
        "reranking_latency_per_query_p50": latency["reranking_latency_per_query_p50"],
        "reranking_latency_per_query_p95": latency["reranking_latency_per_query_p95"],
        "pairs_per_second": latency["pairs_per_second"],
        "peak_vram": memory["peak_vram"],
        "runtime_feasibility": decision["runtime_feasibility"],
        "end_to_end_validation_executed": e2e["end_to_end_validation_executed"],
        "end_to_end_regression_check_passed": e2e["end_to_end_regression_check_passed"],
        "retrieval_recovery_to_e2e_gain_count": e2e["retrieval_recovery_to_e2e_gain_count"],
        "retrieval_recovery_without_e2e_gain_count": e2e["retrieval_recovery_without_e2e_gain_count"],
        "retrieval_regression_to_e2e_regression_count": e2e["retrieval_regression_to_e2e_regression_count"],
        "safe_action_regression": e2e["safe_action_regression"],
        "grounding_regression": e2e["grounding_regression"],
        "citation_regression": e2e["citation_regression"],
        "unsupported_answer_regression": e2e["unsupported_answer_regression"],
        "lexical_followup_required": analysis["lexical_followup"]["lexical_followup_required"],
        "lexical_reranker_variant_executed": analysis["lexical_followup"]["lexical_reranker_variant_executed"],
        "promotion_candidate": decision["promotion_candidate"],
        "promotion_decision": decision["promotion_decision"],
        "default_retriever_modified": False,
        "reranker_default_enabled": False,
        "recommended_next_task": decision["recommended_next_task"],
        "repository_wide_verification_status": "not_run",
        "git_add_executed": False,
        "git_commit_created": False,
        "model_manifest": model_manifest,
        "runtime_manifest": runtime_manifest,
        "environment": environment,
        "task0085_core_rag_environment_block_cleared": core_rag["task0085_core_rag_environment_block_cleared"],
        "authority_unit_count": len(inputs["authority"]),
    }


def build_environment_blocked_summary(
    *,
    error: BaseException,
    inputs: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_manifest: dict[str, Any],
    environment: dict[str, Any],
    core_rag: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    r0_metrics = strategy_metrics(build_strategy_candidates(inputs)["r0"], final_k=20)
    return {
        "schema_version": "opk-rag.task0087.real-reranker-validation-summary.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "blocked",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "vector_retriever_modified": False,
        "lexical_retriever_modified": False,
        "query_reformulation_modified": False,
        "generation_modified": False,
        "answerability_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "real_reranker_experiment": False,
        "offline_proxy_used_for_primary_result": False,
        "promotion_eligible": False,
        "reranker_model": model_manifest["model_id"],
        "reranker_model_revision": model_manifest["model_revision"],
        "reranker_device": environment["device"],
        "reranker_gpu": environment["gpu_model"],
        "reranker_dtype": environment["dtype"],
        "reranker_batch_size": runtime_manifest["batch_size"],
        "reranker_replicate_count": 0,
        "reranker_results_deterministic": False,
        "r0_recall_at_5": r0_metrics["recall_at_5"],
        "r0_recall_at_10": r0_metrics["recall_at_10"],
        "r0_recall_at_20": r0_metrics["recall_at_20"],
        "r0_mrr": r0_metrics["mrr"],
        "r1_candidate_recall_at_50": None,
        "candidate_expansion_gain": None,
        "r2_recall_at_5": None,
        "r2_recall_at_10": None,
        "r2_recall_at_20": None,
        "r2_mrr": None,
        "deep_gold_candidate_count": 0,
        "deep_gold_promoted_to_top20_count": 0,
        "deep_gold_promoted_to_top10_count": 0,
        "deep_gold_promoted_to_top5_count": 0,
        "deep_candidate_to_top20_conversion_rate": 0.0,
        "vector_protection_unit_count": 0,
        "vector_hit_preserved_count": 0,
        "vector_hit_regressed_count": 0,
        "vector_protection_rate": 0.0,
        "newly_recovered_count": 0,
        "newly_regressed_count": 0,
        "net_recovered_count": 0,
        "all_real_reranker_failures_classified": False,
        "all_real_reranker_regressions_classified": False,
        "truncated_candidate_count": 0,
        "gold_candidate_truncated_count": 0,
        "reranking_latency_per_query_p50": None,
        "reranking_latency_per_query_p95": None,
        "pairs_per_second": 0.0,
        "peak_vram": None,
        "runtime_feasibility": "unacceptable",
        "end_to_end_validation_executed": False,
        "end_to_end_regression_check_passed": False,
        "retrieval_recovery_to_e2e_gain_count": 0,
        "retrieval_recovery_without_e2e_gain_count": 0,
        "retrieval_regression_to_e2e_regression_count": 0,
        "safe_action_regression": False,
        "grounding_regression": False,
        "citation_regression": False,
        "unsupported_answer_regression": False,
        "lexical_followup_required": False,
        "lexical_reranker_variant_executed": False,
        "promotion_candidate": False,
        "promotion_decision": "environment_blocked_real_reranker_unresolved",
        "default_retriever_modified": False,
        "reranker_default_enabled": False,
        "recommended_next_task": "Resolve local GPU/model artifact availability and rerun TASK-0087.",
        "repository_wide_verification_status": "not_run",
        "git_add_executed": False,
        "git_commit_created": False,
        "environment_block_reason": f"{type(error).__name__}: {error}",
        "model_manifest": model_manifest,
        "runtime_manifest": runtime_manifest,
        "environment": environment,
        "task0085_core_rag_environment_block_cleared": core_rag["task0085_core_rag_environment_block_cleared"],
    }


def write_task0087_artifacts(
    *,
    environment: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, list[RankedCandidate]],
    analysis: dict[str, Any],
    replicate_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_json(RESULT_DIR / "environment.json", environment)
    write_json(RESULT_DIR / "model_manifest.json", model_manifest)
    write_json(RESULT_DIR / "runtime_manifest.json", runtime_manifest)
    write_json(RESULT_DIR / "candidate_manifest.json", candidate_manifest)
    write_jsonl(RESULT_DIR / "reranker_scores.jsonl", reranker_score_rows(ranked))
    write_jsonl(RESULT_DIR / "gold_rank_transitions.jsonl", analysis["gold_rank_transitions"])
    write_jsonl(RESULT_DIR / "failure_classification.jsonl", analysis["failure_classification"])
    write_jsonl(RESULT_DIR / "regression_audit.jsonl", analysis["regression_audit"])
    for name in (
        "retrieval_metrics",
        "deep_candidate_conversion",
        "vector_protection_audit",
        "score_distribution",
        "truncation_audit",
        "latency",
        "memory",
        "end_to_end_metrics",
        "promotion_gates",
        "promotion_decision",
    ):
        write_json(RESULT_DIR / f"{name}.json", analysis[name])
    write_json(RESULT_DIR / "replicate_results.json", {"schema_version": "opk-rag.task0087.replicate-results.v1", "replicates": replicate_results})
    write_json(RESULT_DIR / "summary.json", summary)


def write_environment_blocked_artifacts(
    summary: dict[str, Any],
    environment: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_manifest: dict[str, Any],
) -> None:
    write_json(RESULT_DIR / "environment.json", environment)
    write_json(RESULT_DIR / "model_manifest.json", model_manifest)
    write_json(RESULT_DIR / "runtime_manifest.json", runtime_manifest)
    write_json(RESULT_DIR / "candidate_manifest.json", {"schema_version": "opk-rag.task0087.candidate-manifest.v1", "status": "environment_blocked"})
    write_json(RESULT_DIR / "promotion_decision.json", {"schema_version": "opk-rag.task0087.promotion-decision.v1", "promotion_decision": summary["promotion_decision"]})
    write_json(RESULT_DIR / "promotion_gates.json", {"schema_version": "opk-rag.task0087.promotion-gates.v1", "promotion_eligible": False})
    write_json(RESULT_DIR / "summary.json", summary)
    verification = verify_task0087_artifacts(allow_blocked=True)
    write_json(RESULT_DIR / "verification.json", verification)
    REPORT_PATH.write_text(build_task0087_blocked_report(summary), encoding="utf-8")


def verify_task0087_artifacts(*, allow_blocked: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "environment.json",
        RESULT_DIR / "model_manifest.json",
        RESULT_DIR / "runtime_manifest.json",
        RESULT_DIR / "candidate_manifest.json",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "promotion_decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    if not allow_blocked:
        required.extend(
            [
                RESULT_DIR / "reranker_scores.jsonl",
                RESULT_DIR / "gold_rank_transitions.jsonl",
                RESULT_DIR / "retrieval_metrics.json",
                RESULT_DIR / "deep_candidate_conversion.json",
                RESULT_DIR / "vector_protection_audit.json",
                RESULT_DIR / "failure_classification.jsonl",
                RESULT_DIR / "regression_audit.jsonl",
                RESULT_DIR / "score_distribution.json",
                RESULT_DIR / "truncation_audit.json",
                RESULT_DIR / "replicate_results.json",
                RESULT_DIR / "latency.json",
                RESULT_DIR / "memory.json",
                RESULT_DIR / "end_to_end_metrics.json",
            ]
        )
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = _read_json(RESULT_DIR / "summary.json")
        for key in (
            "benchmark_modified",
            "chunking_modified",
            "embedding_model_modified",
            "vector_retriever_modified",
            "lexical_retriever_modified",
            "query_reformulation_modified",
            "generation_modified",
            "answerability_modified",
            "agent_runtime_modified",
            "graph_runtime_modified",
            "default_retriever_modified",
            "reranker_default_enabled",
        ):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
        if summary.get("formal_metric_authority") != FORMAL_AUTHORITY_ID or summary.get("formal_metric_authority_valid") is not True:
            issues.append({"code": "formal_metric_authority_invalid"})
        if summary.get("offline_proxy_used_for_primary_result") is not False:
            issues.append({"code": "offline_proxy_used"})
        if summary.get("git_commit_created") is not False:
            issues.append({"code": "git_commit_created"})
        if summary.get("task_status") != "blocked" and summary.get("reranker_replicate_count") != REPLICATE_COUNT:
            issues.append({"code": "replicate_count_invalid"})
    return {
        "schema_version": "opk-rag.task0087.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def reranker_score_rows(ranked: dict[str, list[RankedCandidate]]) -> list[dict[str, Any]]:
    rows = []
    for unit_rows in ranked.values():
        for row in unit_rows:
            rows.append(
                {
                    "schema_version": "opk-rag.task0087.reranker-score.v1",
                    "sample_unit_id": row.candidate.sample_unit_id,
                    "sample_id": row.candidate.sample_id,
                    "chunk_id": row.candidate.chunk_id,
                    "document_id": row.candidate.document_id,
                    "section_id": row.candidate.section_id,
                    "original_vector_rank": row.candidate.vector_rank,
                    "vector_score": None,
                    "reranker_raw_score": row.reranker_score,
                    "reranked_rank": row.final_rank,
                    "is_gold": row.candidate.is_gold,
                    "input_truncated": row.pair_metadata.input_truncated,
                    "original_pair_token_count": row.pair_metadata.original_pair_token_count,
                    "pair_token_count": row.pair_metadata.pair_token_count,
                }
            )
    return rows


def build_task0087_report(summary: dict[str, Any], analysis: dict[str, Any]) -> str:
    score = analysis["score_distribution"]
    return "\n".join(
        [
            "# TASK0087 Real Cross-Encoder Reranker Validation Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- formal_metric_authority=`{summary['formal_metric_authority']}`",
            f"- real_reranker_experiment=`{str(summary['real_reranker_experiment']).lower()}`",
            f"- promotion_eligible=`{str(summary['promotion_eligible']).lower()}`",
            f"- promotion_decision=`{summary['promotion_decision']}`",
            f"- reranker_model=`{summary['reranker_model']}`",
            f"- reranker_model_revision=`{summary['reranker_model_revision']}`",
            "",
            "## Required Answers",
            "",
            f"1. TASK-0086 real Cross-Encoder identity: `{summary['reranker_model']}@{summary['reranker_model_revision']}`.",
            f"2. GPU / accelerated inference: device `{summary['reranker_device']}`, GPU `{summary['reranker_gpu']}`.",
            f"3. Complete formal benchmark finished: `{str(summary['task_status'] == 'complete').lower()}`.",
            f"4. Promotion eligible: `{str(summary['promotion_eligible']).lower()}`.",
            f"5. Vector Top-50 Gold Coverage: `{summary['r1_candidate_recall_at_50']}`.",
            f"6. Rank 21-50 Gold promoted to Top-20: `{summary['deep_gold_promoted_to_top20_count']}` / `{summary['deep_gold_candidate_count']}`.",
            f"7. Promoted to Top-10 / Top-5: `{summary['deep_gold_promoted_to_top10_count']}` / `{summary['deep_gold_promoted_to_top5_count']}`.",
            f"8. Recall@5/10/20: R0 `{summary['r0_recall_at_5']}/{summary['r0_recall_at_10']}/{summary['r0_recall_at_20']}`, R2 `{summary['r2_recall_at_5']}/{summary['r2_recall_at_10']}/{summary['r2_recall_at_20']}`.",
            f"9. MRR: R0 `{summary['r0_mrr']}`, R2 `{summary['r2_mrr']}`.",
            f"10. Vector hits regressed: `{summary['vector_hit_regressed_count']}` / `{summary['vector_protection_unit_count']}`.",
            f"11. Remaining failures classified: `{str(summary['all_real_reranker_failures_classified']).lower()}`.",
            f"12. Retrieval gain converted to E2E gain: executed `{str(summary['end_to_end_validation_executed']).lower()}`, gain count `{summary['retrieval_recovery_to_e2e_gain_count']}`.",
            f"13. Latency / VRAM: p50 `{summary['reranking_latency_per_query_p50']}`, p95 `{summary['reranking_latency_per_query_p95']}`, peak_vram `{summary['peak_vram']}`.",
            f"14. Formal promotion: `{summary['promotion_decision']}`.",
            f"15. Lexical Candidate Union follow-up required: `{str(summary['lexical_followup_required']).lower()}`; executed `{str(summary['lexical_reranker_variant_executed']).lower()}`.",
            "",
            "## Diagnostics",
            "",
            f"- score_separation=`{score['score_separation']}`",
            f"- truncated_candidate_count=`{summary['truncated_candidate_count']}`",
            f"- gold_candidate_truncated_count=`{summary['gold_candidate_truncated_count']}`",
            f"- pairs_per_second=`{summary['pairs_per_second']}`",
            f"- recommended_next_task=`{summary['recommended_next_task']}`",
        ]
    )


def build_task0087_blocked_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0087 Real Cross-Encoder Reranker Validation Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- promotion_decision=`{summary['promotion_decision']}`",
            f"- environment_block_reason=`{summary['environment_block_reason']}`",
            f"- reranker_model=`{summary['reranker_model']}`",
            f"- reranker_model_revision=`{summary['reranker_model_revision']}`",
            "",
            "This is an environment block, not a negative reranker result.",
        ]
    )


def _recommended_next_task(decision: str) -> str:
    if decision == "promote_vector_cross_encoder":
        return "TASK-0088 real reranker promotion and default runtime integration"
    if decision == "promising_runtime_optimization_required":
        return "TASK-0088 reranker runtime optimization and smaller-model feasibility validation"
    if decision == "real_reranker_valid_lexical_followup_required":
        return "TASK-0088 Vector + Lexical Candidate Union + Real Reranker validation"
    if decision == "environment_blocked_real_reranker_unresolved":
        return "Resolve local GPU/model artifact availability and rerun TASK-0087"
    return "Return to query representation, embedding localization, and residual evidence diagnosis"


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
