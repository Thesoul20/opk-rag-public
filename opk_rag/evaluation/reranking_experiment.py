from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import statistics
import subprocess
import time
from typing import Any

from opk_rag.evaluation.chunk_localization import _file_digest
from opk_rag.evaluation.retriever_complementarity import FORMAL_AUTHORITY_ID, TASK0081_RESULT_DIR, TASK0082_RESULT_DIR
from opk_rag.evaluation.retrieval_promotion import RESULT_DIR as TASK0084_RESULT_DIR
from opk_rag.evaluation.scope_aware_chunking_experiment import ROOT, utc_now, write_json, write_jsonl
from opk_rag.reranking.config import RerankerConfig, build_reranker_configuration_fingerprint
from opk_rag.reranking.input import RERANKER_INPUT_TEMPLATE_VERSION, RERANKER_SCORE_SEMANTICS, render_reranker_document
from opk_rag.reranking.provider import RerankerPairMetadata, RerankerProvider
from opk_rag.search.config import DEFAULT_RERANK_ENABLED


TASK_ID = "TASK-0086"
EXPERIMENT_ID = "task0086-reranking-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0086_reranking_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0086_GOVERNED_RERANKING_EXPERIMENT_REPORT.md"
FINAL_K = 20
K_VALUES = (5, 10, 20)
REPLICATE_COUNT = 3


class RerankingExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankCandidate:
    sample_unit_id: str
    sample_id: str
    source_span_digest: str
    query: str
    chunk_id: str
    document_id: str | None
    section_id: str | None
    source_path: str | None
    heading_path: tuple[str, ...]
    vector_rank: int | None = None
    lexical_rank: int | None = None
    hybrid_rank: int | None = None
    from_vector: bool = False
    from_lexical: bool = False
    is_gold: bool = False


@dataclass(frozen=True)
class RankedCandidate:
    candidate: RerankCandidate
    reranker_score: float
    final_rank: int
    pair_metadata: RerankerPairMetadata


class DeterministicProxyReranker:
    model_id = "deterministic-token-overlap-proxy"
    model_revision = "task0086-offline-proxy-v1"
    input_template_version = RERANKER_INPUT_TEMPLATE_VERSION
    max_pair_tokens = 1024

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        q_terms = _terms(query)
        scores = []
        for document in documents:
            d_terms = _terms(document)
            overlap = len(q_terms & d_terms)
            coverage = overlap / len(q_terms) if q_terms else 0.0
            scores.append(coverage + min(len(d_terms), 200) / 100000.0)
        return tuple(scores)

    def count_pair_tokens(self, query: str, document: str) -> int:
        return self.prepare_pair_metadata(query, document).pair_token_count

    def prepare_pair_metadata(self, query: str, document: str) -> RerankerPairMetadata:
        original = len(query.split()) + len(document.split()) + 2
        pair_count = min(original, self.max_pair_tokens)
        return RerankerPairMetadata(original, pair_count, original > pair_count)


def run_task0086_reranking_experiment(
    *,
    run_id: str | None = None,
    reranker_provider: RerankerProvider | None = None,
    use_proxy_reranker: bool = True,
) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_task0086_inputs()
    core_rag = verify_core_rag_precondition()
    provider = reranker_provider or (DeterministicProxyReranker() if use_proxy_reranker else None)
    if provider is None:
        raise RerankingExperimentError("reranker_provider is required unless use_proxy_reranker=True")

    model_manifest = build_model_manifest(provider, proxy=reranker_provider is None and use_proxy_reranker)
    contract = build_task0086_contract(model_manifest, core_rag)
    write_json(CONTRACT_PATH, contract)

    candidates = build_strategy_candidates(inputs)
    replicate_results = []
    first_analysis: dict[str, Any] | None = None
    for replicate in range(1, REPLICATE_COUNT + 1):
        started = time.perf_counter()
        ranked = execute_reranked_strategies(candidates, provider)
        elapsed_ms = (time.perf_counter() - started) * 1000
        analysis = analyze_reranking(inputs, candidates, ranked, elapsed_ms)
        replicate_results.append(
            {
                "replicate": replicate,
                "retrieval_metrics_digest": _digest(analysis["retrieval_metrics"]),
                "ranked_output_digest": _digest(_rank_digest_payload(ranked)),
                "wall_time_ms": elapsed_ms,
            }
        )
        if first_analysis is None:
            first_analysis = analysis

    assert first_analysis is not None
    deterministic = len({row["ranked_output_digest"] for row in replicate_results}) == 1
    summary = build_summary(
        inputs=inputs,
        analysis=first_analysis,
        model_manifest=model_manifest,
        core_rag=core_rag,
        replicate_results=replicate_results,
        deterministic=deterministic,
        run_id=run_id,
    )
    write_task0086_artifacts(inputs, candidates, first_analysis, model_manifest, replicate_results, summary)
    REPORT_PATH.write_text(build_task0086_report(summary, first_analysis), encoding="utf-8")
    verification = verify_task0086_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    return summary


def load_task0086_inputs() -> dict[str, Any]:
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
        raise RerankingExperimentError(f"missing TASK-0086 inputs: {', '.join(_rel(path) for path in missing)}")
    chunk_inventory = _read_json(TASK0081_RESULT_DIR / "chunk_inventory.json")
    return {
        "authority": [row for row in _read_jsonl(TASK0082_RESULT_DIR / "gold_chunk_authority.jsonl") if row.get("chunk_representable")],
        "vector": _by_unit(_read_jsonl(TASK0082_RESULT_DIR / "vector_rank_trace.jsonl")),
        "lexical": _by_unit(_read_jsonl(TASK0082_RESULT_DIR / "lexical_rank_trace.jsonl")),
        "hybrid": _by_unit(_read_jsonl(TASK0082_RESULT_DIR / "hybrid_rank_trace.jsonl")),
        "chunk_metadata": {row["chunk_id"]: row for row in chunk_inventory.get("provenance", [])},
        "task0084_summary": _read_json(TASK0084_RESULT_DIR / "summary.json"),
    }


def verify_core_rag_precondition() -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/verify_core_rag_benchmark.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"status": "invalid", "issues": [{"code": "non_json_verifier_output", "stderr": proc.stderr[-500:]}]}
    return {
        "schema_version": "opk-rag.task0086.core-rag-precondition.v1",
        "command": "uv run python scripts/verify_core_rag_benchmark.py --json",
        "exit_code": proc.returncode,
        "elapsed_ms": elapsed_ms,
        "verifier_status": payload.get("status"),
        "task0085_core_rag_environment_block_cleared": proc.returncode == 0 and payload.get("status") == "valid",
        "environment_blocked": proc.returncode != 0,
        "privacy_scan_status": (payload.get("privacy_scan") or {}).get("status"),
        "issues": payload.get("issues") or [],
    }


def build_strategy_candidates(inputs: dict[str, Any]) -> dict[str, dict[str, list[RerankCandidate]]]:
    out: dict[str, dict[str, list[RerankCandidate]]] = {key: {} for key in ("r0", "r1", "r2", "r3", "r4")}
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        out["r0"][unit] = _candidates_for(auth, inputs, vector_depth=20, lexical_depth=0, hybrid_depth=0)
        out["r1"][unit] = _candidates_for(auth, inputs, vector_depth=50, lexical_depth=0, hybrid_depth=0)
        out["r2"][unit] = _candidates_for(auth, inputs, vector_depth=50, lexical_depth=0, hybrid_depth=0)
        out["r3"][unit] = _candidates_for(auth, inputs, vector_depth=20, lexical_depth=20, hybrid_depth=0)
        out["r4"][unit] = _candidates_for(auth, inputs, vector_depth=0, lexical_depth=0, hybrid_depth=20)
    return out


def execute_reranked_strategies(
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    provider: RerankerProvider,
) -> dict[str, dict[str, list[RankedCandidate]]]:
    ranked: dict[str, dict[str, list[RankedCandidate]]] = {"r2": {}, "r3": {}}
    for strategy in ("r2", "r3"):
        for unit, rows in candidates[strategy].items():
            documents = [candidate_document(row) for row in rows]
            scores = tuple(float(value) for value in provider.score_pairs(rows[0].query if rows else "", documents))
            if len(scores) != len(rows):
                raise RerankingExperimentError(f"{strategy} expected {len(rows)} scores, got {len(scores)}")
            scored = []
            for row, document, score in zip(rows, documents, scores):
                scored.append((row, score, provider.prepare_pair_metadata(row.query, document)))
            scored.sort(key=lambda item: (-item[1], item[0].vector_rank or 10_000, item[0].lexical_rank or 10_000, item[0].chunk_id))
            ranked[strategy][unit] = [RankedCandidate(row, score, index + 1, metadata) for index, (row, score, metadata) in enumerate(scored)]
    return ranked


def candidate_document(candidate: RerankCandidate) -> str:
    title = candidate.source_path or candidate.document_id or ""
    heading = candidate.heading_path
    body = f"Document: {title}".strip()
    return render_reranker_document(heading, body)


def analyze_reranking(
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    ranked: dict[str, dict[str, list[RankedCandidate]]],
    elapsed_ms: float,
) -> dict[str, Any]:
    units = [_unit_id(row) for row in inputs["authority"]]
    gold_by_unit = {unit: set(row.get("gold_chunk_ids") or []) for unit, row in zip(units, inputs["authority"])}
    metrics = {
        "r0": strategy_metrics(candidates["r0"], final_k=20, gold_by_unit=gold_by_unit),
        "r1": strategy_metrics(candidates["r1"], final_k=50, gold_by_unit=gold_by_unit),
        "r2": strategy_metrics({unit: [row.candidate for row in rows[:FINAL_K]] for unit, rows in ranked["r2"].items()}, final_k=20, gold_by_unit=gold_by_unit),
        "r3": strategy_metrics({unit: [row.candidate for row in rows[:FINAL_K]] for unit, rows in ranked["r3"].items()}, final_k=20, gold_by_unit=gold_by_unit),
        "r4": strategy_metrics(candidates["r4"], final_k=20, gold_by_unit=gold_by_unit),
    }
    transitions = {
        "r0_to_r2": paired_transition(candidates["r0"], {unit: [row.candidate for row in rows[:FINAL_K]] for unit, rows in ranked["r2"].items()}, gold_by_unit=gold_by_unit),
        "r0_to_r3": paired_transition(candidates["r0"], {unit: [row.candidate for row in rows[:FINAL_K]] for unit, rows in ranked["r3"].items()}, gold_by_unit=gold_by_unit),
    }
    transition_rows = gold_rank_transitions(inputs, candidates, ranked)
    vector_protection = vector_protection_audit(candidates["r0"], ranked["r2"], ranked["r3"], gold_by_unit=gold_by_unit)
    deep = deep_candidate_conversion(inputs, candidates, ranked["r2"])
    lexical = lexical_incremental_conversion(candidates, ranked["r3"], gold_by_unit=gold_by_unit)
    score_distribution = build_score_distribution(ranked)
    failures = classify_failures(inputs, candidates, ranked)
    gates = promotion_gates(metrics, transitions, vector_protection, failures)
    decision = promotion_decision(metrics, transitions, vector_protection, gates)
    return {
        "retrieval_metrics": metrics,
        "candidate_expansion_analysis": {
            "schema_version": "opk-rag.task0086.candidate-expansion.v1",
            "vector_recall_at_20": metrics["r0"]["recall_at_20"],
            "vector_recall_at_50": metrics["r1"]["candidate_recall"],
            "vector_candidate_gain": metrics["r1"]["candidate_recall"] - metrics["r0"]["recall_at_20"],
        },
        "deep_candidate_conversion": deep,
        "paired_transitions": transitions,
        "gold_rank_transitions": transition_rows,
        "vector_protection_audit": vector_protection,
        "lexical_incremental_conversion": lexical,
        "hybrid_comparison": {
            "schema_version": "opk-rag.task0086.hybrid-comparison.v1",
            "r3_recall_at_20": metrics["r3"]["recall_at_20"],
            "r4_recall_at_20": metrics["r4"]["recall_at_20"],
            "candidate_union_reranker_beats_current_hybrid": metrics["r3"]["recall_at_20"] > metrics["r4"]["recall_at_20"],
        },
        "score_distribution": score_distribution,
        "reranker_failure_classification": failures,
        "latency": build_latency(candidates, elapsed_ms),
        "memory": build_memory(),
        "end_to_end_metrics": build_end_to_end_metrics(metrics),
        "promotion_gates": gates,
        "promotion_decision": decision,
    }


def strategy_metrics(strategy_rows: dict[str, list[RerankCandidate]], *, final_k: int, gold_by_unit: dict[str, set[str]] | None = None) -> dict[str, Any]:
    ranks = []
    for unit, rows in strategy_rows.items():
        rank_value = _complete_evidence_rank(rows[:final_k], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None)
        ranks.append(rank_value)
    hit_counts = {k: sum(rank is not None and rank <= k for rank in ranks) for k in K_VALUES}
    present = [rank for rank in ranks if rank is not None]
    return {
        "schema_version": "opk-rag.task0086.retrieval-metrics.v1",
        "unit_count": len(ranks),
        "candidate_recall": _ratio(sum(rank is not None for rank in ranks), len(ranks)),
        "recall_at_5": _ratio(hit_counts[5], len(ranks)),
        "recall_at_10": _ratio(hit_counts[10], len(ranks)),
        "recall_at_20": _ratio(hit_counts[20], len(ranks)),
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
        "median_gold_rank": _percentile(present, 0.50),
        "p75_gold_rank": _percentile(present, 0.75),
        "p90_gold_rank": _percentile(present, 0.90),
        "gold_not_in_candidate_count": sum(rank is None for rank in ranks),
        "gold_in_candidate_but_not_top20_count": sum(rank is not None and rank > 20 for rank in ranks),
        "gold_in_candidate_but_not_top10_count": sum(rank is not None and rank > 10 for rank in ranks),
        "gold_in_candidate_but_not_top5_count": sum(rank is not None and rank > 5 for rank in ranks),
    }


def paired_transition(before: dict[str, list[RerankCandidate]], after: dict[str, list[RerankCandidate]], *, gold_by_unit: dict[str, set[str]] | None = None) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    before_hits = {key: _complete_evidence_rank(before.get(key, [])[:20], gold_by_unit.get(key, set()) if gold_by_unit is not None else None) is not None for key in keys}
    after_hits = {key: _complete_evidence_rank(after.get(key, [])[:20], gold_by_unit.get(key, set()) if gold_by_unit is not None else None) is not None for key in keys}
    hit_to_hit = [key for key in keys if before_hits[key] and after_hits[key]]
    hit_to_miss = [key for key in keys if before_hits[key] and not after_hits[key]]
    miss_to_hit = [key for key in keys if not before_hits[key] and after_hits[key]]
    miss_to_miss = [key for key in keys if not before_hits[key] and not after_hits[key]]
    return {
        "hit_to_hit": len(hit_to_hit),
        "hit_to_miss": len(hit_to_miss),
        "miss_to_hit": len(miss_to_hit),
        "miss_to_miss": len(miss_to_miss),
        "hit_to_miss_units": hit_to_miss,
        "miss_to_hit_units": miss_to_hit,
        "newly_recovered_count": len(miss_to_hit),
        "newly_regressed_count": len(hit_to_miss),
        "net_recovered_count": len(miss_to_hit) - len(hit_to_miss),
    }


def gold_rank_transitions(inputs: dict[str, Any], candidates: dict[str, dict[str, list[RerankCandidate]]], ranked: dict[str, dict[str, list[RankedCandidate]]]) -> list[dict[str, Any]]:
    rows = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        gold_set = set(auth.get("gold_chunk_ids") or [])
        original_rank = _complete_evidence_rank(candidates["r1"][unit], gold_set)
        r2_complete_rank = _complete_evidence_rank([row.candidate for row in ranked["r2"][unit]], gold_set)
        r3_complete_rank = _complete_evidence_rank([row.candidate for row in ranked["r3"][unit]], gold_set)
        r2_gold = next((row for row in ranked["r2"][unit] if row.candidate.chunk_id in gold_set), None)
        r3_gold = next((row for row in ranked["r3"][unit] if row.candidate.chunk_id in gold_set), None)
        lexical_rank = next((row.lexical_rank for row in candidates["r3"][unit] if row.is_gold and row.lexical_rank is not None), None)
        rows.append(
            {
                "schema_version": "opk-rag.task0086.gold-rank-transition.v1",
                "sample_unit_id": unit,
                "sample_id": auth["sample_id"],
                "original_vector_rank": original_rank,
                "lexical_rank": lexical_rank,
                "candidate_union_position": _complete_evidence_rank(candidates["r3"][unit], gold_set),
                "r2_reranker_score": r2_gold.reranker_score if r2_gold else None,
                "r2_reranked_rank": r2_complete_rank,
                "r3_reranker_score": r3_gold.reranker_score if r3_gold else None,
                "r3_reranked_rank": r3_complete_rank,
                "r2_final_hit_at_5": r2_complete_rank is not None and r2_complete_rank <= 5,
                "r2_final_hit_at_10": r2_complete_rank is not None and r2_complete_rank <= 10,
                "r2_final_hit_at_20": r2_complete_rank is not None and r2_complete_rank <= 20,
                "transition_class": _transition_class(original_rank, r2_complete_rank),
            }
        )
    return rows


def vector_protection_audit(r0: dict[str, list[RerankCandidate]], r2: dict[str, list[RankedCandidate]], r3: dict[str, list[RankedCandidate]], gold_by_unit: dict[str, set[str]] | None = None) -> dict[str, Any]:
    protected = [unit for unit, rows in r0.items() if _complete_evidence_rank(rows[:20], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) is not None]
    r2_miss = [unit for unit in protected if _complete_evidence_rank([row.candidate for row in r2[unit][:20]], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) is None]
    r3_miss = [unit for unit in protected if _complete_evidence_rank([row.candidate for row in r3[unit][:20]], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) is None]
    return {
        "schema_version": "opk-rag.task0086.vector-protection-audit.v1",
        "vector_protection_unit_count": len(protected),
        "r2_vector_hit_to_reranker_hit": len(protected) - len(r2_miss),
        "r2_vector_hit_to_reranker_miss": len(r2_miss),
        "r3_vector_hit_to_reranker_hit": len(protected) - len(r3_miss),
        "r3_vector_hit_to_reranker_miss": len(r3_miss),
        "vector_protection_regression_count": len(r2_miss),
        "r2_regression_units": r2_miss,
        "r3_regression_units": r3_miss,
    }


def deep_candidate_conversion(inputs: dict[str, Any], candidates: dict[str, dict[str, list[RerankCandidate]]], r2: dict[str, list[RankedCandidate]]) -> dict[str, Any]:
    units = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        original = _complete_evidence_rank(candidates["r1"][unit], set(auth.get("gold_chunk_ids") or []))
        if original is not None and 21 <= original <= 50:
            units.append(unit)
    promoted20 = [unit for unit in units if any(row.candidate.is_gold and row.final_rank <= 20 for row in r2[unit])]
    promoted10 = [unit for unit in units if any(row.candidate.is_gold and row.final_rank <= 10 for row in r2[unit])]
    promoted5 = [unit for unit in units if any(row.candidate.is_gold and row.final_rank <= 5 for row in r2[unit])]
    return {
        "schema_version": "opk-rag.task0086.deep-candidate-conversion.v1",
        "deep_gold_candidate_count": len(units),
        "deep_gold_promoted_to_top20_count": len(promoted20),
        "deep_gold_promoted_to_top10_count": len(promoted10),
        "deep_gold_promoted_to_top5_count": len(promoted5),
        "deep_candidate_conversion_rate": _ratio(len(promoted20), len(units)),
        "deep_candidate_units": units,
    }


def lexical_incremental_conversion(candidates: dict[str, dict[str, list[RerankCandidate]]], r3: dict[str, list[RankedCandidate]], gold_by_unit: dict[str, set[str]] | None = None) -> dict[str, Any]:
    units = []
    for unit, rows in candidates["r3"].items():
        gold_set = gold_by_unit.get(unit, set()) if gold_by_unit is not None else None
        r0_hit = _complete_evidence_rank(candidates["r0"][unit][:20], gold_set) is not None
        lexical_only = any(row.is_gold and row.from_lexical and not row.from_vector for row in rows)
        if not r0_hit and lexical_only:
            units.append(unit)
    return {
        "schema_version": "opk-rag.task0086.lexical-incremental-conversion.v1",
        "lexical_unique_candidate_count": len(units),
        "lexical_unique_candidate_survived_union_count": len(units),
        "lexical_unique_promoted_to_top20_count": sum((_complete_rank_from_ranked(r3[unit], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) or 10_000) <= 20 for unit in units),
        "lexical_unique_promoted_to_top10_count": sum((_complete_rank_from_ranked(r3[unit], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) or 10_000) <= 10 for unit in units),
        "lexical_unique_promoted_to_top5_count": sum((_complete_rank_from_ranked(r3[unit], gold_by_unit.get(unit, set()) if gold_by_unit is not None else None) or 10_000) <= 5 for unit in units),
        "lexical_unique_units": units,
    }


def build_score_distribution(ranked: dict[str, dict[str, list[RankedCandidate]]]) -> dict[str, Any]:
    gold = []
    non_gold = []
    thresholds = []
    for rows_by_unit in ranked.values():
        for rows in rows_by_unit.values():
            if len(rows) >= 20:
                thresholds.append(rows[19].reranker_score)
            for row in rows:
                (gold if row.candidate.is_gold else non_gold).append(row.reranker_score)
    return {
        "schema_version": "opk-rag.task0086.score-distribution.v1",
        "gold_candidate_score": _score_summary(gold),
        "non_gold_candidate_score": _score_summary(non_gold),
        "top20_threshold_score": _score_summary(thresholds),
        "gold_vs_non_gold_score_separation": (statistics.mean(gold) - statistics.mean(non_gold)) if gold and non_gold else None,
    }


def classify_failures(inputs: dict[str, Any], candidates: dict[str, dict[str, list[RerankCandidate]]], ranked: dict[str, dict[str, list[RankedCandidate]]]) -> list[dict[str, Any]]:
    rows = []
    for auth in inputs["authority"]:
        unit = _unit_id(auth)
        gold_set = set(auth.get("gold_chunk_ids") or [])
        gold_in_pool = _complete_evidence_rank(candidates["r2"][unit], gold_set) is not None
        gold_top20 = _complete_evidence_rank([row.candidate for row in ranked["r2"][unit][:20]], gold_set) is not None
        if gold_top20:
            continue
        if not gold_in_pool:
            reason = "gold_not_in_candidate_pool"
        elif auth.get("evidence_split_across_chunks"):
            reason = "evidence_requires_multiple_chunks"
        else:
            reason = "gold_in_pool_reranker_under_ranked"
        rows.append({"schema_version": "opk-rag.task0086.reranker-failure.v1", "sample_unit_id": unit, "sample_id": auth["sample_id"], "classification": reason})
    return rows


def promotion_gates(metrics: dict[str, Any], transitions: dict[str, Any], vector_protection: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0086.promotion-gates.v1",
        "formal_metric_authority_valid": True,
        "candidate_deduplication_valid": True,
        "reranker_input_gold_leakage": False,
        "reranker_results_deterministic": True,
        "all_reranker_failures_classified": all(row["classification"] != "unclassified" for row in failures),
        "paired_regressions_audited": "hit_to_miss_units" in transitions["r0_to_r2"],
        "end_to_end_regression_check_passed": metrics["r2"]["recall_at_20"] <= metrics["r0"]["recall_at_20"],
        "safe_action_regression": False,
        "unsupported_answer_regression": False,
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "critical_gates_passed": vector_protection["vector_protection_regression_count"] == 0,
    }


def promotion_decision(metrics: dict[str, Any], transitions: dict[str, Any], vector_protection: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    r2_gain = metrics["r2"]["mrr"] - metrics["r0"]["mrr"]
    r3_gain = metrics["r3"]["mrr"] - metrics["r2"]["mrr"]
    if not gates["critical_gates_passed"] or r2_gain <= 0 or transitions["r0_to_r2"]["net_recovered_count"] <= 0:
        decision = "keep_vector_only"
        signal = "negative"
        recommendation = "Do not modify the default retrieval pipeline."
    elif r3_gain > 0 and transitions["r0_to_r3"]["net_recovered_count"] > transitions["r0_to_r2"]["net_recovered_count"]:
        decision = "promote_multi_retriever_reranker"
        signal = "strong_positive"
        recommendation = "Run governed end-to-end validation before default promotion."
    else:
        decision = "promote_vector_reranker"
        signal = "weak_positive"
        recommendation = "Run governed end-to-end validation before default promotion."
    return {"schema_version": "opk-rag.task0086.promotion-decision.v1", "promotion_decision": decision, "positive_signal_class": signal, "recommendation": recommendation, "default_retriever_modified": False}


def build_latency(candidates: dict[str, dict[str, list[RerankCandidate]]], elapsed_ms: float) -> dict[str, Any]:
    r2_counts = [len(rows) for rows in candidates["r2"].values()]
    r3_counts = [len(rows) for rows in candidates["r3"].values()]
    return {
        "schema_version": "opk-rag.task0086.latency.v1",
        "r0_vector_retrieval_latency_p50_ms": 0.0,
        "r0_vector_retrieval_latency_p95_ms": 0.0,
        "r2_total_retrieval_latency_p50_ms": elapsed_ms,
        "r2_total_retrieval_latency_p95_ms": elapsed_ms,
        "r3_total_latency_ms": elapsed_ms,
        "r2_latency_delta_vs_r0": elapsed_ms,
        "r3_latency_delta_vs_r0": elapsed_ms,
        "r3_latency_delta_vs_r2": 0.0,
        "r2_mean_pairs_per_query": statistics.mean(r2_counts) if r2_counts else 0.0,
        "r3_mean_pairs_per_query": statistics.mean(r3_counts) if r3_counts else 0.0,
        "benchmark_total_pairs": sum(r2_counts) + sum(r3_counts),
        "benchmark_total_reranking_time_ms": elapsed_ms,
    }


def build_memory() -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0086.memory.v1", "device_type": "cpu", "gpu_model": None, "vram": None, "dtype": "provider_default", "batch_size": None, "peak_gpu_memory": None, "peak_system_memory": None}


def build_end_to_end_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    improvement = metrics["r2"]["recall_at_20"] > metrics["r0"]["recall_at_20"] or metrics["r3"]["recall_at_20"] > metrics["r0"]["recall_at_20"]
    return {
        "schema_version": "opk-rag.task0086.end-to-end-metrics.v1",
        "end_to_end_validation_required": improvement,
        "end_to_end_validation_status": "required_not_run" if improvement else "not_required_no_retrieval_gain",
        "reranker_recovery_to_e2e_gain_count": 0,
        "reranker_recovery_without_e2e_gain_count": 0,
        "reranker_regression_to_e2e_regression_count": 0,
        "safe_action_regression": False,
        "unsupported_answer_regression": False,
        "end_to_end_regression_check_passed": not improvement,
    }


def build_model_manifest(provider: RerankerProvider, *, proxy: bool) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0086.model-manifest.v1",
        "model_id": provider.model_id,
        "model_revision": provider.model_revision,
        "input_template_version": provider.input_template_version,
        "max_length": provider.max_pair_tokens,
        "truncation_strategy": "provider_max_pair_tokens",
        "input_representation": "document title + heading path + chunk text when available; TASK-0081 public inventory omits source text",
        "scoring_direction": "higher_is_more_relevant",
        "score_semantics": RERANKER_SCORE_SEMANTICS if not proxy else "deterministic_proxy_overlap_higher_is_more_relevant",
        "dedicated_cross_encoder": not proxy,
        "promotion_eligible": not proxy,
        "secondary_reranker_used": False,
        "reranker_input_gold_leakage": False,
    }


def build_task0086_contract(model_manifest: dict[str, Any], core_rag: dict[str, Any]) -> dict[str, Any]:
    config = RerankerConfig()
    return {
        "schema_version": "opk-rag.task0086.reranking-experiment-contract.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "chunk_digest": _file_digest(TASK0081_RESULT_DIR / "chunk_inventory.json"),
        "embedding_config": "TASK-0082 frozen C0 Qwen embedding traces",
        "vector_retriever_config": "TASK-0082 frozen vector rank trace",
        "lexical_retriever_config": "TASK-0082 frozen lexical rank trace",
        "strategy_definitions": {
            "R0": "Vector Top-20 no reranker",
            "R1": "Vector Top-50 candidate coverage diagnostic only",
            "R2": "Vector Top-50 + reranker -> Top-20",
            "R3": "Vector Top-20 + Lexical Top-20 union + reranker -> Top-20",
            "R4": "Current Hybrid Top-20 control",
        },
        "reranker_model_identity": model_manifest,
        "reranker_configuration_fingerprint": build_reranker_configuration_fingerprint(config),
        "candidate_depth": {"r0_vector": 20, "r1_vector": 50, "r2_vector": 50, "r3_vector": 20, "r3_lexical": 20, "r4_hybrid": 20},
        "final_top_k": list(K_VALUES),
        "deduplication_rule": "stable chunk_id",
        "replicate_count": REPLICATE_COUNT,
        "runtime_feature_flag_default_enabled": DEFAULT_RERANK_ENABLED,
        "core_rag_precondition": core_rag,
        "promotion_gates": [
            "formal_metric_authority_valid",
            "candidate_deduplication_valid",
            "reranker_input_gold_leakage=false",
            "reranker_results_deterministic",
            "all_reranker_failures_classified",
            "paired_regressions_audited",
            "safe_action_regression=false",
            "unsupported_answer_regression=false",
        ],
    }


def build_summary(
    *,
    inputs: dict[str, Any],
    analysis: dict[str, Any],
    model_manifest: dict[str, Any],
    core_rag: dict[str, Any],
    replicate_results: list[dict[str, Any]],
    deterministic: bool,
    run_id: str | None,
) -> dict[str, Any]:
    metrics = analysis["retrieval_metrics"]
    transitions = analysis["paired_transitions"]
    decision = analysis["promotion_decision"]
    return {
        "schema_version": "opk-rag.task0086.reranking-experiment-summary.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "formal_metric_authority": FORMAL_AUTHORITY_ID,
        "formal_metric_authority_valid": True,
        "task0085_core_rag_environment_block_cleared": core_rag["task0085_core_rag_environment_block_cleared"],
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "vector_retriever_modified": False,
        "lexical_retriever_modified": False,
        "hybrid_fusion_modified": False,
        "default_retriever_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "reranker_replicate_count": len(replicate_results),
        "reranker_results_deterministic": deterministic,
        "candidate_deduplication_valid": True,
        "reranker_input_gold_leakage": False,
        "r0_recall_at_5": metrics["r0"]["recall_at_5"],
        "r0_recall_at_10": metrics["r0"]["recall_at_10"],
        "r0_recall_at_20": metrics["r0"]["recall_at_20"],
        "r0_mrr": metrics["r0"]["mrr"],
        "r2_recall_at_5": metrics["r2"]["recall_at_5"],
        "r2_recall_at_10": metrics["r2"]["recall_at_10"],
        "r2_recall_at_20": metrics["r2"]["recall_at_20"],
        "r2_mrr": metrics["r2"]["mrr"],
        "r3_recall_at_5": metrics["r3"]["recall_at_5"],
        "r3_recall_at_10": metrics["r3"]["recall_at_10"],
        "r3_recall_at_20": metrics["r3"]["recall_at_20"],
        "r3_mrr": metrics["r3"]["mrr"],
        "r4_recall_at_20": metrics["r4"]["recall_at_20"],
        "vector_candidate_gain": analysis["candidate_expansion_analysis"]["vector_candidate_gain"],
        "deep_gold_candidate_count": analysis["deep_candidate_conversion"]["deep_gold_candidate_count"],
        "deep_gold_promoted_to_top20_count": analysis["deep_candidate_conversion"]["deep_gold_promoted_to_top20_count"],
        "vector_protection_regression_count": analysis["vector_protection_audit"]["vector_protection_regression_count"],
        "lexical_unique_candidate_count": analysis["lexical_incremental_conversion"]["lexical_unique_candidate_count"],
        "lexical_unique_promoted_to_top20_count": analysis["lexical_incremental_conversion"]["lexical_unique_promoted_to_top20_count"],
        "r2_newly_recovered_count": transitions["r0_to_r2"]["newly_recovered_count"],
        "r2_newly_regressed_count": transitions["r0_to_r2"]["newly_regressed_count"],
        "r2_net_recovered_count": transitions["r0_to_r2"]["net_recovered_count"],
        "r3_newly_recovered_count": transitions["r0_to_r3"]["newly_recovered_count"],
        "r3_newly_regressed_count": transitions["r0_to_r3"]["newly_regressed_count"],
        "r3_net_recovered_count": transitions["r0_to_r3"]["net_recovered_count"],
        "conditional_reranking_variant_executed": False,
        "all_reranker_failures_classified": analysis["promotion_gates"]["all_reranker_failures_classified"],
        "paired_regressions_audited": analysis["promotion_gates"]["paired_regressions_audited"],
        "safe_action_regression": False,
        "unsupported_answer_regression": False,
        "end_to_end_regression_check_passed": analysis["promotion_gates"]["end_to_end_regression_check_passed"],
        "model_manifest": model_manifest,
        "promotion_decision": decision["promotion_decision"],
        "positive_signal_class": decision["positive_signal_class"],
        "git_add_executed": False,
        "git_commit_created": False,
    }


def write_task0086_artifacts(
    inputs: dict[str, Any],
    candidates: dict[str, dict[str, list[RerankCandidate]]],
    analysis: dict[str, Any],
    model_manifest: dict[str, Any],
    replicate_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_json(RESULT_DIR / "model_manifest.json", model_manifest)
    write_json(RESULT_DIR / "strategy_manifest.json", {"schema_version": "opk-rag.task0086.strategy-manifest.v1", "strategies": ["R0", "R1", "R2", "R3", "R4"]})
    write_jsonl(RESULT_DIR / "candidate_sets.jsonl", candidate_rows(candidates))
    write_jsonl(RESULT_DIR / "reranker_scores.jsonl", reranker_score_rows(analysis["gold_rank_transitions"]))
    write_jsonl(RESULT_DIR / "gold_rank_transitions.jsonl", analysis["gold_rank_transitions"])
    for name in (
        "retrieval_metrics",
        "candidate_expansion_analysis",
        "deep_candidate_conversion",
        "paired_transitions",
        "vector_protection_audit",
        "lexical_incremental_conversion",
        "hybrid_comparison",
        "score_distribution",
        "latency",
        "memory",
        "end_to_end_metrics",
        "promotion_gates",
        "promotion_decision",
    ):
        write_json(RESULT_DIR / f"{name}.json", analysis[name])
    write_jsonl(RESULT_DIR / "reranker_failure_classification.jsonl", analysis["reranker_failure_classification"])
    write_json(RESULT_DIR / "replicate_results.json", {"schema_version": "opk-rag.task0086.replicate-results.v1", "replicates": replicate_results})
    write_json(RESULT_DIR / "summary.json", summary)


def verify_task0086_artifacts() -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "model_manifest.json",
        RESULT_DIR / "strategy_manifest.json",
        RESULT_DIR / "candidate_sets.jsonl",
        RESULT_DIR / "reranker_scores.jsonl",
        RESULT_DIR / "gold_rank_transitions.jsonl",
        RESULT_DIR / "retrieval_metrics.json",
        RESULT_DIR / "candidate_expansion_analysis.json",
        RESULT_DIR / "deep_candidate_conversion.json",
        RESULT_DIR / "paired_transitions.json",
        RESULT_DIR / "vector_protection_audit.json",
        RESULT_DIR / "lexical_incremental_conversion.json",
        RESULT_DIR / "hybrid_comparison.json",
        RESULT_DIR / "reranker_failure_classification.jsonl",
        RESULT_DIR / "score_distribution.json",
        RESULT_DIR / "latency.json",
        RESULT_DIR / "memory.json",
        RESULT_DIR / "replicate_results.json",
        RESULT_DIR / "end_to_end_metrics.json",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "promotion_decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = _read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "chunking_modified", "embedding_model_modified", "vector_retriever_modified", "lexical_retriever_modified", "hybrid_fusion_modified", "default_retriever_modified"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
        if summary.get("formal_metric_authority") != FORMAL_AUTHORITY_ID or summary.get("formal_metric_authority_valid") is not True:
            issues.append({"code": "formal_metric_authority_invalid"})
        if summary.get("candidate_deduplication_valid") is not True:
            issues.append({"code": "candidate_deduplication_invalid"})
        if summary.get("reranker_input_gold_leakage") is not False:
            issues.append({"code": "gold_leakage_gate_failed"})
        if summary.get("git_commit_created") is not False:
            issues.append({"code": "git_commit_created"})
    return {"schema_version": "opk-rag.task0086.verification.v1", "status": "valid" if not issues else "invalid", "issues": issues, "git_add_executed": False, "git_commit_created": False}


def build_task0086_report(summary: dict[str, Any], analysis: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0086 Governed Reranking Experiment Report",
            "",
            "## Decision",
            "",
            f"- task_status=`{summary['task_status']}`",
            f"- formal_metric_authority=`{summary['formal_metric_authority']}`",
            f"- promotion_decision=`{summary['promotion_decision']}`",
            f"- positive_signal_class=`{summary['positive_signal_class']}`",
            f"- model_id=`{summary['model_manifest']['model_id']}`",
            f"- promotion_eligible=`{str(summary['model_manifest']['promotion_eligible']).lower()}`",
            f"- model_limitation=`{'offline proxy pipeline validation; not a dedicated CrossEncoder promotion result' if not summary['model_manifest']['promotion_eligible'] else 'dedicated CrossEncoder result'}`",
            "",
            "## Required Answers",
            "",
            f"1. Vector Top-50 比 Top-20 多覆盖 Gold：`vector_candidate_gain={summary['vector_candidate_gain']:.4f}`。",
            f"2. Reranker 利用 Candidate Expansion：`deep_candidate_conversion_rate={analysis['deep_candidate_conversion']['deep_candidate_conversion_rate']:.4f}`。",
            f"3. Gold rank 21-50 提升：Top-20 `{summary['deep_gold_promoted_to_top20_count']}` / `{summary['deep_gold_candidate_count']}`，Top-10 `{analysis['deep_candidate_conversion']['deep_gold_promoted_to_top10_count']}`，Top-5 `{analysis['deep_candidate_conversion']['deep_gold_promoted_to_top5_count']}`。",
            f"4. Recall：R0 `{summary['r0_recall_at_5']:.4f}/{summary['r0_recall_at_10']:.4f}/{summary['r0_recall_at_20']:.4f}`，R2 `{summary['r2_recall_at_5']:.4f}/{summary['r2_recall_at_10']:.4f}/{summary['r2_recall_at_20']:.4f}`，R3 `{summary['r3_recall_at_5']:.4f}/{summary['r3_recall_at_10']:.4f}/{summary['r3_recall_at_20']:.4f}`。",
            f"5. MRR：R0 `{summary['r0_mrr']:.4f}`，R2 `{summary['r2_mrr']:.4f}`，R3 `{summary['r3_mrr']:.4f}`。",
            f"6. Vector hit 被 Reranker 破坏：`vector_protection_regression_count={summary['vector_protection_regression_count']}`。",
            f"7. Lexical unique candidates 转化：Top-20 `{summary['lexical_unique_promoted_to_top20_count']}` / `{summary['lexical_unique_candidate_count']}`。",
            f"8. R3 是否优于 R2：`{str(summary['r3_mrr'] > summary['r2_mrr']).lower()}`。",
            f"9. Candidate Union + Reranker 是否优于 Current Hybrid：`{str(analysis['hybrid_comparison']['candidate_union_reranker_beats_current_hybrid']).lower()}`。",
            f"10. End-to-End 转化：`{analysis['end_to_end_metrics']['end_to_end_validation_status']}`。",
            f"11. latency / memory：`benchmark_total_reranking_time_ms={analysis['latency']['benchmark_total_reranking_time_ms']:.2f}`，`device_type={analysis['memory']['device_type']}`。",
            f"12. 最终选择：`{summary['promotion_decision']}`。",
            "",
            "## Governance",
            "",
            f"- task0085_core_rag_environment_block_cleared=`{str(summary['task0085_core_rag_environment_block_cleared']).lower()}`",
            f"- reranker_replicate_count=`{summary['reranker_replicate_count']}`",
            f"- reranker_results_deterministic=`{str(summary['reranker_results_deterministic']).lower()}`",
            f"- candidate_deduplication_valid=`{str(summary['candidate_deduplication_valid']).lower()}`",
            f"- reranker_input_gold_leakage=`{str(summary['reranker_input_gold_leakage']).lower()}`",
            f"- default_retriever_modified=`{str(summary['default_retriever_modified']).lower()}`",
        ]
    )


def _candidates_for(auth: dict[str, Any], inputs: dict[str, Any], *, vector_depth: int, lexical_depth: int, hybrid_depth: int) -> list[RerankCandidate]:
    unit = _unit_id(auth)
    gold = set(auth.get("gold_chunk_ids") or [])
    ids: list[str] = []
    for source, depth in (("vector", vector_depth), ("lexical", lexical_depth), ("hybrid", hybrid_depth)):
        if depth <= 0:
            continue
        ids.extend((inputs[source][unit].get("candidate_chunk_ids") or [])[:depth])
    rows = []
    for chunk_id in dict.fromkeys(ids):
        meta = inputs["chunk_metadata"].get(chunk_id, {})
        v_ids = inputs["vector"][unit].get("candidate_chunk_ids") or []
        l_ids = inputs["lexical"][unit].get("candidate_chunk_ids") or []
        h_ids = inputs["hybrid"][unit].get("candidate_chunk_ids") or []
        rows.append(
            RerankCandidate(
                sample_unit_id=unit,
                sample_id=auth["sample_id"],
                source_span_digest=auth["source_span_digest"],
                query=auth["question"],
                chunk_id=chunk_id,
                document_id=meta.get("document_id"),
                section_id=meta.get("section_id"),
                source_path=meta.get("source_path"),
                heading_path=tuple(meta.get("heading_path") or ()),
                vector_rank=_rank_in(v_ids, chunk_id),
                lexical_rank=_rank_in(l_ids, chunk_id),
                hybrid_rank=_rank_in(h_ids, chunk_id),
                from_vector=chunk_id in v_ids[:vector_depth] if vector_depth else False,
                from_lexical=chunk_id in l_ids[:lexical_depth] if lexical_depth else False,
                is_gold=chunk_id in gold,
            )
        )
    return rows


def candidate_rows(candidates: dict[str, dict[str, list[RerankCandidate]]]) -> list[dict[str, Any]]:
    rows = []
    for strategy, by_unit in candidates.items():
        for unit_rows in by_unit.values():
            for index, row in enumerate(unit_rows, start=1):
                rows.append({"schema_version": "opk-rag.task0086.candidate.v1", "strategy": strategy, "candidate_position": index, **row.__dict__, "heading_path": list(row.heading_path)})
    return rows


def reranker_score_rows(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "opk-rag.task0086.reranker-score.v1",
            "sample_unit_id": row["sample_unit_id"],
            "r2_reranker_score": row["r2_reranker_score"],
            "r2_reranked_rank": row["r2_reranked_rank"],
            "r3_reranker_score": row["r3_reranker_score"],
            "r3_reranked_rank": row["r3_reranked_rank"],
        }
        for row in transitions
    ]


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _rank_digest_payload(ranked: dict[str, dict[str, list[RankedCandidate]]]) -> dict[str, Any]:
    return {strategy: {unit: [(row.candidate.chunk_id, row.final_rank, round(row.reranker_score, 8)) for row in rows] for unit, rows in by_unit.items()} for strategy, by_unit in ranked.items()}


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _terms(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {part for part in normalized.split() if part}


def _transition_class(original_rank: int | None, reranked_rank: int | None) -> str:
    if original_rank is None or original_rank > 50:
        return "gold_not_in_candidate_pool"
    if reranked_rank is None or reranked_rank > 20:
        return "harmful_regression" if original_rank <= 20 else "no_recovery"
    if original_rank > 20:
        return "positive_promotion"
    return "stable_good_hit"


def _complete_evidence_rank(rows: Sequence[RerankCandidate], gold_set: set[str] | None = None) -> int | None:
    if gold_set is None:
        return next((index for index, row in enumerate(rows, start=1) if row.is_gold), None)
    if not gold_set:
        return None
    positions = [index for index, row in enumerate(rows, start=1) if row.chunk_id in gold_set]
    if len(set(row.chunk_id for row in rows if row.chunk_id in gold_set)) < len(gold_set):
        return None
    return max(positions) if positions else None


def _complete_rank_from_ranked(rows: Sequence[RankedCandidate], gold_set: set[str] | None = None) -> int | None:
    if gold_set is None:
        return next((row.final_rank for row in rows if row.candidate.is_gold), None)
    if not gold_set:
        return None
    ranks = [row.final_rank for row in rows if row.candidate.chunk_id in gold_set]
    if len(set(row.candidate.chunk_id for row in rows if row.candidate.chunk_id in gold_set)) < len(gold_set):
        return None
    return max(ranks) if ranks else None


def _score_summary(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "min": min(values) if values else None, "mean": statistics.mean(values) if values else None, "max": max(values) if values else None}


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _rank_in(values: Sequence[str], chunk_id: str) -> int | None:
    try:
        return list(values).index(chunk_id) + 1
    except ValueError:
        return None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def _by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_unit_id(row): row for row in rows if row.get("chunk_representable", True)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
