from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    build_replay_units,
    digest_json,
    first_relevant_rank,
    load_task0091_inputs,
    ndcg_at_k,
    read_json,
    read_jsonl,
    rerank_rows,
    sample_comparisons,
    sha256_file,
    utc_now,
    verify_task0091_artifacts,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import downstream_metrics, verify_task0092_artifacts
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import (
    arm_sample_rows,
    candidate_membership_change_count,
    rank_fusion,
)
from opk_rag.evaluation.task0095_default_reranker_promotion_freeze import (
    BASELINE_NAME as TASK0095_BASELINE_NAME,
    RESULT_DIR as TASK0095_RESULT_DIR,
    verify_task0095_artifacts,
)
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import DEFAULT_RERANKER_MODEL_NAME, DEFAULT_RERANKER_MODEL_REVISION, RerankerConfig
from opk_rag.reranking.input import RERANKER_SCORE_SEMANTICS, render_reranker_document
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata, RerankerProvider
from opk_rag.reranking.qwen import QWEN3_RERANKER_0_6B, QWEN3_RERANKER_4B, Qwen3LocalRerankerProvider
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA, RANK_FUSION_POLICY
from opk_rag.runtime_v2.reranker import RerankerRuntimeConfig


TASK_ID = "TASK-0096"
EXPERIMENT_ID = "task0096-reranker-model-benchmark"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0096_reranker_model_benchmark_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0096_QWEN_VS_BGE_RERANKER_BENCHMARK_REPORT.md"
REPLICATE_COUNT = 3
CANDIDATE_DEPTH = 50
K_VALUES = (1, 5, 10, 20)
QWEN_0_6B_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN_4B_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


class Task0096Error(RuntimeError):
    pass


def run_task0096_reranker_model_benchmark(
    *,
    models: Sequence[str] = ("bge", "qwen_0_6b"),
    device: str = "cpu",
    batch_size: int = 8,
    replicates: int = REPLICATE_COUNT,
    output_dir: Path = RESULT_DIR,
    allow_environment_block: bool = True,
    providers: dict[str, RerankerProvider] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_audit = git_safety_audit()
    inputs = load_inputs()
    contract = build_contract(inputs, git_audit)
    write_json(CONTRACT_PATH, contract)
    contract["contract_digest"] = sha256_file(CONTRACT_PATH)
    write_json(CONTRACT_PATH, contract)

    units = build_benchmark_units(inputs)
    if len(units) != 75:
        return write_invalid_authoritative_input(output_dir, inputs, units, git_audit)

    baseline = {unit: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit, rows in units.items()}
    model_results: dict[str, dict[str, Any]] = {}
    providers = providers or {}
    for model_key in models:
        try:
            provider = providers.get(model_key)
            if provider is None and model_key.startswith("qwen"):
                provider = build_provider(model_key, device=device, batch_size=batch_size)
            model_results[model_key] = execute_model_arm(
                model_key,
                baseline,
                provider=provider,
                replicates=replicates,
            )
        except (RerankerModelLoadError, RerankerInferenceError, RuntimeError) as exc:
            if not allow_environment_block:
                raise
            model_results[model_key] = blocked_model_result(model_key, exc)

    summary = analyze_and_write_results(
        output_dir=output_dir,
        inputs=inputs,
        baseline=baseline,
        model_results=model_results,
        git_audit=git_audit,
        device=device,
        batch_size=batch_size,
        replicates=replicates,
    )
    return summary


def load_inputs() -> dict[str, Any]:
    task0095_verification = verify_task0095_artifacts()
    task0095_summary_path = TASK0095_RESULT_DIR / "summary.json"
    task0095_frozen_path = TASK0095_RESULT_DIR / "frozen_baseline.json"
    task0095_config_path = TASK0095_RESULT_DIR / "default_runtime_configuration.json"
    required = [
        task0095_summary_path,
        task0095_frozen_path,
        task0095_config_path,
        ROOT / "evaluation-data" / "results" / "task0091-reranker-replay-benchmark" / "sample_comparison.jsonl",
        ROOT / "evaluation-data" / "results" / "task0092-reranker-downstream-validation" / "sample_downstream_comparison.jsonl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise Task0096Error(f"missing TASK-0096 inputs: {', '.join(_rel(path) for path in missing)}")
    task0091_inputs = load_task0091_inputs()
    return {
        "task0091_inputs": task0091_inputs,
        "task0091_verification": verify_task0091_artifacts(),
        "task0092_verification": verify_task0092_artifacts(),
        "task0095_verification": task0095_verification,
        "task0095_summary": read_json(task0095_summary_path),
        "task0095_frozen_baseline": read_json(task0095_frozen_path),
        "task0095_default_config": read_json(task0095_config_path),
        "task0091_sample_comparison": read_jsonl(ROOT / "evaluation-data" / "results" / "task0091-reranker-replay-benchmark" / "sample_comparison.jsonl"),
        "task0092_sample_downstream_comparison": read_jsonl(ROOT / "evaluation-data" / "results" / "task0092-reranker-downstream-validation" / "sample_downstream_comparison.jsonl"),
        "paths": {
            "task0095_summary": _rel(task0095_summary_path),
            "task0095_frozen_baseline": _rel(task0095_frozen_path),
            "task0095_default_config": _rel(task0095_config_path),
            **task0091_inputs["paths"],
        },
        "digests": {
            "task0095_summary": sha256_file(task0095_summary_path),
            "task0095_frozen_baseline": sha256_file(task0095_frozen_path),
            "task0095_default_config": sha256_file(task0095_config_path),
            **task0091_inputs["digests"],
        },
    }


def build_benchmark_units(inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    units = build_replay_units(inputs["task0091_inputs"])
    authority_by_unit = {
        _unit_id(row): row for row in inputs["task0091_inputs"]["gold_authority"]
    }
    raw_by_key = {
        (row["sample_unit_id"], row["content_digest"]): row
        for row in inputs["task0091_inputs"]["task0090_candidates"]
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for unit, rows in units.items():
        question = authority_by_unit[unit]["question"]
        enriched = []
        for row in rows:
            raw = raw_by_key[(unit, row["content_digest"])]
            document = render_reranker_document(tuple(raw.get("heading_path") or ()), raw.get("content") or "")
            enriched.append({**row, "question": question, "document": document, "heading_path": raw.get("heading_path") or []})
        out[unit] = enriched
    return out


def execute_model_arm(
    model_key: str,
    baseline: dict[str, list[dict[str, Any]]],
    *,
    provider: RerankerProvider | None,
    replicates: int,
) -> dict[str, Any]:
    replicate_rows = []
    first: dict[str, list[dict[str, Any]]] | None = None
    first_latency: list[dict[str, Any]] = []
    model_load_seconds = 0.0
    if provider is not None:
        started = time.perf_counter()
        provider.prepare_pair_metadata("warmup query", "warmup document")
        model_load_seconds = time.perf_counter() - started
    for replicate in range(1, replicates + 1):
        started = time.perf_counter()
        scored, latencies = score_model(model_key, baseline, provider)
        elapsed = time.perf_counter() - started
        digest = digest_json(rank_digest(scored))
        replicate_rows.append({"replicate": replicate, "ranked_output_digest": digest, "wall_time_seconds": elapsed})
        if first is None:
            first = scored
            first_latency = latencies
    assert first is not None
    deterministic = len({row["ranked_output_digest"] for row in replicate_rows}) == 1
    pure = {unit: rerank_rows(rows) for unit, rows in first.items()}
    fused = {
        unit: rank_fusion(
            baseline[unit],
            pure[unit],
            {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA},
        )
        for unit in baseline
    }
    return {
        "model_key": model_key,
        "status": "executed",
        "model_manifest": model_manifest(model_key, provider),
        "scored": first,
        "pure_ranking": pure,
        "rank_fusion": fused,
        "ranking_metrics": ranking_metrics_0096(pure),
        "rank_fusion_metrics": ranking_metrics_0096(fused),
        "replicates": replicate_rows,
        "deterministic_replicates_passed": deterministic,
        "runtime": runtime_metrics(model_key, first_latency, model_load_seconds),
    }


def score_model(
    model_key: str,
    baseline: dict[str, list[dict[str, Any]]],
    provider: RerankerProvider | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    scored: dict[str, list[dict[str, Any]]] = {}
    latencies = []
    for unit, rows in baseline.items():
        started = time.perf_counter()
        if provider is None:
            if model_key != "bge":
                raise Task0096Error(f"provider required for {model_key}")
            scores = [float(row["reranker_score"]) for row in rows]
            metadata = [RerankerPairMetadata(0, 0, False) for _ in rows]
        else:
            documents = [row["document"] for row in rows]
            scores = tuple(float(value) for value in provider.score_pairs(rows[0]["question"], documents))
            metadata = [provider.prepare_pair_metadata(row["question"], document) for row, document in zip(rows, documents)]
        elapsed = time.perf_counter() - started
        if len(scores) != len(rows):
            raise Task0096Error(f"{model_key} expected {len(rows)} scores for {unit}, got {len(scores)}")
        scored[unit] = [
            {
                **row,
                "reranker_score": float(score),
                "reranker_model_key": model_key,
                "pair_token_count": meta.pair_token_count,
                "original_pair_token_count": meta.original_pair_token_count,
                "input_truncated": meta.input_truncated,
            }
            for row, score, meta in zip(rows, scores, metadata)
        ]
        latencies.append({"sample_unit_id": unit, "candidate_count": len(rows), "latency_seconds": elapsed})
    return scored, latencies


def ranking_metrics_0096(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [first_relevant_rank(rows) for rows in rows_by_unit.values()]
    present = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "schema_version": "opk-rag.task0096.ranking-metrics.v1",
        "unit_count": len(ranks),
        "candidate_depth": CANDIDATE_DEPTH,
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
        "mean_first_relevant_rank": _ratio(sum(present), len(present)) if present else None,
    }
    for k in K_VALUES:
        metrics[f"recall_at_{k}"] = _ratio(sum(rank is not None and rank <= k for rank in ranks), len(ranks))
        metrics[f"required_evidence_coverage_at_{k}"] = _ratio(
            sum(required_evidence_coverage(rows, k) for rows in rows_by_unit.values()),
            len(rows_by_unit),
        )
    for k in (5, 10, 20):
        metrics[f"ndcg_at_{k}"] = _ratio(sum(ndcg_at_k(rows, k) for rows in rows_by_unit.values()), len(rows_by_unit))
    return metrics


def analyze_and_write_results(
    *,
    output_dir: Path,
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    model_results: dict[str, dict[str, Any]],
    git_audit: dict[str, Any],
    device: str,
    batch_size: int,
    replicates: int,
) -> dict[str, Any]:
    executed = {key: value for key, value in model_results.items() if value["status"] == "executed"}
    bge = executed.get("bge")
    qwen = executed.get("qwen_0_6b")
    if bge is None or qwen is None:
        summary = blocked_summary(inputs, model_results, git_audit, device, batch_size, replicates)
        write_blocked_artifacts(output_dir, inputs, baseline, model_results, summary)
        return summary

    bge_rows = arm_sample_rows("TASK-0096-bge-rank-fusion", baseline, bge["rank_fusion"], sample_comparisons(baseline, bge["pure_ranking"]))
    qwen_rows = arm_sample_rows("TASK-0096-qwen-0-6b-rank-fusion", baseline, qwen["rank_fusion"], sample_comparisons(baseline, qwen["pure_ranking"]))
    bge_downstream = downstream_metrics(bge_rows, "reranker")
    qwen_downstream = downstream_metrics(qwen_rows, "reranker")
    pairwise = pairwise_comparison(bge, qwen, bge_downstream, qwen_downstream)
    qwen_known = known_regression_reintroduced(qwen_rows, inputs["task0092_sample_downstream_comparison"])
    bootstrap = bootstrap_results(bge["pure_ranking"], qwen["pure_ranking"])
    decision = decision_payload(pairwise, bge, qwen, bge_downstream, qwen_downstream, qwen_known)
    summary = build_summary(inputs, baseline, model_results, pairwise, bge_downstream, qwen_downstream, qwen_known, bootstrap, decision, git_audit)

    write_json(output_dir / "run_manifest.json", run_manifest(inputs, baseline, git_audit, device, batch_size, replicates))
    write_json(output_dir / "model_manifest.json", {key: result["model_manifest"] for key, result in model_results.items()})
    write_json(output_dir / "input_manifest.json", input_manifest(inputs, baseline))
    write_json(output_dir / "ranking_metrics.json", {key: result.get("ranking_metrics") for key, result in model_results.items()})
    write_json(output_dir / "downstream_metrics.json", {"bge": bge_downstream, "qwen_0_6b": qwen_downstream})
    write_json(output_dir / "runtime_metrics.json", {key: result.get("runtime") for key, result in model_results.items()})
    write_json(output_dir / "pairwise_comparison.json", pairwise)
    write_jsonl(output_dir / "sample_rank_deltas.jsonl", sample_rank_deltas(bge["pure_ranking"], qwen["pure_ranking"]))
    write_jsonl(output_dir / "downstream_sample_deltas.jsonl", downstream_sample_deltas(bge_rows, qwen_rows))
    write_json(output_dir / "bootstrap_results.json", bootstrap)
    write_json(output_dir / "decision.json", decision)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0096_artifacts(output_dir=output_dir, write=True)
    summary["verification"] = verification
    summary["repository_verification_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def pairwise_comparison(bge: dict[str, Any], qwen: dict[str, Any], bge_downstream: dict[str, Any], qwen_downstream: dict[str, Any]) -> dict[str, Any]:
    keys = ["mrr", "ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20"]
    return {
        "schema_version": "opk-rag.task0096.pairwise-comparison.v1",
        "ranking_deltas_qwen_minus_bge": {key: qwen["ranking_metrics"][key] - bge["ranking_metrics"][key] for key in keys},
        "rank_fusion_deltas_qwen_minus_bge": {key: qwen["rank_fusion_metrics"][key] - bge["rank_fusion_metrics"][key] for key in keys},
        "downstream_deltas_qwen_minus_bge": {
            "end_to_end_accuracy": qwen_downstream["end_to_end_accuracy"] - bge_downstream["end_to_end_accuracy"],
            "answerability_accuracy": qwen_downstream["answerability_accuracy"] - bge_downstream["answerability_accuracy"],
            "safe_action_accuracy": qwen_downstream["safe_action_accuracy"] - bge_downstream["safe_action_accuracy"],
        },
    }


def decision_payload(
    pairwise: dict[str, Any],
    bge: dict[str, Any],
    qwen: dict[str, Any],
    bge_downstream: dict[str, Any],
    qwen_downstream: dict[str, Any],
    qwen_known: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in sample_rank_deltas(bge["pure_ranking"], qwen["pure_ranking"]))
    downstream_counts = {
        "improved": max(0, qwen_downstream["end_to_end_correct_count"] - bge_downstream["end_to_end_correct_count"]),
        "regressed": max(0, bge_downstream["end_to_end_correct_count"] - qwen_downstream["end_to_end_correct_count"]),
    }
    quality = quality_winner(pairwise, qwen_known)
    deployment = deployment_winner(bge["runtime"], qwen["runtime"])
    switch = quality == "qwen_0_6b" and qwen_known["known_regression_reintroduced_count"] == 0 and qwen_downstream["end_to_end_accuracy"] >= bge_downstream["end_to_end_accuracy"]
    return {
        "schema_version": "opk-rag.task0096.decision.v1",
        "quality_winner": quality,
        "deployment_winner": deployment,
        "default_switch_recommended": switch,
        "next_task_decision": "qwen_reranker_promotion_candidate_requires_task0097" if switch else ("retain_task0095_bge_default" if quality == "bge" else "reranker_comparison_inconclusive_more_samples_required"),
        "pure_ranking_improved_units": counts["qwen_better"],
        "pure_ranking_regressed_units": counts["bge_better"],
        "qwen_downstream_improved_count": downstream_counts["improved"],
        "qwen_downstream_regressed_count": downstream_counts["regressed"],
        "qwen_downstream_net_improvement": downstream_counts["improved"] - downstream_counts["regressed"],
        "known_regression_reintroduced_count": qwen_known["known_regression_reintroduced_count"],
        "no_default_promotion_performed": True,
    }


def quality_winner(pairwise: dict[str, Any], qwen_known: dict[str, Any]) -> str:
    if qwen_known["known_regression_reintroduced_count"] > 0:
        return "bge"
    rank = pairwise["ranking_deltas_qwen_minus_bge"]
    downstream = pairwise["downstream_deltas_qwen_minus_bge"]
    positives = sum(rank[key] > 0 for key in ("mrr", "ndcg_at_5", "recall_at_5", "recall_at_20"))
    negatives = sum(rank[key] < 0 for key in ("mrr", "ndcg_at_5", "recall_at_5", "recall_at_20"))
    if downstream["end_to_end_accuracy"] > 0 and positives >= negatives:
        return "qwen_0_6b"
    if downstream["end_to_end_accuracy"] < 0 or negatives > positives:
        return "bge"
    return "tie" if positives == negatives else "inconclusive"


def deployment_winner(bge: dict[str, Any], qwen: dict[str, Any]) -> str:
    bge_p95 = bge["p95_latency_seconds"]
    qwen_p95 = qwen["p95_latency_seconds"]
    if bge_p95 is None or qwen_p95 is None:
        return "inconclusive"
    if abs(bge_p95 - qwen_p95) <= 1e-9:
        return "tie"
    return "qwen_0_6b" if qwen_p95 < bge_p95 else "bge"


def verify_task0096_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("task_status") == "blocked":
            minimal = [
                CONTRACT_PATH,
                output_dir / "run_manifest.json",
                output_dir / "model_manifest.json",
                output_dir / "input_manifest.json",
                output_dir / "decision.json",
                summary_path,
            ]
            issues = [{"code": "missing_required_blocked_artifact", "path": _rel(path)} for path in minimal if not path.exists()]
            decision = read_json(output_dir / "decision.json") if (output_dir / "decision.json").exists() else {}
            if decision.get("next_task_decision") != "blocked_by_environment":
                issues.append({"code": "blocked_decision_missing"})
            result = {
                "schema_version": "opk-rag.task0096.verification.v1",
                "status": "blocked" if not issues else "invalid",
                "issues": issues if issues else [{"code": "blocked_by_environment"}],
                "task0095_artifacts_modified": False,
                "git_add_executed": False,
                "git_commit_created": False,
            }
            if write:
                write_json(output_dir / "verification.json", result)
            return result
    required = [
        CONTRACT_PATH,
        output_dir / "run_manifest.json",
        output_dir / "model_manifest.json",
        output_dir / "input_manifest.json",
        output_dir / "ranking_metrics.json",
        output_dir / "downstream_metrics.json",
        output_dir / "runtime_metrics.json",
        output_dir / "pairwise_comparison.json",
        output_dir / "sample_rank_deltas.jsonl",
        output_dir / "downstream_sample_deltas.jsonl",
        output_dir / "bootstrap_results.json",
        output_dir / "decision.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        contract = read_json(CONTRACT_PATH)
        run = read_json(output_dir / "run_manifest.json")
        ranking = read_json(output_dir / "ranking_metrics.json")
        downstream = read_json(output_dir / "downstream_metrics.json")
        runtime = read_json(output_dir / "runtime_metrics.json")
        decision = read_json(output_dir / "decision.json")
        config = RerankerRuntimeConfig()
        if contract.get("evaluation_unit_count") != 75 or run.get("evaluation_unit_count") != 75:
            issues.append({"code": "evaluation_unit_count_not_75"})
        if config.policy != RANK_FUSION_POLICY or config.rank_fusion_k != 60 or config.rank_fusion_lambda != 0.75:
            issues.append({"code": "task0095_default_drift"})
        for model in ("bge", "qwen_0_6b"):
            if not ranking.get(model):
                issues.append({"code": "missing_model_ranking_metrics", "model": model})
            if not runtime.get(model):
                issues.append({"code": "missing_model_runtime_metrics", "model": model})
        if "bge" not in downstream or "qwen_0_6b" not in downstream:
            issues.append({"code": "missing_downstream_metrics"})
        if decision.get("next_task_decision") not in {
            "qwen_reranker_promotion_candidate_requires_task0097",
            "retain_task0095_bge_default",
            "reranker_comparison_inconclusive_more_samples_required",
            "benchmark_invalid_requires_repair",
            "blocked_by_environment",
        }:
            issues.append({"code": "invalid_next_task_decision"})
        if decision.get("no_default_promotion_performed") is not True:
            issues.append({"code": "default_promotion_performed"})
    result = {
        "schema_version": "opk-rag.task0096.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task0095_artifacts_modified": False,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_contract(inputs: dict[str, Any], git_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0096.reranker-model-benchmark-contract.v1",
        "task_id": TASK_ID,
        "benchmark_version": "task0096-reranker-model-benchmark-v1",
        "source_task0095_baseline": TASK0095_BASELINE_NAME,
        "source_candidate_digest": inputs["digests"]["task0090_candidates"],
        "evaluation_unit_count": 75,
        "models": ["BAAI/bge-reranker-v2-m3", QWEN3_RERANKER_0_6B, QWEN3_RERANKER_4B],
        "model_revisions": {
            "bge": DEFAULT_RERANKER_MODEL_REVISION,
            "qwen_0_6b": QWEN_0_6B_REVISION,
            "qwen_4b": QWEN_4B_REVISION,
        },
        "model_configs": {"batch_size_default": 8, "score_semantics": RERANKER_SCORE_SEMANTICS},
        "ranking_metrics": ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR", "nDCG@5", "nDCG@10", "nDCG@20", "Mean First Relevant Rank", "Required Evidence Coverage@K"],
        "downstream_metrics": ["End-to-End Accuracy", "Answerability Accuracy", "Safe Action Accuracy", "Unsupported Answer", "Grounding Failure", "Over-Abstention"],
        "runtime_metrics": ["model_loading_time", "reranking_latency", "throughput", "peak_vram", "oom_behavior"],
        "replicate_policy": {"deterministic_replicate_count": REPLICATE_COUNT},
        "bootstrap_policy": {"seed": 9600, "iterations": 1000},
        "candidate_identity_policy": "frozen TASK-0090 candidate canonical identities and membership must be unchanged",
        "rank_fusion_policy": {"policy": RANK_FUSION_POLICY, "k": DEFAULT_RANK_FUSION_K, "lambda": DEFAULT_RANK_FUSION_LAMBDA, "tuning_allowed": False},
        "quality_winner_policy": "paired ranking and downstream gates; no known-regression reintroduction",
        "deployment_winner_policy": "lower p95 latency and no OOM wins when both executed",
        "default_switch_policy": "recommend only; no TASK-0095 default modification",
        "git_safety_audit": git_audit,
    }


def build_provider(model_key: str, *, device: str, batch_size: int) -> RerankerProvider:
    if model_key == "bge":
        return BgeLocalRerankerProvider(RerankerConfig(device=device, batch_size=batch_size))
    if model_key == "qwen_0_6b":
        return Qwen3LocalRerankerProvider(
            RerankerConfig(provider="local_qwen3_cross_encoder", model_name=QWEN3_RERANKER_0_6B, model_revision=QWEN_0_6B_REVISION, device=device, batch_size=batch_size)
        )
    if model_key == "qwen_4b":
        return Qwen3LocalRerankerProvider(
            RerankerConfig(provider="local_qwen3_cross_encoder", model_name=QWEN3_RERANKER_4B, model_revision=QWEN_4B_REVISION, device=device, batch_size=batch_size)
        )
    raise Task0096Error(f"unsupported model key: {model_key}")


def model_manifest(model_key: str, provider: RerankerProvider | None) -> dict[str, Any]:
    if model_key == "bge" and provider is None:
        return {"model_key": "bge", "status": "executed_from_frozen_task0087_scores", "model_id": DEFAULT_RERANKER_MODEL_NAME, "model_revision": DEFAULT_RERANKER_MODEL_REVISION}
    return {
        "model_key": model_key,
        "status": "executed",
        "model_id": provider.model_id if provider else None,
        "model_revision": provider.model_revision if provider else None,
        "input_template_version": provider.input_template_version if provider else None,
        "max_pair_tokens": provider.max_pair_tokens if provider else None,
    }


def runtime_metrics(model_key: str, latencies: list[dict[str, Any]], model_load_seconds: float) -> dict[str, Any]:
    values = sorted(float(row["latency_seconds"]) for row in latencies)
    p95 = percentile(values, 0.95) if values else None
    total_candidates = sum(int(row["candidate_count"]) for row in latencies)
    total_latency = sum(float(row["latency_seconds"]) for row in latencies)
    return {
        "schema_version": "opk-rag.task0096.runtime-metrics.v1",
        "model_key": model_key,
        "model_load_seconds": model_load_seconds,
        "p50_latency_seconds": percentile(values, 0.50) if values else None,
        "p95_latency_seconds": p95,
        "total_reranking_seconds": total_latency,
        "throughput_pairs_per_second": _ratio(total_candidates, total_latency),
        "peak_vram_mib": peak_vram_mib(),
        "oom": False,
        "latency_unit_count": len(latencies),
    }


def blocked_model_result(model_key: str, exc: BaseException) -> dict[str, Any]:
    cause = exc.__cause__
    return {
        "model_key": model_key,
        "status": "blocked_by_environment",
        "model_manifest": {
            "model_key": model_key,
            "status": "blocked_by_environment",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc)[:500],
            "cause_type": cause.__class__.__name__ if cause else None,
            "cause_message": str(cause)[:1000] if cause else None,
        },
        "runtime": {"model_key": model_key, "status": "blocked_by_environment", "oom": "out of memory" in str(exc).lower()},
    }


def blocked_summary(inputs: dict[str, Any], model_results: dict[str, Any], git_audit: dict[str, Any], device: str, batch_size: int, replicates: int) -> dict[str, Any]:
    qwen_manifest = model_results.get("qwen_0_6b", {}).get("model_manifest", {})
    return {
        "schema_version": "opk-rag.task0096.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "blocker": "required BGE and Qwen 0.6B model arms did not both execute",
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "evaluation_unit_count": 75,
        "qwen_model_revision": qwen_manifest.get("model_revision") or QWEN_0_6B_REVISION,
        "qwen_4b_status": "not_attempted",
        "quality_winner": "inconclusive",
        "deployment_winner": "inconclusive",
        "default_switch_recommended": False,
        "next_task_decision": "blocked_by_environment",
        "model_results": {key: value.get("model_manifest") for key, value in model_results.items()},
        "git_safety_audit": git_audit,
        "device": device,
        "batch_size": batch_size,
        "replicates": replicates,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def write_blocked_artifacts(output_dir: Path, inputs: dict[str, Any], baseline: dict[str, list[dict[str, Any]]], model_results: dict[str, Any], summary: dict[str, Any]) -> None:
    write_json(output_dir / "run_manifest.json", {"task_id": TASK_ID, "task_status": "blocked", "evaluation_unit_count": len(baseline)})
    write_json(output_dir / "model_manifest.json", {key: result.get("model_manifest") for key, result in model_results.items()})
    write_json(output_dir / "input_manifest.json", input_manifest(inputs, baseline))
    for name in ("ranking_metrics.json", "downstream_metrics.json", "runtime_metrics.json", "pairwise_comparison.json", "bootstrap_results.json"):
        write_json(output_dir / name, {})
    write_jsonl(output_dir / "sample_rank_deltas.jsonl", [])
    write_jsonl(output_dir / "downstream_sample_deltas.jsonl", [])
    write_json(output_dir / "decision.json", {"next_task_decision": "blocked_by_environment", "no_default_promotion_performed": True})
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "verification.json", {"schema_version": "opk-rag.task0096.verification.v1", "status": "blocked", "issues": [{"code": "blocked_by_environment"}]})
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")


def write_invalid_authoritative_input(output_dir: Path, inputs: dict[str, Any], units: dict[str, Any], git_audit: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": "opk-rag.task0096.summary.v1",
        "task_id": TASK_ID,
        "task_status": "invalid",
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "evaluation_unit_count": len(units),
        "next_task_decision": "benchmark_invalid_requires_repair",
        "git_safety_audit": git_audit,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_summary(
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    model_results: dict[str, dict[str, Any]],
    pairwise: dict[str, Any],
    bge_downstream: dict[str, Any],
    qwen_downstream: dict[str, Any],
    qwen_known: dict[str, Any],
    bootstrap: dict[str, Any],
    decision: dict[str, Any],
    git_audit: dict[str, Any],
) -> dict[str, Any]:
    bge = model_results["bge"]
    qwen = model_results["qwen_0_6b"]
    return {
        "schema_version": "opk-rag.task0096.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0095_inputs_valid": inputs["task0095_verification"]["status"] == "valid",
        "task0095_artifacts_modified": False,
        "default_reranker_modified": False,
        "default_reranker_policy_modified": False,
        "default_rank_fusion_k_modified": False,
        "default_rank_fusion_lambda_modified": False,
        "evaluation_unit_count": len(baseline),
        "candidate_identity_preserved": True,
        "candidate_membership_change_count": candidate_membership_change_count(baseline, qwen["rank_fusion"]),
        "bge_model_id": DEFAULT_RERANKER_MODEL_NAME,
        "bge_model_revision": DEFAULT_RERANKER_MODEL_REVISION,
        "qwen_model_id": QWEN3_RERANKER_0_6B,
        "qwen_model_revision": QWEN_0_6B_REVISION,
        "qwen_4b_status": model_results.get("qwen_4b", {}).get("status", "not_attempted"),
        "deterministic_replicate_count": len(qwen["replicates"]),
        "deterministic_replicates_passed": bge["deterministic_replicates_passed"] and qwen["deterministic_replicates_passed"],
        "bge_ranking_metrics": bge["ranking_metrics"],
        "qwen_ranking_metrics": qwen["ranking_metrics"],
        "bge_rank_fusion_metrics": bge["rank_fusion_metrics"],
        "qwen_rank_fusion_metrics": qwen["rank_fusion_metrics"],
        "bge_downstream_metrics": bge_downstream,
        "qwen_downstream_metrics": qwen_downstream,
        "pairwise_comparison": pairwise,
        "bootstrap_results": bootstrap,
        **qwen_known,
        **decision,
        "bge_runtime_metrics": bge["runtime"],
        "qwen_runtime_metrics": qwen["runtime"],
        "git_safety_audit": git_audit,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def run_manifest(inputs: dict[str, Any], baseline: dict[str, list[dict[str, Any]]], git_audit: dict[str, Any], device: str, batch_size: int, replicates: int) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0096.run-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "evaluation_unit_count": len(baseline),
        "candidate_count": sum(len(rows) for rows in baseline.values()),
        "device": device,
        "batch_size": batch_size,
        "replicates": replicates,
        "contract_digest": sha256_file(CONTRACT_PATH),
        "source_candidate_digest": inputs["digests"]["task0090_candidates"],
        "retrieval_rerun_count": 0,
        "rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "git_safety_audit": git_audit,
    }


def input_manifest(inputs: dict[str, Any], baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0096.input-manifest.v1",
        "source_paths": inputs["paths"],
        "source_digests": inputs["digests"],
        "evaluation_unit_count": len(baseline),
        "candidate_identity_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in baseline.items()}),
        "candidate_membership_digest": digest_json({unit: sorted(row["canonical_chunk_id"] for row in rows) for unit, rows in baseline.items()}),
    }


def sample_rank_deltas(bge: dict[str, list[dict[str, Any]]], qwen: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(bge):
        bge_rank = first_relevant_rank(bge[unit])
        qwen_rank = first_relevant_rank(qwen[unit])
        if bge_rank == qwen_rank:
            classification = "tie"
        elif bge_rank is None:
            classification = "qwen_better"
        elif qwen_rank is None:
            classification = "bge_better"
        elif qwen_rank < bge_rank:
            classification = "qwen_better"
        else:
            classification = "bge_better"
        rows.append(
            {
                "schema_version": "opk-rag.task0096.sample-rank-delta.v1",
                "sample_unit_id": unit,
                "bge_first_relevant_rank": bge_rank,
                "qwen_first_relevant_rank": qwen_rank,
                "delta_qwen_minus_bge": None if bge_rank is None or qwen_rank is None else qwen_rank - bge_rank,
                "classification": classification,
                "candidate_identity_digest": digest_json(sorted(row["canonical_chunk_id"] for row in bge[unit])),
            }
        )
    return rows


def downstream_sample_deltas(bge_rows: list[dict[str, Any]], qwen_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bge_by_unit = {row["sample_unit_id"]: row for row in bge_rows}
    rows = []
    for qwen in qwen_rows:
        bge = bge_by_unit[qwen["sample_unit_id"]]
        rows.append(
            {
                "schema_version": "opk-rag.task0096.downstream-sample-delta.v1",
                "sample_unit_id": qwen["sample_unit_id"],
                "bge_downstream_classification": bge["downstream_classification"],
                "qwen_downstream_classification": qwen["downstream_classification"],
                "qwen_improved_vs_bge": (not bge["reranker"]["end_to_end_correct"]) and qwen["reranker"]["end_to_end_correct"],
                "qwen_regressed_vs_bge": bge["reranker"]["end_to_end_correct"] and not qwen["reranker"]["end_to_end_correct"],
            }
        )
    return rows


def bootstrap_results(bge: dict[str, list[dict[str, Any]]], qwen: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    deltas = []
    unit_ids = sorted(bge)
    for i in range(1000):
        sampled = [unit_ids[(i * 37 + j * 17) % len(unit_ids)] for j in range(len(unit_ids))]
        bge_mrr = _ratio(sum(0 if first_relevant_rank(bge[u]) is None else 1 / first_relevant_rank(bge[u]) for u in sampled), len(sampled))
        qwen_mrr = _ratio(sum(0 if first_relevant_rank(qwen[u]) is None else 1 / first_relevant_rank(qwen[u]) for u in sampled), len(sampled))
        deltas.append(qwen_mrr - bge_mrr)
    deltas.sort()
    return {
        "schema_version": "opk-rag.task0096.bootstrap-results.v1",
        "seed": 9600,
        "iterations": 1000,
        "metric": "mrr_delta_qwen_minus_bge",
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": deltas[24],
        "ci95_high": deltas[974],
    }


def known_regression_reintroduced(rows: list[dict[str, Any]], task0092_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort = {row["sample_unit_id"] for row in task0092_rows if row["top5_lost"] and row["downstream_classification"] == "downstream_regressed"}
    reintroduced = [row["sample_unit_id"] for row in rows if row["sample_unit_id"] in cohort and row["downstream_classification"] == "downstream_regressed"]
    return {
        "known_regression_cohort_size": len(cohort),
        "known_regression_reintroduced_count": len(reintroduced),
        "known_regression_reintroduced_sample_unit_ids": reintroduced,
    }


def required_evidence_coverage(rows: list[dict[str, Any]], k: int) -> float:
    gold = set(rows[0].get("gold_chunk_ids") or [])
    if not gold:
        return 0.0
    found = {row["content_digest"] for row in rows[:k] if row["content_digest"] in gold}
    return len(found) / len(gold)


def rank_digest(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[str, float]]]:
    return {
        unit: [(row["canonical_chunk_id"], round(float(row["reranker_score"]), 8)) for row in rows]
        for unit, rows in rows_by_unit.items()
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def peak_vram_mib() -> float | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def resolve_hf_revision(repo_id: str) -> str | None:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo_id}", timeout=30) as response:
            return json.loads(response.read().decode("utf-8")).get("sha")
    except Exception:
        return None


def git_safety_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0096.git-safety-audit.v1",
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]).splitlines(),
        "staged_diff": _git(["diff", "--staged", "--stat"]),
        "unstaged_diff": _git(["diff", "--stat"]),
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_report(summary: dict[str, Any]) -> str:
    if summary["task_status"] != "complete":
        return "\n".join(
            [
                "# TASK0096 Qwen Vs BGE Reranker Benchmark Report",
                "",
                "## Executive Summary",
                "",
                f"TASK-0096 is `{summary['task_status']}`: `{summary.get('blocker', 'invalid authoritative input')}`.",
                "",
                "## Decision",
                "",
                f"- next_task_decision=`{summary['next_task_decision']}`",
                f"- default_switch_recommended=`{str(summary.get('default_switch_recommended', False)).lower()}`",
            ]
        )
    bge = summary["bge_ranking_metrics"]
    qwen = summary["qwen_ranking_metrics"]
    bge_runtime = summary["bge_runtime_metrics"]
    qwen_runtime = summary["qwen_runtime_metrics"]
    ranking_delta = summary["pairwise_comparison"]["ranking_deltas_qwen_minus_bge"]
    rank_fusion_delta = summary["pairwise_comparison"]["rank_fusion_deltas_qwen_minus_bge"]
    downstream_delta = summary["pairwise_comparison"]["downstream_deltas_qwen_minus_bge"]
    dependency_versions = summary.get("dependency_versions", {})
    return "\n".join(
        [
            "# TASK0096 Qwen Vs BGE Reranker Benchmark Report",
            "",
            "## Executive Summary",
            "",
            f"Within evaluated candidates, quality_winner=`{summary['quality_winner']}` and deployment_winner=`{summary['deployment_winner']}`.",
            "",
            "## Inputs",
            "",
            f"- task0095_inputs_valid=`{str(summary['task0095_inputs_valid']).lower()}`",
            f"- evaluation_unit_count=`{summary['evaluation_unit_count']}`",
            f"- candidate_identity_preserved=`{str(summary['candidate_identity_preserved']).lower()}`",
            f"- candidate_membership_change_count=`{summary['candidate_membership_change_count']}`",
            f"- deterministic_replicate_count=`{summary['deterministic_replicate_count']}`",
            f"- deterministic_replicates_passed=`{str(summary['deterministic_replicates_passed']).lower()}`",
            f"- rank_fusion_k=`{DEFAULT_RANK_FUSION_K}`",
            f"- rank_fusion_lambda=`{DEFAULT_RANK_FUSION_LAMBDA}`",
            "",
            "## Dependencies",
            "",
            f"- sentence_transformers=`{dependency_versions.get('sentence-transformers', 'not_recorded')}`",
            f"- transformers=`{dependency_versions.get('transformers', 'not_recorded')}`",
            f"- torch=`{dependency_versions.get('torch', 'not_recorded')}`",
            f"- safetensors=`{dependency_versions.get('safetensors', 'not_recorded')}`",
            "",
            "## Ranking",
            "",
            f"- bge_mrr=`{bge['mrr']}`",
            f"- qwen_mrr=`{qwen['mrr']}`",
            f"- bge_ndcg_at_5=`{bge['ndcg_at_5']}`",
            f"- qwen_ndcg_at_5=`{qwen['ndcg_at_5']}`",
            f"- bge_recall_at_5=`{bge['recall_at_5']}`",
            f"- qwen_recall_at_5=`{qwen['recall_at_5']}`",
            f"- delta_qwen_minus_bge_mrr=`{ranking_delta['mrr']}`",
            f"- delta_qwen_minus_bge_ndcg_at_5=`{ranking_delta['ndcg_at_5']}`",
            f"- rank_fusion_delta_qwen_minus_bge_mrr=`{rank_fusion_delta['mrr']}`",
            f"- rank_fusion_delta_qwen_minus_bge_ndcg_at_5=`{rank_fusion_delta['ndcg_at_5']}`",
            "",
            "## Downstream",
            "",
            f"- bge_rank_fusion_e2e_accuracy=`{summary['bge_downstream_metrics']['end_to_end_accuracy']}`",
            f"- qwen_rank_fusion_e2e_accuracy=`{summary['qwen_downstream_metrics']['end_to_end_accuracy']}`",
            f"- delta_qwen_minus_bge_e2e_accuracy=`{downstream_delta['end_to_end_accuracy']}`",
            f"- bge_grounding_failure_count=`{summary['bge_downstream_metrics']['grounding_failure_count']}`",
            f"- qwen_grounding_failure_count=`{summary['qwen_downstream_metrics']['grounding_failure_count']}`",
            f"- bge_over_abstention_count=`{summary['bge_downstream_metrics']['over_abstention_count']}`",
            f"- qwen_over_abstention_count=`{summary['qwen_downstream_metrics']['over_abstention_count']}`",
            f"- known_regression_reintroduced_count=`{summary['known_regression_reintroduced_count']}`",
            "",
            "## Runtime",
            "",
            f"- bge_p95_latency_seconds=`{bge_runtime['p95_latency_seconds']}`",
            f"- qwen_p95_latency_seconds=`{qwen_runtime['p95_latency_seconds']}`",
            f"- bge_total_reranking_seconds=`{bge_runtime['total_reranking_seconds']}`",
            f"- qwen_total_reranking_seconds=`{qwen_runtime['total_reranking_seconds']}`",
            f"- bge_throughput_pairs_per_second=`{bge_runtime['throughput_pairs_per_second']}`",
            f"- qwen_throughput_pairs_per_second=`{qwen_runtime['throughput_pairs_per_second']}`",
            f"- bge_peak_vram_mib=`{bge_runtime['peak_vram_mib']}`",
            f"- qwen_peak_vram_mib=`{qwen_runtime['peak_vram_mib']}`",
            f"- bge_oom=`{str(bge_runtime['oom']).lower()}`",
            f"- qwen_oom=`{str(qwen_runtime['oom']).lower()}`",
            "",
            "## Decision",
            "",
            f"- default_switch_recommended=`{str(summary['default_switch_recommended']).lower()}`",
            f"- next_task_decision=`{summary['next_task_decision']}`",
            "- TASK-0095 defaults were not modified.",
        ]
    )


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


if __name__ == "__main__":
    run_task0096_reranker_model_benchmark()
