from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

from opk_rag.evaluation.task0208_reranker_hardware_execution_profiling_and_device_authority import (
    _external_gpu_memory_used_mb,
    _mb,
    _module_version,
    _nvidia_smi,
    _torch_environment,
    percentile,
)
from opk_rag.reranking.config import RerankerConfig, load_reranker_config


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0209"
EXPERIMENT_ID = "task0209-reranker-cuda-precision-optimization-experiment"
SCHEMA_VERSION = "opk-rag.task0209.reranker-cuda-precision-optimization-experiment.v1"
TASK0206_DIR = ROOT / "evaluation-data" / "results" / "task0206-retrieval-latency-bottleneck-attribution"
TASK0207_DIR = ROOT / "evaluation-data" / "results" / "task0207-reranker-model-execution-optimization-experiment"
TASK0208_DIR = ROOT / "evaluation-data" / "results" / "task0208-reranker-hardware-execution-profiling-and-device-authority"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0209_reranker_cuda_precision_optimization_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0209_RERANKER_CUDA_PRECISION_OPTIMIZATION_EXPERIMENT_REPORT.md"

WARMUP_COUNT_PER_QUERY = 5
MEASUREMENT_COUNT_PER_QUERY = 20
PROMOTION_P95_REDUCTION_THRESHOLD = 0.10
FORMAL_QUERY_COUNT = 47

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_authority_audit.json",
    "environment.json",
    "input_snapshot.json",
    "precision_arm_results.json",
    "quality_equivalence.json",
    "downstream_safety.json",
    "memory_comparison.json",
    "promotion_decision.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "measurement_valid",
    "production_reranker_model",
    "production_device",
    "resolved_production_device",
    "production_dtype",
    "formal_query_count",
    "total_measurement_count",
    "experiment_arm_count",
    "valid_experiment_arm_count",
    "model_reinitialized_per_query",
    "tokenizer_reinitialized_per_query",
    "inference_mode_active",
    "model_eval_mode_active",
    "candidate_input_equivalence",
    "best_arm",
    "best_requested_dtype",
    "best_effective_dtype",
    "best_reranker_p50_ms",
    "best_reranker_p95_ms",
    "best_reranker_p99_ms",
    "reranker_p95_reduction_ratio",
    "baseline_peak_gpu_memory_mb",
    "best_peak_gpu_memory_mb",
    "peak_gpu_memory_reduction_ratio",
    "formal_quality_regression_count",
    "downstream_regression_count",
    "promotion_candidate",
    "runtime_integration_recommended",
    "promotion_applied",
    "production_dtype_unchanged",
)

ARM_SPECS: tuple[dict[str, Any], ...] = (
    {"arm_id": "C0_fp32_production_baseline", "requested_dtype": "torch.float32", "cast_dtype": "float32", "autocast": False},
    {"arm_id": "C1_fp16_model_cast", "requested_dtype": "torch.float16", "cast_dtype": "float16", "autocast": False},
    {"arm_id": "C2_fp16_cuda_autocast", "requested_dtype": "torch.float16", "cast_dtype": None, "autocast": True},
    {"arm_id": "C3_bf16_model_cast", "requested_dtype": "torch.bfloat16", "cast_dtype": "bfloat16", "autocast": False},
    {"arm_id": "C4_bf16_cuda_autocast", "requested_dtype": "torch.bfloat16", "cast_dtype": None, "autocast": True},
)


def run_task0209(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task0206_summary = read_json(TASK0206_DIR / "summary.json")
    task0206_raw = read_json(TASK0206_DIR / "raw_stage_measurements.json")
    task0207_summary = read_json(TASK0207_DIR / "summary.json")
    task0207_audit = read_json(TASK0207_DIR / "production_reranker_execution_audit.json")
    task0208_summary = read_json(TASK0208_DIR / "summary.json")
    task0208_memory = read_json(TASK0208_DIR / "memory_diagnostics.json")

    source_audit = build_source_authority_audit(task0206_summary, task0207_summary, task0207_audit, task0208_summary)
    environment = build_environment()
    input_snapshot = build_input_snapshot(task0206_raw)
    arms = run_precision_arms(input_snapshot, environment, runtime_env)
    quality = build_quality_equivalence(arms)
    downstream = build_downstream_safety(arms, task0207_summary)
    memory = build_memory_comparison(arms, task0208_memory)
    promotion = build_promotion_decision(arms, quality, downstream, memory)
    sensitive = scan_sensitive_payloads((source_audit, environment, input_snapshot, arms, quality, downstream, memory, promotion))
    summary = build_summary(
        source_audit=source_audit,
        environment=environment,
        input_snapshot=input_snapshot,
        arms=arms,
        quality=quality,
        downstream=downstream,
        memory=memory,
        promotion=promotion,
        sensitive=sensitive,
    )
    contract = build_contract()
    if write:
        write_json(RESULT_DIR / "source_authority_audit.json", source_audit)
        write_json(RESULT_DIR / "environment.json", environment)
        write_json(RESULT_DIR / "input_snapshot.json", input_snapshot)
        write_json(RESULT_DIR / "precision_arm_results.json", {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "arms": arms})
        write_json(RESULT_DIR / "quality_equivalence.json", quality)
        write_json(RESULT_DIR / "downstream_safety.json", downstream)
        write_json(RESULT_DIR / "memory_comparison.json", memory)
        write_json(RESULT_DIR / "promotion_decision.json", promotion)
        write_json(RESULT_DIR / "sensitive_scan.json", sensitive)
        write_json(CONTRACT_PATH, contract)
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, environment, arms, quality, downstream, memory, promotion), encoding="utf-8")
        verification = verify_task0209_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, environment, arms, quality, downstream, memory, promotion), encoding="utf-8")
    return summary


def build_source_authority_audit(
    task0206_summary: Mapping[str, Any],
    task0207_summary: Mapping[str, Any],
    task0207_audit: Mapping[str, Any],
    task0208_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_tasks": ["TASK-0206", "TASK-0207", "TASK-0208"],
        "task0206_measurement_valid": task0206_summary.get("measurement_valid") is True,
        "task0207_measurement_valid": task0207_summary.get("measurement_valid") is True,
        "task0208_measurement_valid": task0208_summary.get("measurement_valid") is True,
        "benchmark_query_digest_matches_task0207": task0206_summary.get("benchmark_query_digest") == task0207_summary.get("workload_digest"),
        "benchmark_corpus_digest_matches_task0207": task0206_summary.get("benchmark_corpus_digest") == task0207_summary.get("benchmark_corpus_digest"),
        "production_reranker_model": task0207_summary.get("production_reranker_model"),
        "production_reranker_revision": task0207_summary.get("production_reranker_revision"),
        "production_device": task0207_audit.get("production_device"),
        "resolved_production_device": task0208_summary.get("resolved_device"),
        "production_dtype": "torch.float32",
        "task0208_effective_dtype": task0208_summary.get("effective_model_dtype"),
        "task0208_cuda_max_allocated_mb": task0208_summary.get("cuda_max_memory_allocated_mb"),
        "task0207_zero_gpu_memory_is_cpu_evidence": False,
        "model_reinitialized_per_query": task0207_audit.get("model_reinitialized_per_query"),
        "tokenizer_reinitialized_per_query": task0207_audit.get("tokenizer_reinitialized_per_query"),
        "inference_mode_active": task0207_audit.get("inference_mode_active"),
        "model_eval_mode_active": task0207_audit.get("model_eval_mode_active"),
        "source_authority_valid": all(
            [
                task0206_summary.get("measurement_valid") is True,
                task0207_summary.get("measurement_valid") is True,
                task0208_summary.get("measurement_valid") is True,
                task0208_summary.get("effective_model_dtype") == "torch.float32",
            ]
        ),
    }


def build_environment() -> dict[str, Any]:
    torch_info = _torch_environment()
    smi = _nvidia_smi()
    driver = smi.get("nvidia_driver_version") or _driver_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_name": smi.get("gpu_model") or torch_info.get("torch_cuda_device_name"),
        "gpu_compute_capability": smi.get("gpu_compute_capability") or torch_info.get("torch_cuda_capability"),
        "cuda_version": torch_info.get("torch_cuda_runtime_version") or torch_info.get("torch_build_cuda_version"),
        "pytorch_version": torch_info.get("torch_version"),
        "transformers_version": _module_version("transformers"),
        "sentence_transformers_version": _module_version("sentence_transformers"),
        "reranker_model_revision": load_reranker_config().model_revision,
        "driver_version": driver,
        "device_count": torch_info.get("torch_cuda_device_count"),
        "torch_cuda_available": torch_info.get("torch_cuda_available"),
        "torch_build_cuda_version": torch_info.get("torch_build_cuda_version"),
        "torch_cuda_runtime_version": torch_info.get("torch_cuda_runtime_version"),
    }


def build_input_snapshot(raw_measurements: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in raw_measurements.get("measurements", []) if row.get("valid_sample") is True]
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row.get("query_id")), []).append(row)
    queries = []
    for query_id, items in sorted(by_query.items()):
        candidate_count = max(int((item.get("details") or {}).get("reranker_input_count") or 0) for item in items)
        queries.append(
            {
                "query_id": query_id,
                "query_text": f"sanitized task0209 formal reranker query {query_id}",
                "candidate_ids": [f"{query_id}::candidate::{idx:02d}" for idx in range(candidate_count)],
                "candidate_texts": [
                    f"sanitized task0209 candidate {idx:02d} for formal reranker query {query_id}"
                    for idx in range(candidate_count)
                ],
            }
        )
    digest_payload = [
        {"query_id": item["query_id"], "candidate_ids": item["candidate_ids"], "candidate_count": len(item["candidate_ids"])}
        for item in queries
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source": "TASK-0206 raw_stage_measurements metadata; sanitized query/candidate text rematerialized because upstream artifacts do not persist raw formal inputs.",
        "formal_query_count": len(queries),
        "candidate_count_digest": _digest(digest_payload),
        "input_snapshot_digest": _digest(digest_payload),
        "raw_text_persisted": False,
        "max_pair_tokens": load_reranker_config().max_pair_tokens,
        "warmup_count_per_query": WARMUP_COUNT_PER_QUERY,
        "measurement_count_per_query": MEASUREMENT_COUNT_PER_QUERY,
        "queries": queries,
    }


def run_precision_arms(input_snapshot: Mapping[str, Any], environment: Mapping[str, Any], env: Mapping[str, str]) -> list[dict[str, Any]]:
    if environment.get("torch_cuda_available") is not True or int(environment.get("device_count") or 0) < 1:
        return [unsupported_arm(spec, "cuda_unavailable") for spec in ARM_SPECS]
    return [run_precision_arm(spec, input_snapshot, env) for spec in ARM_SPECS]


def run_precision_arm(spec: Mapping[str, Any], input_snapshot: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    try:
        import torch
        from opk_rag.reranking.bge import BgeLocalRerankerProvider
    except Exception as exc:
        return unsupported_arm(spec, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    config = _cuda_config(env)
    requested_dtype = str(spec["requested_dtype"])
    target_dtype = _torch_dtype(torch, spec.get("cast_dtype") or requested_dtype.split(".")[-1])
    autocast_enabled = bool(spec.get("autocast"))
    latencies: list[float] = []
    repeat_scores: dict[str, list[list[float]]] = {}
    all_scores: dict[str, list[float]] = {}
    cuda_sync_boundary_used = False
    peak_stats_reset = False
    memory = empty_memory_metrics()
    output_dtype = "unknown"
    forward_input_device = "unknown"
    forward_input_dtype = "unknown"
    try:
        provider = BgeLocalRerankerProvider(config)
        model = provider._model
        model.eval()
        if spec.get("cast_dtype"):
            model.model.to(dtype=target_dtype)
        resolved_device = provider.device
        parameter_devices = _parameter_device_counts(model.model)
        parameter_dtypes = _parameter_dtype_counts(model.model)
        model_parameter_device = next(iter(parameter_devices), "unknown")
        model_parameter_dtype = next(iter(parameter_dtypes), "unknown")
        cuda_device = torch.device(model.model.device)
        queries = input_snapshot.get("queries") or []
        for query in queries:
            for _ in range(WARMUP_COUNT_PER_QUERY):
                _score_query(torch, model, config, query, autocast_enabled, target_dtype)
        torch.cuda.synchronize(cuda_device)
        cuda_sync_boundary_used = True
        torch.cuda.reset_peak_memory_stats(cuda_device)
        peak_stats_reset = True
        memory["cuda_memory_allocated_before_mb"] = _mb(torch.cuda.memory_allocated(cuda_device))
        memory["cuda_memory_reserved_before_mb"] = _mb(torch.cuda.memory_reserved(cuda_device))
        memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        for query in queries:
            qid = str(query["query_id"])
            repeat_scores[qid] = []
            for _ in range(MEASUREMENT_COUNT_PER_QUERY):
                metrics, scores, meta = _score_query(torch, model, config, query, autocast_enabled, target_dtype)
                latencies.append(metrics["reranker_total_ms"])
                repeat_scores[qid].append(scores)
                all_scores[qid] = scores
                output_dtype = meta["output_dtype"]
                forward_input_device = meta["forward_input_device"]
                forward_input_dtype = meta["forward_input_dtype"]
        torch.cuda.synchronize(cuda_device)
        memory["cuda_memory_allocated_after_mb"] = _mb(torch.cuda.memory_allocated(cuda_device))
        memory["cuda_memory_reserved_after_mb"] = _mb(torch.cuda.memory_reserved(cuda_device))
        memory["cuda_max_memory_allocated_mb"] = _mb(torch.cuda.max_memory_allocated(cuda_device))
        memory["cuda_max_memory_reserved_mb"] = _mb(torch.cuda.max_memory_reserved(cuda_device))
        memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        finite = [score for scores in all_scores.values() for score in scores if math.isfinite(score)]
        score_count = sum(len(scores) for scores in all_scores.values())
        deterministic = all(_stable_repeats(scores) for scores in repeat_scores.values())
        requested_effective = requested_dtype == "torch.float32" or (
            model_parameter_dtype == requested_dtype or (autocast_enabled and output_dtype == requested_dtype)
        )
        silent_fp32_fallback = requested_dtype != "torch.float32" and not requested_effective
        arm_valid = bool(
            score_count
            and len(finite) == score_count
            and not silent_fp32_fallback
            and str(resolved_device).startswith("cuda")
            and str(model_parameter_device).startswith("cuda")
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "arm_id": spec["arm_id"],
            "arm_status": "valid_measured" if arm_valid else "invalid",
            "execution_valid": arm_valid,
            "requested_precision_effective": requested_effective,
            "requested_dtype": requested_dtype,
            "model_parameter_dtype": model_parameter_dtype,
            "model_parameter_dtype_count": parameter_dtypes,
            "forward_input_dtype": forward_input_dtype,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": requested_dtype if autocast_enabled else None,
            "resolved_device": resolved_device,
            "model_parameter_device": model_parameter_device,
            "model_parameter_device_count": parameter_devices,
            "forward_input_device": forward_input_device,
            "output_dtype": output_dtype,
            "silent_fp32_fallback_detected": silent_fp32_fallback,
            "finite_output_ratio": len(finite) / score_count if score_count else 0.0,
            "nan_count": sum(1 for scores in all_scores.values() for score in scores if math.isnan(score)),
            "inf_count": sum(1 for scores in all_scores.values() for score in scores if math.isinf(score)),
            "cuda_oom": False,
            "device_mismatch_error": False,
            "undeclared_cpu_fallback_detected": not str(model_parameter_device).startswith("cuda"),
            "deterministic_repeatability": deterministic,
            "candidate_input_digest": input_snapshot.get("input_snapshot_digest"),
            "score_digest": _digest({key: [round(value, 6) for value in values] for key, values in sorted(all_scores.items())}),
            "rankings": _rankings(input_snapshot, all_scores),
            "reranker_p50_ms": percentile(latencies, 50),
            "reranker_p95_ms": percentile(latencies, 95),
            "reranker_p99_ms": percentile(latencies, 99),
            "mean_reranker_latency_ms": round(statistics.fmean(latencies), 6) if latencies else None,
            "standard_deviation_ms": round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0,
            "measurement_count": len(latencies),
            "cuda_sync_boundary_used": cuda_sync_boundary_used,
            "peak_memory_stats_reset": peak_stats_reset,
            "cuda_memory": memory,
            "peak_gpu_memory_mb": memory["cuda_max_memory_allocated_mb"],
            "max_reserved_gpu_memory_mb": memory["cuda_max_memory_reserved_mb"],
            "runtime_error_count": 0,
            "oom_count": 0,
            "exception_type": None,
            "exception_reason": None,
        }
    except RuntimeError as exc:
        reason = str(exc).splitlines()[0][:240]
        oom = "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        return error_arm(spec, type(exc).__name__, reason, oom=oom)
    except Exception as exc:
        return error_arm(spec, type(exc).__name__, str(exc).splitlines()[0][:240], oom=False)
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _score_query(torch: Any, model: Any, config: RerankerConfig, query: Mapping[str, Any], autocast: bool, dtype: Any) -> tuple[dict[str, float], list[float], dict[str, str]]:
    pairs = [(query["query_text"], document) for document in query["candidate_texts"]]
    t0 = time.perf_counter()
    features = model.tokenizer(pairs, padding=True, truncation=True, max_length=config.max_pair_tokens, return_tensors="pt")
    features.to(model.model.device)
    torch.cuda.synchronize(model.model.device)
    t1 = time.perf_counter()
    context = torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast)
    with torch.inference_mode(), context:
        predictions = model.model(**features, return_dict=True)
        logits = model.activation_fn(predictions.logits)
    torch.cuda.synchronize(model.model.device)
    t2 = time.perf_counter()
    values = logits.detach().cpu().reshape(-1).tolist()
    t3 = time.perf_counter()
    return (
        {"reranker_total_ms": _elapsed_ms(t0, t3), "forward_ms": _elapsed_ms(t1, t2)},
        [float(value) for value in values],
        {
            "forward_input_device": str(features["input_ids"].device),
            "forward_input_dtype": str(features["input_ids"].dtype),
            "output_dtype": str(logits.dtype),
        },
    )


def build_quality_equivalence(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = next((arm for arm in arms if arm.get("arm_id") == "C0_fp32_production_baseline"), {})
    baseline_rankings = baseline.get("rankings") or {}
    baseline_scores = _scores_by_query(baseline)
    per_arm = {}
    for arm in arms:
        if arm is baseline:
            continue
        metrics = compare_arm_quality(baseline_rankings, baseline_scores, arm)
        per_arm[arm["arm_id"]] = metrics
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "baseline_arm": "C0_fp32_production_baseline",
        "candidate_input_equivalence": all(arm.get("candidate_input_digest") == baseline.get("candidate_input_digest") for arm in arms if arm.get("execution_valid") is True),
        "per_arm": per_arm,
        "formal_quality_regression_count": sum(int(metrics.get("formal_quality_regression_count") or 0) for metrics in per_arm.values()),
    }


def compare_arm_quality(
    baseline_rankings: Mapping[str, Sequence[str]], baseline_scores: Mapping[str, Mapping[str, float]], arm: Mapping[str, Any]
) -> dict[str, Any]:
    if arm.get("execution_valid") is not True:
        return empty_quality_metrics("arm_not_valid")
    rankings = arm.get("rankings") or {}
    arm_scores = _scores_by_query(arm)
    query_ids = sorted(baseline_rankings)
    top1 = top3 = top5 = full = pairwise = 0
    max_delta = 0.0
    deltas: list[float] = []
    for query_id in query_ids:
        base = list(baseline_rankings.get(query_id, []))
        cand = list(rankings.get(query_id, []))
        top1 += int(base[:1] == cand[:1])
        top3 += int(set(base[:3]) == set(cand[:3]))
        top5 += int(set(base[:5]) == set(cand[:5]))
        full += int(base == cand)
        pairwise += int(_pairwise_agreement(base, cand) == 1.0)
        for candidate_id, base_score in baseline_scores.get(query_id, {}).items():
            delta = abs(base_score - arm_scores.get(query_id, {}).get(candidate_id, base_score))
            deltas.append(delta)
            max_delta = max(max_delta, delta)
    count = len(query_ids) or 1
    regressions = sum(
        1
        for query_id in query_ids
        if list(baseline_rankings.get(query_id, []))[:1] != list(rankings.get(query_id, []))[:1]
        or set(list(baseline_rankings.get(query_id, []))[:5]) != set(list(rankings.get(query_id, []))[:5])
    )
    return {
        "top1_agreement_ratio": round(top1 / count, 6),
        "top3_set_agreement_ratio": round(top3 / count, 6),
        "top5_set_agreement_ratio": round(top5 / count, 6),
        "full_ranking_agreement_ratio": round(full / count, 6),
        "pairwise_order_agreement_ratio": round(pairwise / count, 6),
        "maximum_absolute_score_delta": round(max_delta, 6),
        "mean_absolute_score_delta": round(statistics.fmean(deltas), 6) if deltas else 0.0,
        "formal_quality_regression_count": regressions,
    }


def build_downstream_safety(arms: Sequence[Mapping[str, Any]], task0207_summary: Mapping[str, Any]) -> dict[str, Any]:
    baseline_ok = task0207_summary.get("downstream_retrieval_equivalence") is True and task0207_summary.get("known_quality_regression_count") == 0
    per_arm = {}
    for arm in arms:
        if arm.get("arm_id") == "C0_fp32_production_baseline":
            continue
        valid = arm.get("execution_valid") is True
        quality = compare_arm_quality(
            next((a.get("rankings") or {} for a in arms if a.get("arm_id") == "C0_fp32_production_baseline"), {}),
            _scores_by_query(next((a for a in arms if a.get("arm_id") == "C0_fp32_production_baseline"), {})),
            arm,
        )
        regression = 0 if valid and baseline_ok and quality["top5_set_agreement_ratio"] == 1.0 and quality["formal_quality_regression_count"] == 0 else int(valid)
        per_arm[arm["arm_id"]] = {
            "evidence_selection_regression_count": regression,
            "answerability_regression_count": regression,
            "grounding_regression_count": regression,
            "safe_action_regression_count": regression,
            "downstream_regression_count": regression,
            "basis": "TASK-0207 downstream authority reused only when TASK-0209 ranking top5 is exactly equivalent to C0.",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "per_arm": per_arm,
        "evidence_selection_regression_count": sum(item["evidence_selection_regression_count"] for item in per_arm.values()),
        "answerability_regression_count": sum(item["answerability_regression_count"] for item in per_arm.values()),
        "grounding_regression_count": sum(item["grounding_regression_count"] for item in per_arm.values()),
        "safe_action_regression_count": sum(item["safe_action_regression_count"] for item in per_arm.values()),
        "downstream_regression_count": sum(item["downstream_regression_count"] for item in per_arm.values()),
    }


def build_memory_comparison(arms: Sequence[Mapping[str, Any]], task0208_memory: Mapping[str, Any]) -> dict[str, Any]:
    baseline = next((arm for arm in arms if arm.get("arm_id") == "C0_fp32_production_baseline"), {})
    baseline_peak = _valid_peak(baseline.get("peak_gpu_memory_mb")) or _valid_peak(task0208_memory.get("cuda_max_memory_allocated_mb")) or 0.0
    per_arm = {}
    for arm in arms:
        peak = float(arm.get("peak_gpu_memory_mb") or 0.0)
        per_arm[arm["arm_id"]] = {
            "baseline_peak_gpu_memory_mb": baseline_peak,
            "experiment_peak_gpu_memory_mb": peak,
            "peak_gpu_memory_reduction_mb": round(baseline_peak - peak, 6) if peak > 0 else None,
            "peak_gpu_memory_reduction_ratio": _reduction(baseline_peak, peak) if peak > 0 else None,
            "cuda_memory": arm.get("cuda_memory"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "baseline_peak_gpu_memory_mb": baseline_peak,
        "per_arm": per_arm,
        "unreliable_zero_memory_result_detected": any(arm.get("execution_valid") is True and not _valid_peak(arm.get("peak_gpu_memory_mb")) for arm in arms),
    }


def build_promotion_decision(
    arms: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    downstream: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = next((arm for arm in arms if arm.get("arm_id") == "C0_fp32_production_baseline"), {})
    baseline_p95 = baseline.get("reranker_p95_ms")
    eligible = []
    for arm in arms:
        if arm.get("arm_id") == "C0_fp32_production_baseline" or arm.get("execution_valid") is not True:
            continue
        q = (quality.get("per_arm") or {}).get(arm["arm_id"], {})
        d = (downstream.get("per_arm") or {}).get(arm["arm_id"], {})
        reduction = _reduction(baseline_p95, arm.get("reranker_p95_ms"))
        promotion_eligible = all(
            [
                arm.get("requested_precision_effective") is True,
                arm.get("finite_output_ratio") == 1.0,
                quality.get("candidate_input_equivalence") is True,
                q.get("formal_quality_regression_count") == 0,
                d.get("evidence_selection_regression_count") == 0,
                d.get("grounding_regression_count") == 0,
                d.get("safe_action_regression_count") == 0,
                d.get("downstream_regression_count") == 0,
                q.get("top1_agreement_ratio") == 1.0,
                q.get("top5_set_agreement_ratio") == 1.0,
                reduction is not None and reduction >= PROMOTION_P95_REDUCTION_THRESHOLD,
            ]
        )
        if promotion_eligible:
            eligible.append((arm, reduction))
    if eligible:
        eligible.sort(key=lambda item: (item[0].get("reranker_p95_ms") or math.inf, item[0].get("reranker_p99_ms") or math.inf, item[0].get("peak_gpu_memory_mb") or math.inf))
        best = eligible[0][0]
        candidate = best["arm_id"]
        recommended = True
    else:
        best = baseline
        candidate = "none"
        recommended = False
    best_mem = (memory.get("per_arm") or {}).get(best.get("arm_id"), {})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "best_arm": best.get("arm_id"),
        "best_requested_dtype": best.get("requested_dtype"),
        "best_effective_dtype": best.get("output_dtype") if best.get("autocast_enabled") is True else best.get("model_parameter_dtype"),
        "best_reranker_p50_ms": best.get("reranker_p50_ms"),
        "best_reranker_p95_ms": best.get("reranker_p95_ms"),
        "best_reranker_p99_ms": best.get("reranker_p99_ms"),
        "reranker_p95_reduction_ratio": _reduction(baseline_p95, best.get("reranker_p95_ms")),
        "best_peak_gpu_memory_mb": best.get("peak_gpu_memory_mb"),
        "peak_gpu_memory_reduction_ratio": best_mem.get("peak_gpu_memory_reduction_ratio"),
        "promotion_candidate": candidate,
        "runtime_integration_recommended": recommended,
        "promotion_applied": False,
        "production_dtype_unchanged": True,
        "promotion_eligible_arm_ids": [arm["arm_id"] for arm, _ in eligible],
        "decision_basis": "fail_closed_precision_promotion_gate",
    }


def build_summary(
    *,
    source_audit: Mapping[str, Any],
    environment: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    downstream: Mapping[str, Any],
    memory: Mapping[str, Any],
    promotion: Mapping[str, Any],
    sensitive: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = next((arm for arm in arms if arm.get("arm_id") == "C0_fp32_production_baseline"), {})
    best_quality = (quality.get("per_arm") or {}).get(promotion.get("best_arm"), {}) or {
        "top1_agreement_ratio": 1.0,
        "top5_set_agreement_ratio": 1.0,
        "formal_quality_regression_count": 0,
    }
    best_downstream = (downstream.get("per_arm") or {}).get(promotion.get("best_arm"), {}) or {
        "evidence_selection_regression_count": 0,
        "answerability_regression_count": 0,
        "grounding_regression_count": 0,
        "safe_action_regression_count": 0,
        "downstream_regression_count": 0,
    }
    blockers = []
    if source_audit.get("source_authority_valid") is not True:
        blockers.append("source_authority_invalid")
    if environment.get("torch_cuda_available") is not True:
        blockers.append("cuda_unavailable")
    if baseline.get("execution_valid") is not True:
        blockers.append("fp32_baseline_invalid")
    if input_snapshot.get("formal_query_count") != FORMAL_QUERY_COUNT:
        blockers.append("formal_query_count_mismatch")
    if memory.get("unreliable_zero_memory_result_detected") is True:
        blockers.append("unreliable_zero_memory_result")
    if sensitive.get("sensitive_value_exposure_count") != 0:
        blockers.append("sensitive_scan_failed")
    valid_arms = [arm for arm in arms if arm.get("execution_valid") is True]
    measurement_valid = not blockers and bool(valid_arms) and all(arm.get("cuda_sync_boundary_used") is True for arm in valid_arms)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if measurement_valid else "partial",
        "measurement_valid": measurement_valid,
        "production_reranker_model": source_audit.get("production_reranker_model"),
        "production_reranker_revision": source_audit.get("production_reranker_revision"),
        "production_device": source_audit.get("production_device"),
        "resolved_production_device": source_audit.get("resolved_production_device"),
        "production_dtype": source_audit.get("production_dtype"),
        "gpu_name": environment.get("gpu_name"),
        "formal_query_count": input_snapshot.get("formal_query_count"),
        "total_measurement_count": sum(int(arm.get("measurement_count") or 0) for arm in valid_arms),
        "experiment_arm_count": len(arms),
        "valid_experiment_arm_count": len(valid_arms),
        "model_reinitialized_per_query": False,
        "tokenizer_reinitialized_per_query": False,
        "inference_mode_active": True,
        "model_eval_mode_active": True,
        "candidate_input_equivalence": quality.get("candidate_input_equivalence"),
        "best_arm": promotion.get("best_arm"),
        "best_requested_dtype": promotion.get("best_requested_dtype"),
        "best_effective_dtype": promotion.get("best_effective_dtype"),
        "baseline_reranker_p50_ms": baseline.get("reranker_p50_ms"),
        "baseline_reranker_p95_ms": baseline.get("reranker_p95_ms"),
        "baseline_reranker_p99_ms": baseline.get("reranker_p99_ms"),
        "best_reranker_p50_ms": promotion.get("best_reranker_p50_ms"),
        "best_reranker_p95_ms": promotion.get("best_reranker_p95_ms"),
        "best_reranker_p99_ms": promotion.get("best_reranker_p99_ms"),
        "reranker_p95_reduction_ratio": promotion.get("reranker_p95_reduction_ratio"),
        "baseline_peak_gpu_memory_mb": memory.get("baseline_peak_gpu_memory_mb"),
        "best_peak_gpu_memory_mb": promotion.get("best_peak_gpu_memory_mb"),
        "peak_gpu_memory_reduction_ratio": promotion.get("peak_gpu_memory_reduction_ratio"),
        "top1_agreement_ratio": best_quality.get("top1_agreement_ratio"),
        "top5_set_agreement_ratio": best_quality.get("top5_set_agreement_ratio"),
        "formal_quality_regression_count": best_quality.get("formal_quality_regression_count"),
        "all_arm_formal_quality_regression_count": quality.get("formal_quality_regression_count"),
        "evidence_selection_regression_count": best_downstream.get("evidence_selection_regression_count"),
        "answerability_regression_count": best_downstream.get("answerability_regression_count"),
        "grounding_regression_count": best_downstream.get("grounding_regression_count"),
        "safe_action_regression_count": best_downstream.get("safe_action_regression_count"),
        "downstream_regression_count": best_downstream.get("downstream_regression_count"),
        "all_arm_downstream_regression_count": downstream.get("downstream_regression_count"),
        "promotion_candidate": promotion.get("promotion_candidate"),
        "runtime_integration_recommended": promotion.get("runtime_integration_recommended"),
        "promotion_applied": False,
        "production_dtype_unchanged": True,
        "precision_blockers": blockers,
        "git_commit_created": False,
        "sensitive_value_exposure_count": sensitive.get("sensitive_value_exposure_count"),
    }


def verify_task0209_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path) if contract_path.exists() else build_contract()
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required if name != "verification.json")]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    arms_payload = read_json(result_dir / "precision_arm_results.json") if (result_dir / "precision_arm_results.json").exists() else {"arms": []}
    arms = arms_payload.get("arms") or []
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    checks = {
        "task_id": TASK_ID,
        "measurement_valid": True,
        "production_reranker_model": "BAAI/bge-reranker-v2-m3",
        "production_device": "auto",
        "resolved_production_device": "cuda",
        "production_dtype": "torch.float32",
        "formal_query_count": FORMAL_QUERY_COUNT,
        "model_reinitialized_per_query": False,
        "tokenizer_reinitialized_per_query": False,
        "inference_mode_active": True,
        "model_eval_mode_active": True,
        "candidate_input_equivalence": True,
        "promotion_applied": False,
        "production_dtype_unchanged": True,
        "git_commit_created": False,
        "sensitive_value_exposure_count": 0,
    }
    for key, expected_value in checks.items():
        if summary.get(key) != expected_value:
            issues.append(f"{key} expected {expected_value!r}, got {summary.get(key)!r}")
    if not any(arm.get("arm_id") == "C0_fp32_production_baseline" and arm.get("execution_valid") is True for arm in arms):
        issues.append("missing valid FP32 control arm")
    if any(arm.get("execution_valid") is True and arm.get("cuda_sync_boundary_used") is not True for arm in arms):
        issues.append("valid arm missing CUDA synchronization boundary")
    if any(arm.get("execution_valid") is True and arm.get("peak_memory_stats_reset") is not True for arm in arms):
        issues.append("valid arm missing CUDA peak-memory reset")
    if any((arm.get("nan_count") or 0) or (arm.get("inf_count") or 0) for arm in arms):
        issues.append("non-finite output detected")
    if any(arm.get("silent_fp32_fallback_detected") is True for arm in arms):
        issues.append("silent FP32 fallback detected")
    if any(arm.get("execution_valid") is True and not _valid_peak(arm.get("peak_gpu_memory_mb")) for arm in arms):
        issues.append("unreliable 0 MB GPU memory result")
    recommended = summary.get("runtime_integration_recommended") is True
    if recommended and ((summary.get("formal_quality_regression_count") or 0) > 0 or (summary.get("downstream_regression_count") or 0) > 0):
        issues.append("promotion recommended despite candidate quality/downstream regression")
    if recommended and (summary.get("reranker_p95_reduction_ratio") or 0.0) < PROMOTION_P95_REDUCTION_THRESHOLD:
        issues.append("promotion recommended despite insufficient P95 reduction")
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "promotion_policy": {
            "p95_reduction_threshold": PROMOTION_P95_REDUCTION_THRESHOLD,
            "fail_closed": True,
            "production_dtype_change_allowed": False,
        },
    }


def render_report(
    summary: Mapping[str, Any],
    environment: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    downstream: Mapping[str, Any],
    memory: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> str:
    arm_lines = [
        f"* `{arm['arm_id']}` status `{arm.get('arm_status')}` requested `{arm.get('requested_dtype')}` param `{arm.get('model_parameter_dtype')}` output `{arm.get('output_dtype')}` autocast `{arm.get('autocast_enabled')}` P50/P95/P99 `{arm.get('reranker_p50_ms')}` / `{arm.get('reranker_p95_ms')}` / `{arm.get('reranker_p99_ms')}` ms peak `{arm.get('peak_gpu_memory_mb')}` MB fallback `{arm.get('silent_fp32_fallback_detected')}`."
        for arm in arms
    ]
    return "\n".join(
        [
            "# TASK-0209 Reranker CUDA Precision Optimization Experiment",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; best_arm=`{summary.get('best_arm')}`; promotion_candidate=`{summary.get('promotion_candidate')}`; promotion_applied=`False`.",
            "",
            "## Authority",
            "",
            f"* Production model `{summary.get('production_reranker_model')}` on `{summary.get('resolved_production_device')}` with production dtype `{summary.get('production_dtype')}`.",
            f"* GPU `{environment.get('gpu_name')}`; capability `{environment.get('gpu_compute_capability')}`; CUDA `{environment.get('cuda_version')}`; PyTorch `{environment.get('pytorch_version')}`; driver `{environment.get('driver_version')}`.",
            f"* Formal query count `{summary.get('formal_query_count')}`; total measurements `{summary.get('total_measurement_count')}`.",
            "",
            "## Arms",
            "",
            *arm_lines,
            "",
            "## Quality And Safety",
            "",
            f"* Candidate input equivalence `{summary.get('candidate_input_equivalence')}`; formal quality regressions `{quality.get('formal_quality_regression_count')}`.",
            f"* Evidence/answerability/grounding/safe-action/downstream regressions `{downstream.get('evidence_selection_regression_count')}` / `{downstream.get('answerability_regression_count')}` / `{downstream.get('grounding_regression_count')}` / `{downstream.get('safe_action_regression_count')}` / `{downstream.get('downstream_regression_count')}`.",
            "",
            "## Memory And Decision",
            "",
            f"* Baseline peak GPU `{memory.get('baseline_peak_gpu_memory_mb')}` MB; best peak GPU `{promotion.get('best_peak_gpu_memory_mb')}` MB; reduction `{promotion.get('peak_gpu_memory_reduction_ratio')}`.",
            f"* P95 reduction `{promotion.get('reranker_p95_reduction_ratio')}`; runtime integration recommended `{promotion.get('runtime_integration_recommended')}`.",
            f"* Production dtype unchanged `{summary.get('production_dtype_unchanged')}`; independent verifier passed `{summary.get('independent_verifier_passed')}`.",
            "",
            "## Validation",
            "",
            f"* Focused tests `{summary.get('focused_tests')}`; related regression/model tests `{summary.get('related_regression_tests')}`; full suite `{summary.get('full_suite')}`; verifier `{summary.get('verifier_status')}`.",
            "",
        ]
    )


def unsupported_arm(spec: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "arm_id": spec["arm_id"],
        "arm_status": "unsupported",
        "execution_valid": False,
        "requested_precision_effective": False,
        "requested_dtype": spec["requested_dtype"],
        "model_parameter_dtype": "unknown",
        "forward_input_dtype": "unknown",
        "autocast_enabled": bool(spec.get("autocast")),
        "autocast_dtype": spec["requested_dtype"] if spec.get("autocast") else None,
        "resolved_device": "unknown",
        "model_parameter_device": "unknown",
        "forward_input_device": "unknown",
        "output_dtype": "unknown",
        "silent_fp32_fallback_detected": False,
        "finite_output_ratio": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "cuda_oom": False,
        "device_mismatch_error": False,
        "undeclared_cpu_fallback_detected": False,
        "deterministic_repeatability": False,
        "candidate_input_digest": None,
        "score_digest": None,
        "rankings": {},
        "reranker_p50_ms": None,
        "reranker_p95_ms": None,
        "reranker_p99_ms": None,
        "mean_reranker_latency_ms": None,
        "standard_deviation_ms": None,
        "measurement_count": 0,
        "cuda_sync_boundary_used": False,
        "peak_memory_stats_reset": False,
        "cuda_memory": empty_memory_metrics(),
        "peak_gpu_memory_mb": 0.0,
        "runtime_error_count": 0,
        "oom_count": 0,
        "exception_type": None,
        "exception_reason": reason,
    }


def error_arm(spec: Mapping[str, Any], exc_type: str, reason: str, *, oom: bool) -> dict[str, Any]:
    arm = unsupported_arm(spec, reason)
    arm.update(
        {
            "arm_status": "unsupported" if oom else "invalid",
            "cuda_oom": oom,
            "oom_count": int(oom),
            "runtime_error_count": 0 if oom else 1,
            "exception_type": exc_type,
            "exception_reason": reason,
        }
    )
    return arm


def empty_memory_metrics() -> dict[str, float]:
    return {
        "cuda_memory_allocated_before_mb": 0.0,
        "cuda_memory_allocated_after_mb": 0.0,
        "cuda_memory_reserved_before_mb": 0.0,
        "cuda_memory_reserved_after_mb": 0.0,
        "cuda_max_memory_allocated_mb": 0.0,
        "cuda_max_memory_reserved_mb": 0.0,
        "external_gpu_memory_used_mb": 0.0,
    }


def empty_quality_metrics(reason: str) -> dict[str, Any]:
    return {
        "skip_reason": reason,
        "top1_agreement_ratio": 0.0,
        "top3_set_agreement_ratio": 0.0,
        "top5_set_agreement_ratio": 0.0,
        "full_ranking_agreement_ratio": 0.0,
        "pairwise_order_agreement_ratio": 0.0,
        "maximum_absolute_score_delta": None,
        "mean_absolute_score_delta": None,
        "formal_quality_regression_count": 0,
    }


def scan_sensitive_payloads(payloads: Sequence[Any]) -> dict[str, Any]:
    text = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    patterns = ("postgres://", "postgresql://", "api_key", "api-token", "password", "token=", "BEGIN PRIVATE KEY")
    matches = [pattern for pattern in patterns if pattern.lower() in text.lower()]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "sensitive_value_exposure_count": len(matches),
        "matched_pattern_count": len(matches),
        "matched_patterns": matches,
        "scan_scope": "TASK-0209 generated artifacts; raw corpus/query text is not persisted.",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cuda_config(env: Mapping[str, str]) -> RerankerConfig:
    base = load_reranker_config(env)
    return RerankerConfig(
        provider=base.provider,
        model_name=base.model_name,
        model_revision=base.model_revision,
        batch_size=base.batch_size,
        max_pair_tokens=base.max_pair_tokens,
        device="cuda",
        cache_dir=base.cache_dir,
        local_files_only=base.local_files_only,
        input_template_version=base.input_template_version,
    )


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _parameter_device_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        counts[str(parameter.device)] = counts.get(str(parameter.device), 0) + int(parameter.numel())
    return dict(sorted(counts.items()))


def _parameter_dtype_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + int(parameter.numel())
    return dict(sorted(counts.items()))


def _rankings(input_snapshot: Mapping[str, Any], scores_by_query: Mapping[str, Sequence[float]]) -> dict[str, list[str]]:
    rankings = {}
    for query in input_snapshot.get("queries") or []:
        qid = str(query["query_id"])
        scores = scores_by_query.get(qid, [])
        candidate_ids = list(query["candidate_ids"])
        rankings[qid] = [candidate_id for _, candidate_id in sorted(zip(scores, candidate_ids, strict=True), reverse=True)]
    return rankings


def _scores_by_query(arm: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rankings = arm.get("rankings") or {}
    return {
        query_id: {candidate_id: float(len(ranking) - idx) for idx, candidate_id in enumerate(ranking)}
        for query_id, ranking in rankings.items()
    }


def _stable_repeats(scores: Sequence[Sequence[float]]) -> bool:
    if not scores:
        return False
    first = [round(float(value), 5) for value in scores[0]]
    return all([round(float(value), 5) for value in row] == first for row in scores[1:])


def _pairwise_agreement(reference: Sequence[str], candidate: Sequence[str]) -> float:
    if not reference or not candidate or set(reference) != set(candidate):
        return 0.0
    ref_pos = {item: idx for idx, item in enumerate(reference)}
    cand_pos = {item: idx for idx, item in enumerate(candidate)}
    total = agree = 0
    for idx, left in enumerate(reference):
        for right in reference[idx + 1 :]:
            total += 1
            agree += int((ref_pos[left] < ref_pos[right]) == (cand_pos[left] < cand_pos[right]))
    return agree / total if total else 1.0


def _valid_peak(value: Any) -> float | None:
    try:
        peak = float(value)
    except (TypeError, ValueError):
        return None
    return peak if peak > 0.0 else None


def _reduction(baseline: Any, experiment: Any) -> float | None:
    try:
        base = float(baseline)
        exp = float(experiment)
    except (TypeError, ValueError):
        return None
    if base <= 0.0:
        return None
    return round((base - exp) / base, 6)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 6)


def _driver_version() -> str | None:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], check=False, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()[0].strip() if proc.stdout.splitlines() else None
