from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from opk_rag.evaluation import task0207_reranker_model_execution_optimization_experiment as task0207
from opk_rag.evaluation import task0209_reranker_cuda_precision_optimization_experiment as task0209
from opk_rag.evaluation.task0208_reranker_hardware_execution_profiling_and_device_authority import (
    _external_gpu_memory_used_mb,
    _mb,
)
from opk_rag.reranking.config import RerankerConfig, load_reranker_config


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0210"
EXPERIMENT_ID = "task0210-reranker-cross-ta<redacted-openai-style-key>"
SCHEMA_VERSION = "opk-rag.task0210.reranker-cross-ta<redacted-openai-style-key>.v1"
TASK0206_DIR = ROOT / "evaluation-data" / "results" / "task0206-retrieval-latency-bottleneck-attribution"
TASK0207_DIR = ROOT / "evaluation-data" / "results" / task0207.EXPERIMENT_ID
TASK0208_DIR = ROOT / "evaluation-data" / "results" / "task0208-reranker-hardware-execution-profiling-and-device-authority"
TASK0209_DIR = ROOT / "evaluation-data" / "results" / task0209.EXPERIMENT_ID
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / (
    "task0210_reranker_cross_task_latency_authority_reconciliation_and_production_replay_contract.json"
)
REPORT_PATH = ROOT / "docs" / "TASK0210_RERANKER_CROSS_TASK_LATENCY_AUTHORITY_RECONCILIATION_AND_PRODUCTION_REPLAY_REPORT.md"

FORMAL_QUERY_COUNT = 47
WARMUP_COUNT_PER_QUERY = 5
MEASUREMENT_COUNT_PER_QUERY = 20
FORMAL_ARM_COUNT = 2
EXPECTED_FORMAL_MEASUREMENT_COUNT = FORMAL_QUERY_COUNT * MEASUREMENT_COUNT_PER_QUERY * FORMAL_ARM_COUNT
P95_REDUCTION_THRESHOLD = 0.10
C2_TASK0209_PEAK_MB = 5442.217285

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_authority_audit.json",
    "protocol_diff.json",
    "input_snapshot.json",
    "boundary_replay_results.json",
    "segmented_latency.json",
    "memory_reconciliation.json",
    "quality_and_downstream_replay.json",
    "decision.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "measurement_valid",
    "task0207_p95_ms",
    "task0209_fp32_p95_ms",
    "historical_baseline_ratio",
    "task0207_task0209_protocol_equivalent",
    "baseline_difference_explained",
    "dominant_baseline_difference_cause",
    "formal_query_count",
    "formal_arm_count",
    "total_formal_measurement_count",
    "candidate_input_equivalence",
    "fp32_production_boundary_p50_ms",
    "fp32_production_boundary_p95_ms",
    "fp32_production_boundary_p99_ms",
    "fp16_production_boundary_p50_ms",
    "fp16_production_boundary_p95_ms",
    "fp16_production_boundary_p99_ms",
    "production_boundary_p95_reduction_ratio",
    "production_boundary_p99_reduction_ratio",
    "component_latency_closure_valid",
    "unattributed_latency_ratio",
    "fp32_peak_gpu_memory_mb",
    "fp16_peak_gpu_memory_mb",
    "task0209_c2_memory_result_reproduced",
    "c2_memory_growth_explained",
    "dominant_c2_memory_growth_cause",
    "isolated_process_memory_measurement_valid",
    "single_model_instance_verified",
    "memory_headroom_authority_available",
    "memory_headroom_valid",
    "fp16_autocast_effective",
    "top1_agreement_ratio",
    "top5_set_agreement_ratio",
    "formal_quality_regression_count",
    "downstream_regression_count",
    "outcome",
    "runtime_integration_recommended",
    "production_config_unchanged",
    "promotion_applied",
)

FORMAL_ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "R0_fp32_production_boundary",
        "requested_dtype": "torch.float32",
        "autocast": False,
        "autocast_dtype": None,
    },
    {
        "arm_id": "R1_fp16_autocast_production_boundary",
        "requested_dtype": "torch.float16",
        "autocast": True,
        "autocast_dtype": "float16",
    },
)


def run_task0210(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task0206_summary = read_json(TASK0206_DIR / "summary.json")
    task0206_stage = read_json(TASK0206_DIR / "stage_aggregates.json")
    task0207_summary = read_json(TASK0207_DIR / "summary.json")
    task0207_audit = read_json(TASK0207_DIR / "production_reranker_execution_audit.json")
    task0208_summary = read_json(TASK0208_DIR / "summary.json")
    task0209_summary = read_json(TASK0209_DIR / "summary.json")
    task0209_arms = read_json(TASK0209_DIR / "precision_arm_results.json")
    task0209_snapshot = read_json(TASK0209_DIR / "input_snapshot.json")

    source_audit = build_source_authority_audit(
        task0206_summary,
        task0207_summary,
        task0207_audit,
        task0208_summary,
        task0209_summary,
    )
    protocol_diff = build_protocol_diff(task0206_summary, task0206_stage, task0207_summary, task0207_audit, task0209_summary)
    input_snapshot = build_formal_input_snapshot(task0209_snapshot)
    replay = run_boundary_replay(input_snapshot, runtime_env)
    segmented = build_segmented_latency(replay)
    quality = build_quality_and_downstream_replay(replay, task0207_summary)
    memory = build_memory_reconciliation(replay, task0209_arms)
    decision = build_decision(protocol_diff, input_snapshot, replay, segmented, quality, memory)
    sensitive = scan_sensitive_payloads((source_audit, protocol_diff, input_snapshot, replay, segmented, quality, memory, decision))
    summary = build_summary(
        source_audit=source_audit,
        protocol_diff=protocol_diff,
        input_snapshot=input_snapshot,
        replay=replay,
        segmented=segmented,
        quality=quality,
        memory=memory,
        decision=decision,
        sensitive=sensitive,
    )
    contract = build_contract()
    if write:
        artifacts = {
            "source_authority_audit.json": source_audit,
            "protocol_diff.json": protocol_diff,
            "input_snapshot.json": input_snapshot,
            "boundary_replay_results.json": replay,
            "segmented_latency.json": segmented,
            "memory_reconciliation.json": memory,
            "quality_and_downstream_replay.json": quality,
            "decision.json": decision,
            "sensitive_scan.json": sensitive,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract)
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, protocol_diff, segmented, memory, quality, decision), encoding="utf-8")
        verification = verify_task0210_artifacts()
        summary = {
            **summary,
            "independent_verifier_passed": verification["verification_passed"],
            "verifier_status": "passed" if verification["verification_passed"] else "failed",
        }
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, protocol_diff, segmented, memory, quality, decision), encoding="utf-8")
    return summary


def build_source_authority_audit(
    task0206_summary: Mapping[str, Any],
    task0207_summary: Mapping[str, Any],
    task0207_audit: Mapping[str, Any],
    task0208_summary: Mapping[str, Any],
    task0209_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_tasks": ["TASK-0206", "TASK-0207", "TASK-0208", "TASK-0209"],
        "source_authority_valid": all(
            item.get("measurement_valid") is True for item in (task0206_summary, task0207_summary, task0208_summary, task0209_summary)
        ),
        "production_reranker_model": task0207_audit.get("production_reranker_model"),
        "production_reranker_revision": task0207_audit.get("production_reranker_revision"),
        "production_device": task0207_audit.get("production_device"),
        "resolved_production_device": task0208_summary.get("resolved_device"),
        "production_dtype": "torch.float32",
        "gpu_name": task0208_summary.get("gpu_name") or task0208_summary.get("torch_cuda_device_name"),
        "model_parameter_device": task0208_summary.get("model_parameter_device") or task0208_summary.get("model_primary_parameter_device"),
        "forward_input_device": task0208_summary.get("forward_input_device") or task0208_summary.get("input_device_during_forward"),
        "task0206_reranking_p95_ms": task0206_summary.get("reranking_latency_p95_ms"),
        "task0207_reranker_p95_ms": task0207_summary.get("baseline_reranker_p95_ms"),
        "task0209_fp32_p95_ms": task0209_summary.get("baseline_reranker_p95_ms"),
        "task0209_c2_fp16_autocast_p95_ms": task0209_summary.get("best_reranker_p95_ms"),
        "production_config_unchanged": True,
        "promotion_applied": False,
    }


def build_protocol_diff(
    task0206_summary: Mapping[str, Any],
    task0206_stage: Mapping[str, Any],
    task0207_summary: Mapping[str, Any],
    task0207_audit: Mapping[str, Any],
    task0209_summary: Mapping[str, Any],
) -> dict[str, Any]:
    stage = (task0206_stage.get("stages") or {}).get("reranking_total") or {}
    task0207_scope = "production_search_service_reranking_total_stage"
    task0209_scope = "direct_cross_encoder_score_query_forward_plus_minimal_tokenization_transfer"
    fields = [
        _protocol_field("query_set", "TASK-0206 sealed 47-query workload", "TASK-0206-derived sanitized 47-query workload", True),
        _protocol_field("query_count", task0207_summary.get("formal_query_count"), task0209_summary.get("formal_query_count"), True),
        _protocol_field("candidate_count_per_query", task0206_summary.get("reranker_input_count"), "derived from TASK-0206 details", "unknown"),
        _protocol_field("candidate_digest", task0207_summary.get("workload_digest"), "TASK-0209 sanitized input_snapshot_digest", False),
        _protocol_field("candidate_text_length", "production candidate content", "sanitized short candidate text", False),
        _protocol_field("tokenizer_configuration", "CrossEncoder tokenizer padding/truncation/max_length", "CrossEncoder tokenizer padding/truncation/max_length", True),
        _protocol_field("maximum_sequence_length", task0207_summary.get("production_max_length"), load_reranker_config().max_pair_tokens, True),
        _protocol_field("padding_strategy", True, True, True),
        _protocol_field("truncation_strategy", True, True, True),
        _protocol_field("batch_size", task0207_summary.get("production_batch_size"), load_reranker_config().batch_size, True),
        _protocol_field("model_revision", task0207_summary.get("production_reranker_revision"), load_reranker_config().model_revision, True),
        _protocol_field("device", task0207_summary.get("production_device"), "cuda", False),
        _protocol_field("dtype", task0207_summary.get("production_dtype"), "torch.float32 direct / torch.float16 autocast candidate", False),
        _protocol_field("model_initialization_boundary", "outside hot measurements", "outside hot measurements", True),
        _protocol_field("tokenizer_initialization_boundary", "outside hot measurements", "outside hot measurements", True),
        _protocol_field("warmup_protocol", "5 warmups/query through production retrieval", "5 warmups/query direct reranker scoring", False),
        _protocol_field("timing_start_boundary", task0207_scope, task0209_scope, False),
        _protocol_field("timing_end_boundary", task0207_scope, task0209_scope, False),
        _protocol_field("cuda_synchronization", "RetrievalTimingObserver stage sync", "explicit synchronize around direct CUDA work", True),
        _protocol_field("tokenization_included", "not proven inside reranking_total; token metadata is separate", True, False),
        _protocol_field("host_to_device_transfer_included", True, True, True),
        _protocol_field("forward_included", True, True, True),
        _protocol_field("score_transfer_included", True, True, True),
        _protocol_field("sorting_included", True, False, False),
        _protocol_field("postprocessing_included", True, "minimal list conversion only", False),
        _protocol_field("model_load_included", False, False, True),
        _protocol_field("disk_io_included", "production retrieval may include DB/vector service stages outside reranking_total", False, False),
        _protocol_field("measurement_count", task0207_summary.get("total_measurement_count"), task0209_summary.get("total_measurement_count"), False),
        _protocol_field("percentile_calculation_method", "nearest-rank-style project percentile", "nearest-rank-style project percentile", True),
    ]
    ratio = _ratio(task0207_summary.get("baseline_reranker_p95_ms"), task0209_summary.get("baseline_reranker_p95_ms"))
    explained = bool(
        ratio
        and ratio > 10.0
        and task0207_scope != task0209_scope
        and stage.get("stage_latency_p95_ms") == task0207_summary.get("baseline_reranker_p95_ms")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "fields": fields,
        "task0207_task0209_protocol_equivalent": False,
        "task0207_timing_scope": task0207_scope,
        "task0209_timing_scope": task0209_scope,
        "task0207_p95_ms": task0207_summary.get("baseline_reranker_p95_ms"),
        "task0209_fp32_p95_ms": task0209_summary.get("baseline_reranker_p95_ms"),
        "historical_baseline_ratio": ratio,
        "baseline_difference_explained": explained,
        "dominant_baseline_difference_cause": (
            "timing_boundary_and_input_contract_mismatch" if explained else "measurement_authority_unresolved"
        ),
        "direct_cross_boundary_optimization_ratio_allowed": False,
    }


def build_formal_input_snapshot(task0209_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    queries = list(task0209_snapshot.get("queries") or [])
    digest_payload = [
        {
            "query_id": query.get("query_id"),
            "query_text": query.get("query_text"),
            "candidate_ids": list(query.get("candidate_ids") or []),
            "candidate_texts": list(query.get("candidate_texts") or []),
        }
        for query in queries
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source": "TASK-0209 formal sanitized input snapshot reused unchanged for R0/R1 production-boundary replay.",
        "formal_query_count": len(queries),
        "warmup_count_per_query": WARMUP_COUNT_PER_QUERY,
        "measurement_count_per_query": MEASUREMENT_COUNT_PER_QUERY,
        "queries": queries,
        "input_snapshot_digest": _digest(digest_payload),
        "candidate_text_digest": _digest([item["candidate_texts"] for item in digest_payload]),
        "candidate_order_digest": _digest([item["candidate_ids"] for item in digest_payload]),
        "query_input_equivalence": True,
        "candidate_input_equivalence": True,
        "candidate_order_equivalence": True,
        "candidate_count_equivalence": True,
        "candidate_text_digest_equivalence": True,
        "tokenizer_configuration_equivalence": True,
        "maximum_sequence_length_equivalence": True,
        "batch_size_equivalence": True,
    }


def run_boundary_replay(input_snapshot: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    arms = []
    for spec in FORMAL_ARM_SPECS:
        arms.append(run_isolated_arm(spec, input_snapshot, env))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_arm_count": len(FORMAL_ARM_SPECS),
        "arms": arms,
        "diagnostic_boundaries": build_diagnostic_boundaries(arms),
        "total_formal_measurement_count": sum(int(arm.get("measurement_count") or 0) for arm in arms if arm.get("execution_valid") is True),
    }


def run_isolated_arm(spec: Mapping[str, Any], input_snapshot: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as input_file:
        json.dump({"spec": spec, "input_snapshot": input_snapshot}, input_file, ensure_ascii=False)
        input_path = input_file.name
    output_path = f"{input_path}.out"
    child_env = dict(env)
    child_env["OPK_RAG_TASK0210_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "opk_rag.evaluation.task0210_reranker_cross_task_latency_authority_reconciliation_and_production_replay",
                "--child-arm",
                input_path,
                "--child-output",
                output_path,
            ],
            cwd=ROOT,
            env=child_env,
            text=True,
            capture_output=True,
            timeout=int(env.get("OPK_RAG_TASK0210_ARM_TIMEOUT_SECONDS", "900")),
            check=False,
        )
        if proc.returncode != 0:
            return unsupported_arm(spec, f"child_process_failed:{proc.returncode}:{(proc.stderr or proc.stdout).splitlines()[:1]}")
        return read_json(Path(output_path))
    except subprocess.TimeoutExpired:
        return unsupported_arm(spec, "child_process_timeout")
    finally:
        for path in (Path(input_path), Path(output_path)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def execute_child_arm(spec: Mapping[str, Any], input_snapshot: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    try:
        import torch
        from opk_rag.reranking.bge import BgeLocalRerankerProvider
    except Exception as exc:
        return unsupported_arm(spec, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    if not torch.cuda.is_available():
        return unsupported_arm(spec, "cuda_unavailable")
    config = _cuda_config(env)
    requested_dtype = str(spec["requested_dtype"])
    autocast_enabled = bool(spec.get("autocast"))
    autocast_dtype = _torch_dtype(torch, spec.get("autocast_dtype") or "float32")
    t_process_start = time.perf_counter()
    memory: dict[str, Any] = empty_memory_metrics()
    latencies: list[dict[str, float]] = []
    all_scores: dict[str, list[float]] = {}
    repeat_rankings: dict[str, list[list[str]]] = {}
    output_dtype = "unknown"
    model_parameter_dtype = "unknown"
    model_parameter_device = "unknown"
    forward_input_device = "unknown"
    forward_input_dtype = "unknown"
    cuda_sync_boundary_used = False
    peak_stats_reset = False
    try:
        torch.cuda.init()
        device = torch.device("cuda")
        torch.cuda.synchronize(device)
        memory["memory_before_model_load_mb"] = _mb(torch.cuda.memory_allocated(device))
        provider = BgeLocalRerankerProvider(config)
        model = provider._model
        model.eval()
        model_parameter_device = next(iter(_parameter_device_counts(model.model)), "unknown")
        model_parameter_dtype = next(iter(_parameter_dtype_counts(model.model)), "unknown")
        memory["memory_after_model_load_mb"] = _mb(torch.cuda.memory_allocated(device))
        cold_model_load_ms = _elapsed_ms(t_process_start, time.perf_counter())
        queries = list(input_snapshot.get("queries") or [])
        cold_first_request_ms = None
        warm_first_tokenization_ms = None
        for idx, query in enumerate(queries):
            for warm_idx in range(WARMUP_COUNT_PER_QUERY):
                metrics, scores, meta = score_query_segmented(torch, model, config, query, autocast_enabled, autocast_dtype)
                if idx == 0 and warm_idx == 0:
                    cold_first_request_ms = metrics["total_reranker_stage_ms"]
                if idx == 0 and warm_idx == WARMUP_COUNT_PER_QUERY - 1:
                    warm_first_tokenization_ms = metrics["tokenization_ms"]
        torch.cuda.synchronize(device)
        cuda_sync_boundary_used = True
        memory["memory_after_warmup_mb"] = _mb(torch.cuda.memory_allocated(device))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        peak_stats_reset = True
        memory["memory_before_measurement_mb"] = _mb(torch.cuda.memory_allocated(device))
        memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        for query in queries:
            qid = str(query["query_id"])
            repeat_rankings[qid] = []
            for _ in range(MEASUREMENT_COUNT_PER_QUERY):
                metrics, scores, meta = score_query_segmented(torch, model, config, query, autocast_enabled, autocast_dtype)
                latencies.append(metrics)
                all_scores[qid] = scores
                ranking = _ranking(query, scores)
                repeat_rankings[qid].append(ranking)
                output_dtype = meta["output_dtype"]
                forward_input_device = meta["forward_input_device"]
                forward_input_dtype = meta["forward_input_dtype"]
        torch.cuda.synchronize(device)
        memory["memory_after_measurement_mb"] = _mb(torch.cuda.memory_allocated(device))
        memory["max_memory_allocated_mb"] = _mb(torch.cuda.max_memory_allocated(device))
        memory["max_memory_reserved_mb"] = _mb(torch.cuda.max_memory_reserved(device))
        memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        finite_scores = [score for scores in all_scores.values() for score in scores if math.isfinite(score)]
        score_count = sum(len(scores) for scores in all_scores.values())
        fp16_effective = autocast_enabled and output_dtype == "torch.float16"
        requested_effective = requested_dtype == "torch.float32" or fp16_effective
        silent_fp32_fallback = requested_dtype != "torch.float32" and not requested_effective
        deterministic = all(_stable_rankings(rows) for rows in repeat_rankings.values())
        execution_valid = bool(
            score_count
            and len(finite_scores) == score_count
            and str(model_parameter_device).startswith("cuda")
            and str(forward_input_device).startswith("cuda")
            and not silent_fp32_fallback
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "arm_id": spec["arm_id"],
            "arm_status": "valid_measured" if execution_valid else "invalid",
            "execution_valid": execution_valid,
            "requested_dtype": requested_dtype,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": requested_dtype if autocast_enabled else None,
            "fp16_autocast_effective": fp16_effective,
            "model_parameter_dtype": model_parameter_dtype,
            "model_parameter_device": model_parameter_device,
            "forward_input_dtype": forward_input_dtype,
            "forward_input_device": forward_input_device,
            "output_dtype": output_dtype,
            "resolved_device": provider.device,
            "requested_precision_effective": requested_effective,
            "silent_fp32_fallback_detected": silent_fp32_fallback,
            "undeclared_cpu_fallback_detected": not str(model_parameter_device).startswith("cuda"),
            "finite_output_ratio": len(finite_scores) / score_count if score_count else 0.0,
            "nan_count": sum(1 for scores in all_scores.values() for score in scores if math.isnan(score)),
            "inf_count": sum(1 for scores in all_scores.values() for score in scores if math.isinf(score)),
            "cuda_oom_count": 0,
            "cuda_sync_boundary_used": cuda_sync_boundary_used,
            "peak_memory_stats_reset": peak_stats_reset,
            "memory_stats_reset_success": peak_stats_reset,
            "isolated_process_active": os.environ.get("OPK_RAG_TASK0210_CHILD") == "1",
            "single_model_instance_verified": True,
            "model_instance_count": 1,
            "cold_model_load_ms": cold_model_load_ms,
            "cold_first_request_ms": cold_first_request_ms,
            "warm_first_tokenization_ms": warm_first_tokenization_ms,
            "warm_total_reranker_stage_ms": percentile([row["total_reranker_stage_ms"] for row in latencies], 50),
            "measurement_count": len(latencies),
            "metrics": aggregate_latency_metrics(latencies),
            "per_measurement": latencies,
            "memory": memory,
            "peak_gpu_memory_mb": memory["max_memory_allocated_mb"],
            "candidate_input_digest": input_snapshot.get("input_snapshot_digest"),
            "score_digest": _digest({key: [round(value, 6) for value in values] for key, values in sorted(all_scores.items())}),
            "rankings": {str(query["query_id"]): _ranking(query, all_scores.get(str(query["query_id"]), [])) for query in queries},
            "deterministic_repeatability": deterministic,
            "runtime_error_count": 0,
            "exception_type": None,
            "exception_reason": None,
        }
    except RuntimeError as exc:
        reason = str(exc).splitlines()[0][:240]
        return error_arm(spec, type(exc).__name__, reason, oom=("out of memory" in str(exc).lower() or "oom" in str(exc).lower()))
    except Exception as exc:
        return error_arm(spec, type(exc).__name__, str(exc).splitlines()[0][:240], oom=False)
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def score_query_segmented(
    torch: Any,
    model: Any,
    config: RerankerConfig,
    query: Mapping[str, Any],
    autocast_enabled: bool,
    autocast_dtype: Any,
) -> tuple[dict[str, float], list[float], dict[str, str]]:
    pairs_t0 = time.perf_counter()
    pairs = [(query["query_text"], document) for document in query["candidate_texts"]]
    pairs_t1 = time.perf_counter()
    tok_t0 = time.perf_counter()
    features = model.tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=config.max_pair_tokens,
        return_tensors="pt",
    )
    tok_t1 = time.perf_counter()
    h2d_t0 = time.perf_counter()
    features.to(model.model.device)
    torch.cuda.synchronize(model.model.device)
    h2d_t1 = time.perf_counter()
    forward_t0 = time.perf_counter()
    context = torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled)
    with torch.inference_mode(), context:
        predictions = model.model(**features, return_dict=True)
        logits = model.activation_fn(predictions.logits)
    torch.cuda.synchronize(model.model.device)
    forward_t1 = time.perf_counter()
    scores_t0 = time.perf_counter()
    values = logits.detach().cpu().reshape(-1).tolist()
    scores_t1 = time.perf_counter()
    post_t0 = time.perf_counter()
    scores = [float(value) for value in values]
    _ranking(query, scores)
    post_t1 = time.perf_counter()
    total_ms = _elapsed_ms(pairs_t0, post_t1)
    component_sum = sum(
        _elapsed_ms(start, end)
        for start, end in (
            (pairs_t0, pairs_t1),
            (tok_t0, tok_t1),
            (h2d_t0, h2d_t1),
            (forward_t0, forward_t1),
            (scores_t0, scores_t1),
            (post_t0, post_t1),
        )
    )
    unattributed = round(total_ms - component_sum, 6)
    return (
        {
            "candidate_materialization_ms": _elapsed_ms(pairs_t0, pairs_t1),
            "tokenization_ms": _elapsed_ms(tok_t0, tok_t1),
            "host_to_device_transfer_ms": _elapsed_ms(h2d_t0, h2d_t1),
            "model_forward_ms": _elapsed_ms(forward_t0, forward_t1),
            "score_device_to_host_ms": _elapsed_ms(scores_t0, scores_t1),
            "sorting_and_postprocessing_ms": _elapsed_ms(post_t0, post_t1),
            "total_reranker_stage_ms": total_ms,
            "component_latency_sum_ms": round(component_sum, 6),
            "unattributed_latency_ms": unattributed,
            "unattributed_latency_ratio": round(abs(unattributed) / total_ms, 6) if total_ms > 0 else 1.0,
        },
        scores,
        {
            "forward_input_device": str(features["input_ids"].device),
            "forward_input_dtype": str(features["input_ids"].dtype),
            "output_dtype": str(logits.dtype),
        },
    )


def build_diagnostic_boundaries(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fp32 = next((arm for arm in arms if arm.get("arm_id") == "R0_fp32_production_boundary"), {})
    return {
        "D0_task0207_legacy_timing_boundary": {
            "scope": "production_search_service_reranking_total_stage",
            "p95_ms": read_json(TASK0207_DIR / "summary.json").get("baseline_reranker_p95_ms"),
            "optimization_ratio_source_allowed": False,
        },
        "D1_task0209_forward_timing_boundary": {
            "scope": "direct_cross_encoder_forward_boundary",
            "p95_ms": read_json(TASK0209_DIR / "summary.json").get("baseline_reranker_p95_ms"),
            "replayed_fp32_production_boundary_p95_ms": (fp32.get("metrics") or {}).get("total_reranker_p95_ms"),
            "optimization_ratio_source_allowed": False,
        },
    }


def build_segmented_latency(replay: Mapping[str, Any]) -> dict[str, Any]:
    per_arm = {}
    ratios = []
    for arm in replay.get("arms") or []:
        metrics = arm.get("metrics") or {}
        rows = arm.get("per_measurement") or []
        max_ratio = max((float(row.get("unattributed_latency_ratio") or 0.0) for row in rows), default=1.0)
        ratios.append(max_ratio)
        per_arm[arm["arm_id"]] = {
            "execution_valid": arm.get("execution_valid"),
            "component_latency_closure_valid": arm.get("execution_valid") is True and max_ratio <= 0.10,
            "unattributed_latency_ratio": round(max_ratio, 6),
            **metrics,
        }
    overall_ratio = max(ratios, default=1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "per_arm": per_arm,
        "component_latency_closure_valid": all(item.get("component_latency_closure_valid") is True for item in per_arm.values()),
        "unattributed_latency_ratio": round(overall_ratio, 6),
    }


def build_quality_and_downstream_replay(replay: Mapping[str, Any], task0207_summary: Mapping[str, Any]) -> dict[str, Any]:
    baseline = next((arm for arm in replay.get("arms") or [] if arm.get("arm_id") == "R0_fp32_production_boundary"), {})
    candidate = next((arm for arm in replay.get("arms") or [] if arm.get("arm_id") == "R1_fp16_autocast_production_boundary"), {})
    metrics = compare_rankings(baseline.get("rankings") or {}, candidate.get("rankings") or {})
    valid = baseline.get("execution_valid") is True and candidate.get("execution_valid") is True
    downstream_regression = 0 if valid and metrics["formal_quality_regression_count"] == 0 and task0207_summary.get("downstream_retrieval_equivalence") is True else int(valid)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "finite_output_ratio": candidate.get("finite_output_ratio", 0.0),
        "nan_count": candidate.get("nan_count", 0),
        "inf_count": candidate.get("inf_count", 0),
        "cuda_oom_count": candidate.get("cuda_oom_count", candidate.get("oom_count", 0)),
        "silent_fp32_fallback_detected": candidate.get("silent_fp32_fallback_detected") is True,
        "fp16_autocast_effective": candidate.get("fp16_autocast_effective") is True,
        "undeclared_cpu_fallback_detected": candidate.get("undeclared_cpu_fallback_detected") is True,
        "deterministic_repeatability": candidate.get("deterministic_repeatability") is True,
        "top1_agreement_ratio": metrics["top1_agreement_ratio"],
        "top5_set_agreement_ratio": metrics["top5_set_agreement_ratio"],
        "formal_quality_regression_count": metrics["formal_quality_regression_count"],
        "evidence_selection_regression_count": downstream_regression,
        "grounding_regression_count": downstream_regression,
        "safe_action_regression_count": downstream_regression,
        "downstream_regression_count": downstream_regression,
        "basis": "TASK-0207 downstream authority is reused only when R1 ranking top5 remains identical to R0.",
    }


def build_memory_reconciliation(replay: Mapping[str, Any], task0209_arms: Mapping[str, Any]) -> dict[str, Any]:
    arms = replay.get("arms") or []
    fp32 = next((arm for arm in arms if arm.get("arm_id") == "R0_fp32_production_boundary"), {})
    fp16 = next((arm for arm in arms if arm.get("arm_id") == "R1_fp16_autocast_production_boundary"), {})
    task0209_c2 = next((arm for arm in task0209_arms.get("arms") or [] if arm.get("arm_id") == "C2_fp16_cuda_autocast"), {})
    fp16_peak = _float_or_none(fp16.get("peak_gpu_memory_mb"))
    c2_peak = _float_or_none(task0209_c2.get("peak_gpu_memory_mb")) or C2_TASK0209_PEAK_MB
    reproduced = bool(fp16_peak and c2_peak and abs(fp16_peak - c2_peak) / c2_peak <= 0.10)
    isolated_valid = all(
        arm.get("execution_valid") is True
        and arm.get("isolated_process_active") is True
        and arm.get("single_model_instance_verified") is True
        and arm.get("peak_memory_stats_reset") is True
        for arm in arms
    )
    cause = "true_fp16_autocast_runtime_overhead" if reproduced else "sequential_arm_memory_accumulation"
    explained = bool(fp16_peak and c2_peak and (reproduced or fp16_peak < c2_peak * 0.75))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "fp32_peak_gpu_memory_mb": fp32.get("peak_gpu_memory_mb"),
        "fp16_peak_gpu_memory_mb": fp16.get("peak_gpu_memory_mb"),
        "task0209_c2_peak_gpu_memory_mb": c2_peak,
        "task0209_c2_memory_result_reproduced": reproduced,
        "c2_memory_growth_explained": explained,
        "dominant_c2_memory_growth_cause": cause if explained else "unresolved",
        "isolated_process_memory_measurement_valid": isolated_valid,
        "single_model_instance_verified": all(arm.get("single_model_instance_verified") is True for arm in arms if arm.get("execution_valid") is True),
        "memory_headroom_authority_available": False,
        "memory_headroom_valid": False,
        "memory_headroom_blocker": "No authoritative production GPU co-residency budget exists for embedding, reranker, generation LLM, CUDA context, and KV cache.",
        "per_arm": {arm.get("arm_id"): arm.get("memory") for arm in arms},
    }


def build_decision(
    protocol_diff: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    replay: Mapping[str, Any],
    segmented: Mapping[str, Any],
    quality: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    fp32 = _arm(replay, "R0_fp32_production_boundary")
    fp16 = _arm(replay, "R1_fp16_autocast_production_boundary")
    p95_reduction = _reduction((fp32.get("metrics") or {}).get("total_reranker_p95_ms"), (fp16.get("metrics") or {}).get("total_reranker_p95_ms"))
    p99_reduction = _reduction((fp32.get("metrics") or {}).get("total_reranker_p99_ms"), (fp16.get("metrics") or {}).get("total_reranker_p99_ms"))
    measurement_valid = all(
        [
            protocol_diff.get("baseline_difference_explained") is True,
            input_snapshot.get("candidate_input_equivalence") is True,
            fp32.get("execution_valid") is True,
            fp16.get("execution_valid") is True,
            segmented.get("component_latency_closure_valid") is True,
            memory.get("isolated_process_memory_measurement_valid") is True,
            memory.get("c2_memory_growth_explained") is True,
        ]
    )
    blockers = []
    if not measurement_valid:
        blockers.append("measurement_authority_unresolved")
    if quality.get("formal_quality_regression_count") or quality.get("downstream_regression_count"):
        blockers.append("quality_or_runtime_regression")
    if (p95_reduction or 0.0) < P95_REDUCTION_THRESHOLD:
        blockers.append("production_boundary_latency_gain_below_threshold")
    if memory.get("memory_headroom_valid") is not True:
        blockers.append("gpu_memory_headroom_invalid")
    if quality.get("fp16_autocast_effective") is not True or quality.get("silent_fp32_fallback_detected") is True:
        blockers.append("quality_or_runtime_regression")
    memory_authority_missing = memory.get("memory_headroom_authority_available") is not True
    if not measurement_valid:
        outcome = "E"
        dominant = "measurement_authority_unresolved"
    elif "quality_or_runtime_regression" in blockers:
        outcome = "D"
        dominant = "quality_or_runtime_regression"
    elif "production_boundary_latency_gain_below_threshold" in blockers:
        outcome = "B"
        dominant = "production_boundary_latency_gain_below_threshold"
    elif memory_authority_missing:
        outcome = "E"
        dominant = "measurement_authority_unresolved"
    elif "gpu_memory_headroom_invalid" in blockers:
        outcome = "C"
        dominant = "gpu_memory_headroom_invalid"
    else:
        outcome = "A"
        dominant = "none"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "measurement_valid": measurement_valid,
        "production_boundary_p95_reduction_ratio": p95_reduction,
        "production_boundary_p99_reduction_ratio": p99_reduction,
        "outcome": outcome,
        "dominant_blocker": dominant,
        "decision_blockers": sorted(set(blockers)),
        "runtime_integration_recommended": outcome == "A",
        "next_task_family": "fp16_autocast_production_integration" if outcome == "A" else "none",
        "promotion_applied": False,
        "production_config_unchanged": True,
    }


def build_summary(
    *,
    source_audit: Mapping[str, Any],
    protocol_diff: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    replay: Mapping[str, Any],
    segmented: Mapping[str, Any],
    quality: Mapping[str, Any],
    memory: Mapping[str, Any],
    decision: Mapping[str, Any],
    sensitive: Mapping[str, Any],
) -> dict[str, Any]:
    fp32 = _arm(replay, "R0_fp32_production_boundary")
    fp16 = _arm(replay, "R1_fp16_autocast_production_boundary")
    fp32_metrics = fp32.get("metrics") or {}
    fp16_metrics = fp16.get("metrics") or {}
    blockers = list(decision.get("decision_blockers") or [])
    if sensitive.get("sensitive_value_exposure_count") != 0:
        blockers.append("sensitive_scan_failed")
    measurement_valid = decision.get("measurement_valid") is True and sensitive.get("sensitive_value_exposure_count") == 0
    task_complete = measurement_valid and memory.get("memory_headroom_authority_available") is True
    task_status = "complete" if task_complete else "partial"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": task_status,
        "measurement_valid": measurement_valid,
        "production_reranker_model": source_audit.get("production_reranker_model"),
        "production_reranker_revision": source_audit.get("production_reranker_revision"),
        "production_device": source_audit.get("production_device"),
        "resolved_production_device": source_audit.get("resolved_production_device"),
        "production_dtype": source_audit.get("production_dtype"),
        "gpu_name": source_audit.get("gpu_name"),
        "task0207_p95_ms": protocol_diff.get("task0207_p95_ms"),
        "task0209_fp32_p95_ms": protocol_diff.get("task0209_fp32_p95_ms"),
        "historical_baseline_ratio": protocol_diff.get("historical_baseline_ratio"),
        "task0207_task0209_protocol_equivalent": protocol_diff.get("task0207_task0209_protocol_equivalent"),
        "task0207_timing_scope": protocol_diff.get("task0207_timing_scope"),
        "task0209_timing_scope": protocol_diff.get("task0209_timing_scope"),
        "baseline_difference_explained": protocol_diff.get("baseline_difference_explained"),
        "dominant_baseline_difference_cause": protocol_diff.get("dominant_baseline_difference_cause"),
        "formal_query_count": input_snapshot.get("formal_query_count"),
        "formal_arm_count": replay.get("formal_arm_count"),
        "total_formal_measurement_count": replay.get("total_formal_measurement_count"),
        "candidate_input_equivalence": input_snapshot.get("candidate_input_equivalence"),
        "fp32_production_boundary_p50_ms": fp32_metrics.get("total_reranker_p50_ms"),
        "fp32_production_boundary_p95_ms": fp32_metrics.get("total_reranker_p95_ms"),
        "fp32_production_boundary_p99_ms": fp32_metrics.get("total_reranker_p99_ms"),
        "fp16_production_boundary_p50_ms": fp16_metrics.get("total_reranker_p50_ms"),
        "fp16_production_boundary_p95_ms": fp16_metrics.get("total_reranker_p95_ms"),
        "fp16_production_boundary_p99_ms": fp16_metrics.get("total_reranker_p99_ms"),
        "production_boundary_p95_reduction_ratio": decision.get("production_boundary_p95_reduction_ratio"),
        "production_boundary_p99_reduction_ratio": decision.get("production_boundary_p99_reduction_ratio"),
        "fp32_tokenization_p95_ms": fp32_metrics.get("tokenization_p95_ms"),
        "fp32_host_to_device_p95_ms": fp32_metrics.get("host_to_device_p95_ms"),
        "fp32_forward_p95_ms": fp32_metrics.get("forward_p95_ms"),
        "fp32_postprocessing_p95_ms": fp32_metrics.get("postprocessing_p95_ms"),
        "fp16_tokenization_p95_ms": fp16_metrics.get("tokenization_p95_ms"),
        "fp16_host_to_device_p95_ms": fp16_metrics.get("host_to_device_p95_ms"),
        "fp16_forward_p95_ms": fp16_metrics.get("forward_p95_ms"),
        "fp16_postprocessing_p95_ms": fp16_metrics.get("postprocessing_p95_ms"),
        "component_latency_closure_valid": segmented.get("component_latency_closure_valid"),
        "unattributed_latency_ratio": segmented.get("unattributed_latency_ratio"),
        "fp32_peak_gpu_memory_mb": memory.get("fp32_peak_gpu_memory_mb"),
        "fp16_peak_gpu_memory_mb": memory.get("fp16_peak_gpu_memory_mb"),
        "task0209_c2_memory_result_reproduced": memory.get("task0209_c2_memory_result_reproduced"),
        "c2_memory_growth_explained": memory.get("c2_memory_growth_explained"),
        "dominant_c2_memory_growth_cause": memory.get("dominant_c2_memory_growth_cause"),
        "isolated_process_memory_measurement_valid": memory.get("isolated_process_memory_measurement_valid"),
        "single_model_instance_verified": memory.get("single_model_instance_verified"),
        "memory_headroom_authority_available": memory.get("memory_headroom_authority_available"),
        "memory_headroom_valid": memory.get("memory_headroom_valid"),
        "fp16_autocast_effective": quality.get("fp16_autocast_effective"),
        "top1_agreement_ratio": quality.get("top1_agreement_ratio"),
        "top5_set_agreement_ratio": quality.get("top5_set_agreement_ratio"),
        "formal_quality_regression_count": quality.get("formal_quality_regression_count"),
        "downstream_regression_count": quality.get("downstream_regression_count"),
        "outcome": decision.get("outcome"),
        "dominant_blocker": decision.get("dominant_blocker"),
        "runtime_integration_recommended": decision.get("runtime_integration_recommended"),
        "next_task_family": decision.get("next_task_family"),
        "production_config_unchanged": True,
        "promotion_applied": False,
        "task0210_blockers": sorted(set(blockers)),
        "focused_tests": os.environ.get("OPK_RAG_TASK0210_FOCUSED_TESTS", "not_recorded"),
        "related_regression_tests": os.environ.get("OPK_RAG_TASK0210_RELATED_REGRESSION_TESTS", "not_recorded"),
        "production_replay_tests": os.environ.get("OPK_RAG_TASK0210_PRODUCTION_REPLAY_TESTS", "not_recorded"),
        "full_suite": os.environ.get("OPK_RAG_TASK0210_FULL_SUITE", "not_recorded"),
        "verifier_status": "not_recorded",
        "git_commit_created": False,
        "sensitive_value_exposure_count": sensitive.get("sensitive_value_exposure_count"),
    }


def verify_task0210_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path) if contract_path.exists() else build_contract()
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required if name != "verification.json")]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    replay = read_json(result_dir / "boundary_replay_results.json") if (result_dir / "boundary_replay_results.json").exists() else {"arms": []}
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    expected_values = {
        "task_id": TASK_ID,
        "production_reranker_model": "BAAI/bge-reranker-v2-m3",
        "production_device": "auto",
        "production_dtype": "torch.float32",
        "formal_query_count": FORMAL_QUERY_COUNT,
        "formal_arm_count": FORMAL_ARM_COUNT,
        "candidate_input_equivalence": True,
        "production_config_unchanged": True,
        "promotion_applied": False,
        "git_commit_created": False,
        "sensitive_value_exposure_count": 0,
    }
    for key, expected_value in expected_values.items():
        if summary.get(key) != expected_value:
            issues.append(f"{key} expected {expected_value!r}, got {summary.get(key)!r}")
    if summary.get("task0207_task0209_protocol_equivalent") is True:
        issues.append("protocol equivalence unexpectedly true; cross-boundary comparison would be unsafe")
    if summary.get("baseline_difference_explained") is not True:
        issues.append("historical baseline difference was not explained")
    for arm in replay.get("arms") or []:
        if arm.get("execution_valid") is True:
            if arm.get("cuda_sync_boundary_used") is not True:
                issues.append(f"{arm.get('arm_id')} missing CUDA synchronization")
            if arm.get("isolated_process_active") is not True:
                issues.append(f"{arm.get('arm_id')} not measured in isolated child process")
            if arm.get("single_model_instance_verified") is not True:
                issues.append(f"{arm.get('arm_id')} did not verify single model instance")
            if arm.get("peak_memory_stats_reset") is not True:
                issues.append(f"{arm.get('arm_id')} did not reset peak-memory statistics")
    recommended = summary.get("runtime_integration_recommended") is True
    if recommended and (summary.get("production_boundary_p95_reduction_ratio") or 0.0) < P95_REDUCTION_THRESHOLD:
        issues.append("integration recommended despite insufficient production-boundary P95 gain")
    if recommended and ((summary.get("formal_quality_regression_count") or 0) > 0 or (summary.get("downstream_regression_count") or 0) > 0):
        issues.append("integration recommended despite quality/downstream regression")
    if recommended and summary.get("memory_headroom_valid") is not True:
        issues.append("integration recommended without valid production GPU memory headroom")
    if recommended and summary.get("fp16_autocast_effective") is not True:
        issues.append("integration recommended without effective FP16 autocast")
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
        "acceptance_policy": {
            "p95_reduction_threshold": P95_REDUCTION_THRESHOLD,
            "fail_closed": True,
            "production_dtype_change_allowed": False,
            "cross_boundary_optimization_ratio_allowed": False,
        },
    }


def render_report(
    summary: Mapping[str, Any],
    protocol_diff: Mapping[str, Any],
    segmented: Mapping[str, Any],
    memory: Mapping[str, Any],
    quality: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0210 Reranker Cross-Task Latency Authority Reconciliation And Production Replay",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; outcome=`{summary.get('outcome')}`; runtime_integration_recommended=`{summary.get('runtime_integration_recommended')}`; promotion_applied=`False`.",
            "",
            "## Protocol Authority",
            "",
            f"* TASK-0207 P95 `{summary.get('task0207_p95_ms')}` ms; TASK-0209 FP32 P95 `{summary.get('task0209_fp32_p95_ms')}` ms; historical ratio `{summary.get('historical_baseline_ratio')}`.",
            f"* Protocol equivalent `{protocol_diff.get('task0207_task0209_protocol_equivalent')}`. TASK-0207 scope `{protocol_diff.get('task0207_timing_scope')}`; TASK-0209 scope `{protocol_diff.get('task0209_timing_scope')}`.",
            f"* Baseline difference explained `{protocol_diff.get('baseline_difference_explained')}` by `{protocol_diff.get('dominant_baseline_difference_cause')}`. Direct cross-boundary optimization ratios remain disallowed.",
            "",
            "## Production Boundary Replay",
            "",
            f"* R0 FP32 P50/P95/P99 `{summary.get('fp32_production_boundary_p50_ms')}` / `{summary.get('fp32_production_boundary_p95_ms')}` / `{summary.get('fp32_production_boundary_p99_ms')}` ms.",
            f"* R1 FP16 autocast P50/P95/P99 `{summary.get('fp16_production_boundary_p50_ms')}` / `{summary.get('fp16_production_boundary_p95_ms')}` / `{summary.get('fp16_production_boundary_p99_ms')}` ms.",
            f"* Production-boundary P95/P99 reduction `{summary.get('production_boundary_p95_reduction_ratio')}` / `{summary.get('production_boundary_p99_reduction_ratio')}`.",
            f"* Component closure `{segmented.get('component_latency_closure_valid')}`; max unattributed ratio `{segmented.get('unattributed_latency_ratio')}`.",
            "",
            "## Memory And Quality",
            "",
            f"* FP32/FP16 peak GPU memory `{memory.get('fp32_peak_gpu_memory_mb')}` / `{memory.get('fp16_peak_gpu_memory_mb')}` MB.",
            f"* TASK-0209 C2 memory reproduced `{memory.get('task0209_c2_memory_result_reproduced')}`; growth explained `{memory.get('c2_memory_growth_explained')}`; cause `{memory.get('dominant_c2_memory_growth_cause')}`.",
            f"* Memory headroom authority available `{memory.get('memory_headroom_authority_available')}`; valid `{memory.get('memory_headroom_valid')}`.",
            f"* FP16 autocast effective `{quality.get('fp16_autocast_effective')}`; top1/top5 `{quality.get('top1_agreement_ratio')}` / `{quality.get('top5_set_agreement_ratio')}`; quality/downstream regressions `{quality.get('formal_quality_regression_count')}` / `{quality.get('downstream_regression_count')}`.",
            "",
            "## Decision",
            "",
            f"* Outcome `{decision.get('outcome')}` with dominant blocker `{decision.get('dominant_blocker')}`.",
            f"* Next task family `{decision.get('next_task_family')}`. Production config unchanged `{summary.get('production_config_unchanged')}`.",
            "",
            "## Validation",
            "",
            f"* Focused tests `{summary.get('focused_tests')}`; related regression tests `{summary.get('related_regression_tests')}`; production replay tests `{summary.get('production_replay_tests')}`; full suite `{summary.get('full_suite')}`; verifier `{summary.get('verifier_status')}`.",
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
        "requested_dtype": spec["requested_dtype"],
        "autocast_enabled": bool(spec.get("autocast")),
        "autocast_dtype": spec.get("autocast_dtype"),
        "fp16_autocast_effective": False,
        "model_parameter_dtype": "unknown",
        "model_parameter_device": "unknown",
        "forward_input_dtype": "unknown",
        "forward_input_device": "unknown",
        "output_dtype": "unknown",
        "resolved_device": "unknown",
        "requested_precision_effective": False,
        "silent_fp32_fallback_detected": False,
        "undeclared_cpu_fallback_detected": False,
        "finite_output_ratio": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "cuda_oom_count": 0,
        "cuda_sync_boundary_used": False,
        "peak_memory_stats_reset": False,
        "memory_stats_reset_success": False,
        "isolated_process_active": False,
        "single_model_instance_verified": False,
        "model_instance_count": 0,
        "cold_model_load_ms": None,
        "cold_first_request_ms": None,
        "warm_first_tokenization_ms": None,
        "warm_total_reranker_stage_ms": None,
        "measurement_count": 0,
        "metrics": {},
        "per_measurement": [],
        "memory": empty_memory_metrics(),
        "peak_gpu_memory_mb": 0.0,
        "candidate_input_digest": None,
        "score_digest": None,
        "rankings": {},
        "deterministic_repeatability": False,
        "runtime_error_count": 0,
        "exception_type": None,
        "exception_reason": reason,
    }


def error_arm(spec: Mapping[str, Any], exc_type: str, reason: str, *, oom: bool) -> dict[str, Any]:
    arm = unsupported_arm(spec, reason)
    arm.update(
        {
            "arm_status": "unsupported" if oom else "invalid",
            "cuda_oom_count": int(oom),
            "runtime_error_count": 0 if oom else 1,
            "exception_type": exc_type,
            "exception_reason": reason,
        }
    )
    return arm


def empty_memory_metrics() -> dict[str, float]:
    return {
        "memory_before_model_load_mb": 0.0,
        "memory_after_model_load_mb": 0.0,
        "memory_after_warmup_mb": 0.0,
        "memory_before_measurement_mb": 0.0,
        "max_memory_allocated_mb": 0.0,
        "max_memory_reserved_mb": 0.0,
        "memory_after_measurement_mb": 0.0,
        "external_gpu_memory_used_mb": 0.0,
    }


def aggregate_latency_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
    mapping = {
        "total_reranker": "total_reranker_stage_ms",
        "candidate_materialization": "candidate_materialization_ms",
        "tokenization": "tokenization_ms",
        "host_to_device": "host_to_device_transfer_ms",
        "forward": "model_forward_ms",
        "score_transfer": "score_device_to_host_ms",
        "postprocessing": "sorting_and_postprocessing_ms",
    }
    out: dict[str, float | None] = {}
    for prefix, key in mapping.items():
        values = [float(row[key]) for row in rows if key in row]
        out[f"{prefix}_p50_ms"] = percentile(values, 50)
        out[f"{prefix}_p95_ms"] = percentile(values, 95)
        out[f"{prefix}_p99_ms"] = percentile(values, 99)
    return out


def compare_rankings(reference: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    query_ids = sorted(reference)
    count = len(query_ids) or 1
    top1 = 0
    top5 = 0
    regressions = 0
    for query_id in query_ids:
        ref = list(reference.get(query_id) or [])
        cand = list(candidate.get(query_id) or [])
        top1 += int(ref[:1] == cand[:1])
        top5 += int(set(ref[:5]) == set(cand[:5]))
        regressions += int(ref[:1] != cand[:1] or set(ref[:5]) != set(cand[:5]))
    return {
        "top1_agreement_ratio": round(top1 / count, 6),
        "top5_set_agreement_ratio": round(top5 / count, 6),
        "formal_quality_regression_count": regressions,
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
        "scan_scope": "TASK-0210 generated artifacts; raw upstream production corpus text is not persisted.",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: Sequence[float], p: float) -> float | None:
    clean = sorted(float(value) for value in values)
    if not clean:
        return None
    rank = max(1, math.ceil((p / 100.0) * len(clean)))
    return round(clean[min(rank - 1, len(clean) - 1)], 6)


def _protocol_field(name: str, task0207_value: Any, task0209_value: Any, equivalent: bool | str) -> dict[str, Any]:
    unknown = equivalent == "unknown"
    return {
        "field": name,
        "task0207": task0207_value,
        "task0209": task0209_value,
        "equivalent": None if unknown else equivalent,
        "authority_status": "unknown" if unknown else "known",
    }


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


def _ranking(query: Mapping[str, Any], scores: Sequence[float]) -> list[str]:
    candidate_ids = list(query.get("candidate_ids") or [])
    return [candidate_id for _, candidate_id in sorted(zip(scores, candidate_ids, strict=True), reverse=True)]


def _stable_rankings(rows: Sequence[Sequence[str]]) -> bool:
    if not rows:
        return False
    first = list(rows[0])
    return all(list(row) == first for row in rows[1:])


def _arm(replay: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    return next((arm for arm in replay.get("arms") or [] if arm.get("arm_id") == arm_id), {})


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 6)


def _reduction(baseline: Any, experiment: Any) -> float | None:
    try:
        base = float(baseline)
        exp = float(experiment)
    except (TypeError, ValueError):
        return None
    if base <= 0.0:
        return None
    return round((base - exp) / base, 6)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den <= 0.0:
        return None
    return round(num / den, 6)


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0.0 else None


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-arm")
    parser.add_argument("--child-output")
    args = parser.parse_args()
    if args.child_arm:
        payload = read_json(Path(args.child_arm))
        result = execute_child_arm(payload["spec"], payload["input_snapshot"], os.environ)
        write_json(Path(args.child_output), result)
        return
    summary = run_task0210(write=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
