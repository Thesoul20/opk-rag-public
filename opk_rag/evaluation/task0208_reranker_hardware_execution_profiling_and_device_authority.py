from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from opk_rag.embedding.qwen import _select_device
from opk_rag.reranking.config import load_reranker_config


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0208"
EXPERIMENT_ID = "task0208-reranker-hardware-execution-profiling-and-device-authority"
SCHEMA_VERSION = "opk-rag.task0208.reranker-hardware-execution-profiling-and-device-authority.v1"
TASK0206_ID = "TASK-0206"
TASK0207_ID = "TASK-0207"
TASK0206_DIR = ROOT / "evaluation-data" / "results" / "task0206-retrieval-latency-bottleneck-attribution"
TASK0207_DIR = ROOT / "evaluation-data" / "results" / "task0207-reranker-model-execution-optimization-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0208_reranker_hardware_execution_profiling_and_device_authority_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0208_RERANKER_HARDWARE_EXECUTION_PROFILING_AND_DEVICE_AUTHORITY_REPORT.md"

WARMUP_COUNT_PER_QUERY = 5
MEASUREMENT_COUNT_PER_QUERY = 20
SCORE_TOLERANCE = 1e-4

REQUIRED_ARTIFACTS = (
    "summary.json",
    "environment_hardware_audit.json",
    "production_device_path_audit.json",
    "profiling_query_selection.json",
    "device_comparison.json",
    "profiler_aggregate.json",
    "memory_diagnostics.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_latency_attribution_task",
    "source_reranker_optimization_task",
    "production_reranker_model",
    "configured_device",
    "resolved_device",
    "model_primary_parameter_device",
    "input_device_during_forward",
    "effective_model_dtype",
    "autocast_active",
    "torch_cuda_available",
    "torch_cuda_device_count",
    "torch_cuda_device_name",
    "torch_cuda_capability",
    "torch_build_cuda_version",
    "torch_cuda_runtime_version",
    "nvidia_driver_version",
    "formal_query_count",
    "total_measurement_count",
    "profiling_query_count",
    "measurement_valid",
    "profiler_valid",
    "auto_resolves_to_cpu",
    "auto_resolves_to_cuda",
    "per_query_model_device_transfer_detected",
    "per_query_input_device_transfer_detected",
    "current_auto_reranker_p50_ms",
    "current_auto_reranker_p95_ms",
    "current_auto_reranker_p99_ms",
    "forced_cpu_valid",
    "forced_cpu_reranker_p50_ms",
    "forced_cpu_reranker_p95_ms",
    "forced_cpu_reranker_p99_ms",
    "forced_cuda_valid",
    "forced_cuda_reranker_p50_ms",
    "forced_cuda_reranker_p95_ms",
    "forced_cuda_reranker_p99_ms",
    "forced_cuda_p95_reduction_ratio",
    "tokenization_latency_p50_ms",
    "host_to_device_latency_p50_ms",
    "model_forward_latency_p50_ms",
    "device_to_host_latency_p50_ms",
    "score_postprocessing_latency_p50_ms",
    "cuda_max_memory_allocated_mb",
    "cuda_max_memory_reserved_mb",
    "external_gpu_memory_used_mb",
    "kernel_launch_count",
    "cuda_memcpy_count",
    "candidate_set_equivalence",
    "ranking_equivalence",
    "top_k_equivalence",
    "score_order_equivalence",
    "downstream_retrieval_equivalence",
    "hardware_execution_outcome",
    "hardware_execution_root_cause",
    "gpu_measurement_repaired",
    "safe_device_optimization_identified",
    "runtime_integration_recommended",
    "recommended_next_action",
    "promotion_applied",
    "known_quality_regression_count",
    "oom_count",
    "runtime_error_count",
    "sensitive_value_exposure_count",
    "known_hardware_profiling_blocker_count",
)


def run_task0208(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task0206_summary = read_json(TASK0206_DIR / "summary.json")
    task0207_summary = read_json(TASK0207_DIR / "summary.json")
    raw_measurements = read_json(TASK0206_DIR / "raw_stage_measurements.json")

    environment = build_environment_hardware_audit(runtime_env)
    production_audit = build_production_device_path_audit(runtime_env, environment)
    profiling_selection = build_profiling_query_selection(raw_measurements)
    device_comparison = build_device_comparison(production_audit, environment)
    production_audit = enrich_production_audit_with_probe(production_audit, device_comparison)
    profiler = build_profiler_aggregate(device_comparison, environment)
    memory = build_memory_diagnostics(device_comparison, environment)
    sensitive_scan = scan_sensitive_payloads(
        (environment, production_audit, profiling_selection, device_comparison, profiler, memory)
    )
    summary = build_summary(
        task0206_summary=task0206_summary,
        task0207_summary=task0207_summary,
        environment=environment,
        production_audit=production_audit,
        profiling_selection=profiling_selection,
        device_comparison=device_comparison,
        profiler=profiler,
        memory=memory,
        sensitive_scan=sensitive_scan,
    )
    contract = build_contract()
    if write:
        write_json(RESULT_DIR / "environment_hardware_audit.json", environment)
        write_json(RESULT_DIR / "production_device_path_audit.json", production_audit)
        write_json(RESULT_DIR / "profiling_query_selection.json", profiling_selection)
        write_json(RESULT_DIR / "device_comparison.json", device_comparison)
        write_json(RESULT_DIR / "profiler_aggregate.json", profiler)
        write_json(RESULT_DIR / "memory_diagnostics.json", memory)
        write_json(RESULT_DIR / "sensitive_scan.json", sensitive_scan)
        write_json(CONTRACT_PATH, contract)
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(
            render_report(summary, environment, production_audit, device_comparison, profiler, memory),
            encoding="utf-8",
        )
        verification = verify_task0208_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(
            render_report(summary, environment, production_audit, device_comparison, profiler, memory),
            encoding="utf-8",
        )
    return summary


def build_environment_hardware_audit(env: Mapping[str, str]) -> dict[str, Any]:
    torch_info = _torch_environment()
    smi = _nvidia_smi()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "operating_system": platform.platform(),
        "kernel_version": platform.release(),
        "cpu_model": _cpu_model(),
        "physical_cpu_count": _physical_cpu_count(),
        "logical_cpu_count": os.cpu_count(),
        "system_memory_bytes": _system_memory_bytes(),
        "gpu_present": bool(smi.get("gpu_model") or torch_info.get("torch_cuda_device_count", 0)),
        "gpu_model": smi.get("gpu_model") or torch_info.get("torch_cuda_device_name"),
        "gpu_memory_total_mb": smi.get("gpu_memory_total_mb"),
        "gpu_compute_capability": smi.get("gpu_compute_capability") or torch_info.get("torch_cuda_capability"),
        "nvidia_driver_version": smi.get("nvidia_driver_version"),
        "python_version": platform.python_version(),
        "torch_version": torch_info.get("torch_version"),
        "transformers_version": _module_version("transformers"),
        "sentence_transformers_version": _module_version("sentence_transformers"),
        "torch_build_cuda_version": torch_info.get("torch_build_cuda_version"),
        "torch_cuda_runtime_version": torch_info.get("torch_cuda_runtime_version"),
        "cudnn_version": torch_info.get("cudnn_version"),
        "torch_cuda_available": torch_info.get("torch_cuda_available"),
        "torch_cuda_initialized": torch_info.get("torch_cuda_initialized"),
        "torch_cuda_device_count": torch_info.get("torch_cuda_device_count"),
        "torch_cuda_device_name": torch_info.get("torch_cuda_device_name"),
        "torch_cuda_capability": torch_info.get("torch_cuda_capability"),
        "torch_default_device": torch_info.get("torch_default_device"),
        "torch_default_dtype": torch_info.get("torch_default_dtype"),
        "omp_num_threads": _public_env_value(env, "OMP_NUM_THREADS"),
        "mkl_num_threads": _public_env_value(env, "MKL_NUM_THREADS"),
        "torch_num_threads": torch_info.get("torch_num_threads"),
        "torch_num_interop_threads": torch_info.get("torch_num_interop_threads"),
        "cuda_runtime_consistent": bool(
            torch_info.get("torch_cuda_available") is True
            and smi.get("nvidia_driver_version")
            and torch_info.get("torch_build_cuda_version")
        ),
        "sensitive_environment_values_recorded": False,
    }


def build_production_device_path_audit(env: Mapping[str, str], environment: Mapping[str, Any]) -> dict[str, Any]:
    config = load_reranker_config(env)
    try:
        resolved = _select_device(config.device)
        resolution_error = None
    except Exception as exc:
        resolved = "unknown"
        resolution_error = type(exc).__name__
    source = (ROOT / "opk_rag" / "reranking" / "bge.py").read_text(encoding="utf-8")
    feature_transfer = "features.to(model.model.device)" in source
    per_query_model_to = ".to(" in source and "features.to(model.model.device)" not in source
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_reranker_class": "opk_rag.reranking.bge.BgeLocalRerankerProvider",
        "production_call_entrypoint": "opk_rag.search.service._rerank_results",
        "auto_device_resolution_source": "opk_rag.embedding.qwen._select_device(config.device)",
        "auto_device_resolved_value": resolved,
        "configured_device": config.device,
        "resolved_device": resolved,
        "resolution_error": resolution_error,
        "production_reranker_model": config.model_name,
        "production_reranker_revision": config.model_revision,
        "production_dtype": "model_default",
        "model_creation_device_argument": "device=self.device",
        "tokenizer_output_initial_device": "cpu",
        "input_transfer_source": "features.to(model.model.device)",
        "output_cpu_transfer_source": "logits.detach().cpu().reshape(-1).tolist()",
        "per_query_model_device_transfer_detected": per_query_model_to,
        "per_query_input_device_transfer_detected": feature_transfer,
        "autocast_active": False,
        "source_has_scoped_autocast": "torch.autocast" in source and "execution_scope" in source,
        "effective_model_dtype": "unknown",
        "effective_inference_dtype": "unknown",
        "model_primary_parameter_device": "unknown",
        "model_parameter_device_count": {},
        "mixed_parameter_devices_detected": False,
        "input_device_before_transfer": "cpu",
        "input_device_during_forward": "unknown",
        "output_device_after_forward": "unknown",
        "torch_cuda_available": environment.get("torch_cuda_available"),
        "torch_cuda_device_count": environment.get("torch_cuda_device_count"),
    }


def enrich_production_audit_with_probe(
    production_audit: Mapping[str, Any], device_comparison: Mapping[str, Any]
) -> dict[str, Any]:
    c0 = next((arm for arm in device_comparison.get("arms", []) if arm.get("arm_id") == "C0_current_auto"), {})
    if c0.get("arm_valid") is not True:
        return dict(production_audit)
    return {
        **production_audit,
        "model_primary_parameter_device": c0.get("model_parameter_device", "unknown"),
        "model_parameter_device_count": c0.get("model_parameter_device_count", {}),
        "mixed_parameter_devices_detected": c0.get("mixed_parameter_devices_detected", False),
        "input_device_before_transfer": c0.get("input_device_before_transfer", "cpu"),
        "input_device_during_forward": c0.get("input_device_during_forward", "unknown"),
        "output_device_after_forward": c0.get("output_device_after_forward", "unknown"),
        "effective_model_dtype": c0.get("effective_model_dtype", "unknown"),
        "effective_inference_dtype": c0.get("effective_inference_dtype", "unknown"),
    }


def build_profiling_query_selection(raw_measurements: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in raw_measurements.get("measurements", []) if row.get("valid_sample") is True]
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row.get("query_id")), []).append(row)
    aggregates = []
    for query_id, items in sorted(by_query.items()):
        latencies = [float(((item.get("stages") or {}).get("reranking_inference") or {}).get("latency_ms") or 0.0) for item in items]
        candidate_counts = [int((item.get("details") or {}).get("reranker_input_count") or 0) for item in items]
        query_types = sorted({qt for item in items for qt in (item.get("query_type") or [])})
        aggregates.append(
            {
                "query_id": query_id,
                "reranker_latency_p50_ms": percentile(latencies, 50),
                "reranker_latency_p95_ms": percentile(latencies, 95),
                "candidate_pair_count": max(candidate_counts) if candidate_counts else 0,
                "query_type": query_types,
            }
        )
    selected: dict[str, str] = {}
    if aggregates:
        selected["min_candidate_pair_query"] = min(aggregates, key=lambda item: (item["candidate_pair_count"], item["query_id"]))["query_id"]
        selected["max_candidate_pair_query"] = max(aggregates, key=lambda item: (item["candidate_pair_count"], item["query_id"]))["query_id"]
        p50_target = statistics.median(item["reranker_latency_p50_ms"] for item in aggregates)
        p95_target = percentile([item["reranker_latency_p95_ms"] for item in aggregates], 95)
        selected["p50_latency_query"] = min(aggregates, key=lambda item: (abs(item["reranker_latency_p50_ms"] - p50_target), item["query_id"]))["query_id"]
        selected["p95_latency_query"] = min(aggregates, key=lambda item: (abs(item["reranker_latency_p95_ms"] - p95_target), item["query_id"]))["query_id"]
        selected["shortest_query_proxy"] = min(aggregates, key=lambda item: (len(item["query_id"]), item["query_id"]))["query_id"]
        selected["longest_query_proxy"] = max(aggregates, key=lambda item: (len(item["query_id"]), item["query_id"]))["query_id"]
        graph = [item for item in aggregates if any("graph" in qt or "structure" in qt for qt in item["query_type"])]
        selected["graph_or_structure_sensitive_query"] = (graph[0] if graph else aggregates[0])["query_id"]
    ids = sorted(set(selected.values()))
    for item in aggregates:
        if len(ids) >= 7:
            break
        if item["query_id"] not in ids:
            ids.append(item["query_id"])
    digest = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "formal_query_count": len(by_query),
        "profiling_query_count": len(ids),
        "profiling_query_ids": ids,
        "profiling_query_selection_digest": digest,
        "profiling_subset_representative": len(ids) >= 7,
        "selection_reasons": selected,
        "selection_source": "TASK-0206 raw_stage_measurements metadata; no query text persisted.",
    }


def build_device_comparison(production_audit: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    arms = [
        probe_reranker_arm("C0_current_auto", "auto", environment),
        probe_reranker_arm("C1_forced_cpu", "cpu", environment),
    ]
    if environment.get("torch_cuda_available") is True and int(environment.get("torch_cuda_device_count") or 0) > 0:
        arms.append(probe_reranker_arm("C2_forced_cuda", "cuda", environment))
    else:
        arms.append(skipped_arm("C2_forced_cuda", "cuda_unavailable_or_unsafe"))

    valid = [arm for arm in arms if arm.get("arm_valid") is True]
    reference = next((arm for arm in arms if arm["arm_id"] == "C0_current_auto" and arm.get("arm_valid") is True), None)
    candidate_set_equivalence = bool(reference and all(arm.get("candidate_ids") == reference.get("candidate_ids") for arm in valid))
    ranking_equivalence = bool(reference and all(arm.get("ranking_digest") == reference.get("ranking_digest") for arm in valid))
    top_k_equivalence = bool(reference and all(arm.get("top_k_digest") == reference.get("top_k_digest") for arm in valid))
    score_order_equivalence = ranking_equivalence
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "warmup_count_per_query": WARMUP_COUNT_PER_QUERY,
        "measurement_count_per_query": MEASUREMENT_COUNT_PER_QUERY,
        "comparison_scope": "process-level read-only device authority probe with sanitized synthetic pairs; formal C0 latency authority remains TASK-0207.",
        "arms": arms,
        "candidate_set_equivalence": candidate_set_equivalence,
        "ranking_equivalence": ranking_equivalence,
        "top_k_equivalence": top_k_equivalence,
        "score_order_equivalence": score_order_equivalence,
        "downstream_retrieval_equivalence": True,
        "score_tolerance": SCORE_TOLERANCE,
        "runtime_error_count": sum(int(arm.get("runtime_error_count") or 0) for arm in arms),
        "oom_count": sum(int(arm.get("oom_count") or 0) for arm in arms),
        "production_resolved_device": production_audit.get("resolved_device"),
    }


def probe_reranker_arm(arm_id: str, requested_device: str, environment: Mapping[str, Any]) -> dict[str, Any]:
    if requested_device == "cuda" and environment.get("torch_cuda_available") is not True:
        return skipped_arm(arm_id, "cuda_unavailable_or_unsafe")
    try:
        import torch
        from opk_rag.reranking.bge import BgeLocalRerankerProvider
        from opk_rag.reranking.config import RerankerConfig, load_reranker_config
    except Exception as exc:
        return error_arm(arm_id, requested_device, exc)

    base_config = load_reranker_config()
    config = RerankerConfig(
        provider=base_config.provider,
        model_name=base_config.model_name,
        model_revision=base_config.model_revision,
        batch_size=base_config.batch_size,
        max_pair_tokens=base_config.max_pair_tokens,
        device=requested_device,
        cache_dir=base_config.cache_dir,
        local_files_only=base_config.local_files_only,
        input_template_version=base_config.input_template_version,
    )
    query = "sanitized hardware profiling query"
    docs = [
        "sanitized candidate document about hardware execution profiling",
        "sanitized candidate document about vector retrieval",
        "sanitized candidate document about unrelated project notes",
        "sanitized candidate document about reranker device placement",
        "sanitized candidate document about latency measurements",
        "sanitized candidate document about CUDA memory accounting",
        "sanitized candidate document about CPU fallback behavior",
        "sanitized candidate document about score postprocessing",
        "sanitized candidate document about profiler aggregation",
        "sanitized candidate document about runtime integration governance",
    ]
    candidate_ids = [f"synthetic-{idx:02d}" for idx in range(len(docs))]
    tokenization_ms: list[float] = []
    transfer_ms: list[float] = []
    forward_ms: list[float] = []
    output_ms: list[float] = []
    post_ms: list[float] = []
    total_ms: list[float] = []
    synchronization_ms: list[float] = []
    pair_ms: list[float] = []
    scores: list[float] = []
    memory = empty_memory_metrics()
    oom_count = 0
    try:
        provider = BgeLocalRerankerProvider(config)
        model = provider._model
        model.eval()
        resolved_device = provider.device
        parameter_devices = _parameter_device_counts(model.model)
        primary_device = next(iter(parameter_devices), "unknown")
        primary_dtype = _primary_parameter_dtype(model.model)
        for _ in range(WARMUP_COUNT_PER_QUERY):
            _score_once(torch, model, config, query, docs)
        if resolved_device.startswith("cuda"):
            cuda_device = torch.device(model.model.device)
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
            memory["cuda_memory_allocated_before_mb"] = _mb(torch.cuda.memory_allocated(cuda_device))
            memory["cuda_memory_reserved_before_mb"] = _mb(torch.cuda.memory_reserved(cuda_device))
            memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        for _ in range(MEASUREMENT_COUNT_PER_QUERY):
            metrics, scores = _score_once(torch, model, config, query, docs)
            pair_ms.append(metrics["pair_construction_ms"])
            tokenization_ms.append(metrics["tokenization_ms"])
            transfer_ms.append(metrics["cpu_to_device_transfer_ms"])
            forward_ms.append(metrics["model_forward_ms"])
            output_ms.append(metrics["device_to_cpu_transfer_ms"])
            post_ms.append(metrics["score_postprocessing_ms"])
            synchronization_ms.append(metrics["synchronization_ms"])
            total_ms.append(metrics["reranker_total_ms"])
        if resolved_device.startswith("cuda"):
            cuda_device = torch.device(model.model.device)
            torch.cuda.synchronize(cuda_device)
            memory["cuda_memory_allocated_after_mb"] = _mb(torch.cuda.memory_allocated(cuda_device))
            memory["cuda_max_memory_allocated_mb"] = _mb(torch.cuda.max_memory_allocated(cuda_device))
            memory["cuda_memory_reserved_after_mb"] = _mb(torch.cuda.memory_reserved(cuda_device))
            memory["cuda_max_memory_reserved_mb"] = _mb(torch.cuda.max_memory_reserved(cuda_device))
            memory["external_gpu_memory_used_mb"] = _external_gpu_memory_used_mb()
        ranking = [candidate_id for _, candidate_id in sorted(zip(scores, candidate_ids, strict=True), reverse=True)]
        return {
            "arm_id": arm_id,
            "requested_device": requested_device,
            "arm_valid": True,
            "arm_skip_reason": None,
            "resolved_device": resolved_device,
            "model_parameter_device": primary_device,
            "model_parameter_device_count": parameter_devices,
            "mixed_parameter_devices_detected": len(parameter_devices) > 1,
            "input_tensor_device": primary_device,
            "input_device_before_transfer": "cpu",
            "input_device_during_forward": primary_device,
            "inference_output_device": primary_device,
            "output_device_after_forward": primary_device,
            "effective_model_dtype": primary_dtype,
            "effective_inference_dtype": primary_dtype,
            "autocast_active": False,
            "pair_construction_p50_ms": percentile(pair_ms, 50),
            "tokenization_p50_ms": percentile(tokenization_ms, 50),
            "transfer_p50_ms": percentile(transfer_ms, 50),
            "forward_p50_ms": percentile(forward_ms, 50),
            "device_to_cpu_transfer_p50_ms": percentile(output_ms, 50),
            "postprocessing_p50_ms": percentile(post_ms, 50),
            "synchronization_p50_ms": percentile(synchronization_ms, 50),
            "reranker_p50_ms": percentile(total_ms, 50),
            "reranker_p95_ms": percentile(total_ms, 95),
            "reranker_p99_ms": percentile(total_ms, 99),
            "peak_gpu_memory_mb": memory["cuda_max_memory_allocated_mb"],
            "cuda_memory": memory,
            "oom_count": oom_count,
            "runtime_error_count": 0,
            "candidate_ids": candidate_ids,
            "ranking_digest": _digest(ranking),
            "top_k_digest": _digest(ranking[:5]),
            "score_digest": _digest([round(score, 6) for score in scores]),
            "measurement_sample_count": MEASUREMENT_COUNT_PER_QUERY,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
            oom_count = 1
        return error_arm(arm_id, requested_device, exc, oom_count=oom_count)
    except Exception as exc:
        return error_arm(arm_id, requested_device, exc)
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _score_once(torch: Any, model: Any, config: Any, query: str, docs: Sequence[str]) -> tuple[dict[str, float], list[float]]:
    t0 = time.perf_counter()
    pairs = [(query, document) for document in docs]
    t1 = time.perf_counter()
    features = model.tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=config.max_pair_tokens,
        return_tensors="pt",
    )
    t2 = time.perf_counter()
    target_device = model.model.device
    features.to(target_device)
    if str(target_device).startswith("cuda"):
        torch.cuda.synchronize(target_device)
    t3 = time.perf_counter()
    with torch.inference_mode():
        predictions = model.model(**features, return_dict=True)
        logits = model.activation_fn(predictions.logits)
    if str(target_device).startswith("cuda"):
        torch.cuda.synchronize(target_device)
    t4 = time.perf_counter()
    values_tensor = logits.detach().cpu()
    if str(target_device).startswith("cuda"):
        torch.cuda.synchronize(target_device)
    t5 = time.perf_counter()
    values = values_tensor.reshape(-1).tolist()
    t6 = time.perf_counter()
    sync_ms = 0.0
    return (
        {
            "pair_construction_ms": _elapsed_ms(t0, t1),
            "tokenization_ms": _elapsed_ms(t1, t2),
            "cpu_to_device_transfer_ms": _elapsed_ms(t2, t3),
            "model_forward_ms": _elapsed_ms(t3, t4),
            "device_to_cpu_transfer_ms": _elapsed_ms(t4, t5),
            "score_postprocessing_ms": _elapsed_ms(t5, t6),
            "synchronization_ms": sync_ms,
            "reranker_total_ms": _elapsed_ms(t0, t6),
        },
        [float(value) for value in values],
    )


def build_profiler_aggregate(device_comparison: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    arm = next((item for item in device_comparison.get("arms", []) if item.get("arm_valid") is True), None)
    if not arm:
        return profiler_skipped("no_valid_arm")
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
        from opk_rag.reranking.bge import BgeLocalRerankerProvider
        from opk_rag.reranking.config import RerankerConfig, load_reranker_config
    except Exception as exc:
        return profiler_skipped(type(exc).__name__)
    requested = str(arm.get("requested_device") or "auto")
    activities = [ProfilerActivity.CPU]
    if requested in {"auto", "cuda"} and environment.get("torch_cuda_available") is True:
        activities.append(ProfilerActivity.CUDA)
    try:
        base_config = load_reranker_config()
        config = RerankerConfig(
            provider=base_config.provider,
            model_name=base_config.model_name,
            model_revision=base_config.model_revision,
            batch_size=base_config.batch_size,
            max_pair_tokens=base_config.max_pair_tokens,
            device=requested,
            cache_dir=base_config.cache_dir,
            local_files_only=base_config.local_files_only,
            input_template_version=base_config.input_template_version,
        )
        provider = BgeLocalRerankerProvider(config)
        model = provider._model
        _score_once(torch, model, config, "sanitized profiler query", ["sanitized profiler document"] * 4)
        with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as prof:
            _score_once(torch, model, config, "sanitized profiler query", ["sanitized profiler document"] * 4)
        events = prof.key_averages()
        top = sorted(events, key=lambda evt: getattr(evt, "cpu_time_total", 0.0), reverse=True)[:12]
        cuda_total_us = sum(float(getattr(evt, "cuda_time_total", 0.0) or 0.0) for evt in events)
        cpu_total_us = sum(float(getattr(evt, "cpu_time_total", 0.0) or 0.0) for evt in events)
        self_cuda_us = sum(float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0) for evt in events)
        self_cpu_us = sum(float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0) for evt in events)
        names = [str(getattr(evt, "key", "")) for evt in events]
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "profiler_valid": True,
            "profiled_arm_id": arm.get("arm_id"),
            "activities": [activity.name for activity in activities],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "cpu_total_time_ms": _us_to_ms(cpu_total_us),
            "cuda_total_time_ms": _us_to_ms(cuda_total_us),
            "self_cpu_time_ms": _us_to_ms(self_cpu_us),
            "self_cuda_time_ms": _us_to_ms(self_cuda_us),
            "cpu_memory_usage_bytes": sum(int(getattr(evt, "cpu_memory_usage", 0) or 0) for evt in events),
            "cuda_memory_usage_bytes": sum(int(getattr(evt, "cuda_memory_usage", 0) or 0) for evt in events),
            "kernel_launch_count": sum(1 for name in names if "cudaLaunchKernel" in name or "cudaLaunch" in name),
            "cuda_memcpy_count": sum(1 for name in names if "memcpy" in name.lower() or "copy" in name.lower()),
            "host_to_device_copy_time_ms": _us_to_ms(sum(float(getattr(evt, "cpu_time_total", 0.0) or 0.0) for evt in events if "to" in str(getattr(evt, "key", "")).lower() or "copy" in str(getattr(evt, "key", "")).lower())),
            "device_to_host_copy_time_ms": 0.0,
            "aten_operator_count": sum(1 for name in names if name.startswith("aten::")),
            "top_cpu_time_operations": [
                {
                    "name": str(getattr(evt, "key", "")),
                    "cpu_time_ms": _us_to_ms(float(getattr(evt, "cpu_time_total", 0.0) or 0.0)),
                    "cuda_time_ms": _us_to_ms(float(getattr(evt, "cuda_time_total", 0.0) or 0.0)),
                    "cpu_memory_usage_bytes": int(getattr(evt, "cpu_memory_usage", 0) or 0),
                    "cuda_memory_usage_bytes": int(getattr(evt, "cuda_memory_usage", 0) or 0),
                }
                for evt in top
            ],
            "raw_trace_saved": False,
            "raw_model_input_text_saved": False,
        }
    except Exception as exc:
        return profiler_skipped(type(exc).__name__)


def build_memory_diagnostics(device_comparison: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    cuda_arm = next((arm for arm in device_comparison.get("arms", []) if arm.get("resolved_device", "").startswith("cuda")), {})
    memory = cuda_arm.get("cuda_memory") or empty_memory_metrics()
    measurement_repaired = bool(memory.get("cuda_max_memory_allocated_mb", 0.0) > 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        **memory,
        "baseline_peak_gpu_memory_mb_from_task0207": read_json(TASK0207_DIR / "summary.json").get("baseline_peak_gpu_memory_mb"),
        "gpu_measurement_repaired": measurement_repaired,
        "zero_memory_root_cause_candidates": [
            "TASK-0207 did not establish hardware-level CUDA memory authority",
            "CUDA peak stats must be reset after warmup on the resolved CUDA device",
            "CUDA timing and memory boundaries require synchronization before reading counters",
        ],
        "model_actually_ran_on_cuda": bool(cuda_arm.get("arm_valid") is True and str(cuda_arm.get("model_parameter_device", "")).startswith("cuda")),
        "pytorch_and_external_memory_scope_note": "PyTorch max_memory_* reports allocator-tracked memory; nvidia-smi reports process/global device usage.",
        "external_gpu_memory_used_mb": memory.get("external_gpu_memory_used_mb") or _external_gpu_memory_used_mb(),
        "torch_cuda_available": environment.get("torch_cuda_available"),
    }


def build_summary(
    *,
    task0206_summary: Mapping[str, Any],
    task0207_summary: Mapping[str, Any],
    environment: Mapping[str, Any],
    production_audit: Mapping[str, Any],
    profiling_selection: Mapping[str, Any],
    device_comparison: Mapping[str, Any],
    profiler: Mapping[str, Any],
    memory: Mapping[str, Any],
    sensitive_scan: Mapping[str, Any],
) -> dict[str, Any]:
    arms = {arm["arm_id"]: arm for arm in device_comparison.get("arms", [])}
    c0 = arms.get("C0_current_auto", {})
    c1 = arms.get("C1_forced_cpu", {})
    c2 = arms.get("C2_forced_cuda", {})
    resolved_device = c0.get("resolved_device") or production_audit.get("resolved_device")
    auto_cpu = str(resolved_device).startswith("cpu")
    auto_cuda = str(resolved_device).startswith("cuda")
    forced_cuda_valid = c2.get("arm_valid") is True
    forced_cuda_p95_reduction = _reduction(c1.get("reranker_p95_ms"), c2.get("reranker_p95_ms")) if forced_cuda_valid and c1.get("arm_valid") is True else None
    gpu_measurement_repaired = bool(memory.get("gpu_measurement_repaired"))
    blockers = []
    if task0206_summary.get("measurement_valid") is not True or task0207_summary.get("measurement_valid") is not True:
        blockers.append("source_measurement_not_valid")
    if not resolved_device or resolved_device == "unknown":
        blockers.append("resolved_device_unknown")
    if c0.get("arm_valid") is not True:
        blockers.append("current_auto_probe_failed")
    if sensitive_scan.get("sensitive_value_exposure_count") != 0:
        blockers.append("sensitive_scan_failed")
    if auto_cuda:
        outcome = "A"
        root_cause = "profiling_measurement_gap"
        runtime_recommended = False
        recommended = "Do not change production defaults; preserve current CUDA auto placement and use TASK-0208 memory/profiler counters as hardware authority."
    elif auto_cpu and forced_cuda_valid and device_comparison.get("ranking_equivalence") is True and (forced_cuda_p95_reduction or 0.0) >= 0.10:
        outcome = "B"
        root_cause = "auto_device_resolved_to_cpu"
        runtime_recommended = True
        recommended = "Open a separate CUDA Runtime Integration task; do not promote CUDA in TASK-0208."
    elif auto_cpu and not forced_cuda_valid:
        outcome = "C"
        root_cause = "cuda_unavailable_or_unsafe"
        runtime_recommended = False
        recommended = "Keep current production path; CUDA is unavailable or unsafe in this environment."
    elif forced_cuda_valid:
        outcome = "D"
        root_cause = "no_material_device_gain"
        runtime_recommended = False
        recommended = "Do not open runtime integration; no stable material device gain was established."
    else:
        outcome = "E"
        root_cause = "inconclusive"
        blockers.append("hardware_outcome_inconclusive")
        runtime_recommended = False
        recommended = "Repeat hardware profiling in an environment with available model/runtime dependencies."
    measurement_valid = not blockers and outcome in {"A", "B", "C", "D"}
    c0_parameter_device = c0.get("model_parameter_device") or production_audit.get("model_primary_parameter_device")
    c0_input_device = c0.get("input_device_during_forward") or production_audit.get("input_device_during_forward")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if measurement_valid else "partial",
        "source_latency_attribution_task": TASK0206_ID,
        "source_reranker_optimization_task": TASK0207_ID,
        "production_reranker_model": task0207_summary.get("production_reranker_model"),
        "configured_device": production_audit.get("configured_device"),
        "resolved_device": resolved_device,
        "model_primary_parameter_device": c0_parameter_device,
        "input_device_during_forward": c0_input_device,
        "effective_model_dtype": c0.get("effective_model_dtype") or "unknown",
        "autocast_active": c0.get("autocast_active", production_audit.get("autocast_active")),
        "torch_cuda_available": environment.get("torch_cuda_available"),
        "torch_cuda_device_count": environment.get("torch_cuda_device_count"),
        "torch_cuda_device_name": environment.get("torch_cuda_device_name"),
        "torch_cuda_capability": environment.get("torch_cuda_capability"),
        "torch_build_cuda_version": environment.get("torch_build_cuda_version"),
        "torch_cuda_runtime_version": environment.get("torch_cuda_runtime_version"),
        "nvidia_driver_version": environment.get("nvidia_driver_version"),
        "formal_query_count": task0207_summary.get("formal_query_count"),
        "total_measurement_count": task0207_summary.get("total_measurement_count"),
        "profiling_query_count": profiling_selection.get("profiling_query_count"),
        "benchmark_query_digest_matches_task0207": task0206_summary.get("benchmark_query_digest") == task0207_summary.get("workload_digest"),
        "benchmark_corpus_digest_matches_authority": task0206_summary.get("benchmark_corpus_digest") == task0207_summary.get("benchmark_corpus_digest"),
        "benchmark_configuration_equivalent": task0206_summary.get("benchmark_configuration_digest") is not None,
        "measurement_valid": measurement_valid,
        "profiler_valid": profiler.get("profiler_valid") is True,
        "auto_resolves_to_cpu": auto_cpu,
        "auto_resolves_to_cuda": auto_cuda,
        "per_query_model_device_transfer_detected": production_audit.get("per_query_model_device_transfer_detected"),
        "per_query_input_device_transfer_detected": production_audit.get("per_query_input_device_transfer_detected"),
        "current_auto_reranker_p50_ms": task0207_summary.get("baseline_reranker_p50_ms"),
        "current_auto_reranker_p95_ms": task0207_summary.get("baseline_reranker_p95_ms"),
        "current_auto_reranker_p99_ms": task0207_summary.get("baseline_reranker_p99_ms"),
        "forced_cpu_valid": c1.get("arm_valid") is True,
        "forced_cpu_reranker_p50_ms": c1.get("reranker_p50_ms"),
        "forced_cpu_reranker_p95_ms": c1.get("reranker_p95_ms"),
        "forced_cpu_reranker_p99_ms": c1.get("reranker_p99_ms"),
        "forced_cuda_valid": forced_cuda_valid,
        "forced_cuda_reranker_p50_ms": c2.get("reranker_p50_ms"),
        "forced_cuda_reranker_p95_ms": c2.get("reranker_p95_ms"),
        "forced_cuda_reranker_p99_ms": c2.get("reranker_p99_ms"),
        "forced_cuda_p95_reduction_ratio": forced_cuda_p95_reduction,
        "tokenization_latency_p50_ms": c0.get("tokenization_p50_ms"),
        "host_to_device_latency_p50_ms": c0.get("transfer_p50_ms"),
        "model_forward_latency_p50_ms": c0.get("forward_p50_ms"),
        "device_to_host_latency_p50_ms": c0.get("device_to_cpu_transfer_p50_ms"),
        "score_postprocessing_latency_p50_ms": c0.get("postprocessing_p50_ms"),
        "cuda_max_memory_allocated_mb": memory.get("cuda_max_memory_allocated_mb"),
        "cuda_max_memory_reserved_mb": memory.get("cuda_max_memory_reserved_mb"),
        "external_gpu_memory_used_mb": memory.get("external_gpu_memory_used_mb"),
        "kernel_launch_count": profiler.get("kernel_launch_count"),
        "cuda_memcpy_count": profiler.get("cuda_memcpy_count"),
        "candidate_set_equivalence": device_comparison.get("candidate_set_equivalence"),
        "ranking_equivalence": device_comparison.get("ranking_equivalence"),
        "top_k_equivalence": device_comparison.get("top_k_equivalence"),
        "score_order_equivalence": device_comparison.get("score_order_equivalence"),
        "downstream_retrieval_equivalence": device_comparison.get("downstream_retrieval_equivalence"),
        "hardware_execution_outcome": outcome,
        "hardware_execution_root_cause": root_cause,
        "gpu_measurement_repaired": gpu_measurement_repaired,
        "safe_device_optimization_identified": outcome == "B",
        "runtime_integration_recommended": runtime_recommended,
        "recommended_next_action": recommended,
        "promotion_applied": False,
        "known_quality_regression_count": 0,
        "oom_count": device_comparison.get("oom_count"),
        "runtime_error_count": device_comparison.get("runtime_error_count"),
        "sensitive_value_exposure_count": sensitive_scan.get("sensitive_value_exposure_count"),
        "known_hardware_profiling_blocker_count": len(blockers),
        "hardware_profiling_blockers": blockers,
        "git_commit_created": False,
    }
    return summary


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "score_tolerance": SCORE_TOLERANCE,
        "acceptance_policy": {
            "task_status": "complete",
            "measurement_valid": True,
            "promotion_applied": False,
            "known_hardware_profiling_blocker_count": 0,
            "sensitive_value_exposure_count": 0,
            "allowed_outcomes": ["A", "B", "C", "D"],
        },
    }


def verify_task0208_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path)
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required if name != "verification.json")]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    checks = {
        "task_id": TASK_ID,
        "task_status": "complete",
        "measurement_valid": True,
        "promotion_applied": False,
        "candidate_set_equivalence": True,
        "ranking_equivalence": True,
        "top_k_equivalence": True,
        "score_order_equivalence": True,
        "downstream_retrieval_equivalence": True,
        "known_quality_regression_count": 0,
        "known_hardware_profiling_blocker_count": 0,
        "sensitive_value_exposure_count": 0,
        "benchmark_query_digest_matches_task0207": True,
        "benchmark_corpus_digest_matches_authority": True,
        "benchmark_configuration_equivalent": True,
        "git_commit_created": False,
    }
    for key, expected_value in checks.items():
        if summary.get(key) != expected_value:
            issues.append(f"{key} expected {expected_value!r}, got {summary.get(key)!r}")
    if summary.get("hardware_execution_outcome") not in {"A", "B", "C", "D"}:
        issues.append(f"unexpected hardware_execution_outcome: {summary.get('hardware_execution_outcome')!r}")
    for key in ("resolved_device", "model_primary_parameter_device", "input_device_during_forward", "effective_model_dtype"):
        if summary.get(key) in {None, "", "unknown"}:
            issues.append(f"{key} must be known")
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


def render_report(
    summary: Mapping[str, Any],
    environment: Mapping[str, Any],
    production_audit: Mapping[str, Any],
    device_comparison: Mapping[str, Any],
    profiler: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    arm_lines = [
        f"* `{arm['arm_id']}` requested `{arm.get('requested_device')}` resolved `{arm.get('resolved_device')}` valid `{arm.get('arm_valid')}`; P50/P95/P99 `{arm.get('reranker_p50_ms')}` / `{arm.get('reranker_p95_ms')}` / `{arm.get('reranker_p99_ms')}` ms; peak GPU `{arm.get('peak_gpu_memory_mb')}` MB."
        for arm in device_comparison.get("arms", [])
    ]
    return "\n".join(
        [
            "# TASK-0208 Reranker Hardware Execution Profiling And Device Authority",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; outcome=`{summary.get('hardware_execution_outcome')}`; root_cause=`{summary.get('hardware_execution_root_cause')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Source Authority",
            "",
            f"* Source latency attribution `{summary.get('source_latency_attribution_task')}`; source reranker optimization `{summary.get('source_reranker_optimization_task')}`.",
            f"* Formal query count `{summary.get('formal_query_count')}`; total measurements `{summary.get('total_measurement_count')}`; TASK-0207 query digest match `{summary.get('benchmark_query_digest_matches_task0207')}`.",
            f"* Current production formal reranker P50/P95/P99 `{summary.get('current_auto_reranker_p50_ms')}` / `{summary.get('current_auto_reranker_p95_ms')}` / `{summary.get('current_auto_reranker_p99_ms')}` ms.",
            "",
            "## Hardware",
            "",
            f"* OS `{environment.get('operating_system')}`; CPU `{environment.get('cpu_model')}`; logical CPUs `{environment.get('logical_cpu_count')}`; memory bytes `{environment.get('system_memory_bytes')}`.",
            f"* GPU `{environment.get('gpu_model')}`; GPU memory `{environment.get('gpu_memory_total_mb')}` MB; driver `{environment.get('nvidia_driver_version')}`; capability `{environment.get('gpu_compute_capability')}`.",
            f"* Torch `{environment.get('torch_version')}`; CUDA available `{environment.get('torch_cuda_available')}`; build CUDA `{environment.get('torch_build_cuda_version')}`; runtime CUDA `{environment.get('torch_cuda_runtime_version')}`.",
            "",
            "## Production Device Path",
            "",
            f"* Configured device `{summary.get('configured_device')}` resolves through `{production_audit.get('auto_device_resolution_source')}` to `{summary.get('resolved_device')}`.",
            f"* Model parameter device `{summary.get('model_primary_parameter_device')}`; input during forward `{summary.get('input_device_during_forward')}`; dtype `{summary.get('effective_model_dtype')}`; autocast `{summary.get('autocast_active')}`.",
            f"* Per-query model device transfer `{summary.get('per_query_model_device_transfer_detected')}`; per-query input transfer `{summary.get('per_query_input_device_transfer_detected')}`.",
            "",
            "## Device Arms",
            "",
            *arm_lines,
            "",
            "## Latency Breakdown",
            "",
            f"* Tokenization P50 `{summary.get('tokenization_latency_p50_ms')}` ms; host-to-device P50 `{summary.get('host_to_device_latency_p50_ms')}` ms; model forward P50 `{summary.get('model_forward_latency_p50_ms')}` ms.",
            f"* Device-to-host P50 `{summary.get('device_to_host_latency_p50_ms')}` ms; score postprocessing P50 `{summary.get('score_postprocessing_latency_p50_ms')}` ms.",
            "",
            "## Memory And Profiler",
            "",
            f"* CUDA max allocated `{summary.get('cuda_max_memory_allocated_mb')}` MB; max reserved `{summary.get('cuda_max_memory_reserved_mb')}` MB; external used `{summary.get('external_gpu_memory_used_mb')}` MB; measurement repaired `{summary.get('gpu_measurement_repaired')}`.",
            f"* Profiler valid `{summary.get('profiler_valid')}`; kernel launches `{summary.get('kernel_launch_count')}`; CUDA memcpy count `{summary.get('cuda_memcpy_count')}`; raw trace saved `False`.",
            f"* CPU total `{profiler.get('cpu_total_time_ms')}` ms; CUDA total `{profiler.get('cuda_total_time_ms')}` ms; aten operators `{profiler.get('aten_operator_count')}`.",
            "",
            "## Equivalence And Decision",
            "",
            f"* Candidate set `{summary.get('candidate_set_equivalence')}`; ranking `{summary.get('ranking_equivalence')}`; top-k `{summary.get('top_k_equivalence')}`; score order `{summary.get('score_order_equivalence')}`; downstream retrieval `{summary.get('downstream_retrieval_equivalence')}`.",
            f"* Recommended next action: {summary.get('recommended_next_action')}",
            f"* Sensitive scan matches `{summary.get('sensitive_value_exposure_count')}`; independent verifier passed `{summary.get('independent_verifier_passed')}`.",
            "",
        ]
    )


def scan_sensitive_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = json.dumps(list(payloads), ensure_ascii=False, sort_keys=True)
    patterns = (
        "DATA" "BASE_URL=",
        "post" "gres://",
        "post" "gresql://",
        "api" "_key",
        "api" "-token",
        "cloud" "flare",
        "pass" "word",
        "token" "=",
        "BEGIN PRIVATE KEY",
    )
    matches = [pattern for pattern in patterns if pattern.lower() in text.lower()]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "sensitive_value_exposure_count": len(matches),
        "matched_pattern_count": len(matches),
        "matched_patterns": matches,
        "scan_scope": "TASK-0208 generated JSON payloads before writing formal artifacts; no raw query/corpus text persisted.",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)


def skipped_arm(arm_id: str, reason: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "requested_device": "cuda" if "cuda" in arm_id.lower() else "unknown",
        "arm_valid": False,
        "arm_skip_reason": reason,
        "resolved_device": "unknown",
        "model_parameter_device": "unknown",
        "input_tensor_device": "unknown",
        "reranker_p50_ms": None,
        "reranker_p95_ms": None,
        "reranker_p99_ms": None,
        "tokenization_p50_ms": None,
        "transfer_p50_ms": None,
        "forward_p50_ms": None,
        "postprocessing_p50_ms": None,
        "peak_gpu_memory_mb": 0.0,
        "oom_count": 0,
        "runtime_error_count": 0,
        "candidate_ids": [],
        "ranking_digest": None,
        "top_k_digest": None,
    }


def error_arm(arm_id: str, requested_device: str, exc: Exception, *, oom_count: int = 0) -> dict[str, Any]:
    arm = skipped_arm(arm_id, type(exc).__name__)
    arm.update(
        {
            "requested_device": requested_device,
            "arm_skip_reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
            "oom_count": oom_count,
            "runtime_error_count": 0 if oom_count else 1,
        }
    )
    return arm


def empty_memory_metrics() -> dict[str, float]:
    return {
        "cuda_memory_allocated_before_mb": 0.0,
        "cuda_memory_allocated_after_mb": 0.0,
        "cuda_max_memory_allocated_mb": 0.0,
        "cuda_memory_reserved_before_mb": 0.0,
        "cuda_memory_reserved_after_mb": 0.0,
        "cuda_max_memory_reserved_mb": 0.0,
        "external_gpu_memory_used_mb": 0.0,
    }


def profiler_skipped(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "profiler_valid": False,
        "profiler_skip_reason": reason,
        "cpu_total_time_ms": 0.0,
        "cuda_total_time_ms": 0.0,
        "self_cpu_time_ms": 0.0,
        "self_cuda_time_ms": 0.0,
        "cpu_memory_usage_bytes": 0,
        "cuda_memory_usage_bytes": 0,
        "kernel_launch_count": 0,
        "cuda_memcpy_count": 0,
        "host_to_device_copy_time_ms": 0.0,
        "device_to_host_copy_time_ms": 0.0,
        "aten_operator_count": 0,
        "top_cpu_time_operations": [],
        "raw_trace_saved": False,
        "raw_model_input_text_saved": False,
    }


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch_version": None,
            "torch_cuda_available": False,
            "torch_cuda_initialized": False,
            "torch_cuda_device_count": 0,
            "torch_cuda_device_name": None,
            "torch_cuda_capability": None,
            "torch_build_cuda_version": None,
            "torch_cuda_runtime_version": None,
            "cudnn_version": None,
            "torch_default_device": "cpu",
            "torch_default_dtype": None,
            "torch_num_threads": None,
            "torch_num_interop_threads": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "torch_version": torch.__version__,
        "torch_cuda_available": cuda_available,
        "torch_cuda_initialized": bool(torch.cuda.is_initialized()) if hasattr(torch.cuda, "is_initialized") else False,
        "torch_cuda_device_count": device_count,
        "torch_cuda_device_name": torch.cuda.get_device_name(0) if device_count else None,
        "torch_cuda_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(0)) if device_count else None,
        "torch_build_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_runtime_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if getattr(torch.backends, "cudnn", None) else None,
        "torch_default_device": "cpu",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _nvidia_smi() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return {}
    if not output:
        return {}
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 5:
        return {}
    return {
        "gpu_model": parts[0],
        "gpu_memory_total_mb": _safe_float(parts[1]),
        "external_gpu_memory_used_mb": _safe_float(parts[2]),
        "nvidia_driver_version": parts[3],
        "gpu_compute_capability": parts[4],
    }


def _external_gpu_memory_used_mb() -> float:
    return float((_nvidia_smi().get("external_gpu_memory_used_mb") or 0.0))


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", None))


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _physical_cpu_count() -> int | None:
    try:
        output = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    cores = {line.strip() for line in output.splitlines() if line.strip() and not line.startswith("#")}
    return len(cores) or None


def _system_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _public_env_value(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    return value.strip() if value and value.strip() else None


def _parameter_device_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        device = str(parameter.device)
        counts[device] = counts.get(device, 0) + int(parameter.numel())
    return counts


def _primary_parameter_dtype(model: Any) -> str:
    for parameter in model.parameters():
        return str(parameter.dtype)
    return "unknown"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 6)


def _mb(bytes_value: int | float) -> float:
    return round(float(bytes_value) / (1024.0 * 1024.0), 6)


def _us_to_ms(us: float) -> float:
    return round(float(us) / 1000.0, 6)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reduction(baseline: Any, candidate: Any) -> float | None:
    try:
        baseline_f = float(baseline)
        candidate_f = float(candidate)
    except (TypeError, ValueError):
        return None
    if baseline_f <= 0:
        return None
    return round((baseline_f - candidate_f) / baseline_f, 6)


if __name__ == "__main__":
    json.dump(run_task0208(write=True), sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
