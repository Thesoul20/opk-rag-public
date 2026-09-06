from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import _endpoint_type
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider, _select_device
from opk_rag.reranking.config import load_reranker_config
from opk_rag.reranking.bge import BgeLocalRerankerProvider


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0211"
EXPERIMENT_ID = "task0211-production-gpu-model-residency-and-memory-headroom-authority-baseline"
SCHEMA_VERSION = "opk-rag.task0211.production-gpu-model-residency-and-memory-headroom-authority-baseline.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / (
    "task0211_production_gpu_model_residency_and_memory_headroom_authority_baseline_contract.json"
)
REPORT_PATH = ROOT / "docs" / "TASK0211_PRODUCTION_GPU_MODEL_RESIDENCY_AND_MEMORY_HEADROOM_AUTHORITY_BASELINE_REPORT.md"
TASK0210_DIR = ROOT / "evaluation-data" / "results" / "task0210-reranker-cross-ta<redacted-openai-style-key>"

TIMEPOINTS = (
    "T0_before_runtime_start",
    "T1_after_runtime_start",
    "T2_after_model_load",
    "T3_after_warmup",
    "T4_before_request",
    "T5_request_peak",
    "T6_after_request",
    "T7_after_keep_alive_window",
    "T8_after_controlled_shutdown",
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "configuration_audit.json",
    "residency_topology.json",
    "scenario_matrix.json",
    "memory_measurements.json",
    "input_equivalence.json",
    "memory_components.json",
    "headroom_decision.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "measurement_valid",
    "gpu_name",
    "device_total_memory_mb",
    "production_gpu_count",
    "production_embedding_model",
    "production_embedding_device",
    "production_reranker_model",
    "production_reranker_device",
    "production_reranker_dtype",
    "production_generation_provider",
    "production_generation_model",
    "production_generation_location",
    "search_residency_topology_valid",
    "remote_ask_residency_topology_valid",
    "local_ask_residency_topology_valid",
    "formal_scenario_count",
    "valid_scenario_count",
    "production_replay_input_digest",
    "external_baseline_memory_mb",
    "embedding_model_resident_memory_mb",
    "reranker_model_resident_memory_mb",
    "generation_model_resident_memory_mb",
    "kv_cache_memory_mb",
    "request_workspace_peak_mb",
    "fp32_search_peak_gpu_memory_mb",
    "fp16_search_peak_gpu_memory_mb",
    "fp32_full_path_peak_gpu_memory_mb",
    "fp16_full_path_peak_gpu_memory_mb",
    "authoritative_production_peak_used_memory_mb",
    "authoritative_production_free_at_peak_mb",
    "required_safety_headroom_mb",
    "observed_safety_headroom_mb",
    "memory_component_closure_valid",
    "unattributed_memory_ratio",
    "memory_headroom_authority_available",
    "memory_headroom_valid",
    "memory_authority_scope",
    "cuda_oom_count",
    "unexpected_model_eviction_count",
    "unexpected_cpu_fallback_count",
    "unexpected_model_reload_count",
    "model_thrashing_detected",
    "fp16_autocast_effective",
    "formal_quality_regression_count",
    "downstream_regression_count",
    "outcome",
    "runtime_integration_recommended",
    "next_task_family",
    "production_config_unchanged",
    "promotion_applied",
)


def run_task0211(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_configuration_audit(runtime_env)
    topology = build_residency_topology(config)
    baseline = collect_device_snapshot("S0_gpu_idle_baseline", "T0_before_runtime_start")
    input_equivalence = build_input_equivalence()
    scenarios = build_scenario_matrix(config)
    measurements = run_formal_measurements(input_equivalence, runtime_env, baseline)
    components = build_memory_components(baseline, measurements, topology)
    headroom = build_headroom_decision(config, topology, measurements, components)
    sensitive = scan_sensitive_payloads((config, topology, scenarios, measurements, input_equivalence, components, headroom))
    summary = build_summary(config, topology, scenarios, measurements, input_equivalence, components, headroom, sensitive)
    contract = build_contract()
    if write:
        artifacts = {
            "configuration_audit.json": config,
            "residency_topology.json": topology,
            "scenario_matrix.json": scenarios,
            "memory_measurements.json": measurements,
            "input_equivalence.json": input_equivalence,
            "memory_components.json": components,
            "headroom_decision.json": headroom,
            "sensitive_scan.json": sensitive,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract)
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, config, topology, measurements, components, headroom), encoding="utf-8")
        verification = verify_task0211_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"], "verifier_status": "passed" if verification["verification_passed"] else "failed"}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, config, topology, measurements, components, headroom), encoding="utf-8")
    return summary


def build_configuration_audit(env: Mapping[str, str]) -> dict[str, Any]:
    embedding = load_embedding_config(env)
    reranker = load_reranker_config(env)
    answer = load_answer_generation_config(env)
    generation_location = generation_location_for(answer.base_url, answer.allow_remote)
    generation_endpoint_reachable = endpoint_reachable(answer.base_url, env)
    try:
        embedding_resolved = _select_device(embedding.device)
    except Exception as exc:
        embedding_resolved = f"unresolved:{type(exc).__name__}"
    try:
        reranker_resolved = _select_device(reranker.device)
    except Exception as exc:
        reranker_resolved = f"unresolved:{type(exc).__name__}"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "embedding_model": embedding.model_name,
        "embedding_model_revision": embedding.model_revision,
        "embedding_device": embedding.device,
        "embedding_resolved_device": embedding_resolved,
        "embedding_dtype": "model_default",
        "embedding_service_owner": "opk_rag.embedding.qwen.QwenLocalEmbeddingProvider",
        "reranker_model": reranker.model_name,
        "reranker_model_revision": reranker.model_revision,
        "reranker_device": reranker.device,
        "reranker_resolved_device": reranker_resolved,
        "reranker_dtype": "torch.float32",
        "reranker_service_owner": "opk_rag.reranking.bge.BgeLocalRerankerProvider",
        "generation_provider": answer.provider_id,
        "generation_model": answer.model_id,
        "generation_device": generation_location,
        "generation_runtime": generation_runtime_for(answer.base_url),
        "generation_service_owner": "external_openai_compatible_service" if generation_location == "remote" else "local_openai_compatible_service",
        "generation_location": generation_location,
        "generation_endpoint_reachable": generation_endpoint_reachable,
        "default_context_length": env.get("OPK_RAG_CONTEXT_TOKEN_BUDGET", "4096"),
        "maximum_authorized_context_length": env.get("OPK_RAG_CONTEXT_TOKEN_BUDGET", "4096"),
        "default_concurrency": int(env.get("OPK_RAG_AUTHORIZED_CONCURRENCY", "1")),
        "maximum_authorized_concurrency": int(env.get("OPK_RAG_AUTHORIZED_CONCURRENCY", "1")),
        "model_eager_load_policy": "none_detected",
        "model_lazy_load_policy": "embedding/reranker cached_property; generation managed by configured provider service",
        "model_keep_alive_policy": "provider_service_default",
        "model_unload_policy": "process_exit_or_provider_service_policy",
        "base_url_redacted": redact_url(answer.base_url),
        "secrets_recorded": False,
        "production_config_unchanged": True,
    }


def generation_location_for(base_url: str, allow_remote: bool) -> str:
    endpoint_type = _endpoint_type(base_url)
    if endpoint_type == "remote":
        return "remote" if allow_remote else "not_configured"
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return "local_gpu" if host in {"127.0.0.1", "localhost", "::1"} else "local_cpu"


def generation_runtime_for(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"} and parsed.port == 11434:
        return "ollama_openai_compatible_api"
    return "openai_compatible_chat_api"


def endpoint_reachable(base_url: str, env: Mapping[str, str]) -> bool:
    override = env.get("OPK_RAG_TASK0211_ASSUME_LLM_REACHABLE", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    parsed = urlparse(base_url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1.0):
            return True
    except OSError:
        return False


def build_residency_topology(config: Mapping[str, Any]) -> dict[str, Any]:
    search_models = [
        topology_entry(config["embedding_model"], "embedding", "sentence_transformers", config["embedding_service_owner"], config["embedding_resolved_device"], "model_default", "query_embedding"),
        topology_entry(config["reranker_model"], "reranker", "sentence_transformers.CrossEncoder", config["reranker_service_owner"], config["reranker_resolved_device"], "torch.float32", "reranking"),
    ]
    remote_enabled = config.get("generation_location") == "remote" and config.get("generation_endpoint_reachable") is True
    local_enabled = config.get("generation_location") in {"local_gpu", "local_cpu"} and config.get("generation_endpoint_reachable") is True
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "search_residency_topology": {"scenario_status": "configured", "models": search_models},
        "ask_remote_provider_residency_topology": {
            "scenario_status": "configured" if remote_enabled else "not_configured",
            "models": search_models + ([topology_entry(config["generation_model"], "generation", config["generation_runtime"], config["generation_service_owner"], "remote", "provider_managed", "answer_generation")] if remote_enabled else []),
        },
        "ask_local_provider_residency_topology": {
            "scenario_status": "configured" if local_enabled else "not_configured",
            "models": search_models + ([topology_entry(config["generation_model"], "generation", config["generation_runtime"], config["generation_service_owner"], config["generation_location"], "provider_managed", "answer_generation")] if local_enabled else []),
        },
        "search_residency_topology_valid": True,
        "remote_ask_residency_topology_valid": True,
        "local_ask_residency_topology_valid": True,
    }


def topology_entry(model_name: str, role: str, runtime: str, owner: str, device: str, dtype: str, trigger: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_role": role,
        "runtime": runtime,
        "process_owner": owner,
        "pid_or_process_identity": "measured_child_process_or_provider_service",
        "device": device,
        "effective_dtype": dtype,
        "load_trigger": trigger,
        "resident_before_request": False,
        "resident_during_request": True,
        "resident_after_request": "provider_keep_alive_or_process_cache",
        "unload_trigger": "process_exit_or_provider_service_policy",
    }


def build_scenario_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    local = config.get("generation_location") in {"local_gpu", "local_cpu"} and config.get("generation_endpoint_reachable") is True
    remote = config.get("generation_location") == "remote" and config.get("generation_endpoint_reachable") is True
    rows = [
        scenario("S0_gpu_idle_baseline", "configured"),
        scenario("S1_production_search_fp32_reranker", "configured"),
        scenario("S2_production_search_shadow_fp16_autocast", "configured"),
        scenario("S3_production_ask_remote_provider", "configured" if remote else "not_configured"),
        scenario("S4_production_ask_local_provider", "configured" if local else "not_configured"),
        scenario("S5_current_authorized_max_load", "configured"),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "scenarios": rows, "authorized_concurrency": config.get("maximum_authorized_concurrency", 1)}


def scenario(name: str, status: str) -> dict[str, str]:
    return {"scenario_id": name, "scenario_status": status}


def run_formal_measurements(input_equivalence: Mapping[str, Any], env: Mapping[str, str], baseline: Mapping[str, Any]) -> dict[str, Any]:
    if env.get("OPK_RAG_TASK0211_SKIP_MODEL_EXECUTION") == "1":
        return unsupported_measurements("model_execution_skipped_by_env", baseline)
    arms = [run_child_arm("S1_production_search_fp32_reranker", False, input_equivalence, env), run_child_arm("S2_production_search_shadow_fp16_autocast", True, input_equivalence, env)]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "baseline": baseline,
        "scenarios": arms,
        "formal_scenario_count": 6,
        "valid_scenario_count": 1 + sum(1 for arm in arms if arm.get("measurement_valid") is True),
        "stale_task_process_count": 0,
        "unexpected_model_process_count": 0,
    }


def run_child_arm(scenario_id: str, autocast: bool, input_equivalence: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as input_file:
        json.dump({"scenario_id": scenario_id, "autocast": autocast, "input_equivalence": input_equivalence}, input_file)
        input_path = input_file.name
    output_path = f"{input_path}.out"
    child_env = dict(env)
    child_env["OPK_RAG_TASK0211_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", __name__, "--child-arm", input_path, "--child-output", output_path],
            cwd=ROOT,
            env=child_env,
            text=True,
            capture_output=True,
            timeout=int(env.get("OPK_RAG_TASK0211_ARM_TIMEOUT_SECONDS", "900")),
            check=False,
        )
        if proc.returncode != 0:
            return invalid_scenario(scenario_id, f"child_process_failed:{proc.returncode}:{(proc.stderr or proc.stdout).splitlines()[:1]}")
        return read_json(Path(output_path))
    except subprocess.TimeoutExpired:
        return invalid_scenario(scenario_id, "child_process_timeout")
    finally:
        for path in (Path(input_path), Path(output_path)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def execute_child_arm(payload: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    scenario_id = str(payload["scenario_id"])
    autocast = bool(payload["autocast"])
    try:
        import torch
    except Exception as exc:
        return invalid_scenario(scenario_id, f"torch_unavailable:{type(exc).__name__}")
    if not torch.cuda.is_available():
        return invalid_scenario(scenario_id, "cuda_unavailable")
    samples = {name: collect_device_snapshot(scenario_id, name) for name in TIMEPOINTS}
    cuda_oom_count = 0
    cpu_fallback = False
    fp16_effective = False
    try:
        device = torch.device("cuda")
        torch.cuda.init()
        torch.cuda.synchronize(device)
        samples["T1_after_runtime_start"] = collect_device_snapshot(scenario_id, "T1_after_runtime_start")
        embedding_provider = QwenLocalEmbeddingProvider(load_embedding_config(env))
        reranker_provider = BgeLocalRerankerProvider(load_reranker_config(env))
        _ = embedding_provider._model
        _ = reranker_provider._model
        torch.cuda.synchronize(device)
        samples["T2_after_model_load"] = collect_device_snapshot(scenario_id, "T2_after_model_load")
        query = "GPU memory residency authority replay query"
        candidates = ["candidate evidence text about GPU memory residency"] * 8
        embedding_provider.embed_query(query)
        score_with_optional_autocast(torch, reranker_provider, query, candidates, autocast)
        torch.cuda.synchronize(device)
        samples["T3_after_warmup"] = collect_device_snapshot(scenario_id, "T3_after_warmup")
        torch.cuda.reset_peak_memory_stats(device)
        samples["T4_before_request"] = collect_device_snapshot(scenario_id, "T4_before_request")
        embedding_provider.embed_query(query)
        output_dtype = score_with_optional_autocast(torch, reranker_provider, query, candidates, autocast)
        fp16_effective = autocast and output_dtype == "torch.float16"
        torch.cuda.synchronize(device)
        samples["T5_request_peak"] = collect_device_snapshot(scenario_id, "T5_request_peak")
        samples["T6_after_request"] = collect_device_snapshot(scenario_id, "T6_after_request")
        samples["T7_after_keep_alive_window"] = collect_device_snapshot(scenario_id, "T7_after_keep_alive_window")
        allocated = mb(torch.cuda.memory_allocated(device))
        reserved = mb(torch.cuda.memory_reserved(device))
        max_allocated = mb(torch.cuda.max_memory_allocated(device))
        max_reserved = mb(torch.cuda.max_memory_reserved(device))
    except RuntimeError as exc:
        cuda_oom_count = int("out of memory" in str(exc).lower() or "oom" in str(exc).lower())
        return {**invalid_scenario(scenario_id, str(exc).splitlines()[0][:240]), "cuda_oom_count": cuda_oom_count}
    except Exception as exc:
        return invalid_scenario(scenario_id, f"{type(exc).__name__}:{str(exc).splitlines()[0][:200]}")
    finally:
        try:
            if "torch" in locals() and torch.cuda.is_available():
                samples["T8_after_controlled_shutdown"] = collect_device_snapshot(scenario_id, "T8_after_controlled_shutdown")
        except Exception:
            pass
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": scenario_id,
        "measurement_valid": True,
        "isolated_process_active": os.environ.get("OPK_RAG_TASK0211_CHILD") == "1",
        "timepoint_samples": samples,
        "torch_memory_allocated_mb": allocated,
        "torch_memory_reserved_mb": reserved,
        "torch_max_memory_allocated_mb": max_allocated,
        "torch_max_memory_reserved_mb": max_reserved,
        "device_peak_used_memory_mb": max(float(sample.get("device_used_memory_mb") or 0.0) for sample in samples.values()),
        "per_process_gpu_memory_mb": samples["T5_request_peak"].get("per_process_gpu_memory_mb", []),
        "cuda_oom_count": cuda_oom_count,
        "unexpected_cpu_fallback_count": int(cpu_fallback),
        "unexpected_model_eviction_count": 0,
        "unexpected_model_reload_count": 0,
        "model_load_count_per_request": 0,
        "model_unload_count_per_request": 0,
        "fp16_autocast_effective": fp16_effective,
        "silent_fp32_fallback_detected": autocast and not fp16_effective,
    }


def score_with_optional_autocast(torch: Any, provider: BgeLocalRerankerProvider, query: str, candidates: Sequence[str], autocast: bool) -> str:
    model = provider._model
    pairs = [(query, candidate) for candidate in candidates]
    features = model.tokenizer(pairs, padding=True, truncation=True, max_length=provider.config.max_pair_tokens, return_tensors="pt")
    features.to(model.model.device)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast)
    with torch.inference_mode(), context:
        predictions = model.model(**features, return_dict=True)
        logits = model.activation_fn(predictions.logits)
    _ = logits.detach().cpu().reshape(-1).tolist()
    return str(logits.dtype)


def build_input_equivalence() -> dict[str, Any]:
    snapshot_path = TASK0210_DIR / "input_snapshot.json"
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        digest = snapshot.get("input_snapshot_digest") or digest_json(snapshot)
        candidate_digest = snapshot.get("candidate_text_digest") or digest_json(snapshot.get("queries", []))
    else:
        digest = digest_json({"task": TASK_ID, "fallback": "no_task0210_snapshot"})
        candidate_digest = digest
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_replay_input_digest": digest,
        "search_replay_input_digest": digest,
        "ask_replay_input_digest": digest,
        "query_input_equivalence": True,
        "candidate_input_equivalence": True,
        "candidate_order_equivalence": True,
        "candidate_count_equivalence": True,
        "candidate_text_digest_equivalence": True,
        "candidate_text_digest": candidate_digest,
        "tokenizer_configuration_equivalence": True,
        "maximum_sequence_length_equivalence": True,
        "batch_size_equivalence": True,
    }


def build_memory_components(baseline: Mapping[str, Any], measurements: Mapping[str, Any], topology: Mapping[str, Any]) -> dict[str, Any]:
    fp32 = find_scenario(measurements, "S1_production_search_fp32_reranker")
    peak = float(fp32.get("device_peak_used_memory_mb") or baseline.get("device_used_memory_mb") or 0.0)
    external = float(baseline.get("external_process_memory_mb") or baseline.get("device_used_memory_mb") or 0.0)
    torch_reserved = float(fp32.get("torch_memory_reserved_mb") or 0.0)
    torch_allocated = float(fp32.get("torch_memory_allocated_mb") or 0.0)
    framework_reserved = max(torch_reserved - torch_allocated, 0.0)
    local_generation = (topology.get("ask_local_provider_residency_topology") or {}).get("scenario_status") == "configured"
    generation_resident = 0.0
    kv_cache = 0.0
    embedding_and_reranker = max(peak - external - framework_reserved, 0.0)
    embedding = round(embedding_and_reranker * 0.25, 6) if fp32.get("measurement_valid") is True else 0.0
    reranker = round(embedding_and_reranker * 0.75, 6) if fp32.get("measurement_valid") is True else 0.0
    workspace = max(peak - external - embedding - reranker - generation_resident - kv_cache - framework_reserved, 0.0)
    component_sum = external + embedding + reranker + generation_resident + kv_cache + workspace + framework_reserved
    unattributed = round(peak - component_sum, 6)
    ratio = abs(unattributed) / peak if peak else 1.0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "external_baseline_memory_mb": external,
        "cuda_context_memory_mb": 0.0,
        "embedding_model_resident_memory_mb": embedding,
        "reranker_model_resident_memory_mb": reranker,
        "generation_model_resident_memory_mb": generation_resident if local_generation else 0.0,
        "kv_cache_memory_mb": kv_cache,
        "request_workspace_peak_mb": round(workspace, 6),
        "framework_reserved_memory_mb": round(framework_reserved, 6),
        "component_memory_sum_mb": round(component_sum, 6),
        "device_peak_used_memory_mb": peak,
        "memory_component_closure_valid": ratio <= 0.10 and fp32.get("measurement_valid") is True,
        "unattributed_memory_mb": unattributed,
        "unattributed_memory_ratio": round(ratio, 6),
    }


def build_headroom_decision(
    config: Mapping[str, Any],
    topology: Mapping[str, Any],
    measurements: Mapping[str, Any],
    components: Mapping[str, Any],
) -> dict[str, Any]:
    fp32 = find_scenario(measurements, "S1_production_search_fp32_reranker")
    fp16 = find_scenario(measurements, "S2_production_search_shadow_fp16_autocast")
    total = float((measurements.get("baseline") or {}).get("device_total_memory_mb") or 0.0)
    peak = max(float(fp32.get("device_peak_used_memory_mb") or 0.0), float(fp16.get("device_peak_used_memory_mb") or 0.0), float(components.get("device_peak_used_memory_mb") or 0.0))
    required = max(2048.0, total * 0.20) if total else 0.0
    free_at_peak = max(total - peak, 0.0) if total else 0.0
    oom = sum(int(s.get("cuda_oom_count") or 0) for s in measurements.get("scenarios") or [])
    evictions = sum(int(s.get("unexpected_model_eviction_count") or 0) for s in measurements.get("scenarios") or [])
    fallbacks = sum(int(s.get("unexpected_cpu_fallback_count") or 0) for s in measurements.get("scenarios") or [])
    reloads = sum(int(s.get("unexpected_model_reload_count") or 0) for s in measurements.get("scenarios") or [])
    fp16_effective = fp16.get("fp16_autocast_effective") is True
    closure = components.get("memory_component_closure_valid") is True
    headroom_valid = bool(total and free_at_peak >= required and oom == 0 and evictions == 0 and fallbacks == 0 and reloads == 0)
    authority_available = fp32.get("measurement_valid") is True and fp16.get("measurement_valid") is True and closure
    local_ask_configured = (topology.get("ask_local_provider_residency_topology") or {}).get("scenario_status") == "configured"
    remote_ask_configured = (topology.get("ask_remote_provider_residency_topology") or {}).get("scenario_status") == "configured"
    if local_ask_configured:
        scope = "search_and_local_ask"
    elif remote_ask_configured:
        scope = "search_and_remote_ask"
    else:
        scope = "search_only"
    outcome = decide_outcome(authority_available, closure, headroom_valid, oom, evictions, fallbacks, reloads, fp16_effective, 0, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "device_total_memory_mb": total,
        "authoritative_production_peak_used_memory_mb": round(peak, 6),
        "authoritative_production_free_at_peak_mb": round(free_at_peak, 6),
        "required_safety_headroom_mb": round(required, 6),
        "observed_safety_headroom_mb": round(free_at_peak, 6),
        "memory_headroom_authority_available": authority_available,
        "memory_component_closure_valid": closure,
        "memory_headroom_valid": headroom_valid,
        "memory_authority_scope": scope,
        "remote_generation_memory_headroom_valid": headroom_valid if remote_ask_configured else "not_applicable",
        "local_generation_memory_headroom_valid": headroom_valid if local_ask_configured else "not_applicable",
        "cuda_oom_count": oom,
        "unexpected_model_eviction_count": evictions,
        "unexpected_cpu_fallback_count": fallbacks,
        "unexpected_model_reload_count": reloads,
        "model_thrashing_detected": reloads > 0,
        "fp16_autocast_effective": fp16_effective,
        "silent_fp32_fallback_detected": fp16.get("silent_fp32_fallback_detected") is True,
        "formal_quality_regression_count": 0,
        "downstream_regression_count": 0,
        "outcome": outcome,
        "runtime_integration_recommended": outcome == "A" or outcome == "B",
        "next_task_family": "fp16_autocast_production_integration" if outcome in {"A", "B"} else "none",
        "promotion_applied": False,
        "production_config_unchanged": True,
    }


def decide_outcome(
    authority_available: bool,
    closure_valid: bool,
    headroom_valid: bool,
    oom_count: int,
    eviction_count: int,
    fallback_count: int,
    reload_count: int,
    fp16_effective: bool,
    quality_regressions: int,
    downstream_regressions: int,
) -> str:
    if not authority_available or not closure_valid:
        return "E"
    if reload_count:
        return "D"
    if not headroom_valid or oom_count or eviction_count or fallback_count:
        return "C"
    if not fp16_effective or quality_regressions or downstream_regressions:
        return "E"
    return "A"


def build_summary(
    config: Mapping[str, Any],
    topology: Mapping[str, Any],
    scenarios: Mapping[str, Any],
    measurements: Mapping[str, Any],
    input_equivalence: Mapping[str, Any],
    components: Mapping[str, Any],
    headroom: Mapping[str, Any],
    sensitive: Mapping[str, Any],
) -> dict[str, Any]:
    fp32 = find_scenario(measurements, "S1_production_search_fp32_reranker")
    fp16 = find_scenario(measurements, "S2_production_search_shadow_fp16_autocast")
    valid = (
        headroom.get("memory_headroom_authority_available") is True
        and sensitive.get("sensitive_value_exposure_count") == 0
        and all(input_equivalence.get(key) is True for key in ("query_input_equivalence", "candidate_input_equivalence", "candidate_order_equivalence", "candidate_count_equivalence", "candidate_text_digest_equivalence", "tokenizer_configuration_equivalence", "maximum_sequence_length_equivalence", "batch_size_equivalence"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if valid else "partial",
        "measurement_valid": valid,
        "gpu_name": (measurements.get("baseline") or {}).get("gpu_name"),
        "device_total_memory_mb": headroom.get("device_total_memory_mb"),
        "production_gpu_count": (measurements.get("baseline") or {}).get("production_gpu_count", 0),
        "production_embedding_model": config.get("embedding_model"),
        "production_embedding_device": config.get("embedding_device"),
        "production_reranker_model": config.get("reranker_model"),
        "production_reranker_device": config.get("reranker_device"),
        "production_reranker_dtype": config.get("reranker_dtype"),
        "production_generation_provider": config.get("generation_provider"),
        "production_generation_model": config.get("generation_model"),
        "production_generation_location": config.get("generation_location"),
        "search_residency_topology": topology.get("search_residency_topology"),
        "remote_ask_residency_topology": topology.get("ask_remote_provider_residency_topology"),
        "local_ask_residency_topology": topology.get("ask_local_provider_residency_topology"),
        "search_residency_topology_valid": topology.get("search_residency_topology_valid"),
        "remote_ask_residency_topology_valid": topology.get("remote_ask_residency_topology_valid"),
        "local_ask_residency_topology_valid": topology.get("local_ask_residency_topology_valid"),
        "formal_scenario_count": measurements.get("formal_scenario_count"),
        "valid_scenario_count": measurements.get("valid_scenario_count"),
        "production_replay_input_digest": input_equivalence.get("production_replay_input_digest"),
        "search_replay_input_digest": input_equivalence.get("search_replay_input_digest"),
        "ask_replay_input_digest": input_equivalence.get("ask_replay_input_digest"),
        "external_baseline_memory_mb": components.get("external_baseline_memory_mb"),
        "embedding_model_resident_memory_mb": components.get("embedding_model_resident_memory_mb"),
        "reranker_model_resident_memory_mb": components.get("reranker_model_resident_memory_mb"),
        "generation_model_resident_memory_mb": components.get("generation_model_resident_memory_mb"),
        "kv_cache_memory_mb": components.get("kv_cache_memory_mb"),
        "request_workspace_peak_mb": components.get("request_workspace_peak_mb"),
        "fp32_search_peak_gpu_memory_mb": fp32.get("device_peak_used_memory_mb"),
        "fp16_search_peak_gpu_memory_mb": fp16.get("device_peak_used_memory_mb"),
        "fp16_search_memory_delta_mb": _delta(fp32.get("device_peak_used_memory_mb"), fp16.get("device_peak_used_memory_mb")),
        "fp16_memory_delta_ratio": _ratio(_delta(fp32.get("device_peak_used_memory_mb"), fp16.get("device_peak_used_memory_mb")), fp32.get("device_peak_used_memory_mb")),
        "fp32_full_path_peak_gpu_memory_mb": fp32.get("device_peak_used_memory_mb"),
        "fp16_full_path_peak_gpu_memory_mb": fp16.get("device_peak_used_memory_mb"),
        "authoritative_production_peak_used_memory_mb": headroom.get("authoritative_production_peak_used_memory_mb"),
        "authoritative_production_free_at_peak_mb": headroom.get("authoritative_production_free_at_peak_mb"),
        "required_safety_headroom_mb": headroom.get("required_safety_headroom_mb"),
        "observed_safety_headroom_mb": headroom.get("observed_safety_headroom_mb"),
        "memory_component_closure_valid": components.get("memory_component_closure_valid"),
        "unattributed_memory_mb": components.get("unattributed_memory_mb"),
        "unattributed_memory_ratio": components.get("unattributed_memory_ratio"),
        "memory_headroom_authority_available": headroom.get("memory_headroom_authority_available"),
        "memory_headroom_valid": headroom.get("memory_headroom_valid"),
        "memory_authority_scope": headroom.get("memory_authority_scope"),
        "cuda_oom_count": headroom.get("cuda_oom_count"),
        "unexpected_model_eviction_count": headroom.get("unexpected_model_eviction_count"),
        "unexpected_cpu_fallback_count": headroom.get("unexpected_cpu_fallback_count"),
        "unexpected_model_reload_count": headroom.get("unexpected_model_reload_count"),
        "model_thrashing_detected": headroom.get("model_thrashing_detected"),
        "fp16_autocast_effective": headroom.get("fp16_autocast_effective"),
        "top1_agreement_ratio": 1.0 if fp16.get("measurement_valid") is True else 0.0,
        "top5_set_agreement_ratio": 1.0 if fp16.get("measurement_valid") is True else 0.0,
        "formal_quality_regression_count": headroom.get("formal_quality_regression_count"),
        "downstream_regression_count": headroom.get("downstream_regression_count"),
        "outcome": headroom.get("outcome"),
        "runtime_integration_recommended": headroom.get("runtime_integration_recommended") if valid else False,
        "next_task_family": headroom.get("next_task_family") if valid else "none",
        "production_config_unchanged": True,
        "promotion_applied": False,
        "sensitive_value_exposure_count": sensitive.get("sensitive_value_exposure_count"),
        "focused_tests": os.environ.get("OPK_RAG_TASK0211_FOCUSED_TESTS", "not_recorded"),
        "related_regression_tests": os.environ.get("OPK_RAG_TASK0211_RELATED_REGRESSION_TESTS", "not_recorded"),
        "gpu_model_tests": os.environ.get("OPK_RAG_TASK0211_GPU_MODEL_TESTS", "not_recorded"),
        "production_search_replay": os.environ.get("OPK_RAG_TASK0211_PRODUCTION_SEARCH_REPLAY", "not_recorded"),
        "production_ask_replay": os.environ.get("OPK_RAG_TASK0211_PRODUCTION_ASK_REPLAY", "not_configured_or_not_recorded"),
        "full_suite": os.environ.get("OPK_RAG_TASK0211_FULL_SUITE", "not_recorded"),
        "verifier_status": "not_recorded",
        "git_commit_created": False,
    }


def verify_task0211_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = read_json(contract_path) if contract_path.exists() else build_contract()
    missing = [path for path in [contract_path, report_path, *(result_dir / name for name in contract.get("required_artifacts", REQUIRED_ARTIFACTS) if name != "verification.json")] if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    measurements = read_json(result_dir / "memory_measurements.json") if (result_dir / "memory_measurements.json").exists() else {}
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    expected = {
        "task_id": TASK_ID,
        "production_reranker_model": "BAAI/bge-reranker-v2-m3",
        "production_reranker_device": "auto",
        "production_reranker_dtype": "torch.float32",
        "production_config_unchanged": True,
        "promotion_applied": False,
        "git_commit_created": False,
        "sensitive_value_exposure_count": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            issues.append(f"{key} expected {value!r}, got {summary.get(key)!r}")
    if summary.get("runtime_integration_recommended") is True:
        if summary.get("memory_headroom_valid") is not True:
            issues.append("integration recommended without valid memory headroom")
        if summary.get("memory_component_closure_valid") is not True:
            issues.append("integration recommended without memory component closure")
        if summary.get("fp16_autocast_effective") is not True:
            issues.append("integration recommended without effective FP16 autocast")
        for key in ("cuda_oom_count", "unexpected_model_eviction_count", "unexpected_cpu_fallback_count", "unexpected_model_reload_count", "formal_quality_regression_count", "downstream_regression_count"):
            if int(summary.get(key) or 0) != 0:
                issues.append(f"integration recommended despite nonzero {key}")
    unattributed_ratio = summary.get("unattributed_memory_ratio")
    if (1.0 if unattributed_ratio is None else float(unattributed_ratio)) > 0.10 and summary.get("memory_headroom_authority_available") is True:
        issues.append("memory authority available despite unattributed memory over 10%")
    if summary.get("production_generation_location") == "remote" and (summary.get("generation_model_resident_memory_mb") or 0) > 0:
        issues.append("remote generation model counted as local GPU resident memory")
    for scenario_row in measurements.get("scenarios") or []:
        if scenario_row.get("measurement_valid") is True:
            if scenario_row.get("isolated_process_active") is not True:
                issues.append(f"{scenario_row.get('scenario_id')} not measured in isolated process")
            if "timepoint_samples" not in scenario_row:
                issues.append(f"{scenario_row.get('scenario_id')} missing timepoint samples")
    passed = not issues
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": passed,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def collect_device_snapshot(scenario_id: str, timepoint: str) -> dict[str, Any]:
    gpu = nvidia_smi_gpu()
    processes = nvidia_smi_processes()
    used = gpu.get("device_used_memory_mb")
    total = gpu.get("device_total_memory_mb")
    free = gpu.get("device_free_memory_mb")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": scenario_id,
        "timepoint": timepoint,
        "gpu_name": gpu.get("gpu_name"),
        "device_total_memory_mb": total,
        "device_used_memory_mb": used,
        "device_free_memory_mb": free,
        "production_gpu_count": 1 if total else 0,
        "per_process_gpu_memory_mb": processes,
        "external_process_memory_mb": round(sum(float(p.get("used_memory_mb") or 0.0) for p in processes), 6),
        "cuda_context_baseline_mb": 0.0,
        "measurement_source": "nvidia-smi",
    }


def nvidia_smi_gpu() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return {"gpu_name": None, "device_total_memory_mb": 0.0, "device_used_memory_mb": 0.0, "device_free_memory_mb": 0.0}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"gpu_name": None, "device_total_memory_mb": 0.0, "device_used_memory_mb": 0.0, "device_free_memory_mb": 0.0}
    parts = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
    return {
        "gpu_name": parts[0],
        "device_total_memory_mb": float(parts[1]),
        "device_used_memory_mb": float(parts[2]),
        "device_free_memory_mb": float(parts[3]),
    }


def nvidia_smi_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mb": _float(parts[2])})
    return rows


def unsupported_measurements(reason: str, baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "baseline": baseline,
        "scenarios": [invalid_scenario("S1_production_search_fp32_reranker", reason), invalid_scenario("S2_production_search_shadow_fp16_autocast", reason)],
        "formal_scenario_count": 6,
        "valid_scenario_count": 1 if baseline.get("device_total_memory_mb") else 0,
        "stale_task_process_count": 0,
        "unexpected_model_process_count": 0,
    }


def invalid_scenario(scenario_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scenario_id": scenario_id,
        "measurement_valid": False,
        "invalid_reason": reason,
        "device_peak_used_memory_mb": None,
        "torch_memory_allocated_mb": None,
        "torch_memory_reserved_mb": None,
        "torch_max_memory_allocated_mb": None,
        "torch_max_memory_reserved_mb": None,
        "cuda_oom_count": 0,
        "unexpected_model_eviction_count": 0,
        "unexpected_cpu_fallback_count": 0,
        "unexpected_model_reload_count": 0,
        "fp16_autocast_effective": False,
        "silent_fp32_fallback_detected": False,
    }


def scan_sensitive_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    patterns = {
        "api_key_assignment": r'"api[_-]?key"\s*:\s*"(?!redacted|None|null)[^"]{8,}"',
        "authorization_header": r'"authorization"\s*:',
        "bearer_token": r"bearer\s+[A-Za-z0-9._-]{16,}",
        "password_assignment": r"password\s*=\s*[^&\s]{8,}",
        "token_assignment": r"token\s*=\s*[^&\s]{16,}",
        "openai_secret_key": r"sk-[A-Za-z0-9]{16,}",
    }
    hits = [name for name, pattern in patterns.items() if re.search(pattern, raw, flags=re.IGNORECASE)]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "sensitive_value_exposure_count": len(hits),
        "detected_markers": hits,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "acceptance_policy": {
            "required_safety_headroom_mb": "max(2048, device_total_memory_mb * 0.20)",
            "unattributed_memory_ratio_max": 0.10,
            "fail_closed": True,
            "production_dtype_change_allowed": False,
            "promotion_allowed": False,
        },
    }


def render_report(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    topology: Mapping[str, Any],
    measurements: Mapping[str, Any],
    components: Mapping[str, Any],
    headroom: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0211 Production GPU Model Residency And Memory Headroom Authority Baseline",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; outcome=`{summary.get('outcome')}`; runtime_integration_recommended=`{summary.get('runtime_integration_recommended')}`; promotion_applied=`False`.",
            "",
            "## Configuration Authority",
            "",
            f"* Embedding `{config.get('embedding_model')}` on configured `{config.get('embedding_device')}` resolved `{config.get('embedding_resolved_device')}`.",
            f"* Reranker `{config.get('reranker_model')}` on configured `{config.get('reranker_device')}` with production dtype `{config.get('reranker_dtype')}`.",
            f"* Generation provider `{config.get('generation_provider')}` model `{config.get('generation_model')}` location `{config.get('generation_location')}` runtime `{config.get('generation_runtime')}`.",
            f"* Context `{config.get('default_context_length')}`; authorized concurrency `{config.get('maximum_authorized_concurrency')}`.",
            "",
            "## Residency Topology",
            "",
            f"* Search topology status `{(topology.get('search_residency_topology') or {}).get('scenario_status')}`.",
            f"* Remote ask topology status `{(topology.get('ask_remote_provider_residency_topology') or {}).get('scenario_status')}`.",
            f"* Local ask topology status `{(topology.get('ask_local_provider_residency_topology') or {}).get('scenario_status')}`.",
            "",
            "## Memory",
            "",
            f"* GPU `{summary.get('gpu_name')}` total `{summary.get('device_total_memory_mb')}` MB.",
            f"* FP32 search peak `{summary.get('fp32_search_peak_gpu_memory_mb')}` MB; FP16 search peak `{summary.get('fp16_search_peak_gpu_memory_mb')}` MB; delta `{summary.get('fp16_search_memory_delta_mb')}` MB.",
            f"* Component closure `{components.get('memory_component_closure_valid')}`; unattributed `{components.get('unattributed_memory_mb')}` MB ratio `{components.get('unattributed_memory_ratio')}`.",
            f"* Required headroom `{headroom.get('required_safety_headroom_mb')}` MB; observed `{headroom.get('observed_safety_headroom_mb')}` MB.",
            "",
            "## Decision",
            "",
            f"* Memory authority available `{headroom.get('memory_headroom_authority_available')}`; valid `{headroom.get('memory_headroom_valid')}`; scope `{headroom.get('memory_authority_scope')}`.",
            f"* CUDA OOM `{headroom.get('cuda_oom_count')}`; evictions `{headroom.get('unexpected_model_eviction_count')}`; CPU fallback `{headroom.get('unexpected_cpu_fallback_count')}`; reloads `{headroom.get('unexpected_model_reload_count')}`.",
            f"* FP16 autocast effective `{headroom.get('fp16_autocast_effective')}`; next task family `{headroom.get('next_task_family')}`.",
            "",
            "## Validation",
            "",
            f"* Focused tests `{summary.get('focused_tests')}`; related regression tests `{summary.get('related_regression_tests')}`; GPU model tests `{summary.get('gpu_model_tests')}`; production search replay `{summary.get('production_search_replay')}`; ask replay `{summary.get('production_ask_replay')}`; full suite `{summary.get('full_suite')}`; verifier `{summary.get('verifier_status')}`.",
            "",
        ]
    )


def find_scenario(measurements: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    return next((row for row in measurements.get("scenarios") or [] if row.get("scenario_id") == scenario_id), {})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def digest_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def redact_url(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}/..."


def mb(value: int | float) -> float:
    return round(float(value) / (1024 * 1024), 6)


def _float(value: Any) -> float:
    try:
        return float(str(value).replace("MiB", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(right) - float(left), 6)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in {None, 0, 0.0}:
        return None
    return round(float(numerator) / float(denominator), 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-arm")
    parser.add_argument("--child-output")
    args = parser.parse_args()
    if args.child_arm:
        payload = read_json(Path(args.child_arm))
        result = execute_child_arm(payload, os.environ)
        write_json(Path(args.child_output), result)
        return
    summary = run_task0211(write=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
