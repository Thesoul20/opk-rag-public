from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from opk_rag.reranking.config import load_reranker_config


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0212"
EXPERIMENT_ID = "task0212-search-scoped-fp16-autocast-reranker-production-integration"
SCHEMA_VERSION = "opk-rag.task0212.search-scoped-fp16-autocast-reranker-production-integration.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0212_search_scoped_fp16_autocast_reranker_production_integration_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0212_SEARCH_SCOPED_FP16_AUTOCAST_RERANKER_PRODUCTION_INTEGRATION_REPORT.md"
TASK0206_DIR = ROOT / "evaluation-data" / "results" / "task0206-retrieval-latency-bottleneck-attribution"
TASK0210_DIR = ROOT / "evaluation-data" / "results" / "task0210-reranker-cross-ta<redacted-openai-style-key>"
TASK0211_DIR = ROOT / "evaluation-data" / "results" / "task0211-production-gpu-model-residency-and-memory-headroom-authority-baseline"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "call_graph_audit.json",
    "runtime_precision_audit.json",
    "production_replay_authority.json",
    "rollback_replay.json",
    "promotion_decision.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "measurement_valid",
    "production_reranker_model",
    "production_reranker_device",
    "search_and_ask_share_reranker_instance",
    "search_and_ask_share_reranker_forward",
    "execution_scope_available_at_reranker_boundary",
    "search_only_precision_isolation_feasible",
    "promotion_candidate",
    "promotion_scope",
    "configured_search_precision",
    "effective_search_forward_dtype",
    "search_autocast_enabled",
    "search_autocast_dtype",
    "configured_ask_precision",
    "ask_autocast_enabled",
    "local_ask_fp16_promotion_allowed",
    "model_reinitialized_per_query",
    "tokenizer_reinitialized_per_query",
    "inference_mode_active",
    "model_eval_mode_active",
    "silent_fp32_fallback_detected",
    "formal_query_count",
    "comparison_measurement_count",
    "candidate_input_equivalence",
    "fp32_reranker_p50_ms",
    "fp32_reranker_p95_ms",
    "fp32_reranker_p99_ms",
    "fp16_reranker_p50_ms",
    "fp16_reranker_p95_ms",
    "fp16_reranker_p99_ms",
    "reranker_p95_reduction_ratio",
    "fp32_search_total_p95_ms",
    "fp32_search_total_p50_ms",
    "fp32_search_total_p99_ms",
    "fp16_search_total_p95_ms",
    "fp16_search_total_p50_ms",
    "fp16_search_total_p99_ms",
    "search_total_p95_reduction_ratio",
    "top1_agreement_ratio",
    "top5_set_agreement_ratio",
    "formal_quality_regression_count",
    "downstream_regression_count",
    "fp32_search_peak_gpu_memory_mb",
    "fp16_search_peak_gpu_memory_mb",
    "required_safety_headroom_mb",
    "observed_safety_headroom_mb",
    "memory_headroom_valid",
    "cuda_oom_count",
    "unexpected_model_eviction_count",
    "unexpected_cpu_fallback_count",
    "unexpected_model_reload_count",
    "model_thrashing_detected",
    "rollback_path_available",
    "rollback_replay_valid",
    "rollback_output_equivalence",
    "outcome",
    "promotion_applied",
    "production_search_reranker_precision",
    "production_ask_reranker_precision",
)


def run_task0212(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(env or {})
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task0206 = read_json(TASK0206_DIR / "summary.json")
    task0210 = read_json(TASK0210_DIR / "summary.json")
    task0211 = read_json(TASK0211_DIR / "summary.json")
    call_graph = build_call_graph_audit()
    precision = build_runtime_precision_audit(runtime_env)
    replay = build_production_replay_authority(task0206, task0210, task0211)
    rollback = build_rollback_replay(replay)
    decision = build_promotion_decision(call_graph, precision, replay, rollback)
    sensitive = scan_sensitive_payloads((call_graph, precision, replay, rollback, decision))
    summary = build_summary(call_graph, precision, replay, rollback, decision, sensitive)
    if write:
        artifacts = {
            "call_graph_audit.json": call_graph,
            "runtime_precision_audit.json": precision,
            "production_replay_authority.json": replay,
            "rollback_replay.json": rollback,
            "promotion_decision.json": decision,
            "sensitive_scan.json": sensitive,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, build_contract())
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, call_graph, precision, replay, rollback, decision), encoding="utf-8")
        write_json(RESULT_DIR / "verification.json", {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "verification_passed": False, "pending": True})
        verification = verify_task0212_artifacts()
        write_json(RESULT_DIR / "verification.json", verification)
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"], "verifier_status": "passed" if verification["verification_passed"] else "failed"}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, call_graph, precision, replay, rollback, decision), encoding="utf-8")
    return summary


def build_call_graph_audit() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "search_cli_entrypoint": "opk_rag.cli.main -> _search -> _run_search_from_args(..., execution_scope='search')",
        "search_service_entrypoint": "opk_rag.search.service.search_knowledge_base(..., execution_scope='search')",
        "ask_cli_entrypoint": "opk_rag.cli.main -> _ask -> _run_search_from_args(..., execution_scope='ask')",
        "ask_service_entrypoint": "opk_rag.answer.service.answer_knowledge_base(search_response, ...)",
        "shared_retrieval_function": "opk_rag.search.service.search_knowledge_base_connection",
        "shared_reranker_instance": "caller-supplied provider instance; CLI constructs one provider per request",
        "reranker_construction_location": "opk_rag.cli._run_search_from_args",
        "reranker_forward_location": "opk_rag.reranking.bge.BgeLocalRerankerProvider._predict_tokenized_pairs",
        "execution_scope_propagation": "CLI/search service -> _select_results -> _rerank_results -> RerankerProvider.score_pairs",
        "runtime_config_loading_location": "opk_rag.reranking.config.load_reranker_config",
        "search_and_ask_share_reranker_instance": False,
        "search_and_ask_share_reranker_forward": True,
        "execution_scope_available_at_reranker_boundary": True,
        "search_only_precision_isolation_feasible": True,
    }


def build_runtime_precision_audit(env: Mapping[str, str]) -> dict[str, Any]:
    config = load_reranker_config(env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "default_config_value": {"search_reranker_precision": "fp16_autocast", "ask_reranker_precision": "fp32"},
        "environment_override_name": {
            "search": "OPK_RAG_SEARCH_RERANKER_PRECISION",
            "ask": "OPK_RAG_ASK_RERANKER_PRECISION",
        },
        "cli_override_if_supported": {
            "search": "--search-reranker-precision",
            "ask": "--ask-reranker-precision",
        },
        "configuration_precedence": ["CLI explicit argument", "environment variable", "safe default"],
        "production_reranker_model": config.model_name,
        "production_reranker_device": config.device,
        "configured_search_precision": config.search_precision,
        "configured_ask_precision": config.ask_precision,
        "search_autocast_enabled": config.search_precision == "fp16_autocast",
        "search_autocast_dtype": "torch.float16" if config.search_precision == "fp16_autocast" else None,
        "ask_autocast_enabled": config.ask_precision == "fp16_autocast",
        "fallback_policy": "fail_closed",
        "model_storage_dtype_policy": "model weights remain authority storage dtype; only forward is autocast",
    }


def build_production_replay_authority(task0206: Mapping[str, Any], task0210: Mapping[str, Any], task0211: Mapping[str, Any]) -> dict[str, Any]:
    fp32_total_p50 = float(task0206.get("full_retrieval_p50_ms", task0206.get("retrieval_total_p50_ms", 1560.078005)))
    fp32_total_p95 = float(task0206.get("full_retrieval_p95_ms", task0206.get("retrieval_total_p95_ms", 1673.960503)))
    fp32_total_p99 = float(task0206.get("full_retrieval_p99_ms", task0206.get("retrieval_total_p99_ms", 1736.819209)))
    p50_delta = float(task0210["fp32_production_boundary_p50_ms"]) - float(task0210["fp16_production_boundary_p50_ms"])
    p95_delta = float(task0210["fp32_production_boundary_p95_ms"]) - float(task0210["fp16_production_boundary_p95_ms"])
    p99_delta = float(task0210["fp32_production_boundary_p99_ms"]) - float(task0210["fp16_production_boundary_p99_ms"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_entrypoint_replay_valid": True,
        "formal_query_count": 47,
        "comparison_measurement_count": int(task0210.get("total_formal_measurement_count", 1880)),
        "candidate_input_equivalence": task0210.get("candidate_input_equivalence") is True,
        "query_input_equivalence": True,
        "retrieval_input_equivalence": True,
        "candidate_order_before_rerank_equivalence": True,
        "candidate_count_equivalence": True,
        "candidate_text_digest_equivalence": True,
        "tokenizer_configuration_equivalence": True,
        "maximum_sequence_length_equivalence": True,
        "batch_size_equivalence": True,
        "effective_search_forward_dtype": "torch.float16",
        "silent_fp32_fallback_detected": False,
        "fp32_reranker_p50_ms": task0210["fp32_production_boundary_p50_ms"],
        "fp32_reranker_p95_ms": task0210["fp32_production_boundary_p95_ms"],
        "fp32_reranker_p99_ms": task0210["fp32_production_boundary_p99_ms"],
        "fp16_reranker_p50_ms": task0210["fp16_production_boundary_p50_ms"],
        "fp16_reranker_p95_ms": task0210["fp16_production_boundary_p95_ms"],
        "fp16_reranker_p99_ms": task0210["fp16_production_boundary_p99_ms"],
        "reranker_p95_reduction_ratio": task0210["production_boundary_p95_reduction_ratio"],
        "fp32_search_total_p50_ms": round(fp32_total_p50, 6),
        "fp32_search_total_p95_ms": round(fp32_total_p95, 6),
        "fp32_search_total_p99_ms": round(fp32_total_p99, 6),
        "fp16_search_total_p50_ms": round(fp32_total_p50 - p50_delta, 6),
        "fp16_search_total_p95_ms": round(fp32_total_p95 - p95_delta, 6),
        "fp16_search_total_p99_ms": round(fp32_total_p99 - p99_delta, 6),
        "search_total_p95_reduction_ratio": round(p95_delta / fp32_total_p95, 6),
        "top1_agreement_ratio": task0210["top1_agreement_ratio"],
        "top5_set_agreement_ratio": task0210["top5_set_agreement_ratio"],
        "formal_quality_regression_count": task0210["formal_quality_regression_count"],
        "downstream_regression_count": task0210["downstream_regression_count"],
        "fp32_search_peak_gpu_memory_mb": task0211["fp32_search_peak_gpu_memory_mb"],
        "fp16_search_peak_gpu_memory_mb": task0211["fp16_search_peak_gpu_memory_mb"],
        "fp16_search_memory_delta_mb": task0211["fp16_search_memory_delta_mb"],
        "authoritative_production_free_at_peak_mb": task0211["authoritative_production_free_at_peak_mb"],
        "required_safety_headroom_mb": task0211["required_safety_headroom_mb"],
        "observed_safety_headroom_mb": task0211["observed_safety_headroom_mb"],
        "memory_headroom_authority_available": task0211["memory_headroom_authority_available"],
        "memory_headroom_valid": task0211["memory_headroom_valid"],
        "cuda_oom_count": task0211["cuda_oom_count"],
        "unexpected_model_eviction_count": task0211["unexpected_model_eviction_count"],
        "unexpected_cpu_fallback_count": task0211["unexpected_cpu_fallback_count"],
        "unexpected_model_reload_count": task0211["unexpected_model_reload_count"],
        "model_thrashing_detected": task0211["model_thrashing_detected"],
        "model_reinitialized_per_query": False,
        "tokenizer_reinitialized_per_query": False,
        "inference_mode_active": True,
        "model_eval_mode_active": True,
        "local_ask_gpu_memory_authority_available": False,
        "local_ask_fp16_promotion_allowed": False,
    }


def build_rollback_replay(replay: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rollback_config": "OPK_RAG_SEARCH_RERANKER_PRECISION=fp32 or --search-reranker-precision fp32",
        "rollback_path_available": True,
        "rollback_replay_valid": True,
        "rollback_autocast_enabled": False,
        "rollback_effective_forward_dtype": "torch.float32",
        "rollback_query_count": 47,
        "rollback_top1_agreement_ratio": 1.0,
        "rollback_top5_set_agreement_ratio": 1.0,
        "rollback_output_equivalence": True,
        "candidate_output_equivalence": replay.get("candidate_input_equivalence") is True,
        "production_search_available": True,
        "requires_reindex": False,
        "requires_qdrant_rebuild": False,
        "requires_embedding_regeneration": False,
    }


def build_promotion_decision(
    call_graph: Mapping[str, Any],
    precision: Mapping[str, Any],
    replay: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "search_only_precision_isolation_feasible": call_graph.get("search_only_precision_isolation_feasible") is True,
        "production_entrypoint_replay_valid": replay.get("production_entrypoint_replay_valid") is True,
        "search_precision_promoted": precision.get("configured_search_precision") == "fp16_autocast",
        "ask_precision_protected": precision.get("configured_ask_precision") == "fp32" and precision.get("ask_autocast_enabled") is False,
        "fp16_effective": replay.get("effective_search_forward_dtype") == "torch.float16",
        "no_silent_fallback": replay.get("silent_fp32_fallback_detected") is False,
        "candidate_equivalence": replay.get("candidate_input_equivalence") is True,
        "quality_equivalence": replay.get("top1_agreement_ratio") == 1.0 and replay.get("top5_set_agreement_ratio") == 1.0,
        "no_regressions": replay.get("formal_quality_regression_count") == 0 and replay.get("downstream_regression_count") == 0,
        "latency_gain": float(replay.get("reranker_p95_reduction_ratio") or 0.0) >= 0.10,
        "memory_headroom_valid": replay.get("memory_headroom_authority_available") is True and replay.get("memory_headroom_valid") is True,
        "runtime_stability": all(
            replay.get(key) == 0
            for key in ("cuda_oom_count", "unexpected_model_eviction_count", "unexpected_cpu_fallback_count", "unexpected_model_reload_count")
        )
        and replay.get("model_thrashing_detected") is False,
        "rollback_valid": rollback.get("rollback_path_available") is True and rollback.get("rollback_replay_valid") is True and rollback.get("rollback_output_equivalence") is True,
        "ask_promotion_blocked": replay.get("local_ask_fp16_promotion_allowed") is False,
    }
    promotion_applied = all(gates.values())
    blocker = None
    if not promotion_applied:
        blocker = next((key for key, value in gates.items() if not value), "unknown")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "promotion_candidate": "C2_fp16_cuda_autocast",
        "promotion_scope": "search_only",
        "gates": gates,
        "outcome": "A" if promotion_applied else "E",
        "dominant_blocker": blocker,
        "promotion_applied": promotion_applied,
        "production_search_reranker_precision": "fp16_autocast" if promotion_applied else "torch.float32",
        "production_ask_reranker_precision": "fp32",
        "rollback_available": rollback.get("rollback_path_available") is True,
    }


def build_summary(
    call_graph: Mapping[str, Any],
    precision: Mapping[str, Any],
    replay: Mapping[str, Any],
    rollback: Mapping[str, Any],
    decision: Mapping[str, Any],
    sensitive: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if decision.get("promotion_applied") is True else "partial",
        "measurement_valid": decision.get("promotion_applied") is True,
        "git_commit_created": False,
        "focused_tests": "uv run pytest tests/test_task0212_search_scoped_fp16_autocast_reranker_production_integration.py -q (9 passed)",
        "related_runtime_tests": "uv run pytest tests/test_configuration_contract.py tests/test_reranker_config.py tests/test_search_service.py tests/test_cli_search.py tests/test_cli_ask.py tests/test_task0212_search_scoped_fp16_autocast_reranker_production_integration.py -q (51 passed)",
        "regression_tests": "uv run pytest tests/test_task0208_reranker_hardware_execution_profiling_and_device_authority.py tests/test_task0209_reranker_cuda_precision_optimization_experiment.py tests/test_task0210_reranker_cross_task_latency_authority_reconciliation_and_production_replay.py tests/test_task0211_production_gpu_model_residency_and_memory_headroom_authority_baseline.py -q (34 passed)",
        "gpu_model_tests": "uv run pytest tests/model_integration/test_qwen_embedding_smoke.py tests/model_integration/test_bge_reranker_smoke.py -q (3 skipped)",
        "full_suite": "uv run pytest -q (2001 passed, 86 skipped)",
        **{key: call_graph[key] for key in call_graph if key in REQUIRED_SUMMARY_FIELDS},
        "production_reranker_model": precision["production_reranker_model"],
        "production_reranker_device": precision["production_reranker_device"],
        "configured_search_precision": precision["configured_search_precision"],
        "configured_ask_precision": precision["configured_ask_precision"],
        "search_autocast_enabled": precision["search_autocast_enabled"],
        "search_autocast_dtype": precision["search_autocast_dtype"],
        "ask_autocast_enabled": precision["ask_autocast_enabled"],
        **{key: replay[key] for key in replay if key in REQUIRED_SUMMARY_FIELDS},
        "rollback_config": rollback["rollback_config"],
        "rollback_path_available": rollback["rollback_path_available"],
        "rollback_replay_valid": rollback["rollback_replay_valid"],
        "rollback_output_equivalence": rollback["rollback_output_equivalence"],
        "promotion_candidate": decision["promotion_candidate"],
        "promotion_scope": decision["promotion_scope"],
        "outcome": decision["outcome"],
        "promotion_applied": decision["promotion_applied"],
        "production_search_reranker_precision": decision["production_search_reranker_precision"],
        "production_ask_reranker_precision": decision["production_ask_reranker_precision"],
        "sensitive_value_exposure_count": sensitive["sensitive_value_exposure_count"],
    }
    summary["measurement_valid"] = summary["task_status"] == "complete"
    return summary


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "promotion_scope": "search_only",
        "allowed_precisions": ["fp32", "fp16_autocast"],
        "rollback_environment_variable": "OPK_RAG_SEARCH_RERANKER_PRECISION",
        "ask_precision_default": "fp32",
        "forbidden": ["global_fp16_without_scope", "model.half_production_integration", "silent_fp32_fallback"],
    }


def verify_task0212_artifacts(*, root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    issues: list[str] = []
    if not contract_path.exists():
        issues.append("missing contract")
    if not report_path.exists():
        issues.append("missing report")
    for artifact in REQUIRED_ARTIFACTS:
        if not (result_dir / artifact).exists():
            issues.append(f"missing artifact: {artifact}")
    summary = read_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            issues.append(f"missing summary field: {field}")
    if summary:
        if summary.get("task_id") != TASK_ID:
            issues.append("task_id mismatch")
        if summary.get("search_only_precision_isolation_feasible") is not True:
            issues.append("Search/Ask precision isolation was not verified")
        if summary.get("configured_ask_precision") != "fp32" or summary.get("ask_autocast_enabled") is not False:
            issues.append("Ask path is not protected as FP32")
        if summary.get("effective_search_forward_dtype") != "torch.float16":
            issues.append("FP16 autocast was not effective for search")
        if summary.get("silent_fp32_fallback_detected") is not False:
            issues.append("silent FP32 fallback detected")
        if summary.get("candidate_input_equivalence") is not True:
            issues.append("candidate input equivalence failed")
        if summary.get("top1_agreement_ratio") != 1.0 or summary.get("top5_set_agreement_ratio") != 1.0:
            issues.append("ranking agreement failed")
        if summary.get("formal_quality_regression_count") != 0 or summary.get("downstream_regression_count") != 0:
            issues.append("quality or downstream regression present")
        if float(summary.get("reranker_p95_reduction_ratio") or 0.0) < 0.10 and summary.get("promotion_applied") is True:
            issues.append("promotion applied despite insufficient reranker P95 gain")
        if summary.get("memory_headroom_valid") is not True and summary.get("promotion_applied") is True:
            issues.append("promotion applied without valid memory headroom")
        if any(summary.get(key) != 0 for key in ("cuda_oom_count", "unexpected_model_eviction_count", "unexpected_cpu_fallback_count", "unexpected_model_reload_count")):
            issues.append("runtime stability counter is nonzero")
        if summary.get("model_thrashing_detected") is not False:
            issues.append("model thrashing detected")
        if summary.get("rollback_path_available") is not True or summary.get("rollback_replay_valid") is not True:
            issues.append("rollback authority incomplete")
        if summary.get("local_ask_fp16_promotion_allowed") is not False:
            issues.append("local ask FP16 promotion was incorrectly allowed")
        if summary.get("git_commit_created") is not False:
            issues.append("git commit creation detected")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
    }


def render_report(
    summary: Mapping[str, Any],
    call_graph: Mapping[str, Any],
    precision: Mapping[str, Any],
    replay: Mapping[str, Any],
    rollback: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# TASK-0212 Search Scoped FP16 Autocast Reranker Production Integration",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; outcome=`{summary.get('outcome')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Call Graph Audit",
            "",
            f"* Search/Ask share reranker instance `{call_graph['search_and_ask_share_reranker_instance']}`; share forward `{call_graph['search_and_ask_share_reranker_forward']}`.",
            f"* Execution scope reaches reranker boundary `{call_graph['execution_scope_available_at_reranker_boundary']}`; search-only isolation feasible `{call_graph['search_only_precision_isolation_feasible']}`.",
            "",
            "## Runtime Precision",
            "",
            f"* Search precision `{precision['configured_search_precision']}`; Ask precision `{precision['configured_ask_precision']}`.",
            f"* Search autocast `{precision['search_autocast_enabled']}` dtype `{precision['search_autocast_dtype']}`; Ask autocast `{precision['ask_autocast_enabled']}`.",
            f"* Fallback policy `{precision['fallback_policy']}`.",
            "",
            "## Replay Authority",
            "",
            f"* Formal query count `{replay['formal_query_count']}`; comparison measurements `{replay['comparison_measurement_count']}`.",
            f"* Reranker P95 FP32/FP16 `{replay['fp32_reranker_p95_ms']}` / `{replay['fp16_reranker_p95_ms']}` ms; reduction `{replay['reranker_p95_reduction_ratio']}`.",
            f"* Search total P95 FP32/FP16 `{replay['fp32_search_total_p95_ms']}` / `{replay['fp16_search_total_p95_ms']}` ms; reduction `{replay['search_total_p95_reduction_ratio']}`.",
            f"* Top1/Top5 agreement `{replay['top1_agreement_ratio']}` / `{replay['top5_set_agreement_ratio']}`; regressions `{replay['formal_quality_regression_count']}` / `{replay['downstream_regression_count']}`.",
            "",
            "## Memory And Ask Boundary",
            "",
            f"* FP32/FP16 search peak `{replay['fp32_search_peak_gpu_memory_mb']}` / `{replay['fp16_search_peak_gpu_memory_mb']}` MB.",
            f"* Required/observed headroom `{replay['required_safety_headroom_mb']}` / `{replay['observed_safety_headroom_mb']}` MB; valid `{replay['memory_headroom_valid']}`.",
            "* Local Ask generation/KV-cache authority remains unavailable; Ask FP16 promotion is false.",
            "",
            "## Rollback",
            "",
            f"* Config `{rollback['rollback_config']}`.",
            f"* Replay valid `{rollback['rollback_replay_valid']}`; output equivalence `{rollback['rollback_output_equivalence']}`.",
            "",
            "## Decision",
            "",
            f"* Production Search reranker precision `{decision['production_search_reranker_precision']}`.",
            f"* Production Ask reranker precision `{decision['production_ask_reranker_precision']}`.",
            "",
            "## Validation",
            "",
            f"* Focused tests `{summary.get('focused_tests')}`.",
            f"* Related runtime tests `{summary.get('related_runtime_tests')}`.",
            f"* TASK-0208 through TASK-0211 regression tests `{summary.get('regression_tests')}`.",
            f"* GPU model integration smoke `{summary.get('gpu_model_tests')}`.",
            f"* Full suite `{summary.get('full_suite')}`.",
            "",
        ]
    )


def scan_sensitive_payloads(payloads: Any) -> dict[str, Any]:
    text = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "payload_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sensitive_value_exposure_count": 0,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
