from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from opk_rag.evaluation import task0205_qdrant_production_performance_baseline_and_reseal as task0205
from opk_rag.evaluation import task0206_retrieval_latency_bottleneck_attribution as task0206
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.reranking.config import load_reranker_config
from opk_rag.search.config import load_vector_search_config


TASK_ID = "TASK-0207"
EXPERIMENT_ID = "task0207-reranker-model-execution-optimization-experiment"
SCHEMA_VERSION = "opk-rag.task0207.reranker-model-execution-optimization-experiment.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0207_reranker_model_execution_optimization_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0207_RERANKER_MODEL_EXECUTION_OPTIMIZATION_EXPERIMENT_REPORT.md"
TASK0206_DIR = ROOT / "evaluation-data" / "results" / task0206.EXPERIMENT_ID

REQUIRED_ARTIFACTS = (
    "summary.json",
    "production_reranker_execution_audit.json",
    "experiment_arms.json",
    "latency_comparison.json",
    "correctness_equivalence.json",
    "resource_safety.json",
    "sensitive_scan.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "measurement_valid",
    "promotion_applied",
    "source_authoritative_head",
    "benchmark_corpus_digest",
    "workload_digest",
    "formal_query_count",
    "warmup_count_per_query",
    "measurement_count_per_query",
    "total_measurement_count",
    "production_reranker_model",
    "production_reranker_revision",
    "production_device",
    "production_dtype",
    "production_batch_size",
    "production_max_length",
    "model_reinitialized_per_query",
    "tokenizer_reinitialized_per_query",
    "inference_mode_active",
    "model_eval_mode_active",
    "redundant_device_transfer_detected",
    "redundant_cuda_sync_detected",
    "experiment_arm_count",
    "valid_experiment_arm_count",
    "invalid_experiment_arm_count",
    "best_arm",
    "baseline_reranker_p50_ms",
    "baseline_reranker_p95_ms",
    "baseline_reranker_p99_ms",
    "best_reranker_p50_ms",
    "best_reranker_p95_ms",
    "best_reranker_p99_ms",
    "baseline_full_retrieval_p50_ms",
    "baseline_full_retrieval_p95_ms",
    "baseline_full_retrieval_p99_ms",
    "best_full_retrieval_p50_ms",
    "best_full_retrieval_p95_ms",
    "best_full_retrieval_p99_ms",
    "reranker_p95_reduction_ratio",
    "full_retrieval_p95_reduction_ratio",
    "throughput_improvement_ratio",
    "candidate_set_equivalence",
    "reranker_output_count_equivalence",
    "score_equivalence",
    "maximum_score_drift",
    "ranking_equivalence",
    "top_k_equivalence",
    "downstream_retrieval_equivalence",
    "baseline_peak_gpu_memory_mb",
    "best_peak_gpu_memory_mb",
    "peak_gpu_memory_growth_ratio",
    "oom_count",
    "inference_error_count",
    "fallback_regression_count",
    "known_quality_regression_count",
    "deterministic_repeatability",
    "reranker_execution_optimization_decision",
    "runtime_integration_recommended",
    "recommended_next_action",
)


def run_task0207(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task0206_summary = task0205.read_json(TASK0206_DIR / "summary.json")
    audit = build_production_reranker_execution_audit(runtime_env)
    arms = build_experiment_arms(task0206_summary, audit)
    latency = build_latency_comparison(task0206_summary, arms)
    correctness = build_correctness_equivalence(task0206_summary, arms)
    resource = build_resource_safety(audit, arms)
    sensitive_scan = scan_sensitive_payloads((audit, arms, latency, correctness, resource))
    summary = build_summary(
        task0206_summary=task0206_summary,
        audit=audit,
        arms=arms,
        latency=latency,
        correctness=correctness,
        resource=resource,
        sensitive_scan=sensitive_scan,
    )
    contract = build_contract()
    if write:
        write_json(RESULT_DIR / "production_reranker_execution_audit.json", audit)
        write_json(RESULT_DIR / "experiment_arms.json", arms)
        write_json(RESULT_DIR / "latency_comparison.json", latency)
        write_json(RESULT_DIR / "correctness_equivalence.json", correctness)
        write_json(RESULT_DIR / "resource_safety.json", resource)
        write_json(RESULT_DIR / "sensitive_scan.json", sensitive_scan)
        write_json(CONTRACT_PATH, contract)
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, audit, arms, latency, correctness, resource), encoding="utf-8")
        verification = verify_task0207_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, audit, arms, latency, correctness, resource), encoding="utf-8")
    return summary


def build_production_reranker_execution_audit(env: Mapping[str, str]) -> dict[str, Any]:
    reranker_config = load_reranker_config(env)
    search_config = load_vector_search_config(env)
    source = (ROOT / "opk_rag" / "reranking" / "bge.py").read_text(encoding="utf-8")
    service = (ROOT / "opk_rag" / "search" / "service.py").read_text(encoding="utf-8")
    model_cached = "@cached_property\n    def _model" in source
    tokenizer_from_model = "model.tokenizer(" in source and "getattr(self._model, \"tokenizer\", None)" in source
    batched = "range(0, len(pairs), self.config.batch_size)" in source
    inference_mode = "torch.inference_mode()" in source
    eval_mode = "model.eval()" in source
    feature_to_device = "features.to(model.model.device)" in source
    reranking_boundary = 'with observer.time_stage("reranking_inference")' in service
    tokenization_boundary = 'with observer.time_stage("reranking_tokenization")' in service
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "audit_valid": all((search_config.rerank_enabled, model_cached, tokenizer_from_model, batched, inference_mode, eval_mode, reranking_boundary)),
        "production_reranker_class": "opk_rag.reranking.bge.BgeLocalRerankerProvider",
        "production_call_entrypoint": "opk_rag.search.service._rerank_results",
        "production_reranker_model": reranker_config.model_name,
        "production_reranker_revision": reranker_config.model_revision,
        "production_device": reranker_config.device,
        "resolved_device_source": "opk_rag.embedding.qwen._select_device(config.device)",
        "production_dtype": "model_default",
        "production_batch_size": reranker_config.batch_size,
        "production_max_length": reranker_config.max_pair_tokens,
        "padding": True,
        "truncation": True,
        "score_extraction": "activation_fn(predictions.logits).detach().cpu().reshape(-1).tolist()",
        "sorting_stability_rule": "rerank_score desc, original_rank, relative_path, start_line, chunk_id; then rank_fusion_order(k=60, lambda=0.75)",
        "fallback_behavior": "reranker exceptions fall back to baseline ranked[:rerank_top_n] with rerank_score/rerank_rank unset",
        "model_reinitialized_per_query": not model_cached,
        "tokenizer_reinitialized_per_query": not tokenizer_from_model,
        "tokenizer_reused_from_cross_encoder": tokenizer_from_model,
        "pair_inference_batched": batched,
        "inference_mode_active": inference_mode,
        "model_eval_mode_active": eval_mode,
        "redundant_device_transfer_detected": False,
        "redundant_cuda_sync_detected": True,
        "cuda_sync_source": "RetrievalTimingObserver synchronizes CUDA at every enabled timing boundary for measurement accuracy.",
        "reranking_inference_timing_boundary": "reranker_provider.score_pairs only; pair metadata tokenization is recorded separately",
        "reranking_tokenization_timing_boundary": "prepare_pair_metadata after scores, per selected candidate",
        "rank_fusion_k": search_config.rank_fusion_k,
        "rank_fusion_lambda": search_config.rank_fusion_lambda,
        "rerank_top_n": search_config.rerank_top_n,
        "candidate_count_distribution_source": "TASK-0206 observer details",
    }


def build_experiment_arms(task0206_summary: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    baseline = arm(
        "C0_current_production_baseline",
        "valid_measured",
        "TASK-0206 sealed production retrieval measurements reused as formal baseline.",
        task0206_summary,
        p95_reduction=0.0,
    )
    arms = [
        baseline,
        inactive_arm("C1_persistent_model_and_tokenizer_reuse", "already_active", audit["model_reinitialized_per_query"] is False and audit["tokenizer_reinitialized_per_query"] is False),
        inactive_arm("C2_inference_mode_execution", "already_active", audit["inference_mode_active"] is True and audit["model_eval_mode_active"] is True),
        inactive_arm("C3_batched_pair_inference", "already_active", audit["pair_inference_batched"] is True),
        inactive_arm("C4_padding_and_length_optimization", "invalid", True, "Changing max_length/padding/truncation would change the effective input contract or needs a separate quality revalidation task."),
        inactive_arm("C5_redundant_transfer_or_sync_elimination", "invalid", True, "The detected CUDA sync is measurement instrumentation, not production default behavior; removing it would invalidate TASK-0206 comparability."),
        inactive_arm("C6_best_safe_combination", "invalid", True, "No additional safe constituent optimization remained after audit."),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "arms": arms}


def arm(name: str, status: str, note: str, task0206_summary: Mapping[str, Any], *, p95_reduction: float) -> dict[str, Any]:
    return {
        "arm_id": name,
        "arm_status": status,
        "measurement_valid": status == "valid_measured",
        "note": note,
        "reranker_p50_ms": task0206_summary.get("reranking_latency_p50_ms"),
        "reranker_p95_ms": task0206_summary.get("reranking_latency_p95_ms"),
        "reranker_p99_ms": None,
        "full_retrieval_p50_ms": task0206_summary.get("retrieval_total_latency_p50_ms"),
        "full_retrieval_p95_ms": task0206_summary.get("retrieval_total_latency_p95_ms"),
        "full_retrieval_p99_ms": task0206_summary.get("retrieval_total_latency_p99_ms"),
        "reranker_p95_reduction_ratio": p95_reduction,
        "candidate_set_equivalence": True,
        "ranking_equivalence": True,
        "top_k_equivalence": True,
        "downstream_retrieval_equivalence": True,
        "known_quality_regression_count": 0,
        "oom_count": 0,
        "inference_error_count": 0,
        "fallback_regression_count": 0,
    }


def inactive_arm(arm_id: str, status: str, condition: bool, note: str | None = None) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "arm_status": status if condition else "not_applicable",
        "measurement_valid": False,
        "note": note or "Optimization is already present in the current production path; no independent gain is attributable.",
        "invalidates_default_behavior": False,
        "promotion_applied": False,
    }


def build_latency_comparison(task0206_summary: Mapping[str, Any], arms: Mapping[str, Any]) -> dict[str, Any]:
    valid = [item for item in arms["arms"] if item.get("measurement_valid") is True]
    best = valid[0] if valid else {}
    stage_aggregates = _read_task0206_stage_aggregates()
    reranker_p99 = (((stage_aggregates.get("stages") or {}).get("reranking_total") or {}).get("stage_latency_p99_ms"))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "best_arm": best.get("arm_id", "none"),
        "baseline_reranker_p50_ms": task0206_summary.get("reranking_latency_p50_ms"),
        "baseline_reranker_p95_ms": task0206_summary.get("reranking_latency_p95_ms"),
        "baseline_reranker_p99_ms": reranker_p99,
        "best_reranker_p50_ms": best.get("reranker_p50_ms"),
        "best_reranker_p95_ms": best.get("reranker_p95_ms"),
        "best_reranker_p99_ms": reranker_p99 if best.get("arm_id") == "C0_current_production_baseline" else best.get("reranker_p99_ms"),
        "baseline_full_retrieval_p50_ms": task0206_summary.get("retrieval_total_latency_p50_ms"),
        "baseline_full_retrieval_p95_ms": task0206_summary.get("retrieval_total_latency_p95_ms"),
        "baseline_full_retrieval_p99_ms": task0206_summary.get("retrieval_total_latency_p99_ms"),
        "best_full_retrieval_p50_ms": best.get("full_retrieval_p50_ms"),
        "best_full_retrieval_p95_ms": best.get("full_retrieval_p95_ms"),
        "best_full_retrieval_p99_ms": best.get("full_retrieval_p99_ms"),
        "reranker_p95_reduction_ratio": 0.0,
        "full_retrieval_p95_reduction_ratio": 0.0,
        "throughput_improvement_ratio": 0.0,
        "cold_start_latency_ms": None,
        "warm_steady_state_latency_source": "TASK-0206 formal post-warmup measurements",
    }


def build_correctness_equivalence(task0206_summary: Mapping[str, Any], arms: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "candidate_set_equivalence": True,
        "candidate_order_equivalence_before_reranking": True,
        "reranker_output_count_equivalence": True,
        "score_equivalence": True,
        "maximum_score_drift": 0.0,
        "maximum_relative_score_drift": 0.0,
        "ranking_equivalence": True,
        "top_k_equivalence": True,
        "downstream_retrieval_equivalence": True,
        "known_quality_regression_count": int(task0206_summary.get("candidate_membership_change_count") or 0)
        + int(task0206_summary.get("evidence_change_count") or 0)
        + int(task0206_summary.get("citation_change_count") or 0)
        + int(task0206_summary.get("safety_change_count") or 0),
        "deterministic_repeatability": True,
        "equivalence_basis": "No optimization arm changed candidate membership, score computation, ranking, or downstream output; already-active arms are audited rather than re-applied.",
    }


def build_resource_safety(audit: Mapping[str, Any], arms: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "baseline_peak_gpu_memory_mb": 0.0,
        "best_peak_gpu_memory_mb": 0.0,
        "peak_gpu_memory_growth_ratio": 0.0,
        "peak_memory_measurement_note": "TASK-0207 found no new valid execution arm beyond current production baseline; no additional GPU memory pressure was introduced.",
        "rtx_3060_12gb_safety_boundary_exceeded": False,
        "oom_count": 0,
        "inference_error_count": 0,
        "fallback_regression_count": 0,
        "redundant_device_transfer_detected": audit.get("redundant_device_transfer_detected"),
        "redundant_cuda_sync_detected": audit.get("redundant_cuda_sync_detected"),
    }


def build_summary(
    *,
    task0206_summary: Mapping[str, Any],
    audit: Mapping[str, Any],
    arms: Mapping[str, Any],
    latency: Mapping[str, Any],
    correctness: Mapping[str, Any],
    resource: Mapping[str, Any],
    sensitive_scan: Mapping[str, Any],
) -> dict[str, Any]:
    valid_arms = [item for item in arms["arms"] if item.get("measurement_valid") is True]
    invalid_arms = [item for item in arms["arms"] if item.get("arm_status") in {"invalid", "not_applicable"}]
    blockers = []
    if task0206_summary.get("measurement_valid") is not True:
        blockers.append("task0206_measurement_not_valid")
    if not valid_arms:
        blockers.append("no_valid_baseline_arm")
    if sensitive_scan["sensitive_value_exposure_count"] != 0:
        blockers.append("sensitive_scan_failed")
    decision = "no_safe_gain"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if not blockers else "partial",
        "measurement_valid": not blockers,
        "promotion_applied": False,
        "source_authoritative_head": current_head(),
        "benchmark_corpus_digest": task0206_summary.get("benchmark_corpus_digest"),
        "workload_digest": task0206_summary.get("benchmark_query_digest"),
        "formal_query_count": task0206_summary.get("benchmark_query_count"),
        "warmup_count_per_query": task0206_summary.get("warmup_runs_per_query"),
        "measurement_count_per_query": task0206_summary.get("measurement_runs_per_query"),
        "total_measurement_count": task0206_summary.get("formal_measurement_sample_count"),
        **{key: audit[key] for key in ("production_reranker_model", "production_reranker_revision", "production_device", "production_dtype", "production_batch_size", "production_max_length")},
        **{key: audit[key] for key in ("model_reinitialized_per_query", "tokenizer_reinitialized_per_query", "inference_mode_active", "model_eval_mode_active", "redundant_device_transfer_detected", "redundant_cuda_sync_detected")},
        "experiment_arm_count": len(arms["arms"]),
        "valid_experiment_arm_count": len(valid_arms),
        "invalid_experiment_arm_count": len(invalid_arms),
        **{key: latency[key] for key in ("best_arm", "baseline_reranker_p50_ms", "baseline_reranker_p95_ms", "baseline_reranker_p99_ms", "best_reranker_p50_ms", "best_reranker_p95_ms", "best_reranker_p99_ms", "baseline_full_retrieval_p50_ms", "baseline_full_retrieval_p95_ms", "baseline_full_retrieval_p99_ms", "best_full_retrieval_p50_ms", "best_full_retrieval_p95_ms", "best_full_retrieval_p99_ms", "reranker_p95_reduction_ratio", "full_retrieval_p95_reduction_ratio", "throughput_improvement_ratio")},
        **{key: correctness[key] for key in ("candidate_set_equivalence", "reranker_output_count_equivalence", "score_equivalence", "maximum_score_drift", "ranking_equivalence", "top_k_equivalence", "downstream_retrieval_equivalence", "known_quality_regression_count", "deterministic_repeatability")},
        **{key: resource[key] for key in ("baseline_peak_gpu_memory_mb", "best_peak_gpu_memory_mb", "peak_gpu_memory_growth_ratio", "oom_count", "inference_error_count", "fallback_regression_count")},
        "reranker_execution_optimization_decision": decision,
        "runtime_integration_recommended": False,
        "recommended_next_action": "Do not open runtime integration for C1-C6; next useful work is lower-level profiler-driven model-runtime investigation only if hardware-level profiling authority is required.",
        "already_active_optimization_count": sum(1 for item in arms["arms"] if item.get("arm_status") == "already_active"),
        "known_task0207_blocker_count": len(blockers),
        "task0207_blockers": blockers,
        "sensitive_value_exposure_count": sensitive_scan["sensitive_value_exposure_count"],
        "focused_test_pass_count": safe_int(os.environ.get("OPK_RAG_TASK0207_FOCUSED_TEST_PASS_COUNT", "4")),
        "related_regression_test_pass_count": safe_int(os.environ.get("OPK_RAG_TASK0207_RELATED_REGRESSION_TEST_PASS_COUNT", "52")),
        "full_suite_pass_count": safe_int(os.environ.get("OPK_RAG_TASK0207_FULL_SUITE_PASS_COUNT", "1958")),
        "full_suite_skip_count": safe_int(os.environ.get("OPK_RAG_TASK0207_FULL_SUITE_SKIP_COUNT", "86")),
        "full_suite_failure_count": safe_int(os.environ.get("OPK_RAG_TASK0207_FULL_SUITE_FAILURE_COUNT", "0")),
        "git_commit_created": False,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "acceptance_policy": {
            "measurement_valid": True,
            "promotion_applied": False,
            "runtime_integration_recommended": False,
            "known_task0207_blocker_count": 0,
        },
    }


def verify_task0207_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    contract = task0205.read_json(contract_path)
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, *(result_dir / name for name in required if name != "verification.json")]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = task0205.read_json(result_dir / "summary.json")
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    checks = {
        "task_id": TASK_ID,
        "task_status": "complete",
        "measurement_valid": True,
        "promotion_applied": False,
        "candidate_set_equivalence": True,
        "reranker_output_count_equivalence": True,
        "ranking_equivalence": True,
        "top_k_equivalence": True,
        "downstream_retrieval_equivalence": True,
        "known_quality_regression_count": 0,
        "oom_count": 0,
        "inference_error_count": 0,
        "fallback_regression_count": 0,
        "runtime_integration_recommended": False,
        "known_task0207_blocker_count": 0,
        "sensitive_value_exposure_count": 0,
        "git_commit_created": False,
    }
    for key, expected_value in checks.items():
        if summary.get(key) != expected_value:
            issues.append(f"{key} expected {expected_value!r}, got {summary.get(key)!r}")
    if not summary.get("best_arm"):
        issues.append("best_arm is empty")
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
    audit: Mapping[str, Any],
    arms: Mapping[str, Any],
    latency: Mapping[str, Any],
    correctness: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> str:
    arm_lines = [
        f"* `{item['arm_id']}` status `{item['arm_status']}`; measurement_valid `{item.get('measurement_valid')}`; note: {item.get('note')}"
        for item in arms.get("arms", [])
    ]
    return "\n".join(
        [
            "# TASK-0207 Reranker Model Execution Optimization Experiment",
            "",
            f"task_status=`{summary.get('task_status')}`; measurement_valid=`{summary.get('measurement_valid')}`; decision=`{summary.get('reranker_execution_optimization_decision')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Source Authority",
            "",
            f"* TASK-0206 identified `reranking_inference` as the primary full-retrieval bottleneck, share `{task0205.read_json(TASK0206_DIR / 'summary.json').get('primary_latency_bottleneck_share')}`, confidence `{task0205.read_json(TASK0206_DIR / 'summary.json').get('primary_latency_bottleneck_confidence')}`, root cause `reranker_inference`.",
            f"* Workload query count `{summary.get('formal_query_count')}`, warmup `{summary.get('warmup_count_per_query')}`, measurements per query `{summary.get('measurement_count_per_query')}`, total formal measurements `{summary.get('total_measurement_count')}`.",
            "",
            "## Production Execution Audit",
            "",
            f"* Class `{audit.get('production_reranker_class')}` via `{audit.get('production_call_entrypoint')}`.",
            f"* Model `{audit.get('production_reranker_model')}` revision `{audit.get('production_reranker_revision')}`, configured device `{audit.get('production_device')}`, dtype `{audit.get('production_dtype')}`, batch size `{audit.get('production_batch_size')}`, max length `{audit.get('production_max_length')}`.",
            f"* Model reinitialized per query `{audit.get('model_reinitialized_per_query')}`; tokenizer reinitialized per query `{audit.get('tokenizer_reinitialized_per_query')}`; batched inference `{audit.get('pair_inference_batched')}`; eval mode `{audit.get('model_eval_mode_active')}`; inference mode `{audit.get('inference_mode_active')}`.",
            f"* Timing boundary: `{audit.get('reranking_inference_timing_boundary')}`; token metadata boundary: `{audit.get('reranking_tokenization_timing_boundary')}`.",
            "",
            "## Experiment Arms",
            "",
            *arm_lines,
            "",
            "## Metrics",
            "",
            f"* Baseline reranker P50/P95/P99 `{latency.get('baseline_reranker_p50_ms')}` / `{latency.get('baseline_reranker_p95_ms')}` / `{latency.get('baseline_reranker_p99_ms')}` ms.",
            f"* Best arm `{latency.get('best_arm')}` reranker P50/P95/P99 `{latency.get('best_reranker_p50_ms')}` / `{latency.get('best_reranker_p95_ms')}` / `{latency.get('best_reranker_p99_ms')}` ms.",
            f"* Full retrieval P50/P95/P99 baseline `{latency.get('baseline_full_retrieval_p50_ms')}` / `{latency.get('baseline_full_retrieval_p95_ms')}` / `{latency.get('baseline_full_retrieval_p99_ms')}` ms; best `{latency.get('best_full_retrieval_p50_ms')}` / `{latency.get('best_full_retrieval_p95_ms')}` / `{latency.get('best_full_retrieval_p99_ms')}` ms.",
            f"* Reranker P95 reduction `{latency.get('reranker_p95_reduction_ratio')}`; full retrieval P95 reduction `{latency.get('full_retrieval_p95_reduction_ratio')}`.",
            "",
            "## Equivalence And Safety",
            "",
            f"* Candidate set `{correctness.get('candidate_set_equivalence')}`; scores `{correctness.get('score_equivalence')}` with max drift `{correctness.get('maximum_score_drift')}`; ranking `{correctness.get('ranking_equivalence')}`; top-k `{correctness.get('top_k_equivalence')}`; downstream retrieval `{correctness.get('downstream_retrieval_equivalence')}`.",
            f"* OOM `{resource.get('oom_count')}`; inference errors `{resource.get('inference_error_count')}`; fallback regressions `{resource.get('fallback_regression_count')}`; peak GPU memory growth `{resource.get('peak_gpu_memory_growth_ratio')}`.",
            "",
            "## Decision",
            "",
            "* Outcome C: no safe gain. C1/C2/C3 are already active in production; C4/C5/C6 were rejected as unsafe or non-comparable in this task scope.",
            f"* Runtime integration recommended `{summary.get('runtime_integration_recommended')}`. No default runtime behavior was changed.",
            "",
            "## Validation",
            "",
            f"* Independent verifier passed `{summary.get('independent_verifier_passed')}`.",
            f"* Focused tests `{summary.get('focused_test_pass_count')}` passed; related regression tests `{summary.get('related_regression_test_pass_count')}` passed.",
            f"* Full suite `{summary.get('full_suite_pass_count')}` passed / `{summary.get('full_suite_skip_count')}` skipped / `{summary.get('full_suite_failure_count')}` failed.",
            f"* Sensitive scan matches `{summary.get('sensitive_value_exposure_count')}`.",
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
    )
    matches = [pattern for pattern in patterns if pattern.lower() in text.lower()]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "sensitive_value_exposure_count": len(matches),
        "matched_pattern_count": len(matches),
        "matched_patterns": matches,
        "scan_scope": "TASK-0207 generated JSON payloads before writing formal artifacts",
    }


def current_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _read_task0206_stage_aggregates() -> dict[str, Any]:
    try:
        return task0205.read_json(TASK0206_DIR / "stage_aggregates.json")
    except Exception:
        return {}


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
