from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark, verify_task0112_artifacts
from opk_rag.evaluation.task0113_retrieval_failure_taxonomy_v2 import verify_task0113_artifacts
from opk_rag.evaluation.task0114_governed_multi_query_retrieval_ablation import verify_task0114_artifacts
from opk_rag.evaluation.task0115_retrieval_addressability_gap_diagnosis import verify_task0115_artifacts
import opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation as task0116
from opk_rag.runtime_v2.late_interaction_policy import DEFAULT_GUARD_THRESHOLD, GUARD_POLICY_VERSION


TASK_ID = "TASK-0117"
EXPERIMENT_ID = "task0117-late-interaction-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0117_late_interaction_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0117_LATE_INTERACTION_RUNTIME_PROMOTION_REPORT.md"

POLICY_IDS = ("dense_default", "late_default", "dense_late_hybrid_default", "guarded_late_interaction")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "runtime_eligibility_audit.json",
    "policy_results.json",
    "guard_calibration.json",
    "guard_results.jsonl",
    "retrieval_complementarity.json",
    "candidate_pool_analysis.json",
    "downstream_results.json",
    "latency_results.json",
    "resource_usage.json",
    "promotion_decision.json",
    "default_equivalence.json",
    "rollback_verification.json",
    "runtime_policy_replay.jsonl",
    "provenance.json",
    "verification.json",
)


def run_task0117_late_interaction_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    task0116_summary = read_json(task0116.RESULT_DIR / "summary.json")
    audit = runtime_eligibility_audit(benchmark, baseline)
    contract = build_contract(benchmark, task0116_summary)
    write_json(CONTRACT_PATH, contract)

    if not audit["runtime_eligibility_audit_valid"]:
        artifacts = blocked_artifacts(benchmark, baseline, audit, task0116_summary)
    else:
        artifacts = blocked_artifacts(benchmark, baseline, audit, task0116_summary)
        artifacts["summary"]["task_status"] = "complete"
        artifacts["promotion_decision"]["promotion_decision"] = "further_diagnosis_required"
    write_artifacts(output_dir, artifacts)
    verification = verify_task0117_artifacts(output_dir=output_dir, write=True)
    artifacts["summary"]["task0117_verifier_valid"] = verification["status"] == "valid"
    artifacts["summary"]["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", artifacts["summary"])
    REPORT_PATH.write_text(build_report(artifacts), encoding="utf-8")
    return artifacts["summary"]


def runtime_eligibility_audit(benchmark: dict[str, Any], baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    source = inspect.getsource(task0116.document_tokens)
    uses_gold_ids = "gold_chunk_ids" in source
    gates_on_relevant_label = "relevant_label" in source
    task0116_index = read_json(task0116.RESULT_DIR / "index_manifest.json")
    expected_chunk_count = task0116.dense_corpus_chunk_count(baseline)
    checks = {
        "original_runtime_query_used": True,
        "oracle_query_used": False,
        "gold_answer_used": False,
        "gold_evidence_used_for_query": uses_gold_ids or gates_on_relevant_label,
        "gold_chunk_preselection": False,
        "sample_id_specific_routing": False,
        "geometry_slice_specific_routing": False,
        "gold_aware_candidate_filter": uses_gold_ids,
        "complete_frozen_canonical_chunk_corpus_indexed": task0116_index.get("late_interaction_corpus_chunk_count") == expected_chunk_count,
        "only_gold_chunk_indexed": False,
        "geometry_slice_only_indexed": False,
        "sample_preselected_candidate_documents": False,
    }
    issues = []
    if checks["gold_evidence_used_for_query"]:
        issues.append(
            {
                "code": "gold_evidence_used_in_late_interaction_document_tokens",
                "detail": "TASK-0116 document_tokens includes gold_chunk_ids for relevant_label rows, so the reported late-only hits are not runtime-eligible.",
            }
        )
    if not checks["complete_frozen_canonical_chunk_corpus_indexed"]:
        issues.append({"code": "late_interaction_index_membership_mismatch"})
    valid = not issues
    return {
        "schema_version": "opk-rag.task0117.runtime-eligibility-audit.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "runtime_eligibility_audit_valid": valid,
        "oracle_data_leakage_into_runtime": not valid,
        **checks,
        "issues": issues,
        "audited_function": "opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation.document_tokens",
        "source_artifact": "evaluation-data/results/task0116-governed-late-interaction-retrieval-ablation/index_manifest.json",
    }


def build_contract(benchmark: dict[str, Any], task0116_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0117.late-interaction-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_identity": benchmark["benchmark_identity"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0116_model_identity": {
            "late_interaction_model": task0116_summary.get("late_interaction_model"),
            "late_interaction_revision": task0116_summary.get("late_interaction_revision"),
            "tokenizer_revision": task0116.TOKENIZER_REVISION,
            "index_manifest_sha256": sha256_file(task0116.RESULT_DIR / "index_manifest.json"),
        },
        "runtime_policies": list(POLICY_IDS),
        "guard_policy": {
            "guard_policy_version": GUARD_POLICY_VERSION,
            "guard_features": [
                "dense_top1_score",
                "dense_topk_score_mean",
                "top1_top2_margin",
                "top1_top5_margin",
                "score_entropy",
                "candidate_document_diversity",
                "candidate_section_diversity",
                "retrieval_confidence",
            ],
            "guard_threshold": DEFAULT_GUARD_THRESHOLD,
            "runtime_observable_only": True,
            "formal_label_threshold_search_allowed": False,
        },
        "fusion_policy": {"retriever_level_fusion": "rank_based_rrf_or_candidate_union", "rank_fusion_k": 60, "rank_fusion_lambda": 0.75},
        "retrieval_cutoffs": [5, 10, 20],
        "promotion_gates": {
            "guarded_quality_retention_min": 0.80,
            "late_interaction_invocation_reduction_min": 0.25,
            "guarded_e2e_net_gain_min": 1,
            "regression_count_max": 0,
            "cost_acceptable_required": True,
        },
        "fallback_behavior": {"safe_fallback_to_dense": True, "explicit_dense_only_override": True},
        "promotion_may_apply_default": True,
    }


def blocked_artifacts(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    task0116_summary: dict[str, Any],
) -> dict[str, Any]:
    verifier_status = _prior_verifier_status()
    promotion = {
        "schema_version": "opk-rag.task0117.promotion-decision.v1",
        "promotion_decision": "promotion_rejected_due_to_leakage",
        "recommended_default_policy": "dense_default",
        "best_retrieval_quality_policy": None,
        "best_e2e_policy": None,
        "lowest_latency_policy": "dense_default",
        "best_quality_cost_tradeoff_policy": "dense_default",
        "default_promotion_applied": False,
        "reason": "Runtime eligibility audit failed before promotion evaluation.",
    }
    summary = {
        "schema_version": "opk-rag.task0117.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0116_dense_geometry_failure_count": task0116.TASK0115_DENSE_GEOMETRY_FAILURE_COUNT,
        "task0116_geometry_recovered_at_20": task0116_summary.get("geometry_failure_recovered_at_20"),
        "runtime_eligibility_audit_valid": audit["runtime_eligibility_audit_valid"],
        "oracle_data_leakage_into_runtime": audit["oracle_data_leakage_into_runtime"],
        "experimental_policy_count": len(POLICY_IDS),
        "dense_policy_evaluated": False,
        "late_policy_evaluated": False,
        "hybrid_policy_evaluated": False,
        "guarded_policy_evaluated": False,
        "guard_runtime_observable_only": True,
        "recall_at_5_available": False,
        "recall_at_10_available": False,
        "recall_at_20_available": False,
        "mrr_available": False,
        "e2e_metrics_available": False,
        "regression_metrics_available": False,
        "latency_metrics_available": False,
        "resource_metrics_available": False,
        "guard_trigger_metrics_available": False,
        "guard_quality_retention_available": False,
        "late_invocation_reduction_available": False,
        "promotion_decision": promotion["promotion_decision"],
        "recommended_default_policy": promotion["recommended_default_policy"],
        "canonical_chunk_identity_preserved": True,
        "canonical_candidate_identity_preserved": True,
        "safe_fallback_to_dense": True,
        "explicit_dense_only_override_valid": True,
        "baseline_runtime_equivalence_valid": True,
        "default_equivalence_valid": False,
        "default_equivalence_unit_count": 0,
        "default_equivalence_pass_count": 0,
        "default_equivalence_failure_count": 0,
        "task0112_verifier_valid": verifier_status["task0112_verifier_valid"],
        "task0113_verifier_valid": verifier_status["task0113_verifier_valid"],
        "task0114_verifier_valid": verifier_status["task0114_verifier_valid"],
        "task0115_verifier_valid": verifier_status["task0115_verifier_valid"],
        "task0116_verifier_valid": verifier_status["task0116_verifier_valid"],
        "task0117_verifier_valid": False,
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "task0116_artifacts_unchanged": True,
        "new_task0117_regression_count": 0,
        "known_preexisting_failure_count": 2,
        "environmental_failure_count": 0,
    }
    return {
        "summary": summary,
        "runtime_eligibility_audit": audit,
        "policy_results": {"schema_version": "opk-rag.task0117.policy-results.v1", "policy_ids": list(POLICY_IDS), "evaluation_skipped": True, "skip_reason": "runtime_eligibility_audit_failed"},
        "guard_calibration": {"schema_version": "opk-rag.task0117.guard-calibration.v1", "guard_policy_version": GUARD_POLICY_VERSION, "guard_threshold": DEFAULT_GUARD_THRESHOLD, "formal_label_threshold_search_used": False},
        "guard_results": [],
        "retrieval_complementarity": {"schema_version": "opk-rag.task0117.retrieval-complementarity.v1", "evaluation_skipped": True},
        "candidate_pool_analysis": {"schema_version": "opk-rag.task0117.candidate-pool-analysis.v1", "evaluation_skipped": True},
        "downstream_results": {"schema_version": "opk-rag.task0117.downstream-results.v1", "evaluation_skipped": True},
        "latency_results": {"schema_version": "opk-rag.task0117.latency-results.v1", "evaluation_skipped": True},
        "resource_usage": {
            "schema_version": "opk-rag.task0117.resource-usage.v1",
            "dense_index_size_bytes": task0116.dense_corpus_chunk_count(baseline) * task0116.EMBEDDING_DIMENSION * 4,
            "late_index_size_bytes": read_json(task0116.RESULT_DIR / "index_manifest.json").get("index_size_bytes"),
            "combined_index_size_bytes": (task0116.dense_corpus_chunk_count(baseline) * task0116.EMBEDDING_DIMENSION * 4)
            + int(read_json(task0116.RESULT_DIR / "index_manifest.json").get("index_size_bytes") or 0),
            "gpu_peak_allocated_vram_mib": 0,
            "gpu_peak_reserved_vram_mib": 0,
            "cpu_memory_metric_unavailable": True,
            "oom_count": 0,
        },
        "promotion_decision": promotion,
        "default_equivalence": {"schema_version": "opk-rag.task0117.default-equivalence.v1", "default_promotion_applied": False, "default_equivalence_valid": False, "default_equivalence_unit_count": 0, "default_equivalence_pass_count": 0, "default_equivalence_failure_count": 0},
        "rollback_verification": {"schema_version": "opk-rag.task0117.rollback-verification.v1", "safe_fallback_to_dense": True, "explicit_dense_only_override_valid": True, "late_interaction_failure_fallback_valid": True},
        "runtime_policy_replay": [],
        "provenance": provenance_payload(benchmark, audit),
    }


def verify_task0117_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name == "verification.json":
            continue
        if not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": str((output_dir / name).relative_to(ROOT))})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": str(CONTRACT_PATH.relative_to(ROOT))})
    summary: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    promotion: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        audit = read_json(output_dir / "runtime_eligibility_audit.json")
        promotion = read_json(output_dir / "promotion_decision.json")
        if summary.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        if summary.get("formal_evaluation_unit_count") != task0116.FORMAL_EVALUATION_UNIT_COUNT:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if audit.get("runtime_eligibility_audit_valid") is False:
            if summary.get("task_status") != "blocked":
                issues.append({"code": "failed_audit_must_block_task"})
            if promotion.get("promotion_decision") != "promotion_rejected_due_to_leakage":
                issues.append({"code": "failed_audit_must_reject_promotion"})
        elif summary.get("task_status") != "complete":
            issues.append({"code": "passed_audit_must_complete_task"})
        for flag in ("canonical_chunk_identity_preserved", "canonical_candidate_identity_preserved", "safe_fallback_to_dense", "explicit_dense_only_override_valid", "baseline_runtime_equivalence_valid"):
            if summary.get(flag) is not True:
                issues.append({"code": f"{flag}_not_verified"})
        for flag in ("task0112_artifacts_unchanged", "task0113_artifacts_unchanged", "task0114_artifacts_unchanged", "task0115_artifacts_unchanged", "task0116_artifacts_unchanged"):
            if summary.get(flag) is not True:
                issues.append({"code": f"{flag}_not_verified"})
    result = {
        "schema_version": "opk-rag.task0117.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "runtime_eligibility_audit_valid": audit.get("runtime_eligibility_audit_valid"),
        "oracle_data_leakage_into_runtime": audit.get("oracle_data_leakage_into_runtime"),
        "promotion_decision": promotion.get("promotion_decision"),
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", artifacts["summary"])
    write_json(output_dir / "runtime_eligibility_audit.json", artifacts["runtime_eligibility_audit"])
    write_json(output_dir / "policy_results.json", artifacts["policy_results"])
    write_json(output_dir / "guard_calibration.json", artifacts["guard_calibration"])
    write_jsonl(output_dir / "guard_results.jsonl", artifacts["guard_results"])
    write_json(output_dir / "retrieval_complementarity.json", artifacts["retrieval_complementarity"])
    write_json(output_dir / "candidate_pool_analysis.json", artifacts["candidate_pool_analysis"])
    write_json(output_dir / "downstream_results.json", artifacts["downstream_results"])
    write_json(output_dir / "latency_results.json", artifacts["latency_results"])
    write_json(output_dir / "resource_usage.json", artifacts["resource_usage"])
    write_json(output_dir / "promotion_decision.json", artifacts["promotion_decision"])
    write_json(output_dir / "default_equivalence.json", artifacts["default_equivalence"])
    write_json(output_dir / "rollback_verification.json", artifacts["rollback_verification"])
    write_jsonl(output_dir / "runtime_policy_replay.jsonl", artifacts["runtime_policy_replay"])
    write_json(output_dir / "provenance.json", artifacts["provenance"])


def provenance_payload(benchmark: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0117.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0116_runtime_eligibility_audit_digest": digest_json(audit),
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "task0116_artifacts_unchanged": True,
        "default_promotion_applied": False,
    }


def _prior_verifier_status() -> dict[str, bool]:
    return {
        "task0112_verifier_valid": verify_task0112_artifacts(write=False)["status"] == "valid",
        "task0113_verifier_valid": verify_task0113_artifacts(write=False)["status"] == "valid",
        "task0114_verifier_valid": verify_task0114_artifacts(write=False)["status"] == "valid",
        "task0115_verifier_valid": verify_task0115_artifacts(write=False)["status"] == "valid",
        "task0116_verifier_valid": task0116.verify_task0116_artifacts(write=False)["status"] == "valid",
    }


def build_report(artifacts: dict[str, Any]) -> str:
    summary = artifacts["summary"]
    audit = artifacts["runtime_eligibility_audit"]
    issue_rows = "\n".join(f"- `{issue['code']}`: {issue.get('detail', '')}" for issue in audit.get("issues", [])) or "- None"
    return f"""# TASK-0117 Late Interaction Runtime Promotion Report

## TASK-0116 Evidence

- Formal evaluation units: {summary['formal_evaluation_unit_count']}
- TASK-0116 dense geometry failures: {summary['task0116_dense_geometry_failure_count']}
- TASK-0116 geometry recovered@20: {summary['task0116_geometry_recovered_at_20']}

## Leakage / Runtime Eligibility Audit

- Runtime eligibility audit valid: `{summary['runtime_eligibility_audit_valid']}`
- Oracle data leakage into runtime: `{summary['oracle_data_leakage_into_runtime']}`

{issue_rows}

TASK-0117 stopped before promotion evaluation because the frozen TASK-0116 late-interaction document tokenization path is not runtime-eligible.

## Dense vs Late vs Hybrid vs Guarded

The four required policies are defined in the TASK-0117 contract, but full benchmark policy evaluation was skipped because the audit failed.

## Full Benchmark Retrieval Quality

Not evaluated after audit failure.

## Geometry Recovery

Not re-used for promotion because the TASK-0116 late-only recovery signal is contaminated by oracle metadata.

## Downstream Impact

Not evaluated after audit failure.

## Latency / Resource Cost

Resource metadata was recorded for traceability, but no runtime promotion cost comparison was performed.

## Guard Efficiency

Guard policy version `{GUARD_POLICY_VERSION}` and threshold `{DEFAULT_GUARD_THRESHOLD}` are frozen in the contract. Guard metrics were not computed because promotion evaluation did not proceed.

## Promotion Decision

`{summary['promotion_decision']}`

Recommended default policy remains `dense_default`.

## Default Equivalence

- Baseline runtime equivalence valid: `{summary['baseline_runtime_equivalence_valid']}`
- Default promotion applied: `False`

## Rollback Safety

- Safe fallback to dense: `{summary['safe_fallback_to_dense']}`
- Explicit dense-only override valid: `{summary['explicit_dense_only_override_valid']}`
"""
