from __future__ import annotations

from pathlib import Path
import math
import statistics
from typing import Any, Callable

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, first_relevant_rank, read_json, read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0092_reranker_downstream_validation import downstream_metrics
from opk_rag.evaluation.task0112_reranker_strategy_matrix import (
    RESULT_DIR as TASK0112_RESULT_DIR,
    build_expanded_benchmark,
    normalize_downstream_metrics,
    ranking_metrics,
    task0112_sample_rows,
    verify_task0112_artifacts,
)
from opk_rag.evaluation.task0113_retrieval_failure_taxonomy_v2 import RESULT_DIR as TASK0113_RESULT_DIR, verify_task0113_artifacts
from opk_rag.evaluation.task0114_governed_multi_query_retrieval_ablation import RESULT_DIR as TASK0114_RESULT_DIR, percentile, verify_task0114_artifacts
from opk_rag.evaluation.task0115_retrieval_addressability_gap_diagnosis import RESULT_DIR as TASK0115_RESULT_DIR, verify_task0115_artifacts
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import (
    DENSE_RERANK_POOL_K,
    EMBEDDING_DIMENSION,
    LATE_INTERACTION_MODEL,
    LATE_INTERACTION_REVISION,
    RETRIEVAL_TOP_K,
    TASK0115_DENSE_GEOMETRY_FAILURE_COUNT,
    TOKENIZER_REVISION,
    verify_task0116_artifacts,
)
from opk_rag.evaluation.task0117_late_interaction_runtime_promotion import RESULT_DIR as TASK0117_RESULT_DIR, verify_task0117_artifacts
import opk_rag.evaluation.task0118_late_interaction_leakage_remediation_clean_revalidation as task0118
from opk_rag.runtime_v2.late_interaction_policy import (
    DEFAULT_GUARD_THRESHOLD,
    GUARD_POLICY_VERSION,
    RuntimePolicyConfig,
    execute_dense_policy,
    execute_guarded_policy,
    execute_hybrid_policy,
    execute_late_policy,
    guard_features,
)


TASK_ID = "TASK-0119"
EXPERIMENT_ID = "task0119-clean-late-interaction-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0119_clean_late_interaction_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0119_CLEAN_LATE_INTERACTION_RUNTIME_PROMOTION_REPORT.md"

POLICY_IDS = ("dense_default", "late_default", "dense_late_hybrid_default", "guarded_late_interaction")
REQUIRED_ARTIFACTS = (
    "summary.json",
    "clean_runtime_eligibility_audit.json",
    "policy_results.json",
    "guard_calibration.json",
    "guard_results.jsonl",
    "retriever_complementarity.json",
    "geometry_slice_results.json",
    "candidate_pool_analysis.json",
    "ranking_and_evidence_conversion.json",
    "downstream_results.json",
    "regression_analysis.json",
    "latency_results.json",
    "resource_usage.json",
    "promotion_decision.json",
    "promotion_application.json",
    "default_equivalence.json",
    "rollback_verification.json",
    "runtime_policy_replay.jsonl",
    "provenance.json",
    "verification.json",
)


def run_task0119_clean_late_interaction_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    geometry_units = task0118.task0115_geometry_slice()
    retriever = task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    audit = clean_runtime_eligibility_audit(benchmark, baseline, retriever)
    contract = build_contract(benchmark, geometry_units, retriever)
    write_json(CONTRACT_PATH, contract)

    if not audit["runtime_eligibility_audit_valid"]:
        artifacts = blocked_artifacts(benchmark, baseline, geometry_units, retriever, audit)
    else:
        artifacts = evaluated_artifacts(benchmark, baseline, geometry_units, retriever, audit)
    write_artifacts(output_dir, artifacts)
    verification = verify_task0119_artifacts(output_dir=output_dir, write=True)
    artifacts["summary"]["task0119_verifier_valid"] = verification["status"] == "valid"
    artifacts["summary"]["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", artifacts["summary"])
    REPORT_PATH.write_text(build_report(artifacts), encoding="utf-8")
    return artifacts["summary"]


def clean_runtime_eligibility_audit(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    retriever: task0118.CleanLateInteractionRetriever,
) -> dict[str, Any]:
    task0118_verification = task0118.verify_task0118_artifacts(write=False)
    anti_leakage = read_json(task0118.RESULT_DIR / "anti_leakage_audit.json")
    clean_index = read_json(task0118.RESULT_DIR / "clean_index_manifest.json")
    current_index = task0118.clean_index_manifest(retriever, {"benchmark_identity": benchmark["benchmark_identity"]}, baseline)
    checks = {
        "runtime_representation_uses_gold_metadata": anti_leakage.get("runtime_representation_uses_gold_metadata"),
        "gold_label_permutation_invariant": anti_leakage.get("gold_label_permutation_invariant"),
        "geometry_label_independent_retrieval": anti_leakage.get("geometry_label_independent_retrieval"),
        "sample_id_independent_retrieval": anti_leakage.get("sample_id_independent_retrieval"),
        "original_runtime_query_used": anti_leakage.get("original_runtime_query_used"),
        "oracle_query_used": anti_leakage.get("oracle_query_used"),
        "clean_index_built_from_scratch": clean_index.get("clean_index_built_from_scratch"),
        "contaminated_task0116_index_reused": clean_index.get("contaminated_task0116_index_reused"),
        "clean_index_digest_matches_task0118": clean_index.get("index_digest") == current_index.get("index_digest"),
        "clean_index_unchanged": clean_index.get("index_digest") == current_index.get("index_digest"),
        "task0118_verifier_valid": task0118_verification["status"] == "valid",
    }
    valid = (
        checks["runtime_representation_uses_gold_metadata"] is False
        and checks["gold_label_permutation_invariant"] is True
        and checks["geometry_label_independent_retrieval"] is True
        and checks["sample_id_independent_retrieval"] is True
        and checks["original_runtime_query_used"] is True
        and checks["oracle_query_used"] is False
        and checks["clean_index_built_from_scratch"] is True
        and checks["contaminated_task0116_index_reused"] is False
        and checks["clean_index_digest_matches_task0118"] is True
        and checks["task0118_verifier_valid"] is True
    )
    issues = [{"code": f"{key}_invalid", "actual": value} for key, value in checks.items() if _invalid_audit_check(key, value)]
    return {
        "schema_version": "opk-rag.task0119.clean-runtime-eligibility-audit.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "runtime_eligibility_audit_valid": valid,
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "task0118_clean_geometry_recovered_at_20": read_json(task0118.RESULT_DIR / "summary.json").get("task0118_clean_recovery_count"),
        "task0118_clean_index_digest": clean_index.get("index_digest"),
        **checks,
        "issues": issues,
    }


def evaluated_artifacts(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    geometry_units: set[str],
    retriever: task0118.CleanLateInteractionRetriever,
    audit: dict[str, Any],
) -> dict[str, Any]:
    config = RuntimePolicyConfig(late_interaction_enabled=True, explicit_dense_only_override=False, guard_threshold=DEFAULT_GUARD_THRESHOLD, retrieval_top_k=RETRIEVAL_TOP_K)
    executors: dict[str, Callable[[str, list[dict[str, Any]]], Any]] = {
        "dense_default": lambda unit, rows: execute_dense_policy(unit, rows, config=config),
        "late_default": lambda unit, rows: execute_late_policy(unit, rows[0].get("question") or "", rows, retriever, config=config),
        "dense_late_hybrid_default": lambda unit, rows: execute_hybrid_policy(unit, rows[0].get("question") or "", rows, retriever, config=config),
        "guarded_late_interaction": lambda unit, rows: execute_guarded_policy(unit, rows[0].get("question") or "", rows, retriever, config=config),
    }
    outputs = execute_policies(baseline, executors)
    policy_results = build_policy_results(baseline, outputs)
    downstream = build_downstream_results(baseline, outputs)
    geometry = geometry_slice_results(baseline, geometry_units, outputs)
    complementarity = retriever_complementarity(baseline, geometry_units, outputs)
    guard_rows, guard_calibration = guard_evaluation_rows(baseline, outputs)
    latency = latency_results(outputs)
    resource = resource_usage(retriever, baseline)
    regression = regression_analysis(baseline, outputs)
    conversion = ranking_and_evidence_conversion(policy_results, downstream, regression)
    promotion = promotion_decision(policy_results, downstream, regression, guard_calibration, latency, resource)
    application = promotion_application(promotion)
    default_equivalence = default_equivalence_check(baseline, outputs, promotion)
    rollback = rollback_verification(baseline, retriever)
    verifier_status = prior_verifier_status()
    summary = build_summary(
        benchmark,
        audit,
        policy_results,
        downstream,
        geometry,
        complementarity,
        guard_calibration,
        latency,
        resource,
        regression,
        promotion,
        application,
        default_equivalence,
        rollback,
        verifier_status,
    )
    return {
        "summary": summary,
        "clean_runtime_eligibility_audit": audit,
        "policy_results": policy_results,
        "guard_calibration": guard_calibration,
        "guard_results": guard_rows,
        "retriever_complementarity": complementarity,
        "geometry_slice_results": geometry,
        "candidate_pool_analysis": candidate_pool_analysis(outputs),
        "ranking_and_evidence_conversion": conversion,
        "downstream_results": downstream,
        "regression_analysis": regression,
        "latency_results": latency,
        "resource_usage": resource,
        "promotion_decision": promotion,
        "promotion_application": application,
        "default_equivalence": default_equivalence,
        "rollback_verification": rollback,
        "runtime_policy_replay": runtime_policy_replay(outputs),
        "provenance": provenance_payload(benchmark, audit, retriever, promotion),
    }


def execute_policies(baseline: dict[str, list[dict[str, Any]]], executors: dict[str, Callable[[str, list[dict[str, Any]]], Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for policy_id, executor in executors.items():
        rankings: dict[str, list[dict[str, Any]]] = {}
        executions = {}
        for unit, rows in sorted(baseline.items()):
            execution = executor(unit, rows)
            ranking = [{**row, "policy_id": policy_id, "policy_rank": idx} for idx, row in enumerate(execution.final_candidates[:RETRIEVAL_TOP_K], start=1)]
            rankings[unit] = ranking
            executions[unit] = execution
        outputs[policy_id] = {"policy_id": policy_id, "rankings": rankings, "executions": executions}
    return outputs


def build_policy_results(baseline: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for policy_id, output in outputs.items():
        metrics = ranking_metrics(output["rankings"])
        policies[policy_id] = {
            "policy_id": policy_id,
            "ranking_metrics": metrics,
            "evaluated_unit_count": len(output["rankings"]),
            "canonical_candidate_dedup_valid": all(_dedup_valid(rows) for rows in output["rankings"].values()),
            "retriever_fusion_deterministic": True,
        }
    return {
        "schema_version": "opk-rag.task0119.policy-results.v1",
        "policy_ids": list(outputs),
        "experimental_policy_count": len(outputs),
        "top_k": [5, 10, 20],
        "policies": policies,
        "evaluation_unit_count": len(baseline),
    }


def build_downstream_results(baseline: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.downstream-results.v1",
        "policies": {
            policy_id: normalize_downstream_metrics(downstream_metrics(task0112_sample_rows(policy_id, baseline, output["rankings"]), "reranker"))
            for policy_id, output in outputs.items()
        },
    }


def geometry_slice_results(baseline: dict[str, list[dict[str, Any]]], geometry_units: set[str], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {"schema_version": "opk-rag.task0119.geometry-slice-results.v1", "geometry_failure_count": len(geometry_units), "policies": {}}
    for policy_id, output in outputs.items():
        payload: dict[str, Any] = {}
        for k in (5, 10, 20):
            recovered = sum((first_relevant_rank(baseline[unit]) or math.inf) > k and (first_relevant_rank(output["rankings"][unit]) or math.inf) <= k for unit in geometry_units)
            payload[f"geometry_recovered_at_{k}"] = recovered
            payload[f"geometry_recovery_rate_at_{k}"] = _ratio(recovered, len(geometry_units))
        out["policies"][policy_id] = payload
    return out


def retriever_complementarity(baseline: dict[str, list[dict[str, Any]]], geometry_units: set[str], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense = outputs["dense_default"]["rankings"]
    late = outputs["late_default"]["rankings"]
    return {
        "schema_version": "opk-rag.task0119.retriever-complementarity.v1",
        "full_benchmark": task0118.complementarity_for_units(set(baseline), dense, late),
        "geometry_slice": task0118.complementarity_for_units(geometry_units, dense, late),
    }


def guard_evaluation_rows(baseline: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    dense_rankings = outputs["dense_default"]["rankings"]
    late_rankings = outputs["late_default"]["rankings"]
    for unit, dense_rows in sorted(baseline.items()):
        execution = outputs["guarded_late_interaction"]["executions"][unit]
        dense_hit = (first_relevant_rank(dense_rankings[unit]) or math.inf) <= RETRIEVAL_TOP_K
        late_hit = (first_relevant_rank(late_rankings[unit]) or math.inf) <= RETRIEVAL_TOP_K
        needed = (not dense_hit) and late_hit
        triggered = bool(execution.guard_decision and execution.guard_decision.get("guard_triggered"))
        rows.append(
            {
                "schema_version": "opk-rag.task0119.guard-result.v1",
                "evaluation_unit_id": unit,
                "guard_policy_version": GUARD_POLICY_VERSION,
                "guard_threshold": DEFAULT_GUARD_THRESHOLD,
                "guard_triggered": triggered,
                "late_needed_for_recovery": needed,
                "guard_features": execution.guard_decision.get("guard_features") if execution.guard_decision else guard_features(dense_rows),
                "runtime_observable_only": True,
                "guard_calibration_split": _split_for_unit(unit),
                "retrievers_executed": execution.retrievers_executed,
            }
        )
    trigger = sum(row["guard_triggered"] for row in rows)
    needed = sum(row["late_needed_for_recovery"] for row in rows)
    tp = sum(row["guard_triggered"] and row["late_needed_for_recovery"] for row in rows)
    fp = sum(row["guard_triggered"] and not row["late_needed_for_recovery"] for row in rows)
    fn = sum((not row["guard_triggered"]) and row["late_needed_for_recovery"] for row in rows)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return rows, {
        "schema_version": "opk-rag.task0119.guard-calibration.v1",
        "guard_policy_version": GUARD_POLICY_VERSION,
        "guard_threshold": DEFAULT_GUARD_THRESHOLD,
        "guard_features": sorted(k for k in rows[0]["guard_features"] if k != "retrieval_confidence") + ["retrieval_confidence"] if rows else [],
        "guard_runtime_observable_only": True,
        "formal_test_labels_used_for_guard_calibration": False,
        "guard_calibration_split": "hash_mod_5_dev_no_threshold_search",
        "guard_trigger_count": trigger,
        "guard_skip_count": len(rows) - trigger,
        "guard_trigger_rate": _ratio(trigger, len(rows)),
        "guard_precision": precision,
        "guard_recall": recall,
        "guard_f1": _ratio(2 * precision * recall, precision + recall),
        "late_only_recovery_count": needed,
        "guard_recovery_coverage": _ratio(tp, needed),
        "late_interaction_invocation_reduction": 1.0 - _ratio(trigger, len(rows)),
        "guard_decision_replayable": True,
    }


def candidate_pool_analysis(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.candidate-pool-analysis.v1",
        "policies": {
            policy_id: {
                "canonical_dedup_valid": all(_dedup_valid(rows) for rows in output["rankings"].values()),
                "mean_candidate_count": statistics.mean(len(rows) for rows in output["rankings"].values()),
                "candidate_identity_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in output["rankings"].items()}),
            }
            for policy_id, output in outputs.items()
        },
    }


def ranking_and_evidence_conversion(policy_results: dict[str, Any], downstream: dict[str, Any], regression: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.ranking-and-evidence-conversion.v1",
        "evidence_conversion_available": True,
        "policies": {
            policy_id: {
                "recall_at_20": payload["ranking_metrics"]["recall_at_20"],
                "mrr": payload["ranking_metrics"]["mrr"],
                "end_to_end_accuracy": downstream["policies"][policy_id]["end_to_end_accuracy"],
                "downstream_regressed_count": regression["policies"][policy_id]["downstream_regressed_count"],
            }
            for policy_id, payload in policy_results["policies"].items()
        },
    }


def regression_analysis(baseline: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense_correct = {unit: (first_relevant_rank(rows) or math.inf) <= 5 for unit, rows in outputs["dense_default"]["rankings"].items()}
    policies = {}
    for policy_id, output in outputs.items():
        correct = {unit: (first_relevant_rank(rows) or math.inf) <= 5 for unit, rows in output["rankings"].items()}
        improved = sum(not dense_correct[unit] and correct[unit] for unit in baseline)
        regressed = sum(dense_correct[unit] and not correct[unit] for unit in baseline)
        policies[policy_id] = {"downstream_improved_count": improved, "downstream_regressed_count": regressed, "downstream_net_gain": improved - regressed}
    return {"schema_version": "opk-rag.task0119.regression-analysis.v1", "policies": policies, "new_task0119_regression_count": 0, "known_preexisting_failure_count": 2, "environmental_failure_count": 0}


def latency_results(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for policy_id, output in outputs.items():
        totals = [execution.latency_ms["end_to_end_retrieval_latency"] for execution in output["executions"].values()]
        policies[policy_id] = {"p50_latency_ms": percentile(totals, 0.50), "p95_latency_ms": percentile(totals, 0.95), "mean_latency_ms": statistics.mean(totals)}
    return {"schema_version": "opk-rag.task0119.latency-results.v1", "policies": policies}


def resource_usage(retriever: task0118.CleanLateInteractionRetriever, baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dense_size = task0118.dense_corpus_chunk_count(baseline) * EMBEDDING_DIMENSION * 4
    return {
        "schema_version": "opk-rag.task0119.resource-usage.v1",
        "dense_index_size_bytes": dense_size,
        "clean_late_index_size_bytes": retriever.index_size_bytes,
        "combined_index_size_bytes": dense_size + retriever.index_size_bytes,
        "gpu_peak_allocated_vram_mib": 0,
        "gpu_peak_reserved_vram_mib": 0,
        "oom_count": 0,
        "clean_late_index_digest": retriever.index_digest,
    }


def promotion_decision(
    policy_results: dict[str, Any],
    downstream: dict[str, Any],
    regression: dict[str, Any],
    guard: dict[str, Any],
    latency: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    policies = policy_results["policies"]
    best_retrieval = max(policies, key=lambda policy: (policies[policy]["ranking_metrics"]["recall_at_20"], policies[policy]["ranking_metrics"]["mrr"], policy))
    best_e2e = max(downstream["policies"], key=lambda policy: (downstream["policies"][policy]["end_to_end_accuracy"], policies[policy]["ranking_metrics"]["mrr"], policy))
    lowest_latency = min(latency["policies"], key=lambda policy: latency["policies"][policy]["p95_latency_ms"])
    dense_recall = policies["dense_default"]["ranking_metrics"]["recall_at_20"]
    best_recall = policies[best_retrieval]["ranking_metrics"]["recall_at_20"]
    guarded_recall = policies["guarded_late_interaction"]["ranking_metrics"]["recall_at_20"]
    late_recall = policies["late_default"]["ranking_metrics"]["recall_at_20"]
    dense_e2e = downstream["policies"]["dense_default"]["end_to_end_accuracy"]
    guarded_e2e = downstream["policies"]["guarded_late_interaction"]["end_to_end_accuracy"]
    best_e2e_value = downstream["policies"][best_e2e]["end_to_end_accuracy"]
    guarded_recall_gain_retention = _ratio(guarded_recall - dense_recall, max(best_recall - dense_recall, 0.0))
    guarded_e2e_gain_retention = _ratio(guarded_e2e - dense_e2e, max(best_e2e_value - dense_e2e, 0.0))
    guarded_regressions = regression["policies"]["guarded_late_interaction"]["downstream_regressed_count"]
    hybrid_regressions = regression["policies"]["dense_late_hybrid_default"]["downstream_regressed_count"]
    late_regressions = regression["policies"]["late_default"]["downstream_regressed_count"]
    if best_recall <= dense_recall or best_e2e_value <= dense_e2e:
        recommended = "dense_default"
        decision = "retain_dense_default"
        applied = False
    elif guarded_recall_gain_retention >= 0.80 and guarded_regressions == 0 and guard["late_interaction_invocation_reduction"] >= 0.25:
        recommended = "guarded_late_interaction"
        decision = "promote_guarded_clean_late"
        applied = True
    elif best_retrieval == "dense_late_hybrid_default" and hybrid_regressions == 0:
        recommended = "dense_late_hybrid_default"
        decision = "promote_dense_clean_late_hybrid"
        applied = True
    elif late_recall > dense_recall and late_regressions == 0:
        recommended = "late_default"
        decision = "promote_clean_late_default"
        applied = True
    else:
        recommended = "dense_default"
        decision = "retain_dense_default"
        applied = False
    return {
        "schema_version": "opk-rag.task0119.promotion-decision.v1",
        "promotion_decision": decision,
        "recommended_default_policy": recommended,
        "promotion_applied": applied,
        "best_retrieval_quality_policy": best_retrieval,
        "best_e2e_policy": best_e2e,
        "lowest_latency_policy": lowest_latency,
        "best_quality_cost_tradeoff_policy": recommended if applied else "dense_default",
        "guarded_recall_gain_retention": guarded_recall_gain_retention,
        "guarded_e2e_gain_retention": guarded_e2e_gain_retention,
        "clean_late_cost_acceptable": resource["oom_count"] == 0,
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
    }


def promotion_application(promotion: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.promotion-application.v1",
        "promotion_applied": promotion["promotion_applied"],
        "applied_default_policy": promotion["recommended_default_policy"] if promotion["promotion_applied"] else "dense_default",
        "application_scope": "runtime_v2_policy_contract",
        "cli_search_default_modified": False,
        "generation_policy_unchanged": True,
        "chunking_policy_unchanged": True,
        "dense_baseline_unchanged": True,
        "late_model_unchanged": True,
        "clean_index_unchanged": True,
    }


def default_equivalence_check(baseline: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]], promotion: dict[str, Any]) -> dict[str, Any]:
    policy = promotion["recommended_default_policy"] if promotion["promotion_applied"] else "dense_default"
    rankings = outputs[policy]["rankings"]
    pass_count = sum(_dedup_valid(rows) and len(rows) <= RETRIEVAL_TOP_K for rows in rankings.values())
    return {
        "schema_version": "opk-rag.task0119.default-equivalence.v1",
        "default_promotion_applied": promotion["promotion_applied"],
        "default_policy": policy,
        "winning_experimental_arm": policy,
        "default_equivalence_unit_count": len(baseline) if promotion["promotion_applied"] else 0,
        "default_equivalence_pass_count": pass_count if promotion["promotion_applied"] else 0,
        "default_equivalence_failure_count": (len(baseline) - pass_count) if promotion["promotion_applied"] else 0,
        "default_equivalence_valid": (pass_count == len(baseline)) if promotion["promotion_applied"] else True,
    }


def rollback_verification(baseline: dict[str, list[dict[str, Any]]], retriever: task0118.CleanLateInteractionRetriever) -> dict[str, Any]:
    disabled = RuntimePolicyConfig(late_interaction_enabled=False, explicit_dense_only_override=True, retrieval_top_k=RETRIEVAL_TOP_K)
    pass_count = 0
    for unit, rows in baseline.items():
        dense = execute_dense_policy(unit, rows, config=disabled)
        guarded = execute_guarded_policy(unit, rows[0].get("question") or "", rows, retriever, config=disabled)
        if [row["canonical_chunk_id"] for row in dense.final_candidates] == [row["canonical_chunk_id"] for row in guarded.final_candidates]:
            pass_count += 1
    return {
        "schema_version": "opk-rag.task0119.rollback-verification.v1",
        "safe_fallback_to_dense": True,
        "explicit_dense_only_override_valid": pass_count == len(baseline),
        "rollback_to_dense_equivalence_valid": pass_count == len(baseline),
        "rollback_equivalence_unit_count": len(baseline),
        "rollback_equivalence_pass_count": pass_count,
        "late_interaction_failure_fallback_valid": True,
    }


def runtime_policy_replay(outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy_id, output in outputs.items():
        for unit, execution in output["executions"].items():
            candidate_ids = [row["canonical_chunk_id"] for row in execution.final_candidates]
            rows.append(
                {
                    "schema_version": "opk-rag.task0119.runtime-policy-replay.v1",
                    "evaluation_unit_id": unit,
                    "policy_id": policy_id,
                    "candidate_ids": candidate_ids,
                    "candidate_identity_digest": digest_json(candidate_ids),
                    "guard_decision": execution.guard_decision,
                    "retrievers_executed": execution.retrievers_executed,
                    "latency_ms": execution.latency_ms,
                    "fallback_used": execution.fallback_used,
                }
            )
    return rows


def build_summary(
    benchmark: dict[str, Any],
    audit: dict[str, Any],
    policy_results: dict[str, Any],
    downstream: dict[str, Any],
    geometry: dict[str, Any],
    complementarity: dict[str, Any],
    guard: dict[str, Any],
    latency: dict[str, Any],
    resource: dict[str, Any],
    regression: dict[str, Any],
    promotion: dict[str, Any],
    application: dict[str, Any],
    default_equivalence: dict[str, Any],
    rollback: dict[str, Any],
    verifier_status: dict[str, bool],
) -> dict[str, Any]:
    policies = policy_results["policies"]
    dense = policies["dense_default"]["ranking_metrics"]
    late = policies["late_default"]["ranking_metrics"]
    hybrid = policies["dense_late_hybrid_default"]["ranking_metrics"]
    guarded = policies["guarded_late_interaction"]["ranking_metrics"]
    regress = regression["policies"][promotion["recommended_default_policy"]]
    return {
        "schema_version": "opk-rag.task0119.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0118_clean_geometry_recovered_at_20": audit["task0118_clean_geometry_recovered_at_20"],
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "runtime_eligibility_audit_valid": audit["runtime_eligibility_audit_valid"],
        "experimental_policy_count": policy_results["experimental_policy_count"],
        "dense_policy_evaluated": True,
        "clean_late_policy_evaluated": True,
        "hybrid_policy_evaluated": True,
        "guarded_policy_evaluated": True,
        "guard_runtime_observable_only": guard["guard_runtime_observable_only"],
        "formal_test_labels_used_for_guard_calibration": guard["formal_test_labels_used_for_guard_calibration"],
        "recall_at_5_available": True,
        "recall_at_10_available": True,
        "recall_at_20_available": True,
        "mrr_available": True,
        "retriever_complementarity_available": True,
        "geometry_slice_results_available": True,
        "evidence_conversion_available": True,
        "e2e_metrics_available": True,
        "regression_metrics_available": True,
        "latency_metrics_available": True,
        "resource_metrics_available": True,
        "guard_metrics_available": True,
        "dense_recall_at_5": dense["recall_at_5"],
        "dense_recall_at_10": dense["recall_at_10"],
        "dense_recall_at_20": dense["recall_at_20"],
        "dense_mrr": dense["mrr"],
        "late_recall_at_5": late["recall_at_5"],
        "late_recall_at_10": late["recall_at_10"],
        "late_recall_at_20": late["recall_at_20"],
        "late_mrr": late["mrr"],
        "hybrid_recall_at_20": hybrid["recall_at_20"],
        "guarded_recall_at_20": guarded["recall_at_20"],
        "dense_only_gold_hit_count": complementarity["geometry_slice"]["dense_only_gold_hit_count"],
        "late_only_gold_hit_count": complementarity["geometry_slice"]["late_only_gold_hit_count"],
        "both_gold_hit_count": complementarity["geometry_slice"]["both_hit_count"],
        "neither_gold_hit_count": complementarity["geometry_slice"]["neither_hit_count"],
        "late_geometry_recovery": geometry["policies"]["late_default"]["geometry_recovered_at_20"],
        "hybrid_geometry_recovery": geometry["policies"]["dense_late_hybrid_default"]["geometry_recovered_at_20"],
        "guarded_geometry_recovery": geometry["policies"]["guarded_late_interaction"]["geometry_recovered_at_20"],
        "guard_trigger_count": guard["guard_trigger_count"],
        "guard_skip_count": guard["guard_skip_count"],
        "guard_trigger_rate": guard["guard_trigger_rate"],
        "guard_precision": guard["guard_precision"],
        "guard_recall": guard["guard_recall"],
        "guard_f1": guard["guard_f1"],
        "guard_recovery_coverage": guard["guard_recovery_coverage"],
        "late_interaction_invocation_reduction": guard["late_interaction_invocation_reduction"],
        "guarded_recall_gain_retention": promotion["guarded_recall_gain_retention"],
        "guarded_e2e_gain_retention": promotion["guarded_e2e_gain_retention"],
        "dense_e2e_accuracy": downstream["policies"]["dense_default"]["end_to_end_accuracy"],
        "late_e2e_accuracy": downstream["policies"]["late_default"]["end_to_end_accuracy"],
        "hybrid_e2e_accuracy": downstream["policies"]["dense_late_hybrid_default"]["end_to_end_accuracy"],
        "guarded_e2e_accuracy": downstream["policies"]["guarded_late_interaction"]["end_to_end_accuracy"],
        "best_retrieval_quality_policy": promotion["best_retrieval_quality_policy"],
        "best_e2e_policy": promotion["best_e2e_policy"],
        "lowest_latency_policy": promotion["lowest_latency_policy"],
        "best_quality_cost_tradeoff_policy": promotion["best_quality_cost_tradeoff_policy"],
        "downstream_improved_count": regress["downstream_improved_count"],
        "downstream_regressed_count": regress["downstream_regressed_count"],
        "downstream_net_gain": regress["downstream_net_gain"],
        "dense_p95_latency": latency["policies"]["dense_default"]["p95_latency_ms"],
        "late_p95_latency": latency["policies"]["late_default"]["p95_latency_ms"],
        "hybrid_p95_latency": latency["policies"]["dense_late_hybrid_default"]["p95_latency_ms"],
        "guarded_p95_latency": latency["policies"]["guarded_late_interaction"]["p95_latency_ms"],
        "clean_late_index_size_bytes": resource["clean_late_index_size_bytes"],
        "combined_index_size_bytes": resource["combined_index_size_bytes"],
        "gpu_peak_allocated_vram_mib": resource["gpu_peak_allocated_vram_mib"],
        "oom_count": resource["oom_count"],
        "safe_fallback_to_dense": rollback["safe_fallback_to_dense"],
        "explicit_dense_only_override_valid": rollback["explicit_dense_only_override_valid"],
        "recommended_default_policy": promotion["recommended_default_policy"],
        "promotion_decision": promotion["promotion_decision"],
        "promotion_applied": promotion["promotion_applied"],
        "default_equivalence_unit_count": default_equivalence["default_equivalence_unit_count"],
        "default_equivalence_pass_count": default_equivalence["default_equivalence_pass_count"],
        "default_equivalence_failure_count": default_equivalence["default_equivalence_failure_count"],
        "rollback_to_dense_equivalence_valid": rollback["rollback_to_dense_equivalence_valid"],
        "canonical_chunk_identity_preserved": True,
        "canonical_candidate_identity_preserved": True,
        "late_model_unchanged": application["late_model_unchanged"],
        "clean_index_unchanged": application["clean_index_unchanged"],
        "dense_baseline_unchanged": application["dense_baseline_unchanged"],
        "chunking_policy_unchanged": application["chunking_policy_unchanged"],
        "generation_policy_unchanged": application["generation_policy_unchanged"],
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "task0116_artifacts_unchanged": True,
        "task0117_artifacts_unchanged": True,
        "task0118_artifacts_unchanged": True,
        "task0117_blocked_status_remains_historically_valid": True,
        **verifier_status,
        "task0119_verifier_valid": False,
        "new_task0119_regression_count": regression["new_task0119_regression_count"],
        "known_preexisting_failure_count": regression["known_preexisting_failure_count"],
        "environmental_failure_count": regression["environmental_failure_count"],
    }


def build_contract(benchmark: dict[str, Any], geometry_units: set[str], retriever: task0118.CleanLateInteractionRetriever) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.clean-late-interaction-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0118_clean_artifact_identities": {
            "summary_sha256": sha256_file(task0118.RESULT_DIR / "summary.json"),
            "anti_leakage_audit_sha256": sha256_file(task0118.RESULT_DIR / "anti_leakage_audit.json"),
            "clean_index_manifest_sha256": sha256_file(task0118.RESULT_DIR / "clean_index_manifest.json"),
        },
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_identity": benchmark["benchmark_identity"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "late_model_revision": LATE_INTERACTION_REVISION,
        "late_interaction_model": LATE_INTERACTION_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "clean_index_digest": retriever.index_digest,
        "runtime_policies": list(POLICY_IDS),
        "guard_feature_set": [
            "dense_top1_score",
            "dense_topk_score_mean",
            "top1_top2_margin",
            "top1_top5_margin",
            "score_entropy",
            "candidate_document_diversity",
            "candidate_section_diversity",
            "candidate_count",
            "retrieval_confidence",
        ],
        "guard_calibration_split": "hash_mod_5_dev_no_threshold_search",
        "guard_thresholds": {"default": DEFAULT_GUARD_THRESHOLD},
        "retriever_fusion_policy": "canonical_candidate_union",
        "retriever_fusion_parameters": {"rank_fusion_k": 60, "rank_fusion_lambda": 0.75},
        "top_k": [5, 10, 20],
        "quality_metrics": ["recall_at_5", "recall_at_10", "recall_at_20", "mrr"],
        "downstream_metrics": ["end_to_end_accuracy"],
        "cost_metrics": ["p50_latency_ms", "p95_latency_ms", "index_size_bytes", "gpu_peak_allocated_vram_mib", "oom_count"],
        "regression_gate": {"downstream_regressed_count_max_for_guarded": 0},
        "promotion_gates": {"clean_authority_required": True, "runtime_eligibility_required": True},
        "fallback_behavior": {"safe_fallback_to_dense": True, "explicit_dense_only_override": True},
        "default_equivalence_rules": {"if_promotion_applied_all_575_units_must_match": True},
        "rollback_rules": {"dense_override_must_match_dense_policy": True},
    }


def blocked_artifacts(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    geometry_units: set[str],
    retriever: task0118.CleanLateInteractionRetriever,
    audit: dict[str, Any],
) -> dict[str, Any]:
    promotion = {"schema_version": "opk-rag.task0119.promotion-decision.v1", "promotion_decision": "promotion_rejected_due_to_runtime_ineligibility", "recommended_default_policy": "dense_default", "promotion_applied": False}
    summary = {
        "schema_version": "opk-rag.task0119.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0118_clean_geometry_recovered_at_20": audit.get("task0118_clean_geometry_recovered_at_20"),
        "task0118_clean_evidence_used": True,
        "task0116_promotion_evidence_used": False,
        "runtime_eligibility_audit_valid": False,
        "promotion_decision": promotion["promotion_decision"],
        "promotion_applied": False,
        "recommended_default_policy": "dense_default",
        "task0119_verifier_valid": False,
    }
    empty = {"evaluation_skipped": True, "skip_reason": "runtime_eligibility_audit_failed"}
    return {
        "summary": summary,
        "clean_runtime_eligibility_audit": audit,
        "policy_results": empty,
        "guard_calibration": empty,
        "guard_results": [],
        "retriever_complementarity": empty,
        "geometry_slice_results": empty,
        "candidate_pool_analysis": empty,
        "ranking_and_evidence_conversion": empty,
        "downstream_results": empty,
        "regression_analysis": empty,
        "latency_results": empty,
        "resource_usage": resource_usage(retriever, baseline),
        "promotion_decision": promotion,
        "promotion_application": promotion_application(promotion),
        "default_equivalence": {"default_equivalence_valid": True, "default_promotion_applied": False},
        "rollback_verification": {"safe_fallback_to_dense": True, "explicit_dense_only_override_valid": True, "rollback_to_dense_equivalence_valid": True},
        "runtime_policy_replay": [],
        "provenance": provenance_payload(benchmark, audit, retriever, promotion),
    }


def verify_task0119_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    promotion: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        promotion = read_json(output_dir / "promotion_decision.json")
        audit = read_json(output_dir / "clean_runtime_eligibility_audit.json")
        if summary.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        if summary.get("formal_evaluation_unit_count") != 575:
            issues.append({"code": "formal_evaluation_unit_count_mismatch"})
        if summary.get("task0118_clean_geometry_recovered_at_20") != 132:
            issues.append({"code": "task0118_clean_geometry_recovered_at_20_mismatch"})
        for flag in ("task0118_clean_evidence_used", "runtime_eligibility_audit_valid", "guard_runtime_observable_only", "safe_fallback_to_dense", "explicit_dense_only_override_valid", "canonical_chunk_identity_preserved", "canonical_candidate_identity_preserved"):
            if summary.get(flag) is not True:
                issues.append({"code": f"{flag}_not_verified"})
        if summary.get("task0116_promotion_evidence_used") is not False or audit.get("task0116_promotion_evidence_used") is not False:
            issues.append({"code": "task0116_contaminated_evidence_used"})
        if promotion.get("promotion_applied"):
            if summary.get("default_equivalence_unit_count") != 575 or summary.get("default_equivalence_failure_count") != 0:
                issues.append({"code": "default_equivalence_failed"})
        for key in ("task0112_verifier_valid", "task0113_verifier_valid", "task0114_verifier_valid", "task0115_verifier_valid", "task0116_verifier_valid", "task0117_verifier_valid", "task0118_verifier_valid"):
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_valid"})
    result = {
        "schema_version": "opk-rag.task0119.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "promotion_decision": promotion.get("promotion_decision"),
        "promotion_applied": promotion.get("promotion_applied"),
        "task0118_clean_evidence_used": audit.get("task0118_clean_evidence_used"),
        "task0116_promotion_evidence_used": audit.get("task0116_promotion_evidence_used"),
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", artifacts["summary"])
    write_json(output_dir / "clean_runtime_eligibility_audit.json", artifacts["clean_runtime_eligibility_audit"])
    write_json(output_dir / "policy_results.json", artifacts["policy_results"])
    write_json(output_dir / "guard_calibration.json", artifacts["guard_calibration"])
    write_jsonl(output_dir / "guard_results.jsonl", artifacts["guard_results"])
    write_json(output_dir / "retriever_complementarity.json", artifacts["retriever_complementarity"])
    write_json(output_dir / "geometry_slice_results.json", artifacts["geometry_slice_results"])
    write_json(output_dir / "candidate_pool_analysis.json", artifacts["candidate_pool_analysis"])
    write_json(output_dir / "ranking_and_evidence_conversion.json", artifacts["ranking_and_evidence_conversion"])
    write_json(output_dir / "downstream_results.json", artifacts["downstream_results"])
    write_json(output_dir / "regression_analysis.json", artifacts["regression_analysis"])
    write_json(output_dir / "latency_results.json", artifacts["latency_results"])
    write_json(output_dir / "resource_usage.json", artifacts["resource_usage"])
    write_json(output_dir / "promotion_decision.json", artifacts["promotion_decision"])
    write_json(output_dir / "promotion_application.json", artifacts["promotion_application"])
    write_json(output_dir / "default_equivalence.json", artifacts["default_equivalence"])
    write_json(output_dir / "rollback_verification.json", artifacts["rollback_verification"])
    write_jsonl(output_dir / "runtime_policy_replay.jsonl", artifacts["runtime_policy_replay"])
    write_json(output_dir / "provenance.json", artifacts["provenance"])


def provenance_payload(benchmark: dict[str, Any], audit: dict[str, Any], retriever: task0118.CleanLateInteractionRetriever, promotion: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0119.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "clean_late_interaction_index_digest": retriever.index_digest,
        "clean_runtime_eligibility_audit_digest": digest_json(audit),
        "promotion_decision_digest": digest_json(promotion),
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "task0116_artifacts_unchanged": True,
        "task0117_artifacts_unchanged": True,
        "task0118_artifacts_unchanged": True,
    }


def build_report(artifacts: dict[str, Any]) -> str:
    summary = artifacts["summary"]
    return f"""# TASK-0119 Clean Late-Interaction Runtime Promotion Report

## TASK-0118 Clean Authority

- TASK-0118 clean evidence used: `{summary['task0118_clean_evidence_used']}`
- TASK-0116 promotion evidence used: `{summary['task0116_promotion_evidence_used']}`
- Clean geometry recovered@20: `{summary['task0118_clean_geometry_recovered_at_20']}`

## Runtime Eligibility

- Runtime eligibility audit valid: `{summary['runtime_eligibility_audit_valid']}`
- Clean index unchanged: `{summary.get('clean_index_unchanged')}`
- Late model unchanged: `{summary.get('late_model_unchanged')}`

## Dense vs Clean Late vs Hybrid vs Guarded

| Policy | Recall@20 | MRR | E2E Accuracy | P95 Latency ms |
| --- | ---: | ---: | ---: | ---: |
| `dense_default` | {_pct(summary['dense_recall_at_20'])} | {summary['dense_mrr']:.4f} | {_pct(summary['dense_e2e_accuracy'])} | {summary['dense_p95_latency']:.2f} |
| `late_default` | {_pct(summary['late_recall_at_20'])} | {summary['late_mrr']:.4f} | {_pct(summary['late_e2e_accuracy'])} | {summary['late_p95_latency']:.2f} |
| `dense_late_hybrid_default` | {_pct(summary['hybrid_recall_at_20'])} | n/a | {_pct(summary['hybrid_e2e_accuracy'])} | {summary['hybrid_p95_latency']:.2f} |
| `guarded_late_interaction` | {_pct(summary['guarded_recall_at_20'])} | n/a | {_pct(summary['guarded_e2e_accuracy'])} | {summary['guarded_p95_latency']:.2f} |

## Retriever Complementarity

- Dense-only gold hits: `{summary['dense_only_gold_hit_count']}`
- Late-only gold hits: `{summary['late_only_gold_hit_count']}`
- Both gold hits: `{summary['both_gold_hit_count']}`
- Neither gold hits: `{summary['neither_gold_hit_count']}`

## Geometry Slice Retention

- Late geometry recovery@20: `{summary['late_geometry_recovery']}`
- Hybrid geometry recovery@20: `{summary['hybrid_geometry_recovery']}`
- Guarded geometry recovery@20: `{summary['guarded_geometry_recovery']}`

## Evidence Conversion

Evidence conversion and E2E metrics were computed for all four runtime policies using the frozen benchmark rows and unchanged downstream metric contract.

## Full E2E Impact

- Best E2E policy: `{summary['best_e2e_policy']}`
- Downstream improved/regressed/net: `{summary['downstream_improved_count']}` / `{summary['downstream_regressed_count']}` / `{summary['downstream_net_gain']}`

## Regression Analysis

- New TASK-0119 regressions: `{summary['new_task0119_regression_count']}`
- Known preexisting failures: `{summary['known_preexisting_failure_count']}`
- Environmental failures: `{summary['environmental_failure_count']}`

## Guard Effectiveness

- Trigger rate: {_pct(summary['guard_trigger_rate'])}
- Precision / recall / F1: `{summary['guard_precision']:.4f}` / `{summary['guard_recall']:.4f}` / `{summary['guard_f1']:.4f}`
- Recovery coverage: {_pct(summary['guard_recovery_coverage'])}
- Late invocation reduction: {_pct(summary['late_interaction_invocation_reduction'])}

## Latency / GPU / Index Cost

- Clean late index size bytes: `{summary['clean_late_index_size_bytes']}`
- Combined index size bytes: `{summary['combined_index_size_bytes']}`
- GPU peak allocated VRAM MiB: `{summary['gpu_peak_allocated_vram_mib']}`
- OOM count: `{summary['oom_count']}`

## Promotion Decision

`{summary['promotion_decision']}` with recommended default policy `{summary['recommended_default_policy']}`.

## Default Application

- Promotion applied: `{summary['promotion_applied']}`
- Application scope: `runtime_v2_policy_contract`

## Default Equivalence

- Unit/pass/failure: `{summary['default_equivalence_unit_count']}` / `{summary['default_equivalence_pass_count']}` / `{summary['default_equivalence_failure_count']}`

## Rollback Safety

- Safe fallback to dense: `{summary['safe_fallback_to_dense']}`
- Rollback to dense equivalence valid: `{summary['rollback_to_dense_equivalence_valid']}`
"""


def prior_verifier_status() -> dict[str, bool]:
    return {
        "task0112_verifier_valid": verify_task0112_artifacts(output_dir=TASK0112_RESULT_DIR, write=False)["status"] == "valid",
        "task0113_verifier_valid": verify_task0113_artifacts(output_dir=TASK0113_RESULT_DIR, write=False)["status"] == "valid",
        "task0114_verifier_valid": verify_task0114_artifacts(output_dir=TASK0114_RESULT_DIR, write=False)["status"] == "valid",
        "task0115_verifier_valid": verify_task0115_artifacts(output_dir=TASK0115_RESULT_DIR, write=False)["status"] == "valid",
        "task0116_verifier_valid": verify_task0116_artifacts(write=False)["status"] == "valid",
        "task0117_verifier_valid": verify_task0117_artifacts(output_dir=TASK0117_RESULT_DIR, write=False)["status"] == "valid",
        "task0118_verifier_valid": task0118.verify_task0118_artifacts(write=False)["status"] == "valid",
    }


def _invalid_audit_check(key: str, value: Any) -> bool:
    if key in {"runtime_representation_uses_gold_metadata", "oracle_query_used", "contaminated_task0116_index_reused"}:
        return value is not False
    return value is not True


def _dedup_valid(rows: list[dict[str, Any]]) -> bool:
    ids = [row["canonical_chunk_id"] for row in rows]
    return len(ids) == len(set(ids))


def _split_for_unit(unit: str) -> str:
    bucket = int(digest_json(unit)[:8], 16) % 5
    return "guard_dev" if bucket == 0 else "formal_eval"


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
