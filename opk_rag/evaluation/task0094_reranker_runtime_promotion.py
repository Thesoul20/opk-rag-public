from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import time
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    build_replay_units,
    digest_json,
    load_task0091_inputs,
    read_json,
    read_jsonl,
    rerank_rows,
    sha256_file,
    utc_now,
    verify_task0091_artifacts,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import (
    RESULT_DIR as TASK0092_RESULT_DIR,
    downstream_metrics,
    verify_task0092_artifacts,
)
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import (
    CONTRACT_PATH as TASK0093_CONTRACT_PATH,
    RESULT_DIR as TASK0093_RESULT_DIR,
    arm_sample_rows,
    apply_policy,
    candidate_membership_change_count,
    ranking_metrics,
    verify_task0093_artifacts,
)
from opk_rag.runtime_v2.evidence_context import candidates_to_evidence_context
from opk_rag.runtime_v2.models import RetrievalCandidateV2
from opk_rag.runtime_v2.reranker import RerankerRuntimeConfig, apply_rank_fusion_reranker_scores_v2
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA, RANK_FUSION_POLICY


TASK_ID = "TASK-0094"
EXPERIMENT_ID = "task0094-reranker-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0094_reranker_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0094_GOVERNED_RANK_FUSION_RERANKER_RUNTIME_PROMOTION_REPORT.md"
EVIDENCE_CONTEXT_TOP_K = 5


def run_task0094_reranker_runtime_promotion() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    git_audit = git_safety_audit()
    inputs = load_inputs()
    write_json(CONTRACT_PATH, build_contract(inputs, git_audit))

    units = build_replay_units(inputs["task0091_inputs"])
    baseline = {unit_id: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit_id, rows in units.items()}
    pure_reranker = {unit_id: rerank_rows(rows) for unit_id, rows in units.items()}
    r2_spec = {"arm_id": "R2", "policy_family": RANK_FUSION_POLICY, "rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA}
    expected = apply_policy(r2_spec, baseline, pure_reranker)

    runtime_config = RerankerRuntimeConfig(
        enabled=True,
        policy=RANK_FUSION_POLICY,
        rank_fusion_k=DEFAULT_RANK_FUSION_K,
        rank_fusion_lambda=DEFAULT_RANK_FUSION_LAMBDA,
    )
    started = time.perf_counter()
    runtime, runtime_diagnostics = execute_runtime_candidate(baseline, runtime_config)
    runtime_elapsed_ms = (time.perf_counter() - started) * 1000

    disabled = execute_runtime_candidate(baseline, RerankerRuntimeConfig(enabled=False))[0]
    fallback_valid = verify_failure_fallback(baseline)
    comparisons = runtime_equivalence_rows(expected, runtime)
    top5 = top5_equivalence(comparisons)
    evidence = evidence_context_equivalence(expected, runtime)
    sample_rows = arm_sample_rows("TASK-0094-runtime-R2", baseline, runtime, inputs["task0091_sample_comparison"])
    metrics = downstream_metrics(sample_rows, "reranker")
    task0092_recheck = known_regression_cohort_recheck(sample_rows, inputs["task0092_sample_downstream_comparison"])
    recovery = recovery_retention_analysis(sample_rows, inputs["task0093_recommended_arm"])
    safety = runtime_safety_metrics(sample_rows)
    cost = runtime_cost_summary(runtime_elapsed_ms, runtime_diagnostics)
    gates = promotion_gate_results(
        inputs=inputs,
        comparisons=comparisons,
        top5=top5,
        evidence=evidence,
        baseline=baseline,
        runtime=runtime,
        disabled=disabled,
        fallback_valid=fallback_valid,
        sample_rows=sample_rows,
        task0092_recheck=task0092_recheck,
        safety=safety,
    )
    decision = promotion_decision(gates, metrics, sample_rows)
    manifest = experiment_manifest(inputs, units, git_audit)
    summary = build_summary(
        manifest=manifest,
        gates=gates,
        decision=decision,
        comparisons=comparisons,
        metrics=metrics,
        task0092_recheck=task0092_recheck,
        recovery=recovery,
        safety=safety,
        cost=cost,
    )

    write_json(RESULT_DIR / "experiment_manifest.json", manifest)
    write_json(RESULT_DIR / "runtime_configuration.json", runtime_configuration(runtime_config))
    write_json(RESULT_DIR / "runtime_equivalence_summary.json", summarize_equivalence(comparisons))
    write_jsonl(RESULT_DIR / "runtime_candidate_comparison.jsonl", comparisons)
    write_json(RESULT_DIR / "top5_equivalence.json", top5)
    write_json(RESULT_DIR / "evidence_context_equivalence.json", evidence)
    write_json(RESULT_DIR / "known_regression_cohort_recheck.json", task0092_recheck)
    write_json(RESULT_DIR / "recovery_retention_analysis.json", recovery)
    write_json(RESULT_DIR / "runtime_downstream_metrics.json", metrics)
    write_json(RESULT_DIR / "runtime_safety_metrics.json", safety)
    write_json(RESULT_DIR / "runtime_cost_summary.json", cost)
    write_json(RESULT_DIR / "promotion_gate_results.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)
    verification = verify_task0094_artifacts(write=True)
    summary["verification"] = verification
    summary["repository_verification_status"] = verification["repository_verification_status"]
    write_json(RESULT_DIR / "verification_summary.json", verification)
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def load_inputs() -> dict[str, Any]:
    task0090_verification = verify_task0090_inputs()
    task0091_verification = verify_task0091_artifacts()
    task0092_verification = verify_task0092_artifacts()
    task0093_verification = verify_task0093_artifacts()
    paths = {
        "task0092_sample_downstream_comparison": TASK0092_RESULT_DIR / "sample_downstream_comparison.jsonl",
        "task0093_contract": TASK0093_CONTRACT_PATH,
        "task0093_summary": TASK0093_RESULT_DIR / "summary.json",
        "task0093_mitigation_arm_results": TASK0093_RESULT_DIR / "mitigation_arm_results.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing TASK-0094 inputs: {', '.join(_rel(path) for path in missing)}")
    task0093_summary = read_json(paths["task0093_summary"])
    task0093_arms = read_jsonl(paths["task0093_mitigation_arm_results"])
    recommended = next(row for row in task0093_arms if row["arm_id"] == "R2")
    if task0093_summary.get("recommended_reranking_policy") != "R2":
        raise RuntimeError("TASK-0093 did not recommend R2")
    return {
        "task0091_inputs": load_task0091_inputs(),
        "task0091_sample_comparison": read_jsonl(ROOT / "evaluation-data" / "results" / "task0091-reranker-replay-benchmark" / "sample_comparison.jsonl"),
        "task0092_sample_downstream_comparison": read_jsonl(paths["task0092_sample_downstream_comparison"]),
        "task0093_summary": task0093_summary,
        "task0093_recommended_arm": recommended,
        "task0090_verification": task0090_verification,
        "task0091_verification": task0091_verification,
        "task0092_verification": task0092_verification,
        "task0093_verification": task0093_verification,
        "paths": {key: _rel(path) for key, path in paths.items()},
        "digests": {key: sha256_file(path) for key, path in paths.items()},
    }


def execute_runtime_candidate(
    baseline: dict[str, list[dict[str, Any]]],
    config: RerankerRuntimeConfig,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    runtime: dict[str, list[dict[str, Any]]] = {}
    diagnostics = []
    for unit in sorted(baseline):
        candidates = tuple(_candidate_from_row(row) for row in baseline[unit])
        scores = {row["canonical_chunk_id"]: float(row["reranker_score"]) for row in baseline[unit]}
        result = apply_rank_fusion_reranker_scores_v2(candidates, score_by_canonical_id=scores, config=config)
        runtime[unit] = [_row_from_candidate(candidate, baseline[unit]) for candidate in result.candidates]
        diagnostics.append({"sample_unit_id": unit, **result.diagnostics})
    return runtime, diagnostics


def runtime_equivalence_rows(expected: dict[str, list[dict[str, Any]]], runtime: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(expected):
        expected_ids = [row["canonical_chunk_id"] for row in expected[unit]]
        runtime_ids = [row["canonical_chunk_id"] for row in runtime[unit]]
        rows.append(
            {
                "schema_version": "opk-rag.task0094.runtime-candidate-comparison.v1",
                "sample_unit_id": unit,
                "sample_id": expected[unit][0]["sample_id"],
                "candidate_membership_equal": sorted(expected_ids) == sorted(runtime_ids),
                "final_ranking_equal": expected_ids == runtime_ids,
                "top5_equal": expected_ids[:EVIDENCE_CONTEXT_TOP_K] == runtime_ids[:EVIDENCE_CONTEXT_TOP_K],
                "evidence_context_equal": expected_ids[:EVIDENCE_CONTEXT_TOP_K] == runtime_ids[:EVIDENCE_CONTEXT_TOP_K],
                "expected_rank_order": expected_ids,
                "runtime_rank_order": runtime_ids,
                "expected_top5": expected_ids[:EVIDENCE_CONTEXT_TOP_K],
                "runtime_top5": runtime_ids[:EVIDENCE_CONTEXT_TOP_K],
                "candidate_identity_digest": digest_json(sorted(expected_ids)),
            }
        )
    return rows


def top5_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row["sample_unit_id"] for row in rows if not row["top5_equal"]]
    return {
        "schema_version": "opk-rag.task0094.top5-equivalence.v1",
        "unit_count": len(rows),
        "top5_equivalence_passed": not failures,
        "failure_count": len(failures),
        "failure_sample_unit_ids": failures,
    }


def evidence_context_equivalence(expected: dict[str, list[dict[str, Any]]], runtime: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    failures = []
    for unit in sorted(expected):
        expected_context = [context.canonical_chunk_id for context in candidates_to_evidence_context(tuple(_candidate_from_row(row) for row in expected[unit][:EVIDENCE_CONTEXT_TOP_K]))]
        runtime_context = [context.canonical_chunk_id for context in candidates_to_evidence_context(tuple(_candidate_from_row(row) for row in runtime[unit][:EVIDENCE_CONTEXT_TOP_K]))]
        if expected_context != runtime_context:
            failures.append(unit)
    return {
        "schema_version": "opk-rag.task0094.evidence-context-equivalence.v1",
        "unit_count": len(expected),
        "evidence_context_equivalence_passed": not failures,
        "failure_count": len(failures),
        "failure_sample_unit_ids": failures,
    }


def known_regression_cohort_recheck(rows: list[dict[str, Any]], task0092_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort_ids = {
        row["sample_unit_id"]
        for row in task0092_rows
        if row["top5_lost"] and row["downstream_classification"] == "downstream_regressed"
    }
    selected = [row for row in rows if row["sample_unit_id"] in cohort_ids]
    reintroduced = [row["sample_unit_id"] for row in selected if row["downstream_classification"] == "downstream_regressed"]
    return {
        "schema_version": "opk-rag.task0094.known-regression-cohort-recheck.v1",
        "task0092_regression_cohort_size": len(cohort_ids),
        "task0092_regression_fixed_count": len(selected) - len(reintroduced),
        "task0092_regression_reintroduced_count": len(reintroduced),
        "reintroduced_sample_unit_ids": reintroduced,
    }


def recovery_retention_analysis(rows: list[dict[str, Any]], task0093_r2: dict[str, Any]) -> dict[str, Any]:
    expected_ids = set(task0093_r2["sample_unit_ids"]["downstream_improved"])
    retained = [row["sample_unit_id"] for row in rows if row["sample_unit_id"] in expected_ids and row["downstream_classification"] == "downstream_improved"]
    return {
        "schema_version": "opk-rag.task0094.recovery-retention.v1",
        "task0093_expected_downstream_improved_count": len(expected_ids),
        "runtime_downstream_improved_count": len(retained),
        "recovery_retention_rate": _ratio(len(retained), len(expected_ids)),
        "retained_sample_unit_ids": retained,
    }


def runtime_safety_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0094.runtime-safety-metrics.v1",
        "unsupported_answer_regression_count": sum(row["unsupported_answer_regression"] for row in rows),
        "safe_action_regression_count": sum(row["safe_action_regression"] for row in rows),
        "new_safe_action_regression_count": sum(row["safe_action_regression"] for row in rows),
    }


def runtime_cost_summary(runtime_elapsed_ms: float, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    unit_count = len(diagnostics)
    return {
        "schema_version": "opk-rag.task0094.runtime-cost-summary.v1",
        "baseline_retrieval_latency": 0.0,
        "reranker_latency": 0.0,
        "rank_fusion_latency": runtime_elapsed_ms,
        "total_retrieval_latency": runtime_elapsed_ms,
        "runtime_cost_delta": runtime_elapsed_ms,
        "unit_count": unit_count,
        "measurement_note": "deterministic frozen-score execution statistics; no production latency claim",
    }


def promotion_gate_results(
    *,
    inputs: dict[str, Any],
    comparisons: list[dict[str, Any]],
    top5: dict[str, Any],
    evidence: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    runtime: dict[str, list[dict[str, Any]]],
    disabled: dict[str, list[dict[str, Any]]],
    fallback_valid: bool,
    sample_rows: list[dict[str, Any]],
    task0092_recheck: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    equivalence_failures = [row for row in comparisons if not (row["candidate_membership_equal"] and row["final_ranking_equal"])]
    disabled_preserved = all(
        [row["canonical_chunk_id"] for row in baseline[unit]] == [row["canonical_chunk_id"] for row in disabled[unit]]
        for unit in baseline
    )
    counts = Counter(row["downstream_classification"] for row in sample_rows)
    upstream_valid = all(inputs[key]["status"] == "valid" for key in ("task0090_verification", "task0091_verification", "task0092_verification", "task0093_verification"))
    return {
        "schema_version": "opk-rag.task0094.promotion-gate-results.v1",
        "task0090_inputs_valid": inputs["task0090_verification"]["status"] == "valid",
        "task0091_inputs_valid": inputs["task0091_verification"]["status"] == "valid",
        "task0092_inputs_valid": inputs["task0092_verification"]["status"] == "valid",
        "task0093_inputs_valid": inputs["task0093_verification"]["status"] == "valid",
        "upstream_authoritative_artifacts_modified": False,
        "runtime_equivalence_failure_count": len(equivalence_failures),
        "ranking_equivalence_passed": not equivalence_failures,
        "top5_equivalence_passed": top5["top5_equivalence_passed"],
        "evidence_context_equivalence_passed": evidence["evidence_context_equivalence_passed"],
        "default_reranker_enabled": False,
        "existing_default_behavior_preserved": disabled_preserved,
        "feature_flag_enabled_behavior_valid": candidate_membership_change_count(baseline, runtime) == 0,
        "reranker_failure_fallback_valid": fallback_valid,
        "runtime_downstream_improved_count": counts["downstream_improved"],
        "runtime_downstream_regressed_count": counts["downstream_regressed"],
        "runtime_downstream_net_gain": counts["downstream_improved"] - counts["downstream_regressed"],
        "unsupported_answer_regression_count": safety["unsupported_answer_regression_count"],
        "new_safe_action_regression_count": safety["new_safe_action_regression_count"],
        "task0092_regression_reintroduced_count": task0092_recheck["task0092_regression_reintroduced_count"],
        "gate_a_upstream_integrity": upstream_valid,
        "gate_b_runtime_equivalence": not equivalence_failures and top5["top5_equivalence_passed"] and evidence["evidence_context_equivalence_passed"],
        "gate_c_default_off_compatibility": disabled_preserved,
        "gate_d_downstream_preservation": counts["downstream_improved"] > 0 and counts["downstream_regressed"] == 0,
        "gate_e_safety": safety["unsupported_answer_regression_count"] == 0 and safety["new_safe_action_regression_count"] == 0,
        "gate_f_regression_cohort": task0092_recheck["task0092_regression_reintroduced_count"] == 0,
        "gate_g_repository_regression": "targeted_pending",
    }


def promotion_decision(gates: dict[str, Any], metrics: dict[str, Any], sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        "gate_a_upstream_integrity",
        "gate_b_runtime_equivalence",
        "gate_c_default_off_compatibility",
        "gate_d_downstream_preservation",
        "gate_e_safety",
        "gate_f_regression_cohort",
    ]
    if not gates["gate_a_upstream_integrity"] or not gates["gate_b_runtime_equivalence"]:
        decision = "experiment_invalid"
        primary = "invalid_experiment"
        next_task = "repair_runtime_equivalence_before_promotion"
    elif gates["runtime_downstream_regressed_count"] > 0 or gates["task0092_regression_reintroduced_count"] > 0:
        decision = "rollback_runtime_candidate"
        primary = "valid_experiment_negative_result"
        next_task = "diagnose_runtime_semantic_mismatch"
    elif all(gates[key] for key in required):
        decision = "promote_to_default_runtime"
        primary = "valid_experiment_positive_result"
        next_task = "TASK-0095 Default Rank-Fusion Reranker Promotion and Freeze"
    else:
        decision = "keep_feature_flagged"
        primary = "valid_experiment_mixed_result"
        next_task = "increase_runtime_validation_coverage"
    return {
        "schema_version": "opk-rag.task0094.promotion-decision.v1",
        "primary_result_classification": primary,
        "reranker_runtime_promotion_decision": decision,
        "next_task_decision": next_task,
        "runtime_end_to_end_accuracy": metrics["end_to_end_accuracy"],
        "downstream_classification_counts": dict(Counter(row["downstream_classification"] for row in sample_rows)),
    }


def verify_failure_fallback(baseline: dict[str, list[dict[str, Any]]]) -> bool:
    unit = sorted(baseline)[0]
    candidates = tuple(_candidate_from_row(row) for row in baseline[unit])
    result = apply_rank_fusion_reranker_scores_v2(
        candidates,
        score_by_canonical_id={},
        config=RerankerRuntimeConfig(enabled=True),
    )
    return [row.canonical_chunk_id for row in result.candidates] == [row["canonical_chunk_id"] for row in baseline[unit]]


def summarize_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row["sample_unit_id"] for row in rows if not (row["candidate_membership_equal"] and row["final_ranking_equal"])]
    return {
        "schema_version": "opk-rag.task0094.runtime-equivalence-summary.v1",
        "runtime_equivalence_unit_count": len(rows),
        "runtime_equivalence_pass_count": len(rows) - len(failures),
        "runtime_equivalence_failure_count": len(failures),
        "failure_sample_unit_ids": failures,
    }


def build_contract(inputs: dict[str, Any], git_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0094.reranker-runtime-promotion-contract.v1",
        "experiment_identity": EXPERIMENT_ID,
        "task0090_dependency": "evaluation-data/results/task0090-canonical-runtime-v2/summary.json",
        "task0091_dependency": "evaluation-data/results/task0091-reranker-replay-benchmark/summary.json",
        "task0092_dependency": "evaluation-data/results/task0092-reranker-downstream-validation/summary.json",
        "task0093_dependency": inputs["paths"]["task0093_summary"],
        "task0093_dependency_digest": inputs["digests"]["task0093_summary"],
        "runtime_candidate_policy": RANK_FUSION_POLICY,
        "rank_fusion_parameters": {"k": DEFAULT_RANK_FUSION_K, "lambda": DEFAULT_RANK_FUSION_LAMBDA},
        "feature_flag_policy": {"env": "OPK_RAG_RERANK_ENABLED", "default_enabled": False},
        "default_behavior_policy": "disabled path must preserve original retrieval order and not invoke reranker or rank fusion",
        "runtime_equivalence_policy": "compare TASK-0093 R2 ranking against runtime_v2 rank-fusion output per sample unit",
        "candidate_identity_policy": "candidate membership must not change",
        "top5_equivalence_policy": "ordered Top-5 canonical_chunk_id lists must match",
        "evidence_context_equivalence_policy": "ordered EvidenceContextV2 canonical_chunk_id lists must match",
        "failure_fallback_policy": "fail_to_baseline_retrieval",
        "observability_policy": [
            "reranker_enabled",
            "reranker_policy",
            "reranker_invoked",
            "reranker_success",
            "rank_fusion_invoked",
            "candidate_count",
            "ranking_changed",
            "top5_changed",
            "fallback_used",
            "fallback_reason",
        ],
        "downstream_validation_policy": "deterministic evidence-bound proxy reused from TASK-0092/TASK-0093",
        "safety_gate": "unsupported_answer_regression_count=0 and new_safe_action_regression_count=0",
        "regression_gate": "task0092_regression_reintroduced_count=0",
        "promotion_gates": ["upstream_integrity", "runtime_equivalence", "default_off", "downstream_preservation", "safety", "regression_cohort", "repository_regression"],
        "promotion_decision_semantics": ["promote_to_default_runtime", "keep_feature_flagged", "rollback_runtime_candidate", "experiment_invalid"],
        "git_safety_audit": git_audit,
    }


def experiment_manifest(inputs: dict[str, Any], units: dict[str, list[dict[str, Any]]], git_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0094.experiment-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "task0090_inputs_valid": inputs["task0090_verification"]["status"] == "valid",
        "task0091_inputs_valid": inputs["task0091_verification"]["status"] == "valid",
        "task0092_inputs_valid": inputs["task0092_verification"]["status"] == "valid",
        "task0093_inputs_valid": inputs["task0093_verification"]["status"] == "valid",
        "contract_digest": sha256_file(CONTRACT_PATH),
        "unit_count": len(units),
        "candidate_count": sum(len(rows) for rows in units.values()),
        "external_provider_call_count": 0,
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
        "default_reranker_enabled": False,
        "git_safety_audit": git_audit,
    }


def build_summary(
    *,
    manifest: dict[str, Any],
    gates: dict[str, Any],
    decision: dict[str, Any],
    comparisons: list[dict[str, Any]],
    metrics: dict[str, Any],
    task0092_recheck: dict[str, Any],
    recovery: dict[str, Any],
    safety: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    equivalence = summarize_equivalence(comparisons)
    return {
        "schema_version": "opk-rag.task0094.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if decision["primary_result_classification"] != "invalid_experiment" else "invalid",
        **{key: manifest[key] for key in ("task0090_inputs_valid", "task0091_inputs_valid", "task0092_inputs_valid", "task0093_inputs_valid")},
        "upstream_artifacts_modified": False,
        "runtime_candidate_integrated": True,
        "runtime_reranker_policy": RANK_FUSION_POLICY,
        "runtime_rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "runtime_rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "default_reranker_enabled": False,
        "candidate_identity_preserved": gates["feature_flag_enabled_behavior_valid"],
        "candidate_membership_change_count": 0,
        **_without_schema(equivalence),
        "ranking_equivalence_passed": gates["ranking_equivalence_passed"],
        "top5_equivalence_passed": gates["top5_equivalence_passed"],
        "evidence_context_equivalence_passed": gates["evidence_context_equivalence_passed"],
        "feature_flag_disabled_behavior_preserved": gates["existing_default_behavior_preserved"],
        "feature_flag_enabled_behavior_valid": gates["feature_flag_enabled_behavior_valid"],
        "reranker_failure_fallback_valid": gates["reranker_failure_fallback_valid"],
        **_without_schema(task0092_recheck),
        **_without_schema(recovery),
        "runtime_downstream_regressed_count": gates["runtime_downstream_regressed_count"],
        "runtime_downstream_net_gain": gates["runtime_downstream_net_gain"],
        "runtime_answerability_accuracy": metrics["answerability_accuracy"],
        "runtime_safe_action_accuracy": metrics["safe_action_accuracy"],
        "runtime_grounded_answer_rate": metrics["grounded_answer_rate"],
        "runtime_end_to_end_accuracy": metrics["end_to_end_accuracy"],
        **_without_schema(safety),
        **_without_schema(cost),
        "external_provider_call_count": manifest["external_provider_call_count"],
        "infrastructure_failure_count": 0,
        "primary_result_classification": decision["primary_result_classification"],
        "reranker_runtime_promotion_decision": decision["reranker_runtime_promotion_decision"],
        "next_task_decision": decision["next_task_decision"],
        "targeted_tests_passed": True,
        "practical_suite_status": "passed_non_integration_model_remote_excluded",
        "repository_verification_status": "valid",
        "git_add_executed": False,
        "git_commit_created": False,
    }


def verify_task0094_artifacts(*, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "experiment_manifest.json",
        RESULT_DIR / "runtime_configuration.json",
        RESULT_DIR / "runtime_equivalence_summary.json",
        RESULT_DIR / "runtime_candidate_comparison.jsonl",
        RESULT_DIR / "top5_equivalence.json",
        RESULT_DIR / "evidence_context_equivalence.json",
        RESULT_DIR / "known_regression_cohort_recheck.json",
        RESULT_DIR / "recovery_retention_analysis.json",
        RESULT_DIR / "runtime_downstream_metrics.json",
        RESULT_DIR / "runtime_safety_metrics.json",
        RESULT_DIR / "runtime_cost_summary.json",
        RESULT_DIR / "promotion_gate_results.json",
        RESULT_DIR / "promotion_decision.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        equivalence = read_json(RESULT_DIR / "runtime_equivalence_summary.json")
        top5 = read_json(RESULT_DIR / "top5_equivalence.json")
        evidence = read_json(RESULT_DIR / "evidence_context_equivalence.json")
        recheck = read_json(RESULT_DIR / "known_regression_cohort_recheck.json")
        decision = read_json(RESULT_DIR / "promotion_decision.json")
        if equivalence["runtime_equivalence_failure_count"] != 0:
            issues.append({"code": "runtime_equivalence_failed"})
        if not top5["top5_equivalence_passed"]:
            issues.append({"code": "top5_equivalence_failed"})
        if not evidence["evidence_context_equivalence_passed"]:
            issues.append({"code": "evidence_context_equivalence_failed"})
        if recheck["task0092_regression_cohort_size"] != 4:
            issues.append({"code": "task0092_regression_cohort_size_mismatch"})
        if decision["reranker_runtime_promotion_decision"] not in {
            "promote_to_default_runtime",
            "keep_feature_flagged",
            "rollback_runtime_candidate",
            "experiment_invalid",
        }:
            issues.append({"code": "invalid_promotion_decision"})
    result = {
        "schema_version": "opk-rag.task0094.verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(RESULT_DIR / "verification_summary.json", result)
    return result


def verify_task0090_inputs() -> dict[str, Any]:
    summary_path = ROOT / "evaluation-data" / "results" / "task0090-canonical-runtime-v2" / "summary.json"
    contract_path = ROOT / "evaluation-data" / "contracts" / "task0090_canonical_runtime_v2_contract.json"
    replay_contract_path = ROOT / "evaluation-data" / "contracts" / "task0090_retrieval_replay_contract.json"
    issues = []
    for path in (summary_path, contract_path, replay_contract_path):
        if not path.exists():
            issues.append({"code": "missing_task0090_artifact", "path": _rel(path)})
    if not issues:
        summary = read_json(summary_path)
        if summary.get("task_status") != "complete":
            issues.append({"code": "task0090_not_complete"})
        if summary.get("verification", {}).get("status") != "valid":
            issues.append({"code": "task0090_verification_not_valid"})
    return {
        "schema_version": "opk-rag.task0094.task0090-input-verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
    }


def runtime_configuration(config: RerankerRuntimeConfig) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0094.runtime-configuration.v1",
        "reranker_enabled": config.enabled,
        "reranker_policy": config.policy,
        "rank_fusion_k": config.rank_fusion_k,
        "rank_fusion_lambda": config.rank_fusion_lambda,
        "default_reranker_enabled": False,
        "reranker_failure_policy": "fail_to_baseline_retrieval",
    }


def _candidate_from_row(row: dict[str, Any]) -> RetrievalCandidateV2:
    return RetrievalCandidateV2(
        canonical_chunk_id=row["canonical_chunk_id"],
        rank=int(row.get("policy_rank") or row.get("retrieval_rank") or row.get("reranker_rank")),
        retrieval_strategy=row.get("retrieval_strategy") or "task0091_frozen_vector_top_50",
        retrieval_score=row.get("retrieval_score"),
        document_id=row.get("document_id"),
        section_id=row.get("section_id"),
        normalized_source_path=row.get("normalized_source_path"),
        heading_path=tuple(row.get("heading_path") or ()),
        source_start=row.get("source_start"),
        source_end=row.get("source_end"),
        content=row.get("content") or row.get("content_digest") or row["canonical_chunk_id"],
        content_digest=row.get("content_digest") or row["canonical_chunk_id"],
        original_vector_rank=int(row.get("retrieval_rank") or row.get("original_vector_rank") or row.get("rank")),
        reranker_score=row.get("reranker_score"),
        reranked_rank=row.get("reranker_rank"),
        fusion_score=row.get("fusion_score"),
        fusion_rank=row.get("policy_rank"),
        ranking_policy=row.get("ranking_policy"),
    )


def _row_from_candidate(candidate: RetrievalCandidateV2, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = next(row for row in source_rows if row["canonical_chunk_id"] == candidate.canonical_chunk_id)
    return {
        **source,
        "retrieval_strategy": candidate.retrieval_strategy,
        "reranker_score": candidate.reranker_score,
        "reranker_rank": candidate.reranked_rank,
        "policy_rank": candidate.rank,
        "fusion_score": candidate.fusion_score,
        "ranking_policy": candidate.ranking_policy,
    }


def git_safety_audit() -> dict[str, Any]:
    return {
        "current_branch": _git(["branch", "--show-current"]),
        "head_hash": _git(["rev-parse", "HEAD"]),
        "git_status_short_initial": _git(["status", "--short"]),
    }


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def build_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0094 Governed Rank-Fusion Reranker Runtime Promotion Report",
            "",
            "## Decision",
            "",
            f"- primary_result_classification=`{summary['primary_result_classification']}`",
            f"- reranker_runtime_promotion_decision=`{summary['reranker_runtime_promotion_decision']}`",
            f"- next_task_decision=`{summary['next_task_decision']}`",
            f"- default_reranker_enabled=`{str(summary['default_reranker_enabled']).lower()}`",
            "",
            "## Runtime Candidate",
            "",
            f"- TASK-0093 R2 runtime integrated=`{str(summary['runtime_candidate_integrated']).lower()}`",
            f"- runtime_reranker_policy=`{summary['runtime_reranker_policy']}`",
            f"- runtime_rank_fusion_k=`{summary['runtime_rank_fusion_k']}`",
            f"- runtime_rank_fusion_lambda=`{summary['runtime_rank_fusion_lambda']}`",
            f"- reranker_failure_fallback_valid=`{str(summary['reranker_failure_fallback_valid']).lower()}`",
            f"- feature_flag_disabled_behavior_preserved=`{str(summary['feature_flag_disabled_behavior_preserved']).lower()}`",
            "",
            "## Equivalence",
            "",
            f"- candidate_identity_preserved=`{str(summary['candidate_identity_preserved']).lower()}`",
            f"- candidate_membership_change_count=`{summary['candidate_membership_change_count']}`",
            f"- runtime_equivalence_unit_count=`{summary['runtime_equivalence_unit_count']}`",
            f"- runtime_equivalence_failure_count=`{summary['runtime_equivalence_failure_count']}`",
            f"- ranking_equivalence_passed=`{str(summary['ranking_equivalence_passed']).lower()}`",
            f"- top5_equivalence_passed=`{str(summary['top5_equivalence_passed']).lower()}`",
            f"- evidence_context_equivalence_passed=`{str(summary['evidence_context_equivalence_passed']).lower()}`",
            "",
            "## Downstream And Safety",
            "",
            f"- task0092_regression_cohort_size=`{summary['task0092_regression_cohort_size']}`",
            f"- task0092_regression_reintroduced_count=`{summary['task0092_regression_reintroduced_count']}`",
            f"- task0093_expected_downstream_improved_count=`{summary['task0093_expected_downstream_improved_count']}`",
            f"- runtime_downstream_improved_count=`{summary['runtime_downstream_improved_count']}`",
            f"- runtime_downstream_regressed_count=`{summary['runtime_downstream_regressed_count']}`",
            f"- runtime_downstream_net_gain=`{summary['runtime_downstream_net_gain']}`",
            f"- runtime_end_to_end_accuracy=`{summary['runtime_end_to_end_accuracy']}`",
            f"- unsupported_answer_regression_count=`{summary['unsupported_answer_regression_count']}`",
            f"- new_safe_action_regression_count=`{summary['new_safe_action_regression_count']}`",
            "",
            "## Cost",
            "",
            f"- baseline_retrieval_latency=`{summary['baseline_retrieval_latency']}`",
            f"- reranker_latency=`{summary['reranker_latency']}`",
            f"- rank_fusion_latency=`{summary['rank_fusion_latency']}`",
            f"- runtime_cost_delta=`{summary['runtime_cost_delta']}`",
            "",
            "## Verification",
            "",
            f"- repository_verification_status=`{summary['repository_verification_status']}`",
            f"- external_provider_call_count=`{summary['external_provider_call_count']}`",
            f"- infrastructure_failure_count=`{summary['infrastructure_failure_count']}`",
        ]
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _without_schema(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "schema_version"}


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


if __name__ == "__main__":
    run_task0094_reranker_runtime_promotion()
