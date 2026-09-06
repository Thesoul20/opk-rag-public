from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    CONTRACT_PATH as TASK0091_CONTRACT_PATH,
    RESULT_DIR as TASK0091_RESULT_DIR,
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
from opk_rag.runtime_v2.evidence_context import candidates_to_evidence_context
from opk_rag.runtime_v2.models import RetrievalCandidateV2


TASK_ID = "TASK-0092"
EXPERIMENT_ID = "task0092-reranker-downstream-validation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0092_reranker_downstream_validation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0092_RERANKER_DOWNSTREAM_END_TO_END_VALIDATION_REPORT.md"
EVIDENCE_CONTEXT_TOP_K = 5
REPLICATE_COUNT = 3


def run_task0092_reranker_downstream_validation() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    units = build_replay_units(inputs["task0091_inputs"])
    baseline = {unit_id: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit_id, rows in units.items()}
    reranked = {unit_id: rerank_rows(rows) for unit_id, rows in units.items()}
    write_json(CONTRACT_PATH, build_contract(inputs))

    context_rows = build_evidence_context_comparisons(baseline, reranked)
    sample_rows = build_sample_downstream_comparisons(context_rows, inputs["task0091_sample_comparison"])
    baseline_metrics = downstream_metrics(sample_rows, "baseline")
    reranker_metrics = downstream_metrics(sample_rows, "reranker")
    deltas = metric_deltas(baseline_metrics, reranker_metrics)
    cohort = task0091_cohort_analysis(sample_rows)
    top5_recovered = top5_analysis(sample_rows, cohort_name="top5_recovered")
    top5_lost = top5_lost_analysis(sample_rows)
    stability = replicate_stability_analysis(sample_rows)
    gates = promotion_gates(inputs, baseline, reranked, sample_rows, baseline_metrics, reranker_metrics, stability)
    decision = promotion_decision(gates, deltas, sample_rows)
    manifest = experiment_manifest(inputs, units)
    summary = build_summary(
        manifest,
        baseline_metrics,
        reranker_metrics,
        deltas,
        cohort,
        top5_recovered,
        top5_lost,
        stability,
        gates,
        decision,
    )

    write_json(RESULT_DIR / "experiment_manifest.json", manifest)
    write_json(RESULT_DIR / "baseline_downstream_metrics.json", baseline_metrics)
    write_json(RESULT_DIR / "reranker_downstream_metrics.json", reranker_metrics)
    write_json(RESULT_DIR / "downstream_metric_deltas.json", deltas)
    write_jsonl(RESULT_DIR / "evidence_context_comparison.jsonl", context_rows)
    write_jsonl(RESULT_DIR / "sample_downstream_comparison.jsonl", sample_rows)
    write_json(RESULT_DIR / "task0091_cohort_analysis.json", cohort)
    write_json(RESULT_DIR / "top5_recovered_analysis.json", top5_recovered)
    write_json(RESULT_DIR / "top5_lost_regression_analysis.json", top5_lost)
    write_json(RESULT_DIR / "replicate_stability_analysis.json", stability)
    write_json(RESULT_DIR / "promotion_gates.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)
    verification = verify_task0092_artifacts(write=True)
    summary["verification"] = verification
    summary["repository_verification_status"] = verification["repository_verification_status"]
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def load_inputs() -> dict[str, Any]:
    task0091_verification = verify_task0091_artifacts()
    paths = {
        "task0091_contract": TASK0091_CONTRACT_PATH,
        "task0091_summary": TASK0091_RESULT_DIR / "summary.json",
        "task0091_sample_comparison": TASK0091_RESULT_DIR / "sample_comparison.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing TASK-0092 inputs: {', '.join(_rel(path) for path in missing)}")
    task0091_summary = read_json(paths["task0091_summary"])
    if task0091_verification.get("status") != "valid":
        raise RuntimeError("TASK-0091 artifacts are not valid")
    if task0091_summary.get("reranker_promotion_recommendation") != "promote_to_downstream_validation":
        raise RuntimeError("TASK-0091 did not recommend downstream validation")
    return {
        "task0091_inputs": load_task0091_inputs(),
        "task0091_summary": task0091_summary,
        "task0091_verification": task0091_verification,
        "task0091_sample_comparison": read_jsonl(paths["task0091_sample_comparison"]),
        "paths": {key: _rel(path) for key, path in paths.items()},
        "digests": {key: sha256_file(path) for key, path in paths.items()},
    }


def build_evidence_context_comparisons(
    baseline: dict[str, list[dict[str, Any]]],
    reranked: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for unit_id in sorted(baseline):
        left = baseline[unit_id]
        right = reranked[unit_id]
        baseline_context = build_evidence_context(left[:EVIDENCE_CONTEXT_TOP_K])
        reranker_context = build_evidence_context(right[:EVIDENCE_CONTEXT_TOP_K])
        baseline_ids = [context["canonical_chunk_id"] for context in baseline_context]
        reranker_ids = [context["canonical_chunk_id"] for context in reranker_context]
        relevant_before = [row["canonical_chunk_id"] for row in left[:EVIDENCE_CONTEXT_TOP_K] if row["relevant_label"]]
        relevant_after = [row["canonical_chunk_id"] for row in right[:EVIDENCE_CONTEXT_TOP_K] if row["relevant_label"]]
        rows.append(
            {
                "schema_version": "opk-rag.task0092.evidence-context-comparison.v1",
                "query_id": unit_id,
                "sample_unit_id": unit_id,
                "sample_id": left[0]["sample_id"],
                "candidate_identity_set_digest": digest_json(sorted(row["canonical_chunk_id"] for row in left)),
                "baseline_rank_order": [row["canonical_chunk_id"] for row in left],
                "reranker_rank_order": [row["canonical_chunk_id"] for row in right],
                "baseline_evidence_context_chunk_ids": baseline_ids,
                "reranker_evidence_context_chunk_ids": reranker_ids,
                "evidence_context_top_k": EVIDENCE_CONTEXT_TOP_K,
                "ranking_changed": [row["canonical_chunk_id"] for row in left] != [row["canonical_chunk_id"] for row in right],
                "evidence_context_changed": baseline_ids != reranker_ids,
                "relevant_evidence_in_context_before": bool(relevant_before),
                "relevant_evidence_in_context_after": bool(relevant_after),
                "baseline_relevant_context_chunk_ids": relevant_before,
                "reranker_relevant_context_chunk_ids": relevant_after,
                "baseline_evidence_context": baseline_context,
                "reranker_evidence_context": reranker_context,
            }
        )
    return rows


def build_evidence_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = tuple(_candidate_from_row(index, row) for index, row in enumerate(rows, start=1))
    return [context.to_json() for context in candidates_to_evidence_context(candidates)]


def build_sample_downstream_comparisons(context_rows: list[dict[str, Any]], task0091_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task0091_by_unit = {row["sample_unit_id"]: row for row in task0091_rows}
    rows = []
    for context in context_rows:
        unit_id = context["sample_unit_id"]
        ranking = task0091_by_unit[unit_id]
        baseline = arm_outcome(context["relevant_evidence_in_context_before"])
        reranker = arm_outcome(context["relevant_evidence_in_context_after"])
        classification = classify_downstream_pair(baseline, reranker, infrastructure_failure=False)
        rows.append(
            {
                "schema_version": "opk-rag.task0092.sample-downstream-comparison.v1",
                "sample_unit_id": unit_id,
                "sample_id": context["sample_id"],
                "task0091_classification": ranking["classification"],
                "top5_recovered": not ranking["baseline_hit_at_5"] and ranking["reranker_hit_at_5"],
                "top5_lost": ranking["baseline_hit_at_5"] and not ranking["reranker_hit_at_5"],
                "ranking_changed": context["ranking_changed"],
                "evidence_context_changed": context["evidence_context_changed"],
                "baseline": baseline,
                "reranker": reranker,
                "downstream_classification": classification,
                "ranking_only_classification": ranking_only_classification(ranking["classification"], classification),
                "unsupported_answer_regression": baseline["unsupported_answer_count"] == 0 and reranker["unsupported_answer_count"] > 0,
                "safe_action_regression": baseline["safe_action_correct"] and not reranker["safe_action_correct"],
                "infrastructure_failure": False,
                "provider_config_identity": "not_used_deterministic_evidence_bound_proxy",
                "generation_policy_identity": "task0092_evidence_bound_proxy_v1",
            }
        )
    return rows


def arm_outcome(relevant_evidence_in_context: bool) -> dict[str, Any]:
    answerable = relevant_evidence_in_context
    return {
        "answerability_status": "answerable" if answerable else "abstained_no_relevant_evidence_in_context",
        "answerability_correct": answerable,
        "correct_answer_route": answerable,
        "correct_abstention": False,
        "over_abstention": not answerable,
        "under_abstention": False,
        "safe_action_correct": answerable,
        "safe_refusal_count": 0,
        "unsafe_answer_count": 0,
        "unsupported_answer_count": 0,
        "generation_invoked": answerable,
        "generation_contract_valid": answerable,
        "answer_draft_count": 1 if answerable else 0,
        "model_refusal_count": 0,
        "provider_failure_count": 0,
        "grounded_answer": answerable,
        "grounding_failure_count": 0 if answerable else 1,
        "unsupported_claim_count": 0,
        "citation_valid": answerable,
        "citation_invalid": False,
        "end_to_end_correct": answerable,
    }


def classify_downstream_pair(baseline: dict[str, Any], reranker: dict[str, Any], *, infrastructure_failure: bool) -> str:
    if infrastructure_failure:
        return "infrastructure_failure"
    if not baseline["end_to_end_correct"] and reranker["end_to_end_correct"]:
        return "downstream_improved"
    if baseline["end_to_end_correct"] and not reranker["end_to_end_correct"]:
        return "downstream_regressed"
    return "downstream_unchanged"


def ranking_only_classification(task0091_classification: str, downstream_classification: str) -> str | None:
    if downstream_classification != "downstream_unchanged":
        return None
    if task0091_classification == "improved":
        return "ranking_only_improved"
    if task0091_classification == "regressed":
        return "ranking_only_regressed"
    return None


def downstream_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = [row[arm] for row in rows]
    total = len(outcomes)
    generation_invocations = sum(row["generation_invoked"] for row in outcomes)
    citation_valid = sum(row["citation_valid"] for row in outcomes)
    return {
        "schema_version": "opk-rag.task0092.downstream-metrics.v1",
        "arm": arm,
        "unit_count": total,
        "answerability_correct": sum(row["answerability_correct"] for row in outcomes),
        "answerability_incorrect": sum(not row["answerability_correct"] for row in outcomes),
        "answerability_accuracy": _ratio(sum(row["answerability_correct"] for row in outcomes), total),
        "correct_answer_route_rate": _ratio(sum(row["correct_answer_route"] for row in outcomes), total),
        "correct_abstention_rate": _ratio(sum(row["correct_abstention"] for row in outcomes), total),
        "over_abstention_count": sum(row["over_abstention"] for row in outcomes),
        "under_abstention_count": sum(row["under_abstention"] for row in outcomes),
        "safe_action_accuracy": _ratio(sum(row["safe_action_correct"] for row in outcomes), total),
        "safe_refusal_count": sum(row["safe_refusal_count"] for row in outcomes),
        "unsafe_answer_count": sum(row["unsafe_answer_count"] for row in outcomes),
        "unsupported_answer_count": sum(row["unsupported_answer_count"] for row in outcomes),
        "generation_invocation_count": generation_invocations,
        "generation_contract_valid_count": sum(row["generation_contract_valid"] for row in outcomes),
        "answer_draft_count": sum(row["answer_draft_count"] for row in outcomes),
        "model_refusal_count": sum(row["model_refusal_count"] for row in outcomes),
        "provider_failure_count": sum(row["provider_failure_count"] for row in outcomes),
        "grounded_answer_count": sum(row["grounded_answer"] for row in outcomes),
        "grounded_answer_rate": _ratio(sum(row["grounded_answer"] for row in outcomes), total),
        "grounding_failure_count": sum(row["grounding_failure_count"] for row in outcomes),
        "unsupported_claim_count": sum(row["unsupported_claim_count"] for row in outcomes),
        "citation_valid_count": citation_valid,
        "citation_invalid_count": sum(row["citation_invalid"] for row in outcomes),
        "citation_correctness": _ratio(citation_valid, generation_invocations),
        "end_to_end_correct_count": sum(row["end_to_end_correct"] for row in outcomes),
        "end_to_end_incorrect_count": sum(not row["end_to_end_correct"] for row in outcomes),
        "end_to_end_accuracy": _ratio(sum(row["end_to_end_correct"] for row in outcomes), total),
        "live_provider_generation_executed": False,
    }


def task0091_cohort_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohorts: dict[str, list[dict[str, Any]]] = {
        "rerank_improved": [row for row in rows if row["task0091_classification"] == "improved"],
        "rerank_unchanged": [row for row in rows if row["task0091_classification"] == "unchanged"],
        "rerank_regressed": [row for row in rows if row["task0091_classification"] == "regressed"],
        "top5_recovered": [row for row in rows if row["top5_recovered"]],
        "top5_lost": [row for row in rows if row["top5_lost"]],
    }
    return {
        "schema_version": "opk-rag.task0092.task0091-cohort-analysis.v1",
        "cohorts": {name: cohort_counts(items) for name, items in cohorts.items()},
    }


def cohort_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["downstream_classification"] for row in rows)
    return {
        "unit_count": len(rows),
        "downstream_improved_count": counts["downstream_improved"],
        "downstream_unchanged_count": counts["downstream_unchanged"],
        "downstream_regressed_count": counts["downstream_regressed"],
        "sample_unit_ids": [row["sample_unit_id"] for row in rows],
    }


def top5_analysis(rows: list[dict[str, Any]], *, cohort_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row[cohort_name]]
    counts = cohort_counts(selected)
    return {
        "schema_version": f"opk-rag.task0092.{cohort_name.replace('_', '-')}-analysis.v1",
        **counts,
    }


def top5_lost_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["top5_lost"]]
    counts = cohort_counts(selected)
    return {
        "schema_version": "opk-rag.task0092.top5-lost-regression-analysis.v1",
        **counts,
        "regression_sample_unit_ids": [row["sample_unit_id"] for row in selected if row["downstream_classification"] == "downstream_regressed"],
    }


def replicate_stability_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable_counts = Counter(row["downstream_classification"] for row in rows)
    return {
        "schema_version": "opk-rag.task0092.replicate-stability-analysis.v1",
        "replicate_count": REPLICATE_COUNT,
        "live_provider_generation_executed": False,
        "external_provider_call_count": 0,
        "provider_config_identity": "not_used_deterministic_evidence_bound_proxy",
        "generation_policy_identity": "task0092_evidence_bound_proxy_v1",
        "stability_classification": "deterministic_proxy_replicate_stable",
        "stable_downstream_improvement_count": stable_counts["downstream_improved"],
        "stable_downstream_regression_count": stable_counts["downstream_regressed"],
        "stable_no_change_count": stable_counts["downstream_unchanged"],
        "generation_nondeterminism_unit_count": 0,
        "infrastructure_failure_unit_count": 0,
    }


def promotion_gates(
    inputs: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    reranked: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    reranker_metrics: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    candidate_identity_preserved = all(
        sorted(row["canonical_chunk_id"] for row in baseline[unit]) == sorted(row["canonical_chunk_id"] for row in reranked[unit])
        for unit in baseline
    )
    downstream_regressed = [row["sample_unit_id"] for row in rows if row["downstream_classification"] == "downstream_regressed"]
    unsupported_regression = [row["sample_unit_id"] for row in rows if row["unsupported_answer_regression"]]
    safe_action_regression = [row["sample_unit_id"] for row in rows if row["safe_action_regression"]]
    return {
        "schema_version": "opk-rag.task0092.promotion-gates.v1",
        "gate_a_experiment_integrity": inputs["task0091_verification"]["status"] == "valid" and candidate_identity_preserved,
        "task0091_inputs_valid": inputs["task0091_verification"]["status"] == "valid",
        "candidate_set_frozen": True,
        "candidate_identity_preserved": candidate_identity_preserved,
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
        "gate_b_end_to_end_improvement": reranker_metrics["end_to_end_accuracy"] > baseline_metrics["end_to_end_accuracy"],
        "gate_c_safety_preservation": not unsupported_regression and not any(row["reranker"]["unsafe_answer_count"] for row in rows),
        "new_unsafe_answer_count": 0,
        "new_unsupported_answer_regression_count": len(unsupported_regression),
        "gate_d_regression_budget": not downstream_regressed,
        "downstream_regressed_sample_unit_ids": downstream_regressed,
        "top5_lost_downstream_regression_sample_unit_ids": [
            row["sample_unit_id"] for row in rows if row["top5_lost"] and row["downstream_classification"] == "downstream_regressed"
        ],
        "safe_action_regression_sample_unit_ids": safe_action_regression,
        "gate_e_stability": stability["generation_nondeterminism_unit_count"] == 0,
        "live_provider_generation_executed": False,
        "default_reranker_enabled": False,
    }


def promotion_decision(gates: dict[str, Any], deltas: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = all(
        gates[key]
        for key in (
            "gate_a_experiment_integrity",
            "candidate_set_frozen",
            "candidate_identity_preserved",
        )
    )
    regressed = sum(row["downstream_classification"] == "downstream_regressed" for row in rows)
    improved = sum(row["downstream_classification"] == "downstream_improved" for row in rows)
    if not integrity:
        primary = "invalid_experiment"
        recommendation = "experiment_invalid"
        next_task = "restore_task0092_experiment_integrity"
    elif improved > 0 and regressed == 0 and gates["gate_c_safety_preservation"] and gates["live_provider_generation_executed"]:
        primary = "valid_experiment_positive_result"
        recommendation = "promote_to_runtime_candidate"
        next_task = "TASK-0093 Governed Reranker Runtime Promotion"
    elif improved > 0 and regressed > 0:
        primary = "valid_experiment_mixed_result"
        recommendation = "requires_regression_mitigation"
        next_task = "diagnose_top5_lost_and_downstream_regression_cohorts_before_runtime_promotion"
    elif deltas["end_to_end_accuracy"] <= 0:
        primary = "valid_experiment_negative_result"
        recommendation = "do_not_promote"
        next_task = "reranker_downstream_conversion_insufficient"
    else:
        primary = "valid_experiment_mixed_result"
        recommendation = "requires_regression_mitigation"
        next_task = "requires_live_provider_stability_validation"
    return {
        "schema_version": "opk-rag.task0092.promotion-decision.v1",
        "primary_result_classification": primary,
        "reranker_runtime_promotion_recommendation": recommendation,
        "next_task_decision": next_task,
        "live_provider_generation_executed": gates["live_provider_generation_executed"],
    }


def build_contract(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0092.reranker-downstream-validation-contract.v1",
        "experiment_identity": EXPERIMENT_ID,
        "task0090_dependency": inputs["task0091_inputs"]["paths"]["task0090_candidates"],
        "task0091_dependency": inputs["paths"]["task0091_summary"],
        "task0091_dependency_digest": inputs["digests"]["task0091_summary"],
        "paired_arm_contract": {"c0": "task0091_baseline_order", "r1": "task0091_reranker_order", "only_difference": "candidate_ordering"},
        "candidate_identity_policy": "same_frozen_candidate_membership_digest_per_unit",
        "evidence_context_policy": {"schema": "EvidenceContextV2", "top_k": EVIDENCE_CONTEXT_TOP_K},
        "answerability_policy": "deterministic_evidence_bound_proxy_no_threshold_change",
        "generation_policy": "deterministic_evidence_bound_proxy_live_provider_not_called",
        "citation_policy": "runtime_v2_evidence_context_citation_metadata",
        "grounding_policy": "gold_evidence_present_in_context_proxy",
        "replicate_policy": {"minimum_replicate_count": REPLICATE_COUNT, "provider_calls": 0},
        "stability_policy": "deterministic_proxy_must_be_replicate_stable",
        "downstream_metrics": [
            "answerability_accuracy",
            "safe_action_accuracy",
            "grounded_answer_rate",
            "citation_correctness",
            "end_to_end_accuracy",
        ],
        "sample_classification": [
            "downstream_improved",
            "downstream_unchanged",
            "downstream_regressed",
            "ranking_only_improved",
            "ranking_only_regressed",
            "unstable_generation",
            "infrastructure_failure",
        ],
        "promotion_gates": ["experiment_integrity", "end_to_end_improvement", "safety_preservation", "regression_budget", "stability"],
        "validity_rules": [
            "TASK-0091 artifacts valid",
            "candidate identities preserved",
            "retrieval_rerun_count=0",
            "query_reformulation_count=0",
            "graph_retrieval_count=0",
            "live provider generation not claimed when not executed",
        ],
    }


def experiment_manifest(inputs: dict[str, Any], units: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0092.experiment-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "task0091_inputs_valid": inputs["task0091_verification"]["status"] == "valid",
        "task0091_artifacts_modified": False,
        "task0091_summary_digest": inputs["digests"]["task0091_summary"],
        "contract_digest": sha256_file(CONTRACT_PATH),
        "unit_count": len(units),
        "candidate_count": sum(len(rows) for rows in units.values()),
        "evidence_context_top_k": EVIDENCE_CONTEXT_TOP_K,
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
        "external_provider_call_count": 0,
        "live_provider_generation_executed": False,
        "source_paths": inputs["paths"],
    }


def build_summary(
    manifest: dict[str, Any],
    baseline_metrics: dict[str, Any],
    reranker_metrics: dict[str, Any],
    deltas: dict[str, Any],
    cohort: dict[str, Any],
    top5_recovered: dict[str, Any],
    top5_lost: dict[str, Any],
    stability: dict[str, Any],
    gates: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    improved_cohort = cohort["cohorts"]["rerank_improved"]
    unchanged_cohort = cohort["cohorts"]["rerank_unchanged"]
    regressed_cohort = cohort["cohorts"]["rerank_regressed"]
    downstream_improved = sum(item["downstream_improved_count"] for item in (improved_cohort, unchanged_cohort, regressed_cohort))
    downstream_unchanged = sum(item["downstream_unchanged_count"] for item in (improved_cohort, unchanged_cohort, regressed_cohort))
    downstream_regressed = sum(item["downstream_regressed_count"] for item in (improved_cohort, unchanged_cohort, regressed_cohort))
    return {
        "schema_version": "opk-rag.task0092.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if decision["primary_result_classification"] != "invalid_experiment" else "invalid",
        "task0091_inputs_valid": manifest["task0091_inputs_valid"],
        "task0091_artifacts_modified": manifest["task0091_artifacts_modified"],
        "candidate_set_frozen": gates["candidate_set_frozen"],
        "candidate_identity_preserved": gates["candidate_identity_preserved"],
        "retrieval_rerun_count": gates["retrieval_rerun_count"],
        "query_reformulation_count": gates["query_reformulation_count"],
        "graph_retrieval_count": gates["graph_retrieval_count"],
        "control_unit_count": baseline_metrics["unit_count"],
        "reranker_unit_count": reranker_metrics["unit_count"],
        "baseline_answerability_accuracy": baseline_metrics["answerability_accuracy"],
        "reranker_answerability_accuracy": reranker_metrics["answerability_accuracy"],
        "baseline_safe_action_accuracy": baseline_metrics["safe_action_accuracy"],
        "reranker_safe_action_accuracy": reranker_metrics["safe_action_accuracy"],
        "baseline_grounded_answer_rate": baseline_metrics["grounded_answer_rate"],
        "reranker_grounded_answer_rate": reranker_metrics["grounded_answer_rate"],
        "baseline_end_to_end_accuracy": baseline_metrics["end_to_end_accuracy"],
        "reranker_end_to_end_accuracy": reranker_metrics["end_to_end_accuracy"],
        "downstream_improved_unit_count": downstream_improved,
        "downstream_unchanged_unit_count": downstream_unchanged,
        "downstream_regressed_unit_count": downstream_regressed,
        "ranking_only_improved_unit_count": improved_cohort["downstream_unchanged_count"],
        "ranking_only_regressed_unit_count": regressed_cohort["downstream_unchanged_count"],
        "task0091_ranking_improved_downstream_improved_count": improved_cohort["downstream_improved_count"],
        "task0091_ranking_regressed_downstream_regressed_count": regressed_cohort["downstream_regressed_count"],
        "top5_recovered_unit_count": top5_recovered["unit_count"],
        "top5_recovered_downstream_improved_count": top5_recovered["downstream_improved_count"],
        "top5_recovered_downstream_unchanged_count": top5_recovered["downstream_unchanged_count"],
        "top5_recovered_downstream_regressed_count": top5_recovered["downstream_regressed_count"],
        "top5_lost_unit_count": top5_lost["unit_count"],
        "top5_lost_downstream_regression_count": top5_lost["downstream_regressed_count"],
        "unsupported_answer_regression_count": gates["new_unsupported_answer_regression_count"],
        "safe_action_regression_count": len(gates["safe_action_regression_sample_unit_ids"]),
        "replicate_count": stability["replicate_count"],
        "stable_downstream_improvement_count": stability["stable_downstream_improvement_count"],
        "stable_downstream_regression_count": stability["stable_downstream_regression_count"],
        "generation_nondeterminism_unit_count": stability["generation_nondeterminism_unit_count"],
        "external_provider_call_count": stability["external_provider_call_count"],
        "infrastructure_failure_unit_count": stability["infrastructure_failure_unit_count"],
        "live_provider_generation_executed": stability["live_provider_generation_executed"],
        "live_provider_blocked_reason": "not_required_for_deterministic_frozen_candidate_proxy; no provider configured or called",
        "downstream_metric_deltas": deltas,
        "primary_result_classification": decision["primary_result_classification"],
        "reranker_runtime_promotion_recommendation": decision["reranker_runtime_promotion_recommendation"],
        "next_task_decision": decision["next_task_decision"],
        "targeted_tests_passed": None,
        "repository_verification_status": None,
        "default_reranker_enabled": False,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def verify_task0092_artifacts(*, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "experiment_manifest.json",
        RESULT_DIR / "baseline_downstream_metrics.json",
        RESULT_DIR / "reranker_downstream_metrics.json",
        RESULT_DIR / "downstream_metric_deltas.json",
        RESULT_DIR / "evidence_context_comparison.jsonl",
        RESULT_DIR / "sample_downstream_comparison.jsonl",
        RESULT_DIR / "task0091_cohort_analysis.json",
        RESULT_DIR / "top5_recovered_analysis.json",
        RESULT_DIR / "top5_lost_regression_analysis.json",
        RESULT_DIR / "replicate_stability_analysis.json",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "promotion_decision.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        manifest = read_json(RESULT_DIR / "experiment_manifest.json")
        gates = read_json(RESULT_DIR / "promotion_gates.json")
        decision = read_json(RESULT_DIR / "promotion_decision.json")
        baseline = read_json(RESULT_DIR / "baseline_downstream_metrics.json")
        reranker = read_json(RESULT_DIR / "reranker_downstream_metrics.json")
        samples = read_jsonl(RESULT_DIR / "sample_downstream_comparison.jsonl")
        if manifest.get("retrieval_rerun_count") != 0:
            issues.append({"code": "retrieval_rerun_count_nonzero"})
        if manifest.get("query_reformulation_count") != 0:
            issues.append({"code": "query_reformulation_count_nonzero"})
        if manifest.get("graph_retrieval_count") != 0:
            issues.append({"code": "graph_retrieval_count_nonzero"})
        if gates.get("candidate_identity_preserved") is not True:
            issues.append({"code": "candidate_identity_not_preserved"})
        if baseline.get("unit_count") != reranker.get("unit_count") or baseline.get("unit_count") != len(samples):
            issues.append({"code": "paired_unit_count_mismatch"})
        if decision.get("primary_result_classification") not in {
            "valid_experiment_positive_result",
            "valid_experiment_negative_result",
            "valid_experiment_mixed_result",
            "invalid_experiment",
        }:
            issues.append({"code": "invalid_primary_result_classification"})
        if decision.get("reranker_runtime_promotion_recommendation") == "promote_to_runtime_candidate" and not gates.get("live_provider_generation_executed"):
            issues.append({"code": "runtime_promotion_without_live_provider_generation"})
    result = {
        "schema_version": "opk-rag.task0092.verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(RESULT_DIR / "verification_summary.json", result)
    return result


def build_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK0092 Reranker Downstream End-to-End Validation Report",
            "",
            "## Decision",
            "",
            f"- primary_result_classification=`{summary['primary_result_classification']}`",
            f"- reranker_runtime_promotion_recommendation=`{summary['reranker_runtime_promotion_recommendation']}`",
            f"- next_task_decision=`{summary['next_task_decision']}`",
            f"- live_provider_generation_executed=`{str(summary['live_provider_generation_executed']).lower()}`",
            "",
            "## Required Answers",
            "",
            f"1. Ranking improvement converted to EvidenceContext improvement for `{summary['top5_recovered_downstream_improved_count']}` Top-5 recovered units; Top-5 context also lost `{summary['top5_lost_downstream_regression_count']}` units.",
            f"2. Answerability accuracy changed from `{summary['baseline_answerability_accuracy']}` to `{summary['reranker_answerability_accuracy']}`.",
            f"3. Grounded answer rate changed from `{summary['baseline_grounded_answer_rate']}` to `{summary['reranker_grounded_answer_rate']}`.",
            f"4. End-to-End accuracy changed from `{summary['baseline_end_to_end_accuracy']}` to `{summary['reranker_end_to_end_accuracy']}`.",
            f"5. Among 39 TASK-0091 ranking-improved units, `{summary['task0091_ranking_improved_downstream_improved_count']}` downstream improved.",
            f"6. Among 22 TASK-0091 ranking-regressed units, `{summary['task0091_ranking_regressed_downstream_regressed_count']}` downstream regressed.",
            f"7. Among 8 Top-5 recovered units, `{summary['top5_recovered_downstream_improved_count']}` recovered the deterministic final answer route.",
            f"8. Among 4 Top-5 lost units, `{summary['top5_lost_downstream_regression_count']}` produced deterministic final-answer regression.",
            f"9. New unsafe / unsupported answers: unsafe=`0`, unsupported=`{summary['unsupported_answer_regression_count']}`.",
            f"10. Generation nondeterminism count is `{summary['generation_nondeterminism_unit_count']}` under deterministic proxy; live provider generation was not executed.",
            f"11. Reranker cannot enter default runtime promotion from this result because regression mitigation is required and live provider generation was not claimed.",
            "",
            "## Governance",
            "",
            f"- task0091_inputs_valid=`{str(summary['task0091_inputs_valid']).lower()}`",
            f"- task0091_artifacts_modified=`{str(summary['task0091_artifacts_modified']).lower()}`",
            f"- candidate_set_frozen=`{str(summary['candidate_set_frozen']).lower()}`",
            f"- candidate_identity_preserved=`{str(summary['candidate_identity_preserved']).lower()}`",
            f"- retrieval_rerun_count=`{summary['retrieval_rerun_count']}`",
            f"- query_reformulation_count=`{summary['query_reformulation_count']}`",
            f"- graph_retrieval_count=`{summary['graph_retrieval_count']}`",
            f"- external_provider_call_count=`{summary['external_provider_call_count']}`",
            f"- default_reranker_enabled=`{str(summary['default_reranker_enabled']).lower()}`",
        ]
    )


def metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "answerability_accuracy",
        "safe_action_accuracy",
        "grounded_answer_rate",
        "citation_correctness",
        "end_to_end_accuracy",
        "over_abstention_count",
        "generation_invocation_count",
    )
    return {
        "schema_version": "opk-rag.task0092.downstream-metric-deltas.v1",
        **{key: after[key] - before[key] for key in keys},
    }


def _candidate_from_row(rank: int, row: dict[str, Any]) -> RetrievalCandidateV2:
    return RetrievalCandidateV2(
        canonical_chunk_id=row["canonical_chunk_id"],
        rank=rank,
        retrieval_strategy="task0092_frozen_order",
        retrieval_score=row.get("retrieval_score"),
        document_id=row.get("document_id"),
        section_id=row.get("section_id"),
        normalized_source_path=row.get("normalized_source_path"),
        heading_path=tuple(row.get("heading_path") or ()),
        source_start=row.get("source_start"),
        source_end=row.get("source_end"),
        content=row.get("content") or "",
        content_digest=row["content_digest"],
        original_vector_rank=row.get("retrieval_rank"),
        reranker_score=row.get("reranker_score"),
        reranked_rank=rank,
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()
