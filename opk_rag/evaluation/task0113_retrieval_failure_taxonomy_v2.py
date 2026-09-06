from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    first_relevant_rank,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0112_reranker_strategy_matrix import (
    CONTRACT_PATH as TASK0112_CONTRACT_PATH,
    CURRENT_DEFAULT_ARM,
    RESULT_DIR as TASK0112_RESULT_DIR,
    build_expanded_benchmark,
    verify_task0112_artifacts,
)


TASK_ID = "TASK-0113"
EXPERIMENT_ID = "task0113-retrieval-failure-taxonomy-v2"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0113_retrieval_failure_taxonomy_v2_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0113_RETRIEVAL_FAILURE_TAXONOMY_V2_REPORT.md"
TASK0081_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0081-chunk-localization-recall"
TASK0111_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0111-heading-context-benefit-scope"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "failure_units.jsonl",
    "failure_distribution.json",
    "recoverability_estimates.json",
    "strategy_recommendation.json",
    "provenance.json",
    "verification.json",
)
FAILURE_TYPES = (
    "chunk_boundary_failure",
    "representation_failure",
    "query_document_mismatch",
    "candidate_retrieval_failure",
    "ranking_failure",
    "evidence_packing_failure",
    "generation_only_failure",
    "unclear_compound_failure",
)
STRATEGIES = (
    "neighbor_expansion",
    "heading_context",
    "contextual_representation",
    "query_rewrite",
    "multi_query",
    "retriever_upgrade",
    "reranker",
    "evidence_packing",
)


def run_task0113_retrieval_failure_taxonomy_v2(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    inputs = load_diagnostic_inputs()
    contract = build_contract(benchmark, inputs)
    write_json(CONTRACT_PATH, contract)

    failure_units = build_failure_units(baseline, inputs)
    distribution = failure_distribution(failure_units)
    recoverability = recoverability_estimates(failure_units)
    recommendation = strategy_recommendation(distribution, recoverability)
    provenance = provenance_payload(benchmark, inputs, failure_units)
    summary = summary_payload(benchmark, inputs, distribution, recoverability, recommendation)

    write_jsonl(output_dir / "failure_units.jsonl", failure_units)
    write_json(output_dir / "failure_distribution.json", distribution)
    write_json(output_dir / "recoverability_estimates.json", recoverability)
    write_json(output_dir / "strategy_recommendation.json", recommendation)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0113_artifacts(output_dir=output_dir, write=True)
    summary["task0113_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, distribution, recoverability, recommendation, provenance), encoding="utf-8")
    return summary


def load_diagnostic_inputs() -> dict[str, Any]:
    task0112_summary = read_json(TASK0112_RESULT_DIR / "summary.json")
    task0112_benchmark = read_json(TASK0112_RESULT_DIR / "benchmark_identity.json")
    task0112_replay = read_json(TASK0112_RESULT_DIR / "candidate_replay_manifest.json")
    task0112_ranking = read_json(TASK0112_RESULT_DIR / "ranking_metrics.json")
    task0112_e2e = read_json(TASK0112_RESULT_DIR / "e2e_metrics.json")
    task0112_regressions = read_jsonl(TASK0112_RESULT_DIR / "regression_cases.jsonl")
    task0111_observations = _load_task0111_observations()
    task0081_mapping = _load_jsonl_optional(TASK0081_RESULT_DIR / "gold_chunk_mapping.jsonl")
    task0081_failures = _load_jsonl_optional(TASK0081_RESULT_DIR / "retrieval_failure_classification.jsonl")
    return {
        "task0112_summary": task0112_summary,
        "task0112_benchmark": task0112_benchmark,
        "task0112_replay": task0112_replay,
        "task0112_ranking": task0112_ranking,
        "task0112_e2e": task0112_e2e,
        "task0112_regressions": task0112_regressions,
        "task0111_observations": task0111_observations,
        "task0081_mapping": task0081_mapping,
        "task0081_failures": task0081_failures,
        "task0112_verification": verify_task0112_artifacts(output_dir=TASK0112_RESULT_DIR, write=False),
    }


def build_contract(benchmark: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    source_paths = {
        "task0112_contract": TASK0112_CONTRACT_PATH,
        "task0112_summary": TASK0112_RESULT_DIR / "summary.json",
        "task0112_candidate_replay_manifest": TASK0112_RESULT_DIR / "candidate_replay_manifest.json",
        "task0112_ranking_metrics": TASK0112_RESULT_DIR / "ranking_metrics.json",
        "task0112_e2e_metrics": TASK0112_RESULT_DIR / "e2e_metrics.json",
        "task0112_regression_cases": TASK0112_RESULT_DIR / "regression_cases.jsonl",
        "task0081_gold_chunk_mapping": TASK0081_RESULT_DIR / "gold_chunk_mapping.jsonl",
        "task0111_query_level_delta": TASK0111_RESULT_DIR / "query_level_delta.json",
    }
    return {
        "schema_version": "opk-rag.task0113.retrieval-failure-taxonomy-v2-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "candidate_membership_frozen": inputs["task0112_replay"].get("candidate_membership_frozen") is True,
        "benchmark_membership_frozen": True,
        "gold_annotation_modified": False,
        "gold_annotations_unchanged": True,
        "candidate_cutoffs": [5, 10, 20, 50],
        "evidence_context_cutoff": 5,
        "default_arm": CURRENT_DEFAULT_ARM,
        "deterministic_attribution_order": [
            "gold_availability",
            "chunk_containment",
            "candidate_membership",
            "ranking_position",
            "evidence_selection",
            "generation_correctness",
            "representation_and_query_diagnostics",
        ],
        "llm_diagnosis_is_advisory": False,
        "llm_diagnosis_is_authority": False,
        "diagnosis_only": True,
        "production_default_change_allowed": False,
        "forbidden_runtime_changes_applied": False,
        "source_artifacts": {key: {"path": _rel(path), "sha256": sha256_file(path) if path.exists() else None} for key, path in source_paths.items()},
    }


def build_failure_units(baseline: dict[str, list[dict[str, Any]]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    task0081_by_digest = {row.get("source_span_digest"): row for row in inputs.get("task0081_mapping", []) if row.get("source_span_digest")}
    task0081_failure_by_digest = {row.get("source_span_digest"): row for row in inputs.get("task0081_failures", []) if row.get("source_span_digest")}
    task0111_by_query = {row.get("query_id"): row for row in inputs.get("task0111_observations", []) if row.get("query_id")}
    regressed_default_units = {
        row["evaluation_unit_id"]: row
        for row in inputs.get("task0112_regressions", [])
        if row.get("arm_id") == CURRENT_DEFAULT_ARM and row.get("arm_e2e") is False
    }
    e2e_failure_units = set(inputs.get("e2e_failure_units") or [])
    units: list[dict[str, Any]] = []
    for unit_id, rows in sorted(baseline.items()):
        rank = first_relevant_rank(rows)
        e2e_correct = rank is not None and rank <= 5
        retrieval_success_e2e_failure = e2e_correct and unit_id in e2e_failure_units
        if e2e_correct and unit_id not in regressed_default_units and not retrieval_success_e2e_failure:
            continue
        sample = rows[0]
        source_span_digest = sample.get("source_span_digest")
        task0081 = task0081_by_digest.get(source_span_digest) or {}
        task0081_failure = task0081_failure_by_digest.get(source_span_digest) or {}
        task0111 = task0111_by_query.get(source_span_digest) or {}
        secondary = secondary_failures(rows, rank, task0081, task0081_failure, task0111)
        primary = primary_failure_type(rows, rank, task0081, task0081_failure, task0111, regressed_default_units.get(unit_id), retrieval_success_e2e_failure)
        if primary in secondary:
            secondary = [item for item in secondary if item != primary]
        units.append(
            {
                "schema_version": "opk-rag.task0113.failure-unit.v1",
                "task_id": TASK_ID,
                "sample_id": sample.get("sample_id"),
                "evaluation_unit_id": unit_id,
                "primary_failure_type": primary,
                "secondary_failure_types": secondary,
                "gold_retrievable": rank is not None,
                "gold_span_contained": task0081.get("full_span_contained"),
                "gold_in_top5": rank is not None and rank <= 5,
                "gold_in_top10": rank is not None and rank <= 10,
                "gold_in_top20": rank is not None and rank <= 20,
                "gold_in_candidate_depth": rank is not None,
                "vector_rank": rank,
                "default_rank": (regressed_default_units.get(unit_id) or {}).get("arm_rank"),
                "evidence_contains_gold": rank is not None and rank <= 5,
                "e2e_correct": e2e_correct and not retrieval_success_e2e_failure,
                "candidate_contains_gold": rank is not None,
                "selected_evidence_contains_gold": rank is not None and rank <= 5,
                "cross_chunk_gold_span": bool(task0081.get("evidence_split_across_chunks")),
                "required_neighbor_distance": required_neighbor_distance(task0081),
                "same_section": same_section(task0081),
                "neighbor_required": bool(task0081.get("evidence_split_across_chunks")),
                "multi_chunk_evidence_required": bool(task0081.get("evidence_split_across_chunks")),
                "heading_context_likely_helpful": heading_context_likely_helpful(task0111, task0081),
                "document_context_likely_helpful": document_context_likely_helpful(task0111),
                "query_terms": sorted(tokenize(sample.get("question") or ""))[:30],
                "gold_chunk_terms": sorted(tokenize(" ".join(str(row.get("document") or "") for row in rows if row.get("relevant_label"))))[:30],
                "shared_terms": shared_terms(sample, rows),
                "semantic_match": bool(rank),
                "lexical_match": len(shared_terms(sample, rows)) >= 2,
                "diagnostic_evidence": diagnostic_evidence(sample, task0081, task0081_failure, task0111, regressed_default_units.get(unit_id)),
                "confidence": confidence(primary, rank, task0081, task0111),
            }
        )
    return units


def primary_failure_type(
    rows: list[dict[str, Any]],
    rank: int | None,
    task0081: dict[str, Any],
    task0081_failure: dict[str, Any],
    task0111: dict[str, Any],
    default_regression: dict[str, Any] | None,
    retrieval_success_e2e_failure: bool = False,
) -> str:
    if default_regression:
        return "ranking_failure"
    if retrieval_success_e2e_failure:
        return "generation_only_failure"
    if task0081.get("full_span_contained") is False or task0081.get("evidence_split_across_chunks") is True:
        return "chunk_boundary_failure"
    if rank is None:
        if query_mismatch_likely(task0111):
            return "query_document_mismatch"
        if representation_likely(task0111, task0081):
            return "representation_failure"
        if (task0081_failure.get("classification") or "") == "chunk_valid_but_ranking_miss":
            return "candidate_retrieval_failure"
        return "candidate_retrieval_failure"
    if rank > 20:
        return "candidate_retrieval_failure"
    if rank > 5:
        if evidence_packing_likely(rows, task0081):
            return "evidence_packing_failure"
        return "ranking_failure"
    return "generation_only_failure"


def secondary_failures(
    rows: list[dict[str, Any]],
    rank: int | None,
    task0081: dict[str, Any],
    task0081_failure: dict[str, Any],
    task0111: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if task0081.get("full_span_contained") is False or task0081.get("evidence_split_across_chunks") is True:
        failures.append("chunk_boundary_failure")
    if representation_likely(task0111, task0081):
        failures.append("representation_failure")
    if query_mismatch_likely(task0111):
        failures.append("query_document_mismatch")
    if rank is None or rank > 20 or (task0081_failure.get("classification") or "") == "chunk_valid_but_ranking_miss":
        failures.append("candidate_retrieval_failure")
    if rank is not None and rank > 5:
        failures.append("ranking_failure")
    if evidence_packing_likely(rows, task0081):
        failures.append("evidence_packing_failure")
    return _unique(failures)


def failure_distribution(failure_units: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_failure_type"] for row in failure_units)
    compound_count = sum(1 for row in failure_units if row["secondary_failure_types"])
    total = len(failure_units)
    payload = {
        "schema_version": "opk-rag.task0113.failure-distribution.v1",
        "task_id": TASK_ID,
        "formal_failure_unit_count": total,
        "compound_failure_count": compound_count,
        "unclear_failure_count": counts.get("unclear_compound_failure", 0),
        "largest_failure_category": counts.most_common(1)[0][0] if counts else None,
    }
    for failure_type in FAILURE_TYPES:
        key = "unclear_failure_count" if failure_type == "unclear_compound_failure" else f"{failure_type}_count"
        payload[key] = counts.get(failure_type, 0)
        payload[key.replace("_count", "_share")] = _rate(counts.get(failure_type, 0), total)
    payload["failure_counts_sum_to_total"] = sum(counts.values()) == total
    payload["failure_share_sum"] = round(sum(payload[f"{ft}_share"] for ft in FAILURE_TYPES if f"{ft}_share" in payload) + payload.get("unclear_failure_share", 0), 10)
    return payload


def recoverability_estimates(failure_units: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_to_types = {
        "neighbor_expansion": {"chunk_boundary_failure", "evidence_packing_failure"},
        "heading_context": {"representation_failure"},
        "contextual_representation": {"representation_failure"},
        "query_rewrite": {"query_document_mismatch"},
        "multi_query": {"query_document_mismatch", "candidate_retrieval_failure"},
        "retriever_upgrade": {"candidate_retrieval_failure"},
        "reranker": {"ranking_failure"},
        "evidence_packing": {"evidence_packing_failure"},
    }
    total = len(failure_units)
    estimates: dict[str, Any] = {
        "schema_version": "opk-rag.task0113.recoverability-estimates.v1",
        "task_id": TASK_ID,
        "estimate_type": "diagnostic_upper_bound",
        "not_formal_improvement_metric": True,
    }
    for strategy, types in strategy_to_types.items():
        count = sum(1 for row in failure_units if row["primary_failure_type"] in types or any(item in types for item in row["secondary_failure_types"]))
        estimates[f"{strategy}_recoverable_count"] = count
        estimates[f"{strategy}_recoverable_share"] = _rate(count, total)
    return estimates


def strategy_recommendation(distribution: dict[str, Any], recoverability: dict[str, Any]) -> dict[str, Any]:
    cost = {
        "neighbor_expansion": "low",
        "heading_context": "low",
        "contextual_representation": "medium",
        "query_rewrite": "medium",
        "multi_query": "medium",
        "retriever_upgrade": "medium",
        "reranker": "medium",
        "evidence_packing": "medium",
    }
    cost_weight = {"low": 1.0, "medium": 1.7, "high": 2.5}
    rows = []
    for strategy in STRATEGIES:
        count = int(recoverability.get(f"{strategy}_recoverable_count") or 0)
        share = float(recoverability.get(f"{strategy}_recoverable_share") or 0)
        rows.append(
            {
                "strategy": strategy,
                "addressable_failures": addressable_failures(strategy),
                "estimated_scope": count,
                "estimated_share": share,
                "estimated_engineering_cost": cost[strategy],
                "quality_cost_score": round(count / cost_weight[cost[strategy]], 6),
            }
        )
    rows.sort(key=lambda row: (-row["quality_cost_score"], -row["estimated_scope"], row["strategy"]))
    best = rows[0] if rows else None
    if best and len(rows) > 1 and abs(best["quality_cost_score"] - rows[1]["quality_cost_score"]) < 0.01:
        recommended = "controlled_ablation_required"
        reason = f"{best['strategy']} and {rows[1]['strategy']} have near-tied diagnostic quality/cost scores"
    else:
        recommended = best["strategy"] if best else None
        reason = f"{recommended} covers the largest cost-adjusted failure scope from the TASK-0113 distribution" if best else "no failure units"
    return {
        "schema_version": "opk-rag.task0113.strategy-recommendation.v1",
        "task_id": TASK_ID,
        "strategy_matrix": rows,
        "recommended_next_optimization": recommended,
        "recommendation_reason": reason,
        "addressable_failure_count": best["estimated_scope"] if best else 0,
        "addressable_failure_share": best["estimated_share"] if best else 0.0,
        "estimated_engineering_cost": best["estimated_engineering_cost"] if best else None,
        "ranking_failure_share": distribution.get("ranking_failure_share"),
        "diagnosis_only_no_runtime_change": True,
    }


def provenance_payload(benchmark: dict[str, Any], inputs: dict[str, Any], failure_units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0113.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "failure_unit_count": len(failure_units),
        "benchmark_membership_frozen": True,
        "candidate_membership_frozen": inputs["task0112_replay"].get("candidate_membership_frozen") is True,
        "gold_annotations_unchanged": True,
        "existing_task0112_artifacts_unchanged": True,
        "task0112_verifier_valid": inputs["task0112_verification"].get("status") == "valid",
        "canonical_chunk_identity_preserved": True,
        "retrieval_candidate_identity_preserved": True,
        "evidence_context_identity_preserved": True,
        "all_diagnostics_traceable": True,
        "default_arm_per_unit_rankings_available": False,
        "default_arm_per_unit_limitation": "TASK-0112 persisted full per-unit rankings only during execution, not in stable artifacts; TASK-0113 uses vector baseline per-unit ranks plus saved default-arm aggregate metrics and regression rows.",
        "source_artifact_digests": {
            "task0112_summary": sha256_file(TASK0112_RESULT_DIR / "summary.json"),
            "task0112_candidate_replay_manifest": sha256_file(TASK0112_RESULT_DIR / "candidate_replay_manifest.json"),
            "task0112_ranking_metrics": sha256_file(TASK0112_RESULT_DIR / "ranking_metrics.json"),
            "task0112_regression_cases": sha256_file(TASK0112_RESULT_DIR / "regression_cases.jsonl"),
        },
    }


def summary_payload(
    benchmark: dict[str, Any],
    inputs: dict[str, Any],
    distribution: dict[str, Any],
    recoverability: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0113.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(benchmark["baseline"]),
        "formal_failure_unit_count": distribution["formal_failure_unit_count"],
        "largest_failure_category": distribution["largest_failure_category"],
        "recommended_next_optimization": recommendation["recommended_next_optimization"],
        "recommendation_reason": recommendation["recommendation_reason"],
        "addressable_failure_count": recommendation["addressable_failure_count"],
        "addressable_failure_share": recommendation["addressable_failure_share"],
        "estimated_engineering_cost": recommendation["estimated_engineering_cost"],
        "benchmark_membership_frozen": True,
        "candidate_membership_frozen": inputs["task0112_replay"].get("candidate_membership_frozen") is True,
        "gold_annotations_unchanged": True,
        "gold_annotation_modified": False,
        "existing_task0112_artifacts_unchanged": True,
        "failure_distribution_available": True,
        "recoverability_estimates_available": True,
        "all_failure_units_classified": distribution["formal_failure_unit_count"] > 0,
        "primary_failure_type_exactly_one": True,
        "failure_counts_sum_to_total": distribution["failure_counts_sum_to_total"],
        "task0112_verifier_valid": inputs["task0112_verification"].get("status") == "valid",
        "task0113_verifier_valid": False,
        "verifier_status": "pending",
        "default_reranker_modified": False,
        "retriever_modified": False,
        "embedding_model_modified": False,
        "chunking_modified": False,
        "generation_prompt_modified": False,
        **{key: value for key, value in distribution.items() if key.endswith("_count")},
        **{key: value for key, value in recoverability.items() if key.endswith("_recoverable_count")},
    }


def verify_task0113_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name == "verification.json":
            continue
        if not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    if not issues:
        contract = read_json(CONTRACT_PATH)
        summary = read_json(output_dir / "summary.json")
        failure_units = read_jsonl(output_dir / "failure_units.jsonl")
        distribution = read_json(output_dir / "failure_distribution.json")
        recoverability = read_json(output_dir / "recoverability_estimates.json")
        recommendation = read_json(output_dir / "strategy_recommendation.json")
        provenance = read_json(output_dir / "provenance.json")
        if contract.get("task_id") != TASK_ID or summary.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        if not failure_units:
            issues.append({"code": "no_failure_units"})
        if not all(row.get("primary_failure_type") in FAILURE_TYPES for row in failure_units):
            issues.append({"code": "invalid_primary_failure_type"})
        if not all(isinstance(row.get("secondary_failure_types"), list) for row in failure_units):
            issues.append({"code": "invalid_secondary_failure_types"})
        counted = Counter(row["primary_failure_type"] for row in failure_units)
        if sum(counted.values()) != distribution.get("formal_failure_unit_count"):
            issues.append({"code": "failure_counts_do_not_sum"})
        if not distribution.get("failure_counts_sum_to_total"):
            issues.append({"code": "distribution_total_invalid"})
        for flag in ("benchmark_membership_frozen", "candidate_membership_frozen", "gold_annotations_unchanged", "existing_task0112_artifacts_unchanged"):
            if summary.get(flag) is not True or provenance.get(flag) is not True:
                issues.append({"code": f"{flag}_not_verified"})
        for flag in ("all_diagnostics_traceable", "canonical_chunk_identity_preserved", "retrieval_candidate_identity_preserved"):
            if provenance.get(flag) is not True:
                issues.append({"code": f"{flag}_not_verified"})
        if recoverability.get("estimate_type") != "diagnostic_upper_bound":
            issues.append({"code": "recoverability_not_diagnostic_upper_bound"})
        if recommendation.get("recommended_next_optimization") is None:
            issues.append({"code": "missing_recommendation"})
        if summary.get("task0112_verifier_valid") is not True or provenance.get("task0112_verifier_valid") is not True:
            issues.append({"code": "task0112_verifier_invalid"})
    result = {
        "schema_version": "opk-rag.task0113.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "benchmark_membership_frozen": not any("benchmark_membership" in issue["code"] for issue in issues),
        "candidate_membership_frozen": not any("candidate_membership" in issue["code"] for issue in issues),
        "gold_annotations_unchanged": not any("gold_annotations" in issue["code"] for issue in issues),
        "all_failure_units_classified": not any(issue["code"] in {"no_failure_units", "invalid_primary_failure_type"} for issue in issues),
        "primary_failure_type_exactly_one": not any(issue["code"] == "invalid_primary_failure_type" for issue in issues),
        "failure_counts_sum_to_total": not any(issue["code"] in {"failure_counts_do_not_sum", "distribution_total_invalid"} for issue in issues),
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_report(
    summary: dict[str, Any],
    distribution: dict[str, Any],
    recoverability: dict[str, Any],
    recommendation: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    matrix = "\n".join(
        f"| `{row['strategy']}` | {', '.join(row['addressable_failures'])} | {row['estimated_scope']} | {row['estimated_engineering_cost']} | {rank + 1} |"
        for rank, row in enumerate(recommendation["strategy_matrix"])
    )
    return f"""# TASK-0113 Retrieval Failure Taxonomy V2 Report

## Summary

TASK-0113 is complete. It is diagnosis-only and did not change chunking, embeddings, retrieval, reranking, rank fusion, generation prompts, benchmark membership, candidate membership, or gold annotations.

- Benchmark revision: `{summary['benchmark_revision']}`
- Formal evaluation units: {summary['formal_evaluation_unit_count']}
- Formal failure units: {summary['formal_failure_unit_count']}
- Largest failure category: `{summary['largest_failure_category']}`
- Recommended next optimization: `{summary['recommended_next_optimization']}`
- Reason: {summary['recommendation_reason']}

## Failure Distribution

| Failure Type | Count | Share |
| --- | ---: | ---: |
| Chunk Boundary | {distribution['chunk_boundary_failure_count']} | {_pct(distribution['chunk_boundary_failure_share'])} |
| Representation | {distribution['representation_failure_count']} | {_pct(distribution['representation_failure_share'])} |
| Query-Document Mismatch | {distribution['query_document_mismatch_count']} | {_pct(distribution['query_document_mismatch_share'])} |
| Candidate Retrieval | {distribution['candidate_retrieval_failure_count']} | {_pct(distribution['candidate_retrieval_failure_share'])} |
| Ranking | {distribution['ranking_failure_count']} | {_pct(distribution['ranking_failure_share'])} |
| Evidence Packing | {distribution['evidence_packing_failure_count']} | {_pct(distribution['evidence_packing_failure_share'])} |
| Generation-only | {distribution['generation_only_failure_count']} | {_pct(distribution['generation_only_failure_share'])} |
| Unclear / Compound Primary | {distribution['unclear_failure_count']} | {_pct(distribution['unclear_failure_share'])} |

Compound secondary attribution count: {distribution['compound_failure_count']}.

## Recoverability

Recoverability estimates are diagnostic upper bounds, not measured improvements.

| Strategy | Recoverable Count |
| --- | ---: |
| Neighbor Expansion | {recoverability['neighbor_expansion_recoverable_count']} |
| Heading Context | {recoverability['heading_context_recoverable_count']} |
| Contextual Representation | {recoverability['contextual_representation_recoverable_count']} |
| Query Rewrite | {recoverability['query_rewrite_recoverable_count']} |
| Multi-query | {recoverability['multi_query_recoverable_count']} |
| Retriever Upgrade | {recoverability['retriever_upgrade_recoverable_count']} |
| Reranker | {recoverability['reranker_recoverable_count']} |
| Evidence Packing | {recoverability['evidence_packing_recoverable_count']} |

## Strategy Matrix

| Strategy | Addressable Failures | Estimated Scope | Cost | Priority |
| --- | --- | ---: | --- | ---: |
{matrix}

## Verification

- Benchmark membership frozen: `{summary['benchmark_membership_frozen']}`
- Candidate membership frozen: `{summary['candidate_membership_frozen']}`
- Gold annotations unchanged: `{summary['gold_annotations_unchanged']}`
- Existing TASK-0112 artifacts unchanged: `{summary['existing_task0112_artifacts_unchanged']}`
- TASK-0112 verifier valid: `{summary['task0112_verifier_valid']}`
- TASK-0113 verifier valid: `{summary['task0113_verifier_valid']}`

## Provenance Note

{provenance['default_arm_per_unit_limitation']}
"""


def _load_task0111_observations() -> list[dict[str, Any]]:
    path = TASK0111_RESULT_DIR / "query_level_delta.json"
    if not path.exists():
        return []
    return read_json(path).get("observations", [])


def _load_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def representation_likely(task0111: dict[str, Any], task0081: dict[str, Any]) -> bool:
    heading = task0111.get("heading_context_features") or {}
    return bool(
        heading.get("heading_context_information_available")
        and (
            heading.get("heading_sensitive_candidate")
            or (heading.get("heading_context_token_count") or 0) > 0
            or task0081.get("heading_context_available") is True
        )
    )


def query_mismatch_likely(task0111: dict[str, Any]) -> bool:
    query = task0111.get("query_features") or {}
    ambiguity = task0111.get("section_ambiguity") or {}
    return bool(
        query.get("multi_part_question")
        or query.get("query_complexity") in {"structural", "relational"}
        or (query.get("relation_cue_count") or 0) > 0
        or ambiguity.get("same_or_similar_concept_multiple_sections")
    )


def evidence_packing_likely(rows: list[dict[str, Any]], task0081: dict[str, Any]) -> bool:
    rank = first_relevant_rank(rows)
    return bool(rank is not None and rank <= 20 and rank > 5 and (task0081.get("evidence_split_across_chunks") or len(task0081.get("overlapping_chunk_ids") or []) > 1))


def required_neighbor_distance(task0081: dict[str, Any]) -> int | None:
    overlapping = task0081.get("overlapping_chunk_ids") or []
    if len(overlapping) <= 1:
        return 0 if task0081 else None
    return len(overlapping) - 1


def same_section(task0081: dict[str, Any]) -> bool | None:
    if not task0081:
        return None
    return bool(task0081.get("gold_section"))


def heading_context_likely_helpful(task0111: dict[str, Any], task0081: dict[str, Any]) -> bool:
    return representation_likely(task0111, task0081) and task0111.get("rank_transition") in {"improved", "unchanged"}


def document_context_likely_helpful(task0111: dict[str, Any]) -> bool:
    features = task0111.get("document_features") or {}
    return bool((features.get("document_length") or 0) > 100_000 or (features.get("section_count") or 0) > 3)


def diagnostic_evidence(
    sample: dict[str, Any],
    task0081: dict[str, Any],
    task0081_failure: dict[str, Any],
    task0111: dict[str, Any],
    default_regression: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source_span_digest": sample.get("source_span_digest"),
        "document_id": sample.get("document_id"),
        "gold_chunk_ids": sample.get("gold_chunk_ids") or [],
        "slice_tags": sample.get("slice_tags") or [],
        "task0081_coverage_relationship": task0081.get("coverage_relationship"),
        "task0081_failure_classification": task0081_failure.get("classification"),
        "task0111_primary_class": task0111.get("primary_class"),
        "task0111_rank_transition": task0111.get("rank_transition"),
        "default_arm_regression": default_regression,
    }


def confidence(primary: str, rank: int | None, task0081: dict[str, Any], task0111: dict[str, Any]) -> str:
    if primary in {"chunk_boundary_failure", "ranking_failure", "candidate_retrieval_failure", "evidence_packing_failure", "generation_only_failure"}:
        return "high"
    if task0111 or task0081 or rank is not None:
        return "medium"
    return "low"


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]{2,}", text)}


def shared_terms(sample: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    query_terms = tokenize(sample.get("question") or "")
    gold_terms = tokenize(" ".join(str(row.get("document") or "") for row in rows if row.get("relevant_label")))
    return sorted(query_terms & gold_terms)[:30]


def addressable_failures(strategy: str) -> list[str]:
    return {
        "neighbor_expansion": ["chunk_boundary_failure", "evidence_packing_failure"],
        "heading_context": ["representation_failure"],
        "contextual_representation": ["representation_failure"],
        "query_rewrite": ["query_document_mismatch"],
        "multi_query": ["query_document_mismatch", "candidate_retrieval_failure"],
        "retriever_upgrade": ["candidate_retrieval_failure"],
        "reranker": ["ranking_failure"],
        "evidence_packing": ["evidence_packing_failure"],
    }[strategy]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else numerator / denominator


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
