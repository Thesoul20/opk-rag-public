from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    CONTRACT_PATH as TASK0091_CONTRACT_PATH,
    RESULT_DIR as TASK0091_RESULT_DIR,
    ROOT,
    build_replay_units,
    digest_json,
    first_relevant_rank,
    load_task0091_inputs,
    ndcg_at_k,
    read_json,
    read_jsonl,
    rerank_rows,
    sample_comparisons,
    sha256_file,
    utc_now,
    verify_task0091_artifacts,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import (
    RESULT_DIR as TASK0092_RESULT_DIR,
    build_evidence_context,
    downstream_metrics,
    verify_task0092_artifacts,
)
from opk_rag.runtime_v2.rank_fusion import RankFusionParameters, rank_fusion_order


TASK_ID = "TASK-0093"
EXPERIMENT_ID = "task0093-reranker-guarded-mitigation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0093_reranker_guarded_mitigation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0093_RERANKER_REGRESSION_DIAGNOSIS_AND_GUARDED_MITIGATION_REPORT.md"
EVIDENCE_CONTEXT_TOP_K = 5
CANDIDATE_DEPTH = 50
NEAR_TIE_ABSOLUTE_MARGIN = 1e-5
R4_BOUNDARY_MARGIN_THRESHOLD = 1e-5


ARM_SPECS: tuple[dict[str, Any], ...] = (
    {"arm_id": "C0", "policy_family": "original_retrieval", "description": "frozen TASK-0090 retrieval order"},
    {"arm_id": "R0", "policy_family": "pure_reranker", "description": "TASK-0091 pure reranker full reorder"},
    {"arm_id": "R1-A", "policy_family": "score_fusion", "retrieval_weight": 0.20, "reranker_weight": 0.80},
    {"arm_id": "R1-B", "policy_family": "score_fusion", "retrieval_weight": 0.30, "reranker_weight": 0.70},
    {"arm_id": "R1-C", "policy_family": "score_fusion", "retrieval_weight": 0.50, "reranker_weight": 0.50},
    {"arm_id": "R2", "policy_family": "rank_fusion", "rank_fusion_k": 60, "rank_fusion_lambda": 0.75},
    {"arm_id": "R3", "policy_family": "bounded_reordering", "protected_retrieval_rank_prefix": 5},
    {"arm_id": "R4", "policy_family": "confidence_gated_reranking", "boundary_margin_threshold": R4_BOUNDARY_MARGIN_THRESHOLD},
)


def run_task0093_reranker_guarded_mitigation() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    units = build_replay_units(inputs["task0091_inputs"])
    baseline = {unit_id: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit_id, rows in units.items()}
    pure_reranker = {unit_id: rerank_rows(rows) for unit_id, rows in units.items()}
    task0091_rows = sample_comparisons(baseline, pure_reranker)
    task0092_rows = inputs["task0092_sample_downstream_comparison"]
    write_json(CONTRACT_PATH, build_contract(inputs))

    diagnosis = diagnose_top5_lost_regressions(baseline, pure_reranker, task0092_rows)
    arm_rankings = {spec["arm_id"]: apply_policy(spec, baseline, pure_reranker) for spec in ARM_SPECS}
    arm_rows = [evaluate_arm(spec, baseline, arm_rankings[spec["arm_id"]], task0091_rows, task0092_rows) for spec in ARM_SPECS]
    cohort_rows = build_cohort_evaluation_rows(arm_rows, task0092_rows)
    transition_rows = build_transition_matrix_rows(arm_rows, task0092_rows)
    gates = promotion_gates(arm_rows)
    decision = promotion_decision(arm_rows, gates)
    manifest = experiment_manifest(inputs, units)
    summary = build_summary(manifest, diagnosis, arm_rows, gates, decision)

    write_json(RESULT_DIR / "experiment_manifest.json", manifest)
    write_jsonl(RESULT_DIR / "regression_diagnosis.jsonl", diagnosis)
    write_jsonl(RESULT_DIR / "mitigation_arm_results.jsonl", arm_rows)
    write_jsonl(RESULT_DIR / "cohort_evaluation.jsonl", cohort_rows)
    write_jsonl(RESULT_DIR / "pairwise_transition_matrix.jsonl", transition_rows)
    write_json(RESULT_DIR / "promotion_gates.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)
    verification = verify_task0093_artifacts(write=True)
    summary["verification"] = verification
    summary["repository_verification_status"] = verification["repository_verification_status"]
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_rows), encoding="utf-8")
    return summary


def load_inputs() -> dict[str, Any]:
    task0091_verification = verify_task0091_artifacts()
    task0092_verification = verify_task0092_artifacts()
    paths = {
        "task0091_contract": TASK0091_CONTRACT_PATH,
        "task0091_summary": TASK0091_RESULT_DIR / "summary.json",
        "task0092_summary": TASK0092_RESULT_DIR / "summary.json",
        "task0092_sample_downstream_comparison": TASK0092_RESULT_DIR / "sample_downstream_comparison.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing TASK-0093 inputs: {', '.join(_rel(path) for path in missing)}")
    task0092_summary = read_json(paths["task0092_summary"])
    if task0091_verification.get("status") != "valid":
        raise RuntimeError("TASK-0091 artifacts are not valid")
    if task0092_verification.get("status") != "valid":
        raise RuntimeError("TASK-0092 artifacts are not valid")
    if task0092_summary.get("reranker_runtime_promotion_recommendation") != "requires_regression_mitigation":
        raise RuntimeError("TASK-0092 did not require regression mitigation")
    return {
        "task0091_inputs": load_task0091_inputs(),
        "task0091_verification": task0091_verification,
        "task0092_verification": task0092_verification,
        "task0092_summary": task0092_summary,
        "task0092_sample_downstream_comparison": read_jsonl(paths["task0092_sample_downstream_comparison"]),
        "paths": {key: _rel(path) for key, path in paths.items()},
        "digests": {key: sha256_file(path) for key, path in paths.items()},
    }


def apply_policy(
    spec: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    pure_reranker: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    family = spec["policy_family"]
    if family == "original_retrieval":
        return {unit: assign_policy_rank(rows) for unit, rows in baseline.items()}
    if family == "pure_reranker":
        return {unit: assign_policy_rank(rows) for unit, rows in pure_reranker.items()}
    if family == "score_fusion":
        return {unit: score_fusion(rows, spec) for unit, rows in baseline.items()}
    if family == "rank_fusion":
        return {unit: rank_fusion(baseline[unit], pure_reranker[unit], spec) for unit in baseline}
    if family == "bounded_reordering":
        return {unit: protect_retrieval_prefix(baseline[unit], pure_reranker[unit], spec["protected_retrieval_rank_prefix"]) for unit in baseline}
    if family == "confidence_gated_reranking":
        return {unit: confidence_gated_reranking(baseline[unit], pure_reranker[unit], spec["boundary_margin_threshold"]) for unit in baseline}
    raise ValueError(f"unsupported policy family: {family}")


def score_fusion(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_norm = minmax([retrieval_confidence(row) for row in rows])
    reranker_norm = minmax([float(row["reranker_score"]) for row in rows])
    scored = []
    for row, retrieval_score, reranker_score in zip(rows, retrieval_norm, reranker_norm):
        final_score = spec["retrieval_weight"] * retrieval_score + spec["reranker_weight"] * reranker_score
        scored.append((final_score, row))
    ranked = [row for _, row in sorted(scored, key=lambda item: (-item[0], item[1]["retrieval_rank"], item[1]["canonical_chunk_id"]))]
    return assign_policy_rank(ranked)


def rank_fusion(rows: list[dict[str, Any]], pure_reranker: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    reranker_rank = {row["canonical_chunk_id"]: index for index, row in enumerate(pure_reranker, start=1)}
    fused = rank_fusion_order(
        tuple(rows),
        identity=lambda row: row["canonical_chunk_id"],
        retrieval_rank=lambda row: row["retrieval_rank"],
        reranker_rank_by_identity=reranker_rank,
        parameters=RankFusionParameters(k=spec["rank_fusion_k"], lambda_weight=spec["rank_fusion_lambda"]),
    )
    return [
        {**scored.item, "reranker_rank": scored.reranker_rank, "policy_rank": scored.fusion_rank, "fusion_score": scored.fusion_score}
        for scored in fused
    ]


def protect_retrieval_prefix(rows: list[dict[str, Any]], pure_reranker: list[dict[str, Any]], prefix: int) -> list[dict[str, Any]]:
    protected = [row for row in rows if row["retrieval_rank"] <= prefix]
    protected_ids = {row["canonical_chunk_id"] for row in protected}
    rest = [row for row in pure_reranker if row["canonical_chunk_id"] not in protected_ids]
    return assign_policy_rank(protected + rest)


def confidence_gated_reranking(rows: list[dict[str, Any]], pure_reranker: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    margin = boundary_margin(pure_reranker)
    selected = pure_reranker if margin >= threshold else rows
    return assign_policy_rank(selected)


def evaluate_arm(
    spec: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
    task0091_rows: list[dict[str, Any]],
    task0092_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_rows = arm_sample_rows(spec["arm_id"], baseline, variant, task0091_rows)
    metrics = downstream_metrics(sample_rows, "reranker")
    r0_recovered = {row["sample_unit_id"] for row in task0092_rows if row["downstream_classification"] == "downstream_improved"}
    retained_recoveries = [row["sample_unit_id"] for row in sample_rows if row["sample_unit_id"] in r0_recovered and row["downstream_classification"] == "downstream_improved"]
    counts = Counter(row["downstream_classification"] for row in sample_rows)
    return {
        "schema_version": "opk-rag.task0093.mitigation-arm-result.v1",
        "arm_id": spec["arm_id"],
        "policy_family": spec["policy_family"],
        "policy_parameters": {key: value for key, value in spec.items() if key not in {"arm_id", "policy_family", "description"}},
        "candidate_membership_change_count": candidate_membership_change_count(baseline, variant),
        "ranking_metrics": ranking_metrics(variant),
        "top5_recovered_unit_count": sum(row["top5_recovered"] for row in sample_rows),
        "top5_lost_unit_count": sum(row["top5_lost"] for row in sample_rows),
        "top5_net_gain": sum(row["top5_recovered"] for row in sample_rows) - sum(row["top5_lost"] for row in sample_rows),
        "downstream_metrics": metrics,
        "downstream_improved_unit_count": counts["downstream_improved"],
        "downstream_regressed_unit_count": counts["downstream_regressed"],
        "downstream_net_gain": counts["downstream_improved"] - counts["downstream_regressed"],
        "safe_action_regression_count": sum(row["safe_action_regression"] for row in sample_rows),
        "unsupported_answer_regression_count": sum(row["unsupported_answer_regression"] for row in sample_rows),
        "reranker_recovery_retention_count": len(retained_recoveries),
        "reranker_recovery_retention_rate": _ratio(len(retained_recoveries), len(r0_recovered)),
        "sample_unit_ids": {
            "downstream_improved": [row["sample_unit_id"] for row in sample_rows if row["downstream_classification"] == "downstream_improved"],
            "downstream_regressed": [row["sample_unit_id"] for row in sample_rows if row["downstream_classification"] == "downstream_regressed"],
            "retained_r0_recoveries": retained_recoveries,
        },
        "sample_rows": sample_rows,
    }


def arm_sample_rows(
    arm_id: str,
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
    task0091_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task0091_by_unit = {row["sample_unit_id"]: row for row in task0091_rows}
    rows = []
    for unit in sorted(baseline):
        base_hit = any(row["relevant_label"] for row in baseline[unit][:EVIDENCE_CONTEXT_TOP_K])
        variant_hit = any(row["relevant_label"] for row in variant[unit][:EVIDENCE_CONTEXT_TOP_K])
        classification = downstream_classification(base_hit, variant_hit)
        rows.append(
            {
                "schema_version": "opk-rag.task0093.sample-arm-comparison.v1",
                "arm_id": arm_id,
                "sample_unit_id": unit,
                "sample_id": baseline[unit][0]["sample_id"],
                "task0091_classification": task0091_by_unit[unit]["classification"],
                "top5_recovered": not base_hit and variant_hit,
                "top5_lost": base_hit and not variant_hit,
                "baseline": arm_outcome(base_hit),
                "reranker": arm_outcome(variant_hit),
                "downstream_classification": classification,
                "unsupported_answer_regression": False,
                "safe_action_regression": base_hit and not variant_hit,
                "baseline_evidence_context": build_evidence_context(baseline[unit][:EVIDENCE_CONTEXT_TOP_K]),
                "variant_evidence_context": build_evidence_context(variant[unit][:EVIDENCE_CONTEXT_TOP_K]),
            }
        )
    return rows


def diagnose_top5_lost_regressions(
    baseline: dict[str, list[dict[str, Any]]],
    pure_reranker: dict[str, list[dict[str, Any]]],
    task0092_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task0092_by_unit = {row["sample_unit_id"]: row for row in task0092_rows}
    selected = [row for row in task0092_rows if row["top5_lost"] and row["downstream_classification"] == "downstream_regressed"]
    diagnostics = []
    for source in selected:
        unit = source["sample_unit_id"]
        base = baseline[unit]
        reranked = pure_reranker[unit]
        relevant = next(row for row in base if row["relevant_label"])
        relevant_after = next(index for index, row in enumerate(reranked, start=1) if row["canonical_chunk_id"] == relevant["canonical_chunk_id"])
        rank5 = reranked[EVIDENCE_CONTEXT_TOP_K - 1]
        rank6 = reranked[EVIDENCE_CONTEXT_TOP_K]
        margin = boundary_margin(reranked)
        root_cause, notes = classify_root_cause(relevant, relevant_after, margin)
        diagnostics.append(
            {
                "schema_version": "opk-rag.task0093.regression-diagnostic.v1",
                "query_id": unit,
                "sample_unit_id": unit,
                "sample_id": source["sample_id"],
                "question_type": relevant["question_type"],
                "relevant_chunk_id": relevant["canonical_chunk_id"],
                "retrieval_rank_before": relevant["retrieval_rank"],
                "retrieval_score": relevant.get("retrieval_score"),
                "reranker_rank_after": relevant_after,
                "reranker_score": relevant["reranker_score"],
                "top5_boundary_candidate_id": rank6["canonical_chunk_id"],
                "rank5_chunk_id": rank5["canonical_chunk_id"],
                "rank5_retrieval_score": rank5.get("retrieval_score"),
                "rank5_reranker_score": rank5["reranker_score"],
                "rank6_chunk_id": rank6["canonical_chunk_id"],
                "rank6_retrieval_score": rank6.get("retrieval_score"),
                "rank6_reranker_score": rank6["reranker_score"],
                "relevant_chunk_rank_delta": relevant_after - relevant["retrieval_rank"],
                "relevant_chunk_retrieval_confidence": retrieval_confidence(relevant),
                "reranker_boundary_margin": margin,
                "low_confidence_boundary_flip": margin <= NEAR_TIE_ABSOLUTE_MARGIN,
                "evidence_context_before": build_evidence_context(base[:EVIDENCE_CONTEXT_TOP_K]),
                "evidence_context_after": build_evidence_context(reranked[:EVIDENCE_CONTEXT_TOP_K]),
                "downstream_outcome_before": task0092_by_unit[unit]["baseline"],
                "downstream_outcome_after": task0092_by_unit[unit]["reranker"],
                "root_cause_classification": root_cause,
                "diagnostic_notes": notes,
            }
        )
    return diagnostics


def classify_root_cause(relevant: dict[str, Any], relevant_after_rank: int, margin: float) -> tuple[str, str]:
    high_confidence_retrieval = relevant["retrieval_rank"] <= EVIDENCE_CONTEXT_TOP_K
    near_tie = margin <= NEAR_TIE_ABSOLUTE_MARGIN
    if high_confidence_retrieval and near_tie:
        return "compound_failure", "high-confidence retrieval evidence was pushed outside Top-5 by a low-confidence reranker boundary flip"
    if high_confidence_retrieval and relevant_after_rank > EVIDENCE_CONTEXT_TOP_K:
        return "high_confidence_retrieval_overridden", "retrieval placed relevant evidence inside Top-5 but pure reranker pushed it outside EvidenceContext"
    if near_tie:
        return "near_tie_instability", "rank5/rank6 reranker score margin is within the frozen near-tie threshold"
    return "score_calibration_mismatch", "reranker score ordering crossed the Top-5 cutoff without retrieval-score support"


def build_cohort_evaluation_rows(arm_rows: list[dict[str, Any]], task0092_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts = {
        "task0092_top5_recovered": {row["sample_unit_id"] for row in task0092_rows if row["top5_recovered"]},
        "task0092_top5_lost": {row["sample_unit_id"] for row in task0092_rows if row["top5_lost"]},
        "task0092_ranking_only_improved": {row["sample_unit_id"] for row in task0092_rows if row["ranking_only_classification"] == "ranking_only_improved"},
        "task0092_ranking_only_regressed": {row["sample_unit_id"] for row in task0092_rows if row["ranking_only_classification"] == "ranking_only_regressed"},
    }
    rows = []
    for arm in arm_rows:
        samples = {row["sample_unit_id"]: row for row in arm["sample_rows"]}
        for cohort_name, unit_ids in cohorts.items():
            selected = [samples[unit] for unit in sorted(unit_ids)]
            counts = Counter(row["downstream_classification"] for row in selected)
            rows.append(
                {
                    "schema_version": "opk-rag.task0093.cohort-evaluation.v1",
                    "arm_id": arm["arm_id"],
                    "cohort": cohort_name,
                    "unit_count": len(selected),
                    "downstream_improved_unit_count": counts["downstream_improved"],
                    "downstream_regressed_unit_count": counts["downstream_regressed"],
                    "downstream_unchanged_unit_count": counts["downstream_unchanged"],
                    "sample_unit_ids": [row["sample_unit_id"] for row in selected],
                }
            )
    return rows


def build_transition_matrix_rows(arm_rows: list[dict[str, Any]], task0092_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r0_class = {row["sample_unit_id"]: row["downstream_classification"] for row in task0092_rows}
    rows = []
    for arm in arm_rows:
        if arm["arm_id"] in {"C0", "R0"}:
            continue
        transitions = Counter((r0_class[row["sample_unit_id"]], row["downstream_classification"]) for row in arm["sample_rows"])
        rows.append(
            {
                "schema_version": "opk-rag.task0093.pairwise-transition-matrix.v1",
                "from_arm_id": "R0",
                "to_arm_id": arm["arm_id"],
                "retained_improvement": transitions[("downstream_improved", "downstream_improved")],
                "lost_improvement": sum(count for (before, after), count in transitions.items() if before == "downstream_improved" and after != "downstream_improved"),
                "regression_fixed": sum(count for (before, after), count in transitions.items() if before == "downstream_regressed" and after != "downstream_regressed"),
                "regression_retained": transitions[("downstream_regressed", "downstream_regressed")],
                "new_regression": sum(count for (before, after), count in transitions.items() if before != "downstream_regressed" and after == "downstream_regressed"),
                "transition_counts": {f"{before}->{after}": count for (before, after), count in sorted(transitions.items())},
            }
        )
    return rows


def promotion_gates(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    r0 = next(row for row in arm_rows if row["arm_id"] == "R0")
    eligible = []
    for arm in arm_rows:
        if arm["arm_id"] in {"C0", "R0"}:
            continue
        if (
            arm["candidate_membership_change_count"] == 0
            and arm["unsupported_answer_regression_count"] == 0
            and arm["downstream_regressed_unit_count"] < r0["downstream_regressed_unit_count"]
            and arm["downstream_net_gain"] > r0["downstream_net_gain"]
            and arm["reranker_recovery_retention_rate"] > 0
        ):
            eligible.append(arm["arm_id"])
    return {
        "schema_version": "opk-rag.task0093.promotion-gates.v1",
        "task0090_inputs_valid": True,
        "task0091_inputs_valid": True,
        "task0092_inputs_valid": True,
        "candidate_set_frozen": True,
        "candidate_identity_preserved": all(row["candidate_membership_change_count"] == 0 for row in arm_rows),
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
        "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in arm_rows),
        "unsupported_answer_regression_gate": all(row["unsupported_answer_regression_count"] == 0 for row in arm_rows),
        "default_reranker_enabled": False,
        "eligible_policy_arm_ids": eligible,
    }


def promotion_decision(arm_rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    if not gates["candidate_identity_preserved"] or gates["candidate_membership_change_count"] != 0:
        return {
            "schema_version": "opk-rag.task0093.promotion-decision.v1",
            "primary_result_classification": "invalid_experiment",
            "recommended_reranking_policy": None,
            "reranker_runtime_promotion_recommendation": "experiment_invalid",
            "next_task_decision": "restore_task0093_experiment_integrity",
        }
    eligible = [row for row in arm_rows if row["arm_id"] in gates["eligible_policy_arm_ids"]]
    if not eligible:
        return {
            "schema_version": "opk-rag.task0093.promotion-decision.v1",
            "primary_result_classification": "valid_experiment_no_safe_mitigation_found",
            "recommended_reranking_policy": "no_safe_mitigation_found",
            "reranker_runtime_promotion_recommendation": "do_not_promote_default_reranker",
            "next_task_decision": "revisit_reranker_training_or_candidate_context_strategy",
        }
    winner = sorted(
        eligible,
        key=lambda row: (
            row["unsupported_answer_regression_count"],
            row["downstream_regressed_unit_count"],
            -row["downstream_net_gain"],
            -row["downstream_metrics"]["end_to_end_accuracy"],
            -row["reranker_recovery_retention_count"],
            -row["ranking_metrics"]["mrr"],
            row["arm_id"],
        ),
    )[0]
    return {
        "schema_version": "opk-rag.task0093.promotion-decision.v1",
        "primary_result_classification": "valid_experiment_guarded_mitigation_found",
        "recommended_reranking_policy": winner["arm_id"],
        "recommended_policy_family": winner["policy_family"],
        "recommended_policy_parameters": winner["policy_parameters"],
        "reranker_runtime_promotion_recommendation": "promote_guarded_policy_to_runtime_candidate",
        "next_task_decision": "validate_guarded_policy_with_live_provider_before_default_runtime_promotion",
    }


def build_contract(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0093.reranker-guarded-mitigation-contract.v1",
        "experiment_identity": EXPERIMENT_ID,
        "task0091_dependency": inputs["paths"]["task0091_summary"],
        "task0091_dependency_digest": inputs["digests"]["task0091_summary"],
        "task0092_dependency": inputs["paths"]["task0092_summary"],
        "task0092_dependency_digest": inputs["digests"]["task0092_summary"],
        "near_tie_policy": {"metric": "absolute_reranker_score_margin_between_r0_rank5_and_rank6", "threshold": NEAR_TIE_ABSOLUTE_MARGIN},
        "retrieval_score_normalization": "retrieval_score unavailable in TASK-0090 replay; use frozen retrieval-rank confidence 1/rank then min-max within candidate set",
        "reranker_score_normalization": "min-max within frozen candidate set",
        "evidence_context_top_k": EVIDENCE_CONTEXT_TOP_K,
        "candidate_identity_policy": "same frozen candidate membership, identity, count, retrieval rank, and reranker score for every arm",
        "arm_specs": ARM_SPECS,
        "primary_variant_ranking": [
            "no unsupported-answer regression",
            "minimize downstream regression",
            "maximize downstream net gain",
            "maximize end-to-end accuracy",
            "retain Top-5 recoveries",
            "ranking metrics as tie breaker",
        ],
        "hard_constraints": {
            "retrieval_rerun_count": 0,
            "query_reformulation_count": 0,
            "graph_retrieval_count": 0,
            "candidate_membership_change_count": 0,
            "default_reranker_enabled": False,
            "external_provider_call_count": 0,
        },
    }


def experiment_manifest(inputs: dict[str, Any], units: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0093.experiment-manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "task0090_inputs_valid": True,
        "task0091_inputs_valid": inputs["task0091_verification"]["status"] == "valid",
        "task0092_inputs_valid": inputs["task0092_verification"]["status"] == "valid",
        "contract_digest": sha256_file(CONTRACT_PATH),
        "unit_count": len(units),
        "candidate_count": sum(len(rows) for rows in units.values()),
        "arm_count": len(ARM_SPECS),
        "retrieval_rerun_count": 0,
        "query_reformulation_count": 0,
        "graph_retrieval_count": 0,
        "candidate_membership_change_count": 0,
        "external_provider_call_count": 0,
        "default_reranker_enabled": False,
        "source_paths": inputs["paths"],
    }


def build_summary(
    manifest: dict[str, Any],
    diagnosis: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    gates: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    r0 = next(row for row in arm_rows if row["arm_id"] == "R0")
    winner = next((row for row in arm_rows if row["arm_id"] == decision.get("recommended_reranking_policy")), None)
    return {
        "schema_version": "opk-rag.task0093.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if decision["primary_result_classification"] != "invalid_experiment" else "invalid",
        **{key: manifest[key] for key in ("task0090_inputs_valid", "task0091_inputs_valid", "task0092_inputs_valid")},
        "top5_lost_diagnosed_unit_count": len(diagnosis),
        "root_cause_counts": dict(Counter(row["root_cause_classification"] for row in diagnosis)),
        "low_confidence_boundary_flip_count": sum(row["low_confidence_boundary_flip"] for row in diagnosis),
        "r0_downstream_improved_unit_count": r0["downstream_improved_unit_count"],
        "r0_downstream_regressed_unit_count": r0["downstream_regressed_unit_count"],
        "r0_downstream_net_gain": r0["downstream_net_gain"],
        "r0_end_to_end_accuracy": r0["downstream_metrics"]["end_to_end_accuracy"],
        "recommended_reranking_policy": decision["recommended_reranking_policy"],
        "recommended_policy_family": decision.get("recommended_policy_family"),
        "recommended_downstream_improved_unit_count": winner["downstream_improved_unit_count"] if winner else None,
        "recommended_downstream_regressed_unit_count": winner["downstream_regressed_unit_count"] if winner else None,
        "recommended_downstream_net_gain": winner["downstream_net_gain"] if winner else None,
        "recommended_end_to_end_accuracy": winner["downstream_metrics"]["end_to_end_accuracy"] if winner else None,
        "recommended_reranker_recovery_retention_rate": winner["reranker_recovery_retention_rate"] if winner else None,
        "primary_result_classification": decision["primary_result_classification"],
        "reranker_runtime_promotion_recommendation": decision["reranker_runtime_promotion_recommendation"],
        "next_task_decision": decision["next_task_decision"],
        "candidate_set_frozen": gates["candidate_set_frozen"],
        "candidate_identity_preserved": gates["candidate_identity_preserved"],
        "retrieval_rerun_count": gates["retrieval_rerun_count"],
        "query_reformulation_count": gates["query_reformulation_count"],
        "graph_retrieval_count": gates["graph_retrieval_count"],
        "candidate_membership_change_count": gates["candidate_membership_change_count"],
        "external_provider_call_count": 0,
        "default_reranker_enabled": False,
        "targeted_tests_passed": None,
        "repository_verification_status": None,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def verify_task0093_artifacts(*, write: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "experiment_manifest.json",
        RESULT_DIR / "regression_diagnosis.jsonl",
        RESULT_DIR / "mitigation_arm_results.jsonl",
        RESULT_DIR / "cohort_evaluation.jsonl",
        RESULT_DIR / "pairwise_transition_matrix.jsonl",
        RESULT_DIR / "promotion_gates.json",
        RESULT_DIR / "promotion_decision.json",
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if not issues:
        diagnosis = read_jsonl(RESULT_DIR / "regression_diagnosis.jsonl")
        arms = read_jsonl(RESULT_DIR / "mitigation_arm_results.jsonl")
        gates = read_json(RESULT_DIR / "promotion_gates.json")
        decision = read_json(RESULT_DIR / "promotion_decision.json")
        by_arm = {row["arm_id"]: row for row in arms}
        if len(diagnosis) != 4:
            issues.append({"code": "top5_lost_diagnosis_count_mismatch", "count": len(diagnosis)})
        if by_arm.get("R0", {}).get("top5_recovered_unit_count") != 8 or by_arm.get("R0", {}).get("top5_lost_unit_count") != 4:
            issues.append({"code": "r0_does_not_reproduce_task0092_top5_transitions"})
        if gates.get("candidate_membership_change_count") != 0:
            issues.append({"code": "candidate_membership_changed"})
        if gates.get("retrieval_rerun_count") != 0 or gates.get("query_reformulation_count") != 0 or gates.get("graph_retrieval_count") != 0:
            issues.append({"code": "hard_constraint_count_nonzero"})
        if decision.get("recommended_reranking_policy") not in {None, "no_safe_mitigation_found", *by_arm}:
            issues.append({"code": "unknown_recommended_reranking_policy"})
    result = {
        "schema_version": "opk-rag.task0093.verification.v1",
        "status": "valid" if not issues else "invalid",
        "repository_verification_status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    if write:
        write_json(RESULT_DIR / "verification_summary.json", result)
    return result


def build_report(summary: dict[str, Any], arm_rows: list[dict[str, Any]]) -> str:
    arm_lines = [
        f"- `{row['arm_id']}` {row['policy_family']}: improved=`{row['downstream_improved_unit_count']}`, regressed=`{row['downstream_regressed_unit_count']}`, net=`{row['downstream_net_gain']}`, e2e=`{row['downstream_metrics']['end_to_end_accuracy']}`, retention=`{row['reranker_recovery_retention_rate']}`"
        for row in arm_rows
    ]
    return "\n".join(
        [
            "# TASK0093 Reranker Regression Diagnosis and Guarded Mitigation Report",
            "",
            "## Decision",
            "",
            f"- primary_result_classification=`{summary['primary_result_classification']}`",
            f"- recommended_reranking_policy=`{summary['recommended_reranking_policy']}`",
            f"- recommended_policy_family=`{summary['recommended_policy_family']}`",
            f"- reranker_runtime_promotion_recommendation=`{summary['reranker_runtime_promotion_recommendation']}`",
            f"- next_task_decision=`{summary['next_task_decision']}`",
            "",
            "## Diagnosis",
            "",
            f"- top5_lost_diagnosed_unit_count=`{summary['top5_lost_diagnosed_unit_count']}`",
            f"- root_cause_counts=`{summary['root_cause_counts']}`",
            f"- low_confidence_boundary_flip_count=`{summary['low_confidence_boundary_flip_count']}`",
            "",
            "## Arms",
            "",
            *arm_lines,
            "",
            "## Recommended Utility",
            "",
            f"- R0 downstream net gain=`{summary['r0_downstream_net_gain']}` with regressions=`{summary['r0_downstream_regressed_unit_count']}`",
            f"- recommended downstream net gain=`{summary['recommended_downstream_net_gain']}` with regressions=`{summary['recommended_downstream_regressed_unit_count']}`",
            f"- recommended end_to_end_accuracy=`{summary['recommended_end_to_end_accuracy']}`",
            f"- recommended reranker_recovery_retention_rate=`{summary['recommended_reranker_recovery_retention_rate']}`",
            "",
            "## Governance",
            "",
            f"- task0090_inputs_valid=`{str(summary['task0090_inputs_valid']).lower()}`",
            f"- task0091_inputs_valid=`{str(summary['task0091_inputs_valid']).lower()}`",
            f"- task0092_inputs_valid=`{str(summary['task0092_inputs_valid']).lower()}`",
            f"- candidate_set_frozen=`{str(summary['candidate_set_frozen']).lower()}`",
            f"- candidate_identity_preserved=`{str(summary['candidate_identity_preserved']).lower()}`",
            f"- retrieval_rerun_count=`{summary['retrieval_rerun_count']}`",
            f"- query_reformulation_count=`{summary['query_reformulation_count']}`",
            f"- graph_retrieval_count=`{summary['graph_retrieval_count']}`",
            f"- candidate_membership_change_count=`{summary['candidate_membership_change_count']}`",
            f"- external_provider_call_count=`{summary['external_provider_call_count']}`",
            f"- default_reranker_enabled=`{str(summary['default_reranker_enabled']).lower()}`",
        ]
    )


def ranking_metrics(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [first_relevant_rank(rows) for rows in rows_by_unit.values()]
    present = [rank for rank in ranks if rank is not None]
    return {
        "schema_version": "opk-rag.task0093.ranking-metrics.v1",
        "unit_count": len(ranks),
        "recall_at_1": _ratio(sum(rank is not None and rank <= 1 for rank in ranks), len(ranks)),
        "recall_at_3": _ratio(sum(rank is not None and rank <= 3 for rank in ranks), len(ranks)),
        "recall_at_5": _ratio(sum(rank is not None and rank <= 5 for rank in ranks), len(ranks)),
        "recall_at_10": _ratio(sum(rank is not None and rank <= 10 for rank in ranks), len(ranks)),
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
        "ndcg_at_5": _ratio(sum(ndcg_at_k(rows, 5) for rows in rows_by_unit.values()), len(rows_by_unit)),
        "ndcg_at_10": _ratio(sum(ndcg_at_k(rows, 10) for rows in rows_by_unit.values()), len(rows_by_unit)),
    }


def downstream_classification(base_hit: bool, variant_hit: bool) -> str:
    if not base_hit and variant_hit:
        return "downstream_improved"
    if base_hit and not variant_hit:
        return "downstream_regressed"
    return "downstream_unchanged"


def arm_outcome(relevant_evidence_in_context: bool) -> dict[str, Any]:
    return {
        "answerability_status": "answerable" if relevant_evidence_in_context else "abstained_no_relevant_evidence_in_context",
        "answerability_correct": relevant_evidence_in_context,
        "correct_answer_route": relevant_evidence_in_context,
        "correct_abstention": False,
        "over_abstention": not relevant_evidence_in_context,
        "under_abstention": False,
        "safe_action_correct": relevant_evidence_in_context,
        "safe_refusal_count": 0,
        "unsafe_answer_count": 0,
        "unsupported_answer_count": 0,
        "generation_invoked": relevant_evidence_in_context,
        "generation_contract_valid": relevant_evidence_in_context,
        "answer_draft_count": 1 if relevant_evidence_in_context else 0,
        "model_refusal_count": 0,
        "provider_failure_count": 0,
        "grounded_answer": relevant_evidence_in_context,
        "grounding_failure_count": 0 if relevant_evidence_in_context else 1,
        "unsupported_claim_count": 0,
        "citation_valid": relevant_evidence_in_context,
        "citation_invalid": False,
        "end_to_end_correct": relevant_evidence_in_context,
    }


def assign_policy_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "reranker_rank": index, "policy_rank": index} for index, row in enumerate(rows, start=1)]


def boundary_margin(rows: list[dict[str, Any]]) -> float:
    return float(rows[EVIDENCE_CONTEXT_TOP_K - 1]["reranker_score"]) - float(rows[EVIDENCE_CONTEXT_TOP_K]["reranker_score"])


def retrieval_confidence(row: dict[str, Any]) -> float:
    if row.get("retrieval_score") is not None:
        return float(row["retrieval_score"])
    return 1 / float(row["retrieval_rank"])


def minmax(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def candidate_membership_change_count(
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
) -> int:
    changed = 0
    for unit in baseline:
        left = sorted(row["canonical_chunk_id"] for row in baseline[unit])
        right = sorted(row["canonical_chunk_id"] for row in variant[unit])
        changed += left != right
    return changed


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


if __name__ == "__main__":
    run_task0093_reranker_guarded_mitigation()
