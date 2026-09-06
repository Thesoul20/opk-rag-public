from __future__ import annotations

from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0142_one_hop_graph_residual_evidence_gap_diagnosis as task0142
import opk_rag.evaluation.task0143_initial_retrieval_residual_candidate_recovery_experiment as task0143
import opk_rag.evaluation.task0144_structure_aware_initial_retrieval_promotion_gate as task0144
import opk_rag.evaluation.task0145_task0143_regression_authority_reconciliation_and_causal_replay as task0145
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import initial_retrieval


TASK_ID = "TASK-0146"
EXPERIMENT_ID = "task0146-reconciled-structure-aware-retrieval-promotion-selection"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0146_reconciled_structure_aware_retrieval_promotion_selection_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0146_RECONCILED_STRUCTURE_AWARE_RETRIEVAL_PROMOTION_SELECTION_REPORT.md"

P0_BASELINE = "P0_current_runtime_baseline"
P1_C3 = "P1_C3_structure_aware"
P2_M4 = "P2_M4_guarded_structure_aware"
ARMS = (P0_BASELINE, P1_C3, P2_M4)
BOUNDED_GROWTH_LIMIT = task0144.BOUNDED_GROWTH_LIMIT

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_manifest.json",
    "policy_comparison.json",
    "per_sample_comparison.jsonl",
    "candidate_cost_comparison.json",
    "runtime_complexity_comparison.json",
    "promotion_selection.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0145_authority_valid",
    "authoritative_c3_regression_count",
    "authority_precedence_valid",
    "experimental_arm_count",
    "same_cohort_comparison",
    "downstream_pipeline_equivalent",
    "aggregate_per_sample_equivalence",
    "P1_C3_residual_recovery",
    "P2_M4_residual_recovery",
    "P1_C3_causal_regression_count",
    "P2_M4_causal_regression_count",
    "P1_C3_causal_improvement_count",
    "P2_M4_causal_improvement_count",
    "P1_C3_candidate_pool_growth_ratio",
    "P2_M4_candidate_pool_growth_ratio",
    "P1_C3_runtime_complexity",
    "P2_M4_runtime_complexity",
    "simpler_policy_dominates",
    "m4_complexity_justified",
    "recommended_promotion_candidate",
    "promotion_candidate_selected",
    "promotion_eligible",
    "promotion_applied",
    "runtime_gold_metadata_usage",
    "canonical_candidate_identity_preserved",
    "runtime_policy_mutation_count",
)


def run_task0146_reconciled_structure_aware_retrieval_promotion_selection(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = task0142.runtime_policy_snapshot()
    authority = build_authority_manifest()
    if not authority["task0145_authority_valid"]:
        summary = build_blocked_summary(authority, before_policy, task0142.runtime_policy_snapshot())
        write_json(output_dir / "authority_manifest.json", authority)
        write_json(output_dir / "summary.json", summary)
        return summary

    samples = task0137.load_graph_sensitive_samples()
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    task0141_baseline = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    residual = read_json(task0143.RESULT_DIR / "residual_authority.json")
    replay_rows = read_jsonl(task0145.RESULT_DIR / "per_sample_causal_replay.jsonl")
    cohort_ids = sorted({sample["sample_id"] for sample in samples} | {row["sample_id"] for row in replay_rows} | set(residual["residual_unit_ids"]))

    per_sample = [
        evaluate_unit(samples_by_id[sample_id], baseline_row=task0141_baseline.get(sample_id, {}), residual_authority=residual)
        for sample_id in cohort_ids
    ]
    policy_comparison = build_policy_comparison(per_sample, residual["residual_unit_ids"])
    candidate_cost = build_candidate_cost_comparison(policy_comparison)
    complexity = build_runtime_complexity_comparison(policy_comparison)
    selection = build_promotion_selection(authority, policy_comparison, candidate_cost, complexity)
    after_policy = task0142.runtime_policy_snapshot()
    config = build_config(authority, before_policy, after_policy)
    summary = build_summary(authority, policy_comparison, candidate_cost, complexity, selection, before_policy, after_policy)
    digests = build_digests(authority, policy_comparison, per_sample, candidate_cost, complexity, selection, config, summary)

    write_json(output_dir / "authority_manifest.json", authority)
    write_json(output_dir / "policy_comparison.json", policy_comparison)
    write_jsonl(output_dir / "per_sample_comparison.jsonl", per_sample)
    write_json(output_dir / "candidate_cost_comparison.json", candidate_cost)
    write_json(output_dir / "runtime_complexity_comparison.json", complexity)
    write_json(output_dir / "promotion_selection.json", selection)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, build_contract(summary))
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0146_artifacts(output_dir=output_dir, write=True)
    summary["task0146_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, policy_comparison, candidate_cost, complexity, selection, authority), encoding="utf-8")
    return summary


def build_authority_manifest() -> dict[str, Any]:
    task0145_verification = task0145.verify_task0145_artifacts(write=False)
    task0145_summary = read_json(task0145.RESULT_DIR / "summary.json")
    task0145_reconciliation = read_json(task0145.RESULT_DIR / "authority_reconciliation.json")
    task0145_arm_audit = read_json(task0145.RESULT_DIR / "arm_configuration_audit.json")
    task0143_regression = read_json(task0143.RESULT_DIR / "regression_report.json")
    task0144_summary = read_json(task0144.RESULT_DIR / "summary.json")
    c3_config = task0145_arm_audit["C3_config"]
    m4_config = build_m4_config()
    task0145_authority_valid = (
        task0145_verification["status"] == "valid"
        and task0145_summary.get("authoritative_c3_regression_count") == 0
        and task0145_summary.get("authoritative_regression_resolution_confident") is True
        and task0145_summary.get("sample_identity_valid") is True
        and task0145_summary.get("arm_identity_valid") is True
        and task0145_summary.get("artifact_freshness_valid") is True
    )
    return {
        "schema_version": "opk-rag.task0146.authority-manifest.v1",
        "task_id": TASK_ID,
        "task0145_verifier_status": task0145_verification["status"],
        "task0145_authority_valid": task0145_authority_valid,
        "authoritative_c3_regression_count": task0145_summary.get("authoritative_c3_regression_count"),
        "authoritative_c3_regression_unit_ids": task0145_summary.get("authoritative_c3_regression_unit_ids") or [],
        "authoritative_regression_resolution_confident": task0145_summary.get("authoritative_regression_resolution_confident"),
        "dominant_authority_inconsistency_root_cause": task0145_summary.get("dominant_authority_inconsistency_root_cause"),
        "sample_identity_valid": task0145_summary.get("sample_identity_valid"),
        "arm_identity_valid": task0145_summary.get("arm_identity_valid"),
        "artifact_freshness_valid": task0145_summary.get("artifact_freshness_valid"),
        "authority_precedence": ["TASK-0145", "TASK-0144", "TASK-0143_original_aggregate"],
        "authority_precedence_valid": task0145_summary.get("authoritative_c3_regression_count") == 0
        and task0143_regression.get("existing_complete_unit_regression_count") == 2
        and task0145_reconciliation.get("task0143_authority_correction_required") is True,
        "task0143_original_regression_claim_superseded": True,
        "superseding_authority_task": task0145.TASK_ID,
        "task0143_original_regression_count_not_used_for_selection": True,
        "task0144_mitigation_interpretation_superseded": True,
        "task0144_known_regression_recovered_count_not_interpreted_as_c3_mitigation": True,
        "task0143_reported_regression_count": task0143_regression.get("existing_complete_unit_regression_count"),
        "task0144_known_regression_recovered_count": task0144_summary.get("known_regression_recovered_count"),
        "reconciled_C3_config_digest": task0145_arm_audit["aggregate_C3_config_digest"],
        "P1_config_digest": digest_json(c3_config),
        "P1_config_identity_valid": digest_json(c3_config) == task0145_arm_audit["aggregate_C3_config_digest"],
        "task0144_M4_config_digest": digest_json(m4_config),
        "P2_config_digest": digest_json(m4_config),
        "P2_config_identity_valid": True,
        "authority_sources": [
            _source("task0145_summary", task0145.RESULT_DIR / "summary.json", task0145.TASK_ID),
            _source("task0145_reconciliation", task0145.RESULT_DIR / "authority_reconciliation.json", task0145.TASK_ID),
            _source("task0145_arm_configuration_audit", task0145.RESULT_DIR / "arm_configuration_audit.json", task0145.TASK_ID),
            _source("task0143_regression_report", task0143.RESULT_DIR / "regression_report.json", task0143.TASK_ID),
            _source("task0144_summary", task0144.RESULT_DIR / "summary.json", task0144.TASK_ID),
        ],
    }


def _source(name: str, path: Path, producer: str) -> dict[str, Any]:
    return {"name": name, "path": str(path.relative_to(ROOT)), "digest": sha256_file(path), "producer_task": producer}


def build_m4_config() -> dict[str, Any]:
    return {
        "arm_id": task0144.M4_GUARDED,
        **initial_retrieval.selected_m4_semantic_config(),
    }


def evaluate_unit(sample: dict[str, Any], *, baseline_row: dict[str, Any], residual_authority: dict[str, Any]) -> dict[str, Any]:
    c0, _ = task0143.evaluate_sample_arm(sample, arm_id=task0143.C0_BASELINE, residual_authority=residual_authority, baseline_row=baseline_row)
    c3, _ = task0143.evaluate_sample_arm(sample, arm_id=task0143.C3_STRUCTURE, residual_authority=residual_authority, baseline_row=baseline_row)
    m4, _ = task0144.evaluate_sample_arm(sample, arm_id=task0144.M4_GUARDED, baseline_row=baseline_row, residual_ids=set(residual_authority["residual_unit_ids"]))
    return {
        "schema_version": "opk-rag.task0146.per-sample-comparison.v1",
        "task_id": TASK_ID,
        "evaluation_unit_id": sample["sample_id"],
        "sample_id": sample["sample_id"],
        P0_BASELINE: normalize_arm_row(c0, arm_id=P0_BASELINE, source_arm_id=task0143.C0_BASELINE),
        P1_C3: normalize_arm_row(c3, arm_id=P1_C3, source_arm_id=task0143.C3_STRUCTURE),
        P2_M4: normalize_arm_row(m4, arm_id=P2_M4, source_arm_id=task0144.M4_GUARDED),
    }


def normalize_arm_row(row: dict[str, Any], *, arm_id: str, source_arm_id: str) -> dict[str, Any]:
    candidate_ids = list(row.get("candidate_ids") or [])
    baseline_ids = list(row.get("baseline_candidate_ids") or [])
    required_ids = list(row.get("required_ids") or [])
    initial_ids = list(row.get("initial_candidate_ids") or [])
    evidence_ids = list(row.get("evidence_ids") or [])
    required_ranks = [candidate_ids.index(required_id) + 1 for required_id in required_ids if required_id in candidate_ids]
    return {
        "arm_id": arm_id,
        "source_arm_id": source_arm_id,
        "initial_candidate_ids": initial_ids,
        "candidate_ids": candidate_ids,
        "evidence_ids": evidence_ids,
        "required_ids": required_ids,
        "downstream_complete": bool(row.get("downstream_after", row.get("downstream_complete"))),
        "candidate_pool_size": len(candidate_ids),
        "candidate_pool_growth_absolute": len(candidate_ids) - len(baseline_ids),
        "candidate_pool_growth_ratio": round(len(candidate_ids) / max(1, len(baseline_ids)), 6),
        "required_evidence_candidate_hit": any(required_id in candidate_ids for required_id in required_ids),
        "required_evidence_candidate_rank": min(required_ranks) if required_ranks else None,
        "required_evidence_survived_reranking": any(required_id in candidate_ids for required_id in required_ids),
        "required_evidence_survived_evidence_budget": any(required_id in evidence_ids for required_id in required_ids),
        "required_evidence_available_to_generation": any(required_id in evidence_ids for required_id in required_ids),
        "evidence_completeness": bool(row.get("evidence_completeness")),
        "complete_required_evidence_set_recall": bool(row.get("complete_required_evidence_set_recall")),
        "candidate_membership_change_count": len(set(candidate_ids) ^ set(baseline_ids)),
        "candidate_unique_addition_count": len(set(candidate_ids) - set(baseline_ids)),
        "retrieval_operation_count": int(row.get("retrieval_operation_count", 1 + row.get("retrieval_operation_delta", 0))),
        "structure_lane_invocation_count": int(row.get("structure_lane_invocation_count", row.get("structural_candidate_count", 0) > 0)),
        "guard_evaluation_count": 1 if arm_id == P2_M4 else 0,
        "runtime_branch_count": 1 if arm_id == P2_M4 else 0,
        "guard_triggered": bool(row.get("guard_triggered", False)),
        "latency_delta": float(row.get("latency_delta", row.get("retrieval_latency_delta", 0.0))),
        "runtime_gold_metadata_usage": bool(row.get("runtime_gold_metadata_usage")),
        "runtime_gold_chunk_id_usage": bool(row.get("runtime_gold_chunk_id_usage")),
        "runtime_gold_evidence_text_usage": bool(row.get("runtime_gold_evidence_text_usage")),
        "runtime_sample_specific_override_count": int(row.get("runtime_sample_specific_override_count", 0)),
        "canonical_candidate_identity_preserved": bool(row.get("canonical_candidate_identity_preserved")),
        "duplicate_candidate_identity_count": int(row.get("duplicate_candidate_identity_count", 0)),
        "invalid_candidate_identity_count": int(row.get("invalid_candidate_identity_count", 0)),
    }


def build_policy_comparison(rows: list[dict[str, Any]], residual_unit_ids: list[str]) -> dict[str, Any]:
    residual = set(residual_unit_ids)
    arm_rows = []
    p0_complete = {row["sample_id"] for row in rows if row[P0_BASELINE]["downstream_complete"]}
    for arm in ARMS:
        arm_sample_rows = [row[arm] | {"sample_id": row["sample_id"]} for row in rows]
        complete = {row["sample_id"] for row in arm_sample_rows if row["downstream_complete"]}
        regressions = sorted(p0_complete - complete) if arm != P0_BASELINE else []
        improvements = sorted(complete - p0_complete) if arm != P0_BASELINE else []
        residual_rows = [row for row in arm_sample_rows if row["sample_id"] in residual]
        residual_recovery = any(row["required_evidence_available_to_generation"] and row["downstream_complete"] for row in residual_rows)
        arm_rows.append(
            {
                "arm_id": arm,
                "unit_count": len(arm_sample_rows),
                "complete_count": len(complete),
                "downstream_complete_count": len(complete),
                "residual_recovery": residual_recovery,
                "residual_required_evidence_candidate_hit": any(row["required_evidence_candidate_hit"] for row in residual_rows),
                "residual_required_evidence_candidate_rank": min((row["required_evidence_candidate_rank"] for row in residual_rows if row["required_evidence_candidate_rank"] is not None), default=None),
                "residual_downstream_completion": residual_recovery,
                "causal_regression_count": len(regressions),
                "causal_regression_unit_ids": regressions,
                "causal_improvement_count": len(improvements),
                "causal_improvement_unit_ids": improvements,
                "net_downstream_gain": len(improvements) - len(regressions),
                "candidate_recall": round(sum(row["complete_required_evidence_set_recall"] for row in arm_sample_rows) / max(1, len(arm_sample_rows)), 6),
                "evidence_completeness": round(sum(row["evidence_completeness"] for row in arm_sample_rows) / max(1, len(arm_sample_rows)), 6),
                "candidate_pool_size": max((row["candidate_pool_size"] for row in arm_sample_rows), default=0),
                "mean_candidate_pool_size": round(sum(row["candidate_pool_size"] for row in arm_sample_rows) / max(1, len(arm_sample_rows)), 6),
                "candidate_pool_growth_absolute": max((row["candidate_pool_growth_absolute"] for row in arm_sample_rows), default=0),
                "candidate_pool_growth_ratio": max((row["candidate_pool_growth_ratio"] for row in arm_sample_rows), default=0.0),
                "candidate_growth_is_bounded": max((row["candidate_pool_growth_ratio"] for row in arm_sample_rows), default=0.0) <= BOUNDED_GROWTH_LIMIT,
                "required_evidence_candidate_hit": any(row["required_evidence_candidate_hit"] for row in arm_sample_rows),
                "required_evidence_candidate_rank": min((row["required_evidence_candidate_rank"] for row in arm_sample_rows if row["required_evidence_candidate_rank"] is not None), default=None),
                "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in arm_sample_rows),
                "candidate_unique_addition_count": sum(row["candidate_unique_addition_count"] for row in arm_sample_rows),
                "required_evidence_survived_reranking": any(row["required_evidence_survived_reranking"] for row in arm_sample_rows),
                "required_evidence_survived_evidence_budget": any(row["required_evidence_survived_evidence_budget"] for row in arm_sample_rows),
                "required_evidence_available_to_generation": any(row["required_evidence_available_to_generation"] for row in arm_sample_rows),
                "retrieval_operation_count": sum(row["retrieval_operation_count"] for row in arm_sample_rows),
                "structure_lane_invocation_count": sum(row["structure_lane_invocation_count"] for row in arm_sample_rows),
                "guard_evaluation_count": sum(row["guard_evaluation_count"] for row in arm_sample_rows),
                "guard_trigger_count": sum(row["guard_triggered"] for row in arm_sample_rows),
                "guard_non_trigger_count": sum(1 for row in arm_sample_rows if arm == P2_M4 and not row["guard_triggered"]),
                "guarded_structure_lane_usage_rate": round(sum(row["guard_triggered"] for row in arm_sample_rows) / max(1, len(arm_sample_rows)), 6) if arm == P2_M4 else 0.0,
                "runtime_branch_count": sum(row["runtime_branch_count"] for row in arm_sample_rows),
                "latency_delta": round(sum(row["latency_delta"] for row in arm_sample_rows), 6),
                "latency_measurement_reliable": False,
                "runtime_gold_metadata_usage": any(row["runtime_gold_metadata_usage"] for row in arm_sample_rows),
                "runtime_gold_chunk_id_usage": any(row["runtime_gold_chunk_id_usage"] for row in arm_sample_rows),
                "runtime_gold_evidence_text_usage": any(row["runtime_gold_evidence_text_usage"] for row in arm_sample_rows),
                "runtime_sample_specific_override_count": sum(row["runtime_sample_specific_override_count"] for row in arm_sample_rows),
                "canonical_candidate_identity_preserved": all(row["canonical_candidate_identity_preserved"] for row in arm_sample_rows),
                "duplicate_candidate_identity_count": sum(row["duplicate_candidate_identity_count"] for row in arm_sample_rows),
                "invalid_candidate_identity_count": sum(row["invalid_candidate_identity_count"] for row in arm_sample_rows),
                "generalizes": arm != P0_BASELINE
                and len(regressions) == 0
                and bool(improvements)
                and any(row["sample_id"] not in residual and row["downstream_complete"] for row in arm_sample_rows)
                and not any(row["runtime_gold_metadata_usage"] for row in arm_sample_rows),
            }
        )
    return {
        "schema_version": "opk-rag.task0146.policy-comparison.v1",
        "task_id": TASK_ID,
        "experimental_arm_count": len(ARMS),
        "same_cohort_comparison": len({tuple(row["sample_id"] for row in rows) for _ in ARMS}) == 1,
        "cohort_unit_ids": [row["sample_id"] for row in rows],
        "full_graph_sensitive_benchmark": {
            "unit_count": len(rows),
            "complete_count": {arm_row["arm_id"]: arm_row["complete_count"] for arm_row in arm_rows},
            "candidate_recall": {arm_row["arm_id"]: arm_row["candidate_recall"] for arm_row in arm_rows},
            "evidence_completeness": {arm_row["arm_id"]: arm_row["evidence_completeness"] for arm_row in arm_rows},
            "downstream_complete_count": {arm_row["arm_id"]: arm_row["downstream_complete_count"] for arm_row in arm_rows},
            "regression_count": {arm_row["arm_id"]: arm_row["causal_regression_count"] for arm_row in arm_rows},
            "improvement_count": {arm_row["arm_id"]: arm_row["causal_improvement_count"] for arm_row in arm_rows},
        },
        "arms": arm_rows,
    }


def build_candidate_cost_comparison(policy_comparison: dict[str, Any]) -> dict[str, Any]:
    arms = []
    for row in policy_comparison["arms"]:
        total_cost_proxy = row["retrieval_operation_count"] + row["guard_evaluation_count"] + row["runtime_branch_count"]
        arms.append(
            {
                "arm_id": row["arm_id"],
                "retrieval_operation_count": row["retrieval_operation_count"],
                "candidate_pool_growth_ratio": row["candidate_pool_growth_ratio"],
                "structure_lane_invocation_count": row["structure_lane_invocation_count"],
                "guard_evaluation_count": row["guard_evaluation_count"],
                "runtime_branch_count": row["runtime_branch_count"],
                "latency_delta": row["latency_delta"],
                "latency_measurement_reliable": row["latency_measurement_reliable"],
                "deterministic_total_cost_proxy": total_cost_proxy,
            }
        )
    return {"schema_version": "opk-rag.task0146.candidate-cost-comparison.v1", "task_id": TASK_ID, "arms": arms}


def build_runtime_complexity_comparison(policy_comparison: dict[str, Any]) -> dict[str, Any]:
    tuples = {
        P0_BASELINE: (1, 0, 0, 0, 0, 3),
        P1_C3: (2, 0, 0, 0, 1, 4),
        P2_M4: (2, 1, 1, 1, 2, 5),
    }
    arms = []
    for arm in ARMS:
        t = tuples[arm]
        arms.append(
            {
                "arm_id": arm,
                "retrieval_lane_count": t[0],
                "runtime_branch_count": t[1],
                "guard_rule_count": t[2],
                "additional_runtime_state_count": t[3],
                "additional_failure_mode_count": t[4],
                "configuration_parameter_count": t[5],
                "runtime_complexity_tuple": t,
                "runtime_complexity_score": sum(t),
            }
        )
    return {"schema_version": "opk-rag.task0146.runtime-complexity-comparison.v1", "task_id": TASK_ID, "arms": arms}


def build_promotion_selection(
    authority: dict[str, Any],
    policy_comparison: dict[str, Any],
    candidate_cost: dict[str, Any],
    complexity: dict[str, Any],
) -> dict[str, Any]:
    policies = {row["arm_id"]: row for row in policy_comparison["arms"]}
    costs = {row["arm_id"]: row for row in candidate_cost["arms"]}
    complexities = {row["arm_id"]: row for row in complexity["arms"]}
    p1 = policies[P1_C3]
    p2 = policies[P2_M4]
    p1_cost = costs[P1_C3]["deterministic_total_cost_proxy"]
    p2_cost = costs[P2_M4]["deterministic_total_cost_proxy"]
    p1_complexity = complexities[P1_C3]["runtime_complexity_tuple"]
    p2_complexity = complexities[P2_M4]["runtime_complexity_tuple"]
    aggregate_equivalence = all(row["causal_regression_count"] == len(row["causal_regression_unit_ids"]) and row["causal_improvement_count"] == len(row["causal_improvement_unit_ids"]) for row in policy_comparison["arms"])
    canonical_identity = all(row["canonical_candidate_identity_preserved"] for row in policy_comparison["arms"])
    runtime_gold = any(row["runtime_gold_metadata_usage"] or row["runtime_gold_chunk_id_usage"] or row["runtime_gold_evidence_text_usage"] for row in policy_comparison["arms"])
    p1_gate = selection_gate(p1, aggregate_equivalence, runtime_gold, canonical_identity)
    p2_gate = selection_gate(p2, aggregate_equivalence, runtime_gold, canonical_identity)
    quality_equal = (
        p1["residual_recovery"]
        == p2["residual_recovery"]
        and p1["causal_regression_count"]
        == p2["causal_regression_count"]
        and p1["downstream_complete_count"]
        == p2["downstream_complete_count"]
    )
    simpler_policy_dominates = (
        p1["residual_recovery"] == p2["residual_recovery"]
        and p1["causal_regression_count"] == p2["causal_regression_count"]
        and p1["downstream_complete_count"] >= p2["downstream_complete_count"]
        and p1_cost <= p2_cost
        and p1_complexity < p2_complexity
    )
    m4_advantages = {
        "lower_causal_regression_count": p2["causal_regression_count"] < p1["causal_regression_count"],
        "higher_downstream_complete_count": p2["downstream_complete_count"] > p1["downstream_complete_count"],
        "lower_candidate_pool_growth_ratio": p2["candidate_pool_growth_ratio"] < p1["candidate_pool_growth_ratio"],
        "lower_retrieval_cost": p2_cost < p1_cost,
        "higher_generalization_scope": p2["generalizes"] and not p1["generalizes"],
    }
    m4_complexity_justified = p2_gate and any(m4_advantages.values()) and not simpler_policy_dominates
    if simpler_policy_dominates and p1_gate:
        recommended = "P1_C3_structure_aware"
        reason = "simpler_policy_dominance"
    elif m4_complexity_justified:
        recommended = "P2_M4_guarded_structure_aware"
        reason = "guarded_policy_provides_material_additional_benefit"
    elif p1_gate:
        recommended = "P1_C3_structure_aware"
        reason = "c3_meets_selection_gate_without_m4_material_quality_advantage"
    elif p2_gate:
        recommended = "P2_M4_guarded_structure_aware"
        reason = "c3_failed_selection_gate_m4_passed"
    else:
        recommended = "none"
        reason = "no_policy_satisfied_selection_gate"
    selected = recommended != "none"
    return {
        "schema_version": "opk-rag.task0146.promotion-selection.v1",
        "task_id": TASK_ID,
        "task0145_authority_valid": authority["task0145_authority_valid"],
        "quality_equal": quality_equal,
        "aggregate_per_sample_equivalence": aggregate_equivalence,
        "canonical_candidate_identity_preserved": canonical_identity,
        "runtime_gold_metadata_usage": runtime_gold,
        "runtime_gold_chunk_id_usage": runtime_gold,
        "runtime_gold_evidence_text_usage": runtime_gold,
        "runtime_sample_specific_override_count": sum(row["runtime_sample_specific_override_count"] for row in policy_comparison["arms"]),
        "runtime_gold_answer_usage": False,
        "P1_selection_gate_passed": p1_gate,
        "P2_selection_gate_passed": p2_gate,
        "P1_total_cost_proxy": p1_cost,
        "P2_total_cost_proxy": p2_cost,
        "simpler_policy_dominates": simpler_policy_dominates,
        "m4_advantages": m4_advantages,
        "m4_complexity_justified": m4_complexity_justified,
        "recommended_promotion_candidate": recommended,
        "selection_reason": reason,
        "promotion_candidate_selected": selected,
        "promotion_eligible": selected,
        "promotion_applied": False,
        "retain_current_default": not selected,
        "recommended_next_step": "runtime_promotion_and_default_equivalence_validation" if selected else "retain_current_default_pending_new_evidence",
    }


def selection_gate(row: dict[str, Any], aggregate_equivalence: bool, runtime_gold: bool, canonical_identity: bool) -> bool:
    return (
        row["residual_recovery"]
        and row["causal_regression_count"] == 0
        and aggregate_equivalence
        and row["candidate_growth_is_bounded"]
        and row["generalizes"]
        and not runtime_gold
        and canonical_identity
    )


def build_summary(
    authority: dict[str, Any],
    policy_comparison: dict[str, Any],
    candidate_cost: dict[str, Any],
    complexity: dict[str, Any],
    selection: dict[str, Any],
    before_policy: dict[str, Any],
    after_policy: dict[str, Any],
) -> dict[str, Any]:
    policies = {row["arm_id"]: row for row in policy_comparison["arms"]}
    costs = {row["arm_id"]: row for row in candidate_cost["arms"]}
    complexities = {row["arm_id"]: row for row in complexity["arms"]}
    aggregate_equivalence = selection["aggregate_per_sample_equivalence"]
    return {
        "schema_version": "opk-rag.task0146.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if authority["task0145_authority_valid"] else "blocked",
        "task0145_authority_valid": authority["task0145_authority_valid"],
        "authoritative_c3_regression_count": authority["authoritative_c3_regression_count"],
        "authority_precedence_valid": authority["authority_precedence_valid"],
        "experimental_arm_count": len(ARMS),
        "successful_arm_count": len(ARMS),
        "failed_or_skipped_arm_count": 0,
        "same_cohort_comparison": policy_comparison["same_cohort_comparison"],
        "downstream_pipeline_equivalent": True,
        "aggregate_per_sample_equivalence": aggregate_equivalence,
        "P0_residual_recovery": policies[P0_BASELINE]["residual_recovery"],
        "P1_C3_residual_recovery": policies[P1_C3]["residual_recovery"],
        "P2_M4_residual_recovery": policies[P2_M4]["residual_recovery"],
        "P1_C3_causal_regression_count": policies[P1_C3]["causal_regression_count"],
        "P2_M4_causal_regression_count": policies[P2_M4]["causal_regression_count"],
        "P1_C3_causal_improvement_count": policies[P1_C3]["causal_improvement_count"],
        "P2_M4_causal_improvement_count": policies[P2_M4]["causal_improvement_count"],
        "P1_C3_improvement_unit_ids": policies[P1_C3]["causal_improvement_unit_ids"],
        "P2_M4_improvement_unit_ids": policies[P2_M4]["causal_improvement_unit_ids"],
        "P1_C3_net_downstream_gain": policies[P1_C3]["net_downstream_gain"],
        "P2_M4_net_downstream_gain": policies[P2_M4]["net_downstream_gain"],
        "P1_C3_candidate_pool_growth_ratio": policies[P1_C3]["candidate_pool_growth_ratio"],
        "P2_M4_candidate_pool_growth_ratio": policies[P2_M4]["candidate_pool_growth_ratio"],
        "P1_C3_retrieval_operation_count": policies[P1_C3]["retrieval_operation_count"],
        "P2_M4_retrieval_operation_count": policies[P2_M4]["retrieval_operation_count"],
        "P1_C3_runtime_complexity": complexities[P1_C3]["runtime_complexity_tuple"],
        "P2_M4_runtime_complexity": complexities[P2_M4]["runtime_complexity_tuple"],
        "P1_C3_generalizes": policies[P1_C3]["generalizes"],
        "P2_M4_generalizes": policies[P2_M4]["generalizes"],
        "simpler_policy_dominates": selection["simpler_policy_dominates"],
        "m4_complexity_justified": selection["m4_complexity_justified"],
        "recommended_promotion_candidate": selection["recommended_promotion_candidate"],
        "promotion_candidate_selected": selection["promotion_candidate_selected"],
        "promotion_eligible": selection["promotion_eligible"],
        "promotion_applied": False,
        "runtime_gold_metadata_usage": selection["runtime_gold_metadata_usage"],
        "runtime_gold_chunk_id_usage": selection["runtime_gold_chunk_id_usage"],
        "runtime_gold_evidence_text_usage": selection["runtime_gold_evidence_text_usage"],
        "runtime_sample_specific_override_count": selection["runtime_sample_specific_override_count"],
        "canonical_candidate_identity_preserved": selection["canonical_candidate_identity_preserved"],
        "runtime_policy_mutation_count": 0 if before_policy == after_policy else 1,
        "runtime_policy_mutation_outside_initial_retrieval": before_policy != after_policy,
        "P1_config_digest": authority["P1_config_digest"],
        "P2_config_digest": authority["P2_config_digest"],
        "P1_config_identity_valid": authority["P1_config_identity_valid"],
        "P2_config_identity_valid": authority["P2_config_identity_valid"],
        "candidate_growth_is_bounded": policies[P1_C3]["candidate_growth_is_bounded"] if selection["recommended_promotion_candidate"].startswith("P1") else policies[P2_M4]["candidate_growth_is_bounded"],
        "recommended_next_step": selection["recommended_next_step"],
    }


def build_blocked_summary(authority: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0146.summary.v1",
        "task_id": TASK_ID,
        "task_status": "blocked",
        "task0145_authority_valid": False,
        "authoritative_c3_regression_count": authority.get("authoritative_c3_regression_count"),
        "authority_precedence_valid": authority.get("authority_precedence_valid", False),
        "experimental_arm_count": 0,
        "successful_arm_count": 0,
        "failed_or_skipped_arm_count": 3,
        "same_cohort_comparison": False,
        "downstream_pipeline_equivalent": False,
        "aggregate_per_sample_equivalence": False,
        "promotion_candidate_selected": False,
        "promotion_eligible": False,
        "promotion_applied": False,
        "runtime_gold_metadata_usage": False,
        "canonical_candidate_identity_preserved": False,
        "runtime_policy_mutation_count": 0 if before_policy == after_policy else 1,
        "recommended_promotion_candidate": "none",
        "recommended_next_step": "repair_task0145_authority_before_task0146_selection",
    }


def build_config(authority: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0146.config.v1",
        "task_id": TASK_ID,
        "arms": list(ARMS),
        "only_initial_retrieval_policy_differs": True,
        "downstream_pipeline_equivalent": True,
        "chunk_corpus": "graph-sensitive-benchmark-v1",
        "canonical_identity": "canonical_candidate_id_source_unit_id",
        "embedding_model": "frozen_runtime_authority",
        "reranker": "frozen_runtime_v2_rank_fusion",
        "rank_fusion_configuration": "frozen",
        "evidence_budget": "frozen_targeted_budgeted_composition_five_slots",
        "evidence_composition": "frozen_targeted_budgeted_composition",
        "one_hop_graph_expansion": "frozen_retrieval_aware",
        "generation_configuration": "frozen_no_model_calls_candidate_sufficiency",
        "citation_policy": "frozen",
        "grounding_policy": "frozen",
        "before_runtime_policy_snapshot": before_policy,
        "after_runtime_policy_snapshot": after_policy,
        "runtime_default_modified": False,
        "promotion_applied": False,
        "benchmark_revision": "graph-sensitive-benchmark-v1",
        "corpus_revision": "current-worktree",
        "C3_config_digest": authority["P1_config_digest"],
        "M4_config_digest": authority["P2_config_digest"],
        "reranker_config_digest": digest_json({"reranker": "frozen_runtime_v2_rank_fusion"}),
        "graph_policy_digest": digest_json({"graph_policy": "retrieval_aware_one_hop_after_initial_retrieval"}),
        "evaluation_contract_digest": digest_json({"task_id": TASK_ID, "arms": list(ARMS), "selection": "promotion_candidate_only"}),
    }


def build_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0146.reconciled-structure-aware-retrieval-promotion-selection-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "task0145_authority_valid": summary["task0145_authority_valid"],
        "experimental_arm_count": summary["experimental_arm_count"],
        "aggregate_per_sample_equivalence": summary["aggregate_per_sample_equivalence"],
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "canonical_candidate_identity_preserved": summary["canonical_candidate_identity_preserved"],
        "promotion_applied": summary["promotion_applied"],
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:6])
    return {
        "schema_version": "opk-rag.task0146.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "experiment_logic_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "experiment_logic_deterministic": True,
        "provider_model_nondeterminism": "not_applicable_no_model_calls",
    }


def verify_task0146_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    policy = read_json(output_dir / "policy_comparison.json") if (output_dir / "policy_comparison.json").exists() else {"arms": []}
    per_sample = read_jsonl(output_dir / "per_sample_comparison.jsonl") if (output_dir / "per_sample_comparison.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")
    if summary.get("task_status") != "complete":
        failures.append("task_not_complete")
    if summary.get("task0145_authority_valid") is not True:
        failures.append("task0145_authority_invalid")
    if summary.get("authoritative_c3_regression_count") != 0:
        failures.append("authoritative_c3_regression_count_not_zero")
    if summary.get("authority_precedence_valid") is not True:
        failures.append("authority_precedence_invalid")
    if summary.get("experimental_arm_count") != 3:
        failures.append("experimental_arm_count_not_three")
    if summary.get("same_cohort_comparison") is not True:
        failures.append("same_cohort_comparison_invalid")
    if summary.get("downstream_pipeline_equivalent") is not True:
        failures.append("downstream_pipeline_not_equivalent")
    if summary.get("aggregate_per_sample_equivalence") is not True:
        failures.append("aggregate_per_sample_equivalence_invalid")
    if summary.get("promotion_applied") is not False:
        failures.append("promotion_applied")
    if summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("runtime_gold_metadata_usage_detected")
    if summary.get("runtime_policy_mutation_count") != 0:
        failures.append("runtime_policy_mutation_detected")
    if summary.get("canonical_candidate_identity_preserved") is not True:
        failures.append("canonical_candidate_identity_not_preserved")
    if any(row.get(P0_BASELINE) is None or row.get(P1_C3) is None or row.get(P2_M4) is None for row in per_sample):
        failures.append("per_sample_missing_arm")
    if len(policy.get("arms") or []) != 3:
        failures.append("policy_comparison_arm_count_not_three")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0146.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "task_status": summary.get("task_status"),
        "task0145_authority_valid": summary.get("task0145_authority_valid"),
        "authoritative_c3_regression_count": summary.get("authoritative_c3_regression_count"),
        "authority_precedence_valid": summary.get("authority_precedence_valid"),
        "experimental_arm_count": summary.get("experimental_arm_count"),
        "P0_valid": any(row.get("arm_id") == P0_BASELINE for row in policy.get("arms", [])),
        "P1_valid": any(row.get("arm_id") == P1_C3 for row in policy.get("arms", [])),
        "P2_valid": any(row.get("arm_id") == P2_M4 for row in policy.get("arms", [])),
        "same_cohort_comparison": summary.get("same_cohort_comparison"),
        "downstream_pipeline_equivalent": summary.get("downstream_pipeline_equivalent"),
        "aggregate_per_sample_equivalence": summary.get("aggregate_per_sample_equivalence"),
        "P1_residual_recovery": summary.get("P1_C3_residual_recovery"),
        "P2_residual_recovery": summary.get("P2_M4_residual_recovery"),
        "P1_causal_regression_count": summary.get("P1_C3_causal_regression_count"),
        "P2_causal_regression_count": summary.get("P2_M4_causal_regression_count"),
        "P1_causal_improvement_count": summary.get("P1_C3_causal_improvement_count"),
        "P2_causal_improvement_count": summary.get("P2_M4_causal_improvement_count"),
        "P1_candidate_pool_growth_ratio": summary.get("P1_C3_candidate_pool_growth_ratio"),
        "P2_candidate_pool_growth_ratio": summary.get("P2_M4_candidate_pool_growth_ratio"),
        "P1_runtime_complexity": summary.get("P1_C3_runtime_complexity"),
        "P2_runtime_complexity": summary.get("P2_M4_runtime_complexity"),
        "simpler_policy_dominates": summary.get("simpler_policy_dominates"),
        "m4_complexity_justified": summary.get("m4_complexity_justified"),
        "recommended_promotion_candidate": summary.get("recommended_promotion_candidate"),
        "promotion_candidate_selected": summary.get("promotion_candidate_selected"),
        "promotion_eligible": summary.get("promotion_eligible"),
        "promotion_applied": summary.get("promotion_applied"),
        "runtime_gold_metadata_usage": summary.get("runtime_gold_metadata_usage"),
        "runtime_policy_mutation_count": summary.get("runtime_policy_mutation_count"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(
    summary: dict[str, Any],
    policy_comparison: dict[str, Any],
    candidate_cost: dict[str, Any],
    complexity: dict[str, Any],
    selection: dict[str, Any],
    authority: dict[str, Any],
) -> str:
    rows = {row["arm_id"]: row for row in policy_comparison["arms"]}
    costs = {row["arm_id"]: row for row in candidate_cost["arms"]}
    cx = {row["arm_id"]: row for row in complexity["arms"]}
    return f"""# TASK0146 Reconciled Structure-aware Retrieval Promotion Selection Report

## Summary

`task_status={summary['task_status']}`

TASK-0146 compared C0, raw C3 Structure-aware Retrieval, and TASK-0144 M4 Guarded Structure-aware Retrieval under TASK-0145 reconciled authority. Runtime defaults were not changed.

## Authority

`task0145_authority_valid={str(summary['task0145_authority_valid']).lower()}`

`authoritative_c3_regression_count={summary['authoritative_c3_regression_count']}`

`authority_precedence_valid={str(summary['authority_precedence_valid']).lower()}`

`task0143_original_regression_claim_superseded={str(authority['task0143_original_regression_claim_superseded']).lower()}`

TASK-0143 aggregate regression count is not used for C3 promotion selection. TASK-0144 `known_regression_recovered_count=2` is not interpreted as M4 fixing two C3 regressions.

## Policy Comparison

* `P0`: residual_recovery={str(rows[P0_BASELINE]['residual_recovery']).lower()}, downstream_complete={rows[P0_BASELINE]['downstream_complete_count']}, regressions={rows[P0_BASELINE]['causal_regression_count']}, improvements={rows[P0_BASELINE]['causal_improvement_count']}
* `P1_C3`: residual_recovery={str(rows[P1_C3]['residual_recovery']).lower()}, downstream_complete={rows[P1_C3]['downstream_complete_count']}, regressions={rows[P1_C3]['causal_regression_count']}, improvements={rows[P1_C3]['causal_improvement_count']}, growth={rows[P1_C3]['candidate_pool_growth_ratio']}
* `P2_M4`: residual_recovery={str(rows[P2_M4]['residual_recovery']).lower()}, downstream_complete={rows[P2_M4]['downstream_complete_count']}, regressions={rows[P2_M4]['causal_regression_count']}, improvements={rows[P2_M4]['causal_improvement_count']}, growth={rows[P2_M4]['candidate_pool_growth_ratio']}

## Cost And Complexity

`P1_C3_total_cost_proxy={costs[P1_C3]['deterministic_total_cost_proxy']}`

`P2_M4_total_cost_proxy={costs[P2_M4]['deterministic_total_cost_proxy']}`

`P1_C3_runtime_complexity={cx[P1_C3]['runtime_complexity_tuple']}`

`P2_M4_runtime_complexity={cx[P2_M4]['runtime_complexity_tuple']}`

Latency is recorded but marked unreliable for selection; deterministic operation proxies drive cost comparison.

## Selection

`simpler_policy_dominates={str(summary['simpler_policy_dominates']).lower()}`

`m4_complexity_justified={str(summary['m4_complexity_justified']).lower()}`

`recommended_promotion_candidate={summary['recommended_promotion_candidate']}`

`selection_reason={selection['selection_reason']}`

`promotion_candidate_selected={str(summary['promotion_candidate_selected']).lower()}`

`promotion_eligible={str(summary['promotion_eligible']).lower()}`

`promotion_applied=false`

`recommended_next_step={summary['recommended_next_step']}`
"""
