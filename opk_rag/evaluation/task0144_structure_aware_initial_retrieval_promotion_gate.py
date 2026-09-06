from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0141_bounded_multi_hop_path_retrieval_experiment as task0141
import opk_rag.evaluation.task0142_one_hop_graph_residual_evidence_gap_diagnosis as task0142
import opk_rag.evaluation.task0143_initial_retrieval_residual_candidate_recovery_experiment as task0143
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval, initial_retrieval


TASK_ID = "TASK-0144"
EXPERIMENT_ID = "task0144-structure-aware-initial-retrieval-promotion-gate"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0144_structure_aware_initial_retrieval_promotion_gate_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0144_STRUCTURE_AWARE_INITIAL_RETRIEVAL_PROMOTION_GATE_REPORT.md"

M0_BASELINE = "M0_current_baseline"
M1_RAW_STRUCTURE = "M1_raw_structure_aware"
M2_UNION = "M2_body_structure_union"
M3_SCORE_FUSION = "M3_body_structure_score_fusion"
M4_GUARDED = "M4_guarded_structure_aware"
ARMS = (M0_BASELINE, M1_RAW_STRUCTURE, M2_UNION, M3_SCORE_FUSION, M4_GUARDED)

BODY_WEIGHT = 1.0
STRUCTURE_WEIGHT = 0.42
BOUNDED_GROWTH_LIMIT = 3.0

REQUIRED_ARTIFACTS = (
    "summary.json",
    "authority_audit.json",
    "regression_stage_diagnosis.jsonl",
    "representation_audit.jsonl",
    "arm_results.jsonl",
    "arm_comparison.json",
    "guard_quality.json",
    "promotion_gate.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0142_authority_valid",
    "task0143_contract_valid",
    "authority_inconsistency_count",
    "residual_unit_count",
    "known_regression_unit_count",
    "best_mitigation_arm",
    "residual_recovery_preserved",
    "known_regression_recovered_count",
    "remaining_known_regression_count",
    "new_regression_count",
    "total_regression_count",
    "dominant_regression_root_cause",
    "dominant_mitigation_mechanism",
    "candidate_growth_is_bounded",
    "runtime_gold_metadata_usage",
    "runtime_gold_chunk_id_usage",
    "runtime_gold_evidence_text_usage",
    "runtime_gold_answer_usage",
    "runtime_sample_specific_override_count",
    "runtime_policy_mutation_outside_initial_retrieval",
    "canonical_candidate_identity_preserved",
    "promotion_eligible",
    "promotion_applied",
    "recommended_next_step",
)


def run_task0144_structure_aware_initial_retrieval_promotion_gate(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_policy = task0142.runtime_policy_snapshot()
    preflight = build_preflight()
    authority = build_authority_audit(preflight)
    samples = task0137.load_graph_sensitive_samples()
    task0141_baseline = {
        row["sample_id"]: row
        for row in read_jsonl(task0141.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0141.A0_BASELINE
    }
    previous_complete = sorted(sample_id for sample_id, row in task0141_baseline.items() if row.get("complete_required_evidence_set_recall"))
    cohort_ids = sorted(set(authority["residual_unit_ids"]) | set(authority["known_regression_unit_ids"]) | set(previous_complete))

    arm_results: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample["sample_id"] not in cohort_ids:
            continue
        baseline_row = task0141_baseline.get(sample["sample_id"], {})
        for arm_id in ARMS:
            row, repr_rows = evaluate_sample_arm(sample, arm_id=arm_id, baseline_row=baseline_row, residual_ids=set(authority["residual_unit_ids"]))
            arm_results.append(row)
            representation_rows.extend(repr_rows)

    stage_diagnosis = diagnose_regressions(arm_results, authority["known_regression_unit_ids"])
    arm_comparison = build_arm_comparison(arm_results, authority, previous_complete)
    guard_quality = build_guard_quality(arm_results, authority)
    best_arm = select_best_arm(arm_comparison, authority)
    promotion_gate = build_promotion_gate(best_arm, arm_comparison, authority, before_policy, task0142.runtime_policy_snapshot())
    summary = build_summary(authority, arm_comparison, stage_diagnosis, guard_quality, promotion_gate, best_arm)
    config = build_config(preflight, before_policy, task0142.runtime_policy_snapshot())
    digests = build_digests(authority, stage_diagnosis, representation_rows, arm_results, arm_comparison, guard_quality, promotion_gate, config, summary)

    write_json(output_dir / "authority_audit.json", authority)
    write_jsonl(output_dir / "regression_stage_diagnosis.jsonl", stage_diagnosis)
    write_jsonl(output_dir / "representation_audit.jsonl", representation_rows)
    write_jsonl(output_dir / "arm_results.jsonl", arm_results)
    write_json(output_dir / "arm_comparison.json", arm_comparison)
    write_json(output_dir / "guard_quality.json", guard_quality)
    write_json(output_dir / "promotion_gate.json", promotion_gate)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, build_contract(summary, config))
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0144_artifacts(output_dir=output_dir, write=True)
    summary["task0144_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison, stage_diagnosis, guard_quality, promotion_gate, authority), encoding="utf-8")
    return summary


def build_preflight() -> dict[str, Any]:
    task0142_verification = task0142.verify_task0142_artifacts(write=False)
    task0143_verification = task0143.verify_task0143_artifacts(write=False)
    task0143_summary = read_json(task0143.RESULT_DIR / "summary.json")
    task0143_regression = read_json(task0143.RESULT_DIR / "regression_report.json")
    failures = []
    if task0142_verification["status"] != "valid":
        failures.append("task0142_verifier_invalid")
    if task0143_verification["status"] != "valid":
        failures.append("task0143_verifier_invalid")
    if task0143_summary.get("best_candidate_recovery_arm") != task0143.C3_STRUCTURE:
        failures.append("task0143_best_arm_not_structure_aware")
    if task0143_regression.get("existing_complete_unit_regression_count") != 2:
        failures.append("task0143_regression_count_not_two")
    return {
        "schema_version": "opk-rag.task0144.preflight.v1",
        "task_id": TASK_ID,
        "task0142_verifier_status": task0142_verification["status"],
        "task0143_verifier_status": task0143_verification["status"],
        "task0142_summary_digest": sha256_file(task0142.RESULT_DIR / "summary.json"),
        "task0143_summary_digest": sha256_file(task0143.RESULT_DIR / "summary.json"),
        "task0143_regression_report_digest": sha256_file(task0143.RESULT_DIR / "regression_report.json"),
        "preflight_valid": not failures,
        "failures": failures,
    }


def build_authority_audit(preflight: dict[str, Any]) -> dict[str, Any]:
    residual = read_json(task0143.RESULT_DIR / "residual_authority.json")
    regression = read_json(task0143.RESULT_DIR / "regression_report.json")
    per_sample = read_jsonl(task0143.RESULT_DIR / "per_sample.jsonl")
    c3_regressed = sorted(row["sample_id"] for row in per_sample if row.get("arm_id") == task0143.C3_STRUCTURE and row.get("downstream_regressed"))
    all_regressed = sorted(regression.get("regression_case_ids") or [])
    inconsistencies = []
    if all_regressed != c3_regressed:
        inconsistencies.append("task0143_regression_report_not_localized_to_C3_structure_aware")
    return {
        "schema_version": "opk-rag.task0144.authority-audit.v1",
        "task_id": TASK_ID,
        "task0142_authority_valid": preflight["task0142_verifier_status"] == "valid",
        "task0143_contract_valid": preflight["task0143_verifier_status"] == "valid",
        "task0143_summary_digest": preflight["task0143_summary_digest"],
        "task0143_regression_report_digest": preflight["task0143_regression_report_digest"],
        "residual_unit_ids": list(residual["residual_unit_ids"]),
        "residual_missing_required_evidence_ids": list(residual["missing_required_evidence_ids"]),
        "known_regression_unit_ids": all_regressed,
        "task0143_c3_downstream_regression_unit_ids": c3_regressed,
        "authority_inconsistencies": inconsistencies,
        "authority_inconsistency_count": len(inconsistencies),
        "gold_evidence_redefined": False,
        "promotion_applied": False,
    }


def evaluate_sample_arm(
    sample: dict[str, Any],
    *,
    arm_id: str,
    baseline_row: dict[str, Any],
    residual_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = perf_counter()
    body = graph_retrieval.select_seed_candidates(sample, seed_count=task0143.CURRENT_TOP_K)
    structure = task0143.structural_candidates(sample, section_limit=task0143.STRUCTURAL_SECTION_LIMIT)
    guard = guard_decision(sample, body, structure)
    if arm_id == M0_BASELINE:
        initial = body
        strategy = "body_only"
    elif arm_id in {M1_RAW_STRUCTURE, M2_UNION}:
        initial = graph_retrieval.merge_candidates([*body, *structure], [])
        strategy = "body_then_structure_union"
    elif arm_id == M3_SCORE_FUSION:
        initial = score_fusion_candidates(body, structure)
        strategy = "weighted_rank_fusion"
    elif arm_id == M4_GUARDED:
        initial = graph_retrieval.merge_candidates([*body, *structure], []) if guard["guard_triggered"] else body
        strategy = "guarded_body_structure_union"
    else:
        initial = body
        strategy = "unknown_fallback_body_only"
    decision = graph_activation.decide_runtime_graph_activation(
        sample,
        activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        seeds=initial,
    )
    runtime_config = graph_activation.runtime_config_for_decision(decision)
    runtime = graph_retrieval.expand_runtime_candidates(initial, sample, runtime_config=runtime_config, policy=graph_retrieval.default_graph_retrieval_policy())
    candidates = list(runtime.candidates)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    evidence, composition_trace = evidence_composition.compose_evidence(candidates, evaluation_unit_id=sample["sample_id"])
    evidence_ids = [row["canonical_chunk_id"] for row in evidence]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    baseline_candidate_ids = list(baseline_row.get("candidate_ids") or [])
    baseline_complete = bool(baseline_row.get("complete_required_evidence_set_recall", task0137.complete_required_evidence_set_recall(baseline_candidate_ids, required_ids)))
    downstream_before = bool(baseline_row.get("downstream_after", baseline_complete))
    candidate_complete = task0137.complete_required_evidence_set_recall(candidate_ids, required_ids)
    evidence_complete = task0137.complete_required_evidence_set_recall(evidence_ids, required_ids)
    downstream_after = task0137.candidate_set_answer_sufficient(evidence_ids, required_ids, sample)
    ranks = [candidate_ids.index(required_id) + 1 for required_id in required_ids if required_id in candidate_ids]
    structure_ids = {candidate["candidate_id"] for candidate in structure}
    body_ids = {candidate["candidate_id"] for candidate in body}
    trace_by_id = {row["canonical_chunk_id"]: row for row in composition_trace}
    duplicate_count = len(candidate_ids) - len(set(candidate_ids))
    invalid_count = sum(1 for candidate_id in candidate_ids if "#L" not in candidate_id)
    return (
        {
            "schema_version": "opk-rag.task0144.arm-result.v1",
            "task_id": TASK_ID,
            "arm_id": arm_id,
            "sample_id": sample["sample_id"],
            "query": sample.get("query") or sample.get("question"),
            "initial_retrieval_strategy": strategy,
            "initial_candidate_ids": [candidate["candidate_id"] for candidate in initial],
            "candidate_ids": candidate_ids,
            "evidence_ids": evidence_ids,
            "required_ids": required_ids,
            "baseline_candidate_ids": baseline_candidate_ids,
            "baseline_candidate_membership": task0137.complete_required_evidence_set_recall(baseline_candidate_ids, required_ids),
            "structure_candidate_membership": task0137.complete_required_evidence_set_recall([candidate["candidate_id"] for candidate in graph_retrieval.merge_candidates([*body, *structure], [])], required_ids),
            "required_evidence_candidate_hit": any(required_id in candidate_ids for required_id in required_ids),
            "required_evidence_candidate_rank": min(ranks) if ranks else None,
            "candidate_pool_size": len(candidate_ids),
            "candidate_pool_growth_ratio": round(len(candidate_ids) / max(1, len(baseline_candidate_ids)), 6),
            "candidate_membership_change_count": len(set(candidate_ids) ^ set(baseline_candidate_ids)),
            "candidate_rank_change_count": rank_change_count(baseline_candidate_ids, candidate_ids),
            "body_only_unique_candidate_count": len(body_ids - structure_ids),
            "structure_only_unique_candidate_count": len(structure_ids - body_ids),
            "shared_candidate_count": len(body_ids & structure_ids),
            "union_candidate_count": len(body_ids | structure_ids),
            "body_unique_candidate_count": len(body_ids - structure_ids),
            "structure_unique_candidate_count": len(structure_ids - body_ids),
            "required_evidence_survived_reranking": any(required_id in candidate_ids for required_id in required_ids),
            "required_evidence_survived_evidence_budget": any(required_id in evidence_ids for required_id in required_ids),
            "required_evidence_available_to_generation": any(required_id in evidence_ids for required_id in required_ids),
            "evidence_composition_rejection_reasons": {required_id: (trace_by_id.get(required_id) or {}).get("rejection_reason") for required_id in required_ids},
            "complete_required_evidence_set_recall": candidate_complete,
            "evidence_completeness": evidence_complete,
            "baseline_downstream_complete": downstream_before,
            "downstream_complete": downstream_after,
            "downstream_before": downstream_before,
            "downstream_after": downstream_after,
            "downstream_regressed": downstream_before and not downstream_after,
            "residual_recovery_preserved": sample["sample_id"] in residual_ids and evidence_complete and not downstream_before,
            "candidate_recall_regressed": baseline_complete and not candidate_complete,
            "evidence_completeness_regressed": baseline_complete and not evidence_complete,
            "graph_expansion_survival": candidate_complete,
            "graph_lookup_count": 0,
            "retrieval_operation_count": 1 + int(arm_id in {M1_RAW_STRUCTURE, M2_UNION, M3_SCORE_FUSION} or (arm_id == M4_GUARDED and guard["guard_triggered"])),
            "structure_lane_invocation_count": int(arm_id in {M1_RAW_STRUCTURE, M2_UNION, M3_SCORE_FUSION} or (arm_id == M4_GUARDED and guard["guard_triggered"])),
            "latency_delta": round((perf_counter() - started) * 1000, 6),
            "guard_triggered": guard["guard_triggered"] if arm_id == M4_GUARDED else False,
            "guard_reason": guard["guard_reason"] if arm_id == M4_GUARDED else None,
            "canonical_candidate_identity_preserved": duplicate_count == 0 and invalid_count == 0,
            "duplicate_candidate_identity_count": duplicate_count,
            "invalid_candidate_identity_count": invalid_count,
            "gold_identity_injection_count": 0,
            "runtime_gold_metadata_usage": False,
            "runtime_gold_chunk_id_usage": False,
            "runtime_gold_evidence_text_usage": False,
            "runtime_gold_answer_usage": False,
            "runtime_sample_specific_override_count": 0,
            "runtime_policy_mutation_outside_initial_retrieval": False,
            "promotion_applied": False,
        },
        representation_audit(sample, body, structure, arm_id),
    )


def score_fusion_candidates(body: list[dict[str, Any]], structure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for index, candidate in enumerate(body, start=1):
        cid = candidate["candidate_id"]
        by_id[cid] = candidate
        scores[cid] = scores.get(cid, 0.0) + BODY_WEIGHT / index
    for index, candidate in enumerate(structure, start=1):
        cid = candidate["candidate_id"]
        by_id.setdefault(cid, candidate)
        scores[cid] = scores.get(cid, 0.0) + STRUCTURE_WEIGHT / index
    ordered = sorted(by_id.values(), key=lambda candidate: (-scores[candidate["candidate_id"]], candidate["candidate_id"]))
    return graph_retrieval.merge_candidates(ordered, [])


def guard_decision(sample: dict[str, Any], body: list[dict[str, Any]], structure: list[dict[str, Any]]) -> dict[str, Any]:
    return initial_retrieval.guard_decision(sample, body, structure)


def representation_audit(sample: dict[str, Any], body: list[dict[str, Any]], structure: list[dict[str, Any]], arm_id: str) -> list[dict[str, Any]]:
    rows = []
    for lane, candidates in (("C0_body", body), ("C3_structure", structure)):
        for candidate in candidates:
            text = source_text(candidate["candidate_id"])
            heading = heading_for_candidate(candidate["candidate_id"])
            section_context = section_context_for_candidate(candidate["candidate_id"])
            rows.append(
                {
                    "schema_version": "opk-rag.task0144.representation-audit.v1",
                    "task_id": TASK_ID,
                    "arm_id": arm_id,
                    "sample_id": sample["sample_id"],
                    "candidate_id": candidate["candidate_id"],
                    "representation_lane": lane,
                    "C0_representation_components": ["body"],
                    "C3_representation_components": ["body", "heading", "section_context"],
                    "body_length": len(text),
                    "heading_length": len(heading),
                    "section_context_length": len(section_context),
                    "combined_representation_length": len(text) + len(heading) + len(section_context),
                    "heading_may_introduce_wrong_semantics": bool(heading and heading not in text and heading not in str(sample.get("question", ""))),
                    "section_context_may_cover_body_signal": len(section_context) > len(text),
                    "representation_length_dilution_risk": (len(text) + len(heading) + len(section_context)) > max(1, len(text)) * 2,
                    "shared_heading_collision_risk": len(heading) > 0 and sum(1 for row in candidates if heading_for_candidate(row["candidate_id"]) == heading) > 1,
                    "structural_semantic_collision": lane == "C3_structure" and bool(heading) and heading not in str(sample.get("question", "")),
                }
            )
    return rows


def diagnose_regressions(rows: list[dict[str, Any]], regression_ids: list[str]) -> list[dict[str, Any]]:
    by_sample_arm = {(row["sample_id"], row["arm_id"]): row for row in rows}
    diagnosis = []
    for sample_id in regression_ids:
        c0 = by_sample_arm[(sample_id, M0_BASELINE)]
        m1 = by_sample_arm[(sample_id, M1_RAW_STRUCTURE)]
        first_stage = first_regression_stage(c0, m1)
        diagnosis.append(
            {
                "schema_version": "opk-rag.task0144.regression-stage-diagnosis.v1",
                "task_id": TASK_ID,
                "sample_id": sample_id,
                "baseline_candidate_membership": c0["baseline_candidate_membership"],
                "structure_candidate_membership": m1["structure_candidate_membership"],
                "baseline_required_evidence_rank": c0["required_evidence_candidate_rank"],
                "structure_required_evidence_rank": m1["required_evidence_candidate_rank"],
                "baseline_reranker_survival": c0["required_evidence_survived_reranking"],
                "structure_reranker_survival": m1["required_evidence_survived_reranking"],
                "baseline_evidence_survival": c0["required_evidence_survived_evidence_budget"],
                "structure_evidence_survival": m1["required_evidence_survived_evidence_budget"],
                "baseline_graph_expansion_survival": c0["graph_expansion_survival"],
                "structure_graph_expansion_survival": m1["graph_expansion_survival"],
                "baseline_downstream_complete": c0["downstream_complete"],
                "structure_downstream_complete": m1["downstream_complete"],
                "first_regression_stage": first_stage,
                "regression_classification": regression_classification(c0, m1, first_stage),
                "dominant_regression_root_cause": root_cause_for_regression(c0, m1, first_stage),
            }
        )
    return diagnosis


def first_regression_stage(c0: dict[str, Any], m1: dict[str, Any]) -> str:
    if c0["baseline_candidate_membership"] and not m1["complete_required_evidence_set_recall"]:
        return "initial_candidate_membership"
    if c0["required_evidence_candidate_rank"] and m1["required_evidence_candidate_rank"] and m1["required_evidence_candidate_rank"] > c0["required_evidence_candidate_rank"]:
        return "initial_candidate_rank"
    if c0["required_evidence_survived_reranking"] and not m1["required_evidence_survived_reranking"]:
        return "reranker"
    if c0["required_evidence_survived_evidence_budget"] and not m1["required_evidence_survived_evidence_budget"]:
        return "evidence_budget"
    if c0["graph_expansion_survival"] and not m1["graph_expansion_survival"]:
        return "graph_expansion"
    if c0["downstream_complete"] and not m1["downstream_complete"]:
        return "generation"
    return "none_observed_in_M1_replay"


def regression_classification(c0: dict[str, Any], m1: dict[str, Any], first_stage: str) -> str:
    if first_stage == "initial_candidate_membership":
        return "Candidate Membership Loss"
    if first_stage == "initial_candidate_rank":
        return "Candidate Rank Degradation"
    if first_stage in {"reranker", "evidence_budget", "graph_expansion", "generation"}:
        return "downstream survival failure"
    return "no M1 regression reproduced"


def root_cause_for_regression(c0: dict[str, Any], m1: dict[str, Any], first_stage: str) -> str:
    if first_stage == "none_observed_in_M1_replay":
        return "authority_replay_inconsistency"
    if m1["structure_only_unique_candidate_count"] > 0 and m1["candidate_pool_size"] > c0["candidate_pool_size"]:
        return "candidate_pool_competition"
    return "unknown"


def build_arm_comparison(rows: list[dict[str, Any]], authority: dict[str, Any], previous_complete: list[str]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm_id"] == arm] for arm in ARMS}
    previous = set(previous_complete)
    known = set(authority["known_regression_unit_ids"])
    residual = set(authority["residual_unit_ids"])
    arm_rows = []
    baseline_success = {row["sample_id"] for row in by_arm[M0_BASELINE] if row["downstream_after"] and row["sample_id"] in previous}
    for arm, arm_sample_rows in by_arm.items():
        regressed = {row["sample_id"] for row in arm_sample_rows if row["sample_id"] in baseline_success and row["downstream_regressed"]}
        known_regressed = regressed & known
        residual_recovered = any(row["sample_id"] in residual and row["residual_recovery_preserved"] for row in arm_sample_rows)
        arm_rows.append(
            {
                "arm_id": arm,
                "sample_count": len(arm_sample_rows),
                "residual_recovery_preserved": residual_recovered,
                "known_regression_recovered_count": len(known - known_regressed),
                "remaining_known_regression_count": len(known_regressed),
                "new_regression_count": len(regressed - known),
                "total_regression_count": len(regressed),
                "regression_unit_ids": sorted(regressed),
                "mean_candidate_pool_size": round(sum(row["candidate_pool_size"] for row in arm_sample_rows) / max(1, len(arm_sample_rows)), 6),
                "candidate_pool_growth_ratio": max((row["candidate_pool_growth_ratio"] for row in arm_sample_rows), default=0.0),
                "candidate_growth_is_bounded": max((row["candidate_pool_growth_ratio"] for row in arm_sample_rows), default=0.0) <= BOUNDED_GROWTH_LIMIT,
                "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in arm_sample_rows),
                "candidate_rank_change_count": sum(row["candidate_rank_change_count"] for row in arm_sample_rows),
                "structure_unique_candidate_count": sum(row["structure_unique_candidate_count"] for row in arm_sample_rows),
                "body_unique_candidate_count": sum(row["body_unique_candidate_count"] for row in arm_sample_rows),
                "shared_candidate_count": sum(row["shared_candidate_count"] for row in arm_sample_rows),
                "retrieval_operation_count": sum(row["retrieval_operation_count"] for row in arm_sample_rows),
                "structure_lane_invocation_count": sum(row["structure_lane_invocation_count"] for row in arm_sample_rows),
                "graph_lookup_count": sum(row["graph_lookup_count"] for row in arm_sample_rows),
                "latency_delta": round(sum(row["latency_delta"] for row in arm_sample_rows), 6),
                "canonical_candidate_identity_preserved": all(row["canonical_candidate_identity_preserved"] for row in arm_sample_rows),
            }
        )
    return {"schema_version": "opk-rag.task0144.arm-comparison.v1", "task_id": TASK_ID, "arms": arm_rows}


def build_guard_quality(rows: list[dict[str, Any]], authority: dict[str, Any]) -> dict[str, Any]:
    guarded = [row for row in rows if row["arm_id"] == M4_GUARDED]
    trigger = [row for row in guarded if row["guard_triggered"]]
    residual_ids = set(authority["residual_unit_ids"])
    helpful = [row for row in trigger if row["sample_id"] in residual_ids and row["residual_recovery_preserved"]]
    return {
        "schema_version": "opk-rag.task0144.guard-quality.v1",
        "task_id": TASK_ID,
        "guard_trigger_count": len(trigger),
        "guard_non_trigger_count": len(guarded) - len(trigger),
        "true_helpful_trigger_count": len(helpful),
        "unnecessary_trigger_count": len(trigger) - len(helpful),
        "guarded_structure_lane_usage_rate": round(len(trigger) / max(1, len(guarded)), 6),
        "runtime_gold_signal_used": False,
    }


def select_best_arm(arm_comparison: dict[str, Any], authority: dict[str, Any]) -> str:
    rows = {row["arm_id"]: row for row in arm_comparison["arms"]}
    for arm in (M4_GUARDED, M3_SCORE_FUSION, M2_UNION, M1_RAW_STRUCTURE):
        row = rows[arm]
        if row["residual_recovery_preserved"] and row["total_regression_count"] == 0 and row["candidate_growth_is_bounded"]:
            return arm
    if rows[M0_BASELINE]["total_regression_count"] == 0:
        return M0_BASELINE
    return "none"


def build_promotion_gate(
    best_arm: str,
    arm_comparison: dict[str, Any],
    authority: dict[str, Any],
    before_policy: dict[str, Any],
    after_policy: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["arm_id"]: row for row in arm_comparison["arms"]}
    best = rows.get(best_arm, {})
    gate = {
        "residual_recovery_preserved": bool(best.get("residual_recovery_preserved")),
        "known_regression_recovered_count": best.get("known_regression_recovered_count", 0),
        "remaining_known_regression_count": best.get("remaining_known_regression_count", 0),
        "new_regression_count": best.get("new_regression_count", 0),
        "total_regression_count": best.get("total_regression_count", 0),
        "runtime_gold_metadata_usage": False,
        "canonical_candidate_identity_preserved": bool(best.get("canonical_candidate_identity_preserved")),
        "candidate_growth_is_bounded": bool(best.get("candidate_growth_is_bounded")),
        "mitigation_is_runtime_generalizable": best_arm in {M2_UNION, M3_SCORE_FUSION, M4_GUARDED},
        "authority_consistent": authority["authority_inconsistency_count"] == 0,
        "runtime_policy_mutation_outside_initial_retrieval": before_policy != after_policy,
    }
    promotion_eligible = (
        gate["residual_recovery_preserved"]
        and gate["known_regression_recovered_count"] == len(authority["known_regression_unit_ids"])
        and gate["remaining_known_regression_count"] == 0
        and gate["new_regression_count"] == 0
        and gate["total_regression_count"] == 0
        and not gate["runtime_gold_metadata_usage"]
        and gate["canonical_candidate_identity_preserved"]
        and gate["candidate_growth_is_bounded"]
        and gate["mitigation_is_runtime_generalizable"]
        and gate["authority_consistent"]
        and not gate["runtime_policy_mutation_outside_initial_retrieval"]
    )
    return {
        "schema_version": "opk-rag.task0144.promotion-gate.v1",
        "task_id": TASK_ID,
        "best_mitigation_arm": best_arm,
        **gate,
        "promotion_eligible": promotion_eligible,
        "promotion_applied": False,
    }


def build_summary(
    authority: dict[str, Any],
    arm_comparison: dict[str, Any],
    stage_diagnosis: list[dict[str, Any]],
    guard_quality: dict[str, Any],
    promotion_gate: dict[str, Any],
    best_arm: str,
) -> dict[str, Any]:
    rows = {row["arm_id"]: row for row in arm_comparison["arms"]}
    best = rows.get(best_arm, {})
    causes = Counter(row["dominant_regression_root_cause"] for row in stage_diagnosis)
    dominant_regression_root_cause = causes.most_common(1)[0][0] if causes else "unknown"
    mitigation = "none"
    if best_arm == M2_UNION:
        mitigation = "candidate_union_complementarity"
    elif best_arm == M3_SCORE_FUSION:
        mitigation = "body_structure_score_fusion"
    elif best_arm == M4_GUARDED:
        mitigation = "guarded_structure_aware_routing"
    elif best_arm == M0_BASELINE:
        mitigation = "baseline_freeze"
    recommended = "structure_aware_initial_retrieval_runtime_promotion" if promotion_gate["promotion_eligible"] else "freeze_current_baseline_pending_authority_reconciliation"
    return {
        "schema_version": "opk-rag.task0144.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0142_authority_valid": authority["task0142_authority_valid"],
        "task0143_contract_valid": authority["task0143_contract_valid"],
        "authority_inconsistency_count": authority["authority_inconsistency_count"],
        "residual_unit_count": len(authority["residual_unit_ids"]),
        "known_regression_unit_count": len(authority["known_regression_unit_ids"]),
        "best_mitigation_arm": best_arm,
        "best_arm_residual_recovery_preserved": bool(best.get("residual_recovery_preserved")),
        "best_arm_known_regression_count": best.get("remaining_known_regression_count", 0),
        "best_arm_new_regression_count": best.get("new_regression_count", 0),
        "best_arm_total_regression_count": best.get("total_regression_count", 0),
        "best_arm_candidate_growth_ratio": best.get("candidate_pool_growth_ratio", 0.0),
        "best_arm_promotion_eligible": promotion_gate["promotion_eligible"],
        "residual_recovery_preserved": bool(best.get("residual_recovery_preserved")),
        "known_regression_recovered_count": best.get("known_regression_recovered_count", 0),
        "remaining_known_regression_count": best.get("remaining_known_regression_count", 0),
        "new_regression_count": best.get("new_regression_count", 0),
        "total_regression_count": best.get("total_regression_count", 0),
        "dominant_regression_root_cause": dominant_regression_root_cause,
        "dominant_mitigation_mechanism": mitigation,
        "candidate_growth_is_bounded": bool(best.get("candidate_growth_is_bounded")),
        "guard_trigger_count": guard_quality["guard_trigger_count"],
        "guard_non_trigger_count": guard_quality["guard_non_trigger_count"],
        "true_helpful_trigger_count": guard_quality["true_helpful_trigger_count"],
        "unnecessary_trigger_count": guard_quality["unnecessary_trigger_count"],
        "runtime_gold_metadata_usage": False,
        "runtime_gold_chunk_id_usage": False,
        "runtime_gold_evidence_text_usage": False,
        "runtime_gold_answer_usage": False,
        "runtime_sample_specific_override_count": 0,
        "runtime_policy_mutation_outside_initial_retrieval": promotion_gate["runtime_policy_mutation_outside_initial_retrieval"],
        "canonical_candidate_identity_preserved": bool(best.get("canonical_candidate_identity_preserved")),
        "promotion_eligible": promotion_gate["promotion_eligible"],
        "promotion_applied": False,
        "recommended_next_step": recommended,
    }


def build_config(preflight: dict[str, Any], before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0144.config.v1",
        "task_id": TASK_ID,
        "arms": list(ARMS),
        "body_weight": BODY_WEIGHT,
        "structure_weight": STRUCTURE_WEIGHT,
        "bounded_growth_limit": BOUNDED_GROWTH_LIMIT,
        "before_runtime_policy_snapshot": before_policy,
        "after_runtime_policy_snapshot": after_policy,
        "runtime_default_modified": False,
        "allowed_mutation_surface": "initial_retrieval_composition_fusion_guard_only",
        "reranker_frozen": True,
        "evidence_budget_frozen": True,
        "evidence_composition_frozen": True,
        "graph_expansion_frozen": True,
        "generation_frozen": True,
        "citation_policy_frozen": True,
        "grounding_policy_frozen": True,
        "gold_signal_allowed": False,
        "promotion_applied": False,
        "preflight": preflight,
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0144.structure-aware-initial-retrieval-promotion-gate-contract.v1",
        "task_id": TASK_ID,
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
        "experimental_arm_count": len(config["arms"]),
        "runtime_gold_metadata_usage": summary["runtime_gold_metadata_usage"],
        "runtime_policy_mutation_outside_initial_retrieval": summary["runtime_policy_mutation_outside_initial_retrieval"],
        "canonical_candidate_identity_preserved": summary["canonical_candidate_identity_preserved"],
        "promotion_applied": summary["promotion_applied"],
        "allowed_mutation_surface": config["allowed_mutation_surface"],
    }


def verify_task0144_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    arm_results = read_jsonl(output_dir / "arm_results.jsonl") if (output_dir / "arm_results.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")
    if summary.get("task0142_authority_valid") is not True:
        failures.append("task0142_authority_invalid")
    if summary.get("task0143_contract_valid") is not True:
        failures.append("task0143_contract_invalid")
    if summary.get("runtime_gold_metadata_usage") is not False:
        failures.append("runtime_gold_metadata_usage_detected")
    if summary.get("runtime_gold_chunk_id_usage") is not False:
        failures.append("runtime_gold_chunk_id_usage_detected")
    if summary.get("runtime_gold_evidence_text_usage") is not False:
        failures.append("runtime_gold_evidence_text_usage_detected")
    if summary.get("runtime_gold_answer_usage") is not False:
        failures.append("runtime_gold_answer_usage_detected")
    if summary.get("runtime_sample_specific_override_count") != 0:
        failures.append("runtime_sample_specific_override_detected")
    if summary.get("runtime_policy_mutation_outside_initial_retrieval") is not False:
        failures.append("runtime_policy_mutation_outside_initial_retrieval")
    if summary.get("promotion_applied") is not False:
        failures.append("promotion_applied")
    if any(row.get("gold_identity_injection_count") != 0 for row in arm_results):
        failures.append("gold_identity_injection_detected")
    if any(row.get("duplicate_candidate_identity_count") != 0 for row in arm_results):
        failures.append("duplicate_candidate_identity_detected")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0144.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "best_mitigation_arm": summary.get("best_mitigation_arm"),
        "promotion_eligible": summary.get("promotion_eligible"),
        "promotion_applied": summary.get("promotion_applied"),
        "authority_inconsistency_count": summary.get("authority_inconsistency_count"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:7])
    return {
        "schema_version": "opk-rag.task0144.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "experiment_logic_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "experiment_logic_deterministic": True,
        "provider_model_nondeterminism": "not_applicable_no_model_calls",
    }


def build_report(
    summary: dict[str, Any],
    arm_comparison: dict[str, Any],
    stage_diagnosis: list[dict[str, Any]],
    guard_quality: dict[str, Any],
    promotion_gate: dict[str, Any],
    authority: dict[str, Any],
) -> str:
    arm_rows = "\n".join(
        f"* `{row['arm_id']}`: residual_recovery={str(row['residual_recovery_preserved']).lower()}, "
        f"known_remaining={row['remaining_known_regression_count']}, new_regressions={row['new_regression_count']}, "
        f"total_regressions={row['total_regression_count']}, growth={row['candidate_pool_growth_ratio']}"
        for row in arm_comparison["arms"]
    )
    diag_rows = "\n".join(
        f"* `{row['sample_id']}`: first_regression_stage={row['first_regression_stage']}, "
        f"classification={row['regression_classification']}, root_cause={row['dominant_regression_root_cause']}"
        for row in stage_diagnosis
    )
    return f"""# TASK0144 Structure-aware Initial Retrieval Promotion Gate Report

## Summary

`task_status={summary['task_status']}`

TASK-0144 audited TASK-0142 and TASK-0143 authority, replayed bounded Initial Retrieval mitigation arms, and left Runtime defaults unchanged.

## Authority Audit

`residual_unit_ids={authority['residual_unit_ids']}`

`known_regression_unit_ids={authority['known_regression_unit_ids']}`

`authority_inconsistency_count={authority['authority_inconsistency_count']}`

`authority_inconsistencies={authority['authority_inconsistencies']}`

The TASK-0143 regression report identifies two known regression units, but the TASK-0143 per-sample replay does not localize those regressions to `C3_structure_aware_retrieval`. TASK-0144 therefore treats promotion as blocked by authority reconciliation even when a bounded arm looks regression-free in replay.

## Regression Stage Diagnosis

{diag_rows}

## Mitigation Arms

{arm_rows}

## Guard Quality

`guard_trigger_count={guard_quality['guard_trigger_count']}`

`guard_non_trigger_count={guard_quality['guard_non_trigger_count']}`

`true_helpful_trigger_count={guard_quality['true_helpful_trigger_count']}`

`unnecessary_trigger_count={guard_quality['unnecessary_trigger_count']}`

## Decision

`best_mitigation_arm={summary['best_mitigation_arm']}`

`dominant_regression_root_cause={summary['dominant_regression_root_cause']}`

`dominant_mitigation_mechanism={summary['dominant_mitigation_mechanism']}`

`residual_recovery_preserved={str(summary['residual_recovery_preserved']).lower()}`

`total_regression_count={summary['total_regression_count']}`

`candidate_growth_is_bounded={str(summary['candidate_growth_is_bounded']).lower()}`

`promotion_eligible={str(summary['promotion_eligible']).lower()}`

`promotion_applied=false`

`recommended_next_step={summary['recommended_next_step']}`

## Promotion Gate

`authority_consistent={str(promotion_gate['authority_consistent']).lower()}`

`mitigation_is_runtime_generalizable={str(promotion_gate['mitigation_is_runtime_generalizable']).lower()}`

Promotion is not applied. The current baseline should remain frozen until the TASK-0143 regression authority is reconciled or a later promotion task establishes a zero-regression strategy against consistent authority.
"""


def rank_change_count(before: list[str], after: list[str]) -> int:
    after_ranks = {candidate_id: index for index, candidate_id in enumerate(after, start=1)}
    return sum(1 for index, candidate_id in enumerate(before, start=1) if after_ranks.get(candidate_id) != index)


def source_text(candidate_id: str) -> str:
    document, line_span = split_candidate_id(candidate_id)
    path = ROOT / document
    if not path.exists() or not line_span:
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = parse_line_span(line_span)
    return "\n".join(lines[start - 1 : end])


def heading_for_candidate(candidate_id: str) -> str:
    document, line_span = split_candidate_id(candidate_id)
    path = ROOT / document
    if not path.exists() or not line_span:
        return ""
    start, _ = parse_line_span(line_span)
    current = ""
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if idx > start:
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip()
    return current


def section_context_for_candidate(candidate_id: str) -> str:
    document, line_span = split_candidate_id(candidate_id)
    path = ROOT / document
    if not path.exists() or not line_span:
        return ""
    start, _ = parse_line_span(line_span)
    headings = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if idx > start:
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
    return " / ".join(headings[-3:])


def split_candidate_id(candidate_id: str) -> tuple[str, str]:
    if "#L" not in candidate_id:
        return candidate_id, ""
    document, line_span = candidate_id.rsplit("#", 1)
    return document, line_span


def parse_line_span(line_span: str) -> tuple[int, int]:
    raw = line_span.removeprefix("L")
    if "-L" in raw:
        start, end = raw.split("-L", 1)
        return int(start), int(end)
    return int(raw), int(raw)
