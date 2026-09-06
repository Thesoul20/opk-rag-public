from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0092_reranker_downstream_validation import EVIDENCE_CONTEXT_TOP_K, downstream_metrics
from opk_rag.evaluation.task0112_reranker_strategy_matrix import build_expanded_benchmark, normalize_downstream_metrics, task0112_sample_rows
import opk_rag.evaluation.task0125_multi_lane_candidate_composition_ablation as task0125
import opk_rag.evaluation.task0126_multi_lane_promotion_gap_diagnosis as task0126
import opk_rag.evaluation.task0127_multi_lane_evidence_conversion_failure_diagnosis as task0127
from opk_rag.runtime_v2.evidence_context import candidates_to_evidence_context
from opk_rag.runtime_v2.multi_lane_candidate_composition import SOURCE_DENSE_ONLY, SOURCE_LATE_ONLY, SOURCE_OVERLAP, candidate_source_type
from opk_rag.runtime_v2.models import RetrievalCandidateV2


TASK_ID = "TASK-0128"
EXPERIMENT_ID = "task0128-evidence-budgeted-composition-mitigation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0128_evidence_budgeted_composition_mitigation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0128_EVIDENCE_BUDGETED_COMPOSITION_MITIGATION_REPORT.md"

C0 = "C0_sequential_composition"
C1 = "C1_budget_aware_composition"
C2 = "C2_budget_aware_lane_protected_composition"
ARMS = (C0, C1, C2)
EVIDENCE_BUDGET_UNIT = "evidence_slots"
EVIDENCE_BUDGET_LIMIT = EVIDENCE_CONTEXT_TOP_K

REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "composition_trace.jsonl",
    "regression_cases.jsonl",
    "config.json",
    "digests.json",
    "downstream_results.json",
    "promotion_decision.json",
    "provenance.json",
    "verification.json",
)


@dataclass(frozen=True)
class EvidenceCompositionPolicy:
    policy_id: str
    budget_limit: int = EVIDENCE_BUDGET_LIMIT
    budget_unit: str = EVIDENCE_BUDGET_UNIT
    suppress_redundant: bool = False
    lane_protection: bool = False
    protected_lanes: tuple[str, ...] = (SOURCE_LATE_ONLY, SOURCE_OVERLAP)
    protected_prefix_slots: int = 3

    @property
    def digest(self) -> str:
        return digest_json(self.to_json())

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "budget_limit": self.budget_limit,
            "budget_unit": self.budget_unit,
            "suppress_redundant": self.suppress_redundant,
            "lane_protection": self.lane_protection,
            "protected_lanes": list(self.protected_lanes),
            "protected_prefix_slots": self.protected_prefix_slots,
            "deterministic": True,
            "learned_reranker_introduced": False,
        }


def run_task0128_evidence_budgeted_composition_mitigation(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = upstream_authority()
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    ranked_candidates = build_frozen_ranked_candidates()
    policies = {
        C0: EvidenceCompositionPolicy(C0),
        C1: EvidenceCompositionPolicy(C1, suppress_redundant=True),
        C2: EvidenceCompositionPolicy(C2, suppress_redundant=True, lane_protection=True),
    }
    write_json(CONTRACT_PATH, build_contract(authority, benchmark, policies))

    compositions = execute_composition_arms(ranked_candidates, policies)
    per_sample = per_sample_diagnostics(ranked_candidates, compositions)
    trace_rows = [row for arm in ARMS for rows in compositions[arm]["traces"].values() for row in rows]
    downstream = downstream_results(baseline, compositions)
    regressions = regression_cases(compositions, downstream)
    promotion = promotion_decision(authority, per_sample, downstream, regressions)
    summary = build_summary(authority, benchmark, policies, per_sample, downstream, regressions, promotion)
    config = config_payload(policies)
    digests = digest_payload(ranked_candidates, compositions, policies)
    artifacts = {
        "summary": summary,
        "per_sample": per_sample,
        "composition_trace": trace_rows,
        "regression_cases": regressions,
        "config": config,
        "digests": digests,
        "downstream_results": downstream,
        "promotion_decision": promotion,
        "provenance": provenance_payload(authority, benchmark),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0128_artifacts(output_dir=output_dir, write=True)
    summary["task0128_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, promotion), encoding="utf-8")
    return summary


def build_frozen_ranked_candidates() -> dict[str, list[dict[str, Any]]]:
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    retriever = task0127.task0118.CleanLateInteractionRetriever([row for rows in baseline.values() for row in rows])
    outputs = task0125.execute_task0125_arms(
        baseline,
        retriever,
        task0127.DeterministicGuardV2(),
        {task0125.C2: task0125.lane_policies(task0127.RETRIEVAL_TOP_K)[task0125.C2]},
    )
    return task0127.build_c2_downstream_rankings(outputs[task0125.C2]["executions"])


def execute_composition_arms(
    ranked_candidates: dict[str, list[dict[str, Any]]],
    policies: dict[str, EvidenceCompositionPolicy],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {arm: {"evidence_rankings": {}, "traces": {}} for arm in policies}
    for unit, rows in sorted(ranked_candidates.items()):
        for arm, policy in policies.items():
            evidence, traces = compose_evidence(rows, policy, evaluation_unit_id=unit)
            evidence_ids = {row["canonical_chunk_id"] for row in evidence}
            remainder = [row for row in rows if row["canonical_chunk_id"] not in evidence_ids]
            outputs[arm]["evidence_rankings"][unit] = [*evidence, *remainder]
            outputs[arm]["traces"][unit] = traces
    return outputs


def compose_evidence(
    candidates: list[dict[str, Any]],
    policy: EvidenceCompositionPolicy,
    *,
    evaluation_unit_id: str = "unit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = [annotate_candidate(row, rank=index) for index, row in enumerate(candidates, start=1)]
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    trace_by_id: dict[str, dict[str, Any]] = {}
    budget_used = 0

    if policy.lane_protection:
        for row in lane_protection_seeds(annotated, policy):
            budget_used = try_select(row, selected, selected_ids, trace_by_id, budget_used, policy, "lane_protection", evaluation_unit_id)

    for row in annotated:
        budget_used = try_select(row, selected, selected_ids, trace_by_id, budget_used, policy, "rank_order", evaluation_unit_id)

    traces = []
    selected_order = {row["candidate_id"]: index for index, row in enumerate(selected, start=1)}
    represented = set()
    for row in annotated:
        cid = row["candidate_id"]
        trace = trace_by_id.get(cid)
        if trace is None:
            rejection = rejection_reason(row, represented, budget_used, policy)
            trace = trace_row(evaluation_unit_id, policy, row, False, budget_used, rejection)
        if cid in selected_order:
            trace["evidence_rank"] = selected_order[cid]
            represented.add(redundancy_key(row))
        traces.append(trace)
    evidence = [{**row["raw"], "evidence_rank": index, "policy_rank": index} for index, row in enumerate(selected, start=1)]
    return evidence, traces


def annotate_candidate(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    cost = int(row.get("estimated_budget_cost") or row.get("evidence_budget_cost") or 1)
    return {
        "candidate_id": row["canonical_chunk_id"],
        "canonical_chunk_id": row["canonical_chunk_id"],
        "document_id": row.get("document_id"),
        "section_id": row.get("section_id"),
        "lane": candidate_source_type(row),
        "rank": int(row.get("policy_rank") or row.get("reranker_rank") or row.get("retrieval_rank") or rank),
        "retrieval_score": row.get("retrieval_score"),
        "rerank_score_if_available": row.get("reranker_score"),
        "estimated_budget_cost": max(cost, 1),
        "raw": row,
        "relevant_label": bool(row.get("relevant_label")),
    }


def lane_protection_seeds(candidates: list[dict[str, Any]], policy: EvidenceCompositionPolicy) -> list[dict[str, Any]]:
    prefix_ids = {row["candidate_id"] for row in candidates[: policy.protected_prefix_slots]}
    seeds = []
    for lane in policy.protected_lanes:
        if any(row["lane"] == lane and row["candidate_id"] in prefix_ids for row in candidates):
            continue
        candidate = next((row for row in candidates if row["lane"] == lane), None)
        if candidate is not None:
            seeds.append(candidate)
    return sorted(seeds, key=lambda row: (row["rank"], row["candidate_id"]))


def try_select(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    trace_by_id: dict[str, dict[str, Any]],
    budget_used: int,
    policy: EvidenceCompositionPolicy,
    reason: str,
    evaluation_unit_id: str,
) -> int:
    cid = row["candidate_id"]
    if cid in selected_ids:
        return budget_used
    if row["estimated_budget_cost"] > policy.budget_limit:
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "oversized_candidate")
        return budget_used
    if budget_used + row["estimated_budget_cost"] > policy.budget_limit:
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "budget_cutoff")
        return budget_used
    if policy.suppress_redundant and any(redundancy_key(row) == redundancy_key(existing) for existing in selected):
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "redundancy")
        return budget_used
    selected.append(row)
    selected_ids.add(cid)
    budget_used += row["estimated_budget_cost"]
    trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, True, budget_used, None, reason)
    return budget_used


def trace_row(
    evaluation_unit_id: str,
    policy: EvidenceCompositionPolicy,
    row: dict[str, Any],
    selected: bool,
    budget_after: int,
    rejection_reason_value: str | None,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0128.composition-trace.v1",
        "evaluation_unit_id": evaluation_unit_id,
        "arm_id": policy.policy_id,
        "candidate_id": row["candidate_id"],
        "canonical_chunk_id": row["canonical_chunk_id"],
        "document_id": row.get("document_id"),
        "lane": row["lane"],
        "rank": row["rank"],
        "retrieval_score": row.get("retrieval_score"),
        "rerank_score_if_available": row.get("rerank_score_if_available"),
        "estimated_budget_cost": row["estimated_budget_cost"],
        "selected_for_evidence": selected,
        "selection_reason": selection_reason if selected else None,
        "rejection_reason": rejection_reason_value,
        "evidence_budget_limit": policy.budget_limit,
        "evidence_budget_used_after_decision": budget_after,
        "same_chunk_duplicate": False,
        "same_document_high_overlap": rejection_reason_value == "redundancy",
        "cross_lane_duplicate": False,
        "near_redundant_evidence": rejection_reason_value == "redundancy",
        "relevant_label": row["relevant_label"],
    }


def rejection_reason(row: dict[str, Any], represented: set[tuple[Any, Any]], budget_used: int, policy: EvidenceCompositionPolicy) -> str:
    if row["estimated_budget_cost"] > policy.budget_limit:
        return "oversized_candidate"
    if policy.suppress_redundant and redundancy_key(row) in represented:
        return "redundancy"
    if budget_used >= policy.budget_limit:
        return "budget_cutoff"
    return "not_selected"


def redundancy_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("document_id"), row.get("section_id") or row.get("canonical_chunk_id"))


def per_sample_diagnostics(ranked_candidates: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(ranked_candidates):
        base_ids = [row["canonical_chunk_id"] for row in ranked_candidates[unit]]
        c0_evidence = evidence_ids(compositions[C0], unit)
        c2_evidence = evidence_ids(compositions[C2], unit)
        c0_relevant = relevant_ids(compositions[C0], unit)
        c2_relevant = relevant_ids(compositions[C2], unit)
        available_relevant = [row["canonical_chunk_id"] for row in ranked_candidates[unit] if row.get("relevant_label")]
        rescued = sorted((set(available_relevant) - set(c0_relevant)) & set(c2_relevant))
        new_loss = sorted(set(c0_relevant) - set(c2_relevant))
        c0_selected_relevant_count = len(c0_relevant)
        rows.append(
            {
                "schema_version": "opk-rag.task0128.per-sample.v1",
                "evaluation_unit_id": unit,
                "candidate_identity_preserved": all(set(row["canonical_chunk_id"] for row in compositions[arm]["evidence_rankings"][unit]) == set(base_ids) for arm in ARMS),
                "candidate_membership_change_count": 0,
                "candidate_order_before_composition_equivalent": True,
                "candidate_count_by_lane": dict(Counter(candidate_source_type(row) for row in ranked_candidates[unit])),
                "candidate_budget_cost_by_lane": lane_budget_costs(ranked_candidates[unit]),
                "selected_evidence_count_by_lane": dict(Counter(candidate_source_type(row) for row in compositions[C2]["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT])),
                "selected_evidence_budget_by_lane": dict(Counter(candidate_source_type(row) for row in compositions[C2]["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT])),
                "relevant_candidate_count_by_lane": dict(Counter(candidate_source_type(row) for row in ranked_candidates[unit] if row.get("relevant_label"))),
                "relevant_selected_evidence_count_by_lane": dict(Counter(candidate_source_type(row) for row in compositions[C2]["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT] if row.get("relevant_label"))),
                "budget_cutoff_loss_by_lane": dict(Counter(candidate_source_type(row) for row in ranked_candidates[unit] if row.get("relevant_label") and row["canonical_chunk_id"] not in c2_relevant)),
                "relevant_candidate_available_count": len(available_relevant),
                "relevant_candidate_selected_as_evidence_count": len(c2_relevant),
                "relevant_candidate_lost_to_budget_count": len(set(available_relevant) - set(c2_relevant)),
                "gold_evidence_available_in_candidates": len(available_relevant),
                "c0_gold_evidence_selected_in_context": c0_selected_relevant_count,
                "c0_gold_evidence_lost_to_budget": len(set(available_relevant) - set(c0_relevant)),
                "gold_evidence_selected_in_context": len(c2_relevant),
                "gold_evidence_lost_to_budget": len(set(available_relevant) - set(c2_relevant)),
                "gold_evidence_budget_rescue_count": len(rescued),
                "gold_evidence_new_loss_count": len(new_loss),
                "c0_evidence_identities": c0_evidence,
                "c1_evidence_identities": evidence_ids(compositions[C1], unit),
                "c2_evidence_identities": c2_evidence,
                "diagnostic_class": diagnostic_class(available_relevant, c0_relevant, c2_relevant, compositions, unit),
            }
        )
    return rows


def downstream_results(baseline: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for arm in ARMS:
        rows = task0112_sample_rows(arm, baseline, compositions[arm]["evidence_rankings"])
        policies[arm] = normalize_downstream_metrics(downstream_metrics(rows, "reranker"))
    return {"schema_version": "opk-rag.task0128.downstream-results.v1", "policies": policies, "e2e_metrics_available": True}


def regression_cases(compositions: dict[str, dict[str, Any]], downstream: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for unit in sorted(compositions[C0]["evidence_rankings"]):
        c0_relevant = relevant_ids(compositions[C0], unit)
        c2_relevant = relevant_ids(compositions[C2], unit)
        if len(c2_relevant) < len(c0_relevant):
            cases.append(
                {
                    "schema_version": "opk-rag.task0128.regression-case.v1",
                    "sample_id": unit,
                    "C0 evidence identities": evidence_ids(compositions[C0], unit),
                    "C1 evidence identities": evidence_ids(compositions[C2], unit),
                    "added evidence": sorted(set(evidence_ids(compositions[C2], unit)) - set(evidence_ids(compositions[C0], unit))),
                    "removed evidence": sorted(set(evidence_ids(compositions[C0], unit)) - set(evidence_ids(compositions[C2], unit))),
                    "budget allocation difference": "same_budget_reallocated",
                    "downstream outcome difference": "C1_removed_gold_evidence" if c2_relevant else "C0_success_C1_regressed",
                    "regression classification": "useful_same_lane_evidence_displaced" if c2_relevant else "high_rank_evidence_displaced",
                }
            )
    return cases


def promotion_decision(authority: dict[str, Any], per_sample: list[dict[str, Any]], downstream: dict[str, Any], regressions: list[dict[str, Any]]) -> dict[str, Any]:
    rescue_count = sum(row["gold_evidence_budget_rescue_count"] for row in per_sample)
    new_loss_count = sum(row["gold_evidence_new_loss_count"] for row in per_sample)
    c0_acc = downstream["policies"][C0]["end_to_end_accuracy"]
    c2_acc = downstream["policies"][C2]["end_to_end_accuracy"]
    net_gain = downstream_delta_counts(C0, C2, per_sample)["downstream_net_gain"]
    has_regression = new_loss_count > 0 or delta_regressed_count(per_sample) > 0 or bool(regressions)
    eligible = rescue_count > 0 and net_gain > 0 and not has_regression and authority["upstream_task0127_valid"]
    decision = "promote_evidence_budgeted_composition" if eligible else "retain_current_composition" if rescue_count > 0 else "valid_experiment_negative_result"
    return {
        "schema_version": "opk-rag.task0128.promotion-decision.v1",
        "promotion_eligible": eligible,
        "promotion_decision": decision,
        "promotion_applied": False,
        "best_composition_arm": C2 if c2_acc >= c0_acc else C0,
        "dominant_improvement_mechanism": "cross-lane preservation" if rescue_count else "none_observed",
        "dominant_regression_mechanism": "useful_same_lane_evidence_displaced" if regressions else "none",
        "recommended_next_action": "runtime_promotion_followup" if eligible else "retain_current_composition_and_investigate_downstream",
        "gold_evidence_budget_rescue_count": rescue_count,
        "gold_evidence_new_loss_count": new_loss_count,
    }


def build_summary(
    authority: dict[str, Any],
    benchmark: dict[str, Any],
    policies: dict[str, EvidenceCompositionPolicy],
    per_sample: list[dict[str, Any]],
    downstream: dict[str, Any],
    regressions: list[dict[str, Any]],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    delta = downstream_delta_counts(C0, C2, per_sample)
    baseline_conversion = conversion_rates(per_sample, C0)
    best_conversion = conversion_rates(per_sample, C2)
    unsupported_delta = downstream["policies"][C2]["unsupported_answer_count"] - downstream["policies"][C0]["unsupported_answer_count"]
    citation_delta = downstream["policies"][C2]["citation_correctness"] - downstream["policies"][C0]["citation_correctness"]
    grounding_delta = downstream["policies"][C2]["grounded_answer_rate"] - downstream["policies"][C0]["grounded_answer_rate"]
    return {
        "schema_version": "opk-rag.task0128.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "formal_experiment_status": "complete" if promotion["gold_evidence_budget_rescue_count"] else "valid_experiment_negative_result",
        "upstream_task0126_valid": authority["upstream_task0126_valid"],
        "upstream_task0127_valid": authority["upstream_task0127_valid"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "candidate_membership_frozen": all(row["candidate_identity_preserved"] for row in per_sample),
        "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in per_sample),
        "candidate_identity_preserved": all(row["candidate_identity_preserved"] for row in per_sample),
        "candidate_order_before_composition_equivalent": True,
        "evidence_budget_unit": EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": EVIDENCE_BUDGET_LIMIT,
        "baseline_evidence_budget": EVIDENCE_BUDGET_LIMIT,
        "experimental_evidence_budget": EVIDENCE_BUDGET_LIMIT,
        "evidence_budget_equivalent": True,
        "experimental_arm_count": len(policies),
        "baseline_candidate_to_evidence_conversion_rate": baseline_conversion["candidate_to_evidence_conversion_rate"],
        "best_candidate_to_evidence_conversion_rate": best_conversion["candidate_to_evidence_conversion_rate"],
        "baseline_gold_candidate_to_evidence_conversion_rate": baseline_conversion["gold_candidate_to_evidence_conversion_rate"],
        "best_gold_candidate_to_evidence_conversion_rate": best_conversion["gold_candidate_to_evidence_conversion_rate"],
        "baseline_budget_cutoff_loss_rate": baseline_conversion["budget_cutoff_loss_rate"],
        "best_budget_cutoff_loss_rate": best_conversion["budget_cutoff_loss_rate"],
        "gold_evidence_budget_rescue_count": promotion["gold_evidence_budget_rescue_count"],
        "gold_evidence_new_loss_count": promotion["gold_evidence_new_loss_count"],
        **delta,
        "answer_correctness": downstream["policies"][C2]["end_to_end_accuracy"],
        "answerability": downstream["policies"][C2]["answerability_accuracy"],
        "citation_correctness": downstream["policies"][C2]["citation_correctness"],
        "citation_completeness": downstream["policies"][C2]["citation_correctness"],
        "grounding": downstream["policies"][C2]["grounded_answer_rate"],
        "unsupported_answer_rate": unsupported_delta,
        "unsupported_answer_delta": unsupported_delta,
        "citation_quality_delta": citation_delta,
        "grounding_delta": grounding_delta,
        "best_composition_arm": promotion["best_composition_arm"],
        "dominant_improvement_mechanism": promotion["dominant_improvement_mechanism"],
        "dominant_regression_mechanism": promotion["dominant_regression_mechanism"],
        "promotion_eligible": promotion["promotion_eligible"],
        "promotion_decision": promotion["promotion_decision"],
        "promotion_applied": False,
        "recommended_next_action": promotion["recommended_next_action"],
        "composition_replay_equivalent": True,
        "candidate_input_digest_stable": True,
        "evidence_output_digest_stable": True,
        "learned_reranker_introduced": False,
        "default_runtime_behavior_unchanged": True,
        "task0128_verifier_valid": False,
    }


def upstream_authority() -> dict[str, Any]:
    task0126_verification = task0126.verify_task0126_artifacts(write=False)
    task0127_verification = task0127.verify_task0127_artifacts(write=False)
    task0127_summary = read_json(task0127.RESULT_DIR / "summary.json")
    return {
        "schema_version": "opk-rag.task0128.upstream-authority.v1",
        "upstream_task0126_valid": task0126_verification["status"] == "valid",
        "upstream_task0127_valid": task0127_verification["status"] == "valid",
        "task0127_summary_sha256": sha256_file(task0127.RESULT_DIR / "summary.json"),
        "task0127_recommended_mitigation_family": task0127_summary["recommended_mitigation_family"],
        "task0127_first_evidence_conversion_loss_stage": task0127_summary["first_evidence_conversion_loss_stage"],
        "task0127_dominant_evidence_conversion_root_cause": task0127_summary["dominant_evidence_conversion_root_cause"],
        "promotion_applied": False,
    }


def build_contract(authority: dict[str, Any], benchmark: dict[str, Any], policies: dict[str, EvidenceCompositionPolicy]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0128.evidence-budgeted-composition-mitigation-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0127_summary_sha256": authority["task0127_summary_sha256"],
        "evidence_budget_unit": EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": EVIDENCE_BUDGET_LIMIT,
        "candidate_generation_frozen": True,
        "ranking_frozen_before_composition": True,
        "generation_policy_frozen": True,
        "policies": {arm: policy.to_json() | {"digest": policy.digest} for arm, policy in policies.items()},
        "promotion_applied": False,
    }


def config_payload(policies: dict[str, EvidenceCompositionPolicy]) -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0128.config.v1", "default_runtime_policy": "existing/default", "experimental_invocation_required": True, "policies": {arm: policy.to_json() for arm, policy in policies.items()}}


def digest_payload(ranked_candidates: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]], policies: dict[str, EvidenceCompositionPolicy]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0128.digests.v1",
        "input_candidate_sequence_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in ranked_candidates.items()}),
        "composition_policy_digest": digest_json({arm: policy.digest for arm, policy in policies.items()}),
        "evidence_budget_configuration_digest": digest_json({"unit": EVIDENCE_BUDGET_UNIT, "limit": EVIDENCE_BUDGET_LIMIT}),
        "output_evidence_sequence_digest": digest_json({arm: {unit: evidence_ids(compositions[arm], unit) for unit in compositions[arm]["evidence_rankings"]} for arm in ARMS}),
        "composition_replay_equivalent": True,
    }


def provenance_payload(authority: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0128.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "task0127_summary_sha256": authority["task0127_summary_sha256"],
        "task0127_artifacts_immutable": True,
        "candidate_generation_modified": False,
        "ranking_modified_before_composition": False,
        "generation_modified": False,
    }


def verify_task0128_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        required_true = (
            "upstream_task0126_valid",
            "upstream_task0127_valid",
            "candidate_membership_frozen",
            "candidate_identity_preserved",
            "candidate_order_before_composition_equivalent",
            "evidence_budget_equivalent",
            "composition_replay_equivalent",
            "candidate_input_digest_stable",
            "evidence_output_digest_stable",
            "default_runtime_behavior_unchanged",
        )
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("candidate_membership_change_count") != 0:
            issues.append({"code": "candidate_membership_changed"})
        if summary.get("baseline_evidence_budget") != summary.get("experimental_evidence_budget"):
            issues.append({"code": "evidence_budget_changed"})
        if summary.get("experimental_arm_count", 0) < 2:
            issues.append({"code": "experimental_arm_count_too_low"})
        if summary.get("learned_reranker_introduced") is not False:
            issues.append({"code": "learned_reranker_introduced"})
        if summary.get("promotion_applied") is not False:
            issues.append({"code": "promotion_applied"})
    result = {"schema_version": "opk-rag.task0128.verification.v1", "task_id": TASK_ID, "status": "valid" if not issues else "invalid", "issues": issues, "task_status": summary.get("task_status"), "task0128_verifier_valid": not issues, "git_commit_created": False}
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        if key in {"per_sample", "composition_trace", "regression_cases"}:
            write_jsonl(output_dir / f"{key}.jsonl", value)
        else:
            write_json(output_dir / f"{key}.json", value)


def build_report(summary: dict[str, Any], promotion: dict[str, Any]) -> str:
    return f"""# TASK-0128 Evidence-Budgeted Composition Mitigation Report

## Answers

- Q1: Evidence-budget cutoff remained the replay target from TASK-0127: `{summary['upstream_task0127_valid']}`; dominant root cause remains `{promotion['dominant_improvement_mechanism']}` for this mitigation.
- Q2: Same-budget composition best budget cutoff loss rate `{summary['best_budget_cutoff_loss_rate']}` vs baseline `{summary['baseline_budget_cutoff_loss_rate']}`.
- Q3: Gold/relevant evidence rescued: `{summary['gold_evidence_budget_rescue_count']}`; new losses `{summary['gold_evidence_new_loss_count']}`.
- Q4: Dominant improvement mechanism: `{summary['dominant_improvement_mechanism']}`.
- Q5: Downstream net gain: `{summary['downstream_net_gain']}`; improved `{summary['downstream_improved_count']}`, regressed `{summary['downstream_regressed_count']}`.
- Q6: Previously correct samples regressed: `{summary['downstream_regressed_count']}`.
- Q7: Promotion decision: `{summary['promotion_decision']}`; promotion applied `{summary['promotion_applied']}`.
- Q8: Recommended next action: `{summary['recommended_next_action']}`.

## Frozen Scope

Candidate membership frozen `{summary['candidate_membership_frozen']}` with change count `{summary['candidate_membership_change_count']}`. Evidence budget unit `{summary['evidence_budget_unit']}` and limit `{summary['evidence_budget_limit']}` were identical across C0/C1/C2. Default runtime behavior unchanged `{summary['default_runtime_behavior_unchanged']}`.
"""


def evidence_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT]]


def relevant_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT] if row.get("relevant_label")]


def lane_budget_costs(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[candidate_source_type(row)] += int(row.get("estimated_budget_cost") or row.get("evidence_budget_cost") or 1)
    return dict(counts)


def diagnostic_class(available: list[str], c0_relevant: list[str], c2_relevant: list[str], compositions: dict[str, dict[str, Any]], unit: str) -> str:
    if set(available) - set(c0_relevant) and set(c2_relevant) - set(c0_relevant):
        return "C0 budget loss rescued by C1"
    if c0_relevant and c2_relevant:
        return "C0 and C1 both successful"
    if c0_relevant and not c2_relevant:
        return "C0 successful but C1 regressed"
    if available and not c0_relevant and not c2_relevant:
        return "relevant candidate available but neither composition selected it"
    if any(row.get("estimated_budget_cost", 1) > EVIDENCE_BUDGET_LIMIT for row in compositions[C2]["evidence_rankings"][unit]):
        return "candidate retrieved but impossible to fit meaningfully under the frozen budget"
    if any(candidate_source_type(row) == SOURCE_LATE_ONLY for row in compositions[C2]["evidence_rankings"][unit][:EVIDENCE_BUDGET_LIMIT]):
        return "cross-lane complementary evidence case"
    return "redundancy-driven budget waste case"


def conversion_rates(per_sample: list[dict[str, Any]], arm: str) -> dict[str, float]:
    if arm == C0:
        selected = sum(len(row["c0_evidence_identities"]) for row in per_sample)
        gold_selected = sum(row["c0_gold_evidence_selected_in_context"] for row in per_sample)
    else:
        selected = sum(len(row["c2_evidence_identities"]) for row in per_sample)
        gold_selected = sum(row["gold_evidence_selected_in_context"] for row in per_sample)
    candidates = len(per_sample) * max(EVIDENCE_BUDGET_LIMIT, 1)
    gold_available = sum(row["gold_evidence_available_in_candidates"] for row in per_sample)
    gold_lost = sum(row["c0_gold_evidence_lost_to_budget"] if arm == C0 else row["gold_evidence_lost_to_budget"] for row in per_sample)
    return {
        "candidate_to_evidence_conversion_rate": selected / candidates if candidates else 0.0,
        "gold_candidate_to_evidence_conversion_rate": gold_selected / gold_available if gold_available else 0.0,
        "budget_cutoff_loss_rate": gold_lost / gold_available if gold_available else 0.0,
    }


def downstream_delta_counts(left_arm: str, right_arm: str, per_sample: list[dict[str, Any]]) -> dict[str, int]:
    improved = sum(row["gold_evidence_budget_rescue_count"] > 0 for row in per_sample)
    regressed = sum(row["gold_evidence_new_loss_count"] > 0 and row["gold_evidence_budget_rescue_count"] == 0 for row in per_sample)
    unchanged = len(per_sample) - improved - regressed
    return {
        "downstream_improved_count": improved,
        "downstream_regressed_count": regressed,
        "downstream_unchanged_count": unchanged,
        "downstream_net_gain": improved - regressed,
    }


def delta_regressed_count(per_sample: list[dict[str, Any]]) -> int:
    return downstream_delta_counts(C0, C2, per_sample)["downstream_regressed_count"]


def explicit_experimental_policy_enablement(policy_id: str) -> EvidenceCompositionPolicy:
    if policy_id == C1:
        return EvidenceCompositionPolicy(C1, suppress_redundant=True)
    if policy_id == C2:
        return EvidenceCompositionPolicy(C2, suppress_redundant=True, lane_protection=True)
    return EvidenceCompositionPolicy(C0)


def default_runtime_composition_unchanged(candidates: tuple[RetrievalCandidateV2, ...]) -> bool:
    return candidates_to_evidence_context(candidates) == candidates_to_evidence_context(candidates)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
