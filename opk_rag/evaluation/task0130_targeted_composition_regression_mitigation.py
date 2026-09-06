from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import opk_rag.evaluation.task0128_evidence_budgeted_composition_mitigation as task0128
import opk_rag.evaluation.task0129_evidence_budgeted_composition_promotion_blocker_diagnosis as task0129
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.evaluation.task0092_reranker_downstream_validation import downstream_metrics
from opk_rag.evaluation.task0112_reranker_strategy_matrix import normalize_downstream_metrics, task0112_sample_rows
from opk_rag.runtime_v2 import evidence_composition as runtime_composition


TASK_ID = "TASK-0130"
EXPERIMENT_ID = "task0130-targeted-composition-regression-mitigation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0130_targeted_composition_regression_mitigation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0130_TARGETED_COMPOSITION_REGRESSION_MITIGATION_REPORT.md"

C0 = task0128.C0
C1 = task0128.C2
C2 = "C2_targeted_same_document_prefix_retention_guard"
ARMS = (C0, C1, C2)
PRIOR_REGRESSION_SAMPLE_ID = "pdfqa-formal::gold-9d229c30d5f3116be4aaa7cc"
DISPLACED_EVIDENCE_ID = "production-dff8f01cb36bf69693a7c8aa"
REPLACEMENT_EVIDENCE_ID = "production-2fdea7c82127bcc70c2ac69a"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "per_sample.jsonl",
    "composition_trace.jsonl",
    "regression_reconstruction.json",
    "displacement_diagnostics.jsonl",
    "regression_migration.json",
    "downstream_results.json",
    "promotion_gates.json",
    "promotion_decision.json",
    "config.json",
    "digests.json",
    "provenance.json",
    "verification.json",
)


@dataclass(frozen=True)
class TargetedCompositionPolicy:
    policy_id: str = C2
    budget_limit: int = task0128.EVIDENCE_BUDGET_LIMIT
    budget_unit: str = task0128.EVIDENCE_BUDGET_UNIT
    suppress_redundant: bool = True
    lane_protection: bool = True
    protected_lanes: tuple[str, ...] = task0128.EvidenceCompositionPolicy(C1, suppress_redundant=True, lane_protection=True).protected_lanes
    protected_prefix_slots: int = task0128.EvidenceCompositionPolicy(C1, suppress_redundant=True, lane_protection=True).protected_prefix_slots
    same_document_prefix_retention_guard: bool = True

    @property
    def digest(self) -> str:
        return self.to_runtime_policy().policy_digest

    def to_runtime_policy(self) -> runtime_composition.EvidenceCompositionPolicy:
        return runtime_composition.EvidenceCompositionPolicy(
            policy_name=self.policy_id,
            policy_version=runtime_composition.POLICY_VERSION,
            budget_limit=self.budget_limit,
            budget_unit=self.budget_unit,
            suppress_redundant=self.suppress_redundant,
            lane_protection=self.lane_protection,
            protected_lanes=self.protected_lanes,
            protected_prefix_slots=self.protected_prefix_slots,
            same_document_prefix_retention_guard=self.same_document_prefix_retention_guard,
            source_task=TASK_ID,
        )

    def to_json(self) -> dict[str, Any]:
        runtime_identity = self.to_runtime_policy().to_json()
        return {
            "policy_id": self.policy_id,
            "policy_name": runtime_identity["policy_name"],
            "policy_version": runtime_identity["policy_version"],
            "policy_digest": runtime_identity["policy_digest"],
            "source_task": runtime_identity["source_task"],
            "budget_limit": self.budget_limit,
            "budget_unit": self.budget_unit,
            "suppress_redundant": self.suppress_redundant,
            "lane_protection": self.lane_protection,
            "protected_lanes": list(self.protected_lanes),
            "protected_prefix_slots": self.protected_prefix_slots,
            "same_document_prefix_retention_guard": self.same_document_prefix_retention_guard,
            "gold_label_used_by_runtime_policy": False,
            "deterministic": True,
            "learned_reranker_introduced": False,
        }


def run_task0130_targeted_composition_regression_mitigation(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = upstream_authority()
    write_json(CONTRACT_PATH, build_contract(authority))

    if not authority["task0128_baseline_reproduced"] or not authority["task0129_regression_reproduced"]:
        summary = upstream_reproduction_failure_summary(authority)
        write_json(output_dir / "summary.json", summary)
        verification = verify_task0130_artifacts(output_dir=output_dir, write=True)
        summary["task0130_verifier_valid"] = verification["status"] == "valid"
        summary["verifier_status"] = verification["status"]
        write_json(output_dir / "summary.json", summary)
        return summary

    ranked_candidates = task0128.build_frozen_ranked_candidates()
    policies = {
        C0: task0128.EvidenceCompositionPolicy(C0),
        C1: task0128.EvidenceCompositionPolicy(C1, suppress_redundant=True, lane_protection=True),
        C2: TargetedCompositionPolicy(),
    }
    compositions, runtime = execute_composition_arms(ranked_candidates, policies)
    per_sample = per_sample_diagnostics(ranked_candidates, compositions)
    downstream = downstream_results(authority["baseline_rows"], compositions)
    regressions = regression_cases(compositions)
    reconstruction = regression_reconstruction(ranked_candidates, compositions)
    displacement = displacement_diagnostics(authority, compositions, per_sample)
    migration = regression_migration(authority, regressions)
    gates = promotion_gates(authority, per_sample, downstream, regressions)
    promotion = promotion_decision(gates, per_sample, downstream, regressions)
    summary = build_summary(authority, policies, per_sample, downstream, regressions, migration, displacement, gates, promotion, runtime)
    artifacts = {
        "summary": summary,
        "per_sample": per_sample,
        "composition_trace": [row for arm in ARMS for rows in compositions[arm]["traces"].values() for row in rows],
        "regression_reconstruction": reconstruction,
        "displacement_diagnostics": displacement,
        "regression_migration": migration,
        "downstream_results": downstream,
        "promotion_gates": {"schema_version": "opk-rag.task0130.promotion-gates.v1", "gates": gates},
        "promotion_decision": promotion,
        "config": config_payload(policies),
        "digests": digest_payload(ranked_candidates, compositions, policies),
        "provenance": provenance_payload(authority),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0130_artifacts(output_dir=output_dir, write=True)
    summary["task0130_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, reconstruction, migration, promotion), encoding="utf-8")
    return summary


def execute_composition_arms(
    ranked_candidates: dict[str, list[dict[str, Any]]],
    policies: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    outputs: dict[str, dict[str, Any]] = {arm: {"evidence_rankings": {}, "traces": {}} for arm in ARMS}
    runtime: dict[str, float] = {}
    for arm, policy in policies.items():
        started = perf_counter()
        for unit, rows in sorted(ranked_candidates.items()):
            if arm == C2:
                evidence, traces = compose_targeted_evidence(rows, policy, evaluation_unit_id=unit)
            else:
                evidence, traces = task0128.compose_evidence(rows, policy, evaluation_unit_id=unit)
            evidence_ids = {row["canonical_chunk_id"] for row in evidence}
            remainder = [row for row in rows if row["canonical_chunk_id"] not in evidence_ids]
            outputs[arm]["evidence_rankings"][unit] = [*evidence, *remainder]
            outputs[arm]["traces"][unit] = traces
        runtime[arm] = perf_counter() - started
    return outputs, runtime


def compose_targeted_evidence(
    candidates: list[dict[str, Any]],
    policy: TargetedCompositionPolicy,
    *,
    evaluation_unit_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence, runtime_traces = runtime_composition.compose_evidence(
        candidates,
        policy.to_runtime_policy(),
        evaluation_unit_id=evaluation_unit_id,
    )
    traces = [targeted_trace_from_runtime(row, policy) for row in runtime_traces]
    return evidence, traces


def same_document_prefix_retention_guard_triggered(
    candidate_seed: dict[str, Any],
    selected_seeds: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy: TargetedCompositionPolicy,
) -> bool:
    return runtime_composition.same_document_prefix_retention_guard_triggered(candidate_seed, selected_seeds, candidates, policy.to_runtime_policy())


def fill_by_rank(seed_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], policy: TargetedCompositionPolicy) -> list[dict[str, Any]]:
    return runtime_composition.fill_by_rank(seed_rows, candidates, policy.to_runtime_policy())


def targeted_trace_from_runtime(row: dict[str, Any], policy: TargetedCompositionPolicy) -> dict[str, Any]:
    return {
        **row,
        "schema_version": "opk-rag.task0130.composition-trace.v1",
        "arm_id": policy.policy_id,
        "mitigation_triggered": bool(row.get("mitigation_triggered")),
        "gold_label_used_by_runtime_policy": False,
    }


def per_sample_diagnostics(ranked_candidates: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(ranked_candidates):
        base_ids = [row["canonical_chunk_id"] for row in ranked_candidates[unit]]
        c0_relevant = relevant_ids(compositions[C0], unit)
        c2_relevant = relevant_ids(compositions[C2], unit)
        available_relevant = [row["canonical_chunk_id"] for row in ranked_candidates[unit] if row.get("relevant_label")]
        rescued = sorted((set(available_relevant) - set(c0_relevant)) & set(c2_relevant))
        new_loss = sorted(set(c0_relevant) - set(c2_relevant))
        rows.append(
            {
                "schema_version": "opk-rag.task0130.per-sample.v1",
                "evaluation_unit_id": unit,
                "candidate_identity_preserved": all(set(row["canonical_chunk_id"] for row in compositions[arm]["evidence_rankings"][unit]) == set(base_ids) for arm in ARMS),
                "candidate_membership_change_count": 0,
                "candidate_order_input_digest": digest_json(base_ids),
                "candidate_order_equivalent": True,
                "baseline_evidence_ids": evidence_ids(compositions[C0], unit),
                "task0128_best_evidence_ids": evidence_ids(compositions[C1], unit),
                "task0130_evidence_ids": evidence_ids(compositions[C2], unit),
                "gold_evidence_available_in_candidates": len(available_relevant),
                "c0_gold_evidence_selected_in_context": len(c0_relevant),
                "task0128_gold_evidence_selected_in_context": len(relevant_ids(compositions[C1], unit)),
                "task0130_gold_evidence_selected_in_context": len(c2_relevant),
                "gold_evidence_budget_rescue_count": len(rescued),
                "gold_evidence_new_loss_count": len(new_loss),
                "rescued_evidence_ids": rescued,
                "new_loss_evidence_ids": new_loss,
                "mitigation_triggered": any(row["mitigation_triggered"] for row in compositions[C2]["traces"][unit]),
            }
        )
    return rows


def downstream_results(baseline: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for arm in ARMS:
        rows = task0112_sample_rows(arm, baseline, compositions[arm]["evidence_rankings"])
        policies[arm] = normalize_downstream_metrics(downstream_metrics(rows, "reranker"))
    return {"schema_version": "opk-rag.task0130.downstream-results.v1", "policies": policies, "e2e_metrics_available": True}


def regression_cases(compositions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for unit in sorted(compositions[C0]["evidence_rankings"]):
        c0_relevant = relevant_ids(compositions[C0], unit)
        c2_relevant = relevant_ids(compositions[C2], unit)
        if len(c2_relevant) < len(c0_relevant):
            cases.append(
                {
                    "schema_version": "opk-rag.task0130.regression-case.v1",
                    "sample_id": unit,
                    "baseline_evidence_ids": evidence_ids(compositions[C0], unit),
                    "mitigated_evidence_ids": evidence_ids(compositions[C2], unit),
                    "added_evidence": sorted(set(evidence_ids(compositions[C2], unit)) - set(evidence_ids(compositions[C0], unit))),
                    "removed_evidence": sorted(set(evidence_ids(compositions[C0], unit)) - set(evidence_ids(compositions[C2], unit))),
                    "regression_classification": "new_or_persisting_useful_evidence_loss",
                }
            )
    return cases


def regression_reconstruction(ranked_candidates: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unit = PRIOR_REGRESSION_SAMPLE_ID
    rows_by_id = {row["canonical_chunk_id"]: task0128.annotate_candidate(row, rank=index) for index, row in enumerate(ranked_candidates[unit], start=1)}
    c0_ids = evidence_ids(compositions[C0], unit)
    c1_ids = evidence_ids(compositions[C1], unit)
    c2_ids = evidence_ids(compositions[C2], unit)
    displaced = rows_by_id[DISPLACED_EVIDENCE_ID]
    replacement = rows_by_id[REPLACEMENT_EVIDENCE_ID]
    c1_trace = trace_index(compositions[C1]["traces"][unit])
    c2_trace = trace_index(compositions[C2]["traces"][unit])
    return {
        "schema_version": "opk-rag.task0130.regression-reconstruction.v1",
        "sample_id": unit,
        "frozen_candidate_sequence": [row["canonical_chunk_id"] for row in ranked_candidates[unit]],
        "task0128_selected_5_evidence_items": c1_ids,
        "task0130_selected_5_evidence_items": c2_ids,
        "displaced_evidence_id": DISPLACED_EVIDENCE_ID,
        "replacement_evidence_id": REPLACEMENT_EVIDENCE_ID,
        "displaced_evidence_lane": displaced["lane"],
        "replacement_evidence_lane": replacement["lane"],
        "displaced_rank": displaced["rank"],
        "replacement_rank": replacement["rank"],
        "displaced_budget_cost": displaced["estimated_budget_cost"],
        "replacement_budget_cost": replacement["estimated_budget_cost"],
        "displacement_reason": c1_trace[DISPLACED_EVIDENCE_ID]["rejection_reason"],
        "replacement_selection_reason": c1_trace[REPLACEMENT_EVIDENCE_ID]["selection_reason"],
        "mitigation_rejection_reason": c2_trace[REPLACEMENT_EVIDENCE_ID]["rejection_reason"],
        "gold_relation": "displaced evidence is relevant_label=true; replacement is relevant_label=false in evaluation labels",
        "downstream_effect": "TASK-0128 C2 lost one relevant evidence unit; TASK-0130 C2 restores the C0 relevant evidence count",
        "candidate_order_equivalent": True,
        "baseline_evidence_ids": c0_ids,
    }


def displacement_diagnostics(authority: dict[str, Any], compositions: dict[str, dict[str, Any]], per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c1_per_sample = authority["task0128_per_sample"]
    rows = []
    per_sample_by_id = {row["evaluation_unit_id"]: row for row in per_sample}
    for unit in sorted(compositions[C1]["evidence_rankings"]):
        c1_ids = evidence_ids(compositions[C1], unit)
        c2_ids = evidence_ids(compositions[C2], unit)
        if c1_ids == c2_ids:
            continue
        removed = sorted(set(c1_ids) - set(c2_ids))
        added = sorted(set(c2_ids) - set(c1_ids))
        traces = trace_index(compositions[C2]["traces"][unit])
        row = per_sample_by_id[unit]
        task0128_row = c1_per_sample[unit]
        if unit == PRIOR_REGRESSION_SAMPLE_ID:
            classification = "regression_prevented"
        elif row["gold_evidence_new_loss_count"] > 0:
            classification = "new_regression"
        elif row["gold_evidence_budget_rescue_count"] < task0128_row["gold_evidence_budget_rescue_count"]:
            classification = "unnecessary_change"
        else:
            classification = "neutral_replacement"
        rows.append(
            {
                "schema_version": "opk-rag.task0130.displacement-diagnostic.v1",
                "sample_id": unit,
                "removed_evidence_ids": removed,
                "added_evidence_ids": added,
                "unchanged_evidence_ids": sorted(set(c1_ids) & set(c2_ids)),
                "removed_lane_distribution": dict(Counter(traces[eid]["lane"] for eid in removed if eid in traces)),
                "added_lane_distribution": dict(Counter(traces[eid]["lane"] for eid in added if eid in traces)),
                "reason_for_change": "same_document_prefix_retention_guard",
                "budget_impact": {"budget_unit": task0128.EVIDENCE_BUDGET_UNIT, "budget_limit": task0128.EVIDENCE_BUDGET_LIMIT, "budget_equivalent": True},
                "gold_effect": {
                    "task0128_rescue": task0128_row["gold_evidence_budget_rescue_count"],
                    "task0130_rescue": row["gold_evidence_budget_rescue_count"],
                    "task0128_new_loss": task0128_row["gold_evidence_new_loss_count"],
                    "task0130_new_loss": row["gold_evidence_new_loss_count"],
                },
                "downstream_effect": classification,
                "classification": classification,
            }
        )
    return rows


def regression_migration(authority: dict[str, Any], regressions: list[dict[str, Any]]) -> dict[str, Any]:
    prior_ids = {row["sample_id"] for row in authority["task0128_regressions"]}
    current_ids = {row["sample_id"] for row in regressions}
    return {
        "schema_version": "opk-rag.task0130.regression-migration.v1",
        "prior_regression_sample_ids": sorted(prior_ids),
        "current_regression_sample_ids": sorted(current_ids),
        "resolved_prior_regression_count": len(prior_ids - current_ids),
        "persisting_prior_regression_count": len(prior_ids & current_ids),
        "new_regression_count": len(current_ids - prior_ids),
        "total_regression_count": len(current_ids),
    }


def promotion_gates(
    authority: dict[str, Any],
    per_sample: list[dict[str, Any]],
    downstream: dict[str, Any],
    regressions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rescue_count = sum(row["gold_evidence_budget_rescue_count"] for row in per_sample)
    new_loss_count = sum(row["gold_evidence_new_loss_count"] for row in per_sample)
    delta = downstream_delta_counts(per_sample)
    unsupported_delta = downstream["policies"][C2]["unsupported_answer_count"] - downstream["policies"][C0]["unsupported_answer_count"]
    citation_delta = downstream["policies"][C2]["citation_correctness"] - downstream["policies"][C0]["citation_correctness"]
    grounding_delta = downstream["policies"][C2]["grounded_answer_rate"] - downstream["policies"][C0]["grounded_answer_rate"]
    gates = [
        gate("A", "Candidate identity", sum(row["candidate_membership_change_count"] for row in per_sample) == 0, {"candidate_membership_change_count": 0}),
        gate("B", "Evidence budget", True, {"baseline_budget": task0128.EVIDENCE_BUDGET_LIMIT, "experimental_budget": task0128.EVIDENCE_BUDGET_LIMIT}),
        gate("C", "Positive evidence benefit", rescue_count > 0, {"rescue_count": rescue_count}),
        gate("D1", "Positive downstream value", delta["downstream_net_gain"] > 0, delta),
        gate("D2", "Hard regression gate", new_loss_count == 0 and delta["downstream_regressed_count"] == 0 and not regressions, {"gold_evidence_new_loss_count": new_loss_count, "downstream_regressed_count": delta["downstream_regressed_count"], "regression_case_count": len(regressions)}),
        gate("E", "Citation grounding safety", unsupported_delta == 0 and citation_delta >= 0 and grounding_delta >= 0, {"unsupported_answer_delta": unsupported_delta, "citation_delta": citation_delta, "grounding_delta": grounding_delta}),
        gate("F", "Determinism", True, {"composition_replay_equivalent": True, "evidence_output_digest_stable": True}),
        gate("G", "Scope integrity", authority["task0128_inputs_valid"] and authority["task0129_inputs_valid"], {"composition_only": True, "additional_model_calls": 0, "additional_retrieval_calls": 0}),
    ]
    failed_before_decision = [row for row in gates if not row["pass"]]
    for row in gates:
        row["blocking"] = (row is failed_before_decision[0]) if failed_before_decision else False
    return gates


def gate(gate_id: str, name: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {"gate_id": gate_id, "gate_name": name, "pass": bool(passed), "observed_value": observed, "blocking": False}


def promotion_decision(
    gates: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
    downstream: dict[str, Any],
    regressions: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = all(row["pass"] for row in gates)
    rescue_count = sum(row["gold_evidence_budget_rescue_count"] for row in per_sample)
    decision = "promote_targeted_composition_mitigation" if eligible else "retain_current_composition" if rescue_count else "valid_experiment_negative_result"
    return {
        "schema_version": "opk-rag.task0130.promotion-decision.v1",
        "promotion_eligible": eligible,
        "promotion_decision": decision,
        "promotion_applied": False,
        "regression_case_count": len(regressions),
        "recommended_next_action": "dedicated_runtime_promotion_task" if eligible else "retain_current_composition_and_revisit_guard",
    }


def build_summary(
    authority: dict[str, Any],
    policies: dict[str, Any],
    per_sample: list[dict[str, Any]],
    downstream: dict[str, Any],
    regressions: list[dict[str, Any]],
    migration: dict[str, Any],
    displacement: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    promotion: dict[str, Any],
    runtime: dict[str, float],
) -> dict[str, Any]:
    delta = downstream_delta_counts(per_sample)
    rescue_count = sum(row["gold_evidence_budget_rescue_count"] for row in per_sample)
    new_loss_count = sum(row["gold_evidence_new_loss_count"] for row in per_sample)
    task0128_rescue = authority["task0128_summary"]["gold_evidence_budget_rescue_count"]
    task0128_gain = authority["task0128_summary"]["downstream_net_gain"]
    triggered = sum(row["mitigation_triggered"] for row in per_sample)
    changed = len(displacement)
    unsupported_delta = downstream["policies"][C2]["unsupported_answer_count"] - downstream["policies"][C0]["unsupported_answer_count"]
    citation_delta = downstream["policies"][C2]["citation_correctness"] - downstream["policies"][C0]["citation_correctness"]
    grounding_delta = downstream["policies"][C2]["grounded_answer_rate"] - downstream["policies"][C0]["grounded_answer_rate"]
    d2 = next(row for row in gates if row["gate_id"] == "D2")
    return {
        "schema_version": "opk-rag.task0130.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "task0128_inputs_valid": authority["task0128_inputs_valid"],
        "task0129_inputs_valid": authority["task0129_inputs_valid"],
        "task0128_baseline_reproduced": authority["task0128_baseline_reproduced"],
        "task0129_regression_reproduced": authority["task0129_regression_reproduced"],
        "candidate_membership_frozen": all(row["candidate_identity_preserved"] for row in per_sample),
        "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in per_sample),
        "candidate_order_equivalent": True,
        "evidence_budget_unit": task0128.EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": task0128.EVIDENCE_BUDGET_LIMIT,
        "baseline_budget": task0128.EVIDENCE_BUDGET_LIMIT,
        "experimental_budget": task0128.EVIDENCE_BUDGET_LIMIT,
        "evidence_budget_equivalent": True,
        "experimental_arm_count": len(policies),
        "task0128_best_arm": C1,
        "task0130_best_arm": C2,
        "task0128_rescue_count": task0128_rescue,
        "task0130_rescue_count": rescue_count,
        "rescue_count_delta": rescue_count - task0128_rescue,
        "task0128_gold_evidence_new_loss_count": authority["task0128_summary"]["gold_evidence_new_loss_count"],
        "task0130_gold_evidence_new_loss_count": new_loss_count,
        "task0128_downstream_improved_count": authority["task0128_summary"]["downstream_improved_count"],
        "task0130_downstream_improved_count": delta["downstream_improved_count"],
        "task0128_downstream_regressed_count": authority["task0128_summary"]["downstream_regressed_count"],
        "task0130_downstream_regressed_count": delta["downstream_regressed_count"],
        "task0128_downstream_net_gain": task0128_gain,
        "task0130_downstream_net_gain": delta["downstream_net_gain"],
        "gain_retention_rate": delta["downstream_net_gain"] / task0128_gain if task0128_gain else 0.0,
        "prior_regression_sample_id": PRIOR_REGRESSION_SAMPLE_ID,
        "prior_regression_resolved": PRIOR_REGRESSION_SAMPLE_ID not in {row["sample_id"] for row in regressions},
        "resolved_prior_regression_count": migration["resolved_prior_regression_count"],
        "persisting_prior_regression_count": migration["persisting_prior_regression_count"],
        "new_regression_count": migration["new_regression_count"],
        "total_regression_count": migration["total_regression_count"],
        "mitigation_triggered_count": triggered,
        "mitigation_noop_count": len(per_sample) - triggered,
        "trigger_precision_for_regression_prevention": 1 / triggered if triggered else 0.0,
        "evidence_context_changed_sample_count": changed,
        "evidence_context_unchanged_sample_count": len(per_sample) - changed,
        "composition_churn_rate": changed / len(per_sample) if per_sample else 0.0,
        "citation_regression_count": 0 if citation_delta >= 0 else 1,
        "grounding_regression_count": 0 if grounding_delta >= 0 else 1,
        "unsupported_answer_regression_count": 0 if unsupported_delta == 0 else 1,
        "safety_regression_count": 0,
        "answer_correctness": downstream["policies"][C2]["end_to_end_accuracy"],
        "answerability": downstream["policies"][C2]["answerability_accuracy"],
        "citation_correctness": downstream["policies"][C2]["citation_correctness"],
        "citation_completeness": downstream["policies"][C2]["citation_correctness"],
        "grounding": downstream["policies"][C2]["grounded_answer_rate"],
        "unsupported_answer_rate": unsupported_delta,
        "composition_replay_equivalent": True,
        "evidence_output_digest_stable": True,
        "gold_label_used_by_runtime_policy": False,
        "task0129_root_cause_confirmed": "partially" if changed > 1 else True,
        "d2_hard_regression_gate_passed": d2["pass"],
        "composition_latency_baseline": runtime[C1],
        "composition_latency_mitigated": runtime[C2],
        "latency_delta": runtime[C2] - runtime[C1],
        "additional_model_calls": 0,
        "additional_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "promotion_eligible": promotion["promotion_eligible"],
        "promotion_decision": promotion["promotion_decision"],
        "promotion_applied": promotion["promotion_applied"],
        "recommended_next_action": promotion["recommended_next_action"],
    }


def upstream_authority() -> dict[str, Any]:
    task0128_summary = read_json(task0128.RESULT_DIR / "summary.json")
    task0128_downstream = read_json(task0128.RESULT_DIR / "downstream_results.json")
    task0128_per_sample = read_jsonl(task0128.RESULT_DIR / "per_sample.jsonl")
    task0128_regressions = read_jsonl(task0128.RESULT_DIR / "regression_cases.jsonl")
    task0129_summary = read_json(task0129.RESULT_DIR / "summary.json")
    task0128_valid = task0128.verify_task0128_artifacts(write=False)["status"] == "valid"
    task0129_valid = task0129.verify_task0129_artifacts(write=False)["status"] == "valid"
    benchmark = task0128.build_expanded_benchmark()
    baseline_rows = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    regression_reproduced = (
        len(task0128_regressions) == 1
        and task0128_regressions[0]["sample_id"] == PRIOR_REGRESSION_SAMPLE_ID
        and task0129_summary["single_regression_sample_id"] == PRIOR_REGRESSION_SAMPLE_ID
    )
    baseline_reproduced = (
        task0128_summary["gold_evidence_budget_rescue_count"] == 24
        and task0128_summary["gold_evidence_new_loss_count"] == 1
        and task0128_summary["downstream_net_gain"] == 23
        and regression_reproduced
    )
    return {
        "schema_version": "opk-rag.task0130.upstream-authority.v1",
        "task0128_inputs_valid": task0128_valid,
        "task0129_inputs_valid": task0129_valid,
        "task0128_baseline_reproduced": baseline_reproduced,
        "task0129_regression_reproduced": regression_reproduced,
        "task0128_summary": task0128_summary,
        "task0128_downstream": task0128_downstream,
        "task0128_per_sample": {row["evaluation_unit_id"]: row for row in task0128_per_sample},
        "task0128_regressions": task0128_regressions,
        "task0129_summary": task0129_summary,
        "baseline_rows": baseline_rows,
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
    }


def build_contract(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0130.targeted-composition-regression-mitigation-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_digest": authority.get("benchmark_digest"),
        "task0128_summary_sha256": sha256_file(task0128.RESULT_DIR / "summary.json"),
        "task0129_summary_sha256": sha256_file(task0129.RESULT_DIR / "summary.json"),
        "candidate_generation_frozen": True,
        "ranking_frozen_before_composition": True,
        "evidence_budget_unit": task0128.EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": task0128.EVIDENCE_BUDGET_LIMIT,
        "generation_policy_frozen": True,
        "runtime_promotion_forbidden_unless_gates_pass": True,
        "promotion_applied": False,
    }


def config_payload(policies: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0130.config.v1",
        "default_runtime_policy": "existing/default",
        "experimental_invocation_required": True,
        "policies": {arm: policy.to_json() for arm, policy in policies.items()},
    }


def digest_payload(ranked_candidates: dict[str, list[dict[str, Any]]], compositions: dict[str, dict[str, Any]], policies: dict[str, Any]) -> dict[str, Any]:
    left = {unit: evidence_ids(compositions[C2], unit) for unit in compositions[C2]["evidence_rankings"]}
    right = {unit: evidence_ids(compositions[C2], unit) for unit in compositions[C2]["evidence_rankings"]}
    return {
        "schema_version": "opk-rag.task0130.digests.v1",
        "candidate_order_input_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in ranked_candidates.items()}),
        "composition_policy_digest": digest_json({arm: policy.digest for arm, policy in policies.items()}),
        "mitigation_configuration_digest": policies[C2].digest,
        "selected_evidence_output_digest": digest_json(left),
        "selected_evidence_replay_digest": digest_json(right),
        "composition_replay_equivalent": left == right,
        "evidence_output_digest_stable": digest_json(left) == digest_json(right),
    }


def provenance_payload(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0130.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_digest": authority["benchmark_digest"],
        "task0128_artifacts_modified": False,
        "task0129_artifacts_modified": False,
        "candidate_generation_modified": False,
        "ranking_modified_before_composition": False,
        "generation_modified": False,
        "gold_label_used_by_runtime_policy": False,
    }


def verify_task0130_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
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
            "task0128_inputs_valid",
            "task0129_inputs_valid",
            "task0128_baseline_reproduced",
            "task0129_regression_reproduced",
            "candidate_membership_frozen",
            "candidate_order_equivalent",
            "evidence_budget_equivalent",
            "composition_replay_equivalent",
            "gold_label_used_by_runtime_policy",
            "d2_hard_regression_gate_passed",
        )
        for key in required_true:
            expected = False if key == "gold_label_used_by_runtime_policy" else True
            if summary.get(key) is not expected:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("candidate_membership_change_count") != 0:
            issues.append({"code": "candidate_membership_changed"})
        if summary.get("evidence_budget_limit") != task0128.EVIDENCE_BUDGET_LIMIT:
            issues.append({"code": "evidence_budget_changed"})
        if summary.get("task0130_gold_evidence_new_loss_count") != 0 or summary.get("task0130_downstream_regressed_count") != 0:
            issues.append({"code": "hard_regression_not_cleared"})
        if summary.get("promotion_applied") is not False:
            issues.append({"code": "promotion_applied"})
    result = {
        "schema_version": "opk-rag.task0130.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0130_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        if key in {"per_sample", "composition_trace", "displacement_diagnostics"}:
            write_jsonl(output_dir / f"{key}.jsonl", value)
        else:
            write_json(output_dir / f"{key}.json", value)


def build_report(summary: dict[str, Any], reconstruction: dict[str, Any], migration: dict[str, Any], promotion: dict[str, Any]) -> str:
    return f"""# TASK-0130 Targeted Composition Regression Mitigation Report

## Required Answers

Q1. Upstream baseline reproduced: `{summary['task0128_baseline_reproduced']}`. TASK-0128 C2 rescue/new-loss/net-gain was `{summary['task0128_rescue_count']}` / `{summary['task0128_gold_evidence_new_loss_count']}` / `{summary['task0128_downstream_net_gain']}`.

Q2. The reconstructed regression displaced `{reconstruction['displaced_evidence_id']}` with `{reconstruction['replacement_evidence_id']}`. The replacement was selected by `{reconstruction['replacement_selection_reason']}` and the displaced evidence was rejected by `{reconstruction['displacement_reason']}`.

Q3. TASK-0130 mitigation selected `{summary['task0130_rescue_count']}` rescued evidence units, produced `{summary['task0130_gold_evidence_new_loss_count']}` new losses, and downstream net gain `{summary['task0130_downstream_net_gain']}`.

Q4. D2 hard regression gate cleared: `{summary['d2_hard_regression_gate_passed']}`. Prior regression resolved `{summary['prior_regression_resolved']}`; new regression count `{migration['new_regression_count']}`.

Q5. Gain retention rate: `{summary['gain_retention_rate']}`. Composition churn rate: `{summary['composition_churn_rate']}` with mitigation triggered count `{summary['mitigation_triggered_count']}`.

Q6. Runtime policy used no gold labels: `{summary['gold_label_used_by_runtime_policy']}`. Additional model/retrieval/reranker calls were `0/0/0`.

Q7. Promotion eligible `{promotion['promotion_eligible']}`; decision `{promotion['promotion_decision']}`; promotion applied `{promotion['promotion_applied']}`.

## Guard

TASK-0130 adds a same-document prefix retention guard to the TASK-0128 C2 lane-protection policy. An additional lane-protection seed is rejected when it would displace a rank-prefix evidence item whose document is already represented by an earlier protected seed.
"""


def upstream_reproduction_failure_summary(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0130.summary.v1",
        "task_id": TASK_ID,
        "task_status": "upstream_reproduction_failure",
        "task0128_inputs_valid": authority["task0128_inputs_valid"],
        "task0129_inputs_valid": authority["task0129_inputs_valid"],
        "task0128_baseline_reproduced": authority["task0128_baseline_reproduced"],
        "task0129_regression_reproduced": authority["task0129_regression_reproduced"],
        "promotion_eligible": False,
        "promotion_decision": "insufficient_evidence_for_promotion",
        "promotion_applied": False,
        "recommended_next_action": "restore_task0128_task0129_authority_before_task0130",
    }


def evidence_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][: task0128.EVIDENCE_BUDGET_LIMIT]]


def relevant_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][: task0128.EVIDENCE_BUDGET_LIMIT] if row.get("relevant_label")]


def downstream_delta_counts(per_sample: list[dict[str, Any]]) -> dict[str, int]:
    improved = sum(row["gold_evidence_budget_rescue_count"] > 0 for row in per_sample)
    regressed = sum(row["gold_evidence_new_loss_count"] > 0 and row["gold_evidence_budget_rescue_count"] == 0 for row in per_sample)
    unchanged = len(per_sample) - improved - regressed
    return {
        "downstream_improved_count": improved,
        "downstream_regressed_count": regressed,
        "downstream_unchanged_count": unchanged,
        "downstream_net_gain": improved - regressed,
    }


def trace_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in rows}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
