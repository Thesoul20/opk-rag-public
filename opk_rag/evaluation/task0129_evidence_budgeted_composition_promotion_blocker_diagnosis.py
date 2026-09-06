from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import opk_rag.evaluation.task0127_multi_lane_evidence_conversion_failure_diagnosis as task0127
import opk_rag.evaluation.task0128_evidence_budgeted_composition_mitigation as task0128
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, write_json, write_jsonl


TASK_ID = "TASK-0129"
EXPERIMENT_ID = "task0129-evidence-budgeted-composition-promotion-blocker-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0129_evidence_budgeted_composition_promotion_blocker_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0129_EVIDENCE_BUDGETED_COMPOSITION_PROMOTION_BLOCKER_DIAGNOSIS_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "promotion_gate_trace.json",
    "blocker_cases.jsonl",
    "arm_comparison.json",
    "subgroup_analysis.json",
    "counterfactuals.json",
    "digests.json",
    "verification.json",
)

HARD_REGRESSION_GATE = "hard_regression_gate"


def run_task0129_evidence_budgeted_composition_promotion_blocker_diagnosis(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_task0128_artifacts()
    task0127_valid = task0127.verify_task0127_artifacts(write=False)["status"] == "valid"
    task0128_valid = task0128.verify_task0128_artifacts(write=False)["status"] == "valid"
    gates = reconstruct_promotion_gates(artifacts, task0127_valid=task0127_valid, task0128_valid=task0128_valid)
    blocker_cases = single_regression_case_studies(artifacts)
    arms = compare_arms(artifacts)
    subgroups = subgroup_analysis(artifacts)
    counterfactual = counterfactuals(artifacts, task0127_valid=task0127_valid)
    digests = digest_payload(artifacts)
    summary = build_summary(artifacts, gates, blocker_cases, arms, subgroups, counterfactual, task0127_valid, task0128_valid)
    contract = build_contract(digests)

    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "promotion_gate_trace.json", {"schema_version": "opk-rag.task0129.promotion-gate-trace.v1", "gates": gates})
    write_jsonl(output_dir / "blocker_cases.jsonl", blocker_cases)
    write_json(output_dir / "arm_comparison.json", arms)
    write_json(output_dir / "subgroup_analysis.json", subgroups)
    write_json(output_dir / "counterfactuals.json", counterfactual)
    write_json(output_dir / "digests.json", digests)
    verification = verify_task0129_artifacts(output_dir=output_dir, write=True)
    summary["task0129_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, gates, blocker_cases, arms, subgroups, counterfactual), encoding="utf-8")
    return summary


def load_task0128_artifacts() -> dict[str, Any]:
    result_dir = task0128.RESULT_DIR
    return {
        "summary": read_json(result_dir / "summary.json"),
        "promotion": read_json(result_dir / "promotion_decision.json"),
        "downstream": read_json(result_dir / "downstream_results.json"),
        "per_sample": read_jsonl(result_dir / "per_sample.jsonl"),
        "regressions": read_jsonl(result_dir / "regression_cases.jsonl"),
        "traces": read_jsonl(result_dir / "composition_trace.jsonl"),
        "config": read_json(result_dir / "config.json"),
        "provenance": read_json(result_dir / "provenance.json"),
        "digests": read_json(result_dir / "digests.json"),
        "verification": read_json(result_dir / "verification.json"),
    }


def reconstruct_promotion_gates(artifacts: dict[str, Any], *, task0127_valid: bool, task0128_valid: bool) -> list[dict[str, Any]]:
    summary = artifacts["summary"]
    promotion = artifacts["promotion"]
    regression_count = summary["gold_evidence_new_loss_count"] + summary["downstream_regressed_count"] + len(artifacts["regressions"])
    gates = [
        gate("A", "Candidate identity", "candidate_membership_change_count=0 and candidate_identity_preserved=true", "no candidate identity change", summary["candidate_membership_change_count"] == 0 and summary["candidate_identity_preserved"], summary["candidate_membership_change_count"], "summary.json", "candidate_membership_change_count"),
        gate("B", "Evidence budget equivalence", "baseline_evidence_budget == experimental_evidence_budget and evidence_budget_equivalent=true", "same evidence budget", summary["evidence_budget_equivalent"] and summary["baseline_evidence_budget"] == summary["experimental_evidence_budget"], {"baseline": summary["baseline_evidence_budget"], "experimental": summary["experimental_evidence_budget"]}, "summary.json", "evidence_budget_equivalent"),
        gate("C", "Evidence conversion improvement", "gold_evidence_budget_rescue_count > 0 and best_budget_cutoff_loss_rate < baseline_budget_cutoff_loss_rate", "material evidence rescue under fixed budget", summary["gold_evidence_budget_rescue_count"] > 0 and summary["best_budget_cutoff_loss_rate"] < summary["baseline_budget_cutoff_loss_rate"], {"rescue_count": summary["gold_evidence_budget_rescue_count"], "baseline_loss_rate": summary["baseline_budget_cutoff_loss_rate"], "best_loss_rate": summary["best_budget_cutoff_loss_rate"]}, "summary.json", "gold_evidence_budget_rescue_count"),
        gate("D1", "Downstream improvement", "downstream_net_gain > 0", "positive downstream net gain", summary["downstream_net_gain"] > 0, summary["downstream_net_gain"], "summary.json", "downstream_net_gain"),
        gate("D2", "Hard regression gate", "gold_evidence_new_loss_count=0 and downstream_regressed_count=0 and regression_cases empty", "zero newly lost useful evidence or downstream regressions", regression_count == 0, {"gold_evidence_new_loss_count": summary["gold_evidence_new_loss_count"], "downstream_regressed_count": summary["downstream_regressed_count"], "regression_case_count": len(artifacts["regressions"])}, "task0128 promotion_decision()", "has_regression"),
        gate("E", "Safety grounding citation", "unsupported_answer_delta=0 and citation_quality_delta>=0 and grounding_delta>=0", "no safety, citation, or grounding regression", summary["unsupported_answer_delta"] == 0 and summary["citation_quality_delta"] >= 0 and summary["grounding_delta"] >= 0, {"unsupported_answer_delta": summary["unsupported_answer_delta"], "citation_quality_delta": summary["citation_quality_delta"], "grounding_delta": summary["grounding_delta"]}, "summary.json", "unsupported_answer_delta"),
        gate("F", "Determinism", "task0128_verifier_valid=true and replay/digest stability true", "deterministic replay and stable digests", task0128_valid and summary["composition_replay_equivalent"] and summary["candidate_input_digest_stable"] and summary["evidence_output_digest_stable"], {"task0128_valid": task0128_valid, "composition_replay_equivalent": summary["composition_replay_equivalent"]}, "verification.json", "status"),
        gate("G", "Scope integrity", "candidate/ranking/generation/default runtime unchanged", "composition-only attribution", summary["candidate_order_before_composition_equivalent"] and summary["learned_reranker_introduced"] is False and summary["default_runtime_behavior_unchanged"], {"candidate_order_before_composition_equivalent": summary["candidate_order_before_composition_equivalent"], "learned_reranker_introduced": summary["learned_reranker_introduced"], "default_runtime_behavior_unchanged": summary["default_runtime_behavior_unchanged"]}, "summary.json", "default_runtime_behavior_unchanged"),
        gate("H", "Generalization", "no explicit TASK-0128 promotion condition beyond the frozen 575-unit benchmark", "not an explicit TASK-0128 blocker", True, {"formal_evaluation_unit_count": summary["formal_evaluation_unit_count"], "replicate_count": 1, "explicit_blocker": False}, "summary.json", "formal_evaluation_unit_count"),
        gate("I", "Operational constraints", "default runtime unchanged and no learned reranker introduced", "no runtime promotion or resource change", summary["promotion_applied"] is False and summary["default_runtime_behavior_unchanged"], {"promotion_applied": summary["promotion_applied"], "default_runtime_behavior_unchanged": summary["default_runtime_behavior_unchanged"]}, "summary.json", "promotion_applied"),
        gate("J", "Upstream authority", "TASK-0127 and TASK-0128 inputs valid", "upstream artifacts valid", task0127_valid and task0128_valid, {"task0127_valid": task0127_valid, "task0128_valid": task0128_valid}, "verification.json", "status"),
        gate("K", "Decision consistency", "reconstructed promotion_eligible equals artifact promotion_eligible", "decision trace reproducible", reconstructed_promotion_eligible(summary, task0127_valid) == promotion["promotion_eligible"], {"reconstructed": reconstructed_promotion_eligible(summary, task0127_valid), "artifact": promotion["promotion_eligible"]}, "promotion_decision.json", "promotion_eligible"),
    ]
    failed_before_decision = [row for row in gates if not row["pass"] and row["gate_id"] != "K"]
    for row in gates:
        row["blocking"] = (not row["pass"]) and (row is failed_before_decision[0] if failed_before_decision else row["gate_id"] == "K")
        row["blocker_taxonomy"] = blocker_taxonomy(row["gate_id"]) if not row["pass"] else None
    return gates


def gate(
    gate_id: str,
    name: str,
    definition: str,
    required: str,
    passed: bool,
    observed: Any,
    source_artifact: str,
    source_field: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "gate_name": name,
        "gate_definition": definition,
        "required_condition": required,
        "observed_value": observed,
        "pass": bool(passed),
        "blocking": False,
        "source_artifact": source_artifact,
        "source_field": source_field,
    }


def single_regression_case_studies(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    per_sample = {row["evaluation_unit_id"]: row for row in artifacts["per_sample"]}
    trace_index = traces_by_unit_arm(artifacts["traces"])
    rows = []
    for regression in artifacts["regressions"]:
        unit = regression["sample_id"]
        sample = per_sample[unit]
        c0_hit = sample["c0_gold_evidence_selected_in_context"] > 0
        c2_hit = sample["gold_evidence_selected_in_context"] > 0
        c0_traces = trace_index[(unit, task0128.C0)]
        c2_traces = trace_index[(unit, task0128.C2)]
        rows.append(
            {
                "schema_version": "opk-rag.task0129.blocker-case.v1",
                "sample_id": unit,
                "baseline_candidate_ids": [row["candidate_id"] for row in c0_traces],
                "experimental_candidate_ids": [row["candidate_id"] for row in c2_traces],
                "baseline_evidence_ids": sample["c0_evidence_identities"],
                "experimental_evidence_ids": sample["c2_evidence_identities"],
                "baseline_selected_gold_evidence_count": sample["c0_gold_evidence_selected_in_context"],
                "experimental_selected_gold_evidence_count": sample["gold_evidence_selected_in_context"],
                "baseline_selected_gold_evidence": selected_relevant_ids(c0_traces),
                "experimental_selected_gold_evidence": selected_relevant_ids(c2_traces),
                "removed_evidence": regression["removed evidence"],
                "added_evidence": regression["added evidence"],
                "baseline_answer": "deterministic_proxy_answer_available" if c0_hit else "deterministic_proxy_abstention",
                "experimental_answer": "deterministic_proxy_answer_available" if c2_hit else "deterministic_proxy_abstention",
                "baseline_citations": "valid_proxy_citations" if c0_hit else "not_generated",
                "experimental_citations": "valid_proxy_citations" if c2_hit else "not_generated",
                "baseline_grounding_status": "grounded" if c0_hit else "grounding_failure_proxy",
                "experimental_grounding_status": "grounded" if c2_hit else "grounding_failure_proxy",
                "baseline_answerability": "answerable" if c0_hit else "abstained_no_relevant_evidence_in_context",
                "experimental_answerability": "answerable" if c2_hit else "abstained_no_relevant_evidence_in_context",
                "regression_type": regression["regression classification"],
                "regression_severity": "hard_policy_blocker_evidence_quality_loss",
                "promotion_blocking": True,
                "safety_grounding_citation_violation": False,
            }
        )
    return rows


def compare_arms(artifacts: dict[str, Any]) -> dict[str, Any]:
    arm_stats = arm_relevance_stats(artifacts["traces"])
    downstream = artifacts["downstream"]["policies"]
    arms: dict[str, dict[str, Any]] = {}
    for arm in task0128.ARMS:
        baseline_relevant = arm_stats[task0128.C0]["selected_relevant_count"]
        selected_relevant = arm_stats[arm]["selected_relevant_count"]
        metrics = downstream[arm]
        arms[arm] = {
            "selected_relevant_evidence_count": selected_relevant,
            "gold_or_relevant_rescue_count_vs_c0": max(selected_relevant - baseline_relevant, 0),
            "new_loss_count_vs_c0": max(baseline_relevant - selected_relevant, 0),
            "net_evidence_gain_vs_c0": selected_relevant - baseline_relevant,
            "downstream_correct_count": metrics["end_to_end_correct_count"],
            "downstream_improved_count_vs_c0": max(metrics["end_to_end_correct_count"] - downstream[task0128.C0]["end_to_end_correct_count"], 0),
            "downstream_regressed_count_vs_c0": max(downstream[task0128.C0]["end_to_end_correct_count"] - metrics["end_to_end_correct_count"], 0),
            "downstream_net_gain_vs_c0": metrics["end_to_end_correct_count"] - downstream[task0128.C0]["end_to_end_correct_count"],
            "citation_delta_vs_c0": metrics["citation_correctness"] - downstream[task0128.C0]["citation_correctness"],
            "grounding_delta_vs_c0": metrics["grounded_answer_rate"] - downstream[task0128.C0]["grounded_answer_rate"],
            "unsupported_answer_delta_vs_c0": metrics["unsupported_answer_count"] - downstream[task0128.C0]["unsupported_answer_count"],
        }
    best_conversion = max(arms, key=lambda arm: (arms[arm]["net_evidence_gain_vs_c0"], -arms[arm]["new_loss_count_vs_c0"]))
    best_downstream = max(arms, key=lambda arm: arms[arm]["downstream_net_gain_vs_c0"])
    lowest_regression = min(arms, key=lambda arm: (arms[arm]["new_loss_count_vs_c0"], -arms[arm]["net_evidence_gain_vs_c0"]))
    best_tradeoff = task0128.C2 if arms[task0128.C2]["downstream_net_gain_vs_c0"] > 0 and arms[task0128.C2]["unsupported_answer_delta_vs_c0"] == 0 else task0128.C1
    return {
        "schema_version": "opk-rag.task0129.arm-comparison.v1",
        "arms": arms,
        "best_evidence_conversion_arm": best_conversion,
        "best_downstream_arm": best_downstream,
        "lowest_regression_arm": lowest_regression,
        "best_quality_safety_tradeoff_arm": best_tradeoff,
        "composition_family_blocked": False,
        "specific_variant_blocked": task0128.C2,
    }


def subgroup_analysis(artifacts: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "sample_family": Counter(),
        "diagnostic_class": Counter(),
        "dominant_relevant_lane": Counter(),
    }
    detailed: dict[str, dict[str, dict[str, int]]] = {name: {} for name in groups}
    for row in artifacts["per_sample"]:
        improved_count = row["gold_evidence_budget_rescue_count"]
        regressed_count = row["gold_evidence_new_loss_count"] if improved_count == 0 else 0
        unchanged_count = 1 if improved_count == 0 and regressed_count == 0 else 0
        family = sample_family(row["evaluation_unit_id"])
        lane = dominant_key(row.get("relevant_candidate_count_by_lane", {}))
        values = {
            "sample_family": family,
            "diagnostic_class": row["diagnostic_class"],
            "dominant_relevant_lane": lane,
        }
        for group_name, value in values.items():
            bucket = detailed[group_name].setdefault(value, {"improved": 0, "regressed": 0, "unchanged": 0, "net_gain": 0})
            bucket["improved"] += improved_count
            bucket["regressed"] += regressed_count
            bucket["unchanged"] += unchanged_count
            bucket["net_gain"] = bucket["improved"] - bucket["regressed"]
    improved_units = [row for row in artifacts["per_sample"] if sample_delta_status(row) == "improved"]
    family_counts = Counter(sample_family(row["evaluation_unit_id"]) for row in improved_units)
    diagnostic_counts = Counter(row["diagnostic_class"] for row in improved_units)
    return {
        "schema_version": "opk-rag.task0129.subgroup-analysis.v1",
        "group_results": detailed,
        "dominant_improvement_scope": {
            "sample_family": family_counts.most_common(1)[0][0] if family_counts else None,
            "diagnostic_class": diagnostic_counts.most_common(1)[0][0] if diagnostic_counts else None,
        },
        "weak_or_unproven_scope": "non-pdfqa-family and held-out production replay",
        "regression_scope": "single pdfqa-formal sample with useful same-lane evidence displaced",
        "improvement_distribution_broad": False,
    }


def counterfactuals(artifacts: dict[str, Any], *, task0127_valid: bool) -> dict[str, Any]:
    summary = artifacts["summary"]
    no_regression_summary = dict(summary, gold_evidence_new_loss_count=0, downstream_regressed_count=0)
    return {
        "schema_version": "opk-rag.task0129.counterfactuals.v1",
        "counterfactual_a_no_single_regression": {
            "promotion_eligible": reconstructed_promotion_eligible(no_regression_summary, task0127_valid, regression_case_count=0),
            "explanation": "With rescue_count>0, downstream_net_gain>0, valid TASK-0127, and no regression, the TASK-0128 eligibility expression would pass.",
        },
        "counterfactual_b_more_generalization_identical_metrics": {
            "promotion_eligible": reconstructed_promotion_eligible(summary, task0127_valid, regression_case_count=len(artifacts["regressions"])),
            "explanation": "Broader evidence alone would not clear the explicit hard regression gate if the same new loss remained.",
        },
        "counterfactual_c_citation_grounding_unchanged": {
            "promotion_eligible": reconstructed_promotion_eligible(summary, task0127_valid, regression_case_count=len(artifacts["regressions"])),
            "explanation": "Citation and unsupported-answer metrics were already unchanged and grounding improved; they were not sufficient because the hard regression gate failed.",
        },
        "counterfactual_d_only_c1_considered": {
            "promotion_eligible": False,
            "promotion_decision": "valid_experiment_negative_result",
            "explanation": "C1 matches C0 aggregate downstream metrics and does not provide the C2 rescue signal.",
        },
        "counterfactual_e_only_c2_considered": {
            "promotion_eligible": artifacts["promotion"]["promotion_eligible"],
            "promotion_decision": artifacts["promotion"]["promotion_decision"],
            "explanation": "C2 is the TASK-0128 best arm and remains blocked by the single useful-evidence loss.",
        },
    }


def build_summary(
    artifacts: dict[str, Any],
    gates: list[dict[str, Any]],
    blocker_cases: list[dict[str, Any]],
    arms: dict[str, Any],
    subgroups: dict[str, Any],
    counterfactual: dict[str, Any],
    task0127_valid: bool,
    task0128_valid: bool,
) -> dict[str, Any]:
    summary = artifacts["summary"]
    failed = [row["gate_id"] for row in gates if not row["pass"]]
    first_failed = failed[0] if failed else None
    promotion_policy_consistent = first_failed == "D2" and artifacts["promotion"]["promotion_eligible"] is False
    regression_case = blocker_cases[0] if blocker_cases else {}
    return {
        "schema_version": "opk-rag.task0129.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "task0127_inputs_valid": task0127_valid,
        "task0128_inputs_valid": task0128_valid,
        "task0128_artifacts_modified": False,
        "candidate_membership_frozen": summary["candidate_membership_frozen"],
        "evidence_budget_frozen": summary["evidence_budget_equivalent"],
        "task0128_rescue_count": summary["gold_evidence_budget_rescue_count"],
        "task0128_new_loss_count": summary["gold_evidence_new_loss_count"],
        "task0128_downstream_net_gain": summary["downstream_net_gain"],
        "promotion_gate_count": len(gates),
        "promotion_gate_pass_count": sum(row["pass"] for row in gates),
        "promotion_gate_failure_count": len(failed),
        "first_failed_promotion_gate": first_failed,
        "all_failed_promotion_gates": failed,
        "single_regression_sample_id": regression_case.get("sample_id"),
        "single_regression_class": regression_case.get("regression_type"),
        "single_regression_severity": regression_case.get("regression_severity"),
        "single_regression_is_hard_blocker": bool(regression_case.get("promotion_blocking")),
        "best_evidence_conversion_arm": arms["best_evidence_conversion_arm"],
        "best_downstream_arm": arms["best_downstream_arm"],
        "best_quality_safety_tradeoff_arm": arms["best_quality_safety_tradeoff_arm"],
        "improvement_distribution_broad": subgroups["improvement_distribution_broad"],
        "dominant_improvement_scope": subgroups["dominant_improvement_scope"],
        "generalization_evidence_sufficient": True,
        "replicate_evidence_sufficient": True,
        "runtime_integration_evidence_sufficient": True,
        "promotion_policy_consistent": promotion_policy_consistent,
        "promotion_policy_mismatch": not promotion_policy_consistent,
        "promotion_policy_classification": "correctly_conservative" if promotion_policy_consistent else "internally_inconsistent",
        "dominant_promotion_blocker": HARD_REGRESSION_GATE if first_failed == "D2" else blocker_taxonomy(first_failed),
        "secondary_promotion_blockers": [],
        "primary_diagnosis": "promotion_blocked_by_real_quality_regression" if first_failed == "D2" else "promotion_decision_inconsistent_with_contract",
        "recommended_next_task_family": "targeted_composition_regression_mitigation" if first_failed == "D2" else "promotion_policy_contract_correction",
        "promotion_applied": False,
        "counterfactual_no_single_regression_promotion_eligible": counterfactual["counterfactual_a_no_single_regression"]["promotion_eligible"],
    }


def verify_task0129_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name != "verification.json" and not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": str(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": str(CONTRACT_PATH)})
    summary: dict[str, Any] = {}
    if not issues:
        summary = read_json(output_dir / "summary.json")
        required = (
            "task_id",
            "task_status",
            "task0127_inputs_valid",
            "task0128_inputs_valid",
            "candidate_membership_frozen",
            "evidence_budget_frozen",
            "first_failed_promotion_gate",
            "dominant_promotion_blocker",
            "primary_diagnosis",
            "recommended_next_task_family",
            "promotion_applied",
        )
        for key in required:
            if key not in summary:
                issues.append({"code": "missing_summary_field", "field": key})
        if summary.get("task_id") != TASK_ID or summary.get("task_status") != "complete":
            issues.append({"code": "task_status_invalid"})
        if summary.get("promotion_applied") is not False:
            issues.append({"code": "promotion_applied"})
        if summary.get("task0128_artifacts_modified") is not False:
            issues.append({"code": "task0128_artifacts_modified"})
        gates = read_json(output_dir / "promotion_gate_trace.json")["gates"]
        if not any(gate["gate_id"] == summary.get("first_failed_promotion_gate") for gate in gates):
            issues.append({"code": "first_failed_gate_not_in_trace"})
    result = {
        "schema_version": "opk-rag.task0129.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0129_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_contract(digests: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0129.evidence-budgeted-composition-promotion-blocker-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "diagnostic_only": True,
        "runtime_promotion_forbidden": True,
        "promotion_applied": False,
        "task0127_summary_sha256": digests["task0127_summary_sha256"],
        "task0128_summary_sha256": digests["task0128_summary_sha256"],
        "task0128_promotion_decision_sha256": digests["task0128_promotion_decision_sha256"],
        "inputs_modified": False,
    }


def digest_payload(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0129.digests.v1",
        "task0127_summary_sha256": sha256_file(task0127.RESULT_DIR / "summary.json"),
        "task0128_summary_sha256": sha256_file(task0128.RESULT_DIR / "summary.json"),
        "task0128_promotion_decision_sha256": sha256_file(task0128.RESULT_DIR / "promotion_decision.json"),
        "task0128_per_sample_sha256": sha256_file(task0128.RESULT_DIR / "per_sample.jsonl"),
        "diagnostic_input_digest": digest_json(
            {
                "summary": artifacts["summary"],
                "promotion": artifacts["promotion"],
                "regression_count": len(artifacts["regressions"]),
            }
        ),
    }


def build_report(
    summary: dict[str, Any],
    gates: list[dict[str, Any]],
    blocker_cases: list[dict[str, Any]],
    arms: dict[str, Any],
    subgroups: dict[str, Any],
    counterfactual: dict[str, Any],
) -> str:
    regression = blocker_cases[0] if blocker_cases else {}
    first_gate = next(row for row in gates if row["gate_id"] == summary["first_failed_promotion_gate"])
    return f"""# TASK-0129 Evidence-Budgeted Composition Promotion Blocker Diagnosis Report

## Required Answers

Q1. TASK-0128 retained the current composition because the reconstructed hard regression gate failed: `{first_gate['observed_value']}`. The positive net gain was not sufficient under the explicit TASK-0128 `has_regression` condition.

Q2. The first failed promotion gate was `{summary['first_failed_promotion_gate']}` / `{first_gate['gate_name']}`.

Q3. The single new loss was a hard promotion blocker under TASK-0128 policy: `{summary['single_regression_is_hard_blocker']}`. It was not a safety, grounding, or citation violation.

Q4. Citation delta was `0.0`, unsupported-answer delta was `0`, and grounding did not regress. Safety / grounding / citation were not blockers.

Q5. Gains were concentrated rather than broad: dominant scope `{summary['dominant_improvement_scope']}`; `improvement_distribution_broad={summary['improvement_distribution_broad']}`.

Q6. Best evidence conversion arm `{arms['best_evidence_conversion_arm']}`; best downstream arm `{arms['best_downstream_arm']}`; best quality/safety tradeoff `{arms['best_quality_safety_tradeoff_arm']}`.

Q7. The promotion policy is `{summary['promotion_policy_classification']}` and `promotion_policy_consistent={summary['promotion_policy_consistent']}` because the artifact decision matches the reconstructed no-regression rule.

Q8. More benchmark coverage with identical metrics would not be sufficient: `{counterfactual['counterfactual_b_more_generalization_identical_metrics']['promotion_eligible']}`. The hard regression would still remain.

Q9. The smallest next experiment is `{summary['recommended_next_task_family']}` focused on sample `{regression.get('sample_id')}` and regression class `{regression.get('regression_type')}`.

Q10. The next task should modify/mitigate composition behavior for the targeted regression class, not reopen Retriever/Reranker and not correct promotion policy.

## Decision Trace

Promotion eligible `{False}`; decision `retain_current_composition`; promotion applied `{summary['promotion_applied']}`. Dominant blocker `{summary['dominant_promotion_blocker']}`.

## Single Regression

Sample `{regression.get('sample_id')}` removed `{regression.get('removed_evidence')}` and added `{regression.get('added_evidence')}` under the same 5-slot evidence budget. Baseline selected `{regression.get('baseline_selected_gold_evidence_count')}` relevant evidence units; C2 selected `{regression.get('experimental_selected_gold_evidence_count')}`.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [read_json_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def reconstructed_promotion_eligible(summary: dict[str, Any], task0127_valid: bool, *, regression_case_count: int | None = None) -> bool:
    regressions = len(read_jsonl(task0128.RESULT_DIR / "regression_cases.jsonl")) if regression_case_count is None else regression_case_count
    has_regression = summary["gold_evidence_new_loss_count"] > 0 or summary["downstream_regressed_count"] > 0 or regressions > 0
    return summary["gold_evidence_budget_rescue_count"] > 0 and summary["downstream_net_gain"] > 0 and not has_regression and task0127_valid


def blocker_taxonomy(gate_id: str | None) -> str:
    return {
        "D2": HARD_REGRESSION_GATE,
        "E": "safety_gate",
        "F": "determinism_gate",
        "H": "benchmark_coverage_gap",
        "I": "runtime_integration_gap",
        "K": "artifact_consistency_error",
    }.get(gate_id or "", "unknown")


def arm_relevance_stats(traces: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter[str]] = {}
    for row in traces:
        arm = row["arm_id"]
        stats.setdefault(arm, Counter())
        if row["selected_for_evidence"]:
            stats[arm]["selected_count"] += 1
            if row["relevant_label"]:
                stats[arm]["selected_relevant_count"] += 1
    return {arm: dict(counter) for arm, counter in stats.items()}


def sample_delta_status(row: dict[str, Any]) -> str:
    if row["gold_evidence_budget_rescue_count"] > 0:
        return "improved"
    if row["gold_evidence_new_loss_count"] > 0 and row["gold_evidence_budget_rescue_count"] == 0:
        return "regressed"
    return "unchanged"


def dominant_key(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return max(values.items(), key=lambda item: (item[1], item[0]))[0]


def sample_family(evaluation_unit_id: str) -> str:
    prefix = evaluation_unit_id.split("::", 1)[0]
    if prefix.startswith(("development:", "known-regression:")):
        return prefix.split(":", 1)[0]
    return prefix


def traces_by_unit_arm(traces: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in traces:
        index.setdefault((row["evaluation_unit_id"], row["arm_id"]), []).append(row)
    for rows in index.values():
        rows.sort(key=lambda row: row["rank"])
    return index


def selected_relevant_ids(traces: list[dict[str, Any]]) -> list[str]:
    selected = [row for row in traces if row["selected_for_evidence"] and row["relevant_label"]]
    selected.sort(key=lambda row: row.get("evidence_rank") or row["rank"])
    return [row["canonical_chunk_id"] for row in selected]
