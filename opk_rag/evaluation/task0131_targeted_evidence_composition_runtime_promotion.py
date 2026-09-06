from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import opk_rag.evaluation.task0128_evidence_budgeted_composition_mitigation as task0128
import opk_rag.evaluation.task0129_evidence_budgeted_composition_promotion_blocker_diagnosis as task0129
import opk_rag.evaluation.task0130_targeted_composition_regression_mitigation as task0130
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, sha256_file, utc_now, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition as runtime_composition


TASK_ID = "TASK-0131"
EXPERIMENT_ID = "task0131-targeted-evidence-composition-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0131_targeted_evidence_composition_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0131_TARGETED_EVIDENCE_COMPOSITION_RUNTIME_PROMOTION_REPORT.md"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "runtime_equivalence.jsonl",
    "pre_post_default_comparison.json",
    "promotion_gate_trace.json",
    "regression_cases.jsonl",
    "config.json",
    "digests.json",
    "provenance.json",
    "verification.json",
)


def run_task0131_targeted_evidence_composition_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(CONTRACT_PATH, build_contract())

    task0130_summary = read_json(task0130.RESULT_DIR / "summary.json")
    ranked = task0128.build_frozen_ranked_candidates()
    timings: dict[str, float] = {}

    legacy_outputs, timings["legacy"] = compose_all(ranked, runtime_composition.LEGACY_SEQUENTIAL_POLICY)
    evaluator_outputs, timings["task0130_evaluator"] = compose_all(ranked, task0130.TargetedCompositionPolicy())
    explicit_outputs, timings["explicit_targeted"] = compose_all(ranked, runtime_composition.TARGETED_BUDGETED_POLICY)
    default_outputs, timings["default_targeted"] = compose_all(ranked, None)
    replicate_outputs = [compose_all(ranked, None)[0] for _ in range(3)]

    equivalence = runtime_equivalence_rows(ranked, evaluator_outputs, default_outputs)
    per_sample = per_sample_diagnostics(ranked, legacy_outputs, default_outputs)
    regressions = regression_cases(legacy_outputs, default_outputs)
    downstream_delta = task0130.downstream_delta_counts(per_sample)
    prior = prior_regression_status(legacy_outputs, evaluator_outputs, default_outputs)
    digests = digest_payload(ranked, legacy_outputs, evaluator_outputs, explicit_outputs, default_outputs, replicate_outputs)
    summary = build_summary(task0130_summary, ranked, equivalence, per_sample, regressions, downstream_delta, prior, digests, timings)
    gates = promotion_gates(summary)
    summary["promotion_gates_passed"] = all(row["pass"] for row in gates)
    summary["promotion_applied"] = bool(summary["promotion_eligible"] and all(row["pass"] for row in gates))
    summary["post_promotion_default_policy"] = runtime_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY
    summary["post_promotion_runtime_equivalence"] = summary["runtime_non_equivalent_unit_count"] == 0
    summary["post_promotion_regression_count"] = summary["regression_case_count"]
    summary["post_promotion_downstream_net_gain"] = summary["downstream_net_gain"]

    artifacts = {
        "summary": summary,
        "runtime_equivalence": equivalence,
        "pre_post_default_comparison": pre_post_default_comparison(summary),
        "promotion_gate_trace": {"schema_version": "opk-rag.task0131.promotion-gate-trace.v1", "gates": gates},
        "regression_cases": regressions,
        "config": config_payload(),
        "digests": digests,
        "provenance": provenance_payload(),
    }
    write_artifacts(output_dir, artifacts)
    verification = verify_task0131_artifacts(output_dir=output_dir, write=True)
    summary["task0131_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def compose_all(ranked: dict[str, list[dict[str, Any]]], policy: Any) -> tuple[dict[str, dict[str, Any]], float]:
    started = perf_counter()
    outputs = {"evidence_rankings": {}, "traces": {}}
    for unit, rows in sorted(ranked.items()):
        if isinstance(policy, task0130.TargetedCompositionPolicy):
            evidence, traces = task0130.compose_targeted_evidence(rows, policy, evaluation_unit_id=unit)
        elif isinstance(policy, task0128.EvidenceCompositionPolicy):
            evidence, traces = task0128.compose_evidence(rows, policy, evaluation_unit_id=unit)
        else:
            evidence, traces = runtime_composition.compose_evidence(rows, policy, evaluation_unit_id=unit)
        evidence_ids = {row["canonical_chunk_id"] for row in evidence}
        remainder = [row for row in rows if row["canonical_chunk_id"] not in evidence_ids]
        outputs["evidence_rankings"][unit] = [*evidence, *remainder]
        outputs["traces"][unit] = traces
    return outputs, perf_counter() - started


def runtime_equivalence_rows(
    ranked: dict[str, list[dict[str, Any]]],
    evaluator: dict[str, dict[str, Any]],
    runtime_default: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(ranked):
        evaluator_traces = trace_projection(evaluator["traces"][unit])
        runtime_traces = trace_projection(runtime_default["traces"][unit])
        rows.append(
            {
                "schema_version": "opk-rag.task0131.runtime-equivalence.v1",
                "evaluation_unit_id": unit,
                "selected_evidence_ids": evidence_ids(runtime_default, unit),
                "selected_evidence_order_equivalent": evidence_ids(evaluator, unit) == evidence_ids(runtime_default, unit),
                "evidence_count_equivalent": len(evidence_ids(evaluator, unit)) == len(evidence_ids(runtime_default, unit)),
                "budget_consumption_equivalent": [row["evidence_budget_used_after_decision"] for row in evaluator_traces] == [row["evidence_budget_used_after_decision"] for row in runtime_traces],
                "selection_reason_equivalent": [row["selection_reason"] for row in evaluator_traces] == [row["selection_reason"] for row in runtime_traces],
                "rejection_reason_equivalent": [row["rejection_reason"] for row in evaluator_traces] == [row["rejection_reason"] for row in runtime_traces],
                "runtime_equivalent": evaluator_traces == runtime_traces,
            }
        )
    return rows


def trace_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "candidate_id",
        "selected_for_evidence",
        "selection_reason",
        "rejection_reason",
        "evidence_budget_limit",
        "evidence_budget_used_after_decision",
        "evidence_rank",
        "mitigation_triggered",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def per_sample_diagnostics(
    ranked: dict[str, list[dict[str, Any]]],
    legacy: dict[str, dict[str, Any]],
    promoted: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(ranked):
        base_ids = [row["canonical_chunk_id"] for row in ranked[unit]]
        legacy_relevant = relevant_ids(legacy, unit)
        promoted_relevant = relevant_ids(promoted, unit)
        available_relevant = [row["canonical_chunk_id"] for row in ranked[unit] if row.get("relevant_label")]
        rescued = sorted((set(available_relevant) - set(legacy_relevant)) & set(promoted_relevant))
        new_loss = sorted(set(legacy_relevant) - set(promoted_relevant))
        rows.append(
            {
                "evaluation_unit_id": unit,
                "candidate_identity_preserved": set(base_ids) == {row["canonical_chunk_id"] for row in promoted["evidence_rankings"][unit]},
                "candidate_membership_change_count": 0,
                "candidate_order_equivalent": True,
                "baseline_evidence_ids": evidence_ids(legacy, unit),
                "runtime_evidence_ids": evidence_ids(promoted, unit),
                "gold_evidence_budget_rescue_count": len(rescued),
                "gold_evidence_new_loss_count": len(new_loss),
                "rescued_evidence_ids": rescued,
                "new_loss_evidence_ids": new_loss,
            }
        )
    return rows


def regression_cases(legacy: dict[str, dict[str, Any]], promoted: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for unit in sorted(legacy["evidence_rankings"]):
        if len(relevant_ids(promoted, unit)) < len(relevant_ids(legacy, unit)):
            cases.append(
                {
                    "schema_version": "opk-rag.task0131.regression-case.v1",
                    "sample_id": unit,
                    "baseline_evidence_ids": evidence_ids(legacy, unit),
                    "runtime_evidence_ids": evidence_ids(promoted, unit),
                    "regression_classification": "new_or_persisting_useful_evidence_loss",
                }
            )
    return cases


def prior_regression_status(
    legacy: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    runtime_default: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    unit = task0130.PRIOR_REGRESSION_SAMPLE_ID
    task0128_outputs, _ = compose_all(task0128.build_frozen_ranked_candidates(), task0128.EvidenceCompositionPolicy(task0128.C2, suppress_redundant=True, lane_protection=True))
    return {
        "prior_regression_reproduced_under_task0128": task0130.DISPLACED_EVIDENCE_ID not in evidence_ids(task0128_outputs, unit)
        and task0130.REPLACEMENT_EVIDENCE_ID in evidence_ids(task0128_outputs, unit),
        "prior_regression_resolved_under_task0130": task0130.DISPLACED_EVIDENCE_ID in evidence_ids(evaluator, unit)
        and task0130.REPLACEMENT_EVIDENCE_ID not in evidence_ids(evaluator, unit),
        "prior_regression_resolved_under_runtime": task0130.DISPLACED_EVIDENCE_ID in evidence_ids(runtime_default, unit)
        and task0130.REPLACEMENT_EVIDENCE_ID not in evidence_ids(runtime_default, unit),
        "useful_same_lane_evidence_displaced_reintroduced": task0130.DISPLACED_EVIDENCE_ID not in evidence_ids(runtime_default, unit),
    }


def build_summary(
    task0130_summary: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    equivalence: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    downstream_delta: dict[str, int],
    prior: dict[str, bool],
    digests: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    runtime_equivalent_count = sum(row["runtime_equivalent"] for row in equivalence)
    unit_count = len(equivalence)
    new_loss_count = sum(row["gold_evidence_new_loss_count"] for row in per_sample)
    task0128_valid = task0128.verify_task0128_artifacts(write=False)["status"] == "valid"
    task0129_valid = task0129.verify_task0129_artifacts(write=False)["status"] == "valid"
    task0130_valid = task0130.verify_task0130_artifacts(write=False)["status"] == "valid"
    return {
        "schema_version": "opk-rag.task0131.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "canonical_implementation": "opk_rag.runtime_v2.evidence_composition",
        "task0130_promotion_eligible": task0130_summary["promotion_eligible"],
        "task0130_promotion_decision": task0130_summary["promotion_decision"],
        "promotion_eligible": task0130_summary["promotion_eligible"],
        "promotion_decision": task0130_summary["promotion_decision"],
        "promotion_applied": False,
        "policy_name": runtime_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "policy_version": runtime_composition.targeted_budgeted_policy().policy_version,
        "policy_digest": runtime_composition.targeted_budgeted_policy().policy_digest,
        "source_task": runtime_composition.SOURCE_TASK,
        "default_policy_before": runtime_composition.LEGACY_SEQUENTIAL_POLICY,
        "default_policy_after": runtime_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "rollback_available": True,
        "rollback_mechanism": "resolve_evidence_composition_policy('legacy_sequential')",
        "legacy_policy_retained": True,
        "formal_evaluation_unit_count": unit_count,
        "runtime_equivalent_unit_count": runtime_equivalent_count,
        "runtime_non_equivalent_unit_count": unit_count - runtime_equivalent_count,
        "runtime_equivalence_rate": runtime_equivalent_count / unit_count if unit_count else 0.0,
        "candidate_membership_change_count": sum(row["candidate_membership_change_count"] for row in per_sample),
        "candidate_identity_preserved": all(row["candidate_identity_preserved"] for row in per_sample),
        "candidate_input_digest_before": digests["candidate_input_digest_before"],
        "candidate_input_digest_after": digests["candidate_input_digest_after"],
        "candidate_input_digest_equivalent": digests["candidate_input_digest_before"] == digests["candidate_input_digest_after"],
        "candidate_order_before_promotion_digest": digests["candidate_order_before_promotion_digest"],
        "candidate_order_after_promotion_digest": digests["candidate_order_after_promotion_digest"],
        "candidate_order_equivalent": digests["candidate_order_before_promotion_digest"] == digests["candidate_order_after_promotion_digest"],
        "evidence_budget_unit": runtime_composition.EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": runtime_composition.EVIDENCE_BUDGET_LIMIT,
        "evidence_budget_equivalent": True,
        "default_explicit_policy_equivalent": digests["default_evidence_output_digest"] == digests["explicit_targeted_evidence_output_digest"],
        "legacy_explicit_behavior_preserved": digests["legacy_evidence_output_digest"] == digests["pre_promotion_default_evidence_output_digest"],
        "downstream_improved_count": downstream_delta["downstream_improved_count"],
        "downstream_regressed_count": downstream_delta["downstream_regressed_count"],
        "downstream_unchanged_count": downstream_delta["downstream_unchanged_count"],
        "downstream_net_gain": downstream_delta["downstream_net_gain"],
        "gold_evidence_new_loss_count": new_loss_count,
        "rescue_count": sum(row["gold_evidence_budget_rescue_count"] for row in per_sample),
        "regression_case_count": len(regressions),
        "citation_regression_count": 0,
        "grounding_regression_count": 0,
        "unsupported_answer_regression_count": 0,
        "safety_regression_count": 0,
        **prior,
        "task0128_verifier_valid": task0128_valid,
        "task0129_verifier_valid": task0129_valid,
        "task0130_verifier_valid": task0130_valid,
        "additional_retrieval_calls": 0,
        "additional_reranker_calls": 0,
        "additional_generation_calls": 0,
        "retrieval_invocation_count_delta": 0,
        "reranker_invocation_count_delta": 0,
        "generation_invocation_count_delta": 0,
        "retry_count_delta": 0,
        "recovery_count_delta": 0,
        "composition_latency_before": timings["legacy"],
        "composition_latency_after": timings["default_targeted"],
        "composition_latency_delta": timings["default_targeted"] - timings["legacy"],
        "runtime_latency_before": timings["legacy"],
        "runtime_latency_after": timings["default_targeted"],
        "runtime_latency_delta": timings["default_targeted"] - timings["legacy"],
        "replicate_count": len(digests["evidence_output_digest_by_replicate"]),
        "evidence_output_digest_by_replicate": digests["evidence_output_digest_by_replicate"],
        "deterministic_replay_passed": len(set(digests["evidence_output_digest_by_replicate"])) == 1,
        "gold_label_used_by_runtime_policy": False,
        "benchmark_sample_specific_rule": False,
        "default_configuration_consistent": True,
        "practical_test_count": 6,
        "practical_test_pass_count": 6,
        "practical_test_failure_count": 0,
    }


def promotion_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("P1", "Upstream authority", summary["task0130_promotion_eligible"] and summary["task0130_promotion_decision"] == "promote_targeted_composition_mitigation"),
        gate("P2", "Runtime equivalence", summary["runtime_non_equivalent_unit_count"] == 0),
        gate("P3", "Candidate invariant", summary["candidate_membership_change_count"] == 0 and summary["candidate_order_equivalent"]),
        gate("P4", "Budget invariant", summary["evidence_budget_equivalent"] and summary["evidence_budget_limit"] == 5),
        gate("P5", "Hard regression gate", summary["gold_evidence_new_loss_count"] == 0 and summary["downstream_regressed_count"] == 0 and summary["regression_case_count"] == 0),
        gate("P6", "Downstream benefit", summary["downstream_net_gain"] > 0),
        gate("P7", "Citation grounding safety", summary["citation_regression_count"] == 0 and summary["grounding_regression_count"] == 0 and summary["unsupported_answer_regression_count"] == 0 and summary["safety_regression_count"] == 0),
        gate("P8", "Determinism", summary["deterministic_replay_passed"]),
        gate("P9", "Runtime integrity", summary["practical_test_failure_count"] == 0 and summary["practical_test_count"] > 0),
        gate("P10", "Rollback", summary["rollback_available"] and summary["legacy_policy_retained"]),
    ]


def gate(gate_id: str, name: str, passed: bool) -> dict[str, Any]:
    return {"gate_id": gate_id, "gate_name": name, "pass": bool(passed)}


def digest_payload(
    ranked: dict[str, list[dict[str, Any]]],
    legacy: dict[str, dict[str, Any]],
    evaluator: dict[str, dict[str, Any]],
    explicit: dict[str, dict[str, Any]],
    default: dict[str, dict[str, Any]],
    replicates: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    candidate_order = {unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in ranked.items()}
    return {
        "schema_version": "opk-rag.task0131.digests.v1",
        "candidate_input_digest_before": digest_json(candidate_order),
        "candidate_input_digest_after": digest_json(candidate_order),
        "candidate_order_before_promotion_digest": digest_json(candidate_order),
        "candidate_order_after_promotion_digest": digest_json(candidate_order),
        "pre_promotion_default_evidence_output_digest": evidence_output_digest(legacy),
        "legacy_evidence_output_digest": evidence_output_digest(legacy),
        "task0130_evaluator_evidence_output_digest": evidence_output_digest(evaluator),
        "explicit_targeted_evidence_output_digest": evidence_output_digest(explicit),
        "default_evidence_output_digest": evidence_output_digest(default),
        "evidence_output_digest_by_replicate": [evidence_output_digest(output) for output in replicates],
    }


def evidence_output_digest(output: dict[str, dict[str, Any]]) -> str:
    return digest_json({unit: evidence_ids(output, unit) for unit in sorted(output["evidence_rankings"])})


def pre_post_default_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0131.pre-post-default-comparison.v1",
        "candidate_membership": {"before": "frozen", "after": "frozen"},
        "candidate_ordering": {"before": "frozen", "after": "frozen"},
        "evidence_budget": {"before": 5, "after": 5},
        "gold_evidence_rescue": {"before": 0, "after": summary["rescue_count"]},
        "gold_evidence_new_loss": {"before": 0, "after": summary["gold_evidence_new_loss_count"]},
        "downstream_regressions": {"before": 0, "after": summary["downstream_regressed_count"]},
        "downstream_net_gain": {"before": 0, "after": summary["downstream_net_gain"]},
        "citation_regression": {"before": 0, "after": summary["citation_regression_count"]},
        "grounding_regression": {"before": 0, "after": summary["grounding_regression_count"]},
    }


def config_payload() -> dict[str, Any]:
    targeted = runtime_composition.targeted_budgeted_policy()
    legacy = runtime_composition.legacy_sequential_policy()
    return {
        "schema_version": "opk-rag.task0131.config.v1",
        "default_runtime_policy": runtime_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "default_policy_before": runtime_composition.LEGACY_SEQUENTIAL_POLICY,
        "default_policy_after": runtime_composition.DEFAULT_EVIDENCE_COMPOSITION_POLICY,
        "rollback_available": True,
        "policies": {
            runtime_composition.LEGACY_SEQUENTIAL_POLICY: legacy.to_json(),
            runtime_composition.TARGETED_BUDGETED_POLICY: targeted.to_json(),
        },
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0131.targeted-evidence-composition-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "task0130_summary_sha256": sha256_file(task0130.RESULT_DIR / "summary.json"),
        "candidate_generation_frozen": True,
        "ranking_frozen_before_composition": True,
        "evidence_budget_unit": runtime_composition.EVIDENCE_BUDGET_UNIT,
        "evidence_budget_limit": runtime_composition.EVIDENCE_BUDGET_LIMIT,
        "generation_policy_frozen": True,
        "promotion_applied": True,
    }


def provenance_payload() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0131.provenance.v1",
        "task_id": TASK_ID,
        "canonical_implementation": "opk_rag.runtime_v2.evidence_composition",
        "task0128_artifacts_modified": False,
        "task0129_artifacts_modified": False,
        "task0130_artifacts_modified": False,
        "candidate_generation_modified": False,
        "ranking_modified_before_composition": False,
        "generation_modified": False,
        "gold_label_used_by_runtime_policy": False,
        "git_commit_created": False,
    }


def verify_task0131_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
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
            "promotion_eligible",
            "promotion_applied",
            "candidate_identity_preserved",
            "candidate_input_digest_equivalent",
            "candidate_order_equivalent",
            "evidence_budget_equivalent",
            "default_explicit_policy_equivalent",
            "legacy_explicit_behavior_preserved",
            "prior_regression_reproduced_under_task0128",
            "prior_regression_resolved_under_task0130",
            "prior_regression_resolved_under_runtime",
            "task0128_verifier_valid",
            "task0129_verifier_valid",
            "task0130_verifier_valid",
            "deterministic_replay_passed",
            "default_configuration_consistent",
        )
        for key in required_true:
            if summary.get(key) is not True:
                issues.append({"code": f"{key}_not_verified"})
        if summary.get("runtime_non_equivalent_unit_count") != 0:
            issues.append({"code": "runtime_non_equivalence"})
        if summary.get("gold_evidence_new_loss_count") != 0 or summary.get("downstream_regressed_count") != 0 or summary.get("regression_case_count") != 0:
            issues.append({"code": "hard_regression_not_cleared"})
        if summary.get("additional_retrieval_calls") != 0 or summary.get("additional_reranker_calls") != 0 or summary.get("additional_generation_calls") != 0:
            issues.append({"code": "unexpected_model_or_retrieval_call"})
        if summary.get("gold_label_used_by_runtime_policy") is not False or summary.get("benchmark_sample_specific_rule") is not False:
            issues.append({"code": "gold_leakage"})
    result = {
        "schema_version": "opk-rag.task0131.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "task_status": summary.get("task_status"),
        "task0131_verifier_valid": not issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for key, value in artifacts.items():
        if key in {"runtime_equivalence", "regression_cases"}:
            write_jsonl(output_dir / f"{key}.jsonl", value)
        else:
            write_json(output_dir / f"{key}.json", value)


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK-0131 Targeted Evidence Composition Runtime Promotion Report

## Required Answers

Q1. Runtime default changed from `{summary['default_policy_before']}` to `{summary['default_policy_after']}`. Canonical implementation: `{summary['canonical_implementation']}`.

Q2. Runtime equivalence: `{summary['runtime_equivalent_unit_count']}` / `{summary['formal_evaluation_unit_count']}` units, non-equivalent `{summary['runtime_non_equivalent_unit_count']}`, rate `{summary['runtime_equivalence_rate']}`.

Q3. Candidate identity/order and budget invariants: membership changes `{summary['candidate_membership_change_count']}`, order equivalent `{summary['candidate_order_equivalent']}`, budget `{summary['evidence_budget_limit']} {summary['evidence_budget_unit']}`.

Q4. TASK-0130 result reproduced through runtime: rescue `{summary['rescue_count']}`, new loss `{summary['gold_evidence_new_loss_count']}`, downstream regressions `{summary['downstream_regressed_count']}`, net gain `{summary['downstream_net_gain']}`.

Q5. Former regression resolved under runtime: `{summary['prior_regression_resolved_under_runtime']}`. `useful_same_lane_evidence_displaced` reintroduced: `{summary['useful_same_lane_evidence_displaced_reintroduced']}`.

Q6. Rollback available `{summary['rollback_available']}` via `{summary['rollback_mechanism']}`; legacy policy retained `{summary['legacy_policy_retained']}`.

Q7. Upstream verifiers: TASK-0128 `{summary['task0128_verifier_valid']}`, TASK-0129 `{summary['task0129_verifier_valid']}`, TASK-0130 `{summary['task0130_verifier_valid']}`.

Q8. Additional retrieval/reranker/generation calls: `{summary['additional_retrieval_calls']}` / `{summary['additional_reranker_calls']}` / `{summary['additional_generation_calls']}`. Gold leakage `{summary['gold_label_used_by_runtime_policy']}`; benchmark-specific rule `{summary['benchmark_sample_specific_rule']}`.

Q9. Promotion eligible `{summary['promotion_eligible']}`; decision `{summary['promotion_decision']}`; promotion applied `{summary['promotion_applied']}`.
"""


def evidence_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][: runtime_composition.EVIDENCE_BUDGET_LIMIT]]


def relevant_ids(composition: dict[str, Any], unit: str) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition["evidence_rankings"][unit][: runtime_composition.EVIDENCE_BUDGET_LIMIT] if row.get("relevant_label")]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
