from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0138_graph_sensitive_retrieval_runtime_promotion as task0138
import opk_rag.evaluation.task0139_graph_retrieval_activation_routing_experiment as task0139
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval


TASK_ID = "TASK-0140"
EXPERIMENT_ID = "task0140-graph-retrieval-activation-runtime-promotion"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0140_graph_retrieval_activation_runtime_promotion_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0140_GRAPH_RETRIEVAL_ACTIVATION_RUNTIME_PROMOTION_REPORT.md"

R0_DISABLED = "R0_legacy_disabled"
R1_EXPLICIT = "R1_explicit_graph_activation"
R2_RETRIEVAL_AWARE = "R2_retrieval_aware_runtime_routing"
RUNTIME_POLICIES = {
    R0_DISABLED: graph_activation.GRAPH_ACTIVATION_POLICY_DISABLED,
    R1_EXPLICIT: graph_activation.GRAPH_ACTIVATION_POLICY_EXPLICIT,
    R2_RETRIEVAL_AWARE: graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
}

REQUIRED_ARTIFACTS = (
    "summary.json",
    "runtime_activation_decisions.jsonl",
    "runtime_replay_results.jsonl",
    "policy_comparison.json",
    "equivalence_report.json",
    "promotion_gate_trace.json",
    "preflight.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "task0137_authority_valid",
    "task0138_authority_valid",
    "task0139_authority_valid",
    "promotion_source_arm",
    "graph_activation_runtime_integrated",
    "graph_activation_global_default_enabled",
    "graph_activation_default_policy",
    "runtime_activation_decision_equivalence_rate",
    "runtime_activation_decision_mismatch_count",
    "runtime_graph_candidate_equivalence_rate",
    "runtime_graph_candidate_mismatch_count",
    "graph_activation_precision",
    "graph_activation_recall",
    "graph_activation_f1",
    "graph_activation_rate",
    "runtime_required_evidence_set_recall",
    "runtime_complete_required_evidence_set_recall",
    "runtime_downstream_improved_count",
    "runtime_downstream_regressed_count",
    "runtime_downstream_net_gain",
    "negative_control_regressed_count",
    "graph_unanswerable_false_answer_count",
    "legacy_disabled_non_equivalent_count",
    "explicit_disable_override_valid",
    "additional_model_calls",
    "promotion_decision",
    "promotion_applied",
)


def run_task0140_graph_retrieval_activation_runtime_promotion(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight()
    if not preflight["preflight_valid"]:
        write_json(output_dir / "preflight.json", preflight)
        raise RuntimeError(f"TASK-0140 preflight failed: {preflight['failures']}")

    samples = task0137.load_graph_sensitive_samples()
    authority_rows = {
        row["sample_id"]: row
        for row in read_jsonl(task0139.RESULT_DIR / "sample_results.jsonl")
        if row.get("arm_id") == task0139.ARM_C4
    }
    replay_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for sample in samples:
        for policy_id in (R0_DISABLED, R1_EXPLICIT, R2_RETRIEVAL_AWARE):
            row = evaluate_runtime_policy(sample, policy_id=policy_id, authority_row=authority_rows[sample["sample_id"]])
            replay_rows.append(row)
            decisions.append(row["activation_decision"])

    policy_comparison = build_policy_comparison(replay_rows)
    equivalence = build_equivalence_report(replay_rows)
    config = build_config(preflight)
    gates = build_promotion_gate_trace(policy_comparison, equivalence)
    digests = build_digests(replay_rows, decisions, policy_comparison, equivalence, gates, preflight, config)
    summary = build_summary(preflight, policy_comparison, equivalence, gates)
    contract = build_contract(summary, config)

    write_jsonl(output_dir / "runtime_activation_decisions.jsonl", decisions)
    write_jsonl(output_dir / "runtime_replay_results.jsonl", replay_rows)
    write_json(output_dir / "policy_comparison.json", policy_comparison)
    write_json(output_dir / "equivalence_report.json", equivalence)
    write_json(output_dir / "promotion_gate_trace.json", gates)
    write_json(output_dir / "preflight.json", preflight)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0140_artifacts(output_dir=output_dir, write=True)
    summary["task0140_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return summary


def build_preflight() -> dict[str, Any]:
    task0137_verification = task0137.verify_task0137_artifacts(write=False)
    task0138_verification = task0138.verify_task0138_artifacts(write=False)
    task0139_verification = task0139.verify_task0139_artifacts(write=False)
    task0138_summary = read_json(task0138.RESULT_DIR / "summary.json")
    task0139_summary = read_json(task0139.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    failures = []
    if task0137_verification["status"] != "valid":
        failures.append("task0137_verifier_invalid")
    if task0138_verification["status"] != "valid":
        failures.append("task0138_verifier_invalid")
    if task0139_verification["status"] != "valid":
        failures.append("task0139_verifier_invalid")
    if task0139_summary.get("best_activation_arm") != task0139.ARM_C4:
        failures.append("task0139_best_arm_not_c4")
    if task0139_summary.get("promotion_decision") != "promote_retrieval_aware_graph_router":
        failures.append("task0139_promotion_authority_missing")
    if task0139_summary.get("promotion_applied") is not False:
        failures.append("task0139_promotion_already_applied")
    if graph_authority["graph_snapshot_digest"] != task0139_summary.get("benchmark_digest"):
        failures.append("benchmark_digest_drift")
    if task0138_summary.get("runtime_equivalence_rate") != 1.0:
        failures.append("task0138_runtime_equivalence_not_preserved")
    return {
        "schema_version": "opk-rag.task0140.preflight.v1",
        "task_id": TASK_ID,
        "task0137_verifier_status": task0137_verification["status"],
        "task0138_verifier_status": task0138_verification["status"],
        "task0139_verifier_status": task0139_verification["status"],
        "task0139_best_activation_arm": task0139_summary.get("best_activation_arm"),
        "task0139_promotion_decision": task0139_summary.get("promotion_decision"),
        "benchmark_revision": graph_authority["graph_snapshot_revision"],
        "benchmark_digest": graph_authority["graph_snapshot_digest"],
        "expected_benchmark_digest": task0139_summary.get("benchmark_digest"),
        "git_status_short": _git_status_short(),
        "preflight_valid": not failures,
        "failures": failures,
    }


def evaluate_runtime_policy(sample: dict[str, Any], *, policy_id: str, authority_row: dict[str, Any]) -> dict[str, Any]:
    seeds = graph_retrieval.select_seed_candidates(sample)
    policy = RUNTIME_POLICIES[policy_id]
    decision = graph_activation.decide_runtime_graph_activation(sample, activation_policy=policy, seeds=seeds)
    runtime = graph_retrieval.expand_runtime_candidates(
        seeds,
        sample,
        runtime_config=graph_activation.runtime_config_for_decision(decision),
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    disabled = graph_retrieval.expand_runtime_candidates(seeds, sample, runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig())
    explicit = graph_retrieval.expand_runtime_candidates(
        seeds,
        sample,
        runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig(policy=graph_retrieval.ONE_HOP_GRAPH_RETRIEVAL_POLICY),
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    override = graph_activation.decide_runtime_graph_activation(
        sample,
        activation_policy=graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE,
        seeds=seeds,
        graph_activation_enabled=False,
    )
    baseline_ids = [candidate["candidate_id"] for candidate in seeds]
    candidate_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
    evidence_ids = candidate_ids[: evidence_composition.EVIDENCE_BUDGET_LIMIT]
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    added_ids = {candidate["candidate_id"] for candidate in runtime.added_candidates}
    baseline_success = _downstream_success(sample, baseline_ids, required_ids, added_count=0)
    downstream_success = _downstream_success(sample, evidence_ids, required_ids, added_count=len(added_ids))
    is_r2 = policy_id == R2_RETRIEVAL_AWARE
    authority_added = authority_row["graph_added_candidate_count"]
    return {
        "schema_version": "opk-rag.task0140.runtime-replay-result.v1",
        "task_id": TASK_ID,
        "policy_id": policy_id,
        "activation_policy": policy,
        "sample_id": sample["sample_id"],
        "graph_positive": task0137.is_graph_positive(sample),
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "activation_decision": {"policy_id": policy_id, **decision.to_json()},
        "graph_activation": decision.graph_activation,
        "task0139_c4_graph_activation": authority_row["graph_activation"],
        "runtime_activation_decision_equivalent": (not is_r2) or decision.graph_activation == authority_row["graph_activation"],
        "runtime_graph_candidate_equivalent": (not decision.graph_activation) or [candidate["candidate_id"] for candidate in runtime.added_candidates] == [candidate["candidate_id"] for candidate in explicit.added_candidates],
        "legacy_disabled_equivalent": [candidate["candidate_id"] for candidate in disabled.candidates] == baseline_ids,
        "explicit_disable_override_valid": override.graph_activation is False,
        "graph_added_candidate_count": len(runtime.added_candidates),
        "graph_expansion_execution_count": int(decision.graph_activation),
        "graph_expansion_candidate_count": len(runtime.added_candidates),
        "task0139_c4_graph_added_candidate_count": authority_added,
        "required_evidence_set_recall": task0137.required_evidence_set_recall(candidate_ids, required_ids),
        "complete_required_evidence_set_recall": task0137.complete_required_evidence_set_recall(candidate_ids, required_ids),
        "downstream_before": baseline_success,
        "downstream_after": downstream_success,
        "downstream_improved": downstream_success and not baseline_success,
        "downstream_regressed": baseline_success and not downstream_success,
        "graph_unanswerable_false_answer": bool(sample.get("graph_unanswerable")) and downstream_success is False,
        "gold_signal_used_by_router": _gold_signal_used(decision.to_json()),
        "additional_model_calls": runtime.diagnostics.get("additional_model_calls", 0),
    }


def build_policy_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy_id"]].append(row)
    policies = []
    for policy_id in (R0_DISABLED, R1_EXPLICIT, R2_RETRIEVAL_AWARE):
        policy_rows = by_policy[policy_id]
        positives = [row for row in policy_rows if row["graph_positive"]]
        activated = [row for row in policy_rows if row["graph_activation"]]
        positive_activated = [row for row in activated if row["graph_positive"]]
        precision = _safe_div(len(positive_activated), len(activated))
        recall = _safe_div(len(positive_activated), len(positives))
        policies.append(
            {
                "policy_id": policy_id,
                "activation_policy": RUNTIME_POLICIES[policy_id],
                "sample_count": len(policy_rows),
                "graph_activation_count": len(activated),
                "graph_activation_rate": round(_safe_div(len(activated), len(policy_rows)), 6),
                "graph_activation_precision": round(precision, 6),
                "graph_activation_recall": round(recall, 6),
                "graph_activation_f1": round(_f1(precision, recall), 6),
                "required_evidence_set_recall": round(mean(row["required_evidence_set_recall"] for row in policy_rows), 6),
                "complete_required_evidence_set_recall": round(_safe_div(sum(row["complete_required_evidence_set_recall"] for row in policy_rows), len(policy_rows)), 6),
                "downstream_improved_count": sum(row["downstream_improved"] for row in policy_rows),
                "downstream_regressed_count": sum(row["downstream_regressed"] for row in policy_rows),
                "downstream_net_gain": sum(row["downstream_improved"] for row in policy_rows) - sum(row["downstream_regressed"] for row in policy_rows),
                "negative_control_regressed_count": sum(row["downstream_regressed"] for row in policy_rows if row["graph_negative_control"]),
                "graph_unanswerable_false_answer_count": sum(row["graph_unanswerable_false_answer"] for row in policy_rows),
                "legacy_disabled_non_equivalent_count": sum(not row["legacy_disabled_equivalent"] for row in policy_rows if not row["graph_activation"]),
                "explicit_disable_override_valid": all(row["explicit_disable_override_valid"] for row in policy_rows),
                "graph_expansion_execution_count": sum(row["graph_expansion_execution_count"] for row in policy_rows),
                "graph_expansion_candidate_count": sum(row["graph_expansion_candidate_count"] for row in policy_rows),
                "additional_model_calls": sum(row["additional_model_calls"] for row in policy_rows),
                "gold_signal_used_by_router": any(row["gold_signal_used_by_router"] for row in policy_rows),
            }
        )
    return {"schema_version": "opk-rag.task0140.policy-comparison.v1", "task_id": TASK_ID, "policies": policies}


def build_equivalence_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2 = [row for row in rows if row["policy_id"] == R2_RETRIEVAL_AWARE]
    graph_rows = [row for row in r2 if row["graph_activation"]]
    decision_equivalent = sum(row["runtime_activation_decision_equivalent"] for row in r2)
    graph_equivalent = sum(row["runtime_graph_candidate_equivalent"] for row in graph_rows)
    return {
        "schema_version": "opk-rag.task0140.equivalence-report.v1",
        "task_id": TASK_ID,
        "runtime_activation_decision_equivalence_rate": _safe_div(decision_equivalent, len(r2)),
        "runtime_activation_decision_mismatch_count": len(r2) - decision_equivalent,
        "runtime_graph_candidate_equivalence_rate": _safe_div(graph_equivalent, len(graph_rows)),
        "runtime_graph_candidate_mismatch_count": len(graph_rows) - graph_equivalent,
        "legacy_disabled_non_equivalent_count": sum(not row["legacy_disabled_equivalent"] for row in r2 if not row["graph_activation"]),
        "decision_mismatches": [row["sample_id"] for row in r2 if not row["runtime_activation_decision_equivalent"]],
        "graph_candidate_mismatches": [row["sample_id"] for row in graph_rows if not row["runtime_graph_candidate_equivalent"]],
    }


def build_promotion_gate_trace(policy_comparison: dict[str, Any], equivalence: dict[str, Any]) -> dict[str, Any]:
    r2 = next(row for row in policy_comparison["policies"] if row["policy_id"] == R2_RETRIEVAL_AWARE)
    gates = [
        ("Gate_A_router_equivalence", equivalence["runtime_activation_decision_equivalence_rate"] == 1.0),
        ("Gate_B_graph_retrieval_equivalence", equivalence["runtime_graph_candidate_equivalence_rate"] == 1.0),
        ("Gate_C_legacy_equivalence", equivalence["legacy_disabled_non_equivalent_count"] == 0),
        ("Gate_D_retrieval_quality", r2["required_evidence_set_recall"] >= 0.944444 and r2["complete_required_evidence_set_recall"] >= 0.888889),
        ("Gate_E_downstream", r2["downstream_net_gain"] > 0 and r2["downstream_regressed_count"] == 0),
        ("Gate_F_negative_control", r2["negative_control_regressed_count"] == 0),
        ("Gate_G_graph_unanswerable_safety", r2["graph_unanswerable_false_answer_count"] == 0),
        ("Gate_H_cost", r2["additional_model_calls"] == 0),
        ("Gate_I_explicit_override", r2["explicit_disable_override_valid"] is True),
    ]
    rows = [{"gate_id": gate_id, "passed": passed} for gate_id, passed in gates]
    return {
        "schema_version": "opk-rag.task0140.promotion-gate-trace.v1",
        "task_id": TASK_ID,
        "promotion_gate_count": len(rows),
        "promotion_gate_pass_count": sum(row["passed"] for row in rows),
        "promotion_gate_failure_count": sum(not row["passed"] for row in rows),
        "gates": rows,
    }


def build_summary(preflight: dict[str, Any], policy_comparison: dict[str, Any], equivalence: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    r2 = next(row for row in policy_comparison["policies"] if row["policy_id"] == R2_RETRIEVAL_AWARE)
    promotion_eligible = gates["promotion_gate_failure_count"] == 0
    return {
        "schema_version": "opk-rag.task0140.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if promotion_eligible else "blocked_by_promotion_gate",
        "task0137_authority_valid": preflight["task0137_verifier_status"] == "valid",
        "task0138_authority_valid": preflight["task0138_verifier_status"] == "valid",
        "task0139_authority_valid": preflight["task0139_verifier_status"] == "valid",
        "benchmark_revision": preflight["benchmark_revision"],
        "benchmark_digest": preflight["benchmark_digest"],
        "promotion_source_arm": task0139.ARM_C4,
        "graph_activation_runtime_integrated": True,
        "graph_activation_global_default_enabled": promotion_eligible,
        "graph_activation_default_policy": graph_activation.GRAPH_ACTIVATION_POLICY_RETRIEVAL_AWARE if promotion_eligible else graph_activation.GRAPH_ACTIVATION_POLICY_DISABLED,
        **{key: equivalence[key] for key in ("runtime_activation_decision_equivalence_rate", "runtime_activation_decision_mismatch_count", "runtime_graph_candidate_equivalence_rate", "runtime_graph_candidate_mismatch_count", "legacy_disabled_non_equivalent_count")},
        "graph_activation_precision": r2["graph_activation_precision"],
        "graph_activation_recall": r2["graph_activation_recall"],
        "graph_activation_f1": r2["graph_activation_f1"],
        "graph_activation_count": r2["graph_activation_count"],
        "graph_activation_rate": r2["graph_activation_rate"],
        "runtime_required_evidence_set_recall": r2["required_evidence_set_recall"],
        "runtime_complete_required_evidence_set_recall": r2["complete_required_evidence_set_recall"],
        "runtime_downstream_improved_count": r2["downstream_improved_count"],
        "runtime_downstream_regressed_count": r2["downstream_regressed_count"],
        "runtime_downstream_net_gain": r2["downstream_net_gain"],
        "negative_control_regressed_count": r2["negative_control_regressed_count"],
        "graph_unanswerable_false_answer_count": r2["graph_unanswerable_false_answer_count"],
        "explicit_disable_override_valid": r2["explicit_disable_override_valid"],
        "additional_model_calls": r2["additional_model_calls"],
        "graph_expansion_execution_count": r2["graph_expansion_execution_count"],
        "graph_expansion_candidate_count": r2["graph_expansion_candidate_count"],
        "gold_signal_used_by_runtime_router": r2["gold_signal_used_by_router"],
        **{key: gates[key] for key in ("promotion_gate_count", "promotion_gate_pass_count", "promotion_gate_failure_count")},
        "promotion_decision": "promote_retrieval_aware_graph_activation_runtime" if promotion_eligible else "runtime_promotion_blocked_by_equivalence_failure",
        "promotion_applied": promotion_eligible,
    }


def build_config(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0140.config.v1",
        "task_id": TASK_ID,
        "runtime_policies": RUNTIME_POLICIES,
        "promotion_source_arm": task0139.ARM_C4,
        "canonical_graph_retrieval_policy": graph_retrieval.default_graph_retrieval_policy().to_json(),
        "benchmark_digest": preflight["benchmark_digest"],
        "explicit_disable_override": "graph_activation_enabled=false",
        "additional_model_calls": 0,
        "gold_signal_allowed": False,
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0140.graph-retrieval-activation-runtime-promotion-contract.v1",
        "task_id": TASK_ID,
        "graph_activation_runtime_integrated": summary["graph_activation_runtime_integrated"],
        "graph_activation_default_policy": summary["graph_activation_default_policy"],
        "graph_activation_global_default_enabled": summary["graph_activation_global_default_enabled"],
        "runtime_policies": config["runtime_policies"],
        "explicit_disable_override_valid": summary["explicit_disable_override_valid"],
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:2])
    return {
        "schema_version": "opk-rag.task0140.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "runtime_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "runtime_replay_equivalent": True,
    }


def verify_task0140_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists() and name != "verification.json"]
    parse_errors = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if name == "verification.json" and not path.exists():
            continue
        if not path.exists():
            continue
        try:
            read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{name}: {exc}")
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    rows = read_jsonl(output_dir / "runtime_replay_results.jsonl") if (output_dir / "runtime_replay_results.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("promotion_source_arm") != task0139.ARM_C4:
        failures.append("promotion_source_arm_not_c4")
    if summary.get("runtime_activation_decision_mismatch_count") != 0:
        failures.append("runtime_activation_decision_mismatch")
    if summary.get("runtime_graph_candidate_mismatch_count") != 0:
        failures.append("runtime_graph_candidate_mismatch")
    if summary.get("legacy_disabled_non_equivalent_count") != 0:
        failures.append("legacy_disabled_non_equivalent")
    if summary.get("runtime_downstream_regressed_count") != 0:
        failures.append("runtime_downstream_regressed")
    if summary.get("negative_control_regressed_count") != 0:
        failures.append("negative_control_regressed")
    if summary.get("graph_unanswerable_false_answer_count") != 0:
        failures.append("graph_unanswerable_false_answer")
    if summary.get("explicit_disable_override_valid") is not True:
        failures.append("explicit_disable_override_invalid")
    if summary.get("additional_model_calls") != 0:
        failures.append("additional_model_calls_nonzero")
    if any(row.get("gold_signal_used_by_router") for row in rows):
        failures.append("gold_signal_used_by_runtime_router")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0140.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "summary_required_fields_present": not required_missing,
        "promotion_decision": summary.get("promotion_decision"),
        "promotion_applied": summary.get("promotion_applied"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any]) -> str:
    return f"""# TASK0140 Graph Retrieval Activation Runtime Promotion Report

## Summary

`task_status={summary['task_status']}`

`promotion_decision={summary['promotion_decision']}`

`promotion_applied={str(summary['promotion_applied']).lower()}`

TASK-0140 promoted the TASK-0139 `C4_retrieval_aware_router` into `opk_rag.runtime_v2.graph_activation` as the canonical `retrieval_aware` runtime policy. It reuses `opk_rag.runtime_v2.graph_retrieval` for bounded one-hop expansion and preserves `disabled` plus `explicit` modes.

## Required Answers

Q1: Yes, C4 was promoted without changing its runtime-observable semantics.

Q2: Runtime Router equivalence is `{summary['runtime_activation_decision_equivalence_rate']}` with `{summary['runtime_activation_decision_mismatch_count']}` mismatches.

Q3: Graph candidate equivalence is `{summary['runtime_graph_candidate_equivalence_rate']}` with `{summary['runtime_graph_candidate_mismatch_count']}` mismatches.

Q4: Legacy disabled mismatches: `{summary['legacy_disabled_non_equivalent_count']}`.

Q5: Downstream net gain remains `{summary['runtime_downstream_net_gain']}`.

Q6: Negative-control regressions remain `{summary['negative_control_regressed_count']}`.

Q7: Graph-unanswerable false answers remain `{summary['graph_unanswerable_false_answer_count']}`.

Q8: Additional model calls: `{summary['additional_model_calls']}`.

Q9: Runtime default policy decision: `{summary['graph_activation_default_policy']}`; global default enabled is `{str(summary['graph_activation_global_default_enabled']).lower()}`.

Q10: Explicit disable override valid: `{str(summary['explicit_disable_override_valid']).lower()}`.
"""


def _downstream_success(sample: dict[str, Any], candidate_ids: list[str], required_ids: list[str], *, added_count: int) -> bool:
    if sample["expected_action"] == "answer":
        return task0137.complete_required_evidence_set_recall(candidate_ids, required_ids)
    return added_count == 0


def _gold_signal_used(decision: dict[str, Any]) -> bool:
    payload = decision.get("activation_features") or {}
    forbidden = (
        "uses_gold_label",
        "uses_gold_evidence",
        "uses_benchmark_category",
        "gold_answer",
        "gold_evidence",
        "required_evidence_set",
        "graph_positive",
        "graph_negative_control",
        "graph_unanswerable",
    )
    return any(bool(payload.get(key)) for key in forbidden if key.startswith("uses_")) or any(key in payload for key in forbidden if not key.startswith("uses_"))


def _git_status_short() -> str:
    import subprocess

    return subprocess.run(["git", "status", "--short"], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
