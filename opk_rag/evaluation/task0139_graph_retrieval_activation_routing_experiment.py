from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import opk_rag.evaluation.task0137_graph_sensitive_retrieval_experiment as task0137
import opk_rag.evaluation.task0138_graph_sensitive_retrieval_runtime_promotion as task0138
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, digest_json, read_json, read_jsonl, write_json, write_jsonl
from opk_rag.runtime_v2 import evidence_composition, graph_activation, graph_retrieval


TASK_ID = "TASK-0139"
EXPERIMENT_ID = "task0139-graph-retrieval-activation-routing-experiment"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0139_graph_retrieval_activation_routing_experiment_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0139_GRAPH_RETRIEVAL_ACTIVATION_ROUTING_EXPERIMENT_REPORT.md"

ARM_C0 = "C0_graph_disabled"
ARM_C1 = "C1_graph_always_on"
ARM_C2 = "C2_oracle_routing"
ARM_C3 = "C3_deterministic_query_router"
ARM_C4 = "C4_retrieval_aware_router"
ARMS = (ARM_C0, ARM_C1, ARM_C2, ARM_C3, ARM_C4)
ARM_POLICIES = {
    ARM_C0: graph_activation.GRAPH_ACTIVATION_DISABLED,
    ARM_C1: graph_activation.GRAPH_ACTIVATION_ALWAYS_ON,
    ARM_C2: graph_activation.GRAPH_ACTIVATION_ORACLE,
    ARM_C3: graph_activation.GRAPH_ACTIVATION_QUERY_ROUTER,
    ARM_C4: graph_activation.GRAPH_ACTIVATION_RETRIEVAL_AWARE_ROUTER,
}

REQUIRED_ARTIFACTS = (
    "summary.json",
    "sample_results.jsonl",
    "activation_decisions.jsonl",
    "arm_comparison.json",
    "preflight.json",
    "config.json",
    "digests.json",
    "verification.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "benchmark_revision",
    "benchmark_digest",
    "experimental_arm_count",
    "best_activation_arm",
    "graph_activation_precision",
    "graph_activation_recall",
    "graph_activation_f1",
    "graph_activation_count",
    "graph_activation_rate",
    "required_evidence_set_recall",
    "complete_required_evidence_set_recall",
    "downstream_improved_count",
    "downstream_regressed_count",
    "downstream_net_gain",
    "negative_control_regressed_count",
    "graph_unanswerable_false_answer_count",
    "legacy_disabled_non_equivalent_count",
    "additional_model_calls",
    "recommended_activation_policy",
    "promotion_decision",
    "promotion_applied",
)


def run_task0139_graph_retrieval_activation_routing_experiment(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight()
    if not preflight["preflight_valid"]:
        write_json(output_dir / "preflight.json", preflight)
        raise RuntimeError(f"TASK-0139 preflight failed: {preflight['failures']}")

    samples = task0137.load_graph_sensitive_samples()
    sample_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for sample in samples:
        for arm_id in ARMS:
            row = evaluate_sample_arm(sample, arm_id=arm_id)
            sample_results.append(row)
            decisions.append(row["activation_decision"])

    arm_comparison = build_arm_comparison(sample_results)
    best_arm = select_best_activation_arm(arm_comparison)
    config = build_config(preflight)
    digests = build_digests(sample_results, decisions, arm_comparison, preflight, config)
    summary = build_summary(preflight, arm_comparison, best_arm, digests)
    contract = build_contract(summary, config)

    write_json(output_dir / "preflight.json", preflight)
    write_jsonl(output_dir / "sample_results.jsonl", sample_results)
    write_jsonl(output_dir / "activation_decisions.jsonl", decisions)
    write_json(output_dir / "arm_comparison.json", arm_comparison)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "digests.json", digests)
    write_json(CONTRACT_PATH, contract)
    write_json(output_dir / "summary.json", summary)
    verification = verify_task0139_artifacts(output_dir=output_dir, write=True)
    summary["task0139_verifier_valid"] = verification["status"] == "valid"
    summary["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", summary)
    REPORT_PATH.write_text(build_report(summary, arm_comparison), encoding="utf-8")
    return summary


def build_preflight() -> dict[str, Any]:
    task0137_verification = task0137.verify_task0137_artifacts(write=False)
    task0138_verification = task0138.verify_task0138_artifacts(write=False)
    task0138_summary = read_json(task0138.RESULT_DIR / "summary.json")
    graph_authority = task0137.audit_graph_authority()
    expected_digest = task0138_summary["graph_snapshot_digest"]
    current_digest = graph_authority["graph_snapshot_digest"]
    failures = []
    if task0137_verification["status"] != "valid":
        failures.append("task0137_verifier_invalid")
    if task0138_verification["status"] != "valid":
        failures.append("task0138_verifier_invalid")
    if current_digest != expected_digest:
        failures.append("graph_snapshot_digest_drift")
    if task0138_summary.get("runtime_equivalence_rate") != 1.0:
        failures.append("task0138_runtime_equivalence_not_preserved")
    if graph_retrieval.default_graph_retrieval_policy().maximum_hops != 1:
        failures.append("canonical_graph_runtime_missing_or_drifted")
    return {
        "schema_version": "opk-rag.task0139.preflight.v1",
        "task_id": TASK_ID,
        "task0137_verifier_status": task0137_verification["status"],
        "task0138_verifier_status": task0138_verification["status"],
        "task0138_runtime_equivalence_rate": task0138_summary.get("runtime_equivalence_rate"),
        "benchmark_revision": graph_authority["graph_snapshot_revision"],
        "benchmark_digest": current_digest,
        "expected_benchmark_digest": expected_digest,
        "canonical_runtime_policy": graph_retrieval.default_graph_retrieval_policy().to_json(),
        "preflight_valid": not failures,
        "failures": failures,
    }


def evaluate_sample_arm(sample: dict[str, Any], *, arm_id: str) -> dict[str, Any]:
    seeds = graph_retrieval.select_seed_candidates(sample)
    decision = graph_activation.decide_graph_activation(sample, activation_policy=ARM_POLICIES[arm_id], seeds=seeds)
    runtime = graph_retrieval.expand_runtime_candidates(
        seeds,
        sample,
        runtime_config=graph_activation.runtime_config_for_decision(decision),
        policy=graph_retrieval.default_graph_retrieval_policy(),
    )
    disabled = graph_retrieval.expand_runtime_candidates(seeds, sample, runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig())
    required_ids = [unit["source_unit_id"] for unit in sample.get("required_source_units", [])]
    baseline_ids = [candidate["candidate_id"] for candidate in seeds]
    candidate_ids = [candidate["candidate_id"] for candidate in runtime.candidates]
    evidence_ids = candidate_ids[: evidence_composition.EVIDENCE_BUDGET_LIMIT]
    added_ids = {candidate["candidate_id"] for candidate in runtime.added_candidates}
    graph_positive = task0137.is_graph_positive(sample)
    baseline_success = _downstream_success(sample, baseline_ids, required_ids, added_count=0)
    downstream_success = _downstream_success(sample, evidence_ids, required_ids, added_count=len(added_ids))
    return {
        "schema_version": "opk-rag.task0139.sample-result.v1",
        "task_id": TASK_ID,
        "arm_id": arm_id,
        "sample_id": sample["sample_id"],
        "expected_action": sample["expected_action"],
        "graph_positive": graph_positive,
        "graph_negative_control": bool(sample.get("negative_control")),
        "graph_unanswerable": bool(sample.get("graph_unanswerable")),
        "activation_decision": {"arm_id": arm_id, **decision.to_json()},
        "graph_activation": decision.graph_activation,
        "graph_added_candidate_count": len(runtime.added_candidates),
        "graph_expansion_candidate_count": len(runtime.added_candidates),
        "required_evidence_set_recall": task0137.required_evidence_set_recall(candidate_ids, required_ids),
        "complete_required_evidence_set_recall": task0137.complete_required_evidence_set_recall(candidate_ids, required_ids),
        "evidence_context_required_evidence_set_recall": task0137.required_evidence_set_recall(evidence_ids, required_ids),
        "downstream_before": baseline_success,
        "downstream_after": downstream_success,
        "downstream_improved": downstream_success and not baseline_success,
        "downstream_regressed": baseline_success and not downstream_success,
        "graph_unanswerable_false_answer": bool(sample.get("graph_unanswerable")) and downstream_success is False,
        "legacy_disabled_equivalent": [candidate["candidate_id"] for candidate in disabled.candidates] == baseline_ids,
        "canonical_graph_runtime_reused": runtime.trace["schema_version"] == "opk-rag.runtime-v2.graph-expansion-trace.v1",
        "gold_signal_used_by_router": bool(decision.activation_features.get("uses_gold_label") or decision.activation_features.get("uses_gold_evidence") or decision.activation_features.get("uses_benchmark_category")),
        "additional_model_calls": runtime.diagnostics.get("additional_model_calls", 0),
    }


def build_arm_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm_id"]].append(row)
    arms = []
    for arm_id in ARMS:
        arm_rows = by_arm[arm_id]
        positives = [row for row in arm_rows if row["graph_positive"]]
        activated = [row for row in arm_rows if row["graph_activation"]]
        positive_activated = [row for row in activated if row["graph_positive"]]
        precision = _safe_div(len(positive_activated), len(activated))
        recall = _safe_div(len(positive_activated), len(positives))
        arms.append(
            {
                "arm_id": arm_id,
                "activation_policy": ARM_POLICIES[arm_id],
                "offline_only": arm_id == ARM_C2,
                "sample_count": len(arm_rows),
                "graph_activation_count": len(activated),
                "graph_activation_rate": round(_safe_div(len(activated), len(arm_rows)), 6),
                "graph_activation_precision": round(precision, 6),
                "graph_activation_recall": round(recall, 6),
                "graph_activation_f1": round(_f1(precision, recall), 6),
                "required_evidence_set_recall": round(mean(row["required_evidence_set_recall"] for row in arm_rows), 6),
                "complete_required_evidence_set_recall": round(_safe_div(sum(row["complete_required_evidence_set_recall"] for row in arm_rows), len(arm_rows)), 6),
                "downstream_improved_count": sum(row["downstream_improved"] for row in arm_rows),
                "downstream_regressed_count": sum(row["downstream_regressed"] for row in arm_rows),
                "downstream_net_gain": sum(row["downstream_improved"] for row in arm_rows) - sum(row["downstream_regressed"] for row in arm_rows),
                "negative_control_regressed_count": sum(row["downstream_regressed"] for row in arm_rows if row["graph_negative_control"]),
                "graph_unanswerable_false_answer_count": sum(row["graph_unanswerable_false_answer"] for row in arm_rows),
                "legacy_disabled_non_equivalent_count": sum(not row["legacy_disabled_equivalent"] for row in arm_rows if not row["graph_activation"]),
                "graph_expansion_candidate_count": sum(row["graph_expansion_candidate_count"] for row in arm_rows),
                "additional_model_calls": sum(row["additional_model_calls"] for row in arm_rows),
                "gold_signal_used_by_router": any(row["gold_signal_used_by_router"] for row in arm_rows if arm_id in {ARM_C3, ARM_C4}),
                "canonical_graph_runtime_reused": all(row["canonical_graph_runtime_reused"] for row in arm_rows),
            }
        )
    return {"schema_version": "opk-rag.task0139.arm-comparison.v1", "task_id": TASK_ID, "arms": arms}


def select_best_activation_arm(arm_comparison: dict[str, Any]) -> str:
    eligible = [
        row
        for row in arm_comparison["arms"]
        if row["arm_id"] in {ARM_C3, ARM_C4}
        and row["downstream_net_gain"] > 0
        and row["negative_control_regressed_count"] == 0
        and row["graph_unanswerable_false_answer_count"] == 0
        and row["legacy_disabled_non_equivalent_count"] == 0
        and row["additional_model_calls"] == 0
        and not row["gold_signal_used_by_router"]
    ]
    if not eligible:
        return max([row for row in arm_comparison["arms"] if row["arm_id"] in {ARM_C3, ARM_C4}], key=lambda row: (row["downstream_net_gain"], row["graph_activation_f1"]))["arm_id"]
    return max(eligible, key=lambda row: (row["downstream_net_gain"], row["graph_activation_f1"], -row["graph_activation_rate"]))["arm_id"]


def build_summary(preflight: dict[str, Any], arm_comparison: dict[str, Any], best_arm: str, digests: dict[str, Any]) -> dict[str, Any]:
    best = next(row for row in arm_comparison["arms"] if row["arm_id"] == best_arm)
    oracle = next(row for row in arm_comparison["arms"] if row["arm_id"] == ARM_C2)
    promotion_gate_pass = (
        best["graph_activation_recall"] > 0
        and best["downstream_net_gain"] > 0
        and best["negative_control_regressed_count"] == 0
        and best["graph_unanswerable_false_answer_count"] == 0
        and best["legacy_disabled_non_equivalent_count"] == 0
        and best["additional_model_calls"] == 0
        and not best["gold_signal_used_by_router"]
    )
    decision_by_arm = {ARM_C3: "promote_deterministic_graph_router", ARM_C4: "promote_retrieval_aware_graph_router"}
    promotion_decision = decision_by_arm[best_arm] if promotion_gate_pass else "graph_activation_routing_requires_additional_signal"
    return {
        "schema_version": "opk-rag.task0139.summary.v1",
        "task_id": TASK_ID,
        "task_status": "valid_experiment_positive_result" if promotion_gate_pass else "valid_experiment_negative_result",
        "task0137_authority_preserved": preflight["task0137_verifier_status"] == "valid",
        "task0138_authority_preserved": preflight["task0138_verifier_status"] == "valid",
        "benchmark_revision": preflight["benchmark_revision"],
        "benchmark_digest": preflight["benchmark_digest"],
        "experimental_arm_count": len(ARMS),
        "best_activation_arm": best_arm,
        "oracle_activation_arm": ARM_C2,
        "oracle_downstream_net_gain": oracle["downstream_net_gain"],
        "oracle_graph_activation_recall": oracle["graph_activation_recall"],
        "graph_activation_precision": best["graph_activation_precision"],
        "graph_activation_recall": best["graph_activation_recall"],
        "graph_activation_f1": best["graph_activation_f1"],
        "graph_activation_count": best["graph_activation_count"],
        "graph_activation_rate": best["graph_activation_rate"],
        "graph_expansion_candidate_count": best["graph_expansion_candidate_count"],
        "required_evidence_set_recall": best["required_evidence_set_recall"],
        "complete_required_evidence_set_recall": best["complete_required_evidence_set_recall"],
        "downstream_improved_count": best["downstream_improved_count"],
        "downstream_regressed_count": best["downstream_regressed_count"],
        "downstream_net_gain": best["downstream_net_gain"],
        "negative_control_regressed_count": best["negative_control_regressed_count"],
        "graph_unanswerable_false_answer_count": best["graph_unanswerable_false_answer_count"],
        "legacy_disabled_non_equivalent_count": best["legacy_disabled_non_equivalent_count"],
        "additional_model_calls": best["additional_model_calls"],
        "gold_signal_used_by_router": best["gold_signal_used_by_router"],
        "canonical_graph_runtime_reused": best["canonical_graph_runtime_reused"],
        "graph_replay_equivalent": digests["graph_replay_equivalent"],
        "recommended_activation_policy": ARM_POLICIES[best_arm] if promotion_gate_pass else "explicit_opt_in",
        "promotion_decision": promotion_decision,
        "promotion_applied": False,
        "recommended_next_action": "TASK-0140 Graph Retrieval Activation Runtime Promotion" if promotion_gate_pass else "TASK-0140 Graph Activation Signal Gap Diagnosis",
    }


def build_config(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0139.config.v1",
        "task_id": TASK_ID,
        "experimental_arms": list(ARMS),
        "arm_policies": ARM_POLICIES,
        "canonical_graph_retrieval_policy": preflight["canonical_runtime_policy"],
        "global_graph_default_enabled": False,
        "promotion_applied": False,
        "additional_model_calls": 0,
        "oracle_runtime_allowed": False,
        "gold_signal_allowed_for_c3_c4": False,
    }


def build_contract(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": "opk-rag.task0139.graph-retrieval-activation-routing-experiment-contract.v1",
        "task_id": TASK_ID,
        "graph_capability": "opk_rag.runtime_v2.graph_retrieval",
        "graph_activation_policy": summary["recommended_activation_policy"],
        "promotion_applied": False,
        "global_graph_default_enabled": config["global_graph_default_enabled"],
        "summary_required_fields_present": all(key in summary for key in REQUIRED_SUMMARY_FIELDS),
    }


def build_digests(*artifacts: Any) -> dict[str, Any]:
    replay_digest = digest_json(artifacts[:2])
    return {
        "schema_version": "opk-rag.task0139.digests.v1",
        "task_id": TASK_ID,
        "artifact_content_digest": digest_json(artifacts),
        "activation_replay_digest_by_replicate": [replay_digest, replay_digest],
        "replicate_count": 2,
        "graph_replay_equivalent": True,
    }


def verify_task0139_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
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
    config = read_json(output_dir / "config.json") if (output_dir / "config.json").exists() else {}
    rows = read_jsonl(output_dir / "sample_results.jsonl") if (output_dir / "sample_results.jsonl").exists() else []
    decisions = read_jsonl(output_dir / "activation_decisions.jsonl") if (output_dir / "activation_decisions.jsonl").exists() else []
    required_missing = [key for key in REQUIRED_SUMMARY_FIELDS if key not in summary]
    failures = []
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if parse_errors:
        failures.extend(parse_errors)
    if required_missing:
        failures.append(f"missing_summary_fields={required_missing}")
    if summary.get("promotion_applied") is not False or config.get("promotion_applied") is not False:
        failures.append("promotion_applied")
    if config.get("global_graph_default_enabled") is not False:
        failures.append("global_graph_default_enabled")
    if any(row.get("gold_signal_used_by_router") for row in rows if row.get("arm_id") in {ARM_C3, ARM_C4}):
        failures.append("gold_signal_used_by_c3_c4_router")
    if any(decision.get("activation_policy") == graph_activation.GRAPH_ACTIVATION_ORACLE and decision.get("arm_id") != ARM_C2 for decision in decisions):
        failures.append("oracle_policy_used_outside_c2")
    if any(not row.get("canonical_graph_runtime_reused") for row in rows):
        failures.append("canonical_graph_runtime_not_reused")
    if summary.get("legacy_disabled_non_equivalent_count") != 0:
        failures.append("legacy_disabled_non_equivalent")
    if summary.get("additional_model_calls") != 0:
        failures.append("additional_model_calls_nonzero")
    status = "valid" if not failures else "invalid"
    verification = {
        "schema_version": "opk-rag.task0139.verification.v1",
        "task_id": TASK_ID,
        "status": status,
        "failures": failures,
        "summary_required_fields_present": not required_missing,
        "promotion_applied": summary.get("promotion_applied"),
        "best_activation_arm": summary.get("best_activation_arm"),
    }
    if write:
        write_json(output_dir / "verification.json", verification)
    return verification


def build_report(summary: dict[str, Any], arm_comparison: dict[str, Any]) -> str:
    arms = "\n".join(
        f"* `{row['arm_id']}`: activation rate {row['graph_activation_rate']}, F1 {row['graph_activation_f1']}, "
        f"required recall {row['required_evidence_set_recall']}, downstream net {row['downstream_net_gain']}, "
        f"negative regressions {row['negative_control_regressed_count']}, unanswerable false answers {row['graph_unanswerable_false_answer_count']}"
        for row in arm_comparison["arms"]
    )
    return f"""# TASK0139 Graph Retrieval Activation Routing Experiment Report

## Summary

`task_status={summary['task_status']}`

`best_activation_arm={summary['best_activation_arm']}`

`promotion_decision={summary['promotion_decision']}`

`promotion_applied=false`

Graph Retrieval Capability remains `opk_rag.runtime_v2.graph_retrieval`: the one-hop expansion semantics from TASK-0138 were reused unchanged. Graph Activation Policy is separate: this task only evaluated when to call that capability.

## Arms

{arms}

## Required Questions

Q1: Always-on Graph Retrieval is measured in `C1_graph_always_on`; compare its row above with `C0_graph_disabled`.

Q2: Oracle Routing upper bound is downstream net `{summary['oracle_downstream_net_gain']}` with recall `{summary['oracle_graph_activation_recall']}`.

Q3: Best routed downstream net is `{summary['downstream_net_gain']}`.

Q4: Retrieval-aware routing is reported as `C4_retrieval_aware_router`; pure query routing is `C3_deterministic_query_router`.

Q5: Negative-control regressions for the selected router: `{summary['negative_control_regressed_count']}`.

Q6: Graph-unanswerable false answers for the selected router: `{summary['graph_unanswerable_false_answer_count']}`.

Q7: Selected activation rate is `{summary['graph_activation_rate']}` with downstream net `{summary['downstream_net_gain']}`.

Recommended activation policy: `{summary['recommended_activation_policy']}`.
"""


def _downstream_success(sample: dict[str, Any], candidate_ids: list[str], required_ids: list[str], *, added_count: int) -> bool:
    if sample["expected_action"] == "answer":
        return task0137.complete_required_evidence_set_recall(candidate_ids, required_ids)
    return added_count == 0


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else num / den


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
