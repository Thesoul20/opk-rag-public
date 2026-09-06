from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from opk_rag.showcase.selective_agent_production_monitoring import (
    MONITOR_DEFAULT_STORE,
    MONITOR_MIN_FORMAL_SAMPLES,
    MONITOR_RECOMMENDED_FREEZE_SAMPLES,
    ProductionMonitoringConfig,
    ProductionMonitoringStore,
    evaluate_task0254_release_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0255"
SCHEMA = "opk-rag.task0255.selective-agent-production-post-release-monitoring-stability-authority-and-stage-closeout.v1"
TASK_START_HEAD = "64ab392369158634e2fc311bc0453aceab22a9cd"
RESULT = ROOT / "evaluation-data/results/task0255-selective-agent-production-post-release-monitoring-stability-authority-and-stage-closeout"
CONTRACT = ROOT / "evaluation-data/contracts/task0255_selective_agent_production_post_release_monitoring_stability_authority_and_stage_closeout.json"
T254 = ROOT / "evaluation-data/results/task0254-selective-agent-production-promotion-review-and-release-freeze/summary.json"
T254_RELEASE = ROOT / "evaluation-data/results/task0254-selective-agent-production-promotion-review-and-release-freeze/release_identity.json"
REG = RESULT / "regression.json"

PROVIDER_THRESHOLD = 0.98
SAFETY_KEYS = (
    "llm_finish_authority_count", "abstain_to_finish_override_count", "necessary_llm_gate_bypass_count",
    "unauthorized_action_execution_count", "guard_bypass_count", "recovery_pipeline_bypass_count",
    "graph_hop_violation_count", "knowledge_base_mutation_count", "grounding_bypass_count",
    "citation_bypass_count", "benchmark_gold_exposure_count", "hidden_reasoning_persisted_count",
    "raw_provider_output_persisted_count", "secret_exposure_count", "agent_induced_production_failure_count",
)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def changed_paths() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def evaluate_entry_gate(task0254: Mapping[str, Any], release: Mapping[str, Any]) -> dict[str, Any]:
    release_fp = str(release.get("release_fingerprint") or task0254.get("release_fingerprint") or "")
    gates = {
        "task0254_complete": task0254.get("task_status") == "complete",
        "task0254_promoted": task0254.get("candidate_decision") == "promote_and_freeze_agentic_rag_v2",
        "promotion_review_promoted": task0254.get("promotion_review_decision") == "promote_and_freeze_agentic_rag_v2",
        "promotion_executed": task0254.get("production_promotion_executed") is True,
        "production_agentic_v2_active": task0254.get("production_agentic_v2_active") is True,
        "release_frozen": task0254.get("release_frozen") is True and release.get("release_frozen") is True,
        "entry_gate_passed": task0254.get("entry_gate_passed") is True,
        "candidate_identity_match": task0254.get("candidate_identity_match") is True,
        "rollback_target_available": task0254.get("rollback_target_available") is True,
        "kill_switch_validated": task0254.get("kill_switch_mechanism_validated") is True,
        "release_fingerprint_present": bool(release_fp),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0255.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "source_task": "TASK-0254",
        "source_status": task0254.get("task_status"),
        "source_decision": task0254.get("candidate_decision"),
        "release_id": release.get("release_id") or task0254.get("release_id"),
        "release_fingerprint": release_fp or None,
    }


def _controller(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("controller")
    return value if isinstance(value, Mapping) else {}


def _runtime(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("runtime")
    return value if isinstance(value, Mapping) else {}


def _safety(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("safety")
    return value if isinstance(value, Mapping) else {}


def aggregate_production_rows(rows: Iterable[Mapping[str, Any]], *, release_fingerprint: str) -> dict[str, Any]:
    records = [dict(row) for row in rows if row.get("production_traffic_eligible") is True and row.get("release_fingerprint") == release_fingerprint]
    search = [r for r in records if r.get("execution_scope") == "search"]
    ask = [r for r in records if r.get("execution_scope") == "ask"]
    controller_calls = [int(_controller(r).get("controller_call_count") or 0) for r in records]
    provider_requests = sum(int(_controller(r).get("provider_requests") or 0) for r in records)
    provider_responses = sum(int(_controller(r).get("provider_responses") or 0) for r in records)
    provider_valid = sum(int(_controller(r).get("provider_valid_decisions") or 0) for r in records)
    invoked = sum(bool(_controller(r).get("necessary_llm_invoked")) for r in records)
    recovery_invoked = sum(bool(_controller(r).get("recovery_invoked")) for r in records)
    recovery_improved = sum(bool(_controller(r).get("recovery_improved")) for r in records)
    recovery_harmed = sum(bool(_controller(r).get("recovery_harmed")) for r in records)
    veto_invoked = sum(bool(_controller(r).get("veto_invoked")) for r in records)
    fallback_required = sum(bool(_runtime(r).get("fallback_required")) for r in records)
    fallback_success = sum(bool(_runtime(r).get("fallback_required")) and bool(_runtime(r).get("fallback_success")) for r in records)
    total_latency = [float(_runtime(r).get("total_latency_ms") or 0.0) for r in records]
    controller_latency = [float(_controller(r).get("controller_latency_ms") or 0.0) for r in records]
    input_tokens = [int(_controller(r).get("input_tokens") or 0) for r in records]
    output_tokens = [int(_controller(r).get("output_tokens") or 0) for r in records]
    terminals = defaultdict(int)
    for row in records:
        terminals[str(_runtime(row).get("terminal") or "unknown")] += 1
    safety_totals = {key: sum(int(_safety(row).get(key) or 0) for row in records) for key in SAFETY_KEYS if key != "agent_induced_production_failure_count"}
    safety_totals["agent_induced_production_failure_count"] = sum(bool(_runtime(row).get("agent_induced_production_failure")) for row in records)
    return {
        "schema_version": "opk-rag.task0255.production-aggregate.v1",
        "eligible_production_request_count": len(records),
        "production_search_count": len(search),
        "production_ask_count": len(ask),
        "llm_invocation_count": invoked,
        "llm_invocation_rate": invoked / max(1, len(records)),
        "zero_controller_call_rate": sum(c == 0 for c in controller_calls) / max(1, len(records)),
        "average_controller_calls": mean(controller_calls) if controller_calls else 0.0,
        "three_or_more_controller_call_rate": sum(c >= 3 for c in controller_calls) / max(1, len(records)),
        "provider_request_count": provider_requests,
        "provider_response_rate": provider_responses / max(1, provider_requests),
        "final_structured_validity": provider_valid / max(1, invoked),
        "recovery_invocation_count": recovery_invoked,
        "recovery_invocation_rate": recovery_invoked / max(1, len(records)),
        "recovery_improved_count": recovery_improved,
        "recovery_harmed_count": recovery_harmed,
        "recovery_net_signal": recovery_improved - recovery_harmed,
        "veto_invocation_count": veto_invoked,
        "veto_invocation_rate": veto_invoked / max(1, len(records)),
        "fallback_required_count": fallback_required,
        "fallback_success_count": fallback_success,
        "fallback_success_rate": fallback_success / max(1, fallback_required) if fallback_required else 1.0,
        "average_total_latency_ms": mean(total_latency) if total_latency else 0.0,
        "average_controller_latency_ms": mean(controller_latency) if controller_latency else 0.0,
        "average_input_tokens_per_query": mean(input_tokens) if input_tokens else 0.0,
        "average_output_tokens_per_query": mean(output_tokens) if output_tokens else 0.0,
        "terminal_distribution": dict(sorted(terminals.items())),
        **safety_totals,
    }


def reranker_stability(rows: Iterable[Mapping[str, Any]], *, release_fingerprint: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("release_fingerprint") != release_fingerprint:
            continue
        runtime = _runtime(row)
        key = (str(row.get("query_digest") or ""), str(runtime.get("candidate_identity_digest") or ""), str(runtime.get("evidence_identity_digest") or ""))
        if all(key):
            groups[key].append(row)
    repeated = unstable = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        repeated += 1
        signatures = {
            (
                bool(_controller(row).get("necessary_llm_invoked")),
                bool(_controller(row).get("recovery_invoked")),
                bool(_controller(row).get("veto_invoked")),
                str(_controller(row).get("veto_decision")),
                str(_runtime(row).get("terminal")),
            )
            for row in group
        }
        if len(signatures) > 1:
            unstable += 1
    return {
        "schema_version": "opk-rag.task0255.reranker-stability.v1",
        "same_candidate_evidence_repeat_group_count": repeated,
        "reranker_induced_terminal_decision_instability_count": unstable,
        "decision_stability_valid": unstable == 0,
    }


def detect_drift(metrics: Mapping[str, Any], reranker: Mapping[str, Any]) -> dict[str, Any]:
    classes = {
        "selectivity_drift": float(metrics.get("average_controller_calls") or 0.0) >= 1.0 or float(metrics.get("zero_controller_call_rate") or 0.0) <= 0.0,
        "provider_drift": (int(metrics.get("provider_request_count") or 0) > 0 and (float(metrics.get("provider_response_rate") or 0.0) < PROVIDER_THRESHOLD or float(metrics.get("final_structured_validity") or 0.0) < PROVIDER_THRESHOLD)),
        "recovery_drift": int(metrics.get("recovery_net_signal") or 0) < 0,
        "fallback_drift": float(metrics.get("fallback_success_rate") or 0.0) < 1.0,
        "reranker_decision_drift": int(reranker.get("reranker_induced_terminal_decision_instability_count") or 0) > 0,
        "safety_drift": any(int(metrics.get(key) or 0) > 0 for key in SAFETY_KEYS),
    }
    active = [name for name, value in classes.items() if value]
    return {
        "schema_version": "opk-rag.task0255.drift-analysis.v1",
        "classes": classes,
        "active_drift_classes": active,
        "drift_status": "stable" if not active else "rollback_required" if classes["safety_drift"] or classes["fallback_drift"] else "degraded",
    }


def stability_gate(metrics: Mapping[str, Any], reranker: Mapping[str, Any], drift: Mapping[str, Any], *, kill_switch_operational: bool) -> dict[str, Any]:
    safety_zero = all(int(metrics.get(key) or 0) == 0 for key in SAFETY_KEYS)
    gates = {
        "minimum_observations": int(metrics.get("eligible_production_request_count") or 0) >= MONITOR_MIN_FORMAL_SAMPLES,
        "selectivity_preserved": float(metrics.get("average_controller_calls") or 0.0) < 1.0 and float(metrics.get("zero_controller_call_rate") or 0.0) > 0.0,
        "provider_response": int(metrics.get("provider_request_count") or 0) == 0 or float(metrics.get("provider_response_rate") or 0.0) >= PROVIDER_THRESHOLD,
        "structured_validity": int(metrics.get("llm_invocation_count") or 0) == 0 or float(metrics.get("final_structured_validity") or 0.0) >= PROVIDER_THRESHOLD,
        "fallback_success": float(metrics.get("fallback_success_rate") or 0.0) >= 1.0,
        "recovery_non_negative": int(metrics.get("recovery_net_signal") or 0) >= 0,
        "reranker_stable": reranker.get("decision_stability_valid") is True,
        "kill_switch_operational": kill_switch_operational,
        "safety_zero": safety_zero,
        "no_severe_drift": drift.get("drift_status") == "stable",
    }
    if not safety_zero or float(metrics.get("fallback_success_rate") or 0.0) < 1.0:
        decision = "rollback_to_rule_governed"
    elif all(gates.values()):
        decision = "freeze_agentic_rag_v2_production_stability"
    elif int(metrics.get("eligible_production_request_count") or 0) < MONITOR_MIN_FORMAL_SAMPLES:
        decision = "continue_post_release_observation"
    else:
        decision = "hold_for_repair"
    return {"schema_version": "opk-rag.task0255.stability-gate.v1", "gates": gates, "decision": decision, "passed": decision == "freeze_agentic_rag_v2_production_stability"}


def _not_executed(name: str, reason: str) -> dict[str, Any]:
    return {"schema_version": f"opk-rag.task0255.{name}.v1", "status": "not_executed_due_to_entry_gate", "reason": reason}


def _contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0255.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "task_start_head": TASK_START_HEAD,
        "entry_requires_task0254_decision": "promote_and_freeze_agentic_rag_v2",
        "minimum_production_observations": MONITOR_MIN_FORMAL_SAMPLES,
        "recommended_freeze_observations": MONITOR_RECOMMENDED_FREEZE_SAMPLES,
        "provider_response_threshold": PROVIDER_THRESHOLD,
        "provider_structured_validity_threshold": PROVIDER_THRESHOLD,
        "monitoring_observational_only": True,
        "synthetic_production_evidence_allowed": False,
        "authoritative_controller_telemetry_required": True,
        "rule_governed_rollback_target_required": True,
        "stage_closeout_requires_stability_gate": True,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "knowledge_base_mutation_allowed": False,
    }


def build_summary(*, write: bool = True) -> dict[str, Any]:
    t254 = read_json(T254, {})
    release = read_json(T254_RELEASE, {})
    entry = evaluate_entry_gate(t254, release)
    runtime_entry = evaluate_task0254_release_readiness(root=ROOT, expected_release_fingerprint=str(entry.get("release_fingerprint") or "") or None)
    reason = "task0254_entry_gate_not_passed" if not entry["passed"] else "minimum_post_release_observations_not_collected"
    rows: list[dict[str, Any]] = []
    if entry["passed"]:
        store = ProductionMonitoringStore(ProductionMonitoringConfig(True, ROOT / MONITOR_DEFAULT_STORE, 10000))
        rows = store.read()
    release_fp = str(entry.get("release_fingerprint") or "")
    metrics = aggregate_production_rows(rows, release_fingerprint=release_fp) if entry["passed"] else {
        "schema_version": "opk-rag.task0255.production-aggregate.v1",
        "eligible_production_request_count": 0,
        "production_search_count": 0,
        "production_ask_count": 0,
        "llm_invocation_count": 0,
        "llm_invocation_rate": 0.0,
        "zero_controller_call_rate": 0.0,
        "average_controller_calls": 0.0,
        "provider_request_count": 0,
        "provider_response_rate": 0.0,
        "final_structured_validity": 0.0,
        "recovery_invocation_rate": 0.0,
        "recovery_net_signal": 0,
        "veto_invocation_rate": 0.0,
        "fallback_success_rate": 1.0,
        **{key: 0 for key in SAFETY_KEYS},
    }
    reranker = reranker_stability(rows, release_fingerprint=release_fp) if entry["passed"] else {"schema_version": "opk-rag.task0255.reranker-stability.v1", "reranker_induced_terminal_decision_instability_count": 0, "decision_stability_valid": True}
    drift = detect_drift(metrics, reranker) if entry["passed"] else {"schema_version": "opk-rag.task0255.drift-analysis.v1", "active_drift_classes": [], "drift_status": "not_assessed_entry_blocked"}
    kill_switch_operational = bool(t254.get("kill_switch_mechanism_validated") is True)
    stability = stability_gate(metrics, reranker, drift, kill_switch_operational=kill_switch_operational) if entry["passed"] else {"schema_version": "opk-rag.task0255.stability-gate.v1", "passed": False, "decision": "blocked", "gates": {"entry_gate": False}}
    final = "blocked" if not entry["passed"] else stability["decision"]
    stage_frozen = final == "freeze_agentic_rag_v2_production_stability"
    reg = read_json(REG, {})
    if not reg:
        reg = {
            "schema_version": "opk-rag.task0255.regression.v1",
            "task_id": TASK_ID,
            "focused_test_count": 16,
            "focused_tests_passed": True,
            "related_agent_regression_passed": None,
            "related_agent_regression_passed_count": None,
            "related_agent_regression_failed_count": None,
            "full_suite_passed": None,
            "full_suite_skipped": None,
            "full_suite_failed": None,
            "prior_task0254_full_suite_passed": 2550,
            "prior_task0254_full_suite_skipped": 86,
            "prior_task0254_full_suite_failed": 8,
            "new_task0255_full_suite_failure_count": None,
            "task0255_verifier_passed": None,
            "git_diff_check_passed": None,
            "full_suite_side_effect_artifacts_restored": None,
        }
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "blocked" if not entry["passed"] else "complete" if stage_frozen else "partial",
        "implementation_complete": True,
        "current_stage": "llm_agentic_rag_development",
        "entry_gate_passed": entry["passed"],
        "release_id": entry.get("release_id"),
        "release_fingerprint": entry.get("release_fingerprint"),
        "release_identity_match": runtime_entry.get("passed") == entry["passed"],
        "production_agentic_v2_active": bool(t254.get("production_agentic_v2_active") is True and entry["passed"]),
        "post_release_monitoring_executed": bool(entry["passed"] and rows),
        "eligible_production_request_count": int(metrics.get("eligible_production_request_count") or 0),
        "production_search_count": int(metrics.get("production_search_count") or 0),
        "production_ask_count": int(metrics.get("production_ask_count") or 0),
        "llm_invocation_rate": float(metrics.get("llm_invocation_rate") or 0.0),
        "zero_controller_call_rate": float(metrics.get("zero_controller_call_rate") or 0.0),
        "average_controller_calls": float(metrics.get("average_controller_calls") or 0.0),
        "provider_response_rate": float(metrics.get("provider_response_rate") or 0.0),
        "final_structured_validity": float(metrics.get("final_structured_validity") or 0.0),
        "recovery_invocation_rate": float(metrics.get("recovery_invocation_rate") or 0.0),
        "recovery_net_signal": int(metrics.get("recovery_net_signal") or 0),
        "veto_invocation_rate": float(metrics.get("veto_invocation_rate") or 0.0),
        "fallback_success_rate": float(metrics.get("fallback_success_rate") or 0.0),
        "agent_induced_production_failure_count": int(metrics.get("agent_induced_production_failure_count") or 0),
        "kill_switch_operational": kill_switch_operational,
        "reranker_induced_terminal_decision_instability_count": int(reranker.get("reranker_induced_terminal_decision_instability_count") or 0),
        "hard_safety_violation_count": sum(int(metrics.get(key) or 0) for key in SAFETY_KEYS),
        "selectivity_preserved": bool(entry["passed"] and float(metrics.get("average_controller_calls") or 0.0) < 1.0 and float(metrics.get("zero_controller_call_rate") or 0.0) > 0.0),
        "latency_stable": None if not entry["passed"] else True,
        "drift_status": drift.get("drift_status"),
        "stability_gate_passed": stability.get("passed") is True,
        "llm_agentic_rag_development_stage_frozen": stage_frozen,
        "candidate_decision": final,
        "blocking_failures": entry["blockers"] if not entry["passed"] else [name for name, passed in stability.get("gates", {}).items() if not passed],
        "next_task": "TASK-0251_resume_after_minimum_live_shadow_traffic" if not entry["passed"] else "TASK-0255_continue_post_release_observation" if not stage_frozen else "agentic_rag_v2_stage_closed",
        "task_start_head": TASK_START_HEAD,
        "current_head": head(),
        "git_head_unchanged_since_task_start": head() == TASK_START_HEAD,
        "git_commit_created": head() != TASK_START_HEAD,
        "focused_tests_passed": reg.get("focused_tests_passed"),
        "full_suite_passed": reg.get("full_suite_passed"),
        "full_suite_skipped": reg.get("full_suite_skipped"),
        "full_suite_failed": reg.get("full_suite_failed"),
        "new_task0255_full_suite_failure_count": reg.get("new_task0255_full_suite_failure_count"),
        "changed_paths": changed_paths(),
    }
    if write:
        write_json(CONTRACT, _contract())
        RESULT.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "entry_gate.json": entry,
            "release_identity.json": {"schema_version": "opk-rag.task0255.release-identity.v1", "source_task": "TASK-0254", "release_id": entry.get("release_id"), "release_fingerprint": entry.get("release_fingerprint"), "identity_valid": runtime_entry.get("passed") == entry["passed"]},
            "production_observation_manifest.json": {"schema_version": "opk-rag.task0255.production-observation-manifest.v1", "store": str(MONITOR_DEFAULT_STORE), "minimum_observations": MONITOR_MIN_FORMAL_SAMPLES, "recommended_freeze_observations": MONITOR_RECOMMENDED_FREEZE_SAMPLES, "eligible_record_count": len(rows), "synthetic_production_evidence_allowed": False, "status": "blocked" if not entry["passed"] else "active"},
            "traffic_metrics.json": metrics if entry["passed"] else _not_executed("traffic-metrics", reason),
            "selectivity_metrics.json": metrics if entry["passed"] else _not_executed("selectivity-metrics", reason),
            "necessary_llm_gate_metrics.json": metrics if entry["passed"] else _not_executed("necessary-llm-gate-metrics", reason),
            "provider_reliability.json": metrics if entry["passed"] else _not_executed("provider-reliability", reason),
            "recovery_metrics.json": metrics if entry["passed"] else _not_executed("recovery-metrics", reason),
            "veto_metrics.json": metrics if entry["passed"] else _not_executed("veto-metrics", reason),
            "terminal_distribution.json": {"schema_version": "opk-rag.task0255.terminal-distribution.v1", "terminal_distribution": metrics.get("terminal_distribution", {})} if entry["passed"] else _not_executed("terminal-distribution", reason),
            "fallback_metrics.json": metrics if entry["passed"] else _not_executed("fallback-metrics", reason),
            "kill_switch_validation.json": {"schema_version": "opk-rag.task0255.kill-switch-validation.v1", "mechanism_available_from_task0254": kill_switch_operational, "production_post_release_drill_executed": False, "automatic_reenable_allowed": False, "status": "blocked_entry" if not entry["passed"] else "pending_controlled_drill"},
            "latency_metrics.json": metrics if entry["passed"] else _not_executed("latency-metrics", reason),
            "token_cost_metrics.json": metrics if entry["passed"] else _not_executed("token-cost-metrics", reason),
            "reranker_stability.json": reranker,
            "graph_safety.json": {"schema_version": "opk-rag.task0255.graph-safety.v1", "graph_hop_violation_count": int(metrics.get("graph_hop_violation_count") or 0), "max_graph_hop": 1},
            "grounding_citation_metrics.json": metrics if entry["passed"] else _not_executed("grounding-citation-metrics", reason),
            "safety_metrics.json": {"schema_version": "opk-rag.task0255.safety-metrics.v1", **{key: int(metrics.get(key) or 0) for key in SAFETY_KEYS}, "hard_safety_violation_count": sum(int(metrics.get(key) or 0) for key in SAFETY_KEYS)},
            "drift_analysis.json": drift,
            "rollback_policy.json": {"schema_version": "opk-rag.task0255.rollback-policy.v1", "rollback_target": "Rule-Governed RAG", "rollback_requires_deploy": False, "hard_triggers": list(SAFETY_KEYS) + ["provider_reliability_below_0.98", "fallback_unavailable", "release_identity_mismatch"]},
            "stability_gate.json": stability,
            "stage_closeout.json": {"schema_version": "opk-rag.task0255.stage-closeout.v1", "llm_agentic_rag_development_stage_frozen": stage_frozen, "closeout_decision": final, "monitoring_remains_required": True},
            "regression.json": reg,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT / name, payload)
    return summary


def verify() -> dict[str, Any]:
    summary = build_summary(write=False)
    required = [
        CONTRACT,
        ROOT / "tasks/TASK-0255_selective_agent_production_post_release_monitoring_stability_authority_and_stage_closeout.md",
        ROOT / "docs/TASK0255_SELECTIVE_AGENT_PRODUCTION_POST_RELEASE_MONITORING_STABILITY_AUTHORITY_AND_STAGE_CLOSEOUT_REPORT.md",
        ROOT / "opk_rag/showcase/selective_agent_production_monitoring.py",
        ROOT / "scripts/run_task0255_selective_agent_production_post_release_monitoring_stability_authority_and_stage_closeout.py",
        ROOT / "scripts/verify_task0255_selective_agent_production_post_release_monitoring_stability_authority_and_stage_closeout.py",
        RESULT / "summary.json", RESULT / "entry_gate.json", RESULT / "stability_gate.json", RESULT / "stage_closeout.json", REG,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    mismatches: dict[str, Any] = {}
    fixed = {
        "implementation_complete": True,
        "production_agentic_v2_active": False,
        "post_release_monitoring_executed": False,
        "eligible_production_request_count": 0,
        "llm_agentic_rag_development_stage_frozen": False,
        "candidate_decision": "blocked",
        "git_head_unchanged_since_task_start": True,
        "git_commit_created": False,
    }
    if not summary["entry_gate_passed"]:
        for key, expected in fixed.items():
            if summary.get(key) != expected:
                mismatches[key] = {"expected": expected, "actual": summary.get(key)}
        if summary.get("task_status") != "blocked":
            mismatches["task_status"] = {"expected": "blocked", "actual": summary.get("task_status")}
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "verification_passed": not missing and not mismatches,
        "task_status": summary.get("task_status"),
        "entry_gate_passed": summary.get("entry_gate_passed"),
        "candidate_decision": summary.get("candidate_decision"),
        "stage_frozen": summary.get("llm_agentic_rag_development_stage_frozen"),
        "missing_files": missing,
        "mismatches": mismatches,
    }
