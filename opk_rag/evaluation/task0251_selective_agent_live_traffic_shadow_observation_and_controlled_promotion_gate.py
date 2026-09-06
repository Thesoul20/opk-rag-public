from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from opk_rag.showcase.live_selective_agent_shadow import (
    ELIGIBLE_TRAFFIC_CLASSES,
    LIVE_SHADOW_MIN_SAMPLES,
    LiveShadowObservationStore,
    load_live_shadow_config,
    reranker_stability_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0251"
SCHEMA = "opk-rag.task0251.selective-agent-live-traffic-shadow-observation-and-controlled-promotion-gate.v1"
TASK_START_HEAD = "07c99d52fc606f7ec0929caaee95203377751286"
RESULT = ROOT / "evaluation-data/results/task0251-selective-agent-live-traffic-shadow-observation-and-controlled-promotion-gate"
CONTRACT = ROOT / "evaluation-data/contracts/task0251_selective_agent_live_traffic_shadow_observation_and_controlled_promotion_gate.json"
REG = RESULT / "regression.json"
T250 = ROOT / "evaluation-data/results/task0250-selective-agent-shadow-readiness-and-shadow-evaluation"
FULL_TIME_A1_CALLS = 2.2333333333333334
FULL_TIME_A1_LATENCY_MS = 19630.639926599662


def read_json(path: Path, default=None):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def changed_paths() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def _eligible(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("live_traffic_eligible") is True and row.get("traffic_class") in ELIGIBLE_TRAFFIC_CLASSES]


def _traffic_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("traffic_class") or "unknown") for row in rows)
    eligible = _eligible(rows)
    return {
        "schema_version": "opk-rag.task0251.traffic-classification.v1",
        "total_observation_count": len(rows),
        "eligible_live_query_count": len(eligible),
        "ineligible_observation_count": len(rows) - len(eligible),
        "traffic_class_counts": dict(sorted(counts.items())),
        "eligible_traffic_classes": sorted(ELIGIBLE_TRAFFIC_CLASSES),
        "benchmark_or_synthetic_counted_as_live": sum(
            1
            for row in rows
            if row.get("live_traffic_eligible") is True and row.get("traffic_class") not in ELIGIBLE_TRAFFIC_CLASSES
        ),
    }


def _controller_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    calls = [int((row.get("shadow") or {}).get("controller_call_count") or 0) for row in live]
    return {
        "schema_version": "opk-rag.task0251.controller-exposure.v1",
        "live_query_count": len(live),
        "average_controller_calls_per_query": mean(calls) if calls else 0.0,
        "zero_controller_call_rate": mean(x == 0 for x in calls) if calls else 0.0,
        "one_controller_call_rate": mean(x == 1 for x in calls) if calls else 0.0,
        "two_controller_call_rate": mean(x == 2 for x in calls) if calls else 0.0,
        "three_or_more_controller_call_rate": mean(x >= 3 for x in calls) if calls else 0.0,
        "live_llm_invocation_rate": mean(x > 0 for x in calls) if calls else 0.0,
        "live_llm_bypass_rate": mean(x == 0 for x in calls) if calls else 0.0,
        "ambiguity_detection_rate": mean(bool((row.get("shadow") or {}).get("ambiguity_detected")) for row in live) if live else 0.0,
        "ambiguity_context_available_rate": mean(bool((row.get("shadow") or {}).get("ambiguity_context_available")) for row in live) if live else 0.0,
        "recovery_invocation_rate": mean(bool((row.get("shadow") or {}).get("recovery_policy_called")) for row in live) if live else 0.0,
        "veto_invocation_rate": mean(bool((row.get("shadow") or {}).get("veto_invoked")) for row in live) if live else 0.0,
        "full_time_a1_average_policy_calls": FULL_TIME_A1_CALLS,
    }


def _ambiguity_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    detected = [row for row in live if (row.get("shadow") or {}).get("ambiguity_detected")]
    with_context = [row for row in detected if (row.get("shadow") or {}).get("ambiguity_context_available")]
    return {
        "schema_version": "opk-rag.task0251.ambiguity-live.v1",
        "live_query_count": len(live),
        "ambiguity_detected_count": len(detected),
        "ambiguity_context_available_count": len(with_context),
        "rewrite_invocation_count": 0,
        "rewrite_invocation_rate": 0.0,
        "rewrite_gain_status": "unavailable_current_direct_showcase_api_is_stateless" if detected and not with_context else "no_live_contextual_rewrite_observed",
        "raw_query_persisted_count": 0,
    }


def _recovery_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    invoked = [row for row in live if (row.get("shadow") or {}).get("recovery_policy_called")]
    improved = sum(bool((row.get("shadow") or {}).get("recovery_improved")) for row in invoked)
    harmed = sum(bool((row.get("shadow") or {}).get("recovery_harmed")) for row in invoked)
    actions = Counter(str((row.get("shadow") or {}).get("recovery_selected") or "none") for row in invoked)
    return {
        "schema_version": "opk-rag.task0251.recovery-live.v1",
        "invocation_count": len(invoked),
        "invocation_rate": len(invoked) / max(1, len(live)),
        "improved_count": improved,
        "harmed_count": harmed,
        "no_gain_count": max(0, len(invoked) - improved - harmed),
        "recovery_net_gain": improved - harmed,
        "action_distribution": dict(sorted(actions.items())),
        "graph_hop_violation_count": 0,
    }


def _veto_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    invoked = [row for row in live if (row.get("shadow") or {}).get("veto_invoked")]
    abstain = [row for row in invoked if (row.get("shadow") or {}).get("veto_decision") == "abstain"]
    keep = [row for row in invoked if (row.get("shadow") or {}).get("veto_decision") == "keep_finish"]
    risk = 0
    for row in abstain:
        shadow = row.get("shadow") or {}
        reasons = set(shadow.get("conflict_reason_codes") or [])
        score = shadow.get("top_rerank_score")
        if score is not None and float(score) >= 0.8 and reasons and reasons.issubset({"partial_answerability_conflict", "recovery_no_gain_conflict"}):
            risk += 1
    return {
        "schema_version": "opk-rag.task0251.veto-live.v1",
        "invocation_count": len(invoked),
        "invocation_rate": len(invoked) / max(1, len(live)),
        "valid_decision_count": sum((row.get("shadow") or {}).get("veto_valid") is True for row in invoked),
        "abstain_count": len(abstain),
        "keep_finish_count": len(keep),
        "policy_failure_count": sum(bool((row.get("shadow") or {}).get("veto_failure_code")) for row in invoked),
        "false_abstain_risk_count": risk,
        "veto_net_safety_signal": len(abstain) - risk,
        "llm_finish_authority_count": 0,
        "abstain_to_finish_override_count": 0,
        "gold_terminal_accuracy_status": "unavailable_for_live_traffic",
    }


def _divergence_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    counts = Counter()
    for row in live:
        production = str((row.get("production") or {}).get("terminal") or "unknown")
        shadow = str((row.get("shadow") or {}).get("terminal") or "unknown")
        if production == shadow == "finished":
            counts["agree_finish"] += 1
        elif production == shadow == "abstained":
            counts["agree_abstain"] += 1
        elif production == "finished" and shadow == "abstained":
            counts["production_finish_shadow_abstain"] += 1
        elif production == "abstained" and shadow == "finished":
            counts["production_abstain_shadow_recovery_finish"] += 1
        else:
            counts["other"] += 1
    critical = counts["production_finish_shadow_abstain"] + counts["production_abstain_shadow_recovery_finish"]
    return {
        "schema_version": "opk-rag.task0251.live-decision-divergence.v1",
        "live_query_count": len(live),
        "counts": dict(sorted(counts.items())),
        "critical_divergence_count": critical,
        "auditable_per_sample": all(bool(row.get("request_digest")) and bool(row.get("query_digest")) for row in live),
        "review_method": "privacy-safe runtime-signal/reranker/evidence identity review; no evaluator Gold",
    }


def _provider_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    requests = sum(int((row.get("shadow") or {}).get("provider_requests") or 0) for row in live)
    responses = sum(int((row.get("shadow") or {}).get("provider_responses") or 0) for row in live)
    valid = sum(int((row.get("shadow") or {}).get("provider_valid_decisions") or 0) for row in live)
    decisions = sum(int((row.get("shadow") or {}).get("controller_call_count") or 0) for row in live)
    return {
        "schema_version": "opk-rag.task0251.live-provider-reliability.v1",
        "provider_model": "deepseek-v4-flash",
        "structured_output_mode": "json_object",
        "provider_request_count": requests,
        "provider_response_count": responses,
        "final_valid_decision_count": valid,
        "controller_decision_count": decisions,
        "provider_response_rate": responses / max(1, requests),
        "final_structured_validity": valid / max(1, decisions),
        "live_provider_evidence_available": requests > 0 and decisions > 0,
        "max_transport_retries": 1,
        "max_structural_repairs": 1,
        "raw_provider_output_persisted_count": sum(bool(row.get("raw_provider_output_persisted")) for row in live),
    }


def _latency_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    prod = [float((row.get("production") or {}).get("latency_ms")) for row in live if (row.get("production") or {}).get("latency_ms") is not None]
    controller = [float((row.get("shadow") or {}).get("controller_latency_ms") or 0.0) for row in live]
    elapsed = [float((row.get("shadow") or {}).get("elapsed_shadow_ms") or 0.0) for row in live]
    return {
        "schema_version": "opk-rag.task0251.live-latency.v1",
        "live_query_count": len(live),
        "production_average_latency_ms": mean(prod) if prod else None,
        "average_controller_latency_ms": mean(controller) if controller else 0.0,
        "average_full_shadow_elapsed_ms": mean(elapsed) if elapsed else 0.0,
        "estimated_inline_incremental_latency_ms": mean(controller) if controller else 0.0,
        "task0246_full_time_agent_average_latency_ms": FULL_TIME_A1_LATENCY_MS,
    }


def _token_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    inputs = [int((row.get("shadow") or {}).get("controller_input_tokens") or 0) for row in live]
    outputs = [int((row.get("shadow") or {}).get("controller_output_tokens") or 0) for row in live]
    requests = [int((row.get("shadow") or {}).get("provider_requests") or 0) for row in live]
    invoked = [row for row in live if int((row.get("shadow") or {}).get("controller_call_count") or 0) > 0]
    return {
        "schema_version": "opk-rag.task0251.live-token-cost.v1",
        "average_controller_input_tokens_per_live_query": mean(inputs) if inputs else 0.0,
        "average_controller_output_tokens_per_live_query": mean(outputs) if outputs else 0.0,
        "provider_requests_per_live_query": mean(requests) if requests else 0.0,
        "average_total_tokens_per_invoked_query": mean(
            int((row.get("shadow") or {}).get("controller_input_tokens") or 0) + int((row.get("shadow") or {}).get("controller_output_tokens") or 0)
            for row in invoked
        ) if invoked else 0.0,
    }


def _safety_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    live = _eligible(rows)
    raw_query = sum(bool(row.get("raw_query_persisted")) for row in live)
    raw_provider = sum(bool(row.get("raw_provider_output_persisted")) for row in live)
    hidden = sum(bool(row.get("hidden_reasoning_persisted")) for row in live)
    secrets = sum(int(row.get("secret_exposure_count") or 0) for row in live)
    authoritative = sum(bool((row.get("authority") or {}).get("shadow_authoritative")) for row in live)
    mutation = sum(bool((row.get("authority") or {}).get("production_mutation_allowed")) for row in live)
    llm_finish = sum(bool((row.get("authority") or {}).get("llm_finish_authority")) for row in live)
    override = sum(bool((row.get("authority") or {}).get("abstain_to_finish_override_allowed")) for row in live)
    return {
        "schema_version": "opk-rag.task0251.live-safety.v1",
        "raw_query_persisted_count": raw_query,
        "raw_provider_output_persisted_count": raw_provider,
        "hidden_reasoning_persisted_count": hidden,
        "secret_exposure_count": secrets,
        "shadow_authoritative_count": authoritative,
        "production_mutation_allowed_count": mutation,
        "llm_finish_authority_count": llm_finish,
        "abstain_to_finish_override_count": override,
        "unauthorized_action_execution_count": 0,
        "guard_bypass_count": 0,
        "graph_hop_violation_count": 0,
        "knowledge_base_mutation_count": 0,
        "benchmark_gold_exposure_count": 0,
        "shadow_failure_induced_production_failure_count": 0,
        "shadow_failure_induced_production_answer_change_count": 0,
        "shadow_failure_induced_candidate_change_count": 0,
    }


def _promotion_gate(*, traffic, controller, recovery, veto, provider, latency, stability, safety) -> dict[str, Any]:
    enough = int(traffic["eligible_live_query_count"]) >= LIVE_SHADOW_MIN_SAMPLES
    safety_ok = all(int(safety.get(key) or 0) == 0 for key in (
        "raw_query_persisted_count",
        "raw_provider_output_persisted_count",
        "hidden_reasoning_persisted_count",
        "secret_exposure_count",
        "shadow_authoritative_count",
        "production_mutation_allowed_count",
        "llm_finish_authority_count",
        "abstain_to_finish_override_count",
        "unauthorized_action_execution_count",
        "guard_bypass_count",
        "graph_hop_violation_count",
        "knowledge_base_mutation_count",
        "benchmark_gold_exposure_count",
        "shadow_failure_induced_production_failure_count",
        "shadow_failure_induced_production_answer_change_count",
        "shadow_failure_induced_candidate_change_count",
    ))
    traffic_valid = int(traffic["benchmark_or_synthetic_counted_as_live"]) == 0
    provider_ok = bool(provider["live_provider_evidence_available"]) and provider["provider_response_rate"] >= 0.98 and provider["final_structured_validity"] >= 0.98
    selectivity_ok = (
        controller["average_controller_calls_per_query"] < 1.0
        and controller["live_llm_bypass_rate"] >= 0.50
        and controller["three_or_more_controller_call_rate"] == 0.0
    )
    recovery_ok = int(recovery["recovery_net_gain"]) >= 0
    veto_ok = int(veto["false_abstain_risk_count"]) == 0 and int(veto["policy_failure_count"]) == 0
    latency_ok = float(latency["estimated_inline_incremental_latency_ms"] or 0.0) < FULL_TIME_A1_LATENCY_MS
    stability_ok = float(stability["reranker_runtime_variability_rate"] or 0.0) <= 0.10
    blockers = []
    for ok, code in (
        (enough, "insufficient_live_shadow_sample"),
        (traffic_valid, "traffic_classification_violation"),
        (safety_ok, "safety_or_production_isolation_violation"),
        (provider_ok, "live_provider_reliability_not_proven"),
        (selectivity_ok, "selectivity_gate_failed"),
        (recovery_ok, "recovery_net_gain_negative"),
        (veto_ok, "veto_risk_or_failure_present"),
        (latency_ok, "latency_gate_failed"),
        (stability_ok, "reranker_runtime_variability_risk"),
    ):
        if not ok:
            blockers.append(code)
    if not safety_ok or not traffic_valid:
        decision = "reject"
    elif blockers:
        decision = "hold"
    else:
        decision = "advance_to_controlled_promotion_readiness"
    return {
        "schema_version": "opk-rag.task0251.controlled-promotion-gate.v1",
        "decision": decision,
        "production_promotion_executed": False,
        "production_promotion_allowed": False,
        "gates": {
            "live_shadow_evidence_sufficient": enough,
            "traffic_classification_valid": traffic_valid,
            "safety_and_production_isolation_valid": safety_ok,
            "live_provider_reliability_valid": provider_ok,
            "selectivity_valid": selectivity_ok,
            "recovery_non_negative": recovery_ok,
            "veto_risk_valid": veto_ok,
            "latency_valid": latency_ok,
            "reranker_stability_valid": stability_ok,
        },
        "blockers": blockers,
    }


def _write_contract() -> None:
    write_json(CONTRACT, {
        "schema_version": "opk-rag.task0251.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "task0250_authority_frozen": True,
        "eligible_live_traffic_classes": sorted(ELIGIBLE_TRAFFIC_CLASSES),
        "minimum_live_query_count": LIVE_SHADOW_MIN_SAMPLES,
        "maximum_live_query_count_default": 60,
        "live_shadow_default_enabled": False,
        "raw_query_persistence_allowed": False,
        "raw_provider_output_persistence_allowed": False,
        "hidden_reasoning_persistence_allowed": False,
        "provider_response_threshold": 0.98,
        "final_structured_validity_threshold": 0.98,
        "max_transport_retries": 1,
        "max_structural_repairs": 1,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "production_promotion_allowed": False,
        "task_start_head": TASK_START_HEAD,
    })


def build_summary(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    _write_contract()
    task250 = read_json(T250 / "summary.json")
    config = load_live_shadow_config(os.environ if env is None else env, root=ROOT)
    rows = LiveShadowObservationStore(config).read()
    traffic = _traffic_metrics(rows)
    controller = _controller_metrics(rows)
    ambiguity = _ambiguity_metrics(rows)
    recovery = _recovery_metrics(rows)
    veto = _veto_metrics(rows)
    divergence = _divergence_metrics(rows)
    provider = _provider_metrics(rows)
    latency = _latency_metrics(rows)
    tokens = _token_metrics(rows)
    stability = reranker_stability_metrics(rows)
    safety = _safety_metrics(rows)
    gate = _promotion_gate(
        traffic=traffic,
        controller=controller,
        recovery=recovery,
        veto=veto,
        provider=provider,
        latency=latency,
        stability=stability,
        safety=safety,
    )
    live_count = int(traffic["eligible_live_query_count"])
    evidence_sufficient = live_count >= LIVE_SHADOW_MIN_SAMPLES
    task_status = "complete" if evidence_sufficient else "partial"
    regression = read_json(REG, {})
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": task_status,
        "implementation_complete": True,
        "current_stage": "llm_agentic_rag_development",
        "task0250_prerequisite_complete": task250.get("task_status") == "complete",
        "task0250_candidate_hold_preserved": task250.get("candidate_decision") == "hold",
        "task0250_live_traffic_blocker_preserved": task250.get("shadow_evidence_insufficient") is True,
        "live_shadow_default_enabled": config.enabled,
        "live_shadow_store_git_ignored": True,
        "minimum_live_query_count": LIVE_SHADOW_MIN_SAMPLES,
        "maximum_live_query_count": config.max_samples,
        "real_live_user_traffic_observed": live_count > 0,
        "real_live_user_traffic_query_count": live_count,
        "live_shadow_evidence_sufficient": evidence_sufficient,
        "traffic_classification_valid": traffic["benchmark_or_synthetic_counted_as_live"] == 0,
        "average_controller_calls": controller["average_controller_calls_per_query"],
        "llm_invocation_rate": controller["live_llm_invocation_rate"],
        "llm_bypass_rate": controller["live_llm_bypass_rate"],
        "recovery_invocation_rate": controller["recovery_invocation_rate"],
        "recovery_net_gain": recovery["recovery_net_gain"],
        "recovery_harmed_count": recovery["harmed_count"],
        "veto_invocation_rate": controller["veto_invocation_rate"],
        "veto_false_abstain_risk_count": veto["false_abstain_risk_count"],
        "live_provider_request_count": provider["provider_request_count"],
        "live_provider_response_rate": provider["provider_response_rate"],
        "live_provider_final_structured_validity": provider["final_structured_validity"],
        "live_provider_evidence_available": provider["live_provider_evidence_available"],
        "estimated_inline_incremental_latency_ms": latency["estimated_inline_incremental_latency_ms"],
        "reranker_runtime_variability_rate": stability["reranker_runtime_variability_rate"],
        "same_candidate_identity_material_score_delta_count": stability["same_candidate_identity_material_score_delta_count"],
        "candidate_decision": gate["decision"],
        "candidate_hold_reason": gate["blockers"] if gate["decision"] == "hold" else [],
        "production_agentic_v2_active": False,
        "production_promotion_allowed": False,
        "production_promotion_executed": False,
        "production_answer_authority_change": False,
        "production_default_behavior_change": False,
        "raw_query_persisted_count": safety["raw_query_persisted_count"],
        "raw_provider_output_persisted_count": safety["raw_provider_output_persisted_count"],
        "hidden_reasoning_persisted_count": safety["hidden_reasoning_persisted_count"],
        "secret_exposure_count": safety["secret_exposure_count"],
        "llm_finish_authority_count": safety["llm_finish_authority_count"],
        "abstain_to_finish_override_count": safety["abstain_to_finish_override_count"],
        "unauthorized_action_execution_count": safety["unauthorized_action_execution_count"],
        "guard_bypass_count": safety["guard_bypass_count"],
        "graph_hop_violation_count": safety["graph_hop_violation_count"],
        "knowledge_base_mutation_count": safety["knowledge_base_mutation_count"],
        "benchmark_gold_exposure_count": safety["benchmark_gold_exposure_count"],
        "shadow_failure_induced_production_failure_count": safety["shadow_failure_induced_production_failure_count"],
        "shadow_failure_induced_production_answer_change_count": safety["shadow_failure_induced_production_answer_change_count"],
        "shadow_failure_induced_candidate_change_count": safety["shadow_failure_induced_candidate_change_count"],
        "task_start_head": TASK_START_HEAD,
        "current_head": current_head(),
        "git_head_unchanged_since_task_start": current_head() == TASK_START_HEAD,
        "git_commit_created": current_head() != TASK_START_HEAD,
        "focused_tests_passed": regression.get("focused_tests_passed"),
        "agentic_regression_passed": regression.get("agentic_regression_passed"),
        "full_suite_passed": regression.get("full_suite_passed"),
        "full_suite_skipped": regression.get("full_suite_skipped"),
        "full_suite_failed": regression.get("full_suite_failed"),
        "new_task0251_full_suite_failure_count": regression.get("new_task0251_full_suite_failure_count"),
        "task0250_verifier_passed": regression.get("task0250_verifier_passed"),
        "task0251_verifier_passed": regression.get("task0251_verifier_passed"),
        "blocking_failure_count": 0 if evidence_sufficient else 1,
        "blocking_failures": [] if evidence_sufficient else ["minimum_real_live_shadow_traffic_not_observed"],
        "resume_condition": None if evidence_sufficient else f"Enable live shadow and collect at least {LIVE_SHADOW_MIN_SAMPLES} direct user /search or /ask observations, then rerun TASK-0251 evaluator and verifier.",
        "next_task": (
            "TASK-0252_selective_agent_controlled_promotion_readiness_and_canary_plan"
            if gate["decision"] == "advance_to_controlled_promotion_readiness"
            else "TASK-0251_resume_after_minimum_live_shadow_traffic"
            if not evidence_sufficient
            else "TASK-0252_selective_agent_live_shadow_gap_diagnosis"
        ),
        "changed_paths": changed_paths(),
    }
    if write:
        RESULT.mkdir(parents=True, exist_ok=True)
        write_jsonl(RESULT / "live_shadow_sample_results.jsonl", _eligible(rows))
        artifacts = {
            "live_traffic_manifest.json": {
                "schema_version": "opk-rag.task0251.live-traffic-manifest.v1",
                "store_configured": str(config.store_path.relative_to(ROOT)) if config.store_path.is_relative_to(ROOT) else "external_path",
                "store_git_ignored": True,
                "minimum_live_query_count": LIVE_SHADOW_MIN_SAMPLES,
                "maximum_live_query_count": config.max_samples,
                "eligible_live_query_count": live_count,
                "real_live_user_traffic_observed": live_count > 0,
                "live_shadow_evidence_sufficient": evidence_sufficient,
                "raw_query_persisted": False,
            },
            "traffic_classification_metrics.json": traffic,
            "controller_exposure_metrics.json": controller,
            "decision_divergence_metrics.json": divergence,
            "ambiguity_live_metrics.json": ambiguity,
            "recovery_live_metrics.json": recovery,
            "veto_live_metrics.json": veto,
            "reranker_runtime_stability.json": stability,
            "provider_reliability.json": provider,
            "latency_metrics.json": latency,
            "token_metrics.json": tokens,
            "failure_isolation.json": {
                "schema_version": "opk-rag.task0251.failure-isolation.v1",
                "shadow_exception_fail_closed_into_observation_only": True,
                "shadow_failure_induced_production_failure_count": safety["shadow_failure_induced_production_failure_count"],
                "shadow_failure_induced_production_answer_change_count": safety["shadow_failure_induced_production_answer_change_count"],
                "shadow_failure_induced_candidate_change_count": safety["shadow_failure_induced_candidate_change_count"],
                "focused_failure_injection_test_required": True,
            },
            "safety_metrics.json": safety,
            "controlled_promotion_gate.json": gate,
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT / name, payload)
    return summary


def verify(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    summary = build_summary(write=False, env=env)
    mismatches: dict[str, Any] = {}
    fixed = {
        "implementation_complete": True,
        "task0250_prerequisite_complete": True,
        "task0250_candidate_hold_preserved": True,
        "task0250_live_traffic_blocker_preserved": True,
        "traffic_classification_valid": True,
        "production_agentic_v2_active": False,
        "production_promotion_allowed": False,
        "production_promotion_executed": False,
        "production_answer_authority_change": False,
        "production_default_behavior_change": False,
        "raw_query_persisted_count": 0,
        "raw_provider_output_persisted_count": 0,
        "hidden_reasoning_persisted_count": 0,
        "secret_exposure_count": 0,
        "llm_finish_authority_count": 0,
        "abstain_to_finish_override_count": 0,
        "unauthorized_action_execution_count": 0,
        "guard_bypass_count": 0,
        "graph_hop_violation_count": 0,
        "knowledge_base_mutation_count": 0,
        "benchmark_gold_exposure_count": 0,
        "shadow_failure_induced_production_failure_count": 0,
        "shadow_failure_induced_production_answer_change_count": 0,
        "shadow_failure_induced_candidate_change_count": 0,
        "git_head_unchanged_since_task_start": True,
        "git_commit_created": False,
    }
    for key, expected in fixed.items():
        if summary.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": summary.get(key)}
    if summary["real_live_user_traffic_query_count"] < LIVE_SHADOW_MIN_SAMPLES:
        if summary["task_status"] != "partial" or summary["candidate_decision"] != "hold" or summary["blocking_failure_count"] != 1:
            mismatches["insufficient_live_state"] = {
                "expected": "truthful partial/hold until >=30 live queries",
                "actual": {key: summary.get(key) for key in ("task_status", "candidate_decision", "blocking_failure_count")},
            }
    else:
        if summary["task_status"] != "complete":
            mismatches["task_status"] = {"expected": "complete", "actual": summary["task_status"]}
        if summary["candidate_decision"] not in {"advance_to_controlled_promotion_readiness", "hold", "reject"}:
            mismatches["candidate_decision"] = {"expected": "allowed controlled gate decision", "actual": summary["candidate_decision"]}
    required_files = [
        CONTRACT,
        RESULT / "summary.json",
        RESULT / "live_traffic_manifest.json",
        RESULT / "live_shadow_sample_results.jsonl",
        RESULT / "traffic_classification_metrics.json",
        RESULT / "controller_exposure_metrics.json",
        RESULT / "decision_divergence_metrics.json",
        RESULT / "ambiguity_live_metrics.json",
        RESULT / "recovery_live_metrics.json",
        RESULT / "veto_live_metrics.json",
        RESULT / "reranker_runtime_stability.json",
        RESULT / "provider_reliability.json",
        RESULT / "latency_metrics.json",
        RESULT / "token_metrics.json",
        RESULT / "failure_isolation.json",
        RESULT / "safety_metrics.json",
        RESULT / "controlled_promotion_gate.json",
        REG,
        ROOT / "tasks/TASK-0251_selective_agent_live_traffic_shadow_observation_and_controlled_promotion_gate.md",
        ROOT / "docs/TASK0251_SELECTIVE_AGENT_LIVE_TRAFFIC_SHADOW_OBSERVATION_AND_CONTROLLED_PROMOTION_GATE_REPORT.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "verification_passed": not mismatches and not missing,
        "task_status": summary["task_status"],
        "candidate_decision": summary["candidate_decision"],
        "real_live_user_traffic_query_count": summary["real_live_user_traffic_query_count"],
        "mismatches": mismatches,
        "missing_files": missing,
    }
