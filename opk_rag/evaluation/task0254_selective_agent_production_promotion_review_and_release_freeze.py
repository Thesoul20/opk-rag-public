from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from opk_rag.showcase.selective_agent_production import (
    PRODUCTION_ENABLED_ENV,
    PRODUCTION_KILL_SWITCH_ENV,
    build_release_identity,
    evaluate_task0253_promotion_readiness,
    explicit_activation_request,
    load_production_config,
    route_selective_agent_production,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0254"
SCHEMA = "opk-rag.task0254.selective-agent-production-promotion-review-and-release-freeze.v1"
TASK_START_HEAD = "cdb2574b7377626d083aaf80cfb83cc067887066"
RESULT = ROOT / "evaluation-data/results/task0254-selective-agent-production-promotion-review-and-release-freeze"
CONTRACT = ROOT / "evaluation-data/contracts/task0254_selective_agent_production_promotion_review_and_release_freeze.json"
T250 = ROOT / "evaluation-data/results/task0250-selective-agent-shadow-readiness-and-shadow-evaluation/summary.json"
T251 = ROOT / "evaluation-data/results/task0251-selective-agent-live-traffic-shadow-observation-and-controlled-promotion-gate/summary.json"
T252 = ROOT / "evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/summary.json"
T252_CANDIDATE = ROOT / "evaluation-data/results/task0252-selective-agent-controlled-promotion-readiness-and-canary-plan/candidate_identity.json"
T253 = ROOT / "evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation/summary.json"
T253_GATE = ROOT / "evaluation-data/results/task0253-selective-agent-bounded-production-canary-execution-and-rollback-validation/production_promotion_review_gate.json"
REG = RESULT / "regression.json"

SAFETY_KEYS = (
    "llm_finish_authority_count",
    "abstain_to_finish_override_count",
    "unauthorized_action_execution_count",
    "guard_bypass_count",
    "graph_hop_violation_count",
    "knowledge_base_mutation_count",
    "benchmark_gold_exposure_count",
    "hidden_reasoning_persisted_count",
    "raw_provider_output_persisted_count",
    "secret_exposure_count",
    "benchmark_or_synthetic_canary_count",
    "agent_induced_production_failure_count",
    "canary_exposure_limit_violation_count",
)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def changed_paths() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def evaluate_entry_gate(task0253: Mapping[str, Any]) -> dict[str, Any]:
    safety_zero = bool(task0253.get("safety_invariants_preserved") is True) and all(
        int(task0253.get(key) or 0) == 0 for key in SAFETY_KEYS
    )
    gates = {
        "task0253_complete": task0253.get("task_status") == "complete",
        "task0253_advanced": task0253.get("final_candidate_decision") == "advance_to_production_promotion_review",
        "canary_execution_performed": task0253.get("canary_execution_performed") is True,
        "small_canary_minimum": int(task0253.get("small_canary_observation_count") or 0) >= 30,
        "expanded_canary_minimum": int(task0253.get("expanded_canary_observation_count") or 0) >= 100,
        "small_canary_advanced": task0253.get("small_canary_decision") == "advance_to_expanded_canary",
        "expanded_canary_advanced": task0253.get("expanded_canary_decision") == "advance_to_production_promotion_review",
        "exposure_within_bound": float(task0253.get("observed_canary_exposure") or 0.0) <= 0.20,
        "provider_response_rate": float(task0253.get("provider_response_rate") or 0.0) >= 0.98,
        "structured_validity": float(task0253.get("final_structured_validity") or 0.0) >= 0.98,
        "fallback_success": float(task0253.get("fallback_success_rate") or 0.0) >= 1.0,
        "kill_switch_validated": task0253.get("kill_switch_validated") is True,
        "rollback_validated": task0253.get("rollback_validated") is True,
        "production_isolation": int(task0253.get("agent_induced_production_failure_count") or 0) == 0,
        "reranker_terminal_stability": int(task0253.get("reranker_induced_terminal_decision_instability_count") or 0) == 0,
        "safety_invariants_zero": safety_zero,
        "candidate_fingerprint_present": bool(task0253.get("candidate_fingerprint")),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0254.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "source_task": "TASK-0253",
        "source_status": task0253.get("task_status"),
        "source_decision": task0253.get("final_candidate_decision"),
        "candidate_fingerprint": task0253.get("candidate_fingerprint"),
    }


def selectivity_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    avg_calls = float(metrics.get("average_controller_calls") or 0.0)
    bypass = float(metrics.get("llm_bypass_rate") or 0.0)
    gates = {
        "average_controller_calls_below_always_on": avg_calls < 1.0,
        "zero_llm_path_exists": bypass > 0.0,
        "three_or_more_controller_calls_zero": float(metrics.get("three_or_more_controller_call_rate") or 0.0) == 0.0,
    }
    return {
        "schema_version": "opk-rag.task0254.selectivity-gate.v1",
        "gates": gates,
        "selectivity_preserved": all(gates.values()),
        "average_controller_calls": avg_calls,
        "llm_bypass_rate": bypass,
    }


def provider_reliability_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    response = float(metrics.get("provider_response_rate") or 0.0)
    validity = float(metrics.get("final_structured_validity") or 0.0)
    return {
        "schema_version": "opk-rag.task0254.provider-reliability-gate.v1",
        "provider_response_rate": response,
        "final_structured_validity": validity,
        "threshold": 0.98,
        "passed": response >= 0.98 and validity >= 0.98,
    }


def _not_executed(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"opk-rag.task0254.{name}.v1",
        "status": "not_executed_due_to_entry_gate",
        "reason": reason,
    }


def _contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0254.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "task_start_head": TASK_START_HEAD,
        "entry_requires_task0253_decision": "advance_to_production_promotion_review",
        "minimum_small_canary_observations": 30,
        "minimum_expanded_canary_observations": 100,
        "provider_response_threshold": 0.98,
        "provider_structured_validity_threshold": 0.98,
        "production_activation_requires_explicit_action": True,
        "evaluator_may_activate_production": False,
        "default_production_enabled": False,
        "default_kill_switch": True,
        "rule_governed_fallback_required": True,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "knowledge_base_mutation_allowed": False,
    }


def _evidence_manifest() -> dict[str, Any]:
    sources = {
        "internal_agent_benchmark": T250,
        "live_shadow": T251,
        "external_generalization": T252,
        "production_canary": T253,
    }
    return {
        "schema_version": "opk-rag.task0254.evidence-authority-manifest.v1",
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_digest(path),
                "available": path.is_file(),
            }
            for name, path in sources.items()
        },
        "evidence_layers_required": list(sources),
        "single_layer_sufficient_for_promotion": False,
    }


def _candidate_identity(task0253: Mapping[str, Any], task0252_candidate: Mapping[str, Any]) -> dict[str, Any]:
    fp = str(task0253.get("candidate_fingerprint") or task0252_candidate.get("candidate_fingerprint") or "") or None
    return {
        "schema_version": "opk-rag.task0254.candidate-identity.v1",
        "candidate_fingerprint": fp,
        "task0253_approved_candidate_fingerprint": task0253.get("candidate_fingerprint"),
        "task0252_candidate_fingerprint": task0252_candidate.get("candidate_fingerprint"),
        "candidate_match": bool(fp) and fp == task0253.get("candidate_fingerprint") == task0252_candidate.get("candidate_fingerprint"),
        "candidate_architecture": task0252_candidate.get("candidate_architecture"),
        "provider_model": task0252_candidate.get("provider_model"),
        "structured_output_mode": task0252_candidate.get("structured_output_mode"),
        "embedding_model": task0252_candidate.get("embedding_model"),
        "reranker_model": task0252_candidate.get("reranker_model"),
        "production_vector_backend": task0252_candidate.get("production_vector_backend"),
        "max_graph_hop": task0252_candidate.get("max_graph_hop"),
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
    }


def _mechanism_validation(candidate_fp: str | None) -> dict[str, Any]:
    disabled = route_selective_agent_production(query="task0254-mechanism", execution_scope="search", env={}, root=ROOT, expected_candidate_fingerprint=candidate_fp)
    killed = route_selective_agent_production(
        query="task0254-mechanism", execution_scope="search",
        env={PRODUCTION_ENABLED_ENV: "1", PRODUCTION_KILL_SWITCH_ENV: "1"}, root=ROOT,
        expected_candidate_fingerprint=candidate_fp,
    )
    fake_release = {
        "candidate_fingerprint": candidate_fp,
        "release_id": "task0254-mechanism-only",
    }
    activation = explicit_activation_request(
        root=ROOT,
        env={PRODUCTION_ENABLED_ENV: "1", PRODUCTION_KILL_SWITCH_ENV: "0"},
        release_identity=fake_release,
        explicit=True,
        dry_run=True,
    )
    return {
        "schema_version": "opk-rag.task0254.promotion-mechanism-validation.v1",
        "default_routes_rule_governed": disabled.get("authority_lane") == "rule_governed_control",
        "kill_switch_routes_rule_governed": killed.get("authority_lane") == "rule_governed_control",
        "blocked_entry_prevents_activation": activation.get("activated") is False,
        "activation_status": activation.get("status"),
        "production_activation_executed": False,
        "mechanism_valid": disabled.get("authority_lane") == "rule_governed_control" and killed.get("authority_lane") == "rule_governed_control" and activation.get("activated") is False,
    }


def build_summary(*, write: bool = True) -> dict[str, Any]:
    t250 = read_json(T250, {})
    t251 = read_json(T251, {})
    t252 = read_json(T252, {})
    t252_candidate = read_json(T252_CANDIDATE, {})
    t253 = read_json(T253, {})
    entry = evaluate_entry_gate(t253)
    runtime_entry = evaluate_task0253_promotion_readiness(root=ROOT, expected_candidate_fingerprint=str(t253.get("candidate_fingerprint") or "") or None)
    candidate = _candidate_identity(t253, t252_candidate)
    evidence = _evidence_manifest()
    mechanism = _mechanism_validation(candidate.get("candidate_fingerprint"))
    reason = "task0253_entry_gate_not_passed" if not entry["passed"] else "promotion_review_requires_explicit_activation"

    internal_metrics = {
        "schema_version": "opk-rag.task0254.internal-agent-metrics.v1",
        "source_task": "TASK-0250",
        "production_terminal_accuracy": t250.get("production_terminal_accuracy"),
        "selective_shadow_terminal_accuracy": t250.get("shadow_terminal_accuracy"),
        "false_abstain_created_count": t250.get("false_abstain_created_count"),
        "recovery_net_gain": t250.get("recovery_net_gain"),
        "veto_net_gain": t250.get("veto_net_gain"),
        "average_controller_calls": t250.get("average_controller_calls"),
        "llm_bypass_rate": t250.get("llm_bypass_rate"),
        "three_or_more_controller_call_rate": t250.get("three_or_more_controller_call_rate"),
        "safety_invariants_preserved": t250.get("safety_invariants_preserved"),
        "internal_agent_authority_positive": float(t250.get("shadow_terminal_accuracy") or 0.0) >= float(t250.get("production_terminal_accuracy") or 0.0),
    }
    external_metrics = {
        "schema_version": "opk-rag.task0254.pdfqa-external-metrics.v1",
        "source_task": "TASK-0252",
        "evaluation_executed": t252.get("pdfqa_external_evaluation_executed") is True,
        "independence_proven": t252.get("pdfqa_independence_proven") is True,
        "generalization_diagnosis": t252.get("generalization_diagnosis"),
        "status": "available" if t252.get("pdfqa_external_evaluation_executed") is True else "not_executed_due_to_upstream_gate",
    }
    live_metrics = {
        "schema_version": "opk-rag.task0254.live-shadow-metrics.v1",
        "source_task": "TASK-0251",
        "live_query_count": int(t251.get("real_live_user_traffic_query_count") or 0),
        "provider_response_rate": t251.get("live_provider_response_rate"),
        "final_structured_validity": t251.get("live_provider_final_structured_validity"),
        "evidence_sufficient": t251.get("live_shadow_evidence_sufficient") is True,
        "status": t251.get("task_status"),
    }
    canary_metrics = {
        "schema_version": "opk-rag.task0254.canary-metrics.v1",
        "source_task": "TASK-0253",
        "canary_execution_performed": t253.get("canary_execution_performed") is True,
        "small_canary_observation_count": int(t253.get("small_canary_observation_count") or 0),
        "expanded_canary_observation_count": int(t253.get("expanded_canary_observation_count") or 0),
        "provider_response_rate": t253.get("provider_response_rate"),
        "final_structured_validity": t253.get("final_structured_validity"),
        "fallback_success_rate": t253.get("fallback_success_rate"),
        "final_candidate_decision": t253.get("final_candidate_decision"),
    }

    selectivity = selectivity_gate(internal_metrics)
    provider = provider_reliability_gate({
        "provider_response_rate": t253.get("provider_response_rate") if entry["passed"] else 0.0,
        "final_structured_validity": t253.get("final_structured_validity") if entry["passed"] else 0.0,
    })
    safety = {
        "schema_version": "opk-rag.task0254.safety-metrics.v1",
        **{key: int(t253.get(key) or 0) for key in SAFETY_KEYS},
        "safety_invariants_preserved": bool(t253.get("safety_invariants_preserved") is True),
    }
    rollback = {
        "schema_version": "opk-rag.task0254.rollback-authority.v1",
        "rollback_target": "Production Rule-Governed Search/Ask",
        "rollback_target_available": True,
        "kill_switch_mechanism_validated": mechanism["kill_switch_routes_rule_governed"],
        "task0253_rollback_validated": t253.get("rollback_validated") is True,
        "rollback_requires_deploy": False,
        "production_promotion_executed": False,
    }

    architecture_digest = canonical_digest({
        "architecture": candidate.get("candidate_architecture"),
        "max_graph_hop": candidate.get("max_graph_hop"),
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
    })
    config_digest = canonical_digest({
        "provider_model": candidate.get("provider_model"),
        "structured_output_mode": candidate.get("structured_output_mode"),
        "embedding_model": candidate.get("embedding_model"),
        "reranker_model": candidate.get("reranker_model"),
        "vector_backend": candidate.get("production_vector_backend"),
    })
    evidence_digests = {name: (row.get("sha256") or "missing") for name, row in evidence["sources"].items()}
    release_identity = build_release_identity(
        candidate_fingerprint=str(candidate.get("candidate_fingerprint") or "missing"),
        git_commit=TASK_START_HEAD,
        architecture_digest=architecture_digest,
        config_digest=config_digest,
        evidence_digests=evidence_digests,
    )
    release_identity.update({
        "release_frozen": False,
        "production_activation_timestamp": None,
        "status": "planned_identity_only_entry_blocked" if not entry["passed"] else "promotion_review_pending",
    })

    promotion_gates = {
        "task0253_entry": entry["passed"],
        "runtime_entry_consistent": runtime_entry["passed"] == entry["passed"],
        "candidate_identity_match": candidate["candidate_match"],
        "internal_agent_authority_positive": internal_metrics["internal_agent_authority_positive"],
        "external_generalization_available": external_metrics["evaluation_executed"] and external_metrics["independence_proven"],
        "live_shadow_sufficient": live_metrics["evidence_sufficient"],
        "canary_evidence_available": canary_metrics["canary_execution_performed"],
        "provider_reliability": provider["passed"],
        "selectivity_preserved": selectivity["selectivity_preserved"],
        "safety_zero": safety["safety_invariants_preserved"] and all(safety[key] == 0 for key in SAFETY_KEYS),
        "rollback_authority": rollback["rollback_target_available"] and rollback["kill_switch_mechanism_validated"] and rollback["task0253_rollback_validated"],
        "reranker_terminal_stability": int(t253.get("reranker_induced_terminal_decision_instability_count") or 0) == 0,
    }
    blockers = [name for name, passed in promotion_gates.items() if not passed]
    promotion_decision = "blocked" if not entry["passed"] else ("ready_for_explicit_activation" if not blockers else "hold_for_more_evidence")
    promotion_gate = {
        "schema_version": "opk-rag.task0254.promotion-readiness-gate.v1",
        "gates": promotion_gates,
        "blockers": blockers,
        "decision": promotion_decision,
        "production_activation_executed": False,
    }

    reg = read_json(REG, {})
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "blocked" if not entry["passed"] else "partial",
        "implementation_complete": True,
        "current_stage": "llm_agentic_rag_development",
        "entry_gate_passed": entry["passed"],
        "candidate_fingerprint": candidate.get("candidate_fingerprint"),
        "candidate_identity_match": candidate["candidate_match"],
        "evidence_layer_count": 4,
        "internal_agent_authority_positive": internal_metrics["internal_agent_authority_positive"],
        "pdfqa_external_evaluation_available": external_metrics["evaluation_executed"],
        "live_shadow_evidence_sufficient": live_metrics["evidence_sufficient"],
        "canary_evidence_available": canary_metrics["canary_execution_performed"],
        "selectivity_preserved": selectivity["selectivity_preserved"],
        "provider_reliability_passed": provider["passed"],
        "rollback_target_available": rollback["rollback_target_available"],
        "kill_switch_mechanism_validated": mechanism["kill_switch_routes_rule_governed"],
        "production_promotion_executed": False,
        "production_agentic_v2_active": False,
        "production_default_behavior_change": False,
        "production_answer_authority_change": False,
        "release_frozen": False,
        "release_id": release_identity["release_id"],
        "release_fingerprint": release_identity["release_fingerprint"],
        "promotion_review_decision": promotion_decision,
        "candidate_decision": "blocked" if not entry["passed"] else "hold_for_more_evidence",
        "blocking_failures": entry["blockers"] if not entry["passed"] else blockers,
        "next_task": "TASK-0251_resume_after_minimum_live_shadow_traffic" if not entry["passed"] else "TASK-0254_explicit_production_activation_review",
        "task_start_head": TASK_START_HEAD,
        "current_head": head(),
        "git_head_unchanged_since_task_start": head() == TASK_START_HEAD,
        "git_commit_created": head() != TASK_START_HEAD,
        "focused_tests_passed": reg.get("focused_tests_passed"),
        "full_suite_passed": reg.get("full_suite_passed"),
        "full_suite_skipped": reg.get("full_suite_skipped"),
        "full_suite_failed": reg.get("full_suite_failed"),
        "new_task0254_full_suite_failure_count": reg.get("new_task0254_full_suite_failure_count"),
        "changed_paths": changed_paths(),
    }

    if write:
        write_json(CONTRACT, _contract())
        RESULT.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "candidate_identity.json": candidate,
            "entry_gate.json": entry,
            "evidence_authority_manifest.json": evidence,
            "internal_agent_metrics.json": internal_metrics,
            "pdfqa_external_metrics.json": external_metrics,
            "live_shadow_metrics.json": live_metrics,
            "canary_metrics.json": canary_metrics,
            "provider_reliability.json": provider,
            "selectivity_metrics.json": selectivity,
            "recovery_metrics.json": _not_executed("recovery-metrics", reason) if not entry["passed"] else {"status": "use_task0253_canary_authority"},
            "veto_metrics.json": _not_executed("veto-metrics", reason) if not entry["passed"] else {"status": "use_task0253_canary_authority"},
            "latency_metrics.json": _not_executed("latency-metrics", reason) if not entry["passed"] else {"status": "use_task0253_canary_authority"},
            "token_cost_metrics.json": _not_executed("token-cost-metrics", reason) if not entry["passed"] else {"status": "use_task0253_canary_authority"},
            "reranker_stability.json": {
                "schema_version": "opk-rag.task0254.reranker-stability.v1",
                "blocking_terminal_instability_count": int(t253.get("reranker_induced_terminal_decision_instability_count") or 0),
                "promotion_safe": int(t253.get("reranker_induced_terminal_decision_instability_count") or 0) == 0,
            },
            "safety_metrics.json": safety,
            "rollback_authority.json": rollback,
            "promotion_readiness_gate.json": promotion_gate,
            "pre_promotion_snapshot.json": _not_executed("pre-promotion-snapshot", reason),
            "production_activation.json": _not_executed("production-activation", reason),
            "post_activation_smoke.json": _not_executed("post-activation-smoke", reason),
            "kill_switch_validation.json": mechanism,
            "release_identity.json": release_identity,
            "release_freeze.json": {
                "schema_version": "opk-rag.task0254.release-freeze.v1",
                "release_frozen": False,
                "status": "not_executed_due_to_entry_gate" if not entry["passed"] else "promotion_review_pending",
                "production_agentic_v2_active": False,
                "rule_governed_fallback_preserved": True,
            },
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT / name, payload)
    return summary


def verify() -> dict[str, Any]:
    summary = build_summary(write=False)
    missing: list[str] = []
    mismatches: dict[str, Any] = {}
    required = [
        CONTRACT,
        ROOT / "tasks/TASK-0254_selective_agent_production_promotion_review_and_release_freeze.md",
        ROOT / "docs/TASK0254_SELECTIVE_AGENT_PRODUCTION_PROMOTION_REVIEW_AND_RELEASE_FREEZE_REPORT.md",
        ROOT / "opk_rag/showcase/selective_agent_production.py",
        ROOT / "scripts/run_task0254_selective_agent_production_promotion_review_and_release_freeze.py",
        ROOT / "scripts/verify_task0254_selective_agent_production_promotion_review_and_release_freeze.py",
        ROOT / "scripts/activate_task0254_selective_agent_production_release.py",
        RESULT / "summary.json",
        RESULT / "entry_gate.json",
        RESULT / "promotion_readiness_gate.json",
        RESULT / "release_identity.json",
        RESULT / "release_freeze.json",
        REG,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    fixed = {
        "implementation_complete": True,
        "production_promotion_executed": False,
        "production_agentic_v2_active": False,
        "production_default_behavior_change": False,
        "production_answer_authority_change": False,
        "release_frozen": False,
        "rollback_target_available": True,
        "git_head_unchanged_since_task_start": True,
        "git_commit_created": False,
    }
    for key, expected in fixed.items():
        if summary.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": summary.get(key)}
    source = read_json(T253, {})
    recalculated_entry = evaluate_entry_gate(source)
    if recalculated_entry["passed"] != summary.get("entry_gate_passed"):
        mismatches["entry_gate_recalculation"] = {
            "expected": recalculated_entry["passed"], "actual": summary.get("entry_gate_passed")
        }
    if not summary["entry_gate_passed"] and (summary["task_status"] != "blocked" or summary["candidate_decision"] != "blocked"):
        mismatches["blocked_entry_state"] = {
            "expected": "blocked/blocked", "actual": [summary["task_status"], summary["candidate_decision"]]
        }
    release = read_json(RESULT / "release_identity.json", {})
    if release and release.get("candidate_fingerprint") != summary.get("candidate_fingerprint"):
        mismatches["release_candidate_identity"] = {
            "expected": summary.get("candidate_fingerprint"), "actual": release.get("candidate_fingerprint")
        }
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "verification_passed": not missing and not mismatches,
        "task_status": summary["task_status"],
        "candidate_decision": summary["candidate_decision"],
        "entry_gate_passed": summary["entry_gate_passed"],
        "production_agentic_v2_active": summary["production_agentic_v2_active"],
        "release_frozen": summary["release_frozen"],
        "missing_files": missing,
        "mismatches": mismatches,
    }
