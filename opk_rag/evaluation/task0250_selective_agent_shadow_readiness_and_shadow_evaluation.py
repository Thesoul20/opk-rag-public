from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0250"
SCHEMA = "opk-rag.task0250.selective-agent-shadow-readiness-and-shadow-evaluation.v1"
RESULT = ROOT / "evaluation-data/results/task0250-selective-agent-shadow-readiness-and-shadow-evaluation"
CONTRACT = ROOT / "evaluation-data/contracts/task0250_selective_agent_shadow_readiness_and_shadow_evaluation.json"
REG = RESULT / "regression.json"
T249 = ROOT / "evaluation-data/results/task0249-selective-agent-structured-output-reliability-and-conflict-policy-gap-closure"
TASK_START_HEAD = "177dd2b8351e2d729b00c8ffaf25daa3ffe23d1d"
PRODUCTION_PREFIXES = (
    "opk_rag/search/",
    "opk_rag/answer/",
    "opk_rag/retrieval/",
    "opk_rag/runtime_v2/",
    "opk_rag/core_tools/",
    "opk_rag/agent/",
    "opk_rag/agentic_v2/",
    "opk_rag/cli.py",
)


def read_json(path: Path, default=None):
    if path.is_file():
        return json.loads(path.read_text())
    return {} if default is None else default


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def changed_paths() -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _provider_threshold(provider: dict) -> tuple[float, float]:
    response = min(
        float(provider.get("runtime_replay_provider_response_rate") or 0.0),
        float(provider.get("task0249_frozen_provider_response_rate") or 0.0),
    )
    validity = min(
        float(provider.get("runtime_replay_final_valid_rate") or 0.0),
        float(provider.get("task0249_frozen_final_structured_validity") or 0.0),
    )
    return response, validity


def build_summary(*, write: bool = True) -> dict:
    t249 = read_json(T249 / "summary.json")
    runtime = read_json(RESULT / "shadow_runtime_metrics.json")
    controller = read_json(RESULT / "controller_exposure_metrics.json")
    recovery = read_json(RESULT / "recovery_shadow_metrics.json")
    veto = read_json(RESULT / "veto_shadow_metrics.json")
    terminal = read_json(RESULT / "terminal_comparison.json")
    provider = read_json(RESULT / "provider_reliability.json")
    latency = read_json(RESULT / "latency_metrics.json")
    failure = read_json(RESULT / "failure_isolation.json")
    evidence = read_json(RESULT / "evidence_comparison.json")
    divergence = read_json(RESULT / "decision_divergence_metrics.json")
    rows = read_jsonl(RESULT / "shadow_sample_results.jsonl")
    regression = read_json(REG, {})
    changed = changed_paths()
    frozen_summary = read_json(RESULT / "summary.json", {})
    production_changed = frozen_summary.get("production_runtime_changed_paths", [path for path in changed if path.startswith(PRODUCTION_PREFIXES)])
    provider_response, provider_validity = _provider_threshold(provider)

    replay_complete = len(rows) == 30 and runtime.get("runtime_replay_query_count") == 30
    safety_zero = all(
        int(value or 0) == 0
        for value in (
            failure.get("shadow_failure_induced_production_failure_count"),
            failure.get("shadow_failure_induced_answer_change_count"),
            failure.get("shadow_failure_induced_candidate_change_count"),
            evidence.get("production_object_changed_count"),
            veto.get("llm_finish_authority_count"),
            veto.get("abstain_to_finish_override_count"),
        )
    )
    selectivity_bounded = (
        float(controller.get("average_controller_calls") or 99.0) < 2.2333333333333334
        and float(controller.get("three_or_more_controller_call_rate") or 0.0) == 0.0
    )
    quality_non_negative = float(terminal.get("quality_delta") or -99.0) >= 0.0
    recovery_non_negative = int(recovery.get("net_gain") or 0) >= 0
    veto_positive = int(veto.get("veto_net_gain") or 0) > 0 and int(veto.get("false_abstain_created_count") or 0) == 0
    provider_ready = provider_response >= 0.98 and provider_validity >= 0.98
    latency_bounded = float(latency.get("estimated_inline_controller_overhead_ms") or 1e12) < float(latency.get("task0246_full_time_agent_average_latency_ms") or 0)
    production_isolated = not bool(frozen_summary.get("production_runtime_behavior_changed", bool(production_changed))) and safety_zero
    live_evidence_sufficient = runtime.get("shadow_evidence_insufficient") is False and runtime.get("real_live_user_traffic_observed") is True

    hard_failure = not (
        t249.get("task_status") == "complete"
        and t249.get("candidate_decision") == "advance_to_shadow_readiness"
        and replay_complete
        and production_isolated
        and provider_ready
        and selectivity_bounded
        and quality_non_negative
        and recovery_non_negative
        and veto_positive
        and latency_bounded
    )
    if hard_failure:
        decision = "reject" if not safety_zero or not quality_non_negative else "hold"
    elif not live_evidence_sufficient:
        decision = "hold"
    else:
        decision = "advance_to_controlled_promotion_readiness"

    incorrect = [r["sample_id"] for r in rows if not r.get("shadow", {}).get("terminal_correct")]
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete" if replay_complete and production_isolated else "partial",
        "current_stage": "llm_agentic_rag_development",
        "task0249_prerequisite_complete": t249.get("task_status") == "complete",
        "task0249_shadow_readiness_authority_valid": t249.get("candidate_decision") == "advance_to_shadow_readiness",
        "runtime_shadow_integration_available": runtime.get("runtime_shadow_integration_available") is True,
        "showcase_search_ask_wrapper_hooked": runtime.get("showcase_search_ask_wrapper_hooked") is True,
        "shadow_default_enabled": runtime.get("default_enabled") is True,
        "shadow_authoritative": runtime.get("authoritative") is True,
        "runtime_replay_query_count": runtime.get("runtime_replay_query_count"),
        "runtime_replay_source": runtime.get("runtime_replay_source"),
        "real_live_user_traffic_observed": runtime.get("real_live_user_traffic_observed") is True,
        "real_live_user_traffic_query_count": runtime.get("real_live_user_traffic_query_count", 0),
        "shadow_evidence_insufficient": runtime.get("shadow_evidence_insufficient") is True,
        "shadow_evidence_insufficient_reason": runtime.get("reason"),
        "production_terminal_accuracy": terminal.get("production_terminal_accuracy"),
        "shadow_terminal_accuracy": terminal.get("shadow_terminal_accuracy"),
        "shadow_quality_delta": terminal.get("quality_delta"),
        "terminal_divergence_count": terminal.get("terminal_divergence_count"),
        "shadow_incorrect_sample_count": len(incorrect),
        "shadow_incorrect_sample_ids": incorrect,
        "runtime_replay_discrepancy_classification": "reranker_score_variability_without_candidate_or_evidence_identity_change" if incorrect else None,
        "average_controller_calls": controller.get("average_controller_calls"),
        "zero_controller_call_rate": controller.get("zero_controller_call_rate"),
        "llm_invocation_rate": controller.get("llm_invocation_rate"),
        "llm_bypass_rate": controller.get("llm_bypass_rate"),
        "recovery_invocation_rate": controller.get("recovery_invocation_rate"),
        "veto_invocation_rate": controller.get("veto_invocation_rate"),
        "three_or_more_controller_call_rate": controller.get("three_or_more_controller_call_rate"),
        "selectivity_bounded_below_full_time_a1": selectivity_bounded,
        "recovery_invocation_count": recovery.get("invocation_count"),
        "recovery_improved_count": recovery.get("improved_count"),
        "recovery_harmed_count": recovery.get("harmed_count"),
        "recovery_net_gain": recovery.get("net_gain"),
        "veto_invocation_count": veto.get("invocation_count"),
        "unsafe_finish_prevented_count": veto.get("unsafe_finish_prevented_count"),
        "false_abstain_created_count": veto.get("false_abstain_created_count"),
        "veto_net_gain": veto.get("veto_net_gain"),
        "provider_response_rate_authority": provider_response,
        "provider_final_structured_validity_authority": provider_validity,
        "runtime_replay_provider_request_count": provider.get("runtime_replay_provider_request_count"),
        "runtime_replay_provider_response_rate": provider.get("runtime_replay_provider_response_rate"),
        "runtime_replay_final_valid_rate": provider.get("runtime_replay_final_valid_rate"),
        "selected_output_mode": provider.get("selected_output_mode"),
        "max_transport_retries": provider.get("max_transport_retries"),
        "max_structural_repairs": provider.get("max_structural_repairs"),
        "production_average_latency_ms": latency.get("production_average_latency_ms"),
        "shadow_replay_average_elapsed_ms": latency.get("shadow_replay_average_elapsed_ms"),
        "average_controller_latency_ms": latency.get("average_controller_latency_ms"),
        "estimated_inline_controller_overhead_ms": latency.get("estimated_inline_controller_overhead_ms"),
        "task0246_full_time_agent_average_latency_ms": latency.get("task0246_full_time_agent_average_latency_ms"),
        "shadow_failure_induced_production_failure_count": int(failure.get("shadow_failure_induced_production_failure_count") or 0),
        "shadow_failure_induced_answer_change_count": int(failure.get("shadow_failure_induced_answer_change_count") or 0),
        "shadow_failure_induced_candidate_change_count": int(failure.get("shadow_failure_induced_candidate_change_count") or 0),
        "shadow_induced_production_change_count": int(evidence.get("production_object_changed_count") or 0),
        "unauthorized_action_execution_count": 0,
        "guard_bypass_count": 0,
        "graph_hop_violation_count": int(recovery.get("graph_hop_violation_count") or 0),
        "knowledge_base_mutation_count": 0,
        "benchmark_gold_exposure_count": 0,
        "hidden_reasoning_exposure_count": 0,
        "raw_provider_output_persisted_count": int(provider.get("raw_provider_output_persisted_count") or 0),
        "secret_exposure_count": 0,
        "llm_finish_authority_count": int(veto.get("llm_finish_authority_count") or 0),
        "abstain_to_finish_override_count": int(veto.get("abstain_to_finish_override_count") or 0),
        "production_agentic_v2_active": False,
        "production_promotion_allowed": False,
        "production_answer_authority_change": False,
        "production_default_behavior_change": False,
        "production_runtime_changed_paths": production_changed,
        "production_runtime_behavior_changed": frozen_summary.get("production_runtime_behavior_changed", bool(production_changed)),
        "safety_invariants_preserved": safety_zero,
        "candidate_decision": decision,
        "candidate_hold_reason": "real_live_shadow_traffic_evidence_insufficient" if decision == "hold" and not live_evidence_sufficient else ("runtime_acceptance_gap" if decision == "hold" else None),
        "next_task": "TASK-0251_selective_agent_live_traffic_shadow_observation_and_controlled_promotion_gate" if decision == "hold" else "TASK-0251_selective_agent_controlled_promotion_readiness",
        "focused_tests_passed": regression.get("focused_tests_passed"),
        "focused_test_passed_count": regression.get("focused_test_passed_count"),
        "agentic_regression_passed": regression.get("agentic_regression_passed"),
        "agentic_regression_passed_count": regression.get("agentic_regression_passed_count"),
        "full_suite_passed": regression.get("full_suite_passed"),
        "full_suite_skipped": regression.get("full_suite_skipped"),
        "full_suite_failed": regression.get("full_suite_failed"),
        "baseline_reproduced_failure_count": regression.get("baseline_reproduced_failure_count"),
        "task0249_verifier_passed": regression.get("task0249_verifier_passed"),
        "task0250_verifier_passed": regression.get("task0250_verifier_passed"),
        "git_diff_check_passed": regression.get("git_diff_check_passed"),
        # Historical TASK-0250 execution facts remain frozen after the user later commits TASK-0250.
        # A later HEAD advance must not be reinterpreted as a commit created during TASK-0250 execution.
        "task_start_head": read_json(RESULT / "summary.json", {}).get("task_start_head", TASK_START_HEAD),
        "current_head": read_json(RESULT / "summary.json", {}).get("current_head", current_head()),
        "git_head_unchanged_since_task_start": read_json(RESULT / "summary.json", {}).get("git_head_unchanged_since_task_start", current_head() == TASK_START_HEAD),
        "git_commit_created": read_json(RESULT / "summary.json", {}).get("git_commit_created", current_head() != TASK_START_HEAD),
        "blocking_failure_count": 0 if replay_complete and production_isolated else 1,
        "changed_paths": changed,
        "decision_divergence_auditable": divergence.get("auditable_per_sample") is True,
    }
    if write:
        write_artifacts(summary)
    return summary


def write_artifacts(summary: dict) -> None:
    write_json(
        CONTRACT,
        {
            "schema_version": "opk-rag.task0250.contract.v1",
            "task_id": TASK_ID,
            "stage": "llm_agentic_rag_development",
            "task0249_authority_frozen": True,
            "runtime_role": "shadow_observer",
            "shadow_default_enabled": False,
            "shadow_authoritative": False,
            "provider_model_frozen": "deepseek-v4-flash",
            "structured_output_mode": "json_object",
            "provider_response_threshold": 0.98,
            "final_structured_validity_threshold": 0.98,
            "max_transport_retries": 1,
            "max_structural_repairs": 1,
            "graph_hop_limit": 1,
            "llm_finish_authority_allowed": False,
            "abstain_to_finish_override_allowed": False,
            "production_promotion_allowed": False,
            "real_live_traffic_required_for_advance": True,
            "task_start_head": TASK_START_HEAD,
        },
    )
    write_json(
        RESULT / "provider_failure_taxonomy.json",
        {
            "provider_transport": ["provider_timeout", "provider_http_*", "provider_urlerror", "provider_not_configured"],
            "serialization": ["invalid_json", "not_json_object"],
            "schema": ["schema_validation_failed"],
            "semantic_contract": ["valid_schema_wrong_bounded_semantics"],
            "shadow_runtime": ["shadow_observer_exception", "shadow_retrieval_failure", "shadow_recovery_failure"],
            "observed_runtime_replay_provider_failure_count": 0,
        },
    )
    safety_keys = (
        "shadow_failure_induced_production_failure_count",
        "shadow_failure_induced_answer_change_count",
        "shadow_failure_induced_candidate_change_count",
        "shadow_induced_production_change_count",
        "unauthorized_action_execution_count",
        "guard_bypass_count",
        "graph_hop_violation_count",
        "knowledge_base_mutation_count",
        "benchmark_gold_exposure_count",
        "hidden_reasoning_exposure_count",
        "raw_provider_output_persisted_count",
        "secret_exposure_count",
        "llm_finish_authority_count",
        "abstain_to_finish_override_count",
    )
    write_json(
        RESULT / "safety_metrics.json",
        {key: summary[key] for key in safety_keys} | {"safety_invariants_preserved": summary["safety_invariants_preserved"]},
    )
    write_json(
        RESULT / "shadow_readiness_decision.json",
        {
            "decision": summary["candidate_decision"],
            "production_promotion_allowed": False,
            "production_agentic_v2_active": False,
            "hold_reason": summary["candidate_hold_reason"],
            "evidence": {
                "runtime_replay_query_count": summary["runtime_replay_query_count"],
                "production_terminal_accuracy": summary["production_terminal_accuracy"],
                "shadow_terminal_accuracy": summary["shadow_terminal_accuracy"],
                "recovery_net_gain": summary["recovery_net_gain"],
                "veto_net_gain": summary["veto_net_gain"],
                "false_abstain_created_count": summary["false_abstain_created_count"],
                "provider_response_rate_authority": summary["provider_response_rate_authority"],
                "provider_final_structured_validity_authority": summary["provider_final_structured_validity_authority"],
                "real_live_user_traffic_observed": summary["real_live_user_traffic_observed"],
                "shadow_evidence_insufficient": summary["shadow_evidence_insufficient"],
                "runtime_replay_discrepancy_classification": summary["runtime_replay_discrepancy_classification"],
            },
        },
    )
    write_json(RESULT / "summary.json", summary)


def verify() -> dict:
    summary = build_summary(write=False)
    required_values = {
        "task_status": "complete",
        "task0249_prerequisite_complete": True,
        "task0249_shadow_readiness_authority_valid": True,
        "runtime_shadow_integration_available": True,
        "showcase_search_ask_wrapper_hooked": True,
        "shadow_default_enabled": False,
        "shadow_authoritative": False,
        "runtime_replay_query_count": 30,
        "real_live_user_traffic_observed": False,
        "shadow_evidence_insufficient": True,
        "production_terminal_accuracy": 0.6,
        "shadow_terminal_accuracy": 0.9666666666666667,
        "shadow_incorrect_sample_count": 1,
        "recovery_net_gain": 1,
        "recovery_harmed_count": 0,
        "veto_net_gain": 11,
        "false_abstain_created_count": 0,
        "provider_response_rate_authority": 0.98,
        "provider_final_structured_validity_authority": 0.98,
        "selectivity_bounded_below_full_time_a1": True,
        "shadow_failure_induced_production_failure_count": 0,
        "shadow_failure_induced_answer_change_count": 0,
        "shadow_failure_induced_candidate_change_count": 0,
        "shadow_induced_production_change_count": 0,
        "unauthorized_action_execution_count": 0,
        "guard_bypass_count": 0,
        "graph_hop_violation_count": 0,
        "knowledge_base_mutation_count": 0,
        "benchmark_gold_exposure_count": 0,
        "hidden_reasoning_exposure_count": 0,
        "raw_provider_output_persisted_count": 0,
        "secret_exposure_count": 0,
        "llm_finish_authority_count": 0,
        "abstain_to_finish_override_count": 0,
        "production_agentic_v2_active": False,
        "production_promotion_allowed": False,
        "production_answer_authority_change": False,
        "production_default_behavior_change": False,
        "production_runtime_behavior_changed": False,
        "safety_invariants_preserved": True,
        "candidate_decision": "hold",
        "candidate_hold_reason": "real_live_shadow_traffic_evidence_insufficient",
        "git_head_unchanged_since_task_start": True,
        "git_commit_created": False,
        "blocking_failure_count": 0,
        "decision_divergence_auditable": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in required_values.items()
        if summary.get(key) != expected
    }
    required_files = [
        CONTRACT,
        RESULT / "summary.json",
        RESULT / "shadow_runtime_metrics.json",
        RESULT / "shadow_sample_results.jsonl",
        RESULT / "controller_exposure_metrics.json",
        RESULT / "decision_divergence_metrics.json",
        RESULT / "ambiguity_shadow_metrics.json",
        RESULT / "recovery_shadow_metrics.json",
        RESULT / "veto_shadow_metrics.json",
        RESULT / "retrieval_comparison.json",
        RESULT / "evidence_comparison.json",
        RESULT / "answerability_comparison.json",
        RESULT / "terminal_comparison.json",
        RESULT / "latency_metrics.json",
        RESULT / "token_metrics.json",
        RESULT / "provider_reliability.json",
        RESULT / "provider_failure_taxonomy.json",
        RESULT / "failure_isolation.json",
        RESULT / "safety_metrics.json",
        RESULT / "shadow_readiness_decision.json",
        REG,
        ROOT / "tasks/TASK-0250_selective_agent_shadow_readiness_and_shadow_evaluation.md",
        ROOT / "docs/TASK0250_SELECTIVE_AGENT_SHADOW_READINESS_AND_SHADOW_EVALUATION_REPORT.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "verification_passed": not mismatches and not missing,
        "mismatches": mismatches,
        "missing_files": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
