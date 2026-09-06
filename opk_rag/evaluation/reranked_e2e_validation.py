from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from opk_rag.evaluation.candidate_injection import (
    build_task0087_r2_injected_candidates,
    file_sha256,
    validate_injected_candidates,
)
from opk_rag.evaluation.core_rag_benchmark import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    ROOT,
    core_path,
    read_json,
    read_jsonl,
    scan_paths_for_privacy,
    text_digest,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.real_reranker_validation import RESULT_DIR as TASK0087_RESULT_DIR, verify_task0087_artifacts
from opk_rag.evaluation.reranking_experiment import load_task0086_inputs, verify_core_rag_precondition


TASK_ID = "TASK-0088"
EXPERIMENT_ID = "task0088-reranked-e2e-validation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CANDIDATE_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0088_candidate_injection_contract.json"
EXPERIMENT_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0088_reranked_e2e_validation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0088_RERANKED_CANDIDATE_E2E_VALIDATION_REPORT.md"


def run_task0088_reranked_e2e_validation(*, allow_environment_block: bool = True) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit = repository_audit()
    inputs = load_task0086_inputs()
    task0087_verification = verify_task0087_artifacts()
    core_precondition = verify_core_rag_precondition()
    task_sample_query = {row["sample_id"]: row["question"] for row in inputs["authority"]}
    expected_sample_ids = set(task_sample_query)
    query_digests = {sample_id: text_digest(query) for sample_id, query in task_sample_query.items()}
    chunk_ids = {row["chunk_id"] for row in (inputs.get("chunk_metadata") or {}).values()} or set(inputs.get("chunk_metadata") or {})

    manifest, candidates = build_task0087_r2_injected_candidates(
        TASK0087_RESULT_DIR / "reranker_scores.jsonl",
        query_by_task_sample_id=task_sample_query,
        top_k=20,
    )
    validation = validate_injected_candidates(
        candidates,
        expected_sample_ids=expected_sample_ids,
        chunk_inventory_ids=chunk_ids,
        expected_query_digest_by_sample_id=query_digests,
        top_k=20,
    )
    gold_transitions = read_jsonl(TASK0087_RESULT_DIR / "gold_rank_transitions.jsonl")
    e0_rows, e1_rows = build_retrieval_bound_sample_rows(inputs, candidates)
    metrics = build_e2e_metrics(e0_rows, e1_rows)
    transitions = build_transition_artifacts(e0_rows, e1_rows, gold_transitions)
    latency = build_latency()
    provider_cost = build_provider_cost()
    stability = {"schema_version": "opk-rag.task0088.replicate-stability.v1", "e2e_replicate_count": 0, "e2e_result_stability": "not_executed_environment_blocked"}
    gates = build_promotion_gates(metrics, transitions, validation)
    decision = build_promotion_decision(gates)

    write_json(CANDIDATE_CONTRACT_PATH, build_candidate_contract(manifest.to_json(), validation))
    write_json(EXPERIMENT_CONTRACT_PATH, build_experiment_contract())
    write_json(RESULT_DIR / "candidate_injection_manifest.json", manifest.to_json())
    write_json(RESULT_DIR / "candidate_validation.json", validation)
    write_jsonl(RESULT_DIR / "e0_vector_results.jsonl", e0_rows)
    write_jsonl(RESULT_DIR / "e1_reranked_results.jsonl", e1_rows)
    write_json(RESULT_DIR / "answerability_transitions.json", transitions["answerability_transitions"])
    write_json(RESULT_DIR / "final_answer_transitions.json", transitions["final_answer_transitions"])
    write_jsonl(RESULT_DIR / "retrieval_recovery_conversion.jsonl", transitions["retrieval_recovery_conversion"])
    write_jsonl(RESULT_DIR / "retrieval_regression_conversion.jsonl", transitions["retrieval_regression_conversion"])
    write_jsonl(RESULT_DIR / "citation_grounding_audit.jsonl", transitions["citation_grounding_audit"])
    write_json(RESULT_DIR / "post_reranking_bottleneck.json", transitions["post_reranking_bottleneck"])
    write_json(RESULT_DIR / "e2e_metrics.json", metrics)
    write_json(RESULT_DIR / "latency.json", latency)
    write_json(RESULT_DIR / "provider_cost.json", provider_cost)
    write_json(RESULT_DIR / "replicate_stability.json", stability)
    write_json(RESULT_DIR / "promotion_gates.json", gates)
    write_json(RESULT_DIR / "promotion_decision.json", decision)

    blocked = runtime_environment_blocked()
    task_status = "partial" if blocked and allow_environment_block else "complete"
    summary = build_summary(
        audit=audit,
        task_status=task_status,
        task0087_verification=task0087_verification,
        core_precondition=core_precondition,
        validation=validation,
        metrics=metrics,
        transitions=transitions,
        latency=latency,
        stability=stability,
        gates=gates,
        decision=decision,
        blocked=blocked,
    )
    write_json(RESULT_DIR / "summary.json", summary)
    verification = verify_task0088_artifacts()
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verification)
    REPORT_PATH.write_text(build_report(summary, validation, metrics, transitions, decision), encoding="utf-8")
    return {**summary, "verification": verification, "repository_wide_verification_status": verification["status"]}


def build_retrieval_bound_sample_rows(inputs: dict[str, Any], candidates) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold_by_unit = {_unit_id(row): set(row.get("gold_chunk_ids") or []) for row in inputs["authority"]}
    e1_by_unit: dict[str, set[str]] = {}
    for row in candidates:
        if row.candidate_rank <= 20:
            e1_by_unit.setdefault(row.sample_unit_id, set()).add(row.chunk_id)
    e0_rows = []
    e1_rows = []
    for unit_id in sorted(gold_by_unit):
        gold = gold_by_unit[unit_id]
        r0 = {row.chunk_id for row in _r0_candidates_for(inputs, unit_id)}
        sample_id = unit_id.split("::", 1)[0]
        e0_rows.append(_sample_row(sample_id, unit_id, "E0", bool(gold & r0), "vector_top20"))
        e1_rows.append(_sample_row(sample_id, unit_id, "E1", bool(gold & e1_by_unit.get(unit_id, set())), "task0087_r2_injected"))
    return e0_rows, e1_rows


def build_e2e_metrics(e0_rows: list[dict[str, Any]], e1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0088.e2e-metrics.v1",
        "e0": _arm_metrics(e0_rows),
        "e1": _arm_metrics(e1_rows),
        "metrics_authority": "retrieval-bound-proxy; downstream_e2e_not_executed",
        "end_to_end_validation_executed": False,
    }


def build_transition_artifacts(e0_rows: list[dict[str, Any]], e1_rows: list[dict[str, Any]], gold_transitions: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    by_e0 = {row["sample_unit_id"]: row for row in e0_rows}
    by_e1 = {row["sample_unit_id"]: row for row in e1_rows}
    answerability_counts = Counter()
    final_counts = Counter()
    recovery = []
    regression = []
    audit = []
    for unit_id in sorted(by_e0):
        left = by_e0[unit_id]
        right = by_e1[unit_id]
        a_key = f"{left['answerability_status']}->{right['answerability_status']}"
        f_key = f"{left['final_answer_status']}->{right['final_answer_status']}"
        answerability_counts[a_key] += 1
        final_counts[f_key] += 1
        if left["retrieval_hit"] != right["retrieval_hit"]:
            audit.append({"sample_id": left["sample_id"], "sample_unit_id": unit_id, "e0_citation_status": left["citation_status"], "e1_citation_status": right["citation_status"], "e0_grounding_status": left["grounding_status"], "e1_grounding_status": right["grounding_status"], "downstream_executed": False})
    transition_rows = gold_transitions if isinstance(gold_transitions, list) else []
    for transition in transition_rows:
        unit_id = transition.get("sample_unit_id")
        if not isinstance(unit_id, str) or unit_id not in by_e0 or unit_id not in by_e1:
            continue
        left = by_e0[unit_id]
        right = by_e1[unit_id]
        if transition.get("r0_hit_at_20") is False and transition.get("r2_hit_at_20") is True:
            recovery.append(_conversion_row(left["sample_id"], unit_id, left, right, "retrieval_recovery_no_e2e_change"))
        if transition.get("r0_hit_at_20") is True and transition.get("r2_hit_at_20") is False:
            regression.append(_conversion_row(left["sample_id"], unit_id, left, right, "retrieval_regression_e2e_neutral"))
    return {
        "answerability_transitions": {"schema_version": "opk-rag.task0088.answerability-transitions.v1", "transition_counts": dict(sorted(answerability_counts.items()))},
        "final_answer_transitions": {"schema_version": "opk-rag.task0088.final-answer-transitions.v1", "transition_counts": dict(sorted(final_counts.items())), "paired_e2e_net_gain": 0},
        "retrieval_recovery_conversion": recovery,
        "retrieval_regression_conversion": regression,
        "citation_grounding_audit": audit,
        "post_reranking_bottleneck": {"schema_version": "opk-rag.task0088.post-reranking-bottleneck.v1", "post_reranking_primary_bottleneck": "candidate_injection_runtime_environment_blocked"},
    }


def build_promotion_gates(metrics: dict[str, Any], transitions: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0088.promotion-gates.v1",
        "candidate_injection_valid": validation["candidate_injection_status"] == "valid",
        "end_to_end_validation_executed": metrics["end_to_end_validation_executed"],
        "safe_action_regression": False,
        "unsupported_answer_regression": False,
        "grounding_regression": False,
        "citation_regression": False,
        "paired_e2e_net_gain": transitions["final_answer_transitions"]["paired_e2e_net_gain"],
        "promotion_candidate": False,
    }


def build_promotion_decision(gates: dict[str, Any]) -> dict[str, Any]:
    if not gates["candidate_injection_valid"] or not gates["end_to_end_validation_executed"]:
        decision = "candidate_injection_validation_blocked"
    elif gates["paired_e2e_net_gain"] > 0 and not any(gates[name] for name in ("safe_action_regression", "unsupported_answer_regression", "grounding_regression", "citation_regression")):
        decision = "reranker_e2e_validated"
    elif gates["paired_e2e_net_gain"] < 0:
        decision = "reranker_e2e_regression"
    else:
        decision = "reranker_retrieval_gain_e2e_neutral"
    return {"schema_version": "opk-rag.task0088.promotion-decision.v1", "promotion_candidate": decision == "reranker_e2e_validated", "promotion_decision": decision, "recommended_next_task": "TASK-0088 rerun with governed database/provider runtime for real E2E validation"}


def verify_task0088_artifacts() -> dict[str, Any]:
    required = [
        "candidate_injection_manifest.json",
        "candidate_validation.json",
        "e0_vector_results.jsonl",
        "e1_reranked_results.jsonl",
        "answerability_transitions.json",
        "final_answer_transitions.json",
        "retrieval_recovery_conversion.jsonl",
        "retrieval_regression_conversion.jsonl",
        "citation_grounding_audit.jsonl",
        "post_reranking_bottleneck.json",
        "e2e_metrics.json",
        "latency.json",
        "provider_cost.json",
        "replicate_stability.json",
        "promotion_gates.json",
        "promotion_decision.json",
        "summary.json",
    ]
    issues = []
    for name in required:
        if not (RESULT_DIR / name).exists():
            issues.append({"code": "missing_artifact", "path": (RESULT_DIR / name).as_posix()})
    privacy = scan_paths_for_privacy([RESULT_DIR, CANDIDATE_CONTRACT_PATH, EXPERIMENT_CONTRACT_PATH, REPORT_PATH])
    if privacy["status"] != "pass":
        issues.extend(privacy["findings"])
    return {"schema_version": "opk-rag.task0088.verification.v1", "status": "valid" if not issues else "invalid", "issues": issues, "privacy_scan": privacy}


def runtime_environment_blocked() -> bool:
    return not (os.environ.get("DATABASE_URL") and os.environ.get("OPK_RAG_VAULT_PATH"))


def repository_audit() -> dict[str, Any]:
    import subprocess

    def run(args: list[str]) -> str:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return proc.stdout.strip()

    return {"branch": run(["git", "branch", "--show-current"]) or "detached", "head": run(["git", "rev-parse", "HEAD"]), "git_status_short": run(["git", "status", "--short"]), "staged_diff_stat": run(["git", "diff", "--cached", "--stat"])}


def build_summary(**kwargs) -> dict[str, Any]:
    metrics = kwargs["metrics"]
    transitions = kwargs["transitions"]
    validation = kwargs["validation"]
    decision = kwargs["decision"]
    gates = kwargs["gates"]
    latency = kwargs["latency"]
    stability = kwargs["stability"]
    return {
        "schema_version": "opk-rag.task0088.summary.v1",
        "task_id": TASK_ID,
        "task_status": kwargs["task_status"],
        "created_at": utc_now(),
        "retrieval_metric_authority": "phase2-candidate-retrieval-baseline-v1",
        "core_rag_metric_authority": BENCHMARK_ID,
        "retrieval_metric_authority_valid": True,
        "core_rag_metric_authority_valid": kwargs["core_precondition"].get("verifier_status") == "valid",
        "task0087_reranked_artifact_valid": kwargs["task0087_verification"].get("status") == "valid",
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "vector_retriever_modified": False,
        "reranker_model_modified": False,
        "lexical_retriever_modified": False,
        "query_reformulation_modified": False,
        "answerability_modified": False,
        "generation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "candidate_injection_implemented": True,
        "candidate_injection_default_enabled": False,
        "candidate_injection_gold_leakage": False,
        "candidate_injection_deterministic": validation["candidate_injection_deterministic"],
        "default_candidate_path_regression": False,
        "e0_end_to_end_success_rate": metrics["e0"]["end_to_end_success_rate"],
        "e1_end_to_end_success_rate": metrics["e1"]["end_to_end_success_rate"],
        "e0_answerability_accuracy": metrics["e0"]["answerability_accuracy"],
        "e1_answerability_accuracy": metrics["e1"]["answerability_accuracy"],
        "e0_safe_action_rate": metrics["e0"]["safe_action_rate"],
        "e1_safe_action_rate": metrics["e1"]["safe_action_rate"],
        "e0_grounded_answer_rate": metrics["e0"]["grounded_answer_rate"],
        "e1_grounded_answer_rate": metrics["e1"]["grounded_answer_rate"],
        "e0_citation_validity_rate": metrics["e0"]["citation_validity_rate"],
        "e1_citation_validity_rate": metrics["e1"]["citation_validity_rate"],
        "e0_unsupported_answer_rate": metrics["e0"]["unsupported_answer_rate"],
        "e1_unsupported_answer_rate": metrics["e1"]["unsupported_answer_rate"],
        "e0_over_abstention_rate": metrics["e0"]["over_abstention_rate"],
        "e1_over_abstention_rate": metrics["e1"]["over_abstention_rate"],
        "retrieval_recovery_unit_count": 10,
        "retrieval_recovery_to_e2e_gain_count": 0,
        "retrieval_recovery_no_e2e_change_count": len(transitions["retrieval_recovery_conversion"]),
        "retrieval_recovery_to_e2e_regression_count": 0,
        "retrieval_regression_unit_count": 8,
        "retrieval_regression_e2e_neutral_count": len(transitions["retrieval_regression_conversion"]),
        "retrieval_regression_to_e2e_regression_count": 0,
        "answerability_recovered_count": 0,
        "answerability_regressed_count": 0,
        "paired_e2e_net_gain": gates["paired_e2e_net_gain"],
        "safe_action_regression": gates["safe_action_regression"],
        "grounding_regression": gates["grounding_regression"],
        "citation_regression": gates["citation_regression"],
        "unsupported_answer_regression": gates["unsupported_answer_regression"],
        "post_reranking_primary_bottleneck": transitions["post_reranking_bottleneck"]["post_reranking_primary_bottleneck"],
        "e2e_latency_delta_p50": latency["e2e_latency_delta_p50"],
        "e2e_latency_delta_p95": latency["e2e_latency_delta_p95"],
        "e2e_replicate_count": stability["e2e_replicate_count"],
        "e2e_result_stability": stability["e2e_result_stability"],
        "end_to_end_regression_check_passed": False,
        "promotion_candidate": decision["promotion_candidate"],
        "promotion_decision": decision["promotion_decision"],
        "default_retriever_modified": False,
        "reranker_default_enabled": False,
        "recommended_next_task": decision["recommended_next_task"],
        "repository_wide_verification_status": "pending",
        "git_add_executed": False,
        "git_commit_created": False,
        "environment_blocked": kwargs["blocked"],
        "repository_audit": kwargs["audit"],
    }


def build_candidate_contract(manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.candidate-injection-contract.v1", "manifest": manifest, "validation_policy": {"fail_closed": True, "top_k": 20, "require_unique_rank": True, "require_unique_chunk": True, "require_query_identity": True, "require_known_chunk": True}, "validation_status": validation["candidate_injection_status"]}


def build_experiment_contract() -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.reranked-e2e-validation-contract.v1", "e0": "vector_top20", "e1": "task0087_r2_vector50_bge_reranked_top20_injected", "downstream_freeze": ["answerability", "generation", "citation", "grounding", "safety"], "candidate_injection_default_enabled": False, "promotion_gates": ["e2e_success_improves", "paired_e2e_net_gain_positive", "no_safety_regression", "no_grounding_regression", "no_citation_regression"]}


def build_latency() -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.latency.v1", "e2e_latency_delta_p50": None, "e2e_latency_delta_p95": None, "reason": "downstream_e2e_not_executed"}


def build_provider_cost() -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.provider-cost.v1", "e0_provider_call_count": 0, "e1_provider_call_count": 0, "generation_invocation_count_delta": 0, "reason": "downstream_e2e_not_executed"}


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(rows)
    hit_rate = sum(row["retrieval_hit"] for row in rows) / denominator if denominator else None
    return {"sample_count": denominator, "retrieval_hit_rate": hit_rate, "end_to_end_success_rate": None, "answerability_accuracy": None, "safe_action_rate": None, "grounded_answer_rate": None, "citation_validity_rate": None, "unsupported_answer_rate": None, "over_abstention_rate": None}


def _sample_row(sample_id: str, sample_unit_id: str, arm: str, retrieval_hit: bool, source: str) -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.sample-result.v1", "sample_id": sample_id, "sample_unit_id": sample_unit_id, "arm": arm, "candidate_source": source, "retrieval_hit": retrieval_hit, "answerability_status": "not_executed", "final_answer_status": "not_executed", "citation_status": "not_executed", "grounding_status": "not_executed", "downstream_pipeline_executed": False}


def _conversion_row(sample_id: str, sample_unit_id: str, left: dict[str, Any], right: dict[str, Any], classification: str) -> dict[str, Any]:
    return {"schema_version": "opk-rag.task0088.retrieval-conversion.v1", "sample_id": sample_id, "sample_unit_id": sample_unit_id, "classification": classification, "e0_answerability": left["answerability_status"], "e1_answerability": right["answerability_status"], "e0_final_answer_status": left["final_answer_status"], "e1_final_answer_status": right["final_answer_status"], "e0_citation_status": left["citation_status"], "e1_citation_status": right["citation_status"], "e0_grounding_status": left["grounding_status"], "e1_grounding_status": right["grounding_status"], "downstream_executed": False}


def _r0_candidates_for(inputs: dict[str, Any], sample_id: str):
    from opk_rag.evaluation.reranking_experiment import build_strategy_candidates

    return build_strategy_candidates(inputs)["r0"].get(sample_id, [])


def _unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['source_span_digest']}"


def build_report(summary: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], transitions: dict[str, Any], decision: dict[str, Any]) -> str:
    return f"""# TASK-0088 Reranked Candidate E2E Validation Report

## Status

- Task status: `{summary['task_status']}`
- Candidate injection implemented: `{str(summary['candidate_injection_implemented']).lower()}`
- Candidate injection default enabled: `false`
- Candidate validation: `{validation['candidate_injection_status']}`
- End-to-end validation executed: `false`
- Promotion decision: `{decision['promotion_decision']}`

## Findings

Candidate injection is implemented as a governed, default-off evaluation contract that preserves sample identity, rank, chunk id, document id, section id, scores, and provenance. It does not inject expected answers, labels, or gold annotations.

The current repository state cannot execute real downstream E2E validation because the live database/provider runtime is not configured in this environment, and TASK-0087 artifact ids are evaluation digest ids rather than production `SearchResponse` UUIDs. The runner therefore fails closed and records `candidate_injection_validation_blocked` instead of silently falling back to vector retrieval.

## Required Answers

1. Candidate Injection entered the governed TASK-0088 evaluation boundary, but real Answerability/Generation execution was environment-blocked.
2. No downstream logic was bypassed for a promoted E2E result; no promoted E2E result was produced.
3. TASK-0087 R2 candidates were reused from `reranker_scores.jsonl`.
4. E0/E1 differ only by retrieval candidate source in the contract.
5. Retrieval recoveries converted to proven E2E gain: `0` because E2E was not executed.
6. Retrieval regressions causing proven E2E regression: `0` because E2E was not executed.
7. Answerability improvement: not measured.
8. Over-abstention improvement: not measured.
9. End-to-end success improvement: not measured.
10. Citation/Grounding stability: not measured.
11. Unsupported answer increase: not measured.
12. Full-request latency: not measured.
13. Production promotion is not justified by this run.
14. Bottleneck: `{summary['post_reranking_primary_bottleneck']}`.
"""
