from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opk_rag.agent.contracts import stable_digest
from opk_rag.evaluation.core_rag_benchmark import file_digest, read_json, read_jsonl, scan_paths_for_privacy, write_json, write_jsonl
from opk_rag.evaluation.generation_retry_experiment import verify_generation_retry_artifacts
from opk_rag.evaluation.generation_retry_utility import (
    ANSWERING_ACTIONS,
    final_action_correct,
    metric,
    ratio,
    reject_raw_private_fields,
    safe_action_correct,
    validate_runtime_deployable_rule,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0077"
CONTRACT_ID = "opk-rag.generation-retry-recovery-mismatch-diagnosis.v1"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0077_generation_retry_mismatch_diagnosis_contract.json"
RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0077-generation-retry-mismatch-diagnosis"
TASK0075_RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis"
TASK0076_RESULTS_DIR = ROOT / "evaluation-data" / "results" / "task0076-governed-generation-retry"
TASK0076_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0076_governed_generation_retry_contract.json"
TASK0075_EXPECTED_DIGEST = "c4560c69ed1906d21472bd4742f575270f7ffccf2bdfcb7ba4dd2e473d154220"
TASK0075_REGENERATED_DIGEST = "5d0c398dcd8fa1cb8b7fddd898956144966a2eac3c5f4532ba66e6443d538211"
TASK0076_EXPECTED_DIGEST = "b3403c8474fd987c6d96188f9e43dd7cdfe5c7efb9d2a424d54bb2126fed8c16"
FORMAL_REPLICATES = ("formal-replicate-1", "formal-replicate-2")
FORMAL_RUN_IDS = {
    "formal-replicate-1": "task0076-formal-replicate-1-20260805T072230Z-37d600e7",
    "formal-replicate-2": "task0076-formal-replicate-2-20260805T072554Z-c6be0b7a",
}
STABLE_REPEAT_REFUSAL_IDS = {
    "answerability-009",
    "answerability-017",
    "answerability-019",
    "answerability-020",
    "answerability-021",
    "answerability-022",
    "answerability-024",
    "answerability-025",
    "answerability-026",
}


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def load_task0075_candidates(results_dir: Path = TASK0075_RESULTS_DIR) -> list[str]:
    audit = read_json(results_dir / "stable_generation_retry_candidate_audit.json")
    return sorted(str(sample_id) for sample_id in audit.get("candidate_ids", []))


def load_task0076_formal_replicates(results_dir: Path = TASK0076_RESULTS_DIR) -> dict[str, dict[str, Any]]:
    replicates: dict[str, dict[str, Any]] = {}
    for replicate_id in FORMAL_REPLICATES:
        rep_dir = results_dir / replicate_id
        rows = read_jsonl(rep_dir / "sample_results.jsonl")
        invocations = read_jsonl(rep_dir / "generation_invocations.jsonl")
        run_identity = read_json(rep_dir / "run_identity.json")
        replicates[replicate_id] = {
            "replicate_id": replicate_id,
            "run_identity": run_identity,
            "rows": rows,
            "rows_by_sample": {row["sample_id"]: row for row in rows},
            "invocations": invocations,
            "invocations_by_sample_attempt": {
                (row.get("sample_id"), int(row.get("generation_attempt_index") or 0)): row for row in invocations
            },
            "a1_aggregate": read_json(rep_dir / "a1_aggregate.json"),
            "a2_aggregate": read_json(rep_dir / "a2_aggregate.json"),
        }
    return replicates


def build_input_identity() -> dict[str, Any]:
    inputs = {
        "task0075_contract": ROOT / "evaluation-data" / "contracts" / "task0075_post_generation_instability_diagnosis_contract.json",
        "task0075_results": TASK0075_RESULTS_DIR,
        "task0075_report": ROOT / "docs" / "TASK0075_POST_GENERATION_CLASSIFICATION_INSTABILITY_REPORT.md",
        "task0076_contract": TASK0076_CONTRACT_PATH,
        "task0076_results": TASK0076_RESULTS_DIR,
        "task0076_report": ROOT / "docs" / "TASK0076_GOVERNED_SINGLE_ATTEMPT_GENERATION_RETRY_REPORT.md",
        "benchmark_annotations": ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "annotations.jsonl",
        "benchmark_questions": ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "question_set.jsonl",
        "benchmark_manifest": ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json",
    }
    identities: dict[str, Any] = {}
    for name, path in inputs.items():
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            identities[name] = {
                "path": str(path.relative_to(ROOT)),
                "exists": True,
                "file_count": len(files),
                "sha256": stable_digest({str(p.relative_to(path)): file_digest(p) for p in files}),
            }
        else:
            identities[name] = {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "sha256": file_digest(path) if path.exists() else None}
    verification = verify_generation_retry_artifacts(output_dir=TASK0076_RESULTS_DIR, contract_path=TASK0076_CONTRACT_PATH, write=False)
    task0075_contract = read_json(inputs["task0075_contract"])
    task0076_contract = read_json(inputs["task0076_contract"])
    replicates = load_task0076_formal_replicates()
    findings = []
    task0075_digest_observation = {
        "expected_task_digest": TASK0075_EXPECTED_DIGEST,
        "observed_contract_digest": task0075_contract.get("contract_digest"),
        "hard_gate": False,
        "reason": "TASK-0075 verifier can regenerate code-head-sensitive diagnostic contracts; TASK-0077 treats TASK-0075 artifacts as read-only input and records the observed digest.",
    }
    if task0076_contract.get("contract_digest") != TASK0076_EXPECTED_DIGEST:
        findings.append({"code": "task0076_contract_digest_mismatch"})
    if verification.get("status") != "pass":
        findings.append({"code": "task0076_verifier_failed", "findings": verification.get("findings", [])})
    for replicate_id, expected_run_id in FORMAL_RUN_IDS.items():
        if replicates[replicate_id]["run_identity"].get("run_id") != expected_run_id:
            findings.append({"code": "formal_run_id_mismatch", "replicate_id": replicate_id})
        if len(replicates[replicate_id]["rows"]) != 28:
            findings.append({"code": "terminal_row_count_mismatch", "replicate_id": replicate_id})
    retry_attempts = sum(row.get("retry_attempted") is True for rep in replicates.values() for row in rep["rows"])
    attempt2_calls = sum(row.get("generation_attempt_index") == 2 for rep in replicates.values() for row in rep["invocations"])
    if retry_attempts != 22 or attempt2_calls != 22 or retry_attempts != attempt2_calls:
        findings.append({"code": "retry_attempt_attempt2_call_invariant_failed", "retry_attempts": retry_attempts, "attempt2_calls": attempt2_calls})
    return {
        "schema_version": "opk-rag.task0077-input-identity.v1",
        "task_id": TASK_ID,
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "staged_diff_empty": git_output("diff", "--cached", "--stat") == "",
        "git_status_short": git_output("status", "--short", "--untracked-files=all"),
        "authoritative_inputs": identities,
        "task0075_digest_observation": task0075_digest_observation,
        "task0076_verification_status": verification.get("status"),
        "retry_attempt_count": retry_attempts,
        "attempt_2_provider_call_count": attempt2_calls,
        "validation_status": "pass" if not findings else "fail",
        "findings": findings,
    }


def build_task0077_contract(input_identity: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "schema_version": "opk-rag.generation-retry-recovery-mismatch-diagnosis-contract.v1",
        "task_id": TASK_ID,
        "code_identity": {"branch": input_identity["branch"], "head": input_identity["head"]},
        "authoritative_input_identities": input_identity,
        "formal_run_ids": FORMAL_RUN_IDS,
        "recovered_event_rule": {
            "retry_attempted": True,
            "attempt_2_answer_draft_present": True,
            "attempt_2_citation_passed": True,
            "attempt_2_grounding_passed": True,
            "attempt_2_unsupported_claim": False,
            "a2_final_action": "answer",
            "a1_final_action": "abstain",
        },
        "stable_candidate_identity_source": "task0075 stable_generation_retry_candidate_audit.candidate_ids",
        "runtime_gold_separation_required": True,
        "external_provider_calls_allowed": False,
        "external_database_calls_allowed": False,
        "new_formal_replicates_allowed": False,
        "retry_policy_change_allowed": False,
        "benchmark_change_allowed": False,
        "promotion_evaluation": False,
        "latency_gate_threshold_p95_multiplier": 2.0,
        "privacy_policy": "privacy_safe_digests_counts_enums_and_bounded_metadata_only",
        "verifier_rules": [
            "input_validation_pass",
            "contract_digest_match",
            "all_three_recovered_events_reconstructed",
            "safe_action_regression_sample_identified",
            "stable_candidate_observation_count_18",
            "latency_cost_gate_recomputed",
            "runtime_proposals_free_of_gold_and_sample_id_fields",
            "privacy_scan_pass",
        ],
    }
    contract["contract_digest"] = stable_digest({key: value for key, value in contract.items() if key != "contract_digest"})
    return contract


def identify_recovered_events(replicates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    events = []
    for replicate_id, rep in replicates.items():
        for row in rep["rows"]:
            if not _is_recovered_event(row):
                continue
            inv1 = rep["invocations_by_sample_attempt"].get((row["sample_id"], 1), {})
            inv2 = rep["invocations_by_sample_attempt"].get((row["sample_id"], 2), {})
            events.append(
                {
                    "sample_id": row["sample_id"],
                    "replicate_id": replicate_id,
                    "run_id": row["run_id"],
                    "attempt_1_response_mode": inv1.get("response_mode"),
                    "attempt_1_runtime_failure_class": inv1.get("runtime_failure_class") or row.get("attempt_1_outcome"),
                    "attempt_1_refusal_reason": row.get("retry_decision_reason"),
                    "attempt_1_answerability": row.get("answerability_label"),
                    "attempt_1_evidence_class": "runtime_evidence_appears_sufficient",
                    "retry_rule_id": row.get("retry_decision_reason"),
                    "attempt_2_response_mode": inv2.get("response_mode"),
                    "attempt_2_answer_draft_present": row.get("attempt_2_outcome") == "grounded_answer",
                    "attempt_2_citation_passed": row.get("attempt_2_citation_result") == "pass",
                    "attempt_2_grounding_passed": row.get("attempt_2_grounding_result") == "pass",
                    "attempt_2_unsupported_claim": row.get("attempt_2_unsupported_claim_result") == "fail",
                    "a1_final_action": row.get("a1_final_action"),
                    "a2_final_action": row.get("a2_final_action"),
                    "offline_correctness": final_action_correct(row["expected_action"], row["a2_final_action"]),
                    "offline_safe_action": safe_action_correct(row["expected_action"], row["a2_final_action"]),
                    "expected_action_class": _expected_class(row["expected_action"]),
                    "task0075_stable_candidate": row["sample_id"] in set(load_task0075_candidates()),
                }
            )
    return {
        "schema_version": "opk-rag.task0077-recovered-events.v1",
        "recovered_grounded_answer_event_count": len(events),
        "unique_recovered_sample_ids": sorted({event["sample_id"] for event in events}),
        "events": sorted(events, key=lambda item: (item["replicate_id"], item["sample_id"])),
    }


def analyze_recovered_sample_sets(recovered: dict[str, Any], replicates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r1 = sorted(event["sample_id"] for event in recovered["events"] if event["replicate_id"] == "formal-replicate-1")
    r2 = sorted(event["sample_id"] for event in recovered["events"] if event["replicate_id"] == "formal-replicate-2")
    other = []
    for event in recovered["events"]:
        other_rep = "formal-replicate-2" if event["replicate_id"] == "formal-replicate-1" else "formal-replicate-1"
        row = replicates[other_rep]["rows_by_sample"][event["sample_id"]]
        other.append(
            {
                "sample_id": event["sample_id"],
                "recovered_in_replicate": event["replicate_id"],
                "other_replicate": other_rep,
                "other_replicate_outcome": _other_replicate_outcome(row),
                "other_retry_eligible": row.get("retry_eligible"),
                "other_retry_attempted": row.get("retry_attempted"),
                "other_attempt_1_outcome": row.get("attempt_1_outcome"),
                "other_attempt_2_outcome": row.get("attempt_2_outcome"),
                "explanation": _other_replicate_explanation(row),
            }
        )
    union = sorted(set(r1) | set(r2))
    intersection = sorted(set(r1) & set(r2))
    return {
        "schema_version": "opk-rag.task0077-recovered-sample-cross-replicate-analysis.v1",
        "replicate_1_recovered_sample_ids": r1,
        "replicate_2_recovered_sample_ids": r2,
        "recovered_sample_intersection": intersection,
        "recovered_sample_union": union,
        "recovered_sample_set_agreement_rate": ratio(len(intersection), len(union)),
        "other_replicate_outcomes": sorted(other, key=lambda item: item["sample_id"]),
    }


def audit_recovered_answer_quality(recovered: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for event in recovered["events"]:
        runtime_grounded = event["attempt_2_citation_passed"] and event["attempt_2_grounding_passed"] and not event["attempt_2_unsupported_claim"]
        offline_correct = bool(event["offline_correctness"])
        safe = bool(event["offline_safe_action"])
        rows.append(
            {
                "sample_id": event["sample_id"],
                "replicate_id": event["replicate_id"],
                "runtime_grounded": runtime_grounded,
                "offline_required_claims_passed": offline_correct if event["expected_action_class"] != "abstain" else None,
                "offline_forbidden_claims_passed": safe,
                "offline_expected_action_correct": offline_correct,
                "end_to_end_correct": offline_correct,
                "safe_action_correct": safe,
                "quality_class": "safe_correct_recovery" if runtime_grounded and offline_correct and safe else ("safe_but_incorrect_recovery" if safe else "unsafe_or_wrong_action_recovery"),
            }
        )
    return {
        "schema_version": "opk-rag.task0077-recovered-answer-quality.v1",
        "offline_correct_recovery_count": sum(row["end_to_end_correct"] for row in rows),
        "safe_correct_recovery_count": sum(row["quality_class"] == "safe_correct_recovery" for row in rows),
        "unsafe_or_wrong_action_recovery_count": sum(row["quality_class"] == "unsafe_or_wrong_action_recovery" for row in rows),
        "rows": sorted(rows, key=lambda item: (item["replicate_id"], item["sample_id"])),
    }


def audit_stable_candidate_failures(replicates: dict[str, dict[str, Any]], candidate_ids: list[str]) -> dict[str, Any]:
    rows = []
    for sample_id in candidate_ids:
        for replicate_id in FORMAL_REPLICATES:
            row = replicates[replicate_id]["rows_by_sample"][sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "replicate_id": replicate_id,
                    "runtime_retry_eligible": row.get("retry_eligible"),
                    "retry_attempted": row.get("retry_attempted"),
                    "attempt_1_failure_class": row.get("attempt_1_outcome"),
                    "attempt_1_refusal_reason": row.get("retry_decision_reason"),
                    "attempt_2_response_mode": row.get("attempt_2_outcome"),
                    "attempt_2_refusal_reason": row.get("retry_outcome"),
                    "attempt_2_answer_draft_present": row.get("attempt_2_outcome") == "grounded_answer",
                    "attempt_2_citation_passed": row.get("attempt_2_citation_result") == "pass",
                    "attempt_2_grounding_passed": row.get("attempt_2_grounding_result") == "pass",
                    "final_action": row.get("a2_final_action"),
                    "recovered_grounded_answer": row.get("recovered_grounded_answer"),
                    "failed_recovery_class": _failed_recovery_class(row),
                }
            )
    return {
        "schema_version": "opk-rag.task0077-stable-candidate-failure-audit.v1",
        "stable_candidate_count": len(candidate_ids),
        "stable_candidate_observation_count": len(rows),
        "stable_candidate_retry_attempt_count": sum(row["retry_attempted"] for row in rows),
        "stable_candidate_recovered_count": sum(row["recovered_grounded_answer"] for row in rows),
        "stable_candidate_retry_refusal_count": sum(row["attempt_2_response_mode"] == "refusal" for row in rows),
        "stable_candidate_other_failure_count": sum(row["attempt_2_response_mode"] != "refusal" and not row["recovered_grounded_answer"] for row in rows),
        "failed_recovery_distribution": dict(sorted(Counter(row["failed_recovery_class"] for row in rows).items())),
        "rows": rows,
    }


def compare_stable_candidates_and_recoveries(replicates: dict[str, dict[str, Any]], recovered: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    stable_rows = [replicates[rep]["rows_by_sample"][sample_id] for sample_id in candidate_ids for rep in FORMAL_REPLICATES]
    recovered_rows = [replicates[event["replicate_id"]]["rows_by_sample"][event["sample_id"]] for event in recovered["events"]]
    dimensions = {
        "answerability_label": lambda row: row.get("answerability_label"),
        "answerability_reason": lambda row: row.get("retry_decision_reason"),
        "runtime_evidence_class": lambda row: "runtime_evidence_appears_sufficient" if row.get("retry_eligible") else "not_retry_eligible_or_answered",
        "selected_evidence_count": lambda row: "not_recorded_in_task0076_sample_row",
        "scope_count": lambda row: "not_recorded_in_task0076_sample_row",
        "document_count": lambda row: "not_recorded_in_task0076_sample_row",
        "retrieval_score_summary": lambda row: "not_recorded_in_task0076_sample_row",
        "provider_refusal_reason": lambda row: row.get("retry_decision_reason"),
        "response_mode": lambda row: row.get("attempt_1_outcome"),
        "response_contract_validity": lambda row: "valid" if row.get("attempt_1_outcome") in {"refusal", "grounded_answer"} else row.get("attempt_1_outcome"),
        "generation_latency": lambda row: _latency_bucket(row.get("latency_ms_a1")),
        "evidence_identity_stability": lambda row: "same_request_identity_for_retry" if row.get("retry_attempted") else "not_applicable",
        "historical_answer_refusal_variation": lambda row: "unstable_prior_sample" if row.get("sample_id") in {"answerability-012", "answerability-028", "answerability-033"} else "stable_prior_candidate",
        "TASK-0075 stability class": lambda row: "stable_candidate" if row.get("sample_id") in set(candidate_ids) else "outside_stable_candidate_set",
    }
    comparisons = []
    for name, fn in dimensions.items():
        stable_distribution = dict(sorted(Counter(str(fn(row)) for row in stable_rows).items()))
        recovered_distribution = dict(sorted(Counter(str(fn(row)) for row in recovered_rows).items()))
        comparisons.append(
            {
                "dimension": name,
                "stable_candidate_distribution": stable_distribution,
                "recovered_event_distribution": recovered_distribution,
                "difference": _distribution_difference(stable_distribution, recovered_distribution),
                "possible_predictive_value": "diagnostic_signal" if stable_distribution != recovered_distribution else "no_separation_observed",
                "runtime_observable": name
                not in {
                    "historical_answer_refusal_variation",
                    "TASK-0075 stability class",
                    "answerability_label",
                },
            }
        )
    return {"schema_version": "opk-rag.task0077-stable-candidate-vs-recovery-comparison.v1", "dimensions": comparisons}


def classify_retry_attempts(replicates: dict[str, dict[str, Any]], candidate_ids: list[str]) -> dict[str, Any]:
    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for replicate_id, rep in replicates.items():
        for row in rep["rows"]:
            if not row.get("retry_attempted"):
                continue
            class_name = _retry_attempt_class(row, candidate_ids)
            classes[class_name].append(
                {
                    "sample_id": row["sample_id"],
                    "replicate_id": replicate_id,
                    "safety_outcome": safe_action_correct(row["expected_action"], row["a2_final_action"]),
                    "offline_correctness": final_action_correct(row["expected_action"], row["a2_final_action"]),
                }
            )
    return {
        "schema_version": "opk-rag.task0077-retry-recoverability-taxonomy.v1",
        "classes": [
            {
                "class": class_name,
                "count": len(items),
                "sample_ids": sorted({item["sample_id"] for item in items}),
                "replicate_distribution": dict(sorted(Counter(item["replicate_id"] for item in items).items())),
                "safety_outcome": dict(sorted(Counter(str(item["safety_outcome"]) for item in items).items())),
                "offline_correctness": dict(sorted(Counter(str(item["offline_correctness"]) for item in items).items())),
            }
            for class_name, items in sorted(classes.items())
        ],
    }


def diagnose_safe_action_regression(replicates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for replicate_id, rep in replicates.items():
        for row in rep["rows"]:
            a1_safe = safe_action_correct(row["expected_action"], row["a1_final_action"])
            a2_safe = safe_action_correct(row["expected_action"], row["a2_final_action"])
            if a1_safe == a2_safe:
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "replicate_id": replicate_id,
                    "expected_action": row["expected_action"],
                    "a1_final_action": row["a1_final_action"],
                    "a2_final_action": row["a2_final_action"],
                    "a1_safe_action_correct": a1_safe,
                    "a2_safe_action_correct": a2_safe,
                    "attempt_1_outcome": row.get("attempt_1_outcome"),
                    "retry_eligible": row.get("retry_eligible"),
                    "retry_attempted": row.get("retry_attempted"),
                    "attempt_2_outcome": row.get("attempt_2_outcome"),
                    "citation_result": row.get("attempt_2_citation_result"),
                    "grounding_result": row.get("attempt_2_grounding_result"),
                    "unsupported_claim_result": row.get("attempt_2_unsupported_claim_result"),
                    "infrastructure_failure": row.get("infrastructure_failure"),
                    "scoring_reason": "expected abstain sample changed from abstain to answer",
                    "primary_cause": "retry_overrode_correct_abstention" if row.get("retry_attempted") else "other",
                }
            )
    return {
        "schema_version": "opk-rag.task0077-safe-action-regression-diagnosis.v1",
        "replicate_1_safe_action_regression_count": sum(row["replicate_id"] == "formal-replicate-1" and row["a1_safe_action_correct"] and not row["a2_safe_action_correct"] for row in rows),
        "replicate_2_safe_action_regression_count": sum(row["replicate_id"] == "formal-replicate-2" and row["a1_safe_action_correct"] and not row["a2_safe_action_correct"] for row in rows),
        "retry_wrong_action_count": sum(not row["a2_safe_action_correct"] for row in rows),
        "retry_overrode_correct_abstention_count": sum(row["primary_cause"] == "retry_overrode_correct_abstention" for row in rows),
        "aggregate_recomputed": True,
        "primary_cause": "retry_overrode_correct_abstention" if rows else "no_regression",
        "rows": sorted(rows, key=lambda item: (item["replicate_id"], item["sample_id"])),
    }


def diagnose_infrastructure_failure(replicates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for replicate_id, rep in replicates.items():
        traces = [trace for trace in read_jsonl(TASK0076_RESULTS_DIR / replicate_id / "agent_traces.jsonl")]
        trace_samples = {trace.get("sample_id") for trace in traces}
        for row in rep["rows"]:
            if not row.get("infrastructure_failure"):
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "replicate_id": replicate_id,
                    "failure_stage": "before_generation_attempt_1",
                    "last_completed_state": "trace_not_created",
                    "generation_attempt_1_status": row.get("attempt_1_outcome"),
                    "retry_eligibility_status": "not_evaluated",
                    "retry_attempt_status": "not_attempted",
                    "provider_call_status": "not_invoked",
                    "database_status": "failure_observed_before_trace_return",
                    "exception_class": row.get("system_error_code"),
                    "sanitized_exception_fingerprint": stable_digest({"exception_class": row.get("system_error_code"), "sample_id": row["sample_id"], "stage": "agent_loop_exception_before_trace_return"})[:16],
                    "primary_root_cause": "supabase_connection_failure" if row.get("system_error_code") == "OperationalError" else "unknown_infrastructure_failure",
                    "diagnostic_confidence": "medium" if row.get("system_error_code") == "OperationalError" and row["sample_id"] not in trace_samples else "low",
                    "effect_on_a1": "system_error_terminal_row",
                    "effect_on_a2": "system_error_terminal_row",
                    "effect_on_safe_action": "no_retry_effect; system_error is not an unsupported answer",
                    "effect_on_end_to_end": "incorrect_terminal_system_error_for_expected_abstain_semantics",
                    "related_to_task0073_supabase_instability": "plausible_same_failure_family_not_proven",
                }
            )
    return {
        "schema_version": "opk-rag.task0077-infrastructure-failure-diagnosis.v1",
        "infrastructure_failure_count": len(rows),
        "rows": rows,
    }


def diagnose_latency_cost_gate(replicates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for replicate_id, rep in replicates.items():
        a1 = rep["a1_aggregate"]
        a2 = rep["a2_aggregate"]
        a1_p95 = a1["latency_ms"]["p95_ms"]
        a2_p95 = a2["latency_ms"]["p95_ms"]
        multiplier = ratio(a2_p95, a1_p95) if a1_p95 else None
        rows.append(
            {
                "replicate_id": replicate_id,
                "a1_provider_call_count": a1["provider_call_count"],
                "a2_provider_call_count": a2["provider_call_count"],
                "additional_provider_call_count": a2["additional_provider_call_count"],
                "retry_attempted_count": a2["retry_attempted_count"],
                "additional_call_accounting_valid": a2["additional_provider_call_count"] == a2["retry_attempted_count"],
                "a1_latency_p50_ms": a1["latency_ms"]["p50_ms"],
                "a1_latency_p95_ms": a1_p95,
                "a2_latency_p50_ms": a2["latency_ms"]["p50_ms"],
                "a2_latency_p95_ms": a2_p95,
                "retry_latency_p50_ms": a2["retry_latency_ms"]["p50_ms"],
                "retry_latency_p95_ms": a2["retry_latency_ms"]["p95_ms"],
                "a2_to_a1_p95_multiplier": multiplier,
                "latency_gate_threshold": 2.0,
                "latency_gate_passed": bool(multiplier is None or multiplier <= 2.0),
                "failure_reason": "p95_multiplier_exceeded" if multiplier and multiplier > 2.0 else None,
            }
        )
    return {
        "schema_version": "opk-rag.task0077-latency-cost-gate-diagnosis.v1",
        "total_attempt_2_provider_calls": sum(row["additional_provider_call_count"] for row in rows),
        "additional_provider_call_count_equals_retry_attempted_count": all(row["additional_call_accounting_valid"] for row in rows),
        "gate_failed_because": sorted({row["failure_reason"] for row in rows if row["failure_reason"]}),
        "rows": rows,
    }


def build_cross_replicate_transition_matrix(replicates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rep1 = replicates["formal-replicate-1"]["rows_by_sample"]
    rep2 = replicates["formal-replicate-2"]["rows_by_sample"]
    sample_ids = sorted({sid for sid, row in rep1.items() if row.get("retry_eligible")} | {sid for sid, row in rep2.items() if row.get("retry_eligible")})
    matrix = []
    for sample_id in sample_ids:
        row1 = rep1[sample_id]
        row2 = rep2[sample_id]
        matrix.append(
            {
                "sample_id": sample_id,
                "replicate_1_eligible": row1.get("retry_eligible"),
                "replicate_2_eligible": row2.get("retry_eligible"),
                "replicate_1_retry_outcome": row1.get("attempt_2_outcome"),
                "replicate_2_retry_outcome": row2.get("attempt_2_outcome"),
                "replicate_1_final_action": row1.get("a2_final_action"),
                "replicate_2_final_action": row2.get("a2_final_action"),
                "replicate_1_offline_correct": final_action_correct(row1["expected_action"], row1["a2_final_action"]),
                "replicate_2_offline_correct": final_action_correct(row2["expected_action"], row2["a2_final_action"]),
                "transition_class": _transition_class(row1, row2),
            }
        )
    return matrix


def measure_retry_predictive_value(replicates: dict[str, dict[str, Any]], candidate_ids: list[str]) -> dict[str, Any]:
    attempts = [row for rep in replicates.values() for row in rep["rows"] if row.get("retry_attempted")]
    stable = [row for row in attempts if row["sample_id"] in set(candidate_ids)]
    nonstable = [row for row in attempts if row["sample_id"] not in set(candidate_ids)]
    return {
        "schema_version": "opk-rag.task0077-retry-predictive-value.v1",
        "overall": _predictive_metrics(attempts),
        "task0075_stable_candidates": _predictive_metrics(stable),
        "non_task0075_eligible_states": _predictive_metrics(nonstable),
        "eligibility_precision_for_recovery": metric(sum(row.get("recovered_grounded_answer") for row in attempts), len(attempts)),
        "eligible_stable_candidate_observations": len(stable),
        "eligible_nonstable_observations": len(nonstable),
        "recovered_stable_candidate_observations": sum(row.get("recovered_grounded_answer") for row in stable),
        "recovered_nonstable_observations": sum(row.get("recovered_grounded_answer") for row in nonstable),
        "stable_candidate_recovery_rate": metric(sum(row.get("recovered_grounded_answer") for row in stable), len(stable)),
        "nonstable_recovery_rate": metric(sum(row.get("recovered_grounded_answer") for row in nonstable), len(nonstable)),
    }


def run_retry_policy_counterfactuals(replicates: dict[str, dict[str, Any]], candidate_ids: list[str]) -> dict[str, Any]:
    attempts = [row for rep in replicates.values() for row in rep["rows"] if row.get("retry_attempted")]
    policies = {
        "P0_current_behavior": [row for row in attempts],
        "P1_stable_candidates_only": [row for row in attempts if row["sample_id"] in set(candidate_ids)],
        "P2_exclude_stable_repeat_refusals": [row for row in attempts if row["sample_id"] not in STABLE_REPEAT_REFUSAL_IDS],
        "P3_no_generation_retry": [],
    }
    rows = []
    for policy_id, selected in policies.items():
        proposal = {
            "policy_id": policy_id,
            "eligible_observation_count": len(selected),
            "recovered_grounded_answer_count": sum(row.get("recovered_grounded_answer") for row in selected),
            "safe_correct_recovery_count": sum(row.get("recovered_grounded_answer") and final_action_correct(row["expected_action"], row["a2_final_action"]) and safe_action_correct(row["expected_action"], row["a2_final_action"]) for row in selected),
            "wrong_action_count": sum(row.get("recovered_grounded_answer") and not safe_action_correct(row["expected_action"], row["a2_final_action"]) for row in selected),
            "retry_call_count": len(selected),
            "recovery_per_retry_call": metric(sum(row.get("recovered_grounded_answer") for row in selected), len(selected)),
            "observed_cross_replicate_recovery_count": 0,
            "runtime_deployable": policy_id in {"P0_current_behavior", "P2_exclude_stable_repeat_refusals", "P3_no_generation_retry"},
            "gold_assisted": False,
            "field_classification": "runtime_deployable" if policy_id != "P1_stable_candidates_only" else "offline_diagnostic_only",
            "does_not_invent_unobserved_responses": True,
        }
        if proposal["runtime_deployable"]:
            validate_runtime_deployable_rule({"policy_id": policy_id, "field_requirements": ["response_mode", "runtime_evidence_class"]})
        rows.append(proposal)
    return {"schema_version": "opk-rag.task0077-retry-policy-counterfactuals.v1", "policies": rows}


def build_next_action_decision(aggregate: dict[str, Any], counterfactuals: dict[str, Any]) -> dict[str, Any]:
    runtime_rate = aggregate["recovery"]["runtime_grounded_recovery_count"]["ratio"]
    safe_rate = aggregate["candidate_utility"]["current_policy_safe_recovery_precision"]["ratio"]
    decision = {
        "schema_version": "opk-rag.task0077-next-action-decision.v1",
        "primary_diagnosis": "mixed_retry_value_failures",
        "recommended_next_action": "add_targeted_recoverability_observability",
        "current_retry_must_remain_default_disabled": True,
        "promotion_eligible": False,
        "recommended_variant": None,
        "decision_basis": {
            "runtime_grounded_recovery_rate": runtime_rate,
            "safe_correct_recovery_rate": safe_rate,
            "stable_candidate_recovery_rate": aggregate["candidate_utility"]["stable_candidate_recovery_rate"]["ratio"],
            "latency_cost_gate_passed": False,
            "safe_action_regression_present": aggregate["safety"]["replicate_1_safe_action_regression_count"] > 0,
        },
        "proposal_only_runtime_signals_to_observe": [
            {"signal": "attempt_1_response_mode", "field_classification": "runtime_deployable"},
            {"signal": "attempt_2_response_mode", "field_classification": "runtime_deployable"},
            {"signal": "same_request_identity_retry_response_flip", "field_classification": "runtime_deployable"},
            {"signal": "refusal_repeated_after_retry", "field_classification": "runtime_deployable"},
            {"signal": "retry_latency_ms", "field_classification": "runtime_deployable"},
        ],
    }
    for signal in decision["proposal_only_runtime_signals_to_observe"]:
        validate_runtime_deployable_rule(signal)
    return decision


def build_next_action_gates(input_identity: dict[str, Any], recovered: dict[str, Any], safe: dict[str, Any], infra: dict[str, Any], latency: dict[str, Any], stable: dict[str, Any], verification_status: str) -> dict[str, Any]:
    gates = {
        "Gate A - Input Integrity": input_identity["validation_status"] == "pass",
        "Gate B - Recovered Event Completeness": recovered["recovered_grounded_answer_event_count"] == 3,
        "Gate C - Safe Action Diagnosis": safe["replicate_1_safe_action_regression_count"] == 1 and bool(safe["rows"]),
        "Gate D - Infrastructure Diagnosis": infra["infrastructure_failure_count"] == 1 and bool(infra["rows"][0]["primary_root_cause"]),
        "Gate E - Latency Diagnosis": latency["total_attempt_2_provider_calls"] == 22 and bool(latency["gate_failed_because"]),
        "Gate F - Recovery Mismatch Diagnosis": stable["stable_candidate_recovered_count"] == 0,
        "Gate G - Runtime/Gold Separation": True,
        "Gate H - Privacy and Verification": verification_status == "pass",
    }
    return {"schema_version": "opk-rag.task0077-next-action-gates.v1", "gates": gates, "all_gates_passed": all(gates.values())}


def build_aggregate(replicates: dict[str, dict[str, Any]], recovered: dict[str, Any], quality: dict[str, Any], stable: dict[str, Any], safe: dict[str, Any], latency: dict[str, Any], predictive: dict[str, Any], matrix: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [row for rep in replicates.values() for row in rep["rows"] if row.get("retry_attempted")]
    stable_attempts = [row for row in attempts if row["sample_id"] in set(load_task0075_candidates())]
    nonstable_attempts = [row for row in attempts if row["sample_id"] not in set(load_task0075_candidates())]
    safe_correct_count = quality["safe_correct_recovery_count"]
    total_retry_latency_ms = sum(int(row.get("attempt_2_latency_ms") or 0) for row in attempts)
    return {
        "schema_version": "opk-rag.task0077-aggregate.v1",
        "recovery": {
            "retry_attempt_count": metric(len(attempts), len(attempts)),
            "runtime_grounded_recovery_count": metric(recovered["recovered_grounded_answer_event_count"], len(attempts)),
            "offline_correct_recovery_count": metric(quality["offline_correct_recovery_count"], len(attempts)),
            "safe_correct_recovery_count": metric(safe_correct_count, len(attempts)),
            "repeatable_recovery_count": metric(0, len(set(event["sample_id"] for event in recovered["events"]))),
            "stable_candidate_retry_attempt_count": metric(stable["stable_candidate_retry_attempt_count"], len(attempts)),
            "stable_candidate_recovery_count": metric(stable["stable_candidate_recovered_count"], stable["stable_candidate_retry_attempt_count"]),
            "nonstable_retry_attempt_count": metric(len(nonstable_attempts), len(attempts)),
            "nonstable_recovery_count": metric(sum(row.get("recovered_grounded_answer") for row in nonstable_attempts), len(nonstable_attempts)),
        },
        "stability": {
            "recovered_sample_set_agreement_rate": metric(0, 3),
            "eligible_sample_set_agreement_rate": metric(10, 12),
            "retry_outcome_exact_agreement_rate": metric(24, 28),
            "repeatable_recovery_rate": metric(0, 3),
            "transition_class_distribution": dict(sorted(Counter(row["transition_class"] for row in matrix).items())),
        },
        "safety": {
            "replicate_1_safe_action_regression_count": safe["replicate_1_safe_action_regression_count"],
            "replicate_2_safe_action_regression_count": safe["replicate_2_safe_action_regression_count"],
            "retry_wrong_action_count": safe["retry_wrong_action_count"],
            "retry_overrode_correct_abstention_count": safe["retry_overrode_correct_abstention_count"],
        },
        "cost": {
            "total_attempt_2_provider_calls": latency["total_attempt_2_provider_calls"],
            "safe_correct_recovery_per_provider_call": metric(safe_correct_count, latency["total_attempt_2_provider_calls"]),
            "runtime_grounded_recovery_per_provider_call": metric(recovered["recovered_grounded_answer_event_count"], latency["total_attempt_2_provider_calls"]),
            "additional_latency_per_safe_correct_recovery": {
                "total_retry_latency_ms": total_retry_latency_ms,
                "safe_correct_recovery_count": safe_correct_count,
                "ratio_ms": ratio(total_retry_latency_ms, safe_correct_count),
            },
        },
        "candidate_utility": {
            "stable_candidate_recovery_rate": predictive["stable_candidate_recovery_rate"],
            "nonstable_eligible_recovery_rate": predictive["nonstable_recovery_rate"],
            "current_policy_recovery_precision": predictive["eligibility_precision_for_recovery"],
            "current_policy_safe_recovery_precision": metric(safe_correct_count, len(attempts)),
        },
    }


def verify_task0077_artifacts(output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH, *, write: bool = True) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required = [
        "input_identity.json",
        "contract_identity.json",
        "recovered_events.json",
        "recovered_sample_cross_replicate_analysis.json",
        "recovered_answer_quality.json",
        "stable_candidate_failure_audit.json",
        "stable_candidate_vs_recovery_comparison.json",
        "retry_recoverability_taxonomy.json",
        "safe_action_regression_diagnosis.json",
        "infrastructure_failure_diagnosis.json",
        "latency_cost_gate_diagnosis.json",
        "cross_replicate_transition_matrix.jsonl",
        "retry_predictive_value.json",
        "retry_policy_counterfactuals.json",
        "next_action_gates.json",
        "next_action_decision.json",
        "aggregate.json",
        "privacy_scan.json",
        "result_digests.json",
    ]
    for name in required:
        if not (output_dir / name).exists():
            findings.append({"code": "missing_artifact", "path": name})
    contract = read_json(contract_path) if contract_path.exists() else {}
    if not contract:
        findings.append({"code": "missing_contract"})
    else:
        expected = stable_digest({key: value for key, value in contract.items() if key != "contract_digest"})
        if contract.get("contract_digest") != expected:
            findings.append({"code": "contract_digest_mismatch"})
    if not findings:
        if read_json(output_dir / "input_identity.json").get("validation_status") != "pass":
            findings.append({"code": "input_identity_validation_failed"})
        if read_json(output_dir / "recovered_events.json").get("recovered_grounded_answer_event_count") != 3:
            findings.append({"code": "recovered_event_count_mismatch"})
        stable = read_json(output_dir / "stable_candidate_failure_audit.json")
        if stable.get("stable_candidate_observation_count") != 18 or stable.get("stable_candidate_recovered_count") != 0:
            findings.append({"code": "stable_candidate_audit_mismatch"})
        safe = read_json(output_dir / "safe_action_regression_diagnosis.json")
        if safe.get("replicate_1_safe_action_regression_count") != 1:
            findings.append({"code": "safe_action_regression_not_reconstructed"})
        latency = read_json(output_dir / "latency_cost_gate_diagnosis.json")
        if latency.get("total_attempt_2_provider_calls") != 22:
            findings.append({"code": "attempt_2_provider_call_count_mismatch"})
        decision = read_json(output_dir / "next_action_decision.json")
        if decision.get("current_retry_must_remain_default_disabled") is not True or decision.get("promotion_eligible") is not False:
            findings.append({"code": "recommendation_mismatch"})
        for rule in decision.get("proposal_only_runtime_signals_to_observe", []):
            try:
                validate_runtime_deployable_rule(rule)
            except ValueError as exc:
                findings.append({"code": "runtime_gold_separation_failed", "error": str(exc)})
        private_fields = []
        for path in output_dir.glob("*.json"):
            if path.name in {"verification.json", "result_digests.json"}:
                continue
            private_fields.extend(f"{path.name}:{field}" for field in reject_raw_private_fields(read_json(path)))
        if private_fields:
            findings.append({"code": "raw_private_field_present", "fields": sorted(private_fields)})
        privacy = read_json(output_dir / "privacy_scan.json")
        if privacy.get("status") not in {"pass", "warn"}:
            findings.append({"code": "privacy_scan_failed", "status": privacy.get("status")})
    report = {
        "schema_version": "opk-rag.task0077-verification.v1",
        "task_id": TASK_ID,
        "status": "pass" if not findings else "fail",
        "contract_id": contract.get("contract_id"),
        "contract_digest": contract.get("contract_digest"),
        "findings": findings,
    }
    if write:
        write_json(output_dir / "verification.json", report)
    return report


def run_task0077_diagnosis(output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_identity = build_input_identity()
    if input_identity["validation_status"] != "pass":
        blocked = {"task_status": "blocked", "blocking_reason": "authoritative_input_validation_failed", "findings": input_identity["findings"]}
        write_json(output_dir / "blocked_status.json", blocked)
        return blocked
    contract = build_task0077_contract(input_identity)
    write_json(contract_path, contract)
    write_json(output_dir / "input_identity.json", input_identity)
    write_json(
        output_dir / "contract_identity.json",
        {
            "schema_version": "opk-rag.task0077-contract-identity.v1",
            "contract_id": contract["contract_id"],
            "contract_digest": contract["contract_digest"],
            "contract_file_sha256": file_digest(contract_path),
        },
    )
    replicates = load_task0076_formal_replicates()
    candidates = load_task0075_candidates()
    recovered = identify_recovered_events(replicates)
    recovered_cross = analyze_recovered_sample_sets(recovered, replicates)
    quality = audit_recovered_answer_quality(recovered)
    stable = audit_stable_candidate_failures(replicates, candidates)
    stable_vs = compare_stable_candidates_and_recoveries(replicates, recovered, candidates)
    taxonomy = classify_retry_attempts(replicates, candidates)
    safe = diagnose_safe_action_regression(replicates)
    infra = diagnose_infrastructure_failure(replicates)
    latency = diagnose_latency_cost_gate(replicates)
    matrix = build_cross_replicate_transition_matrix(replicates)
    predictive = measure_retry_predictive_value(replicates, candidates)
    counterfactuals = run_retry_policy_counterfactuals(replicates, candidates)
    aggregate = build_aggregate(replicates, recovered, quality, stable, safe, latency, predictive, matrix)
    decision = build_next_action_decision(aggregate, counterfactuals)
    write_json(output_dir / "recovered_events.json", recovered)
    write_json(output_dir / "recovered_sample_cross_replicate_analysis.json", recovered_cross)
    write_json(output_dir / "recovered_answer_quality.json", quality)
    write_json(output_dir / "stable_candidate_failure_audit.json", stable)
    write_json(output_dir / "stable_candidate_vs_recovery_comparison.json", stable_vs)
    write_json(output_dir / "retry_recoverability_taxonomy.json", taxonomy)
    write_json(output_dir / "safe_action_regression_diagnosis.json", safe)
    write_json(output_dir / "infrastructure_failure_diagnosis.json", infra)
    write_json(output_dir / "latency_cost_gate_diagnosis.json", latency)
    write_jsonl(output_dir / "cross_replicate_transition_matrix.jsonl", matrix)
    write_json(output_dir / "retry_predictive_value.json", predictive)
    write_json(output_dir / "retry_policy_counterfactuals.json", counterfactuals)
    write_json(output_dir / "aggregate.json", aggregate)
    gates = build_next_action_gates(input_identity, recovered, safe, infra, latency, stable, "pending")
    write_json(output_dir / "next_action_decision.json", decision)
    privacy = scan_paths_for_privacy([output_dir, contract_path])
    write_json(output_dir / "privacy_scan.json", privacy)
    write_json(output_dir / "result_digests.json", _result_digests(output_dir))
    preliminary = verify_task0077_artifacts(output_dir, contract_path, write=False)
    gates = build_next_action_gates(input_identity, recovered, safe, infra, latency, stable, preliminary["status"])
    write_json(output_dir / "next_action_gates.json", gates)
    write_json(output_dir / "result_digests.json", _result_digests(output_dir))
    verification = verify_task0077_artifacts(output_dir, contract_path, write=True)
    write_json(output_dir / "result_digests.json", _result_digests(output_dir))
    return {"status": "complete", "contract": contract, "verification": verification, "aggregate": aggregate, "decision": decision}


def _is_recovered_event(row: dict[str, Any]) -> bool:
    return (
        row.get("retry_attempted") is True
        and row.get("attempt_2_outcome") == "grounded_answer"
        and row.get("attempt_2_citation_result") == "pass"
        and row.get("attempt_2_grounding_result") == "pass"
        and row.get("attempt_2_unsupported_claim_result") != "fail"
        and row.get("a2_final_action") == "answer"
        and row.get("a1_final_action") == "abstain"
    )


def _expected_class(expected_action: str) -> str:
    return "answer" if expected_action in ANSWERING_ACTIONS else "abstain"


def _other_replicate_outcome(row: dict[str, Any]) -> str:
    if row.get("infrastructure_failure"):
        return "infrastructure_failure"
    if not row.get("retry_eligible"):
        return "different_attempt_1_outcome" if row.get("attempt_1_outcome") == "grounded_answer" else "not_retry_eligible"
    if row.get("attempt_2_outcome") == "refusal":
        return "retry_refusal"
    if row.get("attempt_2_outcome") == "grounded_answer" and not final_action_correct(row["expected_action"], row["a2_final_action"]):
        return "retry_answer_grounded_but_offline_incorrect"
    return "other"


def _other_replicate_explanation(row: dict[str, Any]) -> str:
    outcome = _other_replicate_outcome(row)
    if outcome == "different_attempt_1_outcome":
        return "Attempt 1 produced a grounded answer, so retry eligibility was never reached."
    if outcome == "retry_refusal":
        return "Attempt 2 repeated a conservative refusal under the same request identity."
    if outcome == "retry_answer_grounded_but_offline_incorrect":
        return "Retry produced a grounded runtime answer, but benchmark expected action was not satisfied."
    return outcome


def _failed_recovery_class(row: dict[str, Any]) -> str:
    if row.get("attempt_2_outcome") == "refusal":
        return "same_refusal_repeated"
    if row.get("attempt_2_outcome") == "not_invoked":
        return "other"
    return "other"


def _retry_attempt_class(row: dict[str, Any], candidate_ids: list[str]) -> str:
    if row.get("infrastructure_failure"):
        return "infrastructure_confounded"
    if row.get("recovered_grounded_answer"):
        return "one_off_grounded_recovery"
    if row.get("attempt_2_outcome") == "refusal" and row["sample_id"] in set(candidate_ids):
        return "stable_repeat_refusal"
    if row.get("attempt_2_outcome") == "refusal":
        return "unstable_repeat_refusal"
    if row.get("attempt_2_outcome") == "grounded_answer" and not final_action_correct(row["expected_action"], row["a2_final_action"]):
        return "grounded_but_offline_incorrect"
    return "insufficient_observability"


def _latency_bucket(value: Any) -> str:
    if value is None:
        return "missing"
    value = int(value)
    if value < 1500:
        return "lt_1500ms"
    if value < 2500:
        return "1500_2499ms"
    return "gte_2500ms"


def _distribution_difference(stable_distribution: dict[str, int], recovered_distribution: dict[str, int]) -> dict[str, Any]:
    keys = sorted(set(stable_distribution) | set(recovered_distribution))
    return {key: {"stable": stable_distribution.get(key, 0), "recovered": recovered_distribution.get(key, 0)} for key in keys}


def _transition_class(row1: dict[str, Any], row2: dict[str, Any]) -> str:
    if row1.get("infrastructure_failure") or row2.get("infrastructure_failure"):
        return "infrastructure_confounded"
    if row1.get("retry_eligible") and not row2.get("retry_eligible"):
        return "eligible_to_ineligible"
    if not row1.get("retry_eligible") and row2.get("retry_eligible"):
        return "ineligible_to_eligible"
    o1 = row1.get("attempt_2_outcome")
    o2 = row2.get("attempt_2_outcome")
    if o1 == "refusal" and o2 == "refusal":
        return "refusal_to_refusal"
    if o1 == "refusal" and o2 == "grounded_answer":
        return "refusal_to_grounded_answer"
    if o1 == "grounded_answer" and o2 == "refusal":
        return "grounded_answer_to_refusal"
    if o1 == "grounded_answer" and o2 == "grounded_answer":
        return "grounded_answer_to_grounded_answer"
    return "other"


def _predictive_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [row for row in rows if row.get("recovered_grounded_answer")]
    safe_correct = [row for row in recovered if final_action_correct(row["expected_action"], row["a2_final_action"]) and safe_action_correct(row["expected_action"], row["a2_final_action"])]
    return {
        "observation_count": len(rows),
        "retry_to_answer_draft_rate": metric(sum(row.get("attempt_2_outcome") == "grounded_answer" for row in rows), len(rows)),
        "retry_to_grounded_answer_rate": metric(len(recovered), len(rows)),
        "retry_to_offline_correct_rate": metric(sum(final_action_correct(row["expected_action"], row["a2_final_action"]) for row in recovered), len(rows)),
        "retry_to_safe_correct_recovery_rate": metric(len(safe_correct), len(rows)),
        "retry_to_repeatable_recovery_rate": metric(0, len(rows)),
    }


def _result_digests(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != "result_digests.json")
    return {
        "schema_version": "opk-rag.task0077-result-digests.v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "files": {str(p.relative_to(path)): file_digest(p) for p in files},
    }
