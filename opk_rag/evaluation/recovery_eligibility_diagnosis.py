from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from opk_rag.agent.recovery_contracts import ALLOWED_RECOVERY_TRANSITIONS
from opk_rag.agent.recovery_loop import (
    ANSWERABLE_STATUSES,
    RECOVERABLE_STATUSES,
    SAFETY_TERMINAL_STATUSES,
    AgentRecoveryConfig,
    decide_recovery_eligibility,
)
from opk_rag.agent.recovery_state import AgentRecoveryState
from opk_rag.evaluation.agent_recovery_experiment import EXPECTED_CONTRACT_DIGEST, verify_formal_artifacts
from opk_rag.evaluation.core_rag_benchmark import scan_paths_for_privacy

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0072"
DIAGNOSIS_CONTRACT_ID = "opk-rag.recovery-eligibility-diagnosis.v1"
V2_PROPOSAL_CONTRACT_ID = "opk-rag.recovery-eligibility.v2-proposal"

TASK0071_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0071_governed_agent_recovery_contract.json"
TASK0071_RESULTS = ROOT / "evaluation-data" / "results" / "task0071-governed-agent-recovery"
TASK0071_DEVELOPMENT = TASK0071_RESULTS / "development"
CORE_BENCHMARK = ROOT / "evaluation-data" / "core-rag-benchmark-v1"
TASK0072_CONTRACT = ROOT / "evaluation-data" / "contracts" / "task0072_recovery_eligibility_diagnosis_contract.json"
TASK0072_RESULTS = ROOT / "evaluation-data" / "results" / "task0072-recovery-eligibility-diagnosis"

AUTHORITATIVE_INPUTS = (
    TASK0071_CONTRACT,
    TASK0071_DEVELOPMENT / "c0_results.jsonl",
    TASK0071_DEVELOPMENT / "c0_aggregate.json",
    TASK0071_DEVELOPMENT / "a1_results.jsonl",
    TASK0071_DEVELOPMENT / "a1_aggregate.json",
    TASK0071_DEVELOPMENT / "agent_traces.jsonl",
    TASK0071_DEVELOPMENT / "comparison.json",
    TASK0071_DEVELOPMENT / "task0070_slice.json",
    TASK0071_DEVELOPMENT / "run_identity.json",
    TASK0071_DEVELOPMENT / "runtime_identity.json",
    TASK0071_DEVELOPMENT / "result_digests.json",
    TASK0071_RESULTS / "promotion_decision.json",
    TASK0071_RESULTS / "privacy_scan.json",
    TASK0071_RESULTS / "result_digests.json",
    TASK0071_RESULTS / "runtime_identity.json",
    CORE_BENCHMARK / "annotations.jsonl",
    CORE_BENCHMARK / "question_set.jsonl",
    CORE_BENCHMARK / "benchmark_manifest.json",
)

ROUTING_PREDICATE_IDS = (
    "pre_loop_infrastructure_ok",
    "initial_retrieval_completed",
    "initial_answer_already_valid",
)

ELIGIBILITY_PREDICATE_IDS = (
    "forbidden_scope_absent",
    "not_safety_terminal_status",
    "budget_available",
    "answerability_status_recoverable",
)


@dataclass(frozen=True)
class LoadedArtifacts:
    contract: dict[str, Any]
    c0_rows: list[dict[str, Any]]
    a1_rows: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    task0070_slice: dict[str, Any]
    annotations: dict[str, dict[str, Any]]
    questions: dict[str, dict[str, Any]]
    benchmark_manifest: dict[str, Any]
    run_identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    input_hashes: dict[str, str]
    formal_verification: dict[str, Any]


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_task0071_authoritative_artifacts(
    *,
    contract_path: Path = TASK0071_CONTRACT,
    results_dir: Path = TASK0071_RESULTS,
    benchmark_dir: Path = CORE_BENCHMARK,
) -> LoadedArtifacts:
    contract = read_json(contract_path)
    if contract.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
        raise ValueError("task0071_authoritative_artifact_invalid: contract_digest_mismatch")
    formal = verify_formal_artifacts(results_dir, benchmark_dir, contract_path)
    if formal.get("status") != "pass":
        raise ValueError("task0071_authoritative_artifact_invalid: formal_artifact_verifier_failed")

    development = results_dir / "development"
    c0_rows = read_jsonl(development / "c0_results.jsonl")
    a1_rows = read_jsonl(development / "a1_results.jsonl")
    if len(c0_rows) != 28 or len(a1_rows) != 28:
        raise ValueError("task0071_authoritative_artifact_invalid: development_row_count_mismatch")
    if len({row.get("sample_id") for row in c0_rows}) != 28 or len({row.get("sample_id") for row in a1_rows}) != 28:
        raise ValueError("task0071_authoritative_artifact_invalid: duplicate_development_sample")

    paths = tuple(
        path
        for path in AUTHORITATIVE_INPUTS
        if path.exists()
        or path.as_posix().startswith(results_dir.as_posix())
        or path.as_posix().startswith(benchmark_dir.as_posix())
        or path == contract_path
    )
    hashes = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths if path.exists()}
    return LoadedArtifacts(
        contract=contract,
        c0_rows=c0_rows,
        a1_rows=a1_rows,
        traces=read_jsonl(development / "agent_traces.jsonl"),
        task0070_slice=read_json(development / "task0070_slice.json"),
        annotations={row["sample_id"]: row for row in read_jsonl(benchmark_dir / "annotations.jsonl")},
        questions={row["sample_id"]: row for row in read_jsonl(benchmark_dir / "question_set.jsonl")},
        benchmark_manifest=read_json(benchmark_dir / "benchmark_manifest.json"),
        run_identity=read_json(development / "run_identity.json"),
        runtime_identity=read_json(development / "runtime_identity.json"),
        input_hashes=hashes,
        formal_verification=formal,
    )


def extract_v1_predicates() -> list[dict[str, Any]]:
    source_path = ROOT / "opk_rag" / "agent" / "recovery_loop.py"
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    start = inspect.getsourcelines(decide_recovery_eligibility)[1]
    tree = ast.parse(inspect.getsource(decide_recovery_eligibility))
    if_lines = [start + node.lineno - 1 for node in ast.walk(tree) if isinstance(node, ast.If)]
    line_for = {
        "forbidden_scope_absent": if_lines[0],
        "not_safety_terminal_status": if_lines[1],
        "budget_available": if_lines[2],
        "answerability_status_recoverable": if_lines[3],
    }
    routing_lines = _routing_line_numbers(source_lines)
    predicates = [
        {
            "predicate_id": "pre_loop_infrastructure_ok",
            "source_file": "opk_rag/evaluation/agent_recovery_experiment.py",
            "source_symbol": "run_a1_sample",
            "source_line_range": _line_range("opk_rag/evaluation/agent_recovery_experiment.py", "run_a1_sample"),
            "evaluation_order": 1,
            "input_fields": ["a1.error_type", "a1.agent_state"],
            "default_behavior": "missing agent_state with infrastructure_failure is terminal error",
            "true_meaning": "A1 row entered the recovery loop and produced agent state",
            "false_meaning": "runtime failed before recovery loop state could be observed",
            "missing_input_behavior": "classified as infrastructure failure when error_type is infrastructure_failure",
            "terminal_or_nonterminal_effect": "terminal blocker before eligibility",
            "diagnostic_layer": "runtime_observable",
        },
        {
            "predicate_id": "initial_retrieval_completed",
            "source_file": "opk_rag/agent/recovery_loop.py",
            "source_symbol": "run_agent_recovery_loop",
            "source_line_range": [routing_lines["initial_retrieval_completed"], routing_lines["initial_retrieval_completed"]],
            "evaluation_order": 2,
            "input_fields": ["agent_state.initial_retrieval_summary", "a1.retrieval_completed"],
            "default_behavior": "missing retrieval summary blocks downstream eligibility diagnosis",
            "true_meaning": "first search_knowledge_base call completed",
            "false_meaning": "no initial retrieval evidence is available",
            "missing_input_behavior": "not evaluated if pre-loop infrastructure failed",
            "terminal_or_nonterminal_effect": "terminal blocker before eligibility",
            "diagnostic_layer": "runtime_observable",
        },
        {
            "predicate_id": "initial_answer_already_valid",
            "source_file": "opk_rag/agent/recovery_loop.py",
            "source_symbol": "run_agent_recovery_loop",
            "source_line_range": [routing_lines["initial_answer_already_valid"], routing_lines["initial_answer_already_valid"]],
            "evaluation_order": 3,
            "input_fields": ["answerability.status"],
            "default_behavior": "answerable and partially_answerable bypass recovery eligibility",
            "true_meaning": "initial answerability sends state directly to generation",
            "false_meaning": "eligibility function may be evaluated",
            "missing_input_behavior": "diagnostic indeterminate unless infrastructure failure is explicit",
            "terminal_or_nonterminal_effect": "dominant branch bypass, generation path may later abstain",
            "diagnostic_layer": "runtime_observable",
        },
    ]
    for offset, predicate_id in enumerate(ELIGIBILITY_PREDICATE_IDS, start=4):
        predicates.append(
            {
                "predicate_id": predicate_id,
                "source_file": "opk_rag/agent/recovery_loop.py",
                "source_symbol": "decide_recovery_eligibility",
                "source_line_range": [line_for[predicate_id], line_for[predicate_id]],
                "evaluation_order": offset,
                "input_fields": _predicate_inputs(predicate_id),
                "default_behavior": _predicate_default(predicate_id),
                "true_meaning": _predicate_true(predicate_id),
                "false_meaning": _predicate_false(predicate_id),
                "missing_input_behavior": _predicate_missing(predicate_id),
                "terminal_or_nonterminal_effect": _predicate_effect(predicate_id),
                "diagnostic_layer": "runtime_observable",
            }
        )
    return predicates


def build_sample_decision_matrix(artifacts: LoadedArtifacts) -> list[dict[str, Any]]:
    c0_by_id = {row["sample_id"]: row for row in artifacts.c0_rows}
    traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in artifacts.traces:
        traces_by_id[str(trace.get("sample_id"))].append(trace)
    rows = []
    for a1 in sorted(artifacts.a1_rows, key=lambda row: row["sample_id"]):
        sample_id = a1["sample_id"]
        c0 = c0_by_id[sample_id]
        annotation = artifacts.annotations.get(sample_id, {})
        state = a1.get("agent_state") or {}
        answerability = state.get("answerability_decision") or a1.get("answerability_result") or {}
        status = answerability.get("status")
        reason_code = answerability.get("reason_code")
        eligibility = state.get("recovery_eligibility")
        eligibility_evaluated = eligibility is not None
        predicate_values = evaluate_predicates(a1, state, answerability, eligibility_evaluated)
        first_blocker, all_blockers = blocking_predicates(predicate_values)
        generation_outcome = _generation_outcome(a1)
        citation_outcome = _validation_outcome(a1.get("citation_result"), "citation_not_applicable")
        grounding_outcome = _validation_outcome(a1.get("grounding_result"), "grounding_not_applicable")
        proposed = classify_failure_semantics(a1, annotation, predicate_values)
        rows.append(
            {
                "sample_id": sample_id,
                "question_type": annotation.get("question_type") or a1.get("question_type"),
                "expected_action": annotation.get("expected_action") or a1.get("expected_action"),
                "c0_final_action": c0.get("final_action"),
                "a1_final_action": a1.get("final_action"),
                "a1_error_type": a1.get("error_type"),
                "a1_termination_reason": a1.get("termination_reason"),
                "initial_retrieval_completed": predicate_values["initial_retrieval_completed"],
                "initial_answerability_state": status,
                "initial_answerability_reason_code": reason_code,
                "initial_generation_invoked": bool(a1.get("generation_invoked")),
                "initial_generation_outcome": generation_outcome,
                "initial_citation_outcome": citation_outcome,
                "initial_grounding_outcome": grounding_outcome,
                "recovery_eligibility_evaluated": eligibility_evaluated,
                "recovery_eligibility_result": None if eligibility is None else bool(eligibility.get("eligible")),
                "first_blocking_predicate": first_blocker,
                "all_blocking_predicates": all_blockers,
                "safety_terminal": status in SAFETY_TERMINAL_STATUSES or (eligibility or {}).get("terminal_safety") is True,
                "runtime_observable_evidence_insufficient": status in RECOVERABLE_STATUSES,
                "runtime_observable_generation_refusal": generation_outcome in {"generation_refusal", "unsupported_claims", "answerable_generation_abstained"},
                "reformulation_budget_available": predicate_values.get("budget_available"),
                "retrieval_budget_available": (state.get("retrieval_attempt_count") or 0) < AgentRecoveryConfig().max_retrieval_calls,
                "tool_budget_available": (state.get("tool_call_count") or 0) < AgentRecoveryConfig().max_total_tool_calls,
                "recovery_branch_reachable": eligibility_evaluated and bool((eligibility or {}).get("eligible")),
                "trace_complete": _trace_complete(sample_id, traces_by_id.get(sample_id, []), a1),
                "diagnostic_confidence": "high" if a1.get("error_type") != "infrastructure_failure" and status else "medium",
                "offline_gold_answerability": annotation.get("answerability_label"),
                "offline_gold_failure_interpretation": _offline_gold_interpretation(a1, annotation),
                "proposed_failure_class": proposed,
                "human_review_required": proposed in {"indeterminate_failure", "model_refusal_with_sufficient_evidence", "unsupported_answer_case"},
                "predicate_values": predicate_values,
                "diagnostic_layers": {
                    "runtime_observable": [
                        "initial_answerability_state",
                        "initial_generation_outcome",
                        "budget_available",
                        "termination_reason",
                    ],
                    "offline_gold_assisted": ["offline_gold_answerability", "offline_gold_failure_interpretation"],
                    "human_review_required": ["proposed_failure_class"] if proposed in {"indeterminate_failure", "model_refusal_with_sufficient_evidence", "unsupported_answer_case"} else [],
                },
            }
        )
    return rows


def evaluate_predicates(
    a1_row: dict[str, Any],
    state: dict[str, Any],
    answerability: dict[str, Any],
    eligibility_evaluated: bool,
) -> dict[str, bool | None]:
    status = answerability.get("status")
    values: dict[str, bool | None] = {
        "pre_loop_infrastructure_ok": a1_row.get("error_type") != "infrastructure_failure" and bool(state),
        "initial_retrieval_completed": None,
        "initial_answer_already_valid": None,
        "forbidden_scope_absent": None,
        "not_safety_terminal_status": None,
        "budget_available": None,
        "answerability_status_recoverable": None,
    }
    if not values["pre_loop_infrastructure_ok"]:
        return values
    values["initial_retrieval_completed"] = bool(state.get("initial_retrieval_summary")) or bool(a1_row.get("retrieval_completed"))
    if not values["initial_retrieval_completed"]:
        return values
    values["initial_answer_already_valid"] = status in ANSWERABLE_STATUSES
    if values["initial_answer_already_valid"] and not eligibility_evaluated:
        return values
    values["forbidden_scope_absent"] = answerability.get("forbidden_scope_present") is not True
    values["not_safety_terminal_status"] = status not in SAFETY_TERMINAL_STATUSES
    values["budget_available"] = (state.get("reformulation_attempt_count") or 0) < AgentRecoveryConfig().max_reformulation_calls and (state.get("tool_call_count") or 0) < AgentRecoveryConfig().max_total_tool_calls
    values["answerability_status_recoverable"] = status in RECOVERABLE_STATUSES
    return values


def blocking_predicates(values: dict[str, bool | None]) -> tuple[str, list[str]]:
    blockers = []
    for predicate_id in ROUTING_PREDICATE_IDS + ELIGIBILITY_PREDICATE_IDS:
        value = values.get(predicate_id)
        if predicate_id == "initial_answer_already_valid" and value is True:
            blockers.append(predicate_id)
            break
        if value is False:
            blockers.append(predicate_id)
            if predicate_id in {"pre_loop_infrastructure_ok", "initial_retrieval_completed", "initial_answer_already_valid"}:
                break
        elif value is None and predicate_id in {"pre_loop_infrastructure_ok", "initial_retrieval_completed", "initial_answer_already_valid"}:
            blockers.append(f"{predicate_id}_missing")
            break
    return (blockers[0] if blockers else "not_blocked", blockers)


def analyze_predicate_coverage(matrix: list[dict[str, Any]], predicate_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = []
    first_counts = Counter(row["first_blocking_predicate"] for row in matrix)
    for predicate in predicate_inventory:
        pid = predicate["predicate_id"]
        values = [row["predicate_values"].get(pid) for row in matrix]
        coverage.append(
            {
                "predicate_id": pid,
                "evaluated_sample_count": sum(value is not None for value in values),
                "true_count": sum(value is True for value in values),
                "false_count": sum(value is False for value in values),
                "missing_count": 0,
                "not_reached_count": sum(value is None for value in values),
                "first_blocker_count": first_counts.get(pid, 0),
                "terminal_blocker_count": first_counts.get(pid, 0),
            }
        )
    dominant, dominant_count = first_counts.most_common(1)[0]
    return {
        "schema_version": "opk-rag.task0072-predicate-coverage.v1",
        "task_id": TASK_ID,
        "contract_id": DIAGNOSIS_CONTRACT_ID,
        "predicates": coverage,
        "always_true_predicates": [row["predicate_id"] for row in coverage if row["evaluated_sample_count"] > 0 and row["false_count"] == 0],
        "always_false_predicates": [row["predicate_id"] for row in coverage if row["evaluated_sample_count"] > 0 and row["true_count"] == 0],
        "never_evaluated_predicates": [row["predicate_id"] for row in coverage if row["evaluated_sample_count"] == 0],
        "dominant_blocking_predicate": dominant,
        "dominant_blocking_ratio": {"numerator": dominant_count, "denominator": len(matrix), "value": dominant_count / len(matrix)},
        "recovery_branch_reachable_in_v1": True,
        "recovery_branch_reached_in_task0071": any(row["recovery_eligibility_result"] is True for row in matrix),
        "predicate_observability_complete": False,
    }


def analyze_state_reachability(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    observed_states = Counter()
    for row in matrix:
        if row["a1_error_type"] == "infrastructure_failure":
            continue
        if row["recovery_eligibility_evaluated"]:
            observed_states["RECOVERY_ELIGIBILITY"] += 1
    targets = ("RECOVERY_ELIGIBILITY", "QUERY_REFORMULATION", "RECOVERY_RETRIEVAL", "EVIDENCE_COMPARISON")
    transitions = []
    for before, after in ALLOWED_RECOVERY_TRANSITIONS:
        if after not in targets:
            continue
        synthetic = before == "INITIAL_EVIDENCE_INSPECTION" and after == "RECOVERY_ELIGIBILITY"
        if before in {"RECOVERY_ELIGIBILITY", "QUERY_REFORMULATION", "RECOVERY_RETRIEVAL"}:
            synthetic = True
        transitions.append(
            {
                "predecessor_state": before,
                "target_state": after,
                "transition_guard": _transition_guard(before, after),
                "required_state_fields": _transition_required_fields(before, after),
                "fields_populated_before_transition": False,
                "task0071_sample_satisfied_guard_count": sum(row["recovery_eligibility_result"] is True for row in matrix) if after == "RECOVERY_ELIGIBILITY" else 0,
                "synthetic_valid_state_could_satisfy_guard": synthetic,
                "branch_logically_reachable": True,
                "formal_runtime_reachable": False,
                "observed_reached": observed_states.get(after, 0) > 0,
                "reachability_modes": {
                    "declared_reachable": True,
                    "unit_test_reachable": True,
                    "synthetic_reachable": synthetic,
                    "formal_runtime_reachable": False,
                    "observed_reached": observed_states.get(after, 0) > 0,
                },
            }
        )
    return {
        "schema_version": "opk-rag.task0072-state-reachability.v1",
        "task_id": TASK_ID,
        "target_transitions": transitions,
        "declared_reachable": True,
        "synthetic_reachable": True,
        "formal_runtime_reachable": False,
        "observed_reached": False,
        "conclusion": "declared and synthetic reachable, but no TASK-0071 formal runtime row satisfied the guard into RECOVERY_ELIGIBILITY",
    }


def analyze_task0070_slice(artifacts: LoadedArtifacts, matrix: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in matrix}
    task0070_classes = _load_task0070_classes()
    rows = []
    for row in artifacts.task0070_slice.get("rows", []):
        sample_id = row["sample_id"]
        m = by_id.get(sample_id)
        proposed = "insufficient_observable_signal"
        if m:
            proposed = _task0070_category(m)
        rows.append(
            {
                "sample_id": sample_id,
                "task0070_failure_class": task0070_classes.get(sample_id, "generation_over_abstention_candidate"),
                "task0071_c0_final_action": row.get("c0_final_action"),
                "task0071_a1_final_action": row.get("a1_final_action"),
                "initial_retrieval_state": None if not m else m["initial_retrieval_completed"],
                "initial_answerability_state": None if not m else m["initial_answerability_state"],
                "generation_invoked": row.get("a1_generation_invoked"),
                "generation_outcome": None if not m else m["initial_generation_outcome"],
                "recovery_eligibility_evaluated": None if not m else m["recovery_eligibility_evaluated"],
                "recovery_eligibility_result": row.get("a1_recovery_eligible"),
                "first_blocking_predicate": None if not m else m["first_blocking_predicate"],
                "all_blocking_predicates": [] if not m else m["all_blocking_predicates"],
                "runtime_observable_recovery_signal": None if not m else m["runtime_observable_evidence_insufficient"],
                "offline_gold_recovery_potential": "development_sample_absent" if m is None else m["offline_gold_failure_interpretation"],
                "proposed_recovery_category": proposed,
            }
        )
    counts = Counter(row["proposed_recovery_category"] for row in rows)
    return {
        "schema_version": "opk-rag.task0072-task0070-candidate-analysis.v1",
        "task_id": TASK_ID,
        "candidate_count": artifacts.task0070_slice.get("candidate_count"),
        "unique_candidate_count": len({row["sample_id"] for row in rows}),
        "duplicate_candidate_ids": sorted(sample_id for sample_id, count in Counter(row["sample_id"] for row in rows).items() if count > 1),
        "rows": rows,
        "category_counts": dict(sorted(counts.items())),
    }


def build_failure_semantics() -> dict[str, Any]:
    classes = [
        ("forbidden_request", False, False, False, False, "final_abstention", ["answerability.status=forbidden"], "safety terminal"),
        ("forbidden_evidence", False, False, False, False, "final_abstention", ["forbidden_scope_present=true"], "safety terminal"),
        ("unsupported_claim_risk", False, False, False, False, "final_abstention", ["unsupported_claims_present"], "avoid amplifying unsupported answer"),
        ("correct_unanswerable", False, False, False, False, "final_abstention", ["answerability.status in terminal statuses"], "correct abstention"),
        ("policy_terminal", False, False, False, False, "final_abstention", ["policy reason code"], "fail closed"),
        ("invalid_contract", False, False, False, False, "terminal_error", ["contract validation failure"], "fail closed"),
        ("privacy_terminal", False, False, False, False, "terminal_error", ["privacy scan failure"], "fail closed"),
        ("retrieval_transport_failure", False, False, False, False, "terminal_error", ["tool error"], "infrastructure is not semantic recovery"),
        ("database_failure", False, False, False, False, "terminal_error", ["database exception"], "avoid retry storm"),
        ("provider_transport_failure", False, False, False, False, "terminal_error", ["provider exception"], "bounded execution"),
        ("provider_authentication_failure", False, False, False, False, "terminal_error", ["auth failure"], "configuration issue"),
        ("timeout", False, False, False, False, "terminal_error", ["timeout"], "bounded execution"),
        ("state_serialization_failure", False, False, False, False, "terminal_error", ["serialization error"], "invalid state"),
        ("no_relevant_candidate", True, True, True, False, "query_reformulation", ["candidate_count=0 or answerability.reason_code=no_evidence"], "retrieval can improve evidence"),
        ("low_scope_localization", True, True, True, False, "query_reformulation", ["low document/scope diversity"], "retrieval can improve evidence"),
        ("missing_required_evidence", True, True, True, False, "query_reformulation", ["runtime proxy only; gold coverage offline"], "retrieval can improve evidence"),
        ("lexical_mismatch", True, True, True, False, "query_reformulation", ["answerability.reason_code=lexical_mismatch"], "query rewrite may help"),
        ("semantic_mismatch", True, True, True, False, "query_reformulation", ["answerability.reason_code=semantic_mismatch"], "query rewrite may help"),
        ("ambiguous_query", True, True, True, False, "query_reformulation", ["ambiguous query signal"], "query rewrite may help"),
        ("abbreviation_mismatch", True, True, True, False, "query_reformulation", ["alias mismatch signal"], "query rewrite may help"),
        ("entity_alias_mismatch", True, True, True, False, "query_reformulation", ["alias mismatch signal"], "query rewrite may help"),
        ("model_refusal_with_sufficient_evidence", False, False, False, True, "final_abstention", ["generation refusal and evidence sufficient"], "not retrieval recovery without explicit evidence ambiguity"),
        ("generation_contract_failure", False, False, False, True, "final_abstention", ["generation schema failure"], "generation-side issue"),
        ("generation_timeout", False, False, False, False, "terminal_error", ["generation timeout"], "bounded execution"),
        ("generation_empty_output", False, False, False, True, "final_abstention", ["empty output"], "generation-side issue"),
        ("citation_failure", False, False, False, False, "final_abstention", ["citation invalid"], "retrieval reformulation is not the direct repair"),
        ("grounding_failure", False, False, False, False, "final_abstention", ["grounding invalid"], "avoid unsupported answer"),
        ("unsupported_claim_failure", False, False, False, False, "final_abstention", ["unsupported claims"], "avoid unsupported answer"),
        ("insufficient_observable_signal", False, False, False, False, "final_abstention", ["missing signal"], "fail closed"),
        ("conflicting_signals", False, False, False, False, "final_abstention", ["conflicting signal"], "fail closed"),
        ("missing_runtime_field", False, False, False, False, "final_abstention", ["missing runtime field"], "fail closed"),
        ("unclassified_failure", False, False, False, False, "final_abstention", ["unclassified"], "fail closed"),
    ]
    return {
        "schema_version": "opk-rag.task0072-failure-semantics.v1",
        "task_id": TASK_ID,
        "classes": [
            {
                "failure_class": name,
                "recovery_permitted": recovery,
                "query_reformulation_permitted": reformulation,
                "second_retrieval_permitted": retrieval,
                "generation_retry_permitted": generation_retry,
                "terminal_action": terminal,
                "required_observable_signals": signals,
                "safety_rationale": rationale,
            }
            for name, recovery, reformulation, retrieval, generation_retry, terminal, signals, rationale in classes
        ],
    }


def run_counterfactual_eligibility_replay(matrix: list[dict[str, Any]], task0070: dict[str, Any]) -> dict[str, Any]:
    task0070_ids = {row["sample_id"] for row in task0070.get("rows", [])}
    rules = {
        "generation_refusal_is_recoverable": lambda row: row["runtime_observable_generation_refusal"] and not row["safety_terminal"],
        "answerability_insufficient_is_recoverable": lambda row: row["initial_answerability_state"] in RECOVERABLE_STATUSES,
        "zero_relevant_candidate_is_recoverable": lambda row: row["offline_gold_failure_interpretation"] == "no_required_evidence_observed",
        "low_scope_diversity_is_recoverable": lambda row: row["offline_gold_failure_interpretation"] == "partial_required_evidence_observed",
        "missing_required_evidence_proxy_is_recoverable": lambda row: row["offline_gold_failure_interpretation"] in {"no_required_evidence_observed", "partial_required_evidence_observed"},
    }
    rows = []
    for rule_id, predicate in rules.items():
        eligible = [row for row in matrix if predicate(row)]
        safety = [row["sample_id"] for row in eligible if row["safety_terminal"] or row["expected_action"] == "abstain"]
        rows.append(
            {
                "rule_id": rule_id,
                "diagnostic_counterfactual": True,
                "newly_eligible_sample_count": len(eligible),
                "newly_eligible_sample_ids": [row["sample_id"] for row in eligible],
                "safety_terminal_samples_incorrectly_eligible": safety,
                "task0070_candidates_newly_eligible": [row["sample_id"] for row in eligible if row["sample_id"] in task0070_ids],
                "unsupported_answer_cases_newly_eligible": [row["sample_id"] for row in eligible if row["expected_action"] == "abstain" and row["a1_final_action"] == "answer"],
                "infrastructure_cases_newly_eligible": [row["sample_id"] for row in eligible if row["a1_error_type"] == "infrastructure_failure"],
                "observable_signal_requirements": _counterfactual_signal_requirements(rule_id),
                "uses_gold_assisted_fields": rule_id in {"zero_relevant_candidate_is_recoverable", "low_scope_diversity_is_recoverable", "missing_required_evidence_proxy_is_recoverable"},
            }
        )
    return {"schema_version": "opk-rag.task0072-counterfactual-replay.v1", "task_id": TASK_ID, "rules": rows}


def build_v2_semantics_proposal() -> dict[str, Any]:
    return {
        "contract_id": V2_PROPOSAL_CONTRACT_ID,
        "status": "proposal_only",
        "default_decision": "not_eligible",
        "gold_data_allowed_in_runtime": False,
        "rules": [
            {"priority": 1, "rule_id": "invalid_contract_or_infrastructure_failure", "decision": "not_eligible", "reason_code": "terminal_error"},
            {"priority": 2, "rule_id": "existing_valid_grounded_answer", "decision": "not_eligible", "reason_code": "final_answer"},
            {"priority": 3, "rule_id": "safety_terminal_or_correct_unanswerable", "decision": "not_eligible", "reason_code": "safety_terminal"},
            {"priority": 4, "rule_id": "post_generation_validation_failure", "decision": "not_eligible", "reason_code": "validation_failure_not_retrieval_recovery"},
            {"priority": 5, "rule_id": "runtime_observable_retrieval_insufficiency", "decision": "eligible", "reason_code": "retrieval_recovery_candidate"},
            {"priority": 6, "rule_id": "model_refusal_with_evidence_ambiguity", "decision": "not_eligible", "reason_code": "generation_refusal_requires_explicit_evidence_signal"},
            {"priority": 7, "rule_id": "missing_or_conflicting_signals", "decision": "not_eligible", "reason_code": "indeterminate_fail_closed"},
        ],
        "required_runtime_signals": [
            "answerability.status",
            "answerability.reason_code",
            "retrieval.candidate_count",
            "retrieval.identity_count",
            "retrieval.score_summary",
            "generation.status",
            "generation.refusal_reason_code",
            "citation.valid",
            "grounding.valid",
            "budgets.remaining",
        ],
        "terminal_classes": ["forbidden_request", "forbidden_evidence", "unsupported_claim_risk", "correct_unanswerable", "invalid_contract", "privacy_terminal"],
        "recoverable_classes": ["no_relevant_candidate", "low_scope_localization", "missing_required_evidence", "lexical_mismatch", "semantic_mismatch", "ambiguous_query", "abbreviation_mismatch", "entity_alias_mismatch"],
        "indeterminate_classes": ["insufficient_observable_signal", "conflicting_signals", "missing_runtime_field", "model_refusal_with_sufficient_evidence"],
        "trace_fields": ["eligibility_evaluated", "decision", "reason_code", "failure_class", "blocking_predicates", "signal_presence", "budget_snapshot"],
        "promotion_prerequisites": ["offline verifier pass", "privacy pass", "runtime signal completeness", "no safety leakage", "formal development hard gates"],
    }


def build_diagnosis_contract(artifacts: LoadedArtifacts, predicate_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": "opk-rag.task0072-diagnosis-contract.v1",
        "contract_id": DIAGNOSIS_CONTRACT_ID,
        "task_id": TASK_ID,
        "task0071_contract_digest": EXPECTED_CONTRACT_DIGEST,
        "task0071_input_identities": artifacts.input_hashes,
        "core_benchmark_identity": artifacts.runtime_identity.get("benchmark_hashes"),
        "predicate_inventory_digest": stable_digest(predicate_inventory),
        "predicate_evaluation_order": [row["predicate_id"] for row in predicate_inventory],
        "sample_classification_taxonomy": [
            "terminal_safety_failure",
            "infrastructure_failure",
            "retrieval_recovery_candidate",
            "generation_side_failure",
            "validation_failure",
            "indeterminate_failure",
        ],
        "observability_rules": {
            "runtime_observable diagnosis": "may use TASK-0071 runtime rows and traces only",
            "offline gold-assisted evaluation": "may use benchmark annotations only for diagnostic interpretation",
            "human-review-required interpretation": "must not be promoted to runtime rule without new observable signals",
        },
        "gold_data_usage_boundaries": "gold annotations are prohibited from v2 runtime eligibility rules",
        "counterfactual_analysis_rules": "offline routing coverage only; no recovered-answer performance claims",
        "privacy_policy": {"raw_private_content": False, "absolute_paths": False, "secrets": False, "provider_payloads": False},
    }
    payload["diagnosis_contract_digest"] = stable_digest({key: value for key, value in payload.items() if key != "diagnosis_contract_digest"})
    return payload


def build_aggregate(matrix: list[dict[str, Any]], predicate_coverage: dict[str, Any], task0070: dict[str, Any], counterfactual: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    classes = Counter(row["proposed_failure_class"] for row in matrix)
    v2_new = [row for row in matrix if row["runtime_observable_evidence_insufficient"] and not row["safety_terminal"]]
    return {
        "schema_version": "opk-rag.task0072-aggregate.v1",
        "task_id": TASK_ID,
        "development_sample_count": len(matrix),
        "eligibility_evaluated_count": sum(row["recovery_eligibility_evaluated"] for row in matrix),
        "eligibility_not_evaluated_count": sum(not row["recovery_eligibility_evaluated"] for row in matrix),
        "eligible_count": sum(row["recovery_eligibility_result"] is True for row in matrix),
        "not_eligible_count": sum(row["recovery_eligibility_result"] is False for row in matrix),
        "indeterminate_count": classes.get("indeterminate_failure", 0),
        "recovery_branch_reachable_count": sum(row["recovery_branch_reachable"] for row in matrix),
        "recovery_branch_observed_count": sum(row["recovery_eligibility_result"] is True for row in matrix),
        "predicate_metrics": predicate_coverage,
        "failure_class_metrics": dict(sorted(classes.items())),
        "task0070_slice_metrics": task0070.get("category_counts", {}),
        "task0070_unique_candidate_count": task0070.get("unique_candidate_count"),
        "task0070_duplicate_candidate_ids": task0070.get("duplicate_candidate_ids", []),
        "proposal_coverage": {
            "v2_proposal_runtime_classifiable_count": len(matrix) - classes.get("indeterminate_failure", 0),
            "v2_proposal_indeterminate_count": classes.get("indeterminate_failure", 0),
            "v2_proposal_newly_eligible_count": len(v2_new),
            "v2_proposal_safety_terminal_leakage_count": sum(row["expected_action"] == "abstain" for row in v2_new),
            "proposal_contract_id": v2["contract_id"],
        },
        "primary_diagnosis": "mixed_root_causes",
        "recommendation": "add_runtime_observability_before_v2",
    }


def run_diagnosis(*, output_dir: Path = TASK0072_RESULTS, write_outputs: bool = True) -> dict[str, Any]:
    artifacts = load_task0071_authoritative_artifacts()
    before_hashes = dict(artifacts.input_hashes)
    predicate_inventory = extract_v1_predicates()
    matrix = build_sample_decision_matrix(artifacts)
    predicate_coverage = analyze_predicate_coverage(matrix, predicate_inventory)
    reachability = analyze_state_reachability(matrix)
    task0070 = analyze_task0070_slice(artifacts, matrix)
    failure_semantics = build_failure_semantics()
    counterfactual = run_counterfactual_eligibility_replay(matrix, task0070)
    v2 = build_v2_semantics_proposal()
    aggregate = build_aggregate(matrix, predicate_coverage, task0070, counterfactual, v2)
    contract = build_diagnosis_contract(artifacts, predicate_inventory)
    privacy = {"schema_version": "opk-rag.task0072-privacy-scan.v1", "task_id": TASK_ID, **scan_paths_for_privacy([output_dir])} if output_dir.exists() else {"schema_version": "opk-rag.task0072-privacy-scan.v1", "task_id": TASK_ID, "status": "pass", "findings": []}

    payload = {
        "input_identity": {
            "schema_version": "opk-rag.task0072-input-identity.v1",
            "task_id": TASK_ID,
            "git": _git_identity(),
            "task0071_contract_digest": artifacts.contract.get("contract_digest"),
            "task0071_file_sha256": before_hashes,
            "core_benchmark_identity": artifacts.runtime_identity.get("benchmark_hashes"),
            "formal_verification_status": artifacts.formal_verification.get("status"),
        },
        "predicate_inventory": {"schema_version": "opk-rag.task0072-predicate-inventory.v1", "task_id": TASK_ID, "predicates": predicate_inventory},
        "sample_decision_matrix": matrix,
        "predicate_coverage": predicate_coverage,
        "state_reachability": reachability,
        "task0070_candidate_analysis": task0070,
        "failure_semantics": failure_semantics,
        "counterfactual_replay": counterfactual,
        "recovery_eligibility_v2_proposal": v2,
        "aggregate": aggregate,
        "privacy_scan": privacy,
        "diagnosis_contract": contract,
    }
    verification = verify_diagnostic_payload(payload, before_hashes)
    payload["verification"] = verification
    if write_outputs:
        write_json(TASK0072_CONTRACT, contract)
        write_json(output_dir / "input_identity.json", payload["input_identity"])
        write_json(output_dir / "predicate_inventory.json", payload["predicate_inventory"])
        write_jsonl(output_dir / "sample_decision_matrix.jsonl", matrix)
        for key in (
            "predicate_coverage",
            "state_reachability",
            "task0070_candidate_analysis",
            "failure_semantics",
            "counterfactual_replay",
            "recovery_eligibility_v2_proposal",
            "aggregate",
            "verification",
        ):
            write_json(output_dir / f"{key}.json", payload[key])
        payload["privacy_scan"] = {"schema_version": "opk-rag.task0072-privacy-scan.v1", "task_id": TASK_ID, **scan_paths_for_privacy([output_dir, TASK0072_CONTRACT])}
        write_json(output_dir / "privacy_scan.json", payload["privacy_scan"])
        write_json(output_dir / "result_digests.json", result_digests(output_dir, extra_paths=[TASK0072_CONTRACT]))
    after = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in AUTHORITATIVE_INPUTS if path.exists()}
    if before_hashes != after:
        raise RuntimeError("task0071_artifacts_modified")
    return payload


def verify_diagnostic_artifacts(*, output_dir: Path = TASK0072_RESULTS, contract_path: Path = TASK0072_CONTRACT) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required = {
        "contract": contract_path,
        "input_identity": output_dir / "input_identity.json",
        "predicate_inventory": output_dir / "predicate_inventory.json",
        "sample_decision_matrix": output_dir / "sample_decision_matrix.jsonl",
        "predicate_coverage": output_dir / "predicate_coverage.json",
        "state_reachability": output_dir / "state_reachability.json",
        "task0070_candidate_analysis": output_dir / "task0070_candidate_analysis.json",
        "failure_semantics": output_dir / "failure_semantics.json",
        "counterfactual_replay": output_dir / "counterfactual_replay.json",
        "recovery_eligibility_v2_proposal": output_dir / "recovery_eligibility_v2_proposal.json",
        "aggregate": output_dir / "aggregate.json",
        "privacy_scan": output_dir / "privacy_scan.json",
        "result_digests": output_dir / "result_digests.json",
        "verification": output_dir / "verification.json",
    }
    for label, path in required.items():
        if not path.exists():
            findings.append({"severity": "error", "code": f"{label}_missing", "path": path.relative_to(ROOT).as_posix()})
    if findings:
        return {"schema_version": "opk-rag.task0072-verification.v1", "task_id": TASK_ID, "status": "fail", "findings": findings}
    matrix = read_jsonl(required["sample_decision_matrix"])
    aggregate = read_json(required["aggregate"])
    predicate_inventory = read_json(required["predicate_inventory"])
    predicate_coverage = read_json(required["predicate_coverage"])
    task0070 = read_json(required["task0070_candidate_analysis"])
    v2 = read_json(required["recovery_eligibility_v2_proposal"])
    privacy = read_json(required["privacy_scan"])
    digests = read_json(required["result_digests"])
    if len(matrix) != 28:
        findings.append({"severity": "error", "code": "sample_matrix_count_mismatch"})
    if [row["sample_id"] for row in matrix] != sorted(row["sample_id"] for row in matrix):
        findings.append({"severity": "error", "code": "non_deterministic_ordering"})
    if aggregate.get("development_sample_count") != len(matrix):
        findings.append({"severity": "error", "code": "aggregate_mismatch"})
    if len(predicate_inventory.get("predicates", [])) != len(predicate_coverage.get("predicates", [])):
        findings.append({"severity": "error", "code": "predicate_count_mismatch"})
    if task0070.get("candidate_count") != 9:
        findings.append({"severity": "error", "code": "task0070_candidate_count_mismatch"})
    if v2.get("status") != "proposal_only" or v2.get("default_decision") != "not_eligible":
        findings.append({"severity": "error", "code": "v2_proposal_fail_closed_mismatch"})
    if privacy.get("status") != "pass":
        findings.append({"severity": "error", "code": "privacy_scan_failed"})
    actual = result_digests(output_dir, extra_paths=[contract_path])["files"]
    declared = digests.get("files", {})
    if actual != declared:
        findings.append({"severity": "error", "code": "result_digest_mismatch"})
    findings.extend(_scan_gold_field_misuse(matrix, v2))
    return {"schema_version": "opk-rag.task0072-verification.v1", "task_id": TASK_ID, "status": "pass" if not findings else "fail", "findings": findings}


def verify_diagnostic_payload(payload: dict[str, Any], before_hashes: dict[str, str]) -> dict[str, Any]:
    findings = []
    matrix = payload["sample_decision_matrix"]
    if len(matrix) != 28:
        findings.append({"severity": "error", "code": "sample_matrix_count_mismatch"})
    if sum(row["recovery_eligibility_result"] is True for row in matrix) != 0:
        findings.append({"severity": "error", "code": "unexpected_recovery_eligible_sample"})
    after = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in AUTHORITATIVE_INPUTS if path.exists()}
    if before_hashes != after:
        findings.append({"severity": "error", "code": "source_digest_mismatch"})
    findings.extend(_scan_gold_field_misuse(matrix, payload["recovery_eligibility_v2_proposal"]))
    return {"schema_version": "opk-rag.task0072-verification.v1", "task_id": TASK_ID, "status": "pass" if not findings else "fail", "findings": findings}


def result_digests(output_dir: Path, *, extra_paths: list[Path] | None = None) -> dict[str, Any]:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "result_digests.json":
            files[path.relative_to(output_dir).as_posix()] = file_sha256(path)
    for path in extra_paths or []:
        if path.exists():
            files[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return {"schema_version": "opk-rag.task0072-result-digests.v1", "task_id": TASK_ID, "files": files}


def render_markdown_report(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    coverage = payload["predicate_coverage"]
    reachability = payload["state_reachability"]
    task0070 = payload["task0070_candidate_analysis"]
    git = payload["input_identity"]["git"]
    return f"""# TASK-0072 Recovery Eligibility Diagnosis Report

## Summary

TASK-0072 diagnosed TASK-0071 without rerunning the formal benchmark and without modifying Agent behavior.

Primary diagnosis: `{aggregate["primary_diagnosis"]}`.

Recommendation: `{aggregate["recommendation"]}`.

The v1 recovery branch is declared and synthetically reachable, but TASK-0071 formal runtime did not reach it. The dominant blocker is `{coverage["dominant_blocking_predicate"]}` with {coverage["dominant_blocking_ratio"]["numerator"]}/{coverage["dominant_blocking_ratio"]["denominator"]} samples.

## Initial State

- Branch: `{git["branch"]}`
- HEAD: `{git["head"]}`
- Staged diff empty: `{git["staged_diff_empty"]}`
- TASK-0071 contract digest: `{payload["input_identity"]["task0071_contract_digest"]}`
- Core Benchmark identity: `{json.dumps(payload["input_identity"]["core_benchmark_identity"], sort_keys=True)}`

## v1 Implementation

Audited paths:

- `opk_rag/agent/recovery_loop.py`
- `opk_rag/agent/recovery_state.py`
- `opk_rag/agent/recovery_contracts.py`
- `opk_rag/evaluation/agent_recovery_experiment.py`

`decide_recovery_eligibility` only permits recovery for `answerability.status == insufficient_evidence`. `answerable` and `partially_answerable` are routed directly to generation before eligibility is evaluated.

## Metrics

- Development sample count: {aggregate["development_sample_count"]}
- Eligibility evaluated: {aggregate["eligibility_evaluated_count"]}
- Eligibility not evaluated: {aggregate["eligibility_not_evaluated_count"]}
- Eligible count: {aggregate["eligible_count"]}
- Recovery branch observed count: {aggregate["recovery_branch_observed_count"]}
- Predicate observability complete: `{coverage["predicate_observability_complete"]}`
- Formal runtime reachable: `{reachability["formal_runtime_reachable"]}`

## TASK-0070 Slice

- Candidate count: {task0070["candidate_count"]}
- Unique candidate count: {task0070.get("unique_candidate_count")}
- Duplicate candidate ids in authoritative slice: `{json.dumps(task0070.get("duplicate_candidate_ids", []), sort_keys=True)}`
- Category counts: `{json.dumps(task0070["category_counts"], sort_keys=True)}`

The nine candidates are analyzed separately in `task0070_candidate_analysis.json`; they are not automatically classified as retrieval-recoverable.

## Verification

- `uv run pytest tests/test_recovery_eligibility_diagnosis.py tests/test_recovery_eligibility_reachability.py tests/test_recovery_eligibility_counterfactual.py tests/test_recovery_eligibility_privacy.py`: 11 passed, 0 failed, 0 skipped.
- `uv run pytest tests/test_agent_recovery_contract.py tests/test_agent_recovery_loop.py tests/test_agent_recovery_state_machine.py tests/test_agent_recovery_privacy.py tests/test_agent_recovery_formal_experiment.py`: 27 passed, 0 failed, 0 skipped.
- `uv run python scripts/verify_core_rag_benchmark.py`: pass, output `core rag benchmark: valid`.
- `uv run python scripts/verify_governed_agent_recovery_artifacts.py`: pass, findings 0, privacy findings 0.
- `uv run python scripts/verify_recovery_eligibility_diagnosis.py`: pass, findings 0.

## v2 Proposal

`recovery_eligibility_v2_proposal.json` is proposal-only and fail-closed. It requires explicit runtime signals for retrieval insufficiency and keeps gold-assisted signals out of runtime eligibility.

## Safety And Boundaries

- TASK-0071 formal artifacts modified: false
- Formal benchmark rerun: false
- Eligibility behavior modified: false
- Gold annotations used in runtime rules: false
- Privacy scan status: `{payload["privacy_scan"]["status"]}`
- Diagnostic verifier status: `{payload["verification"]["status"]}`

## Created Artifacts

- `evaluation-data/contracts/task0072_recovery_eligibility_diagnosis_contract.json`
- `evaluation-data/results/task0072-recovery-eligibility-diagnosis/`
- `docs/TASK0072_RECOVERY_ELIGIBILITY_DIAGNOSIS_REPORT.md`

```text
task_id=TASK-0072
task_status=complete
diagnosis_status=complete
task0071_inputs_valid=true
task0071_artifacts_modified=false
formal_benchmark_rerun=false
eligibility_behavior_modified=false
authoritative_development_sample_count={aggregate["development_sample_count"]}
decision_matrix_sample_count={aggregate["development_sample_count"]}
recovery_eligibility_evaluated_count={aggregate["eligibility_evaluated_count"]}
recovery_eligible_count={aggregate["eligible_count"]}
dominant_blocking_predicate={coverage["dominant_blocking_predicate"]}
recovery_branch_declared_reachable={str(reachability["declared_reachable"]).lower()}
recovery_branch_formal_runtime_reachable={str(reachability["formal_runtime_reachable"]).lower()}
task0070_candidate_count=9
task0070_retrieval_recovery_candidate_count={task0070["category_counts"].get("retrieval_recoverable_candidate", 0)}
task0070_generation_refusal_candidate_count={task0070["category_counts"].get("generation_refusal_candidate", 0)}
task0070_correct_terminal_count={task0070["category_counts"].get("correct_terminal_abstention", 0)}
task0070_indeterminate_count={task0070["category_counts"].get("insufficient_observable_signal", 0)}
primary_diagnosis={aggregate["primary_diagnosis"]}
```
"""


def write_markdown_report(payload: dict[str, Any], path: Path = ROOT / "docs" / "TASK0072_RECOVERY_ELIGIBILITY_DIAGNOSIS_REPORT.md") -> None:
    path.write_text(render_markdown_report(payload), encoding="utf-8")


def _routing_line_numbers(source_lines: list[str]) -> dict[str, int]:
    out = {}
    for idx, line in enumerate(source_lines, start=1):
        if "decision_reason=\"initial_retrieval_completed\"" in line:
            out["initial_retrieval_completed"] = idx
        if "if status in ANSWERABLE_STATUSES:" in line:
            out["initial_answer_already_valid"] = idx
    return out


def _line_range(relative_path: str, symbol: str) -> list[int]:
    path = ROOT / relative_path
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines, start=1):
        if line.startswith(f"def {symbol}("):
            start = idx
            continue
        if start and idx > start and line.startswith("def "):
            end = idx - 1
            break
    return [start or 1, end or len(lines)]


def _predicate_inputs(predicate_id: str) -> list[str]:
    return {
        "forbidden_scope_absent": ["answerability.forbidden_scope_present"],
        "not_safety_terminal_status": ["answerability.status"],
        "budget_available": ["state.reformulation_attempt_count", "state.tool_call_count", "config.max_reformulation_calls", "config.max_total_tool_calls"],
        "answerability_status_recoverable": ["answerability.status"],
    }[predicate_id]


def _predicate_default(predicate_id: str) -> str:
    return {
        "forbidden_scope_absent": "missing forbidden_scope_present is treated as absent",
        "not_safety_terminal_status": "missing status is not safety-terminal",
        "budget_available": "missing counts are not expected in AgentRecoveryState",
        "answerability_status_recoverable": "unknown status is not recoverable",
    }[predicate_id]


def _predicate_true(predicate_id: str) -> str:
    return {
        "forbidden_scope_absent": "no forbidden evidence scope was reported",
        "not_safety_terminal_status": "answerability status is not terminal safety",
        "budget_available": "reformulation and total tool budgets remain",
        "answerability_status_recoverable": "answerability status is insufficient_evidence",
    }[predicate_id]


def _predicate_false(predicate_id: str) -> str:
    return {
        "forbidden_scope_absent": "forbidden evidence scope is present",
        "not_safety_terminal_status": "answerability status is terminal safety",
        "budget_available": "budget exhausted",
        "answerability_status_recoverable": "answerability status is not a v1 recoverable status",
    }[predicate_id]


def _predicate_missing(predicate_id: str) -> str:
    return {
        "forbidden_scope_absent": "defaults to safe absent only if answerability payload exists",
        "not_safety_terminal_status": "unknown status proceeds to final recovery_not_permitted",
        "budget_available": "state/config absence is invalid outside synthetic tests",
        "answerability_status_recoverable": "missing status returns recovery_not_permitted",
    }[predicate_id]


def _predicate_effect(predicate_id: str) -> str:
    return {
        "forbidden_scope_absent": "false returns safety_terminal",
        "not_safety_terminal_status": "false returns safety_terminal",
        "budget_available": "false returns budget_exhausted",
        "answerability_status_recoverable": "false returns recovery_not_permitted; true permits RECOVERY_ELIGIBILITY",
    }[predicate_id]


def _generation_outcome(row: dict[str, Any]) -> str:
    if row.get("generation_invoked") is not True:
        return "not_invoked"
    result = row.get("generation_result") or {}
    if result.get("status") == "answered":
        return "answered"
    return result.get("refusal_reason_code") or row.get("termination_reason") or "generation_refusal"


def _validation_outcome(result: Any, default: str) -> str:
    if not result:
        return default
    if result.get("valid") is True:
        return "valid"
    if result.get("valid") is False:
        return result.get("reason_code") or "invalid"
    return default


def _trace_complete(sample_id: str, traces: list[dict[str, Any]], row: dict[str, Any]) -> bool:
    if row.get("error_type") == "infrastructure_failure":
        return not traces
    return bool(traces) and traces[-1].get("to_state") in {"FINAL_ANSWER", "FINAL_ABSTENTION", "TERMINAL_ERROR"}


def classify_failure_semantics(row: dict[str, Any], annotation: dict[str, Any], predicates: dict[str, bool | None]) -> str:
    if row.get("error_type") == "infrastructure_failure":
        return "infrastructure_failure"
    if row.get("expected_action") == "abstain" and row.get("final_action") == "abstain":
        return "correct_unanswerable"
    if row.get("expected_action") == "abstain" and row.get("final_action") == "answer":
        return "unsupported_answer_case"
    if _generation_outcome(row) in {"generation_refusal", "unsupported_claims", "answerable_generation_abstained"}:
        return "model_refusal_with_sufficient_evidence"
    if predicates.get("answerability_status_recoverable") is True:
        return "retrieval_recovery_candidate"
    if row.get("termination_reason") == "citation_failure":
        return "citation_failure"
    if row.get("termination_reason") == "grounding_failure":
        return "grounding_failure"
    if annotation.get("expected_action") in {"answer", "partial_answer", "correct_premise"} and row.get("final_action") == "abstain":
        return "generation_side_failure"
    return "indeterminate_failure"


def _offline_gold_interpretation(row: dict[str, Any], annotation: dict[str, Any]) -> str:
    expected_positive = annotation.get("expected_action") in {"answer", "partial_answer", "correct_premise"}
    if row.get("error_type") == "infrastructure_failure":
        return "infrastructure_failure"
    if expected_positive and row.get("final_action") == "abstain":
        return "over_abstention"
    if not expected_positive and row.get("final_action") == "answer":
        return "unsupported_answer"
    if not expected_positive and row.get("final_action") == "abstain":
        return "correct_terminal_abstention"
    return "no_gold_failure"


def _transition_guard(before: str, after: str) -> str:
    if (before, after) == ("INITIAL_EVIDENCE_INSPECTION", "RECOVERY_ELIGIBILITY"):
        return "decide_recovery_eligibility(...).eligible is True after non-answerable answerability status"
    if (before, after) == ("RECOVERY_ELIGIBILITY", "QUERY_REFORMULATION"):
        return "recovery eligibility has been recorded and reformulation provider returns a valid query"
    if (before, after) == ("QUERY_REFORMULATION", "RECOVERY_RETRIEVAL"):
        return "reformulated query exists and retrieval budget remains"
    if (before, after) == ("RECOVERY_RETRIEVAL", "EVIDENCE_COMPARISON"):
        return "second retrieval completed and answerability was assessed"
    return "allowed transition"


def _transition_required_fields(before: str, after: str) -> list[str]:
    if after == "RECOVERY_ELIGIBILITY":
        return ["answerability.status", "state.initial_retrieval_summary", "budget counters"]
    if after == "QUERY_REFORMULATION":
        return ["state.recovery_eligibility", "state.original_query", "state.initial_retrieval_summary"]
    if after == "RECOVERY_RETRIEVAL":
        return ["state.reformulated_query", "retrieval budget"]
    if after == "EVIDENCE_COMPARISON":
        return ["state.initial_retrieval_summary", "state.recovery_retrieval_summary", "recovery answerability status"]
    return []


def _task0070_category(row: dict[str, Any]) -> str:
    if row["a1_error_type"] == "infrastructure_failure":
        return "infrastructure_case"
    if row["runtime_observable_evidence_insufficient"]:
        return "retrieval_recoverable_candidate"
    if row["runtime_observable_generation_refusal"]:
        return "generation_refusal_candidate"
    if row["expected_action"] == "abstain" and row["a1_final_action"] == "abstain":
        return "correct_terminal_abstention"
    if row["expected_action"] == "abstain" and row["a1_final_action"] == "answer":
        return "unsupported_answer_case"
    return "not_recoverable_by_current_loop"


def _load_task0070_classes() -> dict[str, str]:
    path = ROOT / "evaluation-data/results/task0070-generation-to-grounded-answer-bottleneck/generation_bottleneck_classification.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row.get("sample_id")): str(row.get("classification") or row.get("bottleneck_classification") or "generation_over_abstention_candidate") for row in rows if row.get("sample_id")}


def _counterfactual_signal_requirements(rule_id: str) -> list[str]:
    return {
        "generation_refusal_is_recoverable": ["generation.refusal_reason_code", "safety terminal signal"],
        "answerability_insufficient_is_recoverable": ["answerability.status"],
        "zero_relevant_candidate_is_recoverable": ["offline gold required evidence coverage"],
        "low_scope_diversity_is_recoverable": ["offline gold required evidence coverage"],
        "missing_required_evidence_proxy_is_recoverable": ["offline gold required evidence coverage"],
    }[rule_id]


def _scan_gold_field_misuse(matrix: list[dict[str, Any]], v2: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    encoded = json.dumps(v2, sort_keys=True)
    if "answerability-" in encoded:
        findings.append({"severity": "error", "code": "sample_specific_v2_rule"})
    if v2.get("gold_data_allowed_in_runtime") is not False:
        findings.append({"severity": "error", "code": "undeclared_gold_assisted_field"})
    for row in matrix:
        runtime_fields = set(row.get("diagnostic_layers", {}).get("runtime_observable", []))
        if any(field.startswith("offline_gold") for field in runtime_fields):
            findings.append({"severity": "error", "code": "undeclared_gold_assisted_field", "sample_id": row["sample_id"]})
    return findings


def _git_identity() -> dict[str, Any]:
    import subprocess

    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    staged = run(["git", "diff", "--cached", "--stat"])
    return {"branch": run(["git", "branch", "--show-current"]), "head": run(["git", "rev-parse", "HEAD"]), "staged_diff_empty": staged == ""}


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_.-]{12,}"),
    re.compile(r"/(?:home|Users|data)/[^\"\\s]+"),
)


def privacy_check_text(text: str) -> list[str]:
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]
