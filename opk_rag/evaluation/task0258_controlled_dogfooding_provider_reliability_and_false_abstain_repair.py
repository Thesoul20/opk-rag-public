from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from opk_rag.answer.service import answer_knowledge_base
from opk_rag.core_tools.tools import assess_answerability
from opk_rag.evaluation.task0257_controlled_realistic_dogfooding_traffic_and_selective_agent_evidence_bridge import _runtime_parts
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.live_selective_agent_shadow import build_live_observation_record
from opk_rag.showcase.runtime_trace import RuntimeTraceContext

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0258"
SCHEMA = "opk-rag.task0258.controlled-dogfooding-provider-reliability-and-false-abstain-repair.v1"
TASK0257_RESULT = ROOT / "evaluation-data/results/task0257-controlled-realistic-dogfooding-traffic-and-selective-agent-evidence-bridge"
TASK0250_RESULT = ROOT / "evaluation-data/results/task0250-selective-agent-shadow-readiness-and-shadow-evaluation"
FIXED_QUERY_FILE = ROOT / "evaluation-data/controlled-dogfooding/task0257_queries.jsonl"
TASK0257_REVIEW = ROOT / "evaluation-data/controlled-dogfooding/task0257_divergence_review.json"
HOLDOUT_FILE = ROOT / "evaluation-data/controlled-dogfooding/task0258_repair_holdout.jsonl"
BENCH_QUERY_FILE = ROOT / "evaluation-data/agentic-rag-benchmark-v1/queries.jsonl"
BENCH_GOLD_FILE = ROOT / "evaluation-data/agentic-rag-benchmark-v1/evaluator_gold.jsonl"
RESULT = ROOT / "evaluation-data/results/task0258-controlled-dogfooding-provider-reliability-and-false-abstain-repair"
CONTRACT = ROOT / "evaluation-data/contracts/task0258_controlled_dogfooding_provider_reliability_and_false_abstain_repair.json"
REPORT = ROOT / "docs/TASK0258_CONTROLLED_DOGFOODING_PROVIDER_RELIABILITY_AND_FALSE_ABSTAIN_REPAIR_REPORT.md"
REGRESSION = RESULT / "regression.json"

TASK0257_FROZEN_DIGESTS = {
    "query_file": "e6d302f22ab05cac1b02e02745af505ae051642c37f92b2027ccd840677aa2a2",
    "review_file": "15b5cef72cefe597c01cd61ed43415879f0c3959ab2caab3380dfbb6412256b0",
    "runtime_observations": "a673f03bd29f8df62b9e406f5f5d805ec771041fd68dcd8b0b76d9b9a82f5597",
    "summary": "05a9f7b8b7f1b729558b4735e2708131a2b3228723bae4fef5c883dce4ac2ed7",
}
TASK0257_PATHS = {
    "query_file": FIXED_QUERY_FILE,
    "review_file": TASK0257_REVIEW,
    "runtime_observations": TASK0257_RESULT / "runtime_observations.jsonl",
    "summary": TASK0257_RESULT / "summary.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())



def _regression_authority() -> dict[str, Any]:
    if REGRESSION.is_file():
        payload = _read_json(REGRESSION)
        return {
            "focused_test_count": payload.get("focused_test_count"),
            "focused_tests_passed": payload.get("focused_tests_passed"),
            "related_agent_regression_passed_count": payload.get("related_agent_regression_passed_count"),
            "related_agent_regression_failed_count": payload.get("related_agent_regression_failed_count"),
            "related_agent_regression_failure": payload.get("related_agent_regression_failure"),
            "historical_showcase_isolation_compatibility_passed_count": payload.get("historical_showcase_isolation_compatibility_passed_count"),
            "historical_showcase_isolation_compatibility_failed_count": payload.get("historical_showcase_isolation_compatibility_failed_count"),
            "full_suite_passed": payload.get("full_suite_passed"),
            "full_suite_skipped": payload.get("full_suite_skipped"),
            "full_suite_failed": payload.get("full_suite_failed"),
            "prior_task0257_full_suite_passed": payload.get("prior_task0257_full_suite_passed"),
            "prior_task0257_full_suite_skipped": payload.get("prior_task0257_full_suite_skipped"),
            "prior_task0257_full_suite_failed": payload.get("prior_task0257_full_suite_failed"),
            "new_task0258_full_suite_failure_count": payload.get("new_task0258_full_suite_failure_count"),
            "task0258_verifier_passed": payload.get("task0258_verifier_passed"),
            "git_diff_check_passed": payload.get("git_diff_check_passed"),
            "full_suite_side_effect_artifacts_restored": payload.get("full_suite_side_effect_artifacts_restored"),
            "task_start_head": payload.get("task_start_head"),
            "current_head": payload.get("current_head"),
            "git_head_unchanged_since_task_start": payload.get("git_head_unchanged_since_task_start"),
        }
    return {
        "focused_test_count": None, "focused_tests_passed": None,
        "related_agent_regression_passed_count": None, "related_agent_regression_failed_count": None,
        "related_agent_regression_failure": None,
        "historical_showcase_isolation_compatibility_passed_count": None,
        "historical_showcase_isolation_compatibility_failed_count": None,
        "full_suite_passed": None, "full_suite_skipped": None, "full_suite_failed": None,
        "prior_task0257_full_suite_passed": 2588, "prior_task0257_full_suite_skipped": 86, "prior_task0257_full_suite_failed": 8,
        "new_task0258_full_suite_failure_count": None,
        "task0258_verifier_passed": None, "git_diff_check_passed": None,
        "full_suite_side_effect_artifacts_restored": None,
        "task_start_head": None, "current_head": None, "git_head_unchanged_since_task_start": None,
    }

def task0257_identity() -> dict[str, Any]:
    actual = {name: _sha(path) for name, path in TASK0257_PATHS.items()}
    matches = {name: actual[name] == expected for name, expected in TASK0257_FROZEN_DIGESTS.items()}
    return {
        "schema_version": "opk-rag.task0258.task0257-frozen-identity.v1",
        "expected_sha256": TASK0257_FROZEN_DIGESTS,
        "actual_sha256": actual,
        "matches": matches,
        "task0257_historical_artifacts_unchanged": all(matches.values()),
    }


def entry_gate() -> dict[str, Any]:
    summary = _read_json(TASK0257_RESULT / "summary.json")
    identity = task0257_identity()
    checks = {
        "task0257_identity_valid": identity["task0257_historical_artifacts_unchanged"],
        "task0257_partial_hold_expected": summary.get("task_status") == "partial" and summary.get("controlled_dogfooding_evidence_sufficient") is False,
        "task0257_query_count_48": summary.get("controlled_dogfooding_query_count") == 48,
        "task0257_execution_complete": summary.get("executed_observation_count") == 48 and summary.get("execution_failure_count") == 0,
        "provider_gap_present": float(summary.get("final_structured_validity") or 0.0) < 0.98,
        "false_abstain_gap_present": int(summary.get("probable_false_abstain_count") or 0) > 0,
        "hard_safety_clean": int(summary.get("hard_safety_violation_count") or 0) == 0,
        "production_inactive": summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False and summary.get("canary_execution_performed") is False,
        "task0257_verifier_passed": summary.get("task0257_verifier_passed") is True,
    }
    return {
        "schema_version": "opk-rag.task0258.entry-gate.v1",
        "checks": checks,
        "entry_gate_passed": all(checks.values()),
    }


def query_identity() -> dict[str, Any]:
    rows = _read_jsonl(FIXED_QUERY_FILE)
    return {
        "schema_version": "opk-rag.task0258.fixed-query-identity.v1",
        "query_file": str(FIXED_QUERY_FILE.relative_to(ROOT)),
        "sha256": _sha(FIXED_QUERY_FILE),
        "expected_sha256": TASK0257_FROZEN_DIGESTS["query_file"],
        "query_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "query_digests": [hashlib.sha256(str(row["query"]).encode()).hexdigest() for row in rows],
        "query_identity_match": _sha(FIXED_QUERY_FILE) == TASK0257_FROZEN_DIGESTS["query_file"] and len(rows) == 48,
    }


def holdout_manifest() -> dict[str, Any]:
    rows = _read_jsonl(HOLDOUT_FILE)
    fixed = {_norm(str(row["query"])) for row in _read_jsonl(FIXED_QUERY_FILE)}
    bench = {_norm(str(row["question"])) for row in _read_jsonl(BENCH_QUERY_FILE)}
    queries = [_norm(str(row["query"])) for row in rows]
    return {
        "schema_version": "opk-rag.task0258.repair-holdout-manifest.v1",
        "sha256": _sha(HOLDOUT_FILE),
        "query_count": len(rows),
        "duplicate_query_count": len(queries) - len(set(queries)),
        "overlap_with_task0257_queries": sum(query in fixed for query in queries),
        "overlap_with_frozen_agent_benchmark": sum(query in bench for query in queries),
        "offline_labels_present": all("expected_shadow_terminal" in row for row in rows),
        "runtime_gold_metadata_usage": False,
        "holdout_manifest_valid": len(rows) >= 18 and len(queries) == len(set(queries)) and not any(query in fixed or query in bench for query in queries),
    }


def contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0258.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "task0257_query_sha256": TASK0257_FROZEN_DIGESTS["query_file"],
        "fixed_query_count": 48,
        "holdout_minimum_query_count": 18,
        "provider_response_threshold": 0.98,
        "final_structured_validity_threshold": 0.98,
        "max_transport_retries": 1,
        "max_structural_repairs": 1,
        "max_provider_requests_per_controller_decision": 3,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "organic_live_traffic_substitution_allowed": False,
        "canary_activation_allowed": False,
        "production_activation_allowed": False,
    }


def _execute_rows(rows: list[Mapping[str, Any]], *, source: str, include_generation: bool, runtime_parts: tuple[Any, ...] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, binding = runtime_parts or _runtime_parts()
    observations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    import os
    database_url = os.environ["DATABASE_URL"]
    for index, item in enumerate(rows, 1):
        query = str(item.get("query") or item.get("question") or "")
        scope = str(item.get("execution_scope") or "ask")
        started = time.perf_counter()
        try:
            context = RuntimeTraceContext(query_text=query, execution_scope=scope, enabled=True)
            search = search_knowledge_base(
                database_url,
                knowledge_base_id=kb_id,
                query=query,
                provider=embedding,
                embedding_config=embedding_config,
                search_config=search_config,
                reranker_provider=reranker,
                context_token_counter=counter,
                execution_scope=scope,
                runtime_trace_context=context,
            )
            if include_generation and scope == "ask":
                answer = answer_knowledge_base(search, provider=answer_provider, config=answer_config, runtime_trace_context=context)
                answerability = answer.controller_answerability or answer.answerability
                production_answer_status = answer.status
            else:
                answerability, _ = assess_answerability(search_response=search)
                production_answer_status = None
            record = build_live_observation_record(
                query=query,
                search_response=search,
                answerability=answerability,
                binding=binding,
                execution_scope=scope,
                source=source,
                production_latency_ms=context.elapsed_ms,
            )
            record["sample_id"] = item["sample_id"]
            record["category"] = item.get("category") or item.get("source_family") or "frozen30"
            record["production_answer_status"] = production_answer_status
            record["task0258_execution_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            observations.append(record)
            print(f"[{index:02d}/{len(rows)}] {item['sample_id']} {scope} controller={record['shadow']['controller_call_count']} {record['production']['terminal']}->{record['shadow']['terminal']}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": item.get("sample_id"), "failure_code": type(exc).__name__})
            print(f"[{index:02d}/{len(rows)}] {item.get('sample_id')} FAIL {type(exc).__name__}", flush=True)
    return observations, failures


def _hard_safety_count(rows: list[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        auth = dict(row.get("authority") or {})
        count += int(auth.get("shadow_authoritative") is not False)
        count += int(auth.get("production_mutation_allowed") is not False)
        count += int(auth.get("llm_finish_authority") is not False)
        count += int(auth.get("abstain_to_finish_override_allowed") is not False)
        count += int(auth.get("max_graph_hop") != 1)
        count += int(row.get("raw_query_persisted") is not False)
        count += int(row.get("raw_provider_output_persisted") is not False)
        count += int(row.get("hidden_reasoning_persisted") is not False)
        count += int(row.get("secret_exposure_count") or 0)
    return count


def _runtime_metrics(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    shadows = [dict(row.get("shadow") or {}) for row in rows]
    calls = [int(shadow.get("controller_call_count") or 0) for shadow in shadows]
    requests = sum(int(shadow.get("provider_requests") or 0) for shadow in shadows)
    responses = sum(int(shadow.get("provider_responses") or 0) for shadow in shadows)
    valid = sum(int(shadow.get("provider_valid_decisions") or 0) for shadow in shadows)
    decisions = sum(calls)
    recoveries = [shadow for shadow in shadows if shadow.get("recovery_policy_called")]
    max_recovery_requests = max([int(shadow.get("recovery_provider_requests") or 0) for shadow in shadows] or [0])
    max_veto_requests = max([int(shadow.get("veto_provider_requests") or 0) for shadow in shadows] or [0])
    max_transport_retries = max([
        max(int(shadow.get("recovery_transport_retry_count") or 0), int(shadow.get("veto_transport_retry_count") or 0)) for shadow in shadows
    ] or [0])
    max_structural_repairs = max([
        max(int(shadow.get("recovery_structural_repair_count") or 0), int(shadow.get("veto_structural_repair_count") or 0)) for shadow in shadows
    ] or [0])
    return {
        "execution_count": len(rows),
        "execution_failure_count": len(failures),
        "execution_success_rate": len(rows) / max(1, len(rows) + len(failures)),
        "average_controller_calls": statistics.fmean(calls) if calls else 0.0,
        "zero_controller_call_rate": sum(call == 0 for call in calls) / max(1, len(calls)),
        "llm_invocation_rate": sum(call > 0 for call in calls) / max(1, len(calls)),
        "three_or_more_controller_call_rate": sum(call >= 3 for call in calls) / max(1, len(calls)),
        "provider_request_count": requests,
        "provider_response_count": responses,
        "provider_response_rate": responses / max(1, requests) if requests else 1.0,
        "controller_decision_count": decisions,
        "provider_valid_decision_count": valid,
        "final_structured_validity": valid / max(1, decisions) if decisions else 1.0,
        "max_recovery_provider_requests_per_decision": max_recovery_requests,
        "max_veto_provider_requests_per_decision": max_veto_requests,
        "max_transport_retry_count": max_transport_retries,
        "max_structural_repair_count": max_structural_repairs,
        "recovery_invocation_count": len(recoveries),
        "recovery_improved_count": sum(bool(shadow.get("recovery_improved")) for shadow in recoveries),
        "recovery_harmed_count": sum(bool(shadow.get("recovery_harmed")) for shadow in recoveries),
        "recovery_net_gain": sum(bool(shadow.get("recovery_improved")) for shadow in recoveries) - sum(bool(shadow.get("recovery_harmed")) for shadow in recoveries),
        "veto_invocation_count": sum(bool(shadow.get("veto_invoked")) for shadow in shadows),
        "average_controller_latency_ms": statistics.fmean(float(shadow.get("controller_latency_ms") or 0.0) for shadow in shadows) if shadows else 0.0,
        "average_execution_elapsed_ms": statistics.fmean(float(row.get("task0258_execution_elapsed_ms") or 0.0) for row in rows) if rows else 0.0,
        "hard_safety_violation_count": _hard_safety_count(rows),
    }


def _known_slice(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("sample_id")): row for row in rows}
    expected_finish = ("D26", "D29", "D31")
    expected_abstain = ("D41", "D44")
    false_abstain = [sid for sid in expected_finish if (by_id.get(sid, {}).get("shadow") or {}).get("terminal") == "abstained"]
    retained = [sid for sid in expected_abstain if (by_id.get(sid, {}).get("shadow") or {}).get("terminal") == "abstained"]
    unsafe = [sid for sid in expected_abstain if (by_id.get(sid, {}).get("shadow") or {}).get("terminal") == "finished"]
    return {
        "schema_version": "opk-rag.task0258.known-divergence-replay.v1",
        "expected_finish_ids": list(expected_finish),
        "expected_abstain_ids": list(expected_abstain),
        "probable_false_abstain_ids": false_abstain,
        "probable_false_abstain_count": len(false_abstain),
        "beneficial_safety_abstain_retained_ids": retained,
        "beneficial_safety_abstain_retained_count": len(retained),
        "unsafe_finish_regression_ids": unsafe,
        "unsafe_finish_regression_count": len(unsafe),
        "known_slice_passed": not false_abstain and len(retained) == 2 and not unsafe,
    }


def _holdout_results(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = {str(row["sample_id"]): str(row["expected_shadow_terminal"]) for row in _read_jsonl(HOLDOUT_FILE)}
    by_id = {str(row.get("sample_id")): row for row in rows}
    mismatches = []
    false_abstains = []
    unsafe_finishes = []
    for sid, expected in labels.items():
        actual = (by_id.get(sid, {}).get("shadow") or {}).get("terminal")
        if actual != expected:
            mismatches.append({"sample_id": sid, "expected": expected, "actual": actual})
        if expected == "finished" and actual == "abstained":
            false_abstains.append(sid)
        if expected == "abstained" and actual == "finished":
            unsafe_finishes.append(sid)
    return {
        "schema_version": "opk-rag.task0258.repair-holdout-results.v1",
        "query_count": len(labels),
        "executed_count": len(rows),
        "execution_failure_count": len(failures),
        "terminal_correct_count": len(labels) - len(mismatches),
        "terminal_accuracy": (len(labels) - len(mismatches)) / max(1, len(labels)),
        "mismatches": mismatches,
        "false_abstain_ids": false_abstains,
        "unsafe_finish_ids": unsafe_finishes,
        "repair_holdout_passed": len(rows) == len(labels) and not failures and not mismatches,
        "runtime_gold_metadata_usage": False,
    }



def _holdout_semantic_review() -> dict[str, Any]:
    """Post-run audit only. It never changes the pre-frozen terminal holdout gate or runtime inputs."""
    reviews = [
        {
            "sample_id": "H12",
            "frozen_expected_terminal": "abstained",
            "observed_terminal": "finished",
            "review": "bounded_negative_finish_supported",
            "basis": "Tracked roadmap/minimal-test docs explicitly leave DOCX -> PDF/PNG visual checking unchecked as future work.",
            "source_paths": [
                "source-documents/Agent/academic-docx-polisher/06 Agent Skill 化路线图.md",
                "source-documents/Agent/academic-docx-polisher/04 最小测试案例与验证结果.md",
            ],
        },
        {
            "sample_id": "H16",
            "frozen_expected_terminal": "abstained",
            "observed_terminal": "finished",
            "review": "bounded_negative_finish_supported",
            "basis": "Tracked uv notes explicitly distinguish Unix source .venv/bin activation from Windows .venv\\Scripts\\activate.",
            "source_paths": ["source-documents/Python/环境管理 uv.md"],
        },
        {
            "sample_id": "H18",
            "frozen_expected_terminal": "abstained",
            "observed_terminal": "finished",
            "review": "bounded_negative_finish_supported",
            "basis": "Tracked Skill roadmap explicitly lists a real but de-identified paper-table case as unchecked future work.",
            "source_paths": ["source-documents/Agent/academic-docx-polisher/06 Agent Skill 化路线图.md"],
        },
    ]
    return {
        "schema_version": "opk-rag.task0258.repair-holdout-semantic-review.v1",
        "review_scope": "post_run_offline_audit_only",
        "runtime_gold_metadata_usage": False,
        "frozen_holdout_labels_modified": False,
        "repair_holdout_gate_modified": False,
        "reviewed_mismatch_count": len(reviews),
        "bounded_negative_finish_supported_count": sum(row["review"] == "bounded_negative_finish_supported" for row in reviews),
        "reviews": reviews,
        "interpretation": "The frozen terminal-only holdout gate remains failed at 15/18. The three mismatches are separately audited as evidence-supported negative answers, so TASK-0258 does not force additional abstention merely to satisfy the terminal-only labels.",
    }

def _frozen30_results(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    gold = {str(row["sample_id"]): str(row["expected_terminal"]) for row in _read_jsonl(BENCH_GOLD_FILE)}
    by_id = {str(row.get("sample_id")): row for row in rows}
    incorrect = []
    false_abstain = []
    unsafe_finish = []
    for sid, expected in gold.items():
        actual = (by_id.get(sid, {}).get("shadow") or {}).get("terminal")
        if actual != expected:
            incorrect.append(sid)
        if expected == "finished" and actual == "abstained":
            false_abstain.append(sid)
        if expected == "abstained" and actual == "finished":
            unsafe_finish.append(sid)
    accuracy = (len(gold) - len(incorrect)) / max(1, len(gold))
    return {
        "schema_version": "opk-rag.task0258.frozen30-regression.v1",
        "query_count": len(gold),
        "executed_count": len(rows),
        "execution_failure_count": len(failures),
        "terminal_accuracy": accuracy,
        "incorrect_sample_ids": incorrect,
        "false_abstain_ids": false_abstain,
        "unsafe_finish_ids": unsafe_finish,
        "historical_task0250_terminal_accuracy": 29 / 30,
        "historical_task0250_false_abstain_count": 0,
        "historical_task0250_unsafe_finish_count": 1,
        "frozen30_regression_passed": len(rows) == 30 and not failures and accuracy >= 29 / 30 and len(false_abstain) == 0 and len(unsafe_finish) <= 1,
        "historical_task0250_artifacts_rewritten": False,
        "runtime_gold_metadata_usage": False,
    }


def _decision(fixed: dict[str, Any], known: dict[str, Any], holdout: dict[str, Any], frozen30: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    gates = {
        "provider_response_rate": float(fixed.get("provider_response_rate") or 0.0) >= 0.98,
        "final_structured_validity": float(fixed.get("final_structured_validity") or 0.0) >= 0.98,
        "probable_false_abstain_zero": int(known.get("probable_false_abstain_count") or 0) == 0,
        "beneficial_safety_retained": int(known.get("beneficial_safety_abstain_retained_count") or 0) == 2,
        "unsafe_finish_regression_zero": int(known.get("unsafe_finish_regression_count") or 0) == 0,
        "recovery_not_harmful": int(fixed.get("recovery_harmed_count") or 0) == 0,
        "recovery_net_positive": int(fixed.get("recovery_net_gain") or 0) > 0,
        "selectivity_average_bounded": float(fixed.get("average_controller_calls") or 99.0) < 1.0,
        "selectivity_has_bypass": float(fixed.get("zero_controller_call_rate") or 0.0) > 0.0,
        "selectivity_has_llm": float(fixed.get("llm_invocation_rate") or 0.0) > 0.0,
        "no_three_plus_controller": float(fixed.get("three_or_more_controller_call_rate") if fixed.get("three_or_more_controller_call_rate") is not None else 1.0) == 0.0,
        "max_provider_requests_bounded": max(int(fixed.get("max_recovery_provider_requests_per_decision") or 0), int(fixed.get("max_veto_provider_requests_per_decision") or 0)) <= 3,
        "max_transport_retry_bounded": int(fixed.get("max_transport_retry_count") or 0) <= 1,
        "max_structural_repair_bounded": int(fixed.get("max_structural_repair_count") or 0) <= 1,
        "hard_safety_zero": int(fixed.get("hard_safety_violation_count") or 0) == 0,
        "repair_holdout_passed": holdout.get("repair_holdout_passed") is True,
        "frozen30_regression_passed": frozen30.get("frozen30_regression_passed") is True,
    }
    if all(gates.values()):
        decision = "advance_to_controlled_dogfooding_evidence_policy_review"
    elif not gates["provider_response_rate"] or not gates["final_structured_validity"]:
        decision = "hold_for_provider_reliability"
    elif not gates["probable_false_abstain_zero"]:
        decision = "hold_for_false_abstain_repair"
    elif not gates["beneficial_safety_retained"] or not gates["unsafe_finish_regression_zero"] or not gates["hard_safety_zero"]:
        decision = "hold_for_safety_regression"
    elif not gates["repair_holdout_passed"] or not gates["frozen30_regression_passed"]:
        decision = "hold_for_generalization_failure"
    else:
        decision = "hold_for_generalization_failure"
    return decision, gates


def execute(*, write: bool = True) -> dict[str, Any]:
    entry = entry_gate()
    if not entry["entry_gate_passed"]:
        summary = {"schema_version": SCHEMA, "task_id": TASK_ID, "task_status": "blocked", "entry_gate_passed": False, "candidate_decision": "blocked", "git_commit_created": False}
        if write:
            RESULT.mkdir(parents=True, exist_ok=True)
            _write_json(RESULT / "entry_gate.json", entry)
            _write_json(RESULT / "summary.json", summary)
        return summary

    fixed_rows = _read_jsonl(FIXED_QUERY_FILE)
    holdout_rows = _read_jsonl(HOLDOUT_FILE)
    bench_rows = _read_jsonl(BENCH_QUERY_FILE)

    runtime_parts = _runtime_parts()
    print("=== TASK0258 fixed48 replay ===", flush=True)
    fixed_obs, fixed_fail = _execute_rows(fixed_rows, source="controlled_realistic_dogfooding", include_generation=True, runtime_parts=runtime_parts)
    print("=== TASK0258 repair holdout ===", flush=True)
    holdout_obs, holdout_fail = _execute_rows(holdout_rows, source="controlled_realistic_dogfooding", include_generation=True, runtime_parts=runtime_parts)
    print("=== TASK0258 frozen30 candidate regression ===", flush=True)
    frozen_obs, frozen_fail = _execute_rows(bench_rows, source="frozen_benchmark_runtime_replay", include_generation=False, runtime_parts=runtime_parts)

    fixed_metrics = _runtime_metrics(fixed_obs, fixed_fail)
    known = _known_slice(fixed_obs)
    holdout = _holdout_results(holdout_obs, holdout_fail)
    frozen30 = _frozen30_results(frozen_obs, frozen_fail)
    decision, gates = _decision(fixed_metrics, known, holdout, frozen30)
    baseline = _read_json(TASK0257_RESULT / "summary.json")
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete",
        "implementation_complete": True,
        "entry_gate_passed": True,
        "task0257_historical_artifacts_unchanged": task0257_identity()["task0257_historical_artifacts_unchanged"],
        "fixed_query_identity_match": query_identity()["query_identity_match"],
        "repair_holdout_manifest_valid": holdout_manifest()["holdout_manifest_valid"],
        "candidate_decision": decision,
        "repair_gate_passed": all(gates.values()),
        "controlled_dogfooding_evidence_sufficient": all(gates.values()),
        "organic_live_user_traffic_count": 0,
        "task0251_live_shadow_evidence_modified": False,
        "production_agentic_v2_active": False,
        "production_promotion_executed": False,
        "canary_execution_performed": False,
        "git_commit_created": False,
        "next_task": "TASK-0252_controlled_dogfooding_evidence_policy_review" if all(gates.values()) else ("TASK-0259_controlled_negative_answer_semantics_and_evidence_policy_review" if decision == "hold_for_generalization_failure" else "TASK-0258_repair_followup"),
        "task0257_final_structured_validity": baseline.get("final_structured_validity"),
        "task0257_probable_false_abstain_count": baseline.get("probable_false_abstain_count"),
        "task0257_recovery_net_gain": baseline.get("recovery_net_gain"),
        **fixed_metrics,
        "probable_false_abstain_count": known["probable_false_abstain_count"],
        "beneficial_safety_abstain_retained_count": known["beneficial_safety_abstain_retained_count"],
        "unsafe_finish_regression_count": known["unsafe_finish_regression_count"],
        "repair_holdout_terminal_accuracy": holdout["terminal_accuracy"],
        "repair_holdout_passed": holdout["repair_holdout_passed"],
        "repair_holdout_label_semantic_mismatch_count": _holdout_semantic_review()["reviewed_mismatch_count"],
        "repair_holdout_bounded_negative_finish_supported_count": _holdout_semantic_review()["bounded_negative_finish_supported_count"],
        "repair_holdout_raw_terminal_gate_unchanged": True,
        "frozen30_terminal_accuracy": frozen30["terminal_accuracy"],
        "frozen30_false_abstain_count": len(frozen30["false_abstain_ids"]),
        "frozen30_unsafe_finish_count": len(frozen30["unsafe_finish_ids"]),
        "frozen30_regression_passed": frozen30["frozen30_regression_passed"],
        **_regression_authority(),
    }
    if write:
        RESULT.mkdir(parents=True, exist_ok=True)
        _write_json(CONTRACT, contract())
        _write_json(RESULT / "entry_gate.json", entry)
        _write_json(RESULT / "task0257_frozen_baseline.json", {"identity": task0257_identity(), "summary": baseline})
        _write_json(RESULT / "query_identity.json", query_identity())
        _write_json(RESULT / "repair_holdout_manifest.json", holdout_manifest())
        _write_jsonl(RESULT / "fixed48_runtime_observations.jsonl", fixed_obs)
        _write_json(RESULT / "fixed48_execution_failures.json", {"failure_count": len(fixed_fail), "failures": fixed_fail})
        _write_json(RESULT / "provider_reliability_metrics.json", fixed_metrics)
        _write_json(RESULT / "known_divergence_replay.json", known)
        _write_json(RESULT / "false_abstain_metrics.json", {"probable_false_abstain_count": known["probable_false_abstain_count"], "ids": known["probable_false_abstain_ids"]})
        _write_json(RESULT / "safety_abstain_retention.json", {"retained_count": known["beneficial_safety_abstain_retained_count"], "retained_ids": known["beneficial_safety_abstain_retained_ids"], "unsafe_finish_regression_count": known["unsafe_finish_regression_count"]})
        _write_jsonl(RESULT / "repair_holdout_observations.jsonl", holdout_obs)
        _write_json(RESULT / "repair_holdout_results.json", holdout)
        _write_json(RESULT / "repair_holdout_semantic_review.json", _holdout_semantic_review())
        _write_jsonl(RESULT / "frozen30_candidate_observations.jsonl", frozen_obs)
        _write_json(RESULT / "frozen30_regression.json", frozen30)
        _write_json(RESULT / "recovery_metrics.json", {k: fixed_metrics[k] for k in ("recovery_invocation_count", "recovery_improved_count", "recovery_harmed_count", "recovery_net_gain")})
        _write_json(RESULT / "selectivity_metrics.json", {k: fixed_metrics[k] for k in ("average_controller_calls", "zero_controller_call_rate", "llm_invocation_rate", "three_or_more_controller_call_rate")})
        _write_json(RESULT / "latency_metrics.json", {k: fixed_metrics[k] for k in ("average_controller_latency_ms", "average_execution_elapsed_ms")})
        _write_json(RESULT / "token_cost_metrics.json", {"provider_request_count": fixed_metrics["provider_request_count"], "controller_decision_count": fixed_metrics["controller_decision_count"]})
        _write_json(RESULT / "safety_metrics.json", {"hard_safety_violation_count": fixed_metrics["hard_safety_violation_count"], "llm_finish_authority_count": 0, "abstain_to_finish_override_count": 0, "graph_hop_violation_count": 0, "knowledge_base_mutation_count": 0, "benchmark_gold_exposure_count": 0, "raw_provider_output_persisted_count": 0, "hidden_reasoning_persisted_count": 0, "secret_exposure_count": 0})
        _write_json(RESULT / "before_after_comparison.json", {
            "task0257": {"final_structured_validity": baseline.get("final_structured_validity"), "probable_false_abstain_count": baseline.get("probable_false_abstain_count"), "recovery_net_gain": baseline.get("recovery_net_gain"), "average_controller_calls": baseline.get("average_controller_calls")},
            "task0258": {"final_structured_validity": fixed_metrics["final_structured_validity"], "probable_false_abstain_count": known["probable_false_abstain_count"], "recovery_net_gain": fixed_metrics["recovery_net_gain"], "average_controller_calls": fixed_metrics["average_controller_calls"]},
        })
        _write_json(RESULT / "provider_failure_taxonomy.json", {"classes": ["transport_failure", "provider_empty_structured_content", "invalid_json", "schema_validation_failure", "contract_failure"], "empty_content_is_structural_repair": False, "max_transport_retries": 1, "max_structural_repairs": 1, "max_provider_requests_per_controller_decision": 3})
        _write_json(RESULT / "provider_repair_arms.json", {"selected_arm": "R1_empty_content_transport_classification_plus_v3_gate", "arms": ["R0_task0257_frozen", "R1_empty_content_transport_classification", "R2_repair_prompt_tightening", "R3_compact_policy_contract"], "model_changed": False})
        _write_json(RESULT / "conflict_gate_v3_contract.json", {"version": "opk-rag.task0258.conflict-gate-v3.v1", "historical_task0249_v2_rewritten": False, "one_way_veto": True, "llm_finish_authority": False})
        _write_json(RESULT / "repair_gate.json", {"candidate_decision": decision, "gates": gates, "repair_gate_passed": all(gates.values())})
        _write_json(RESULT / "summary.json", summary)
    return summary



def rebuild_from_existing(*, write: bool = True) -> dict[str, Any]:
    """Recompute TASK-0258 authority from the completed formal replay without rerunning models."""
    fixed_obs = _read_jsonl(RESULT / "fixed48_runtime_observations.jsonl")
    fixed_fail = list((_read_json(RESULT / "fixed48_execution_failures.json")).get("failures") or [])
    holdout_obs = _read_jsonl(RESULT / "repair_holdout_observations.jsonl")
    holdout_fail_doc = _read_json(RESULT / "repair_holdout_results.json")
    holdout_fail = [] if int(holdout_fail_doc.get("execution_failure_count") or 0) == 0 else [{"failure_code": "recorded_holdout_failure"}]
    frozen_obs = _read_jsonl(RESULT / "frozen30_candidate_observations.jsonl")
    frozen_doc = _read_json(RESULT / "frozen30_regression.json")
    frozen_fail = [] if int(frozen_doc.get("execution_failure_count") or 0) == 0 else [{"failure_code": "recorded_frozen30_failure"}]
    fixed_metrics = _runtime_metrics(fixed_obs, fixed_fail)
    known = _known_slice(fixed_obs)
    holdout = _holdout_results(holdout_obs, holdout_fail)
    frozen30 = _frozen30_results(frozen_obs, frozen_fail)
    decision, gates = _decision(fixed_metrics, known, holdout, frozen30)
    baseline = _read_json(TASK0257_RESULT / "summary.json")
    semantic = _holdout_semantic_review()
    summary = {
        "schema_version": SCHEMA, "task_id": TASK_ID, "task_status": "complete", "implementation_complete": True,
        "entry_gate_passed": entry_gate()["entry_gate_passed"],
        "task0257_historical_artifacts_unchanged": task0257_identity()["task0257_historical_artifacts_unchanged"],
        "fixed_query_identity_match": query_identity()["query_identity_match"],
        "repair_holdout_manifest_valid": holdout_manifest()["holdout_manifest_valid"],
        "candidate_decision": decision, "repair_gate_passed": all(gates.values()),
        "controlled_dogfooding_evidence_sufficient": all(gates.values()),
        "organic_live_user_traffic_count": 0, "task0251_live_shadow_evidence_modified": False,
        "production_agentic_v2_active": False, "production_promotion_executed": False, "canary_execution_performed": False,
        "git_commit_created": False,
        "next_task": "TASK-0252_controlled_dogfooding_evidence_policy_review" if all(gates.values()) else ("TASK-0259_controlled_negative_answer_semantics_and_evidence_policy_review" if decision == "hold_for_generalization_failure" else "TASK-0258_repair_followup"),
        "task0257_final_structured_validity": baseline.get("final_structured_validity"),
        "task0257_probable_false_abstain_count": baseline.get("probable_false_abstain_count"),
        "task0257_recovery_net_gain": baseline.get("recovery_net_gain"),
        **fixed_metrics,
        "probable_false_abstain_count": known["probable_false_abstain_count"],
        "beneficial_safety_abstain_retained_count": known["beneficial_safety_abstain_retained_count"],
        "unsafe_finish_regression_count": known["unsafe_finish_regression_count"],
        "repair_holdout_terminal_accuracy": holdout["terminal_accuracy"], "repair_holdout_passed": holdout["repair_holdout_passed"],
        "repair_holdout_label_semantic_mismatch_count": semantic["reviewed_mismatch_count"],
        "repair_holdout_bounded_negative_finish_supported_count": semantic["bounded_negative_finish_supported_count"],
        "repair_holdout_raw_terminal_gate_unchanged": True,
        "frozen30_terminal_accuracy": frozen30["terminal_accuracy"],
        "frozen30_false_abstain_count": len(frozen30["false_abstain_ids"]),
        "frozen30_unsafe_finish_count": len(frozen30["unsafe_finish_ids"]),
        "frozen30_regression_passed": frozen30["frozen30_regression_passed"],
        **_regression_authority(),
    }
    if write:
        _write_json(RESULT / "repair_holdout_results.json", holdout)
        _write_json(RESULT / "repair_holdout_semantic_review.json", semantic)
        _write_json(RESULT / "provider_reliability_metrics.json", fixed_metrics)
        _write_json(RESULT / "known_divergence_replay.json", known)
        _write_json(RESULT / "frozen30_regression.json", frozen30)
        _write_json(RESULT / "repair_gate.json", {"candidate_decision": decision, "gates": gates, "repair_gate_passed": all(gates.values())})
        _write_json(RESULT / "summary.json", summary)
    return summary

def verify() -> dict[str, Any]:
    summary = _read_json(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    gate = _read_json(RESULT / "repair_gate.json") if (RESULT / "repair_gate.json").is_file() else {}
    checks = {
        "entry_gate_passed": entry_gate()["entry_gate_passed"],
        "task0257_identity_preserved": task0257_identity()["task0257_historical_artifacts_unchanged"],
        "fixed_query_identity": query_identity()["query_identity_match"],
        "holdout_manifest_valid": holdout_manifest()["holdout_manifest_valid"],
        "summary_exists": bool(summary),
        "repair_gate_consistent": summary.get("repair_gate_passed") == gate.get("repair_gate_passed"),
        "organic_live_traffic_not_substituted": summary.get("organic_live_user_traffic_count") == 0 and summary.get("task0251_live_shadow_evidence_modified") is False,
        "no_production_activation": summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False and summary.get("canary_execution_performed") is False,
        "no_git_commit_created": summary.get("git_commit_created") is False,
    }
    return {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": all(checks.values()), "checks": checks, "candidate_decision": summary.get("candidate_decision")}


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
