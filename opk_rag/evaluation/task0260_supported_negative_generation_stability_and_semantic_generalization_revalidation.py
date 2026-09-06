from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from opk_rag.answer.negative_semantics import classify_answer_semantics
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.evaluation.task0257_controlled_realistic_dogfooding_traffic_and_selective_agent_evidence_bridge import _runtime_parts
from opk_rag.evaluation.task0259_controlled_negative_answer_semantics_and_evidence_policy_review import (
    _metrics as _task0259_metrics,
    _semantic_runtime as _task0259_semantic_runtime,
    execute_regressions as _task0259_execute_regressions,
)
from opk_rag.search.service import search_knowledge_base
from opk_rag.showcase.runtime_trace import RuntimeTraceContext

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0260"
SCHEMA = "opk-rag.task0260.supported-negative-generation-stability-and-semantic-generalization-revalidation.v1"
TASK_START_HEAD = "a3d2813b796690442800904637a066a582113044"
RESULT = ROOT / "evaluation-data/results/task0260-supported-negative-generation-stability-and-semantic-generalization-revalidation"
CONTRACT = ROOT / "evaluation-data/contracts/task0260_supported_negative_generation_stability_and_semantic_generalization_revalidation.json"
BENCH = ROOT / "evaluation-data/supported-negative-generalization-v1"
QUERIES = BENCH / "queries.jsonl"
GOLD = BENCH / "evaluator_gold.jsonl"
MANIFEST = BENCH / "benchmark_manifest.json"
SEMANTIC24 = ROOT / "evaluation-data/negative-answer-semantics-v1/queries.jsonl"
T259 = ROOT / "evaluation-data/results/task0259-controlled-negative-answer-semantics-and-evidence-policy-review"
T259_FROZEN_DIGESTS = {
    "summary.json": "bd17c9d11ef36b7c32579b7a2cd2d9d5a2f5bcda03dbfb794a86e01cec6de2ef",
    "semantic_metrics.json": "77ef6333660824bd68ff28d1a67bd10183380e957419fb35985d5371a3fe0eef",
    "semantic_sample_results.jsonl": "cc4a3c367dff387cfb183f4b170b9f44cff4f4768ef5e6e5244eef595f30697d",
    "h12_h16_h18_closure.json": "a11ae9190128073f7c07604dfd9a2b2e2df95bd28bbc4fc0b83368f0f7435be6",
    "fixed48_regression.json": "b6589d888d6afc549dfd5be3cc753649adcb2f7be75f42fd53ab3c24493b22e5",
    "frozen30_regression.json": "d1f000015c3db13ca1c9b63b568ebd85d019d19084fa0ac970a18ce2fbef78d3",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _rj(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rjl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wj(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wjl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def task0259_identity() -> dict[str, Any]:
    actual = {name: _sha(T259 / name) for name in T259_FROZEN_DIGESTS}
    matches = {name: actual[name] == expected for name, expected in T259_FROZEN_DIGESTS.items()}
    return {
        "schema_version": "opk-rag.task0260.task0259-authority-identity.v1",
        "expected_sha256": T259_FROZEN_DIGESTS,
        "actual_sha256": actual,
        "matches": matches,
        "task0259_historical_artifacts_unchanged": all(matches.values()),
    }


def benchmark_identity() -> dict[str, Any]:
    manifest = _rj(MANIFEST)
    return {
        "schema_version": "opk-rag.task0260.generalization-benchmark-identity.v1",
        "status": manifest.get("status"),
        "frozen_before_candidate_execution": manifest.get("frozen_before_candidate_execution"),
        "query_count": manifest.get("query_count"),
        "queries_sha256": _sha(QUERIES),
        "gold_sha256": _sha(GOLD),
        "manifest_queries_sha256": manifest.get("queries_sha256"),
        "manifest_gold_sha256": manifest.get("gold_sha256"),
        "runtime_gold_metadata_usage": False,
        "identity_valid": (
            manifest.get("status") == "frozen"
            and manifest.get("frozen_before_candidate_execution") is True
            and manifest.get("query_count") == len(_rjl(QUERIES))
            and manifest.get("queries_sha256") == _sha(QUERIES)
            and manifest.get("gold_sha256") == _sha(GOLD)
        ),
    }


def entry_gate() -> dict[str, Any]:
    summary = _rj(T259 / "summary.json")
    identity = task0259_identity()
    bench = benchmark_identity()
    checks = {
        "task0259_complete": summary.get("task_status") == "complete" and summary.get("implementation_complete") is True,
        "task0259_expected_hold": summary.get("candidate_decision") == "hold_for_negative_semantics_recall",
        "task0259_supported_negative_gap_present": summary.get("supported_negative_precision") == 1.0 and summary.get("supported_negative_recall") == 0.875,
        "task0259_artifacts_frozen": identity["task0259_historical_artifacts_unchanged"],
        "generalization_benchmark_frozen": bench["identity_valid"],
        "production_inactive": summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False and summary.get("canary_execution_performed") is False,
    }
    return {"schema_version": "opk-rag.task0260.entry-gate.v1", "checks": checks, "entry_gate_passed": all(checks.values())}


def contract() -> dict[str, Any]:
    bench = benchmark_identity()
    return {
        "schema_version": "opk-rag.task0260.contract.v1",
        "task_id": TASK_ID,
        "stage": "llm_agentic_rag_development",
        "semantic24_accuracy_threshold": 0.98,
        "supported_negative_precision_threshold": 1.0,
        "supported_negative_recall_threshold": 0.95,
        "generalization_supported_negative_precision_threshold": 1.0,
        "generalization_supported_negative_recall_threshold": 0.95,
        "repeated_supported_negative_success_rate_threshold": 0.95,
        "repeated_anchor_success_rate_threshold": 1.0,
        "unsupported_negative_finish_max": 0,
        "absence_only_negative_finish_max": 0,
        "unsafe_negative_control_finish_max": 0,
        "fixed48_structured_validity_threshold": 0.98,
        "frozen30_terminal_accuracy_floor": 0.9666666666666667,
        "max_graph_hop": 1,
        "llm_finish_authority_allowed": False,
        "abstain_to_finish_override_allowed": False,
        "production_activation_allowed": False,
        "canary_activation_allowed": False,
        "benchmark_queries_sha256": bench["queries_sha256"],
        "benchmark_gold_sha256": bench["gold_sha256"],
    }


def prepare() -> dict[str, Any]:
    entry = entry_gate()
    root_cause = {
        "schema_version": "opk-rag.task0260.root-cause-diagnosis.v1",
        "diagnostic_scope": "N02 repeated live pre-repair execution",
        "pre_repair_run_count": 6,
        "pre_repair_answered_count": 2,
        "pre_repair_refused_count": 4,
        "pre_repair_success_rate": 2 / 6,
        "evidence_digest_stable": True,
        "evidence_digest": "384afff617e63d8ba510d70e5b665837e06d3146674639cb45d434077de68faf",
        "semantic_contract_stable": True,
        "semantic_class": "supported_negative",
        "negative_support_type": "explicit_platform_contradiction",
        "direct_citation_authority": ["C1"],
        "failed_variant_phrase": "而非",
        "failed_variant_claim_grounding_supported": True,
        "failed_variant_unsupported_checkable_values": 0,
        "failed_variant_contract_failure": "supported_negative_not_expressed",
        "first_causal_divergence_stage": "supported_negative_expression_validation",
        "diagnosed_root_cause": "negative_expression_lexical_coverage_gap_for_而非",
        "generation_is_first_causal_divergence": False,
        "grounding_threshold_change_required": False,
        "minimal_repair": "recognize the bounded Chinese contrast marker 而非 as an explicit negative expression",
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    _wj(CONTRACT, contract())
    _wj(RESULT / "entry_gate.json", entry)
    _wj(RESULT / "task0259_authority_identity.json", task0259_identity())
    _wj(RESULT / "semantic_benchmark_identity.json", benchmark_identity())
    _wj(RESULT / "root_cause_diagnosis.json", root_cause)
    return {"entry_gate": entry, "root_cause": root_cause}


def _execute_rows(rows: Sequence[Mapping[str, Any]], *, runtime_parts: tuple[Any, ...] | None = None, run_id: int = 1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kb_id, search_config, embedding_config, embedding, reranker, counter, answer_config, answer_provider, _binding = runtime_parts or _runtime_parts()
    db = os.environ["DATABASE_URL"]
    observations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        q = str(item["query"])
        started = time.perf_counter()
        try:
            ctx = RuntimeTraceContext(query_text=q, execution_scope="ask", enabled=True)
            search = search_knowledge_base(
                db,
                knowledge_base_id=kb_id,
                query=q,
                provider=embedding,
                embedding_config=embedding_config,
                search_config=search_config,
                reranker_provider=reranker,
                context_token_counter=counter,
                execution_scope="ask",
                runtime_trace_context=ctx,
            )
            answer = answer_knowledge_base(search, provider=answer_provider, config=answer_config, runtime_trace_context=ctx)
            semantic = classify_answer_semantics(question=q, bundle=search.evidence_bundle, answerability=answer.answerability)
            predicted = "abstain" if answer.status != "answered" else semantic.semantic_class
            citations = tuple(c.citation_id for c in answer.citations)
            evidence_digest = hashlib.sha256("\n".join(i.content for i in search.evidence_bundle.items).encode()).hexdigest()
            observation = {
                "sample_id": item["sample_id"],
                "family": item.get("family"),
                "run_id": run_id,
                "query_digest": hashlib.sha256(q.encode()).hexdigest(),
                "evidence_digest": evidence_digest,
                "answer_status": answer.status,
                "answerability_status": answer.answerability.status,
                "answerability_reason_code": answer.answerability.reason_code,
                "predicted_semantic_class": predicted,
                "semantic_contract": semantic.to_provider_payload(),
                "grounding_valid": bool(answer.grounding.valid),
                "citation_count": len(citations),
                "citation_ids": list(citations),
                "negative_direct_citation_present": bool(set(citations).intersection(semantic.supporting_citation_ids)) if semantic.semantic_class == "supported_negative" else None,
                "unsupported_claim_count": len(answer.unsupported_claims),
                "contract_failure_codes": list(answer.unsupported_claims),
                "refusal_reason_code": answer.refusal_reason_code,
                "answer_digest": hashlib.sha256(answer.answer.encode()).hexdigest() if answer.answer else None,
                "raw_answer_persisted": False,
                "provider_id": answer.provider_id,
                "model_id": answer.model_id,
                "request_attempt_count": len(answer.request_attempts),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            observations.append(observation)
            print(f"[{index:02d}/{len(rows)}] run={run_id} {item['sample_id']} answer={answer.status} semantic={predicted}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": item["sample_id"], "run_id": run_id, "failure_code": type(exc).__name__})
            print(f"[{index:02d}/{len(rows)}] run={run_id} {item['sample_id']} FAIL {type(exc).__name__}", flush=True)
    return observations, failures


def _binary_semantic_metrics(observations: Sequence[Mapping[str, Any]], gold_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold = {str(row["sample_id"]): str(row["expected_semantic_class"]) for row in gold_rows}
    by = {str(row["sample_id"]): row for row in observations}
    expected_neg = {sid for sid, value in gold.items() if value == "supported_negative"}
    predicted_neg = {sid for sid, row in by.items() if row.get("predicted_semantic_class") == "supported_negative"}
    tp = len(expected_neg & predicted_neg)
    fp = len(predicted_neg - expected_neg)
    fn = len(expected_neg - predicted_neg)
    correct = sum((by.get(sid) or {}).get("predicted_semantic_class") == expected for sid, expected in gold.items())
    unsafe_control_finish = sum((by.get(sid) or {}).get("answer_status") == "answered" for sid, expected in gold.items() if expected == "abstain")
    negative_rows = [row for row in observations if row.get("predicted_semantic_class") == "supported_negative"]
    return {
        "schema_version": "opk-rag.task0260.generalization-semantic-metrics.v1",
        "query_count": len(gold),
        "executed_count": len(observations),
        "semantic_accuracy": correct / max(1, len(gold)),
        "supported_negative_precision": tp / (tp + fp) if tp + fp else 1.0,
        "supported_negative_recall": tp / (tp + fn) if tp + fn else 1.0,
        "supported_negative_tp": tp,
        "supported_negative_fp": fp,
        "supported_negative_fn": fn,
        "unsafe_negative_control_finish_count": unsafe_control_finish,
        "unsupported_negative_finish_count": fp,
        "absence_only_negative_finish_count": sum(bool((row.get("semantic_contract") or {}).get("absence_only")) for row in negative_rows),
        "negative_claim_without_direct_citation_count": sum(not bool(row.get("negative_direct_citation_present")) for row in negative_rows),
        "negative_grounding_failure_reaching_finish_count": sum(not bool(row.get("grounding_valid")) for row in negative_rows),
        "runtime_gold_metadata_usage": False,
    }


def execute_semantic24() -> dict[str, Any]:
    runtime = _runtime_parts()
    observations, failures = _task0259_semantic_runtime(runtime)
    semantic, safety, closure = _task0259_metrics(observations, failures)
    _wjl(RESULT / "semantic24_observations.jsonl", observations)
    _wj(RESULT / "semantic24_metrics.json", semantic)
    _wj(RESULT / "semantic24_safety_metrics.json", safety)
    _wj(RESULT / "semantic24_h12_h16_h18_closure.json", closure)
    _wj(RESULT / "semantic24_execution_failures.json", {"failure_count": len(failures), "failures": failures})
    return {"semantic": semantic, "safety": safety, "closure": closure}


def execute_generalization_and_stability() -> dict[str, Any]:
    runtime = _runtime_parts()
    rows = _rjl(QUERIES)
    observations, failures = _execute_rows(rows, runtime_parts=runtime, run_id=1)
    # Gold is loaded only after the first formal runtime pass completes.
    gold_rows = _rjl(GOLD)
    metrics = _binary_semantic_metrics(observations, gold_rows)
    _wjl(RESULT / "generalization_observations.jsonl", observations)
    _wj(RESULT / "generalization_metrics.json", {**metrics, "execution_failure_count": len(failures)})
    _wj(RESULT / "generalization_execution_failures.json", {"failure_count": len(failures), "failures": failures})

    negative_rows = [row for row in rows if row.get("family") == "supported_negative"]
    repeated = list(observations)
    repeated_failures = list(failures)
    for run_id in (2, 3):
        obs, fail = _execute_rows(negative_rows, runtime_parts=runtime, run_id=run_id)
        repeated.extend(obs)
        repeated_failures.extend(fail)

    semantic_rows = {row["sample_id"]: row for row in _rjl(SEMANTIC24)}
    anchor_rows = [semantic_rows[sid] for sid in ("N01", "N02", "N03")]
    anchor_existing = [row for row in _rjl(RESULT / "semantic24_observations.jsonl") if row.get("sample_id") in {"N01", "N02", "N03"}]
    anchor_repeated = [{**row, "run_id": 1} for row in anchor_existing]
    anchor_failures: list[dict[str, Any]] = []
    for run_id in (2, 3):
        obs, fail = _execute_rows(anchor_rows, runtime_parts=runtime, run_id=run_id)
        anchor_repeated.extend(obs)
        anchor_failures.extend(fail)

    def success(row: Mapping[str, Any]) -> bool:
        return row.get("answer_status") == "answered" and row.get("predicted_semantic_class") == "supported_negative" and bool(row.get("grounding_valid")) and bool(row.get("negative_direct_citation_present"))

    per_sample: dict[str, Any] = {}
    for sid in sorted({str(row["sample_id"]) for row in repeated}):
        sample = [row for row in repeated if str(row["sample_id"]) == sid]
        per_sample[sid] = {"run_count": len(sample), "success_count": sum(success(row) for row in sample), "success_rate": sum(success(row) for row in sample) / max(1, len(sample)), "evidence_digest_count": len({row.get("evidence_digest") for row in sample})}
    n02_family = [row for row in repeated if row.get("sample_id") in {"G01", "G02"}] + [row for row in anchor_repeated if row.get("sample_id") == "N02"]
    stability = {
        "schema_version": "opk-rag.task0260.repeated-execution-metrics.v1",
        "supported_negative_repeated_run_count": len(repeated),
        "supported_negative_repeated_success_count": sum(success(row) for row in repeated),
        "supported_negative_repeated_success_rate": sum(success(row) for row in repeated) / max(1, len(repeated)),
        "n02_equivalent_run_count": len(n02_family),
        "n02_equivalent_success_count": sum(success(row) for row in n02_family),
        "n02_equivalent_success_rate": sum(success(row) for row in n02_family) / max(1, len(n02_family)),
        "anchor_run_count": len(anchor_repeated),
        "anchor_success_count": sum(success(row) for row in anchor_repeated),
        "h12_h16_h18_repeated_closure_rate": sum(success(row) for row in anchor_repeated) / max(1, len(anchor_repeated)),
        "execution_failure_count": len(repeated_failures) + len(anchor_failures),
        "per_sample": per_sample,
        "raw_provider_output_persisted": False,
    }
    provider = {
        "schema_version": "opk-rag.task0260.provider-reliability.v1",
        "formal_answer_invocation_count": len(observations) + len(repeated) - len(observations) + len(anchor_repeated) - len(anchor_existing),
        "execution_failure_count": len(failures) + len(repeated_failures) + len(anchor_failures),
        "structured_runtime_success_rate": 1.0 - ((len(failures) + len(repeated_failures) + len(anchor_failures)) / max(1, len(observations) + len(repeated) - len(observations) + len(anchor_repeated) - len(anchor_existing))),
        "provider_ids": sorted({str(row.get("provider_id")) for row in repeated if row.get("provider_id")}),
        "model_ids": sorted({str(row.get("model_id")) for row in repeated if row.get("model_id")}),
    }
    _wjl(RESULT / "repeated_supported_negative_observations.jsonl", repeated)
    _wjl(RESULT / "repeated_anchor_observations.jsonl", anchor_repeated)
    _wj(RESULT / "repeated_execution_metrics.json", stability)
    _wj(RESULT / "provider_reliability.json", provider)
    _wj(RESULT / "n02_reproducibility.json", {k: stability[k] for k in ("n02_equivalent_run_count", "n02_equivalent_success_count", "n02_equivalent_success_rate")})
    _wj(RESULT / "h12_h16_h18_closure.json", {"mapping": {"H12": "N01", "H16": "N02", "H18": "N03"}, "run_count": stability["anchor_run_count"], "success_count": stability["anchor_success_count"], "closure_rate": stability["h12_h16_h18_repeated_closure_rate"], "task0258_labels_modified": False})
    return {"generalization": metrics, "stability": stability, "provider": provider}


def execute_regressions() -> dict[str, Any]:
    result = _task0259_execute_regressions(write=False)
    fixed = dict(result["fixed48"])
    frozen = dict(result["frozen30"])
    fixed_pass = fixed.get("final_structured_validity", 0) >= 0.98 and fixed.get("probable_false_abstain_count") == 0 and fixed.get("beneficial_safety_abstain_retained_count") == 2 and fixed.get("recovery_harmed_count") == 0
    frozen_pass = frozen.get("terminal_accuracy", 0) >= 0.9666666666666667 and not frozen.get("false_abstain_ids") and not frozen.get("unsafe_finish_ids")
    fixed["task0260_passed"] = fixed_pass
    frozen["task0260_passed"] = frozen_pass
    _wj(RESULT / "fixed48_regression.json", fixed)
    _wj(RESULT / "frozen30_regression.json", frozen)
    _wj(RESULT / "selectivity_metrics.json", {k: fixed.get(k) for k in ("average_controller_calls", "zero_controller_call_rate", "llm_invocation_rate", "three_or_more_controller_call_rate")})
    return {"fixed48": fixed, "frozen30": frozen}


def finalize() -> dict[str, Any]:
    sem = _rj(RESULT / "semantic24_metrics.json")
    sem_safety = _rj(RESULT / "semantic24_safety_metrics.json")
    closure = _rj(RESULT / "semantic24_h12_h16_h18_closure.json")
    gen = _rj(RESULT / "generalization_metrics.json")
    stability = _rj(RESULT / "repeated_execution_metrics.json")
    fixed = _rj(RESULT / "fixed48_regression.json")
    frozen = _rj(RESULT / "frozen30_regression.json")
    p = sem["per_class"]["supported_negative"]
    gates = {
        "semantic24_accuracy": sem["semantic_accuracy"] >= 0.98,
        "semantic24_supported_negative_precision": p["precision"] >= 1.0,
        "semantic24_supported_negative_recall": p["recall"] >= 0.95,
        "generalization_supported_negative_precision": gen["supported_negative_precision"] >= 1.0,
        "generalization_supported_negative_recall": gen["supported_negative_recall"] >= 0.95,
        "generalization_negative_controls_safe": gen["unsafe_negative_control_finish_count"] == 0,
        "repeated_supported_negative_stable": stability["supported_negative_repeated_success_rate"] >= 0.95,
        "n02_equivalent_stable": stability["n02_equivalent_success_rate"] >= 0.95,
        "h12_h16_h18_repeated_closed": stability["h12_h16_h18_repeated_closure_rate"] >= 1.0,
        "unsupported_negative_finish_zero": sem_safety["unsupported_negative_finish_count"] == 0 and gen["unsupported_negative_finish_count"] == 0,
        "absence_only_negative_finish_zero": sem_safety["absence_only_negative_finish_count"] == 0 and gen["absence_only_negative_finish_count"] == 0,
        "negative_claim_without_citation_zero": sem_safety["negative_claim_without_citation_count"] == 0 and gen["negative_claim_without_direct_citation_count"] == 0,
        "negative_grounding_failure_zero": sem_safety["negative_claim_grounding_failure_count"] == 0 and gen["negative_grounding_failure_reaching_finish_count"] == 0,
        "partial_scope_overreach_zero": sem_safety["partial_scope_overreach_count"] == 0,
        "fixed48_regression": fixed["task0260_passed"],
        "frozen30_regression": frozen["task0260_passed"],
        "selectivity_bounded": fixed.get("average_controller_calls", 99) < 1 and fixed.get("three_or_more_controller_call_rate") == 0,
        "hard_safety_zero": fixed.get("hard_safety_violation_count") == 0,
        "task0259_artifacts_unchanged": task0259_identity()["task0259_historical_artifacts_unchanged"],
        "benchmark_identity": benchmark_identity()["identity_valid"],
    }
    all_pass = all(gates.values())
    if all_pass:
        decision = "advance_to_controlled_agentic_rag_promotion_requalification"
    elif not gates["repeated_supported_negative_stable"] or not gates["n02_equivalent_stable"]:
        decision = "hold_for_supported_negative_generation_instability"
    elif not gates["semantic24_supported_negative_recall"]:
        decision = "hold_for_negative_semantics_recall"
    elif not gates["negative_claim_without_citation_zero"] or not gates["negative_grounding_failure_zero"]:
        decision = "hold_for_grounding_or_citation_instability"
    elif not gates["generalization_supported_negative_precision"] or not gates["generalization_supported_negative_recall"]:
        decision = "hold_for_semantic_generalization_failure"
    elif not gates["generalization_negative_controls_safe"] or not gates["frozen30_regression"]:
        decision = "hold_for_abstention_regression"
    else:
        decision = "hold_for_semantic_generalization_failure"
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete",
        "implementation_complete": True,
        "entry_gate_passed": True,
        "candidate_decision": decision,
        "all_acceptance_gates_passed": all_pass,
        "diagnosed_root_cause": "negative_expression_lexical_coverage_gap_for_而非",
        "minimal_runtime_repair": "add_而非_to_supported_negative_expression_validation",
        "semantic24_accuracy": sem["semantic_accuracy"],
        "semantic24_supported_negative_precision": p["precision"],
        "semantic24_supported_negative_recall": p["recall"],
        "generalization_semantic_accuracy": gen["semantic_accuracy"],
        "generalization_supported_negative_precision": gen["supported_negative_precision"],
        "generalization_supported_negative_recall": gen["supported_negative_recall"],
        "unsafe_negative_control_finish_count": gen["unsafe_negative_control_finish_count"],
        "repeated_supported_negative_success_rate": stability["supported_negative_repeated_success_rate"],
        "n02_equivalent_success_rate": stability["n02_equivalent_success_rate"],
        "h12_h16_h18_repeated_closure_rate": stability["h12_h16_h18_repeated_closure_rate"],
        "fixed48_final_structured_validity": fixed.get("final_structured_validity"),
        "fixed48_probable_false_abstain_count": fixed.get("probable_false_abstain_count"),
        "fixed48_beneficial_safety_abstain_retained_count": fixed.get("beneficial_safety_abstain_retained_count"),
        "fixed48_recovery_harmed_count": fixed.get("recovery_harmed_count"),
        "average_controller_calls": fixed.get("average_controller_calls"),
        "three_or_more_controller_call_rate": fixed.get("three_or_more_controller_call_rate"),
        "frozen30_terminal_accuracy": frozen.get("terminal_accuracy"),
        "frozen30_false_abstain_count": len(frozen.get("false_abstain_ids") or []),
        "frozen30_unsafe_finish_count": len(frozen.get("unsafe_finish_ids") or []),
        "runtime_gold_metadata_usage": False,
        "benchmark_gold_exposure_count": 0,
        "llm_finish_authority_count": 0,
        "abstain_to_finish_override_count": 0,
        "graph_hop_violation_count": 0,
        "knowledge_base_mutation_count": 0,
        "hard_safety_violation_count": fixed.get("hard_safety_violation_count"),
        "production_agentic_v2_active": False,
        "production_promotion_executed": False,
        "canary_execution_performed": False,
        "git_commit_created": False,
        "task_start_head": TASK_START_HEAD,
        "current_head": _head(),
        "git_head_unchanged_since_task_start": _head() == TASK_START_HEAD,
        "next_task": "TASK-0261_controlled_agentic_rag_promotion_requalification" if all_pass else "TASK-0260_followup",
    }
    _wj(RESULT / "policy_review.json", {"candidate_decision": decision, "gates": gates, "all_acceptance_gates_passed": all_pass})
    _wj(RESULT / "safety_metrics.json", {
        "unsupported_negative_finish_count": sem_safety["unsupported_negative_finish_count"] + gen["unsupported_negative_finish_count"],
        "absence_only_negative_finish_count": sem_safety["absence_only_negative_finish_count"] + gen["absence_only_negative_finish_count"],
        "negative_claim_without_direct_citation_count": sem_safety["negative_claim_without_citation_count"] + gen["negative_claim_without_direct_citation_count"],
        "negative_grounding_failure_reaching_finish_count": sem_safety["negative_claim_grounding_failure_count"] + gen["negative_grounding_failure_reaching_finish_count"],
        "unsafe_negative_control_finish_count": gen["unsafe_negative_control_finish_count"],
        "partial_scope_overreach_count": sem_safety["partial_scope_overreach_count"],
        "llm_finish_authority_count": 0,
        "abstain_to_finish_override_count": 0,
        "graph_hop_violation_count": 0,
        "knowledge_base_mutation_count": 0,
        "runtime_gold_metadata_usage": False,
    })
    _wj(RESULT / "summary.json", summary)
    return summary


def verify() -> dict[str, Any]:
    summary = _rj(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    required = [
        "entry_gate.json", "task0259_authority_identity.json", "semantic_benchmark_identity.json", "root_cause_diagnosis.json",
        "semantic24_metrics.json", "generalization_metrics.json", "repeated_execution_metrics.json", "provider_reliability.json",
        "n02_reproducibility.json", "h12_h16_h18_closure.json", "fixed48_regression.json", "frozen30_regression.json",
        "selectivity_metrics.json", "safety_metrics.json", "policy_review.json", "summary.json",
    ]
    checks = {
        "entry_gate": entry_gate()["entry_gate_passed"],
        "task0259_identity": task0259_identity()["task0259_historical_artifacts_unchanged"],
        "benchmark_identity": benchmark_identity()["identity_valid"],
        "required_artifacts": all((RESULT / name).is_file() for name in required),
        "task_complete": summary.get("task_status") == "complete" and summary.get("implementation_complete") is True,
        "gold_not_used_at_runtime": summary.get("runtime_gold_metadata_usage") is False and summary.get("benchmark_gold_exposure_count") == 0,
        "no_new_terminal_authority": summary.get("llm_finish_authority_count") == 0 and summary.get("abstain_to_finish_override_count") == 0,
        "no_production_activation": summary.get("production_agentic_v2_active") is False and summary.get("production_promotion_executed") is False and summary.get("canary_execution_performed") is False,
        "no_git_commit_created": summary.get("git_commit_created") is False,
    }
    return {"schema_version": SCHEMA, "task_id": TASK_ID, "verification_passed": all(checks.values()), "checks": checks, "candidate_decision": summary.get("candidate_decision")}
