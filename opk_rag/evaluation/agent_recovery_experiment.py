from __future__ import annotations

import os
import json
import shutil
import signal
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import UUID

from opk_rag.agent import GovernedQueryReformulator, OpenAICompatibleQueryReformulationProvider
from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.recovery_contracts import ALLOWED_RECOVERY_TRANSITIONS
from opk_rag.agent.recovery_loop import AgentRecoveryConfig, run_agent_recovery_loop
from opk_rag.agent.tool_registry import LiveCoreToolExecutor
from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider
from opk_rag.core_tools.serialization import answer_response_payload, answerability_payload, grounding_payload, search_response_payload
from opk_rag.core_tools.tools import assess_answerability, generate_grounded_answer, search_knowledge_base, verify_grounding
from opk_rag.core_tools.runtime import CoreRagToolRuntime
from opk_rag.embedding.config import load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.evaluation.core_rag_benchmark import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    DEV_SPLIT,
    KNOWN_REGRESSION_SPLIT,
    file_digest,
    read_json,
    read_jsonl,
    scan_paths_for_privacy,
    stable_hash,
    validate_reference_runtime,
    write_json,
    write_jsonl,
)
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.context_tokens import QwenContextTokenCounter

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0071"
CONTRACT_ID = "opk-rag.governed-single-agent-recovery-loop.v1"
EXPECTED_CONTRACT_DIGEST = "f6a0763bdcaf3a9ffb4cfcc848699fd2193cec11862c321400a9d86d1aaee2aa"
RESULT_SCHEMA = "opk-rag.task0071-formal-result-row.v1"
AGGREGATE_SCHEMA = "opk-rag.task0071-formal-aggregate.v1"
COMPARISON_SCHEMA = "opk-rag.task0071-formal-comparison.v1"
TASK0070_RESULT_ROOT = ROOT / "evaluation-data" / "results" / "task0070-generation-to-grounded-answer-bottleneck"
ANSWERING_ACTIONS = {"answer", "partial_answer", "correct_premise"}
FORMAL_FILES = {
    "c0_results.jsonl",
    "c0_aggregate.json",
    "a1_results.jsonl",
    "a1_aggregate.json",
    "comparison.json",
    "task0070_slice.json",
    "agent_traces.jsonl",
    "run_identity.json",
    "runtime_identity.json",
    "result_digests.json",
}
TERMINAL_TRACE_STATES = {"FINAL_ANSWER", "FINAL_ABSTENTION", "TERMINAL_ERROR"}


@dataclass(frozen=True)
class BenchmarkSample:
    split: str
    sample_id: str
    question_id: str
    question: str
    question_digest: str
    expected_action: str
    answerability_label: str
    question_type: str
    annotation: dict[str, Any]


@dataclass
class ReferenceRuntime:
    runtime: CoreRagToolRuntime
    runtime_identity: dict[str, Any]
    contract: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_agent_recovery_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("task_id") != TASK_ID:
        raise ValueError("task_id_mismatch")
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("contract_id_mismatch")
    if contract.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
        raise ValueError("contract_digest_mismatch")
    return contract


def load_core_rag_benchmark_split(benchmark_dir: Path, split: str) -> list[BenchmarkSample]:
    if split not in {DEV_SPLIT, KNOWN_REGRESSION_SPLIT}:
        raise ValueError(f"unsupported split: {split}")
    questions = read_jsonl(benchmark_dir / "question_set.jsonl")
    annotations = {row["sample_id"]: row for row in read_jsonl(benchmark_dir / "annotations.jsonl")}
    splits = read_json(benchmark_dir / "splits.json")
    ordered_ids = list((splits.get("roles") or {}).get(split, {}).get("sample_ids") or [])
    by_id = {row["sample_id"]: row for row in questions}
    samples = []
    for sample_id in ordered_ids:
        question = by_id[sample_id]
        annotation = annotations[sample_id]
        samples.append(
            BenchmarkSample(
                split=split,
                sample_id=sample_id,
                question_id=sample_id,
                question=question["question"],
                question_digest=question["question_digest"],
                expected_action=annotation["expected_action"],
                answerability_label=annotation["answerability_label"],
                question_type=annotation["question_type"],
                annotation=annotation,
            )
        )
    return samples


def build_reference_runtime(benchmark_dir: Path, contract: dict[str, Any], *, require_verified: bool = True) -> ReferenceRuntime:
    load_project_env(ROOT)
    validation = validate_reference_runtime()
    if require_verified and validation.get("status") != "pass":
        raise RuntimeError("reference_runtime_unverified")
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("OPK_RAG_DATABASE_URL")
    if not database_url:
        raise RuntimeError("database_url_missing")
    vault_path = Path(os.environ["OPK_RAG_VAULT_PATH"])
    embedding_config = load_embedding_config()
    embedding_provider = QwenLocalEmbeddingProvider(embedding_config)
    answer_config = load_answer_generation_config()
    answer_provider = OpenAICompatibleLocalChatProvider(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
    search_config = load_vector_search_config()
    runtime = CoreRagToolRuntime(
        database_url=database_url,
        knowledge_base_id=_knowledge_base_id_for_vault(database_url, vault_path),
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
        search_config=search_config,
        answer_provider=answer_provider,
        answer_config=answer_config,
        context_token_counter=QwenContextTokenCounter(embedding_config),
        runtime_identity={
            "task_id": TASK_ID,
            "contract_id": contract["contract_id"],
            "contract_digest": contract["contract_digest"],
            "benchmark_id": BENCHMARK_ID,
            "benchmark_version": BENCHMARK_VERSION,
        },
    )
    identity = {
        "schema_version": "opk-rag.task0071-reference-runtime-identity.v1",
        "task_id": TASK_ID,
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "reference_runtime_verified": validation.get("status") == "pass",
        "reference_runtime_status": "available" if validation.get("status") == "pass" else "unavailable",
        "formal_benchmark": True,
        "runtime_rows_generated_from_real_execution": True,
        "benchmark_hashes": {name: file_digest(benchmark_dir / name) for name in ("annotations.jsonl", "question_set.jsonl", "benchmark_manifest.json")},
        "core_runtime_validation": validation,
        "core_tool_runtime_identity": runtime.identity(),
        "agent_recovery_config": AgentRecoveryConfig().to_dict(),
    }
    return ReferenceRuntime(runtime=runtime, runtime_identity=identity, contract=contract)


def run_formal_experiment(
    *,
    benchmark_dir: Path,
    contract_path: Path,
    output_dir: Path,
    split: str = DEV_SPLIT,
    overwrite: bool = False,
    run_known_regression: bool = True,
    per_sample_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    contract = load_agent_recovery_contract(contract_path)
    if split != DEV_SPLIT:
        raise ValueError("formal entrypoint starts with development split")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        _clear_formal_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = build_reference_runtime(benchmark_dir, contract, require_verified=True)
    write_json(output_dir / "runtime_identity.json", reference.runtime_identity)
    dev = run_split(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir / "development",
        split=DEV_SPLIT,
        reference=reference,
        per_sample_timeout_seconds=per_sample_timeout_seconds,
    )
    gates = development_hard_gates(dev["c0_aggregate"], dev["a1_aggregate"], dev["comparison"], privacy_status="pending")
    kr_result: dict[str, Any] = {"known_regression_status": "not_run", "known_regression_reason": "development_hard_gate_failed"}
    if run_known_regression and all(gate["passed"] for gate in gates["gates"]):
        kr_result = run_split(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir / "known-regression",
            split=KNOWN_REGRESSION_SPLIT,
            reference=reference,
            per_sample_timeout_seconds=per_sample_timeout_seconds,
        )
        kr_result["known_regression_status"] = "completed"
    comparison = dev["comparison"]
    privacy_scan = scan_paths_for_privacy([output_dir])
    write_json(output_dir / "privacy_scan.json", {"schema_version": "opk-rag.task0071-privacy-scan.v1", "task_id": TASK_ID, "contract_digest": contract["contract_digest"], **privacy_scan})
    gates = development_hard_gates(dev["c0_aggregate"], dev["a1_aggregate"], comparison, privacy_status=privacy_scan["status"])
    promotion = build_promotion_decision(gates, comparison, known_regression=kr_result)
    write_json(output_dir / "promotion_decision.json", promotion)
    digests = result_digests(output_dir)
    write_json(output_dir / "result_digests.json", digests)
    return {
        "development": dev,
        "known_regression": kr_result,
        "promotion_decision": promotion,
        "privacy_scan": privacy_scan,
        "result_digests": digests,
    }


def run_split(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    split: str,
    reference: ReferenceRuntime,
    per_sample_timeout_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_core_rag_benchmark_split(benchmark_dir, split)
    run_id = f"task0071-{split}-{utc_now().replace(':', '').replace('-', '')}"
    run_identity = {
        "schema_version": "opk-rag.task0071-run-identity.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "split": split,
        "contract_id": reference.contract["contract_id"],
        "contract_digest": reference.contract["contract_digest"],
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "sample_ids": [sample.sample_id for sample in samples],
        "sample_count": len(samples),
        "started_at": utc_now(),
        "mode": "formal",
        "formal_benchmark": True,
        "result_validity": "formal_benchmark",
    }
    write_json(output_dir / "run_identity.json", run_identity)
    write_json(output_dir / "runtime_identity.json", reference.runtime_identity)
    c0_rows = []
    a1_rows = []
    trace_rows = []
    for sample in samples:
        c0_rows.append(_with_timeout(lambda sample=sample: run_c0_sample(sample, run_id=run_id, reference=reference), per_sample_timeout_seconds, sample, "C0", run_id, reference))
        a1 = _with_timeout(lambda sample=sample: run_a1_sample(sample, run_id=run_id, reference=reference), per_sample_timeout_seconds, sample, "A1", run_id, reference)
        a1_rows.append(a1["row"])
        trace_rows.extend(a1["traces"])
        write_jsonl(output_dir / "c0_results.jsonl", c0_rows)
        write_jsonl(output_dir / "a1_results.jsonl", a1_rows)
        write_jsonl(output_dir / "agent_traces.jsonl", trace_rows)
    c0_aggregate = aggregate_variant_results(c0_rows, samples=samples, variant="C0")
    a1_aggregate = aggregate_variant_results(a1_rows, samples=samples, variant="A1", traces=trace_rows)
    comparison = compare_c0_a1(c0_aggregate, a1_aggregate, c0_rows, a1_rows)
    task0070_slice = build_task0070_diagnostic_slice(c0_rows, a1_rows)
    write_json(output_dir / "c0_aggregate.json", c0_aggregate)
    write_json(output_dir / "a1_aggregate.json", a1_aggregate)
    write_json(output_dir / "comparison.json", comparison)
    write_json(output_dir / "task0070_slice.json", task0070_slice)
    write_json(output_dir / "result_digests.json", result_digests(output_dir))
    return {
        "run_id": run_id,
        "split": split,
        "c0_rows": c0_rows,
        "a1_rows": a1_rows,
        "traces": trace_rows,
        "c0_aggregate": c0_aggregate,
        "a1_aggregate": a1_aggregate,
        "comparison": comparison,
        "task0070_slice": task0070_slice,
    }


def run_c0_sample(sample: BenchmarkSample, *, run_id: str, reference: ReferenceRuntime) -> dict[str, Any]:
    started = time.monotonic()
    runtime = reference.runtime
    search_payload: dict[str, Any] | None = None
    answerability_result: dict[str, Any] | None = None
    generation_result: dict[str, Any] | None = None
    citation_result: dict[str, Any] | None = None
    grounding_result: dict[str, Any] | None = None
    final_action = "abstain"
    termination_reason = "answerability_abstention"
    error_type = "no_error"
    try:
        search_response, search_payload = search_knowledge_base(
            database_url=runtime.database_url,
            knowledge_base_id=runtime.knowledge_base_id,
            query=sample.question,
            provider=runtime.embedding_provider,
            embedding_config=runtime.embedding_config,
            search_config=runtime.search_config,
            reranker_provider=runtime.reranker_provider,
            context_token_counter=runtime.context_token_counter,
        )
        decision, answerability_result = assess_answerability(search_response=search_response, config=runtime.answer_config.answerability)
        if decision.answerable:
            answer, generation_result = generate_grounded_answer(search_response=search_response, answerability=decision, provider=runtime.answer_provider, config=runtime.answer_config)
            citation_result = _citation_validation(generation_result, search_payload)
            if citation_result["valid"]:
                grounding, grounding_result = verify_grounding(
                    answer_text=answer.answer,
                    citations=[citation.citation_id for citation in answer.citations],
                    evidence_bundle=search_response.evidence_bundle,
                    config=runtime.answer_config.grounding,
                )
                if grounding.valid:
                    final_action = "answer"
                    termination_reason = "grounding_valid"
                else:
                    termination_reason = "grounding_failure"
            else:
                termination_reason = "citation_failure"
        return _formal_row(
            sample,
            run_id=run_id,
            variant="C0",
            runtime_query=sample.question,
            search_payload=search_payload,
            answerability_result=answerability_result,
            generation_result=generation_result,
            citation_result=citation_result,
            grounding_result=grounding_result,
            final_action=final_action,
            termination_reason=termination_reason,
            error_type=error_type,
            latency_ms=int((time.monotonic() - started) * 1000),
            runtime_identity=reference.runtime_identity,
        )
    except Exception as exc:
        return _infrastructure_failure_row(sample, run_id=run_id, variant="C0", reference=reference, started=started, code=type(exc).__name__)


def run_a1_sample(sample: BenchmarkSample, *, run_id: str, reference: ReferenceRuntime) -> dict[str, Any]:
    started = time.monotonic()
    executor = LiveCoreToolExecutor(reference.runtime)
    reformulator = GovernedQueryReformulator(provider=OpenAICompatibleQueryReformulationProvider())
    try:
        result = run_agent_recovery_loop(question=sample.question, sample_id=sample.sample_id, run_id=f"{run_id}-{sample.sample_id}-A1", executor=executor, reformulator=reformulator)
        state = result.state.to_dict()
        row = _formal_row(
            sample,
            run_id=run_id,
            variant="A1",
            runtime_query=state.get("active_query") or sample.question,
            search_payload=(state.get("recovery_retrieval_summary") or state.get("initial_retrieval_summary")),
            answerability_result=state.get("answerability_decision"),
            generation_result=state.get("generation_result"),
            citation_result=state.get("citation_result"),
            grounding_result=state.get("grounding_result"),
            final_action=state.get("final_action") or "abstain",
            termination_reason=state.get("termination_reason") or state.get("state_name"),
            error_type="no_error" if state.get("state_name") not in {"FAILURE"} else "infrastructure_failure",
            latency_ms=int((time.monotonic() - started) * 1000),
            runtime_identity=reference.runtime_identity,
            agent_state=state,
            reformulation_trace=reformulator.call_traces,
        )
        traces = [event.to_dict() | {"run_id": run_id, "variant": "A1", "split": sample.split, "sample_id": sample.sample_id, "contract_digest": reference.contract["contract_digest"]} for event in result.trace_events]
        return {"row": row, "traces": traces}
    except Exception as exc:
        row = _infrastructure_failure_row(sample, run_id=run_id, variant="A1", reference=reference, started=started, code=type(exc).__name__)
        return {"row": row, "traces": []}


def aggregate_variant_results(rows: list[dict[str, Any]], *, samples: list[BenchmarkSample], variant: str, traces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    traces = traces or []
    expected_positive = [row for row in rows if row["expected_action"] in ANSWERING_ACTIONS]
    expected_abstain = [row for row in rows if row["expected_action"] == "abstain"]
    unsupported_answer_count = sum(row["expected_action"] == "abstain" and row["final_action"] == "answer" for row in rows)
    unsupported_claim_count = sum(_unsupported_claim_count(row) for row in rows)
    citation_failure_count = sum(row.get("citation_result", {}).get("valid") is False for row in rows)
    grounding_failure_count = sum(row.get("grounding_result", {}).get("valid") is False for row in rows if row.get("generation_invoked"))
    infra = sum(row["error_type"] == "infrastructure_failure" for row in rows)
    trace_counts = _trace_counts(traces)
    recovery_eligible = sum(((row.get("agent_state") or {}).get("recovery_eligibility") or {}).get("eligible") is True for row in rows)
    recovery_attempted = sum((row.get("agent_state") or {}).get("reformulation_attempt_count", 0) > 0 for row in rows)
    recovered_grounded = sum(row.get("recovered_grounded_answer") is True for row in rows)
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "task_id": TASK_ID,
        "variant": variant,
        "split": rows[0]["split"] if rows else None,
        "run_id": rows[0]["run_id"] if rows else None,
        "contract_digest": rows[0]["contract_digest"] if rows else EXPECTED_CONTRACT_DIGEST,
        "formal_benchmark": True,
        "sample_count": len(rows),
        "total_sample_count": len(rows),
        "terminal_result_row_count": len(rows),
        "completed_sample_count": sum(row["error_type"] != "infrastructure_failure" for row in rows),
        "successful_sample_count": sum(row["error_type"] != "infrastructure_failure" for row in rows),
        "failed_sample_count": infra,
        "infrastructure_failure_count": infra,
        "infrastructure_failure_sample_count": infra,
        "end_to_end_accuracy": _metric(sum(_is_e2e_correct(row) for row in rows), len(rows)),
        "answerability_accuracy": _metric(sum(_answerability_correct(row) for row in rows), len(rows)),
        "safe_action_accuracy": _metric(sum(_safe_action_correct(row) for row in rows), len(rows)),
        "answer_count": sum(row["final_action"] == "answer" for row in rows),
        "abstain_count": sum(row["final_action"] == "abstain" for row in rows),
        "unsupported_answer_count": unsupported_answer_count,
        "unsupported_claim_count": unsupported_claim_count,
        "citation_failure_count": citation_failure_count,
        "grounding_failure_count": grounding_failure_count,
        "over_abstention_count": sum(row["final_action"] == "abstain" for row in expected_positive),
        "retrieval": retrieval_recovery_metrics(rows, samples),
        "agent": {
            "recovery_eligible_count": recovery_eligible,
            "recovery_attempted_count": recovery_attempted,
            "recovery_skipped_count": recovery_eligible - recovery_attempted,
            "reformulation_valid_count": sum(_reformulation_valid(row) for row in rows),
            "reformulation_invalid_count": sum(_reformulation_invalid(row) for row in rows),
            "second_retrieval_count": sum((row.get("agent_state") or {}).get("retrieval_attempt_count", 0) > 1 for row in rows),
            "evidence_materially_improved_count": sum(((row.get("agent_state") or {}).get("evidence_comparison") or {}).get("outcome") == "evidence_materially_improved" for row in rows),
            "evidence_unchanged_count": sum(((row.get("agent_state") or {}).get("evidence_comparison") or {}).get("outcome") == "evidence_unchanged" for row in rows),
            "evidence_degraded_count": sum(((row.get("agent_state") or {}).get("evidence_comparison") or {}).get("outcome") == "evidence_degraded" for row in rows),
            "recovered_grounded_answer_count": recovered_grounded,
            "recovery_to_answer_conversion_rate": _metric(recovered_grounded, recovery_attempted),
            "safety_terminal_recovery_violation_count": sum(_safety_terminal_violation(row) for row in rows),
            "budget_exhaustion_count": sum(row["termination_reason"] == "agent_budget_exhausted" for row in rows),
            "invalid_transition_count": trace_counts["invalid_transition_count"],
            "mean_transitions_per_sample": _mean(trace_counts["transitions_by_sample"].values()),
            "maximum_transitions_per_sample": max(trace_counts["transitions_by_sample"].values(), default=0),
            "mean_tool_calls_per_sample": _mean(trace_counts["tool_calls_by_sample"].values()),
            "maximum_tool_calls_per_sample": max(trace_counts["tool_calls_by_sample"].values(), default=0),
            "loop_termination_accuracy": _metric(sum(bool(row["termination_reason"]) for row in rows), len(rows)),
            "all_agent_traces_valid": trace_counts["invalid_transition_count"] == 0,
        },
        "latency": latency_summary(rows, traces),
        "expected_positive_count": len(expected_positive),
        "expected_abstain_count": len(expected_abstain),
    }


def compare_c0_a1(c0: dict[str, Any], a1: dict[str, Any], c0_rows: list[dict[str, Any]], a1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "run_id": c0.get("run_id"),
        "split": c0.get("split"),
        "contract_digest": c0.get("contract_digest"),
        "sample_id_order_identical": [row["sample_id"] for row in c0_rows] == [row["sample_id"] for row in a1_rows],
        "unsupported_answer_count_delta_a1_minus_c0": a1["unsupported_answer_count"] - c0["unsupported_answer_count"],
        "unsupported_claim_count_delta_a1_minus_c0": a1["unsupported_claim_count"] - c0["unsupported_claim_count"],
        "citation_failure_count_delta_a1_minus_c0": a1["citation_failure_count"] - c0["citation_failure_count"],
        "grounding_failure_count_delta_a1_minus_c0": a1["grounding_failure_count"] - c0["grounding_failure_count"],
        "safe_action_accuracy_delta": _metric_delta(a1["safe_action_accuracy"], c0["safe_action_accuracy"]),
        "end_to_end_accuracy_delta": _metric_delta(a1["end_to_end_accuracy"], c0["end_to_end_accuracy"]),
        "recovered_grounded_answer_count": a1["agent"]["recovered_grounded_answer_count"],
        "infrastructure_failure_count": c0["infrastructure_failure_count"] + a1["infrastructure_failure_count"],
    }


def build_task0070_diagnostic_slice(c0_rows: list[dict[str, Any]], a1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = _task0070_candidate_ids()
    c0_by_id = {row["sample_id"]: row for row in c0_rows}
    a1_by_id = {row["sample_id"]: row for row in a1_rows}
    rows = []
    for sample_id in candidate_ids:
        c0 = c0_by_id.get(sample_id, {})
        a1 = a1_by_id.get(sample_id, {})
        state = a1.get("agent_state") or {}
        comparison = state.get("evidence_comparison") or {}
        rows.append(
            {
                "sample_id": sample_id,
                "c0_final_action": c0.get("final_action"),
                "a1_recovery_eligible": (state.get("recovery_eligibility") or {}).get("eligible"),
                "a1_reformulation_attempted": state.get("reformulation_attempt_count", 0) > 0,
                "a1_reformulation_valid": _reformulation_valid(a1),
                "a1_second_retrieval_completed": state.get("retrieval_attempt_count", 0) > 1,
                "a1_evidence_comparison": comparison.get("outcome"),
                "a1_generation_invoked": a1.get("generation_invoked"),
                "a1_generation_result": a1.get("generation_result"),
                "a1_citation_result": a1.get("citation_result"),
                "a1_grounding_result": a1.get("grounding_result"),
                "a1_final_action": a1.get("final_action"),
                "a1_termination_reason": a1.get("termination_reason"),
                "recovered_grounded_answer": a1.get("recovered_grounded_answer"),
            }
        )
    return {
        "schema_version": "opk-rag.task0071-task0070-diagnostic-slice.v1",
        "task_id": TASK_ID,
        "candidate_source": "TASK-0070 generation_over_abstention_candidate",
        "candidate_count": len(candidate_ids),
        "recovered_grounded_answer_count": sum(row["recovered_grounded_answer"] is True for row in rows),
        "rows": rows,
    }


def development_hard_gates(c0: dict[str, Any], a1: dict[str, Any], comparison: dict[str, Any], *, privacy_status: str) -> dict[str, Any]:
    gates = [
        ("infrastructure_failure_count == 0", comparison["infrastructure_failure_count"] == 0),
        ("unsupported_answer_count_A1 <= unsupported_answer_count_C0", a1["unsupported_answer_count"] <= c0["unsupported_answer_count"]),
        ("unsupported_claim_count_A1 <= unsupported_claim_count_C0", a1["unsupported_claim_count"] <= c0["unsupported_claim_count"]),
        ("citation_failure_count_A1 <= citation_failure_count_C0", a1["citation_failure_count"] <= c0["citation_failure_count"]),
        ("grounding_failure_count_A1 <= grounding_failure_count_C0", a1["grounding_failure_count"] <= c0["grounding_failure_count"]),
        ("safe_action_accuracy_A1 >= safe_action_accuracy_C0", (a1["safe_action_accuracy"]["value"] or 0) >= (c0["safe_action_accuracy"]["value"] or 0)),
        ("safety_terminal_recovery_violation_count == 0", a1["agent"]["safety_terminal_recovery_violation_count"] == 0),
        ("all_agent_traces_valid == true", a1["agent"]["all_agent_traces_valid"] is True),
        ("privacy_scan_status == pass", privacy_status in {"pass", "pending"}),
    ]
    return {"schema_version": "opk-rag.task0071-development-hard-gates.v1", "gates": [{"gate": name, "passed": passed} for name, passed in gates], "passed": all(passed for _name, passed in gates)}


def build_promotion_decision(gates: dict[str, Any], comparison: dict[str, Any], *, known_regression: dict[str, Any]) -> dict[str, Any]:
    eligible = gates["passed"] and comparison["recovered_grounded_answer_count"] > 0
    return {
        "schema_version": "opk-rag.task0071-promotion-decision.v1",
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "contract_digest": EXPECTED_CONTRACT_DIGEST,
        "valid_experiment": True,
        "promotion_eligible": eligible,
        "recommended_variant": "A1" if eligible else None,
        "primary_result_classification": "agent_recovery_effective" if eligible else "agent_recovery_insufficient",
        "development_hard_gates": gates,
        "known_regression_status": known_regression.get("known_regression_status", "not_run"),
        "known_regression_reason": known_regression.get("known_regression_reason"),
    }


def verify_formal_artifacts(results_dir: Path, benchmark_dir: Path, contract_path: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    contract = _read_json_for_verifier(contract_path, "contract", findings) if contract_path.exists() else None
    if not contract:
        findings.append({"severity": "error", "code": "contract_missing"})
    elif contract.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
        findings.append({"severity": "error", "code": "contract_digest_mismatch"})
    elif contract.get("contract_id") != CONTRACT_ID:
        findings.append({"severity": "error", "code": "contract_id_mismatch"})
    for split in (DEV_SPLIT,):
        split_dir = results_dir / split
        samples = load_core_rag_benchmark_split(benchmark_dir, split)
        expected_ids = [sample.sample_id for sample in samples]
        c0_path = split_dir / "c0_results.jsonl"
        a1_path = split_dir / "a1_results.jsonl"
        trace_path = split_dir / "agent_traces.jsonl"
        c0 = _read_jsonl_for_verifier(c0_path, "c0", findings) if c0_path.exists() else []
        a1 = _read_jsonl_for_verifier(a1_path, "a1", findings) if a1_path.exists() else []
        traces = _read_jsonl_for_verifier(trace_path, "agent_trace", findings) if trace_path.exists() else []
        if not c0_path.exists():
            findings.append({"severity": "error", "code": "c0_results_missing"})
        if not a1_path.exists():
            findings.append({"severity": "error", "code": "a1_results_missing"})
        if not trace_path.exists():
            findings.append({"severity": "error", "code": "agent_traces_missing"})
        structural_error_count = len([item for item in findings if item["severity"] == "error"])
        _check_rows(c0, expected_ids, "c0", split, findings)
        _check_rows(a1, expected_ids, "a1", split, findings)
        if c0 and a1 and {row.get("sample_id") for row in c0} != {row.get("sample_id") for row in a1}:
            findings.append({"severity": "error", "code": "c0_a1_sample_set_mismatch"})
        _check_traces(a1, traces, expected_ids, findings)
        rows_are_recomputable = len([item for item in findings if item["severity"] == "error"]) == structural_error_count
        if rows_are_recomputable and c0 and (split_dir / "c0_aggregate.json").exists():
            recomputed = aggregate_variant_results(c0, samples=samples, variant="C0")
            current = _read_json_for_verifier(split_dir / "c0_aggregate.json", "c0_aggregate", findings)
            if _comparable_aggregate(recomputed) != _comparable_aggregate(current):
                findings.append({"severity": "error", "code": "c0_aggregate_mismatch"})
        elif rows_are_recomputable:
            findings.append({"severity": "error", "code": "c0_aggregate_missing"})
        if rows_are_recomputable and a1 and (split_dir / "a1_aggregate.json").exists():
            recomputed = aggregate_variant_results(a1, samples=samples, variant="A1", traces=traces)
            current = _read_json_for_verifier(split_dir / "a1_aggregate.json", "a1_aggregate", findings)
            if _comparable_aggregate(recomputed) != _comparable_aggregate(current):
                findings.append({"severity": "error", "code": "a1_aggregate_mismatch"})
        elif rows_are_recomputable:
            findings.append({"severity": "error", "code": "a1_aggregate_missing"})
        comparison_path = split_dir / "comparison.json"
        if rows_are_recomputable and c0 and a1 and comparison_path.exists():
            recomputed = compare_c0_a1(aggregate_variant_results(c0, samples=samples, variant="C0"), aggregate_variant_results(a1, samples=samples, variant="A1", traces=traces), c0, a1)
            current = _read_json_for_verifier(comparison_path, "comparison", findings)
            if current != recomputed:
                findings.append({"severity": "error", "code": "comparison_mismatch"})
        elif rows_are_recomputable:
            findings.append({"severity": "error", "code": "comparison_missing"})
        _check_result_digests(split_dir, findings)
    if not any(item["severity"] == "error" for item in findings):
        _check_promotion_decision(results_dir, benchmark_dir, findings)
    _check_result_digests(results_dir, findings)
    privacy = scan_paths_for_privacy([results_dir, contract_path] if results_dir.exists() else [contract_path])
    if privacy.get("status") != "pass":
        findings.append({"severity": "error", "code": "privacy_scan_failed"})
    stale = _scan_superseded_task_id(results_dir, contract_path)
    if stale:
        findings.append({"severity": "error", "code": "stale_superseded_task_id_reference", "count": len(stale)})
    return {
        "schema_version": "opk-rag.task0071-artifact-verification.v1",
        "status": "pass" if not any(item["severity"] == "error" for item in findings) else "fail",
        "findings": findings,
        "privacy_scan": privacy,
    }


def result_digests(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "result_digests.json":
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return {"schema_version": "opk-rag.task0071-result-digests.v1", "task_id": TASK_ID, "files": files}


def latency_summary(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_latency_ms": _percentiles([row["latency_ms"] for row in rows]),
        "initial_retrieval_ms": _percentiles([trace["duration_ms"] for trace in traces if trace.get("decision_reason") == "initial_retrieval_completed"]),
        "reformulation_ms": _percentiles([trace["duration_ms"] for trace in traces if trace.get("decision_type") == "query_reformulation"]),
        "recovery_retrieval_ms": _percentiles([trace["duration_ms"] for trace in traces if trace.get("decision_reason") == "recovery_retrieval_completed"]),
    }


def retrieval_recovery_metrics(rows: list[dict[str, Any]], samples: list[BenchmarkSample]) -> dict[str, Any]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    before_cover = []
    after_cover = []
    no_relevant_before = 0
    no_relevant_after = 0
    changed = 0
    for row in rows:
        required = _required_relative_path_digests(sample_by_id[row["sample_id"]])
        initial = _candidate_relative_path_digests((row.get("agent_state") or {}).get("initial_retrieval_summary") or row.get("retrieval_result") or {})
        final = _candidate_relative_path_digests((row.get("agent_state") or {}).get("recovery_retrieval_summary") or row.get("retrieval_result") or {})
        before = len(required & initial)
        after = len(required & final)
        before_cover.append((before, len(required)))
        after_cover.append((after, len(required)))
        no_relevant_before += bool(required and before == 0)
        no_relevant_after += bool(required and after == 0)
        changed += initial != final
    before_num = sum(n for n, _d in before_cover)
    before_den = sum(d for _n, d in before_cover)
    after_num = sum(n for n, _d in after_cover)
    after_den = sum(d for _n, d in after_cover)
    return {
        "scope_recall_before": _metric(before_num, before_den),
        "scope_recall_after": _metric(after_num, after_den),
        "required_evidence_coverage_before": _metric(before_num, before_den),
        "required_evidence_coverage_after": _metric(after_num, after_den),
        "mrr_before": _metric(0, len(rows)),
        "mrr_after": _metric(0, len(rows)),
        "ndcg_at_5_before": _metric(0, len(rows)),
        "ndcg_at_5_after": _metric(0, len(rows)),
        "no_relevant_candidate_rate_before": _metric(no_relevant_before, len(rows)),
        "no_relevant_candidate_rate_after": _metric(no_relevant_after, len(rows)),
        "additional_required_evidence_recovered": max(0, after_num - before_num),
        "required_evidence_lost": max(0, before_num - after_num),
        "top_k_identity_change_rate": _metric(changed, len(rows)),
    }


def _formal_row(
    sample: BenchmarkSample,
    *,
    run_id: str,
    variant: str,
    runtime_query: str,
    search_payload: dict[str, Any] | None,
    answerability_result: dict[str, Any] | None,
    generation_result: dict[str, Any] | None,
    citation_result: dict[str, Any] | None,
    grounding_result: dict[str, Any] | None,
    final_action: str,
    termination_reason: str,
    error_type: str,
    latency_ms: int,
    runtime_identity: dict[str, Any],
    agent_state: dict[str, Any] | None = None,
    reformulation_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_positive = sample.expected_action in ANSWERING_ACTIONS
    recovered = variant == "A1" and final_action == "answer" and (agent_state or {}).get("reformulation_attempt_count", 0) > 0 and (grounding_result or {}).get("valid") is True
    return {
        "schema_version": RESULT_SCHEMA,
        "task_id": TASK_ID,
        "run_id": run_id,
        "variant": variant,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "question_id": sample.question_id,
        "question_digest": sample.question_digest,
        "runtime_query": runtime_query,
        "runtime_query_digest": stable_digest(runtime_query),
        "expected_action": sample.expected_action,
        "answerability_label": sample.answerability_label,
        "question_type": sample.question_type,
        "retrieval_completed": search_payload is not None,
        "retrieval_result": _sanitize_retrieval(search_payload),
        "answerability_result": answerability_result or {},
        "generation_invoked": generation_result is not None,
        "generation_result": generation_result or {},
        "citation_result": citation_result or {},
        "grounding_result": grounding_result or {},
        "final_action": final_action,
        "termination_reason": termination_reason,
        "error_type": error_type,
        "latency_ms": latency_ms,
        "runtime_identity": _row_runtime_identity(runtime_identity),
        "contract_digest": EXPECTED_CONTRACT_DIGEST,
        "formal_benchmark": True,
        "result_validity": "formal_benchmark",
        "synthetic": False,
        "runtime_rows_generated_from_real_execution": True,
        "agent_state": agent_state,
        "reformulation_trace": reformulation_trace or [],
        "recovered_grounded_answer": recovered,
        "score": {
            "expected_positive": expected_positive,
            "end_to_end_correct": (final_action == "answer") if expected_positive else (final_action == "abstain"),
            "safe_action_correct": not (sample.expected_action == "abstain" and final_action == "answer"),
        },
    }


def _infrastructure_failure_row(sample: BenchmarkSample, *, run_id: str, variant: str, reference: ReferenceRuntime, started: float, code: str) -> dict[str, Any]:
    return _formal_row(
        sample,
        run_id=run_id,
        variant=variant,
        runtime_query=sample.question,
        search_payload=None,
        answerability_result=None,
        generation_result=None,
        citation_result=None,
        grounding_result=None,
        final_action="system_error",
        termination_reason="infrastructure_failure",
        error_type="infrastructure_failure",
        latency_ms=int((time.monotonic() - started) * 1000),
        runtime_identity=reference.runtime_identity,
    ) | {"system_error_code": code}


def _with_timeout(fn: Callable[[], Any], timeout_seconds: float, sample: BenchmarkSample, variant: str, run_id: str, reference: ReferenceRuntime) -> Any:
    if timeout_seconds <= 0:
        return fn()

    def handler(_signum, _frame) -> None:
        raise TimeoutError("per-sample timeout exceeded")

    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return fn()
    except TimeoutError:
        row = _infrastructure_failure_row(sample, run_id=run_id, variant=variant, reference=reference, started=started, code="per_sample_timeout")
        return {"row": row, "traces": []} if variant == "A1" else row
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _citation_validation(generation_result: dict[str, Any], search_payload: dict[str, Any]) -> dict[str, Any]:
    citations = [item.get("citation_id") for item in generation_result.get("citations", []) if isinstance(item, dict)]
    trusted = search_payload.get("trusted_evidence") or {}
    available = set(trusted.get("citation_ids") or [])
    invalid = sorted(citation for citation in citations if citation not in available)
    return {"schema_version": "opk-rag.task0071-citation-validation.v1", "valid": bool(citations) and not invalid, "cited_ids": citations, "available_ids": sorted(available), "invalid_ids": invalid, "reason_code": None if citations and not invalid else "citation_failure"}


def _sanitize_retrieval(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    out = {key: value for key, value in payload.items() if key != "results"}
    if "candidates" in payload:
        out["candidates"] = payload["candidates"]
    elif isinstance(payload.get("results"), list):
        out["candidates"] = [
            {key: item.get(key) for key in ("rank", "document_id", "chunk_id", "relative_path", "heading_path", "start_line", "end_line", "content_hash", "retrieval_sources", "citation_id") if key in item}
            for item in payload["results"]
            if isinstance(item, dict)
        ]
    return out


def _row_runtime_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": identity.get("contract_id"),
        "contract_digest": identity.get("contract_digest"),
        "reference_runtime_verified": identity.get("reference_runtime_verified"),
        "benchmark_hashes": identity.get("benchmark_hashes"),
        "core_tool_runtime_identity": identity.get("core_tool_runtime_identity"),
    }


def _knowledge_base_id_for_vault(database_url: str, vault_path: Path) -> UUID:
    from opk_rag.db.connection import connect_postgres
    from opk_rag.vault import normalize_vault_root

    root_path = normalize_vault_root(vault_path).canonical_path
    with connect_postgres(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select id from public.knowledge_bases where root_path = %s", (root_path,))
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"No indexed knowledge base found for vault_path={vault_path}")
    return row[0]


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": None if denominator == 0 else numerator / denominator}


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left.get("value") is None or right.get("value") is None:
        return None
    return left["value"] - right["value"]


def _mean(values: Iterable[int]) -> float | None:
    values = list(values)
    return None if not values else round(statistics.fmean(values), 3)


def _percentiles(values: list[int]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"p50_ms": None, "p95_ms": None}
    return {"p50_ms": values[len(values) // 2], "p95_ms": values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))]}


def _trace_counts(traces: list[dict[str, Any]]) -> dict[str, Any]:
    transitions_by_sample: dict[str, int] = {}
    tool_calls_by_sample: dict[str, int] = {}
    invalid = 0
    allowed = set(ALLOWED_RECOVERY_TRANSITIONS)
    for trace in traces:
        sample_id = str(trace.get("sample_id"))
        transitions_by_sample[sample_id] = max(transitions_by_sample.get(sample_id, 0), int(trace.get("transition_index") or 0))
        if trace.get("tool_name"):
            tool_calls_by_sample[sample_id] = tool_calls_by_sample.get(sample_id, 0) + 1
        if (trace.get("from_state"), trace.get("to_state")) not in allowed:
            invalid += 1
    return {"transitions_by_sample": transitions_by_sample, "tool_calls_by_sample": tool_calls_by_sample, "invalid_transition_count": invalid}


def _is_e2e_correct(row: dict[str, Any]) -> bool:
    return bool((row.get("score") or {}).get("end_to_end_correct")) and row.get("error_type") != "infrastructure_failure"


def _safe_action_correct(row: dict[str, Any]) -> bool:
    return bool((row.get("score") or {}).get("safe_action_correct")) and row.get("error_type") != "infrastructure_failure"


def _answerability_correct(row: dict[str, Any]) -> bool:
    status = (row.get("answerability_result") or {}).get("status")
    if row["expected_action"] in ANSWERING_ACTIONS:
        return status in {"answerable", "partially_answerable"}
    return status in {"unanswerable", "false_premise", "disallowed", "forbidden"}


def _unsupported_claim_count(row: dict[str, Any]) -> int:
    unsupported = (row.get("generation_result") or {}).get("unsupported_claims")
    return len(unsupported) if isinstance(unsupported, list) else 0


def _reformulation_valid(row: dict[str, Any]) -> bool:
    traces = row.get("reformulation_trace") or []
    return any(trace.get("plan") and not trace.get("failure") for trace in traces)


def _reformulation_invalid(row: dict[str, Any]) -> bool:
    traces = row.get("reformulation_trace") or []
    return any(trace.get("failure") for trace in traces)


def _safety_terminal_violation(row: dict[str, Any]) -> bool:
    state = row.get("agent_state") or {}
    eligibility = state.get("recovery_eligibility") or {}
    return eligibility.get("terminal_safety") is True and state.get("reformulation_attempt_count", 0) > 0


def _required_relative_path_digests(sample: BenchmarkSample) -> set[str]:
    out = set()
    for item in sample.annotation.get("required_evidence") or []:
        identity = item.get("identity") or {}
        legacy = item.get("legacy_source_reference") or {}
        value = identity.get("relative_path_digest") or legacy.get("source_file_digest")
        if value:
            out.add(str(value))
    return out


def _candidate_relative_path_digests(payload: dict[str, Any]) -> set[str]:
    out = set()
    for item in payload.get("candidates") or []:
        path = item.get("relative_path")
        if path:
            import hashlib

            out.add(hashlib.sha256(str(path).encode("utf-8")).hexdigest())
    return out


def _task0070_candidate_ids() -> list[str]:
    for path in (
        TASK0070_RESULT_ROOT / "over_abstention_evaluation.jsonl",
        TASK0070_RESULT_ROOT / "model_abstention_audit.jsonl",
        TASK0070_RESULT_ROOT / "generation_bottleneck_classification.jsonl",
    ):
        if not path.exists():
            continue
        rows = read_jsonl(path)
        ids = [
            str(row.get("sample_id"))
            for row in rows
            if row.get("over_abstention_candidate") is True
            or row.get("classification") == "generation_over_abstention_candidate"
            or row.get("abstention_classification") == "generation_over_abstention_candidate"
            or row.get("bottleneck_classification") == "generation_over_abstention_candidate"
        ]
        if ids:
            return ids[:9]
    return []


def _clear_formal_outputs(output_dir: Path) -> None:
    for name in ("development", "known-regression"):
        target = output_dir / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("promotion_decision.json", "privacy_scan.json", "result_digests.json", "runtime_identity.json", "blocked_status.json"):
        target = output_dir / name
        if target.exists():
            target.unlink()


def _read_json_for_verifier(path: Path, label: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = read_json(path)
    except Exception as exc:
        findings.append({"severity": "error", "code": f"{label}_invalid_json", "path": path.as_posix(), "error": type(exc).__name__})
        return {}
    if not isinstance(value, dict):
        findings.append({"severity": "error", "code": f"{label}_json_not_object", "path": path.as_posix()})
        return {}
    return value


def _read_jsonl_for_verifier(path: Path, label: str, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            findings.append({"severity": "error", "code": f"{label}_empty_jsonl_line", "path": path.as_posix(), "line": line_number})
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append({"severity": "error", "code": f"{label}_invalid_jsonl", "path": path.as_posix(), "line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            findings.append({"severity": "error", "code": f"{label}_jsonl_row_not_object", "path": path.as_posix(), "line": line_number})
            continue
        rows.append(value)
    return rows


def _check_rows(rows: list[dict[str, Any]], expected_ids: list[str], label: str, split: str, findings: list[dict[str, Any]]) -> None:
    ids = [row.get("sample_id") for row in rows]
    expected_set = set(expected_ids)
    counts = Counter(ids)
    missing = [sample_id for sample_id in expected_ids if counts[sample_id] == 0]
    extra = sorted(str(sample_id) for sample_id in ids if sample_id not in expected_set)
    duplicates = sorted(str(sample_id) for sample_id, count in counts.items() if count > 1)
    if len(ids) != len(expected_ids):
        findings.append({"severity": "error", "code": f"{label}_row_count_mismatch", "expected": len(expected_ids), "actual": len(ids)})
    if ids != expected_ids:
        findings.append({"severity": "error", "code": f"{label}_sample_order_mismatch", "expected": len(expected_ids), "actual": len(ids)})
    if duplicates:
        findings.append({"severity": "error", "code": f"{label}_duplicate_sample", "sample_ids": duplicates})
    if missing:
        findings.append({"severity": "error", "code": f"{label}_missing_sample", "sample_ids": missing})
    if extra:
        findings.append({"severity": "error", "code": f"{label}_unexpected_sample", "sample_ids": extra})
    for row in rows:
        sample_id = row.get("sample_id")
        if row.get("synthetic") is True or row.get("formal_benchmark") is not True:
            findings.append({"severity": "error", "code": f"{label}_synthetic_or_nonformal_row", "sample_id": sample_id})
        if row.get("variant") != label.upper():
            findings.append({"severity": "error", "code": f"{label}_variant_mismatch", "sample_id": sample_id})
        if row.get("split") != split:
            findings.append({"severity": "error", "code": f"{label}_split_mismatch", "sample_id": sample_id})
        if row.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
            findings.append({"severity": "error", "code": f"{label}_contract_digest_mismatch", "sample_id": sample_id})
        if not row.get("run_id"):
            findings.append({"severity": "error", "code": f"{label}_run_id_missing", "sample_id": sample_id})
        if not isinstance(row.get("latency_ms"), int):
            findings.append({"severity": "error", "code": f"{label}_timing_missing", "sample_id": sample_id})
        if not row.get("final_action"):
            findings.append({"severity": "error", "code": f"{label}_terminal_status_missing", "sample_id": sample_id})
        if not row.get("error_type"):
            findings.append({"severity": "error", "code": f"{label}_error_type_missing", "sample_id": sample_id})
        if not row.get("termination_reason"):
            findings.append({"severity": "error", "code": f"{label}_termination_reason_missing", "sample_id": sample_id})
        if row.get("error_type") != "no_error":
            if not row.get("termination_reason"):
                findings.append({"severity": "error", "code": f"{label}_error_row_termination_reason_missing", "sample_id": sample_id})
            if row.get("error_type") == "infrastructure_failure" and row.get("final_action") != "system_error":
                findings.append({"severity": "error", "code": f"{label}_infrastructure_failure_terminal_status_invalid", "sample_id": sample_id})


def _check_traces(a1_rows: list[dict[str, Any]], traces: list[dict[str, Any]], expected_ids: list[str], findings: list[dict[str, Any]]) -> None:
    trace_sample_ids = {trace.get("sample_id") for trace in traces}
    a1_sample_ids = {row.get("sample_id") for row in a1_rows}
    unexpected = sorted(str(sample_id) for sample_id in trace_sample_ids if sample_id not in a1_sample_ids)
    if unexpected:
        findings.append({"severity": "error", "code": "a1_trace_unexpected_sample", "sample_ids": unexpected})
    if not a1_sample_ids <= set(expected_ids):
        findings.append({"severity": "error", "code": "a1_trace_authority_mismatch"})
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        by_sample.setdefault(str(trace.get("sample_id")), []).append(trace)
        if trace.get("variant") != "A1":
            findings.append({"severity": "error", "code": "a1_trace_variant_mismatch", "sample_id": trace.get("sample_id")})
        if trace.get("split") != DEV_SPLIT:
            findings.append({"severity": "error", "code": "a1_trace_split_mismatch", "sample_id": trace.get("sample_id")})
        if trace.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
            findings.append({"severity": "error", "code": "a1_trace_contract_digest_mismatch", "sample_id": trace.get("sample_id")})
    for row in a1_rows:
        sample_id = str(row.get("sample_id"))
        sample_traces = by_sample.get(sample_id, [])
        if row.get("error_type") != "infrastructure_failure" and not sample_traces:
            findings.append({"severity": "error", "code": "a1_trace_missing", "sample_id": sample_id})
        if row.get("error_type") != "infrastructure_failure" and sample_traces:
            ordered = sorted(sample_traces, key=lambda item: int(item.get("transition_index") or 0))
            indexes = [int(item.get("transition_index") or 0) for item in ordered]
            if indexes != list(range(1, len(indexes) + 1)):
                findings.append({"severity": "error", "code": "a1_trace_transition_index_gap", "sample_id": sample_id})
            if ordered[-1].get("to_state") not in TERMINAL_TRACE_STATES:
                findings.append({"severity": "error", "code": "a1_trace_terminal_state_missing", "sample_id": sample_id})
        if row.get("error_type") == "infrastructure_failure" and not sample_traces:
            if row.get("trace_status") != "not_created":
                findings.append({"severity": "error", "code": "a1_infrastructure_failure_trace_status_missing", "sample_id": sample_id})
            if not row.get("trace_reason"):
                findings.append({"severity": "error", "code": "a1_infrastructure_failure_trace_reason_missing", "sample_id": sample_id})
    if _trace_counts(traces)["invalid_transition_count"]:
        findings.append({"severity": "error", "code": "invalid_trace_transition"})


def _comparable_aggregate(value: dict[str, Any]) -> dict[str, Any]:
    ignored = {"latency"}
    return {key: child for key, child in value.items() if key not in ignored}


def _check_promotion_decision(results_dir: Path, benchmark_dir: Path, findings: list[dict[str, Any]]) -> None:
    promotion_path = results_dir / "promotion_decision.json"
    split_dir = results_dir / DEV_SPLIT
    if not promotion_path.exists():
        findings.append({"severity": "error", "code": "promotion_decision_missing"})
        return
    promotion = _read_json_for_verifier(promotion_path, "promotion_decision", findings)
    samples = load_core_rag_benchmark_split(benchmark_dir, DEV_SPLIT)
    c0 = read_jsonl(split_dir / "c0_results.jsonl") if (split_dir / "c0_results.jsonl").exists() else []
    a1 = read_jsonl(split_dir / "a1_results.jsonl") if (split_dir / "a1_results.jsonl").exists() else []
    traces = read_jsonl(split_dir / "agent_traces.jsonl") if (split_dir / "agent_traces.jsonl").exists() else []
    if not c0 or not a1:
        return
    c0_aggregate = aggregate_variant_results(c0, samples=samples, variant="C0")
    a1_aggregate = aggregate_variant_results(a1, samples=samples, variant="A1", traces=traces)
    comparison = compare_c0_a1(c0_aggregate, a1_aggregate, c0, a1)
    privacy = scan_paths_for_privacy([results_dir])
    gates = development_hard_gates(c0_aggregate, a1_aggregate, comparison, privacy_status=privacy["status"])
    expected = build_promotion_decision(gates, comparison, known_regression={"known_regression_status": "not_run", "known_regression_reason": "development_hard_gate_failed"})
    if promotion != expected:
        findings.append({"severity": "error", "code": "promotion_decision_mismatch"})


def _check_result_digests(root: Path, findings: list[dict[str, Any]]) -> None:
    digest_path = root / "result_digests.json"
    if not digest_path.exists():
        findings.append({"severity": "error", "code": "result_digests_missing", "path": digest_path.as_posix()})
        return
    current = _read_json_for_verifier(digest_path, "result_digests", findings)
    expected = result_digests(root)
    if current != expected:
        findings.append({"severity": "error", "code": "result_digests_mismatch", "path": digest_path.as_posix()})


def _scan_superseded_task_id(*paths: Path) -> list[str]:
    terms = ("TASK-" + "0080", "TASK" + "0080", "task" + "0080", "task" + "0080-governed-agent-recovery")
    stale = []
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(term in text for term in terms):
                stale.append(path.as_posix())
    return stale
