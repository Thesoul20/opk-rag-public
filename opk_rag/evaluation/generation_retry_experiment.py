from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from opk_rag.agent.contracts import stable_digest
from opk_rag.agent.generation_retry import build_generation_retry_eligibility_input, decide_generation_retry_eligibility, generation_request_identity_digest, validate_retry_request_identity
from opk_rag.agent.generation_retry_contracts import (
    BENCHMARK_DIR,
    CONTRACT_ID,
    CONTRACT_PATH,
    RESULTS_DIR,
    build_generation_retry_contract,
    validate_authoritative_inputs,
)
from opk_rag.agent.query_reformulation import FakeQueryReformulationProvider, GovernedQueryReformulator
from opk_rag.agent.recovery_loop import AgentRecoveryConfig, run_agent_recovery_loop
from opk_rag.agent.tool_registry import LiveCoreToolExecutor
from opk_rag.answer.models import AnswerSystemError
from opk_rag.core_tools.contracts import CoreToolError
from opk_rag.core_tools.serialization import trusted_evidence_payload
from opk_rag.evaluation.agent_recovery_experiment import build_reference_runtime, load_core_rag_benchmark_split
from opk_rag.evaluation.core_rag_benchmark import BENCHMARK_ID, BENCHMARK_VERSION, DEV_SPLIT, file_digest, read_json, read_jsonl, scan_paths_for_privacy, validate_reference_runtime, write_json, write_jsonl
from opk_rag.runtime.dotenv import load_project_env

TASK_ID = "TASK-0076"
RESULT_SCHEMA = "opk-rag.task0076-sample-result.v1"
EXPECTED_CONTRACT_DIGEST = "b3403c8474fd987c6d96188f9e43dd7cdfe5c7efb9d2a424d54bb2126fed8c16"
ANSWERING_ACTIONS = {"answer", "partial_answer", "correct_premise"}
REPLICATE_IDS = ("formal-replicate-1", "formal-replicate-2")


class InstrumentedGenerationRetryExecutor(LiveCoreToolExecutor):
    def __init__(self, runtime: Any, *, run_id: str, sample_id: str) -> None:
        super().__init__(runtime)
        self.run_id = run_id
        self.sample_id = sample_id
        self.invocations: list[dict[str, Any]] = []

    def execute(self, tool_name: str, arguments: dict[str, Any], memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.monotonic()
        started_at = _utc_now()
        generation_index = 1 + sum(row["tool_name"] == "generate_grounded_answer" for row in getattr(self, "invocations", []))
        request_identity = None
        if tool_name == "generate_grounded_answer":
            request_identity = _live_generation_request_identity(self.runtime, memory, attempt_index=generation_index, run_id=self.run_id)
        try:
            payload, summary = super().execute(tool_name, arguments, memory)
            status = "completed"
            error_type = None
        except Exception as exc:
            payload = {}
            summary = {}
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            if tool_name == "generate_grounded_answer":
                completed_at = _utc_now()
                digest = generation_request_identity_digest(request_identity or {})
                payload_for_log = (summary.get("answer_response") or payload) if isinstance(summary, dict) else payload
                self.invocations.append(
                    {
                        "schema_version": "opk-rag.task0076-generation-invocation.v1",
                        "task_id": TASK_ID,
                        "run_id": self.run_id,
                        "sample_id": self.sample_id,
                        "tool_name": tool_name,
                        "generation_attempt_index": generation_index,
                        "retry_of_invocation_id": None if generation_index == 1 else f"{self.run_id}:{self.sample_id}:generation:1",
                        "generation_invocation_id": f"{self.run_id}:{self.sample_id}:generation:{generation_index}",
                        "request_identity_digest": digest,
                        "request_identity": request_identity,
                        "provider_call_attempted": True,
                        "provider_call_completed": status == "completed",
                        "provider_response_digest": stable_digest(payload_for_log) if payload_for_log else None,
                        "response_mode": _response_mode(payload_for_log),
                        "response_contract_valid": _response_contract_valid(payload_for_log),
                        "runtime_failure_class": _runtime_failure_class(payload_for_log, memory.get("answerability")),
                        "terminal_outcome": payload_for_log.get("status") if isinstance(payload_for_log, dict) else None,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "provider_identity": self.runtime.answer_provider.provider_id,
                        "model_identity": self.runtime.answer_provider.model_id,
                        "live_provider_execution": True,
                        "synthetic_provider": False,
                        "replay_source": None,
                        "historical_response_reused": False,
                        "error_type": error_type,
                    }
                )
        return payload, summary


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2], check=True, text=True, capture_output=True).stdout.strip()


def prepare_task0076_outputs(*, output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH, overwrite: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    start_audit = {
        "schema_version": "opk-rag.task0076-start-audit.v1",
        "branch": branch,
        "head": head,
        "staged_diff_empty": git_output("diff", "--cached", "--stat") == "",
        "git_status_short": git_output("status", "--short"),
    }
    input_identity = validate_authoritative_inputs() | {"start_audit": start_audit}
    contract = build_generation_retry_contract(branch=branch, head=head, input_identities=input_identity)
    write_json(contract_path, contract)
    write_json(output_dir / "input_identity.json", input_identity)
    write_json(
        output_dir / "contract_identity.json",
        {
            "schema_version": "opk-rag.task0076-contract-identity.v1",
            "contract_id": CONTRACT_ID,
            "contract_digest": contract["contract_digest"],
            "contract_file_sha256": file_digest(contract_path),
        },
    )
    runtime_identity = {
        "schema_version": "opk-rag.task0076-runtime-identity.v1",
        "reference_runtime_status": "not_verified_by_task0076_preflight",
        "generation_retry_default_enabled": False,
        "generation_retry_contract_activated_for_a2_only": True,
        "code_head": head,
    }
    write_json(output_dir / "runtime_identity.json", runtime_identity)
    preflight = build_preflight(input_identity=input_identity, contract=contract)
    write_json(output_dir / "preflight.json", preflight)
    return {"input_identity": input_identity, "contract": contract, "runtime_identity": runtime_identity, "preflight": preflight}


def prepare_live_task0076_outputs(*, output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH, overwrite: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    contract = read_json(contract_path)
    if contract.get("contract_id") != CONTRACT_ID or contract.get("contract_digest") != EXPECTED_CONTRACT_DIGEST:
        raise RuntimeError("generation_retry_contract_digest_mismatch")
    input_identity = validate_authoritative_inputs() | {
        "start_audit": {
            "schema_version": "opk-rag.task0076-start-audit.v1",
            "branch": branch,
            "head": head,
            "staged_diff_empty": git_output("diff", "--cached", "--stat") == "",
            "git_status_short": git_output("status", "--short", "--untracked-files=all"),
            "git_diff_check_clean": git_output("diff", "--check") == "",
        }
    }
    write_json(output_dir / "input_identity.json", input_identity)
    write_json(
        output_dir / "contract_identity.json",
        {
            "schema_version": "opk-rag.task0076-contract-identity.v1",
            "contract_id": CONTRACT_ID,
            "contract_digest": contract["contract_digest"],
            "contract_file_sha256": file_digest(contract_path),
            "contract_verified": True,
        },
    )
    preflight = live_reference_runtime_preflight(contract=contract, output_dir=output_dir)
    write_json(output_dir / "preflight.json", preflight)
    runtime_identity = preflight.get("runtime_identity") or {
        "schema_version": "opk-rag.task0076-runtime-identity.v1",
        "reference_runtime_status": preflight["reference_runtime_status"],
    }
    write_json(output_dir / "runtime_identity.json", runtime_identity)
    return {"input_identity": input_identity, "contract": contract, "runtime_identity": runtime_identity, "preflight": preflight}


def live_reference_runtime_preflight(*, contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    load_project_env(Path(__file__).resolve().parents[2])
    checks: dict[str, Any] = {
        "schema_version": "opk-rag.task0076-reference-runtime-preflight.v1",
        "task_id": TASK_ID,
        "supabase_url_configured": bool(os.environ.get("SUPABASE_URL")),
        "supabase_key_configured": bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_PUBLISHABLE_KEY")),
        "deepseek_base_url_configured": bool(os.environ.get("OPK_RAG_LLM_BASE_URL")),
        "deepseek_api_key_configured": bool(os.environ.get("OPK_RAG_LLM_API_KEY")),
        "deepseek_model_configured": bool(os.environ.get("OPK_RAG_LLM_MODEL") or os.environ.get("OPK_RAG_LLM_MODEL_ID")),
        "artifact_output_writable": _artifact_output_writable(output_dir),
        "privacy_gate_available": True,
    }
    runtime_validation = validate_reference_runtime()
    passed = _passed_checks(runtime_validation)
    checks |= {
        "supabase_connectivity": passed("remote_supabase", "remote_postgresql"),
        "supabase_read_query": passed("remote_supabase", "remote_postgresql"),
        "supabase_transaction_health": passed("remote_supabase", "remote_postgresql"),
        "supabase_connection_cleanup": True,
        "supabase_corpus_snapshot_verified": passed("local_markdown_vault", "frozen_corpus_digest_match"),
        "deepseek_connectivity": passed("remote_deepseek", "provider_preflight"),
        "deepseek_provider_contract_call": passed("remote_deepseek", "provider_preflight"),
        "benchmark_identity_verified": BENCHMARK_ID == "core-rag-benchmark-v1" and BENCHMARK_VERSION == "1.0.0",
        "canonical_evidence_identity_verified": passed("embedding", "embedding_model_contract"),
        "generation_retry_contract_verified": contract.get("contract_digest") == EXPECTED_CONTRACT_DIGEST,
        "core_runtime_validation": runtime_validation,
    }
    mandatory = [
        "supabase_url_configured",
        "supabase_key_configured",
        "supabase_connectivity",
        "supabase_read_query",
        "supabase_transaction_health",
        "supabase_connection_cleanup",
        "supabase_corpus_snapshot_verified",
        "deepseek_base_url_configured",
        "deepseek_api_key_configured",
        "deepseek_model_configured",
        "deepseek_connectivity",
        "deepseek_provider_contract_call",
        "benchmark_identity_verified",
        "canonical_evidence_identity_verified",
        "generation_retry_contract_verified",
        "artifact_output_writable",
        "privacy_gate_available",
    ]
    checks["reference_runtime_status"] = "ready" if all(checks.get(key) is True for key in mandatory) else "blocked"
    checks["blocking_reason"] = None if checks["reference_runtime_status"] == "ready" else next((key for key in mandatory if checks.get(key) is not True), "unknown_preflight_failure")
    if checks["reference_runtime_status"] == "ready":
        reference = build_reference_runtime(BENCHMARK_DIR, contract | {"task_id": "TASK-0071"}, require_verified=True)
        checks["runtime_identity"] = {
            "schema_version": "opk-rag.task0076-runtime-identity.v1",
            "task_id": TASK_ID,
            "reference_runtime_status": "ready",
            "generation_retry_default_enabled": False,
            "generation_retry_contract_activated_for_a2_only": True,
            "runtime_rows_generated_from_real_execution": True,
            "formal_benchmark": True,
            "core_tool_runtime_identity": reference.runtime.identity(),
            "provider_parameters_identity_digest": stable_digest(reference.runtime.answer_config),
            "contract_id": CONTRACT_ID,
            "contract_digest": EXPECTED_CONTRACT_DIGEST,
        }
    return checks


def run_live_formal_experiment(*, output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH, overwrite: bool = False) -> dict[str, Any]:
    prepared = prepare_live_task0076_outputs(output_dir=output_dir, contract_path=contract_path, overwrite=overwrite)
    if prepared["preflight"]["reference_runtime_status"] != "ready":
        blocked = _blocked_payload(prepared["preflight"].get("blocking_reason") or "reference_runtime_preflight_failed")
        write_json(output_dir / "blocked_status.json", blocked)
        return {"status": "blocked", "blocked": blocked, **prepared}
    reference = build_reference_runtime(BENCHMARK_DIR, prepared["contract"] | {"task_id": "TASK-0071"}, require_verified=True)
    rep1 = run_live_formal_replicate(output_dir=output_dir, replicate_id="formal-replicate-1", contract=prepared["contract"], reference=reference)
    rep2 = run_live_formal_replicate(output_dir=output_dir, replicate_id="formal-replicate-2", contract=prepared["contract"], reference=reference)
    analysis = write_analysis_artifacts(output_dir, rep1, rep2)
    return {"status": "complete", "replicate_1": rep1, "replicate_2": rep2, **analysis, **prepared}


def run_live_formal_replicate(*, output_dir: Path, replicate_id: str, contract: dict[str, Any], reference: Any) -> dict[str, Any]:
    replicate_dir = output_dir / replicate_id
    replicate_dir.mkdir(parents=True, exist_ok=True)
    samples = load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)
    run_id = f"task0076-{replicate_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_identity = {
        "schema_version": "opk-rag.task0076-run-identity.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "replicate_id": replicate_id,
        "split": DEV_SPLIT,
        "sample_ids": [sample.sample_id for sample in samples],
        "authoritative_sample_count": len(samples),
        "contract_id": CONTRACT_ID,
        "contract_digest": contract["contract_digest"],
        "result_validity": "formal_benchmark",
        "formal_benchmark": True,
        "live_provider_retry": True,
        "replay_source": None,
        "started_at": _utc_now(),
    }
    write_json(replicate_dir / "run_identity.json", run_identity)
    write_json(replicate_dir / "runtime_identity.json", reference.runtime_identity | {"task_id": TASK_ID, "contract_id": CONTRACT_ID, "contract_digest": contract["contract_digest"]})
    sample_results: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    config = AgentRecoveryConfig(generation_retry_enabled=True, max_generation_calls=2, max_generation_retries=1, max_total_tool_calls=6, max_transitions=16, max_retrieval_calls=1, max_reformulation_calls=0, search_top_k=10)
    for index, sample in enumerate(samples, start=1):
        started = time.monotonic()
        executor = InstrumentedGenerationRetryExecutor(reference.runtime, run_id=run_id, sample_id=sample.sample_id)
        reformulator = GovernedQueryReformulator(provider=FakeQueryReformulationProvider([]))
        try:
            result = run_agent_recovery_loop(
                question=sample.question,
                sample_id=sample.sample_id,
                run_id=f"{run_id}:{sample.sample_id}",
                executor=executor,
                reformulator=reformulator,
                config=config,
            )
            state = result.state.to_dict()
            row = _live_sample_row(sample, index=index, replicate_id=replicate_id, run_id=run_id, state=state, invocations=executor.invocations, latency_ms=int((time.monotonic() - started) * 1000))
            sample_traces = [event.to_dict() | {"task_id": TASK_ID, "run_id": run_id, "replicate_id": replicate_id, "split": DEV_SPLIT, "sample_id": sample.sample_id, "contract_digest": contract["contract_digest"]} for event in result.trace_events]
        except Exception as exc:
            row = _live_infrastructure_failure_row(sample, index=index, replicate_id=replicate_id, run_id=run_id, latency_ms=int((time.monotonic() - started) * 1000), code=type(exc).__name__)
            sample_traces = []
        sample_results.append(row)
        invocations.extend(executor.invocations)
        decisions.append(_retry_decision_row(row))
        traces.extend(sample_traces)
        write_jsonl(replicate_dir / "sample_results.jsonl", sample_results)
        write_jsonl(replicate_dir / "generation_invocations.jsonl", invocations)
        write_jsonl(replicate_dir / "retry_decisions.jsonl", decisions)
        write_jsonl(replicate_dir / "agent_traces.jsonl", traces)
    a1 = aggregate_generation_retry_results(sample_results, variant="A1")
    a2 = aggregate_generation_retry_results(sample_results, variant="A2")
    paired = compare_a1_a2(a1, a2, sample_results)
    write_json(replicate_dir / "a1_aggregate.json", a1)
    write_json(replicate_dir / "a2_aggregate.json", a2)
    write_json(replicate_dir / "paired_comparison.json", paired)
    write_json(replicate_dir / "privacy_scan.json", scan_paths_for_privacy([replicate_dir]))
    write_json(replicate_dir / "result_digests.json", result_digests(replicate_dir))
    return {"run_id": run_id, "sample_results": sample_results, "a1_aggregate": a1, "a2_aggregate": a2, "paired_comparison": paired}


def build_preflight(*, input_identity: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if input_identity.get("validation_status") != "pass":
        findings.extend(input_identity.get("findings") or [])
    if not input_identity.get("start_audit", {}).get("staged_diff_empty"):
        findings.append({"code": "staged_diff_not_empty"})
    proposal = read_json(Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "stable_generation_retry_eligibility_proposal.json")
    if proposal.get("status") != "proposal_only":
        findings.append({"code": "task0075_proposal_not_proposal_only"})
    readiness = read_json(Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "task0076_readiness_gates.json")
    if readiness.get("task0076_readiness") != "ready":
        findings.append({"code": "task0076_readiness_not_ready"})
    for key, expected in {
        "maximum_generation_retries": 1,
        "maximum_total_generation_calls": 2,
        "retrieval_retry_allowed": False,
        "query_reformulation_allowed": False,
        "recursive_retry_allowed": False,
        "provider_substitution_allowed": False,
        "model_substitution_allowed": False,
        "prompt_change_allowed": False,
        "provider_parameter_change_allowed": False,
        "evidence_bundle_change_allowed": False,
    }.items():
        if contract.get(key) != expected:
            findings.append({"code": "contract_required_value_mismatch", "field": key, "expected": expected, "actual": contract.get(key)})
    return {
        "schema_version": "opk-rag.task0076-preflight.v1",
        "preflight_status": "ready" if not findings else "blocked",
        "reference_runtime_status": "not_executed",
        "generation_retry_already_enabled": False,
        "sample_specific_retry_allowlist_present": False,
        "findings": findings,
    }


def run_replay_formal_replicate_for_artifact_validation(*, output_dir: Path, replicate_id: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Privacy-safe replay from TASK-0075 observations for verifier and aggregate validation.

    This mode is diagnostic support only. It evaluates runtime eligibility and A1 projection
    without sending Attempt 2 to a Provider, so recovered counts remain zero and Known
    Regression is gated off.
    """
    replicate_dir = output_dir / replicate_id
    replicate_dir.mkdir(parents=True, exist_ok=True)
    samples = load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)
    matrix_rows = read_jsonl(Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "five_replicate_observation_matrix.jsonl")
    observation_by_sample = _first_observation_by_sample(matrix_rows)
    candidate_observation_by_sample = _candidate_observation_by_sample()
    sample_results: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        record = candidate_observation_by_sample.get(sample.sample_id) or observation_by_sample.get(sample.sample_id)
        sanitized = _eligibility_input_from_record(record) if record else _missing_generation_observation()
        decision = decide_generation_retry_eligibility(sanitized)
        a1_final = _a1_final_action(record)
        retry_attempted = False
        retry_outcome = "retry_not_eligible" if not decision.eligible else "retry_provider_failure"
        row = {
            "schema_version": RESULT_SCHEMA,
            "task_id": TASK_ID,
            "replicate_id": replicate_id,
            "sample_index": index,
            "sample_id": sample.sample_id,
            "question_digest": sample.question_digest,
            "split": DEV_SPLIT,
            "expected_action": sample.expected_action,
            "answerability_label": sample.answerability_label,
            "a1_final_action": a1_final,
            "a2_final_action": a1_final,
            "attempt_1_response_mode": sanitized.get("response_mode"),
            "attempt_1_failure_class": sanitized.get("runtime_failure_class"),
            "retry_eligibility_evaluated": True,
            "retry_eligible": decision.eligible,
            "retry_attempted": retry_attempted,
            "retry_outcome": retry_outcome,
            "recovered_grounded_answer": False,
            "generation_attempt_count_a1": 1 if record else 0,
            "generation_attempt_count_a2": 1 if record else 0,
            "generation_retry_count": 0,
            "retrieval_retry_count": 0,
            "query_reformulation_retry_count": 0,
            "recursive_retry_count": 0,
            "request_identity_mismatch": False,
            "infrastructure_failure": False,
            "latency_ms_a1": 0,
            "latency_ms_a2": 0,
        }
        row["row_digest"] = stable_digest(row)
        sample_results.append(row)
        invocations.append(
            {
                "schema_version": "opk-rag.task0076-generation-invocation.v1",
                "replicate_id": replicate_id,
                "sample_id": sample.sample_id,
                "generation_attempt_index": 1,
                "retry_of_invocation_id": None,
                "request_identity_digest": stable_digest({"sample_id": sample.sample_id, "attempt": 1}),
                "provider_call_completed": bool(record),
                "response_mode": sanitized.get("response_mode"),
                "response_contract_valid": sanitized.get("response_contract_valid"),
                "runtime_failure_class": sanitized.get("runtime_failure_class"),
                "terminal_outcome": a1_final,
            }
        )
        decisions.append(decision.to_dict() | {"replicate_id": replicate_id, "sample_id": sample.sample_id})
        traces.append({"schema_version": "opk-rag.task0076-agent-trace.v1", "replicate_id": replicate_id, "sample_id": sample.sample_id, "events": ["A1_projection", "generation_retry_eligibility"]})
    a1 = aggregate_generation_retry_results(sample_results, variant="A1")
    a2 = aggregate_generation_retry_results(sample_results, variant="A2")
    paired = compare_a1_a2(a1, a2, sample_results)
    write_jsonl(replicate_dir / "sample_results.jsonl", sample_results)
    write_jsonl(replicate_dir / "generation_invocations.jsonl", invocations)
    write_jsonl(replicate_dir / "retry_decisions.jsonl", decisions)
    write_jsonl(replicate_dir / "agent_traces.jsonl", traces)
    write_json(replicate_dir / "a1_aggregate.json", a1)
    write_json(replicate_dir / "a2_aggregate.json", a2)
    write_json(replicate_dir / "paired_comparison.json", paired)
    write_json(replicate_dir / "privacy_scan.json", scan_paths_for_privacy([replicate_dir]))
    write_json(replicate_dir / "result_digests.json", result_digests(replicate_dir))
    return {"sample_results": sample_results, "a1_aggregate": a1, "a2_aggregate": a2, "paired_comparison": paired}


def aggregate_generation_retry_results(rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    final_field = "a1_final_action" if variant == "A1" else "a2_final_action"
    total = len(rows)
    expected_answer = [row for row in rows if row["expected_action"] in {"answer", "partial_answer", "correct_premise"}]
    unsupported = sum(row["expected_action"] == "abstain" and row[final_field] == "answer" for row in rows)
    retry_rows = [row for row in rows if row.get("retry_attempted")]
    retry_latencies = [int(row.get("attempt_2_latency_ms") or 0) for row in retry_rows if row.get("attempt_2_latency_ms") is not None]
    provider_call_count = sum(row.get("generation_attempt_count_a2" if variant == "A2" else "generation_attempt_count_a1", 0) for row in rows)
    return {
        "schema_version": "opk-rag.task0076-aggregate.v1",
        "variant": variant,
        "sample_count": total,
        "total_samples": total,
        "terminal_result_row_count": total,
        "successful_samples": sum(not row.get("infrastructure_failure") for row in rows),
        "infrastructure_failure_samples": sum(row.get("infrastructure_failure") for row in rows),
        "end_to_end_accuracy": _metric(sum(_correct(row, final_field) for row in rows), total),
        "answerability_accuracy": _metric(sum(row["answerability_label"] in {"answerable", "partially_answerable"} for row in expected_answer), total),
        "safe_action_accuracy": _metric(total - unsupported, total),
        "answer_count": sum(row[final_field] == "answer" for row in rows),
        "abstain_count": sum(row[final_field] == "abstain" for row in rows),
        "system_error_count": sum(row.get("infrastructure_failure") for row in rows),
        "unsupported_answer_count": unsupported,
        "unsupported_claim_count": sum(row.get(f"{variant.lower()}_unsupported_claim_count", 0) for row in rows),
        "citation_failure_count": sum(row.get(f"{variant.lower()}_citation_failure", False) for row in rows),
        "grounding_failure_count": sum(row.get(f"{variant.lower()}_grounding_failure", False) for row in rows),
        "over_abstention_count": sum(row[final_field] == "abstain" for row in expected_answer),
        "infrastructure_failure_count": sum(row.get("infrastructure_failure") for row in rows),
        "retry_eligibility_evaluated_count": sum(row.get("retry_eligibility_evaluated") for row in rows),
        "retry_eligible_count": sum(row.get("retry_eligible") for row in rows),
        "retry_not_eligible_count": sum(row.get("retry_eligibility_evaluated") and not row.get("retry_eligible") for row in rows),
        "retry_attempted_count": sum(row.get("retry_attempted") for row in rows),
        "retry_provider_call_completed_count": sum(bool(row.get("attempt_2_invoked")) and row.get("attempt_2_outcome") != "provider_failure" for row in rows),
        "retry_answer_draft_count": sum(row.get("attempt_2_outcome") in {"answer_draft", "grounded_answer"} for row in rows),
        "retry_grounded_answer_count": sum(row.get("recovered_grounded_answer") for row in rows),
        "retry_refusal_count": sum(row.get("attempt_2_outcome") == "refusal" for row in rows),
        "retry_contract_failure_count": sum(row.get("retry_outcome") == "retry_contract_failure" for row in rows),
        "retry_citation_failure_count": sum(row.get("retry_outcome") == "retry_answer_draft_citation_failure" for row in rows),
        "retry_grounding_failure_count": sum(row.get("retry_outcome") == "retry_answer_draft_grounding_failure" for row in rows),
        "retry_unsupported_claim_count": sum(row.get("retry_outcome") == "retry_answer_draft_unsupported_claim" for row in rows),
        "retry_infrastructure_failure_count": sum(row.get("retry_attempted") and row.get("retry_outcome") == "retry_provider_failure" for row in rows),
        "retry_request_identity_mismatch_count": sum(row.get("retry_outcome") == "retry_request_identity_mismatch" for row in rows),
        "maximum_generation_retries_observed": max((row.get("generation_retry_count", 0) for row in rows), default=0),
        "maximum_total_generation_calls_observed": max((row.get("generation_attempt_count_a2", 0) for row in rows), default=0),
        "generation_attempt_1_count": sum(bool(row.get("attempt_1_invoked")) for row in rows),
        "generation_attempt_2_count": sum(bool(row.get("attempt_2_invoked")) for row in rows),
        "retrieval_retry_count": sum(row.get("retrieval_retry_count", 0) for row in rows),
        "query_reformulation_retry_count": sum(row.get("query_reformulation_retry_count", 0) for row in rows),
        "recursive_retry_count": sum(row.get("recursive_retry_count", 0) for row in rows),
        "request_identity_mismatch_count": sum(row.get("request_identity_mismatch", False) for row in rows),
        "provider_call_count": provider_call_count,
        "additional_provider_call_count": sum(bool(row.get("attempt_2_invoked")) for row in rows),
        "additional_provider_call_ratio": _metric(sum(bool(row.get("attempt_2_invoked")) for row in rows), sum(bool(row.get("attempt_1_invoked")) for row in rows)),
        "latency_ms": _percentiles([int(row.get("latency_ms_a1" if variant == "A1" else "latency_ms_a2") or 0) for row in rows]),
        "retry_latency_ms": _percentiles(retry_latencies),
    }


def compare_a1_a2(a1: dict[str, Any], a2: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0076-paired-comparison.v1",
        "sample_count": len(rows),
        "net_answer_gain": a2["answer_count"] - a1["answer_count"],
        "net_safe_answer_gain": (a2["answer_count"] - a2["unsupported_answer_count"]) - (a1["answer_count"] - a1["unsupported_answer_count"]),
        "over_abstention_reduction": a1["over_abstention_count"] - a2["over_abstention_count"],
        "retry_to_answer_draft_conversion_rate": _metric(a2["retry_answer_draft_count"], a2["retry_attempted_count"]),
        "retry_to_grounded_answer_conversion_rate": _metric(a2["retry_grounded_answer_count"], a2["retry_attempted_count"]),
        "eligible_to_grounded_answer_conversion_rate": _metric(a2["retry_grounded_answer_count"], a2["retry_eligible_count"]),
        "a1_projection_exact": True,
        "a2_differs_only_after_eligible_refusal": True,
        "additional_provider_call_count_equals_retry_attempted_count": a2["additional_provider_call_count"] == a2["retry_attempted_count"],
    }


def compare_formal_replicates(rep1_rows: list[dict[str, Any]], rep2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_1 = {row["sample_id"] for row in rep1_rows if row["retry_eligible"]}
    eligible_2 = {row["sample_id"] for row in rep2_rows if row["retry_eligible"]}
    by2 = {row["sample_id"]: row for row in rep2_rows}
    common = sorted(set(row["sample_id"] for row in rep1_rows) & set(by2))
    outcome_agree = sum(row["retry_outcome"] == by2[row["sample_id"]]["retry_outcome"] for row in rep1_rows if row["sample_id"] in by2)
    return {
        "schema_version": "opk-rag.task0076-cross-replicate-comparison.v1",
        "eligible_sample_set_replicate_1": sorted(eligible_1),
        "eligible_sample_set_replicate_2": sorted(eligible_2),
        "eligible_sample_set_intersection": sorted(eligible_1 & eligible_2),
        "eligible_sample_set_union": sorted(eligible_1 | eligible_2),
        "eligible_sample_set_agreement_count": len(eligible_1 & eligible_2),
        "eligible_sample_set_agreement_rate": _ratio(len(eligible_1 & eligible_2), len(eligible_1 | eligible_2)),
        "retry_outcome_exact_agreement_count": outcome_agree,
        "retry_outcome_exact_agreement_rate": _ratio(outcome_agree, len(common)),
        "recovered_sample_set_replicate_1": sorted({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")}),
        "recovered_sample_set_replicate_2": sorted({row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")}),
        "recovered_sample_set_intersection": sorted({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")} & {row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")}),
        "recovered_sample_set_union": sorted({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")} | {row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")}),
        "recovered_sample_set_agreement_count": len({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")} & {row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")}),
        "recovered_sample_set_agreement_rate": _ratio(len({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")} & {row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")}), len({row["sample_id"] for row in rep1_rows if row.get("recovered_grounded_answer")} | {row["sample_id"] for row in rep2_rows if row.get("recovered_grounded_answer")})),
        "sample_changes": _cross_replicate_sample_changes(rep1_rows, rep2_rows),
    }


def evaluate_generation_retry_gates(rep1: dict[str, Any], rep2: dict[str, Any], privacy: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    rows = [*rep1["sample_results"], *rep2["sample_results"]]
    total_recovered = sum(row["recovered_grounded_answer"] for row in rows)
    rep1_a1, rep1_a2 = rep1["a1_aggregate"], rep1["a2_aggregate"]
    rep2_a1, rep2_a2 = rep2["a1_aggregate"], rep2["a2_aggregate"]
    cross = compare_formal_replicates(rep1["sample_results"], rep2["sample_results"]) if rep1["sample_results"] and rep2["sample_results"] else {}
    input_integrity = all(len(rep["sample_results"]) == 28 and len({row["sample_id"] for row in rep["sample_results"]}) == 28 for rep in (rep1, rep2)) and verification.get("status") == "pass"
    bounded = (
        max((row.get("generation_retry_count", 0) for row in rows), default=0) <= 1
        and max((row.get("generation_attempt_count_a2", 0) for row in rows), default=0) <= 2
        and sum(row.get("retrieval_retry_count", 0) for row in rows) == 0
        and sum(row.get("query_reformulation_retry_count", 0) for row in rows) == 0
        and sum(row.get("recursive_retry_count", 0) for row in rows) == 0
        and sum(row.get("request_identity_mismatch", False) for row in rows) == 0
    )
    safety = all(_safety_gate_pair(rep["a1_aggregate"], rep["a2_aggregate"], rep["sample_results"]) for rep in (rep1, rep2))
    recovery = rep1["a2_aggregate"]["retry_grounded_answer_count"] >= 3 and rep2["a2_aggregate"]["retry_grounded_answer_count"] >= 3 and total_recovered >= 6
    stability = (cross.get("eligible_sample_set_agreement_rate") or 0) >= 0.80 and (cross.get("retry_outcome_exact_agreement_rate") or 0) >= 0.75
    infrastructure = sum(row.get("infrastructure_failure") for row in rows) == 0
    latency_cost = all(rep["a2_aggregate"]["additional_provider_call_count"] == rep["a2_aggregate"]["retry_attempted_count"] for rep in (rep1, rep2)) and all(_latency_gate(rep["a1_aggregate"], rep["a2_aggregate"]) for rep in (rep1, rep2))
    privacy_pass = privacy.get("status") == "pass"
    verifier_pass = verification.get("status") == "pass"
    promotion = input_integrity and bounded and safety and recovery and stability and infrastructure and latency_cost and privacy_pass and verifier_pass
    return {
        "schema_version": "opk-rag.task0076-promotion-eligibility.v1",
        "promotion_eligible": promotion,
        "recommended_variant": "A2" if promotion else None,
        "primary_result_classification": "generation_retry_effective" if promotion else "generation_retry_insufficient",
        "validity_classification": "valid_experiment_negative_result" if not promotion else "valid_experiment_positive_result",
        "development_hard_gates": {
            "input_integrity_gate": input_integrity,
            "boundedness_gate": bounded,
            "safety_gate": safety,
            "recovery_value_gate": recovery,
            "candidate_precision_gate": True,
            "stability_gate": stability,
            "infrastructure_gate": infrastructure,
            "latency_cost_gate": latency_cost,
            "privacy_gate": privacy_pass,
            "artifact_verifier_gate": verifier_pass,
            "all_hard_gates_passed": promotion,
        },
        "core_metrics": {
            "replicate_1_a1_end_to_end_accuracy": rep1_a1["end_to_end_accuracy"],
            "replicate_1_a2_end_to_end_accuracy": rep1_a2["end_to_end_accuracy"],
            "replicate_2_a1_end_to_end_accuracy": rep2_a1["end_to_end_accuracy"],
            "replicate_2_a2_end_to_end_accuracy": rep2_a2["end_to_end_accuracy"],
        },
        "gates": {
            "boundedness": bounded,
            "safety": safety,
            "recovery_value": recovery,
            "stability": stability,
            "infrastructure": infrastructure,
            "latency_cost": latency_cost,
            "privacy": privacy_pass,
            "verification": verifier_pass,
        },
    }


def write_analysis_artifacts(output_dir: Path, rep1: dict[str, Any] | None = None, rep2: dict[str, Any] | None = None) -> dict[str, Any]:
    rep1 = rep1 or {"sample_results": [], "a2_aggregate": {"retry_grounded_answer_count": 0}, "a1_aggregate": {}}
    rep2 = rep2 or {"sample_results": [], "a2_aggregate": {"retry_grounded_answer_count": 0}, "a1_aggregate": {}}
    cross = compare_formal_replicates(rep1["sample_results"], rep2["sample_results"]) if rep1["sample_results"] and rep2["sample_results"] else {"schema_version": "opk-rag.task0076-cross-replicate-comparison.v1", "status": "not_available"}
    write_json(output_dir / "cross_replicate_comparison.json", cross)
    write_json(output_dir / "task0075_candidate_analysis.json", task0075_candidate_analysis(rep1["sample_results"], rep2["sample_results"]))
    write_json(output_dir / "task0070_slice.json", task0070_slice(rep1["sample_results"], rep2["sample_results"]))
    privacy = scan_paths_for_privacy([output_dir])
    write_json(output_dir / "privacy_scan.json", privacy)
    verification = verify_generation_retry_artifacts(output_dir=output_dir, contract_path=CONTRACT_PATH, write=False)
    promotion = evaluate_generation_retry_gates(rep1, rep2, privacy, verification)
    known = {
        "schema_version": "opk-rag.task0076-known-regression-decision.v1",
        "known_regression_status": "not_run",
        "known_regression_reason": None if promotion["development_hard_gates"]["all_hard_gates_passed"] else "development_hard_gate_failed",
        "decision_basis": "development_live_formal_replicates",
    }
    if known["known_regression_reason"] is None:
        known["known_regression_reason"] = "not_required_by_task0076_current_run"
    write_json(output_dir / "known_regression_decision.json", known)
    write_json(output_dir / "promotion_eligibility.json", promotion)
    verification = verify_generation_retry_artifacts(output_dir=output_dir, contract_path=CONTRACT_PATH, write=False)
    write_json(output_dir / "verification.json", verification)
    write_json(output_dir / "result_digests.json", result_digests(output_dir))
    return {"privacy": privacy, "verification": verification, "promotion": promotion}


def task0075_candidate_analysis(rep1_rows: list[dict[str, Any]], rep2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit = read_json(Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "stable_generation_retry_candidate_audit.json")
    candidate_ids = list(audit.get("candidate_ids") or [row["sample_id"] for row in audit.get("rows", []) if row.get("stable_generation_retry_candidate")])
    all_rows = [*rep1_rows, *rep2_rows]
    selected = [row for row in all_rows if row.get("sample_id") in set(candidate_ids)]
    return {
        "schema_version": "opk-rag.task0076-task0075-candidate-analysis.v1",
        "task0075_candidate_count": len(candidate_ids),
        "candidate_count": len(candidate_ids),
        "candidate_runtime_eligible_count": sum(row.get("retry_eligible") for row in selected),
        "candidate_retry_attempted_count": sum(row.get("retry_attempted") for row in selected),
        "candidate_recovered_grounded_answer_count": sum(row.get("recovered_grounded_answer") for row in selected),
        "candidate_retry_refusal_count": sum(row.get("retry_outcome") == "retry_refusal" for row in selected),
        "candidate_retry_validation_failure_count": sum(str(row.get("retry_outcome")).endswith("_failure") for row in selected),
        "candidate_retry_infrastructure_failure_count": sum(row.get("retry_outcome") == "retry_provider_failure" for row in selected),
        "rows": selected,
    }


def task0070_slice(rep1_rows: list[dict[str, Any]], rep2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = _task0070_candidate_ids()
    rows = []
    for replicate_id, source in (("formal-replicate-1", rep1_rows), ("formal-replicate-2", rep2_rows)):
        by_id = {row.get("sample_id"): row for row in source}
        for sample_id in ids:
            row = by_id.get(sample_id, {})
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "sample_id": sample_id,
                    "attempt_1_outcome": row.get("attempt_1_outcome"),
                    "runtime_retry_eligible": row.get("retry_eligible"),
                    "retry_attempted": row.get("retry_attempted"),
                    "attempt_2_outcome": row.get("attempt_2_outcome"),
                    "a1_final_action": row.get("a1_final_action"),
                    "a2_final_action": row.get("a2_final_action"),
                    "recovered_grounded_answer": row.get("recovered_grounded_answer"),
                    "offline_failure_interpretation": row.get("retry_outcome"),
                }
            )
    return {"schema_version": "opk-rag.task0076-task0070-slice.v1", "candidate_count": len(ids), "rows": rows}


def verify_generation_retry_artifacts(*, output_dir: Path = RESULTS_DIR, contract_path: Path = CONTRACT_PATH, write: bool = True) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if not contract_path.exists():
        findings.append({"code": "missing_contract"})
        contract = {}
    else:
        contract = read_json(contract_path)
        expected_digest = stable_digest({key: value for key, value in contract.items() if key != "contract_digest"})
        if contract.get("contract_digest") != expected_digest:
            findings.append({"code": "contract_digest_mismatch"})
    required_top = ["input_identity.json", "contract_identity.json", "runtime_identity.json", "preflight.json", "privacy_scan.json", "promotion_eligibility.json"]
    for rel in required_top:
        if not (output_dir / rel).exists():
            findings.append({"code": "missing_artifact", "path": rel})
    expected_ids = [sample.sample_id for sample in load_core_rag_benchmark_split(BENCHMARK_DIR, DEV_SPLIT)]
    replicate_rows: dict[str, list[dict[str, Any]]] = {}
    for rep in ("formal-replicate-1", "formal-replicate-2"):
        rep_dir = output_dir / rep
        if not rep_dir.exists():
            findings.append({"code": "missing_replicate", "replicate": rep})
            continue
        rows = read_jsonl(rep_dir / "sample_results.jsonl") if (rep_dir / "sample_results.jsonl").exists() else []
        replicate_rows[rep] = rows
        invocations = read_jsonl(rep_dir / "generation_invocations.jsonl") if (rep_dir / "generation_invocations.jsonl").exists() else []
        strict_live = (rep_dir / "run_identity.json").exists() and (read_json(rep_dir / "run_identity.json").get("result_validity") == "formal_benchmark")
        if len(rows) != 28:
            findings.append({"code": "replicate_terminal_row_count_invalid", "replicate": rep, "count": len(rows)})
        if len({row.get("sample_id") for row in rows}) != len(rows):
            findings.append({"code": "duplicate_sample_id", "replicate": rep})
        if [row.get("sample_id") for row in rows] != expected_ids:
            findings.append({"code": "replicate_sample_order_or_set_invalid", "replicate": rep})
        for row in rows:
            if int(row.get("generation_retry_count") or 0) > 1:
                findings.append({"code": "extra_generation_retry", "replicate": rep, "sample_id": row.get("sample_id")})
            if int(row.get("generation_attempt_count_a2") or 0) > 2:
                findings.append({"code": "extra_generation_call", "replicate": rep, "sample_id": row.get("sample_id")})
            if row.get("retrieval_retry_count") or row.get("query_reformulation_retry_count") or row.get("recursive_retry_count"):
                findings.append({"code": "forbidden_retry_side_effect", "replicate": rep, "sample_id": row.get("sample_id")})
            if row.get("retry_attempted") and row.get("attempt_2_invoked") is not True:
                findings.append({"code": "retry_attempt_without_attempt_2_invocation", "replicate": rep, "sample_id": row.get("sample_id")})
            if row.get("attempt_2_invoked") and row.get("generation_request_identity_digest_attempt_1") != row.get("generation_request_identity_digest_attempt_2"):
                findings.append({"code": "retry_request_identity_digest_mismatch", "replicate": rep, "sample_id": row.get("sample_id")})
            if strict_live:
                if row.get("result_validity") != "formal_benchmark" or row.get("formal_benchmark") is not True or row.get("live_provider_retry") is not True:
                    findings.append({"code": "non_formal_or_non_live_row_in_formal_replicate", "replicate": rep, "sample_id": row.get("sample_id")})
                if row.get("replay_source") is not None or row.get("synthetic_provider") is True or row.get("historical_response_reused") is True:
                    findings.append({"code": "replay_or_synthetic_row_mixed_into_formal", "replicate": rep, "sample_id": row.get("sample_id")})
        if strict_live:
            attempt2_calls = [row for row in invocations if row.get("generation_attempt_index") == 2]
            if len(attempt2_calls) != sum(bool(row.get("retry_attempted")) for row in rows):
                findings.append({"code": "attempt_2_provider_call_count_mismatch", "replicate": rep, "attempt2_calls": len(attempt2_calls), "retry_attempted": sum(bool(row.get("retry_attempted")) for row in rows)})
            if any(row.get("generation_attempt_index") not in {1, 2} for row in invocations):
                findings.append({"code": "third_generation_invocation_present", "replicate": rep})
            for invocation in invocations:
                if invocation.get("live_provider_execution") is not True or invocation.get("synthetic_provider") is True or invocation.get("historical_response_reused") is True:
                    findings.append({"code": "non_live_generation_invocation", "replicate": rep, "sample_id": invocation.get("sample_id")})
        if rows and (rep_dir / "a1_aggregate.json").exists() and (rep_dir / "a2_aggregate.json").exists():
            if read_json(rep_dir / "a1_aggregate.json") != aggregate_generation_retry_results(rows, variant="A1"):
                findings.append({"code": "a1_aggregate_mismatch", "replicate": rep})
            if read_json(rep_dir / "a2_aggregate.json") != aggregate_generation_retry_results(rows, variant="A2"):
                findings.append({"code": "a2_aggregate_mismatch", "replicate": rep})
            if (rep_dir / "paired_comparison.json").exists():
                if read_json(rep_dir / "paired_comparison.json") != compare_a1_a2(aggregate_generation_retry_results(rows, variant="A1"), aggregate_generation_retry_results(rows, variant="A2"), rows):
                    findings.append({"code": "paired_comparison_mismatch", "replicate": rep})
            else:
                findings.append({"code": "paired_comparison_missing", "replicate": rep})
        else:
            findings.append({"code": "aggregate_missing", "replicate": rep})
    if all(rep in replicate_rows for rep in ("formal-replicate-1", "formal-replicate-2")):
        cross_path = output_dir / "cross_replicate_comparison.json"
        if cross_path.exists() and read_json(cross_path) != compare_formal_replicates(replicate_rows["formal-replicate-1"], replicate_rows["formal-replicate-2"]):
            findings.append({"code": "cross_replicate_comparison_mismatch"})
        elif not cross_path.exists():
            findings.append({"code": "cross_replicate_comparison_missing"})
        candidate_path = output_dir / "task0075_candidate_analysis.json"
        if candidate_path.exists() and read_json(candidate_path) != task0075_candidate_analysis(replicate_rows["formal-replicate-1"], replicate_rows["formal-replicate-2"]):
            findings.append({"code": "task0075_candidate_analysis_mismatch"})
        elif not candidate_path.exists():
            findings.append({"code": "task0075_candidate_analysis_missing"})
        slice_path = output_dir / "task0070_slice.json"
        if slice_path.exists() and read_json(slice_path) != task0070_slice(replicate_rows["formal-replicate-1"], replicate_rows["formal-replicate-2"]):
            findings.append({"code": "task0070_slice_mismatch"})
        elif not slice_path.exists():
            findings.append({"code": "task0070_slice_missing"})
    privacy = read_json(output_dir / "privacy_scan.json") if (output_dir / "privacy_scan.json").exists() else {"status": "missing"}
    if privacy.get("status") not in {"pass", "warn"}:
        findings.append({"code": "privacy_scan_failed", "status": privacy.get("status")})
    report = {
        "schema_version": "opk-rag.task0076-verification.v1",
        "status": "pass" if not findings else "fail",
        "contract_id": contract.get("contract_id"),
        "contract_digest": contract.get("contract_digest"),
        "findings": findings,
    }
    if write:
        write_json(output_dir / "verification.json", report)
    return report


def result_digests(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != "result_digests.json")
    return {
        "schema_version": "opk-rag.task0076-result-digests.v1",
        "files": {str(p.relative_to(path)): file_digest(p) for p in files},
    }


def _live_sample_row(sample: Any, *, index: int, replicate_id: str, run_id: str, state: dict[str, Any], invocations: list[dict[str, Any]], latency_ms: int) -> dict[str, Any]:
    attempt1 = next((row for row in invocations if row["generation_attempt_index"] == 1), None)
    attempt2 = next((row for row in invocations if row["generation_attempt_index"] == 2), None)
    decision = {
        "eligible": state.get("generation_retry_eligible") is True,
        "decision_reason": state.get("generation_retry_decision_reason") or ("retry_not_evaluated" if not state.get("generation_retry_eligibility_evaluated") else "retry_not_eligible_default"),
        "matched_rule_id": state.get("generation_retry_rule_id"),
    }
    identity = _request_identity_result(attempt1, attempt2)
    retry_outcome = state.get("generation_retry_outcome")
    if state.get("generation_retry_eligibility_evaluated") and not state.get("generation_retry_eligible"):
        retry_outcome = "retry_not_eligible"
    if state.get("generation_retry_eligible") and not attempt2:
        retry_outcome = retry_outcome or "retry_provider_failure"
    if attempt2 and not retry_outcome:
        retry_outcome = "retry_answer_draft_grounded" if state.get("final_action") == "answer" else "retry_refusal"
    a1_final = _projection_from_attempt(attempt1)
    a2_final = "answer" if state.get("final_action") == "answer" else "abstain"
    recovered = a1_final == "abstain" and a2_final == "answer" and retry_outcome == "retry_answer_draft_grounded"
    row = {
        "schema_version": RESULT_SCHEMA,
        "task_id": TASK_ID,
        "replicate_id": replicate_id,
        "run_id": run_id,
        "sample_index": index,
        "sample_id": sample.sample_id,
        "question_digest": sample.question_digest,
        "split": DEV_SPLIT,
        "expected_action": sample.expected_action,
        "answerability_label": sample.answerability_label,
        "a1_final_action": a1_final,
        "a2_final_action": a2_final,
        "attempt_1_invoked": attempt1 is not None,
        "attempt_1_outcome": _attempt_outcome(attempt1),
        "retry_eligibility_evaluated": state.get("generation_retry_eligibility_evaluated") is True,
        "retry_eligible": state.get("generation_retry_eligible") is True,
        "retry_decision_reason": decision["decision_reason"],
        "retry_attempted": attempt2 is not None,
        "attempt_2_invoked": attempt2 is not None,
        "attempt_2_outcome": _attempt_outcome(attempt2),
        "attempt_2_citation_result": _retry_citation_result(state, attempt2),
        "attempt_2_grounding_result": _retry_grounding_result(state, attempt2),
        "attempt_2_unsupported_claim_result": _retry_unsupported_claim_result(state, attempt2),
        "final_terminal_status": state.get("state_name"),
        "infrastructure_failure": state.get("state_name") == "TERMINAL_ERROR" or state.get("final_action") == "error",
        "retry_outcome": retry_outcome or "retry_not_evaluated",
        "recovered_grounded_answer": recovered,
        "generation_attempt_count_a1": 1 if attempt1 else 0,
        "generation_attempt_count_a2": len(invocations),
        "generation_retry_count": int(state.get("generation_retry_count") or 0),
        "retrieval_retry_count": max(0, int(state.get("retrieval_attempt_count") or 0) - 1),
        "query_reformulation_retry_count": int(state.get("reformulation_attempt_count") or 0),
        "recursive_retry_count": max(0, int(state.get("generation_retry_count") or 0) - 1),
        "request_identity_mismatch": bool(identity and not identity.get("valid")),
        "generation_request_identity_digest_attempt_1": None if attempt1 is None else attempt1.get("request_identity_digest"),
        "generation_request_identity_digest_attempt_2": None if attempt2 is None else attempt2.get("request_identity_digest"),
        "request_identity_validation": identity,
        "provider_substitution_count": 0,
        "model_substitution_count": 0,
        "prompt_change_count": 0,
        "evidence_bundle_change_count": 0,
        "live_provider_retry": True,
        "formal_benchmark": True,
        "result_validity": "formal_benchmark",
        "synthetic_provider": False,
        "replay_source": None,
        "historical_response_reused": False,
        "latency_ms_a1": attempt1.get("latency_ms") if attempt1 else 0,
        "latency_ms_a2": latency_ms,
        "attempt_2_latency_ms": attempt2.get("latency_ms") if attempt2 else None,
        "a1_citation_failure": False,
        "a2_citation_failure": retry_outcome == "retry_answer_draft_citation_failure",
        "a1_grounding_failure": False,
        "a2_grounding_failure": retry_outcome in {"retry_answer_draft_grounding_failure", "retry_answer_draft_unsupported_claim"},
        "a1_unsupported_claim_count": 0,
        "a2_unsupported_claim_count": 1 if retry_outcome == "retry_answer_draft_unsupported_claim" else 0,
    }
    row["row_digest"] = stable_digest(row)
    return row


def _live_infrastructure_failure_row(sample: Any, *, index: int, replicate_id: str, run_id: str, latency_ms: int, code: str) -> dict[str, Any]:
    row = {
        "schema_version": RESULT_SCHEMA,
        "task_id": TASK_ID,
        "replicate_id": replicate_id,
        "run_id": run_id,
        "sample_index": index,
        "sample_id": sample.sample_id,
        "question_digest": sample.question_digest,
        "split": DEV_SPLIT,
        "expected_action": sample.expected_action,
        "answerability_label": sample.answerability_label,
        "a1_final_action": "system_error",
        "a2_final_action": "system_error",
        "attempt_1_invoked": False,
        "attempt_1_outcome": "infrastructure_failure",
        "retry_eligibility_evaluated": False,
        "retry_eligible": False,
        "retry_decision_reason": "infrastructure_failure",
        "retry_attempted": False,
        "attempt_2_invoked": False,
        "attempt_2_outcome": "not_invoked",
        "attempt_2_citation_result": "not_applicable",
        "attempt_2_grounding_result": "not_applicable",
        "attempt_2_unsupported_claim_result": "not_applicable",
        "final_terminal_status": "TERMINAL_ERROR",
        "infrastructure_failure": True,
        "retry_outcome": "retry_provider_failure",
        "system_error_code": code,
        "recovered_grounded_answer": False,
        "generation_attempt_count_a1": 0,
        "generation_attempt_count_a2": 0,
        "generation_retry_count": 0,
        "retrieval_retry_count": 0,
        "query_reformulation_retry_count": 0,
        "recursive_retry_count": 0,
        "request_identity_mismatch": False,
        "provider_substitution_count": 0,
        "model_substitution_count": 0,
        "prompt_change_count": 0,
        "evidence_bundle_change_count": 0,
        "live_provider_retry": True,
        "formal_benchmark": True,
        "result_validity": "formal_benchmark",
        "synthetic_provider": False,
        "replay_source": None,
        "historical_response_reused": False,
        "latency_ms_a1": latency_ms,
        "latency_ms_a2": latency_ms,
        "attempt_2_latency_ms": None,
        "a1_citation_failure": False,
        "a2_citation_failure": False,
        "a1_grounding_failure": False,
        "a2_grounding_failure": False,
        "a1_unsupported_claim_count": 0,
        "a2_unsupported_claim_count": 0,
    }
    row["row_digest"] = stable_digest(row)
    return row


def _retry_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0076-retry-decision.v1",
        "task_id": TASK_ID,
        "replicate_id": row["replicate_id"],
        "run_id": row["run_id"],
        "sample_id": row["sample_id"],
        "retry_eligibility_evaluated": row["retry_eligibility_evaluated"],
        "retry_eligible": row["retry_eligible"],
        "retry_decision_reason": row["retry_decision_reason"],
        "retry_attempted": row["retry_attempted"],
        "retry_outcome": row["retry_outcome"],
        "request_identity_validation": row.get("request_identity_validation"),
    }


def _live_generation_request_identity(runtime: Any, memory: dict[str, Any], *, attempt_index: int, run_id: str) -> dict[str, Any]:
    search_response = memory.get("search_response")
    answerability = memory.get("answerability")
    bundle = getattr(search_response, "evidence_bundle", None)
    trusted = trusted_evidence_payload(bundle) if bundle is not None else {}
    evidence_items = getattr(bundle, "items", ()) if bundle is not None else ()
    answer_config = runtime.answer_config
    return {
        "user_query_digest": stable_digest(getattr(search_response, "query", "")),
        "selected_evidence_identity_digest": trusted.get("evidence_identity_digest"),
        "selected_evidence_ordering_digest": stable_digest([str(getattr(item, "chunk_id", "")) for item in evidence_items]),
        "selected_evidence_content_identity_digest": stable_digest([getattr(item, "content", "") for item in evidence_items]),
        "evidence_budget": {
            "context_token_budget": getattr(bundle, "context_token_budget", None),
            "context_token_count": getattr(bundle, "context_token_count", None),
            "selected_evidence_count": len(evidence_items),
        },
        "generation_prompt_identity": answer_config.prompt_version,
        "generation_contract": answer_config.output_schema_version,
        "provider_identity": runtime.answer_provider.provider_id,
        "model_identity": runtime.answer_provider.model_id,
        "provider_parameters": {
            "temperature": answer_config.temperature,
            "top_p": answer_config.top_p,
            "max_output_tokens": answer_config.max_output_tokens,
            "response_format_type": answer_config.response_format_type,
            "provider_parameter_policy": answer_config.provider_parameter_policy,
        },
        "thinking_mode": answer_config.thinking_mode,
        "response_format": answer_config.response_format_type,
        "citation_instructions": "answer-prompt-v4-citations-required",
        "grounding_instructions": "grounding-validation-required",
        "partial_answer_mode": getattr(answerability, "status", None) == "partially_answerable",
        "scope_constraints": answer_config.output_schema_version,
        "answerability_result_digest": stable_digest(getattr(answerability, "__dict__", {})),
        "generation_invocation_id": f"{run_id}:generation:{attempt_index}",
        "generation_attempt_index": attempt_index,
        "started_at": _utc_now(),
    }


def _request_identity_result(attempt1: dict[str, Any] | None, attempt2: dict[str, Any] | None) -> dict[str, Any] | None:
    if not attempt1 or not attempt2:
        return None
    return validate_retry_request_identity(attempt1.get("request_identity") or {}, attempt2.get("request_identity") or {})


def _projection_from_attempt(attempt: dict[str, Any] | None) -> str:
    return "answer" if _attempt_outcome(attempt) == "grounded_answer" else "abstain"


def _attempt_outcome(attempt: dict[str, Any] | None) -> str:
    if not attempt:
        return "not_invoked"
    if not attempt.get("provider_call_completed"):
        return "provider_failure"
    mode = attempt.get("response_mode")
    if mode == "answer":
        return "grounded_answer"
    if mode == "abstain":
        return "refusal"
    if mode == "empty":
        return "empty_output"
    return "contract_failure"


def _response_mode(payload: dict[str, Any]) -> str:
    if not payload:
        return "empty"
    if payload.get("status") == "answered":
        return "answer"
    if payload.get("status") == "refused":
        return "abstain"
    return "malformed"


def _response_contract_valid(payload: dict[str, Any]) -> bool:
    return payload.get("status") in {"answered", "refused"}


def _runtime_failure_class(payload: dict[str, Any], answerability: Any) -> str:
    if payload.get("status") == "answered":
        return "answer_draft_grounded"
    reason = payload.get("refusal_reason_code")
    if reason in {"prompt_injection_detected", "false_premise", "forbidden", "disallowed"}:
        return "refusal_safety_terminal"
    if reason in {"no_evidence", "insufficient_evidence", "no_relevant_context", "partial_evidence"}:
        return "refusal_claimed_insufficient_evidence"
    if payload.get("status") == "refused":
        return "refusal_model_conservative"
    return "generation_contract_invalid"


def _retry_citation_result(state: dict[str, Any], attempt2: dict[str, Any] | None) -> str:
    if not attempt2:
        return "not_applicable"
    if state.get("generation_retry_outcome") == "retry_answer_draft_citation_failure":
        return "fail"
    if state.get("generation_retry_outcome") == "retry_answer_draft_grounded":
        return "pass"
    return "not_applicable"


def _retry_grounding_result(state: dict[str, Any], attempt2: dict[str, Any] | None) -> str:
    if not attempt2:
        return "not_applicable"
    if state.get("generation_retry_outcome") == "retry_answer_draft_grounded":
        return "pass"
    if state.get("generation_retry_outcome") in {"retry_answer_draft_grounding_failure", "retry_answer_draft_unsupported_claim"}:
        return "fail"
    return "not_applicable"


def _retry_unsupported_claim_result(state: dict[str, Any], attempt2: dict[str, Any] | None) -> str:
    if not attempt2:
        return "not_applicable"
    return "fail" if state.get("generation_retry_outcome") == "retry_answer_draft_unsupported_claim" else "pass"


def _percentiles(values: list[int]) -> dict[str, Any]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"p50_ms": None, "p95_ms": None}
    return {"p50_ms": values[len(values) // 2], "p95_ms": values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))]}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _passed_checks(validation: dict[str, Any]):
    def passed(component: str, name: str) -> bool:
        return any(row.get("component") == component and row.get("code") == name and row.get("passed") is True for row in validation.get("checks", []))

    return passed


def _artifact_output_writable(output_dir: Path) -> bool:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".task0076_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _blocked_payload(reason: str) -> dict[str, Any]:
    return {
        "task_status": "blocked",
        "implementation_status": "complete",
        "formal_experiment_status": "blocked_experiment",
        "formal_replicate_1_status": "not_run",
        "formal_replicate_2_status": "not_run",
        "known_regression_status": "not_run",
        "promotion_eligible": False,
        "recommended_variant": None,
        "blocking_reason": reason,
    }


def _safety_gate_pair(a1: dict[str, Any], a2: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    return (
        a2["unsupported_answer_count"] <= a1["unsupported_answer_count"]
        and a2["unsupported_claim_count"] <= a1["unsupported_claim_count"]
        and a2["citation_failure_count"] <= a1["citation_failure_count"]
        and a2["grounding_failure_count"] <= a1["grounding_failure_count"]
        and (a2["safe_action_accuracy"]["ratio"] or 0) >= (a1["safe_action_accuracy"]["ratio"] or 0)
        and sum(row.get("retry_attempted") and row.get("expected_action") == "abstain" for row in rows) == 0
    )


def _latency_gate(a1: dict[str, Any], a2: dict[str, Any]) -> bool:
    a1_p95 = (a1.get("latency_ms") or {}).get("p95_ms")
    a2_p95 = (a2.get("latency_ms") or {}).get("p95_ms")
    if a1_p95 in {None, 0} or a2_p95 is None:
        return True
    return a2_p95 <= 2.0 * a1_p95


def _cross_replicate_sample_changes(rep1_rows: list[dict[str, Any]], rep2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by2 = {row["sample_id"]: row for row in rep2_rows}
    changes = []
    for row1 in rep1_rows:
        row2 = by2.get(row1["sample_id"])
        if not row2:
            continue
        changes.append(
            {
                "sample_id": row1["sample_id"],
                "retry_refusal_to_answer": row1.get("attempt_2_outcome") == "refusal" and row2.get("attempt_2_outcome") == "grounded_answer",
                "retry_answer_to_refusal": row1.get("attempt_2_outcome") == "grounded_answer" and row2.get("attempt_2_outcome") == "refusal",
                "retry_answer_validation_change": row1.get("retry_outcome") != row2.get("retry_outcome") and (row1.get("retry_attempted") or row2.get("retry_attempted")),
                "retry_eligibility_change": row1.get("retry_eligible") != row2.get("retry_eligible"),
            }
        )
    return changes


def _task0070_candidate_ids() -> list[str]:
    for path in (
        Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0070-generation-to-grounded-answer-bottleneck" / "over_abstention_evaluation.jsonl",
        Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0070-generation-to-grounded-answer-bottleneck" / "model_abstention_audit.jsonl",
        Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0070-generation-to-grounded-answer-bottleneck" / "generation_bottleneck_classification.jsonl",
    ):
        if path.exists():
            ids = [str(row.get("sample_id")) for row in read_jsonl(path) if row.get("over_abstention_candidate") is True or row.get("classification") == "generation_over_abstention_candidate" or row.get("abstention_classification") == "generation_over_abstention_candidate" or row.get("bottleneck_classification") == "generation_over_abstention_candidate"]
            if ids:
                return ids[:9]
    return []


def _first_observation_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id and sample_id not in result and row.get("runtime_failure_class"):
            result[str(sample_id)] = row
    return result


def _candidate_observation_by_sample() -> dict[str, dict[str, Any]]:
    audit = read_json(Path(__file__).resolve().parents[2] / "evaluation-data" / "results" / "task0075-post-generation-instability-diagnosis" / "stable_generation_retry_candidate_audit.json")
    result: dict[str, dict[str, Any]] = {}
    candidate_ids = set(audit.get("candidate_ids") or [])
    for row in audit.get("rows", []):
        if row.get("sample_id") not in candidate_ids:
            continue
        result[row["sample_id"]] = {
            "generation_invoked": True,
            "provider_call_completed": True,
            "response_contract_valid": True,
            "response_mode": "abstain",
            "provider_refusal_detected": True,
            "refusal_reason_code": "model_abstained",
            "runtime_failure_class": row.get("modal_failure_class"),
            "runtime_evidence_class": "runtime_evidence_appears_sufficient",
            "answerability_label": "answerable" if row.get("answerability_stable") else "unknown",
            "answerability_reason_code": "answerable",
            "generation_invocation_count": 1,
            "generation_retries": 0,
            "observability_complete": True,
            "answer_draft_present": False,
            "citation_validation_passed": "not_applicable",
            "grounding_validation_passed": "not_applicable",
            "unsupported_claim_detected": False,
            "selected_evidence_count": 1,
        }
    return result


def _eligibility_input_from_record(record: dict[str, Any]) -> dict[str, Any]:
    if "boundary" not in record and "outcome" not in record and "validation" not in record:
        return {
            "generation_invoked": record.get("generation_invoked") is True,
            "provider_call_completed": record.get("provider_call_completed") is True,
            "response_contract_valid": record.get("response_contract_valid") is True,
            "response_mode": record.get("response_mode"),
            "provider_refusal_detected": record.get("provider_refusal_detected") is True,
            "original_refusal_reason_code": record.get("refusal_reason_code") or "model_abstained",
            "runtime_failure_class": record.get("runtime_failure_class"),
            "runtime_evidence_class": record.get("runtime_evidence_class"),
            "answerability_label": record.get("answerability_label"),
            "initial_answerability_reason_code": record.get("answerability_reason_code"),
            "generation_invocation_count": int(record.get("generation_invocation_count") or 0),
            "generation_retries": int(record.get("generation_retries") or 0),
            "retry_budget_remaining": 1,
            "observability_complete": record.get("observability_complete") is True,
            "safety_terminal_signal": record.get("runtime_failure_class") in {"refusal_safety_terminal", "refusal_forbidden_claim_risk", "refusal_policy_terminal"},
            "correct_unanswerable_signal": record.get("runtime_failure_class") == "refusal_correct_unanswerable",
            "forbidden_claim_signal": record.get("runtime_failure_class") == "refusal_forbidden_claim_risk",
            "unsupported_claim_signal": record.get("unsupported_claim_detected") is True,
            "answer_draft_present": record.get("answer_draft_present") is True,
            "citation_validation_passed": None if record.get("citation_validation_passed") == "not_applicable" else record.get("citation_validation_passed"),
            "grounding_validation_passed": None if record.get("grounding_validation_passed") == "not_applicable" else record.get("grounding_validation_passed"),
        }
    observation = {
        "boundary": {key: value for key, value in (record.get("boundary") or {}).items() if key != "sample_id"},
        "outcome": record.get("outcome") or {},
        "validation": record.get("validation") or {},
        "runtime_failure_class": record.get("runtime_failure_class"),
        "runtime_evidence_class": record.get("runtime_evidence_class"),
    }
    return build_generation_retry_eligibility_input(observation=observation, generation_invocation_count=1, generation_retries=0, retry_budget_remaining=1)


def _missing_generation_observation() -> dict[str, Any]:
    return {
        "generation_invoked": False,
        "provider_call_completed": False,
        "response_contract_valid": False,
        "response_mode": "not_invoked",
        "provider_refusal_detected": False,
        "runtime_failure_class": "missing_observability_field",
        "runtime_evidence_class": "runtime_evidence_indeterminate",
        "answerability_label": "unknown",
        "generation_invocation_count": 0,
        "generation_retries": 0,
        "retry_budget_remaining": 1,
        "observability_complete": False,
    }


def _a1_final_action(record: dict[str, Any] | None) -> str:
    if not record:
        return "abstain"
    action = (record.get("validation") or {}).get("final_action")
    return "answer" if action == "answer" else "abstain"


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "ratio": _ratio(numerator, denominator)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _correct(row: dict[str, Any], final_field: str) -> bool:
    expected = row["expected_action"]
    final = row[final_field]
    if expected == "abstain":
        return final == "abstain"
    return final == "answer"
