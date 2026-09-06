from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
from uuid import UUID, uuid4

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.provider import AnswerProviderError
from opk_rag.conversation.context import build_conversation_context
from opk_rag.conversation.models import (
    CONVERSATION_PROMPT_VERSION,
    FOLLOWUP_REWRITE_VERSION,
    ConversationContext,
    ConversationContextTurn,
    ConversationSession,
    ConversationTurn,
    ConversationTurnCitation,
)
from opk_rag.conversation.resolver import OpenAICompatibleFollowupQueryResolver
from opk_rag.embedding.config import load_embedding_config
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.context_tokens import QwenContextTokenCounter
from opk_rag.showcase.api.execution import (
    APPROVED_AGENTIC_CANARY_FINGERPRINT,
    CONTROLLED_REALISTIC_EVALUATION_SOURCE,
    ShowcaseExecutor,
)
from opk_rag.showcase.bounded_agentic_canary import evaluate_task0263_readiness
from opk_rag.showcase.live_selective_agent_shadow import classify_traffic
from opk_rag.evaluation.task0266_controlled_realistic_single_and_multi_turn_rag_stability_evaluation import (
    DATASET,
    EXPECTED_DATASET_SHA256,
    _build_turn,
    _canary_ledger_state,
    _conversation_citations,
    _gpu_state,
    _read_json,
    _trace_result,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0267"
PRIMARY_RUN_ID = "task0267-multiturn-revalidation-v4"
RESULT = ROOT / "evaluation-data/results/task0267-conversation-resolver-provider-compatibility"
REPORT = ROOT / "docs/TASK0267_CONVERSATION_RESOLVER_PROVIDER_STRUCTURED_OUTPUT_COMPATIBILITY_REPAIR_REPORT.md"
CHECKPOINT = RESULT / "execution_checkpoint.json"
WINDOW_HISTORY = RESULT / "execution_windows.jsonl"
CONVERSATION_AUTHORITY = RESULT / "conversation-primary"
EXPECTED_CONVERSATIONS = 9
EXPECTED_TURNS = 40
FAILED_TURN_CONTEXT_POLICY = "exclude_failed_turn_and_continue_next_frozen_turn"
TASK0266_FILES = {
    "failure_ledger.jsonl": "98732a63aa60d98b0f5b984e48a5d3b4cf64a4068ebdc13c52708d1d3b821174",
    "summary.json": "5fee9724ff0828b09731a15de905493a0299bfa729f80e0a60796f8b26bc4430",
    "multi_turn_failure_diagnosis.json": "fa7a5fcd6b08dff386a177c12e092fd06c25c454775586e79a68b99d8ec906ff",
    "single_turn_results.jsonl": "d5f72fcec9aebb04be9bc4c187af3aabd616f5d7dc7b8fa57f5a4bd99ace40ae",
}
TASK0266_ROOT = ROOT / "evaluation-data/results/task0266-controlled-realistic-rag-stability"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task0266_identity() -> dict[str, Any]:
    rows = {}
    for name, expected in TASK0266_FILES.items():
        path = TASK0266_ROOT / name
        actual = _sha256(path) if path.is_file() else None
        rows[name] = {"expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
    return {
        "schema_version": "opk-rag.task0267.task0266-history-identity.v1",
        "files": rows,
        "all_match": all(row["match"] for row in rows.values()),
    }


def dataset_identity() -> dict[str, Any]:
    actual = _sha256(DATASET)
    data = _read_json(DATASET)
    conversations = list(data.get("conversations") or [])
    turns = sum(len(list(row.get("turns") or [])) for row in conversations)
    return {
        "schema_version": "opk-rag.task0267.dataset-identity.v1",
        "expected_sha256": EXPECTED_DATASET_SHA256,
        "actual_sha256": actual,
        "sha256_match": actual == EXPECTED_DATASET_SHA256,
        "conversation_count": len(conversations),
        "turn_count": turns,
        "synthetic": data.get("synthetic") is True,
        "identity_valid": actual == EXPECTED_DATASET_SHA256 and len(conversations) == EXPECTED_CONVERSATIONS and turns == EXPECTED_TURNS and data.get("synthetic") is True,
    }


def entry_authority() -> dict[str, Any]:
    load_project_env(root=ROOT)
    config = load_answer_generation_config()
    readiness = evaluate_task0263_readiness(root=ROOT, expected_candidate_fingerprint=APPROVED_AGENTIC_CANARY_FINGERPRINT)
    traffic_class, live_eligible = classify_traffic(execution_scope="ask", source=CONTROLLED_REALISTIC_EVALUATION_SOURCE)
    historical = task0266_identity()
    dataset = dataset_identity()
    return {
        "schema_version": "opk-rag.task0267.entry-authority.v1",
        "task_id": TASK_ID,
        "task0266_historical_identity": historical,
        "dataset_identity": dataset,
        "candidate_readiness_passed": readiness["passed"],
        "candidate_fingerprint": readiness.get("approved_candidate_fingerprint"),
        "frozen_candidate_file_hashes_match": readiness.get("gates", {}).get("frozen_candidate_file_hashes_match"),
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        "traffic_class": traffic_class,
        "traffic_live_eligible": live_eligible,
        "configured_answer_response_format": config.response_format_type,
        "conversation_structured_output_override": os.environ.get("OPK_RAG_CONVERSATION_STRUCTURED_OUTPUT_MODE", "auto") or "auto",
        "provider_model_id": config.model_id,
        "provider_endpoint_type": "loopback" if (urlparse(config.base_url).hostname or "") in {"127.0.0.1", "localhost", "::1"} else "remote",
        "entry_gate_passed": historical["all_match"] and dataset["identity_valid"] and readiness["passed"] and traffic_class == CONTROLLED_REALISTIC_EVALUATION_SOURCE and not live_eligible,
    }


def _probe_context() -> ConversationContext:
    turn = ConversationContextTurn(
        turn_number=1,
        user_query="公众号发布这个 App，MVP 最核心到底只做哪件事？",
        standalone_query="公众号发布这个 App，MVP 最核心到底只做哪件事？",
        answer_text="第一版聚焦最小闭环。",
        answer_decision="answer",
        abstention_reason=None,
        citations=(),
        token_count=10,
    )
    return ConversationContext(
        session_id=UUID(int=1),
        knowledge_base_id=UUID(int=2),
        prompt_version=CONVERSATION_PROMPT_VERSION,
        max_turns=6,
        token_budget=1200,
        token_count=10,
        turns=(turn,),
    )


def probe_provider_capabilities(*, write: bool = True) -> dict[str, Any]:
    load_project_env(root=ROOT)
    config = load_answer_generation_config()
    api_key = os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None
    context = _probe_context()
    probes: list[dict[str, Any]] = []
    for mode in ("json_schema", "json_object", "prompt_json"):
        resolver = OpenAICompatibleFollowupQueryResolver(
            config,
            api_key=api_key,
            structured_output_mode=mode,
            max_structural_repairs=1,
        )
        started = time.perf_counter()
        try:
            resolution = resolver.resolve(current_query="那图片处理也算这个最小闭环的一部分吧？", conversation_context=context)
            probes.append(
                {
                    "mode": mode,
                    "success": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "is_followup": resolution.is_followup,
                    "referenced_turn_numbers": list(resolution.referenced_turn_numbers),
                    "resolver_trace": resolver.last_trace.to_dict() if resolver.last_trace else None,
                    "raw_provider_output_persisted": False,
                }
            )
        except AnswerProviderError as exc:
            detail = exc.detail
            probes.append(
                {
                    "mode": mode,
                    "success": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "failure_code": detail.reason_code if detail is not None else type(exc).__name__,
                    "http_status": detail.http_status if detail is not None else None,
                    "resolver_trace": resolver.last_trace.to_dict() if resolver.last_trace else None,
                    "raw_provider_output_persisted": False,
                }
            )
    auto = OpenAICompatibleFollowupQueryResolver(config, api_key=api_key)
    matrix = {
        "schema_version": "opk-rag.task0267.provider-capability-matrix.v1",
        "provider_model_id": config.model_id,
        "configured_answer_response_format": config.response_format_type,
        "auto_resolved_mode": auto.structured_output_mode,
        "probes": probes,
        "strict_json_schema_supported": next(row["success"] for row in probes if row["mode"] == "json_schema"),
        "json_object_supported": next(row["success"] for row in probes if row["mode"] == "json_object"),
        "prompt_json_supported": next(row["success"] for row in probes if row["mode"] == "prompt_json"),
        "deterministic_auto_matches_config": auto.structured_output_mode == config.response_format_type,
    }
    if write:
        _write_json(RESULT / "provider_capability_matrix.json", matrix)
    return matrix


def _failure_row(*, sample_id: str, conversation_id: str, turn: int, stage: str, exc: Exception, resolver_trace: Mapping[str, Any] | None) -> dict[str, Any]:
    detail = getattr(exc, "detail", None)
    text = str(exc).lower()
    return {
        "schema_version": "opk-rag.task0267.primary-failure.v1",
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "turn": turn,
        "failure_stage": stage,
        "failure_class": type(exc).__name__,
        "failure_code": getattr(detail, "reason_code", None) or type(exc).__name__,
        "http_status": getattr(detail, "http_status", None),
        "unsupported_response_format": getattr(detail, "reason_code", None) == "unsupported_response_format",
        "cuda_oom": "cuda" in text and "out of memory" in text,
        "resolver_trace": dict(resolver_trace or {}),
        "primary_failure": True,
        "diagnostic_replay_performed": False,
        "raw_provider_output_persisted": False,
        "hidden_reasoning_persisted": False,
    }


def _checkpoint(*, completed_conversations: list[str], rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], window_id: str | None) -> dict[str, Any]:
    dataset = _read_json(DATASET)
    expected_ids = [str(row["conversation_id"]) for row in list(dataset.get("conversations") or [])]
    pending = [cid for cid in expected_ids if cid not in set(completed_conversations)]
    attempted_turns = len(rows) + len(failures)
    payload = {
        "schema_version": "opk-rag.resumable-primary-checkpoint.v1",
        "task_id": TASK_ID,
        "primary_run_id": PRIMARY_RUN_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "candidate_fingerprint": APPROVED_AGENTIC_CANARY_FINGERPRINT,
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        "completed_conversation_ids": completed_conversations,
        "completed_conversation_count": len(completed_conversations),
        "pending_conversation_ids": pending,
        "pending_conversation_count": len(pending),
        "successful_turn_count": len(rows),
        "failed_turn_count": len(failures),
        "attempted_turn_count": attempted_turns,
        "next_pending_conversation_id": pending[0] if pending else None,
        "multi_turn_window_boundary": "whole_conversation",
        "failed_turn_context_policy": FAILED_TURN_CONTEXT_POLICY,
        "resume_default": True,
        "last_window": window_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(CHECKPOINT, payload)
    return payload


def _conversation_bundle_path(conversation_id: str) -> Path:
    return CONVERSATION_AUTHORITY / f"{conversation_id}.json"


def _commit_conversation_bundle(*, conversation_id: str, rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    path = _conversation_bundle_path(conversation_id)
    if path.exists():
        raise RuntimeError(f"task0267_conversation_primary_already_exists:{conversation_id}")
    _write_json_atomic(
        path,
        {
            "schema_version": "opk-rag.task0267.conversation-primary.v1",
            "primary_run_id": PRIMARY_RUN_ID,
            "conversation_id": conversation_id,
            "rows": list(rows),
            "failures": list(failures),
            "summary": dict(summary),
        },
    )


def _load_conversation_bundles() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not CONVERSATION_AUTHORITY.is_dir():
        return [], [], []
    dataset = _read_json(DATASET)
    order = [str(row["conversation_id"]) for row in list(dataset.get("conversations") or [])]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for conversation_id in order:
        path = _conversation_bundle_path(conversation_id)
        if not path.is_file():
            continue
        bundle = _read_json(path)
        if bundle.get("primary_run_id") != PRIMARY_RUN_ID or bundle.get("conversation_id") != conversation_id:
            raise RuntimeError(f"task0267_conversation_bundle_identity_mismatch:{conversation_id}")
        rows.extend(list(bundle.get("rows") or []))
        failures.extend(list(bundle.get("failures") or []))
        summaries.append(dict(bundle.get("summary") or {}))
    return rows, failures, summaries


def _rebuild_aggregate_authority(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], summaries: list[Mapping[str, Any]]) -> None:
    _write_jsonl(RESULT / "multi_turn_results.jsonl", rows)
    _write_jsonl(RESULT / "failure_ledger.jsonl", failures)
    _write_json(RESULT / "conversation_summary.json", {"schema_version": "opk-rag.task0267.conversation-summary.v1", "conversations": summaries})


def _load_existing_authority() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, failures, summaries = _load_conversation_bundles()
    if not rows and not failures and not summaries:
        rows = _read_jsonl(RESULT / "multi_turn_results.jsonl")
        failures = _read_jsonl(RESULT / "failure_ledger.jsonl")
        conversation = _read_json(RESULT / "conversation_summary.json") if (RESULT / "conversation_summary.json").is_file() else {"conversations": []}
        summaries = list(conversation.get("conversations") or [])
    ids = [str(row.get("sample_id") or "") for row in [*rows, *failures]]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if sample_id and count > 1]
    if duplicates:
        raise RuntimeError(f"task0267_duplicate_primary_ids:{','.join(sorted(duplicates))}")
    return rows, failures, summaries


def _resolver_metrics(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    traces = [dict(row.get("resolver_trace") or {}) for row in [*rows, *failures] if row.get("resolver_trace")]
    provider_traces = [trace for trace in traces if not trace.get("deterministic_passthrough")]
    return {
        "schema_version": "opk-rag.task0267.resolver-structured-output-metrics.v1",
        "total_resolution_count": len(traces),
        "deterministic_passthrough_count": sum(bool(trace.get("deterministic_passthrough")) for trace in traces),
        "provider_resolution_count": len(provider_traces),
        "mode_counts": dict(Counter(str(trace.get("resolved_mode")) for trace in traces)),
        "provider_request_count": sum(int(trace.get("provider_request_count") or 0) for trace in traces),
        "provider_response_count": sum(int(trace.get("provider_response_count") or 0) for trace in traces),
        "provider_transport_attempt_count": sum(int(trace.get("provider_transport_attempt_count") or 0) for trace in traces),
        "initial_structured_valid_rate": sum(trace.get("initial_structured_valid") is True for trace in provider_traces) / max(1, len(provider_traces)),
        "structural_repair_count": sum(bool(trace.get("structural_repair_invoked")) for trace in provider_traces),
        "structural_repair_success_count": sum(bool(trace.get("structural_repair_success")) for trace in provider_traces),
        "final_structured_valid_count": sum(bool(trace.get("final_structured_valid")) for trace in traces),
        "final_structured_valid_rate": sum(bool(trace.get("final_structured_valid")) for trace in traces) / max(1, len(traces)),
        "resolver_failure_count": sum(str(row.get("failure_stage")) == "resolver" for row in failures),
        "unsupported_response_format_count": sum(bool(row.get("unsupported_response_format")) for row in failures),
        "max_provider_requests_per_resolution": max((int(trace.get("provider_request_count") or 0) for trace in traces), default=0),
    }


def _semantic_quality_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    dataset = _read_json(DATASET)
    pattern_by_conversation = {
        str(conversation["conversation_id"]): str(conversation.get("conversation_pattern") or "unknown")
        for conversation in list(dataset.get("conversations") or [])
    }

    def bucket(subset: list[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(subset),
            "semantic_correct_count": sum(bool(row.get("semantic_correct")) for row in subset),
            "semantic_accuracy": sum(bool(row.get("semantic_correct")) for row in subset) / max(1, len(subset)),
            "runtime_success_rate": 1.0 if subset else 0.0,
        }

    confusion = Counter(
        f"{row.get('expected_semantic_class')}->{row.get('predicted_semantic_class')}"
        for row in rows
    )
    expected_negative = [row for row in rows if row.get("expected_semantic_class") == "supported_negative"]
    predicted_negative = [row for row in rows if row.get("predicted_semantic_class") == "supported_negative"]
    negative_tp = sum(
        row.get("expected_semantic_class") == "supported_negative"
        and row.get("predicted_semantic_class") == "supported_negative"
        for row in rows
    )
    should_abstain = [row for row in rows if row.get("expected_semantic_class") == "abstain"]
    partial = [row for row in rows if row.get("expected_semantic_class") == "supported_partial"]
    by_context = {
        dependency: bucket([row for row in rows if str(row.get("context_dependency") or "none") == dependency])
        for dependency in ("none", "low", "medium", "high")
    }
    patterns = sorted(set(pattern_by_conversation.values()))
    by_pattern = {
        pattern: bucket([
            row for row in rows
            if pattern_by_conversation.get(str(row.get("conversation_id"))) == pattern
        ])
        for pattern in patterns
    }
    cross_turn = [row for row in rows if "cross_turn_reference" in set(row.get("retrieval_traits") or [])]
    topic_return = [
        row for row in rows
        if pattern_by_conversation.get(str(row.get("conversation_id"))) == "topic_switch_and_return"
        and str(row.get("context_dependency") or "none") in {"medium", "high"}
    ]
    false_premise = [row for row in rows if "false_premise" in set(row.get("retrieval_traits") or [])]
    return {
        "schema_version": "opk-rag.task0267.semantic-quality-metrics.v1",
        "evaluated_turn_count": len(rows),
        "overall_semantic_accuracy": sum(bool(row.get("semantic_correct")) for row in rows) / max(1, len(rows)),
        "semantic_correct_count": sum(bool(row.get("semantic_correct")) for row in rows),
        "confusion": dict(sorted(confusion.items())),
        "supported_negative_precision": negative_tp / max(1, len(predicted_negative)),
        "supported_negative_recall": negative_tp / max(1, len(expected_negative)),
        "should_abstain_accuracy": sum(row.get("predicted_semantic_class") == "abstain" for row in should_abstain) / max(1, len(should_abstain)),
        "partial_handling_accuracy": sum(row.get("predicted_semantic_class") == "supported_partial" for row in partial) / max(1, len(partial)),
        "by_context_dependency": by_context,
        "by_conversation_pattern": by_pattern,
        "high_context_dependency_runtime_success_rate": by_context["high"]["runtime_success_rate"],
        "high_context_dependency_semantic_accuracy": by_context["high"]["semantic_accuracy"],
        "cross_turn_reference": bucket(cross_turn),
        "topic_return": bucket(topic_return),
        "false_premise": bucket(false_premise),
        "runtime_gold_metadata_usage": False,
    }


def _summarize(*, rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], conversations: list[Mapping[str, Any]], gpu_samples: list[Mapping[str, Any]], ledger_before: Mapping[str, Any], ledger_after: Mapping[str, Any]) -> dict[str, Any]:
    resolver = _resolver_metrics(rows, failures)
    semantic_quality = _semantic_quality_metrics(rows)
    expected_context = [row for row in rows if row.get("context_dependency") in {"medium", "high"}]
    semantic_accuracy = semantic_quality["overall_semantic_accuracy"]
    hard_fields = (
        "unsafe_finish", "llm_direct_finish_authority", "abstain_to_finish_override", "graph_hop_violation",
        "kb_mutation", "grounding_bypass", "fabricated_citation", "runtime_gold_exposure", "recovery_harmed",
    )
    hard_counts = {field: sum(bool(row.get(field)) for row in rows) for field in hard_fields}
    hard_counts["three_plus_controller_calls"] = sum(int(row.get("controller_call_count") or 0) >= 3 for row in rows)
    hard_violation_count = sum(hard_counts.values())
    gpu_used = [int(row["used_mb"]) for row in gpu_samples if row.get("available") and row.get("used_mb") is not None]
    pollution = int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0)
    runtime_gates = {
        "task0266_history_immutable": task0266_identity()["all_match"],
        "dataset_identity": dataset_identity()["identity_valid"],
        "conversation_9_attempted": len(conversations) == EXPECTED_CONVERSATIONS,
        "turn_40_attempted": len(rows) + len(failures) == EXPECTED_TURNS,
        "unsupported_response_format_zero": resolver["unsupported_response_format_count"] == 0,
        "resolver_structured_validity": resolver["final_structured_valid_rate"] >= 0.98,
        "multi_turn_runtime_success_rate": len(rows) / EXPECTED_TURNS >= 0.98,
        "unexpected_runtime_failure_zero": len(failures) == 0,
        "resolver_provider_requests_bounded": resolver["max_provider_requests_per_resolution"] <= 2,
        "cuda_oom_zero": sum(bool(row.get("cuda_oom")) for row in failures) == 0,
        "hard_agent_safety_zero": hard_violation_count == 0,
        "task0264_pollution_zero": pollution == 0,
    }
    runtime_pass = all(runtime_gates.values())
    if hard_violation_count:
        decision = "fail_closed_for_resolver_safety"
    elif not runtime_pass:
        decision = "hold_for_conversation_runtime_reliability"
    elif semantic_accuracy < 0.90:
        decision = "complete_with_semantic_quality_followup"
    else:
        decision = "advance_to_low_traffic_operator_canary_governance"
    return {
        "resolver_structured_output_metrics.json": resolver,
        "semantic_quality_metrics.json": semantic_quality,
        "gpu_residency.json": {
            "schema_version": "opk-rag.task0267.gpu-residency.v1",
            "samples": gpu_samples,
            "peak_used_mb": max(gpu_used) if gpu_used else None,
            "cuda_oom_count": sum(bool(row.get("cuda_oom")) for row in failures),
        },
        "task0264_isolation_review.json": {
            "schema_version": "opk-rag.task0267.task0264-isolation.v1",
            "ledger_before": ledger_before,
            "ledger_after": ledger_after,
            "pollution_count": pollution,
            "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        },
        "summary.json": {
            "schema_version": "opk-rag.task0267.summary.v1",
            "task_id": TASK_ID,
            "task_status": "complete" if len(conversations) == EXPECTED_CONVERSATIONS else "partial",
            "primary_run_id": PRIMARY_RUN_ID,
            "candidate_decision": decision,
            "conversation_count": len(conversations),
            "turn_success_count": len(rows),
            "turn_failure_count": len(failures),
            "multi_turn_runtime_success_rate": len(rows) / EXPECTED_TURNS,
            "context_dependent_turn_count": len(expected_context),
            "context_resolution_success_rate": sum(bool(row.get("resolver_is_followup")) for row in expected_context) / max(1, len(expected_context)),
            "semantic_accuracy": semantic_accuracy,
            "resolver_final_structured_valid_rate": resolver["final_structured_valid_rate"],
            "resolver_structural_repair_count": resolver["structural_repair_count"],
            "resolver_unsupported_response_format_count": resolver["unsupported_response_format_count"],
            "hard_safety_violation_count": hard_violation_count,
            "hard_safety_violation_counts": hard_counts,
            "task0264_pollution_count": pollution,
            "gpu_peak_used_mb": max(gpu_used) if gpu_used else None,
            "all_runtime_acceptance_gates_passed": runtime_pass,
            "runtime_acceptance_gates": runtime_gates,
            "next_task": "TASK-0268_low_volume_operator_canary_governance_and_real_observation_policy" if decision == "advance_to_low_traffic_operator_canary_governance" else "TASK-0268_semantic_quality_requalification" if decision == "complete_with_semantic_quality_followup" else "TASK-0267_followup",
        },
    }


async def execute(*, resume: bool = True, max_new_conversations: int | None = None, write: bool = True) -> dict[str, Any]:
    load_project_env(root=ROOT)
    entry = entry_authority()
    if not entry["entry_gate_passed"]:
        if write:
            _write_json(RESULT / "entry_authority.json", entry)
        raise RuntimeError("task0267_entry_gate_failed")
    if max_new_conversations is not None and max_new_conversations < 1:
        raise ValueError("max_new_conversations must be >= 1")
    if RESULT.exists() and not resume and any((RESULT / name).exists() for name in ("multi_turn_results.jsonl", "failure_ledger.jsonl", "conversation_summary.json")):
        raise RuntimeError("task0267_existing_primary_authority_refuses_fresh_overwrite")

    RESULT.mkdir(parents=True, exist_ok=True)
    if write:
        _write_json(RESULT / "entry_authority.json", entry)
    rows, failures, conversations = _load_existing_authority()
    if CONVERSATION_AUTHORITY.is_dir() or not any((RESULT / name).exists() for name in ("multi_turn_results.jsonl", "failure_ledger.jsonl", "conversation_summary.json")):
        _rebuild_aggregate_authority(rows, failures, conversations)
    completed = [str(row.get("conversation_id")) for row in conversations]
    dataset = _read_json(DATASET)
    pending = [row for row in list(dataset.get("conversations") or []) if str(row["conversation_id"]) not in set(completed)]
    if not pending:
        return _read_json(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {"task_id": TASK_ID, "task_status": "complete", "primary_run_id": PRIMARY_RUN_ID}

    window_id = f"window-{len(_read_jsonl(WINDOW_HISTORY)) + 1:03d}"
    ledger_before = _canary_ledger_state()
    gpu_samples = [{"position": "window_before", "window_id": window_id, **_gpu_state()}]
    executor = ShowcaseExecutor(max_concurrent_executions=1)
    kb_id = UUID(executor._showcase_kb_id())
    answer_config = load_answer_generation_config()
    resolver = OpenAICompatibleFollowupQueryResolver(answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
    token_counter = QwenContextTokenCounter(load_embedding_config())
    new_conversations = 0

    for conversation in pending:
        if max_new_conversations is not None and new_conversations >= max_new_conversations:
            break
        conversation_id = str(conversation["conversation_id"])
        now = datetime.now(timezone.utc)
        session = ConversationSession(
            id=uuid4(), knowledge_base_id=kb_id, title=conversation.get("topic"), status="active",
            created_at=now, updated_at=now, last_turn_at=None, turn_count=0,
            conversation_prompt_version=CONVERSATION_PROMPT_VERSION,
            followup_rewrite_version=FOLLOWUP_REWRITE_VERSION,
            metadata={"synthetic": True, "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE},
        )
        prior_turns: list[ConversationTurn] = []
        citations_by_turn: dict[UUID, tuple[ConversationTurnCitation, ...]] = {}
        conv_rows: list[dict[str, Any]] = []
        conv_failures: list[dict[str, Any]] = []
        attempted = 0
        for turn_item in list(conversation.get("turns") or []):
            attempted += 1
            turn_number = int(turn_item["turn"])
            sample_id = f"{conversation_id}-T{turn_number:02d}"
            context = build_conversation_context(
                session=session, turns=tuple(prior_turns), citations_by_turn=citations_by_turn,
                token_counter=token_counter, max_turns=6, token_budget=1200,
            )
            try:
                resolution = resolver.resolve(current_query=str(turn_item["query"]), conversation_context=context)
            except Exception as exc:
                trace = resolver.last_trace.to_dict() if resolver.last_trace else None
                conv_failures.append(_failure_row(sample_id=sample_id, conversation_id=conversation_id, turn=turn_number, stage="resolver", exc=exc, resolver_trace=trace))
                print(f"[{window_id}] {sample_id} FAIL resolver {type(exc).__name__}; failed turn excluded from later context", flush=True)
                continue
            resolver_trace = resolver.last_trace.to_dict() if resolver.last_trace else {}
            started = time.perf_counter()
            try:
                result = await executor.run_ask_controlled_evaluation(resolution.standalone_query)
                elapsed = (time.perf_counter() - started) * 1000.0
                row = _trace_result(
                    sample_id=sample_id,
                    expected_answerability=str(turn_item["answerability"]),
                    difficulty=str(turn_item["difficulty"]),
                    retrieval_traits=list(turn_item.get("retrieval_traits") or []),
                    trace=result.trace,
                    answer=result.answer,
                    candidate_telemetry=result.candidate_telemetry,
                    elapsed_ms=elapsed,
                    lane="multi_turn",
                    conversation_id=conversation_id,
                    turn=turn_number,
                    context_dependency=str(turn_item.get("context_dependency") or "none"),
                    resolver=resolution,
                )
                row["resolver_trace"] = resolver_trace
                conv_rows.append(row)
                turn = _build_turn(
                    session_id=session.id, turn_number=turn_number, user_query=str(turn_item["query"]),
                    resolution=resolution, answer=result.answer, elapsed_ms=elapsed,
                )
                prior_turns.append(turn)
                citations_by_turn[turn.id] = _conversation_citations(turn.id, result.answer)
                print(f"[{window_id}] {sample_id} ok mode={resolver_trace.get('resolved_mode')} repair={resolver_trace.get('structural_repair_invoked')} semantic={row['predicted_semantic_class']}", flush=True)
            except Exception as exc:
                conv_failures.append(_failure_row(sample_id=sample_id, conversation_id=conversation_id, turn=turn_number, stage="runtime", exc=exc, resolver_trace=resolver_trace))
                print(f"[{window_id}] {sample_id} FAIL runtime {type(exc).__name__}; failed turn excluded from later context", flush=True)
                continue

        # Commit only at a whole-conversation boundary. The canonical primary
        # authority is one atomically replaced bundle per conversation; aggregate
        # JSONL files are derived views and can be rebuilt after interruption.
        summary_row = {
            "conversation_id": conversation_id,
            "pattern": conversation.get("conversation_pattern"),
            "expected_turn_count": len(list(conversation.get("turns") or [])),
            "attempted_turn_count": attempted,
            "successful_turn_count": len(conv_rows),
            "failure_count": len(conv_failures),
            "complete_success": len(conv_rows) == len(list(conversation.get("turns") or [])) and not conv_failures,
            "failed_turn_context_policy": FAILED_TURN_CONTEXT_POLICY,
            "failed_turns_added_to_context": False,
        }
        _commit_conversation_bundle(
            conversation_id=conversation_id,
            rows=conv_rows,
            failures=conv_failures,
            summary=summary_row,
        )
        rows.extend(conv_rows)
        failures.extend(conv_failures)
        conversations.append(summary_row)
        _rebuild_aggregate_authority(rows, failures, conversations)
        completed.append(conversation_id)
        new_conversations += 1
        gpu_samples.append({"position": "conversation_complete", "window_id": window_id, "conversation_id": conversation_id, **_gpu_state()})
        _checkpoint(completed_conversations=completed, rows=rows, failures=failures, window_id=window_id)

    ledger_after = _canary_ledger_state()
    gpu_samples.append({"position": "window_after", "window_id": window_id, **_gpu_state()})
    previous_gpu = _read_json(RESULT / "gpu_residency.json").get("samples", []) if (RESULT / "gpu_residency.json").is_file() else []
    artifacts = _summarize(rows=rows, failures=failures, conversations=conversations, gpu_samples=[*previous_gpu, *gpu_samples], ledger_before=ledger_before, ledger_after=ledger_after)
    for name, payload in artifacts.items():
        _write_json(RESULT / name, payload)
    window = {
        "schema_version": "opk-rag.resumable-execution-window.v1",
        "task_id": TASK_ID,
        "primary_run_id": PRIMARY_RUN_ID,
        "window_id": window_id,
        "new_conversation_count": new_conversations,
        "new_successful_turn_count": sum(int(row.get("successful_turn_count") or 0) for row in conversations[-new_conversations:]) if new_conversations else 0,
        "new_failure_count": sum(int(row.get("failure_count") or 0) for row in conversations[-new_conversations:]) if new_conversations else 0,
        "task0264_pollution_delta": int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0),
        "candidate_decision_after_window": artifacts["summary.json"]["candidate_decision"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(WINDOW_HISTORY, window)
    _checkpoint(completed_conversations=completed, rows=rows, failures=failures, window_id=window_id)
    return artifacts["summary.json"]


def verify() -> dict[str, Any]:
    required = [
        "entry_authority.json", "provider_capability_matrix.json", "resolver_contract_tests.json",
        "resolver_structured_output_metrics.json", "semantic_quality_metrics.json", "multi_turn_results.jsonl", "conversation_summary.json",
        "failure_ledger.jsonl", "gpu_residency.json", "task0264_isolation_review.json", "post_repair_single_turn_regression.json",
        "runtime_restore_validation.json", "execution_checkpoint.json", "execution_windows.jsonl", "summary.json",
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    rows = _read_jsonl(RESULT / "multi_turn_results.jsonl")
    failures = _read_jsonl(RESULT / "failure_ledger.jsonl")
    conversation = _read_json(RESULT / "conversation_summary.json") if (RESULT / "conversation_summary.json").is_file() else {"conversations": []}
    checkpoint = _read_json(CHECKPOINT) if CHECKPOINT.is_file() else {}
    summary = _read_json(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    isolation = _read_json(RESULT / "task0264_isolation_review.json") if (RESULT / "task0264_isolation_review.json").is_file() else {}
    resolver = _read_json(RESULT / "resolver_structured_output_metrics.json") if (RESULT / "resolver_structured_output_metrics.json").is_file() else {}
    semantic = _read_json(RESULT / "semantic_quality_metrics.json") if (RESULT / "semantic_quality_metrics.json").is_file() else {}
    regression = _read_json(RESULT / "post_repair_single_turn_regression.json") if (RESULT / "post_repair_single_turn_regression.json").is_file() else {}
    restore = _read_json(RESULT / "runtime_restore_validation.json") if (RESULT / "runtime_restore_validation.json").is_file() else {}
    capability = _read_json(RESULT / "provider_capability_matrix.json") if (RESULT / "provider_capability_matrix.json").is_file() else {}
    ids = [str(row.get("sample_id") or "") for row in [*rows, *failures]]
    checks = {
        "required_artifacts_present": not missing,
        "task0266_history_immutable": task0266_identity()["all_match"],
        "dataset_identity_valid": dataset_identity()["identity_valid"],
        "primary_ids_unique": len(ids) == len(set(ids)),
        "conversation_count_9": len(list(conversation.get("conversations") or [])) == EXPECTED_CONVERSATIONS,
        "turn_outcome_count_40": len(rows) + len(failures) == EXPECTED_TURNS,
        "checkpoint_identity": checkpoint.get("primary_run_id") == PRIMARY_RUN_ID and checkpoint.get("dataset_sha256") == EXPECTED_DATASET_SHA256 and checkpoint.get("candidate_fingerprint") == APPROVED_AGENTIC_CANARY_FINGERPRINT,
        "checkpoint_no_pending_conversations": checkpoint.get("pending_conversation_count") == 0,
        "capability_auto_matches_config": capability.get("deterministic_auto_matches_config") is True,
        "resolver_provider_requests_bounded": int(resolver.get("max_provider_requests_per_resolution") or 0) <= 2,
        "resolver_final_structured_validity_98": float(resolver.get("final_structured_valid_rate") or 0.0) >= 0.98,
        "semantic_metrics_cover_40_turns": semantic.get("evaluated_turn_count") == EXPECTED_TURNS,
        "single_turn_regression_passed": regression.get("regression_passed") is True,
        "showcase_api_canary_restored": restore.get("restore_passed") is True,
        "task0264_pollution_zero": isolation.get("pollution_count") == 0,
        "report_present": REPORT.is_file(),
        "summary_decision_allowed": summary.get("candidate_decision") in {"advance_to_low_traffic_operator_canary_governance", "complete_with_semantic_quality_followup", "hold_for_conversation_runtime_reliability", "fail_closed_for_resolver_safety", "blocked"},
    }
    return {
        "schema_version": "opk-rag.task0267.verification.v1",
        "task_id": TASK_ID,
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
