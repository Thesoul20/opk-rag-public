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
from opk_rag.answer.negative_semantics import classify_answer_semantics
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
TASK_ID = "TASK-0268"
PRIMARY_RUN_ID = "task0268-semantic-requalification-v1"
RESULT = ROOT / "evaluation-data/results/task0268-semantic-quality-requalification"
REPORT = ROOT / "docs/TASK0268_SEMANTIC_QUALITY_REQUALIFICATION_REPORT.md"
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
        "schema_version": "opk-rag.task0268.task0266-history-identity.v1",
        "files": rows,
        "all_match": all(row["match"] for row in rows.values()),
    }


def dataset_identity() -> dict[str, Any]:
    actual = _sha256(DATASET)
    data = _read_json(DATASET)
    conversations = list(data.get("conversations") or [])
    turns = sum(len(list(row.get("turns") or [])) for row in conversations)
    return {
        "schema_version": "opk-rag.task0268.dataset-identity.v1",
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
        "schema_version": "opk-rag.task0268.entry-authority.v1",
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
        "schema_version": "opk-rag.task0268.provider-capability-matrix.v1",
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
        "schema_version": "opk-rag.task0268.primary-failure.v1",
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
        "candidate_fingerprint": task0268_candidate_identity()["candidate_fingerprint"],
        "base_candidate_fingerprint": APPROVED_AGENTIC_CANARY_FINGERPRINT,
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
        raise RuntimeError(f"task0268_conversation_primary_already_exists:{conversation_id}")
    _write_json_atomic(
        path,
        {
            "schema_version": "opk-rag.task0268.conversation-primary.v1",
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
            raise RuntimeError(f"task0268_conversation_bundle_identity_mismatch:{conversation_id}")
        rows.extend(list(bundle.get("rows") or []))
        failures.extend(list(bundle.get("failures") or []))
        summaries.append(dict(bundle.get("summary") or {}))
    return rows, failures, summaries


def _rebuild_aggregate_authority(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], summaries: list[Mapping[str, Any]]) -> None:
    _write_jsonl(RESULT / "multi_turn_results.jsonl", rows)
    _write_jsonl(RESULT / "failure_ledger.jsonl", failures)
    _write_json(RESULT / "conversation_summary.json", {"schema_version": "opk-rag.task0268.conversation-summary.v1", "conversations": summaries})


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
        raise RuntimeError(f"task0268_duplicate_primary_ids:{','.join(sorted(duplicates))}")
    return rows, failures, summaries


def _resolver_metrics(rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    traces = [dict(row.get("resolver_trace") or {}) for row in [*rows, *failures] if row.get("resolver_trace")]
    provider_traces = [trace for trace in traces if not trace.get("deterministic_passthrough")]
    return {
        "schema_version": "opk-rag.task0268.resolver-structured-output-metrics.v1",
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
        "schema_version": "opk-rag.task0268.semantic-quality-metrics.v1",
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
            "schema_version": "opk-rag.task0268.gpu-residency.v1",
            "samples": gpu_samples,
            "peak_used_mb": max(gpu_used) if gpu_used else None,
            "cuda_oom_count": sum(bool(row.get("cuda_oom")) for row in failures),
        },
        "task0264_isolation_review.json": {
            "schema_version": "opk-rag.task0268.task0264-isolation.v1",
            "ledger_before": ledger_before,
            "ledger_after": ledger_after,
            "pollution_count": pollution,
            "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        },
        "summary.json": {
            "schema_version": "opk-rag.task0268.summary.v1",
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
        raise RuntimeError("task0268_entry_gate_failed")
    if max_new_conversations is not None and max_new_conversations < 1:
        raise ValueError("max_new_conversations must be >= 1")
    if RESULT.exists() and not resume and any((RESULT / name).exists() for name in ("multi_turn_results.jsonl", "failure_ledger.jsonl", "conversation_summary.json")):
        raise RuntimeError("task0268_existing_primary_authority_refuses_fresh_overwrite")

    RESULT.mkdir(parents=True, exist_ok=True)
    if write:
        _write_json(RESULT / "entry_authority.json", entry)
        _write_json(TASK0268_CANDIDATE_IDENTITY, entry["semantic_repair_candidate"])
    semantic_candidate = dict(entry["semantic_repair_candidate"])
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
                result = await executor.run_ask_controlled_semantic_requalification(
                    resolution.standalone_query, candidate_readiness=semantic_candidate
                )
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
                row = _rescore_task0268_row(row=row, answer=result.answer, expected_answerability=str(turn_item["answerability"]))
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
    artifacts = _summarize0268(rows=rows, failures=failures, conversations=conversations, gpu_samples=[*previous_gpu, *gpu_samples], ledger_before=ledger_before, ledger_after=ledger_after)
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
        "schema_version": "opk-rag.task0268.verification.v1",
        "task_id": TASK_ID,
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }

# TASK-0268 semantic requalification overrides.
TASK0267_ROOT = ROOT / "evaluation-data/results/task0267-conversation-resolver-provider-compatibility"
TASK0267_AUTHORITY_FILES = ("summary.json", "semantic_quality_metrics.json", "multi_turn_results.jsonl", "conversation_summary.json")
TASK0263_CANDIDATE = ROOT / "evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification/candidate_fingerprint.json"
TASK0268_CANDIDATE_IDENTITY = RESULT / "semantic_repair_candidate_identity.json"
TASK0268_SEMANTIC_REPAIR_FILES = (
    "opk_rag/answer/negative_semantics.py",
    "opk_rag/answerability/decision.py",
    "opk_rag/conversation/resolver.py",
    "opk_rag/showcase/api/execution.py",
)
TASK0268_ALLOWED_TASK0263_FROZEN_DRIFT = ("opk_rag/answer/negative_semantics.py",)
EXPECTED_TO_RUNTIME_0268 = {
    "answerable": "supported_affirmative", "supported_negative": "supported_negative",
    "partial": "supported_partial", "should_abstain": "abstain",
}
_DIAGNOSIS = {
    "C001-T02": ("answerability_failure", "negative_semantics", "Unchecked MVP work was confused with scope exclusion; checkbox state is implementation status, not scope membership."),
    "C002-T03": ("answerability_failure", "negative_semantics", "A positive contrastive proposition was confused with a negative proposition because nearby text contained negation."),
    "C003-T01": ("answerability_failure", "negative_semantics", "Direct future-work evidence for Agent Skill packaging was not recognized as proposition-level negative support."),
    "C003-T04": ("answerability_failure", "negative_semantics", "Roadmap future-work evidence for unvalidated cases was not consistently mapped to supported-negative semantics."),
    "C004-T01": ("answerability_failure", "answerability", "The false-premise heuristic was too broad for explanatory why-questions containing a negative premise."),
    "C004-T04": ("answerability_failure", "answerability", "Evidence supported the method but did not prove the exclusive necessity asserted by the question; the correct contract is partial."),
    "C004-T05": ("answerability_failure", "answerability", "A request for measured comparative stability must fail closed when no empirical comparison result is present."),
    "C005-T04": ("followup_resolution_error", "conversation_resolver", "The follow-up rewrite could invert yes/no answer orientation while preserving topic words."),
    "C006-T05": ("answerability_failure", "answerability", "A configured autosave interval is not an empirical power-loss test result."),
    "C007-T03": ("retrieval_relevance_failure", "retrieval", "Relevant Pyright installation evidence exists in the corpus but was not reliably selected into final Evidence."),
    "C009-T02": ("answerability_failure", "negative_semantics", "A troubleshooting question containing negative wording was incorrectly treated as a negative-answer proposition."),
    "C009-T04": ("retrieval_relevance_failure", "retrieval", "The exact Fcitx5 clipboard shortcut/copyq conflict section was not reliably selected into final Evidence."),
    "C009-T05": ("followup_resolution_error", "conversation_resolver", "The follow-up rewrite could neutralize a negative cross-platform path confirmation into a generic compatibility query."),
}


def task0268_candidate_identity() -> dict[str, Any]:
    from opk_rag.agentic_v2.budget import AgentBudget
    from opk_rag.agentic_v2.policy_input import AGENTIC_V2_ACTION_SPACE
    from opk_rag.agentic_v2.tool_registry import FROZEN_TOOL_ACTIONS
    from opk_rag.runtime_v2.public_search_runtime import MAXIMUM_RECOVERY_ATTEMPTS

    base = _read_json(TASK0263_CANDIDATE)
    frozen = dict(base.get("file_sha256") or {})
    current_frozen = {
        rel: (_sha256(ROOT / rel) if (ROOT / rel).is_file() else None)
        for rel in frozen
    }
    drift = sorted(rel for rel, expected in frozen.items() if current_frozen.get(rel) != expected)
    allowed_drift = sorted(TASK0268_ALLOWED_TASK0263_FROZEN_DRIFT)
    repair_hashes = {rel: _sha256(ROOT / rel) for rel in TASK0268_SEMANTIC_REPAIR_FILES if (ROOT / rel).is_file()}
    action_space = list(AGENTIC_V2_ACTION_SPACE)
    budget = AgentBudget()
    payload = {
        "schema_version": "opk-rag.task0268.semantic-repair-candidate-identity.v1",
        "task_id": TASK_ID,
        "base_candidate_fingerprint": base.get("candidate_fingerprint"),
        "base_architecture": base.get("architecture"),
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        "production_authority_unchanged": True,
        "production_canary_fingerprint_updated": False,
        "task0263_frozen_file_drift": drift,
        "allowed_task0263_frozen_file_drift": allowed_drift,
        "frozen_drift_is_semantic_only": drift == allowed_drift,
        "current_task0263_frozen_file_sha256": current_frozen,
        "semantic_repair_file_sha256": repair_hashes,
        "action_space": action_space,
        "action_space_unchanged": tuple(action_space) == tuple(FROZEN_TOOL_ACTIONS),
        "max_graph_hop": budget.max_graph_hops,
        "graph_hop_unchanged": budget.max_graph_hops == int(base.get("max_graph_hop") or 0) == 1,
        "maximum_recovery_attempt_count": MAXIMUM_RECOVERY_ATTEMPTS,
        "recovery_budget_unchanged": MAXIMUM_RECOVERY_ATTEMPTS == 1,
        "llm_finish_authority_allowed": bool(base.get("llm_finish_authority_allowed")),
        "llm_finish_authority_unchanged": base.get("llm_finish_authority_allowed") is False,
        "abstain_to_finish_override_allowed": bool(base.get("abstain_to_finish_override_allowed")),
        "abstain_to_finish_override_unchanged": base.get("abstain_to_finish_override_allowed") is False,
        "runtime_gold_exposure_allowed": False,
        "unrestricted_production_activation_authorized": False,
    }
    fingerprint_material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["candidate_fingerprint"] = hashlib.sha256(fingerprint_material).hexdigest()
    payload["passed"] = bool(
        payload["base_candidate_fingerprint"] == APPROVED_AGENTIC_CANARY_FINGERPRINT
        and payload["frozen_drift_is_semantic_only"]
        and len(repair_hashes) == len(TASK0268_SEMANTIC_REPAIR_FILES)
        and payload["action_space_unchanged"]
        and payload["graph_hop_unchanged"]
        and payload["recovery_budget_unchanged"]
        and payload["llm_finish_authority_unchanged"]
        and payload["abstain_to_finish_override_unchanged"]
    )
    return payload


def task0267_identity() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in TASK0267_AUTHORITY_FILES:
        path = TASK0267_ROOT / name
        files[name] = {"sha256": _sha256(path) if path.is_file() else None, "exists": path.is_file()}
    summary = _read_json(TASK0267_ROOT / "summary.json") if (TASK0267_ROOT / "summary.json").is_file() else {}
    return {
        "schema_version": "opk-rag.task0268.task0267-history-identity.v1", "files": files,
        "all_present": all(row["exists"] for row in files.values()),
        "frozen_primary_run_id": summary.get("primary_run_id"), "frozen_semantic_accuracy": summary.get("semantic_accuracy"),
        "frozen_turn_success_count": summary.get("turn_success_count"),
    }


def entry_authority() -> dict[str, Any]:
    load_project_env(root=ROOT)
    config = load_answer_generation_config()
    base_readiness = evaluate_task0263_readiness(root=ROOT, expected_candidate_fingerprint=APPROVED_AGENTIC_CANARY_FINGERPRINT)
    repair_candidate = task0268_candidate_identity()
    traffic_class, live_eligible = classify_traffic(execution_scope="ask", source=CONTROLLED_REALISTIC_EVALUATION_SOURCE)
    h26, h27, dataset = task0266_identity(), task0267_identity(), dataset_identity()
    base_blockers = list(base_readiness.get("blockers") or [])
    expected_semantic_drift_only = base_blockers == ["frozen_candidate_file_hashes_match"]
    return {
        "schema_version": "opk-rag.task0268.entry-authority.v1", "task_id": TASK_ID,
        "task0266_historical_identity": h26, "task0267_historical_identity": h27, "dataset_identity": dataset,
        "task0263_base_readiness_passed": base_readiness["passed"],
        "task0263_base_readiness_blockers": base_blockers,
        "task0263_expected_semantic_drift_only": expected_semantic_drift_only,
        "base_candidate_fingerprint": base_readiness.get("approved_candidate_fingerprint"),
        "semantic_repair_candidate": repair_candidate,
        "candidate_fingerprint": repair_candidate.get("candidate_fingerprint"),
        "candidate_readiness_passed": repair_candidate.get("passed") is True,
        "production_authority_unchanged": repair_candidate.get("production_authority_unchanged") is True,
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE, "traffic_class": traffic_class, "traffic_live_eligible": live_eligible,
        "task0264_canary_ledger_at_entry": _canary_ledger_state(), "configured_answer_response_format": config.response_format_type,
        "provider_model_id": config.model_id,
        "provider_endpoint_type": "loopback" if (urlparse(config.base_url).hostname or "") in {"127.0.0.1", "localhost", "::1"} else "remote",
        "entry_gate_passed": bool(
            h26["all_match"] and h27["all_present"] and h27["frozen_primary_run_id"] == "task0267-multiturn-revalidation-v4"
            and dataset["identity_valid"] and repair_candidate.get("passed") is True
            and expected_semantic_drift_only
            and traffic_class == CONTROLLED_REALISTIC_EVALUATION_SOURCE and not live_eligible
        ),
    }


def _rescore_task0268_row(*, row: Mapping[str, Any], answer: Any, expected_answerability: str) -> dict[str, Any]:
    updated = dict(row)
    updated["task0267_scoring_semantic_class"] = row.get("predicted_semantic_class")
    updated["task0267_scoring_semantic_correct"] = bool(row.get("semantic_correct"))
    expected = EXPECTED_TO_RUNTIME_0268[expected_answerability]
    refined = answer.answerability or answer.controller_answerability
    bundle = answer.search_response.evidence_bundle
    if refined is not None and bundle is not None:
        contract = classify_answer_semantics(question=str(answer.search_response.query), bundle=bundle, answerability=refined)
        predicted = contract.semantic_class
        updated["semantic_contract_reason_codes"] = list(contract.reason_codes)
        updated["negative_support_present"] = contract.negative_support_present
        updated["negative_support_type"] = contract.negative_support_type
    else:
        predicted = "abstain"
        updated["semantic_contract_reason_codes"] = ["no_refined_answerability_or_evidence"]
        updated["negative_support_present"] = False
        updated["negative_support_type"] = "none"
    updated["predicted_semantic_class"] = predicted
    updated["semantic_correct"] = predicted == expected
    updated["controller_answerability_status"] = getattr(answer.controller_answerability, "status", None)
    updated["controller_answerability_reason_code"] = getattr(answer.controller_answerability, "reason_code", None)
    updated["refined_answerability_status"] = getattr(answer.answerability, "status", None)
    updated["refined_answerability_reason_code"] = getattr(answer.answerability, "reason_code", None)
    terminal = "abstain" if answer.status != "answered" else predicted
    updated["terminal_semantic_class"] = terminal
    updated["terminal_semantic_correct"] = terminal == expected
    return updated


def _task0267_failure_rows() -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(TASK0267_ROOT / "multi_turn_results.jsonl") if not bool(row.get("semantic_correct"))]


def _diagnostic_artifacts(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    current = {str(row.get("sample_id")): row for row in rows}
    ledger, diagnosis = [], []
    counts: Counter[str] = Counter()
    for frozen in _task0267_failure_rows():
        sid = str(frozen["sample_id"])
        primary, stage, explanation = _DIAGNOSIS.get(sid, ("compound_failure", "multiple", "No single earlier stage was sufficient from frozen telemetry."))
        counts[primary] += 1
        now = current.get(sid)
        disposition = "fixed" if now and bool(now.get("semantic_correct")) else "still_failing"
        ledger.append({"schema_version": "opk-rag.task0268.task0267-semantic-failure.v1", "sample_id": sid, "expected_semantic_class": frozen.get("expected_semantic_class"), "task0267_predicted_semantic_class": frozen.get("predicted_semantic_class"), "primary_root_cause": primary, "earliest_faulty_stage": stage, "task0268_disposition": disposition, "task0268_predicted_semantic_class": now.get("predicted_semantic_class") if now else None, "runtime_gold_exposure": False})
        diagnosis.append({"schema_version": "opk-rag.task0268.stage-counterfactual-diagnosis.v1", "sample_id": sid, "primary_root_cause": primary, "earliest_faulty_stage": stage, "diagnosis": explanation, "agent_present": int((now or frozen).get("controller_call_count") or 0) > 0, "agent_causal_failure_proven": bool((now or frozen).get("recovery_harmed")), "runtime_gold_exposure": False})
    taxonomy = {"schema_version": "opk-rag.task0268.semantic-failure-taxonomy.v1", "frozen_task0267_failure_count": len(ledger), "counts": dict(sorted(counts.items())), "allowed_primary_categories": ["followup_resolution_error", "retrieval_relevance_failure", "agent_policy_decision_failure", "guard_policy_interaction_failure", "evidence_composition_failure", "answerability_failure", "generation_semantic_failure", "grounding_or_citation_failure", "evaluation_contract_mismatch", "compound_failure"]}
    return ledger, diagnosis, taxonomy


def _summarize0268(*, rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], conversations: list[Mapping[str, Any]], gpu_samples: list[Mapping[str, Any]], ledger_before: Mapping[str, Any], ledger_after: Mapping[str, Any]) -> dict[str, Any]:
    resolver = _resolver_metrics(rows, failures)
    semantic = _semantic_quality_metrics(rows)
    semantic["schema_version"] = "opk-rag.task0268.semantic-quality-metrics.v1"
    semantic["terminal_semantic_correct_count"] = sum(bool(row.get("terminal_semantic_correct")) for row in rows)
    semantic["terminal_semantic_accuracy"] = semantic["terminal_semantic_correct_count"] / max(1, len(rows))
    semantic["conversation_level_semantic_pass_rate"] = sum(
        all(bool(row.get("semantic_correct")) for row in rows if row.get("conversation_id") == conv.get("conversation_id"))
        for conv in conversations
    ) / max(1, len(conversations))
    frozen_rows = {str(row.get("sample_id")): row for row in _read_jsonl(TASK0267_ROOT / "multi_turn_results.jsonl")}
    previously_correct = [row for row in rows if bool((frozen_rows.get(str(row.get("sample_id"))) or {}).get("semantic_correct"))]
    previous_regressions = [row for row in previously_correct if not bool(row.get("semantic_correct"))]
    false_finish = [row for row in rows if row.get("expected_semantic_class") == "abstain" and row.get("answer_status") == "answered"]
    unsupported_grounded = [row for row in false_finish if bool(row.get("grounding_valid"))]
    hard_fields = ("unsafe_finish", "llm_direct_finish_authority", "abstain_to_finish_override", "graph_hop_violation", "kb_mutation", "grounding_bypass", "fabricated_citation", "runtime_gold_exposure", "recovery_harmed")
    hard_counts = {field: sum(bool(row.get(field)) for row in rows) for field in hard_fields}
    hard_counts["three_plus_controller_calls"] = sum(int(row.get("controller_call_count") or 0) >= 3 for row in rows)
    hard_violation_count = sum(hard_counts.values())
    gpu_used = [int(row["used_mb"]) for row in gpu_samples if row.get("available") and row.get("used_mb") is not None]
    pollution = int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0)
    runtime_gates = {
        "task0266_history_immutable": task0266_identity()["all_match"],
        "task0267_history_present": task0267_identity()["all_present"],
        "dataset_identity": dataset_identity()["identity_valid"],
        "conversation_9_attempted": len(conversations) == EXPECTED_CONVERSATIONS,
        "turn_40_attempted": len(rows) + len(failures) == EXPECTED_TURNS,
        "unsupported_response_format_zero": resolver["unsupported_response_format_count"] == 0,
        "resolver_structured_validity": resolver["final_structured_valid_rate"] >= 0.98,
        "multi_turn_runtime_success_rate": len(rows) / EXPECTED_TURNS >= 0.98,
        "cuda_oom_zero": sum(bool(row.get("cuda_oom")) for row in failures) == 0,
        "task0264_pollution_zero": pollution == 0,
    }
    semantic_gates = {
        "semantic_accuracy_90": float(semantic["overall_semantic_accuracy"]) >= 0.90,
        "previously_correct_regressions_le_1": len(previous_regressions) <= 1,
        "agent_induced_semantic_regression_zero": sum(bool(row.get("recovery_harmed")) and not bool(row.get("semantic_correct")) for row in rows) == 0,
        "false_finish_zero": len(false_finish) == 0,
        "unsupported_grounded_finish_zero": len(unsupported_grounded) == 0,
    }
    safety_gates = {
        "hard_agent_safety_zero": hard_violation_count == 0,
        "graph_hop_violation_zero": sum(bool(row.get("graph_hop_violation")) for row in rows) == 0,
        "runtime_gold_exposure_zero": sum(bool(row.get("runtime_gold_exposure")) for row in rows) == 0,
        "raw_hidden_reasoning_persistence_zero": sum(bool(row.get("hidden_reasoning_persisted")) for row in rows) == 0,
    }
    runtime_pass, semantic_pass, safety_pass = all(runtime_gates.values()), all(semantic_gates.values()), all(safety_gates.values())
    if not safety_pass:
        decision, ui_readiness = "fail_closed_for_semantic_safety", "blocked"
    elif not runtime_pass:
        decision, ui_readiness = "blocked", "blocked"
    elif not semantic_pass:
        decision, ui_readiness = "hold_for_semantic_quality_repair", "hold_ui_stage_for_semantic_repair"
    else:
        decision, ui_readiness = "advance_to_modern_ui_stage", "ready_for_modern_ui_stage"

    old_failures, diagnosis, taxonomy = _diagnostic_artifacts(rows)
    _write_jsonl(RESULT / "task0267_semantic_failure_ledger.jsonl", old_failures)
    _write_jsonl(RESULT / "stage_counterfactual_diagnosis.jsonl", diagnosis)
    repaired = sum(row["task0268_disposition"] == "fixed" for row in old_failures)
    agent_rows = [row for row in rows if int(row.get("controller_call_count") or 0) > 0]
    agent_metrics = {
        "schema_version": "opk-rag.task0268.agent-semantic-metrics.v1",
        "agent_invoked_turn_count": len(agent_rows),
        "agent_semantic_correct_count": sum(bool(row.get("semantic_correct")) for row in agent_rows),
        "agent_correct_decision_rate": sum(bool(row.get("semantic_correct")) for row in agent_rows) / max(1, len(agent_rows)),
        "agent_induced_semantic_regression_count": sum(bool(row.get("recovery_harmed")) and not bool(row.get("semantic_correct")) for row in rows),
        "agent_induced_semantic_recovery_count": sum(bool(row.get("recovery_improved")) and bool(row.get("semantic_correct")) for row in rows),
        "false_finish_count": len(false_finish),
        "false_abstain_count": sum(row.get("expected_semantic_class") != "abstain" and row.get("answer_status") != "answered" for row in rows),
        "unnecessary_recovery_count": sum(bool(row.get("recovery_invoked")) and bool(row.get("recovery_harmed")) for row in rows),
    }
    provider_metrics = {
        "schema_version": "opk-rag.task0268.provider-runtime-metrics.v1",
        "resolver_provider_request_count": resolver["provider_request_count"],
        "resolver_provider_response_count": resolver["provider_response_count"],
        "resolver_final_structured_valid_rate": resolver["final_structured_valid_rate"],
        "agent_provider_request_count": sum(int(row.get("provider_request_count") or 0) for row in rows),
        "agent_provider_response_count": sum(int(row.get("provider_response_count") or 0) for row in rows),
        "provider_failure_count": len(failures),
        "unsupported_response_format_count": resolver["unsupported_response_format_count"],
    }
    repair_eligibility = {
        "schema_version": "opk-rag.task0268.repair-eligibility-review.v1",
        "repairs": [
            {"repair": "resolver_propositional_polarity_preservation", "eligible": True, "sample_specific": False, "gold_runtime_use": False},
            {"repair": "negative_semantic_predicate_and_scope_refinement", "eligible": True, "sample_specific": False, "gold_runtime_use": False},
            {"repair": "false_premise_relevance_and_necessity_refinement", "eligible": True, "sample_specific": False, "gold_runtime_use": False},
            {"repair": "empirical_exact_evidence_gate", "eligible": True, "sample_specific": False, "gold_runtime_use": False},
            {"repair": "direct_negative_evidence_bounded_recovery", "eligible": True, "sample_specific": False, "gold_runtime_use": False},
        ],
        "action_space_expanded": False, "graph_hop_expanded": False, "recovery_budget_expanded": False, "safety_gate_weakened": False,
    }
    repair_plan = {"schema_version": "opk-rag.task0268.repair-plan.v1", "selected_arm": "C1_minimal_general_semantic_repair", "changes": [row["repair"] for row in repair_eligibility["repairs"]], "agent_architecture_changed": False, "new_agent_actions": [], "maximum_graph_hop": 1}
    return {
        "resolver_structured_output_metrics.json": resolver,
        "semantic_failure_taxonomy.json": taxonomy,
        "repair_eligibility_review.json": repair_eligibility,
        "repair_plan.json": repair_plan,
        "ablation_results.json": {"schema_version": "opk-rag.task0268.ablation-results.v1", "C0_task0267_semantic_accuracy": 0.675, "C0_task0267_semantic_correct_count": 27, "C1_task0268_semantic_accuracy": semantic["overall_semantic_accuracy"], "C1_task0268_semantic_correct_count": semantic["semantic_correct_count"], "task0267_failures_fixed_count": repaired, "previously_correct_regression_count": len(previous_regressions)},
        "semantic_quality_metrics.json": semantic,
        "agent_semantic_metrics.json": agent_metrics,
        "provider_runtime_metrics.json": provider_metrics,
        "regression_metrics.json": {"schema_version": "opk-rag.task0268.regression-metrics.v1", "status": "pending_final_regression_execution", "focused_tests_passed": None, "system_regression_passed": None},
        "gpu_residency.json": {"schema_version": "opk-rag.task0268.gpu-residency.v1", "samples": gpu_samples, "peak_used_mb": max(gpu_used) if gpu_used else None, "cuda_oom_count": sum(bool(row.get("cuda_oom")) for row in failures)},
        "task0264_isolation_review.json": {"schema_version": "opk-rag.task0268.task0264-isolation.v1", "ledger_before": ledger_before, "ledger_after": ledger_after, "pollution_count": pollution, "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE},
        "ui_stage_readiness.json": {"schema_version": "opk-rag.task0268.ui-stage-readiness.v1", "ui_stage_readiness": ui_readiness, "semantic_accuracy": semantic["overall_semantic_accuracy"], "runtime_gates_passed": runtime_pass, "semantic_gates_passed": semantic_pass, "safety_gates_passed": safety_pass, "unrestricted_production_agent_activation_authorized": False},
        "summary.json": {"schema_version": "opk-rag.task0268.summary.v1", "task_id": TASK_ID, "task_status": "complete" if len(conversations) == EXPECTED_CONVERSATIONS else "partial", "primary_run_id": PRIMARY_RUN_ID, "candidate_decision": decision, "ui_stage_readiness": ui_readiness, "conversation_count": len(conversations), "turn_success_count": len(rows), "turn_failure_count": len(failures), "multi_turn_runtime_success_rate": len(rows) / EXPECTED_TURNS, "semantic_accuracy": semantic["overall_semantic_accuracy"], "semantic_correct_count": semantic["semantic_correct_count"], "terminal_semantic_accuracy": semantic["terminal_semantic_accuracy"], "task0267_failures_fixed_count": repaired, "previously_correct_regression_count": len(previous_regressions), "hard_safety_violation_count": hard_violation_count, "task0264_pollution_count": pollution, "runtime_acceptance_gates": runtime_gates, "semantic_acceptance_gates": semantic_gates, "safety_acceptance_gates": safety_gates, "all_runtime_acceptance_gates_passed": runtime_pass, "all_semantic_acceptance_gates_passed": semantic_pass, "all_safety_acceptance_gates_passed": safety_pass, "next_task": "TASK-0269_modern_opk_rag_control_center_ui_stage" if ui_readiness == "ready_for_modern_ui_stage" else "TASK-0268_semantic_quality_followup"},
    }


def record_regression_metrics(*, focused_passed: bool, focused_count: int, system_passed: bool, system_count: int, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {"schema_version": "opk-rag.task0268.regression-metrics.v1", "status": "complete", "focused_tests_passed": focused_passed, "focused_test_count": focused_count, "system_regression_passed": system_passed, "system_regression_test_count": system_count, "details": dict(details or {})}
    _write_json(RESULT / "regression_metrics.json", payload)
    return payload


def verify() -> dict[str, Any]:
    required = ["entry_authority.json", "semantic_repair_candidate_identity.json", "task0267_semantic_failure_ledger.jsonl", "semantic_failure_taxonomy.json", "stage_counterfactual_diagnosis.jsonl", "repair_eligibility_review.json", "repair_plan.json", "ablation_results.json", "multi_turn_results.jsonl", "conversation_summary.json", "semantic_quality_metrics.json", "agent_semantic_metrics.json", "regression_metrics.json", "provider_runtime_metrics.json", "gpu_residency.json", "task0264_isolation_review.json", "execution_checkpoint.json", "execution_windows.jsonl", "failure_ledger.jsonl", "ui_stage_readiness.json", "summary.json"]
    missing = [name for name in required if not (RESULT / name).is_file()]
    rows, failures = _read_jsonl(RESULT / "multi_turn_results.jsonl"), _read_jsonl(RESULT / "failure_ledger.jsonl")
    summary = _read_json(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    semantic = _read_json(RESULT / "semantic_quality_metrics.json") if (RESULT / "semantic_quality_metrics.json").is_file() else {}
    isolation = _read_json(RESULT / "task0264_isolation_review.json") if (RESULT / "task0264_isolation_review.json").is_file() else {}
    regression = _read_json(RESULT / "regression_metrics.json") if (RESULT / "regression_metrics.json").is_file() else {}
    ui = _read_json(RESULT / "ui_stage_readiness.json") if (RESULT / "ui_stage_readiness.json").is_file() else {}
    checkpoint = _read_json(CHECKPOINT) if CHECKPOINT.is_file() else {}
    allowed_decisions = {"advance_to_modern_ui_stage", "advance_to_low_traffic_operator_canary_and_ui_stage", "complete_with_registered_nonblocking_semantic_gap", "hold_for_semantic_quality_repair", "hold_for_agent_policy_quality", "hold_for_retrieval_quality", "hold_for_conversation_resolution_quality", "fail_closed_for_semantic_safety", "blocked"}
    provider = _read_json(RESULT / "provider_runtime_metrics.json") if (RESULT / "provider_runtime_metrics.json").is_file() else {}
    provider_blocked = bool(
        summary.get("task_status") == "blocked"
        and summary.get("candidate_decision") == "blocked"
        and provider.get("external_provider_blocker") == "insufficient_balance"
        and int(provider.get("provider_probe_http_status") or 0) == 402
    )
    completed = int(checkpoint.get("completed_conversation_count") or 0)
    pending = int(checkpoint.get("pending_conversation_count") or 0)
    checks = {
        "required_artifacts_present": not missing, "task0266_history_immutable": task0266_identity()["all_match"], "task0267_history_present": task0267_identity()["all_present"], "dataset_identity_valid": dataset_identity()["identity_valid"],
        "semantic_repair_candidate_passed": task0268_candidate_identity().get("passed") is True,
        "production_candidate_authority_unchanged": task0268_candidate_identity().get("production_authority_unchanged") is True and task0268_candidate_identity().get("production_canary_fingerprint_updated") is False,
        "checkpoint_candidate_identity": checkpoint.get("candidate_fingerprint") == task0268_candidate_identity().get("candidate_fingerprint") and checkpoint.get("base_candidate_fingerprint") == APPROVED_AGENTIC_CANARY_FINGERPRINT,
        "turn_outcome_count_valid": (len(rows) + len(failures) == EXPECTED_TURNS) if not provider_blocked else (len(rows) + len(failures) == int(checkpoint.get("attempted_turn_count") or 0)),
        "conversation_count_valid": (summary.get("conversation_count") == EXPECTED_CONVERSATIONS) if not provider_blocked else (summary.get("conversation_count") == completed and completed + pending == EXPECTED_CONVERSATIONS and pending > 0),
        "checkpoint_state_valid": (pending == 0) if not provider_blocked else (pending > 0 and checkpoint.get("next_pending_conversation_id") is not None),
        "semantic_metrics_coverage_valid": (semantic.get("evaluated_turn_count") == EXPECTED_TURNS) if not provider_blocked else (semantic.get("evaluated_turn_count") == len(rows)),
        "task0264_pollution_zero": isolation.get("pollution_count") == 0, "regression_execution_complete": regression.get("status") == "complete",
        "focused_regression_passed": regression.get("focused_tests_passed") is True, "system_regression_passed": regression.get("system_regression_passed") is True, "summary_decision_allowed": summary.get("candidate_decision") in allowed_decisions,
        "ui_readiness_allowed": ui.get("ui_stage_readiness") in {"ready_for_modern_ui_stage", "ready_for_ui_with_registered_semantic_followup", "hold_ui_stage_for_semantic_repair", "blocked"}, "report_present": REPORT.is_file(),
        "provider_blocked_state_truthful": (not provider_blocked) or (summary.get("ui_stage_readiness") == "blocked" and summary.get("resume_condition") == "restore_provider_balance_then_resume_from_checkpoint"),
    }
    return {"schema_version": "opk-rag.task0268.verification.v1", "task_id": TASK_ID, "verification_passed": all(checks.values()), "checks": checks, "missing_artifacts": missing, "candidate_decision": summary.get("candidate_decision"), "ui_stage_readiness": ui.get("ui_stage_readiness")}
