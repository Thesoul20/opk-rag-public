from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4

from opk_rag.answer.config import load_answer_generation_config
from opk_rag.answer.negative_semantics import classify_answer_semantics
from opk_rag.conversation.context import build_conversation_context
from opk_rag.conversation.models import (
    CONVERSATION_PROMPT_VERSION,
    FOLLOWUP_REWRITE_VERSION,
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
from opk_rag.showcase.bounded_agentic_canary import evaluate_task0263_readiness, load_canary_config
from opk_rag.showcase.live_selective_agent_shadow import classify_traffic


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0266"
SCHEMA = "opk-rag.task0266.controlled-realistic-stability.v1"
DATASET = ROOT / "evaluation-data/datasets/realistic-synthetic-traffic/realistic_rag_traffic_v1.json"
MANIFEST = ROOT / "evaluation-data/datasets/realistic-synthetic-traffic/manifest_v1.json"
EXPECTED_DATASET_SHA256 = "1a076de04d2a8932fd4d72ebe6bee264665a213dc64d67005eb2a8399938fa5d"
RESULT = ROOT / "evaluation-data/results/task0266-controlled-realistic-rag-stability"
REPORT = ROOT / "docs/TASK0266_CONTROLLED_REALISTIC_SINGLE_AND_MULTI_TURN_RAG_STABILITY_REPORT.md"
CHECKPOINT = RESULT / "execution_checkpoint.json"
WINDOW_HISTORY = RESULT / "execution_windows.jsonl"
PRIMARY_RUN_ID = "task0266-primary-v1"
EXPECTED_SINGLE = 60
EXPECTED_MULTI = 40
EXPECTED_CONVERSATIONS = 9
EXPECTED_TOTAL = 100
EXPECTED_TO_RUNTIME = {
    "answerable": "supported_affirmative",
    "supported_negative": "supported_negative",
    "partial": "supported_partial",
    "should_abstain": "abstain",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _expected_primary_ids(dataset: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    single_ids = [str(row["id"]) for row in list(dataset.get("single_turn_queries") or [])]
    multi_ids = [
        f"{conversation['conversation_id']}-T{int(turn['turn']):02d}"
        for conversation in list(dataset.get("conversations") or [])
        for turn in list(conversation.get("turns") or [])
    ]
    return single_ids, multi_ids


def _primary_authority_index(
    *,
    dataset: Mapping[str, Any],
    single_results: list[Mapping[str, Any]],
    multi_results: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    single_expected, multi_expected = _expected_primary_ids(dataset)
    expected = set(single_expected) | set(multi_expected)
    success_ids = [str(row.get("sample_id") or "") for row in [*single_results, *multi_results]]
    failure_ids = [str(row.get("sample_id") or "") for row in failures]
    all_ids = [*success_ids, *failure_ids]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(all_ids).items() if sample_id and count > 1)
    unexpected_ids = sorted(sample_id for sample_id in all_ids if sample_id not in expected)
    attempted = set(all_ids)
    pending_single = [sample_id for sample_id in single_expected if sample_id not in attempted]
    pending_multi = [sample_id for sample_id in multi_expected if sample_id not in attempted]
    return {
        "single_expected_ids": single_expected,
        "multi_expected_ids": multi_expected,
        "success_ids": success_ids,
        "failure_ids": failure_ids,
        "duplicate_ids": duplicate_ids,
        "unexpected_ids": unexpected_ids,
        "pending_single_ids": pending_single,
        "pending_multi_ids": pending_multi,
        "attempted_count": len(attempted & expected),
        "authority_valid": not duplicate_ids and not unexpected_ids,
    }


def _checkpoint_payload(
    *,
    dataset: Mapping[str, Any],
    identity: Mapping[str, Any],
    single_results: list[Mapping[str, Any]],
    multi_results: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    window_state: str,
    last_window: str | None = None,
) -> dict[str, Any]:
    index = _primary_authority_index(
        dataset=dataset,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
    )
    single_failures = sum(str(row.get("lane")) == "single_turn" for row in failures)
    multi_failures = sum(str(row.get("lane")) == "multi_turn" for row in failures)
    return {
        "schema_version": "opk-rag.resumable-primary-checkpoint.v1",
        "task_id": TASK_ID,
        "primary_run_id": PRIMARY_RUN_ID,
        "dataset_sha256": identity.get("actual_sha256"),
        "candidate_fingerprint": APPROVED_AGENTIC_CANARY_FINGERPRINT,
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        "window_state": window_state,
        "last_window": last_window,
        "single_turn": {
            "expected": EXPECTED_SINGLE,
            "success": len(single_results),
            "failure": single_failures,
            "attempted": len(single_results) + single_failures,
            "pending": len(index["pending_single_ids"]),
        },
        "multi_turn": {
            "expected": EXPECTED_MULTI,
            "success": len(multi_results),
            "failure": multi_failures,
            "attempted": len(multi_results) + multi_failures,
            "pending": len(index["pending_multi_ids"]),
        },
        "total": {
            "expected": EXPECTED_TOTAL,
            "attempted": index["attempted_count"],
            "pending": len(index["pending_single_ids"]) + len(index["pending_multi_ids"]),
        },
        "next_pending_sample_id": (index["pending_single_ids"] + index["pending_multi_ids"] or [None])[0],
        "primary_authority_valid": index["authority_valid"],
        "duplicate_primary_ids": index["duplicate_ids"],
        "unexpected_primary_ids": index["unexpected_ids"],
        "resume_default": True,
        "completed_primary_is_never_replayed_by_resume": True,
        "failed_primary_is_never_replayed_by_resume": True,
        "multi_turn_window_boundary": "whole_conversation",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_checkpoint(
    *,
    dataset: Mapping[str, Any],
    identity: Mapping[str, Any],
    single_results: list[Mapping[str, Any]],
    multi_results: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    window_state: str,
    last_window: str | None = None,
) -> dict[str, Any]:
    payload = _checkpoint_payload(
        dataset=dataset,
        identity=identity,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
        window_state=window_state,
        last_window=last_window,
    )
    _write_json(CHECKPOINT, payload)
    return payload


def _assert_resume_compatible(*, checkpoint: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if checkpoint.get("primary_run_id") != PRIMARY_RUN_ID:
        raise RuntimeError("task0266_resume_primary_run_id_mismatch")
    if checkpoint.get("dataset_sha256") != identity.get("actual_sha256"):
        raise RuntimeError("task0266_resume_dataset_identity_mismatch")
    if checkpoint.get("candidate_fingerprint") != APPROVED_AGENTIC_CANARY_FINGERPRINT:
        raise RuntimeError("task0266_resume_candidate_fingerprint_mismatch")
    if checkpoint.get("traffic_source") != CONTROLLED_REALISTIC_EVALUATION_SOURCE:
        raise RuntimeError("task0266_resume_traffic_source_mismatch")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_identity() -> dict[str, Any]:
    dataset = _read_json(DATASET)
    manifest = _read_json(MANIFEST)
    single = list(dataset.get("single_turn_queries") or [])
    conversations = list(dataset.get("conversations") or [])
    multi_count = sum(len(list(row.get("turns") or [])) for row in conversations)
    digest = _sha256(DATASET)
    schema_errors: list[str] = []
    if dataset.get("dataset_type") != "realistic_synthetic_rag_traffic":
        schema_errors.append("dataset_type")
    if dataset.get("synthetic") is not True:
        schema_errors.append("synthetic")
    if len(single) != EXPECTED_SINGLE:
        schema_errors.append("single_turn_count")
    if len(conversations) != EXPECTED_CONVERSATIONS:
        schema_errors.append("conversation_count")
    if multi_count != EXPECTED_MULTI:
        schema_errors.append("multi_turn_count")
    ids = [str(row.get("id") or "") for row in single]
    if ids != [f"S{i:03d}" for i in range(1, EXPECTED_SINGLE + 1)]:
        schema_errors.append("single_turn_ids")
    for conversation in conversations:
        turns = list(conversation.get("turns") or [])
        if [int(turn.get("turn") or 0) for turn in turns] != list(range(1, len(turns) + 1)):
            schema_errors.append(f"turn_sequence:{conversation.get('conversation_id')}")
        if conversation.get("synthetic") is not True:
            schema_errors.append(f"conversation_synthetic:{conversation.get('conversation_id')}")
    return {
        "schema_version": "opk-rag.task0266.dataset-identity.v1",
        "dataset_file": str(DATASET.relative_to(ROOT)),
        "manifest_file": str(MANIFEST.relative_to(ROOT)),
        "expected_sha256": EXPECTED_DATASET_SHA256,
        "actual_sha256": digest,
        "sha256_match": digest == EXPECTED_DATASET_SHA256,
        "manifest_sha256_match": manifest.get("dataset_sha256") == digest,
        "single_turn_count": len(single),
        "multi_turn_count": multi_count,
        "conversation_count": len(conversations),
        "total_query_count": len(single) + multi_count,
        "synthetic": dataset.get("synthetic") is True,
        "eligible_for_task0264_real_canary": manifest.get("eligible_for_task0264_real_canary"),
        "schema_errors": schema_errors,
        "identity_valid": digest == EXPECTED_DATASET_SHA256 and manifest.get("dataset_sha256") == digest and not schema_errors,
    }


def _canary_ledger_state() -> dict[str, Any]:
    config = load_canary_config(root=ROOT)
    path = config.store_path
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {"path": str(path), "exists": False, "line_count": 0, "sha256": None, "size_bytes": 0}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "line_count": sum(1 for line in data.splitlines() if line.strip()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _gpu_state() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = proc.stdout.strip().splitlines()[0]
        total, used, free = [int(value.strip()) for value in first.split(",")[:3]]
        return {"available": True, "total_mb": total, "used_mb": used, "free_mb": free}
    except Exception as exc:
        return {"available": False, "failure_code": type(exc).__name__}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ms": round(_percentile(values, 0.50), 3) if values else None,
        "p95_ms": round(_percentile(values, 0.95), 3) if values else None,
        "p99_ms": round(_percentile(values, 0.99), 3) if values else None,
        "max_ms": round(max(values), 3) if values else None,
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
    }


def _trace_result(
    *,
    sample_id: str,
    expected_answerability: str,
    difficulty: str,
    retrieval_traits: list[str],
    trace: Mapping[str, Any],
    answer: Any,
    candidate_telemetry: Mapping[str, Any],
    elapsed_ms: float,
    lane: str,
    conversation_id: str | None = None,
    turn: int | None = None,
    context_dependency: str | None = None,
    resolver: Any | None = None,
) -> dict[str, Any]:
    semantic = classify_answer_semantics(
        question=str(answer.search_response.query),
        bundle=answer.search_response.evidence_bundle,
        answerability=answer.controller_answerability or answer.answerability,
    )
    runtime_class = semantic.semantic_class
    expected_class = EXPECTED_TO_RUNTIME[expected_answerability]
    evaluation = dict(trace.get("controlled_realistic_evaluation") or {})
    canary = dict(trace.get("selective_agent_canary") or {})
    retrieval = dict(trace.get("retrieval") or {})
    rerank = dict(trace.get("rerank") or {})
    evidence = dict(trace.get("evidence") or {})
    timings = dict(trace.get("timings") or {})
    grounding = dict(trace.get("grounding") or {})
    outcome = dict(trace.get("outcome") or {})
    trace_meta = dict(trace.get("trace") or {})
    citations = tuple(answer.citations or ())
    row = {
        "schema_version": "opk-rag.task0266.primary-observation.v1",
        "sample_id": sample_id,
        "lane": lane,
        "conversation_id": conversation_id,
        "turn": turn,
        "difficulty": difficulty,
        "expected_answerability": expected_answerability,
        "expected_semantic_class": expected_class,
        "predicted_semantic_class": runtime_class,
        "semantic_correct": runtime_class == expected_class,
        "retrieval_traits": retrieval_traits,
        "context_dependency": context_dependency,
        "query_digest": _digest_text(str(answer.search_response.query)),
        "trace_id": trace_meta.get("trace_id"),
        "trace_status": trace_meta.get("status"),
        "answer_status": answer.status,
        "refusal_reason_code": answer.refusal_reason_code,
        "grounding_valid": bool(answer.grounding.valid),
        "citation_count": len(citations),
        "citation_paths": sorted({citation.relative_path for citation in citations}),
        "unsupported_claim_count": len(tuple(answer.unsupported_claims or ())),
        "total_elapsed_ms": round(elapsed_ms, 3),
        "trace_total_ms": timings.get("total_ms"),
        "reranking_ms": timings.get("reranking_ms"),
        "generation_ms": timings.get("generation_ms"),
        "retrieval_candidate_count": retrieval.get("initial_candidate_count"),
        "rerank_input_count": rerank.get("input_candidate_count"),
        "evidence_count": evidence.get("evidence_count"),
        "graph_hop_depth": (trace.get("runtime") or {}).get("graph_hop_depth"),
        "task0264_traffic_eligible": canary.get("traffic_eligible"),
        "task0264_selected": canary.get("selected"),
        "task0264_recorded": evaluation.get("task0264_observation_recorded"),
        "evaluation_candidate_executed": evaluation.get("candidate_executor_executed"),
        "candidate_fingerprint": evaluation.get("candidate_fingerprint"),
        "controller_call_count": int(candidate_telemetry.get("controller_call_count") or 0),
        "provider_request_count": int(candidate_telemetry.get("provider_request_count") or 0),
        "provider_response_count": int(candidate_telemetry.get("provider_response_count") or 0),
        "provider_valid_decision_count": int(candidate_telemetry.get("provider_valid_decision_count") or 0),
        "recovery_invoked": bool(candidate_telemetry.get("recovery_invoked")),
        "recovery_selected": candidate_telemetry.get("recovery_selected"),
        "recovery_harmed": bool(candidate_telemetry.get("recovery_harmed")),
        "recovery_improved": bool(candidate_telemetry.get("recovery_improved")),
        "veto_invoked": bool(candidate_telemetry.get("veto_invoked")),
        "veto_decision": candidate_telemetry.get("veto_decision"),
        "unsafe_finish": bool(candidate_telemetry.get("unsafe_finish")),
        "llm_direct_finish_authority": bool(candidate_telemetry.get("llm_direct_finish_authority")),
        "abstain_to_finish_override": bool(candidate_telemetry.get("abstain_to_finish_override")),
        "graph_hop_violation": bool(candidate_telemetry.get("graph_hop_violation")),
        "kb_mutation": bool(candidate_telemetry.get("kb_mutation")),
        "grounding_bypass": bool(candidate_telemetry.get("grounding_bypass")),
        "fabricated_citation": bool(candidate_telemetry.get("fabricated_citation")),
        "runtime_gold_exposure": bool(candidate_telemetry.get("runtime_gold_exposure")),
        "answer_digest": _digest_text(answer.answer) if answer.answer else None,
        "raw_answer_persisted": False,
        "raw_provider_output_persisted": False,
        "hidden_reasoning_persisted": False,
        "grounding_stage_state": grounding.get("stage_state"),
        "final_decision": outcome.get("final_decision"),
    }
    if resolver is not None:
        row.update(
            {
                "resolver_is_followup": bool(resolver.is_followup),
                "resolver_referenced_turn_numbers": list(resolver.referenced_turn_numbers),
                "resolver_referenced_citation_ids": list(resolver.referenced_citation_ids),
                "resolver_model_id": resolver.model_id,
                "resolver_latency_ms": resolver.latency_ms,
                "standalone_query_digest": _digest_text(resolver.standalone_query),
            }
        )
    return row


def _conversation_citations(turn_id: UUID, answer: Any) -> tuple[ConversationTurnCitation, ...]:
    rows = []
    for index, citation in enumerate(tuple(answer.citations or ()), 1):
        rows.append(
            ConversationTurnCitation(
                turn_id=turn_id,
                citation_id=citation.citation_id,
                document_id=UUID(str(citation.document_id)),
                chunk_id=UUID(str(citation.chunk_id)),
                relative_path=citation.relative_path,
                heading_path=tuple(citation.heading_path),
                start_line=citation.start_line,
                end_line=citation.end_line,
                content_hash=_digest_text(citation.snippet or ""),
                context_rank=index,
                created_at=datetime.now(timezone.utc),
                source_status="source_current",
            )
        )
    return tuple(rows)


def _build_turn(*, session_id: UUID, turn_number: int, user_query: str, resolution: Any, answer: Any, elapsed_ms: float) -> ConversationTurn:
    now = datetime.now(timezone.utc)
    return ConversationTurn(
        id=uuid4(),
        session_id=session_id,
        turn_number=turn_number,
        client_request_id=None,
        user_query=user_query,
        standalone_query=resolution.standalone_query,
        rewrite_status="resolved" if resolution.is_followup else "not_needed",
        answer_decision="answer" if answer.status == "answered" else "abstain",
        answer_text=answer.answer or None,
        abstention_reason=answer.refusal_reason_code,
        turn_status="completed" if answer.status == "answered" else "abstained",
        retrieval_mode=answer.search_response.retrieval_mode,
        reranking_enabled=True,
        provider_id=answer.provider_id,
        model_id=answer.model_id,
        model_version=answer.model_revision,
        prompt_fingerprint=None,
        input_token_count=answer.prompt_tokens,
        output_token_count=answer.completion_tokens,
        latency_ms=int(elapsed_ms),
        error_code=None,
        created_at=now,
        completed_at=now,
        metadata={},
    )


def _failure_row(*, sample_id: str, lane: str, exc: Exception, conversation_id: str | None = None, turn: int | None = None) -> dict[str, Any]:
    text = str(exc)
    lowered = text.lower()
    return {
        "schema_version": "opk-rag.task0266.failure.v1",
        "sample_id": sample_id,
        "lane": lane,
        "conversation_id": conversation_id,
        "turn": turn,
        "failure_class": type(exc).__name__,
        "cuda_oom": "out of memory" in lowered and "cuda" in lowered,
        "embedding_model_load_failure": "embeddingmodelloaderror" in lowered,
        "reranker_model_load_failure": "rerankermodelloaderror" in lowered,
        "timeout": "timeout" in type(exc).__name__.lower() or "timeout" in lowered,
        "primary_failure": True,
        "diagnostic_replay_performed": False,
    }


async def execute(
    *,
    write: bool = True,
    resume: bool = True,
    execution_lane: str = "auto",
    max_new_primary: int | None = None,
    max_new_conversations: int | None = None,
) -> dict[str, Any]:
    """Execute TASK-0266 as a resumable logical primary run.

    A tool/chat window is only an execution window; it is not a new benchmark run.
    Existing success *and* failure primary authority is immutable and skipped on
    resume. Multi-turn work is admitted only at whole-conversation boundaries so
    an interrupted window never resumes a conversation without its transient
    context.
    """
    if execution_lane not in {"auto", "single", "multi"}:
        raise ValueError("task0266_invalid_execution_lane")
    if max_new_primary is not None and max_new_primary < 1:
        raise ValueError("task0266_max_new_primary_must_be_positive")
    if max_new_conversations is not None and max_new_conversations < 1:
        raise ValueError("task0266_max_new_conversations_must_be_positive")

    load_project_env(root=ROOT)
    identity = dataset_identity()
    if not identity["identity_valid"]:
        if write:
            _write_json(RESULT / "dataset_identity.json", identity)
        raise RuntimeError("task0266_dataset_identity_invalid")

    readiness = evaluate_task0263_readiness(root=ROOT, expected_candidate_fingerprint=APPROVED_AGENTIC_CANARY_FINGERPRINT)
    if not readiness["passed"]:
        raise RuntimeError("task0266_frozen_candidate_readiness_failed")
    traffic_class, live_eligible = classify_traffic(execution_scope="ask", source=CONTROLLED_REALISTIC_EVALUATION_SOURCE)
    if traffic_class != CONTROLLED_REALISTIC_EVALUATION_SOURCE or live_eligible:
        raise RuntimeError("task0266_traffic_classification_invalid")

    dataset = _read_json(DATASET)
    RESULT.mkdir(parents=True, exist_ok=True)
    authority_paths = (
        RESULT / "single_turn_results.jsonl",
        RESULT / "multi_turn_results.jsonl",
        RESULT / "failure_ledger.jsonl",
    )
    authority_exists = any(path.is_file() and path.stat().st_size > 0 for path in authority_paths)
    if not resume and authority_exists:
        raise RuntimeError("task0266_fresh_execution_would_overwrite_existing_primary_authority")

    single_results = _read_jsonl(RESULT / "single_turn_results.jsonl") if resume else []
    multi_results = _read_jsonl(RESULT / "multi_turn_results.jsonl") if resume else []
    failures = _read_jsonl(RESULT / "failure_ledger.jsonl") if resume else []
    if resume and CHECKPOINT.is_file():
        _assert_resume_compatible(checkpoint=_read_json(CHECKPOINT), identity=identity)

    authority = _primary_authority_index(
        dataset=dataset,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
    )
    if not authority["authority_valid"]:
        raise RuntimeError("task0266_primary_authority_invalid_for_resume")

    # A multi-turn conversation is a transaction boundary. If a previous process
    # died after committing only some turns, resuming from redacted artifacts
    # would lose the transient answer/context state and would not be equivalent.
    # Fail closed rather than silently reconstructing or rewriting the history.
    attempted_ids = set(authority["success_ids"]) | set(authority["failure_ids"])
    for conversation in list(dataset.get("conversations") or []):
        conversation_id = str(conversation["conversation_id"])
        conversation_ids = {
            f"{conversation_id}-T{int(turn['turn']):02d}" for turn in list(conversation.get("turns") or [])
        }
        attempted_in_conversation = conversation_ids & attempted_ids
        if attempted_in_conversation and attempted_in_conversation != conversation_ids:
            raise RuntimeError(f"task0266_partial_conversation_authority_not_resumable:{conversation_id}")

    if write:
        for path in authority_paths:
            if not path.exists():
                _write_jsonl(path, [])
        _write_json(RESULT / "dataset_identity.json", identity)
        if not (RESULT / "execution_config.json").is_file():
            answer_config_for_metadata = load_answer_generation_config()
            _write_json(
                RESULT / "execution_config.json",
                {
                    "schema_version": "opk-rag.task0266.execution-config.v2",
                    "task_id": TASK_ID,
                    "primary_run_id": PRIMARY_RUN_ID,
                    "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
                    "synthetic": True,
                    "public_api_default_source_unchanged": True,
                    "runtime_interface": "ShowcaseExecutor_internal_controlled_evaluation_path",
                    "reason_internal_interface_required": "Public /ask intentionally has no caller-controlled source header; internal path preserves synthetic traffic identity and TASK-0264 isolation while reusing the same runtime components.",
                    "candidate_fingerprint": APPROVED_AGENTIC_CANARY_FINGERPRINT,
                    "candidate_readiness": readiness,
                    "conversation_resolver": answer_config_for_metadata.model_id,
                    "conversation_context_max_turns": 6,
                    "conversation_context_token_budget": 1200,
                    "primary_run_immutable": True,
                    "resumable_execution": True,
                    "resume_default": True,
                    "completed_and_failed_primary_rows_are_skipped": True,
                    "multi_turn_resume_boundary": "whole_conversation",
                },
            )

    checkpoint = _write_checkpoint(
        dataset=dataset,
        identity=identity,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
        window_state="loaded",
        last_window=None,
    ) if write else _checkpoint_payload(
        dataset=dataset,
        identity=identity,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
        window_state="loaded",
    )

    # No-op resume is intentional: once all 100 primary samples have authority,
    # the runner must never replay them simply because a new tool window opened.
    if checkpoint["total"]["pending"] == 0:
        if write:
            _write_checkpoint(
                dataset=dataset,
                identity=identity,
                single_results=single_results,
                multi_results=multi_results,
                failures=failures,
                window_state="primary_complete",
                last_window="resume_noop_all_primary_already_attempted",
            )
        if (RESULT / "summary.json").is_file():
            return _read_json(RESULT / "summary.json")
        return rebuild_from_existing(write=write)

    existing_conversation_summary = (
        _read_json(RESULT / "conversation_summary.json") if (RESULT / "conversation_summary.json").is_file() else {}
    )
    conversation_summaries = list(existing_conversation_summary.get("conversations") or [])
    completed_conversation_ids = {
        str(row.get("conversation_id")) for row in conversation_summaries if row.get("conversation_id")
    }
    prior_gpu = _read_json(RESULT / "gpu_residency.json") if (RESULT / "gpu_residency.json").is_file() else {}
    gpu_samples: list[dict[str, Any]] = list(prior_gpu.get("samples") or [])
    window_history = _read_jsonl(WINDOW_HISTORY)
    window_id = f"window-{len(window_history) + 1:03d}"
    window_started_at = datetime.now(timezone.utc).isoformat()
    ledger_before = _canary_ledger_state()
    if not gpu_samples:
        gpu_samples.append({"position": "before", "window_id": window_id, **_gpu_state()})
    else:
        gpu_samples.append({"position": "window_before", "window_id": window_id, **_gpu_state()})

    executor = ShowcaseExecutor(max_concurrent_executions=1)
    kb_id = UUID(executor._showcase_kb_id())
    answer_config = load_answer_generation_config()
    resolver = OpenAICompatibleFollowupQueryResolver(
        answer_config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None
    )
    token_counter = QwenContextTokenCounter(load_embedding_config())

    new_primary_attempts = 0
    new_conversations = 0
    new_successes = 0
    new_failures = 0
    stop_for_window_limit = False

    if execution_lane in {"auto", "single"}:
        pending_single = set(authority["pending_single_ids"])
        for index, item in enumerate(list(dataset["single_turn_queries"]), 1):
            sample_id = str(item["id"])
            if sample_id not in pending_single:
                continue
            if max_new_primary is not None and new_primary_attempts >= max_new_primary:
                stop_for_window_limit = True
                break
            started = time.perf_counter()
            try:
                result = await executor.run_ask_controlled_evaluation(str(item["query"]))
                elapsed = (time.perf_counter() - started) * 1000.0
                row = _trace_result(
                    sample_id=sample_id,
                    expected_answerability=str(item["answerability"]),
                    difficulty=str(item["difficulty"]),
                    retrieval_traits=list(item.get("retrieval_traits") or []),
                    trace=result.trace,
                    answer=result.answer,
                    candidate_telemetry=result.candidate_telemetry,
                    elapsed_ms=elapsed,
                    lane="single_turn",
                )
                single_results.append(row)
                if write:
                    _append_jsonl(RESULT / "single_turn_results.jsonl", row)
                new_successes += 1
                print(f"[single {index:02d}/{EXPECTED_SINGLE}] {sample_id} ok semantic={row['predicted_semantic_class']} calls={row['controller_call_count']}", flush=True)
            except Exception as exc:
                failure = _failure_row(sample_id=sample_id, lane="single_turn", exc=exc)
                failures.append(failure)
                if write:
                    _append_jsonl(RESULT / "failure_ledger.jsonl", failure)
                new_failures += 1
                print(f"[single {index:02d}/{EXPECTED_SINGLE}] {sample_id} FAIL {type(exc).__name__}", flush=True)
            new_primary_attempts += 1
            if new_primary_attempts == 1 and not any(row.get("position") == "after_warmup" for row in gpu_samples):
                gpu_samples.append({"position": "after_warmup", "window_id": window_id, "query_index": index, **_gpu_state()})
            if new_primary_attempts % 5 == 0:
                gpu_samples.append({"position": "periodic", "window_id": window_id, "query_index": index, **_gpu_state()})
            if write:
                _write_checkpoint(
                    dataset=dataset,
                    identity=identity,
                    single_results=single_results,
                    multi_results=multi_results,
                    failures=failures,
                    window_state="in_progress",
                    last_window=window_id,
                )

    authority = _primary_authority_index(
        dataset=dataset,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
    )

    if execution_lane in {"auto", "multi"} and not stop_for_window_limit:
        pending_multi = set(authority["pending_multi_ids"])
        multi_ordinal = 0
        for conversation in list(dataset["conversations"]):
            conversation_id = str(conversation["conversation_id"])
            turns = list(conversation.get("turns") or [])
            conversation_ids = [f"{conversation_id}-T{int(turn['turn']):02d}" for turn in turns]
            if not any(sample_id in pending_multi for sample_id in conversation_ids):
                continue
            # Whole-conversation admission: never split a new conversation just to
            # obey a numeric window size. If the window already did work, stop and
            # let the next window own this complete conversation.
            if max_new_conversations is not None and new_conversations >= max_new_conversations:
                stop_for_window_limit = True
                break
            if max_new_primary is not None and new_primary_attempts > 0 and new_primary_attempts + len(turns) > max_new_primary:
                stop_for_window_limit = True
                break

            now = datetime.now(timezone.utc)
            session = ConversationSession(
                id=uuid4(),
                knowledge_base_id=kb_id,
                title=conversation.get("topic"),
                status="active",
                created_at=now,
                updated_at=now,
                last_turn_at=None,
                turn_count=0,
                conversation_prompt_version=CONVERSATION_PROMPT_VERSION,
                followup_rewrite_version=FOLLOWUP_REWRITE_VERSION,
                metadata={"synthetic": True, "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE},
            )
            prior_turns: list[ConversationTurn] = []
            citations_by_turn: dict[UUID, tuple[ConversationTurnCitation, ...]] = {}
            staged_rows: list[dict[str, Any]] = []
            staged_failures: list[dict[str, Any]] = []
            context_failures = 0
            for turn_item in turns:
                multi_ordinal += 1
                turn_number = int(turn_item["turn"])
                sample_id = f"{conversation_id}-T{turn_number:02d}"
                started = time.perf_counter()
                try:
                    context = build_conversation_context(
                        session=session,
                        turns=tuple(prior_turns),
                        citations_by_turn=citations_by_turn,
                        token_counter=token_counter,
                        max_turns=6,
                        token_budget=1200,
                    )
                    resolution = resolver.resolve(current_query=str(turn_item["query"]), conversation_context=context)
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
                    staged_rows.append(row)
                    expected_dependency = str(turn_item.get("context_dependency") or "none")
                    if expected_dependency in {"medium", "high"} and prior_turns and not resolution.is_followup:
                        context_failures += 1
                    turn = _build_turn(
                        session_id=session.id,
                        turn_number=turn_number,
                        user_query=str(turn_item["query"]),
                        resolution=resolution,
                        answer=result.answer,
                        elapsed_ms=elapsed,
                    )
                    prior_turns.append(turn)
                    citations_by_turn[turn.id] = _conversation_citations(turn.id, result.answer)
                    print(f"[multi {conversation_id} T{turn_number:02d}] ok followup={resolution.is_followup} semantic={row['predicted_semantic_class']}", flush=True)
                except Exception as exc:
                    failure = _failure_row(
                        sample_id=sample_id,
                        lane="multi_turn",
                        exc=exc,
                        conversation_id=conversation_id,
                        turn=turn_number,
                    )
                    staged_failures.append(failure)
                    print(f"[multi {conversation_id} T{turn_number:02d}] FAIL {type(exc).__name__}", flush=True)

            # Commit a complete conversation batch only after every frozen turn has
            # been attempted. This is the resume boundary.
            multi_results.extend(staged_rows)
            failures.extend(staged_failures)
            if write:
                for row in staged_rows:
                    _append_jsonl(RESULT / "multi_turn_results.jsonl", row)
                for failure in staged_failures:
                    _append_jsonl(RESULT / "failure_ledger.jsonl", failure)
            new_successes += len(staged_rows)
            new_failures += len(staged_failures)
            new_primary_attempts += len(turns)
            new_conversations += 1
            if conversation_id not in completed_conversation_ids:
                conversation_summaries.append(
                    {
                        "conversation_id": conversation_id,
                        "pattern": conversation.get("conversation_pattern"),
                        "turn_count": len(turns),
                        "attempted_turn_count": len(turns),
                        "successful_turn_count": len(staged_rows),
                        "context_resolution_failure_count": context_failures,
                    }
                )
                completed_conversation_ids.add(conversation_id)
            gpu_samples.append(
                {"position": "conversation_boundary", "window_id": window_id, "conversation_id": conversation_id, **_gpu_state()}
            )
            if write:
                _write_json(
                    RESULT / "conversation_summary.json",
                    {
                        "schema_version": "opk-rag.task0266.conversation-summary.v1",
                        "conversation_count": len(conversation_summaries),
                        "conversations": conversation_summaries,
                        "attempted_turn_count": sum(int(row.get("attempted_turn_count") or 0) for row in conversation_summaries),
                        "successful_turn_count": sum(int(row.get("successful_turn_count") or 0) for row in conversation_summaries),
                    },
                )
                _write_checkpoint(
                    dataset=dataset,
                    identity=identity,
                    single_results=single_results,
                    multi_results=multi_results,
                    failures=failures,
                    window_state="in_progress",
                    last_window=window_id,
                )

    ledger_after = _canary_ledger_state()
    gpu_samples.append({"position": "window_final", "window_id": window_id, **_gpu_state()})
    authority = _primary_authority_index(
        dataset=dataset,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
    )
    checkpoint = _checkpoint_payload(
        dataset=dataset,
        identity=identity,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
        window_state="primary_complete" if not authority["pending_single_ids"] and not authority["pending_multi_ids"] else "window_complete",
        last_window=window_id,
    )
    if write:
        _write_json(CHECKPOINT, checkpoint)
        _append_jsonl(
            WINDOW_HISTORY,
            {
                "schema_version": "opk-rag.resumable-execution-window.v1",
                "task_id": TASK_ID,
                "primary_run_id": PRIMARY_RUN_ID,
                "window_id": window_id,
                "started_at": window_started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "execution_lane": execution_lane,
                "new_primary_attempt_count": new_primary_attempts,
                "new_success_count": new_successes,
                "new_failure_count": new_failures,
                "new_conversation_count": new_conversations,
                "stopped_for_window_limit": stop_for_window_limit,
                "ledger_before": ledger_before,
                "ledger_after": ledger_after,
                "task0264_pollution_delta": int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0),
                "gpu_sample_count": len([row for row in gpu_samples if row.get("window_id") == window_id]),
                "pending_after": checkpoint["total"]["pending"],
            },
        )

    if checkpoint["total"]["pending"] > 0:
        return {
            "schema_version": "opk-rag.task0266.execution-window-result.v1",
            "task_id": TASK_ID,
            "task_status": "in_progress",
            "primary_run_id": PRIMARY_RUN_ID,
            "window_id": window_id,
            "new_primary_attempt_count": new_primary_attempts,
            "pending_primary_count": checkpoint["total"]["pending"],
            "next_pending_sample_id": checkpoint["next_pending_sample_id"],
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        }

    window_history = _read_jsonl(WINDOW_HISTORY)
    initial_ledger = dict(window_history[0].get("ledger_before") or {}) if window_history else ledger_before
    artifacts = evaluate_results(
        identity=identity,
        single_results=single_results,
        multi_results=multi_results,
        failures=failures,
        conversation_summaries=conversation_summaries,
        gpu_samples=gpu_samples,
        ledger_before=initial_ledger,
        ledger_after=ledger_after,
    )
    if write:
        _write_jsonl(RESULT / "single_turn_results.jsonl", single_results)
        _write_jsonl(RESULT / "multi_turn_results.jsonl", multi_results)
        _write_jsonl(RESULT / "failure_ledger.jsonl", failures)
        for name, payload in artifacts.items():
            _write_json(RESULT / name, payload)
    return artifacts["summary.json"]


def evaluate_results(
    *,
    identity: Mapping[str, Any],
    single_results: list[Mapping[str, Any]],
    multi_results: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    conversation_summaries: list[Mapping[str, Any]],
    gpu_samples: list[Mapping[str, Any]],
    ledger_before: Mapping[str, Any],
    ledger_after: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [*single_results, *multi_results]
    calls = [int(row.get("controller_call_count") or 0) for row in rows]
    provider_req = sum(int(row.get("provider_request_count") or 0) for row in rows)
    provider_resp = sum(int(row.get("provider_response_count") or 0) for row in rows)
    provider_valid = sum(int(row.get("provider_valid_decision_count") or 0) for row in rows)
    controller_decisions = sum(calls)
    provider_response_rate = provider_resp / provider_req if provider_req else 1.0
    structured_validity = provider_valid / controller_decisions if controller_decisions else 1.0
    hard_fields = (
        "unsafe_finish",
        "llm_direct_finish_authority",
        "abstain_to_finish_override",
        "graph_hop_violation",
        "kb_mutation",
        "grounding_bypass",
        "fabricated_citation",
        "runtime_gold_exposure",
    )
    hard_counts = {field: sum(bool(row.get(field)) for row in rows) for field in hard_fields}
    hard_counts["three_plus_controller_calls"] = sum(int(row.get("controller_call_count") or 0) >= 3 for row in rows)
    hard_counts["recovery_harmed"] = sum(bool(row.get("recovery_harmed")) for row in rows)
    hard_counts["runtime_graph_hop_gt_1"] = sum(int(row.get("graph_hop_depth") or 0) > 1 for row in rows)
    hard_violation_count = sum(hard_counts.values())

    expected_neg = {str(row["sample_id"]) for row in rows if row.get("expected_semantic_class") == "supported_negative"}
    predicted_neg = {str(row["sample_id"]) for row in rows if row.get("predicted_semantic_class") == "supported_negative"}
    tp = len(expected_neg & predicted_neg)
    fp = len(predicted_neg - expected_neg)
    fn = len(expected_neg - predicted_neg)
    semantic_correct = sum(bool(row.get("semantic_correct")) for row in rows)
    should_abstain = [row for row in rows if row.get("expected_semantic_class") == "abstain"]
    partial = [row for row in rows if row.get("expected_semantic_class") == "supported_partial"]
    context_rows = [row for row in multi_results if row.get("context_dependency") in {"medium", "high"}]
    context_success = sum(bool(row.get("resolver_is_followup")) for row in context_rows)

    answerability_metrics = {
        "schema_version": "opk-rag.task0266.answerability-metrics.v1",
        "executed_count": len(rows),
        "semantic_correct_count": semantic_correct,
        "overall_semantic_accuracy": semantic_correct / max(1, len(rows)),
        "single_turn_semantic_accuracy": sum(bool(row.get("semantic_correct")) for row in single_results) / max(1, len(single_results)),
        "multi_turn_semantic_accuracy": sum(bool(row.get("semantic_correct")) for row in multi_results) / max(1, len(multi_results)),
        "context_dependent_turn_accuracy": sum(bool(row.get("semantic_correct")) for row in context_rows) / max(1, len(context_rows)),
        "supported_negative_precision": tp / (tp + fp) if tp + fp else 1.0,
        "supported_negative_recall": tp / (tp + fn) if tp + fn else 1.0,
        "should_abstain_accuracy": sum(row.get("predicted_semantic_class") == "abstain" for row in should_abstain) / max(1, len(should_abstain)),
        "partial_handling_accuracy": sum(row.get("predicted_semantic_class") == "supported_partial" for row in partial) / max(1, len(partial)),
        "confusion": dict(Counter(f"{row.get('expected_semantic_class')}->{row.get('predicted_semantic_class')}" for row in rows)),
        "runtime_gold_metadata_usage": False,
    }
    agent_metrics = {
        "schema_version": "opk-rag.task0266.agent-behavior.v1",
        "query_count": len(rows),
        "agent_invocation_rate": sum(call > 0 for call in calls) / max(1, len(calls)),
        "average_controller_calls": statistics.fmean(calls) if calls else 0.0,
        "controller_calls_total": sum(calls),
        "recovery_count": sum(bool(row.get("recovery_invoked")) for row in rows),
        "recovery_success_count": sum(bool(row.get("recovery_improved")) for row in rows),
        "recovery_harmed_count": sum(bool(row.get("recovery_harmed")) for row in rows),
        "veto_count": sum(bool(row.get("veto_invoked")) for row in rows),
        "hard_violation_count": hard_violation_count,
        "hard_violation_counts": hard_counts,
    }
    provider = {
        "schema_version": "opk-rag.task0266.provider-reliability.v1",
        "controller_provider_request_count": provider_req,
        "controller_provider_response_count": provider_resp,
        "controller_provider_valid_decision_count": provider_valid,
        "controller_decision_count": controller_decisions,
        "provider_response_rate": provider_response_rate,
        "structured_validity_rate": structured_validity,
        "threshold": 0.98,
        "primary_execution_failure_count": len(failures),
        "timeout_count": sum(bool(row.get("timeout")) for row in failures),
    }
    grounding = {
        "schema_version": "opk-rag.task0266.grounding-citation.v1",
        "answered_count": sum(row.get("answer_status") == "answered" for row in rows),
        "grounding_pass_count": sum(bool(row.get("grounding_valid")) for row in rows),
        "grounding_pass_rate": sum(bool(row.get("grounding_valid")) for row in rows) / max(1, len(rows)),
        "answered_without_citation_count": sum(row.get("answer_status") == "answered" and int(row.get("citation_count") or 0) == 0 for row in rows),
        "fabricated_citation_count": hard_counts["fabricated_citation"],
        "grounding_bypass_count": hard_counts["grounding_bypass"],
    }
    latency_groups = {
        "overall": [float(row.get("total_elapsed_ms") or 0.0) for row in rows],
        "single_turn": [float(row.get("total_elapsed_ms") or 0.0) for row in single_results],
        "multi_turn": [float(row.get("total_elapsed_ms") or 0.0) for row in multi_results],
        "agent_invoked": [float(row.get("total_elapsed_ms") or 0.0) for row in rows if int(row.get("controller_call_count") or 0) > 0],
        "agent_not_invoked": [float(row.get("total_elapsed_ms") or 0.0) for row in rows if int(row.get("controller_call_count") or 0) == 0],
        "recovery": [float(row.get("total_elapsed_ms") or 0.0) for row in rows if row.get("recovery_invoked")],
        "no_recovery": [float(row.get("total_elapsed_ms") or 0.0) for row in rows if not row.get("recovery_invoked")],
    }
    latency = {"schema_version": "opk-rag.task0266.latency.v1", **{key: _latency_summary(values) for key, values in latency_groups.items()}}
    gpu_used = [int(row.get("used_mb")) for row in gpu_samples if row.get("available") and row.get("used_mb") is not None]
    periodic = [int(row.get("used_mb")) for row in gpu_samples if row.get("position") == "periodic" and row.get("available")]
    monotonic_growth = len(periodic) >= 4 and all(b >= a for a, b in zip(periodic, periodic[1:])) and (periodic[-1] - periodic[0] > 512)
    gpu = {
        "schema_version": "opk-rag.task0266.gpu-residency.v1",
        "samples": gpu_samples,
        "peak_used_mb": max(gpu_used) if gpu_used else None,
        "final_used_mb": next((row.get("used_mb") for row in reversed(gpu_samples) if row.get("available")), None),
        "cuda_oom_count": sum(bool(row.get("cuda_oom")) for row in failures),
        "embedding_model_load_failure_count": sum(bool(row.get("embedding_model_load_failure")) for row in failures),
        "reranker_model_load_failure_count": sum(bool(row.get("reranker_model_load_failure")) for row in failures),
        "material_monotonic_growth_detected": monotonic_growth,
    }
    isolation = {
        "schema_version": "opk-rag.task0266.task0264-isolation.v1",
        "traffic_source": CONTROLLED_REALISTIC_EVALUATION_SOURCE,
        "traffic_class": classify_traffic(execution_scope="ask", source=CONTROLLED_REALISTIC_EVALUATION_SOURCE)[0],
        "live_traffic_eligible": classify_traffic(execution_scope="ask", source=CONTROLLED_REALISTIC_EVALUATION_SOURCE)[1],
        "all_primary_rows_task0264_ineligible": all(row.get("task0264_traffic_eligible") is False and row.get("task0264_selected") is False for row in rows),
        "all_primary_rows_task0264_unrecorded": all(row.get("task0264_recorded") is False for row in rows),
        "ledger_before": ledger_before,
        "ledger_after": ledger_after,
        "ledger_line_count_delta": int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0),
        "ledger_sha256_unchanged": ledger_before.get("sha256") == ledger_after.get("sha256"),
        "pollution_count": 0 if int(ledger_after.get("line_count") or 0) == int(ledger_before.get("line_count") or 0) else abs(int(ledger_after.get("line_count") or 0) - int(ledger_before.get("line_count") or 0)),
    }
    context_review = {
        "schema_version": "opk-rag.task0266.multi-turn-context-review.v1",
        "multi_turn_executed_count": len(multi_results),
        "context_dependent_turn_count": len(context_rows),
        "resolver_followup_count": sum(bool(row.get("resolver_is_followup")) for row in multi_results),
        "context_resolution_success_count": context_success,
        "context_resolution_failure_count": len(context_rows) - context_success,
        "context_resolution_success_rate": context_success / max(1, len(context_rows)),
        "conversation_summaries": conversation_summaries,
    }
    conversation_summary = {
        "schema_version": "opk-rag.task0266.conversation-summary.v1",
        "conversation_count": len(conversation_summaries),
        "conversations": conversation_summaries,
        "attempted_turn_count": sum(int(row.get("attempted_turn_count") or 0) for row in conversation_summaries),
        "successful_turn_count": sum(int(row.get("successful_turn_count") or 0) for row in conversation_summaries),
    }

    gates = {
        "dataset_identity": identity.get("identity_valid") is True,
        "single_turn_60_attempted": len(single_results) + sum(row.get("lane") == "single_turn" for row in failures) == EXPECTED_SINGLE,
        "multi_turn_40_attempted": len(multi_results) + sum(row.get("lane") == "multi_turn" for row in failures) == EXPECTED_MULTI,
        "conversation_9_attempted": len(conversation_summaries) == EXPECTED_CONVERSATIONS,
        "unexpected_runtime_crash_zero": len(failures) == 0,
        "cuda_oom_zero": gpu["cuda_oom_count"] == 0,
        "embedding_model_load_failure_zero": gpu["embedding_model_load_failure_count"] == 0,
        "reranker_model_load_failure_zero": gpu["reranker_model_load_failure_count"] == 0,
        "gpu_material_monotonic_growth_zero": gpu["material_monotonic_growth_detected"] is False,
        "provider_response_rate": provider_response_rate >= 0.98,
        "structured_validity_rate": structured_validity >= 0.98,
        "hard_agent_invariants_zero": hard_violation_count == 0,
        "grounding_bypass_zero": grounding["grounding_bypass_count"] == 0,
        "fabricated_citation_zero": grounding["fabricated_citation_count"] == 0,
        "task0264_primary_rows_ineligible": isolation["all_primary_rows_task0264_ineligible"],
        "task0264_primary_rows_unrecorded": isolation["all_primary_rows_task0264_unrecorded"],
        "task0264_ledger_pollution_zero": isolation["pollution_count"] == 0,
    }
    stability_pass = all(gates.values())
    quality_followup = answerability_metrics["overall_semantic_accuracy"] < 1.0 or context_review["context_resolution_success_rate"] < 1.0
    if hard_violation_count:
        decision = "fail_closed_and_investigate"
    elif not stability_pass:
        decision = "hold_for_runtime_reliability"
    elif quality_followup:
        decision = "complete_with_quality_followup"
    else:
        decision = "pass_realistic_stability_gate"
    summary = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "task_status": "complete" if stability_pass else "partial",
        "implementation_complete": True,
        "controlled_realistic_evaluation_executed": True,
        "synthetic": True,
        "candidate_decision": decision,
        "dataset_sha256": identity.get("actual_sha256"),
        "single_turn_primary_success_count": len(single_results),
        "multi_turn_primary_success_count": len(multi_results),
        "primary_execution_failure_count": len(failures),
        "conversation_count": len(conversation_summaries),
        "provider_response_rate": provider_response_rate,
        "structured_validity_rate": structured_validity,
        "average_controller_calls": agent_metrics["average_controller_calls"],
        "agent_invocation_rate": agent_metrics["agent_invocation_rate"],
        "recovery_count": agent_metrics["recovery_count"],
        "recovery_harmed_count": agent_metrics["recovery_harmed_count"],
        "hard_safety_violation_count": hard_violation_count,
        "overall_semantic_accuracy": answerability_metrics["overall_semantic_accuracy"],
        "single_turn_semantic_accuracy": answerability_metrics["single_turn_semantic_accuracy"],
        "multi_turn_semantic_accuracy": answerability_metrics["multi_turn_semantic_accuracy"],
        "supported_negative_precision": answerability_metrics["supported_negative_precision"],
        "supported_negative_recall": answerability_metrics["supported_negative_recall"],
        "should_abstain_accuracy": answerability_metrics["should_abstain_accuracy"],
        "partial_handling_accuracy": answerability_metrics["partial_handling_accuracy"],
        "context_resolution_success_rate": context_review["context_resolution_success_rate"],
        "cuda_oom_count": gpu["cuda_oom_count"],
        "gpu_peak_used_mb": gpu["peak_used_mb"],
        "task0264_real_canary_observation_pollution": isolation["pollution_count"],
        "all_acceptance_gates_passed": stability_pass,
        "acceptance_gates": gates,
        "next_task": "TASK-0267_low_traffic_operator_canary_governance_and_real_observation_policy" if stability_pass else "TASK-0267_conversation_resolver_provider_structured_output_compatibility_repair",
    }
    return {
        "conversation_summary.json": conversation_summary,
        "multi_turn_context_review.json": context_review,
        "agent_behavior_metrics.json": agent_metrics,
        "answerability_metrics.json": answerability_metrics,
        "grounding_citation_metrics.json": grounding,
        "provider_reliability.json": provider,
        "gpu_residency.json": gpu,
        "latency_metrics.json": latency,
        "task0264_isolation_review.json": isolation,
        "summary.json": summary,
    }


def rebuild_from_existing(*, write: bool = True) -> dict[str, Any]:
    identity = dataset_identity()
    single = _read_jsonl(RESULT / "single_turn_results.jsonl")
    multi = _read_jsonl(RESULT / "multi_turn_results.jsonl")
    failures = _read_jsonl(RESULT / "failure_ledger.jsonl")
    conversation = _read_json(RESULT / "conversation_summary.json") if (RESULT / "conversation_summary.json").is_file() else {"conversations": []}
    gpu = _read_json(RESULT / "gpu_residency.json") if (RESULT / "gpu_residency.json").is_file() else {"samples": []}
    isolation = _read_json(RESULT / "task0264_isolation_review.json") if (RESULT / "task0264_isolation_review.json").is_file() else {"ledger_before": {}, "ledger_after": {}}
    artifacts = evaluate_results(
        identity=identity,
        single_results=single,
        multi_results=multi,
        failures=failures,
        conversation_summaries=list(conversation.get("conversations") or []),
        gpu_samples=list(gpu.get("samples") or []),
        ledger_before=dict(isolation.get("ledger_before") or {}),
        ledger_after=dict(isolation.get("ledger_after") or {}),
    )
    if write:
        for name, payload in artifacts.items():
            _write_json(RESULT / name, payload)
    return artifacts["summary.json"]


def verify() -> dict[str, Any]:
    identity = dataset_identity()
    required = [
        "dataset_identity.json",
        "execution_config.json",
        "single_turn_results.jsonl",
        "multi_turn_results.jsonl",
        "conversation_summary.json",
        "multi_turn_context_review.json",
        "agent_behavior_metrics.json",
        "answerability_metrics.json",
        "grounding_citation_metrics.json",
        "provider_reliability.json",
        "gpu_residency.json",
        "latency_metrics.json",
        "failure_ledger.jsonl",
        "task0264_isolation_review.json",
        "execution_checkpoint.json",
        "execution_windows.jsonl",
        "multi_turn_failure_diagnosis.json",
        "summary.json",
    ]
    missing = [name for name in required if not (RESULT / name).is_file()]
    summary = _read_json(RESULT / "summary.json") if (RESULT / "summary.json").is_file() else {}
    single = _read_jsonl(RESULT / "single_turn_results.jsonl")
    multi = _read_jsonl(RESULT / "multi_turn_results.jsonl")
    failures = _read_jsonl(RESULT / "failure_ledger.jsonl")
    isolation = _read_json(RESULT / "task0264_isolation_review.json") if (RESULT / "task0264_isolation_review.json").is_file() else {}
    checkpoint = _read_json(CHECKPOINT) if CHECKPOINT.is_file() else {}
    windows = _read_jsonl(WINDOW_HISTORY)
    diagnosis = _read_json(RESULT / "multi_turn_failure_diagnosis.json") if (RESULT / "multi_turn_failure_diagnosis.json").is_file() else {}
    authority = _primary_authority_index(dataset=_read_json(DATASET), single_results=single, multi_results=multi, failures=failures)
    checks = {
        "required_artifacts_present": not missing,
        "dataset_identity_valid": identity["identity_valid"],
        "dataset_digest_frozen": identity["actual_sha256"] == EXPECTED_DATASET_SHA256,
        "single_turn_primary_attempts_60": len(single) + sum(row.get("lane") == "single_turn" for row in failures) == EXPECTED_SINGLE,
        "multi_turn_primary_attempts_40": len(multi) + sum(row.get("lane") == "multi_turn" for row in failures) == EXPECTED_MULTI,
        "all_rows_synthetic_canary_ineligible": all(row.get("task0264_traffic_eligible") is False and row.get("task0264_selected") is False and row.get("task0264_recorded") is False for row in [*single, *multi]),
        "task0264_pollution_zero": isolation.get("pollution_count") == 0,
        "candidate_fingerprint_stable": all(row.get("candidate_fingerprint") == APPROVED_AGENTIC_CANARY_FINGERPRINT for row in [*single, *multi]),
        "primary_authority_unique_and_expected": authority["authority_valid"] is True,
        "checkpoint_primary_run_identity": checkpoint.get("primary_run_id") == PRIMARY_RUN_ID,
        "checkpoint_dataset_identity": checkpoint.get("dataset_sha256") == EXPECTED_DATASET_SHA256,
        "checkpoint_primary_complete": checkpoint.get("total", {}).get("attempted") == EXPECTED_TOTAL and checkpoint.get("total", {}).get("pending") == 0,
        "checkpoint_resume_immutable": checkpoint.get("completed_primary_is_never_replayed_by_resume") is True and checkpoint.get("failed_primary_is_never_replayed_by_resume") is True,
        "window_history_present": bool(windows),
        "window_history_canary_pollution_zero": all(int(row.get("task0264_pollution_delta") or 0) == 0 for row in windows),
        "multi_turn_failure_diagnosed": diagnosis.get("primary_failure_count") == 40 and diagnosis.get("classification") == "production_conversation_resolver_provider_contract_incompatibility",
        "multi_turn_failure_not_mislabeled_evaluator_bug": diagnosis.get("evaluator_bug") is False,
        "summary_decision_allowed": summary.get("candidate_decision") in {"pass_realistic_stability_gate", "complete_with_quality_followup", "hold_for_runtime_reliability", "fail_closed_and_investigate", "blocked"},
        "summary_gate_consistent": summary.get("all_acceptance_gates_passed") == all(dict(summary.get("acceptance_gates") or {}).values()),
    }
    return {
        "schema_version": "opk-rag.task0266.verification.v1",
        "task_id": TASK_ID,
        "verification_passed": all(checks.values()),
        "checks": checks,
        "missing_artifacts": missing,
        "candidate_decision": summary.get("candidate_decision"),
    }
