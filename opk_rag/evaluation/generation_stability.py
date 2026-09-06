from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.models import AnswerGenerationRequest, AnswerSystemError
from opk_rag.answer.prompt import EVIDENCE_SERIALIZATION_VERSION, SYSTEM_PROMPT, render_user_prompt
from opk_rag.answer.provider import OpenAICompatibleLocalChatProvider, build_chat_completion_payload
from opk_rag.answer.service import answer_knowledge_base
from opk_rag.answerability import AnswerabilityPolicy
from opk_rag.evaluation.fixtures import read_fixture
from opk_rag.search.models import EvidenceBundle, EvidenceItem, EvidenceSignals, SearchResponse, SearchResult

GENERATION_STABILITY_SCHEMA_VERSION = "generation-stability-result.v1"
GENERATION_FINGERPRINT_SCHEMA_VERSION = "generation-run-fingerprint.v1"
GENERATION_CONTRACT_VERSION = "answer-response-v3+answer-prompt-v4+evidence-context-v1"
VARIANCE_STAGE_PRIORITY = (
    "input_or_fixture_mismatch",
    "prompt_mismatch",
    "provider_configuration_mismatch",
    "provider_request_failure",
    "raw_generation_variance",
    "generation_parse_variance",
    "generation_contract_variance",
    "citation_variance",
    "grounding_variance",
    "final_action_variance",
    "unclassified_variance",
)


def run_generation_stability(
    *,
    dataset: list[dict[str, Any]],
    fixture_path: Path,
    split: str,
    run_count: int,
    config: AnswerGenerationConfig,
    sample_ids: set[str] | None = None,
    experiment_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if run_count < 1:
        raise ValueError("run_count must be >= 1.")
    fixture_rows = read_fixture(fixture_path)
    fixture_identity = fixture_file_identity(fixture_path)
    records = [row for row in dataset if row.get("split") == split]
    if sample_ids is not None:
        records = [row for row in records if row["id"] in sample_ids]
    fixture_by_id = {row["sample_id"]: row for row in fixture_rows}
    missing = [record["id"] for record in records if record["id"] not in fixture_by_id]
    if missing:
        raise ValueError(f"Fixture is missing dataset sample IDs: {missing}")

    experiment = experiment_id or f"task0028-{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    provider = OpenAICompatibleLocalChatProvider(config, api_key=os.environ.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
    rows: list[dict[str, Any]] = []
    for run_index in range(1, run_count + 1):
        for record in records:
            fixture = fixture_by_id[record["id"]]
            search_response = search_response_from_fixture(fixture)
            row = run_fixture_sample(
                record=record,
                fixture=fixture,
                fixture_identity=fixture_identity,
                experiment_id=experiment,
                run_index=run_index,
                provider=provider,
                config=config,
                search_response=search_response,
            )
            rows.append(row)
    summary = summarize_stability(rows, sample_count=len(records), run_count=run_count, config=config, fixture_path=fixture_path)
    return rows, summary


def run_fixture_sample(
    *,
    record: dict[str, Any],
    fixture: dict[str, Any],
    fixture_identity: dict[str, Any],
    experiment_id: str,
    run_index: int,
    provider: OpenAICompatibleLocalChatProvider,
    config: AnswerGenerationConfig,
    search_response: SearchResponse,
) -> dict[str, Any]:
    started = time.perf_counter()
    answerability = AnswerabilityPolicy(config.answerability).evaluate(
        question=search_response.query,
        evidence_bundle=search_response.evidence_bundle,
        search_response=search_response,
        prompt_injection_detected=False,
    )
    request_answerability = answerability if answerability.status == "partially_answerable" or answerability.reason_code == "false_premise" else None
    user_prompt = render_user_prompt(search_response.query, search_response.evidence_bundle, output_schema_version=config.output_schema_version, answerability=request_answerability)
    request = AnswerGenerationRequest(
        query=search_response.query,
        evidence_bundle=search_response.evidence_bundle,
        prompt_version=config.prompt_version,
        output_schema_version=config.output_schema_version,
        answerability=request_answerability,
    )
    provider_payload = build_chat_completion_payload(config, request)
    fingerprint = generation_fingerprint(
        experiment_id=experiment_id,
        run_index=run_index,
        sample_id=record["id"],
        split=str(record.get("split") or fixture.get("split") or ""),
        fixture=fixture,
        fixture_identity=fixture_identity,
        config=config,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        provider_payload=provider_payload,
    )
    system_error: str | None = None
    answer = None
    try:
        answer = answer_knowledge_base(search_response, provider=provider, config=config)
        answer_payload = answer_payload_for_stability(answer)
    except AnswerSystemError as exc:
        system_error = exc.code
        answer_payload = system_error_payload_for_stability(exc.code, search_response, provider_error_detail=exc.provider_error_detail)
    duration = time.perf_counter() - started
    parsed_output = answer_payload.get("model_decision") if isinstance(answer_payload.get("model_decision"), dict) else None
    raw_output = answer_payload.get("raw_generation_text")
    grounding = answer_payload.get("grounding") if isinstance(answer_payload.get("grounding"), dict) else {}
    citation_status = citation_status_from_answer(answer_payload)
    generation_status = generation_status_from_answer(answer_payload)
    final_action = final_action_from_answer(answer_payload)
    failure_stage = failure_stage_from_statuses(answer_payload, generation_status, citation_status, grounding)
    generation_reason = generation_reason_from_answer(answer_payload, generation_status)
    return {
        "schema_version": GENERATION_STABILITY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_index": run_index,
        "sample_id": record["id"],
        "split": record.get("split"),
        "fixture_identity": fixture_identity,
        "question_hash": fingerprint["question_hash"],
        "evidence_bundle_hash": fingerprint["evidence_bundle_hash"],
        "evidence_item_hashes": fingerprint["evidence_bundle_ordered_item_hashes"],
        "prompt_hash": fingerprint["complete_message_sequence_hash"],
        "system_prompt_hash": fingerprint["rendered_system_prompt_hash"],
        "user_prompt_hash": fingerprint["rendered_user_prompt_hash"],
        "message_sequence_hash": fingerprint["complete_message_sequence_hash"],
        "fingerprint": fingerprint,
        "provider_fingerprint": fingerprint["provider_fingerprint"],
        "generation_parameters": fingerprint["generation_parameters"],
        "provider_request_summary": provider_request_summary(provider_payload, answer_payload.get("request_attempts") or ()),
        "request_attempts": answer_payload.get("request_attempts") or [],
        "finish_reason": answer_payload.get("finish_reason"),
        "output_truncated": bool(answer_payload.get("output_truncated")),
        "empty_output": bool(answer_payload.get("empty_output")),
        "raw_output": raw_output,
        "raw_output_hash": stable_hash(raw_output) if raw_output is not None else None,
        "normalized_output_hash": stable_hash(normalize_output(raw_output)) if raw_output is not None else None,
        "parsed_output": parsed_output,
        "parsed_output_hash": stable_hash(parsed_output) if parsed_output is not None else None,
        "generation_status": generation_status,
        "generation_reason": generation_reason,
        "generation_abstention_reason": abstention_reason(answer_payload, generation_status),
        "citation_status": citation_status,
        "citation_reason": citation_reason_from_answer(answer_payload, citation_status),
        "grounding_status": str(grounding.get("reason_code") or grounding.get("status") or "unavailable"),
        "grounding_reason": grounding.get("reason"),
        "final_action": final_action,
        "failure_stage": failure_stage,
        "error_type": error_type_from_status(system_error, final_action, generation_status, citation_status, grounding),
        "system_error_code": system_error,
        "provider_error_reason_code": provider_error_reason(answer_payload),
        "provider_http_status": provider_http_status(answer_payload),
        "provider_sdk_error_type": provider_sdk_error_type(answer_payload),
        "provider_error_summary": provider_error_summary(answer_payload),
        "answer_payload": answer_payload,
        "duration": duration,
    }


def generation_fingerprint(
    *,
    experiment_id: str,
    run_index: int,
    sample_id: str,
    split: str,
    fixture: dict[str, Any],
    fixture_identity: dict[str, Any],
    config: AnswerGenerationConfig,
    system_prompt: str,
    user_prompt: str,
    provider_payload: dict[str, Any],
) -> dict[str, Any]:
    messages = provider_payload.get("messages") if isinstance(provider_payload.get("messages"), list) else []
    generation_parameters = generation_parameters_payload(config)
    provider_fingerprint = provider_fingerprint_payload(config)
    return {
        "schema_version": GENERATION_FINGERPRINT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_index": run_index,
        "sample_id": sample_id,
        "split": split,
        "fixture_schema_version": fixture.get("fixture_schema_version"),
        "fixture_file_identity": fixture_identity,
        "question_hash": stable_hash(fixture.get("question")),
        "evidence_bundle_hash": stable_hash(fixture.get("evidence_bundle") or []),
        "evidence_bundle_ordered_item_hashes": [stable_hash(item) for item in fixture.get("evidence_bundle") or []],
        "rendered_system_prompt_hash": stable_hash(system_prompt),
        "rendered_user_prompt_hash": stable_hash(user_prompt),
        "complete_message_sequence_hash": stable_hash(messages),
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "provider_type": config.provider_id,
        "provider_base_url_identity": safe_base_url_identity(config.base_url),
        "model_name": config.model_id,
        "model_revision": config.model_revision or "unknown",
        "tokenizer_revision": "unknown",
        "generation_parameters": generation_parameters,
        "retry_configuration": {"max_retries": config.max_retries, "timeout_seconds": config.timeout_seconds, "retry_backoff": "none"},
        "runtime_environment": runtime_environment(),
        "provider_fingerprint": provider_fingerprint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "request_payload_hash": stable_hash(redact_provider_payload(provider_payload)),
    }


def summarize_stability(rows: list[dict[str, Any]], *, sample_count: int, run_count: int, config: AnswerGenerationConfig, fixture_path: Path) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    per_sample = []
    for sample_id, sample_rows in sorted(by_sample.items()):
        sample_summary = per_sample_stability(sample_id, sample_rows, run_count)
        per_sample.append(sample_summary)
    fully_stable = [row for row in per_sample if row["fully_stable"]]
    per_run = []
    for run_index in range(1, run_count + 1):
        run_rows = [row for row in rows if row["run_index"] == run_index]
        per_run.append(
            {
                "run_index": run_index,
                "grounded_answers": sum(row["final_action"] == "answer" and row["grounding_status"] == "grounded" for row in run_rows),
                "abstains": sum(row["final_action"] == "abstain" for row in run_rows),
                "unsupported_answers": sum(row["error_type"] == "unsupported_answer" for row in run_rows),
                "infrastructure_failures": sum(row["error_type"] == "infrastructure_failure" for row in run_rows),
            }
        )
    return {
        "schema_version": "generation-stability-summary.v1",
        "fixture_path": fixture_path.as_posix(),
        "sample_count": sample_count,
        "run_count": run_count,
        "fully_stable_sample_count": len(fully_stable),
        "fully_stable_sample_rate": len(fully_stable) / sample_count if sample_count else None,
        "raw_exact_match_rate": mean(row["raw_output_agreement"] for row in per_sample),
        "normalized_match_rate": mean(row["normalized_output_agreement"] for row in per_sample),
        "generation_status_agreement": mean(row["generation_status_agreement"] for row in per_sample),
        "citation_status_agreement": mean(row["citation_status_agreement"] for row in per_sample),
        "grounding_status_agreement": mean(row["grounding_status_agreement"] for row in per_sample),
        "final_action_agreement": mean(row["final_action_agreement"] for row in per_sample),
        "per_run_grounded_answers": [row["grounded_answers"] for row in per_run],
        "per_run_abstains": [row["abstains"] for row in per_run],
        "per_run_unsupported_answers": [row["unsupported_answers"] for row in per_run],
        "per_run_infrastructure_failures": [row["infrastructure_failures"] for row in per_run],
        "grounded_answer_rate_variance": rate_variance(per_run, "grounded_answers"),
        "abstain_rate_variance": rate_variance(per_run, "abstains"),
        "unsupported_answer_rate_variance": rate_variance(per_run, "unsupported_answers"),
        "infrastructure_failure_rate_variance": rate_variance(per_run, "infrastructure_failures"),
        "first_variance_stage_distribution": count_values(row["first_variance_stage"] for row in per_sample if row["first_variance_stage"]),
        "provider_configuration_summary": provider_fingerprint_payload(config),
        "unsupported_provider_controls": unsupported_provider_controls(config),
        "known_limitations": [
            "OpenAI-compatible endpoint support for seed/top_k is provider-specific; unsupported controls are reported and are not assumed effective.",
            "temperature=0 is recorded as deterministic evaluation configuration but is not treated as a proof of byte-level deterministic model output.",
        ],
        "per_sample": per_sample,
    }


def per_sample_stability(sample_id: str, rows: list[dict[str, Any]], expected_runs: int) -> dict[str, Any]:
    def agreement(key: str) -> float:
        values = [stable_hash(row.get(key)) for row in rows]
        if len(values) != expected_runs or not values:
            return 0.0
        return max(values.count(value) for value in set(values)) / expected_runs

    unique_counts = {
        "unique_raw_output_count": unique_count(row.get("raw_output_hash") for row in rows),
        "unique_normalized_output_count": unique_count(row.get("normalized_output_hash") for row in rows),
        "unique_generation_status_count": unique_count(row.get("generation_status") for row in rows),
        "unique_citation_status_count": unique_count(row.get("citation_status") for row in rows),
        "unique_grounding_status_count": unique_count(row.get("grounding_status") for row in rows),
        "unique_final_action_count": unique_count(row.get("final_action") for row in rows),
    }
    first_stage = first_variance_stage(rows, expected_runs)
    return {
        "sample_id": sample_id,
        "run_count": len(rows),
        **unique_counts,
        "fully_stable": len(rows) == expected_runs and all(value == 1 for value in unique_counts.values()),
        "first_variance_stage": first_stage,
        "raw_output_agreement": agreement("raw_output_hash"),
        "normalized_output_agreement": agreement("normalized_output_hash"),
        "generation_status_agreement": agreement("generation_status"),
        "citation_status_agreement": agreement("citation_status"),
        "grounding_status_agreement": agreement("grounding_status"),
        "final_action_agreement": agreement("final_action"),
    }


def first_variance_stage(rows: list[dict[str, Any]], expected_runs: int) -> str | None:
    if len(rows) != expected_runs:
        return "unclassified_variance"
    checks = (
        ("input_or_fixture_mismatch", ("question_hash", "evidence_bundle_hash")),
        ("prompt_mismatch", ("system_prompt_hash", "user_prompt_hash", "message_sequence_hash")),
        ("provider_configuration_mismatch", ("provider_fingerprint", "generation_parameters")),
        ("provider_request_failure", ("system_error_code",)),
        ("raw_generation_variance", ("raw_output_hash",)),
        ("generation_parse_variance", ("parsed_output_hash",)),
        ("generation_contract_variance", ("generation_status", "generation_reason")),
        ("citation_variance", ("citation_status", "citation_reason")),
        ("grounding_variance", ("grounding_status", "grounding_reason")),
        ("final_action_variance", ("final_action",)),
    )
    for stage, keys in checks:
        for key in keys:
            if unique_count(stable_hash(row.get(key)) for row in rows) > 1:
                return stage
    return None


def search_response_from_fixture(row: dict[str, Any]) -> SearchResponse:
    retrieval_config = row.get("retrieval_configuration") or {}
    kb_id = UUID(str(retrieval_config.get("knowledge_base_id") or "00000000-0000-0000-0000-000000000000"))
    results = tuple(search_result_from_payload(payload) for payload in row.get("candidates") or [])
    bundle_items = tuple(evidence_item_from_payload(payload) for payload in row.get("evidence_bundle") or [])
    bundle = EvidenceBundle(
        query=row["question"],
        normalized_query=row["question"],
        knowledge_base_id=kb_id,
        context_token_budget=int(retrieval_config.get("context_token_budget") or 0),
        context_token_count=int(retrieval_config.get("context_token_count") or sum(item.context_token_count for item in bundle_items)),
        total_token_count=int(retrieval_config.get("context_token_count") or sum(item.context_token_count for item in bundle_items)),
        context_tokenizer_id=str(retrieval_config.get("context_tokenizer_id") or "unknown"),
        context_tokenizer_revision=retrieval_config.get("context_tokenizer_revision"),
        items=bundle_items,
    )
    return SearchResponse(
        query=row["question"],
        normalized_query=row["question"],
        knowledge_base_id=kb_id,
        model_id=str(row.get("embedding_model_identifier") or retrieval_config.get("embedding_model_id") or "unknown"),
        retrieval_mode=retrieval_config.get("retrieval_mode") or "vector",
        query_template_version=str(retrieval_config.get("query_template_version") or "unknown"),
        query_instruction="from frozen retrieval fixture",
        requested_top_k=int(retrieval_config.get("requested_top_k") or len(results)),
        candidate_k=int(retrieval_config.get("candidate_k") or len(results)),
        candidate_count=len(results),
        vector_candidate_count=int(retrieval_config.get("vector_candidate_count") or 0),
        bm25_candidate_count=int(retrieval_config.get("bm25_candidate_count") or 0),
        threshold_filtered_count=0,
        deduplicated_count=0,
        result_count=len(results),
        query_token_count=0,
        query_input_token_count=0,
        query_input_hash=stable_hash(row["question"]),
        lexical_query_terms=(),
        lexical_ready=False,
        retrieval_degraded=False,
        results=results,
        reranker_enabled=bool(retrieval_config.get("reranker_enabled")),
        final_top_k=int(retrieval_config.get("final_top_k") or len(results)),
        context_token_budget=bundle.context_token_budget,
        context_token_count=bundle.context_token_count,
        evidence_bundle=bundle,
        evidence_signals=evidence_signals(results, bundle),
    )


def search_result_from_payload(payload: dict[str, Any]) -> SearchResult:
    return SearchResult(
        rank=int(payload.get("rank") or 0),
        document_id=UUID(str(payload["document_id"])),
        chunk_id=UUID(str(payload["chunk_id"])),
        relative_path=str(payload.get("relative_path") or ""),
        heading_path=tuple(payload.get("heading_path") or ()),
        content=str(payload.get("content") or ""),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        similarity=float(payload.get("similarity") or 0.0),
        vector_similarity=payload.get("vector_similarity"),
        bm25_score=payload.get("bm25_score"),
        vector_rank=payload.get("vector_rank"),
        bm25_rank=payload.get("bm25_rank"),
        rrf_score=payload.get("rrf_score"),
        rerank_score=payload.get("rerank_score"),
        rerank_rank=payload.get("rerank_rank"),
        context_token_count=payload.get("context_token_count"),
        selected_for_context=bool(payload.get("selected_for_context")),
        context_rank=payload.get("context_rank"),
        retrieval_sources=tuple(payload.get("retrieval_sources") or ()),
        matched_terms=tuple(payload.get("matched_terms") or ()),
    )


def evidence_item_from_payload(payload: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        context_rank=int(payload.get("context_rank") or 0),
        result_rank=int(payload.get("result_rank") or payload.get("rank") or 0),
        chunk_id=UUID(str(payload["chunk_id"])),
        document_id=UUID(str(payload["document_id"])),
        relative_path=str(payload.get("relative_path") or ""),
        heading_path=tuple(payload.get("heading_path") or ()),
        content=str(payload.get("content") or ""),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        context_token_count=int(payload.get("context_token_count") or 0),
        reranker_pair_token_count=payload.get("reranker_pair_token_count"),
        reranker_original_pair_token_count=payload.get("reranker_original_pair_token_count"),
        reranker_input_truncated=bool(payload.get("reranker_input_truncated", False)),
        rerank_score=payload.get("rerank_score"),
        retrieval_sources=tuple(payload.get("retrieval_sources") or ()),
    )


def evidence_signals(results: tuple[SearchResult, ...], bundle: EvidenceBundle) -> EvidenceSignals:
    vector_scores = [row.vector_similarity for row in results if row.vector_similarity is not None]
    bm25_scores = [row.bm25_score for row in results if row.bm25_score is not None]
    rrf_scores = [row.rrf_score for row in results if row.rrf_score is not None]
    rerank_scores = [item.rerank_score for item in bundle.items if item.rerank_score is not None]
    return EvidenceSignals(
        top_reranker_score=rerank_scores[0] if rerank_scores else None,
        second_reranker_score=rerank_scores[1] if len(rerank_scores) > 1 else None,
        top1_top2_margin=(rerank_scores[0] - rerank_scores[1]) if len(rerank_scores) > 1 else None,
        max_vector_similarity=max(vector_scores) if vector_scores else None,
        max_bm25_score=max(bm25_scores) if bm25_scores else None,
        max_rrf_score=max(rrf_scores) if rrf_scores else None,
        dual_channel_candidate_count=sum({"vector", "bm25"}.issubset(set(row.retrieval_sources)) for row in results),
        selected_source_document_count=len({item.document_id for item in bundle.items}),
        selected_chunk_count=len(bundle.items),
        selected_context_token_count=bundle.context_token_count,
        query_term_coverage=0.0,
        exact_identifier_match=False,
        any_reranker_input_truncated=any(item.reranker_input_truncated for item in bundle.items),
        reranker_score_min=min(rerank_scores) if rerank_scores else None,
        reranker_score_max=max(rerank_scores) if rerank_scores else None,
        reranker_score_mean=(sum(rerank_scores) / len(rerank_scores)) if rerank_scores else None,
    )


def deterministic_config(config: AnswerGenerationConfig, *, seed: int | None = 0) -> AnswerGenerationConfig:
    return replace(config, temperature=0.0, top_p=1.0, seed=seed, max_retries=0)


def generation_parameters_payload(config: AnswerGenerationConfig) -> dict[str, Any]:
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": "not_applicable",
        "typical_p": "not_applicable",
        "repetition_penalty": config.repetition_penalty,
        "presence_penalty": "not_applicable",
        "frequency_penalty": "not_applicable",
        "seed": config.seed,
        "max_tokens": config.max_output_tokens,
        "stop_sequences": list(config.stop_sequences),
        "streaming": False,
        "response_format": config.response_format_type,
        "reasoning_or_thinking_mode": config.thinking_mode or "not_configured",
        "provider_parameter_policy": config.provider_parameter_policy,
    }


def provider_fingerprint_payload(config: AnswerGenerationConfig) -> dict[str, Any]:
    return {
        "provider_type": config.provider_id,
        "base_url_identity": safe_base_url_identity(config.base_url),
        "model_name": config.model_id,
        "model_revision": config.model_revision or "unknown",
        "tokenizer_revision": "unknown",
        "generation_parameters_hash": stable_hash(generation_parameters_payload(config)),
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "retry_backoff": "none",
        "provider_parameter_policy": config.provider_parameter_policy,
        "endpoint_type": "loopback" if safe_base_url_identity(config.base_url)["endpoint_type"] == "loopback" else "remote",
        "backend": "openai-compatible-chat-completions",
        "device": "provider-managed",
        "dtype": "provider-managed",
        "context_window": "unknown",
        "batch_parameters": "not_applicable",
        "quantization": "unknown",
    }


def unsupported_provider_controls(config: AnswerGenerationConfig) -> list[dict[str, str]]:
    unsupported = [
        ("model_revision", "not reported by generic OpenAI-compatible chat API" if config.model_revision is None else "configured by environment"),
        ("tokenizer_revision", "not reported by provider"),
        ("min_p", "not part of current provider payload"),
        ("typical_p", "not part of current provider payload"),
        ("presence_penalty", "not part of current provider payload"),
        ("frequency_penalty", "not part of current provider payload"),
        ("context_window", "not reported by provider"),
        ("device_dtype_backend_server_flags", "not introspectable through chat completions endpoint"),
    ]
    if config.provider_parameter_policy == "deepseek_v4":
        unsupported.extend(
            [
                ("seed", "unknown by official DeepSeek chat-completion docs; not sent under deepseek_v4 policy"),
                ("top_k", "unsupported by official DeepSeek chat-completion docs; not sent under deepseek_v4 policy"),
                ("repetition_penalty", "unsupported by official DeepSeek chat-completion docs; not sent under deepseek_v4 policy"),
                ("presence_penalty", "unsupported in DeepSeek thinking mode; not part of current provider payload"),
                ("frequency_penalty", "unsupported in DeepSeek thinking mode; not part of current provider payload"),
            ]
        )
    elif config.seed is None:
        unsupported.append(("seed", "not sent unless configured; provider support is unknown"))
    return [{"control": key, "status": value} for key, value in unsupported]


def answer_payload_for_stability(answer) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answerable": answer.answerable,
        "answer": answer.answer,
        "citations": [{"citation_id": citation.citation_id, "relative_path": citation.relative_path, "start_line": citation.start_line, "end_line": citation.end_line} for citation in answer.citations],
        "grounding": {
            "valid": answer.grounding.valid,
            "status": answer.grounding.status,
            "reason_code": answer.grounding.reason_code,
            "reason": answer.grounding.reason,
            "citation_coverage": answer.grounding.citation_coverage,
            "diagnostics": answer.grounding.diagnostics,
        },
        "unsupported_claims": list(answer.unsupported_claims),
        "answerability": {
            "status": answer.answerability.status,
            "answerable": answer.answerability.answerable,
            "reason_code": answer.answerability.reason_code,
            "confidence": answer.answerability.confidence,
            "diagnostics": answer.answerability.diagnostics,
        },
        "model_decision": answer.model_decision,
        "system_error": False,
        "refusal_reason_code": answer.refusal_reason_code,
        "provider_id": answer.provider_id,
        "model_id": answer.model_id,
        "model_revision": answer.model_revision,
        "generation_latency_ms": answer.generation_latency_ms,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "total_tokens": answer.total_tokens,
        "raw_generation_text": answer.raw_generation_text,
        "finish_reason": answer.finish_reason,
        "output_truncated": answer.output_truncated,
        "empty_output": answer.empty_output,
        "request_attempts": list(answer.request_attempts),
    }


def system_error_payload_for_stability(code: str, search_response: SearchResponse, *, provider_error_detail: dict | None = None) -> dict[str, Any]:
    attempts = provider_error_detail.get("attempts") if isinstance(provider_error_detail, dict) else []
    return {
        "status": "system_error",
        "answerable": False,
        "answer": "",
        "citations": [],
        "grounding": {"valid": False, "status": "error", "reason_code": "system_error", "reason": code},
        "unsupported_claims": [],
        "system_error": True,
        "system_error_code": code,
        "provider_error_detail": provider_error_detail,
        "refusal_reason_code": None,
        "raw_generation_text": None,
        "finish_reason": None,
        "output_truncated": False,
        "empty_output": False,
        "request_attempts": attempts if isinstance(attempts, list) else [],
    }


def final_action_from_answer(answer: dict[str, Any]) -> str:
    if answer.get("system_error"):
        return "system_error"
    return "answer" if answer.get("status") == "answered" and answer.get("answerable") is True else "abstain"


def generation_status_from_answer(answer: dict[str, Any]) -> str:
    if answer.get("system_error"):
        return "provider_failure"
    if answer.get("output_truncated"):
        return "truncation"
    if answer.get("empty_output"):
        return "empty_generation"
    decision = answer.get("model_decision") if isinstance(answer.get("model_decision"), dict) else {}
    if decision.get("decision") == "abstain":
        return "abstention"
    if answer.get("raw_generation_text") is not None and not isinstance(decision, dict):
        return "parse_failure"
    if answer.get("status") == "answered":
        return "success"
    if answer.get("generation_latency_ms") is not None:
        return "contract_failure"
    return "not_started"


def citation_status_from_answer(answer: dict[str, Any]) -> str:
    citations = answer.get("citations")
    grounding = answer.get("grounding") if isinstance(answer.get("grounding"), dict) else {}
    reason = grounding.get("reason_code")
    if answer.get("status") != "answered":
        return "not_evaluated"
    if not citations:
        return "missing_citation"
    if reason == "citation_set_mismatch":
        return "citation_set_mismatch"
    if reason in {"invalid_citation_format", "unknown_citation", "citation_out_of_range", "historical_citation_reference", "invalid_evidence_reference"}:
        return str(reason)
    return "citation_success"


def failure_stage_from_statuses(answer: dict[str, Any], generation_status: str, citation_status: str, grounding: dict[str, Any]) -> str:
    if answer.get("system_error"):
        return "infrastructure"
    if generation_status in {"abstention", "empty_generation", "parse_failure", "truncation", "contract_failure"}:
        return "generation"
    if citation_status not in {"citation_success", "not_evaluated"}:
        return "citation"
    if answer.get("status") == "refused" and answer.get("refusal_reason_code") in {"unsupported_claims", "ungrounded_answer"}:
        return "grounding"
    if answer.get("status") == "answered" and grounding.get("valid") is not True:
        return "grounding"
    return "none"


def error_type_from_status(system_error: str | None, final_action: str, generation_status: str, citation_status: str, grounding: dict[str, Any]) -> str:
    if system_error:
        return "infrastructure_failure"
    if generation_status in {"abstention", "empty_generation", "parse_failure", "truncation", "contract_failure"}:
        return "generation_abstention" if final_action == "abstain" else "generation_failure"
    if citation_status not in {"citation_success", "not_evaluated"}:
        return "citation_failure"
    if final_action == "answer" and grounding.get("valid") is not True:
        return "unsupported_answer"
    return "no_error"


def generation_reason_from_answer(answer: dict[str, Any], status: str) -> str | None:
    if status == "provider_failure":
        return str(provider_error_reason(answer) or answer.get("system_error_code") or "provider_failure")
    if status == "truncation":
        return "output_truncation"
    decision = answer.get("model_decision") if isinstance(answer.get("model_decision"), dict) else {}
    return str(answer.get("refusal_reason_code") or decision.get("reason") or status)


def provider_error_reason(answer: dict[str, Any]) -> str | None:
    detail = answer.get("provider_error_detail") if isinstance(answer.get("provider_error_detail"), dict) else {}
    value = detail.get("reason_code")
    return str(value) if value is not None else None


def provider_http_status(answer: dict[str, Any]) -> int | None:
    detail = answer.get("provider_error_detail") if isinstance(answer.get("provider_error_detail"), dict) else {}
    value = detail.get("http_status")
    return int(value) if value is not None else None


def provider_sdk_error_type(answer: dict[str, Any]) -> str | None:
    detail = answer.get("provider_error_detail") if isinstance(answer.get("provider_error_detail"), dict) else {}
    value = detail.get("sdk_error_type")
    return str(value) if value is not None else None


def provider_error_summary(answer: dict[str, Any]) -> str | None:
    detail = answer.get("provider_error_detail") if isinstance(answer.get("provider_error_detail"), dict) else {}
    value = detail.get("error_summary")
    return str(value) if value is not None else None


def abstention_reason(answer: dict[str, Any], generation_status: str) -> str | None:
    if generation_status == "truncation":
        return "output_truncation"
    if generation_status == "empty_generation":
        return "empty_generation"
    if generation_status == "parse_failure":
        return "parse_interpreted_as_abstention"
    decision = answer.get("model_decision") if isinstance(answer.get("model_decision"), dict) else {}
    if decision.get("decision") != "abstain":
        return None
    reason = str(answer.get("refusal_reason_code") or decision.get("reason") or "")
    mapping = {
        "model_abstained": "explicit_model_abstention",
        "answerable_generation_abstained": "explicit_model_abstention",
        "insufficient_evidence": "insufficient_evidence_claimed_by_model",
        "no_evidence": "insufficient_evidence_claimed_by_model",
        "partial_supported_scope_missing": "unsupported_scope_only",
    }
    return mapping.get(reason, "unknown_generation_abstention")


def citation_reason_from_answer(answer: dict[str, Any], citation_status: str) -> str | None:
    if citation_status == "not_evaluated":
        return "generation did not produce an answer"
    grounding = answer.get("grounding") if isinstance(answer.get("grounding"), dict) else {}
    return str(grounding.get("reason") or citation_status)


def fixture_file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": path.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def provider_request_summary(payload: dict[str, Any], attempts: Any) -> dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    return {
        "message_roles": [message.get("role") for message in messages if isinstance(message, dict)],
        "message_hashes": [stable_hash(message.get("content")) for message in messages if isinstance(message, dict)],
        "message_character_counts": [len(str(message.get("content") or "")) for message in messages if isinstance(message, dict)],
        "generation_parameters": {key: payload.get(key) for key in ("temperature", "top_p", "top_k", "seed", "max_tokens", "stop", "repetition_penalty") if key in payload},
        "response_format": payload.get("response_format", {}).get("type") if isinstance(payload.get("response_format"), dict) else None,
        "request_attempt_count": len(attempts) if isinstance(attempts, (list, tuple)) else 0,
        "final_attempt_used": attempts[-1] if isinstance(attempts, (list, tuple)) and attempts else None,
    }


def safe_base_url_identity(base_url: str) -> dict[str, str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "unknown"
    endpoint_type = "loopback" if host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.") else "remote"
    safe = f"{parsed.scheme}://{host}:{parsed.port or ''}{parsed.path}".rstrip(":")
    return {"endpoint_type": endpoint_type, "sha256": stable_hash(safe), "host_class": "loopback" if endpoint_type == "loopback" else "remote"}


def redact_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(stable_json(payload))
    for key in ("api_key", "authorization", "Authorization"):
        redacted.pop(key, None)
    return redacted


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("opk-rag", "psycopg", "sentence-transformers", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": "provider-managed",
        "dtype": "provider-managed",
        "backend": "openai-compatible-chat-completions",
    }


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {
        "revision": run(["git", "rev-parse", "HEAD"]) or "unknown",
        "dirty": bool(run(["git", "status", "--short"])),
    }


def compare_legacy_runs(run_a: list[dict[str, Any]], run_b: list[dict[str, Any]]) -> dict[str, Any]:
    by_a = {row["id"]: row for row in run_a}
    by_b = {row["id"]: row for row in run_b}
    changed = []
    for sample_id in sorted(set(by_a) & set(by_b)):
        left = by_a[sample_id]
        right = by_b[sample_id]
        fields = {
            "generation_status_changed": left.get("generation_status") != right.get("generation_status"),
            "citation_status_changed": left.get("citation_status") != right.get("citation_status"),
            "grounding_status_changed": left.get("grounding_status") != right.get("grounding_status"),
            "final_action_changed": left.get("final_action") != right.get("final_action"),
            "error_type_changed": left.get("error_type") != right.get("error_type"),
        }
        if any(fields.values()):
            changed.append(
                {
                    "sample_id": sample_id,
                    **fields,
                    "evidence_confirmed": "not recoverable from legacy result",
                    "prompt_confirmed": "not recoverable from legacy result",
                    "provider_configuration_confirmed": "not recoverable from legacy result",
                    "raw_output_changed": "not recoverable from legacy result",
                    "first_recoverable_variance_stage": legacy_first_stage(fields),
                    "run_a": legacy_stage_payload(left),
                    "run_b": legacy_stage_payload(right),
                }
            )
    return {
        "sample_count_a": len(run_a),
        "sample_count_b": len(run_b),
        "common_sample_count": len(set(by_a) & set(by_b)),
        "changed_count": len(changed),
        "changed": changed,
    }


def legacy_first_stage(fields: dict[str, bool]) -> str:
    if fields["generation_status_changed"]:
        return "generation_parse_or_raw_generation_variance"
    if fields["citation_status_changed"]:
        return "citation_variance"
    if fields["grounding_status_changed"]:
        return "grounding_variance"
    if fields["final_action_changed"]:
        return "final_action_variance"
    return "unclassified_variance"


def legacy_stage_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("generation_status", "citation_status", "grounding_status", "final_action", "failure_stage", "error_type", "provider_id", "model_id", "model_revision", "prompt_version", "output_schema_version")}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def normalize_output(value: str | None) -> str | None:
    if value is None:
        return None
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n"))


def unique_count(values) -> int:
    return len({stable_json(value) for value in values})


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def mean(values) -> float | None:
    data = [float(value) for value in values if value is not None]
    return sum(data) / len(data) if data else None


def rate_variance(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [int(row[key]) for row in rows]
    if not values:
        return {"mean": None, "min": None, "max": None, "range": None}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "range": max(values) - min(values)}
