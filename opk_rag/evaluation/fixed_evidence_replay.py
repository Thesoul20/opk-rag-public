from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any
from uuid import UUID

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.evidence_context import build_evidence_context
from opk_rag.answer.models import AnswerGenerationRequest, AnswerGeneratorProvider, AnswerSystemError
from opk_rag.answer.prompt import EVIDENCE_SERIALIZATION_VERSION, SYSTEM_PROMPT, render_user_prompt
from opk_rag.answer.provider import build_chat_completion_payload
from opk_rag.answer.service import answer_knowledge_base_with_decision
from opk_rag.answerability import AnswerabilityDecision
from opk_rag.evaluation.generation_stability import (
    answer_payload_for_stability,
    citation_status_from_answer,
    error_type_from_status,
    generation_parameters_payload,
    generation_reason_from_answer,
    generation_status_from_answer,
    normalize_output,
    provider_fingerprint_payload,
    provider_request_summary,
    stable_hash,
    system_error_payload_for_stability,
)
from opk_rag.search.models import EvidenceBundle, EvidenceItem, EvidenceSignals, SearchResponse, SearchResult

FIXED_REPLAY_INPUT_SCHEMA_VERSION = "task0044-fixed-evidence-input.v1"
FIXED_REPLAY_RESULT_SCHEMA_VERSION = "task0044-fixed-evidence-run.v1"
FIXED_REPLAY_REPORT_SCHEMA_VERSION = "task0044-fixed-evidence-report.v1"
FROZEN_RECONSTRUCTION_SOURCE = "task0044_frozen_reconstruction"

PRIVATE_KEYS = {
    "question",
    "selected_evidence",
    "evidence_bundle",
    "generation_context",
    "rendered_prompt",
    "supported_scope",
    "unsupported_scope",
    "premise_correction",
    "raw_generation_text",
    "raw_output",
    "answer_text",
    "private_answer_payload",
    "raw_provider_output",
}


@dataclass(frozen=True)
class FixedReplayInput:
    sample_id: str
    question: str
    expected_answerability: str
    expected_final_action: str
    answerability_decision: AnswerabilityDecision
    search_response: SearchResponse
    prompt_version: str
    response_contract_version: str
    evidence_context_version: str
    provider_model: str
    provider_parameters: dict[str, Any]
    input_hashes: dict[str, str]
    source: str


def load_fixed_replay_inputs(path: Path) -> list[FixedReplayInput]:
    rows = read_jsonl(path)
    return [fixed_replay_input_from_row(row) for row in rows]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixed_replay_input_from_row(row: dict[str, Any]) -> FixedReplayInput:
    sample_id = _required_str(row, "sample_id")
    question = _required_str(row, "question")
    evidence_payload = row.get("selected_evidence") or row.get("evidence_bundle")
    if not isinstance(evidence_payload, list):
        raise ValueError(f"{sample_id}: selected_evidence must be a list")
    answerability = answerability_decision_from_payload(row.get("answerability_decision") or {})
    selected_citation_ids = tuple(str(value) for value in row.get("selected_citation_ids") or _citation_ids_for_evidence(evidence_payload))
    search_response = search_response_from_fixed_input(row, evidence_payload)
    prompt_version = str(row.get("prompt_version") or "answer-prompt-v4")
    response_contract_version = str(row.get("response_contract_version") or row.get("output_schema_version") or "answer-response-v3")
    evidence_context_version = str(row.get("evidence_context_version") or EVIDENCE_SERIALIZATION_VERSION)
    generation_context = row.get("generation_context")
    if generation_context is None:
        generation_context = build_evidence_context(
            question=question,
            bundle=search_response.evidence_bundle,
            answerability=answerability if answerability.status == "partially_answerable" or answerability.reason_code == "false_premise" else None,
            output_schema_version=response_contract_version,
        )
    input_hashes = input_hash_contract(
        question=question,
        answerability_decision=row.get("answerability_decision") or answerability_payload(answerability),
        evidence_payload=evidence_payload,
        selected_citation_ids=selected_citation_ids,
        generation_context=generation_context,
        prompt_version=prompt_version,
        response_contract_version=response_contract_version,
        evidence_context_version=evidence_context_version,
        provider_parameters=row.get("provider_parameters") or {},
        search_response=search_response,
        answerability=answerability,
    )
    supplied_hashes = row.get("input_hashes") if isinstance(row.get("input_hashes"), dict) else {}
    mismatches = {key: {"expected": supplied_hashes[key], "actual": value} for key, value in input_hashes.items() if key in supplied_hashes and supplied_hashes[key] != value}
    if mismatches:
        raise ValueError(f"{sample_id}: input hash mismatch: {mismatches}")
    return FixedReplayInput(
        sample_id=sample_id,
        question=question,
        expected_answerability=_required_str(row, "expected_answerability"),
        expected_final_action=_required_str(row, "expected_final_action"),
        answerability_decision=answerability,
        search_response=search_response,
        prompt_version=prompt_version,
        response_contract_version=response_contract_version,
        evidence_context_version=evidence_context_version,
        provider_model=str(row.get("provider_model") or ""),
        provider_parameters=dict(row.get("provider_parameters") or {}),
        input_hashes=input_hashes,
        source=str(row.get("source") or FROZEN_RECONSTRUCTION_SOURCE),
    )


def input_hash_contract(
    *,
    question: str,
    answerability_decision: dict[str, Any],
    evidence_payload: list[dict[str, Any]],
    selected_citation_ids: tuple[str, ...],
    generation_context: Any,
    prompt_version: str,
    response_contract_version: str,
    evidence_context_version: str,
    provider_parameters: dict[str, Any],
    search_response: SearchResponse,
    answerability: AnswerabilityDecision,
) -> dict[str, str]:
    request_answerability = answerability if answerability.status == "partially_answerable" or answerability.reason_code == "false_premise" else None
    request = AnswerGenerationRequest(
        query=question,
        evidence_bundle=search_response.evidence_bundle,
        prompt_version=prompt_version,
        output_schema_version=response_contract_version,
        answerability=request_answerability,
    )
    rendered_prompt = render_user_prompt(question, search_response.evidence_bundle, output_schema_version=response_contract_version, answerability=request_answerability)
    return {
        "question_hash": stable_hash(question),
        "answerability_decision_hash": stable_hash(answerability_decision),
        "evidence_bundle_hash": stable_hash(evidence_payload),
        "evidence_order_hash": stable_hash([item.get("chunk_id") for item in evidence_payload]),
        "generation_context_hash": stable_hash(generation_context),
        "rendered_prompt_hash": stable_hash({"system": SYSTEM_PROMPT, "user": rendered_prompt}),
        "provider_request_contract_hash": stable_hash(
            {
                "prompt_version": prompt_version,
                "response_contract_version": response_contract_version,
                "evidence_context_version": evidence_context_version,
                "selected_citation_ids": list(selected_citation_ids),
                "provider_parameters": provider_parameters,
                "request_shape": {
                    "query_hash": stable_hash(request.query),
                    "evidence_bundle_hash": stable_hash(evidence_payload),
                    "answerability_hash": stable_hash(answerability_payload(answerability)),
                },
            }
        ),
    }


def run_fixed_replay(
    *,
    inputs: list[FixedReplayInput],
    output_path: Path,
    provider: AnswerGeneratorProvider,
    config: AnswerGenerationConfig,
    runs: int,
    resume: bool = False,
    overwrite: bool = False,
    provider_parameter_policy: str | None = None,
) -> list[dict[str, Any]]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if output_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"Refusing to overwrite existing output without --resume or --overwrite: {output_path}")
    if overwrite and output_path.exists():
        output_path.unlink()
    existing = read_jsonl(output_path) if resume and output_path.exists() else []
    completed = {(str(row.get("sample_id")), int(row.get("run_index"))) for row in existing if row.get("sample_id") and row.get("run_index")}
    rows = list(existing)
    for sample in sorted(inputs, key=lambda item: item.sample_id):
        for run_index in range(1, runs + 1):
            if resume and (sample.sample_id, run_index) in completed:
                continue
            row = run_fixed_replay_sample(
                sample=sample,
                run_index=run_index,
                provider=provider,
                config=config,
                provider_parameter_policy=provider_parameter_policy or config.provider_parameter_policy,
            )
            append_jsonl(output_path, row)
            rows.append(row)
    return rows


def run_fixed_replay_sample(
    *,
    sample: FixedReplayInput,
    run_index: int,
    provider: AnswerGeneratorProvider,
    config: AnswerGenerationConfig,
    provider_parameter_policy: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_answerability = sample.answerability_decision if sample.answerability_decision.status == "partially_answerable" or sample.answerability_decision.reason_code == "false_premise" else None
    request = AnswerGenerationRequest(
        query=sample.question,
        evidence_bundle=sample.search_response.evidence_bundle,
        prompt_version=sample.prompt_version,
        output_schema_version=sample.response_contract_version,
        answerability=request_answerability,
    )
    provider_payload = build_chat_completion_payload(config, request)
    provider_request_hash = stable_hash(provider_payload)
    system_error: str | None = None
    try:
        answer = answer_knowledge_base_with_decision(sample.search_response, answerability=sample.answerability_decision, provider=provider, config=config)
        answer_payload = answer_payload_for_stability(answer)
    except AnswerSystemError as exc:
        system_error = exc.code
        answer_payload = system_error_payload_for_stability(exc.code, sample.search_response, provider_error_detail=exc.provider_error_detail)
    raw_output = answer_payload.get("raw_generation_text")
    normalized_output = normalize_output(raw_output) if isinstance(raw_output, str) else None
    grounding = answer_payload.get("grounding") if isinstance(answer_payload.get("grounding"), dict) else {}
    generation_status = generation_status_from_answer(answer_payload)
    citation_status = citation_status_from_answer(answer_payload)
    final_action = final_action_from_answer(answer_payload)
    primary_outcome = primary_outcome_from_answer(answer_payload, generation_status, citation_status, grounding, final_action)
    citations = citation_set_from_answer(answer_payload)
    model_decision = answer_payload.get("model_decision") if isinstance(answer_payload.get("model_decision"), dict) else {}
    contract_repairs = list(model_decision.get("contract_repairs") or ())
    row = {
        "schema_version": FIXED_REPLAY_RESULT_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "run_index": run_index,
        "input_hashes": sample.input_hashes,
        "provider_request_hash": provider_request_hash,
        "provider_parameter_policy": provider_parameter_policy,
        "provider_parameters": generation_parameters_payload(config),
        "provider_fingerprint": provider_fingerprint_payload(config),
        "provider_request_summary": provider_request_summary(provider_payload, answer_payload.get("request_attempts") or []),
        "raw_output_hash": stable_hash(raw_output) if raw_output is not None else None,
        "normalized_output_hash": stable_hash(normalized_output) if normalized_output is not None else None,
        "answer_text_hash": stable_hash(answer_payload.get("answer")) if answer_payload.get("answer") else None,
        "citation_set": citations,
        "citation_set_hash": stable_hash(sorted(citations)),
        "parse_status": "valid_json" if isinstance(model_decision, dict) and model_decision else ("not_started" if system_error else "invalid_model_output"),
        "contract_repairs": contract_repairs,
        "generation_status": generation_status,
        "generation_reason": generation_reason_from_answer(answer_payload, generation_status),
        "citation_status": citation_status,
        "grounding_status": str(grounding.get("reason_code") or grounding.get("status") or "unavailable"),
        "unsupported_claim_status": "present" if answer_payload.get("unsupported_claims") else "absent",
        "unsupported_claim_category": classify_unsupported_claim(answer_payload),
        "final_action": final_action,
        "primary_outcome": primary_outcome,
        "infrastructure_failure": bool(system_error),
        "system_error_code": system_error,
        "request_attempts": answer_payload.get("request_attempts") or [],
        "raw_private": {
            "raw_generation_text": raw_output,
            "answer": answer_payload.get("answer"),
            "model_decision": model_decision,
            "unsupported_claims": answer_payload.get("unsupported_claims") or [],
        },
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }
    return row


def summarize_fixed_replay(rows: list[dict[str, Any]], *, expected_runs: int | None = None) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    samples = []
    for sample_id, sample_rows in sorted(by_sample.items()):
        samples.append(summarize_sample(sample_id, sample_rows, expected_runs=expected_runs))
    valid_rows = [row for row in rows if not row.get("infrastructure_failure")]
    return {
        "schema_version": FIXED_REPLAY_REPORT_SCHEMA_VERSION,
        "sample_count": len(by_sample),
        "provider_call_count": len(rows),
        "valid_run_count": len(valid_rows),
        "infrastructure_failure_count": len(rows) - len(valid_rows),
        "final_action_distribution": count_field(valid_rows, "final_action"),
        "primary_outcome_distribution": count_field(valid_rows, "primary_outcome"),
        "final_action_stability_rate_mean": _mean(row["final_action_stability_rate"] for row in samples),
        "primary_outcome_stability_rate_mean": _mean(row["primary_outcome_stability_rate"] for row in samples),
        "safe_outcome_rate": _rate(sum(row.get("primary_outcome") in {"safe_full", "safe_partial", "correct_abstention"} for row in valid_rows), len(valid_rows)),
        "unsupported_claim_rate": _rate(sum(row.get("unsupported_claim_status") == "present" for row in valid_rows), len(valid_rows)),
        "explicit_abstention_rate": _rate(sum(row.get("primary_outcome") == "explicit_model_abstention" for row in valid_rows), len(valid_rows)),
        "contract_following_rate": _rate(sum(row.get("parse_status") == "valid_json" for row in valid_rows), len(valid_rows)),
        "invalid_model_output_rate": _rate(sum(row.get("system_error_code") == "invalid_model_output" or row.get("parse_status") == "invalid_model_output" for row in rows), len(rows)),
        "validator_determinism": validator_determinism(rows),
        "high_variance_sample_ids": [row["sample_id"] for row in samples if row["high_variance"]],
        "per_sample": samples,
    }


def summarize_sample(sample_id: str, rows: list[dict[str, Any]], *, expected_runs: int | None = None) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("infrastructure_failure")]
    final_counts = count_field(valid, "final_action")
    outcome_counts = count_field(valid, "primary_outcome")
    citation_sets = [tuple(row.get("citation_set") or ()) for row in valid]
    unsupported_runs = [row for row in valid if row.get("unsupported_claim_status") == "present"]
    final_rate = _most_common_rate(final_counts, len(valid))
    outcome_rate = _most_common_rate(outcome_counts, len(valid))
    return {
        "sample_id": sample_id,
        "run_count": len(rows),
        "expected_run_count": expected_runs,
        "valid_run_count": len(valid),
        "infrastructure_failure_count": len(rows) - len(valid),
        "final_action_distribution": final_counts,
        "primary_outcome_distribution": outcome_counts,
        "final_action_stability_rate": final_rate,
        "primary_outcome_stability_rate": outcome_rate,
        "safe_outcome_rate": _rate(sum(row.get("primary_outcome") in {"safe_full", "safe_partial", "correct_abstention"} for row in valid), len(valid)),
        "unsupported_claim_rate": _rate(len(unsupported_runs), len(valid)),
        "citation_count_distribution": dict(sorted(Counter(len(value) for value in citation_sets).items())),
        "unique_citation_sets": len({stable_hash(sorted(value)) for value in citation_sets}),
        "citation_set_stability": average_pairwise_jaccard(citation_sets),
        "citation_variance_class": citation_variance_class(citation_sets, valid),
        "validator_nondeterminism": bool(validator_determinism(rows)["defect_groups"]),
        "primary_variance_category": primary_variance_category(valid, rows),
        "high_variance": high_variance(valid),
    }


def validator_determinism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("input_hashes", {}).get("provider_request_contract_hash"), row.get("raw_output_hash"), row.get("normalized_output_hash"))
        grouped.setdefault(key, []).append(row)
    defects = []
    checked = 0
    for key, group in grouped.items():
        if len(group) < 2 or key[1] is None or key[2] is None:
            continue
        checked += 1
        statuses = {
            stable_hash(
                {
                    "parse_status": row.get("parse_status"),
                    "contract_repairs": row.get("contract_repairs"),
                    "citation_status": row.get("citation_status"),
                    "grounding_status": row.get("grounding_status"),
                    "unsupported_claim_status": row.get("unsupported_claim_status"),
                    "final_action": row.get("final_action"),
                    "primary_outcome": row.get("primary_outcome"),
                }
            )
            for row in group
        }
        if len(statuses) > 1:
            defects.append({"sample_ids": sorted({str(row.get("sample_id")) for row in group}), "raw_output_hash": key[1], "normalized_output_hash": key[2]})
    return {"checked_identical_input_output_groups": checked, "defect_groups": defects, "deterministic": not defects}


def build_public_report(rows: list[dict[str, Any]], *, expected_runs: int | None = None, parameter_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    report = summarize_fixed_replay(rows, expected_runs=expected_runs)
    report["parameter_audit"] = parameter_audit or {}
    report["privacy"] = {
        "contains_question_text": False,
        "contains_evidence_text": False,
        "contains_scope_text": False,
        "contains_provider_raw_output": False,
        "private_payload_removed": True,
    }
    report["per_sample"] = [{key: value for key, value in row.items() if key not in PRIVATE_KEYS} for row in report["per_sample"]]
    return report


def assert_public_report_is_private(payload: Any) -> None:
    if isinstance(payload, dict):
        forbidden = PRIVATE_KEYS & set(payload)
        if forbidden:
            raise ValueError(f"public report contains private keys: {sorted(forbidden)}")
        for value in payload.values():
            assert_public_report_is_private(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_public_report_is_private(value)


def answerability_decision_from_payload(payload: dict[str, Any]) -> AnswerabilityDecision:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    return AnswerabilityDecision(
        status=str(payload.get("status") or "answerable"),
        reason_code=str(payload.get("reason_code") or "answerable"),
        reason=str(payload.get("reason") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        evidence_chunk_ids=tuple(str(value) for value in payload.get("evidence_chunk_ids") or ()),
        evidence_score=payload.get("evidence_score") if isinstance(payload.get("evidence_score"), float) or payload.get("evidence_score") is None else None,
        evidence_count=int(payload.get("evidence_count") or 0),
        considered_evidence_count=int(payload.get("considered_evidence_count") or payload.get("evidence_count") or 0),
        diagnostics=diagnostics,
    )


def answerability_payload(decision: AnswerabilityDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "evidence_chunk_ids": list(decision.evidence_chunk_ids),
        "evidence_score": decision.evidence_score,
        "evidence_count": decision.evidence_count,
        "considered_evidence_count": decision.considered_evidence_count,
        "diagnostics": decision.diagnostics,
    }


def search_response_from_fixed_input(row: dict[str, Any], evidence_payload: list[dict[str, Any]]) -> SearchResponse:
    question = _required_str(row, "question")
    retrieval_config = row.get("retrieval_configuration") if isinstance(row.get("retrieval_configuration"), dict) else {}
    kb_id = UUID(str(retrieval_config.get("knowledge_base_id") or row.get("knowledge_base_id") or "00000000-0000-0000-0000-000000000000"))
    results = tuple(search_result_from_evidence_payload(payload, index) for index, payload in enumerate(evidence_payload, start=1))
    bundle_items = tuple(evidence_item_from_fixed_payload(payload, index) for index, payload in enumerate(evidence_payload, start=1))
    context_count = sum(item.context_token_count for item in bundle_items)
    bundle = EvidenceBundle(
        query=question,
        normalized_query=question,
        knowledge_base_id=kb_id,
        context_token_budget=int(retrieval_config.get("context_token_budget") or context_count),
        context_token_count=int(retrieval_config.get("context_token_count") or context_count),
        total_token_count=int(retrieval_config.get("total_token_count") or context_count),
        context_tokenizer_id=str(retrieval_config.get("context_tokenizer_id") or "unknown"),
        context_tokenizer_revision=retrieval_config.get("context_tokenizer_revision"),
        items=bundle_items,
    )
    return SearchResponse(
        query=question,
        normalized_query=question,
        knowledge_base_id=kb_id,
        model_id=str(row.get("embedding_model_identifier") or retrieval_config.get("embedding_model_id") or "unknown"),
        retrieval_mode=str(retrieval_config.get("retrieval_mode") or "hybrid"),
        query_template_version=str(retrieval_config.get("query_template_version") or "frozen"),
        query_instruction="from fixed evidence replay input",
        requested_top_k=int(retrieval_config.get("requested_top_k") or len(results)),
        candidate_k=int(retrieval_config.get("candidate_k") or len(results)),
        candidate_count=len(results),
        vector_candidate_count=int(retrieval_config.get("vector_candidate_count") or 0),
        bm25_candidate_count=int(retrieval_config.get("bm25_candidate_count") or 0),
        threshold_filtered_count=int(retrieval_config.get("threshold_filtered_count") or 0),
        deduplicated_count=int(retrieval_config.get("deduplicated_count") or len(results)),
        result_count=len(results),
        query_token_count=int(retrieval_config.get("query_token_count") or 0),
        query_input_token_count=int(retrieval_config.get("query_input_token_count") or 0),
        query_input_hash=stable_hash(question),
        lexical_query_terms=tuple(retrieval_config.get("lexical_query_terms") or ()),
        lexical_ready=bool(retrieval_config.get("lexical_ready", False)),
        retrieval_degraded=bool(retrieval_config.get("retrieval_degraded", False)),
        results=results,
        reranker_enabled=bool(retrieval_config.get("reranker_enabled", False)),
        final_top_k=int(retrieval_config.get("final_top_k") or len(results)),
        context_token_budget=bundle.context_token_budget,
        context_token_count=bundle.context_token_count,
        evidence_bundle=bundle,
        evidence_signals=evidence_signals(results, bundle),
    )


def evidence_item_from_fixed_payload(payload: dict[str, Any], index: int) -> EvidenceItem:
    return EvidenceItem(
        context_rank=int(payload.get("context_rank") or index),
        result_rank=int(payload.get("result_rank") or payload.get("rank") or index),
        chunk_id=UUID(str(payload["chunk_id"])),
        document_id=UUID(str(payload["document_id"])),
        relative_path=str(payload.get("relative_path") or ""),
        heading_path=tuple(payload.get("heading_path") or ()),
        content=str(payload.get("content") or ""),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        context_token_count=int(payload.get("context_token_count") or len(str(payload.get("content") or "").split())),
        reranker_pair_token_count=payload.get("reranker_pair_token_count"),
        reranker_original_pair_token_count=payload.get("reranker_original_pair_token_count"),
        reranker_input_truncated=bool(payload.get("reranker_input_truncated", False)),
        rerank_score=payload.get("rerank_score"),
        retrieval_sources=tuple(payload.get("retrieval_sources") or ()),
    )


def search_result_from_evidence_payload(payload: dict[str, Any], index: int) -> SearchResult:
    return SearchResult(
        rank=int(payload.get("rank") or index),
        document_id=UUID(str(payload["document_id"])),
        chunk_id=UUID(str(payload["chunk_id"])),
        relative_path=str(payload.get("relative_path") or ""),
        heading_path=tuple(payload.get("heading_path") or ()),
        content=str(payload.get("content") or ""),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        similarity=float(payload.get("similarity") or payload.get("vector_similarity") or 0.0),
        vector_similarity=payload.get("vector_similarity"),
        bm25_score=payload.get("bm25_score"),
        vector_rank=payload.get("vector_rank"),
        bm25_rank=payload.get("bm25_rank"),
        rrf_score=payload.get("rrf_score"),
        rerank_score=payload.get("rerank_score"),
        rerank_rank=payload.get("rerank_rank"),
        context_token_count=payload.get("context_token_count"),
        selected_for_context=True,
        context_rank=int(payload.get("context_rank") or index),
        retrieval_sources=tuple(payload.get("retrieval_sources") or ()),
    )


def evidence_signals(results: tuple[SearchResult, ...], bundle: EvidenceBundle) -> EvidenceSignals:
    rerank_scores = [item.rerank_score for item in bundle.items if item.rerank_score is not None]
    vector_scores = [row.vector_similarity for row in results if row.vector_similarity is not None]
    bm25_scores = [row.bm25_score for row in results if row.bm25_score is not None]
    rrf_scores = [row.rrf_score for row in results if row.rrf_score is not None]
    return EvidenceSignals(
        top_reranker_score=rerank_scores[0] if rerank_scores else None,
        second_reranker_score=rerank_scores[1] if len(rerank_scores) > 1 else None,
        top1_top2_margin=rerank_scores[0] - rerank_scores[1] if len(rerank_scores) > 1 else None,
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
        reranker_score_mean=sum(rerank_scores) / len(rerank_scores) if rerank_scores else None,
    )


def final_action_from_answer(answer: dict[str, Any]) -> str:
    if answer.get("system_error"):
        return "system_error"
    if answer.get("status") != "answered":
        return "abstain"
    decision = answer.get("model_decision") if isinstance(answer.get("model_decision"), dict) else {}
    if decision.get("answerability_status") == "partially_answerable":
        return "partial_answer"
    return "answer"


def primary_outcome_from_answer(answer: dict[str, Any], generation_status: str, citation_status: str, grounding: dict[str, Any], final_action: str) -> str:
    if answer.get("system_error"):
        return "infrastructure_failure"
    if generation_status == "provider_failure":
        return "infrastructure_failure"
    if answer.get("unsupported_claims"):
        return "unsupported_claim_rejection"
    if answer.get("refusal_reason_code") == "ungrounded_answer":
        return "grounding_rejection"
    if answer.get("refusal_reason_code") == "invalid_citations":
        return "citation_rejection"
    if generation_status in {"empty_generation", "parse_failure", "truncation", "contract_failure"}:
        return "invalid_model_output"
    if citation_status not in {"citation_success", "not_evaluated"}:
        return "citation_rejection"
    if final_action == "abstain":
        reason = str(answer.get("refusal_reason_code") or "")
        if reason in {"model_abstained", "answerable_generation_abstained", "partial_supported_scope_missing"}:
            return "explicit_model_abstention"
        return "correct_abstention"
    if grounding.get("valid") is not True:
        return "grounding_rejection"
    return "safe_partial" if final_action == "partial_answer" else "safe_full"


def citation_set_from_answer(answer: dict[str, Any]) -> list[str]:
    citations = answer.get("citations")
    if isinstance(citations, list) and citations and isinstance(citations[0], dict):
        return sorted({str(item.get("citation_id")) for item in citations if item.get("citation_id")})
    decision = answer.get("model_decision") if isinstance(answer.get("model_decision"), dict) else {}
    raw = decision.get("citations") if isinstance(decision.get("citations"), list) else []
    return sorted({str(value) for value in raw})


def average_pairwise_jaccard(citation_sets: list[tuple[str, ...]]) -> float | None:
    if len(citation_sets) < 2:
        return None
    scores = []
    for left_index, left in enumerate(citation_sets):
        for right in citation_sets[left_index + 1 :]:
            scores.append(jaccard(set(left), set(right)))
    return sum(scores) / len(scores) if scores else None


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def citation_variance_class(citation_sets: list[tuple[str, ...]], rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("citation_status")) for row in rows}
    if any("unknown" in status for status in statuses):
        return "unknown_citation_variance"
    if any("missing" in status for status in statuses):
        return "missing_citation_variance"
    unique = {tuple(sorted(value)) for value in citation_sets}
    if len(unique) <= 1:
        return "stable_citations"
    if len({len(value) for value in unique}) > 1:
        return "citation_subset_variance"
    return "citation_order_only_variance" if len({stable_hash(sorted(value)) for value in citation_sets}) == 1 else "citation_set_variance"


def high_variance(valid_rows: list[dict[str, Any]]) -> bool:
    if len({row.get("final_action") for row in valid_rows}) > 1:
        return True
    if len({row.get("primary_outcome") for row in valid_rows}) > 1:
        return True
    if len({stable_hash(sorted(row.get("citation_set") or ())) for row in valid_rows}) > 1:
        return True
    unsupported = {row.get("unsupported_claim_status") for row in valid_rows}
    if {"present", "absent"}.issubset(unsupported):
        return True
    return False


def primary_variance_category(valid_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> str:
    if not valid_rows and all(row.get("infrastructure_failure") for row in all_rows):
        return "infrastructure_variance"
    if len({row.get("primary_outcome") for row in valid_rows}) <= 1:
        outcome = next(iter({row.get("primary_outcome") for row in valid_rows}), "other")
        return {
            "safe_full": "stable_safe_full",
            "safe_partial": "stable_safe_partial",
            "correct_abstention": "stable_correct_abstention",
            "explicit_model_abstention": "stable_explicit_model_abstention",
            "unsupported_claim_rejection": "stable_unsupported_claim",
            "grounding_rejection": "stable_grounding_rejection",
        }.get(str(outcome), "other")
    outcomes = {row.get("primary_outcome") for row in valid_rows}
    if "unsupported_claim_rejection" in outcomes:
        return "intermittent_unsupported_claim"
    if "invalid_model_output" in outcomes:
        return "intermittent_invalid_model_output"
    if "safe_partial" in outcomes:
        return "intermittent_partial_answer"
    if "explicit_model_abstention" in outcomes:
        return "intermittent_explicit_abstention"
    if any(outcome in {"safe_full", "safe_partial"} for outcome in outcomes):
        return "intermittent_safe_answer"
    return "other"


def classify_unsupported_claim(answer: dict[str, Any]) -> str | None:
    claims = answer.get("unsupported_claims")
    if not claims:
        return None
    text = " ".join(str(value) for value in claims).casefold()
    if any(marker in text for marker in ("因为", "导致", "所以", "therefore", "cause")):
        return "unsupported_causal_inference"
    if any(marker in text for marker in ("未来", "将会", "后续", "future")):
        return "unsupported_future_prediction"
    if any(char.isdigit() for char in text) or "%" in text:
        return "unsupported_quantification"
    if any(marker in text for marker in ("全部", "所有", "总是", "一定", "always")):
        return "unsupported_status_generalization"
    if any(marker in text for marker in ("/", ".md", "http", "doc-")):
        return "unsupported_entity_link"
    return "other"


def count_field(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _most_common_rate(counts: dict[str, int], denominator: int) -> float | None:
    if denominator == 0:
        return None
    return max(counts.values()) / denominator if counts else 0.0


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def _mean(values) -> float | None:
    data = [float(value) for value in values if value is not None]
    return sum(data) / len(data) if data else None


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value


def _citation_ids_for_evidence(evidence_payload: list[dict[str, Any]]) -> list[str]:
    return [f"C{index}" for index, _ in enumerate(evidence_payload, start=1)]
