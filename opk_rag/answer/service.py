from __future__ import annotations

from dataclasses import replace
import re
import unicodedata

from opk_rag.answer.config import AnswerGenerationConfig, SCOPE_UNITS_OUTPUT_SCHEMA_VERSION
from opk_rag.answer.citation_parser import parse_citation_markers
from opk_rag.answer.evidence_context import build_evidence_context, evidence_context_diagnostics
from opk_rag.answerability import AnswerabilityConfig, AnswerabilityDecision, AnswerabilityPolicy
from opk_rag.answer.grounding import GroundingValidationDecision, GroundingValidator
from opk_rag.answer.models import AnswerGenerationRequest, AnswerGeneratorProvider, AnswerResponse, AnswerSystemError
from opk_rag.answer.negative_semantics import classify_answer_semantics, refine_answerability_for_semantics, validate_supported_negative_answer
from opk_rag.answer.prompt import EVIDENCE_SERIALIZATION_VERSION
from opk_rag.answer.provider import AnswerProviderError
from opk_rag.answer.refusal import refusal_text
from opk_rag.answer.scope_execution import build_scope_execution_contract, validate_scope_execution_output
from opk_rag.answer.validation import build_citations, claim_grounding_diagnostics, unsupported_claims
from opk_rag.search.models import SearchResponse
from opk_rag.showcase.runtime_trace import RuntimeTraceContext, build_runtime_trace_from_live_execution


def answer_knowledge_base(
    search_response: SearchResponse,
    *,
    provider: AnswerGeneratorProvider,
    config: AnswerGenerationConfig,
    runtime_trace_context: RuntimeTraceContext | None = None,
) -> AnswerResponse:
    policy = AnswerabilityPolicy(_effective_answerability_config(config))
    controller_answerability = policy.evaluate(
        question=search_response.query,
        evidence_bundle=search_response.evidence_bundle,
        search_response=search_response,
        prompt_injection_detected=_contains_prompt_injection_query(search_response.query),
    )
    answerability = controller_answerability
    if search_response.evidence_bundle is not None:
        answerability = refine_answerability_for_semantics(
            question=search_response.query,
            bundle=search_response.evidence_bundle,
            decision=controller_answerability,
        )
    if not answerability.answerable:
        response = _with_runtime_trace(
            _refusal(search_response, config=config, provider=provider, reason_code=answerability.reason_code, answerability=answerability, model_decision={"answerability": _decision_payload(answerability)}),
            runtime_trace_context=runtime_trace_context,
            config=config,
        )
        return replace(response, controller_answerability=controller_answerability)
    response = answer_knowledge_base_with_decision(
        search_response,
        answerability=answerability,
        provider=provider,
        config=config,
        runtime_trace_context=runtime_trace_context,
    )
    return replace(response, controller_answerability=controller_answerability)


def answer_knowledge_base_with_decision(
    search_response: SearchResponse,
    *,
    answerability: AnswerabilityDecision,
    provider: AnswerGeneratorProvider,
    config: AnswerGenerationConfig,
    runtime_trace_context: RuntimeTraceContext | None = None,
) -> AnswerResponse:
    if not answerability.answerable:
        return _with_runtime_trace(
            _refusal(search_response, config=config, provider=provider, reason_code=answerability.reason_code, answerability=answerability, model_decision={"answerability": _decision_payload(answerability)}),
            runtime_trace_context=runtime_trace_context,
            config=config,
        )

    bundle = search_response.evidence_bundle
    assert bundle is not None
    answer_semantics = classify_answer_semantics(question=search_response.query, bundle=bundle, answerability=answerability)
    scope_execution_contract = build_scope_execution_contract(answerability, bundle, question=search_response.query) if config.output_schema_version == SCOPE_UNITS_OUTPUT_SCHEMA_VERSION else None
    evidence_context = build_evidence_context(
        question=search_response.query,
        bundle=bundle,
        answerability=answerability,
        output_schema_version=config.output_schema_version,
    )

    request = AnswerGenerationRequest(
        query=search_response.query,
        evidence_bundle=bundle,
        prompt_version=config.prompt_version,
        output_schema_version=config.output_schema_version,
        answerability=answerability,
        scope_execution_contract=scope_execution_contract,
    )
    try:
        raw = provider.generate_answer(request)
    except AnswerProviderError as exc:
        detail = _provider_error_detail_payload(exc)
        reason = detail.get("reason_code") if detail else None
        message = str(exc).lower()
        if reason in {"connection_timeout", "read_timeout"} or "timed out" in message:
            raise AnswerSystemError("timeout", "LLM endpoint timed out.", provider_error_detail=detail) from exc
        if reason == "model_not_found" or "404" in message and "model" in message:
            raise AnswerSystemError("model_not_found", "LLM model was not found by the configured runtime.", provider_error_detail=detail) from exc
        raise AnswerSystemError("provider_error", "LLM provider failed.", provider_error_detail=detail) from exc

    parsed = raw.parsed
    if not isinstance(parsed, dict):
        raise AnswerSystemError("invalid_model_output", "LLM returned invalid JSON after configured retries.")
    try:
        model_decision = _parse_model_decision(parsed)
    except ValueError as exc:
        raise AnswerSystemError("invalid_model_output", "LLM returned JSON that does not match the answer schema.") from exc
    if scope_execution_contract is not None:
        scope_validation = validate_scope_execution_output(parsed, scope_execution_contract)
        model_decision = {
            **model_decision,
            "answer_semantics": answer_semantics.to_provider_payload(),
            "scope_execution_contract": scope_execution_contract.to_provider_payload(),
            "scope_execution_validation": _scope_validation_payload(scope_validation),
        }
        if scope_execution_contract.answer_mode == "abstain":
            return _with_runtime_trace(_refusal(search_response, config=config, provider=provider, raw=raw, reason_code="model_abstained", answerability=answerability, model_decision=model_decision), runtime_trace_context=runtime_trace_context, config=config)
        if not scope_validation.valid:
            return _with_runtime_trace(_refusal(search_response, config=config, provider=provider, raw=raw, reason_code=_scope_validation_reason_code(scope_validation.reason_code), answerability=answerability, model_decision={**model_decision, "unsupported_claims": (scope_validation.reason_code,)}), runtime_trace_context=runtime_trace_context, config=config)
        parsed = {**parsed, "answer": scope_validation.answer_text, "citations": list(scope_validation.citations)}
        model_decision = {**model_decision, "decision": "answer", "answer": scope_validation.answer_text, "citations": list(scope_validation.citations)}
    model_decision = {**model_decision, "answer_semantics": answer_semantics.to_provider_payload()}
    context_diagnostics = evidence_context_diagnostics(
        bundle=bundle,
        context=evidence_context,
        provider_response_mode=model_decision["decision"],
    )
    model_decision = {**model_decision, "evidence_context_diagnostics": context_diagnostics}
    if model_decision["decision"] == "abstain":
        reason = _generation_abstention_reason(answerability, model_decision["reason"])
        if reason == "no_evidence" and search_response.evidence_bundle and search_response.evidence_bundle.items:
            reason = "insufficient_evidence"
        return _with_runtime_trace(_refusal(search_response, config=config, provider=provider, raw=raw, reason_code=reason, answerability=answerability, model_decision=model_decision), runtime_trace_context=runtime_trace_context, config=config)
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise AnswerSystemError("invalid_model_output", "LLM returned an empty answer for an answer decision.")
    validator = GroundingValidator(config.grounding)
    try:
        grounding = validator.validate(answer_text=answer, evidence_bundle=bundle, structured_citations=model_decision["citations"])
    except Exception as exc:
        raise AnswerSystemError("validation_error", "Grounding validator failed.") from exc
    if not grounding.valid:
        return _with_runtime_trace(_refusal(
            search_response,
            config=config,
            provider=provider,
            raw=raw,
            reason_code="ungrounded_answer",
            answerability=answerability,
            grounding=grounding,
            model_decision={**model_decision, "grounding": _grounding_payload(grounding)},
        ), runtime_trace_context=runtime_trace_context, config=config)
    try:
        citation_ids = parsed.get("citations") if grounding.status == "disabled" else list(grounding.cited_ids)
        citations = build_citations(citation_ids, bundle)
    except ValueError as exc:
        raise AnswerSystemError("validation_error", "Grounding validator produced citations that cannot be mapped to evidence.") from exc
    unsupported = unsupported_claims(answer, citations)
    claim_diagnostics = claim_grounding_diagnostics(answer, citations)
    semantic_failures = validate_supported_negative_answer(answer_text=answer, citation_ids=tuple(c.citation_id for c in citations), contract=answer_semantics)
    contract_failures = (*_generation_contract_failures(search_response.query, answerability, answer, model_decision, unsupported), *semantic_failures)
    if contract_failures:
        unsupported = (*unsupported, *contract_failures)
    if unsupported:
        reason_code = _contract_reason_code(contract_failures) if contract_failures else "unsupported_claims"
        grounding = GroundingValidationDecision(
            valid=False,
            status="refused",
            reason_code=reason_code,
            reason="Answer contains checkable values not present in cited evidence.",
            cited_ids=grounding.cited_ids,
            valid_cited_ids=grounding.valid_cited_ids,
            invalid_cited_ids=grounding.invalid_cited_ids,
            available_evidence_ids=grounding.available_evidence_ids,
            citation_coverage=grounding.citation_coverage,
            diagnostics={**grounding.diagnostics, "unsupported_claims": list(unsupported), "contract_failures": list(contract_failures), "claim_diagnostics": [_claim_diagnostic_payload(diagnostic) for diagnostic in claim_diagnostics]},
        )
        return _with_runtime_trace(_refusal(search_response, config=config, provider=provider, raw=raw, reason_code=reason_code, answerability=answerability, grounding=grounding, model_decision={**model_decision, "unsupported_claims": list(unsupported), "contract_failures": list(contract_failures), "grounding": _grounding_payload(grounding)}), runtime_trace_context=runtime_trace_context, config=config)
    return _with_runtime_trace(AnswerResponse(
        status="answered",
        answerable=True,
        answer=answer.strip(),
        refusal_reason_code=None,
        answerability=_mark_llm_called(answerability),
        grounding=grounding,
        citations=citations,
        unsupported_claims=(),
        search_response=search_response,
        provider_id=raw.provider_id,
        model_id=raw.model_id,
        model_revision=raw.model_revision,
        model_license=config.model_license,
        runtime_base_url=config.base_url,
        runtime_endpoint_type=raw.endpoint_type,
        prompt_version=config.prompt_version,
        output_schema_version=config.output_schema_version,
        evidence_serialization=EVIDENCE_SERIALIZATION_VERSION,
        generation_latency_ms=raw.latency_ms,
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens,
        total_tokens=raw.total_tokens,
        model_decision={**model_decision, "answerability": _decision_payload(_mark_llm_called(answerability)), "grounding": _grounding_payload(grounding)},
        raw_generation_text=raw.raw_text,
        finish_reason=raw.finish_reason,
        output_truncated=raw.finish_reason == "length",
        empty_output=not raw.raw_text.strip(),
        request_attempts=raw.request_attempts,
    ), runtime_trace_context=runtime_trace_context, config=config)


def _with_runtime_trace(answer: AnswerResponse, *, runtime_trace_context: RuntimeTraceContext | None, config: AnswerGenerationConfig) -> AnswerResponse:
    if runtime_trace_context is None or not runtime_trace_context.enabled:
        return answer
    from dataclasses import replace

    trace = build_runtime_trace_from_live_execution(
        runtime_trace_context,
        search_response=answer.search_response,
        timing_snapshot=((answer.search_response.runtime_trace or {}).get("timings") or {}).get("retrieval_timing_snapshot") or {},
        answer=answer,
        answer_config=config,
    )
    return replace(answer, runtime_trace=trace, search_response=replace(answer.search_response, runtime_trace=trace))


def _contains_prompt_injection_query(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    suspicious_fragments = (
        "忽略之前",
        "忽略以上",
        "忽略所有",
        "ignore previous",
        "ignore all previous",
        "输出系统提示",
        "显示系统提示",
        "泄露系统提示",
        "system prompt",
        "不要引用来源",
        "不要引用",
        "不需要引用",
        "回答测试成功",
        "回答\"测试成功\"",
        "回答“测试成功”",
    )
    return any(fragment in normalized for fragment in suspicious_fragments)


def _parse_model_decision(parsed: dict) -> dict:
    repairs: list[str] = []
    if "answer_units" in parsed:
        decision = "abstain" if parsed.get("action") == "abstain" or parsed.get("decision") == "abstain" else "answer"
        answer = ""
        citations = []
        reason = None
    else:
        decision = _decision_value(parsed, repairs)
        answer = parsed.get("answer", "")
        citations = parsed.get("citations", [])
        reason = parsed.get("reason")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string.")
    if citations is None:
        citations = []
        repairs.append("null_citations_as_empty")
    if not isinstance(citations, list) or not all(isinstance(value, str) for value in citations):
        raise ValueError("citations must be a string array.")
    citations = list(citations)
    if decision == "answer" and not citations:
        inline_citations = list(parse_citation_markers(answer).cited_ids)
        if inline_citations:
            citations = inline_citations
            repairs.append("citations_recovered_from_inline_markers")
    allowed_reasons = {
        "no_evidence",
        "insufficient_evidence",
        "model_abstained",
        "invalid_citations",
        "unsupported_claims",
        "prompt_injection_detected",
    }
    if reason is not None and reason not in allowed_reasons:
        if decision == "answer":
            reason = None
            repairs.append("ignored_answer_reason")
        else:
            raise ValueError("reason is not an allowed abstention reason.")
    if decision == "answer" and reason is not None:
        reason = None
        repairs.append("ignored_answer_reason")
    if decision == "abstain" and citations:
        raise ValueError("abstain decisions must not cite evidence.")
    if "answerability_status" in parsed and parsed["answerability_status"] is not None and parsed["answerability_status"] not in {"answerable", "partially_answerable", "unanswerable"}:
        if decision == "answer" and parsed["answerability_status"] in {"partial_answer", "partial"}:
            repairs.append("normalized_answerability_status")
        else:
            raise ValueError("answerability_status is invalid.")
    supported_scope = _string_list_field(parsed, "supported_scope", repairs)
    unsupported_scope = _string_list_field(parsed, "unsupported_scope", repairs)
    if "premise_correction" in parsed and parsed["premise_correction"] is not None and not isinstance(parsed["premise_correction"], str):
        raise ValueError("premise_correction must be string or null.")
    decision_payload = {
        "decision": decision,
        "answer": answer,
        "citations": citations,
        "reason": reason,
        "answerability_status": _answerability_status(parsed),
        "supported_scope": supported_scope,
        "unsupported_scope": unsupported_scope,
        "premise_correction": parsed.get("premise_correction"),
    }
    if repairs:
        decision_payload["contract_repairs"] = tuple(dict.fromkeys(repairs))
    return decision_payload


def _decision_value(parsed: dict, repairs: list[str]) -> str:
    if "decision" in parsed:
        raw = parsed["decision"]
    elif "action" in parsed:
        raw = parsed["action"]
        repairs.append("action_field_as_decision")
    elif "status" in parsed:
        raw = parsed["status"]
        repairs.append("status_field_as_decision")
    else:
        raise ValueError("model output must contain decision.")
    if raw in {"answer", "abstain"}:
        return raw
    if raw in {"partial_answer", "partial", "answered"}:
        repairs.append("normalized_answer_decision")
        return "answer"
    if raw in {"refuse", "refused", "no_answer"}:
        repairs.append("normalized_abstain_decision")
        return "abstain"
    raise ValueError("decision must be answer or abstain.")


def _string_list_field(parsed: dict, key: str, repairs: list[str]) -> list[str]:
    if key not in parsed or parsed[key] is None:
        return []
    value = parsed[key]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        repairs.append(f"{key}_string_as_singleton")
        return [stripped]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string array.")
    return value


def _answerability_status(parsed: dict) -> str | None:
    status = parsed.get("answerability_status")
    if status in {"partial_answer", "partial"}:
        return "partially_answerable"
    return status


def _effective_answerability_config(config: AnswerGenerationConfig) -> AnswerabilityConfig:
    baseline = AnswerabilityConfig()
    if config.answerability == baseline and (config.min_evidence_items != baseline.min_evidence or config.min_context_tokens != baseline.min_context_tokens):
        return replace(config.answerability, min_evidence=config.min_evidence_items, min_context_tokens=config.min_context_tokens)
    return config.answerability


def _refusal(search_response: SearchResponse, *, config: AnswerGenerationConfig, provider, reason_code, answerability: AnswerabilityDecision, model_decision, raw=None, grounding: GroundingValidationDecision | None = None) -> AnswerResponse:
    endpoint_type = getattr(provider, "endpoint_type", "loopback")
    llm_called = raw is not None
    decision = _mark_llm_called(answerability) if llm_called else answerability
    grounding_decision = grounding or GroundingValidator(config.grounding).not_applicable(search_response.evidence_bundle)
    return AnswerResponse(
        status="refused",
        answerable=False,
        answer=refusal_text(decision, reason_code=reason_code),
        refusal_reason_code=reason_code,
        answerability=decision,
        grounding=grounding_decision,
        citations=(),
        unsupported_claims=tuple(model_decision.get("unsupported_claims", ())) if isinstance(model_decision, dict) else (),
        search_response=search_response,
        provider_id=getattr(provider, "provider_id", config.provider_id),
        model_id=getattr(provider, "model_id", config.model_id),
        model_revision=getattr(provider, "model_revision", config.model_revision),
        model_license=config.model_license,
        runtime_base_url=config.base_url,
        runtime_endpoint_type=endpoint_type,
        prompt_version=config.prompt_version,
        output_schema_version=config.output_schema_version,
        evidence_serialization=EVIDENCE_SERIALIZATION_VERSION,
        generation_latency_ms=getattr(raw, "latency_ms", None),
        prompt_tokens=getattr(raw, "prompt_tokens", None),
        completion_tokens=getattr(raw, "completion_tokens", None),
        total_tokens=getattr(raw, "total_tokens", None),
        model_decision={**model_decision, "answerability": _decision_payload(decision), "grounding": _grounding_payload(grounding_decision)}
        if isinstance(model_decision, dict)
        else {"answerability": _decision_payload(decision), "grounding": _grounding_payload(grounding_decision)},
        raw_generation_text=getattr(raw, "raw_text", None),
        finish_reason=getattr(raw, "finish_reason", None),
        output_truncated=getattr(raw, "finish_reason", None) == "length",
        empty_output=not bool(str(getattr(raw, "raw_text", "")).strip()) if raw is not None else False,
        request_attempts=getattr(raw, "request_attempts", ()),
    )


def _mark_llm_called(decision: AnswerabilityDecision) -> AnswerabilityDecision:
    diagnostics = {**decision.diagnostics, "answer_llm_called": True}
    return AnswerabilityDecision(
        status=decision.status,
        reason_code=decision.reason_code,
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_chunk_ids=decision.evidence_chunk_ids,
        evidence_score=decision.evidence_score,
        evidence_count=decision.evidence_count,
        considered_evidence_count=decision.considered_evidence_count,
        diagnostics=diagnostics,
    )


def _provider_error_detail_payload(exc: AnswerProviderError) -> dict | None:
    detail = getattr(exc, "detail", None)
    if detail is None:
        return None
    return {
        "reason_code": detail.reason_code,
        "http_status": detail.http_status,
        "sdk_error_type": detail.sdk_error_type,
        "error_summary": detail.error_summary,
        "attempts": list(detail.attempts),
    }


def _decision_payload(decision: AnswerabilityDecision) -> dict[str, object]:
    return {
        "answerable": decision.answerable,
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


def _grounding_payload(decision: GroundingValidationDecision) -> dict[str, object]:
    return {
        "valid": decision.valid,
        "status": decision.status,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "cited_ids": list(decision.cited_ids),
        "valid_cited_ids": list(decision.valid_cited_ids),
        "invalid_cited_ids": list(decision.invalid_cited_ids),
        "available_evidence_ids": list(decision.available_evidence_ids),
        "citation_coverage": decision.citation_coverage,
        "diagnostics": decision.diagnostics,
    }


def _scope_validation_payload(validation) -> dict[str, object]:
    return {
        "valid": validation.valid,
        "reason_code": validation.reason_code,
        "reason": validation.reason,
        "answer_unit_count": validation.diagnostics.get("answer_unit_count"),
        "valid_unit_count": validation.diagnostics.get("valid_unit_count"),
        "citation_count": len(validation.citations),
        "diagnostics": validation.diagnostics,
    }


def _scope_validation_reason_code(reason_code: str) -> str:
    if reason_code in {"missing_citation", "unknown_citation"}:
        return "invalid_citations"
    return "generation_overreach"


def _claim_diagnostic_payload(diagnostic) -> dict[str, object]:
    return {
        "claim_id": diagnostic.claim_id,
        "claim": diagnostic.claim,
        "citation_ids": list(diagnostic.citation_ids),
        "checkable_values": list(diagnostic.checkable_values),
        "unsupported_values": list(diagnostic.unsupported_values),
        "support_status": diagnostic.support_status,
        "validator_status": diagnostic.validator_status,
        "reason": diagnostic.reason,
    }


def _generation_abstention_reason(answerability: AnswerabilityDecision, model_reason: str | None) -> str:
    if answerability.reason_code == "false_premise":
        return "false_premise_not_corrected"
    if answerability.status == "partially_answerable":
        return "partial_supported_scope_missing"
    if answerability.status == "answerable":
        return "answerable_generation_abstained"
    return model_reason or "model_abstained"


def _generation_contract_failures(question: str, answerability: AnswerabilityDecision, answer: str, model_decision: dict, unsupported_claims: tuple[str, ...]) -> tuple[str, ...]:
    failures: list[str] = []
    if answerability.status == "partially_answerable" and answerability.reason_code != "false_premise":
        if not _has_partial_disclosure(answer):
            failures.append("partial_disclosure_missing")
        if not _has_supported_cited_statement(answer):
            failures.append("partial_supported_scope_missing")
        fabricated = _partial_unsupported_scope_fabricated(answer, model_decision)
        if fabricated:
            failures.append(f"partial_unsupported_scope_fabricated: {fabricated}")
    if answerability.reason_code == "false_premise" and _requires_false_premise_correction(question):
        if not _corrects_false_premise(answer):
            failures.append("false_premise_not_corrected")
        if _follows_false_premise(question, answer):
            failures.append("false_premise_followed")
    return tuple(dict.fromkeys(failures))


def _contract_reason_code(failures: tuple[str, ...]) -> str:
    for failure in failures:
        code = failure.split(":", 1)[0]
        if code in {
            "partial_supported_scope_missing",
            "partial_unsupported_scope_fabricated",
            "partial_disclosure_missing",
            "false_premise_not_corrected",
            "false_premise_followed",
            "citation_not_supporting_claim",
            "generation_overreach",
        }:
            return code
    return "unsupported_claims"


def _has_partial_disclosure(answer: str) -> bool:
    return any(marker in answer for marker in ("当前知识库没有提供", "知识库没有提供", "未提供", "没有提供", "未说明", "没有说明", "缺失", "无法确定"))


def _has_supported_cited_statement(answer: str) -> bool:
    return any(re.search(r"\[C[1-9][0-9]*\]", line) and not _is_disclosure_only(line) for line in answer.splitlines())


def _is_disclosure_only(text: str) -> bool:
    return _has_partial_disclosure(text) and not re.search(r"\[C[1-9][0-9]*\]", text)


def _partial_unsupported_scope_fabricated(answer: str, model_decision: dict) -> str | None:
    unsupported_scope = tuple(str(value).strip() for value in model_decision.get("unsupported_scope", ()) if str(value).strip())
    if not unsupported_scope:
        return None
    normalized_answer = _normalize_text(answer)
    for scope in unsupported_scope:
        normalized_scope = _normalize_text(scope)
        if normalized_scope and normalized_scope in normalized_answer and not _has_partial_disclosure(answer):
            return scope
    return None


def _corrects_false_premise(answer: str) -> bool:
    normalized = answer.casefold()
    correction_markers = ("前提不成立", "前提错误", "不符合", "不是", "并非", "未", "没有", "不直接", "短期不做", "暂缓")
    return any(marker in normalized for marker in correction_markers)


def _requires_false_premise_correction(question: str) -> bool:
    normalized = question.casefold()
    explicit_patterns = (
        "既然",
        "选择直接群发",
        "直接群发作为发布目标",
        "已经封装成",
        "测试失败",
        "失败了",
        "完全不兼容",
        "迁移所有环境",
    )
    return any(pattern in normalized for pattern in explicit_patterns)


def _follows_false_premise(question: str, answer: str) -> bool:
    normalized_question = _normalize_text(question)
    normalized_answer = _normalize_text(answer)
    if "直接群发" in normalized_question and "因为" in normalized_answer and not any(marker in normalized_answer for marker in ("不直接群发", "前提不成立", "前提错误")):
        return True
    return False


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
