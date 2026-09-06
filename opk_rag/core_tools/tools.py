from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from opk_rag.answer.config import AnswerGenerationConfig
from opk_rag.answer.grounding import GroundingValidator
from opk_rag.answer.models import AnswerGeneratorProvider, AnswerSystemError
from opk_rag.answer.service import _contains_prompt_injection_query, answer_knowledge_base_with_decision
from opk_rag.answerability import AnswerabilityConfig, AnswerabilityDecision, AnswerabilityPolicy
from opk_rag.core_tools.contracts import CoreToolError, ToolName
from opk_rag.core_tools.serialization import (
    answer_response_payload,
    answerability_payload,
    evidence_bundle_payload,
    evidence_identity_digest,
    grounding_payload,
    search_response_payload,
    trusted_evidence_payload,
)
from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingProvider
from opk_rag.reranking.provider import RerankerProvider
from opk_rag.search.config import VectorSearchConfig
from opk_rag.search.models import EvidenceBundle, SearchResponse
from opk_rag.search.service import KnowledgeBaseNotFoundError
from opk_rag.search.service import search_knowledge_base as core_search_knowledge_base

MAX_TOOL_CANDIDATES = 100
MAX_EXPANSION_ITEMS = 20


def search_knowledge_base(
    *,
    database_url: str,
    knowledge_base_id: UUID,
    query: str,
    provider: EmbeddingProvider | None,
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
    reranker_provider: RerankerProvider | None = None,
    context_token_counter=None,
) -> tuple[SearchResponse, dict[str, Any]]:
    _validate_query(query)
    _validate_candidate_bounds(search_config)
    try:
        response = core_search_knowledge_base(
            database_url,
            knowledge_base_id=knowledge_base_id,
            query=query,
            provider=provider,
            embedding_config=embedding_config,
            search_config=search_config,
            reranker_provider=reranker_provider,
            context_token_counter=context_token_counter,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise CoreToolError("knowledge_base_not_found", str(exc)) from exc
    except Exception:
        raise
    return response, search_response_payload(response)


def expand_document_section(
    *,
    evidence_bundle: EvidenceBundle,
    chunk_id: str,
    max_items: int = 1,
) -> tuple[EvidenceBundle, dict[str, Any]]:
    if max_items < 1 or max_items > MAX_EXPANSION_ITEMS:
        raise CoreToolError("candidate_limit_exceeded", f"max_items must be between 1 and {MAX_EXPANSION_ITEMS}.")
    matched = [item for item in evidence_bundle.items if str(item.chunk_id) == chunk_id]
    if not matched:
        raise CoreToolError("evidence_not_found", "chunk_id is not present in trusted current-turn evidence.")
    expanded_items = tuple(item for item in evidence_bundle.items if item.document_id == matched[0].document_id)[:max_items]
    expanded = replace(
        evidence_bundle,
        items=expanded_items,
        context_token_count=sum(item.context_token_count for item in expanded_items),
        total_token_count=sum(item.context_token_count for item in expanded_items),
    )
    return expanded, evidence_bundle_payload(expanded)


def assess_answerability(
    *,
    search_response: SearchResponse,
    config: AnswerabilityConfig | None = None,
) -> tuple[AnswerabilityDecision, dict[str, Any]]:
    bundle = _trusted_bundle(search_response)
    decision = AnswerabilityPolicy(config).evaluate(
        question=search_response.query,
        evidence_bundle=bundle,
        search_response=search_response,
        prompt_injection_detected=_contains_prompt_injection_query(search_response.query),
    )
    return decision, answerability_payload(decision)


def generate_grounded_answer(
    *,
    search_response: SearchResponse,
    answerability: AnswerabilityDecision,
    provider: AnswerGeneratorProvider,
    config: AnswerGenerationConfig,
):
    _trusted_bundle(search_response)
    if not answerability.answerable:
        raise CoreToolError("generation_not_permitted", "Answerability did not permit generation.", detail={"reason_code": answerability.reason_code})
    if not set(answerability.evidence_chunk_ids).issubset({str(item.chunk_id) for item in search_response.evidence_bundle.items}):
        raise CoreToolError("invalid_evidence_identity", "Answerability references evidence outside the current trusted bundle.")
    try:
        response = answer_knowledge_base_with_decision(
            search_response,
            answerability=answerability,
            provider=provider,
            config=config,
        )
    except AnswerSystemError as exc:
        code = "provider_failure" if exc.code in {"provider_error", "timeout", "model_not_found"} else "answerability_contract_violation"
        raise CoreToolError(code, str(exc), detail={"system_error_code": exc.code}) from exc
    return response, answer_response_payload(response)


def verify_grounding(
    *,
    answer_text: str,
    citations: list[str],
    evidence_bundle: EvidenceBundle,
    config,
):
    available = {f"C{index}" for index, _item in enumerate(evidence_bundle.items, start=1)}
    if not set(citations).issubset(available):
        raise CoreToolError("invalid_evidence_identity", "Citations must refer only to current trusted evidence.")
    try:
        decision = GroundingValidator(config).validate(
            answer_text=answer_text,
            evidence_bundle=evidence_bundle,
            structured_citations=citations,
        )
    except Exception as exc:
        raise CoreToolError("grounding_validation_failure", str(exc)) from exc
    return decision, grounding_payload(decision)


def tool_output(tool_name: ToolName, payload: dict[str, Any], *, bundle: EvidenceBundle | None = None) -> dict[str, Any]:
    output = dict(payload)
    if bundle is not None:
        output["trusted_evidence"] = trusted_evidence_payload(bundle)
    output["tool_name"] = tool_name
    return output


def assert_trusted_evidence(bundle: EvidenceBundle, expected_digest: str) -> None:
    actual = evidence_identity_digest(bundle)
    if actual != expected_digest:
        raise CoreToolError("invalid_evidence_identity", "Trusted evidence digest mismatch.", detail={"actual": actual, "expected": expected_digest})


def _validate_query(query: str) -> None:
    if not query.strip():
        raise CoreToolError("invalid_query", "query must not be empty.")


def _validate_candidate_bounds(config: VectorSearchConfig) -> None:
    if config.candidate_k > MAX_TOOL_CANDIDATES or config.bm25_candidate_k > MAX_TOOL_CANDIDATES:
        raise CoreToolError("candidate_limit_exceeded", f"candidate counts must be <= {MAX_TOOL_CANDIDATES}.")


def _trusted_bundle(search_response: SearchResponse) -> EvidenceBundle:
    if search_response.evidence_bundle is None:
        raise CoreToolError("invalid_evidence_identity", "SearchResponse does not contain an evidence bundle.")
    if search_response.evidence_bundle.knowledge_base_id != search_response.knowledge_base_id:
        raise CoreToolError("invalid_evidence_identity", "Evidence bundle knowledge_base_id does not match SearchResponse.")
    if search_response.evidence_bundle.query != search_response.query:
        raise CoreToolError("invalid_evidence_identity", "Evidence bundle query does not match SearchResponse.")
    return search_response.evidence_bundle
