from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from opk_rag.answer.models import AnswerResponse, Citation
from opk_rag.answerability import AnswerabilityDecision
from opk_rag.search.models import EvidenceBundle, EvidenceItem, SearchResponse, SearchResult


def evidence_identity_digest(bundle: EvidenceBundle) -> str:
    payload = [
        {
            "context_rank": item.context_rank,
            "chunk_id": str(item.chunk_id),
            "document_id": str(item.document_id),
            "relative_path": item.relative_path,
            "heading_path": list(item.heading_path),
            "start_line": item.start_line,
            "end_line": item.end_line,
            "content_hash": hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
        }
        for item in bundle.items
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trusted_evidence_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    from opk_rag.core_tools.contracts import TrustedEvidence

    trusted = TrustedEvidence(
        knowledge_base_id=str(bundle.knowledge_base_id),
        query=bundle.query,
        evidence_identity_digest=evidence_identity_digest(bundle),
        citation_ids=tuple(f"C{index}" for index, _item in enumerate(bundle.items, start=1)),
        chunk_ids=tuple(str(item.chunk_id) for item in bundle.items),
        document_ids=tuple(str(item.document_id) for item in bundle.items),
        context_token_count=bundle.context_token_count,
        context_token_budget=bundle.context_token_budget,
    )
    return trusted.to_dict()


def search_response_payload(response: SearchResponse) -> dict[str, Any]:
    return {
        "query_hash": _digest(response.query),
        "knowledge_base_id": str(response.knowledge_base_id),
        "retrieval_mode": response.retrieval_mode,
        "requested_top_k": response.requested_top_k,
        "candidate_k": response.candidate_k,
        "candidate_count": response.candidate_count,
        "result_count": response.result_count,
        "context_token_count": response.context_token_count,
        "reranker_enabled": response.reranker_enabled,
        "results": [_search_result_payload(result) for result in response.results],
        "trusted_evidence": None if response.evidence_bundle is None else trusted_evidence_payload(response.evidence_bundle),
    }


def evidence_bundle_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "query_hash": _digest(bundle.query),
        "knowledge_base_id": str(bundle.knowledge_base_id),
        "context_token_budget": bundle.context_token_budget,
        "context_token_count": bundle.context_token_count,
        "total_token_count": bundle.total_token_count,
        "items": [_evidence_item_payload(item, index) for index, item in enumerate(bundle.items, start=1)],
        "trusted_evidence": trusted_evidence_payload(bundle),
    }


def answerability_payload(decision: AnswerabilityDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason_code": decision.reason_code,
        "confidence": decision.confidence,
        "evidence_chunk_ids": list(decision.evidence_chunk_ids),
        "evidence_score": decision.evidence_score,
        "evidence_count": decision.evidence_count,
        "considered_evidence_count": decision.considered_evidence_count,
        "diagnostics": _redact_diagnostics(decision.diagnostics),
    }


def answer_response_payload(response: AnswerResponse) -> dict[str, Any]:
    return {
        "status": response.status,
        "answerable": response.answerable,
        "answer_hash": _digest(response.answer),
        "refusal_reason_code": response.refusal_reason_code,
        "provider_id": response.provider_id,
        "model_id": response.model_id,
        "runtime_endpoint_type": response.runtime_endpoint_type,
        "prompt_version": response.prompt_version,
        "output_schema_version": response.output_schema_version,
        "grounding": grounding_payload(response.grounding),
        "citations": [citation_payload(citation) for citation in response.citations],
        "unsupported_claims": list(response.unsupported_claims),
        "answerability": answerability_payload(response.answerability),
    }


def grounding_payload(decision) -> dict[str, Any]:
    return {
        "valid": decision.valid,
        "status": decision.status,
        "reason_code": decision.reason_code,
        "cited_ids": list(decision.cited_ids),
        "valid_cited_ids": list(decision.valid_cited_ids),
        "invalid_cited_ids": list(decision.invalid_cited_ids),
        "available_evidence_ids": list(decision.available_evidence_ids),
        "citation_coverage": decision.citation_coverage,
        "diagnostics": _redact_diagnostics(decision.diagnostics),
    }


def citation_payload(citation: Citation) -> dict[str, Any]:
    return {
        "citation_id": citation.citation_id,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "relative_path": citation.relative_path,
        "heading_path": list(citation.heading_path),
        "start_line": citation.start_line,
        "end_line": citation.end_line,
        "snippet_hash": _digest(citation.snippet),
    }


def stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _search_result_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "document_id": str(result.document_id),
        "chunk_id": str(result.chunk_id),
        "relative_path": result.relative_path,
        "heading_path": list(result.heading_path),
        "start_line": result.start_line,
        "end_line": result.end_line,
        "similarity": result.similarity,
        "vector_similarity": result.vector_similarity,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "context_rank": result.context_rank,
        "selected_for_context": result.selected_for_context,
        "retrieval_sources": list(result.retrieval_sources),
        "content_hash": _digest(result.content),
    }


def _evidence_item_payload(item: EvidenceItem, citation_index: int) -> dict[str, Any]:
    return {
        "citation_id": f"C{citation_index}",
        "context_rank": item.context_rank,
        "result_rank": item.result_rank,
        "chunk_id": str(item.chunk_id),
        "document_id": str(item.document_id),
        "relative_path": item.relative_path,
        "heading_path": list(item.heading_path),
        "start_line": item.start_line,
        "end_line": item.end_line,
        "context_token_count": item.context_token_count,
        "retrieval_sources": list(item.retrieval_sources),
        "content_hash": _digest(item.content),
    }


def _redact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for key, value in diagnostics.items():
        if key in {"query_terms", "covered_query_terms", "missing_exact_requirements", "conflicting_values"}:
            redacted[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            redacted[key] = value
        elif isinstance(value, (list, tuple)):
            redacted[key] = [item for item in value if isinstance(item, (str, int, float, bool)) or item is None]
        elif isinstance(value, dict):
            redacted[key] = _redact_diagnostics(value)
    return redacted


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value
