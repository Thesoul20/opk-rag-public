from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.agent.query_privacy import query_digest

RETRIEVAL_ROUND_CONTRACT_VERSION = "opk-rag.agent-retrieval-round.v1"


@dataclass(frozen=True)
class RetrievalRound:
    round_index: int
    query_source: str
    query_digest: str
    query_length: int
    search_status: str
    result_count: int
    new_document_count: int
    new_evidence_identity_count: int
    duplicate_evidence_count: int
    answerability_before: str | None
    answerability_after: str | None
    latency_ms: int
    failure: dict[str, Any] | None = None
    schema_version: str = RETRIEVAL_ROUND_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_index": self.round_index,
            "query_source": self.query_source,
            "query_digest": self.query_digest,
            "query_length": self.query_length,
            "search_status": self.search_status,
            "result_count": self.result_count,
            "new_document_count": self.new_document_count,
            "new_evidence_identity_count": self.new_evidence_identity_count,
            "duplicate_evidence_count": self.duplicate_evidence_count,
            "answerability_before": self.answerability_before,
            "answerability_after": self.answerability_after,
            "latency_ms": self.latency_ms,
            "failure": self.failure,
        }


def build_retrieval_round(
    *,
    round_index: int,
    query: str,
    query_source: str,
    result_count: int,
    previous_document_ids: set[str] | None = None,
    previous_evidence_ids: set[str] | None = None,
    document_ids: list[str] | tuple[str, ...] = (),
    evidence_ids: list[str] | tuple[str, ...] = (),
    answerability_before: str | None = None,
    answerability_after: str | None = None,
    latency_ms: int = 0,
    failure: dict[str, Any] | None = None,
) -> RetrievalRound:
    previous_document_ids = previous_document_ids or set()
    previous_evidence_ids = previous_evidence_ids or set()
    current_documents = set(str(value) for value in document_ids)
    current_evidence = set(str(value) for value in evidence_ids)
    return RetrievalRound(
        round_index=round_index,
        query_source=query_source,
        query_digest=query_digest(query.strip()),
        query_length=len(query.strip()),
        search_status="failure" if failure else "success",
        result_count=result_count,
        new_document_count=len(current_documents - previous_document_ids),
        new_evidence_identity_count=len(current_evidence - previous_evidence_ids),
        duplicate_evidence_count=len(current_evidence & previous_evidence_ids),
        answerability_before=answerability_before,
        answerability_after=answerability_after,
        latency_ms=latency_ms,
        failure=failure,
    )


def retrieval_round_contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": RETRIEVAL_ROUND_CONTRACT_VERSION,
        "required_fields": [
            "round_index",
            "query_source",
            "query_digest",
            "query_length",
            "search_status",
            "result_count",
            "new_document_count",
            "new_evidence_identity_count",
            "duplicate_evidence_count",
            "answerability_before",
            "answerability_after",
            "latency_ms",
            "failure",
        ],
        "stores_full_query": False,
    }
