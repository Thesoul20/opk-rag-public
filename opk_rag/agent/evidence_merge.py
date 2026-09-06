from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVIDENCE_MERGE_CONTRACT_VERSION = "opk-rag.agent-evidence-merge.v1"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_identity: str
    document_id: str
    chunk_id: str
    rank: int
    retrieval_route: str = "search_knowledge_base"
    source_round: int = 1
    query_digest: str | None = None
    scope_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_identity": self.evidence_identity,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "retrieval_route": self.retrieval_route,
            "source_round": self.source_round,
            "query_digest": self.query_digest,
            "scope_id": self.scope_id,
            "metadata": self.metadata,
        }


def merge_evidence_rounds(
    first_round: list[EvidenceRecord],
    second_round: list[EvidenceRecord],
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    seen_evidence: dict[str, EvidenceRecord] = {}
    seen_chunks: set[str] = set()
    seen_documents: set[str] = set()
    seen_scopes: set[str] = set()
    classifications: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for record in [*first_round, *second_round]:
        classification = _classify(record, seen_evidence, seen_chunks, seen_documents, seen_scopes)
        first_seen = classification != "same_evidence_identity"
        if first_seen and (max_items is None or len(merged) < max_items):
            merged.append({**record.to_dict(), "first_seen": True, "merge_class": classification})
            seen_evidence[record.evidence_identity] = record
            seen_chunks.add(record.chunk_id)
            seen_documents.add(record.document_id)
            if record.scope_id:
                seen_scopes.add(record.scope_id)
        elif not first_seen:
            original = seen_evidence.get(record.evidence_identity)
            if original is not None and original.document_id != record.document_id:
                conflicts.append(
                    {
                        "reason_code": "evidence_identity_collision",
                        "evidence_identity": record.evidence_identity,
                        "first_document_id": original.document_id,
                        "duplicate_document_id": record.document_id,
                    }
                )
        classifications.append(
            {
                "evidence_identity": record.evidence_identity,
                "document_id": record.document_id,
                "chunk_id": record.chunk_id,
                "source_round": record.source_round,
                "merge_class": classification,
                "first_seen": first_seen,
            }
        )

    return {
        "schema_version": EVIDENCE_MERGE_CONTRACT_VERSION,
        "merged_evidence": merged,
        "classifications": classifications,
        "conflicts": conflicts,
        "new_document_count": sum(1 for item in classifications if item["source_round"] > 1 and item["merge_class"] == "new_document"),
        "new_evidence_identity_count": sum(1 for item in classifications if item["source_round"] > 1 and item["first_seen"]),
        "duplicate_evidence_count": sum(1 for item in classifications if item["source_round"] > 1 and item["merge_class"] == "same_evidence_identity"),
        "max_items": max_items,
    }


def records_from_trusted_evidence(trusted: dict[str, Any], *, source_round: int, query_digest: str | None = None) -> list[EvidenceRecord]:
    chunk_ids = [str(value) for value in trusted.get("chunk_ids") or []]
    document_ids = [str(value) for value in trusted.get("document_ids") or []]
    evidence_digest = str(trusted.get("evidence_identity_digest") or "")
    records: list[EvidenceRecord] = []
    for index, chunk_id in enumerate(chunk_ids):
        document_id = document_ids[index] if index < len(document_ids) else ""
        identity = f"{evidence_digest}:{chunk_id}" if evidence_digest else chunk_id
        records.append(EvidenceRecord(evidence_identity=identity, document_id=document_id, chunk_id=chunk_id, rank=index + 1, source_round=source_round, query_digest=query_digest))
    return records


def _classify(record: EvidenceRecord, seen_evidence: dict[str, EvidenceRecord], seen_chunks: set[str], seen_documents: set[str], seen_scopes: set[str]) -> str:
    if record.evidence_identity in seen_evidence:
        return "same_evidence_identity"
    if record.chunk_id in seen_chunks:
        return "duplicate_chunk"
    if record.scope_id and record.scope_id not in seen_scopes:
        return "new_scope"
    if record.document_id in seen_documents:
        return "same_document_new_chunk"
    return "new_document"
