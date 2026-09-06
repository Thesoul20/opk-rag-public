from __future__ import annotations

from opk_rag.runtime_v2.models import EvidenceContextV2, RetrievalCandidateV2


GOLD_LEAKAGE_KEYS = {"gold", "gold_chunk_ids", "expected_answer", "expected", "benchmark_classification", "is_gold"}


def candidates_to_evidence_context(candidates: tuple[RetrievalCandidateV2, ...]) -> tuple[EvidenceContextV2, ...]:
    return tuple(_candidate_to_context(candidate) for candidate in candidates)


def evidence_context_has_gold_leakage(contexts: tuple[EvidenceContextV2, ...]) -> bool:
    for context in contexts:
        payload = context.to_json()
        if _contains_gold_key(payload):
            return True
    return False


def _candidate_to_context(candidate: RetrievalCandidateV2) -> EvidenceContextV2:
    rank = candidate.reranked_rank or candidate.rank
    return EvidenceContextV2(
        canonical_chunk_id=candidate.canonical_chunk_id,
        text=candidate.content,
        source_path=candidate.normalized_source_path,
        heading_path=candidate.heading_path,
        provenance={
            "canonical_chunk_id": candidate.canonical_chunk_id,
            "document_id": candidate.document_id,
            "section_id": candidate.section_id,
            "source_start": candidate.source_start,
            "source_end": candidate.source_end,
        },
        rank=rank,
        citation_metadata={
            "source_path": candidate.normalized_source_path,
            "heading_path": list(candidate.heading_path),
            "source_start": candidate.source_start,
            "source_end": candidate.source_end,
            "rank": rank,
        },
    )


def _contains_gold_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in GOLD_LEAKAGE_KEYS:
                return True
            if _contains_gold_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_gold_key(item) for item in value)
    return False
