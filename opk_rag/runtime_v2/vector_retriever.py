from __future__ import annotations

from dataclasses import replace
from typing import Any

from opk_rag.runtime_v2.models import CanonicalChunkV2, RetrievalCandidateV2


def build_vector_candidates_v2(
    *,
    sample_unit_id: str,
    trace_row: dict[str, Any],
    chunks_by_frozen_id: dict[str, CanonicalChunkV2],
    top_k: int,
) -> tuple[RetrievalCandidateV2, ...]:
    del sample_unit_id
    candidates = []
    for rank, frozen_chunk_id in enumerate((trace_row.get("candidate_chunk_ids") or [])[:top_k], start=1):
        chunk = chunks_by_frozen_id[str(frozen_chunk_id)]
        candidates.append(
            RetrievalCandidateV2(
                canonical_chunk_id=chunk.canonical_chunk_id,
                rank=rank,
                retrieval_strategy=f"vector_top_{top_k}",
                retrieval_score=None,
                document_id=chunk.document_id,
                section_id=chunk.section_id,
                normalized_source_path=chunk.normalized_source_path,
                heading_path=chunk.heading_path,
                source_start=chunk.source_start,
                source_end=chunk.source_end,
                content=chunk.content,
                content_digest=chunk.content_digest,
                original_vector_rank=rank,
            )
        )
    return tuple(candidates)


def reranked_candidate_v2(candidate: RetrievalCandidateV2, *, score: float, reranked_rank: int, final_strategy: str) -> RetrievalCandidateV2:
    return replace(candidate, rank=reranked_rank, retrieval_strategy=final_strategy, reranker_score=score, reranked_rank=reranked_rank)
