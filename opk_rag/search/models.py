from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
from uuid import UUID


@dataclass(frozen=True)
class EvidenceItem:
    context_rank: int
    result_rank: int
    chunk_id: UUID
    document_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    context_token_count: int
    reranker_pair_token_count: int | None
    reranker_original_pair_token_count: int | None
    reranker_input_truncated: bool
    rerank_score: float | None
    retrieval_sources: tuple[Literal["vector", "bm25", "structure", "graph"], ...]

    @property
    def token_count(self) -> int:
        return self.context_token_count


@dataclass(frozen=True)
class EvidenceBundle:
    query: str
    normalized_query: str
    knowledge_base_id: UUID
    context_token_budget: int
    context_token_count: int
    total_token_count: int
    context_tokenizer_id: str
    context_tokenizer_revision: str | None
    items: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class EvidenceSignals:
    top_reranker_score: float | None
    second_reranker_score: float | None
    top1_top2_margin: float | None
    max_vector_similarity: float | None
    max_bm25_score: float | None
    max_rrf_score: float | None
    dual_channel_candidate_count: int
    selected_source_document_count: int
    selected_chunk_count: int
    selected_context_token_count: int
    query_term_coverage: float
    exact_identifier_match: bool
    any_reranker_input_truncated: bool
    reranker_score_min: float | None
    reranker_score_max: float | None
    reranker_score_mean: float | None

    @property
    def top_rerank_score(self) -> float | None:
        return self.top_reranker_score

    @property
    def max_rerank_margin(self) -> float | None:
        return self.top1_top2_margin

    @property
    def selected_evidence_count(self) -> int:
        return self.selected_chunk_count


@dataclass(frozen=True)
class SearchResult:
    rank: int
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    similarity: float
    vector_similarity: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    original_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    pair_token_count: int | None = None
    reranker_pair_token_count: int | None = None
    reranker_original_pair_token_count: int | None = None
    reranker_input_truncated: bool = False
    context_token_count: int | None = None
    selected_for_context: bool = False
    context_rank: int | None = None
    retrieval_sources: tuple[Literal["vector", "bm25", "structure", "graph"], ...] = ()
    matched_terms: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(frozen=True)
class SearchResponse:
    query: str
    normalized_query: str
    knowledge_base_id: UUID
    model_id: str
    retrieval_mode: Literal["vector", "bm25", "hybrid"]
    query_template_version: str
    query_instruction: str
    requested_top_k: int
    candidate_k: int
    candidate_count: int
    vector_candidate_count: int
    bm25_candidate_count: int
    threshold_filtered_count: int
    deduplicated_count: int
    result_count: int
    query_token_count: int
    query_input_token_count: int
    query_input_hash: str
    lexical_query_terms: tuple[str, ...]
    lexical_ready: bool
    retrieval_degraded: bool
    results: tuple[SearchResult, ...]
    reranker_enabled: bool = False
    reranker_model_id: str | None = None
    reranker_model_revision: str | None = None
    reranker_input_template_version: str | None = None
    reranker_score_semantics: str | None = None
    rerank_top_n: int = 0
    rerank_candidate_count: int = 0
    final_top_k: int = 0
    context_token_budget: int = 0
    context_token_count: int = 0
    evidence_bundle: EvidenceBundle | None = None
    evidence_signals: EvidenceSignals | None = None
    shadow_observation: Mapping[str, object] | None = None
    graph_trace: Mapping[str, Any] | None = None
    guard_trace: Mapping[str, Any] | None = None
    runtime_trace: Mapping[str, Any] | None = None
