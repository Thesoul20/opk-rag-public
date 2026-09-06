from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Literal, Mapping

from opk_rag.embedding.query import QUERY_INPUT_TEMPLATE_VERSION, QUERY_INSTRUCTION

DEFAULT_SEARCH_TOP_K = 5
DEFAULT_SEARCH_CANDIDATE_K = 20
DEFAULT_RERANK_TOP_N = 10
DEFAULT_SEARCH_MAX_TOP_K = 100
DEFAULT_QUERY_MAX_TOKENS = 512
DEFAULT_OVERLAP_DEDUPLICATION_THRESHOLD = 0.8
DEFAULT_RETRIEVAL_MODE = "vector"
DEFAULT_BM25_CANDIDATE_K = 20
DEFAULT_RRF_K = 60.0
DEFAULT_RRF_VECTOR_WEIGHT = 1.0
DEFAULT_RRF_BM25_WEIGHT = 1.0
DEFAULT_RERANK_ENABLED = True
DEFAULT_RERANKER_POLICY = "rank_fusion"
DEFAULT_RANK_FUSION_K = 60
DEFAULT_RANK_FUSION_LAMBDA = 0.75
DEFAULT_CONTEXT_TOKEN_BUDGET = 4096
DEFAULT_CONTEXT_MAX_CHUNKS = 5
DEFAULT_CONTEXT_MAX_CHUNKS_PER_DOCUMENT = 2


class VectorSearchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VectorSearchConfig:
    mode: Literal["vector", "bm25", "hybrid"] = DEFAULT_RETRIEVAL_MODE
    top_k: int = DEFAULT_SEARCH_TOP_K
    candidate_k: int = DEFAULT_SEARCH_CANDIDATE_K
    rerank_top_n: int = DEFAULT_RERANK_TOP_N
    bm25_candidate_k: int = DEFAULT_BM25_CANDIDATE_K
    min_similarity: float | None = None
    deduplicate: bool = True
    max_query_tokens: int = DEFAULT_QUERY_MAX_TOKENS
    query_instruction: str = QUERY_INSTRUCTION
    query_template_version: str = QUERY_INPUT_TEMPLATE_VERSION
    max_top_k: int = DEFAULT_SEARCH_MAX_TOP_K
    overlap_deduplication_threshold: float = DEFAULT_OVERLAP_DEDUPLICATION_THRESHOLD
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: float = DEFAULT_RRF_K
    rrf_vector_weight: float = DEFAULT_RRF_VECTOR_WEIGHT
    rrf_bm25_weight: float = DEFAULT_RRF_BM25_WEIGHT
    rerank_enabled: bool = DEFAULT_RERANK_ENABLED
    reranker_policy: Literal["rank_fusion"] = DEFAULT_RERANKER_POLICY
    rank_fusion_k: int = DEFAULT_RANK_FUSION_K
    rank_fusion_lambda: float = DEFAULT_RANK_FUSION_LAMBDA
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET
    context_max_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS
    context_max_chunks_per_document: int = DEFAULT_CONTEXT_MAX_CHUNKS_PER_DOCUMENT

    def __post_init__(self) -> None:
        if self.mode not in {"vector", "bm25", "hybrid"}:
            raise VectorSearchConfigError("mode must be one of: vector, bm25, hybrid.")
        if self.top_k < 1:
            raise VectorSearchConfigError("top_k must be >= 1.")
        if self.max_top_k < 1:
            raise VectorSearchConfigError("max_top_k must be >= 1.")
        if self.top_k > self.max_top_k:
            raise VectorSearchConfigError(f"top_k must be <= {self.max_top_k}.")
        if self.candidate_k < self.top_k:
            raise VectorSearchConfigError("candidate_k must be >= top_k.")
        if self.rerank_top_n < self.top_k:
            raise VectorSearchConfigError("rerank_top_n must be >= top_k.")
        if self.candidate_k < self.rerank_top_n:
            raise VectorSearchConfigError("candidate_k must be >= rerank_top_n.")
        if self.candidate_k > self.max_top_k:
            raise VectorSearchConfigError(f"candidate_k must be <= {self.max_top_k}.")
        if self.bm25_candidate_k < self.top_k:
            raise VectorSearchConfigError("bm25_candidate_k must be >= top_k.")
        if self.bm25_candidate_k < self.rerank_top_n:
            raise VectorSearchConfigError("bm25_candidate_k must be >= rerank_top_n.")
        if self.bm25_candidate_k > self.max_top_k:
            raise VectorSearchConfigError(f"bm25_candidate_k must be <= {self.max_top_k}.")
        if self.min_similarity is not None and not -1.0 <= self.min_similarity <= 1.0:
            raise VectorSearchConfigError("min_similarity must be between -1.0 and 1.0.")
        if self.max_query_tokens < 1:
            raise VectorSearchConfigError("max_query_tokens must be >= 1.")
        if not self.query_instruction.strip():
            raise VectorSearchConfigError("query_instruction must not be empty.")
        if not self.query_template_version.strip():
            raise VectorSearchConfigError("query_template_version must not be empty.")
        if not 0.0 <= self.overlap_deduplication_threshold <= 1.0:
            raise VectorSearchConfigError("overlap_deduplication_threshold must be between 0.0 and 1.0.")
        if self.bm25_k1 <= 0:
            raise VectorSearchConfigError("bm25_k1 must be > 0.")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise VectorSearchConfigError("bm25_b must be between 0.0 and 1.0.")
        if self.rrf_k <= 0:
            raise VectorSearchConfigError("rrf_k must be > 0.")
        if self.rrf_vector_weight < 0 or self.rrf_bm25_weight < 0:
            raise VectorSearchConfigError("RRF weights must be >= 0.")
        if self.mode == "hybrid" and self.rrf_vector_weight == 0 and self.rrf_bm25_weight == 0:
            raise VectorSearchConfigError("At least one RRF weight must be > 0.")
        if self.reranker_policy not in {"rank_fusion"}:
            raise VectorSearchConfigError("reranker_policy must be one of: rank_fusion.")
        if self.rank_fusion_k <= 0:
            raise VectorSearchConfigError("rank_fusion_k must be > 0.")
        if not math.isfinite(self.rank_fusion_lambda) or self.rank_fusion_lambda < 0:
            raise VectorSearchConfigError("rank_fusion_lambda must be finite and >= 0.")
        if self.context_token_budget < 1:
            raise VectorSearchConfigError("context_token_budget must be >= 1.")
        if self.context_max_chunks < 1:
            raise VectorSearchConfigError("context_max_chunks must be >= 1.")
        if self.context_max_chunks_per_document < 1:
            raise VectorSearchConfigError("context_max_chunks_per_document must be >= 1.")


def load_vector_search_config(env: Mapping[str, str] = os.environ) -> VectorSearchConfig:
    min_similarity = env.get("OPK_RAG_SEARCH_MIN_SIMILARITY", "").strip()
    return VectorSearchConfig(
        mode=env.get("OPK_RAG_SEARCH_MODE", DEFAULT_RETRIEVAL_MODE).strip(),
        top_k=int(env.get("OPK_RAG_SEARCH_TOP_K", str(DEFAULT_SEARCH_TOP_K))),
        candidate_k=int(env.get("OPK_RAG_SEARCH_CANDIDATE_K", str(DEFAULT_SEARCH_CANDIDATE_K))),
        rerank_top_n=int(env.get("OPK_RAG_RERANK_TOP_N", str(DEFAULT_RERANK_TOP_N))),
        bm25_candidate_k=int(env.get("OPK_RAG_SEARCH_BM25_CANDIDATE_K", str(DEFAULT_BM25_CANDIDATE_K))),
        min_similarity=float(min_similarity) if min_similarity else None,
        deduplicate=_bool_env(env.get("OPK_RAG_SEARCH_DEDUPLICATE", "true")),
        max_query_tokens=int(env.get("OPK_RAG_QUERY_MAX_TOKENS", str(DEFAULT_QUERY_MAX_TOKENS))),
        query_template_version=env.get("OPK_RAG_QUERY_TEMPLATE_VERSION", QUERY_INPUT_TEMPLATE_VERSION).strip(),
        bm25_k1=float(env.get("OPK_RAG_BM25_K1", "1.2")),
        bm25_b=float(env.get("OPK_RAG_BM25_B", "0.75")),
        rrf_k=float(env.get("OPK_RAG_RRF_K", str(DEFAULT_RRF_K))),
        rrf_vector_weight=float(env.get("OPK_RAG_RRF_VECTOR_WEIGHT", str(DEFAULT_RRF_VECTOR_WEIGHT))),
        rrf_bm25_weight=float(env.get("OPK_RAG_RRF_BM25_WEIGHT", str(DEFAULT_RRF_BM25_WEIGHT))),
        rerank_enabled=_bool_env(env.get("OPK_RAG_RERANK_ENABLED", str(DEFAULT_RERANK_ENABLED).lower())),
        reranker_policy=env.get("OPK_RAG_RERANKER_POLICY", DEFAULT_RERANKER_POLICY).strip(),
        rank_fusion_k=int(env.get("OPK_RAG_RANK_FUSION_K", str(DEFAULT_RANK_FUSION_K))),
        rank_fusion_lambda=float(env.get("OPK_RAG_RANK_FUSION_LAMBDA", str(DEFAULT_RANK_FUSION_LAMBDA))),
        context_token_budget=int(env.get("OPK_RAG_CONTEXT_TOKEN_BUDGET", str(DEFAULT_CONTEXT_TOKEN_BUDGET))),
        context_max_chunks=int(env.get("OPK_RAG_CONTEXT_MAX_CHUNKS", str(DEFAULT_CONTEXT_MAX_CHUNKS))),
        context_max_chunks_per_document=int(
            env.get("OPK_RAG_CONTEXT_MAX_CHUNKS_PER_DOCUMENT", str(DEFAULT_CONTEXT_MAX_CHUNKS_PER_DOCUMENT))
        ),
    )


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise VectorSearchConfigError(f"Invalid boolean value: {value}")
