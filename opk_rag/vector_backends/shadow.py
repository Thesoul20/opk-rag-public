from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
import time
from typing import Any, Literal

from opk_rag.db.models import ChunkSearchRow
from opk_rag.vector_backends.base import VectorBackend, VectorBackendCandidate, VectorBackendSearchFilter
from opk_rag.vector_backends.qdrant_backend import QdrantBackendConfig, QdrantVectorBackend

DifferenceCategory = Literal[
    "exact_match",
    "tie_order_difference",
    "score_precision_difference",
    "backend_search_difference",
    "candidate_loss",
    "candidate_addition",
    "identity_error",
    "shadow_failure",
    "unknown",
]


@dataclass(frozen=True)
class CanonicalVectorCandidate:
    chunk_id: str
    document_id: str
    rank: int
    canonical_similarity_score: float
    backend: str
    embedding_revision: str | None = None
    point_id: str | None = None


@dataclass(frozen=True)
class ShadowVectorConfig:
    authoritative_backend: str = "postgres_pgvector"
    shadow_backend: str = "qdrant"
    enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "opk_rag_shadow_v1"
    qdrant_prefer_grpc: bool = False
    qdrant_shadow_timeout_ms: int = 1000


def load_shadow_vector_config(env: Mapping[str, str] = os.environ) -> ShadowVectorConfig:
    return ShadowVectorConfig(
        authoritative_backend=env.get("OPK_RAG_VECTOR_BACKEND", "postgres_pgvector").strip() or "postgres_pgvector",
        shadow_backend=env.get("OPK_RAG_VECTOR_SHADOW_BACKEND", "qdrant").strip() or "qdrant",
        enabled=_bool_env(env.get("OPK_RAG_VECTOR_SHADOW_ENABLED", "false")),
        qdrant_url=env.get("OPK_RAG_QDRANT_URL", "http://localhost:6333").strip() or "http://localhost:6333",
        qdrant_api_key=env.get("OPK_RAG_QDRANT_API_KEY", "").strip() or None,
        qdrant_collection=env.get("OPK_RAG_VECTOR_SHADOW_COLLECTION", "opk_rag_shadow_v1").strip() or "opk_rag_shadow_v1",
        qdrant_prefer_grpc=_bool_env(env.get("OPK_RAG_QDRANT_PREFER_GRPC", "false")),
        qdrant_shadow_timeout_ms=int(env.get("OPK_RAG_QDRANT_SHADOW_TIMEOUT_MS", "1000")),
    )


def qdrant_backend_from_shadow_config(config: ShadowVectorConfig, *, vector_size: int) -> QdrantVectorBackend:
    return QdrantVectorBackend(
        QdrantBackendConfig(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection=config.qdrant_collection,
            vector_size=vector_size,
            distance="Cosine",
            prefer_grpc=config.qdrant_prefer_grpc,
            timeout_seconds=max(config.qdrant_shadow_timeout_ms / 1000.0, 0.001),
        )
    )


def canonical_candidates_from_pgvector(rows: Sequence[ChunkSearchRow]) -> tuple[CanonicalVectorCandidate, ...]:
    return tuple(
        CanonicalVectorCandidate(
            chunk_id=str(row.chunk_id),
            document_id=str(row.document_id),
            rank=rank,
            canonical_similarity_score=float(row.similarity),
            backend="postgres_pgvector",
        )
        for rank, row in enumerate(rows, start=1)
    )


def canonical_candidates_from_backend(candidates: Sequence[VectorBackendCandidate], *, backend: str) -> tuple[CanonicalVectorCandidate, ...]:
    return tuple(
        CanonicalVectorCandidate(
            chunk_id=str(candidate.chunk_id),
            document_id=str(candidate.document_id),
            rank=int(candidate.rank),
            canonical_similarity_score=float(candidate.canonical_similarity_score),
            backend=backend,
            embedding_revision=str(candidate.embedding_revision),
            point_id=str(candidate.point_id),
        )
        for candidate in candidates
    )


def compare_shadow_candidates(
    authoritative: Sequence[CanonicalVectorCandidate],
    shadow: Sequence[CanonicalVectorCandidate],
    *,
    score_precision_epsilon: float = 0.0001,
) -> dict[str, Any]:
    auth_by_id = {candidate.chunk_id: candidate for candidate in authoritative}
    shadow_by_id = {candidate.chunk_id: candidate for candidate in shadow}
    auth_ids = set(auth_by_id)
    shadow_ids = set(shadow_by_id)
    common_ids = sorted(auth_ids & shadow_ids)
    missing = sorted(auth_ids - shadow_ids)
    additional = sorted(shadow_ids - auth_ids)
    rank_deltas = [abs(auth_by_id[chunk_id].rank - shadow_by_id[chunk_id].rank) for chunk_id in common_ids]
    score_deltas = [
        abs(auth_by_id[chunk_id].canonical_similarity_score - shadow_by_id[chunk_id].canonical_similarity_score)
        for chunk_id in common_ids
    ]
    overlap_count = len(common_ids)
    denominator = max(len(auth_ids), 1)
    category = classify_shadow_difference(
        missing=missing,
        additional=additional,
        rank_deltas=rank_deltas,
        score_deltas=score_deltas,
        score_precision_epsilon=score_precision_epsilon,
    )
    return {
        "top_k_overlap_count": overlap_count,
        "top_k_overlap_ratio": round(overlap_count / denominator, 6),
        "missing_from_shadow": missing,
        "additional_in_shadow": additional,
        "candidate_missing_count": len(missing),
        "candidate_additional_count": len(additional),
        "mean_absolute_rank_delta": round(sum(rank_deltas) / len(rank_deltas), 6) if rank_deltas else 0.0,
        "max_rank_delta": max(rank_deltas) if rank_deltas else 0,
        "mean_absolute_score_delta": round(sum(score_deltas) / len(score_deltas), 9) if score_deltas else 0.0,
        "max_absolute_score_delta": round(max(score_deltas), 9) if score_deltas else 0.0,
        "difference_category": category,
    }


def classify_shadow_difference(
    *,
    missing: Sequence[str],
    additional: Sequence[str],
    rank_deltas: Sequence[int],
    score_deltas: Sequence[float],
    score_precision_epsilon: float,
) -> DifferenceCategory:
    if missing:
        return "candidate_loss"
    if additional:
        return "candidate_addition"
    if rank_deltas and max(rank_deltas) > 0:
        if score_deltas and max(score_deltas) <= score_precision_epsilon:
            return "tie_order_difference"
        return "backend_search_difference"
    if score_deltas and max(score_deltas) > score_precision_epsilon:
        return "backend_search_difference"
    if score_deltas and max(score_deltas) > 0:
        return "score_precision_difference"
    return "exact_match"


class ShadowVectorRetriever:
    def __init__(
        self,
        *,
        shadow_backend: VectorBackend,
        config: ShadowVectorConfig,
        observer: Callable[[Mapping[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.shadow_backend = shadow_backend
        self.config = config
        self.observer = observer
        self.clock = clock

    def observe(
        self,
        *,
        query_id: str,
        query_embedding: Sequence[float],
        authoritative_candidates: Sequence[CanonicalVectorCandidate],
        top_k: int,
        search_filter: VectorBackendSearchFilter | None,
    ) -> dict[str, Any]:
        started = self.clock()
        status = "success"
        shadow_candidates: tuple[CanonicalVectorCandidate, ...] = ()
        failure_type: str | None = None
        try:
            raw_shadow = self.shadow_backend.search(query_embedding, top_k=top_k, search_filter=search_filter)
            shadow_candidates = canonical_candidates_from_backend(raw_shadow, backend=self.config.shadow_backend)
            identity_errors = [
                candidate
                for candidate in shadow_candidates
                if not candidate.chunk_id or not candidate.document_id or candidate.chunk_id == "None" or candidate.document_id == "None"
            ]
            if identity_errors:
                status = "invalid_result"
                failure_type = "qdrant_identity_mapping_error"
                comparison = _failed_comparison("identity_error")
            else:
                comparison = compare_shadow_candidates(authoritative_candidates, shadow_candidates)
        except TimeoutError:
            status = "timeout"
            failure_type = "qdrant_timeout"
            comparison = _failed_comparison("shadow_failure")
        except Exception as exc:
            status = "failure"
            failure_type = _classify_shadow_exception(exc)
            comparison = _failed_comparison("shadow_failure")
        latency_ms = (self.clock() - started) * 1000.0
        record = {
            "query_id": query_id,
            "authoritative_backend": self.config.authoritative_backend,
            "shadow_backend": self.config.shadow_backend,
            "pgvector_candidate_count": len(authoritative_candidates),
            "qdrant_candidate_count": len(shadow_candidates),
            "top_k": top_k,
            "shadow_latency": round(latency_ms, 6),
            "shadow_status": status,
            "shadow_failure_type": failure_type,
            **comparison,
        }
        if self.observer is not None:
            self.observer(record)
        return record


def maybe_observe_qdrant_shadow(
    *,
    env: Mapping[str, str],
    query_id: str,
    query_embedding: Sequence[float],
    authoritative_rows: Sequence[ChunkSearchRow],
    top_k: int,
    embedding_revision: str | None,
    vector_size: int,
    observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    config = load_shadow_vector_config(env)
    if not config.enabled:
        return None
    if config.authoritative_backend != "postgres_pgvector" or config.shadow_backend != "qdrant":
        return {"query_id": query_id, "shadow_status": "disabled", "difference_category": "unknown"}
    backend = qdrant_backend_from_shadow_config(config, vector_size=vector_size)
    try:
        return ShadowVectorRetriever(shadow_backend=backend, config=config, observer=observer).observe(
            query_id=query_id,
            query_embedding=query_embedding,
            authoritative_candidates=canonical_candidates_from_pgvector(authoritative_rows),
            top_k=top_k,
            search_filter=VectorBackendSearchFilter(embedding_revision=embedding_revision) if embedding_revision else None,
        )
    finally:
        backend.close()


def _failed_comparison(category: DifferenceCategory) -> dict[str, Any]:
    return {
        "top_k_overlap_count": 0,
        "top_k_overlap_ratio": 0.0,
        "missing_from_shadow": [],
        "additional_in_shadow": [],
        "candidate_missing_count": 0,
        "candidate_additional_count": 0,
        "mean_absolute_rank_delta": 0.0,
        "max_rank_delta": 0,
        "mean_absolute_score_delta": 0.0,
        "max_absolute_score_delta": 0.0,
        "difference_category": category,
    }


def _classify_shadow_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timeout" in message or "timed out" in message:
        return "qdrant_timeout"
    if "connect" in name or "connect" in message or "unavailable" in message:
        return "qdrant_connection_error"
    if "keyerror" in name or "chunk_id" in message or "document_id" in message:
        return "qdrant_invalid_result"
    return "qdrant_query_error"


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")
