from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class VectorBackendPoint:
    point_id: str
    chunk_id: str
    document_id: str
    embedding_revision: str
    vector: tuple[float, ...]
    payload: dict[str, object]


@dataclass(frozen=True)
class VectorBackendSearchFilter:
    knowledge_base_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    embedding_revision: str | None = None
    source_path: str | None = None

    def as_payload_conditions(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "knowledge_base_id": self.knowledge_base_id,
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "embedding_revision": self.embedding_revision,
                "source_path": self.source_path,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class VectorBackendCandidate:
    rank: int
    point_id: str
    chunk_id: str
    document_id: str
    embedding_revision: str
    raw_backend_score: float
    backend_score_semantics: str
    canonical_similarity_score: float
    payload: Mapping[str, object]


class VectorBackend(Protocol):
    backend_id: str

    def health_check(self) -> Mapping[str, object]:
        ...

    def initialize(self) -> None:
        ...

    def create_collection(self, *, recreate: bool = False) -> None:
        ...

    def collection_exists(self) -> bool:
        ...

    def upsert(self, points: Sequence[VectorBackendPoint], *, batch_size: int = 64) -> Mapping[str, object]:
        ...

    def delete(self, point_ids: Sequence[str]) -> Mapping[str, object]:
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        search_filter: VectorBackendSearchFilter | None = None,
        exact: bool = False,
    ) -> tuple[VectorBackendCandidate, ...]:
        ...

    def count(self) -> int:
        ...

    def close(self) -> None:
        ...


def canonical_similarity_score(score_semantics: str, raw_score: float) -> float:
    if not math.isfinite(raw_score):
        raise ValueError("raw_score must be finite")
    if score_semantics in {"pgvector_similarity", "qdrant_cosine_similarity"}:
        return raw_score
    if score_semantics == "cosine_distance":
        return 1.0 - raw_score
    raise ValueError(f"Unsupported score semantics: {score_semantics}")
