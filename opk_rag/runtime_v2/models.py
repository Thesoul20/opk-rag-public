from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


RUNTIME_ARCHITECTURE_VERSION = "canonical_retrieval_runtime_v2"


@dataclass(frozen=True)
class CorpusSnapshotV2:
    corpus_snapshot_id: str
    corpus_digest: str
    source_manifest: dict[str, Any]
    chunking_contract_version: str
    embedding_contract_version: str

    @property
    def schema_version(self) -> str:
        return "opk-rag.canonical-retrieval-runtime-v2.corpus-snapshot.v1"

    def to_json(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class CanonicalChunkV2:
    canonical_chunk_id: str
    corpus_snapshot_id: str
    document_id: str | None
    section_id: str | None
    normalized_source_path: str | None
    heading_path: tuple[str, ...]
    source_start: int | None
    source_end: int | None
    content: str
    content_digest: str
    token_count: int | None = None

    @property
    def schema_version(self) -> str:
        return "opk-rag.canonical-retrieval-runtime-v2.chunk.v1"

    def to_json(self) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version, **asdict(self)}
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class RetrievalCandidateV2:
    canonical_chunk_id: str
    rank: int
    retrieval_strategy: str
    retrieval_score: float | None
    document_id: str | None
    section_id: str | None
    normalized_source_path: str | None
    heading_path: tuple[str, ...]
    source_start: int | None
    source_end: int | None
    content: str
    content_digest: str
    original_vector_rank: int | None = None
    reranker_score: float | None = None
    reranked_rank: int | None = None
    fusion_score: float | None = None
    fusion_rank: int | None = None
    ranking_policy: str | None = None

    @property
    def schema_version(self) -> str:
        return "opk-rag.canonical-retrieval-runtime-v2.retrieval-candidate.v1"

    def to_json(self) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version, **asdict(self)}
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class EvidenceContextV2:
    canonical_chunk_id: str
    text: str
    source_path: str | None
    heading_path: tuple[str, ...]
    provenance: dict[str, Any]
    rank: int
    citation_metadata: dict[str, Any]

    @property
    def schema_version(self) -> str:
        return "opk-rag.canonical-retrieval-runtime-v2.evidence-context.v1"

    def to_json(self) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version, **asdict(self)}
        payload["heading_path"] = list(self.heading_path)
        return payload
