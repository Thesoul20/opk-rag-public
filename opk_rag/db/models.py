from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class KnowledgeBase:
    id: UUID
    name: str
    root_path: str
    description: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class IndexConfiguration:
    id: UUID
    schema_version: str
    parser_version: str
    chunking_version: str
    embedding_provider: str
    embedding_model: str
    model_revision: str | None
    embedding_dimension: int
    normalize: bool
    distance_metric: str
    configuration_fingerprint: str
    created_at: datetime


@dataclass(frozen=True)
class IndexRun:
    id: UUID
    knowledge_base_id: UUID
    index_configuration_id: UUID | None
    run_type: str
    status: str
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class DocumentForChunking:
    id: UUID
    knowledge_base_id: UUID
    relative_path: str
    file_name: str
    content_hash: str
    parser_version: str
    chunking_version: str
    index_status: str


@dataclass(frozen=True)
class DocumentForEmbedding:
    id: UUID
    knowledge_base_id: UUID
    relative_path: str
    index_status: str


@dataclass(frozen=True)
class StoredChunk:
    id: UUID
    document_id: UUID
    index_configuration_id: UUID
    chunk_index: int
    content: str
    content_hash: str
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None
    token_count: int | None
    embedding: object | None
    embedding_model: str | None
    embedding_dimension: int | None
    metadata: dict
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChunkSearchRow:
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    similarity: float


@dataclass(frozen=True)
class LexicalIndexChunk:
    chunk_id: UUID
    document_id: UUID
    knowledge_base_id: UUID
    heading_path: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class LexicalReplaceResult:
    new_terms: int
    removed_terms: int


@dataclass(frozen=True)
class LexicalReadiness:
    ready: bool
    indexed_chunk_count: int
    expected_chunk_count: int
    stale_chunk_count: int
    missing_chunk_count: int
    corpus_stats_present: bool


@dataclass(frozen=True)
class LexicalIndexStatus:
    total_chunks: int
    pending_chunks: int
    indexed_chunk_count: int
    unique_term_count: int
    average_document_length: float
    lexical_ready: bool


@dataclass(frozen=True)
class BM25SearchRow:
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    bm25_score: float
    matched_terms: tuple[str, ...]
