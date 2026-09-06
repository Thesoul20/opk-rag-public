from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from psycopg.types.json import Jsonb

from opk_rag.db.models import (
    BM25SearchRow,
    ChunkSearchRow,
    DocumentForChunking,
    DocumentForEmbedding,
    IndexConfiguration,
    IndexRun,
    KnowledgeBase,
    LexicalIndexChunk,
    LexicalIndexStatus,
    LexicalReadiness,
    LexicalReplaceResult,
    StoredChunk,
)
from opk_rag.vault.models import ExistingDocumentState, FileSnapshot, ScanFailure

if TYPE_CHECKING:
    from opk_rag.chunking.models import MarkdownChunk, ParsedMarkdownDocument


class RepositoryError(RuntimeError):
    pass


class IndexConfigurationNotFoundError(RepositoryError):
    pass


@dataclass(frozen=True)
class IndexRunStats:
    documents_discovered: int
    documents_created: int
    documents_updated: int
    documents_deleted: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int = 0
    chunks_deleted: int = 0


@dataclass(frozen=True)
class ChunkEmbeddingRecord:
    chunk_id: UUID
    embedding: tuple[float, ...]
    embedding_model: str
    embedding_dimension: int
    token_count: int
    metadata: dict


class KnowledgeBaseRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def get_by_root_path(self, root_path: str) -> KnowledgeBase | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, name, root_path, description, created_at, updated_at
                from public.knowledge_bases
                where root_path = %s
                """,
                (root_path,),
            )
            row = cursor.fetchone()
        return _map_knowledge_base(row) if row is not None else None

    def get_by_id(self, knowledge_base_id: UUID) -> KnowledgeBase | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, name, root_path, description, created_at, updated_at
                from public.knowledge_bases
                where id = %s
                """,
                (knowledge_base_id,),
            )
            row = cursor.fetchone()
        return _map_knowledge_base(row) if row is not None else None

    def create(self, name: str, root_path: str, description: str | None = None) -> KnowledgeBase:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into public.knowledge_bases (name, root_path, description)
                values (%s, %s, %s)
                on conflict (root_path) do nothing
                returning id, name, root_path, description, created_at, updated_at
                """,
                (name, root_path, description),
            )
            row = cursor.fetchone()
        if row is None:
            existing = self.get_by_root_path(root_path)
            if existing is None:  # pragma: no cover - defensive race guard
                raise RepositoryError(f"Knowledge base was not created and cannot be found: {root_path}")
            return existing
        return _map_knowledge_base(row)

    def get_or_create(self, name: str, root_path: str, description: str | None = None) -> KnowledgeBase:
        existing = self.get_by_root_path(root_path)
        if existing is not None:
            return existing
        return self.create(name=name, root_path=root_path, description=description)


class DocumentStateRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def list_states(self, knowledge_base_id: UUID) -> tuple[ExistingDocumentState, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  id,
                  relative_path,
                  content_hash,
                  file_size,
                  source_modified_at,
                  index_status,
                  parser_version,
                  chunking_version
                from public.documents
                where knowledge_base_id = %s
                order by relative_path
                """,
                (knowledge_base_id,),
            )
            rows = cursor.fetchall()
        return tuple(_map_document_state(row) for row in rows)

    def get_state_by_path(self, knowledge_base_id: UUID, relative_path: str) -> ExistingDocumentState | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  id,
                  relative_path,
                  content_hash,
                  file_size,
                  source_modified_at,
                  index_status,
                  parser_version,
                  chunking_version
                from public.documents
                where knowledge_base_id = %s and relative_path = %s
                """,
                (knowledge_base_id, relative_path),
            )
            row = cursor.fetchone()
        return _map_document_state(row) if row is not None else None

    def insert_pending(
        self,
        knowledge_base_id: UUID,
        snapshot: FileSnapshot,
        parser_version: str,
        chunking_version: str,
    ) -> UUID:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into public.documents (
                  knowledge_base_id,
                  relative_path,
                  file_name,
                  content_hash,
                  file_size,
                  source_modified_at,
                  parser_version,
                  chunking_version,
                  index_status,
                  last_indexed_at,
                  last_error
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', null, null)
                returning id
                """,
                (
                    knowledge_base_id,
                    snapshot.relative_path,
                    snapshot.file_name,
                    snapshot.content_hash,
                    snapshot.size_bytes,
                    snapshot.modified_at,
                    parser_version,
                    chunking_version,
                ),
            )
            return cursor.fetchone()[0]

    def update_snapshot(
        self,
        document_id: UUID,
        snapshot: FileSnapshot,
        parser_version: str,
        chunking_version: str,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.documents
                set file_name = %s,
                    content_hash = %s,
                    file_size = %s,
                    source_modified_at = %s,
                    parser_version = %s,
                    chunking_version = %s,
                    index_status = 'pending',
                    last_error = null
                where id = %s
                """,
                (
                    snapshot.file_name,
                    snapshot.content_hash,
                    snapshot.size_bytes,
                    snapshot.modified_at,
                    parser_version,
                    chunking_version,
                    document_id,
                ),
            )

    def mark_deleted(self, document_id: UUID) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.documents
                set index_status = 'deleted'
                where id = %s and index_status <> 'deleted'
                """,
                (document_id,),
            )

    def list_pending_chunk_documents(
        self,
        index_configuration: IndexConfiguration,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> tuple[DocumentForChunking, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  id,
                  knowledge_base_id,
                  relative_path,
                  file_name,
                  content_hash,
                  parser_version,
                  chunking_version,
                  index_status
                from public.documents
                where parser_version = %s
                  and chunking_version = %s
                  and index_status = 'pending'
                  and (%s::uuid is null or knowledge_base_id = %s)
                order by relative_path
                """,
                (
                    index_configuration.parser_version,
                    index_configuration.chunking_version,
                    knowledge_base_id,
                    knowledge_base_id,
                ),
            )
            rows = cursor.fetchall()
        return tuple(_map_document_for_chunking(row) for row in rows)

    def list_chunked_documents_for_embedding(
        self,
        index_configuration: IndexConfiguration,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> tuple[DocumentForEmbedding, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select id, knowledge_base_id, relative_path, index_status
                from public.documents
                where parser_version = %s
                  and chunking_version = %s
                  and index_status = 'chunked'
                  and (%s::uuid is null or knowledge_base_id = %s)
                order by relative_path
                """,
                (
                    index_configuration.parser_version,
                    index_configuration.chunking_version,
                    knowledge_base_id,
                    knowledge_base_id,
                ),
            )
            rows = cursor.fetchall()
        return tuple(_map_document_for_embedding(row) for row in rows)

    def mark_chunked(self, document_id: UUID, title: str | None, frontmatter: dict) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.documents
                set title = %s,
                    index_status = 'chunked',
                    last_error = null,
                    metadata = metadata || %s::jsonb
                where id = %s
                """,
                (title, _json({"frontmatter": frontmatter}), document_id),
            )

    def mark_failed(self, document_id: UUID, error_message: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.documents
                set index_status = 'failed',
                    last_error = %s
                where id = %s
                """,
                (error_message, document_id),
            )

    def mark_embedding_failed(self, document_id: UUID, error_message: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.documents
                set last_error = %s,
                    metadata = metadata || %s::jsonb
                where id = %s and index_status = 'chunked'
                """,
                (error_message, _json({"embedding_error": error_message}), document_id),
            )


class IndexConfigurationRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def get_by_fingerprint(self, configuration_fingerprint: str) -> IndexConfiguration | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  id,
                  schema_version,
                  parser_version,
                  chunking_version,
                  embedding_provider,
                  embedding_model,
                  model_revision,
                  embedding_dimension,
                  normalize,
                  distance_metric,
                  configuration_fingerprint,
                  created_at
                from public.index_configurations
                where configuration_fingerprint = %s
                """,
                (configuration_fingerprint,),
            )
            row = cursor.fetchone()
        return _map_index_configuration(row) if row is not None else None

    def require_by_fingerprint(self, configuration_fingerprint: str) -> IndexConfiguration:
        config = self.get_by_fingerprint(configuration_fingerprint)
        if config is None:
            raise IndexConfigurationNotFoundError(
                f"Index configuration not found for fingerprint: {configuration_fingerprint}"
            )
        return config

    def create(self, config, configuration_fingerprint: str) -> IndexConfiguration:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into public.index_configurations (
                  schema_version,
                  parser_version,
                  chunking_version,
                  embedding_provider,
                  embedding_model,
                  model_revision,
                  embedding_dimension,
                  normalize,
                  distance_metric,
                  configuration_fingerprint
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (configuration_fingerprint) do nothing
                returning
                  id,
                  schema_version,
                  parser_version,
                  chunking_version,
                  embedding_provider,
                  embedding_model,
                  model_revision,
                  embedding_dimension,
                  normalize,
                  distance_metric,
                  configuration_fingerprint,
                  created_at
                """,
                (
                    config.schema_version,
                    config.parser_version,
                    config.chunking_version,
                    config.provider,
                    config.model_name,
                    config.model_revision,
                    config.dimension,
                    config.normalize,
                    config.distance_metric,
                    configuration_fingerprint,
                ),
            )
            row = cursor.fetchone()
        if row is None:
            return self.require_by_fingerprint(configuration_fingerprint)
        return _map_index_configuration(row)

    def get_or_create(self, config, configuration_fingerprint: str) -> IndexConfiguration:
        existing = self.get_by_fingerprint(configuration_fingerprint)
        if existing is not None:
            return existing
        return self.create(config, configuration_fingerprint)


class IndexRunRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def create_running(
        self,
        knowledge_base_id: UUID,
        index_configuration_id: UUID,
        run_type: str,
        stats: IndexRunStats,
        metadata: dict | None = None,
    ) -> IndexRun:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into public.index_runs (
                  knowledge_base_id,
                  index_configuration_id,
                  run_type,
                  status,
                  documents_discovered,
                  documents_created,
                  documents_updated,
                  documents_deleted,
                  documents_skipped,
                  documents_failed,
                  chunks_created,
                  chunks_deleted,
                  metadata
                )
                values (%s, %s, %s, 'running', %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                returning id, knowledge_base_id, index_configuration_id, run_type, status, started_at, finished_at
                """,
                (
                    knowledge_base_id,
                    index_configuration_id,
                    run_type,
                    stats.documents_discovered,
                    stats.documents_created,
                    stats.documents_updated,
                    stats.documents_deleted,
                    stats.documents_skipped,
                    stats.documents_failed,
                    stats.chunks_created,
                    stats.chunks_deleted,
                    _json(metadata or {}),
                ),
            )
            row = cursor.fetchone()
        return _map_index_run(row)

    def mark_partial(self, run_id: UUID, stats: IndexRunStats, metadata: dict | None = None) -> None:
        self._finish(run_id, "partial", stats, None, metadata)

    def mark_completed(self, run_id: UUID, stats: IndexRunStats, metadata: dict | None = None) -> None:
        self._finish(run_id, "completed", stats, None, metadata)

    def mark_failed(self, run_id: UUID, stats: IndexRunStats, error_message: str, metadata: dict | None = None) -> None:
        self._finish(run_id, "failed", stats, error_message, metadata)

    def record_failure(self, run_id: UUID, failure: ScanFailure, retryable: bool = False) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into public.index_failures (
                  run_id, relative_path, stage, error_type, error_message, retryable, metadata
                )
                values (%s, %s, %s, %s, %s, %s, '{}'::jsonb)
                """,
                (
                    run_id,
                    failure.relative_path,
                    failure.stage,
                    failure.error_type,
                    failure.message,
                    retryable,
                ),
            )

    def _finish(
        self,
        run_id: UUID,
        status: str,
        stats: IndexRunStats,
        error_message: str | None,
        metadata: dict | None,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.index_runs
                set status = %s,
                    finished_at = now(),
                    documents_discovered = %s,
                    documents_created = %s,
                    documents_updated = %s,
                    documents_deleted = %s,
                    documents_skipped = %s,
                    documents_failed = %s,
                    chunks_created = %s,
                    chunks_deleted = %s,
                    error_message = %s,
                    metadata = metadata || %s::jsonb
                where id = %s
                """,
                (
                    status,
                    stats.documents_discovered,
                    stats.documents_created,
                    stats.documents_updated,
                    stats.documents_deleted,
                    stats.documents_skipped,
                    stats.documents_failed,
                    stats.chunks_created,
                    stats.chunks_deleted,
                    error_message,
                    _json(metadata or {}),
                    run_id,
                ),
            )


class ChunkRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def list_by_document(self, document_id: UUID) -> tuple[StoredChunk, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  id,
                  document_id,
                  index_configuration_id,
                  chunk_index,
                  content,
                  content_hash,
                  heading_path,
                  start_line,
                  end_line,
                  token_count,
                  embedding,
                  embedding_model,
                  embedding_dimension,
                  metadata,
                  created_at,
                  updated_at
                from public.chunks
                where document_id = %s
                order by chunk_index
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
        return tuple(_map_stored_chunk(row) for row in rows)

    def delete_by_document(self, document_id: UUID) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("delete from public.chunks where document_id = %s", (document_id,))
            return cursor.rowcount

    def delete_for_deleted_documents(self, knowledge_base_id: UUID) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                delete from public.chunks c
                using public.documents d
                where c.document_id = d.id
                  and d.knowledge_base_id = %s
                  and d.index_status = 'deleted'
                """,
                (knowledge_base_id,),
            )
            return cursor.rowcount

    def replace_document_chunks(
        self,
        document: DocumentForChunking,
        index_configuration: IndexConfiguration,
        chunks: tuple["MarkdownChunk", ...],
        parsed_document: "ParsedMarkdownDocument",
    ) -> int:
        self._lock_document(document.id)
        deleted = self.delete_by_document(document.id)
        if not chunks:
            return deleted
        self._validate_chunk_sequence(chunks)
        rows = [
            (
                document.id,
                index_configuration.id,
                chunk.chunk_index,
                chunk.content,
                chunk.content_hash,
                list(chunk.heading_path),
                chunk.start_line,
                chunk.end_line,
                None,
                _json(
                    {
                        **chunk.metadata,
                        "character_count": chunk.char_count,
                        "content_hash": document.content_hash,
                        "document_title": parsed_document.title,
                        "frontmatter_keys": sorted(parsed_document.frontmatter.keys()),
                        "embedding_status": "embedding_pending",
                    }
                ),
            )
            for chunk in chunks
        ]
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                insert into public.chunks (
                  document_id,
                  index_configuration_id,
                  chunk_index,
                  content,
                  content_hash,
                  heading_path,
                  start_line,
                  end_line,
                  token_count,
                  metadata
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                rows,
            )
        return deleted

    def save_embedding_batch(self, records: tuple[ChunkEmbeddingRecord, ...]) -> None:
        if not records:
            return
        rows = [
            (
                _vector_literal(record.embedding),
                record.embedding_model,
                record.embedding_dimension,
                record.token_count,
                _json(
                    {
                        **record.metadata,
                        "embedding_status": "embedding_complete",
                        "embedding_completed_at": "now",
                    }
                ),
                record.chunk_id,
            )
            for record in records
        ]
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                update public.chunks
                set embedding = %s::extensions.vector,
                    embedding_model = %s,
                    embedding_dimension = %s,
                    token_count = %s,
                    metadata = metadata || (%s::jsonb - 'embedding_completed_at')
                               || jsonb_build_object('embedding_completed_at', now())
                where id = %s
                """,
                rows,
            )

    def mark_embedding_batch_failed(self, chunk_ids: tuple[UUID, ...], error_message: str) -> None:
        if not chunk_ids:
            return
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                update public.chunks
                set metadata = metadata || %s::jsonb
                where id = any(%s)
                """,
                (_json({"embedding_status": "embedding_failed", "embedding_error": error_message}), list(chunk_ids)),
            )

    def mark_document_indexed_if_complete(
        self,
        document_id: UUID,
        prepared_inputs,
        index_configuration: IndexConfiguration,
        config,
    ) -> bool:
        expected = tuple((item.chunk_id, item.text_hash) for item in prepared_inputs)
        self._lock_document(document_id)
        with self.connection.cursor() as cursor:
            if expected:
                cursor.execute(
                    """
                    with expected(chunk_id, embedding_input_hash) as (
                      select * from unnest(%s::uuid[], %s::text[])
                    ),
                    incomplete as (
                      select c.id
                      from public.chunks c
                      left join expected e on e.chunk_id = c.id
                      where c.document_id = %s
                        and (
                          e.chunk_id is null
                          or c.embedding is null
                          or c.embedding_model is distinct from %s
                          or c.embedding_dimension is distinct from %s
                          or c.index_configuration_id is distinct from %s
                          or c.metadata ->> 'embedding_status' is distinct from 'embedding_complete'
                          or c.metadata ->> 'embedding_input_hash' is distinct from e.embedding_input_hash
                          or c.metadata ->> 'embedding_input_template_version' is distinct from %s
                          or c.metadata ->> 'retrieval_representation_policy' is distinct from %s
                          or coalesce(c.metadata ->> 'embedding_model_revision', '') is distinct from %s
                          or c.metadata ->> 'embedding_provider' is distinct from %s
                        )
                    )
                    update public.documents
                    set index_status = 'indexed',
                        last_indexed_at = now(),
                        last_error = null,
                        metadata = metadata || %s::jsonb
                    where id = %s
                      and index_status = 'chunked'
                      and not exists (select 1 from incomplete)
                    returning id
                    """,
                    (
                        [str(chunk_id) for chunk_id, _ in expected],
                        [text_hash for _, text_hash in expected],
                        document_id,
                        config.model_name,
                        config.dimension,
                        index_configuration.id,
                        config.input_template_version,
                        config.retrieval_representation_policy,
                        config.model_revision or "",
                        config.provider,
                        _json({"embedding_status": "embedding_complete"}),
                        document_id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    update public.documents
                    set index_status = 'indexed',
                        last_indexed_at = now(),
                        last_error = null,
                        metadata = metadata || %s::jsonb
                    where id = %s
                      and index_status = 'chunked'
                      and not exists (select 1 from public.chunks where document_id = %s)
                    returning id
                    """,
                    (_json({"embedding_status": "embedding_complete"}), document_id, document_id),
                )
            return cursor.fetchone() is not None

    def _lock_document(self, document_id: UUID) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("select id from public.documents where id = %s for update", (document_id,))
            if cursor.fetchone() is None:
                raise RepositoryError(f"Document not found for chunk replacement: {document_id}")

    @staticmethod
    def _validate_chunk_sequence(chunks: tuple["MarkdownChunk", ...]) -> None:
        expected = tuple(range(len(chunks)))
        actual = tuple(chunk.chunk_index for chunk in chunks)
        if actual != expected:
            raise RepositoryError(f"Chunk indexes must be contiguous from 0: {actual}")


class ChunkSearchRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def search_chunks_by_vector(
        self,
        *,
        query_embedding,
        knowledge_base_id: UUID,
        match_count: int,
        index_configuration_id: UUID | None = None,
    ) -> tuple[ChunkSearchRow, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select document_id,
                       chunk_id,
                       relative_path,
                       heading_path,
                       content,
                       start_line,
                       end_line,
                       similarity
                from public.match_chunks(%s::extensions.vector, %s, %s, %s)
                """,
                (_vector_literal(tuple(float(value) for value in query_embedding)), knowledge_base_id, match_count, index_configuration_id),
            )
            rows = cursor.fetchall()
        return tuple(_map_chunk_search_row(row) for row in rows)

    def get_chunks_by_ids_for_search(
        self,
        *,
        chunk_ids: Sequence[UUID],
        knowledge_base_id: UUID,
        similarity_by_chunk_id: Mapping[UUID, float],
    ) -> tuple[ChunkSearchRow, ...]:
        if not chunk_ids:
            return ()
        unique_ids = tuple(dict.fromkeys(chunk_ids))
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select d.id as document_id,
                       c.id as chunk_id,
                       d.relative_path,
                       c.heading_path,
                       c.content,
                       c.start_line,
                       c.end_line
                from public.chunks c
                join public.documents d on d.id = c.document_id
                where d.knowledge_base_id = %s
                  and d.index_status = 'indexed'
                  and c.id = any(%s::uuid[])
                """,
                (knowledge_base_id, list(unique_ids)),
            )
            rows = cursor.fetchall()
        by_chunk_id = {
            row[1]: ChunkSearchRow(
                document_id=row[0],
                chunk_id=row[1],
                relative_path=row[2],
                heading_path=tuple(row[3] or ()),
                content=row[4],
                start_line=row[5],
                end_line=row[6],
                similarity=float(similarity_by_chunk_id[row[1]]),
            )
            for row in rows
            if row[1] in similarity_by_chunk_id
        }
        return tuple(by_chunk_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_chunk_id)


class LexicalIndexRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def list_chunks_requiring_lexical_index(
        self,
        *,
        knowledge_base_id: UUID,
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
        limit: int,
    ) -> tuple[LexicalIndexChunk, ...]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select c.id, d.id, d.knowledge_base_id, c.heading_path, c.content
                from public.chunks c
                join public.documents d on d.id = c.document_id
                left join public.chunk_lexical_statistics cls on cls.chunk_id = c.id
                left join lateral (
                  select coalesce(sum(term_frequency), 0)::integer as term_total,
                         count(*)::integer as term_rows
                  from public.chunk_lexical_terms clt
                  where clt.chunk_id = c.id
                ) term_summary on true
                where d.knowledge_base_id = %s
                  and d.index_status = 'indexed'
                  and (
                    cls.chunk_id is null
                    or cls.tokenizer_id is distinct from %s
                    or cls.tokenizer_version is distinct from %s
                    or cls.index_configuration_fingerprint is distinct from %s
                    or cls.document_length <= 0
                    or term_summary.term_rows = 0
                    or term_summary.term_total <> cls.document_length
                  )
                order by d.relative_path, c.chunk_index
                limit least(greatest(%s, 1), 1000)
                """,
                (knowledge_base_id, tokenizer_id, tokenizer_version, configuration_fingerprint, limit),
            )
            rows = cursor.fetchall()
        return tuple(_map_lexical_index_chunk(row) for row in rows)

    def get_lexical_index_status(
        self,
        *,
        knowledge_base_id: UUID,
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
    ) -> LexicalIndexStatus:
        readiness = self.get_lexical_readiness(
            knowledge_base_id=knowledge_base_id,
            tokenizer_id=tokenizer_id,
            tokenizer_version=tokenizer_version,
            configuration_fingerprint=configuration_fingerprint,
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  coalesce(kbls.indexed_chunk_count, 0)::integer,
                  coalesce(count(distinct kblt.term), 0)::integer,
                  coalesce(kbls.average_document_length, 0)::double precision
                from (select %s::uuid as knowledge_base_id) kb
                left join public.knowledge_base_lexical_statistics kbls
                  on kbls.knowledge_base_id = kb.knowledge_base_id
                 and kbls.tokenizer_id = %s
                 and kbls.tokenizer_version = %s
                 and kbls.index_configuration_fingerprint = %s
                left join public.knowledge_base_lexical_terms kblt
                  on kblt.knowledge_base_id = kb.knowledge_base_id
                group by kbls.indexed_chunk_count, kbls.average_document_length
                """,
                (knowledge_base_id, tokenizer_id, tokenizer_version, configuration_fingerprint),
            )
            row = cursor.fetchone()
        return LexicalIndexStatus(
            total_chunks=readiness.expected_chunk_count,
            pending_chunks=readiness.missing_chunk_count + readiness.stale_chunk_count,
            indexed_chunk_count=int(row[0]),
            unique_term_count=int(row[1]),
            average_document_length=float(row[2]),
            lexical_ready=readiness.ready,
        )

    def delete_knowledge_base_lexical_index(self, knowledge_base_id: UUID) -> int:
        deleted = 0
        with self.connection.cursor() as cursor:
            cursor.execute("delete from public.chunk_lexical_terms where knowledge_base_id = %s", (knowledge_base_id,))
            deleted += cursor.rowcount
            cursor.execute("delete from public.chunk_lexical_statistics where knowledge_base_id = %s", (knowledge_base_id,))
            deleted += cursor.rowcount
            cursor.execute("delete from public.knowledge_base_lexical_terms where knowledge_base_id = %s", (knowledge_base_id,))
            deleted += cursor.rowcount
            cursor.execute("delete from public.knowledge_base_lexical_statistics where knowledge_base_id = %s", (knowledge_base_id,))
            deleted += cursor.rowcount
        return deleted

    def replace_chunk_lexical_terms(
        self,
        *,
        chunk: LexicalIndexChunk,
        term_frequencies: Mapping[str, int],
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
    ) -> LexicalReplaceResult:
        terms = {term: int(freq) for term, freq in term_frequencies.items() if term.strip() and int(freq) > 0}
        if not terms:
            raise RepositoryError("Cannot index a chunk with no lexical terms.")
        with self.connection.cursor() as cursor:
            cursor.execute(
                "select count(*) from public.chunk_lexical_terms where chunk_id = %s",
                (chunk.chunk_id,),
            )
            removed_terms = int(cursor.fetchone()[0])
            cursor.execute("delete from public.chunk_lexical_terms where chunk_id = %s", (chunk.chunk_id,))
            rows = [
                (chunk.chunk_id, chunk.knowledge_base_id, term, frequency)
                for term, frequency in sorted(terms.items())
            ]
            cursor.executemany(
                """
                insert into public.chunk_lexical_terms (chunk_id, knowledge_base_id, term, term_frequency)
                values (%s, %s, %s, %s)
                """,
                rows,
            )
            cursor.execute(
                """
                insert into public.chunk_lexical_statistics (
                  chunk_id,
                  knowledge_base_id,
                  document_length,
                  tokenizer_id,
                  tokenizer_version,
                  index_configuration_fingerprint,
                  indexed_at
                )
                values (%s, %s, %s, %s, %s, %s, now())
                on conflict (chunk_id) do update
                set knowledge_base_id = excluded.knowledge_base_id,
                    document_length = excluded.document_length,
                    tokenizer_id = excluded.tokenizer_id,
                    tokenizer_version = excluded.tokenizer_version,
                    index_configuration_fingerprint = excluded.index_configuration_fingerprint,
                    indexed_at = now()
                """,
                (
                    chunk.chunk_id,
                    chunk.knowledge_base_id,
                    sum(terms.values()),
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                ),
            )
        return LexicalReplaceResult(new_terms=len(terms), removed_terms=removed_terms)

    def replace_chunks_lexical_terms(
        self,
        *,
        chunks: Sequence[tuple[LexicalIndexChunk, Mapping[str, int]]],
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
    ) -> LexicalReplaceResult:
        prepared: list[tuple[LexicalIndexChunk, dict[str, int]]] = []
        for chunk, frequencies in chunks:
            terms = {term: int(freq) for term, freq in frequencies.items() if term.strip() and int(freq) > 0}
            if not terms:
                raise RepositoryError("Cannot index a chunk with no lexical terms.")
            prepared.append((chunk, terms))
        if not prepared:
            return LexicalReplaceResult(new_terms=0, removed_terms=0)

        chunk_ids = [chunk.chunk_id for chunk, _ in prepared]
        term_rows = [
            {
                "chunk_id": str(chunk.chunk_id),
                "knowledge_base_id": str(chunk.knowledge_base_id),
                "term": term,
                "term_frequency": frequency,
            }
            for chunk, terms in prepared
            for term, frequency in sorted(terms.items())
        ]
        statistic_rows = [
            {
                "chunk_id": str(chunk.chunk_id),
                "knowledge_base_id": str(chunk.knowledge_base_id),
                "document_length": sum(terms.values()),
                "tokenizer_id": tokenizer_id,
                "tokenizer_version": tokenizer_version,
                "index_configuration_fingerprint": configuration_fingerprint,
            }
            for chunk, terms in prepared
        ]

        with self.connection.cursor() as cursor:
            cursor.execute(
                "select count(*) from public.chunk_lexical_terms where chunk_id = any(%s)",
                (chunk_ids,),
            )
            removed_terms = int(cursor.fetchone()[0])
            cursor.execute("delete from public.chunk_lexical_terms where chunk_id = any(%s)", (chunk_ids,))
            cursor.execute(
                """
                insert into public.chunk_lexical_terms (chunk_id, knowledge_base_id, term, term_frequency)
                select chunk_id, knowledge_base_id, term, term_frequency
                from jsonb_to_recordset(%s::jsonb) as rows(
                  chunk_id uuid,
                  knowledge_base_id uuid,
                  term text,
                  term_frequency integer
                )
                """,
                (Jsonb(term_rows),),
            )
            cursor.execute(
                """
                insert into public.chunk_lexical_statistics (
                  chunk_id,
                  knowledge_base_id,
                  document_length,
                  tokenizer_id,
                  tokenizer_version,
                  index_configuration_fingerprint,
                  indexed_at
                )
                select
                  chunk_id,
                  knowledge_base_id,
                  document_length,
                  tokenizer_id,
                  tokenizer_version,
                  index_configuration_fingerprint,
                  now()
                from jsonb_to_recordset(%s::jsonb) as rows(
                  chunk_id uuid,
                  knowledge_base_id uuid,
                  document_length integer,
                  tokenizer_id text,
                  tokenizer_version text,
                  index_configuration_fingerprint text
                )
                on conflict (chunk_id) do update
                set knowledge_base_id = excluded.knowledge_base_id,
                    document_length = excluded.document_length,
                    tokenizer_id = excluded.tokenizer_id,
                    tokenizer_version = excluded.tokenizer_version,
                    index_configuration_fingerprint = excluded.index_configuration_fingerprint,
                    indexed_at = now()
                """,
                (Jsonb(statistic_rows),),
            )
        return LexicalReplaceResult(new_terms=len(term_rows), removed_terms=removed_terms)

    def delete_non_indexed_chunk_statistics(self, knowledge_base_id: UUID) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                delete from public.chunk_lexical_statistics cls
                using public.chunks c
                join public.documents d on d.id = c.document_id
                where cls.chunk_id = c.id
                  and d.knowledge_base_id = %s
                  and d.index_status <> 'indexed'
                """,
                (knowledge_base_id,),
            )
            return cursor.rowcount

    def refresh_knowledge_base_lexical_statistics(
        self,
        *,
        knowledge_base_id: UUID,
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
    ) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (str(knowledge_base_id),))
            cursor.execute("delete from public.knowledge_base_lexical_terms where knowledge_base_id = %s", (knowledge_base_id,))
            cursor.execute(
                """
                insert into public.knowledge_base_lexical_terms (knowledge_base_id, term, document_frequency)
                select knowledge_base_id, term, count(*)::integer
                from public.chunk_lexical_terms
                where knowledge_base_id = %s
                group by knowledge_base_id, term
                having count(*) > 0
                """,
                (knowledge_base_id,),
            )
            cursor.execute(
                """
                insert into public.knowledge_base_lexical_statistics (
                  knowledge_base_id,
                  indexed_chunk_count,
                  total_document_length,
                  average_document_length,
                  tokenizer_id,
                  tokenizer_version,
                  index_configuration_fingerprint,
                  updated_at
                )
                select %s,
                       count(*)::integer,
                       coalesce(sum(document_length), 0)::integer,
                       coalesce(avg(document_length), 0)::double precision,
                       %s,
                       %s,
                       %s,
                       now()
                from public.chunk_lexical_statistics
                where knowledge_base_id = %s
                  and tokenizer_id = %s
                  and tokenizer_version = %s
                  and index_configuration_fingerprint = %s
                on conflict (knowledge_base_id) do update
                set indexed_chunk_count = excluded.indexed_chunk_count,
                    total_document_length = excluded.total_document_length,
                    average_document_length = excluded.average_document_length,
                    tokenizer_id = excluded.tokenizer_id,
                    tokenizer_version = excluded.tokenizer_version,
                    index_configuration_fingerprint = excluded.index_configuration_fingerprint,
                    updated_at = now()
                """,
                (
                    knowledge_base_id,
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                    knowledge_base_id,
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                ),
            )
            return True

    def get_lexical_readiness(
        self,
        *,
        knowledge_base_id: UUID,
        tokenizer_id: str,
        tokenizer_version: str,
        configuration_fingerprint: str,
    ) -> LexicalReadiness:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                with expected as (
                  select c.id
                  from public.chunks c
                  join public.documents d on d.id = c.document_id
                  where d.knowledge_base_id = %s
                    and d.index_status = 'indexed'
                ),
                current_stats as (
                  select cls.chunk_id
                  from public.chunk_lexical_statistics cls
                  where cls.knowledge_base_id = %s
                    and cls.tokenizer_id = %s
                    and cls.tokenizer_version = %s
                    and cls.index_configuration_fingerprint = %s
                    and cls.document_length > 0
                ),
                stale as (
                  select cls.chunk_id
                  from public.chunk_lexical_statistics cls
                  where cls.knowledge_base_id = %s
                    and (
                      cls.tokenizer_id is distinct from %s
                      or cls.tokenizer_version is distinct from %s
                      or cls.index_configuration_fingerprint is distinct from %s
                      or cls.document_length <= 0
                    )
                ),
                corpus as (
                  select *
                  from public.knowledge_base_lexical_statistics
                  where knowledge_base_id = %s
                    and tokenizer_id = %s
                    and tokenizer_version = %s
                    and index_configuration_fingerprint = %s
                )
                select
                  (select count(*) from current_stats)::integer,
                  (select count(*) from expected)::integer,
                  (select count(*) from stale)::integer,
                  (select count(*) from expected e left join current_stats cs on cs.chunk_id = e.id where cs.chunk_id is null)::integer,
                  exists(select 1 from corpus),
                  coalesce((select indexed_chunk_count from corpus), -1)::integer
                """,
                (
                    knowledge_base_id,
                    knowledge_base_id,
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                    knowledge_base_id,
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                    knowledge_base_id,
                    tokenizer_id,
                    tokenizer_version,
                    configuration_fingerprint,
                ),
            )
            row = cursor.fetchone()
        indexed_count = int(row[0])
        expected_count = int(row[1])
        stale_count = int(row[2])
        missing_count = int(row[3])
        corpus_present = bool(row[4])
        corpus_count = int(row[5])
        return LexicalReadiness(
            ready=expected_count > 0
            and indexed_count == expected_count
            and stale_count == 0
            and missing_count == 0
            and corpus_present
            and corpus_count == indexed_count,
            indexed_chunk_count=indexed_count,
            expected_chunk_count=expected_count,
            stale_chunk_count=stale_count,
            missing_chunk_count=missing_count,
            corpus_stats_present=corpus_present,
        )


class BM25SearchRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def search_chunks_by_bm25(
        self,
        *,
        query_terms: Sequence[str],
        knowledge_base_id: UUID,
        match_count: int,
        k1: float,
        b: float,
        configuration_fingerprint: str,
    ) -> tuple[BM25SearchRow, ...]:
        terms = tuple(dict.fromkeys(term for term in query_terms if term.strip()))
        if not terms:
            return ()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                select document_id,
                       chunk_id,
                       relative_path,
                       heading_path,
                       content,
                       start_line,
                       end_line,
                       bm25_score,
                       matched_terms
                from public.search_chunks_bm25(%s::text[], %s, %s, %s, %s, %s)
                """,
                (list(terms), knowledge_base_id, match_count, k1, b, configuration_fingerprint),
            )
            rows = cursor.fetchall()
        return tuple(_map_bm25_search_row(row) for row in rows)


def _json(value: dict) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def _map_knowledge_base(row) -> KnowledgeBase:
    return KnowledgeBase(
        id=row[0],
        name=row[1],
        root_path=row[2],
        description=row[3],
        created_at=row[4],
        updated_at=row[5],
    )


def _map_document_state(row) -> ExistingDocumentState:
    return ExistingDocumentState(
        document_id=row[0],
        relative_path=row[1],
        content_hash=row[2],
        file_size=row[3],
        source_modified_at=row[4],
        index_status=row[5],
        parser_version=row[6],
        chunking_version=row[7],
    )


def _map_document_for_chunking(row) -> DocumentForChunking:
    return DocumentForChunking(
        id=row[0],
        knowledge_base_id=row[1],
        relative_path=row[2],
        file_name=row[3],
        content_hash=row[4],
        parser_version=row[5],
        chunking_version=row[6],
        index_status=row[7],
    )


def _map_document_for_embedding(row) -> DocumentForEmbedding:
    return DocumentForEmbedding(
        id=row[0],
        knowledge_base_id=row[1],
        relative_path=row[2],
        index_status=row[3],
    )


def _map_index_configuration(row) -> IndexConfiguration:
    return IndexConfiguration(
        id=row[0],
        schema_version=row[1],
        parser_version=row[2],
        chunking_version=row[3],
        embedding_provider=row[4],
        embedding_model=row[5],
        model_revision=row[6],
        embedding_dimension=row[7],
        normalize=row[8],
        distance_metric=row[9],
        configuration_fingerprint=row[10],
        created_at=row[11],
    )


def _map_index_run(row) -> IndexRun:
    return IndexRun(
        id=row[0],
        knowledge_base_id=row[1],
        index_configuration_id=row[2],
        run_type=row[3],
        status=row[4],
        started_at=row[5],
        finished_at=row[6],
    )


def _map_stored_chunk(row) -> StoredChunk:
    return StoredChunk(
        id=row[0],
        document_id=row[1],
        index_configuration_id=row[2],
        chunk_index=row[3],
        content=row[4],
        content_hash=row[5],
        heading_path=tuple(row[6]),
        start_line=row[7],
        end_line=row[8],
        token_count=row[9],
        embedding=_map_vector(row[10]),
        embedding_model=row[11],
        embedding_dimension=row[12],
        metadata=dict(row[13]) if row[13] is not None else {},
        created_at=row[14],
        updated_at=row[15],
    )


def _map_chunk_search_row(row) -> ChunkSearchRow:
    return ChunkSearchRow(
        document_id=row[0],
        chunk_id=row[1],
        relative_path=row[2],
        heading_path=tuple(row[3] or ()),
        content=row[4],
        start_line=row[5],
        end_line=row[6],
        similarity=float(row[7]),
    )


def _map_lexical_index_chunk(row) -> LexicalIndexChunk:
    return LexicalIndexChunk(
        chunk_id=row[0],
        document_id=row[1],
        knowledge_base_id=row[2],
        heading_path=tuple(row[3] or ()),
        content=row[4],
    )


def _map_bm25_search_row(row) -> BM25SearchRow:
    return BM25SearchRow(
        document_id=row[0],
        chunk_id=row[1],
        relative_path=row[2],
        heading_path=tuple(row[3] or ()),
        content=row[4],
        start_line=row[5],
        end_line=row[6],
        bm25_score=float(row[7]),
        matched_terms=tuple(row[8] or ()),
    )


def _map_vector(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        if not stripped:
            return ()
        return tuple(float(part) for part in stripped.split(","))
    return tuple(float(part) for part in value)
