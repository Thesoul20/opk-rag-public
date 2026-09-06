from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from uuid import UUID

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.service import ChunkSyncResult, sync_document_chunks
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import ChunkRepository, IndexConfigurationRepository, IndexRunRepository, IndexRunStats
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.embedding.provider import EmbeddingProvider
from opk_rag.embedding.service import EmbeddingRunResult, embed_pending_documents
from opk_rag.indexing.models import DocumentSyncResult
from opk_rag.indexing.sync import sync_vault_documents
from opk_rag.lexical.config import LexicalIndexConfig
from opk_rag.lexical.service import LexicalIndexRunResult, index_knowledge_base_lexical
from opk_rag.vault import normalize_vault_root


@dataclass(frozen=True)
class IndexFailureSummary:
    relative_path: str
    stage: str
    message: str


@dataclass(frozen=True)
class EndToEndIndexReport:
    vault_path: str
    knowledge_base_id: UUID
    index_run_id: UUID | None
    configuration_fingerprint: str
    scanned_files: int
    added_files: int
    updated_files: int
    unchanged_files: int
    deleted_files: int
    restored_files: int
    failed_files: int
    documents_written: int
    chunks_written: int
    chunks_deleted: int
    embeddings_written: int
    embeddings_skipped: int
    lexical_records_updated: int
    lexical_records_removed: int
    lexical_ready: bool
    duration_seconds: float
    failures: tuple[IndexFailureSummary, ...]

    @property
    def successful(self) -> bool:
        return self.failed_files == 0 and not self.failures


def index_vault_end_to_end(
    database_url: str,
    vault_path: str | Path,
    *,
    embedding_config: EmbeddingConfig,
    embedding_provider: EmbeddingProvider,
    chunking_config: ChunkingConfig | None = None,
    lexical_config: LexicalIndexConfig | None = None,
    lexical_batch_size: int = 100,
    allow_create_knowledge_base: bool = True,
    knowledge_base_name: str | None = None,
    knowledge_base_description: str | None = None,
) -> EndToEndIndexReport:
    started = monotonic()
    root = normalize_vault_root(vault_path)
    fingerprint = build_configuration_fingerprint(embedding_config)
    _ensure_index_configuration(database_url, embedding_config, fingerprint)

    sync_result = sync_vault_documents(
        database_url,
        root.path,
        fingerprint,
        allow_create_knowledge_base=allow_create_knowledge_base,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_description=knowledge_base_description,
    )
    if sync_result.knowledge_base_id is None:
        raise RuntimeError("Document sync did not return a knowledge_base_id.")

    deleted_chunk_count = _delete_chunks_for_deleted_documents(database_url, sync_result.knowledge_base_id)
    chunk_result = sync_document_chunks(
        database_url,
        root.path,
        fingerprint,
        config=chunking_config,
        knowledge_base_id=sync_result.knowledge_base_id,
    )
    embedding_result = embed_pending_documents(
        database_url,
        fingerprint,
        embedding_provider,
        config=embedding_config,
        knowledge_base_id=sync_result.knowledge_base_id,
    )
    lexical_result = index_knowledge_base_lexical(
        database_url,
        knowledge_base_id=sync_result.knowledge_base_id,
        config=lexical_config or LexicalIndexConfig(),
        batch_size=lexical_batch_size,
    )

    failures = _failure_summaries(sync_result, chunk_result, embedding_result, lexical_result)
    failed_files = sync_result.plan.stats.failures + chunk_result.failed + embedding_result.failed_documents
    report = EndToEndIndexReport(
        vault_path=root.canonical_path,
        knowledge_base_id=sync_result.knowledge_base_id,
        index_run_id=sync_result.index_run_id,
        configuration_fingerprint=fingerprint,
        scanned_files=sync_result.plan.stats.discovered,
        added_files=sync_result.plan.stats.added,
        updated_files=sync_result.plan.stats.modified,
        unchanged_files=sync_result.plan.stats.unchanged,
        deleted_files=sync_result.plan.stats.deleted,
        restored_files=sync_result.plan.stats.restored,
        failed_files=failed_files,
        documents_written=sync_result.plan.stats.added + sync_result.plan.stats.modified + sync_result.plan.stats.restored,
        chunks_written=chunk_result.chunks_created,
        chunks_deleted=chunk_result.chunks_deleted + deleted_chunk_count,
        embeddings_written=embedding_result.embedded_chunks,
        embeddings_skipped=embedding_result.skipped_chunks,
        lexical_records_updated=lexical_result.indexed_chunks,
        lexical_records_removed=lexical_result.removed_terms,
        lexical_ready=lexical_result.ready,
        duration_seconds=monotonic() - started,
        failures=failures,
    )
    _finish_index_run(database_url, report)
    return report


def _ensure_index_configuration(database_url: str, config: EmbeddingConfig, fingerprint: str) -> None:
    with connect_postgres(database_url) as connection:
        with connection.transaction():
            IndexConfigurationRepository(connection).get_or_create(config, fingerprint)


def _delete_chunks_for_deleted_documents(database_url: str, knowledge_base_id: UUID) -> int:
    with connect_postgres(database_url) as connection:
        with connection.transaction():
            return ChunkRepository(connection).delete_for_deleted_documents(knowledge_base_id)


def _finish_index_run(database_url: str, report: EndToEndIndexReport) -> None:
    if report.index_run_id is None:
        return
    stats = IndexRunStats(
        documents_discovered=report.scanned_files,
        documents_created=report.added_files,
        documents_updated=report.updated_files + report.restored_files,
        documents_deleted=report.deleted_files,
        documents_skipped=report.unchanged_files,
        documents_failed=report.failed_files,
        chunks_created=report.chunks_written,
        chunks_deleted=report.chunks_deleted,
    )
    metadata = {
        "configuration_fingerprint": report.configuration_fingerprint,
        "embeddings_written": report.embeddings_written,
        "embeddings_skipped": report.embeddings_skipped,
        "lexical_records_updated": report.lexical_records_updated,
        "lexical_records_removed": report.lexical_records_removed,
        "lexical_ready": report.lexical_ready,
    }
    with connect_postgres(database_url) as connection:
        with connection.transaction():
            run_repo = IndexRunRepository(connection)
            if report.successful:
                run_repo.mark_completed(report.index_run_id, stats, metadata=metadata)
            else:
                run_repo.mark_partial(report.index_run_id, stats, metadata=metadata)


def _failure_summaries(
    sync_result: DocumentSyncResult,
    chunk_result: ChunkSyncResult,
    embedding_result: EmbeddingRunResult,
    lexical_result: LexicalIndexRunResult,
) -> tuple[IndexFailureSummary, ...]:
    failures = [
        IndexFailureSummary(
            relative_path=failure.relative_path,
            stage=failure.stage,
            message=f"{failure.error_type}: {failure.message}",
        )
        for failure in sync_result.plan.scan_failures
    ]
    if chunk_result.failed:
        failures.append(IndexFailureSummary("*", "chunk", f"{chunk_result.failed} document(s) failed during chunking."))
    if embedding_result.failed_documents:
        failures.append(
            IndexFailureSummary("*", "embedding", f"{embedding_result.failed_documents} document(s) failed during embedding.")
        )
    if lexical_result.failed_chunks:
        failures.append(IndexFailureSummary("*", "lexical", f"{lexical_result.failed_chunks} chunk(s) failed during lexical indexing."))
    return tuple(failures)
