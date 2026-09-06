from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from uuid import UUID

from opk_rag.db.connection import connect_postgres
from opk_rag.db.models import StoredChunk
from opk_rag.db.repositories import ChunkEmbeddingRecord, ChunkRepository, DocumentStateRepository, IndexConfigurationRepository
from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.input import EmbeddingInputError, PreparedEmbeddingInput, prepare_embedding_input
from opk_rag.embedding.provider import EmbeddingOutputValidationError, EmbeddingProvider


RETRYABLE_ERROR_NAMES = (
    "OperationalError",
    "InterfaceError",
)


@dataclass(frozen=True)
class EmbeddingRunResult:
    processed_documents: int = 0
    indexed_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0


def embed_pending_documents(
    database_url: str,
    configuration_fingerprint: str,
    provider: EmbeddingProvider,
    *,
    config: EmbeddingConfig,
    knowledge_base_id: UUID | None = None,
    max_documents: int | None = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.25,
) -> EmbeddingRunResult:
    with connect_postgres(database_url) as connection:
        index_config = IndexConfigurationRepository(connection).require_by_fingerprint(configuration_fingerprint)
        documents = DocumentStateRepository(connection).list_chunked_documents_for_embedding(
            index_config,
            knowledge_base_id=knowledge_base_id,
        )
        if max_documents is not None:
            documents = documents[:max_documents]

        total = EmbeddingRunResult()
        for document in documents:
            result = embed_document_chunks(
                connection,
                document.id,
                provider,
                config=config,
                configuration_fingerprint=configuration_fingerprint,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            total = _merge_results(total, result)
        return total


def embed_document_chunks(
    connection,
    document_id: UUID,
    provider: EmbeddingProvider,
    *,
    config: EmbeddingConfig,
    configuration_fingerprint: str,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.25,
) -> EmbeddingRunResult:
    index_config = IndexConfigurationRepository(connection).require_by_fingerprint(configuration_fingerprint)
    chunk_repo = ChunkRepository(connection)
    doc_repo = DocumentStateRepository(connection)
    chunks = chunk_repo.list_by_document(document_id)
    if not chunks:
        with connection.transaction():
            if chunk_repo.mark_document_indexed_if_complete(document_id, (), index_config, config):
                return EmbeddingRunResult(processed_documents=1, indexed_documents=1)
        return EmbeddingRunResult(processed_documents=1, failed_documents=1)

    try:
        prepared = tuple(prepare_embedding_input(chunk, config, count_tokens=provider.count_tokens) for chunk in chunks)
    except EmbeddingInputError as exc:
        connection.rollback()
        with connection.transaction():
            doc_repo.mark_embedding_failed(document_id, _format_error("embedding_input_invalid", exc))
        return EmbeddingRunResult(
            processed_documents=1,
            failed_documents=1,
            total_chunks=len(chunks),
            failed_chunks=len(chunks),
        )

    pending = tuple(
        item for item, chunk in zip(prepared, chunks, strict=True) if _needs_embedding(chunk, item, provider, config)
    )
    skipped_chunks = len(prepared) - len(pending)
    embedded_chunks = 0
    failed_chunks = 0

    for batch in _batches(pending, config.batch_size):
        try:
            vectors = provider.embed_documents([item.text for item in batch])
            _validate_vectors(vectors, expected_count=len(batch), dimension=config.dimension, normalize=config.normalize)
            records = tuple(
                ChunkEmbeddingRecord(
                    chunk_id=item.chunk_id,
                    embedding=tuple(float(value) for value in vector),
                    embedding_model=provider.model_id,
                    embedding_dimension=config.dimension,
                    token_count=item.content_token_count,
                    metadata=item.metadata,
                )
                for item, vector in zip(batch, vectors, strict=True)
            )
            _with_retries(
                lambda: _save_embedding_batch(connection, chunk_repo, records),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            embedded_chunks += len(batch)
        except EmbeddingOutputValidationError as exc:
            connection.rollback()
            failed_chunks += len(batch)
            _mark_batch_failed(connection, chunk_repo, doc_repo, document_id, batch, "embedding_validation_failed", exc)
        except Exception as exc:
            connection.rollback()
            failed_chunks += len(batch)
            _mark_batch_failed(connection, chunk_repo, doc_repo, document_id, batch, "embedding_inference_failed", exc)

    indexed = False
    if failed_chunks == 0:
        try:
            with connection.transaction():
                indexed = chunk_repo.mark_document_indexed_if_complete(document_id, prepared, index_config, config)
        except Exception as exc:
            connection.rollback()
            with connection.transaction():
                doc_repo.mark_embedding_failed(document_id, _format_error("embedding_persistence_failed", exc))
            failed_chunks = len(prepared) - embedded_chunks - skipped_chunks

    return EmbeddingRunResult(
        processed_documents=1,
        indexed_documents=1 if indexed else 0,
        failed_documents=0 if indexed else 1,
        total_chunks=len(prepared),
        embedded_chunks=embedded_chunks,
        skipped_chunks=skipped_chunks,
        failed_chunks=failed_chunks,
    )


def _needs_embedding(
    chunk: StoredChunk,
    prepared: PreparedEmbeddingInput,
    provider: EmbeddingProvider,
    config: EmbeddingConfig,
) -> bool:
    if chunk.embedding is None or chunk.embedding_model is None or chunk.embedding_dimension is None:
        return True
    if chunk.embedding_model != provider.model_id or chunk.embedding_dimension != config.dimension:
        return True
    if chunk.metadata.get("embedding_status") != "embedding_complete":
        return True
    if chunk.metadata.get("embedding_input_hash") != prepared.text_hash:
        return True
    if chunk.metadata.get("embedding_input_template_version") != config.input_template_version:
        return True
    if chunk.metadata.get("retrieval_representation_policy") != config.retrieval_representation_policy:
        return True
    if chunk.metadata.get("retrieval_text_digest") != prepared.retrieval_text_digest:
        return True
    if chunk.metadata.get("embedding_model_revision") != config.model_revision:
        return True
    try:
        _validate_vectors((chunk.embedding,), expected_count=1, dimension=config.dimension, normalize=config.normalize)
    except EmbeddingOutputValidationError:
        return True
    return False


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimension: int,
    normalize: bool,
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingOutputValidationError(f"Expected {expected_count} embedding vectors, got {len(vectors)}.")
    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise EmbeddingOutputValidationError(
                f"Embedding vector {index} has dimension {len(vector)}, expected {dimension}."
            )
        if not vector:
            raise EmbeddingOutputValidationError(f"Embedding vector {index} is empty.")
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values):
            raise EmbeddingOutputValidationError(f"Embedding vector {index} contains non-finite values.")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise EmbeddingOutputValidationError(f"Embedding vector {index} is all zeros.")
        if normalize and not math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            raise EmbeddingOutputValidationError(f"Embedding vector {index} is not normalized; norm={norm:.6f}.")


def _save_embedding_batch(connection, chunk_repo: ChunkRepository, records: tuple[ChunkEmbeddingRecord, ...]) -> None:
    with connection.transaction():
        chunk_repo.save_embedding_batch(records)
    connection.commit()


def _mark_batch_failed(
    connection,
    chunk_repo: ChunkRepository,
    doc_repo: DocumentStateRepository,
    document_id: UUID,
    batch: tuple[PreparedEmbeddingInput, ...],
    stage: str,
    exc: Exception,
) -> None:
    with connection.transaction():
        chunk_repo.mark_embedding_batch_failed(tuple(item.chunk_id for item in batch), _format_error(stage, exc))
        doc_repo.mark_embedding_failed(document_id, _format_error(stage, exc))


def _with_retries(action, *, max_retries: int, retry_backoff_seconds: float) -> None:
    attempts = 0
    while True:
        try:
            action()
            return
        except Exception as exc:
            if attempts >= max_retries or type(exc).__name__ not in RETRYABLE_ERROR_NAMES:
                raise
            attempts += 1
            time.sleep(retry_backoff_seconds * attempts)


def _batches(items: tuple[PreparedEmbeddingInput, ...], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _format_error(stage: str, exc: Exception) -> str:
    return f"{stage}: {type(exc).__name__}: {exc}"


def _merge_results(left: EmbeddingRunResult, right: EmbeddingRunResult) -> EmbeddingRunResult:
    return EmbeddingRunResult(
        processed_documents=left.processed_documents + right.processed_documents,
        indexed_documents=left.indexed_documents + right.indexed_documents,
        skipped_documents=left.skipped_documents + right.skipped_documents,
        failed_documents=left.failed_documents + right.failed_documents,
        total_chunks=left.total_chunks + right.total_chunks,
        embedded_chunks=left.embedded_chunks + right.embedded_chunks,
        skipped_chunks=left.skipped_chunks + right.skipped_chunks,
        failed_chunks=left.failed_chunks + right.failed_chunks,
    )
