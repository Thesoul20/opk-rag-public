from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import UUID

from opk_rag.chunking.chunker import chunk_markdown_document
from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.parser import MarkdownParseError, parse_markdown_file
from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import ChunkRepository, DocumentStateRepository, IndexConfigurationRepository
from opk_rag.vault import normalize_vault_root


@dataclass(frozen=True)
class ChunkSyncResult:
    processed: int
    skipped: int
    failed: int
    chunks_created: int
    chunks_deleted: int


def sync_document_chunks(
    database_url: str,
    vault_path: str | Path,
    configuration_fingerprint: str,
    *,
    config: ChunkingConfig | None = None,
    knowledge_base_id: UUID | None = None,
) -> ChunkSyncResult:
    root = normalize_vault_root(vault_path)
    config = config or ChunkingConfig()
    processed = 0
    failed = 0
    chunks_created = 0
    chunks_deleted = 0

    with connect_postgres(database_url) as connection:
        index_config = IndexConfigurationRepository(connection).require_by_fingerprint(configuration_fingerprint)
        documents = DocumentStateRepository(connection).list_pending_chunk_documents(
            index_config,
            knowledge_base_id=knowledge_base_id,
        )
        connection.commit()
        doc_repo = DocumentStateRepository(connection)
        chunk_repo = ChunkRepository(connection)

        for document in documents:
            source_path = root.path / document.relative_path
            try:
                _ensure_inside_root(source_path, root.path)
                if _file_sha256(source_path) != document.content_hash:
                    raise RuntimeError(f"Source file changed after scan: {document.relative_path}")
                parsed = parse_markdown_file(source_path)
                chunks = chunk_markdown_document(parsed, config)
                with connection.transaction():
                    deleted = chunk_repo.replace_document_chunks(document, index_config, chunks, parsed)
                    doc_repo.mark_chunked(document.id, parsed.title, parsed.frontmatter)
                connection.commit()
                processed += 1
                chunks_created += len(chunks)
                chunks_deleted += deleted
            except Exception as exc:
                connection.rollback()
                failed += 1
                message = _format_error(exc)
                with connection.transaction():
                    doc_repo.mark_failed(document.id, message)
                connection.commit()

    return ChunkSyncResult(
        processed=processed,
        skipped=0,
        failed=failed,
        chunks_created=chunks_created,
        chunks_deleted=chunks_deleted,
    )


def _format_error(exc: Exception) -> str:
    if isinstance(exc, MarkdownParseError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _ensure_inside_root(path: Path, root: Path) -> None:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
