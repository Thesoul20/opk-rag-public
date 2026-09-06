from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import LexicalIndexRepository
from opk_rag.lexical.config import LexicalIndexConfig, build_lexical_configuration_fingerprint
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer, term_frequencies


@dataclass(frozen=True)
class LexicalIndexRunResult:
    total_chunks: int
    pending_chunks: int
    indexed_chunks: int
    skipped_chunks: int
    failed_chunks: int
    new_terms: int
    removed_terms: int
    indexed_chunk_count: int
    unique_term_count: int
    average_document_length: float
    tokenizer_id: str
    tokenizer_version: str
    configuration_fingerprint: str
    corpus_stats_changed: bool
    ready: bool


def index_knowledge_base_lexical(
    database_url: str,
    *,
    knowledge_base_id: UUID,
    config: LexicalIndexConfig | None = None,
    batch_size: int = 100,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> LexicalIndexRunResult:
    with connect_postgres(database_url) as connection:
        with connection.transaction():
            return index_knowledge_base_lexical_connection(
                connection,
                knowledge_base_id=knowledge_base_id,
                config=config or LexicalIndexConfig(),
                batch_size=batch_size,
                dry_run=dry_run,
                force_rebuild=force_rebuild,
            )


def index_knowledge_base_lexical_connection(
    connection,
    *,
    knowledge_base_id: UUID,
    config: LexicalIndexConfig,
    batch_size: int = 100,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> LexicalIndexRunResult:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    tokenizer = JiebaLexicalTokenizer()
    fingerprint = build_lexical_configuration_fingerprint(config)
    repo = LexicalIndexRepository(connection)
    if force_rebuild and not dry_run:
        repo.delete_knowledge_base_lexical_index(knowledge_base_id)
    if dry_run:
        status = repo.get_lexical_index_status(
            knowledge_base_id=knowledge_base_id,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_version=tokenizer.version,
            configuration_fingerprint=fingerprint,
        )
        return LexicalIndexRunResult(
            total_chunks=status.total_chunks,
            pending_chunks=status.pending_chunks,
            indexed_chunks=0,
            skipped_chunks=status.total_chunks - status.pending_chunks,
            failed_chunks=0,
            new_terms=0,
            removed_terms=0,
            indexed_chunk_count=status.indexed_chunk_count,
            unique_term_count=status.unique_term_count,
            average_document_length=status.average_document_length,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_version=tokenizer.version,
            configuration_fingerprint=fingerprint,
            corpus_stats_changed=False,
            ready=status.lexical_ready,
        )
    stale_deleted = repo.delete_non_indexed_chunk_statistics(knowledge_base_id)

    indexed = 0
    failed = 0
    new_terms = 0
    removed_terms = stale_deleted
    while True:
        pending = repo.list_chunks_requiring_lexical_index(
            knowledge_base_id=knowledge_base_id,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_version=tokenizer.version,
            configuration_fingerprint=fingerprint,
            limit=batch_size,
        )
        if not pending:
            break
        batch_indexed = 0
        batch_terms = []
        for chunk in pending:
            try:
                text = "\n".join((*chunk.heading_path, chunk.content))
                tokens = tokenizer.tokenize_document(text)
                if not tokens:
                    failed += 1
                    continue
                batch_terms.append((chunk, term_frequencies(tokens)))
            except Exception:
                failed += 1
        if batch_terms:
            try:
                result = repo.replace_chunks_lexical_terms(
                    chunks=batch_terms,
                    tokenizer_id=tokenizer.tokenizer_id,
                    tokenizer_version=tokenizer.version,
                    configuration_fingerprint=fingerprint,
                )
                indexed += len(batch_terms)
                batch_indexed += len(batch_terms)
                new_terms += result.new_terms
                removed_terms += result.removed_terms
            except Exception:
                for chunk, frequencies in batch_terms:
                    try:
                        result = repo.replace_chunk_lexical_terms(
                            chunk=chunk,
                            term_frequencies=frequencies,
                            tokenizer_id=tokenizer.tokenizer_id,
                            tokenizer_version=tokenizer.version,
                            configuration_fingerprint=fingerprint,
                        )
                        indexed += 1
                        batch_indexed += 1
                        new_terms += result.new_terms
                        removed_terms += result.removed_terms
                    except Exception:
                        failed += 1
        if batch_indexed == 0:
            break

    changed = repo.refresh_knowledge_base_lexical_statistics(
        knowledge_base_id=knowledge_base_id,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_version=tokenizer.version,
        configuration_fingerprint=fingerprint,
    )
    readiness = repo.get_lexical_readiness(
        knowledge_base_id=knowledge_base_id,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_version=tokenizer.version,
        configuration_fingerprint=fingerprint,
    )
    status = repo.get_lexical_index_status(
        knowledge_base_id=knowledge_base_id,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_version=tokenizer.version,
        configuration_fingerprint=fingerprint,
    )
    return LexicalIndexRunResult(
        total_chunks=status.total_chunks,
        pending_chunks=status.pending_chunks,
        indexed_chunks=indexed,
        skipped_chunks=max(status.total_chunks - indexed - status.pending_chunks, 0),
        failed_chunks=failed,
        new_terms=new_terms,
        removed_terms=removed_terms,
        indexed_chunk_count=status.indexed_chunk_count,
        unique_term_count=status.unique_term_count,
        average_document_length=status.average_document_length,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_version=tokenizer.version,
        configuration_fingerprint=fingerprint,
        corpus_stats_changed=changed,
        ready=readiness.ready,
    )
