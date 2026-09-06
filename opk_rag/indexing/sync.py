from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from opk_rag.db.connection import connect_postgres
from opk_rag.db.repositories import (
    DocumentStateRepository,
    IndexConfigurationRepository,
    IndexRunRepository,
    IndexRunStats,
    KnowledgeBaseRepository,
)
from opk_rag.indexing.models import DocumentSyncPlan, DocumentSyncResult
from opk_rag.indexing.planner import build_document_sync_plan
from opk_rag.vault import VaultScanner, normalize_vault_root

DEFAULT_RUN_TYPE = "incremental"


class KnowledgeBaseNotFoundError(RuntimeError):
    pass


class DocumentSyncError(RuntimeError):
    pass


def sync_vault_documents(
    database_url: str,
    vault_path: str | Path,
    configuration_fingerprint: str,
    *,
    allow_create_knowledge_base: bool = False,
    knowledge_base_name: str | None = None,
    knowledge_base_description: str | None = None,
    dry_run: bool = False,
    scanner: VaultScanner | None = None,
    failure_injector: Callable[[str], None] | None = None,
) -> DocumentSyncResult:
    root = normalize_vault_root(vault_path)
    scanner = scanner or VaultScanner()

    with connect_postgres(database_url) as connection:
        kb_repo = KnowledgeBaseRepository(connection)
        config_repo = IndexConfigurationRepository(connection)
        doc_repo = DocumentStateRepository(connection)

        index_config = config_repo.require_by_fingerprint(configuration_fingerprint)
        knowledge_base = kb_repo.get_by_root_path(root.canonical_path)
        if knowledge_base is None and not dry_run:
            if not allow_create_knowledge_base:
                raise KnowledgeBaseNotFoundError(f"Knowledge base not found for root path: {root.canonical_path}")
            knowledge_base = kb_repo.create(
                name=knowledge_base_name or root.path.name,
                root_path=root.canonical_path,
                description=knowledge_base_description,
            )
            connection.commit()

        existing_states = doc_repo.list_states(knowledge_base.id) if knowledge_base is not None else ()
        scan_result = scanner.scan(root.path)
        plan = build_document_sync_plan(
            scan_result,
            existing_states,
            knowledge_base_id=knowledge_base.id if knowledge_base is not None else None,
            dry_run=dry_run,
            parser_version=index_config.parser_version,
            chunking_version=index_config.chunking_version,
        )

        if dry_run:
            return DocumentSyncResult(
                index_run_id=None,
                knowledge_base_id=knowledge_base.id if knowledge_base is not None else None,
                plan=plan,
                run_status="dry_run",
                dry_run=True,
            )

        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge base not found for root path: {root.canonical_path}")

        run_repo = IndexRunRepository(connection)
        parser_version = index_config.parser_version
        chunking_version = index_config.chunking_version
        running_stats = _stats_for_run(plan)
        index_run = run_repo.create_running(
            knowledge_base_id=knowledge_base.id,
            index_configuration_id=index_config.id,
            run_type=DEFAULT_RUN_TYPE,
            stats=running_stats,
            metadata={"dry_run": False, "deletion_suppressed": plan.deletion_suppressed},
        )
        connection.commit()

        try:
            with connection.transaction():
                for snapshot in plan.added:
                    doc_repo.insert_pending(knowledge_base.id, snapshot, parser_version, chunking_version)
                for item in (*plan.modified, *plan.restored):
                    if item.existing.document_id is None:
                        raise DocumentSyncError(f"Existing document has no id: {item.existing.relative_path}")
                    doc_repo.update_snapshot(item.existing.document_id, item.snapshot, parser_version, chunking_version)
                for state in plan.deleted:
                    if state.document_id is None:
                        raise DocumentSyncError(f"Existing document has no id: {state.relative_path}")
                    doc_repo.mark_deleted(state.document_id)
                if failure_injector is not None:
                    failure_injector("before_run_partial")
                for failure in plan.scan_failures:
                    run_repo.record_failure(index_run.id, failure)
                run_repo.mark_partial(
                    index_run.id,
                    _stats_for_run(plan),
                    metadata={"restored": len(plan.restored), "deletion_suppressed": plan.deletion_suppressed},
                )
        except Exception as exc:
            connection.rollback()
            failed_stats = _stats_for_run(plan)
            with connection.transaction():
                run_repo.mark_failed(
                    index_run.id,
                    failed_stats,
                    _redact_error(str(exc)),
                    metadata={"failed_stage": "document_state_sync"},
                )
            raise

        return DocumentSyncResult(
            index_run_id=index_run.id,
            knowledge_base_id=knowledge_base.id,
            plan=plan,
            run_status="partial",
            dry_run=False,
        )


def _stats_for_run(plan: DocumentSyncPlan) -> IndexRunStats:
    stats = plan.stats
    return IndexRunStats(
        documents_discovered=stats.discovered,
        documents_created=stats.added,
        documents_updated=stats.modified + stats.restored,
        documents_deleted=stats.deleted,
        documents_skipped=stats.unchanged,
        documents_failed=stats.failures,
    )


def _redact_error(message: str) -> str:
    return message.replace("postgresql://", "postgresql://***@").replace("postgres://", "postgres://***@")
