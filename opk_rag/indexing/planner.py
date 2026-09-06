from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from opk_rag.indexing.models import DocumentSyncPlan, DocumentUpdatePlan
from opk_rag.vault import ExistingDocumentState, FileSnapshot, ScanFailure, ScanResult, detect_changes


def build_document_sync_plan(
    scan_result: ScanResult,
    existing_documents: Iterable[ExistingDocumentState],
    knowledge_base_id: UUID | None = None,
    dry_run: bool = False,
    parser_version: str | None = None,
    chunking_version: str | None = None,
) -> DocumentSyncPlan:
    existing = tuple(sorted(existing_documents, key=lambda state: state.relative_path))
    existing_by_path = {state.relative_path: state for state in existing}
    changes = detect_changes(scan_result.files, existing)
    failed_paths = _failed_file_paths(scan_result.failures)
    deletion_suppressed = _deletion_is_unsafe(scan_result.failures)

    added = []
    modified = []
    restored = []
    unchanged = []

    for snapshot in changes.added:
        added.append(snapshot)

    for snapshot in changes.modified:
        state = existing_by_path[snapshot.relative_path]
        if state.index_status == "deleted":
            restored.append(DocumentUpdatePlan(existing=state, snapshot=snapshot))
        else:
            modified.append(DocumentUpdatePlan(existing=state, snapshot=snapshot))

    for snapshot in changes.unchanged:
        state = existing_by_path[snapshot.relative_path]
        if state.index_status == "deleted":
            restored.append(DocumentUpdatePlan(existing=state, snapshot=snapshot))
        elif state.index_status == "failed":
            modified.append(DocumentUpdatePlan(existing=state, snapshot=snapshot))
        elif _version_mismatch(state, parser_version, chunking_version):
            modified.append(DocumentUpdatePlan(existing=state, snapshot=snapshot))
        else:
            unchanged.append(state)

    deleted = []
    if not deletion_suppressed:
        for state in changes.deleted:
            if state.index_status == "deleted":
                unchanged.append(state)
            elif state.relative_path not in failed_paths:
                deleted.append(state)
            else:
                unchanged.append(state)

    return DocumentSyncPlan(
        knowledge_base_id=knowledge_base_id,
        added=tuple(sorted(added, key=lambda snapshot: snapshot.relative_path)),
        modified=tuple(sorted(modified, key=lambda item: item.snapshot.relative_path)),
        deleted=tuple(sorted(deleted, key=lambda state: state.relative_path)),
        unchanged=tuple(sorted(unchanged, key=lambda state: state.relative_path)),
        restored=tuple(sorted(restored, key=lambda item: item.snapshot.relative_path)),
        scan_failures=tuple(sorted(scan_result.failures, key=lambda item: (item.relative_path, item.stage, item.error_type))),
        deletion_suppressed=deletion_suppressed,
        dry_run=dry_run,
    )


def _failed_file_paths(failures: tuple[ScanFailure, ...]) -> set[str]:
    return {failure.relative_path for failure in failures if failure.stage in {"hash", "metadata"}}


def _deletion_is_unsafe(failures: tuple[ScanFailure, ...]) -> bool:
    return any(failure.stage == "discover" for failure in failures)


def _version_mismatch(
    state: ExistingDocumentState,
    parser_version: str | None,
    chunking_version: str | None,
) -> bool:
    return (
        parser_version is not None
        and state.parser_version is not None
        and state.parser_version != parser_version
    ) or (
        chunking_version is not None
        and state.chunking_version is not None
        and state.chunking_version != chunking_version
    )
