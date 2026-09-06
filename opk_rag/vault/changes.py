from __future__ import annotations

from collections.abc import Iterable

from opk_rag.vault.models import ChangeSet, ExistingDocumentState, FileSnapshot


def detect_changes(
    current_files: Iterable[FileSnapshot],
    existing_documents: Iterable[ExistingDocumentState],
) -> ChangeSet:
    current_by_path = {snapshot.relative_path: snapshot for snapshot in current_files}
    existing_by_path = {state.relative_path: state for state in existing_documents}

    added = []
    modified = []
    unchanged = []
    deleted = []

    for relative_path in sorted(current_by_path):
        snapshot = current_by_path[relative_path]
        existing = existing_by_path.get(relative_path)
        if existing is None:
            added.append(snapshot)
        elif snapshot.content_hash == existing.content_hash:
            unchanged.append(snapshot)
        else:
            modified.append(snapshot)

    for relative_path in sorted(existing_by_path):
        if relative_path not in current_by_path:
            deleted.append(existing_by_path[relative_path])

    return ChangeSet(
        added=tuple(added),
        modified=tuple(modified),
        deleted=tuple(deleted),
        unchanged=tuple(unchanged),
    )
