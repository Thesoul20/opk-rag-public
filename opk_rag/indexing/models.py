from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from opk_rag.vault.models import ExistingDocumentState, FileSnapshot, ScanFailure


@dataclass(frozen=True)
class DocumentUpdatePlan:
    existing: ExistingDocumentState
    snapshot: FileSnapshot


@dataclass(frozen=True)
class DocumentSyncStats:
    discovered: int
    added: int
    modified: int
    deleted: int
    unchanged: int
    restored: int
    failures: int

    @property
    def actionable(self) -> int:
        return self.added + self.modified + self.deleted + self.restored


@dataclass(frozen=True)
class DocumentSyncPlan:
    knowledge_base_id: UUID | None
    added: tuple[FileSnapshot, ...]
    modified: tuple[DocumentUpdatePlan, ...]
    deleted: tuple[ExistingDocumentState, ...]
    unchanged: tuple[ExistingDocumentState, ...]
    restored: tuple[DocumentUpdatePlan, ...]
    scan_failures: tuple[ScanFailure, ...]
    deletion_suppressed: bool
    dry_run: bool = False

    @property
    def stats(self) -> DocumentSyncStats:
        return DocumentSyncStats(
            discovered=len(self.added) + len(self.modified) + len(self.unchanged) + len(self.restored),
            added=len(self.added),
            modified=len(self.modified),
            deleted=len(self.deleted),
            unchanged=len(self.unchanged),
            restored=len(self.restored),
            failures=len(self.scan_failures),
        )


@dataclass(frozen=True)
class DocumentSyncResult:
    index_run_id: UUID | None
    knowledge_base_id: UUID | None
    plan: DocumentSyncPlan
    run_status: str
    dry_run: bool
