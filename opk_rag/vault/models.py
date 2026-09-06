from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class VaultRoot:
    path: Path
    canonical_path: str
    original_path: str


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    absolute_path: Path
    file_name: str
    size_bytes: int
    modified_at: datetime
    content_hash: str


@dataclass(frozen=True)
class ExistingDocumentState:
    relative_path: str
    content_hash: str
    file_size: int | None = None
    source_modified_at: datetime | None = None
    document_id: UUID | None = None
    index_status: str | None = None
    parser_version: str | None = None
    chunking_version: str | None = None


@dataclass(frozen=True)
class ScanFailure:
    relative_path: str
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ScanStats:
    directories_seen: int = 0
    candidate_files_seen: int = 0
    markdown_files_seen: int = 0
    ignored_entries: int = 0
    snapshots_created: int = 0
    failures_seen: int = 0


@dataclass(frozen=True)
class ScanResult:
    root: VaultRoot
    files: tuple[FileSnapshot, ...]
    failures: tuple[ScanFailure, ...]
    stats: ScanStats
    ignored_dir_names: tuple[str, ...]
    supported_extensions: tuple[str, ...]
    follow_directory_symlinks: bool
    follow_file_symlinks: bool


@dataclass(frozen=True)
class ChangeSet:
    added: tuple[FileSnapshot, ...]
    modified: tuple[FileSnapshot, ...]
    deleted: tuple[ExistingDocumentState, ...]
    unchanged: tuple[FileSnapshot, ...]
