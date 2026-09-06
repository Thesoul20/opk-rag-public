from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from opk_rag.vault.ignore import IgnoreRules
from opk_rag.vault.models import FileSnapshot, ScanFailure, ScanResult, ScanStats
from opk_rag.vault.paths import normalize_vault_root

HashFunction = Callable[[Path], str]


@dataclass(frozen=True)
class VaultScanConfig:
    supported_extensions: tuple[str, ...] = (".md",)
    ignore_rules: IgnoreRules = IgnoreRules()
    follow_directory_symlinks: bool = False
    follow_file_symlinks: bool = False
    hash_chunk_size: int = 1024 * 1024
    max_hash_attempts: int = 2

    def normalized_extensions(self) -> tuple[str, ...]:
        return tuple(sorted({extension.lower() for extension in self.supported_extensions}))


def hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class VaultScanner:
    def __init__(self, config: VaultScanConfig | None = None, hash_function: HashFunction | None = None) -> None:
        self.config = config or VaultScanConfig()
        self._hash_function = hash_function

    def scan(self, vault_path: str | Path) -> ScanResult:
        root = normalize_vault_root(vault_path)
        files: list[FileSnapshot] = []
        failures: list[ScanFailure] = []
        stats = ScanStats(directories_seen=1)

        stats = self._walk_directory(root.path, root.path, files, failures, stats)

        files.sort(key=lambda snapshot: snapshot.relative_path)
        failures.sort(key=lambda failure: (failure.relative_path, failure.stage, failure.error_type))

        stats = replace(
            stats,
            snapshots_created=len(files),
            failures_seen=len(failures),
        )

        return ScanResult(
            root=root,
            files=tuple(files),
            failures=tuple(failures),
            stats=stats,
            ignored_dir_names=tuple(sorted(self.config.ignore_rules.ignored_dir_names)),
            supported_extensions=self.config.normalized_extensions(),
            follow_directory_symlinks=self.config.follow_directory_symlinks,
            follow_file_symlinks=self.config.follow_file_symlinks,
        )

    def _walk_directory(
        self,
        root: Path,
        directory: Path,
        files: list[FileSnapshot],
        failures: list[ScanFailure],
        stats: ScanStats,
    ) -> ScanStats:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            failures.append(self._failure(root, directory, "discover", exc))
            return stats

        for entry in entries:
            entry_path = Path(entry.path)
            relative_path = self._relative_path(root, entry_path)

            if entry.is_symlink():
                if not os.path.exists(entry.path):
                    failures.append(
                        ScanFailure(
                            relative_path=relative_path,
                            stage="discover",
                            error_type="broken_symlink",
                            message="Symbolic link target does not exist.",
                        )
                    )
                else:
                    stats = replace(stats, ignored_entries=stats.ignored_entries + 1)
                continue

            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                failures.append(self._failure(root, entry_path, "metadata", exc))
                continue

            if is_dir:
                if self.config.ignore_rules.ignores_dir(entry.name, relative_path):
                    stats = replace(stats, ignored_entries=stats.ignored_entries + 1)
                    continue
                stats = replace(stats, directories_seen=stats.directories_seen + 1)
                stats = self._walk_directory(root, entry_path, files, failures, stats)
                continue

            if not is_file:
                stats = replace(stats, ignored_entries=stats.ignored_entries + 1)
                continue

            stats = replace(stats, candidate_files_seen=stats.candidate_files_seen + 1)
            if self.config.ignore_rules.ignores_file(entry.name, relative_path):
                stats = replace(stats, ignored_entries=stats.ignored_entries + 1)
                continue

            if entry_path.suffix.lower() not in self.config.normalized_extensions():
                continue

            stats = replace(stats, markdown_files_seen=stats.markdown_files_seen + 1)
            snapshot = self._snapshot_file(root, entry_path, relative_path, failures)
            if snapshot is not None:
                files.append(snapshot)

        return stats

    def _snapshot_file(
        self,
        root: Path,
        path: Path,
        relative_path: str,
        failures: list[ScanFailure],
    ) -> FileSnapshot | None:
        attempts = max(1, self.config.max_hash_attempts)
        for attempt in range(attempts):
            try:
                before = path.stat()
                content_hash = self._hash(path)
                after = path.stat()
            except OSError as exc:
                failures.append(self._failure(root, path, "hash", exc))
                return None

            if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                return FileSnapshot(
                    relative_path=relative_path,
                    absolute_path=path,
                    file_name=path.name,
                    size_bytes=after.st_size,
                    modified_at=datetime.fromtimestamp(after.st_mtime, tz=timezone.utc),
                    content_hash=content_hash,
                )

            if attempt == attempts - 1:
                failures.append(
                    ScanFailure(
                        relative_path=relative_path,
                        stage="hash",
                        error_type="unstable_file",
                        message="File changed while it was being hashed.",
                    )
                )
                return None

        return None

    def _hash(self, path: Path) -> str:
        if self._hash_function is not None:
            return self._hash_function(path)
        return hash_file_sha256(path, chunk_size=self.config.hash_chunk_size)

    @staticmethod
    def _relative_path(root: Path, path: Path) -> str:
        return path.relative_to(root).as_posix()

    @staticmethod
    def _failure(root: Path, path: Path, stage: str, exc: OSError) -> ScanFailure:
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.name
        return ScanFailure(
            relative_path=relative_path,
            stage=stage,
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
