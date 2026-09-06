"""Local Vault discovery and change detection."""

from opk_rag.vault.changes import detect_changes
from opk_rag.vault.models import (
    ChangeSet,
    ExistingDocumentState,
    FileSnapshot,
    ScanFailure,
    ScanResult,
    ScanStats,
    VaultRoot,
)
from opk_rag.vault.paths import VaultPathError, normalize_vault_root
from opk_rag.vault.scanner import VaultScanConfig, VaultScanner, hash_file_sha256

__all__ = [
    "ChangeSet",
    "ExistingDocumentState",
    "FileSnapshot",
    "ScanFailure",
    "ScanResult",
    "ScanStats",
    "VaultPathError",
    "VaultRoot",
    "VaultScanConfig",
    "VaultScanner",
    "detect_changes",
    "hash_file_sha256",
    "normalize_vault_root",
]
