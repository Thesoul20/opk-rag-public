from __future__ import annotations

import os
from pathlib import Path

from opk_rag.vault.models import VaultRoot


class VaultPathError(ValueError):
    """Raised when a Vault root path is missing, invalid, or inaccessible."""


def normalize_vault_root(path: str | Path) -> VaultRoot:
    if isinstance(path, str):
        if not path.strip():
            raise VaultPathError("Vault path must not be empty.")
        original_path = path
    elif isinstance(path, Path):
        original_path = str(path)
    else:
        raise VaultPathError("Vault path must be a string or pathlib.Path.")

    try:
        expanded = Path(path).expanduser()
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise VaultPathError(f"Vault path does not exist: {Path(path).expanduser()}") from exc
    except RuntimeError as exc:
        raise VaultPathError(f"Vault path could not be resolved: {Path(path).expanduser()}") from exc
    except OSError as exc:
        raise VaultPathError(f"Vault path could not be resolved: {Path(path).expanduser()}") from exc

    if not resolved.is_dir():
        raise VaultPathError(f"Vault path is not a directory: {resolved}")

    if not os.access(resolved, os.R_OK | os.X_OK):
        raise VaultPathError(f"Vault path is not readable: {resolved}")

    return VaultRoot(path=resolved, canonical_path=resolved.as_posix(), original_path=original_path)
