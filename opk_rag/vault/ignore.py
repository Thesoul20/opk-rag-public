from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch


DEFAULT_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".obsidian",
        ".trash",
        ".Trash",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)

DEFAULT_IGNORED_FILE_NAMES = frozenset({".DS_Store"})


@dataclass(frozen=True)
class IgnoreRules:
    ignored_dir_names: frozenset[str] = DEFAULT_IGNORED_DIR_NAMES
    ignored_file_names: frozenset[str] = DEFAULT_IGNORED_FILE_NAMES
    ignored_globs: tuple[str, ...] = field(default_factory=tuple)
    ignore_hidden_dirs: bool = True

    def ignores_dir(self, name: str, relative_path: str) -> bool:
        if name in self.ignored_dir_names:
            return True
        if self.ignore_hidden_dirs and name.startswith("."):
            return True
        return self._matches_glob(relative_path)

    def ignores_file(self, name: str, relative_path: str) -> bool:
        if name in self.ignored_file_names:
            return True
        return self._matches_glob(relative_path)

    def _matches_glob(self, relative_path: str) -> bool:
        return any(fnmatch(relative_path, pattern) for pattern in self.ignored_globs)
