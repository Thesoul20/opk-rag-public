from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EnvironmentLoadResult:
    env_path: Path
    found: bool
    loaded: bool
    loaded_keys: tuple[str, ...] = ()
    skipped_existing_keys: tuple[str, ...] = ()


def load_project_env(root: Path | None = None, *, env_filename: str = ".env") -> EnvironmentLoadResult:
    repository_root = Path(root).expanduser().resolve() if root is not None else _repository_root()
    env_path = repository_root / env_filename
    if not env_path.exists():
        return EnvironmentLoadResult(env_path=env_path, found=False, loaded=False)

    loaded_keys: list[str] = []
    skipped_existing_keys: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if not key:
            continue
        if key in os.environ:
            skipped_existing_keys.append(key)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded_keys.append(key)

    return EnvironmentLoadResult(
        env_path=env_path,
        found=True,
        loaded=True,
        loaded_keys=tuple(loaded_keys),
        skipped_existing_keys=tuple(skipped_existing_keys),
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
