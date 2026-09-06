from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

from opk_rag.embedding.config import DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_REVISION


METADATA_CACHE_FILES: tuple[str, ...] = (
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODEL_ARTIFACT_CACHE_FILES: tuple[str, ...] = ("model.safetensors",)


REQUIRED_QWEN3_EMBEDDING_CACHE_FILES: tuple[str, ...] = METADATA_CACHE_FILES + MODEL_ARTIFACT_CACHE_FILES

INCOMPLETE_SUFFIXES: tuple[str, ...] = (".incomplete", ".tmp", ".part")


@dataclass(frozen=True)
class EmbeddingModelCacheAuthority:
    model_id: str = DEFAULT_EMBEDDING_MODEL_NAME
    revision: str | None = DEFAULT_EMBEDDING_MODEL_REVISION
    cache_dir: Path | None = None
    required_files: Sequence[str] = REQUIRED_QWEN3_EMBEDDING_CACHE_FILES

    def audit(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        environment = env or os.environ
        cache_root = resolve_huggingface_hub_cache(self.cache_dir, environment)
        snapshot_path = resolve_snapshot_path(cache_root, self.model_id, self.revision)
        exists = snapshot_path.exists()
        resolved_revision = snapshot_path.name if exists and snapshot_path.name != "<unresolved>" else None
        revision_match = bool(self.revision and resolved_revision == self.revision)
        missing_files: list[str] = []
        zero_byte_files: list[str] = []
        broken_symlinks: list[str] = []
        for relative in self.required_files:
            path = snapshot_path / relative
            if not path.exists():
                missing_files.append(relative)
                if path.is_symlink():
                    broken_symlinks.append(relative)
                continue
            try:
                if path.stat().st_size <= 0:
                    zero_byte_files.append(relative)
            except OSError:
                missing_files.append(relative)
        incomplete_files = sorted(
            str(path.relative_to(snapshot_path))
            for path in snapshot_path.rglob("*")
            if exists and path.is_file() and path.name.endswith(INCOMPLETE_SUFFIXES)
        ) if exists else []
        required_metadata_present = not any(relative in missing_files for relative in METADATA_CACHE_FILES)
        required_model_artifacts_present = not any(relative in missing_files for relative in MODEL_ARTIFACT_CACHE_FILES)
        complete = exists and revision_match and not missing_files and not zero_byte_files and not broken_symlinks and not incomplete_files
        return {
            "embedding_model_id": self.model_id,
            "embedding_model_revision": self.revision,
            "requested_revision": self.revision,
            "resolved_revision": resolved_revision,
            "revision_match": revision_match,
            "cache_root": str(cache_root),
            "snapshot_path": str(snapshot_path),
            "cache_snapshot_resolved": exists and resolved_revision is not None,
            "model_cache_exists": exists,
            "cache_exists": exists,
            "model_cache_complete": complete,
            "cache_complete": complete,
            "required_metadata_present": required_metadata_present,
            "required_model_artifacts_present": required_model_artifacts_present,
            "required_files": list(self.required_files),
            "required_metadata_files": list(METADATA_CACHE_FILES),
            "required_model_artifact_files": list(MODEL_ARTIFACT_CACHE_FILES),
            "missing_required_files": missing_files,
            "zero_byte_files": zero_byte_files,
            "broken_symlinks": broken_symlinks,
            "incomplete_files": incomplete_files,
            "cache_authority_valid": complete,
            "safe_to_use_local_files_only": complete,
        }


def resolve_huggingface_hub_cache(cache_dir: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser()
    environment = env or os.environ
    if environment.get("HUGGINGFACE_HUB_CACHE", "").strip():
        return Path(environment["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if environment.get("HF_HOME", "").strip():
        return Path(environment["HF_HOME"]).expanduser() / "hub"
    if environment.get("XDG_CACHE_HOME", "").strip():
        return Path(environment["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_snapshot_path(cache_root: Path, model_id: str, revision: str | None) -> Path:
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    if revision:
        return repo_dir / "snapshots" / revision
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        resolved = refs_main.read_text(encoding="utf-8").strip()
        if resolved:
            return repo_dir / "snapshots" / resolved
    return repo_dir / "snapshots" / "<unresolved>"
