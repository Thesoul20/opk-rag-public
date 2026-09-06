from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


DEFAULT_EMBEDDING_PROVIDER = "local_qwen"
DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_EMBEDDING_MAX_INPUT_TOKENS = 8192
DEFAULT_EMBEDDING_INPUT_TEMPLATE_VERSION = "embedding-input-v1"
DEFAULT_SCHEMA_VERSION = "2026-07-17"
DEFAULT_PARSER_VERSION = "markdown-parser-v1"
DEFAULT_CHUNKING_VERSION = "markdown-chunker-v1"
DEFAULT_RETRIEVAL_REPRESENTATION_POLICY = "content_only"
RETRIEVAL_REPRESENTATION_POLICIES = frozenset({"content_only", "heading_context_enriched"})
DEFAULT_MAX_HEADING_CONTEXT_TOKENS = 64


class EmbeddingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    model_revision: str | None = DEFAULT_EMBEDDING_MODEL_REVISION
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    max_input_tokens: int = DEFAULT_EMBEDDING_MAX_INPUT_TOKENS
    device: str = "auto"
    normalize: bool = True
    cache_dir: Path | None = None
    input_template_version: str = DEFAULT_EMBEDDING_INPUT_TEMPLATE_VERSION
    retrieval_representation_policy: str = DEFAULT_RETRIEVAL_REPRESENTATION_POLICY
    max_heading_context_tokens: int = DEFAULT_MAX_HEADING_CONTEXT_TOKENS
    schema_version: str = DEFAULT_SCHEMA_VERSION
    parser_version: str = DEFAULT_PARSER_VERSION
    chunking_version: str = DEFAULT_CHUNKING_VERSION
    distance_metric: str = "cosine"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.provider, "provider")
        _require_nonempty(self.model_name, "model_name")
        if self.model_revision is not None:
            _require_nonempty(self.model_revision, "model_revision")
        _require_nonempty(self.input_template_version, "input_template_version")
        _require_nonempty(self.retrieval_representation_policy, "retrieval_representation_policy")
        _require_nonempty(self.schema_version, "schema_version")
        _require_nonempty(self.parser_version, "parser_version")
        _require_nonempty(self.chunking_version, "chunking_version")
        if self.dimension != DEFAULT_EMBEDDING_DIMENSION:
            raise EmbeddingConfigError(f"Embedding dimension must be {DEFAULT_EMBEDDING_DIMENSION}.")
        if self.batch_size < 1:
            raise EmbeddingConfigError("batch_size must be >= 1.")
        if self.max_input_tokens < 1:
            raise EmbeddingConfigError("max_input_tokens must be >= 1.")
        if self.max_heading_context_tokens < 0:
            raise EmbeddingConfigError("max_heading_context_tokens must be >= 0.")
        if self.retrieval_representation_policy not in RETRIEVAL_REPRESENTATION_POLICIES:
            allowed = ", ".join(sorted(RETRIEVAL_REPRESENTATION_POLICIES))
            raise EmbeddingConfigError(f"retrieval_representation_policy must be one of: {allowed}.")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise EmbeddingConfigError("device must be one of: auto, cpu, mps, cuda.")
        if self.distance_metric != "cosine":
            raise EmbeddingConfigError("Only cosine distance is supported.")

    @property
    def model_id(self) -> str:
        if self.model_revision is None:
            return self.model_name
        return f"{self.model_name}@{self.model_revision}"

    @property
    def cache_dir_string(self) -> str | None:
        return str(self.cache_dir) if self.cache_dir is not None else None

    @property
    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "chunking_version": self.chunking_version,
            "embedding_provider": self.provider,
            "embedding_model": self.model_name,
            "model_revision": self.model_revision,
            "embedding_dimension": self.dimension,
            "normalize": self.normalize,
            "distance_metric": self.distance_metric,
            "input_template_version": self.input_template_version,
            "retrieval_representation_policy": self.retrieval_representation_policy,
            "max_heading_context_tokens": self.max_heading_context_tokens,
            "max_input_tokens": self.max_input_tokens,
        }


def load_embedding_config(env: Mapping[str, str] = os.environ) -> EmbeddingConfig:
    cache_dir = env.get("OPK_RAG_EMBEDDING_CACHE_DIR", "").strip()
    revision = env.get("OPK_RAG_EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_MODEL_REVISION).strip()
    model_name = (
        env.get("OPK_RAG_EMBEDDING_MODEL", "").strip()
        or env.get("OPK_RAG_EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip()
    )
    return EmbeddingConfig(
        provider=env.get("OPK_RAG_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip(),
        model_name=model_name,
        model_revision=revision or None,
        dimension=int(env.get("OPK_RAG_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))),
        batch_size=int(env.get("OPK_RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
        max_input_tokens=int(env.get("OPK_RAG_EMBEDDING_MAX_INPUT_TOKENS", str(DEFAULT_EMBEDDING_MAX_INPUT_TOKENS))),
        device=env.get("OPK_RAG_EMBEDDING_DEVICE", "auto").strip(),
        normalize=_bool_env(env.get("OPK_RAG_EMBEDDING_NORMALIZE", "true")),
        cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
        input_template_version=env.get(
            "OPK_RAG_EMBEDDING_INPUT_TEMPLATE_VERSION", DEFAULT_EMBEDDING_INPUT_TEMPLATE_VERSION
        ).strip(),
        retrieval_representation_policy=env.get(
            "OPK_RAG_RETRIEVAL_REPRESENTATION_POLICY", DEFAULT_RETRIEVAL_REPRESENTATION_POLICY
        ).strip(),
        max_heading_context_tokens=int(
            env.get("OPK_RAG_MAX_HEADING_CONTEXT_TOKENS", str(DEFAULT_MAX_HEADING_CONTEXT_TOKENS))
        ),
        schema_version=env.get("OPK_RAG_SCHEMA_VERSION", DEFAULT_SCHEMA_VERSION).strip(),
        parser_version=env.get("OPK_RAG_PARSER_VERSION", DEFAULT_PARSER_VERSION).strip(),
        chunking_version=env.get("OPK_RAG_CHUNKING_VERSION", DEFAULT_CHUNKING_VERSION).strip(),
        distance_metric=env.get("OPK_RAG_EMBEDDING_DISTANCE_METRIC", "cosine").strip(),
        local_files_only=_bool_env(env.get("OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY", "false")),
    )


def build_configuration_fingerprint(config: EmbeddingConfig) -> str:
    payload = json.dumps(config.fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_legacy_configuration_fingerprint(config: EmbeddingConfig) -> str:
    payload = dict(config.fingerprint_payload)
    payload.pop("retrieval_representation_policy", None)
    payload.pop("max_heading_context_tokens", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise EmbeddingConfigError(f"{field_name} must not be empty.")


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise EmbeddingConfigError(f"Invalid boolean value: {value}")
