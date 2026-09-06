from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Mapping

from opk_rag.reranking.input import RERANKER_INPUT_TEMPLATE_VERSION

DEFAULT_RERANKER_PROVIDER = "local_bge_cross_encoder"
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_PAIR_TOKENS = 1024
DEFAULT_SEARCH_RERANKER_PRECISION = "fp16_autocast"
DEFAULT_ASK_RERANKER_PRECISION = "fp32"

RerankerPrecision = Literal["fp32", "fp16_autocast"]
RerankerExecutionScope = Literal["search", "ask"]


class RerankerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = DEFAULT_RERANKER_PROVIDER
    model_name: str = DEFAULT_RERANKER_MODEL_NAME
    model_revision: str | None = DEFAULT_RERANKER_MODEL_REVISION
    batch_size: int = DEFAULT_RERANKER_BATCH_SIZE
    max_pair_tokens: int = DEFAULT_RERANKER_MAX_PAIR_TOKENS
    device: str = "auto"
    cache_dir: Path | None = None
    local_files_only: bool = False
    input_template_version: str = RERANKER_INPUT_TEMPLATE_VERSION
    search_precision: RerankerPrecision = DEFAULT_SEARCH_RERANKER_PRECISION
    ask_precision: RerankerPrecision = DEFAULT_ASK_RERANKER_PRECISION

    def __post_init__(self) -> None:
        _require_nonempty(self.provider, "provider")
        _require_nonempty(self.model_name, "model_name")
        if self.model_revision is not None:
            _require_nonempty(self.model_revision, "model_revision")
        if self.batch_size < 1:
            raise RerankerConfigError("batch_size must be >= 1.")
        if self.max_pair_tokens < 1:
            raise RerankerConfigError("max_pair_tokens must be >= 1.")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise RerankerConfigError("device must be one of: auto, cpu, mps, cuda.")
        _require_nonempty(self.input_template_version, "input_template_version")
        _validate_precision(self.search_precision, "search_precision")
        _validate_precision(self.ask_precision, "ask_precision")

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
            "reranker_provider": self.provider,
            "reranker_model": self.model_name,
            "reranker_model_revision": self.model_revision,
            "input_template_version": self.input_template_version,
            "max_pair_tokens": self.max_pair_tokens,
            "search_precision": self.search_precision,
            "ask_precision": self.ask_precision,
        }


def load_reranker_config(env: Mapping[str, str] = os.environ) -> RerankerConfig:
    cache_dir = env.get("OPK_RAG_RERANKER_CACHE_DIR", "").strip()
    revision = env.get("OPK_RAG_RERANKER_MODEL_REVISION", DEFAULT_RERANKER_MODEL_REVISION).strip()
    return RerankerConfig(
        provider=env.get("OPK_RAG_RERANKER_PROVIDER", DEFAULT_RERANKER_PROVIDER).strip(),
        model_name=env.get("OPK_RAG_RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL_NAME).strip(),
        model_revision=revision or None,
        batch_size=int(env.get("OPK_RAG_RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE))),
        max_pair_tokens=int(env.get("OPK_RAG_RERANKER_MAX_PAIR_TOKENS", str(DEFAULT_RERANKER_MAX_PAIR_TOKENS))),
        device=env.get("OPK_RAG_RERANKER_DEVICE", "auto").strip(),
        cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
        local_files_only=_bool_env(env.get("OPK_RAG_RERANKER_LOCAL_FILES_ONLY", "false")),
        input_template_version=env.get("OPK_RAG_RERANKER_INPUT_TEMPLATE_VERSION", RERANKER_INPUT_TEMPLATE_VERSION).strip(),
        search_precision=_precision_env(
            env.get("OPK_RAG_SEARCH_RERANKER_PRECISION", DEFAULT_SEARCH_RERANKER_PRECISION),
            "OPK_RAG_SEARCH_RERANKER_PRECISION",
        ),
        ask_precision=_precision_env(
            env.get("OPK_RAG_ASK_RERANKER_PRECISION", DEFAULT_ASK_RERANKER_PRECISION),
            "OPK_RAG_ASK_RERANKER_PRECISION",
        ),
    )


def build_reranker_configuration_fingerprint(config: RerankerConfig) -> str:
    payload = json.dumps(config.fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise RerankerConfigError(f"{field_name} must not be empty.")


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RerankerConfigError(f"Invalid boolean value: {value}")


def _precision_env(value: str, name: str) -> RerankerPrecision:
    normalized = value.strip().lower()
    if normalized in {"fp32", "float32", "torch.float32"}:
        return "fp32"
    if normalized in {"fp16_autocast", "cuda_fp16_autocast"}:
        return "fp16_autocast"
    raise RerankerConfigError(f"{name} must be one of: fp32, fp16_autocast.")


def _validate_precision(value: str, field_name: str) -> None:
    if value not in {"fp32", "fp16_autocast"}:
        raise RerankerConfigError(f"{field_name} must be one of: fp32, fp16_autocast.")
