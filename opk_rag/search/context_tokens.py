from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

from opk_rag.embedding.config import EmbeddingConfig


class ContextTokenCounter(Protocol):
    @property
    def tokenizer_id(self) -> str:
        ...

    @property
    def tokenizer_revision(self) -> str | None:
        ...

    def count_tokens(self, text: str) -> int:
        ...


@dataclass(frozen=True)
class SimpleContextTokenCounter:
    tokenizer_id: str = "simple-character-counter"
    tokenizer_revision: str | None = None

    def count_tokens(self, text: str) -> int:
        return len(text)


class QwenContextTokenCounter:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def tokenizer_id(self) -> str:
        return self.config.model_name

    @property
    def tokenizer_revision(self) -> str | None:
        return self.config.model_revision

    @cached_property
    def _tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError("transformers is required for Qwen context token counting.") from exc
        kwargs = {
            "revision": self.config.model_revision,
            "trust_remote_code": False,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = str(Path(self.config.cache_dir).expanduser())
        return AutoTokenizer.from_pretrained(self.config.model_name, **kwargs)

    def count_tokens(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)
