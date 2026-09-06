from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingModelLoadError(EmbeddingProviderError):
    pass


class EmbeddingInferenceError(EmbeddingProviderError):
    pass


class EmbeddingOutputValidationError(EmbeddingProviderError):
    pass


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    # Query encoding is intentionally separate from document encoding because
    # Qwen query embeddings require query-side instruction semantics.
    def embed_query(self, query: str, *, instruction: str | None = None) -> Sequence[float]:
        ...

    def count_tokens(self, text: str) -> int:
        ...
