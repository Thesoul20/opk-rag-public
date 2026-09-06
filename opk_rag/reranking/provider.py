from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from opk_rag.reranking.config import RerankerExecutionScope


@dataclass(frozen=True)
class RerankerPairMetadata:
    original_pair_token_count: int
    pair_token_count: int
    input_truncated: bool


class RerankerProvider(Protocol):
    @property
    def model_id(self) -> str:
        ...

    @property
    def model_revision(self) -> str | None:
        ...

    @property
    def input_template_version(self) -> str:
        ...

    @property
    def max_pair_tokens(self) -> int:
        ...

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        execution_scope: RerankerExecutionScope = "search",
    ) -> Sequence[float]:
        ...

    def count_pair_tokens(self, query: str, document: str) -> int:
        ...

    def prepare_pair_metadata(self, query: str, document: str) -> RerankerPairMetadata:
        ...


class RerankerModelLoadError(RuntimeError):
    pass


class RerankerInferenceError(RuntimeError):
    pass
