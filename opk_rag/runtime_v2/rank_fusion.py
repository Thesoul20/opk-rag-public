from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, TypeVar


T = TypeVar("T")

DEFAULT_RANK_FUSION_K = 60
DEFAULT_RANK_FUSION_LAMBDA = 0.75
RANK_FUSION_POLICY = "rank_fusion"


class RankFusionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RankFusionParameters:
    k: int = DEFAULT_RANK_FUSION_K
    lambda_weight: float = DEFAULT_RANK_FUSION_LAMBDA

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise RankFusionConfigError("rank_fusion_k must be > 0.")
        if not math.isfinite(self.lambda_weight) or self.lambda_weight < 0:
            raise RankFusionConfigError("rank_fusion_lambda must be a finite value >= 0.")


@dataclass(frozen=True)
class RankFusionScoredItem(Generic[T]):
    item: T
    identity: str
    retrieval_rank: int
    reranker_rank: int
    fusion_rank: int
    fusion_score: float


def rank_fusion_score(*, retrieval_rank: int, reranker_rank: int, parameters: RankFusionParameters) -> float:
    if retrieval_rank <= 0:
        raise RankFusionConfigError("retrieval_rank must be > 0.")
    if reranker_rank <= 0:
        raise RankFusionConfigError("reranker_rank must be > 0.")
    return 1 / (parameters.k + retrieval_rank) + parameters.lambda_weight / (parameters.k + reranker_rank)


def rank_fusion_order(
    items: tuple[T, ...],
    *,
    identity: Callable[[T], str],
    retrieval_rank: Callable[[T], int],
    reranker_rank_by_identity: dict[str, int],
    parameters: RankFusionParameters,
) -> tuple[RankFusionScoredItem[T], ...]:
    scored: list[tuple[float, int, str, T, int]] = []
    for item in items:
        item_identity = identity(item)
        item_retrieval_rank = int(retrieval_rank(item))
        item_reranker_rank = int(reranker_rank_by_identity[item_identity])
        score = rank_fusion_score(
            retrieval_rank=item_retrieval_rank,
            reranker_rank=item_reranker_rank,
            parameters=parameters,
        )
        scored.append((score, item_retrieval_rank, item_identity, item, item_reranker_rank))
    ranked = sorted(scored, key=lambda row: (-row[0], row[1], row[2]))
    return tuple(
        RankFusionScoredItem(
            item=item,
            identity=item_identity,
            retrieval_rank=item_retrieval_rank,
            reranker_rank=item_reranker_rank,
            fusion_rank=index,
            fusion_score=score,
        )
        for index, (score, item_retrieval_rank, item_identity, item, item_reranker_rank) in enumerate(ranked, start=1)
    )
