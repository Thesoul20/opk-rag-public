from __future__ import annotations

from dataclasses import dataclass
from opk_rag.runtime_v2.models import RetrievalCandidateV2
from opk_rag.runtime_v2.rank_fusion import RANK_FUSION_POLICY, RankFusionConfigError, RankFusionParameters, rank_fusion_order
from opk_rag.runtime_v2.vector_retriever import reranked_candidate_v2


class RerankerRuntimeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    enabled: bool = True
    policy: str = RANK_FUSION_POLICY
    rank_fusion_k: int = 60
    rank_fusion_lambda: float = 0.75

    def __post_init__(self) -> None:
        if self.policy not in {RANK_FUSION_POLICY}:
            raise RerankerRuntimeConfigError(f"unsupported reranker runtime policy: {self.policy}")
        try:
            RankFusionParameters(k=self.rank_fusion_k, lambda_weight=self.rank_fusion_lambda)
        except RankFusionConfigError as exc:
            raise RerankerRuntimeConfigError(str(exc)) from exc


@dataclass(frozen=True)
class RerankerRuntimeResult:
    candidates: tuple[RetrievalCandidateV2, ...]
    diagnostics: dict[str, object]


def apply_frozen_reranker_scores_v2(
    candidates: tuple[RetrievalCandidateV2, ...],
    *,
    score_by_canonical_id: dict[str, float],
    final_top_k: int,
) -> tuple[RetrievalCandidateV2, ...]:
    scored = []
    for candidate in candidates:
        if candidate.canonical_chunk_id not in score_by_canonical_id:
            continue
        scored.append((candidate, score_by_canonical_id[candidate.canonical_chunk_id]))
    scored.sort(key=lambda item: (-item[1], item[0].original_vector_rank or 10_000, item[0].canonical_chunk_id))
    return tuple(
        reranked_candidate_v2(candidate, score=score, reranked_rank=index + 1, final_strategy="vector_top_50_bge_reranked_top_20")
        for index, (candidate, score) in enumerate(scored[:final_top_k])
    )


def apply_rank_fusion_reranker_scores_v2(
    candidates: tuple[RetrievalCandidateV2, ...],
    *,
    score_by_canonical_id: dict[str, float],
    config: RerankerRuntimeConfig,
) -> RerankerRuntimeResult:
    original = tuple(_with_retrieval_rank(candidate, index + 1) for index, candidate in enumerate(candidates))
    diagnostics: dict[str, object] = {
        "reranker_enabled": config.enabled,
        "reranker_policy": config.policy,
        "rank_fusion_k": config.rank_fusion_k,
        "rank_fusion_lambda": config.rank_fusion_lambda,
        "reranker_invoked": False,
        "reranker_success": False,
        "rank_fusion_invoked": False,
        "candidate_count": len(original),
        "candidate_membership_change_count": 0,
        "ranking_changed": False,
        "top5_changed": False,
        "fallback_used": False,
        "fallback_reason": None,
        "reranker_failure_policy": "fail_to_baseline_retrieval",
    }
    if not config.enabled:
        return RerankerRuntimeResult(candidates=original, diagnostics=diagnostics)

    diagnostics["reranker_invoked"] = True
    try:
        reranker_rank_by_id = _reranker_rank_by_identity(original, score_by_canonical_id)
        fused = rank_fusion_order(
            original,
            identity=lambda candidate: candidate.canonical_chunk_id,
            retrieval_rank=lambda candidate: candidate.original_vector_rank or candidate.rank,
            reranker_rank_by_identity=reranker_rank_by_id,
            parameters=RankFusionParameters(k=config.rank_fusion_k, lambda_weight=config.rank_fusion_lambda),
        )
    except Exception as exc:
        fallback = tuple(_with_retrieval_rank(candidate, index + 1) for index, candidate in enumerate(original))
        diagnostics.update({"fallback_used": True, "fallback_reason": exc.__class__.__name__, "reranker_success": False})
        return RerankerRuntimeResult(candidates=fallback, diagnostics=diagnostics)

    ranked = tuple(
        _replace_candidate(
            scored.item,
            rank=scored.fusion_rank,
            retrieval_strategy="vector_top_50_bge_reranked_rank_fusion",
            reranker_score=float(score_by_canonical_id[scored.identity]),
            reranked_rank=scored.reranker_rank,
            fusion_score=scored.fusion_score,
            fusion_rank=scored.fusion_rank,
            ranking_policy=config.policy,
        )
        for scored in fused
    )
    original_ids = [candidate.canonical_chunk_id for candidate in original]
    ranked_ids = [candidate.canonical_chunk_id for candidate in ranked]
    diagnostics.update(
        {
            "reranker_success": True,
            "rank_fusion_invoked": True,
            "candidate_membership_change_count": int(sorted(original_ids) != sorted(ranked_ids)),
            "ranking_changed": original_ids != ranked_ids,
            "top5_changed": original_ids[:5] != ranked_ids[:5],
        }
    )
    return RerankerRuntimeResult(candidates=ranked, diagnostics=diagnostics)


def _reranker_rank_by_identity(candidates: tuple[RetrievalCandidateV2, ...], score_by_canonical_id: dict[str, float]) -> dict[str, int]:
    missing = [candidate.canonical_chunk_id for candidate in candidates if candidate.canonical_chunk_id not in score_by_canonical_id]
    if missing:
        raise KeyError(f"missing reranker scores for {len(missing)} candidates")
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(score_by_canonical_id[candidate.canonical_chunk_id]),
            candidate.original_vector_rank or candidate.rank,
            candidate.canonical_chunk_id,
        ),
    )
    return {candidate.canonical_chunk_id: index for index, candidate in enumerate(ranked, start=1)}


def _with_retrieval_rank(candidate: RetrievalCandidateV2, rank: int) -> RetrievalCandidateV2:
    original_rank = candidate.original_vector_rank or candidate.rank or rank
    return _replace_candidate(candidate, rank=rank, original_vector_rank=original_rank)


def _replace_candidate(candidate: RetrievalCandidateV2, **changes: object) -> RetrievalCandidateV2:
    from dataclasses import replace

    return replace(candidate, **changes)
