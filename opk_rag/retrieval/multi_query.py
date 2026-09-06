from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryVariant:
    query_id: str
    query_type: str
    text: str
    source_query_id: str | None
    generation_policy: str


@dataclass(frozen=True)
class MultiQueryPlan:
    original_query: str
    variants: tuple[QueryVariant, ...]
    enabled: bool = False
    policy: str = "deterministic_retrieval_rewrite_v1"

    @property
    def query_count(self) -> int:
        return len(self.variants)


@dataclass(frozen=True)
class MultiQueryCandidateHit:
    canonical_chunk_id: str
    query_id: str
    query_type: str
    query_rank: int
    query_score: float
    candidate: dict[str, Any]


@dataclass(frozen=True)
class MultiQueryRetrievalResult:
    unit_id: str
    plan: MultiQueryPlan
    candidates: tuple[dict[str, Any], ...]
    raw_candidate_count: int
    unique_candidate_count: int
    duplicate_candidate_count: int


@dataclass(frozen=True)
class MultiQueryConfig:
    enabled: bool = False
    query_count: int = 1
    retrieval_top_k: int = 20
    query_generation_policy: str = "deterministic_retrieval_rewrite_v1"
    query_fusion_policy: str = "best_rank"
    query_fusion_k: int = 60


def default_multi_query_config() -> MultiQueryConfig:
    return MultiQueryConfig(enabled=False)


def build_query_plan(query: str, *, enabled: bool, query_count: int, policy: str = "deterministic_retrieval_rewrite_v1") -> MultiQueryPlan:
    variants = [QueryVariant("q0", "original", query.strip(), None, policy)]
    if enabled:
        for index, (query_type, text) in enumerate(_alternative_texts(query), start=1):
            if len(variants) >= query_count:
                break
            variants.append(QueryVariant(f"q{index}", query_type, text, "q0", policy))
    return MultiQueryPlan(original_query=query, variants=tuple(variants), enabled=enabled, policy=policy)


def deduplicate_candidate_hits(unit_id: str, plan: MultiQueryPlan, hits: list[MultiQueryCandidateHit]) -> MultiQueryRetrievalResult:
    grouped: dict[str, list[MultiQueryCandidateHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.canonical_chunk_id, []).append(hit)
    candidates = []
    for chunk_id, chunk_hits in sorted(grouped.items(), key=lambda item: (_best_rank(item[1]), -_best_score(item[1]), item[0])):
        best = sorted(chunk_hits, key=lambda hit: (hit.query_rank, -hit.query_score, hit.query_id))[0]
        candidate = dict(best.candidate)
        candidate["canonical_chunk_id"] = chunk_id
        candidate["retrieved_by_query_ids"] = sorted({hit.query_id for hit in chunk_hits})
        candidate["retrieved_by_query_types"] = sorted({hit.query_type for hit in chunk_hits})
        candidate["retrieval_query_count"] = len({hit.query_id for hit in chunk_hits})
        candidate["best_query_rank"] = _best_rank(chunk_hits)
        candidate["best_query_score"] = _best_score(chunk_hits)
        candidate["query_fusion_score"] = _fusion_score(chunk_hits)
        candidate["query_fusion_rank"] = len(candidates) + 1
        candidate["retrieval_rank"] = len(candidates) + 1
        candidates.append(candidate)
    return MultiQueryRetrievalResult(
        unit_id=unit_id,
        plan=plan,
        candidates=tuple(candidates),
        raw_candidate_count=len(hits),
        unique_candidate_count=len(candidates),
        duplicate_candidate_count=len(hits) - len(candidates),
    )


def _best_rank(hits: list[MultiQueryCandidateHit]) -> int:
    return min(hit.query_rank for hit in hits)


def _best_score(hits: list[MultiQueryCandidateHit]) -> float:
    return max(hit.query_score for hit in hits)


def _fusion_score(hits: list[MultiQueryCandidateHit]) -> float:
    return sum(1.0 / (60 + hit.query_rank) for hit in hits)


def _alternative_texts(query: str) -> tuple[tuple[str, str], ...]:
    stripped = " ".join(query.strip().split())
    normalized = stripped.replace("哪些", "什么").replace("如何", "怎么").replace("为什么", "原因")
    expansion_terms = []
    synonym_map = {
        "配置": "设置 配置文件 config",
        "文件": "路径 位置",
        "影响": "作用 范围 生效",
        "范围": "scope 生效范围",
        "错误": "失败 异常 原因",
        "检索": "召回 retrieval search",
        "引用": "citation evidence 证据",
        "模型": "model provider reranker embedding",
    }
    for key, value in synonym_map.items():
        if key in stripped:
            expansion_terms.append(value)
    expansion = " ".join([stripped, *expansion_terms]).strip()
    document_style = stripped.replace("？", "").replace("?", "")
    return (
        ("rewrite", normalized),
        ("alternative", expansion or stripped),
        ("alternative", document_style),
    )

