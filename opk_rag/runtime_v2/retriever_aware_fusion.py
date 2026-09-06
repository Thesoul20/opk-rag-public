from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import digest_json
from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import FUSION_K, merge_provenance


RETRIEVER_AWARE_FUSION_POLICY_VERSION = "task0123-retriever-aware-fusion-v1"


@dataclass(frozen=True)
class RetrieverAwareFusionPolicy:
    policy_version: str = RETRIEVER_AWARE_FUSION_POLICY_VERSION
    dense_protected_prefix_size: int = 5
    fusion_k: int = FUSION_K
    final_top_k: int = 20

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": "opk-rag.task0123.retriever-aware-fusion-policy.v1",
            **asdict(self),
            "policy": "dense_protected_core_plus_overlap_promotion_plus_late_recovery_admission",
            "primary_mitigation_count": 1,
            "parameter_selection_basis": "TASK-0122 found dense supporting candidates at ranks 1-5 displaced from EvidenceContext top-5.",
            "fusion_runtime_observable_only": True,
            "retriever_aware_fusion_gold_independent": True,
            "retriever_aware_fusion_deterministic": True,
            "no_regression_specific_branch": True,
            "tuning_allowed": False,
        }
        return payload | {"fusion_policy_digest": digest_json(payload)}

    @property
    def digest(self) -> str:
        return self.to_json()["fusion_policy_digest"]


def retriever_aware_fusion(
    dense: list[dict[str, Any]],
    late: list[dict[str, Any]],
    *,
    policy: RetrieverAwareFusionPolicy | None = None,
) -> list[dict[str, Any]]:
    cfg = policy or RetrieverAwareFusionPolicy()
    merged = merge_provenance([*dense, *late])
    annotated = [_annotate(row, cfg) for row in merged]
    ranked = sorted(annotated, key=lambda row: _sort_key(row, cfg))
    out = []
    for rank, row in enumerate(ranked[: cfg.final_top_k], start=1):
        item = dict(row)
        item["retriever_fusion_rank"] = rank
        item["fusion_rank"] = rank
        item["ranking_policy"] = cfg.policy_version
        out.append(item)
    return out


def candidate_membership_digest(rows: list[dict[str, Any]]) -> str:
    return digest_json(sorted(row["canonical_chunk_id"] for row in rows))


def fusion_input_membership_equivalent(
    dense: list[dict[str, Any]],
    late: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    mitigated: list[dict[str, Any]],
) -> bool:
    source_union = {row["canonical_chunk_id"] for row in merge_provenance([*dense, *late])}
    return {row["canonical_chunk_id"] for row in existing} <= source_union and {row["canonical_chunk_id"] for row in mitigated} <= source_union


def _annotate(row: dict[str, Any], policy: RetrieverAwareFusionPolicy) -> dict[str, Any]:
    item = dict(row)
    sources = set(item.get("retrieval_sources") or [])
    dense_rank = item.get("dense_rank")
    late_rank = item.get("late_interaction_rank")
    is_dense = "dense" in sources and dense_rank is not None
    is_late = "late_interaction" in sources and late_rank is not None
    item["retriever_source_class"] = "dense+late_interaction" if is_dense and is_late else "dense" if is_dense else "late_interaction"
    item["retriever_overlap_candidate"] = is_dense and is_late
    item["dense_protected_core_candidate"] = is_dense and int(dense_rank) <= policy.dense_protected_prefix_size
    item["late_recovery_candidate"] = is_late and not is_dense
    item["fusion_score"] = (1 / (policy.fusion_k + int(dense_rank)) if dense_rank else 0.0) + (1 / (policy.fusion_k + int(late_rank)) if late_rank else 0.0)
    return item


def _sort_key(row: dict[str, Any], policy: RetrieverAwareFusionPolicy) -> tuple[Any, ...]:
    dense_rank = int(row.get("dense_rank") or 999999)
    late_rank = int(row.get("late_interaction_rank") or 999999)
    if row["retriever_overlap_candidate"]:
        tier = 0
    elif row["dense_protected_core_candidate"]:
        tier = 1
    else:
        tier = 2
    return (
        tier,
        -float(row.get("fusion_score") or 0.0),
        min(dense_rank, late_rank),
        dense_rank,
        late_rank,
        row["canonical_chunk_id"],
    )
