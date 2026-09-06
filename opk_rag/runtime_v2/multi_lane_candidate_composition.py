from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from opk_rag.evaluation.task0116_governed_late_interaction_retrieval_ablation import merge_provenance, union_candidates


SOURCE_DENSE_ONLY = "dense_only"
SOURCE_LATE_ONLY = "late_only"
SOURCE_OVERLAP = "dense_late_overlap"
SOURCE_TYPES = (SOURCE_DENSE_ONLY, SOURCE_LATE_ONLY, SOURCE_OVERLAP)


@dataclass(frozen=True)
class LaneAllocationPolicy:
    policy_id: str
    dense_lane_capacity: int
    late_lane_capacity: int
    overlap_lane_capacity: int
    shared_fill_capacity: int
    total_budget: int
    rationale: str
    empty_lane_fill_policy: str = "unused_lane_capacity_flows_to_shared_fill"
    spillover_semantics: str = "source_local_rank_then_canonical_id"

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        capacities = (
            self.dense_lane_capacity,
            self.late_lane_capacity,
            self.overlap_lane_capacity,
            self.shared_fill_capacity,
        )
        if any(value < 0 for value in capacities):
            raise ValueError("lane capacities must be non-negative")
        if sum(capacities) != self.total_budget:
            raise ValueError("lane capacities must sum to total_budget")
        if self.policy_id != "C0" and self.late_lane_capacity <= 0:
            raise ValueError("formal multi-lane arms must reserve late-only capacity")

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "dense_lane_capacity": self.dense_lane_capacity,
            "late_lane_capacity": self.late_lane_capacity,
            "overlap_lane_capacity": self.overlap_lane_capacity,
            "shared_fill_capacity": self.shared_fill_capacity,
            "total_budget": self.total_budget,
            "rationale": self.rationale,
            "empty_lane_fill_policy": self.empty_lane_fill_policy,
            "spillover_semantics": self.spillover_semantics,
        }


@dataclass(frozen=True)
class MultiLaneCompositionResult:
    candidates: list[dict[str, Any]]
    metadata: dict[str, Any]


class MultiLaneCandidateComposer:
    """Deterministic source-aware composer for reranker admission experiments."""

    def __init__(self, policy: LaneAllocationPolicy) -> None:
        self.policy = policy

    def compose(self, dense_candidates: list[dict[str, Any]], late_candidates: list[dict[str, Any]]) -> MultiLaneCompositionResult:
        start = time.perf_counter()
        union = union_candidates(dense_candidates, late_candidates, policy="candidate_union")
        lanes = {
            SOURCE_DENSE_ONLY: _source_local_sort([row for row in union if candidate_source_type(row) == SOURCE_DENSE_ONLY]),
            SOURCE_LATE_ONLY: _source_local_sort([row for row in union if candidate_source_type(row) == SOURCE_LATE_ONLY]),
            SOURCE_OVERLAP: _source_local_sort([row for row in union if candidate_source_type(row) == SOURCE_OVERLAP]),
        }
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        lane_used = {SOURCE_DENSE_ONLY: 0, SOURCE_LATE_ONLY: 0, SOURCE_OVERLAP: 0, "shared_fill": 0}
        lane_spillover = {SOURCE_DENSE_ONLY: 0, SOURCE_LATE_ONLY: 0, SOURCE_OVERLAP: 0}

        for source_type, capacity in (
            (SOURCE_DENSE_ONLY, self.policy.dense_lane_capacity),
            (SOURCE_LATE_ONLY, self.policy.late_lane_capacity),
            (SOURCE_OVERLAP, self.policy.overlap_lane_capacity),
        ):
            picked = _take(lanes[source_type], capacity, selected_ids)
            selected.extend(picked)
            selected_ids.update(row["canonical_chunk_id"] for row in picked)
            lane_used[source_type] = len(picked)
            lane_spillover[source_type] = max(capacity - len(picked), 0)

        fill_budget = self.policy.shared_fill_capacity + sum(lane_spillover.values())
        fill_pool = _source_local_sort([row for row in union if row["canonical_chunk_id"] not in selected_ids])
        fill = _take(fill_pool, fill_budget, selected_ids)
        selected.extend(fill)
        selected_ids.update(row["canonical_chunk_id"] for row in fill)
        lane_used["shared_fill"] = len(fill)

        final = selected[: self.policy.total_budget]
        final_ids = [row["canonical_chunk_id"] for row in final]
        metadata = {
            "schema_version": "opk-rag.multi-lane-composition.v1",
            "policy": self.policy.to_json(),
            "policy_digest": self.policy.digest,
            "candidate_union_count": len(union),
            "candidate_union_digest": candidate_membership_digest(union),
            "reranker_input_candidate_ids": final_ids,
            "reranker_input_budget": self.policy.total_budget,
            "reranker_input_count": len(final),
            "candidate_source_counts": {source: sum(candidate_source_type(row) == source for row in final) for source in SOURCE_TYPES},
            "dense_lane_used_slots": lane_used[SOURCE_DENSE_ONLY],
            "late_lane_used_slots": lane_used[SOURCE_LATE_ONLY],
            "overlap_lane_used_slots": lane_used[SOURCE_OVERLAP],
            "shared_fill_used_slots": lane_used["shared_fill"],
            "unused_slot_count": max(self.policy.total_budget - len(final), 0),
            "lane_spillover_count": sum(lane_spillover.values()),
            "lane_spillover": lane_spillover,
            "spillover_destination": "shared_fill",
            "composition_latency_ms": (time.perf_counter() - start) * 1000,
            "canonical_candidate_identity_preserved": len(final_ids) == len(set(final_ids)),
            "duplicate_slot_consumption": len(final_ids) != len(set(final_ids)),
            "candidate_source_taxonomy_complete": all(candidate_source_type(row) in SOURCE_TYPES for row in union),
            "multi_lane_composition_deterministic": True,
            "multi_lane_composition_gold_independent": True,
            "multi_lane_runtime_observable_only": True,
        }
        return MultiLaneCompositionResult(candidates=final, metadata=metadata)


def candidate_source_type(row: dict[str, Any]) -> str:
    sources = set(row.get("retrieval_sources") or [])
    if "dense" in sources and "late_interaction" in sources:
        return SOURCE_OVERLAP
    if "late_interaction" in sources:
        return SOURCE_LATE_ONLY
    return SOURCE_DENSE_ONLY


def candidate_membership_digest(rows: list[dict[str, Any]]) -> str:
    payload = sorted(row["canonical_chunk_id"] for row in merge_provenance(rows))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _source_local_sort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            min(row.get("dense_rank") or 999999, row.get("late_interaction_rank") or 999999, row.get("retrieval_rank") or 999999),
            row.get("dense_rank") or 999999,
            row.get("late_interaction_rank") or 999999,
            row["canonical_chunk_id"],
        ),
    )


def _take(rows: list[dict[str, Any]], capacity: int, selected_ids: set[str]) -> list[dict[str, Any]]:
    picked = []
    for row in rows:
        if len(picked) >= capacity:
            break
        cid = row["canonical_chunk_id"]
        if cid in selected_ids:
            continue
        picked.append(row)
    return picked
