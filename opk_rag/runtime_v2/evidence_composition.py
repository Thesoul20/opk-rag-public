from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from opk_rag.runtime_v2.evidence_context import candidates_to_evidence_context
from opk_rag.runtime_v2.models import EvidenceContextV2, RetrievalCandidateV2
from opk_rag.runtime_v2.multi_lane_candidate_composition import SOURCE_LATE_ONLY, SOURCE_OVERLAP, candidate_source_type


LEGACY_SEQUENTIAL_POLICY = "legacy_sequential"
TARGETED_BUDGETED_POLICY = "targeted_budgeted_composition"
DEFAULT_EVIDENCE_COMPOSITION_POLICY = TARGETED_BUDGETED_POLICY
EVIDENCE_BUDGET_UNIT = "evidence_slots"
EVIDENCE_BUDGET_LIMIT = 5
POLICY_VERSION = "opk-rag.evidence-composition.targeted-budgeted.v1"
SOURCE_TASK = "TASK-0130"


@dataclass(frozen=True)
class EvidenceCompositionPolicy:
    policy_name: str = DEFAULT_EVIDENCE_COMPOSITION_POLICY
    policy_version: str = POLICY_VERSION
    budget_limit: int = EVIDENCE_BUDGET_LIMIT
    budget_unit: str = EVIDENCE_BUDGET_UNIT
    suppress_redundant: bool = True
    lane_protection: bool = True
    protected_lanes: tuple[str, ...] = (SOURCE_LATE_ONLY, SOURCE_OVERLAP)
    protected_prefix_slots: int = 3
    same_document_prefix_retention_guard: bool = True
    source_task: str = SOURCE_TASK

    @property
    def policy_digest(self) -> str:
        return stable_digest(self._identity_payload())

    @property
    def digest(self) -> str:
        return self.policy_digest

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_digest": None,
            "source_task": self.source_task,
            "budget_limit": self.budget_limit,
            "budget_unit": self.budget_unit,
            "suppress_redundant": self.suppress_redundant,
            "lane_protection": self.lane_protection,
            "protected_lanes": list(self.protected_lanes),
            "protected_prefix_slots": self.protected_prefix_slots,
            "same_document_prefix_retention_guard": self.same_document_prefix_retention_guard,
            "gold_label_used_by_runtime_policy": False,
            "benchmark_sample_specific_rule": False,
            "deterministic": True,
            "additional_retrieval_calls": 0,
            "additional_reranker_calls": 0,
            "additional_generation_calls": 0,
        } | {"policy_digest": stable_digest({k: v for k, v in self._identity_payload().items()})}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "source_task": self.source_task,
            "budget_limit": self.budget_limit,
            "budget_unit": self.budget_unit,
            "suppress_redundant": self.suppress_redundant,
            "lane_protection": self.lane_protection,
            "protected_lanes": list(self.protected_lanes),
            "protected_prefix_slots": self.protected_prefix_slots,
            "same_document_prefix_retention_guard": self.same_document_prefix_retention_guard,
        }


def legacy_sequential_policy() -> EvidenceCompositionPolicy:
    return EvidenceCompositionPolicy(
        policy_name=LEGACY_SEQUENTIAL_POLICY,
        policy_version="opk-rag.evidence-composition.legacy-sequential.v1",
        suppress_redundant=False,
        lane_protection=False,
        protected_lanes=(),
        same_document_prefix_retention_guard=False,
        source_task="pre-TASK-0131-runtime-default",
    )


def targeted_budgeted_policy() -> EvidenceCompositionPolicy:
    return EvidenceCompositionPolicy()


def resolve_evidence_composition_policy(policy: str | EvidenceCompositionPolicy | None = None) -> EvidenceCompositionPolicy:
    if isinstance(policy, EvidenceCompositionPolicy):
        return policy
    if policy is None or policy == DEFAULT_EVIDENCE_COMPOSITION_POLICY or policy == TARGETED_BUDGETED_POLICY:
        return targeted_budgeted_policy()
    if policy == LEGACY_SEQUENTIAL_POLICY:
        return legacy_sequential_policy()
    raise ValueError(f"unknown evidence composition policy: {policy}")


def compose_evidence(
    candidates: list[dict[str, Any]],
    policy: str | EvidenceCompositionPolicy | None = None,
    *,
    evaluation_unit_id: str = "unit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = resolve_evidence_composition_policy(policy)
    annotated = [annotate_candidate(row, rank=index) for index, row in enumerate(candidates, start=1)]
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    trace_by_id: dict[str, dict[str, Any]] = {}
    budget_used = 0

    if resolved.lane_protection:
        for row in lane_protection_seeds(annotated, resolved):
            if selected and same_document_prefix_retention_guard_triggered(row, selected, annotated, resolved):
                trace_by_id[row["candidate_id"]] = trace_row(
                    evaluation_unit_id,
                    resolved,
                    row,
                    False,
                    budget_used,
                    "same_document_prefix_retention_guard",
                    mitigation_triggered=True,
                )
                continue
            budget_used = try_select(row, selected, selected_ids, trace_by_id, budget_used, resolved, "lane_protection", evaluation_unit_id)

    for row in annotated:
        budget_used = try_select(row, selected, selected_ids, trace_by_id, budget_used, resolved, "rank_order", evaluation_unit_id)

    traces = []
    selected_order = {row["candidate_id"]: index for index, row in enumerate(selected, start=1)}
    represented = set()
    for row in annotated:
        cid = row["candidate_id"]
        trace = trace_by_id.get(cid)
        if trace is None:
            rejection = rejection_reason(row, represented, budget_used, resolved)
            trace = trace_row(evaluation_unit_id, resolved, row, False, budget_used, rejection)
        if cid in selected_order:
            trace["evidence_rank"] = selected_order[cid]
            represented.add(redundancy_key(row))
        traces.append(trace)
    evidence = [{**row["raw"], "evidence_rank": index, "policy_rank": index} for index, row in enumerate(selected, start=1)]
    return evidence, traces


def compose_evidence_contexts(
    candidates: tuple[RetrievalCandidateV2, ...],
    policy: str | EvidenceCompositionPolicy | None = None,
    *,
    evaluation_unit_id: str = "unit",
) -> tuple[EvidenceContextV2, ...]:
    rows = [_candidate_v2_to_row(candidate) for candidate in candidates]
    evidence, _traces = compose_evidence(rows, policy, evaluation_unit_id=evaluation_unit_id)
    by_id = {candidate.canonical_chunk_id: candidate for candidate in candidates}
    selected = tuple(by_id[row["canonical_chunk_id"]] for row in evidence)
    return candidates_to_evidence_context(selected)


def annotate_candidate(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    cost = int(row.get("estimated_budget_cost") or row.get("evidence_budget_cost") or 1)
    return {
        "candidate_id": row["canonical_chunk_id"],
        "canonical_chunk_id": row["canonical_chunk_id"],
        "document_id": row.get("document_id"),
        "section_id": row.get("section_id"),
        "lane": candidate_source_type(row),
        "rank": int(row.get("policy_rank") or row.get("reranker_rank") or row.get("retrieval_rank") or rank),
        "retrieval_score": row.get("retrieval_score"),
        "rerank_score_if_available": row.get("reranker_score"),
        "estimated_budget_cost": max(cost, 1),
        "raw": row,
        "relevant_label": bool(row.get("relevant_label")),
    }


def lane_protection_seeds(candidates: list[dict[str, Any]], policy: EvidenceCompositionPolicy) -> list[dict[str, Any]]:
    prefix_ids = {row["candidate_id"] for row in candidates[: policy.protected_prefix_slots]}
    seeds = []
    for lane in policy.protected_lanes:
        if any(row["lane"] == lane and row["candidate_id"] in prefix_ids for row in candidates):
            continue
        candidate = next((row for row in candidates if row["lane"] == lane), None)
        if candidate is not None:
            seeds.append(candidate)
    return sorted(seeds, key=lambda row: (row["rank"], row["candidate_id"]))


def try_select(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    trace_by_id: dict[str, dict[str, Any]],
    budget_used: int,
    policy: EvidenceCompositionPolicy,
    reason: str,
    evaluation_unit_id: str,
) -> int:
    cid = row["candidate_id"]
    existing_trace = trace_by_id.get(cid)
    if existing_trace is not None and existing_trace.get("mitigation_triggered"):
        return budget_used
    if cid in selected_ids:
        return budget_used
    if row["estimated_budget_cost"] > policy.budget_limit:
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "oversized_candidate")
        return budget_used
    if budget_used + row["estimated_budget_cost"] > policy.budget_limit:
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "budget_cutoff")
        return budget_used
    if policy.suppress_redundant and any(redundancy_key(row) == redundancy_key(existing) for existing in selected):
        trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, False, budget_used, "redundancy")
        return budget_used
    selected.append(row)
    selected_ids.add(cid)
    budget_used += row["estimated_budget_cost"]
    trace_by_id[cid] = trace_row(evaluation_unit_id, policy, row, True, budget_used, None, reason)
    return budget_used


def same_document_prefix_retention_guard_triggered(
    candidate_seed: dict[str, Any],
    selected_seeds: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy: EvidenceCompositionPolicy,
) -> bool:
    if not policy.same_document_prefix_retention_guard:
        return False
    current = fill_by_rank(selected_seeds, candidates, policy)
    trial = fill_by_rank([*selected_seeds, candidate_seed], candidates, policy)
    trial_ids = {row["candidate_id"] for row in trial}
    selected_seed_docs = {row.get("document_id") for row in selected_seeds if row.get("document_id")}
    for dropped in current:
        if dropped["candidate_id"] in trial_ids:
            continue
        if dropped["rank"] > policy.budget_limit:
            continue
        if dropped.get("document_id") in selected_seed_docs:
            return True
    return False


def fill_by_rank(seed_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], policy: EvidenceCompositionPolicy) -> list[dict[str, Any]]:
    selected = list(seed_rows)
    selected_ids = {row["candidate_id"] for row in selected}
    budget_used = sum(row["estimated_budget_cost"] for row in selected)
    for row in candidates:
        if row["candidate_id"] in selected_ids:
            continue
        if row["estimated_budget_cost"] > policy.budget_limit:
            continue
        if budget_used + row["estimated_budget_cost"] > policy.budget_limit:
            continue
        if policy.suppress_redundant and any(redundancy_key(row) == redundancy_key(existing) for existing in selected):
            continue
        selected.append(row)
        selected_ids.add(row["candidate_id"])
        budget_used += row["estimated_budget_cost"]
    return selected


def trace_row(
    evaluation_unit_id: str,
    policy: EvidenceCompositionPolicy,
    row: dict[str, Any],
    selected: bool,
    budget_after: int,
    rejection_reason_value: str | None,
    selection_reason: str | None = None,
    *,
    mitigation_triggered: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.runtime-v2.evidence-composition-trace.v1",
        "evaluation_unit_id": evaluation_unit_id,
        "arm_id": policy.policy_name,
        "policy_name": policy.policy_name,
        "policy_version": policy.policy_version,
        "policy_digest": policy.policy_digest,
        "source_task": policy.source_task,
        "candidate_id": row["candidate_id"],
        "canonical_chunk_id": row["canonical_chunk_id"],
        "document_id": row.get("document_id"),
        "lane": row["lane"],
        "rank": row["rank"],
        "retrieval_score": row.get("retrieval_score"),
        "rerank_score_if_available": row.get("rerank_score_if_available"),
        "estimated_budget_cost": row["estimated_budget_cost"],
        "selected_for_evidence": selected,
        "selection_reason": selection_reason if selected else None,
        "rejection_reason": rejection_reason_value,
        "evidence_budget_limit": policy.budget_limit,
        "evidence_budget_used_after_decision": budget_after,
        "same_chunk_duplicate": False,
        "same_document_high_overlap": rejection_reason_value == "redundancy",
        "cross_lane_duplicate": False,
        "near_redundant_evidence": rejection_reason_value == "redundancy",
        "relevant_label": row["relevant_label"],
        "mitigation_triggered": mitigation_triggered,
        "gold_label_used_by_runtime_policy": False,
        "benchmark_sample_specific_rule": False,
    }


def rejection_reason(row: dict[str, Any], represented: set[tuple[Any, Any]], budget_used: int, policy: EvidenceCompositionPolicy) -> str:
    if row["estimated_budget_cost"] > policy.budget_limit:
        return "oversized_candidate"
    if policy.suppress_redundant and redundancy_key(row) in represented:
        return "redundancy"
    if budget_used >= policy.budget_limit:
        return "budget_cutoff"
    return "not_selected"


def redundancy_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("document_id"), row.get("section_id") or row.get("canonical_chunk_id"))


def selected_evidence_ids(composition: list[dict[str, Any]]) -> list[str]:
    return [row["canonical_chunk_id"] for row in composition[:EVIDENCE_BUDGET_LIMIT]]


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _candidate_v2_to_row(candidate: RetrievalCandidateV2) -> dict[str, Any]:
    rank = candidate.reranked_rank or candidate.fusion_rank or candidate.rank
    sources = ["late_interaction"] if candidate.retrieval_strategy == "late_interaction" else ["dense"]
    return {
        "canonical_chunk_id": candidate.canonical_chunk_id,
        "document_id": candidate.document_id,
        "section_id": candidate.section_id,
        "retrieval_rank": rank,
        "policy_rank": rank,
        "retrieval_score": candidate.retrieval_score,
        "reranker_score": candidate.reranker_score,
        "estimated_budget_cost": 1,
        "retrieval_sources": sources,
    }
