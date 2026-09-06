from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from opk_rag.agent.contracts import stable_digest

EVIDENCE_COMPARISON_CONTRACT_VERSION = "opk-rag.agent-evidence-comparison.v1"

EvidenceOutcome = Literal[
    "evidence_materially_improved",
    "evidence_changed_not_improved",
    "evidence_unchanged",
    "evidence_degraded",
    "evidence_comparison_invalid",
]


@dataclass(frozen=True)
class EvidenceComparisonResult:
    candidate_identity_overlap: int
    new_candidate_count: int
    removed_candidate_count: int
    rank_changes: dict[str, int]
    document_diversity_change: int
    scope_diversity_change: int
    top_k_identity_changed: bool
    retrieval_score_change: float | None
    answerability_change: str | None
    outcome: EvidenceOutcome
    contract_version: str = EVIDENCE_COMPARISON_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "candidate_identity_overlap": self.candidate_identity_overlap,
            "new_candidate_count": self.new_candidate_count,
            "removed_candidate_count": self.removed_candidate_count,
            "rank_changes": self.rank_changes,
            "document_diversity_change": self.document_diversity_change,
            "scope_diversity_change": self.scope_diversity_change,
            "top_k_identity_changed": self.top_k_identity_changed,
            "retrieval_score_change": self.retrieval_score_change,
            "answerability_change": self.answerability_change,
            "outcome": self.outcome,
        }

    @property
    def identity_digest(self) -> str:
        return stable_digest(self.to_dict())


def compare_retrieval_evidence(
    initial_summary: dict[str, Any],
    recovery_summary: dict[str, Any],
    *,
    initial_answerability: str | None = None,
    recovery_answerability: str | None = None,
) -> EvidenceComparisonResult:
    initial = _candidate_rows(initial_summary)
    recovery = _candidate_rows(recovery_summary)
    if not recovery:
        return EvidenceComparisonResult(
            candidate_identity_overlap=0,
            new_candidate_count=0,
            removed_candidate_count=len(initial),
            rank_changes={},
            document_diversity_change=-len(_documents(initial)),
            scope_diversity_change=-len(_scopes(initial)),
            top_k_identity_changed=bool(initial),
            retrieval_score_change=None,
            answerability_change=_answerability_change(initial_answerability, recovery_answerability),
            outcome="evidence_comparison_invalid",
        )

    initial_ids = [_identity(row) for row in initial]
    recovery_ids = [_identity(row) for row in recovery]
    initial_set = set(initial_ids)
    recovery_set = set(recovery_ids)
    overlap = len(initial_set & recovery_set)
    new_count = len(recovery_set - initial_set)
    removed_count = len(initial_set - recovery_set)
    rank_changes = _rank_changes(initial_ids, recovery_ids)
    top_changed = (initial_ids[:1] or [None])[0] != (recovery_ids[:1] or [None])[0]
    score_change = _score(recovery_summary) - _score(initial_summary) if _score(initial_summary) is not None and _score(recovery_summary) is not None else None
    answerability_change = _answerability_change(initial_answerability, recovery_answerability)
    outcome = _classify_outcome(
        overlap=overlap,
        initial_count=len(initial_set),
        new_count=new_count,
        removed_count=removed_count,
        top_changed=top_changed,
        answerability_change=answerability_change,
    )
    return EvidenceComparisonResult(
        candidate_identity_overlap=overlap,
        new_candidate_count=new_count,
        removed_candidate_count=removed_count,
        rank_changes=rank_changes,
        document_diversity_change=len(_documents(recovery)) - len(_documents(initial)),
        scope_diversity_change=len(_scopes(recovery)) - len(_scopes(initial)),
        top_k_identity_changed=top_changed,
        retrieval_score_change=score_change,
        answerability_change=answerability_change,
        outcome=outcome,
    )


def evidence_summary_from_tool(summary: dict[str, Any]) -> dict[str, Any]:
    trusted = summary.get("trusted_evidence") or {}
    chunk_ids = [str(value) for value in trusted.get("chunk_ids", []) if value is not None]
    document_ids = [str(value) for value in trusted.get("document_ids", []) if value is not None]
    citation_ids = [str(value) for value in trusted.get("citation_ids", []) if value is not None]
    candidates = []
    for index, chunk_id in enumerate(chunk_ids):
        candidates.append(
            {
                "rank": index + 1,
                "chunk_id": chunk_id,
                "document_id": document_ids[index] if index < len(document_ids) else None,
                "citation_id": citation_ids[index] if index < len(citation_ids) else None,
            }
        )
    return {
        "result_count": int(summary.get("result_count") or len(candidates)),
        "trusted_evidence_digest": trusted.get("evidence_identity_digest"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "identity_digest": stable_digest({"trusted": trusted, "result_count": summary.get("result_count")}),
    }


def _candidate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("candidates") or []
    return [row for row in rows if isinstance(row, dict)]


def _identity(row: dict[str, Any]) -> str:
    chunk = row.get("chunk_id")
    doc = row.get("document_id")
    return f"{doc or 'unknown-doc'}::{chunk or row.get('citation_id') or row.get('rank')}"


def _documents(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["document_id"]) for row in rows if row.get("document_id")}


def _scopes(rows: list[dict[str, Any]]) -> set[str]:
    scopes = set()
    for row in rows:
        scope = row.get("scope_id") or row.get("heading_path") or row.get("chunk_id")
        if isinstance(scope, list):
            scope = "/".join(str(part) for part in scope)
        if scope:
            scopes.add(str(scope))
    return scopes


def _rank_changes(initial_ids: list[str], recovery_ids: list[str]) -> dict[str, int]:
    initial_rank = {identity: index for index, identity in enumerate(initial_ids, start=1)}
    changes = {}
    for index, identity in enumerate(recovery_ids, start=1):
        if identity in initial_rank and initial_rank[identity] != index:
            changes[stable_digest(identity)] = initial_rank[identity] - index
    return changes


def _score(summary: dict[str, Any]) -> float | None:
    value = summary.get("retrieval_score")
    return value if isinstance(value, (int, float)) else None


def _answerability_change(initial: str | None, recovery: str | None) -> str | None:
    if initial is None or recovery is None:
        return None
    if initial == recovery:
        return "unchanged"
    return f"{initial}_to_{recovery}"


def _classify_outcome(*, overlap: int, initial_count: int, new_count: int, removed_count: int, top_changed: bool, answerability_change: str | None) -> EvidenceOutcome:
    if new_count == 0 and removed_count == 0 and not top_changed:
        return "evidence_unchanged"
    if answerability_change in {"insufficient_evidence_to_answerable", "unanswerable_to_answerable", "insufficient_evidence_to_partially_answerable"}:
        return "evidence_materially_improved"
    if new_count > 0 and (overlap > 0 or initial_count == 0):
        return "evidence_changed_not_improved"
    if removed_count > 0 and new_count == 0:
        return "evidence_degraded"
    return "evidence_changed_not_improved"
