from __future__ import annotations

from collections import Counter
from typing import Any

STATUS_FIELDS = (
    "retrieval_status",
    "decision_status",
    "generation_status",
    "citation_status",
    "grounding_status",
    "final_action",
    "primary_failure_attribution",
)


def compare_runs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_ids = [str(_sample_id(row)) for row in left]
    right_ids = [str(_sample_id(row)) for row in right]
    left_by_id = dict(zip(left_ids, left, strict=True))
    right_by_id = dict(zip(right_ids, right, strict=True))
    common_ids = [sample_id for sample_id in left_ids if sample_id in right_by_id]
    sample_changes = []
    for sample_id in common_ids:
        before = left_by_id[sample_id]
        after = right_by_id[sample_id]
        candidate_changed = _candidate_ids(before) != _candidate_ids(after)
        evidence_changed = _evidence_ids(before) != _evidence_ids(after)
        status_changes = {
            field: {"before": before.get(field), "after": after.get(field)}
            for field in STATUS_FIELDS
            if before.get(field) != after.get(field)
        }
        metric_changes = _metric_changes(before, after)
        if candidate_changed or evidence_changed or status_changes or metric_changes:
            sample_changes.append(
                {
                    "id": sample_id,
                    "retrieval_changed": candidate_changed,
                    "evidence_bundle_changed": evidence_changed,
                    "generation_changed_with_same_evidence": (not evidence_changed) and any(field in status_changes for field in ("generation_status", "final_action", "citation_status", "grounding_status")),
                    "upstream_retrieval_changed": candidate_changed or evidence_changed,
                    "status_changes": status_changes,
                    "metric_changes": metric_changes,
                    "candidate_ids_before": _candidate_ids(before),
                    "candidate_ids_after": _candidate_ids(after),
                    "evidence_ids_before": _evidence_ids(before),
                    "evidence_ids_after": _evidence_ids(after),
                }
            )
    return {
        "left_count": len(left),
        "right_count": len(right),
        "sample_id_order_identical": left_ids == right_ids,
        "missing_from_right": [sample_id for sample_id in left_ids if sample_id not in right_by_id],
        "missing_from_left": [sample_id for sample_id in right_ids if sample_id not in left_by_id],
        "changed_sample_count": len(sample_changes),
        "retrieval_changed_count": sum(row["retrieval_changed"] for row in sample_changes),
        "evidence_bundle_changed_count": sum(row["evidence_bundle_changed"] for row in sample_changes),
        "generation_changed_with_same_evidence_count": sum(row["generation_changed_with_same_evidence"] for row in sample_changes),
        "status_change_distribution": dict(sorted(Counter(field for row in sample_changes for field in row["status_changes"]).items())),
        "changes": sample_changes,
    }


def _sample_id(row: dict[str, Any]) -> object:
    return row.get("id") or row.get("sample_id")


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    return [str(item.get("chunk_id") or item.get("relative_path")) for item in candidates]


def _evidence_ids(row: dict[str, Any]) -> list[str]:
    evidence = row.get("evidence_bundle") or row.get("evidence_items") or []
    if not isinstance(evidence, list):
        return []
    return [str(item.get("chunk_id") or item.get("relative_path")) for item in evidence]


def _metric_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = ("first_relevant_rank", "candidate_hit", "evidence_bundle_hit", "reciprocal_rank")
    return {key: {"before": before.get(key), "after": after.get(key)} for key in keys if before.get(key) != after.get(key)}
