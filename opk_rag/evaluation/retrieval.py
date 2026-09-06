from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.search.models import EvidenceBundle, SearchResponse, SearchResult

RETRIEVAL_GOLD_SCHEMA_VERSION = "retrieval-gold.v1"
RETRIEVAL_FIXTURE_SCHEMA_VERSION = "retrieval-fixture.v1"
DEFAULT_HIT_KS = (1, 3, 5, 10)
DEFAULT_RECALL_KS = (3, 5, 10)


@dataclass(frozen=True)
class GoldEvidenceItem:
    source_file: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    heading: str | None = None
    excerpt: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    citation_id: str | None = None


@dataclass(frozen=True)
class GoldEvidenceContract:
    sample_id: str
    retrieval_evaluable: bool
    reason: str | None
    items: tuple[GoldEvidenceItem, ...]


def gold_contract_from_record(record: dict[str, Any]) -> GoldEvidenceContract:
    items = tuple(
        GoldEvidenceItem(
            source_file=_normalize_source_path(row.get("source_file")),
            heading=_none_if_empty(row.get("heading")),
            excerpt=_none_if_empty(row.get("excerpt")),
        )
        for row in record.get("evidence", ())
        if isinstance(row, dict) and row.get("source_file")
    )
    if not items:
        reason = "no_gold_evidence_for_no_evidence_sample" if record.get("answerability_type") == "no_evidence" else "missing_structured_evidence"
        return GoldEvidenceContract(str(record["id"]), False, reason, ())
    return GoldEvidenceContract(str(record["id"]), True, None, items)


def candidate_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "document_id": str(result.document_id),
        "chunk_id": str(result.chunk_id),
        "relative_path": result.relative_path,
        "heading_path": list(result.heading_path),
        "start_line": result.start_line,
        "end_line": result.end_line,
        "similarity": result.similarity,
        "vector_similarity": result.vector_similarity,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "vector_rank": result.vector_rank,
        "bm25_rank": result.bm25_rank,
        "rerank_score": result.rerank_score,
        "rerank_rank": result.rerank_rank,
        "selected_for_context": result.selected_for_context,
        "context_rank": result.context_rank,
        "context_token_count": result.context_token_count,
        "retrieval_sources": list(result.retrieval_sources),
        "matched_terms": list(result.matched_terms),
        "content": result.content,
    }


def evidence_item_payload(item) -> dict[str, Any]:
    return {
        "context_rank": item.context_rank,
        "result_rank": item.result_rank,
        "document_id": str(item.document_id),
        "chunk_id": str(item.chunk_id),
        "relative_path": item.relative_path,
        "heading_path": list(item.heading_path),
        "start_line": item.start_line,
        "end_line": item.end_line,
        "context_token_count": item.context_token_count,
        "retrieval_sources": list(item.retrieval_sources),
        "content": item.content,
    }


def score_retrieval_response(record: dict[str, Any], response: SearchResponse, *, hit_ks: tuple[int, ...] = DEFAULT_HIT_KS, recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS) -> dict[str, Any]:
    return score_retrieval_payload(
        record,
        candidates=[candidate_payload(result) for result in response.results],
        evidence_bundle=[evidence_item_payload(item) for item in response.evidence_bundle.items] if response.evidence_bundle else [],
        retrieval_config=response_config(response),
        hit_ks=hit_ks,
        recall_ks=recall_ks,
    )


def score_retrieval_payload(
    record: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    evidence_bundle: list[dict[str, Any]],
    retrieval_config: dict[str, Any] | None = None,
    hit_ks: tuple[int, ...] = DEFAULT_HIT_KS,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
) -> dict[str, Any]:
    gold = gold_contract_from_record(record)
    matched_candidates = _matched_positions(gold.items, candidates) if gold.retrieval_evaluable else []
    matched_bundle = _matched_positions(gold.items, evidence_bundle) if gold.retrieval_evaluable else []
    first_rank = min((position for _, position, _ in matched_candidates), default=None)
    unique_gold_count = _unique_gold_count(gold.items)
    matched_gold_by_k = {
        k: {gold_key for gold_key, position, _ in matched_candidates if position <= k}
        for k in recall_ks
    }
    hit_at = {str(k): bool(first_rank is not None and first_rank <= k) for k in hit_ks}
    recall_at = {
        str(k): (len(matched_gold_by_k[k]) / unique_gold_count if unique_gold_count else None)
        for k in recall_ks
    }
    bundle_gold_keys = {gold_key for gold_key, _, _ in matched_bundle}
    return {
        "schema_version": RETRIEVAL_GOLD_SCHEMA_VERSION,
        "id": record["id"],
        "question": record["question"],
        "split": record.get("split"),
        "expected_answerability": record.get("expected_answerability"),
        "answerability_type": record.get("answerability_type"),
        "expected_action": record.get("expected_action"),
        "retrieval_evaluable": gold.retrieval_evaluable,
        "exclusion_reason": gold.reason,
        "gold_evidence": [_gold_payload(item) for item in gold.items],
        "candidate_count": len(candidates),
        "evidence_bundle_count": len(evidence_bundle),
        "first_relevant_rank": first_rank,
        "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
        "hit_at": hit_at,
        "recall_at": recall_at,
        "candidate_hit": bool(matched_candidates),
        "evidence_bundle_hit": bool(bundle_gold_keys),
        "candidate_gold_matches": [
            {"gold_key": gold_key, "rank": rank, "chunk_id": candidate.get("chunk_id"), "relative_path": candidate.get("relative_path")}
            for gold_key, rank, candidate in matched_candidates
        ],
        "evidence_bundle_gold_matches": [
            {"gold_key": gold_key, "context_rank": rank, "chunk_id": item.get("chunk_id"), "relative_path": item.get("relative_path")}
            for gold_key, rank, item in matched_bundle
        ],
        "candidates": candidates,
        "evidence_bundle": evidence_bundle,
        "retrieval_config": retrieval_config or {},
        "diagnostics": {
            "unique_gold_evidence_count": unique_gold_count,
            "duplicate_candidate_rate": _duplicate_rate(candidates),
            "unique_document_count": len({candidate.get("relative_path") for candidate in candidates if candidate.get("relative_path")}),
            "vector_only_hit": _source_hit(matched_candidates, "vector", only=True),
            "lexical_only_hit": _source_hit(matched_candidates, "bm25", only=True),
            "fused_hit": _source_hit(matched_candidates, "vector", only=False) and _source_hit(matched_candidates, "bm25", only=False),
        },
    }


def aggregate_retrieval_results(rows: list[dict[str, Any]], *, hit_ks: tuple[int, ...] = DEFAULT_HIT_KS, recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("retrieval_evaluable")]
    excluded = [row for row in rows if not row.get("retrieval_evaluable")]
    denominator = len(evaluable)
    metrics = {
        f"hit_at_{k}": _mean_bool(row["hit_at"][str(k)] for row in evaluable) for k in hit_ks
    }
    metrics.update({f"recall_at_{k}": _mean_float(row["recall_at"][str(k)] for row in evaluable) for k in recall_ks})
    metrics["mrr"] = _mean_float(row["reciprocal_rank"] for row in evaluable)
    metrics["evidence_bundle_hit_rate"] = _mean_bool(row["evidence_bundle_hit"] for row in evaluable)
    first_ranks = [row["first_relevant_rank"] for row in evaluable if row["first_relevant_rank"] is not None]
    metrics["average_first_relevant_rank"] = (sum(first_ranks) / len(first_ranks)) if first_ranks else None
    return {
        "sample_count": len(rows),
        "retrieval_evaluable_count": denominator,
        "excluded_count": len(excluded),
        "excluded_samples": [{"id": row["id"], "reason": row.get("exclusion_reason")} for row in excluded],
        "metrics": metrics,
        "candidate_hit_bundle_miss_ids": [row["id"] for row in evaluable if row["candidate_hit"] and not row["evidence_bundle_hit"]],
        "retrieval_miss_ids": [row["id"] for row in evaluable if not row["candidate_hit"]],
        "bundle_hit_ids": [row["id"] for row in evaluable if row["evidence_bundle_hit"]],
        "candidate_hit_ids": [row["id"] for row in evaluable if row["candidate_hit"]],
    }


def response_config(response: SearchResponse) -> dict[str, Any]:
    return {
        "knowledge_base_id": str(response.knowledge_base_id),
        "embedding_model_id": response.model_id,
        "retrieval_mode": response.retrieval_mode,
        "query_template_version": response.query_template_version,
        "requested_top_k": response.requested_top_k,
        "candidate_k": response.candidate_k,
        "vector_candidate_count": response.vector_candidate_count,
        "bm25_candidate_count": response.bm25_candidate_count,
        "final_top_k": response.final_top_k,
        "context_token_budget": response.context_token_budget,
        "context_token_count": response.context_token_count,
        "reranker_enabled": response.reranker_enabled,
    }


def _matched_positions(gold_items: tuple[GoldEvidenceItem, ...], payloads: list[dict[str, Any]]) -> list[tuple[str, int, dict[str, Any]]]:
    matches: list[tuple[str, int, dict[str, Any]]] = []
    seen: set[tuple[str, str | None]] = set()
    for index, payload in enumerate(payloads, start=1):
        for gold in gold_items:
            if _matches(gold, payload):
                gold_key = _gold_key(gold)
                candidate_key = str(payload.get("chunk_id") or f"{payload.get('relative_path')}:{payload.get('start_line')}:{payload.get('end_line')}")
                key = (gold_key, candidate_key)
                if key not in seen:
                    matches.append((gold_key, index, payload))
                    seen.add(key)
    return matches


def _matches(gold: GoldEvidenceItem, payload: dict[str, Any]) -> bool:
    if gold.chunk_id and gold.chunk_id == str(payload.get("chunk_id")):
        return True
    if gold.document_id and gold.document_id != str(payload.get("document_id")):
        return False
    if gold.source_file and _normalize_source_path(payload.get("relative_path")) != _normalize_source_path(gold.source_file):
        return False
    if gold.start_line is not None and payload.get("end_line") is not None and int(payload["end_line"]) < gold.start_line:
        return False
    if gold.end_line is not None and payload.get("start_line") is not None and int(payload["start_line"]) > gold.end_line:
        return False
    if gold.heading:
        heading_path = " > ".join(str(value) for value in payload.get("heading_path", ()))
        heading_matches = bool(heading_path) and (gold.heading in heading_path or gold.heading == heading_path.split(" > ")[-1])
        content = str(payload.get("content") or "")
        if not heading_matches and gold.heading not in content:
            return False
    if gold.excerpt and gold.excerpt not in str(payload.get("content") or ""):
        return False
    return bool(gold.source_file or gold.document_id or gold.chunk_id)


def _gold_key(gold: GoldEvidenceItem) -> str:
    return "|".join(
        part
        for part in (
            _normalize_source_path(gold.source_file),
            gold.heading or "",
            gold.excerpt or "",
            gold.chunk_id or "",
            gold.citation_id or "",
        )
        if part
    )


def _gold_payload(item: GoldEvidenceItem) -> dict[str, Any]:
    return {
        "source_file": item.source_file,
        "document_id": item.document_id,
        "chunk_id": item.chunk_id,
        "heading": item.heading,
        "excerpt": item.excerpt,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "citation_id": item.citation_id,
    }


def _unique_gold_count(items: tuple[GoldEvidenceItem, ...]) -> int:
    return len({_gold_key(item) for item in items})


def _normalize_source_path(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("source-documents/"):
        text = text[len("source-documents/") :]
    return text or None


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _duplicate_rate(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    keys = [candidate.get("chunk_id") or (candidate.get("relative_path"), candidate.get("start_line"), candidate.get("end_line")) for candidate in candidates]
    return (len(keys) - len(set(keys))) / len(keys)


def _source_hit(matches: list[tuple[str, int, dict[str, Any]]], source: str, *, only: bool) -> bool:
    for _, _, candidate in matches:
        sources = set(candidate.get("retrieval_sources") or ())
        if source in sources and (not only or len(sources) == 1):
            return True
    return False


def _mean_bool(values) -> float | None:
    data = [bool(value) for value in values]
    return (sum(data) / len(data)) if data else None


def _mean_float(values) -> float | None:
    data = [float(value) for value in values if value is not None]
    return (sum(data) / len(data)) if data else None
