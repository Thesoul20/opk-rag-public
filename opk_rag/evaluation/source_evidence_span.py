from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

from opk_rag.evaluation.evidence_identity import (
    EvidenceIdentity,
    deduplicate_evidence_identities,
    digest_json,
    normalize_gold_evidence_identity,
)
from opk_rag.evaluation.scope_structure_audit import IndexedScopeChunk, resolve_gold_scope_to_indexed_chunks


SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION = "opk-rag.canonical-source-evidence-span.v1"
SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION = "opk-rag.canonical-source-evidence-span-set.v1"
DEFAULT_ALLOWED_GAP = 1


@dataclass(frozen=True)
class SourceEvidenceSpan:
    required_evidence_unit_id: str
    dataset_id: str
    sample_id: str
    document_identity_digest: str
    source_line_start: int
    source_line_end: int
    heading_path_digest: str | None
    source_span_digest: str
    legacy_evidence_identity_digest: str
    legacy_chunk_identity_digests: tuple[str, ...]
    legacy_chunk_indexes: tuple[int, ...]
    old_chunk_count: int

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["legacy_chunk_identity_digests"] = list(self.legacy_chunk_identity_digests)
        payload["legacy_chunk_indexes"] = list(self.legacy_chunk_indexes)
        return payload


@dataclass(frozen=True)
class SourceEvidenceSpanSet:
    required_evidence_unit_id: str
    dataset_id: str
    sample_id: str
    document_identity_digest: str
    spans: tuple[tuple[int, int], ...]
    heading_path_digest: str | None
    source_span_digest: str
    legacy_evidence_identity_digest: str
    legacy_chunk_identity_digests: tuple[str, ...]
    legacy_chunk_indexes: tuple[int, ...]
    old_chunk_count: int

    @property
    def source_line_start(self) -> int:
        return self.spans[0][0]

    @property
    def source_line_end(self) -> int:
        return self.spans[-1][1]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION,
            "required_evidence_unit_id": self.required_evidence_unit_id,
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "document_identity_digest": self.document_identity_digest,
            "spans": [{"source_line_start": start, "source_line_end": end} for start, end in self.spans],
            "heading_path_digest": self.heading_path_digest,
            "source_span_digest": self.source_span_digest,
            "legacy_evidence_identity_digest": self.legacy_evidence_identity_digest,
            "legacy_chunk_identity_digests": list(self.legacy_chunk_identity_digests),
            "legacy_chunk_indexes": list(self.legacy_chunk_indexes),
            "old_chunk_count": self.old_chunk_count,
        }


SourceEvidenceResolution = SourceEvidenceSpan | SourceEvidenceSpanSet


def build_source_evidence_span(
    *,
    sample_id: str,
    dataset_id: str,
    unit_ordinal: int,
    gold: EvidenceIdentity,
    chunks: tuple[IndexedScopeChunk, ...],
) -> SourceEvidenceSpan:
    resolved = resolve_legacy_scope_to_source_span(gold, chunks)
    if not resolved:
        raise ValueError(f"unable to resolve source span for {sample_id} evidence unit {unit_ordinal}")
    document_digests = {chunk.document_identity_digest for chunk in resolved}
    if len(document_digests) != 1 or None in document_digests:
        raise ValueError(f"resolved legacy chunks do not identify one source document for {sample_id}")
    line_start, line_end = merge_contiguous_scope_spans(resolved)
    heading_path_digest = gold.heading_path_digest or resolved[0].heading_path_digest
    document_identity_digest = next(iter(document_digests))
    required_evidence_unit_id = digest_json(
        {
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "unit_ordinal": unit_ordinal,
            "legacy_identity": gold.to_json(),
        }
    )
    source_span_digest = digest_json(
        {
            "schema_version": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
            "document_identity_digest": document_identity_digest,
            "source_line_start": line_start,
            "source_line_end": line_end,
            "heading_path_digest": heading_path_digest,
            "required_evidence_unit_id": required_evidence_unit_id,
        }
    )
    span = SourceEvidenceSpan(
        required_evidence_unit_id=required_evidence_unit_id,
        dataset_id=dataset_id,
        sample_id=sample_id,
        document_identity_digest=document_identity_digest,
        source_line_start=line_start,
        source_line_end=line_end,
        heading_path_digest=heading_path_digest,
        source_span_digest=source_span_digest,
        legacy_evidence_identity_digest=digest_json(gold.to_json()),
        legacy_chunk_identity_digests=tuple(digest_json(chunk.identity) for chunk in resolved),
        legacy_chunk_indexes=tuple(chunk.chunk_index for chunk in resolved),
        old_chunk_count=len(resolved),
    )
    validate_source_span(span)
    return span


def build_source_evidence_resolution(
    *,
    sample_id: str,
    dataset_id: str,
    unit_ordinal: int,
    gold: EvidenceIdentity,
    chunks: tuple[IndexedScopeChunk, ...],
    allowed_gap: int = DEFAULT_ALLOWED_GAP,
) -> SourceEvidenceResolution:
    resolved = resolve_legacy_scope_to_source_span(gold, chunks)
    if not resolved:
        raise ValueError(f"unable to resolve source span for {sample_id} evidence unit {unit_ordinal}")
    document_digests = {chunk.document_identity_digest for chunk in resolved}
    if len(document_digests) != 1 or None in document_digests:
        raise ValueError(f"resolved legacy chunks do not identify one source document for {sample_id}")
    ordered = order_scope_chunks(resolved)
    heading_path_digest = gold.heading_path_digest or ordered[0].heading_path_digest
    document_identity_digest = next(iter(document_digests))
    required_evidence_unit_id = digest_json(
        {
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "unit_ordinal": unit_ordinal,
            "legacy_identity": gold.to_json(),
        }
    )
    merged = merge_line_ranges(ordered, allowed_gap=allowed_gap)
    legacy_chunk_identity_digests = tuple(digest_json(chunk.identity) for chunk in ordered)
    legacy_chunk_indexes = tuple(chunk.chunk_index for chunk in ordered)
    if len(merged) == 1:
        line_start, line_end = merged[0]
        source_span_digest = digest_json(
            {
                "schema_version": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
                "document_identity_digest": document_identity_digest,
                "source_line_start": line_start,
                "source_line_end": line_end,
                "heading_path_digest": heading_path_digest,
                "required_evidence_unit_id": required_evidence_unit_id,
            }
        )
        span = SourceEvidenceSpan(
            required_evidence_unit_id=required_evidence_unit_id,
            dataset_id=dataset_id,
            sample_id=sample_id,
            document_identity_digest=document_identity_digest,
            source_line_start=line_start,
            source_line_end=line_end,
            heading_path_digest=heading_path_digest,
            source_span_digest=source_span_digest,
            legacy_evidence_identity_digest=digest_json(gold.to_json()),
            legacy_chunk_identity_digests=legacy_chunk_identity_digests,
            legacy_chunk_indexes=legacy_chunk_indexes,
            old_chunk_count=len(ordered),
        )
        validate_source_span(span)
        return span
    source_span_digest = digest_json(
        {
            "schema_version": SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION,
            "document_identity_digest": document_identity_digest,
            "spans": merged,
            "heading_path_digest": heading_path_digest,
            "required_evidence_unit_id": required_evidence_unit_id,
        }
    )
    span_set = SourceEvidenceSpanSet(
        required_evidence_unit_id=required_evidence_unit_id,
        dataset_id=dataset_id,
        sample_id=sample_id,
        document_identity_digest=document_identity_digest,
        spans=tuple(merged),
        heading_path_digest=heading_path_digest,
        source_span_digest=source_span_digest,
        legacy_evidence_identity_digest=digest_json(gold.to_json()),
        legacy_chunk_identity_digests=legacy_chunk_identity_digests,
        legacy_chunk_indexes=legacy_chunk_indexes,
        old_chunk_count=len(ordered),
    )
    validate_source_span_set(span_set)
    return span_set


def resolve_legacy_scope_to_source_span(
    gold: EvidenceIdentity,
    chunks: Iterable[IndexedScopeChunk],
) -> tuple[IndexedScopeChunk, ...]:
    resolved = resolve_gold_scope_to_indexed_chunks(gold, chunks)
    with_lines = tuple(chunk for chunk in resolved if chunk.start_line is not None and chunk.end_line is not None)
    if with_lines:
        return tuple(sorted(with_lines, key=lambda chunk: (chunk.start_line or 0, chunk.end_line or 0, chunk.chunk_index)))
    return tuple(resolved)


def resolve_scope_document(gold: EvidenceIdentity, chunks: Iterable[IndexedScopeChunk]) -> str | None:
    resolved = resolve_legacy_scope_to_source_span(gold, chunks)
    document_digests = {chunk.document_identity_digest for chunk in resolved if chunk.document_identity_digest}
    return next(iter(document_digests)) if len(document_digests) == 1 else None


def resolve_scope_chunks(gold: EvidenceIdentity, chunks: Iterable[IndexedScopeChunk]) -> tuple[IndexedScopeChunk, ...]:
    return resolve_legacy_scope_to_source_span(gold, chunks)


def order_scope_chunks(chunks: Iterable[IndexedScopeChunk]) -> tuple[IndexedScopeChunk, ...]:
    deduped: dict[str, IndexedScopeChunk] = {}
    for chunk in chunks:
        key = str(chunk.chunk_id) if chunk.chunk_id else digest_json(chunk.identity)
        deduped.setdefault(key, chunk)
    rows = tuple(deduped.values())
    if any(chunk.start_line is None or chunk.end_line is None for chunk in rows):
        raise ValueError("all resolved chunks must have source line ranges")
    return tuple(sorted(rows, key=lambda chunk: (chunk.start_line or 0, chunk.end_line or 0, chunk.chunk_index)))


def normalize_line_ranges(chunks: Iterable[IndexedScopeChunk]) -> tuple[tuple[int, int], ...]:
    return tuple((int(chunk.start_line or 0), int(chunk.end_line or 0)) for chunk in order_scope_chunks(chunks))


def merge_line_ranges(chunks: Iterable[IndexedScopeChunk], *, allowed_gap: int = DEFAULT_ALLOWED_GAP) -> list[tuple[int, int]]:
    if allowed_gap < 0:
        raise ValueError("allowed_gap must be non-negative")
    ranges = list(normalize_line_ranges(chunks))
    if not ranges:
        raise ValueError("cannot merge an empty chunk span")
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if start <= 0 or end < start:
            raise ValueError("source span has invalid source line range")
        if not merged or start > merged[-1][1] + allowed_gap + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def merge_contiguous_ranges(chunks: tuple[IndexedScopeChunk, ...], *, allowed_gap: int = DEFAULT_ALLOWED_GAP) -> tuple[int, int]:
    merged = merge_line_ranges(chunks, allowed_gap=allowed_gap)
    if len(merged) != 1:
        raise ValueError("resolved source spans are not contiguous")
    return merged[0]


def merge_contiguous_scope_spans(chunks: tuple[IndexedScopeChunk, ...]) -> tuple[int, int]:
    if not chunks:
        raise ValueError("cannot merge an empty chunk span")
    starts = [chunk.start_line for chunk in chunks]
    ends = [chunk.end_line for chunk in chunks]
    if any(value is None for value in starts + ends):
        raise ValueError("all resolved chunks must have source line ranges")
    ordered = sorted(chunks, key=lambda chunk: (chunk.start_line or 0, chunk.end_line or 0, chunk.chunk_index))
    for left, right in zip(ordered, ordered[1:]):
        if (right.start_line or 0) > (left.end_line or 0) + 1:
            raise ValueError("resolved source spans are not contiguous")
    return int(ordered[0].start_line), int(max(chunk.end_line or 0 for chunk in ordered))


def validate_source_span(span: SourceEvidenceSpan) -> None:
    if not span.document_identity_digest:
        raise ValueError("source span missing document_identity_digest")
    if span.source_line_start <= 0 or span.source_line_end < span.source_line_start:
        raise ValueError("source span has invalid source line range")
    if not span.source_span_digest or not span.required_evidence_unit_id:
        raise ValueError("source span missing stable digests")


def validate_source_span_set(span_set: SourceEvidenceSpanSet) -> None:
    if not span_set.document_identity_digest:
        raise ValueError("source span set missing document_identity_digest")
    if not span_set.spans:
        raise ValueError("source span set has no spans")
    previous_end = 0
    for start, end in span_set.spans:
        if start <= 0 or end < start:
            raise ValueError("source span set has invalid source line range")
        if start <= previous_end:
            raise ValueError("source span set spans must be ordered and non-overlapping")
        previous_end = end
    if not span_set.source_span_digest or not span_set.required_evidence_unit_id:
        raise ValueError("source span set missing stable digests")


def match_shadow_chunk_to_source_span(
    chunk: Any,
    span: SourceEvidenceResolution,
    *,
    use_complete_span: bool = True,
) -> str:
    if getattr(chunk, "document_identity_digest", None) != span.document_identity_digest:
        return "no_overlap"
    source = chunk.complete_source_span if use_complete_span else chunk.primary_source_span
    ranges = _resolution_ranges(span)
    if not any(_line_overlap(source["start_line"], source["end_line"], start, end) for start, end in ranges):
        return "no_overlap"
    if len(ranges) == 1 and source["start_line"] <= ranges[0][0] and source["end_line"] >= ranges[0][1]:
        return "full_span_containment"
    return "partial_overlap"


def calculate_span_union_coverage(chunks: Iterable[Any], span: SourceEvidenceResolution) -> str:
    intervals: list[tuple[int, int]] = []
    ranges = _resolution_ranges(span)
    for chunk in chunks:
        if getattr(chunk, "document_identity_digest", None) != span.document_identity_digest:
            continue
        source = chunk.complete_source_span
        for start, end in ranges:
            overlap_start = max(source["start_line"], start)
            overlap_end = min(source["end_line"], end)
            if overlap_start <= overlap_end:
                intervals.append((overlap_start, overlap_end))
    if not intervals:
        return "no_overlap"
    merged = _merge_intervals(intervals)
    if _covers_all_ranges(merged, ranges):
        return "full_span_containment" if len(intervals) == 1 else "multi_candidate_complete_coverage"
    return "partial_overlap"


def build_public_source_span_manifest(spans: Iterable[SourceEvidenceSpan]) -> dict[str, Any]:
    rows = [span.to_json() for span in spans]
    unique = {row["source_span_digest"] for row in rows}
    multi_old = sum(1 for row in rows if row["old_chunk_count"] > 1)
    return {
        "schema_version": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
        "required_evidence_unit_count": len(rows),
        "resolved_source_span_count": len(rows),
        "unresolved_source_span_count": 0,
        "unique_source_span_count": len(unique),
        "multi_old_chunk_span_count": multi_old,
        "publishes_source_text": False,
        "uses_legacy_chunk_identity_for_scoring": False,
        "span_digest": digest_json(rows),
        "spans": rows,
    }


def build_public_source_span_resolution_manifest(spans: Iterable[SourceEvidenceResolution]) -> dict[str, Any]:
    rows = [span.to_json() for span in spans]
    unique = {row["source_span_digest"] for row in rows}
    multi_old = sum(1 for row in rows if row["old_chunk_count"] > 1)
    span_set_count = sum(1 for row in rows if row.get("schema_version") == SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION)
    contiguous_count = len(rows) - span_set_count
    return {
        "schema_version": "opk-rag.canonical-source-evidence-resolution.v1",
        "span_schema_version": SOURCE_EVIDENCE_SPAN_SCHEMA_VERSION,
        "span_set_schema_version": SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION,
        "required_evidence_unit_count": len(rows),
        "resolved_source_span_count": len(rows),
        "unresolved_source_span_count": 0,
        "unique_source_span_count": len(unique),
        "contiguous_span_count": contiguous_count,
        "span_set_count": span_set_count,
        "multi_old_chunk_span_count": multi_old,
        "publishes_source_text": False,
        "uses_legacy_chunk_identity_for_scoring": False,
        "span_digest": digest_json(rows),
        "spans": rows,
    }


def build_source_evidence_spans_from_samples(
    samples: Iterable[dict[str, Any]],
    chunks: tuple[IndexedScopeChunk, ...],
) -> tuple[SourceEvidenceSpan, ...]:
    spans: list[SourceEvidenceSpan] = []
    for sample in samples:
        identities = deduplicate_evidence_identities(
            normalize_gold_evidence_identity(unit) for unit in (sample.get("required_evidence") or [])
        )
        for ordinal, gold in enumerate(identities):
            spans.append(
                build_source_evidence_span(
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    unit_ordinal=ordinal,
                    gold=gold,
                    chunks=chunks,
                )
            )
    return tuple(spans)


def build_source_evidence_resolutions_from_samples(
    samples: Iterable[dict[str, Any]],
    chunks: tuple[IndexedScopeChunk, ...],
    *,
    allowed_gap: int = DEFAULT_ALLOWED_GAP,
) -> tuple[SourceEvidenceResolution, ...]:
    spans: list[SourceEvidenceResolution] = []
    for sample in samples:
        identities = deduplicate_evidence_identities(
            normalize_gold_evidence_identity(unit) for unit in (sample.get("required_evidence") or [])
        )
        for ordinal, gold in enumerate(identities):
            spans.append(
                build_source_evidence_resolution(
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    unit_ordinal=ordinal,
                    gold=gold,
                    chunks=chunks,
                    allowed_gap=allowed_gap,
                )
            )
    return tuple(spans)


def source_text_digest(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _line_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start) + 1)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _resolution_ranges(span: SourceEvidenceResolution) -> tuple[tuple[int, int], ...]:
    if isinstance(span, SourceEvidenceSpanSet) or hasattr(span, "spans"):
        return span.spans
    return ((span.source_line_start, span.source_line_end),)


def _covers_all_ranges(merged: list[tuple[int, int]], ranges: tuple[tuple[int, int], ...]) -> bool:
    for start, end in ranges:
        if not any(left <= start and right >= end for left, right in merged):
            return False
    return True
