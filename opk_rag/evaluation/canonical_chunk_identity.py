from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from opk_rag.evaluation.candidate_injection import InjectedRetrievalCandidate
from opk_rag.search.models import SearchResult


CANONICAL_CHUNK_SCHEMA_VERSION = "opk-rag.canonical-chunk-identity.v1"
CANONICAL_MAPPING_SCHEMA_VERSION = "opk-rag.canonical-chunk-mapping.v1"
STRICT_VERIFICATION_FACTORS = (
    "source_path",
    "document_id",
    "section_id",
    "heading_path",
    "start_offset",
    "end_offset",
    "chunk_text_digest",
    "corpus_snapshot_digest",
)


class CanonicalChunkIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalChunkRecord:
    canonical_chunk_id: str
    evaluation_chunk_digest: str | None
    runtime_chunk_uuid: str | None
    content_digest: str | None
    source_path: str | None
    document_id: str | None
    section_id: str | None
    heading_path: tuple[str, ...]
    start_offset: int | None
    end_offset: int | None
    chunk_text_digest: str | None
    corpus_snapshot_digest: str | None
    chunk_text: str | None = None
    runtime_document_uuid: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    token_count: int | None = None

    @property
    def schema_version(self) -> str:
        return CANONICAL_CHUNK_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version, **asdict(self)}
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class ChunkResolution:
    status: str
    evaluation_chunk_digest: str | None
    runtime_chunk_uuid: str | None
    canonical_chunk_id: str | None
    issue_code: str | None
    verification_factors: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verification_factors"] = list(self.verification_factors)
        return payload


@dataclass(frozen=True)
class CanonicalChunkMapping:
    records: tuple[CanonicalChunkRecord, ...]
    resolutions: tuple[ChunkResolution, ...]
    corpus_snapshot_digest: str | None

    def to_json(self) -> dict[str, Any]:
        coverage = canonical_mapping_coverage(self.resolutions)
        return {
            "schema_version": CANONICAL_MAPPING_SCHEMA_VERSION,
            "canonical_identity_resolution_strict": True,
            "identity_verification_factor_count": len(STRICT_VERIFICATION_FACTORS),
            "corpus_snapshot_digest": self.corpus_snapshot_digest,
            "canonical_mapping_coverage": coverage,
            "records": [record.to_json() for record in self.records],
            "resolutions": [resolution.to_json() for resolution in self.resolutions],
        }

    def evaluation_to_canonical(self) -> dict[str, CanonicalChunkRecord]:
        return {record.evaluation_chunk_digest: record for record in self.records if record.evaluation_chunk_digest}

    def canonical_to_runtime(self) -> dict[str, str]:
        return {record.canonical_chunk_id: record.runtime_chunk_uuid for record in self.records if record.runtime_chunk_uuid}

    def runtime_to_canonical(self) -> dict[str, CanonicalChunkRecord]:
        return {record.runtime_chunk_uuid: record for record in self.records if record.runtime_chunk_uuid}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    return digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def canonical_chunk_id_for(record: CanonicalChunkRecord | dict[str, Any]) -> str:
    payload = record.to_json() if isinstance(record, CanonicalChunkRecord) else dict(record)
    authority = {
        "schema_version": CANONICAL_CHUNK_SCHEMA_VERSION,
        "corpus_snapshot_digest": payload.get("corpus_snapshot_digest"),
        "source_path": _norm_path(payload.get("source_path")),
        "heading_path": tuple(payload.get("heading_path") or ()),
        "chunk_text_digest": payload.get("chunk_text_digest"),
    }
    return digest_json(authority)


def build_corpus_snapshot_digest(records: Iterable[dict[str, Any]]) -> str:
    rows = []
    for row in records:
        rows.append(
            {
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "section_id": row.get("section_id"),
                "source_path": _norm_path(row.get("source_path") or row.get("relative_path")),
                "heading_path": tuple(row.get("heading_path") or ()),
                "start_offset": _int_or_none(row.get("start_offset")),
                "end_offset": _int_or_none(row.get("end_offset")),
                "content_digest": row.get("content_digest") or row.get("content_hash"),
            }
        )
    return digest_json(rows)


def build_evaluation_records(
    chunk_inventory: dict[str, Any],
    *,
    vault_path: Path | None = None,
    corpus_snapshot_digest: str | None = None,
) -> tuple[CanonicalChunkRecord, ...]:
    provenance = list(chunk_inventory.get("provenance") or [])
    snapshot = corpus_snapshot_digest or build_corpus_snapshot_digest(provenance)
    records = []
    for row in provenance:
        source_path = _text(row.get("source_path") or row.get("relative_path"))
        chunk_text = _read_chunk_text(vault_path, source_path, _int_or_none(row.get("start_offset")), _int_or_none(row.get("end_offset")))
        chunk_digest = digest_text(chunk_text) if chunk_text is not None else _text(row.get("chunk_text_digest") or row.get("content_digest") or row.get("content_hash"))
        seed = {
            "evaluation_chunk_digest": _text(row.get("chunk_id")),
            "runtime_chunk_uuid": _text(row.get("runtime_chunk_uuid")),
            "content_digest": _text(row.get("content_digest") or row.get("content_hash")),
            "source_path": source_path,
            "document_id": _text(row.get("document_id")),
            "section_id": _text(row.get("section_id")),
            "heading_path": tuple(str(item) for item in (row.get("heading_path") or ())),
            "start_offset": _int_or_none(row.get("start_offset")),
            "end_offset": _int_or_none(row.get("end_offset")),
            "chunk_text_digest": chunk_digest,
            "corpus_snapshot_digest": snapshot,
            "chunk_text": chunk_text,
            "runtime_document_uuid": _text(row.get("runtime_document_uuid")),
            "start_line": _int_or_none(row.get("start_line")),
            "end_line": _int_or_none(row.get("end_line")),
            "token_count": _int_or_none(row.get("token_count")),
        }
        record = CanonicalChunkRecord(canonical_chunk_id="", **seed)
        records.append(replace(record, canonical_chunk_id=canonical_chunk_id_for(record)))
    return tuple(records)


def build_runtime_record(
    row: Any,
    *,
    corpus_snapshot_digest: str | None,
    runtime_chunk_uuid: str | None = None,
) -> CanonicalChunkRecord:
    payload = row if isinstance(row, dict) else row.__dict__
    source_path = _text(payload.get("source_path") or payload.get("relative_path"))
    content = _text(payload.get("content"))
    content_digest = _text(payload.get("content_digest") or payload.get("content_hash"))
    chunk_digest = _text(payload.get("chunk_text_digest")) or (digest_text(content) if content is not None else content_digest)
    record = CanonicalChunkRecord(
        canonical_chunk_id="",
        evaluation_chunk_digest=_text(payload.get("evaluation_chunk_digest")),
        runtime_chunk_uuid=runtime_chunk_uuid or _text(payload.get("runtime_chunk_uuid") or payload.get("chunk_id")),
        content_digest=content_digest,
        source_path=source_path,
        document_id=_text(payload.get("evaluation_document_id") or payload.get("document_identity_digest") or payload.get("document_id")),
        section_id=_text(payload.get("evaluation_section_id") or payload.get("section_id")),
        heading_path=tuple(str(item) for item in (payload.get("heading_path") or ())),
        start_offset=_int_or_none(payload.get("start_offset")),
        end_offset=_int_or_none(payload.get("end_offset")),
        chunk_text_digest=chunk_digest,
        corpus_snapshot_digest=corpus_snapshot_digest,
        chunk_text=content,
        runtime_document_uuid=_text(payload.get("runtime_document_uuid") or payload.get("document_id")),
        start_line=_int_or_none(payload.get("start_line")),
        end_line=_int_or_none(payload.get("end_line")),
        token_count=_int_or_none(payload.get("token_count")),
    )
    return replace(record, canonical_chunk_id=canonical_chunk_id_for(record))


def build_strict_mapping(
    evaluation_records: Iterable[CanonicalChunkRecord],
    *,
    runtime_records: Iterable[CanonicalChunkRecord] | None = None,
) -> CanonicalChunkMapping:
    evaluation = tuple(evaluation_records)
    runtime = tuple(_evaluation_as_runtime_records(evaluation) if runtime_records is None else runtime_records)
    by_eval_digest = _group(evaluation, "evaluation_chunk_digest")
    by_canonical_runtime = _group(runtime, "canonical_chunk_id")
    resolutions = []
    records = []
    for digest, candidates in sorted(by_eval_digest.items()):
        if len(candidates) != 1:
            resolutions.append(ChunkResolution("unresolved", digest, None, None, "ambiguous_evaluation_identity", ()))
            continue
        evaluation_record = candidates[0]
        matches = by_canonical_runtime.get(evaluation_record.canonical_chunk_id, [])
        if not matches:
            provenance_matches = [record for record in runtime if _same_provenance_without_content(evaluation_record, record)]
            if len(provenance_matches) == 1:
                resolutions.append(
                    ChunkResolution(
                        "digest_mismatch",
                        digest,
                        provenance_matches[0].runtime_chunk_uuid,
                        evaluation_record.canonical_chunk_id,
                        "content_digest_mismatch",
                        (),
                    )
                )
                continue
            if len(provenance_matches) > 1:
                resolutions.append(ChunkResolution("ambiguous", digest, None, evaluation_record.canonical_chunk_id, "ambiguous_runtime_match", ()))
                continue
            resolutions.append(ChunkResolution("unresolved", digest, None, evaluation_record.canonical_chunk_id, "missing_runtime_match", ()))
            continue
        if len(matches) > 1:
            resolutions.append(ChunkResolution("ambiguous", digest, None, evaluation_record.canonical_chunk_id, "ambiguous_runtime_match", ()))
            continue
        runtime_record = matches[0]
        issue = _strict_mismatch(evaluation_record, runtime_record)
        if issue is not None:
            resolutions.append(ChunkResolution("digest_mismatch", digest, runtime_record.runtime_chunk_uuid, evaluation_record.canonical_chunk_id, issue, ()))
            continue
        merged = replace(
            evaluation_record,
            runtime_chunk_uuid=runtime_record.runtime_chunk_uuid,
            runtime_document_uuid=runtime_record.runtime_document_uuid,
            chunk_text=runtime_record.chunk_text or evaluation_record.chunk_text,
            start_line=runtime_record.start_line,
            end_line=runtime_record.end_line,
        )
        records.append(merged)
        resolutions.append(
            ChunkResolution(
                "resolved",
                digest,
                merged.runtime_chunk_uuid,
                merged.canonical_chunk_id,
                None,
                tuple(field for field in STRICT_VERIFICATION_FACTORS if _factor_value(merged, field) is not None),
            )
        )
    snapshot_counts = Counter(record.corpus_snapshot_digest for record in records if record.corpus_snapshot_digest)
    snapshot = snapshot_counts.most_common(1)[0][0] if snapshot_counts else None
    return CanonicalChunkMapping(tuple(records), tuple(resolutions), snapshot)


def canonical_mapping_coverage(resolutions: Iterable[ChunkResolution]) -> dict[str, Any]:
    rows = list(resolutions)
    total = len(rows)
    resolved = sum(row.status == "resolved" for row in rows)
    unresolved = sum(row.issue_code == "missing_runtime_match" for row in rows)
    ambiguous = sum(row.issue_code in {"ambiguous_runtime_match", "ambiguous_evaluation_identity"} for row in rows)
    mismatch = sum(row.issue_code in {"content_digest_mismatch", "provenance_mismatch", "corpus_snapshot_mismatch"} for row in rows)
    return {
        "total_candidate_count": total,
        "uniquely_resolved_count": resolved,
        "unresolved_count": unresolved,
        "ambiguous_count": ambiguous,
        "digest_mismatch_count": mismatch,
        "coverage": resolved / total if total else 0.0,
    }


def resolve_injected_candidates_to_runtime(
    candidates: Iterable[InjectedRetrievalCandidate],
    mapping: CanonicalChunkMapping,
) -> tuple[list[InjectedRetrievalCandidate], dict[str, Any]]:
    candidate_rows = list(candidates)
    by_eval = mapping.evaluation_to_canonical()
    issue_rows = []
    resolved = []
    for candidate in candidate_rows:
        record = by_eval.get(candidate.chunk_id)
        if record is None or not record.runtime_chunk_uuid:
            issue_rows.append({"code": "missing_runtime_match", "sample_unit_id": candidate.sample_unit_id, "chunk_id": candidate.chunk_id})
            continue
        resolved.append(
            replace(
                candidate,
                chunk_id=record.runtime_chunk_uuid,
                provenance={
                    **candidate.provenance,
                    "evaluation_chunk_digest": candidate.chunk_id,
                    "canonical_chunk_id": record.canonical_chunk_id,
                    "runtime_chunk_uuid": record.runtime_chunk_uuid,
                    "canonical_identity_schema_version": CANONICAL_CHUNK_SCHEMA_VERSION,
                },
            )
        )
    return resolved, {
        "schema_version": "opk-rag.task0089.candidate-runtime-resolution.v1",
        "candidate_count": len(candidate_rows),
        "resolved_candidate_count": len(resolved),
        "fail_closed": bool(issue_rows),
        "issues": issue_rows,
    }


def reconstruct_search_results(
    candidates: Iterable[InjectedRetrievalCandidate],
    mapping: CanonicalChunkMapping,
) -> tuple[SearchResult, ...]:
    by_runtime = mapping.runtime_to_canonical()
    results = []
    for candidate in sorted(candidates, key=lambda row: (row.sample_id, row.candidate_rank, row.chunk_id)):
        record = by_runtime.get(candidate.chunk_id)
        if record is None:
            raise CanonicalChunkIdentityError(f"missing canonical runtime record for {candidate.chunk_id}")
        results.append(
            SearchResult(
                rank=candidate.candidate_rank,
                document_id=_uuid(record.runtime_document_uuid or record.runtime_chunk_uuid),
                chunk_id=_uuid(record.runtime_chunk_uuid),
                relative_path=record.source_path or "",
                heading_path=record.heading_path,
                content=record.chunk_text or "",
                start_line=record.start_line,
                end_line=record.end_line,
                similarity=candidate.retrieval_score or 0.0,
                vector_similarity=candidate.retrieval_score,
                vector_rank=candidate.provenance.get("original_vector_rank") if isinstance(candidate.provenance.get("original_vector_rank"), int) else None,
                rerank_score=candidate.reranker_score,
                rerank_rank=candidate.candidate_rank,
                retrieval_sources=("vector",),
                metadata={
                    "candidate_injection": True,
                    "evaluation_chunk_digest": candidate.provenance.get("evaluation_chunk_digest"),
                    "canonical_chunk_id": candidate.provenance.get("canonical_chunk_id"),
                },
            )
        )
    return tuple(results)


def _evaluation_as_runtime_records(records: Iterable[CanonicalChunkRecord]) -> tuple[CanonicalChunkRecord, ...]:
    return tuple(replace(record, runtime_chunk_uuid=record.runtime_chunk_uuid or record.evaluation_chunk_digest) for record in records)


def _strict_mismatch(left: CanonicalChunkRecord, right: CanonicalChunkRecord) -> str | None:
    if left.corpus_snapshot_digest and right.corpus_snapshot_digest and left.corpus_snapshot_digest != right.corpus_snapshot_digest:
        return "corpus_snapshot_mismatch"
    if left.chunk_text_digest and right.chunk_text_digest and left.chunk_text_digest != right.chunk_text_digest:
        return "content_digest_mismatch"
    for field in ("source_path", "document_id", "section_id", "heading_path", "start_offset", "end_offset"):
        if _factor_value(left, field) is not None and _factor_value(right, field) is not None and _factor_value(left, field) != _factor_value(right, field):
            return "provenance_mismatch"
    return None


def _same_provenance_without_content(left: CanonicalChunkRecord, right: CanonicalChunkRecord) -> bool:
    comparable = ("source_path", "heading_path", "start_offset", "end_offset")
    matched = 0
    for field in comparable:
        left_value = _factor_value(left, field)
        right_value = _factor_value(right, field)
        if left_value is None or right_value is None:
            continue
        if left_value != right_value:
            return False
        matched += 1
    return matched >= 2


def _group(records: Iterable[CanonicalChunkRecord], field: str) -> dict[str, list[CanonicalChunkRecord]]:
    grouped: dict[str, list[CanonicalChunkRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, field)
        if value:
            grouped[str(value)].append(record)
    return grouped


def _read_chunk_text(vault_path: Path | None, source_path: str | None, start: int | None, end: int | None) -> str | None:
    if vault_path is None or source_path is None or start is None or end is None:
        return None
    path = vault_path / source_path
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if start < 0 or end < start or end > len(text):
        return None
    return text[start:end]


def _factor_value(record: CanonicalChunkRecord, field: str) -> Any:
    value = getattr(record, field)
    if field == "source_path":
        return _norm_path(value)
    return value


def _norm_path(value: Any) -> str | None:
    text = _text(value)
    return text.replace("\\", "/").strip("/") if text else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _uuid(value: str | None) -> UUID:
    if value is None:
        raise CanonicalChunkIdentityError("missing runtime UUID")
    try:
        return UUID(value)
    except ValueError:
        return UUID(digest_text(value)[:32])
