from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import UUID, uuid5, NAMESPACE_URL

from opk_rag.chunking.chunker import chunk_markdown_document
from opk_rag.chunking.parser import parse_markdown_file
from opk_rag.evaluation.candidate_retrieval_baseline import (
    GOLD_IDENTITY_PATHS,
    ROOT,
    _load_public_samples,
    _reject_forbidden_path,
    read_json,
)
from opk_rag.evaluation.evidence_identity import (
    EvidenceIdentity,
    deduplicate_evidence_identities,
    digest_json,
    identity_from_runtime_evidence_item,
    match_evidence_identity,
    normalize_gold_evidence_identity,
)


SOURCE_ROOT = ROOT / "source-documents"
STRUCTURE_AUDIT_ID = "phase2-scope-structure-audit-v1"
STRUCTURE_AUDIT_SCHEMA_VERSION = "opk-rag.scope-structure-audit-summary.v1"
STRUCTURE_AUDIT_PATH = ROOT / "evaluation-data" / "results" / "phase2_scope_structure_audit_v1.json"


class ScopeStructureAuditError(ValueError):
    pass


@dataclass(frozen=True)
class IndexedScopeChunk:
    document_id: UUID
    chunk_id: UUID
    relative_path: str
    document_title: str | None
    heading_path: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    chunk_index: int
    identity: dict[str, Any]
    document_identity_digest: str | None
    scope_identity_digest: str | None
    chunk_content_digest: str | None
    heading_path_digest: str | None
    block_types: tuple[str, ...]

    def public_json(self) -> dict[str, Any]:
        return {
            "document_identity_digest": self.document_identity_digest,
            "scope_identity_digest": self.scope_identity_digest,
            "chunk_content_digest": self.chunk_content_digest,
            "heading_path_digest": self.heading_path_digest,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "block_types": list(self.block_types),
        }


@dataclass(frozen=True)
class ScopeStructureRecord:
    sample_id: str
    dataset_id: str
    evidence_identity_digest: str
    relationship: str
    structure_class: str
    block_types: tuple[str, ...]
    resolved_chunk_count: int
    heading_dependent: bool
    neighbor_dependent: bool

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["block_types"] = list(self.block_types)
        return payload


def load_indexed_scope_chunks(source_root: Path = SOURCE_ROOT) -> tuple[IndexedScopeChunk, ...]:
    _reject_forbidden_path(source_root)
    if not source_root.exists():
        raise ScopeStructureAuditError(f"source corpus does not exist: {source_root}")
    rows: list[IndexedScopeChunk] = []
    for path in sorted(source_root.rglob("*.md")):
        relative_path = path.relative_to(source_root).as_posix()
        document_id = uuid5(NAMESPACE_URL, f"opk-rag:task0056-document:{relative_path}")
        document = parse_markdown_file(path)
        for chunk in chunk_markdown_document(document):
            chunk_id = uuid5(NAMESPACE_URL, f"opk-rag:task0056-chunk:{relative_path}:{chunk.chunk_index}:{chunk.content_hash}")
            identity = identity_from_runtime_evidence_item(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "relative_path": relative_path,
                    "heading_path": chunk.heading_path,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
            ).to_json()
            rows.append(
                IndexedScopeChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    relative_path=relative_path,
                    document_title=document.title,
                    heading_path=chunk.heading_path,
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_index=chunk.chunk_index,
                    identity=identity,
                    document_identity_digest=identity.get("document_identity_digest"),
                    scope_identity_digest=identity.get("scope_identity_digest"),
                    chunk_content_digest=identity.get("chunk_content_digest"),
                    heading_path_digest=identity.get("heading_path_digest"),
                    block_types=classify_block_structure(chunk.content),
                )
            )
    return tuple(rows)


def load_searchable_scope_chunks_from_database(connection, knowledge_base_id, *, index_configuration_id=None) -> tuple[IndexedScopeChunk, ...]:
    rows: list[IndexedScopeChunk] = []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select
              d.id,
              c.id,
              d.relative_path,
              d.title,
              c.heading_path,
              c.content,
              c.start_line,
              c.end_line,
              c.chunk_index
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.knowledge_base_id = %s
              and d.index_status = 'indexed'
              and c.embedding is not null
              and (%s::uuid is null or c.index_configuration_id = %s)
            order by d.relative_path, c.chunk_index, c.id
            """,
            (knowledge_base_id, index_configuration_id, index_configuration_id),
        )
        db_rows = cursor.fetchall()
    for row in db_rows:
        identity = identity_from_runtime_evidence_item(
            {
                "document_id": row[0],
                "chunk_id": row[1],
                "relative_path": row[2],
                "heading_path": tuple(row[4] or ()),
                "content": row[5],
                "start_line": row[6],
                "end_line": row[7],
            }
        ).to_json()
        rows.append(
            IndexedScopeChunk(
                document_id=row[0],
                chunk_id=row[1],
                relative_path=row[2],
                document_title=row[3],
                heading_path=tuple(row[4] or ()),
                content=row[5],
                start_line=row[6],
                end_line=row[7],
                chunk_index=row[8],
                identity=identity,
                document_identity_digest=identity.get("document_identity_digest"),
                scope_identity_digest=identity.get("scope_identity_digest"),
                chunk_content_digest=identity.get("chunk_content_digest"),
                heading_path_digest=identity.get("heading_path_digest"),
                block_types=classify_block_structure(row[5]),
            )
        )
    return tuple(rows)


def load_gold_scope_identities(dataset_ids: Iterable[str] = ("development", "known-regression")) -> list[tuple[str, str, EvidenceIdentity]]:
    records: list[tuple[str, str, EvidenceIdentity]] = []
    for sample in _load_public_samples(list(dataset_ids)):
        for identity in deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in (sample.get("required_evidence") or [])):
            records.append((sample["sample_id"], sample["dataset_id"], identity))
    return records


def resolve_gold_scope_to_indexed_chunks(gold: EvidenceIdentity, chunks: Iterable[IndexedScopeChunk]) -> tuple[IndexedScopeChunk, ...]:
    exact: list[IndexedScopeChunk] = []
    compatible: list[IndexedScopeChunk] = []
    document_only: list[IndexedScopeChunk] = []
    for chunk in chunks:
        runtime = normalize_gold_evidence_identity(chunk.identity)
        match = match_evidence_identity(gold, runtime)
        if match.matched and match.level.value in {"exact_chunk_match", "exact_scope_match"}:
            exact.append(chunk)
        elif match.matched and match.level.value == "compatible_scope_chunk_match":
            compatible.append(chunk)
        elif gold.document_identity_digest and gold.document_identity_digest == chunk.document_identity_digest:
            document_only.append(chunk)
    return tuple(exact or compatible or document_only)


def classify_scope_chunk_relationship(gold: EvidenceIdentity, resolved: tuple[IndexedScopeChunk, ...]) -> str:
    if not resolved:
        return "chunk_not_found"
    exact = [
        chunk for chunk in resolved
        if gold.chunk_content_digest == chunk.chunk_content_digest or gold.scope_identity_digest == chunk.scope_identity_digest
    ]
    if len(exact) == 1:
        return "exact_single_chunk"
    if len(resolved) == 1:
        chunk = resolved[0]
        if gold.heading_path_digest and gold.heading_path_digest == chunk.heading_path_digest and not gold.chunk_content_digest:
            return "single_chunk_heading_dependent"
        return "single_chunk_document_context_dependent"
    indexes = sorted(chunk.chunk_index for chunk in resolved)
    if indexes == list(range(indexes[0], indexes[-1] + 1)):
        return "adjacent_two_chunks" if len(indexes) == 2 else "adjacent_multiple_chunks"
    return "identity_mismatch"


def classify_block_structure(content: str) -> tuple[str, ...]:
    types: set[str] = set()
    lines = content.splitlines()
    if any(re.match(r"^\s*(?:[-*+]|\d+[.])\s+", line) for line in lines):
        types.add("list_block")
    if any("|" in line and re.search(r"\|?\s*:?-{3,}:?\s*\|", line) for line in lines):
        types.add("table_block")
    if any(re.match(r"^\s*(```|~~~)", line) for line in lines):
        types.add("code_block")
    if any(re.match(r"^\s*>", line) for line in lines):
        types.add("quote_block")
    if not types:
        types.add("paragraph_block")
    return tuple(sorted(types))


def detect_cross_chunk_scope(relationship: str) -> bool:
    return relationship in {"adjacent_two_chunks", "adjacent_multiple_chunks"}


def detect_heading_dependency(relationship: str, chunk: IndexedScopeChunk | None) -> bool:
    return relationship in {"single_chunk_heading_dependent", "heading_only"} or bool(chunk and chunk.heading_path)


def detect_neighbor_dependency(relationship: str) -> bool:
    return relationship in {"adjacent_two_chunks", "adjacent_multiple_chunks"}


def build_scope_structure_summary(
    chunks: tuple[IndexedScopeChunk, ...] | None = None,
    *,
    dataset_ids: Iterable[str] = ("development", "known-regression"),
) -> dict[str, Any]:
    chunks = chunks or load_indexed_scope_chunks()
    records: list[ScopeStructureRecord] = []
    for sample_id, dataset_id, gold in load_gold_scope_identities(dataset_ids):
        resolved = resolve_gold_scope_to_indexed_chunks(gold, chunks)
        relationship = classify_scope_chunk_relationship(gold, resolved)
        chunk = resolved[0] if resolved else None
        block_types = tuple(sorted({block for item in resolved for block in item.block_types})) or ("unverifiable",)
        if relationship == "chunk_not_found":
            structure_class = "unverifiable"
        elif len(block_types) > 1:
            structure_class = "mixed_block"
        else:
            structure_class = block_types[0].replace("_block", "_block")
        records.append(
            ScopeStructureRecord(
                sample_id=sample_id,
                dataset_id=dataset_id,
                evidence_identity_digest=digest_json(gold.to_json()),
                relationship=relationship,
                structure_class=structure_class,
                block_types=block_types,
                resolved_chunk_count=len(resolved),
                heading_dependent=detect_heading_dependency(relationship, chunk),
                neighbor_dependent=detect_neighbor_dependency(relationship),
            )
        )
    return aggregate_scope_structure_records(records, chunks)


def aggregate_scope_structure_records(records: list[ScopeStructureRecord], chunks: tuple[IndexedScopeChunk, ...]) -> dict[str, Any]:
    relationship_counts = Counter(record.relationship for record in records)
    block_counts = Counter(block for record in records for block in record.block_types)
    resolved = sum(record.resolved_chunk_count > 0 for record in records)
    multi = sum(detect_cross_chunk_scope(record.relationship) for record in records)
    total = len(records)
    evidence_counts = Counter(record.evidence_identity_digest for record in records)
    unique_evidence_units = len(evidence_counts)
    by_identity: dict[str, list[ScopeStructureRecord]] = {}
    for record in records:
        by_identity.setdefault(record.evidence_identity_digest, []).append(record)
    unique_multi = sum(any(detect_cross_chunk_scope(record.relationship) for record in grouped) for grouped in by_identity.values())
    unique_single = sum(
        bool(grouped) and not any(detect_cross_chunk_scope(record.relationship) for record in grouped)
        for grouped in by_identity.values()
    )
    adjacent_multi = sum(record.relationship in {"adjacent_two_chunks", "adjacent_multiple_chunks"} for record in records)
    non_adjacent_multi = sum(record.resolved_chunk_count > 1 and not detect_cross_chunk_scope(record.relationship) for record in records)
    heading_multi_overlap = sum(record.heading_dependent and detect_cross_chunk_scope(record.relationship) for record in records)
    return {
        "schema_version": STRUCTURE_AUDIT_SCHEMA_VERSION,
        "audit_id": STRUCTURE_AUDIT_ID,
        "status": "completed",
        "dataset_ids": ["development", "known-regression"],
        "required_evidence_unit_count": total,
        "indexed_chunk_count": len(chunks),
        "gold_scope_resolved_count": resolved,
        "gold_scope_unresolved_count": total - resolved,
        "single_chunk_count": sum(record.relationship.startswith("single_chunk") or record.relationship == "exact_single_chunk" for record in records),
        "multi_chunk_count": multi,
        "multi_chunk_required_evidence_rate": (multi / total) if total else 0.0,
        "multi_chunk_basis": "required evidence unit resolves to two or more contiguous adjacent indexed chunks by chunk_index",
        "multi_chunk_contiguous_adjacent_count": adjacent_multi,
        "multi_chunk_non_contiguous_count": non_adjacent_multi,
        "unique_scope_identity_count": unique_evidence_units,
        "unique_multi_chunk_scope_count": unique_multi,
        "unique_single_chunk_scope_count": unique_single,
        "unique_multi_chunk_scope_rate": (unique_multi / unique_evidence_units) if unique_evidence_units else 0.0,
        "unique_scope_boundary_issue": "confirmed" if unique_evidence_units and unique_multi / unique_evidence_units >= 0.20 else "not_confirmed" if resolved else "insufficient_evidence",
        "chunk_boundary_issue_basis": "required_evidence_unit",
        "duplicate_required_evidence_unit_source": "multiple public claims may cite the same canonical scope identity",
        "duplicate_scope_counting_warning": "duplicate required evidence units are not separate unique boundary failures",
        "duplicate_required_evidence_unit_count": sum(count - 1 for count in evidence_counts.values() if count > 1),
        "duplicate_scope_counted_multiple_times": bool(sum(count - 1 for count in evidence_counts.values() if count > 1)),
        "denominator_basis": "required_evidence_unit",
        "heading_dependency_in_multi_chunk_count": heading_multi_overlap,
        "heading_dependency_counted_as_multi_chunk": False,
        "required_evidence_unit_denominator": total,
        "scope_identity_denominator": unique_evidence_units,
        "required_evidence_unit_scope_identity_denominator_consistent": total == unique_evidence_units,
        "heading_dependent_count": sum(record.heading_dependent for record in records),
        "neighbor_dependent_count": sum(record.neighbor_dependent for record in records),
        "structured_block_count": sum(any(block != "paragraph_block" for block in record.block_types) for record in records),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "structured_block_distribution": dict(sorted(block_counts.items())),
        "chunk_boundary_issue": "confirmed" if total and multi / total >= 0.20 else "not_confirmed" if resolved else "insufficient_evidence",
        "contains_sealed_holdout_data": False,
        "publishes_private_text": False,
        "records": [record.to_json() for record in records],
    }


def write_scope_structure_audit(summary: dict[str, Any], path: Path = STRUCTURE_AUDIT_PATH) -> None:
    _reject_forbidden_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gold_identity_map_digests() -> dict[str, str]:
    return {dataset_id: _sha256(path) for dataset_id, path in GOLD_IDENTITY_PATHS.items() if path.exists()}


def _sha256(path: Path) -> str:
    import hashlib

    _reject_forbidden_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
