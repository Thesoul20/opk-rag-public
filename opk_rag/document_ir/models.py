from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CANONICAL_DOCUMENT_IR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: str
    origin: str
    page_number: int | None = None
    units: str | None = None


@dataclass(frozen=True)
class SourceLocation:
    source_ref: str | None = None
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    bbox: BoundingBox | None = None
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    element_ref: str | None = None


@dataclass(frozen=True)
class RepresentationMetadata:
    format: str
    mime_type: str | None = None
    source_name: str | None = None
    source_relative_path: str | None = None
    source_digest: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    parser_config_digest: str | None = None
    authority: dict[str, Any] = field(default_factory=dict)
    upstream_revision: str | None = None
    language: str | None = None
    created_at_if_source_provides: str | None = None
    semantic_metadata: dict[str, Any] = field(default_factory=dict)
    operational_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalSection:
    section_id: str
    parent_section_id: str | None
    level: int
    title: str | None
    ordinal: int
    path: tuple[str, ...]
    block_ids: tuple[str, ...] = ()
    source_location: SourceLocation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalBlock:
    block_id: str
    block_type: str
    text: str | None
    section_id: str | None
    parent_block_id: str | None
    ordinal: int
    reading_order: int
    source_location: SourceLocation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalTableCell:
    row_index: int
    column_index: int
    row_span: int = 1
    column_span: int = 1
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalTable:
    table_id: str
    block_id: str
    rows: int
    columns: int
    cells: tuple[CanonicalTableCell, ...] = ()
    caption: str | None = None
    source_location: SourceLocation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFigure:
    figure_id: str
    block_id: str
    caption: str | None = None
    alt_text: str | None = None
    source_location: SourceLocation | None = None
    asset_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    schema_version: int
    document_id: str
    knowledge_id: str
    representation_id: str
    representation_type: str
    title: str | None
    metadata: dict[str, Any]
    representation_metadata: RepresentationMetadata
    sections: tuple[CanonicalSection, ...]
    blocks: tuple[CanonicalBlock, ...]
    tables: tuple[CanonicalTable, ...] = ()
    figures: tuple[CanonicalFigure, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    content_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from opk_rag.document_ir.serialization import document_to_dict

        return document_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalDocument":
        from opk_rag.document_ir.serialization import document_from_dict

        return document_from_dict(payload)
