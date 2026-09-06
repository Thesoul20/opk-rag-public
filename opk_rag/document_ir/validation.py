from __future__ import annotations

import math
from typing import Any

from opk_rag.document_ir.models import (
    CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
    BoundingBox,
    CanonicalDocument,
    SourceLocation,
)
from opk_rag.document_ir.serialization import normalize_representation_type


class DocumentIRValidationError(ValueError):
    pass


def validate_document(document: CanonicalDocument) -> None:
    issues: list[str] = []
    if document.schema_version != CANONICAL_DOCUMENT_IR_SCHEMA_VERSION:
        issues.append(f"unsupported schema_version: {document.schema_version!r}")
    for label, value in (
        ("document_id", document.document_id),
        ("knowledge_id", document.knowledge_id),
        ("representation_id", document.representation_id),
        ("representation_type", document.representation_type),
    ):
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{label} must be a non-empty string")
    try:
        normalize_representation_type(document.representation_type)
        normalize_representation_type(document.representation_metadata.format)
    except ValueError as exc:
        issues.append(str(exc))

    section_ids = [section.section_id for section in document.sections]
    block_ids = [block.block_id for block in document.blocks]
    table_ids = [table.table_id for table in document.tables]
    figure_ids = [figure.figure_id for figure in document.figures]
    _duplicates(section_ids, "section", issues)
    _duplicates(block_ids, "block", issues)
    _duplicates(table_ids, "table", issues)
    _duplicates(figure_ids, "figure", issues)

    section_id_set = set(section_ids)
    block_id_set = set(block_ids)
    for section in document.sections:
        if section.parent_section_id is not None and section.parent_section_id not in section_id_set:
            issues.append(f"section parent missing: {section.section_id}")
        if section.level < 0:
            issues.append(f"section level invalid: {section.section_id}")
        if section.ordinal < 0:
            issues.append(f"section ordinal invalid: {section.section_id}")
        for block_id in section.block_ids:
            if block_id not in block_id_set:
                issues.append(f"section references missing block: {section.section_id}:{block_id}")
        _validate_location(section.source_location, issues)
        _validate_json_payload(section.metadata, f"section.metadata:{section.section_id}", issues)
    _detect_cycles({section.section_id: section.parent_section_id for section in document.sections}, "section", issues)

    for block in document.blocks:
        if block.section_id is not None and block.section_id not in section_id_set:
            issues.append(f"block section missing: {block.block_id}")
        if block.parent_block_id is not None and block.parent_block_id not in block_id_set:
            issues.append(f"block parent missing: {block.block_id}")
        if not isinstance(block.block_type, str) or not block.block_type.strip():
            issues.append(f"block_type invalid: {block.block_id}")
        if block.ordinal < 0:
            issues.append(f"block ordinal invalid: {block.block_id}")
        if block.reading_order < 0:
            issues.append(f"block reading_order invalid: {block.block_id}")
        _validate_location(block.source_location, issues)
        _validate_json_payload(block.metadata, f"block.metadata:{block.block_id}", issues)
    _detect_cycles({block.block_id: block.parent_block_id for block in document.blocks}, "block", issues)

    for table in document.tables:
        if table.block_id not in block_id_set:
            issues.append(f"table references missing block: {table.table_id}")
        if table.rows < 0 or table.columns < 0:
            issues.append(f"table dimensions invalid: {table.table_id}")
        occupied: dict[tuple[int, int], int] = {}
        for index, cell in enumerate(table.cells):
            if cell.row_index < 0 or cell.column_index < 0:
                issues.append(f"table cell index invalid: {table.table_id}:{index}")
            if cell.row_span <= 0 or cell.column_span <= 0:
                issues.append(f"table cell span invalid: {table.table_id}:{index}")
            if cell.row_index + cell.row_span > table.rows or cell.column_index + cell.column_span > table.columns:
                issues.append(f"table cell outside dimensions: {table.table_id}:{index}")
            for row in range(cell.row_index, cell.row_index + cell.row_span):
                for column in range(cell.column_index, cell.column_index + cell.column_span):
                    previous = occupied.setdefault((row, column), index)
                    if previous != index:
                        issues.append(f"table cell coverage overlap: {table.table_id}:{index}")
        _validate_location(table.source_location, issues)
        _validate_json_payload(table.metadata, f"table.metadata:{table.table_id}", issues)

    for figure in document.figures:
        if figure.block_id not in block_id_set:
            issues.append(f"figure references missing block: {figure.figure_id}")
        if isinstance(figure.asset_reference, str) and figure.asset_reference.startswith("data:"):
            issues.append(f"figure asset_reference must not embed binary data: {figure.figure_id}")
        _validate_location(figure.source_location, issues)
        _validate_json_payload(figure.metadata, f"figure.metadata:{figure.figure_id}", issues)

    _validate_json_payload(document.metadata, "document.metadata", issues)
    _validate_json_payload(document.source, "document.source", issues)
    _validate_json_payload(document.provenance, "document.provenance", issues)
    _validate_json_payload(document.representation_metadata.authority, "representation_metadata.authority", issues)
    _validate_json_payload(document.representation_metadata.semantic_metadata, "representation_metadata.semantic_metadata", issues)
    _validate_json_payload(document.representation_metadata.operational_metadata, "representation_metadata.operational_metadata", issues)

    if issues:
        raise DocumentIRValidationError("; ".join(issues))


def _duplicates(values: list[str], label: str, issues: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{label} id must be non-empty")
            continue
        if value in seen:
            issues.append(f"duplicate {label} id: {value}")
        seen.add(value)


def _detect_cycles(parents: dict[str, str | None], label: str, issues: list[str]) -> None:
    for node in parents:
        visiting: set[str] = set()
        current: str | None = node
        while current is not None:
            if current in visiting:
                issues.append(f"{label} parent cycle detected: {node}")
                break
            visiting.add(current)
            current = parents.get(current)


def _validate_location(location: SourceLocation | None, issues: list[str]) -> None:
    if location is None:
        return
    for start_name, end_name in (("page_start", "page_end"), ("char_start", "char_end"), ("line_start", "line_end")):
        start = getattr(location, start_name)
        end = getattr(location, end_name)
        if start is not None and start < 0:
            issues.append(f"{start_name} cannot be negative")
        if end is not None and end < 0:
            issues.append(f"{end_name} cannot be negative")
        if start is not None and end is not None and start > end:
            issues.append(f"{start_name} cannot be greater than {end_name}")
    for name in ("page_number",):
        value = getattr(location, name)
        if value is not None and value <= 0:
            issues.append(f"{name} must be positive")
    _validate_bbox(location.bbox, issues)


def _validate_bbox(bbox: BoundingBox | None, issues: list[str]) -> None:
    if bbox is None:
        return
    for name in ("coordinate_space", "origin"):
        value = getattr(bbox, name)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"bbox {name} must be non-empty")
    for name in ("x0", "y0", "x1", "y1"):
        value = getattr(bbox, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(f"bbox {name} must be finite")
    if bbox.x0 > bbox.x1 or bbox.y0 > bbox.y1:
        issues.append("bbox coordinates must be ordered")
    if bbox.page_number is not None and bbox.page_number <= 0:
        issues.append("bbox page_number must be positive")


def _validate_json_payload(value: Any, label: str, issues: list[str]) -> None:
    from opk_rag.document_ir.serialization import canonical_json

    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        issues.append(f"{label} is not canonical JSON serializable: {exc}")
