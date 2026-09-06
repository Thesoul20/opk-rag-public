from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from typing import Any, TypeVar


T = TypeVar("T")


KNOWN_REPRESENTATION_ALIASES = {
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "htm": "html",
    "tex": "tex",
    "latex": "tex",
    "pdf": "pdf",
    "xlsx": "xlsx",
    "xls": "xlsx",
}


def normalize_representation_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("representation type must be non-empty")
    return KNOWN_REPRESENTATION_ALIASES.get(normalized, normalized)


def canonical_json(value: Any) -> str:
    payload = _json_safe(value)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def document_to_dict(document: Any, *, semantic: bool = False) -> dict[str, Any]:
    from opk_rag.document_ir.models import CanonicalDocument
    from opk_rag.document_ir.validation import validate_document

    if not isinstance(document, CanonicalDocument):
        raise TypeError("document_to_dict expects CanonicalDocument")
    validate_document(document)
    payload = _model_to_dict(document, semantic=semantic)
    payload["sections"] = sorted(payload["sections"], key=lambda item: (item["ordinal"], item["section_id"]))
    payload["blocks"] = sorted(payload["blocks"], key=lambda item: (item["reading_order"], item["ordinal"], item["block_id"]))
    payload["tables"] = sorted(payload["tables"], key=lambda item: item["table_id"])
    payload["figures"] = sorted(payload["figures"], key=lambda item: item["figure_id"])
    if semantic:
        payload.pop("content_digest", None)
        payload["representation_metadata"].pop("operational_metadata", None)
    return payload


def document_from_dict(payload: dict[str, Any]) -> Any:
    from opk_rag.document_ir.models import (
        BoundingBox,
        CanonicalBlock,
        CanonicalDocument,
        CanonicalFigure,
        CanonicalSection,
        CanonicalTable,
        CanonicalTableCell,
        RepresentationMetadata,
        SourceLocation,
    )

    def loc(value: dict[str, Any] | None) -> SourceLocation | None:
        if value is None:
            return None
        bbox_payload = value.get("bbox")
        bbox = BoundingBox(**bbox_payload) if isinstance(bbox_payload, dict) else None
        return SourceLocation(**{**value, "bbox": bbox})

    def table(value: dict[str, Any]) -> CanonicalTable:
        cells = tuple(CanonicalTableCell(**cell) for cell in value.get("cells", ()))
        return CanonicalTable(**{**value, "cells": cells, "source_location": loc(value.get("source_location"))})

    document = CanonicalDocument(
        schema_version=payload.get("schema_version"),
        document_id=payload.get("document_id"),
        knowledge_id=payload.get("knowledge_id"),
        representation_id=payload.get("representation_id"),
        representation_type=payload.get("representation_type"),
        title=payload.get("title"),
        metadata=dict(payload.get("metadata") or {}),
        representation_metadata=RepresentationMetadata(**payload.get("representation_metadata", {})),
        sections=tuple(CanonicalSection(**{**item, "block_ids": tuple(item.get("block_ids") or ()), "path": tuple(item.get("path") or ()), "source_location": loc(item.get("source_location"))}) for item in payload.get("sections", ())),
        blocks=tuple(CanonicalBlock(**{**item, "source_location": loc(item.get("source_location"))}) for item in payload.get("blocks", ())),
        tables=tuple(table(item) for item in payload.get("tables", ())),
        figures=tuple(CanonicalFigure(**{**item, "source_location": loc(item.get("source_location"))}) for item in payload.get("figures", ())),
        source=dict(payload.get("source") or {}),
        provenance=dict(payload.get("provenance") or {}),
        content_digest=payload.get("content_digest"),
    )
    from opk_rag.document_ir.validation import validate_document

    validate_document(document)
    return document


def _model_to_dict(value: Any, *, semantic: bool) -> Any:
    if is_dataclass(value):
        return {field.name: _model_to_dict(getattr(value, field.name), semantic=semantic) for field in fields(value)}
    if isinstance(value, tuple):
        return [_model_to_dict(item, semantic=semantic) for item in value]
    if isinstance(value, list):
        return [_model_to_dict(item, semantic=semantic) for item in value]
    if isinstance(value, dict):
        return {str(key): _model_to_dict(item, semantic=semantic) for key, item in value.items()}
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _model_to_dict(value, semantic=False)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN or Infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"value is not canonical JSON serializable: {type(value).__name__}")
