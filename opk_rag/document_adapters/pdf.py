from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from opk_rag.document_adapters.base import ParsedDocumentParts, StructuredDocumentSource, build_canonical_document, stable_id
from opk_rag.document_adapters.errors import AdapterParseError, UnsupportedRepresentationError
from opk_rag.document_ir import BoundingBox, CanonicalBlock, CanonicalSection, SourceLocation


PYMUPDF4LLM_ADAPTER_CONFIG: dict[str, Any] = {
    "config_version": 1,
    "parser": "pymupdf4llm",
    "parser_options": {
        "page_chunks": True,
        "write_images": False,
        "embed_images": False,
        "table_strategy": "parser_default",
    },
    "ocr_enabled": False,
    "vision_enabled": False,
}


class PdfDocumentAdapter:
    adapter_name = "opk-rag-pdf-pymupdf4llm-adapter"
    adapter_version = "1"
    representation_types = ("pdf",)

    def supports(self, representation_type: str) -> bool:
        return str(representation_type).strip().lower() in self.representation_types

    def adapt(self, source: StructuredDocumentSource):
        if not self.supports(source.representation_type):
            raise UnsupportedRepresentationError(f"unsupported representation: {source.representation_type}")
        payload = self.parse(source)
        return self.to_canonical_document(source, payload)

    def parse(self, source: StructuredDocumentSource) -> dict[str, Any]:
        try:
            import pymupdf4llm  # type: ignore
        except ModuleNotFoundError as exc:
            raise AdapterParseError("pymupdf4llm is not installed") from exc

        try:
            markdown = pymupdf4llm.to_markdown(
                source.source_bytes,
                page_chunks=True,
                write_images=False,
                embed_images=False,
            )
        except Exception as exc:
            raise AdapterParseError(f"pymupdf4llm parse failed: {type(exc).__name__}: {exc}") from exc
        return normalize_pymupdf4llm_payload(markdown)

    def to_canonical_document(self, source: StructuredDocumentSource, payload: dict[str, Any]):
        parts = parsed_pdf_payload_to_parts(
            payload,
            source_ref=source.source_ref,
            representation_id=source.representation_id,
        )
        return build_canonical_document(
            source=source,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            parts=parts,
            adapter_config=PYMUPDF4LLM_ADAPTER_CONFIG,
        )


def normalize_pymupdf4llm_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return {
            "schema_version": "opk-rag.pymupdf4llm-normalized-output.v1",
            "text": payload,
            "pages": [{"page_number": 1, "text_length": len(payload), "block_count": 1}],
            "blocks": [{"page_number": 1, "block_index": 0, "block_type": "text", "text": payload}],
            "tables": [],
            "figures": [],
        }
    if isinstance(payload, list):
        pages: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        full_text: list[str] = []
        for index, page in enumerate(payload, start=1):
            if not isinstance(page, dict):
                continue
            text = str(page.get("text") or "")
            page_number = int(page.get("metadata", {}).get("page") or page.get("page") or index)
            pages.append({"page_number": page_number, "text_length": len(text), "block_count": 1})
            blocks.append({"page_number": page_number, "block_index": 0, "block_type": "text", "text": text})
            full_text.append(text)
        return {
            "schema_version": "opk-rag.pymupdf4llm-normalized-output.v1",
            "text": "\n".join(full_text),
            "pages": pages,
            "blocks": blocks,
            "tables": [],
            "figures": [],
        }
    if isinstance(payload, dict):
        return payload
    raise AdapterParseError(f"unsupported pymupdf4llm output type: {type(payload).__name__}")


def parsed_pdf_payload_to_parts(
    payload: dict[str, Any],
    *,
    source_ref: str,
    representation_id: str,
) -> ParsedDocumentParts:
    pages = _normalize_pages(payload)
    raw_blocks = [block for block in payload.get("blocks", []) if isinstance(block, dict)]
    if not raw_blocks and str(payload.get("text") or "").strip():
        raw_blocks = [{"page_number": 1, "block_index": 0, "block_type": "text", "text": payload["text"]}]

    sections = [CanonicalSection("root", None, 0, None, 0, (), ())]
    blocks: list[CanonicalBlock] = []
    for reading_order, raw in enumerate(raw_blocks):
        text = str(raw.get("text") or "").strip()
        block_type = _canonical_block_type(raw.get("block_type"), text)
        page_number = _page_number(raw, default=1)
        block_ordinal = int(raw.get("block_index") if raw.get("block_index") is not None else reading_order)
        bbox = _bbox(raw.get("bbox"), page_number=page_number)
        metadata = {
            "pdf_block_ordinal": block_ordinal,
            "parser_block_type": str(raw.get("block_type") or ""),
        }
        block_id = stable_id("pdfblk", representation_id, page_number, block_ordinal, reading_order, block_type, text)
        blocks.append(
            CanonicalBlock(
                block_id=block_id,
                block_type=block_type,
                text=text,
                section_id="root",
                parent_block_id=None,
                ordinal=block_ordinal,
                reading_order=reading_order,
                source_location=SourceLocation(
                    source_ref=source_ref,
                    page_number=page_number,
                    page_start=page_number,
                    page_end=page_number,
                    element_ref=f"page:{page_number}:block:{block_ordinal}",
                    bbox=bbox,
                ),
                metadata=metadata,
            )
        )

    sections[0] = replace(sections[0], block_ids=tuple(block.block_id for block in blocks))
    title = _infer_title(blocks, fallback=Path(source_ref).stem)
    return ParsedDocumentParts(
        title=title,
        sections=tuple(sections),
        blocks=tuple(blocks),
        tables=(),
        figures=(),
        metadata={
            "page_count": len(pages) or int(payload.get("page_count") or 0),
            "pages": pages,
            "parser_output_schema_version": payload.get("schema_version"),
            "canonical_pdf_adapter_v1": True,
        },
    )


def _normalize_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page in payload.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        page_number = _page_number(page, default=len(pages) + 1)
        pages.append(
            {
                "page_number": page_number,
                "width": page.get("width"),
                "height": page.get("height"),
                "block_count": int(page.get("block_count") or 0),
                "text_length": int(page.get("text_length") or 0),
            }
        )
    if not pages and payload.get("page_count"):
        pages = [{"page_number": page_number, "width": None, "height": None, "block_count": 0, "text_length": 0} for page_number in range(1, int(payload["page_count"]) + 1)]
    return pages


def _canonical_block_type(value: Any, text: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"heading", "paragraph", "list", "table", "code", "other"}:
        return raw
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.startswith("#") or first.startswith("**") and len(first) < 120:
        return "heading"
    if first.startswith(("- ", "* ")) or first[:2].rstrip(".").isdigit():
        return "list"
    if "|" in text and "\n" in text:
        return "table"
    if raw in {"text", "span", "block"}:
        return "paragraph"
    return "other"


def _page_number(value: dict[str, Any], *, default: int) -> int:
    raw = value.get("page_number", value.get("page", default))
    try:
        page = int(raw)
    except (TypeError, ValueError):
        page = default
    return page if page > 0 else default


def _bbox(value: Any, *, page_number: int) -> BoundingBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, coordinate_space="pdf-page", origin="top-left", page_number=page_number, units="pt")


def _infer_title(blocks: list[CanonicalBlock], *, fallback: str) -> str | None:
    for block in blocks[:20]:
        text = (block.text or "").strip()
        if text and len(text) <= 160:
            return text.splitlines()[0].strip("#* ")
    return fallback or None
