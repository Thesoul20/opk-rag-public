from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opk_rag.chunking.models import MarkdownSection, ParsedMarkdownDocument
from opk_rag.document_ir import CanonicalDocument


class CanonicalDocumentChunkingInputError(ValueError):
    """Raised when a CanonicalDocument cannot be safely adapted for the existing chunker."""


@dataclass(frozen=True)
class CanonicalChunkingInput:
    """Minimal DTO accepted by the existing markdown chunker.

    The production chunker currently requires a ParsedMarkdownDocument with sections and
    textual section content.  This DTO keeps canonical identity beside that object so
    Document -> Chunking Input -> Chunk persistence can be audited without changing the
    frozen chunking policy or reimplementing the chunker.
    """

    canonical_document_id: str
    source_path: Path | None
    parsed_document: ParsedMarkdownDocument
    metadata: dict[str, Any]
    canonical_content_digest: str
    chunking_input_content_digest: str

    @property
    def content(self) -> str:
        return self.parsed_document.body


def canonical_document_to_chunking_input(document: CanonicalDocument) -> CanonicalChunkingInput:
    """Adapt a CanonicalDocument to the existing ParsedMarkdownDocument chunker input.

    Contract repair policy:
    - deterministic: sort canonical blocks by reading_order and group by canonical section;
    - fail-closed: reject missing identity, missing content_digest, or empty textual content;
    - content ownership: rendered chunking text is exactly the non-empty CanonicalBlock.text
      payload joined in reading order; no benchmark/gold metadata is consulted;
    - policy invariant: this only builds ParsedMarkdownDocument and does not alter
      ChunkingConfig or chunk_markdown_document.
    """

    if not isinstance(document, CanonicalDocument):
        raise CanonicalDocumentChunkingInputError("expected CanonicalDocument")
    if not document.document_id:
        raise CanonicalDocumentChunkingInputError("canonical document missing document_id")
    if not document.content_digest:
        raise CanonicalDocumentChunkingInputError("canonical document missing content_digest")

    sections_by_id = {section.section_id: section for section in document.sections}
    section_rows: dict[str, list[tuple[int, str, int | None, int | None]]] = {}
    root_key = "root"
    for block in sorted(document.blocks, key=lambda item: item.reading_order):
        text = block.text if isinstance(block.text, str) else ""
        if not text.strip():
            continue
        section_id = block.section_id if block.section_id in sections_by_id else root_key
        loc = block.source_location
        section_rows.setdefault(section_id or root_key, []).append(
            (block.reading_order, text, loc.line_start if loc else None, loc.line_end if loc else None)
        )

    if not section_rows:
        raise CanonicalDocumentChunkingInputError("canonical document has no non-empty text blocks")

    markdown_sections: list[MarkdownSection] = []
    body_parts: list[str] = []
    for section_id, rows in sorted(section_rows.items(), key=lambda item: min(row[0] for row in item[1])):
        section = sections_by_id.get(section_id)
        content = "\n\n".join(row[1] for row in sorted(rows, key=lambda row: row[0]))
        body_parts.append(content)
        starts = [row[2] for row in rows if isinstance(row[2], int)]
        ends = [row[3] for row in rows if isinstance(row[3], int)]
        start_line = min(starts) if starts else 1
        end_line = max(ends) if ends else start_line + max(0, len(content.splitlines()) - 1)
        markdown_sections.append(
            MarkdownSection(
                heading_path=tuple(section.path) if section else (),
                content=content,
                start_line=start_line,
                end_line=end_line,
            )
        )

    source_ref = document.representation_metadata.source_relative_path or document.source.get("source_ref")
    source_path = Path(source_ref) if isinstance(source_ref, str) and source_ref else None
    frontmatter = document.metadata.get("frontmatter") if isinstance(document.metadata.get("frontmatter"), dict) else {}
    body = "\n\n".join(body_parts)
    parsed = ParsedMarkdownDocument(
        source_path=source_path,
        frontmatter=dict(frontmatter),
        body=body,
        title=document.title,
        sections=tuple(markdown_sections),
    )
    digest = _text_digest(body)
    return CanonicalChunkingInput(
        canonical_document_id=document.document_id,
        source_path=source_path,
        parsed_document=parsed,
        metadata={
            "canonical_document_id": document.document_id,
            "canonical_content_digest": document.content_digest,
            "knowledge_id": document.knowledge_id,
            "representation_id": document.representation_id,
            "representation_type": document.representation_type,
            "source_relative_path": str(source_path) if source_path else None,
            "adapter": "canonical_document_to_chunking_input.v1",
            "normalization_policy": "none; CanonicalBlock.text joined by section and reading_order",
        },
        canonical_content_digest=document.content_digest,
        chunking_input_content_digest=digest,
    )


def _text_digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
