from __future__ import annotations

import re

from opk_rag.chunking.parser import MarkdownParseError, parse_markdown
from opk_rag.document_adapters.base import ParsedDocumentParts, StructuredDocumentSource, build_canonical_document, stable_id
from opk_rag.document_adapters.errors import AdapterParseError
from opk_rag.document_ir import CanonicalBlock, CanonicalSection, SourceLocation


class MarkdownDocumentAdapter:
    adapter_name = "opk-rag-markdown-structured-adapter"
    adapter_version = "1"
    representation_types = ("markdown", "md")

    def adapt(self, source: StructuredDocumentSource):
        try:
            parsed = parse_markdown(source.text(), source_path=source.source_ref)
        except (UnicodeDecodeError, MarkdownParseError) as exc:
            raise AdapterParseError(str(exc)) from exc

        sections: list[CanonicalSection] = [
            CanonicalSection("root", None, 0, None, 0, (), ()),
        ]
        section_ids_by_path: dict[tuple[str, ...], str] = {(): "root"}
        section_block_ids: dict[str, list[str]] = {"root": []}
        blocks: list[CanonicalBlock] = []
        reading_order = 0

        for parsed_section in parsed.sections:
            path = parsed_section.heading_path
            parent_path: tuple[str, ...] = ()
            for level, title in enumerate(path, start=1):
                current_path = path[:level]
                if current_path not in section_ids_by_path:
                    section_id = stable_id("sec", source.representation_id, current_path, level)
                    parent_id = section_ids_by_path[parent_path]
                    section_ids_by_path[current_path] = section_id
                    section_block_ids[section_id] = []
                    sections.append(
                        CanonicalSection(
                            section_id=section_id,
                            parent_section_id=parent_id,
                            level=level,
                            title=title,
                            ordinal=len(sections),
                            path=current_path,
                            block_ids=(),
                            source_location=SourceLocation(source_ref=source.source_ref, line_start=parsed_section.start_line),
                        )
                    )
                parent_path = current_path
            section_id = section_ids_by_path.get(path, "root")
            for block_type, text, start_line, end_line in _markdown_blocks(parsed_section.content, parsed_section.start_line):
                block_id = stable_id("blk", source.representation_id, reading_order, block_type, text)
                blocks.append(
                    CanonicalBlock(
                        block_id=block_id,
                        block_type=block_type,
                        text=text,
                        section_id=section_id,
                        parent_block_id=None,
                        ordinal=len(section_block_ids[section_id]),
                        reading_order=reading_order,
                        source_location=SourceLocation(source_ref=source.source_ref, line_start=start_line, line_end=end_line),
                    )
                )
                section_block_ids[section_id].append(block_id)
                reading_order += 1

        sections = [section if section.section_id not in section_block_ids else _with_block_ids(section, section_block_ids[section.section_id]) for section in sections]
        return build_canonical_document(
            source=source,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            parts=ParsedDocumentParts(
                title=parsed.title,
                sections=tuple(sections),
                blocks=tuple(blocks),
                metadata={"frontmatter": parsed.frontmatter},
            ),
        )


def _with_block_ids(section: CanonicalSection, block_ids: list[str]) -> CanonicalSection:
    from dataclasses import replace

    return replace(section, block_ids=tuple(block_ids))


def _markdown_blocks(content: str, base_line: int) -> tuple[tuple[str, str, int, int], ...]:
    lines = content.splitlines()
    blocks: list[tuple[str, str, int, int]] = []
    pending: list[str] = []
    pending_start = base_line
    in_fence = False
    fence_start = base_line
    fence_lines: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal pending, pending_start
        text = "\n".join(pending).strip()
        if text:
            blocks.append((_classify_markdown_block(text), text, pending_start, end_line))
        pending = []

    for offset, line in enumerate(lines):
        line_no = base_line + offset
        if re.match(r"^[ \t]*(```+|~~~+)", line):
            if in_fence:
                fence_lines.append(line)
                blocks.append(("code", "\n".join(fence_lines).strip(), fence_start, line_no))
                fence_lines = []
                in_fence = False
            else:
                flush(line_no - 1)
                in_fence = True
                fence_start = line_no
                fence_lines = [line]
            continue
        if in_fence:
            fence_lines.append(line)
            continue
        if not line.strip():
            flush(line_no - 1)
            pending_start = line_no + 1
            continue
        if not pending:
            pending_start = line_no
        pending.append(line)
    if in_fence:
        blocks.append(("code", "\n".join(fence_lines).strip(), fence_start, base_line + len(lines) - 1))
    flush(base_line + len(lines) - 1)
    return tuple(blocks)


def _classify_markdown_block(text: str) -> str:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.startswith("#"):
        return "heading"
    if re.match(r"^([-*+]|\d+[.])\s+", first):
        return "list"
    if "|" in text and "\n" in text:
        return "table"
    return "paragraph"
