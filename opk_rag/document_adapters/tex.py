from __future__ import annotations

import re
from dataclasses import replace

from opk_rag.document_adapters.base import ParsedDocumentParts, StructuredDocumentSource, build_canonical_document, stable_id
from opk_rag.document_adapters.errors import AdapterParseError
from opk_rag.document_ir import CanonicalBlock, CanonicalFigure, CanonicalSection, CanonicalTable, CanonicalTableCell, SourceLocation


class TeXDocumentAdapter:
    adapter_name = "opk-rag-tex-structured-adapter"
    adapter_version = "1"
    representation_types = ("tex", "latex")

    def adapt(self, source: StructuredDocumentSource):
        try:
            text = source.text()
        except UnicodeDecodeError as exc:
            raise AdapterParseError("tex source is not valid utf-8") from exc
        parts = _parse_tex(text, source.source_ref, source.representation_id)
        return build_canonical_document(
            source=source,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            parts=parts,
        )


TOKEN_RE = re.compile(
    r"\\(?P<section>section|subsection|subsubsection)\*?\{(?P<title>[^{}]+)\}"
    r"|\\begin\{(?P<begin>equation|align|table|tabular|figure)\}"
    r"|\\end\{(?P<end>equation|align|table|tabular|figure)\}"
    r"|\\caption\{(?P<caption>[^{}]*)\}",
    re.DOTALL,
)


def _parse_tex(text: str, source_ref: str, representation_id: str) -> ParsedDocumentParts:
    sections: list[CanonicalSection] = [CanonicalSection("root", None, 0, None, 0, (), ())]
    section_block_ids: dict[str, list[str]] = {"root": []}
    section_stack: list[tuple[int, str, tuple[str, ...]]] = [(0, "root", ())]
    blocks: list[CanonicalBlock] = []
    tables: list[CanonicalTable] = []
    figures: list[CanonicalFigure] = []
    env_stack: list[tuple[str, int]] = []
    reading_order = 0
    title: str | None = None

    def current_section_id() -> str:
        return section_stack[-1][1]

    def add_block(block_type: str, block_text: str | None, metadata: dict | None = None) -> str:
        nonlocal reading_order
        section_id = current_section_id()
        block_id = stable_id("blk", representation_id, reading_order, block_type, block_text)
        blocks.append(CanonicalBlock(block_id, block_type, block_text, section_id, None, len(section_block_ids[section_id]), reading_order, SourceLocation(source_ref=source_ref), metadata or {}))
        section_block_ids[section_id].append(block_id)
        reading_order += 1
        return block_id

    def flush_paragraph(start: int, end: int) -> None:
        paragraph = _clean_tex_text(text[start:end])
        if paragraph:
            add_block("paragraph", paragraph)

    cursor = 0
    for match in TOKEN_RE.finditer(text):
        if not env_stack:
            flush_paragraph(cursor, match.start())
        if match.group("section"):
            level = {"section": 1, "subsection": 2, "subsubsection": 3}[match.group("section")]
            heading = match.group("title").strip()
            section_stack = [item for item in section_stack if item[0] < level]
            parent_id = section_stack[-1][1] if section_stack else "root"
            parent_path = section_stack[-1][2] if section_stack else ()
            path = parent_path + (heading,)
            section_id = stable_id("sec", representation_id, path, len(sections))
            sections.append(CanonicalSection(section_id, parent_id, level, heading, len(sections), path, ()))
            section_block_ids[section_id] = []
            section_stack.append((level, section_id, path))
            title = title or heading
            add_block("heading", heading)
        elif match.group("begin"):
            env_stack.append((match.group("begin"), match.end()))
        elif match.group("end"):
            if not env_stack or env_stack[-1][0] != match.group("end"):
                raise AdapterParseError(f"malformed tex environment: unexpected end {match.group('end')}")
            env, start = env_stack.pop()
            body = text[start:match.start()]
            if not env_stack:
                if env in {"equation", "align"}:
                    add_block("formula", _clean_tex_text(body), metadata={"tex_environment": env})
                elif env in {"table", "tabular"}:
                    block_id = add_block("table", None, metadata={"tex_environment": env})
                    table = _tex_table(block_id, body, representation_id, len(tables), source_ref)
                    tables.append(table)
                elif env == "figure":
                    caption = _caption(body)
                    block_id = add_block("figure", caption, metadata={"tex_environment": env})
                    figures.append(CanonicalFigure(stable_id("fig", representation_id, len(figures)), block_id, caption=caption, source_location=SourceLocation(source_ref=source_ref)))
        cursor = match.end()
    if env_stack:
        raise AdapterParseError(f"malformed tex environment: unclosed {env_stack[-1][0]}")
    flush_paragraph(cursor, len(text))
    sections = [replace(section, block_ids=tuple(section_block_ids.get(section.section_id, ()))) for section in sections]
    return ParsedDocumentParts(title=title, sections=tuple(sections), blocks=tuple(blocks), tables=tuple(tables), figures=tuple(figures), metadata={"tex_static_parse": True})


def _clean_tex_text(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


def _caption(text: str) -> str | None:
    match = re.search(r"\\caption\{([^{}]*)\}", text)
    return match.group(1).strip() if match else None


def _tex_table(block_id: str, body: str, representation_id: str, index: int, source_ref: str) -> CanonicalTable:
    body = re.sub(r"\\caption\{[^{}]*\}", "", body)
    rows_raw = [row for row in re.split(r"\\\\", body) if row.strip()]
    rows = [[_clean_tex_text(cell) for cell in row.split("&")] for row in rows_raw]
    rows = [[cell for cell in row if cell] for row in rows if any(cell for cell in row)]
    if not rows:
        rows = [[""]]
    columns = max(len(row) for row in rows)
    cells = tuple(CanonicalTableCell(row_index=r, column_index=c, text=value) for r, row in enumerate(rows) for c, value in enumerate(row))
    return CanonicalTable(stable_id("tbl", representation_id, index), block_id, len(rows), columns, cells, caption=_caption(body), source_location=SourceLocation(source_ref=source_ref))
