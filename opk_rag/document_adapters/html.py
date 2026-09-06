from __future__ import annotations

from dataclasses import replace
from html.parser import HTMLParser

from opk_rag.document_adapters.base import ParsedDocumentParts, StructuredDocumentSource, build_canonical_document, stable_id
from opk_rag.document_adapters.errors import AdapterParseError
from opk_rag.document_ir import CanonicalBlock, CanonicalFigure, CanonicalSection, CanonicalTable, CanonicalTableCell, SourceLocation


class HTMLDocumentAdapter:
    adapter_name = "opk-rag-html-structured-adapter"
    adapter_version = "1"
    representation_types = ("html", "htm")

    def adapt(self, source: StructuredDocumentSource):
        parser = _StructuredHTMLParser(source.source_ref, source.representation_id)
        try:
            parser.feed(source.text())
            parser.close()
        except (UnicodeDecodeError, AdapterParseError):
            raise
        except Exception as exc:
            raise AdapterParseError(str(exc)) from exc
        parts = parser.parts()
        return build_canonical_document(
            source=source,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            parts=parts,
        )


class _StructuredHTMLParser(HTMLParser):
    _TRACKED = {"html", "body", "section", "article", "figure", "table", "tr", "td", "th"}

    def __init__(self, source_ref: str, representation_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_ref = source_ref
        self.representation_id = representation_id
        self.sections: list[CanonicalSection] = [CanonicalSection("root", None, 0, None, 0, (), ())]
        self.section_block_ids: dict[str, list[str]] = {"root": []}
        self.section_stack: list[tuple[int, str, tuple[str, ...]]] = [(0, "root", ())]
        self.blocks: list[CanonicalBlock] = []
        self.tables: list[CanonicalTable] = []
        self.figures: list[CanonicalFigure] = []
        self.stack: list[str] = []
        self.capture: tuple[str, list[str], dict[str, str]] | None = None
        self.current_table: list[list[str]] | None = None
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.current_figure: dict[str, str | None] | None = None
        self.reading_order = 0
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in self._TRACKED:
            self.stack.append(tag)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "figcaption", "title"}:
            self.capture = (tag, [], attrs_dict)
        elif tag == "table":
            if self.current_table is not None:
                raise AdapterParseError("nested html tables are unsupported")
            self.current_table = []
        elif tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []
        elif tag == "figure":
            self.current_figure = {"caption": None, "alt_text": None, "asset_reference": None}
        elif tag == "img" and self.current_figure is not None:
            self.current_figure["alt_text"] = attrs_dict.get("alt") or None
            self.current_figure["asset_reference"] = attrs_dict.get("src") or None

    def handle_endtag(self, tag: str) -> None:
        if self.capture and self.capture[0] == tag:
            _, chunks, attrs = self.capture
            text = " ".join("".join(chunks).split())
            self.capture = None
            if tag == "title":
                self.title = text or self.title
            elif tag.startswith("h") and len(tag) == 2:
                self._add_heading(int(tag[1]), text)
            elif tag == "p" and text:
                self._add_block("paragraph", text)
            elif tag == "li" and text:
                self._add_block("list", text, metadata={"list_item": True})
            elif tag == "figcaption" and self.current_figure is not None:
                self.current_figure["caption"] = text or None
            return
        if tag in {"td", "th"} and self.current_cell is not None:
            text = " ".join("".join(self.current_cell).split())
            if self.current_row is not None:
                self.current_row.append(text)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_table is not None:
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.current_table is not None:
            self._add_table(self.current_table)
            self.current_table = None
        elif tag == "figure" and self.current_figure is not None:
            self._add_figure(self.current_figure)
            self.current_figure = None
        if tag in self._TRACKED:
            if not self.stack or self.stack[-1] != tag:
                raise AdapterParseError(f"malformed html: unexpected closing tag {tag}")
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)
        if self.capture is not None:
            self.capture[1].append(data)

    def parts(self) -> ParsedDocumentParts:
        if self.stack:
            raise AdapterParseError(f"malformed html: unclosed tags {self.stack}")
        sections = tuple(replace(section, block_ids=tuple(self.section_block_ids.get(section.section_id, ()))) for section in self.sections)
        return ParsedDocumentParts(title=self.title or _first_section_title(sections), sections=sections, blocks=tuple(self.blocks), tables=tuple(self.tables), figures=tuple(self.figures))

    def _add_heading(self, level: int, text: str) -> None:
        if not text:
            return
        self.section_stack = [item for item in self.section_stack if item[0] < level]
        parent_id = self.section_stack[-1][1] if self.section_stack else "root"
        parent_path = self.section_stack[-1][2] if self.section_stack else ()
        path = parent_path + (text,)
        section_id = stable_id("sec", self.representation_id, path, len(self.sections))
        self.sections.append(CanonicalSection(section_id, parent_id, level, text, len(self.sections), path, ()))
        self.section_block_ids[section_id] = []
        self.section_stack.append((level, section_id, path))
        self._add_block("heading", text)

    def _add_block(self, block_type: str, text: str | None, metadata: dict | None = None) -> str:
        section_id = self.section_stack[-1][1]
        block_id = stable_id("blk", self.representation_id, self.reading_order, block_type, text)
        self.blocks.append(CanonicalBlock(block_id, block_type, text, section_id, None, len(self.section_block_ids[section_id]), self.reading_order, SourceLocation(source_ref=self.source_ref), metadata or {}))
        self.section_block_ids[section_id].append(block_id)
        self.reading_order += 1
        return block_id

    def _add_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        columns = max(len(row) for row in rows)
        block_id = self._add_block("table", None)
        cells = tuple(CanonicalTableCell(row_index=r, column_index=c, text=value) for r, row in enumerate(rows) for c, value in enumerate(row))
        self.tables.append(CanonicalTable(stable_id("tbl", self.representation_id, len(self.tables)), block_id, len(rows), columns, cells, source_location=SourceLocation(source_ref=self.source_ref)))

    def _add_figure(self, figure: dict[str, str | None]) -> None:
        block_id = self._add_block("figure", figure.get("caption") or figure.get("alt_text"))
        self.figures.append(CanonicalFigure(stable_id("fig", self.representation_id, len(self.figures)), block_id, caption=figure.get("caption"), alt_text=figure.get("alt_text"), asset_reference=figure.get("asset_reference"), source_location=SourceLocation(source_ref=self.source_ref)))


def _first_section_title(sections: tuple[CanonicalSection, ...]) -> str | None:
    for section in sections:
        if section.title:
            return section.title
    return None
