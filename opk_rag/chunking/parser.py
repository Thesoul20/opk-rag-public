from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from opk_rag.chunking.models import MarkdownSection, ParsedMarkdownDocument


class MarkdownParseError(ValueError):
    pass


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```+|~~~+)")


def parse_markdown_file(path: str | Path) -> ParsedMarkdownDocument:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MarkdownParseError(f"{source_path}: file is not valid UTF-8") from exc
    return parse_markdown(text, source_path=source_path)


def parse_markdown(text: str, source_path: str | Path | None = None) -> ParsedMarkdownDocument:
    path = Path(source_path) if source_path is not None else None
    lines = text.splitlines()
    frontmatter, body_start = _parse_frontmatter(lines, path)
    body_lines = lines[body_start:]
    sections = _parse_sections(body_lines, body_start + 1)
    title = _document_title(frontmatter, sections)
    body = "\n".join(body_lines)
    if text.endswith("\n") and body:
        body += "\n"
    return ParsedMarkdownDocument(
        source_path=path,
        frontmatter=frontmatter,
        body=body,
        title=title,
        sections=sections,
    )


def _parse_frontmatter(lines: list[str], source_path: Path | None) -> tuple[dict, int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        location = f"{source_path}:1" if source_path else "line 1"
        raise MarkdownParseError(f"Unclosed YAML frontmatter starting at {location}.")

    raw = "\n".join(lines[1:closing_index])
    try:
        parsed = yaml.safe_load(raw) if raw.strip() else None
    except yaml.YAMLError as exc:
        location = str(source_path) if source_path else "markdown"
        raise MarkdownParseError(f"{location}: invalid frontmatter: {exc}") from exc
    if parsed is None:
        return {}, closing_index + 1
    if not isinstance(parsed, dict):
        location = str(source_path) if source_path else "markdown"
        raise MarkdownParseError(f"{location}: invalid frontmatter: root object must be a mapping")

    return _json_safe_mapping(parsed), closing_index + 1


def _json_safe_mapping(value: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _json_safe_mapping(value)
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _parse_sections(body_lines: list[str], first_body_line_number: int) -> tuple[MarkdownSection, ...]:
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()
    current_start = first_body_line_number
    in_fence = False
    fence_marker = ""

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_start
        content = "\n".join(current_lines).strip("\n")
        if content.strip():
            sections.append(MarkdownSection(current_path, content, current_start, end_line))
        current_lines = []

    for offset, line in enumerate(body_lines):
        line_number = first_body_line_number + offset
        fence_match = _FENCE_RE.match(line)
        heading_match = None if in_fence else _HEADING_RE.match(line)

        if heading_match is not None:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, heading))
            current_path = tuple(item[1] for item in heading_stack)
            current_start = line_number
            current_lines = [line]
        else:
            if not current_lines:
                current_start = line_number
            current_lines.append(line)

        if fence_match is not None:
            marker = fence_match.group(1)
            marker_prefix = marker[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker_prefix * len(marker)
            elif marker_prefix == fence_marker[0] and len(marker) >= len(fence_marker):
                in_fence = False
                fence_marker = ""

    flush(first_body_line_number + len(body_lines) - 1)
    return tuple(sections)


def _document_title(frontmatter: dict, sections: tuple[MarkdownSection, ...]) -> str | None:
    title = frontmatter.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    for section in sections:
        if section.heading_path:
            return section.heading_path[-1]
    return None
