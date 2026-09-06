from __future__ import annotations

import hashlib
import re

from opk_rag.chunking.models import ChunkingConfig, MarkdownChunk, MarkdownSection, ParsedMarkdownDocument


_BOUNDARY_PATTERNS = (
    re.compile(r"\n{2,}"),
    re.compile(r"\n(?=(?:[-*+]|\d+[.])\s+)"),
    re.compile(r"(?<=[。！？.!?])\s+"),
)


def chunk_markdown_document(
    document: ParsedMarkdownDocument,
    config: ChunkingConfig | None = None,
) -> tuple[MarkdownChunk, ...]:
    config = config or ChunkingConfig()
    raw_chunks: list[tuple[str, tuple[str, ...], int | None, int | None]] = []
    pending_content = ""
    pending_path: tuple[str, ...] = ()
    pending_start: int | None = None
    pending_end: int | None = None

    for section in document.sections:
        section_content = section.content.strip()
        if not section_content:
            continue
        if len(section_content) > config.max_size:
            if pending_content:
                raw_chunks.append((pending_content, pending_path, pending_start, pending_end))
                pending_content = ""
                pending_start = pending_end = None
            for part in _split_oversized_section(section, config.max_size):
                raw_chunks.append(part)
            continue

        candidate = _join_blocks(pending_content, section_content)
        if pending_content and len(candidate) > config.max_size:
            raw_chunks.append((pending_content, pending_path, pending_start, pending_end))
            pending_content = section_content
            pending_path = section.heading_path
            pending_start = section.start_line
            pending_end = section.end_line
        else:
            pending_content = candidate
            pending_path = pending_path or section.heading_path
            pending_start = pending_start if pending_start is not None else section.start_line
            pending_end = section.end_line
            if len(pending_content) >= config.target_size:
                raw_chunks.append((pending_content, pending_path, pending_start, pending_end))
                pending_content = ""
                pending_path = ()
                pending_start = pending_end = None

    if pending_content:
        raw_chunks.append((pending_content, pending_path, pending_start, pending_end))

    chunks = []
    for index, (content, heading_path, start_line, end_line) in enumerate(raw_chunks):
        normalized = normalize_chunk_content(content)
        if not normalized:
            continue
        chunks.append(
            MarkdownChunk(
                chunk_index=len(chunks),
                content=normalized,
                content_hash=chunk_content_hash(normalized),
                heading_path=heading_path,
                start_line=start_line,
                end_line=end_line,
                char_count=len(normalized),
                metadata={
                    "length_unit": config.length_unit,
                    "target_size": config.target_size,
                    "max_size": config.max_size,
                    "overlap": config.overlap,
                    "source_chunk_index": index,
                },
            )
        )
    return tuple(chunks)


def normalize_chunk_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def chunk_content_hash(content: str) -> str:
    return hashlib.sha256(normalize_chunk_content(content).encode("utf-8")).hexdigest()


def _valid_line_range(start_line: int | None, end_line: int | None) -> tuple[int | None, int | None]:
    if start_line is None or end_line is None:
        return None, None
    if end_line < start_line:
        return None, None
    return start_line, end_line


def _split_oversized_section(
    section: MarkdownSection,
    max_size: int,
) -> tuple[tuple[str, tuple[str, ...], int | None, int | None], ...]:
    blocks = _split_blocks_preserving_code(section.content)
    chunks: list[tuple[str, tuple[str, ...], int | None, int | None]] = []
    pending = ""
    pending_start: int | None = None
    pending_end: int | None = None

    for block, start_line, end_line in blocks:
        absolute_start = section.start_line + start_line - 1
        absolute_end = section.start_line + end_line - 1
        if len(block) > max_size and _is_fenced_code_block(block):
            if pending:
                pending_start, pending_end = _valid_line_range(pending_start, pending_end)
                chunks.append((pending, section.heading_path, pending_start, pending_end))
                pending = ""
                pending_start = pending_end = None
            chunks.append((block, section.heading_path, absolute_start, absolute_end))
            continue

        pieces = (block,) if len(block) <= max_size else _split_text_by_boundaries(block, max_size)
        for piece_index, piece in enumerate(pieces):
            piece_start = absolute_start if piece_index == 0 else None
            piece_end = absolute_end if piece_index == len(pieces) - 1 else None
            candidate = _join_blocks(pending, piece)
            if pending and len(candidate) > max_size:
                pending_start, pending_end = _valid_line_range(pending_start, pending_end)
                chunks.append((pending, section.heading_path, pending_start, pending_end))
                pending = piece
                pending_start = piece_start
                pending_end = piece_end
            else:
                pending = candidate
                pending_start = pending_start if pending_start is not None else piece_start
                pending_end = piece_end

    if pending:
        pending_start, pending_end = _valid_line_range(pending_start, pending_end)
        chunks.append((pending, section.heading_path, pending_start, pending_end))
    return tuple(chunks)


def _split_blocks_preserving_code(content: str) -> tuple[tuple[str, int, int], ...]:
    lines = content.splitlines()
    blocks: list[tuple[str, int, int]] = []
    current: list[str] = []
    start_line = 1
    in_fence = False
    fence_char = ""
    fence_length = 0

    def flush(end_line: int) -> None:
        nonlocal current, start_line
        block = "\n".join(current).strip("\n")
        if block.strip():
            blocks.append((block, start_line, end_line))
        current = []

    for offset, line in enumerate(lines, start=1):
        if not current:
            start_line = offset
        fence_match = re.match(r"^[ \t]*(```+|~~~+)", line)
        if not in_fence and not line.strip():
            flush(offset - 1)
            continue
        current.append(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                in_fence = False
                flush(offset)
    flush(len(lines))
    return tuple(blocks)


def _split_text_by_boundaries(text: str, max_size: int) -> tuple[str, ...]:
    pieces = (text,)
    for pattern in _BOUNDARY_PATTERNS:
        next_pieces: list[str] = []
        changed = False
        for piece in pieces:
            if len(piece) <= max_size:
                next_pieces.append(piece)
                continue
            split_piece = _pack_pattern_splits(piece, pattern, max_size)
            changed = changed or len(split_piece) > 1
            next_pieces.extend(split_piece)
        pieces = tuple(next_pieces)
        if all(len(piece) <= max_size for piece in pieces):
            return pieces
        if changed:
            continue
    hard: list[str] = []
    for piece in pieces:
        while len(piece) > max_size:
            hard.append(piece[:max_size])
            piece = piece[max_size:]
        if piece:
            hard.append(piece)
    return tuple(hard)


def _pack_pattern_splits(text: str, pattern: re.Pattern[str], max_size: int) -> tuple[str, ...]:
    atoms = [atom for atom in pattern.split(text) if atom]
    if len(atoms) <= 1:
        return (text,)
    chunks: list[str] = []
    pending = ""
    for atom in atoms:
        candidate = _join_blocks(pending, atom)
        if pending and len(candidate) > max_size:
            chunks.append(pending)
            pending = atom
        else:
            pending = candidate
    if pending:
        chunks.append(pending)
    return tuple(chunks)


def _join_blocks(left: str, right: str) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    return f"{left.rstrip()}\n\n{right.strip()}"


def _is_fenced_code_block(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith(("```", "~~~")) and stripped.count("```") >= 2 or stripped.count("~~~") >= 2
