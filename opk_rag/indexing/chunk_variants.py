from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from opk_rag.chunking.chunker import chunk_markdown_document, normalize_chunk_content
from opk_rag.chunking.models import ChunkingConfig
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    SourceDocument,
    ShadowChunk,
    build_shadow_chunk_identity,
)
from opk_rag.evaluation.scope_structure_audit import classify_block_structure
from opk_rag.embedding.input import normalize_embedding_text


TASK0081_VARIANT_IDS = ("C0", "C1", "C2", "C3", "C4")


@dataclass(frozen=True)
class Task0081ChunkVariant:
    id: str
    label: str
    max_chunk_tokens: int
    target_chunk_tokens: int
    min_chunk_tokens: int
    overlap_tokens: int
    heading_depth: int
    include_heading_in_embedding: bool
    include_document_title: bool
    preserve_list_block: bool
    preserve_table_block: bool
    preserve_code_block: bool
    parent_context_enriched: bool = False
    production_default_change: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


TASK0081_VARIANTS: dict[str, Task0081ChunkVariant] = {
    "C0": Task0081ChunkVariant(
        id="C0",
        label="Frozen current production chunking",
        max_chunk_tokens=1800,
        target_chunk_tokens=1200,
        min_chunk_tokens=1,
        overlap_tokens=0,
        heading_depth=6,
        include_heading_in_embedding=True,
        include_document_title=False,
        preserve_list_block=False,
        preserve_table_block=False,
        preserve_code_block=True,
    ),
    "C1": Task0081ChunkVariant(
        id="C1",
        label="Markdown structure-aware chunking",
        max_chunk_tokens=1200,
        target_chunk_tokens=900,
        min_chunk_tokens=1,
        overlap_tokens=0,
        heading_depth=6,
        include_heading_in_embedding=True,
        include_document_title=False,
        preserve_list_block=True,
        preserve_table_block=True,
        preserve_code_block=True,
    ),
    "C2": Task0081ChunkVariant(
        id="C2",
        label="Structure-aware smaller leaf chunk",
        max_chunk_tokens=780,
        target_chunk_tokens=560,
        min_chunk_tokens=1,
        overlap_tokens=0,
        heading_depth=6,
        include_heading_in_embedding=True,
        include_document_title=False,
        preserve_list_block=True,
        preserve_table_block=True,
        preserve_code_block=True,
    ),
    "C3": Task0081ChunkVariant(
        id="C3",
        label="Structure-aware smaller leaf chunk with controlled boundary overlap",
        max_chunk_tokens=780,
        target_chunk_tokens=560,
        min_chunk_tokens=1,
        overlap_tokens=96,
        heading_depth=6,
        include_heading_in_embedding=True,
        include_document_title=False,
        preserve_list_block=True,
        preserve_table_block=True,
        preserve_code_block=True,
    ),
    "C4": Task0081ChunkVariant(
        id="C4",
        label="Parent-context enriched leaf chunk",
        max_chunk_tokens=780,
        target_chunk_tokens=560,
        min_chunk_tokens=1,
        overlap_tokens=96,
        heading_depth=6,
        include_heading_in_embedding=True,
        include_document_title=True,
        preserve_list_block=True,
        preserve_table_block=True,
        preserve_code_block=True,
        parent_context_enriched=True,
    ),
}


def task0081_variant_contracts() -> list[dict[str, Any]]:
    return [TASK0081_VARIANTS[variant_id].to_json() for variant_id in TASK0081_VARIANT_IDS]


def build_task0081_chunks(documents: tuple[SourceDocument, ...], variant_id: str) -> tuple[ShadowChunk, ...]:
    variant = TASK0081_VARIANTS[variant_id]
    if variant_id == "C0":
        return _build_c0_chunks(documents)
    chunks = _build_structure_chunks(documents, variant)
    if variant.overlap_tokens:
        chunks = _apply_controlled_overlap(documents, chunks, variant)
    return chunks


def render_task0081_embedding_input(chunk: ShadowChunk, variant_id: str) -> str:
    variant = TASK0081_VARIANTS[variant_id]
    content = normalize_embedding_text(chunk.content)
    context_parts: list[str] = []
    if variant.include_document_title and chunk.document_title:
        context_parts.append(chunk.document_title)
    if variant.include_heading_in_embedding and chunk.heading_path:
        context_parts.extend(chunk.heading_path[: variant.heading_depth])
    context = normalize_embedding_text("\n".join(context_parts))
    return f"{context}\n\n{content}" if context else content


def _build_c0_chunks(documents: tuple[SourceDocument, ...]) -> tuple[ShadowChunk, ...]:
    out: list[ShadowChunk] = []
    for document in documents:
        for ordinal, chunk in enumerate(chunk_markdown_document(document.parsed, ChunkingConfig())):
            start = chunk.start_line or 1
            end = chunk.end_line or start
            out.append(
                build_shadow_chunk_identity(
                    variant_id="C0",
                    document=document,
                    ordinal=ordinal,
                    content=chunk.content,
                    heading_path=chunk.heading_path,
                    primary_span={"start_line": start, "end_line": end},
                    complete_span={"start_line": start, "end_line": end},
                    block_types=classify_block_structure(chunk.content),
                )
            )
    return tuple(out)


def _build_structure_chunks(documents: tuple[SourceDocument, ...], variant: Task0081ChunkVariant) -> tuple[ShadowChunk, ...]:
    out: list[ShadowChunk] = []
    for document in documents:
        ordinal = 0
        pending: list[tuple[str, int, int, tuple[str, ...]]] = []
        pending_heading: tuple[str, ...] = ()
        for section in document.parsed.sections:
            if pending and pending_heading != section.heading_path:
                out.append(_flush(document, variant.id, ordinal, pending_heading, pending))
                ordinal += 1
                pending = []
            for text, start, end, block_types in _section_blocks(section.content, section.start_line):
                atoms = _split_large_block(text, start, end, variant.max_chunk_tokens)
                for atom_text, atom_start, atom_end in atoms:
                    atom_types = classify_block_structure(atom_text)
                    if not pending:
                        pending_heading = section.heading_path
                    candidate = "\n\n".join(item[0] for item in pending + [(atom_text, atom_start, atom_end, atom_types)]).strip()
                    if pending and len(candidate) > variant.max_chunk_tokens:
                        out.append(_flush(document, variant.id, ordinal, pending_heading, pending))
                        ordinal += 1
                        pending = [(atom_text, atom_start, atom_end, atom_types)]
                        pending_heading = section.heading_path
                    else:
                        pending.append((atom_text, atom_start, atom_end, atom_types))
                    if pending and len("\n\n".join(item[0] for item in pending)) >= variant.target_chunk_tokens:
                        out.append(_flush(document, variant.id, ordinal, pending_heading, pending))
                        ordinal += 1
                        pending = []
        if pending:
            out.append(_flush(document, variant.id, ordinal, pending_heading, pending))
    return tuple(out)


def _apply_controlled_overlap(
    documents: tuple[SourceDocument, ...],
    chunks: tuple[ShadowChunk, ...],
    variant: Task0081ChunkVariant,
) -> tuple[ShadowChunk, ...]:
    by_doc: dict[str, list[ShadowChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_identity_digest, []).append(chunk)
    out: list[ShadowChunk] = []
    for document in documents:
        previous: ShadowChunk | None = None
        for ordinal, chunk in enumerate(by_doc.get(document.document_identity_digest, [])):
            overlap = ""
            complete_span = dict(chunk.primary_source_span)
            if previous and _same_section(previous, chunk):
                overlap = _tail_at_boundary(previous.content, variant.overlap_tokens)
                complete_span = {
                    "start_line": min(previous.primary_source_span["start_line"], chunk.primary_source_span["start_line"]),
                    "end_line": max(previous.primary_source_span["end_line"], chunk.primary_source_span["end_line"]),
                }
            content = f"{overlap}\n\n{chunk.content}".strip() if overlap else chunk.content
            out.append(
                build_shadow_chunk_identity(
                    variant_id=variant.id,
                    document=document,
                    ordinal=ordinal,
                    content=content,
                    heading_path=chunk.heading_path,
                    primary_span=chunk.primary_source_span,
                    complete_span=complete_span,
                    block_types=chunk.block_types,
                    overlap_token_count=len(overlap),
                )
            )
            previous = chunk
    return tuple(out)


def _section_blocks(content: str, section_start_line: int) -> list[tuple[str, int, int, tuple[str, ...]]]:
    lines = content.splitlines()
    blocks: list[tuple[str, int, int, tuple[str, ...]]] = []
    current: list[str] = []
    start = 1
    in_fence = False
    fence_char = ""

    def flush(end_offset: int) -> None:
        nonlocal current
        text = "\n".join(current).strip("\n")
        if text.strip():
            absolute_start = section_start_line + start - 1
            absolute_end = max(absolute_start, section_start_line + end_offset - 1)
            blocks.append((text, absolute_start, absolute_end, classify_block_structure(text)))
        current = []

    for offset, line in enumerate(lines, start=1):
        if not current:
            start = offset
        fence = re.match(r"^[ \t]*(```+|~~~+)", line)
        if not in_fence and not line.strip():
            flush(offset - 1)
            continue
        current.append(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
                flush(offset)
    flush(len(lines))
    return blocks


def _split_large_block(text: str, start: int, end: int, max_size: int) -> list[tuple[str, int, int]]:
    if len(text) <= max_size:
        return [(text, start, end)]
    if _is_atomic_block(text):
        return [(text, start, end)]
    pieces: list[tuple[str, int, int]] = []
    remaining = text
    while len(remaining) > max_size:
        cut = _best_boundary(remaining, max_size)
        pieces.append((remaining[:cut].strip(), start, end))
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append((remaining, start, end))
    return pieces


def _is_atomic_block(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("```", "~~~")):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    return len(lines) > 1 and all(_is_list_line(line) or _is_table_line(line) for line in lines)


def _best_boundary(text: str, max_size: int) -> int:
    window = text[:max_size]
    boundaries = [window.rfind(marker) for marker in ("\n\n", "\n- ", "\n* ", "\n1. ", "。", "！", "？", ". ", "! ", "? ")]
    boundary = max(boundaries)
    return boundary + 1 if boundary > max_size // 3 else max_size


def _flush(
    document: SourceDocument,
    variant_id: str,
    ordinal: int,
    heading_path: tuple[str, ...],
    blocks: list[tuple[str, int, int, tuple[str, ...]]],
) -> ShadowChunk:
    content = normalize_chunk_content("\n\n".join(block[0] for block in blocks))
    return build_shadow_chunk_identity(
        variant_id=variant_id,
        document=document,
        ordinal=ordinal,
        content=content,
        heading_path=heading_path,
        primary_span={"start_line": min(block[1] for block in blocks), "end_line": max(block[2] for block in blocks)},
        complete_span={"start_line": min(block[1] for block in blocks), "end_line": max(block[2] for block in blocks)},
        block_types=tuple(sorted({item for block in blocks for item in block[3]})),
    )


def _same_section(left: ShadowChunk, right: ShadowChunk) -> bool:
    return left.document_identity_digest == right.document_identity_digest and left.heading_path == right.heading_path


def _tail_at_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    tail = text[-max_chars:]
    for marker in ("\n\n", "。", "！", "？", ". ", "! ", "? "):
        index = tail.find(marker)
        if index >= 0 and index < len(tail) - 1:
            return tail[index + len(marker) :].strip()
    return tail.strip()


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]|\d+[.])\s+", line))


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")
