from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from opk_rag.chunking.chunker import normalize_chunk_content
from opk_rag.document_ir import CanonicalBlock, CanonicalDocument
from opk_rag.document_ir.serialization import digest_json


REPRESENTATION_CHUNKING_VERSION = "representation-aware-deterministic-v1"
ATOMIC_BLOCK_TYPES = {"table", "code"}


@dataclass(frozen=True)
class RepresentationChunkingConfig:
    target_chars: int = 1200
    max_chars: int = 1800
    minimum_target_chars: int = 240
    length_unit: str = "character"
    strategy_version: str = REPRESENTATION_CHUNKING_VERSION

    def __post_init__(self) -> None:
        if self.length_unit != "character":
            raise ValueError("Only character length is supported by representation-aware chunking.")
        if self.minimum_target_chars <= 0:
            raise ValueError("minimum_target_chars must be positive.")
        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive.")
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if self.minimum_target_chars > self.target_chars:
            raise ValueError("minimum_target_chars must be <= target_chars.")
        if self.target_chars > self.max_chars:
            raise ValueError("target_chars must be <= max_chars.")

    @property
    def config_digest(self) -> str:
        return digest_json(
            {
                "target_chars": self.target_chars,
                "max_chars": self.max_chars,
                "minimum_target_chars": self.minimum_target_chars,
                "length_unit": self.length_unit,
                "strategy_version": self.strategy_version,
            }
        )


@dataclass(frozen=True)
class CanonicalBlockView:
    block_id: str
    ordinal: int
    text: str
    block_type: str
    page_number: int | None
    section_id: str | None
    section_path: tuple[str, ...]
    heading_context: tuple[str, ...]
    source_ref: str | None


@dataclass(frozen=True)
class RepresentationChunk:
    chunk_id: str
    strategy_id: str
    strategy_version: str
    config_digest: str
    document_id: str
    chunk_index: int
    content_text: str
    retrieval_text: str
    source_block_ids: tuple[str, ...]
    source_page_numbers: tuple[int, ...]
    source_refs: tuple[str, ...]
    start_block_ordinal: int
    end_block_ordinal: int
    heading_context: tuple[str, ...]
    section_ids: tuple[str, ...]
    section_path: tuple[str, ...]
    block_types: tuple[str, ...]
    fallback_split: bool = False
    cross_section_merge: bool = False

    @property
    def provenance_valid(self) -> bool:
        return bool(self.document_id and self.source_block_ids and self.start_block_ordinal <= self.end_block_ordinal)


def canonical_block_views(document: CanonicalDocument) -> tuple[CanonicalBlockView, ...]:
    sections = {section.section_id: section for section in document.sections}
    heading_context: list[str] = []
    views: list[CanonicalBlockView] = []
    for block in sorted(document.blocks, key=lambda item: item.reading_order):
        text = normalize_chunk_content(block.text or "")
        block_type = normalized_block_type(block)
        section = sections.get(block.section_id or "")
        section_path = tuple(section.path) if section else tuple(heading_context)
        if block_type == "heading" and text:
            level = heading_level(text)
            title_lines = [line.strip() for line in text.strip("#* ").splitlines() if line.strip()]
            title = (title_lines[0] if title_lines else text)[:160]
            heading_context = heading_context[: max(0, level - 1)] + [title]
            section_path = tuple(heading_context)
        views.append(
            CanonicalBlockView(
                block_id=block.block_id,
                ordinal=block.reading_order,
                text=text,
                block_type=block_type,
                page_number=block.source_location.page_number if block.source_location else None,
                section_id=block.section_id,
                section_path=section_path,
                heading_context=tuple(heading_context),
                source_ref=block.source_location.source_ref if block.source_location else None,
            )
        )
    return tuple(view for view in views if view.text)


def build_block_aware_chunks(
    document: CanonicalDocument,
    config: RepresentationChunkingConfig | None = None,
    *,
    strategy_id: str = "block_aware_bounded_merge_v1",
) -> tuple[RepresentationChunk, ...]:
    config = config or RepresentationChunkingConfig()
    return _bounded_merge_chunks(document, canonical_block_views(document), config, strategy_id=strategy_id, respect_sections=False)


def build_section_aware_chunks(
    document: CanonicalDocument,
    config: RepresentationChunkingConfig | None = None,
    *,
    strategy_id: str = "section_aware_bounded_merge_v1",
) -> tuple[RepresentationChunk, ...]:
    config = config or RepresentationChunkingConfig()
    return _bounded_merge_chunks(document, canonical_block_views(document), config, strategy_id=strategy_id, respect_sections=True)


def enrich_heading_context(chunk: RepresentationChunk, *, strategy_id: str = "heading_context_enriched_v1") -> RepresentationChunk:
    prefix = "\n> ".join(chunk.heading_context)
    retrieval_text = f"{prefix}\n\n{chunk.content_text}" if prefix else chunk.content_text
    return RepresentationChunk(
        **{
            **chunk.__dict__,
            "chunk_id": stable_chunk_id(
                document_id=chunk.document_id,
                strategy_id=strategy_id,
                source_block_ids=chunk.source_block_ids,
                config_digest=chunk.config_digest,
                content_text=chunk.content_text,
                retrieval_text=retrieval_text,
            ),
            "strategy_id": strategy_id,
            "retrieval_text": retrieval_text,
        }
    )


def _bounded_merge_chunks(
    document: CanonicalDocument,
    blocks: tuple[CanonicalBlockView, ...],
    config: RepresentationChunkingConfig,
    *,
    strategy_id: str,
    respect_sections: bool,
) -> tuple[RepresentationChunk, ...]:
    chunks: list[RepresentationChunk] = []
    pending: list[CanonicalBlockView] = []

    def flush(*, cross_section_merge: bool = False) -> None:
        nonlocal pending
        if pending:
            chunks.append(_make_chunk(document.document_id, strategy_id, config, len(chunks), pending, cross_section_merge=cross_section_merge))
            pending = []

    for block in blocks:
        if len(block.text) > config.max_chars:
            flush()
            for piece_index, piece in enumerate(split_oversized_block(block.text, config.max_chars)):
                piece_block = CanonicalBlockView(
                    block_id=block.block_id,
                    ordinal=block.ordinal,
                    text=piece,
                    block_type=block.block_type,
                    page_number=block.page_number,
                    section_id=block.section_id,
                    section_path=block.section_path,
                    heading_context=block.heading_context,
                    source_ref=block.source_ref,
                )
                chunks.append(
                    _make_chunk(
                        document.document_id,
                        strategy_id,
                        config,
                        len(chunks),
                        [piece_block],
                        fallback_split=True,
                        split_index=piece_index,
                    )
                )
            continue

        if block.block_type in ATOMIC_BLOCK_TYPES:
            flush()
            chunks.append(_make_chunk(document.document_id, strategy_id, config, len(chunks), [block]))
            continue

        if not pending:
            pending = [block]
            continue

        candidate_text = join_block_text([*pending, block])
        section_change = _section_key(pending[-1]) != _section_key(block)
        if respect_sections and section_change:
            flush()
            pending = [block]
            continue
        if len(candidate_text) > config.max_chars:
            flush()
            pending = [block]
            continue

        pending.append(block)
        if len(join_block_text(pending)) >= config.target_chars:
            flush(cross_section_merge=not respect_sections and _has_cross_section(pending))

    flush(cross_section_merge=not respect_sections and _has_cross_section(pending))
    return tuple(chunks)


def _make_chunk(
    document_id: str,
    strategy_id: str,
    config: RepresentationChunkingConfig,
    chunk_index: int,
    blocks: Iterable[CanonicalBlockView],
    *,
    fallback_split: bool = False,
    cross_section_merge: bool = False,
    split_index: int | None = None,
) -> RepresentationChunk:
    selected = tuple(blocks)
    content_text = join_block_text(selected)
    retrieval_text = content_text
    source_block_ids = tuple(block.block_id for block in selected)
    pages = tuple(sorted({block.page_number for block in selected if isinstance(block.page_number, int)}))
    refs = tuple(sorted({block.source_ref for block in selected if block.source_ref}))
    section_ids = tuple(sorted({block.section_id for block in selected if block.section_id}))
    block_types = tuple(sorted({block.block_type for block in selected}))
    heading_context = selected[0].heading_context if selected else ()
    section_path = selected[0].section_path if selected else ()
    identity_blocks = source_block_ids if split_index is None else (*source_block_ids, f"split:{split_index}")
    return RepresentationChunk(
        chunk_id=stable_chunk_id(
            document_id=document_id,
            strategy_id=strategy_id,
            source_block_ids=identity_blocks,
            config_digest=config.config_digest,
            content_text=content_text,
            retrieval_text=retrieval_text,
        ),
        strategy_id=strategy_id,
        strategy_version=config.strategy_version,
        config_digest=config.config_digest,
        document_id=document_id,
        chunk_index=chunk_index,
        content_text=content_text,
        retrieval_text=retrieval_text,
        source_block_ids=source_block_ids,
        source_page_numbers=pages,
        source_refs=refs,
        start_block_ordinal=min((block.ordinal for block in selected), default=0),
        end_block_ordinal=max((block.ordinal for block in selected), default=0),
        heading_context=heading_context,
        section_ids=section_ids,
        section_path=section_path,
        block_types=block_types,
        fallback_split=fallback_split,
        cross_section_merge=cross_section_merge or len(section_ids) > 1,
    )


def stable_chunk_id(
    *,
    document_id: str,
    strategy_id: str,
    source_block_ids: tuple[str, ...],
    config_digest: str,
    content_text: str,
    retrieval_text: str,
) -> str:
    payload = "\x1f".join([document_id, strategy_id, ",".join(source_block_ids), config_digest, content_text, retrieval_text])
    return f"{strategy_id.split('_')[0]}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def split_oversized_block(text: str, max_chars: int) -> tuple[str, ...]:
    normalized = normalize_chunk_content(text)
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    pieces = _pack_units(paragraphs, max_chars) if len(paragraphs) > 1 else (normalized,)
    if all(len(piece) <= max_chars for piece in pieces):
        return pieces
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", normalized) if part.strip()]
    pieces = _pack_units(sentences, max_chars) if len(sentences) > 1 else (normalized,)
    if all(len(piece) <= max_chars for piece in pieces):
        return pieces
    hard: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            hard.append(piece[:max_chars].strip())
            piece = piece[max_chars:]
        if piece.strip():
            hard.append(piece.strip())
    return tuple(hard)


def _pack_units(units: list[str], max_chars: int) -> tuple[str, ...]:
    pieces: list[str] = []
    pending = ""
    for unit in units:
        candidate = join_text(pending, unit)
        if pending and len(candidate) > max_chars:
            pieces.append(pending)
            pending = unit
        else:
            pending = candidate
    if pending:
        pieces.append(pending)
    return tuple(pieces)


def join_block_text(blocks: Iterable[CanonicalBlockView]) -> str:
    return join_text("", "\n\n".join(block.text.strip() for block in blocks if block.text.strip()))


def join_text(left: str, right: str) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    return f"{left.rstrip()}\n\n{right.strip()}"


def normalized_block_type(block: CanonicalBlock) -> str:
    text = (block.text or "").strip()
    raw = str(block.block_type or "").lower()
    if raw in {"heading", "paragraph", "list", "table", "code"}:
        return raw
    first = text.lstrip().splitlines()[0] if text else ""
    if first.startswith("#") or (first.startswith("**") and len(first) < 120):
        return "heading"
    if first.startswith(("- ", "* ")) or re.match(r"^\d+[.)]\s+", first):
        return "list"
    if "|" in text and "\n" in text:
        return "table"
    if first.startswith(("```", "~~~")):
        return "code"
    return "paragraph"


def heading_level(text: str) -> int:
    stripped = text.lstrip()
    hashes = len(stripped) - len(stripped.lstrip("#"))
    return min(hashes, 6) if hashes else 1


def _section_key(block: CanonicalBlockView) -> tuple[str | None, tuple[str, ...]]:
    return block.section_id, block.section_path


def _has_cross_section(blocks: Iterable[CanonicalBlockView]) -> bool:
    keys = {_section_key(block) for block in blocks}
    return len(keys) > 1
