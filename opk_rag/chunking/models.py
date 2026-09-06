from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PARSER_VERSION = "markdown-parser-v1"
CHUNKING_VERSION = "markdown-heading-char-v1"


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: tuple[str, ...]
    content: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ParsedMarkdownDocument:
    source_path: Path | None
    frontmatter: dict
    body: str
    title: str | None
    sections: tuple[MarkdownSection, ...]


@dataclass(frozen=True)
class ChunkingConfig:
    target_size: int = 1200
    max_size: int = 1800
    overlap: int = 0
    length_unit: str = "character"

    def __post_init__(self) -> None:
        if self.length_unit != "character":
            raise ValueError("Only character length is supported by the current deterministic chunker.")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive.")
        if self.max_size <= 0:
            raise ValueError("max_size must be positive.")
        if self.target_size > self.max_size:
            raise ValueError("target_size must be less than or equal to max_size.")
        if self.overlap != 0:
            raise ValueError("overlap must be 0 until a deterministic overlap policy is introduced.")


@dataclass(frozen=True)
class MarkdownChunk:
    chunk_index: int
    content: str
    content_hash: str
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None
    char_count: int
    metadata: dict = field(default_factory=dict)
