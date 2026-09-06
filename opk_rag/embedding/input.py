from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from opk_rag.db.models import StoredChunk
from opk_rag.embedding.config import EmbeddingConfig


class EmbeddingInputError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedEmbeddingInput:
    chunk_id: object
    text: str
    text_hash: str
    retrieval_text_digest: str
    content_token_count: int
    input_token_count: int
    metadata: dict[str, object]


def prepare_embedding_input(
    chunk: StoredChunk,
    config: EmbeddingConfig,
    *,
    count_tokens,
) -> PreparedEmbeddingInput:
    text = render_embedding_input(chunk, config)
    input_token_count = count_tokens(text)
    if input_token_count > config.max_input_tokens:
        raise EmbeddingInputError(
            f"Embedding input exceeds max_input_tokens for chunk {chunk.id}: "
            f"{input_token_count} > {config.max_input_tokens}"
        )
    content_token_count = count_tokens(normalize_embedding_text(chunk.content))
    text_hash = build_embedding_input_hash(text, config)
    retrieval_text_digest = build_retrieval_text_digest(text, config)
    return PreparedEmbeddingInput(
        chunk_id=chunk.id,
        text=text,
        text_hash=text_hash,
        retrieval_text_digest=retrieval_text_digest,
        content_token_count=content_token_count,
        input_token_count=input_token_count,
        metadata={
            "embedding_input_hash": text_hash,
            "embedding_input_template_version": config.input_template_version,
            "embedding_input_token_count": input_token_count,
            "embedding_input_truncated": False,
            "embedding_model_revision": config.model_revision,
            "embedding_provider": config.provider,
            "embedding_normalize": config.normalize,
            "retrieval_representation_policy": config.retrieval_representation_policy,
            "retrieval_text_digest": retrieval_text_digest,
            "max_heading_context_tokens": config.max_heading_context_tokens,
            "heading_context_missing": not bool(normalize_heading_context(chunk.heading_path, config)),
        },
    )


def render_embedding_input(chunk: StoredChunk, config: EmbeddingConfig | None = None) -> str:
    config = config or EmbeddingConfig()
    content = normalize_embedding_text(chunk.content)
    if not content:
        raise EmbeddingInputError(f"Embedding input is empty for chunk {chunk.id}.")
    heading_context = normalize_heading_context(chunk.heading_path, config)
    if config.retrieval_representation_policy == "heading_context_enriched" and heading_context:
        heading = "\n".join(heading_context)
        text = f"{heading}\n\n{content}"
    else:
        text = content
    if not text.strip():
        raise EmbeddingInputError(f"Embedding input is empty for chunk {chunk.id}.")
    return text


def normalize_heading_context(heading_path: tuple[str, ...], config: EmbeddingConfig) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for part in heading_path:
        value = normalize_embedding_text(part)
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    if config.max_heading_context_tokens == 0:
        return ()
    budget = config.max_heading_context_tokens
    selected: list[str] = []
    used = 0
    for value in reversed(normalized):
        token_count = len(value.split()) or 1
        if selected and used + token_count > budget:
            break
        if not selected and token_count > budget:
            value = " ".join(value.split()[-budget:]) if value.split() else value
            token_count = len(value.split()) or 1
        selected.append(value)
        used += token_count
    return tuple(reversed(selected))


def normalize_embedding_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_retrieval_text_digest(text: str, config: EmbeddingConfig) -> str:
    payload = {
        "retrieval_text": text,
        "retrieval_representation_policy": config.retrieval_representation_policy,
        "max_heading_context_tokens": config.max_heading_context_tokens,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_embedding_input_hash(text: str, config: EmbeddingConfig) -> str:
    payload = {
        "text": text,
        "input_template_version": config.input_template_version,
        "embedding_provider": config.provider,
        "embedding_model": config.model_name,
        "model_revision": config.model_revision,
        "embedding_dimension": config.dimension,
        "normalize": config.normalize,
        "distance_metric": config.distance_metric,
        "max_input_tokens": config.max_input_tokens,
        "retrieval_representation_policy": config.retrieval_representation_policy,
        "retrieval_text_digest": build_retrieval_text_digest(text, config),
        "max_heading_context_tokens": config.max_heading_context_tokens,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
