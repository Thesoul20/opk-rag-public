from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from opk_rag.embedding.config import EmbeddingConfig

QUERY_INPUT_TEMPLATE_VERSION = "query-input-v1"
QUERY_INSTRUCTION = "Given a user query, retrieve relevant passages from a personal Chinese Markdown knowledge base."


class QueryInputError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedQueryInput:
    query: str
    text: str
    text_hash: str
    query_token_count: int
    input_token_count: int
    instruction: str
    template_version: str
    truncated: bool = False


def prepare_query_input(
    query: str,
    config: EmbeddingConfig,
    *,
    count_tokens,
    instruction: str = QUERY_INSTRUCTION,
    template_version: str = QUERY_INPUT_TEMPLATE_VERSION,
    max_query_tokens: int | None = None,
) -> PreparedQueryInput:
    normalized = normalize_query(query)
    instruction = instruction.strip()
    template_version = template_version.strip()
    if not instruction:
        raise QueryInputError("Query instruction must not be empty.")
    if not template_version:
        raise QueryInputError("Query template version must not be empty.")

    query_token_count = count_tokens(normalized)
    limit = max_query_tokens if max_query_tokens is not None else config.max_input_tokens
    if query_token_count > limit:
        raise QueryInputError(f"Query exceeds max_query_tokens: {query_token_count} > {limit}.")

    text = render_query_input(normalized, instruction=instruction)
    input_token_count = count_tokens(text)
    if input_token_count > config.max_input_tokens:
        raise QueryInputError(f"Query embedding input exceeds max_input_tokens: {input_token_count} > {config.max_input_tokens}.")

    return PreparedQueryInput(
        query=normalized,
        text=text,
        text_hash=build_query_input_hash(text, config, template_version=template_version),
        query_token_count=query_token_count,
        input_token_count=input_token_count,
        instruction=instruction,
        template_version=template_version,
    )


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise QueryInputError("Query must be a string.")
    normalized = query.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise QueryInputError("Query must not be empty.")
    return normalized


def render_query_input(query: str, *, instruction: str = QUERY_INSTRUCTION) -> str:
    normalized = normalize_query(query)
    instruction = instruction.strip()
    if not instruction:
        raise QueryInputError("Query instruction must not be empty.")
    return f"Instruct: {instruction}\nQuery:{normalized}"


def build_query_input_hash(text: str, config: EmbeddingConfig, *, template_version: str) -> str:
    payload = {
        "text": text,
        "query_template_version": template_version,
        "embedding_provider": config.provider,
        "embedding_model": config.model_name,
        "model_revision": config.model_revision,
        "embedding_dimension": config.dimension,
        "normalize": config.normalize,
        "distance_metric": config.distance_metric,
        "max_input_tokens": config.max_input_tokens,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
