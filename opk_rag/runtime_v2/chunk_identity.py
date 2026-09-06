from __future__ import annotations

import hashlib
import json
from typing import Any


CANONICAL_IDENTITY_SCHEMA_VERSION = "opk-rag.canonical-retrieval-runtime-v2.chunk-identity.v1"


def digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\", "/").strip("/")


def canonical_chunk_id_for(
    *,
    corpus_snapshot_id: str,
    normalized_source_path: str | None,
    source_start: int | None,
    source_end: int | None,
    content_digest: str,
) -> str:
    authority = {
        "schema_version": CANONICAL_IDENTITY_SCHEMA_VERSION,
        "corpus_snapshot_id": corpus_snapshot_id,
        "normalized_source_path": normalized_path(normalized_source_path),
        "source_start": source_start,
        "source_end": source_end,
        "content_digest": content_digest,
    }
    return digest_json(authority)
