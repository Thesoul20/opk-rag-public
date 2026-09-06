from __future__ import annotations

import hashlib

from opk_rag.document_ir.serialization import digest_json, document_to_dict


def source_bytes_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_content_digest(document: object) -> str:
    return digest_json(document_to_dict(document, semantic=True))
