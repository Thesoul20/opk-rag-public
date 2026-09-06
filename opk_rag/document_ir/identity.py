from __future__ import annotations

from typing import Any

from opk_rag.document_ir.serialization import digest_json, normalize_representation_type


IDENTITY_SCHEMA_VERSION = "opk-rag.canonical-document-ir.identity.v1"


def knowledge_identity_for(*, authority: str, stable_reference: str, namespace: str = "knowledge") -> str:
    return digest_json(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_kind": "knowledge",
            "namespace": _clean(namespace),
            "authority": _clean(authority),
            "stable_reference": _clean(stable_reference),
        }
    )


def representation_identity_for(
    *,
    knowledge_id: str,
    representation_type: str,
    source_ref: str,
    source_digest: str | None,
    upstream_revision: str | None = None,
    extra_authority: dict[str, Any] | None = None,
) -> str:
    return digest_json(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_kind": "representation",
            "knowledge_id": _clean(knowledge_id),
            "representation_type": normalize_representation_type(representation_type),
            "source_ref": _clean(source_ref),
            "source_digest": _clean(source_digest),
            "upstream_revision": _clean(upstream_revision),
            "extra_authority": extra_authority or {},
        }
    )


def document_identity_for(*, representation_id: str, canonical_content_digest: str) -> str:
    return digest_json(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_kind": "canonical_document",
            "representation_id": _clean(representation_id),
            "canonical_content_digest": _clean(canonical_content_digest),
        }
    )


def section_identity_for(*, document_id: str, path: tuple[str, ...], ordinal: int) -> str:
    return digest_json(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_kind": "canonical_section",
            "document_id": _clean(document_id),
            "path": list(path),
            "ordinal": ordinal,
        }
    )


def block_identity_for(
    *,
    document_id: str,
    section_id: str | None,
    block_type: str,
    ordinal: int,
    reading_order: int,
    text_digest: str | None,
) -> str:
    return digest_json(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_kind": "canonical_block",
            "document_id": _clean(document_id),
            "section_id": _clean(section_id),
            "block_type": normalize_representation_type(block_type),
            "ordinal": ordinal,
            "reading_order": reading_order,
            "text_digest": _clean(text_digest),
        }
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip()
