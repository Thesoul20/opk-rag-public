from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from opk_rag.document_adapters.errors import EmptyAdapterOutputError
from opk_rag.document_ir import (
    CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalSection,
    CanonicalTable,
    RepresentationMetadata,
    canonical_content_digest,
    document_identity_for,
    source_bytes_digest,
    validate_document,
)
from opk_rag.document_ir.serialization import digest_json, normalize_representation_type


@dataclass(frozen=True)
class StructuredDocumentSource:
    source_bytes: bytes
    knowledge_id: str
    representation_id: str
    representation_type: str
    source_ref: str
    mime_type: str | None = None
    source_name: str | None = None
    source_relative_path: str | None = None
    upstream_revision: str | None = None
    authority: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return self.source_bytes.decode("utf-8")


class StructuredDocumentAdapter(Protocol):
    adapter_name: str
    adapter_version: str
    representation_types: tuple[str, ...]

    def adapt(self, source: StructuredDocumentSource) -> CanonicalDocument:
        ...


@dataclass(frozen=True)
class ParsedDocumentParts:
    title: str | None
    sections: tuple[CanonicalSection, ...]
    blocks: tuple[CanonicalBlock, ...]
    tables: tuple[CanonicalTable, ...] = ()
    figures: tuple[CanonicalFigure, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def adapter_config_digest(config: dict[str, Any] | None = None) -> str:
    return digest_json(config or {"config_version": 1})


def build_canonical_document(
    *,
    source: StructuredDocumentSource,
    adapter_name: str,
    adapter_version: str,
    parts: ParsedDocumentParts,
    adapter_config: dict[str, Any] | None = None,
) -> CanonicalDocument:
    if source.source_bytes and not parts.blocks:
        raise EmptyAdapterOutputError("non-empty source produced no canonical blocks")

    representation_type = normalize_representation_type(source.representation_type)
    source_digest = source_bytes_digest(source.source_bytes)
    config_digest = adapter_config_digest(adapter_config)
    source_ref = source.source_ref
    metadata = {
        **parts.metadata,
        "structured_source_reference_arm": True,
    }
    representation_metadata = RepresentationMetadata(
        format=representation_type,
        mime_type=source.mime_type,
        source_name=source.source_name or source_ref.rsplit("/", 1)[-1],
        source_relative_path=source.source_relative_path or source_ref,
        source_digest=source_digest,
        parser_name=adapter_name,
        parser_version=adapter_version,
        parser_config_digest=config_digest,
        authority=dict(source.authority),
        upstream_revision=source.upstream_revision,
        semantic_metadata={
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "adapter_config_digest": config_digest,
            "source_digest": source_digest,
            "representation_type": representation_type,
        },
        operational_metadata={},
    )
    document = CanonicalDocument(
        schema_version=CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
        document_id="pending-document-id",
        knowledge_id=source.knowledge_id,
        representation_id=source.representation_id,
        representation_type=representation_type,
        title=parts.title,
        metadata=metadata,
        representation_metadata=representation_metadata,
        sections=parts.sections,
        blocks=parts.blocks,
        tables=parts.tables,
        figures=parts.figures,
        source={"source_ref": source_ref, "source_digest": source_digest},
        provenance={
            "knowledge_id": source.knowledge_id,
            "representation_id": source.representation_id,
            "representation_type": representation_type,
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "adapter_config_digest": config_digest,
            "source_digest": source_digest,
        },
    )
    digest = canonical_content_digest(document)
    document = replace(
        document,
        document_id=document_identity_for(
            representation_id=source.representation_id,
            canonical_content_digest=digest,
        ),
        content_digest=digest,
    )
    validate_document(document)
    return document


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{digest_json([str(part) for part in parts])[:24]}"
