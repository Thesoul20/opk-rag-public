from opk_rag.document_ir.digest import canonical_content_digest, source_bytes_digest
from opk_rag.document_ir.identity import (
    block_identity_for,
    document_identity_for,
    knowledge_identity_for,
    representation_identity_for,
)
from opk_rag.document_ir.models import (
    CANONICAL_DOCUMENT_IR_SCHEMA_VERSION,
    BoundingBox,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalSection,
    CanonicalTable,
    CanonicalTableCell,
    RepresentationMetadata,
    SourceLocation,
)
from opk_rag.document_ir.serialization import canonical_json, document_from_dict, document_to_dict
from opk_rag.document_ir.validation import DocumentIRValidationError, validate_document

__all__ = [
    "CANONICAL_DOCUMENT_IR_SCHEMA_VERSION",
    "BoundingBox",
    "CanonicalBlock",
    "CanonicalDocument",
    "CanonicalFigure",
    "CanonicalSection",
    "CanonicalTable",
    "CanonicalTableCell",
    "DocumentIRValidationError",
    "RepresentationMetadata",
    "SourceLocation",
    "block_identity_for",
    "canonical_content_digest",
    "canonical_json",
    "document_from_dict",
    "document_identity_for",
    "document_to_dict",
    "knowledge_identity_for",
    "representation_identity_for",
    "source_bytes_digest",
    "validate_document",
]
