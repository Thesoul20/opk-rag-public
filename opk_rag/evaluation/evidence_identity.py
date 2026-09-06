from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION = "opk-rag.canonical-evidence-identity.v1"


class MatchLevel(str, Enum):
    EXACT_CHUNK = "exact_chunk_match"
    EXACT_SCOPE = "exact_scope_match"
    COMPATIBLE_SCOPE_CHUNK = "compatible_scope_chunk_match"
    EXACT_DOCUMENT = "exact_document_match"
    COMPATIBLE_DOCUMENT_CHUNK = "compatible_document_chunk_match"
    SOURCE_DIGEST = "source_digest_match"
    NO_MATCH = "no_match"
    UNVERIFIABLE = "unverifiable"


STABLE_FIELDS = (
    "knowledge_base_digest",
    "document_identity_digest",
    "source_digest",
    "relative_path_digest",
    "chunk_content_digest",
    "heading_path_digest",
    "scope_identity_digest",
)
VOLATILE_FIELDS = ("document_id", "chunk_id", "scope_id")


@dataclass(frozen=True)
class EvidenceIdentity:
    knowledge_base_digest: str | None = None
    source_digest: str | None = None
    document_identity_digest: str | None = None
    relative_path_digest: str | None = None
    chunk_content_digest: str | None = None
    heading_path_digest: str | None = None
    scope_identity_digest: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    scope_id: str | None = None
    citation_id: str | None = None
    granularity: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class CanonicalEvidenceIdentity:
    knowledge_base_digest: str | None = None
    source_digest: str | None = None
    document_identity_digest: str | None = None
    relative_path_digest: str | None = None
    chunk_content_digest: str | None = None
    heading_path_digest: str | None = None
    scope_identity_digest: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None

    @property
    def schema_version(self) -> str:
        return CANONICAL_EVIDENCE_IDENTITY_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version}
        payload.update({key: value for key, value in asdict(self).items() if value is not None})
        return payload


@dataclass(frozen=True)
class IdentityMatch:
    matched: bool
    level: MatchLevel
    used_fields: tuple[str, ...] = ()
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "level": self.level.value,
            "used_fields": list(self.used_fields),
            "reason": self.reason,
        }


def normalize_gold_evidence_identity(value: Any, *, default_knowledge_base_digest: str | None = None) -> EvidenceIdentity:
    payload = _coerce_payload(value)
    return EvidenceIdentity(
        knowledge_base_digest=_text(payload.get("knowledge_base_digest") or default_knowledge_base_digest),
        document_id=_text(payload.get("document_id")),
        document_identity_digest=_text(payload.get("document_identity_digest") or payload.get("document_identity_digest".removesuffix("s")) or _first(payload.get("document_identity_digests"))),
        source_digest=_text(payload.get("source_digest") or _first(payload.get("source_digests"))),
        relative_path_digest=_text(payload.get("relative_path_digest")),
        chunk_id=_text(payload.get("chunk_id")),
        chunk_content_digest=_text(payload.get("chunk_content_digest") or payload.get("content_sha256") or _first(payload.get("chunk_content_digests"))),
        heading_path_digest=_text(payload.get("heading_path_digest")),
        scope_identity_digest=_text(payload.get("scope_identity_digest")),
        scope_id=_text(payload.get("scope_id") or _first(payload.get("scope_ids"))),
        citation_id=_text(payload.get("citation_id")),
        granularity=_text(payload.get("granularity")) or _infer_granularity(payload),
    )


def normalize_runtime_evidence_identity(value: Any, *, default_knowledge_base_digest: str | None = None) -> EvidenceIdentity:
    payload = _coerce_payload(value)
    return EvidenceIdentity(
        knowledge_base_digest=_text(payload.get("knowledge_base_digest") or default_knowledge_base_digest),
        document_id=_text(payload.get("document_id")),
        document_identity_digest=_text(payload.get("document_identity_digest")),
        source_digest=_text(payload.get("source_digest")),
        relative_path_digest=_text(payload.get("relative_path_digest")),
        chunk_id=_text(payload.get("chunk_id")),
        chunk_content_digest=_text(payload.get("chunk_content_digest") or payload.get("content_sha256") or payload.get("content_hash")),
        heading_path_digest=_text(payload.get("heading_path_digest")),
        scope_identity_digest=_text(payload.get("scope_identity_digest")),
        scope_id=_text(payload.get("scope_id")),
        citation_id=_text(payload.get("citation_id")),
        granularity=_text(payload.get("granularity")) or _infer_granularity(payload),
    )


def canonical_identity_from_payload(value: Any, *, default_knowledge_base_digest: str | None = None) -> CanonicalEvidenceIdentity:
    identity = normalize_runtime_evidence_identity(value, default_knowledge_base_digest=default_knowledge_base_digest)
    return CanonicalEvidenceIdentity(
        knowledge_base_digest=identity.knowledge_base_digest,
        source_digest=identity.source_digest,
        document_identity_digest=identity.document_identity_digest,
        relative_path_digest=identity.relative_path_digest,
        chunk_content_digest=identity.chunk_content_digest,
        heading_path_digest=identity.heading_path_digest,
        scope_identity_digest=identity.scope_identity_digest,
        document_id=identity.document_id,
        chunk_id=identity.chunk_id,
    )


def identity_from_runtime_evidence_item(item: Any, *, knowledge_base_id: Any | None = None) -> CanonicalEvidenceIdentity:
    payload = _coerce_payload(item)
    relative_path = _text(payload.get("relative_path"))
    heading_path = payload.get("heading_path")
    if isinstance(heading_path, str):
        heading_values = tuple(part.strip() for part in heading_path.split(">") if part.strip())
    elif isinstance(heading_path, (list, tuple)):
        heading_values = tuple(str(part).strip() for part in heading_path if str(part).strip())
    else:
        heading_values = ()
    content = payload.get("content")
    source_digest = _text(payload.get("source_digest") or payload.get("source_content_digest"))
    chunk_digest = _text(payload.get("chunk_content_digest") or payload.get("content_sha256") or payload.get("content_hash"))
    relative_path_digest = _text(payload.get("relative_path_digest")) or (digest_text(_normalize_relative_path(relative_path)) if relative_path else None)
    heading_path_digest = _text(payload.get("heading_path_digest")) or (digest_json([heading_values[-1]]) if heading_values else None)
    if chunk_digest is None and isinstance(content, str):
        chunk_digest = digest_text(content)
    document_identity_digest = _text(payload.get("document_identity_digest"))
    if document_identity_digest is None and relative_path_digest:
        document_identity_digest = digest_json({"relative_path_digest": relative_path_digest})
    scope_identity_digest = _text(payload.get("scope_identity_digest"))
    if scope_identity_digest is None and any((source_digest, relative_path_digest, heading_path_digest, chunk_digest)):
        scope_identity_digest = digest_json(
            {
                "source_digest": source_digest,
                "relative_path_digest": relative_path_digest,
                "heading_path_digest": heading_path_digest,
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
                "chunk_content_digest": chunk_digest,
            }
        )
    return CanonicalEvidenceIdentity(
        knowledge_base_digest=_text(payload.get("knowledge_base_digest")) or (digest_text(str(knowledge_base_id)) if knowledge_base_id is not None else None),
        source_digest=source_digest,
        document_identity_digest=document_identity_digest,
        relative_path_digest=relative_path_digest,
        chunk_content_digest=chunk_digest,
        heading_path_digest=heading_path_digest,
        scope_identity_digest=scope_identity_digest,
        document_id=_text(payload.get("document_id")),
        chunk_id=_text(payload.get("chunk_id")),
    )


def identity_from_gold_annotation(item: Any, *, knowledge_base_id: Any | None = None) -> CanonicalEvidenceIdentity:
    return identity_from_runtime_evidence_item(item, knowledge_base_id=knowledge_base_id)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    return digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def validate_evidence_identity(identity: EvidenceIdentity) -> dict[str, Any]:
    stable = [field for field in STABLE_FIELDS if getattr(identity, field)]
    volatile = [field for field in VOLATILE_FIELDS if getattr(identity, field)]
    if stable:
        return {"status": "verifiable", "stable_fields": stable, "volatile_fields": volatile}
    if volatile:
        return {"status": "volatile_only", "stable_fields": stable, "volatile_fields": volatile}
    if identity.citation_id:
        return {"status": "unverifiable", "stable_fields": [], "volatile_fields": [], "reason": "citation_id_is_request_local"}
    return {"status": "unverifiable", "stable_fields": [], "volatile_fields": [], "reason": "missing_identity_fields"}


def match_evidence_identity(gold: EvidenceIdentity, runtime: EvidenceIdentity) -> IdentityMatch:
    if gold.knowledge_base_digest and runtime.knowledge_base_digest and gold.knowledge_base_digest != runtime.knowledge_base_digest:
        return IdentityMatch(False, MatchLevel.NO_MATCH, ("knowledge_base_digest",), "knowledge base digest differs")
    if gold.chunk_id and runtime.chunk_id and gold.chunk_id == runtime.chunk_id:
        return IdentityMatch(True, MatchLevel.EXACT_CHUNK, ("chunk_id",), "chunk ids match")
    if gold.chunk_content_digest and runtime.chunk_content_digest and gold.chunk_content_digest == runtime.chunk_content_digest:
        return IdentityMatch(True, MatchLevel.EXACT_CHUNK, ("chunk_content_digest",), "chunk content digests match")
    if gold.scope_identity_digest and runtime.scope_identity_digest and gold.scope_identity_digest == runtime.scope_identity_digest:
        return IdentityMatch(True, MatchLevel.EXACT_SCOPE, ("scope_identity_digest",), "scope identity digests match")
    if gold.scope_id and runtime.scope_id and gold.scope_id == runtime.scope_id:
        return IdentityMatch(True, MatchLevel.EXACT_SCOPE, ("scope_id",), "scope ids match")
    if gold.granularity == "scope" and not (gold.scope_id or gold.scope_identity_digest or gold.heading_path_digest):
        return IdentityMatch(False, MatchLevel.UNVERIFIABLE, (), "scope gold lacks scope_id, scope_identity_digest, or heading_path_digest")
    if (gold.scope_id or gold.scope_identity_digest or gold.heading_path_digest) and _same_document(gold, runtime) and _same_heading(gold, runtime):
        return IdentityMatch(True, MatchLevel.COMPATIBLE_SCOPE_CHUNK, _document_fields(gold, runtime) + ("heading_path_digest",), "scope-compatible chunk matched document and heading digest")
    if gold.granularity == "scope" or gold.scope_id:
        if validate_evidence_identity(runtime)["status"] == "unverifiable":
            return IdentityMatch(False, MatchLevel.UNVERIFIABLE, (), "runtime identity is unverifiable")
        return IdentityMatch(False, MatchLevel.NO_MATCH, (), "scope gold did not match exact scope or compatible heading chunk")
    if _same_document(gold, runtime):
        return IdentityMatch(True, MatchLevel.EXACT_DOCUMENT, _document_fields(gold, runtime), "document identity matches")
    if (gold.granularity == "document" or gold.document_identity_digest or gold.document_id) and runtime.chunk_id and _same_document(gold, runtime):
        return IdentityMatch(True, MatchLevel.COMPATIBLE_DOCUMENT_CHUNK, _document_fields(gold, runtime), "document-level gold covered by chunk from the same document")
    if gold.source_digest and runtime.source_digest and gold.source_digest == runtime.source_digest:
        return IdentityMatch(True, MatchLevel.SOURCE_DIGEST, ("source_digest",), "source digest matches")
    if validate_evidence_identity(gold)["status"] == "unverifiable":
        return IdentityMatch(False, MatchLevel.UNVERIFIABLE, (), "gold identity is unverifiable")
    if validate_evidence_identity(runtime)["status"] == "unverifiable":
        return IdentityMatch(False, MatchLevel.UNVERIFIABLE, (), "runtime identity is unverifiable")
    return IdentityMatch(False, MatchLevel.NO_MATCH, (), "no compatible identity fields matched")


def match_required_evidence_set(required: Iterable[EvidenceIdentity], runtime: Iterable[EvidenceIdentity]) -> dict[str, Any]:
    gold_units = deduplicate_evidence_identities(required)
    runtime_units = deduplicate_evidence_identities(runtime)
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    for gold in gold_units:
        if _coverage_unverifiable(gold):
            unverifiable.append({"gold": gold.to_json(), "reason": "gold_unverifiable"})
            continue
        best = _best_match(gold, runtime_units)
        if best and best[1].matched:
            matched.append({"gold": gold.to_json(), "runtime": best[0].to_json(), "match": best[1].to_json()})
        elif best and best[1].level == MatchLevel.UNVERIFIABLE:
            unmatched.append({"gold": gold.to_json(), "match": best[1].to_json()})
        else:
            unmatched.append({"gold": gold.to_json(), "match": IdentityMatch(False, MatchLevel.NO_MATCH, (), "no runtime evidence matched").to_json()})
    denominator = len(gold_units) - len(unverifiable)
    return {
        "required_total": len(gold_units),
        "verifiable_required_total": denominator,
        "matched_count": len(matched),
        "unverifiable_required_units": len(unverifiable),
        "coverage": (len(matched) / denominator) if denominator else None,
        "matched": matched,
        "unmatched": unmatched,
        "unverifiable": unverifiable,
    }


def deduplicate_evidence_identities(values: Iterable[EvidenceIdentity]) -> tuple[EvidenceIdentity, ...]:
    seen: set[tuple[Any, ...]] = set()
    out: list[EvidenceIdentity] = []
    for identity in values:
        key = _identity_key(identity)
        if key in seen:
            continue
        seen.add(key)
        out.append(identity)
    return tuple(out)


def explain_identity_mismatch(gold: EvidenceIdentity, runtime: EvidenceIdentity) -> dict[str, Any]:
    match = match_evidence_identity(gold, runtime)
    return {"gold": gold.to_json(), "runtime": runtime.to_json(), "match": match.to_json()}


def _best_match(gold: EvidenceIdentity, runtime_units: tuple[EvidenceIdentity, ...]) -> tuple[EvidenceIdentity, IdentityMatch] | None:
    best: tuple[EvidenceIdentity, IdentityMatch] | None = None
    order = {level: index for index, level in enumerate(MatchLevel)}
    for runtime in runtime_units:
        current = match_evidence_identity(gold, runtime)
        if best is None or order[current.level] < order[best[1].level]:
            best = (runtime, current)
    return best


def _coverage_unverifiable(identity: EvidenceIdentity) -> bool:
    if validate_evidence_identity(identity)["status"] == "unverifiable":
        return True
    if identity.granularity == "scope" and not (identity.scope_id or identity.scope_identity_digest or identity.heading_path_digest or identity.chunk_id or identity.chunk_content_digest):
        return True
    return False


def _same_document(gold: EvidenceIdentity, runtime: EvidenceIdentity) -> bool:
    for field in ("document_identity_digest", "document_id", "relative_path_digest"):
        left = getattr(gold, field)
        right = getattr(runtime, field)
        if left and right and left == right:
            return True
    return False


def _document_fields(gold: EvidenceIdentity, runtime: EvidenceIdentity) -> tuple[str, ...]:
    return tuple(field for field in ("document_identity_digest", "document_id", "relative_path_digest") if getattr(gold, field) and getattr(runtime, field) and getattr(gold, field) == getattr(runtime, field))


def _same_heading(gold: EvidenceIdentity, runtime: EvidenceIdentity) -> bool:
    return bool(gold.heading_path_digest and runtime.heading_path_digest and gold.heading_path_digest == runtime.heading_path_digest)


def _identity_key(identity: EvidenceIdentity) -> tuple[Any, ...]:
    data = identity.to_json()
    data.pop("citation_id", None)
    return tuple(sorted(data.items()))


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, EvidenceIdentity):
        return value.to_json()
    if isinstance(value, CanonicalEvidenceIdentity):
        return value.to_json()
    if isinstance(value, dict):
        return value
    return {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _infer_granularity(payload: dict[str, Any]) -> str | None:
    if payload.get("scope_id") or payload.get("scope_ids") or payload.get("scope_identity_digest"):
        return "scope"
    if payload.get("chunk_id") or payload.get("chunk_content_digest") or payload.get("content_sha256"):
        return "chunk"
    if payload.get("document_id") or payload.get("document_identity_digest") or payload.get("relative_path_digest") or payload.get("source_digest"):
        return "document"
    return None


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return None


def _normalize_relative_path(value: str | None) -> str:
    text = unicodedata.normalize("NFC", str(value or "").strip()).replace("\\", "/")
    if text.startswith("source-documents/"):
        text = text[len("source-documents/") :]
    while text.startswith("./"):
        text = text[2:]
    return text
