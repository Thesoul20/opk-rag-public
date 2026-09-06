from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from opk_rag.runtime_v2.chunk_identity import canonical_chunk_id_for, digest_json, normalized_path
from opk_rag.runtime_v2.models import CanonicalChunkV2, CorpusSnapshotV2


CHUNKING_CONTRACT_VERSION = "TASK-0081-C0-source-span-localization"
EMBEDDING_CONTRACT_VERSION = "TASK-0082-frozen-qwen-vector-rank-trace"


def build_corpus_snapshot_v2(chunk_inventory: dict[str, Any]) -> CorpusSnapshotV2:
    provenance = list(chunk_inventory.get("provenance") or [])
    source_manifest = {
        "document_count": chunk_inventory.get("document_count"),
        "section_count": chunk_inventory.get("section_count"),
        "chunk_count": chunk_inventory.get("chunk_count"),
        "source_paths": sorted({normalized_path(row.get("source_path")) for row in provenance if row.get("source_path")}),
    }
    corpus_digest = digest_json(
        [
            {
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "section_id": row.get("section_id"),
                "source_path": normalized_path(row.get("source_path")),
                "heading_path": list(row.get("heading_path") or []),
                "start_offset": row.get("start_offset"),
                "end_offset": row.get("end_offset"),
            }
            for row in provenance
        ]
    )
    snapshot_id = digest_json(
        {
            "architecture": "canonical_retrieval_runtime_v2",
            "corpus_digest": corpus_digest,
            "chunking_contract_version": CHUNKING_CONTRACT_VERSION,
            "embedding_contract_version": EMBEDDING_CONTRACT_VERSION,
        }
    )
    return CorpusSnapshotV2(
        corpus_snapshot_id=snapshot_id,
        corpus_digest=corpus_digest,
        source_manifest=source_manifest,
        chunking_contract_version=CHUNKING_CONTRACT_VERSION,
        embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
    )


def build_canonical_chunks_v2(chunk_inventory: dict[str, Any], snapshot: CorpusSnapshotV2) -> tuple[CanonicalChunkV2, ...]:
    chunks = []
    for row in chunk_inventory.get("provenance") or []:
        source_path = normalized_path(row.get("source_path"))
        content_digest = str(row.get("chunk_id"))
        canonical_id = canonical_chunk_id_for(
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            normalized_source_path=source_path,
            source_start=_int_or_none(row.get("start_offset")),
            source_end=_int_or_none(row.get("end_offset")),
            content_digest=content_digest,
        )
        chunks.append(
            CanonicalChunkV2(
                canonical_chunk_id=canonical_id,
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                document_id=_text(row.get("document_id")),
                section_id=_text(row.get("section_id")),
                normalized_source_path=source_path,
                heading_path=tuple(str(item) for item in (row.get("heading_path") or ())),
                source_start=_int_or_none(row.get("start_offset")),
                source_end=_int_or_none(row.get("end_offset")),
                content="",
                content_digest=content_digest,
                token_count=_int_or_none(row.get("token_count")),
            )
        )
    return tuple(chunks)


def chunk_by_frozen_id(chunks: Iterable[CanonicalChunkV2]) -> dict[str, CanonicalChunkV2]:
    return {chunk.content_digest: chunk for chunk in chunks}


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
