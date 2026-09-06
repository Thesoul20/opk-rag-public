from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from uuid import UUID

from opk_rag.db.connection import connect_postgres
from opk_rag.vector_backends.base import VectorBackendPoint
from opk_rag.vector_backends.qdrant_backend import (
    QdrantVectorBackend,
    deterministic_qdrant_point_id,
    load_qdrant_config,
)


@dataclass(frozen=True)
class QdrantMaterializationResult:
    source_point_count: int
    upserted_point_count: int
    deleted_stale_point_count: int
    final_knowledge_base_point_count: int
    collection: str

    @property
    def valid(self) -> bool:
        return self.final_knowledge_base_point_count == self.source_point_count


def materialize_knowledge_base_qdrant(
    database_url: str,
    *,
    knowledge_base_id: UUID,
    embedding_revision: str,
    vector_size: int,
    env: Mapping[str, str] = os.environ,
) -> QdrantMaterializationResult:
    """Reconcile one PostgreSQL knowledge base into its authoritative Qdrant collection."""
    points = load_materialization_points(
        database_url,
        knowledge_base_id=knowledge_base_id,
        embedding_revision=embedding_revision,
        vector_size=vector_size,
    )
    backend = QdrantVectorBackend(load_qdrant_config(env))
    try:
        backend.initialize()
        existing_ids = _knowledge_base_point_ids(backend, knowledge_base_id)
        upsert = backend.upsert(points, batch_size=64) if points else {"inserted_point_count": 0}
        expected_ids = {point.point_id for point in points}
        stale_ids = tuple(sorted(existing_ids - expected_ids))
        if stale_ids:
            backend.delete(stale_ids)
        final_ids = _knowledge_base_point_ids(backend, knowledge_base_id)
    finally:
        backend.close()

    result = QdrantMaterializationResult(
        source_point_count=len(points),
        upserted_point_count=int(upsert["inserted_point_count"]),
        deleted_stale_point_count=len(stale_ids),
        final_knowledge_base_point_count=len(final_ids),
        collection=backend.config.collection,
    )
    if not result.valid:
        raise RuntimeError(
            "Qdrant materialization count mismatch: "
            f"source={result.source_point_count}, final={result.final_knowledge_base_point_count}"
        )
    return result


def load_materialization_points(
    database_url: str,
    *,
    knowledge_base_id: UUID,
    embedding_revision: str,
    vector_size: int,
) -> tuple[VectorBackendPoint, ...]:
    with connect_postgres(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select
                  c.id::text,
                  c.document_id::text,
                  c.chunk_index,
                  d.relative_path,
                  c.heading_path,
                  c.embedding::text
                from public.chunks c
                join public.documents d on d.id = c.document_id
                join public.index_configurations ic on ic.id = c.index_configuration_id
                where d.knowledge_base_id = %s
                  and ic.configuration_fingerprint = %s
                  and c.embedding is not null
                  and d.index_status = 'indexed'
                order by d.relative_path, c.chunk_index, c.id
                """,
                (knowledge_base_id, embedding_revision),
            )
            rows = cursor.fetchall()
    return rows_to_points(
        rows,
        knowledge_base_id=knowledge_base_id,
        embedding_revision=embedding_revision,
        vector_size=vector_size,
    )


def rows_to_points(
    rows,
    *,
    knowledge_base_id: UUID,
    embedding_revision: str,
    vector_size: int,
) -> tuple[VectorBackendPoint, ...]:
    points: list[VectorBackendPoint] = []
    for chunk_id, document_id, chunk_index, relative_path, heading_path, embedding_literal in rows:
        vector = tuple(float(value) for value in json.loads(str(embedding_literal)))
        if len(vector) != vector_size:
            raise ValueError(
                f"Chunk {chunk_id} embedding dimension {len(vector)} does not match {vector_size}."
            )
        points.append(
            VectorBackendPoint(
                point_id=deterministic_qdrant_point_id(str(chunk_id)),
                chunk_id=str(chunk_id),
                document_id=str(document_id),
                embedding_revision=embedding_revision,
                vector=vector,
                payload={
                    "knowledge_base_id": str(knowledge_base_id),
                    "chunk_index": int(chunk_index),
                    "source_path": str(relative_path),
                    "section": " > ".join(heading_path or ()),
                },
            )
        )
    return tuple(points)


def _knowledge_base_point_ids(
    backend: QdrantVectorBackend,
    knowledge_base_id: UUID,
) -> set[str]:
    from qdrant_client import models

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=str(knowledge_base_id)),
            )
        ]
    )
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = backend.client.scroll(
            collection_name=backend.config.collection,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            break
    return ids
