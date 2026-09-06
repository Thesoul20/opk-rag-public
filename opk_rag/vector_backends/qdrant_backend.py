from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import uuid
from typing import Any

from opk_rag.vector_backends.base import (
    VectorBackendCandidate,
    VectorBackendPoint,
    VectorBackendSearchFilter,
    canonical_similarity_score,
)

QDRANT_POINT_NAMESPACE = uuid.UUID("7c0d2a91-4b3e-5a69-a901-019600000196")
DEFAULT_QDRANT_COLLECTION = "opk_rag_chunks_task0196"


@dataclass(frozen=True)
class QdrantBackendConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = DEFAULT_QDRANT_COLLECTION
    vector_size: int = 1024
    distance: str = "Cosine"
    prefer_grpc: bool = False
    timeout_seconds: float = 10.0
    trust_env: bool = False


def load_qdrant_config(env: Mapping[str, str] = os.environ) -> QdrantBackendConfig:
    return QdrantBackendConfig(
        url=env.get("OPK_RAG_QDRANT_URL", "http://localhost:6333").strip(),
        api_key=env.get("OPK_RAG_QDRANT_API_KEY", "").strip() or None,
        collection=env.get("OPK_RAG_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip() or DEFAULT_QDRANT_COLLECTION,
        prefer_grpc=_bool_env(env.get("OPK_RAG_QDRANT_PREFER_GRPC", "false")),
        trust_env=_bool_env(env.get("OPK_RAG_QDRANT_TRUST_ENV", "false")),
    )


def deterministic_qdrant_point_id(canonical_chunk_id: str) -> str:
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, str(canonical_chunk_id)))


class QdrantVectorBackend:
    backend_id = "qdrant"

    def __init__(self, config: QdrantBackendConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("qdrant-client is required for Qdrant runtime experiments.") from exc
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
                timeout=self.config.timeout_seconds,
                trust_env=self.config.trust_env,
            )
        return self._client

    def health_check(self) -> Mapping[str, object]:
        try:
            collections = self.client.get_collections()
            return {
                "backend": self.backend_id,
                "qdrant_server_reachable": True,
                "python_client_connectivity": True,
                "collection_api_connectivity": True,
                "collection_count": len(getattr(collections, "collections", []) or []),
                "primary_transport": "grpc" if self.config.prefer_grpc else "rest",
            }
        except Exception as exc:
            return {
                "backend": self.backend_id,
                "qdrant_server_reachable": False,
                "python_client_connectivity": False,
                "collection_api_connectivity": False,
                "primary_transport": "grpc" if self.config.prefer_grpc else "rest",
                "error": type(exc).__name__ + ": " + str(exc),
            }

    def initialize(self) -> None:
        self.create_collection(recreate=False)
        self.create_payload_indexes()

    def create_collection(self, *, recreate: bool = False) -> None:
        from qdrant_client import models

        if recreate and self.collection_exists():
            self.client.delete_collection(collection_name=self.config.collection)
        if self.collection_exists():
            return
        self.client.create_collection(
            collection_name=self.config.collection,
            vectors_config=models.VectorParams(size=self.config.vector_size, distance=models.Distance.COSINE),
        )

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(collection_name=self.config.collection))

    def create_payload_indexes(self) -> Mapping[str, object]:
        from qdrant_client import models

        created: list[str] = []
        for field_name in ("knowledge_base_id", "document_id", "chunk_id", "embedding_revision", "source_path"):
            self.client.create_payload_index(
                collection_name=self.config.collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            created.append(field_name)
        return {"payload_index_creation_valid": True, "indexed_fields": created}

    def upsert(self, points: Sequence[VectorBackendPoint], *, batch_size: int = 64) -> Mapping[str, object]:
        from qdrant_client import models

        batch_count = 0
        inserted = 0
        failed = 0
        for offset in range(0, len(points), batch_size):
            batch = points[offset : offset + batch_size]
            batch_count += 1
            try:
                self.client.upsert(
                    collection_name=self.config.collection,
                    points=[
                        models.PointStruct(id=point.point_id, vector=list(point.vector), payload=_payload_for_point(point))
                        for point in batch
                    ],
                    wait=True,
                )
                inserted += len(batch)
            except Exception:
                failed += len(batch)
                raise
        return {
            "batch_size": batch_size,
            "batch_count": batch_count,
            "inserted_point_count": inserted,
            "failed_point_count": failed,
        }

    def delete(self, point_ids: Sequence[str]) -> Mapping[str, object]:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.config.collection,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )
        return {"deleted_point_count": len(point_ids)}

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        search_filter: VectorBackendSearchFilter | None = None,
        exact: bool = False,
    ) -> tuple[VectorBackendCandidate, ...]:
        from qdrant_client import models

        query_filter = _to_qdrant_filter(search_filter)
        params = models.SearchParams(exact=True) if exact else None
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.config.collection,
                query=list(query_vector),
                limit=top_k,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
            )
            scored = getattr(response, "points", response)
        else:  # pragma: no cover - older qdrant-client compatibility
            scored = self.client.search(
                collection_name=self.config.collection,
                query_vector=list(query_vector),
                limit=top_k,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
            )
        candidates = []
        for rank, item in enumerate(scored, start=1):
            payload = dict(getattr(item, "payload", None) or {})
            raw_score = float(getattr(item, "score"))
            candidates.append(
                VectorBackendCandidate(
                    rank=rank,
                    point_id=str(getattr(item, "id")),
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload["document_id"]),
                    embedding_revision=str(payload["embedding_revision"]),
                    raw_backend_score=raw_score,
                    backend_score_semantics="qdrant_cosine_similarity",
                    canonical_similarity_score=canonical_similarity_score("qdrant_cosine_similarity", raw_score),
                    payload=payload,
                )
            )
        return tuple(candidates)

    def count(self) -> int:
        response = self.client.count(collection_name=self.config.collection, exact=True)
        return int(getattr(response, "count"))

    def create_snapshot(self) -> Mapping[str, object]:
        snapshot = self.client.create_snapshot(collection_name=self.config.collection, wait=True)
        return {"snapshot_create_valid": True, "snapshot_name": getattr(snapshot, "name", None)}

    def list_snapshots(self) -> Mapping[str, object]:
        snapshots = self.client.list_snapshots(collection_name=self.config.collection)
        return {"snapshot_list_valid": True, "snapshot_count": len(snapshots)}

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()


def _payload_for_point(point: VectorBackendPoint) -> dict[str, object]:
    payload = dict(point.payload)
    payload.update(
        {
            "chunk_id": point.chunk_id,
            "document_id": point.document_id,
            "embedding_revision": point.embedding_revision,
        }
    )
    return payload


def _to_qdrant_filter(search_filter: VectorBackendSearchFilter | None) -> Any:
    if search_filter is None:
        return None
    conditions = search_filter.as_payload_conditions()
    if not conditions:
        return None
    from qdrant_client import models

    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in conditions.items()]
    )


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")
