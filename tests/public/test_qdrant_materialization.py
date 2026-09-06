from __future__ import annotations

from uuid import UUID

import pytest

from opk_rag.vector_backends.materialization import QdrantMaterializationResult, rows_to_points
from opk_rag.vector_backends.qdrant_backend import deterministic_qdrant_point_id


def test_rows_to_points_preserves_qdrant_relational_identity() -> None:
    kb_id = UUID("11111111-1111-4111-8111-111111111111")
    rows = [
        (
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            0,
            "public/demo.md",
            ["Title", "Section"],
            "[1.0,0.0]",
        )
    ]
    points = rows_to_points(
        rows,
        knowledge_base_id=kb_id,
        embedding_revision="fingerprint",
        vector_size=2,
    )
    assert len(points) == 1
    assert points[0].point_id == deterministic_qdrant_point_id(points[0].chunk_id)
    assert points[0].payload["knowledge_base_id"] == str(kb_id)
    assert points[0].payload["source_path"] == "public/demo.md"
    assert points[0].payload["section"] == "Title > Section"


def test_rows_to_points_fails_closed_on_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="embedding dimension"):
        rows_to_points(
            [
                (
                    "22222222-2222-4222-8222-222222222222",
                    "33333333-3333-4333-8333-333333333333",
                    0,
                    "public/demo.md",
                    [],
                    "[1.0,0.0]",
                )
            ],
            knowledge_base_id=UUID("11111111-1111-4111-8111-111111111111"),
            embedding_revision="fingerprint",
            vector_size=3,
        )


def test_materialization_result_requires_exact_point_count() -> None:
    assert QdrantMaterializationResult(4, 4, 0, 4, "chunks").valid is True
    assert QdrantMaterializationResult(4, 4, 0, 3, "chunks").valid is False
