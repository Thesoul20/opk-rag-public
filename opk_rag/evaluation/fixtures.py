from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opk_rag.evaluation.retrieval import RETRIEVAL_FIXTURE_SCHEMA_VERSION


class RetrievalFixtureError(ValueError):
    pass


def fixture_row(*, sample: dict[str, Any], retrieval_result: dict[str, Any], run_id: str, retrieval_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_schema_version": RETRIEVAL_FIXTURE_SCHEMA_VERSION,
        "run_id": run_id,
        "sample_id": sample["id"],
        "question": sample["question"],
        "split": sample.get("split"),
        "retrieval_configuration": retrieval_config,
        "embedding_model_identifier": retrieval_config.get("embedding_model_id"),
        "candidates": retrieval_result["candidates"],
        "evidence_bundle": retrieval_result["evidence_bundle"],
        "gold_evidence": retrieval_result["gold_evidence"],
        "gold_match_result": {
            "retrieval_evaluable": retrieval_result["retrieval_evaluable"],
            "candidate_hit": retrieval_result["candidate_hit"],
            "evidence_bundle_hit": retrieval_result["evidence_bundle_hit"],
            "first_relevant_rank": retrieval_result["first_relevant_rank"],
            "candidate_gold_matches": retrieval_result["candidate_gold_matches"],
            "evidence_bundle_gold_matches": retrieval_result["evidence_bundle_gold_matches"],
        },
    }


def write_fixture(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise RetrievalFixtureError(f"Refusing to overwrite existing fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_fixture(path: Path, *, expected_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("fixture_schema_version") != RETRIEVAL_FIXTURE_SCHEMA_VERSION:
            raise RetrievalFixtureError(f"Unsupported retrieval fixture schema version for sample {row.get('sample_id')}: {row.get('fixture_schema_version')}")
        if expected_config is not None:
            actual = row.get("retrieval_configuration") or {}
            mismatches = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected_config.items()
                if actual.get(key) != value
            }
            if mismatches:
                raise RetrievalFixtureError(f"Retrieval fixture configuration mismatch for sample {row.get('sample_id')}: {mismatches}")
    return rows

