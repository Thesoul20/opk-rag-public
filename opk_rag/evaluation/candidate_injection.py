from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CONTRACT_SCHEMA_VERSION = "opk-rag.task0088.candidate-injection-contract.v1"
CANDIDATE_SCHEMA_VERSION = "opk-rag.task0088.injected-candidate.v1"
DEFAULT_INJECTION_ENABLED = False


class CandidateInjectionValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class InjectedRetrievalCandidate:
    sample_id: str
    sample_unit_id: str
    query_text_digest: str | None
    retrieval_strategy_id: str
    candidate_rank: int
    chunk_id: str
    document_id: str | None
    section_id: str | None
    retrieval_score: float | None
    reranker_score: float | None
    provenance: dict[str, Any]
    source_span_digest: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "sample_unit_id": self.sample_unit_id,
            "query_text_digest": self.query_text_digest,
            "retrieval_strategy_id": self.retrieval_strategy_id,
            "candidate_rank": self.candidate_rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "retrieval_score": self.retrieval_score,
            "reranker_score": self.reranker_score,
            "provenance": self.provenance,
            "source_span_digest": self.source_span_digest,
        }


@dataclass(frozen=True)
class CandidateInjectionManifest:
    retrieval_strategy_id: str
    source_task_id: str
    source_artifact: str
    source_artifact_sha256: str
    top_k: int
    candidate_count: int
    sample_count: int
    deterministic: bool
    default_enabled: bool = DEFAULT_INJECTION_ENABLED

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "retrieval_strategy_id": self.retrieval_strategy_id,
            "source_task_id": self.source_task_id,
            "source_artifact": self.source_artifact,
            "source_artifact_sha256": self.source_artifact_sha256,
            "top_k": self.top_k,
            "candidate_count": self.candidate_count,
            "sample_count": self.sample_count,
            "deterministic": self.deterministic,
            "default_enabled": self.default_enabled,
            "gold_leakage": False,
        }


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateInjectionValidationError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise CandidateInjectionValidationError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def build_task0087_r2_injected_candidates(
    artifact_path: Path,
    *,
    query_by_task_sample_id: dict[str, str],
    top_k: int = 20,
) -> tuple[CandidateInjectionManifest, list[InjectedRetrievalCandidate]]:
    rows = read_jsonl(artifact_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rank = row.get("reranked_rank")
        if not isinstance(rank, int) or rank < 1 or rank > top_k:
            continue
        grouped[str(row.get("sample_id"))].append(row)

    candidates: list[InjectedRetrievalCandidate] = []
    for sample_id, sample_rows in sorted(grouped.items()):
        query = query_by_task_sample_id.get(sample_id)
        query_hash = text_digest(query) if query is not None else None
        for row in sorted(sample_rows, key=lambda item: (item["reranked_rank"], str(item.get("chunk_id")))):
            candidates.append(
                InjectedRetrievalCandidate(
                    sample_id=sample_id,
                    sample_unit_id=str(row.get("sample_unit_id") or ""),
                    query_text_digest=query_hash,
                    retrieval_strategy_id="task0087-r2-vector50-bge-reranked-top20",
                    candidate_rank=int(row["reranked_rank"]),
                    chunk_id=str(row.get("chunk_id") or ""),
                    document_id=str(row.get("document_id")) if row.get("document_id") is not None else None,
                    section_id=str(row.get("section_id")) if row.get("section_id") is not None else None,
                    retrieval_score=float(row["vector_score"]) if isinstance(row.get("vector_score"), (int, float)) else None,
                    reranker_score=float(row["reranker_raw_score"]) if isinstance(row.get("reranker_raw_score"), (int, float)) else None,
                    provenance={
                        "source_task_id": "TASK-0087",
                        "source_artifact": artifact_path.as_posix(),
                        "source_schema_version": row.get("schema_version"),
                        "original_vector_rank": row.get("original_vector_rank"),
                        "reranker_model": "BAAI/bge-reranker-v2-m3",
                    },
                    source_span_digest=_source_span_from_unit(str(row.get("sample_unit_id") or "")),
                )
            )

    manifest = CandidateInjectionManifest(
        retrieval_strategy_id="task0087-r2-vector50-bge-reranked-top20",
        source_task_id="TASK-0087",
        source_artifact=_display_path(artifact_path),
        source_artifact_sha256=file_sha256(artifact_path),
        top_k=top_k,
        candidate_count=len(candidates),
        sample_count=len(grouped),
        deterministic=True,
    )
    return manifest, candidates


def validate_injected_candidates(
    candidates: Iterable[InjectedRetrievalCandidate],
    *,
    expected_sample_ids: set[str],
    chunk_inventory_ids: set[str],
    expected_query_digest_by_sample_id: dict[str, str] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    rows = list(candidates)
    issues: list[dict[str, Any]] = []
    by_sample: dict[str, list[InjectedRetrievalCandidate]] = defaultdict(list)
    for row in rows:
        by_sample[row.sample_id].append(row)
        if row.sample_id not in expected_sample_ids:
            issues.append({"code": "sample_identity_mismatch", "sample_id": row.sample_id})
        if not row.chunk_id or row.chunk_id not in chunk_inventory_ids:
            issues.append({"code": "unknown_chunk", "sample_id": row.sample_id, "chunk_id": row.chunk_id})
        if not row.provenance or not row.provenance.get("source_task_id"):
            issues.append({"code": "invalid_provenance", "sample_id": row.sample_id, "chunk_id": row.chunk_id})
        if row.candidate_rank < 1 or row.candidate_rank > top_k:
            issues.append({"code": "rank_out_of_range", "sample_id": row.sample_id, "rank": row.candidate_rank})
        if expected_query_digest_by_sample_id and row.query_text_digest != expected_query_digest_by_sample_id.get(row.sample_id):
            issues.append({"code": "query_identity_mismatch", "sample_id": row.sample_id})

    missing_samples = sorted(expected_sample_ids - set(by_sample))
    issues.extend({"code": "sample_missing", "sample_id": sample_id} for sample_id in missing_samples)

    for sample_id, sample_rows in sorted(by_sample.items()):
        rank_counts = Counter(row.candidate_rank for row in sample_rows)
        chunk_counts = Counter(row.chunk_id for row in sample_rows)
        for rank, count in sorted(rank_counts.items()):
            if count > 1:
                issues.append({"code": "duplicate_rank", "sample_id": sample_id, "rank": rank})
        for chunk_id, count in sorted(chunk_counts.items()):
            if count > 1:
                issues.append({"code": "duplicate_chunk", "sample_id": sample_id, "chunk_id": chunk_id})
        if len(sample_rows) > top_k:
            issues.append({"code": "top_k_count_exceeded", "sample_id": sample_id, "count": len(sample_rows), "top_k": top_k})

    return {
        "schema_version": "opk-rag.task0088.candidate-validation.v1",
        "candidate_injection_status": "valid" if not issues else "invalid",
        "candidate_count": len(rows),
        "sample_count": len(by_sample),
        "expected_sample_count": len(expected_sample_ids),
        "top_k": top_k,
        "fail_closed": bool(issues),
        "issues": issues,
        "candidate_injection_default_enabled": DEFAULT_INJECTION_ENABLED,
        "candidate_injection_gold_leakage": False,
        "candidate_injection_deterministic": True,
    }


def _source_span_from_unit(sample_unit_id: str) -> str | None:
    if "::" not in sample_unit_id:
        return None
    return sample_unit_id.rsplit("::", 1)[-1] or None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
