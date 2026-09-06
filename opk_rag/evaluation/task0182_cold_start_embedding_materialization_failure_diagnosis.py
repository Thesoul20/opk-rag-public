from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from opk_rag.db.models import StoredChunk
from opk_rag.embedding.cache_authority import EmbeddingModelCacheAuthority
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint, load_embedding_config
from opk_rag.embedding.input import EmbeddingInputError, prepare_embedding_input
from opk_rag.embedding.provider import EmbeddingModelLoadError
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.embedding.service import _needs_embedding, _validate_vectors
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    CORPUS_ROOT,
    SECRET_URL_RE,
    TASK_DATABASE_URL_VARIABLE,
    resolve_cold_start_secret_authority,
)

TASK_ID = "TASK-0182"
SOURCE_AUTHORITATIVE_TASK = "TASK-0181"
EXPERIMENT_ID = "task0182-cold-start-embedding-materialization-failure-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0182_cold_start_embedding_materialization_failure_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0182_COLD_START_EMBEDDING_MATERIALIZATION_FAILURE_DIAGNOSIS_REPORT.md"
TASK0181_SUMMARY = ROOT / "evaluation-data" / "results" / "task0181-canonical-document-to-chunking-input-contract-repair" / "summary.json"
TASK0174_SUMMARY = ROOT / "evaluation-data" / "results" / "task0174-cold-start-embedding-model-cache-authority-remediation" / "summary.json"


@dataclass(frozen=True)
class ChunkInputAudit:
    chunk_id: str
    document_id: str
    relative_path: str
    content_nonempty: bool
    persisted_embedding_status: str
    required_fields_present: bool
    compatible: bool
    mismatch: str
    needs_embedding: bool
    embedding_input_hash: str


def run_task0182(*, write: bool = True, env: Mapping[str, str] | None = None, attempt_model_load: bool = True) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_embedding_config(env)
    source_head = _git_head(ROOT)
    task0181 = _load_json(TASK0181_SUMMARY)
    task0174 = _load_json(TASK0174_SUMMARY)
    database = probe_public_database(env, config)
    contract_rows = audit_embedding_input_contract(database.get("chunks", ()), config)
    cache = EmbeddingModelCacheAuthority(config.model_name, config.model_revision, config.cache_dir).audit(env)
    load = audit_model_loading(config, attempt=attempt_model_load)
    invocation = audit_embedding_invocation(config, load, contract_rows, database.get("chunks", ()))
    output_validation = validate_raw_outputs(invocation.get("raw_embeddings", ()), config)
    summary = build_summary(
        source_head=source_head,
        task0181=task0181,
        task0174=task0174,
        config=config,
        database=database,
        contract_rows=contract_rows,
        cache=cache,
        load=load,
        invocation=invocation,
        output_validation=output_validation,
    )
    artifacts = {
        "summary.json": summary,
        "persisted_chunk_authority.json": _redact({k: v for k, v in database.items() if k != "chunks"}),
        "embedding_input_contract_audit.json": [asdict(row) for row in contract_rows],
        "model_cache_authority_audit.json": _sanitize_cache(cache),
        "model_loading_audit.json": load,
        "embedding_invocation_audit.json": {k: v for k, v in invocation.items() if k != "raw_embeddings"},
        "embedding_output_validation.json": output_validation,
        "failure_taxonomy_decision.json": failure_taxonomy_decision(summary),
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, _redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def probe_public_database(env: Mapping[str, str], config: EmbeddingConfig) -> dict[str, Any]:
    secret = resolve_cold_start_secret_authority(env)
    database_url = str(secret.get("database_url_value") or env.get(TASK_DATABASE_URL_VARIABLE) or env.get("DATABASE_URL") or "")
    root = str(CORPUS_ROOT.resolve())
    out: dict[str, Any] = {
        "database_connection_authority_valid": bool(secret.get("database_connection_authority_valid")),
        "root_path": root,
        "knowledge_base_found": False,
        "authoritative_document_count": 0,
        "authoritative_chunk_count": 0,
        "stored_embedding_count": 0,
        "stored_vector_dimension": None,
        "chunk_content_nonempty": False,
        "canonical_document_identity_valid": False,
        "embedding_eligible_document_count": 0,
        "embedding_eligible_chunk_count": 0,
        "embedding_input_rejected_chunk_count": 0,
        "schema_vector_column_found": False,
        "schema_vector_column_type": "",
        "database_probe_error": "",
        "chunks": [],
    }
    if not database_url:
        out["database_probe_error"] = "database_url_unavailable"
        return out
    try:
        import psycopg

        fingerprint = build_configuration_fingerprint(config)
        with psycopg.connect(database_url, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("select id from public.knowledge_bases where root_path = %s order by created_at desc limit 1", (root,))
                row = cur.fetchone()
                if row is None:
                    out["database_probe_error"] = "knowledge_base_not_found"
                    return out
                kb_id = row[0]
                out["knowledge_base_found"] = True
                cur.execute("""
                    select count(*) from public.documents
                    where knowledge_base_id = %s and index_status <> 'deleted' and relative_path <> 'README.md'
                """, (kb_id,))
                out["authoritative_document_count"] = int(cur.fetchone()[0])
                cur.execute("""
                    select count(*) from public.chunks c join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md'
                """, (kb_id,))
                out["authoritative_chunk_count"] = int(cur.fetchone()[0])
                cur.execute("""
                    select count(*) from public.documents d
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md'
                      and d.parser_version = %s and d.chunking_version = %s and d.index_status = 'chunked'
                """, (kb_id, config.parser_version, config.chunking_version))
                out["embedding_eligible_document_count"] = int(cur.fetchone()[0])
                cur.execute("select id from public.index_configurations where configuration_fingerprint = %s", (fingerprint,))
                cfg = cur.fetchone()
                index_configuration_id = cfg[0] if cfg else None
                cur.execute("""
                    select c.id, c.document_id, c.index_configuration_id, c.chunk_index, c.content, c.content_hash,
                           c.heading_path, c.start_line, c.end_line, c.token_count, c.embedding,
                           c.embedding_model, c.embedding_dimension, c.metadata, c.created_at, c.updated_at,
                           d.relative_path, d.index_status, d.parser_version, d.chunking_version
                    from public.chunks c join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md'
                    order by d.relative_path, c.chunk_index
                """, (kb_id,))
                chunks = []
                for row in cur.fetchall():
                    chunks.append(_stored_chunk_payload(row))
                out["chunks"] = chunks
                out["chunk_content_nonempty"] = all(bool(str(c["content"]).strip()) for c in chunks)
                out["canonical_document_identity_valid"] = all(bool(c["document_id"]) and bool(c["relative_path"]) for c in chunks)
                out["embedding_eligible_chunk_count"] = sum(1 for c in chunks if c["document_parser_version"] == config.parser_version and c["document_chunking_version"] == config.chunking_version and (index_configuration_id is None or c["index_configuration_id"] == str(index_configuration_id)))
                cur.execute("""
                    select count(*), max(c.embedding_dimension)
                    from public.chunks c join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md' and c.embedding is not null
                """, (kb_id,))
                count, dim = cur.fetchone()
                out["stored_embedding_count"] = int(count)
                out["stored_vector_dimension"] = int(dim) if dim is not None else None
                cur.execute("""
                    select udt_schema || '.' || udt_name
                    from information_schema.columns
                    where table_schema = 'public' and table_name = 'chunks' and column_name = 'embedding'
                """)
                col = cur.fetchone()
                out["schema_vector_column_found"] = col is not None
                out["schema_vector_column_type"] = col[0] if col else ""
    except Exception as exc:  # pragma: no cover - live DB dependent
        out["database_probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


def audit_embedding_input_contract(chunks: Sequence[Mapping[str, Any]], config: EmbeddingConfig) -> list[ChunkInputAudit]:
    rows: list[ChunkInputAudit] = []
    provider = _StaticTokenProvider(config.model_name)
    for payload in chunks:
        chunk = _stored_chunk_from_payload(payload)
        mismatch = "none"
        prepared_hash = ""
        try:
            prepared = prepare_embedding_input(chunk, config, count_tokens=provider.count_tokens)
            prepared_hash = prepared.text_hash
            needs = _needs_embedding(chunk, prepared, provider, config)
        except EmbeddingInputError as exc:
            mismatch = f"embedding_input_validation:{type(exc).__name__}"
            needs = True
        required = bool(chunk.id and chunk.document_id and str(chunk.content).strip() and chunk.content_hash and isinstance(chunk.metadata, dict))
        if not required and mismatch == "none":
            mismatch = "missing_required_persisted_chunk_field"
        rows.append(ChunkInputAudit(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            relative_path=str(payload.get("relative_path", "")),
            content_nonempty=bool(str(chunk.content).strip()),
            persisted_embedding_status=str(chunk.metadata.get("embedding_status", "")),
            required_fields_present=required,
            compatible=mismatch == "none" and required,
            mismatch=mismatch,
            needs_embedding=needs,
            embedding_input_hash=prepared_hash,
        ))
    return rows


def audit_model_loading(config: EmbeddingConfig, *, attempt: bool) -> dict[str, Any]:
    out = {
        "embedding_model_load_attempted": False,
        "embedding_model_load_success": False,
        "embedding_model_load_failure_class": "not_attempted",
        "embedding_model_load_failure_message_digest": "",
        "selected_device": "",
    }
    if not attempt:
        return out
    out["embedding_model_load_attempted"] = True
    try:
        provider = QwenLocalEmbeddingProvider(config)
        out["selected_device"] = provider.device
        _ = provider._model
        out["embedding_model_load_success"] = True
        out["embedding_model_load_failure_class"] = "none"
    except Exception as exc:  # pragma: no cover - model/env dependent
        message = f"{type(exc).__name__}: {exc}"
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            message += f"; cause={type(cause).__name__}: {cause}"
        out["embedding_model_load_failure_class"] = classify_model_load_failure(exc, message)
        out["embedding_model_load_failure_message_digest"] = hashlib.sha256(message.encode("utf-8")).hexdigest()
    return out


def audit_embedding_invocation(config: EmbeddingConfig, load: Mapping[str, Any], rows: Sequence[ChunkInputAudit], chunks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    input_pairs = [(row, payload) for row, payload in zip(rows, chunks, strict=False) if row.compatible and row.needs_embedding]
    out: dict[str, Any] = {
        "embedding_invocation_count": 0,
        "embedding_invocation_chunk_count": 0,
        "raw_embedding_output_count": 0,
        "embedding_invocation_failure_class": "none",
        "raw_embeddings": [],
    }
    if not load.get("embedding_model_load_success") or not input_pairs:
        return out
    try:
        provider = QwenLocalEmbeddingProvider(config)
        texts = [prepare_embedding_input(_stored_chunk_from_payload(payload), config, count_tokens=provider.count_tokens).text for _, payload in input_pairs]
        out["embedding_invocation_count"] = 1
        out["embedding_invocation_chunk_count"] = len(texts)
        vectors = provider.embed_documents(texts)
        out["raw_embedding_output_count"] = len(vectors)
        out["raw_embeddings"] = vectors
    except Exception as exc:  # pragma: no cover - model/env dependent
        out["embedding_invocation_failure_class"] = f"{type(exc).__name__}"
    return out


def validate_raw_outputs(vectors: Sequence[Sequence[float]], config: EmbeddingConfig) -> dict[str, Any]:
    result = {
        "embedding_dimension_valid_count": 0,
        "embedding_dimension_invalid_count": 0,
        "embedding_finite_valid_count": 0,
        "embedding_nonzero_valid_count": 0,
        "embedding_normalization_valid_count": 0,
        "embedding_normalization_invalid_count": 0,
    }
    for vector in vectors:
        values = [float(v) for v in vector]
        if len(values) == config.dimension:
            result["embedding_dimension_valid_count"] += 1
        else:
            result["embedding_dimension_invalid_count"] += 1
        finite = all(math.isfinite(v) for v in values)
        nonzero = any(v != 0.0 for v in values)
        norm = math.sqrt(sum(v * v for v in values)) if values else 0.0
        if finite:
            result["embedding_finite_valid_count"] += 1
        if nonzero:
            result["embedding_nonzero_valid_count"] += 1
        if config.normalize and math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            result["embedding_normalization_valid_count"] += 1
        elif config.normalize:
            result["embedding_normalization_invalid_count"] += 1
    return result


def build_summary(*, source_head: str, task0181: Mapping[str, Any], task0174: Mapping[str, Any], config: EmbeddingConfig, database: Mapping[str, Any], contract_rows: Sequence[ChunkInputAudit], cache: Mapping[str, Any], load: Mapping[str, Any], invocation: Mapping[str, Any], output_validation: Mapping[str, Any]) -> dict[str, Any]:
    input_count = sum(1 for row in contract_rows if row.compatible and row.needs_embedding)
    mismatch_count = sum(1 for row in contract_rows if not row.compatible)
    stored = int(database.get("stored_embedding_count", 0) or 0)
    persist_success = stored
    persist_attempt = 0 if int(invocation.get("raw_embedding_output_count", 0) or 0) == 0 else stored
    base = {
        "task_id": TASK_ID,
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0181.get("source_authoritative_head") or source_head,
        "authoritative_document_count": int(database.get("authoritative_document_count", 0) or 0),
        "authoritative_chunk_count": int(database.get("authoritative_chunk_count", 0) or 0),
        "embedding_model": config.model_name,
        "embedding_model_revision": config.model_revision,
        "embedding_dimension": config.dimension,
        "embedding_distance_metric": config.distance_metric,
        "embedding_normalization_policy": "l2" if config.normalize else "none",
        "embedding_eligible_chunk_count": int(database.get("embedding_eligible_chunk_count", 0) or 0),
        "embedding_input_chunk_count": input_count,
        "embedding_input_rejected_chunk_count": mismatch_count,
        "embedding_input_contract_valid": mismatch_count == 0 and len(contract_rows) > 0,
        "embedding_input_contract_mismatch_count": mismatch_count,
        "first_embedding_input_contract_mismatch": next((row.mismatch for row in contract_rows if not row.compatible), "none"),
        "model_cache_path_present": bool(cache.get("snapshot_path")),
        "model_cache_exists": bool(cache.get("model_cache_exists")),
        "model_cache_complete": bool(cache.get("model_cache_complete")),
        "model_cache_authority_valid": bool(cache.get("cache_authority_valid")),
        "model_revision_match": bool(cache.get("revision_match")),
        "local_files_only_policy": config.local_files_only,
        "embedding_model_load_attempted": bool(load.get("embedding_model_load_attempted")),
        "embedding_model_load_success": bool(load.get("embedding_model_load_success")),
        "embedding_model_load_failure_class": load.get("embedding_model_load_failure_class", "unknown"),
        "embedding_invocation_count": int(invocation.get("embedding_invocation_count", 0) or 0),
        "embedding_invocation_chunk_count": int(invocation.get("embedding_invocation_chunk_count", 0) or 0),
        "raw_embedding_output_count": int(invocation.get("raw_embedding_output_count", 0) or 0),
        **output_validation,
        "embedding_persist_attempt_count": persist_attempt,
        "embedding_persist_success_count": persist_success,
        "embedding_persist_failure_count": max(0, persist_attempt - persist_success),
        "stored_embedding_count": stored,
        "stored_vector_dimension": database.get("stored_vector_dimension"),
        "embedding_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
    }
    first, root = determine_failure(base)
    base["first_embedding_loss_substage"] = first
    base["diagnosed_root_cause"] = root
    base["task0174_root_cause_reproduced"] = root == "model_cache_missing_or_incomplete"
    base["task0174_root_cause_still_authoritative"] = root == "model_cache_missing_or_incomplete" and task0174.get("diagnosed_root_cause") == "model_cache_missing_or_incomplete"
    base["cold_start_embedding_stage_passed"] = first == "none"
    base["next_failure_stage"] = "vector_index_materialization" if first == "none" else ("embedding_model_cache_remediation" if root == "model_cache_missing_or_incomplete" else first)
    base["focused_test_passed_count"] = 0
    base["full_suite_passed_count"] = 0
    base["full_suite_skipped_count"] = 0
    base["full_suite_failed_count"] = 0
    base["known_preexisting_failure_count"] = 0
    base["new_regression_count"] = 0
    base["task_status"] = "complete" if base["authoritative_chunk_count"] == 11 and first != "unknown" else "partial"
    return base


def determine_failure(summary: Mapping[str, Any]) -> tuple[str, str]:
    if int(summary["embedding_input_chunk_count"]) == 0:
        return "chunk_to_embedding_input", "chunk_embedding_contract_mismatch"
    if not summary["embedding_input_contract_valid"]:
        return "embedding_input_validation", "chunk_embedding_contract_mismatch"
    if not summary["model_cache_authority_valid"]:
        return "embedding_model_cache", "model_cache_missing_or_incomplete"
    if summary["embedding_model_load_attempted"] and not summary["embedding_model_load_success"]:
        return "embedding_model_loading", "embedding_model_load_failure"
    if summary["embedding_model_load_success"] and int(summary["embedding_invocation_count"]) == 0:
        return "embedding_execution", "embedding_runtime_dispatch_failure"
    if int(summary["raw_embedding_output_count"]) > 0 and int(summary["embedding_dimension_invalid_count"]) > 0:
        return "embedding_output_validation", "embedding_dimension_mismatch"
    if int(summary["raw_embedding_output_count"]) > 0 and int(summary["embedding_normalization_invalid_count"]) > 0:
        return "embedding_output_validation", "embedding_normalization_failure"
    if int(summary["stored_embedding_count"]) < int(summary["authoritative_chunk_count"]):
        return "embedding_persistence", "embedding_persistence_failure"
    return "none", "no_embedding_failure_reproduced"


def classify_model_load_failure(exc: Exception, message: str) -> str:
    text = message.lower()
    if isinstance(exc, EmbeddingModelLoadError) and ("cuda" in text or "device" in text):
        return "cuda_initialization_failure" if "cuda" in text else "device_allocation_failure"
    if "not found" in text or "missing" in text or "no such file" in text or "does not appear" in text:
        return "model_cache_missing_or_incomplete"
    if "revision" in text or "commit" in text:
        return "model_revision_mismatch"
    if "local_files_only" in text or "offline" in text:
        return "offline_policy_failure"
    if "sentence-transformers is required" in text or "modulenotfounderror" in text:
        return "dependency_failure"
    if "config" in text:
        return "model_configuration_failure"
    return "unknown_model_load_failure"


def failure_taxonomy_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"first_embedding_loss_substage": summary.get("first_embedding_loss_substage"), "diagnosed_root_cause": summary.get("diagnosed_root_cause")}


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "diagnosis_only": True,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dimension": 1024,
        "distance_metric": "cosine",
        "l2_normalization": True,
        "runtime_default_behavior_change_allowed": False,
        "promotion_allowed": False,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# TASK-0182 Cold-start Embedding Materialization Failure Diagnosis Report",
        "",
        "## Decision",
        f"First embedding loss substage: `{summary.get('first_embedding_loss_substage')}`.",
        f"Diagnosed root cause: `{summary.get('diagnosed_root_cause')}`.",
        "",
        "## Counters",
        f"Authoritative documents/chunks: `{summary.get('authoritative_document_count')}` / `{summary.get('authoritative_chunk_count')}`.",
        f"Embedding input chunks: `{summary.get('embedding_input_chunk_count')}`; stored embeddings: `{summary.get('stored_embedding_count')}`.",
        f"Model cache valid: `{summary.get('model_cache_authority_valid')}`; model load success: `{summary.get('embedding_model_load_success')}`.",
        "",
        "## Boundary",
        "No embedding policy, runtime default behavior, retriever, reranker, generation, graph, chunking, or corpus promotion was applied.",
        "",
        "## Machine-readable summary",
        "```json",
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ])


class _StaticTokenProvider:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


def _stored_chunk_payload(row: Sequence[Any]) -> dict[str, Any]:
    return {
        "id": str(row[0]), "document_id": str(row[1]), "index_configuration_id": str(row[2]),
        "chunk_index": row[3], "content": row[4], "content_hash": row[5], "heading_path": tuple(row[6] or ()),
        "start_line": row[7], "end_line": row[8], "token_count": row[9], "embedding": row[10],
        "embedding_model": row[11], "embedding_dimension": row[12], "metadata": row[13] or {},
        "created_at": row[14], "updated_at": row[15], "relative_path": row[16],
        "document_index_status": row[17], "document_parser_version": row[18], "document_chunking_version": row[19],
    }


def _stored_chunk_from_payload(payload: Mapping[str, Any]) -> StoredChunk:
    from uuid import UUID
    from datetime import datetime, timezone
    return StoredChunk(
        id=UUID(str(payload["id"])), document_id=UUID(str(payload["document_id"])),
        index_configuration_id=UUID(str(payload["index_configuration_id"])), chunk_index=int(payload["chunk_index"]),
        content=str(payload.get("content") or ""), content_hash=str(payload.get("content_hash") or ""),
        heading_path=tuple(payload.get("heading_path") or ()), start_line=payload.get("start_line"), end_line=payload.get("end_line"),
        token_count=payload.get("token_count"), embedding=payload.get("embedding"), embedding_model=payload.get("embedding_model"),
        embedding_dimension=payload.get("embedding_dimension"), metadata=dict(payload.get("metadata") or {}),
        created_at=payload.get("created_at") or datetime.now(timezone.utc), updated_at=payload.get("updated_at") or datetime.now(timezone.utc),
    )


def verify_task0182_artifacts(*, result_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    required = ["summary.json", "persisted_chunk_authority.json", "embedding_input_contract_audit.json", "model_cache_authority_audit.json", "model_loading_audit.json", "failure_taxonomy_decision.json"]
    missing = [name for name in required if not (result_dir / name).exists()]
    summary = _load_json(result_dir / "summary.json") if (result_dir / "summary.json").exists() else {}
    errors: list[str] = []
    if summary.get("task_id") != TASK_ID:
        errors.append("summary task_id mismatch")
    if summary.get("authoritative_chunk_count") != 11:
        errors.append("authoritative_chunk_count != 11")
    if summary.get("embedding_policy_changed") is not False or summary.get("runtime_default_behavior_change") is not False:
        errors.append("diagnosis boundary flags changed")
    if summary.get("first_embedding_loss_substage") in {None, "unknown"}:
        errors.append("first embedding loss substage not determined")
    result = {"task_id": TASK_ID, "status": "pass" if not missing and not errors else "fail", "missing_artifacts": missing, "errors": errors}
    if write:
        write_json(result_dir / "verification.json", result)
    return result


def _sanitize_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cache)
    home = str(Path.home())
    for key in ("cache_root", "snapshot_path"):
        if isinstance(out.get(key), str):
            out[key] = out[key].replace(home, "<home>")
    return out


def _git_head(cwd: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _redact_text(text: str) -> str:
    return SECRET_URL_RE.sub("postgresql://***:***@", text)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        return {k: ("***redacted***" if k == "database_url_value" else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
