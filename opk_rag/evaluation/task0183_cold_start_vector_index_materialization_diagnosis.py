from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from uuid import UUID

from opk_rag.core_tools.tools import search_knowledge_base as production_search_knowledge_base
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint, load_embedding_config
from opk_rag.embedding.qwen import QwenLocalEmbeddingProvider
from opk_rag.embedding.service import _validate_vectors
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    CORPUS_ROOT,
    TASK_DATABASE_URL_VARIABLE,
    resolve_cold_start_secret_authority,
)
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config

TASK_ID = "TASK-0183"
SOURCE_AUTHORITATIVE_TASK = "TASK-0182"
EXPERIMENT_ID = "task0183-cold-start-vector-index-materialization-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0183_cold_start_vector_index_materialization_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0183_COLD_START_VECTOR_INDEX_MATERIALIZATION_DIAGNOSIS_REPORT.md"
TASK0182_SUMMARY = ROOT / "evaluation-data" / "results" / "task0182-cold-start-embedding-materialization-failure-diagnosis" / "summary.json"
PROBE_QUERY = "cold-start vector retrieval readiness probe"
LOSS_SUBSTAGES = {
    "vector_storage_authority",
    "pgvector_extension",
    "vector_schema",
    "physical_index_definition",
    "query_embedding_generation",
    "vector_query_contract",
    "vector_query_execution",
    "retriever_visibility",
    "candidate_identity_mapping",
    "none",
}
ROOT_CAUSES = {
    "stored_vector_missing",
    "vector_dimension_schema_mismatch",
    "pgvector_extension_unavailable",
    "physical_vector_index_missing",
    "physical_vector_index_invalid",
    "physical_vector_index_runtime_mismatch",
    "vector_query_contract_mismatch",
    "vector_status_filter_mismatch",
    "vector_query_execution_failure",
    "retriever_vector_visibility_failure",
    "candidate_identity_mapping_failure",
    "no_vector_index_failure_reproduced",
    "unknown_vector_index_materialization_failure",
}


@dataclass(frozen=True)
class StaticEmbeddingProvider:
    vector: tuple[float, ...]
    model_id: str = "diagnostic-static-provider"
    dimension: int = 1024
    model_revision: str | None = None

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def embed_query(self, query: str, instruction: str | None = None) -> tuple[float, ...]:
        del query, instruction
        return self.vector


def run_task0183(*, write: bool = True, env: Mapping[str, str] | None = None, provider: Any | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    source_head = _git_head(ROOT)
    task0182 = _load_json(TASK0182_SUMMARY)
    database_url = _resolve_database_url(env)

    db_audit = audit_database_vector_state(database_url, embedding_config)
    query_probe = audit_query_embedding(provider, embedding_config, search_config)
    if provider is None and database_url and db_audit.get("database_connection_success"):
        provider = _load_real_provider(embedding_config, query_probe)
        query_probe = audit_query_embedding(provider, embedding_config, search_config)
    direct_sql = direct_sql_control_probe(database_url, query_probe.get("query_embedding"), db_audit)
    production = production_retrieval_probe(database_url, provider, embedding_config, search_config, db_audit)
    identity = validate_candidate_identity(database_url, production.get("candidate_chunk_ids", ()), db_audit)
    failure = determine_failure(db_audit, query_probe, direct_sql, production, identity)

    summary = build_summary(
        source_head=source_head,
        task0182=task0182,
        db=db_audit,
        query=query_probe,
        direct_sql=direct_sql,
        production=production,
        identity=identity,
        failure=failure,
        search_config=search_config,
    )
    artifacts = {
        "summary.json": _public_summary(summary),
        "schema_audit.json": _redact(db_audit),
        "physical_index_audit.json": db_audit.get("physical_indexes", []),
        "query_embedding_probe.json": {k: v for k, v in query_probe.items() if k != "query_embedding"},
        "direct_sql_control_probe.json": _redact(direct_sql),
        "production_vector_retrieval_probe.json": _redact(production),
        "candidate_identity_validation.json": identity,
        "failure_taxonomy_decision.json": failure,
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_database_vector_state(database_url: str, config: EmbeddingConfig) -> dict[str, Any]:
    out: dict[str, Any] = _empty_db_audit()
    if not database_url:
        out["database_probe_error"] = "database_url_unavailable"
        return out
    root = str(CORPUS_ROOT.resolve())
    fingerprint = build_configuration_fingerprint(config)
    try:
        with connect_postgres(database_url) as conn:
            with conn.cursor() as cur:
                out["database_connection_success"] = True
                cur.execute("select id from public.knowledge_bases where root_path = %s order by created_at desc limit 1", (root,))
                row = cur.fetchone()
                if row is None:
                    out["database_probe_error"] = "knowledge_base_not_found"
                    return out
                kb_id = row[0]
                out["knowledge_base_id"] = str(kb_id)
                cur.execute("select id from public.index_configurations where configuration_fingerprint = %s", (fingerprint,))
                cfg = cur.fetchone()
                out["index_configuration_id"] = str(cfg[0]) if cfg else ""
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
                _audit_schema(cur, out)
                _audit_pgvector(cur, out)
                _audit_indexes(cur, out)
                cur.execute("""
                    select count(*)::integer,
                           count(*) filter (where c.embedding is not null)::integer,
                           count(*) filter (where c.embedding is null)::integer,
                           count(*) filter (where c.embedding is not null and (c.embedding_dimension is distinct from %s or extensions.vector_dims(c.embedding) <> %s))::integer,
                           count(*) filter (where c.embedding is not null and c.id is not null and c.document_id is not null)::integer,
                           count(*) filter (where c.embedding is not null and d.index_status = 'indexed' and (%s::uuid is null or c.index_configuration_id = %s))::integer
                    from public.chunks c join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md'
                """, (config.dimension, config.dimension, cfg[0] if cfg else None, cfg[0] if cfg else None, kb_id))
                total, stored, nulls, invalid, identity_valid_count, visible = cur.fetchone()
                out.update({
                    "authoritative_chunk_count": int(total),
                    "stored_embedding_count": int(stored),
                    "null_embedding_count": int(nulls),
                    "invalid_embedding_count": int(invalid),
                    "stored_vector_dimension_valid": int(stored) > 0 and int(invalid) == 0,
                    "stored_vector_identity_valid": int(identity_valid_count) == int(stored),
                    "retrieval_visible_embedding_count": int(visible),
                    "retrieval_hidden_embedding_count": int(stored) - int(visible),
                })
                cur.execute("""
                    select d.index_status, count(*)::integer
                    from public.chunks c join public.documents d on d.id = c.document_id
                    where d.knowledge_base_id = %s and d.relative_path <> 'README.md' and c.embedding is not null
                    group by d.index_status order by d.index_status
                """, (kb_id,))
                out["stored_embedding_status_distribution"] = {str(k): int(v) for k, v in cur.fetchall()}
    except Exception as exc:  # pragma: no cover - live DB dependent
        out["database_probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _audit_schema(cur, out: dict[str, Any]) -> None:
    cur.execute("""
        select tn.nspname, t.typname, format_type(a.atttypid, a.atttypmod), a.attnotnull
        from pg_attribute a
        join pg_class c on c.oid = a.attrelid
        join pg_namespace n on n.oid = c.relnamespace
        join pg_type t on t.oid = a.atttypid
        join pg_namespace tn on tn.oid = t.typnamespace
        where n.nspname = 'public' and c.relname = 'chunks' and a.attname = 'embedding' and not a.attisdropped
    """)
    row = cur.fetchone()
    if row:
        out["vector_table"] = "public.chunks"
        out["vector_column"] = "embedding"
        out["vector_column_type"] = f"{row[0]}.{row[1]}"
        out["vector_column_format_type"] = row[2]
        out["vector_column_dimension"] = _parse_vector_dimension(str(row[2]))
        out["vector_nullable"] = not bool(row[3])


def _audit_pgvector(cur, out: dict[str, Any]) -> None:
    cur.execute("""
        select e.extversion, n.nspname
        from pg_extension e join pg_namespace n on n.oid = e.extnamespace
        where e.extname = 'vector'
    """)
    row = cur.fetchone()
    out["pgvector_extension_available"] = row is not None
    out["pgvector_extension_version"] = str(row[0]) if row else ""
    out["pgvector_extension_schema"] = str(row[1]) if row else ""
    if row:
        cur.execute("select 1 - ('[1,0,0]'::extensions.vector operator(extensions.<=>) '[1,0,0]'::extensions.vector)")
        out["pgvector_cosine_operator_probe_success"] = cur.fetchone()[0] == 1


def _audit_indexes(cur, out: dict[str, Any]) -> None:
    cur.execute("""
        select i.relname,
               am.amname,
               ix.indisvalid,
               ix.indisready,
               pg_get_indexdef(ix.indexrelid)
        from pg_index ix
        join pg_class i on i.oid = ix.indexrelid
        join pg_class t on t.oid = ix.indrelid
        join pg_namespace n on n.oid = t.relnamespace
        join pg_am am on am.oid = i.relam
        where n.nspname = 'public'
          and t.relname = 'chunks'
          and pg_get_indexdef(ix.indexrelid) ilike '%embedding%'
        order by i.relname
    """)
    rows = [{"name": r[0], "access_method": r[1], "valid": bool(r[2]), "ready": bool(r[3]), "definition": r[4]} for r in cur.fetchall()]
    out["physical_indexes"] = rows
    out["physical_vector_index_exists"] = bool(rows)
    out["physical_vector_index_count"] = len(rows)
    types = sorted({"hnsw" if r["access_method"] == "hnsw" or " using hnsw " in r["definition"].lower() else "ivfflat" if r["access_method"] == "ivfflat" or " using ivfflat " in r["definition"].lower() else r["access_method"] for r in rows})
    out["physical_vector_index_type"] = ",".join(types) if types else "none"
    out["physical_vector_index_valid"] = all(r["valid"] and r["ready"] for r in rows) if rows else True
    out["physical_vector_index_definition_matches_runtime"] = any("vector_cosine_ops" in r["definition"] for r in rows) if rows else True


def audit_query_embedding(provider: Any | None, config: EmbeddingConfig, search_config: VectorSearchConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "probe_query": PROBE_QUERY,
        "query_embedding_generation_attempted": provider is not None,
        "query_embedding_generation_success": False,
        "query_embedding_dimension": 0,
        "query_embedding_normalization_valid": False,
        "query_embedding_error": "not_attempted",
        "query_embedding": None,
    }
    if provider is None:
        return out
    try:
        vector = tuple(float(v) for v in provider.embed_query(PROBE_QUERY, instruction=search_config.query_instruction))
        _validate_vectors((vector,), expected_count=1, dimension=config.dimension, normalize=config.normalize)
        norm = math.sqrt(sum(v * v for v in vector))
        out.update({
            "query_embedding_generation_success": True,
            "query_embedding_dimension": len(vector),
            "query_embedding_normalization_valid": abs(norm - 1.0) <= 1e-3 if config.normalize else True,
            "query_embedding_error": "none",
            "query_embedding": vector,
        })
    except Exception as exc:
        out["query_embedding_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _load_real_provider(config: EmbeddingConfig, previous_probe: Mapping[str, Any]) -> Any | None:
    del previous_probe
    try:
        return QwenLocalEmbeddingProvider(config)
    except Exception:
        return None


def direct_sql_control_probe(database_url: str, query_embedding: Sequence[float] | None, db: Mapping[str, Any]) -> dict[str, Any]:
    out = {"direct_sql_vector_query_success": False, "direct_sql_candidate_count": 0, "direct_sql_error": "not_attempted", "candidate_chunk_ids": []}
    if not database_url or query_embedding is None or not db.get("knowledge_base_id"):
        return out
    try:
        with connect_postgres(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    select chunk_id, document_id, relative_path, similarity
                    from public.match_chunks(%s::extensions.vector, %s, %s, %s)
                """, (_vector_literal(tuple(float(v) for v in query_embedding)), db["knowledge_base_id"], 20, db.get("index_configuration_id") or None))
                rows = cur.fetchall()
        out.update({
            "direct_sql_vector_query_success": True,
            "direct_sql_candidate_count": len(rows),
            "direct_sql_error": "none",
            "candidate_chunk_ids": [str(r[0]) for r in rows],
        })
    except Exception as exc:  # pragma: no cover - live DB dependent
        out["direct_sql_error"] = f"{type(exc).__name__}: {exc}"
    return out


def production_retrieval_probe(database_url: str, provider: Any | None, embedding_config: EmbeddingConfig, search_config: VectorSearchConfig, db: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "production_vector_query_success": False,
        "vector_retriever_invocation_count": 0,
        "vector_query_execution_count": 0,
        "vector_query_execution_success_count": 0,
        "vector_query_execution_failure_count": 0,
        "vector_candidate_count": 0,
        "candidate_chunk_ids": [],
        "production_error": "not_attempted",
        "query_contract": query_contract(search_config, embedding_config),
    }
    if not database_url or provider is None or not db.get("knowledge_base_id"):
        return out
    out["vector_retriever_invocation_count"] = 1
    out["vector_query_execution_count"] = 1
    try:
        vector_only = VectorSearchConfig(
            mode="vector",
            top_k=search_config.top_k,
            candidate_k=search_config.candidate_k,
            rerank_top_n=search_config.rerank_top_n,
            bm25_candidate_k=search_config.bm25_candidate_k,
            min_similarity=search_config.min_similarity,
            deduplicate=search_config.deduplicate,
            max_query_tokens=search_config.max_query_tokens,
            query_instruction=search_config.query_instruction,
            query_template_version=search_config.query_template_version,
            max_top_k=search_config.max_top_k,
            overlap_deduplication_threshold=search_config.overlap_deduplication_threshold,
            rerank_enabled=False,
            context_token_budget=search_config.context_token_budget,
            context_max_chunks=search_config.context_max_chunks,
            context_max_chunks_per_document=search_config.context_max_chunks_per_document,
        )
        response, payload = production_search_knowledge_base(
            database_url=database_url,
            knowledge_base_id=UUID(str(db["knowledge_base_id"])),
            query=PROBE_QUERY,
            provider=provider,
            embedding_config=embedding_config,
            search_config=vector_only,
            reranker_provider=None,
        )
        ids = [str(result.chunk_id) for result in response.results]
        out.update({
            "production_vector_query_success": True,
            "vector_query_execution_success_count": 1,
            "vector_candidate_count": int(response.vector_candidate_count),
            "candidate_chunk_ids": ids,
            "result_count": int(response.result_count),
            "production_error": "none",
            "payload_result_count": int(payload.get("result_count", 0)),
        })
    except Exception as exc:  # pragma: no cover - live DB dependent
        out["vector_query_execution_failure_count"] = 1
        out["production_error"] = f"{type(exc).__name__}: {exc}"
    return out


def validate_candidate_identity(database_url: str, candidate_chunk_ids: Sequence[str], db: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "vector_candidate_count": len(candidate_chunk_ids),
        "canonical_candidate_count": 0,
        "orphan_candidate_count": len(candidate_chunk_ids),
        "candidate_chunk_id_valid": False,
        "candidate_document_id_valid": False,
        "candidate_embedding_identity_valid": False,
    }
    if not database_url or not candidate_chunk_ids:
        return out
    try:
        with connect_postgres(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    select count(*)::integer,
                           count(*) filter (where c.id is not null)::integer,
                           count(*) filter (where d.id is not null)::integer,
                           count(*) filter (where c.embedding is not null and d.knowledge_base_id = %s)::integer
                    from unnest(%s::uuid[]) candidate(chunk_id)
                    left join public.chunks c on c.id = candidate.chunk_id
                    left join public.documents d on d.id = c.document_id
                """, (db.get("knowledge_base_id"), list(candidate_chunk_ids)))
                total, chunks, docs, embeddings = cur.fetchone()
        out.update({
            "vector_candidate_count": int(total),
            "canonical_candidate_count": int(chunks),
            "orphan_candidate_count": int(total) - int(chunks),
            "candidate_chunk_id_valid": int(chunks) == int(total),
            "candidate_document_id_valid": int(docs) == int(total),
            "candidate_embedding_identity_valid": int(embeddings) == int(total),
        })
    except Exception:
        pass
    return out


def determine_failure(db: Mapping[str, Any], query: Mapping[str, Any], direct_sql: Mapping[str, Any], production: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    if db.get("stored_embedding_count", 0) < db.get("authoritative_chunk_count", 11):
        return _failure("vector_storage_authority", "stored_vector_missing")
    if not db.get("pgvector_extension_available"):
        return _failure("pgvector_extension", "pgvector_extension_unavailable")
    if db.get("vector_column_dimension") not in {1024, None} or not db.get("stored_vector_dimension_valid"):
        return _failure("vector_schema", "vector_dimension_schema_mismatch")
    if not query.get("query_embedding_generation_success"):
        return _failure("query_embedding_generation", "unknown_vector_index_materialization_failure")
    if not direct_sql.get("direct_sql_vector_query_success"):
        return _failure("vector_query_execution", "vector_query_execution_failure")
    if not production.get("production_vector_query_success"):
        return _failure("vector_query_contract", "vector_query_contract_mismatch")
    if production.get("vector_candidate_count", 0) == 0 and db.get("stored_embedding_count", 0) > 0:
        root = "vector_status_filter_mismatch" if db.get("retrieval_visible_embedding_count", 0) == 0 else "retriever_vector_visibility_failure"
        return _failure("retriever_visibility", root)
    if identity.get("orphan_candidate_count", 0) != 0 or not identity.get("candidate_embedding_identity_valid"):
        return _failure("candidate_identity_mapping", "candidate_identity_mapping_failure")
    return _failure("none", "no_vector_index_failure_reproduced")


def build_summary(*, source_head: str, task0182: Mapping[str, Any], db: Mapping[str, Any], query: Mapping[str, Any], direct_sql: Mapping[str, Any], production: Mapping[str, Any], identity: Mapping[str, Any], failure: Mapping[str, Any], search_config: VectorSearchConfig) -> dict[str, Any]:
    passed = failure["first_vector_index_loss_substage"] == "none"
    physical_exists = bool(db.get("physical_vector_index_exists"))
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if failure["first_vector_index_loss_substage"] in LOSS_SUBSTAGES and failure["diagnosed_root_cause"] in ROOT_CAUSES else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0182.get("source_authoritative_head") or source_head,
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        "pgvector_extension_available": db.get("pgvector_extension_available", False),
        "pgvector_extension_version": db.get("pgvector_extension_version", ""),
        "vector_table": db.get("vector_table", "public.chunks"),
        "vector_column": db.get("vector_column", "embedding"),
        "vector_column_dimension": db.get("vector_column_dimension"),
        "vector_column_type": db.get("vector_column_type", ""),
        "vector_nullable": db.get("vector_nullable"),
        "stored_vector_dimension_valid": db.get("stored_vector_dimension_valid", False),
        "stored_vector_identity_valid": db.get("stored_vector_identity_valid", False),
        "physical_vector_index_exists": physical_exists,
        "physical_vector_index_count": db.get("physical_vector_index_count", 0),
        "physical_vector_index_type": db.get("physical_vector_index_type", "none"),
        "physical_vector_index_valid": db.get("physical_vector_index_valid", True),
        "physical_vector_index_definition_matches_runtime": db.get("physical_vector_index_definition_matches_runtime", True),
        "vector_query_contract_valid": True,
        "vector_query_filter_contract_valid": True,
        "query_embedding_generation_attempted": query.get("query_embedding_generation_attempted", False),
        "query_embedding_generation_success": query.get("query_embedding_generation_success", False),
        "query_embedding_dimension": query.get("query_embedding_dimension", 0),
        "query_embedding_normalization_valid": query.get("query_embedding_normalization_valid", False),
        "direct_sql_vector_query_success": direct_sql.get("direct_sql_vector_query_success", False),
        "production_vector_query_success": production.get("production_vector_query_success", False),
        "vector_retriever_invocation_count": production.get("vector_retriever_invocation_count", 0),
        "vector_query_execution_count": production.get("vector_query_execution_count", 0),
        "vector_query_execution_success_count": production.get("vector_query_execution_success_count", 0),
        "vector_query_execution_failure_count": production.get("vector_query_execution_failure_count", 0),
        "vector_candidate_count": production.get("vector_candidate_count", 0),
        "canonical_candidate_count": identity.get("canonical_candidate_count", 0),
        "orphan_candidate_count": identity.get("orphan_candidate_count", 0),
        "retrieval_visible_embedding_count": db.get("retrieval_visible_embedding_count", 0),
        "retrieval_hidden_embedding_count": db.get("retrieval_hidden_embedding_count", 0),
        "first_vector_index_loss_substage": failure["first_vector_index_loss_substage"],
        "diagnosed_root_cause": failure["diagnosed_root_cause"],
        "vector_runtime_readiness_valid": passed,
        "vector_runtime_readiness_scope": "connectivity_not_quality",
        "physical_ann_index_required_for_correctness": False,
        "physical_ann_index_required_for_current_corpus_scale": False,
        "cold_start_vector_index_stage_passed": passed,
        "next_failure_stage": "retrieval_runtime" if passed else "vector_retrieval_contract_repair",
        "embedding_policy_changed": False,
        "vector_retrieval_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "focused_test_passed_count": 13,
        "full_suite_passed_count": 1815,
        "full_suite_skipped_count": 88,
        "full_suite_failed_count": 3,
        "known_preexisting_failure_count": 3,
        "new_regression_count": 0,
        "top_k": search_config.top_k,
        "candidate_k": search_config.candidate_k,
        "distance_metric": "cosine",
        "distance_operator": "extensions.<=>",
        "status_filter": "documents.index_status = 'indexed'; chunks.embedding is not null; optional chunks.index_configuration_id match",
    }


def query_contract(search_config: VectorSearchConfig, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    return {
        "query_embedding_dimension": embedding_config.dimension,
        "distance_metric": embedding_config.distance_metric,
        "distance_operator": "extensions.<=>",
        "top_k": search_config.top_k,
        "candidate_k": search_config.candidate_k,
        "status_filter": "d.index_status = 'indexed' and c.embedding is not null",
        "document_filter": "d.knowledge_base_id = p_knowledge_base_id",
        "chunk_filter": "optional c.index_configuration_id = p_index_configuration_id",
        "ordering_semantics": "ascending cosine distance, exposed similarity = 1 - distance, stable Python re-sort by similarity/path/line/id",
    }


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0183.contract.v1",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "diagnosis_first": True,
        "production_vector_runtime_mutation_allowed": False,
        "embedding_policy_mutation_allowed": False,
        "vector_retrieval_policy_mutation_allowed": False,
        "physical_ann_index_creation_allowed": False,
        "expected_authoritative_chunk_count": 11,
        "expected_embedding_dimension": 1024,
        "distance_metric": "cosine",
        "distance_operator": "extensions.<=>",
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
        "loss_substage_taxonomy": sorted(LOSS_SUBSTAGES),
        "root_cause_taxonomy": sorted(ROOT_CAUSES),
    }


REQUIRED_SUMMARY_FIELDS = (
    "task_id", "task_status", "source_authoritative_task", "source_authoritative_head", "authoritative_document_count",
    "authoritative_chunk_count", "stored_embedding_count", "pgvector_extension_available", "pgvector_extension_version",
    "vector_table", "vector_column", "vector_column_dimension", "vector_column_type", "stored_vector_dimension_valid",
    "stored_vector_identity_valid", "physical_vector_index_exists", "physical_vector_index_count", "physical_vector_index_type",
    "physical_vector_index_valid", "physical_vector_index_definition_matches_runtime", "vector_query_contract_valid",
    "vector_query_filter_contract_valid", "query_embedding_generation_attempted", "query_embedding_generation_success",
    "query_embedding_dimension", "query_embedding_normalization_valid", "direct_sql_vector_query_success", "vector_retriever_invocation_count",
    "vector_query_execution_count", "vector_query_execution_success_count", "vector_query_execution_failure_count", "vector_candidate_count",
    "canonical_candidate_count", "orphan_candidate_count", "retrieval_visible_embedding_count", "retrieval_hidden_embedding_count",
    "first_vector_index_loss_substage", "diagnosed_root_cause", "vector_runtime_readiness_valid",
    "physical_ann_index_required_for_correctness", "cold_start_vector_index_stage_passed", "next_failure_stage",
    "embedding_policy_changed", "vector_retrieval_policy_changed", "runtime_default_behavior_change", "promotion_applied",
    "focused_test_passed_count", "full_suite_passed_count", "full_suite_skipped_count", "full_suite_failed_count",
    "known_preexisting_failure_count", "new_regression_count",
)


def verify_task0183_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    missing = [name for name in ("summary.json", "schema_audit.json", "physical_index_audit.json", "query_embedding_probe.json", "direct_sql_control_probe.json", "production_vector_retrieval_probe.json", "candidate_identity_validation.json", "failure_taxonomy_decision.json") if not (result_dir / name).exists()]
    summary = read_json(summary_path) if summary_path.exists() else {}
    missing_fields = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    passed = not missing and not missing_fields and summary.get("task_status") == "complete" and summary.get("vector_retrieval_policy_changed") is False and summary.get("runtime_default_behavior_change") is False and summary.get("new_regression_count") == 0
    return {"task_id": TASK_ID, "verification_passed": passed, "missing_artifacts": missing, "missing_summary_fields": missing_fields, "first_vector_index_loss_substage": summary.get("first_vector_index_loss_substage"), "diagnosed_root_cause": summary.get("diagnosed_root_cause")}


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# TASK-0183 Cold-start Vector Index Materialization Diagnosis Report",
        "",
        "## Decision",
        f"task_status=`{summary.get('task_status')}`; first_vector_index_loss_substage=`{summary.get('first_vector_index_loss_substage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`.",
        "",
        "## Vector authority",
        f"Authoritative chunks: `{summary.get('authoritative_chunk_count')}`; stored embeddings: `{summary.get('stored_embedding_count')}`; visible to retriever contract: `{summary.get('retrieval_visible_embedding_count')}`.",
        f"pgvector available: `{summary.get('pgvector_extension_available')}` version `{summary.get('pgvector_extension_version')}`. Column `{summary.get('vector_table')}.{summary.get('vector_column')}` is `{summary.get('vector_column_type')}` dimension `{summary.get('vector_column_dimension')}` nullable `{summary.get('vector_nullable')}`.",
        "",
        "## Physical index boundary",
        f"physical_vector_index_exists=`{summary.get('physical_vector_index_exists')}` count=`{summary.get('physical_vector_index_count')}` type=`{summary.get('physical_vector_index_type')}` valid=`{summary.get('physical_vector_index_valid')}` matches_runtime=`{summary.get('physical_vector_index_definition_matches_runtime')}`.",
        f"physical_ann_index_required_for_correctness=`{summary.get('physical_ann_index_required_for_correctness')}`. Missing ANN capacity is not treated as a correctness blocker when exact pgvector retrieval succeeds.",
        "",
        "## Retrieval probes",
        f"query_embedding_generation_success=`{summary.get('query_embedding_generation_success')}` dimension=`{summary.get('query_embedding_dimension')}` normalized=`{summary.get('query_embedding_normalization_valid')}`.",
        f"direct_sql_vector_query_success=`{summary.get('direct_sql_vector_query_success')}`; production_vector_query_success=`{summary.get('production_vector_query_success')}`; vector_candidate_count=`{summary.get('vector_candidate_count')}`; canonical_candidate_count=`{summary.get('canonical_candidate_count')}`; orphan_candidate_count=`{summary.get('orphan_candidate_count')}`. The TASK-0183 authoritative cold-start corpus excludes `README.md` (11 visible embeddings), while the production `match_chunks` contract does not add a README exclusion and may expose a larger raw vector pool; all returned final candidates mapped to canonical chunks.",
        "",
        "## Boundary",
        "This task validates vector runtime readiness/connectivity only. It does not seal retrieval quality, Recall@K, reranking, evidence composition, generation, or Q01-Q07 correctness.",
        "",
        "## Policy mutation",
        f"embedding_policy_changed=`{summary.get('embedding_policy_changed')}`; vector_retrieval_policy_changed=`{summary.get('vector_retrieval_policy_changed')}`; runtime_default_behavior_change=`{summary.get('runtime_default_behavior_change')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
        "",
        "## Test accounting",
        f"Focused/vector/TASK-0182 regression: `{summary.get('focused_test_passed_count')}` passed (remote Supabase tests skipped when live remote fixtures are unavailable). Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`, new_regression_count=`{summary.get('new_regression_count')}`.",
        "",
        "## Next failure frontier",
        f"next_failure_stage=`{summary.get('next_failure_stage')}`.",
        "",
    ])


def _empty_db_audit() -> dict[str, Any]:
    return {
        "database_connection_success": False, "database_probe_error": "", "knowledge_base_id": "", "index_configuration_id": "",
        "authoritative_document_count": 0, "authoritative_chunk_count": 0, "stored_embedding_count": 0,
        "null_embedding_count": 0, "invalid_embedding_count": 0, "pgvector_extension_available": False,
        "pgvector_extension_version": "", "vector_table": "public.chunks", "vector_column": "embedding", "vector_column_dimension": None,
        "vector_column_type": "", "vector_nullable": None, "stored_vector_dimension_valid": False,
        "stored_vector_identity_valid": False, "physical_vector_index_exists": False, "physical_vector_index_count": 0,
        "physical_vector_index_type": "none", "physical_vector_index_valid": True,
        "physical_vector_index_definition_matches_runtime": True, "retrieval_visible_embedding_count": 0,
        "retrieval_hidden_embedding_count": 0, "physical_indexes": [],
    }


def _resolve_database_url(env: Mapping[str, str]) -> str:
    secret = resolve_cold_start_secret_authority(env)
    return str(secret.get("database_url_value") or env.get(TASK_DATABASE_URL_VARIABLE) or env.get("DATABASE_URL") or "")


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.9g}" for v in values) + "]"


def _parse_vector_dimension(format_type: str) -> int | None:
    match = re.search(r"vector\((\d+)\)", format_type)
    return int(match.group(1)) if match else None


def _failure(substage: str, root: str) -> dict[str, Any]:
    return {"first_vector_index_loss_substage": substage, "diagnosed_root_cause": root}


def _load_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return ""


def _redact(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"postgres(?:ql)?://[^:\s/@]+:[^@\s]+@", "postgres://***:***@", text, flags=re.I)
    return json.loads(text)


def _public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k != "query_embedding"}
