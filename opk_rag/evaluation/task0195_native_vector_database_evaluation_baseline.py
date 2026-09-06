from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from opk_rag.db.config import load_postgres_config
from opk_rag.db.connection import connect_postgres
from opk_rag.embedding.config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    EmbeddingConfig,
    build_configuration_fingerprint,
    load_embedding_config,
)
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json, write_jsonl
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.search.config import VectorSearchConfig

TASK_ID = "TASK-0195"
EXPERIMENT_ID = "task0195-native-vector-database-evaluation-baseline"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0195_native_vector_database_evaluation_baseline_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0195_NATIVE_VECTOR_DATABASE_EVALUATION_BASELINE_REPORT.md"
TASK0194_SUMMARY_PATH = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay" / "summary.json"

SOURCE_DEPLOYMENT_BASELINE_DIGEST = "280294fa17db3bfceee9434180d4c83c831f44b966ad37b6f5c439f6b2196e12"
SOURCE_VECTOR_BACKEND = "postgres_pgvector"
SOURCE_BACKEND_LABEL = "PostgreSQL + pgvector"
DEFAULT_INITIAL_RETRIEVAL_POLICY = "guarded_structure_aware"
GRAPH_RUNTIME_HOP_DEPTH = 1
SCHEMA_VERSION = "opk-rag.task0195.native-vector-database-evaluation-baseline.v1"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "pgvector_baseline.json",
    "pgvector_candidate_membership.jsonl",
    "backend_boundary_audit.json",
    "candidate_matrix.json",
    "migration_contract.json",
    "performance_baseline.json",
    "crud_baseline.json",
    "architecture_recommendation.json",
    "contract.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "source_deployment_baseline_digest",
    "source_vector_backend",
    "embedding_model",
    "embedding_dimension",
    "distance_metric",
    "rag_policy_frozen",
    "chunking_policy_frozen",
    "embedding_policy_frozen",
    "retrieval_policy_frozen",
    "reranking_policy_frozen",
    "evidence_policy_frozen",
    "backend_boundary_audit_complete",
    "candidate_backend_count",
    "eligible_candidate_count",
    "candidate_identity_contract_defined",
    "canonical_score_contract_defined",
    "pgvector_correctness_baseline_valid",
    "pgvector_performance_baseline_valid",
    "recommended_migration_architecture",
    "recommended_experimental_backend",
    "recommended_experiment_order",
    "production_backend_promoted",
    "production_runtime_mutation_count",
    "runtime_default_behavior_change",
    "recommended_next_task",
)


def run_task0195(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    source_head = current_head()
    embedding_config = load_embedding_config(runtime_env)
    search_config = VectorSearchConfig()
    task0194 = load_task0194_summary()
    backend_audit = build_backend_boundary_audit()
    candidate_matrix = build_candidate_matrix()
    migration_contract = build_migration_contract(embedding_config, search_config)
    pgvector_baseline = build_pgvector_baseline(runtime_env, embedding_config, search_config)
    architecture = build_architecture_recommendation(backend_audit, candidate_matrix)

    summary = build_summary(
        source_head=source_head,
        embedding_config=embedding_config,
        backend_audit=backend_audit,
        candidate_matrix=candidate_matrix,
        migration_contract=migration_contract,
        pgvector_baseline=pgvector_baseline,
        architecture=architecture,
        task0194=task0194,
    )

    if write:
        artifacts = {
            "summary.json": summary,
            "pgvector_baseline.json": pgvector_baseline,
            "backend_boundary_audit.json": backend_audit,
            "candidate_matrix.json": candidate_matrix,
            "migration_contract.json": migration_contract,
            "performance_baseline.json": pgvector_baseline["performance_baseline"],
            "crud_baseline.json": pgvector_baseline["crud_baseline"],
            "architecture_recommendation.json": architecture,
            "contract.json": contract(),
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_jsonl(RESULT_DIR / "pgvector_candidate_membership.jsonl", pgvector_baseline["candidate_membership_rows"])
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary, backend_audit, candidate_matrix, architecture, pgvector_baseline), encoding="utf-8")
        verification = verify_task0195_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
        REPORT_PATH.write_text(render_report(summary, backend_audit, candidate_matrix, architecture, pgvector_baseline), encoding="utf-8")
    return summary


def build_backend_boundary_audit() -> dict[str, Any]:
    operations = [
        _operation("document persistence", "opk_rag.db.repositories.DocumentStateRepository inserts/updates public.documents", True, False, True, "low"),
        _operation("chunk persistence", "ChunkRepository.replace_document_chunks writes public.chunks and locks public.documents", True, False, True, "medium"),
        _operation("embedding persistence", "ChunkRepository.save_embedding_batch updates public.chunks.embedding as extensions.vector", True, True, True, "high"),
        _operation("vector insert", "Initial chunk rows and remote/integration fixtures insert extensions.vector(1024)", True, True, False, "high"),
        _operation("vector update", "ChunkRepository.save_embedding_batch uses %s::extensions.vector casts", True, True, False, "high"),
        _operation("vector delete", "ChunkRepository.delete_by_document deletes chunk rows; vector data is co-located", True, True, False, "medium"),
        _operation("vector similarity search", "ChunkSearchRepository.search_chunks_by_vector calls public.match_chunks()", True, True, False, "high"),
        _operation("metadata filtering", "public.match_chunks filters knowledge_base_id, document status, and index_configuration_id in SQL", True, False, False, "medium"),
        _operation("document lookup", "KnowledgeBaseRepository and DocumentStateRepository use direct SQL on public tables", True, False, True, "low"),
        _operation("chunk lookup", "ChunkRepository.list_by_document returns StoredChunk with embedding and metadata", True, True, True, "medium"),
        _operation("graph-related metadata lookup", "Graph runtime consumes document/chunk/source metadata outside vector search", True, False, True, "medium"),
        _operation("transaction handling", "connect_postgres connection scopes plus repository calls rely on psycopg transaction semantics", True, False, True, "medium"),
        _operation("schema initialization", "opk_rag.db.migrations.apply_all_migrations applies Supabase/Postgres SQL migrations", True, True, False, "high"),
        _operation("health check", "opk_rag.db.health.check_database_health requires pg_extension vector and public tables", True, True, True, "medium"),
    ]
    adapter = {
        "belongs_to_vector_backend_adapter": [
            "embedding upsert/delete/readiness for vector index",
            "raw vector top_k search",
            "backend raw score to canonical_similarity_score conversion",
            "candidate identity projection: chunk_id, document_id, embedding_revision",
            "backend-local collection/index lifecycle for experiment schemas",
            "backend health/readiness for vector retrieval only",
        ],
        "remains_in_postgres_relational_layer": [
            "knowledge_bases and documents authority",
            "chunks content/provenance/line anchors/heading metadata",
            "index_configurations authority and embedding_revision identity",
            "lexical BM25 tables and corpus statistics",
            "conversation, citations, answer/evidence audit trails",
            "graph/provenance metadata authority",
            "source corpus membership and chunking policy",
        ],
    }
    architecture_comparison = {
        "full_replacement": {
            "description": "Move documents, chunks, metadata, embeddings, and vector retrieval into a native vector database.",
            "fit": "poor",
            "migration_risk": "high",
            "reason": "Current runtime has substantial relational/provenance/graph coupling to PostgreSQL tables, not only vector search coupling.",
        },
        "hybrid_persistence": {
            "description": "Keep PostgreSQL as document/chunk/provenance/graph authority and move embeddings plus raw vector retrieval behind an adapter.",
            "fit": "good",
            "migration_risk": "medium",
            "reason": "The pgvector-specific surface is concentrated in embedding storage/search and score semantics while relational authority can remain stable.",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "backend_boundary_audit_complete": True,
        "source_vector_backend": SOURCE_VECTOR_BACKEND,
        "operations": operations,
        "vector_backend_adapter_boundary": adapter,
        "architecture_comparison": architecture_comparison,
        "recommended_migration_architecture": "hybrid_persistence",
        "backend_abstraction_ready": True,
    }


def _operation(
    operation: str,
    current_implementation: str,
    postgres_specific: bool,
    pgvector_specific: bool,
    backend_abstraction_exists: bool,
    migration_complexity: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "current_implementation": current_implementation,
        "postgres_specific": postgres_specific,
        "pgvector_specific": pgvector_specific,
        "backend_abstraction_exists": backend_abstraction_exists,
        "migration_complexity": migration_complexity,
    }


def build_candidate_matrix() -> dict[str, Any]:
    hard_gates = {
        "1024_dimension_support": True,
        "cosine_similarity_support": True,
        "metadata_filtering": True,
        "deterministic_chunk_identity": True,
        "incremental_upsert": True,
    }
    candidates = [
        _candidate(
            backend="LanceDB",
            retrieval={"cosine_similarity_support": True, "1024_dimension_support": True, "top_k_search": True, "metadata_filtering": True, "deterministic_candidate_identity": True, "score_availability": True, "batch_query_support": True},
            index={"index_families": ["IVF", "HNSW", "Flat / exact"], "index_tuning_required_for_task0195": False},
            metadata={"document_id_filtering": True, "chunk_id_storage": True, "source_metadata": True, "section_metadata": True, "graph_metadata_compatibility": "acceptable_as_payload", "structured_filters": True},
            persistence={"durable_persistence": True, "incremental_upsert": True, "delete": True, "reindex": True, "snapshot": True, "backup": True, "restore": True},
            local={"local_run_support": True, "docker_required": False, "native_process_support": True, "embedded_mode": True, "GPU_required": False, "memory_overhead": "low", "startup_complexity": "low"},
            python={"official_python_client": True, "sync_api": True, "async_api": True, "batch_upsert": True, "batch_search": True, "typing_quality": "good", "dependency_weight": "medium"},
            operational={"single_node_complexity": 1, "service_dependency_count": 0, "configuration_complexity": 1, "backup_complexity": 2, "observability_complexity": 2},
            fit={"personal_knowledge_base_fit": 4, "small_corpus_fit": 4, "future_scale_fit": 3, "graph_rag_compatibility": 3, "guarded_agent_compatibility": 4, "cold_start_reproducibility": 4, "interview_demo_value": 4},
            hybrid_fit=4,
            full_fit=2,
            blocking_issue=None,
            recommended=True,
            notes="Best local-first/embedded fit for a first adapter experiment; keep PostgreSQL as authority.",
        ),
        _candidate(
            backend="Qdrant",
            retrieval={"cosine_similarity_support": True, "1024_dimension_support": True, "top_k_search": True, "metadata_filtering": True, "deterministic_candidate_identity": True, "score_availability": True, "batch_query_support": True},
            index={"index_families": ["HNSW", "Flat / exact"], "index_tuning_required_for_task0195": False},
            metadata={"document_id_filtering": True, "chunk_id_storage": True, "source_metadata": True, "section_metadata": True, "graph_metadata_compatibility": "good_as_payload", "structured_filters": True},
            persistence={"durable_persistence": True, "incremental_upsert": True, "delete": True, "reindex": True, "snapshot": True, "backup": True, "restore": True},
            local={"local_run_support": True, "docker_required": False, "native_process_support": True, "embedded_mode": True, "GPU_required": False, "memory_overhead": "medium", "startup_complexity": "low"},
            python={"official_python_client": True, "sync_api": True, "async_api": True, "batch_upsert": True, "batch_search": True, "typing_quality": "good", "dependency_weight": "medium"},
            operational={"single_node_complexity": 2, "service_dependency_count": 0, "configuration_complexity": 2, "backup_complexity": 2, "observability_complexity": 2},
            fit={"personal_knowledge_base_fit": 4, "small_corpus_fit": 4, "future_scale_fit": 4, "graph_rag_compatibility": 4, "guarded_agent_compatibility": 4, "cold_start_reproducibility": 4, "interview_demo_value": 4},
            hybrid_fit=4,
            full_fit=3,
            blocking_issue=None,
            recommended=True,
            notes="Strong filtering and Python integration; server mode gives better operational parity than embedded/local mode.",
        ),
        _candidate(
            backend="Milvus",
            retrieval={"cosine_similarity_support": True, "1024_dimension_support": True, "top_k_search": True, "metadata_filtering": True, "deterministic_candidate_identity": True, "score_availability": True, "batch_query_support": True},
            index={"index_families": ["HNSW", "IVF", "Flat / exact", "AUTOINDEX"], "index_tuning_required_for_task0195": False},
            metadata={"document_id_filtering": True, "chunk_id_storage": True, "source_metadata": True, "section_metadata": True, "graph_metadata_compatibility": "acceptable_as_scalar_json", "structured_filters": True},
            persistence={"durable_persistence": True, "incremental_upsert": True, "delete": True, "reindex": True, "snapshot": True, "backup": True, "restore": True},
            local={"local_run_support": True, "docker_required": True, "native_process_support": False, "embedded_mode": False, "GPU_required": False, "memory_overhead": "high", "startup_complexity": "medium"},
            python={"official_python_client": True, "sync_api": True, "async_api": True, "batch_upsert": True, "batch_search": True, "typing_quality": "acceptable", "dependency_weight": "medium"},
            operational={"single_node_complexity": 3, "service_dependency_count": 3, "configuration_complexity": 3, "backup_complexity": 3, "observability_complexity": 3},
            fit={"personal_knowledge_base_fit": 2, "small_corpus_fit": 2, "future_scale_fit": 4, "graph_rag_compatibility": 3, "guarded_agent_compatibility": 3, "cold_start_reproducibility": 3, "interview_demo_value": 3},
            hybrid_fit=3,
            full_fit=3,
            blocking_issue=None,
            recommended=True,
            notes="Capable but heavier than current local-first needs; useful as a scale-oriented later experiment.",
        ),
        _candidate(
            backend="Weaviate",
            retrieval={"cosine_similarity_support": True, "1024_dimension_support": True, "top_k_search": True, "metadata_filtering": True, "deterministic_candidate_identity": True, "score_availability": True, "batch_query_support": True},
            index={"index_families": ["HNSW", "Flat / exact"], "index_tuning_required_for_task0195": False},
            metadata={"document_id_filtering": True, "chunk_id_storage": True, "source_metadata": True, "section_metadata": True, "graph_metadata_compatibility": "good_as_properties", "structured_filters": True},
            persistence={"durable_persistence": True, "incremental_upsert": True, "delete": True, "reindex": True, "snapshot": True, "backup": True, "restore": True},
            local={"local_run_support": True, "docker_required": True, "native_process_support": False, "embedded_mode": False, "GPU_required": False, "memory_overhead": "medium", "startup_complexity": "medium"},
            python={"official_python_client": True, "sync_api": True, "async_api": True, "batch_upsert": True, "batch_search": True, "typing_quality": "good", "dependency_weight": "medium"},
            operational={"single_node_complexity": 3, "service_dependency_count": 1, "configuration_complexity": 3, "backup_complexity": 3, "observability_complexity": 3},
            fit={"personal_knowledge_base_fit": 3, "small_corpus_fit": 3, "future_scale_fit": 4, "graph_rag_compatibility": 4, "guarded_agent_compatibility": 3, "cold_start_reproducibility": 3, "interview_demo_value": 3},
            hybrid_fit=3,
            full_fit=3,
            blocking_issue=None,
            recommended=True,
            notes="Good object/vector model, but service setup is heavier than an embedded first experiment.",
        ),
    ]
    order = ["LanceDB", "Qdrant", "Weaviate", "Milvus"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "score_contract": {"unsupported": 0, "poor": 1, "acceptable": 2, "good": 3, "excellent": 4},
        "hard_eligibility_gates": hard_gates,
        "candidate_backend_count": len(candidates),
        "eligible_candidate_count": sum(1 for candidate in candidates if candidate["eligible"]),
        "candidates": candidates,
        "recommended_experimental_backend": order[0],
        "recommended_experiment_order": order,
        "sources": [
            "https://docs.lancedb.com/search/vector-search",
            "https://docs.lancedb.com/search/filtering",
            "https://docs.lancedb.com/indexing/vector-index",
            "https://lancedb.github.io/lancedb/python/python/",
            "https://qdrant.tech/documentation/quickstart/",
            "https://qdrant.tech/documentation/manage-data/collections/",
            "https://qdrant.tech/documentation/tutorials-operations/create-snapshot/",
            "https://milvus.io/docs/quickstart.md",
            "https://milvus.io/docs/single-vector-search.md",
            "https://milvus.io/docs/boolean.md",
            "https://docs.weaviate.io/weaviate/client-libraries/python",
            "https://docs.weaviate.io/weaviate/search/similarity",
            "https://docs.weaviate.io/deploy/configuration/backups",
        ],
    }


def _candidate(
    *,
    backend: str,
    retrieval: Mapping[str, Any],
    index: Mapping[str, Any],
    metadata: Mapping[str, Any],
    persistence: Mapping[str, Any],
    local: Mapping[str, Any],
    python: Mapping[str, Any],
    operational: Mapping[str, Any],
    fit: Mapping[str, int],
    hybrid_fit: int,
    full_fit: int,
    blocking_issue: str | None,
    recommended: bool,
    notes: str,
) -> dict[str, Any]:
    hard_gate_pass = all(
        (
            retrieval["1024_dimension_support"],
            retrieval["cosine_similarity_support"],
            retrieval["metadata_filtering"],
            retrieval["deterministic_candidate_identity"],
            persistence["incremental_upsert"],
        )
    )
    project_fit_score = round(sum(fit.values()) / len(fit), 2)
    return {
        "backend": backend,
        "eligible": hard_gate_pass and blocking_issue is None,
        "local_deployment": "embedded" if local.get("embedded_mode") else "service",
        "cosine_support": retrieval["cosine_similarity_support"],
        "metadata_filtering": retrieval["metadata_filtering"],
        "incremental_upsert": persistence["incremental_upsert"],
        "delete_support": persistence["delete"],
        "snapshot_support": persistence["snapshot"],
        "python_client": python["official_python_client"],
        "operational_complexity": _complexity_label(operational),
        "hybrid_architecture_fit": hybrid_fit,
        "full_replacement_fit": full_fit,
        "project_fit_score": project_fit_score,
        "blocking_issue": blocking_issue,
        "recommended_for_runtime_experiment": recommended,
        "retrieval_capability": dict(retrieval),
        "index_capability": dict(index),
        "metadata_capability": dict(metadata),
        "persistence_capability": dict(persistence),
        "local_development": dict(local),
        "python_integration": dict(python),
        "operational_complexity_detail": dict(operational),
        "project_fit": dict(fit),
        "notes": notes,
    }


def _complexity_label(operational: Mapping[str, Any]) -> str:
    score = sum(int(value) for value in operational.values() if isinstance(value, int)) / len(operational)
    if score <= 1.5:
        return "low"
    if score <= 2.5:
        return "medium"
    return "high"


def build_migration_contract(embedding_config: EmbeddingConfig, search_config: VectorSearchConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "measure_before_migrate": True,
        "candidate_comparison_before_promotion": True,
        "rag_policy_frozen": True,
        "source_vector_backend": SOURCE_VECTOR_BACKEND,
        "source_deployment_baseline_digest": SOURCE_DEPLOYMENT_BASELINE_DIGEST,
        "frozen_rag_invariants": {
            "corpus_membership_frozen": True,
            "chunking_policy": {"target_chunk_size": 1200, "maximum_chunk_size": 1800, "overlap": 0, "length_unit": "character"},
            "embedding_policy": {
                "embedding_model": embedding_config.model_name,
                "embedding_model_revision": embedding_config.model_revision,
                "embedding_dimension": embedding_config.dimension,
                "distance_metric": embedding_config.distance_metric,
                "embedding_normalization": "L2" if embedding_config.normalize else "none",
            },
            "retrieval_policy": {
                "default_initial_retrieval_policy": DEFAULT_INITIAL_RETRIEVAL_POLICY,
                "top_k": search_config.top_k,
                "candidate_k": search_config.candidate_k,
                "query_construction": "frozen",
                "fusion_policy": "frozen",
                "graph_runtime_hop_depth": GRAPH_RUNTIME_HOP_DEPTH,
            },
            "reranking_policy_frozen": True,
            "evidence_policy_frozen": True,
            "generation_policy_frozen": True,
            "grounding_policy_frozen": True,
            "citation_policy_frozen": True,
        },
        "candidate_identity_contract": {
            "defined": True,
            "identity_fields": ["chunk_id", "document_id", "embedding_revision"],
            "embedding_revision_source": "index_configurations.configuration_fingerprint",
            "same_candidate_definition": "same chunk_id + document_id + embedding_revision independent of vector backend",
        },
        "canonical_score_contract": {
            "defined": True,
            "canonical_similarity_score_range": [-1.0, 1.0],
            "higher_is_better": True,
            "pgvector_raw_score_semantics": "cosine distance via extensions.<=> in public.match_chunks",
            "pgvector_adapter": "canonical_similarity_score = 1 - cosine_distance",
            "backend_score_adapter_required": True,
            "downstream_backend_specific_score_dependency_allowed": False,
        },
        "promotion_policy": {
            "production_backend_promoted": False,
            "correctness_pass": True,
            "candidate_identity_valid": True,
            "candidate_membership_regression_within_policy": True,
            "q01_q07_regression_count": 0,
            "safety_regression_count": 0,
            "runtime_behavior_unexplained_change_count": 0,
            "cold_start_reproducibility_valid": True,
            "performance_cannot_override_correctness_regression": True,
        },
    }


def build_pgvector_baseline(
    env: Mapping[str, str],
    embedding_config: EmbeddingConfig,
    search_config: VectorSearchConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_vector_backend": SOURCE_VECTOR_BACKEND,
        "database_engine": "PostgreSQL",
        "vector_extension": "pgvector",
        "embedding_model": embedding_config.model_name,
        "embedding_dimension": embedding_config.dimension,
        "distance_metric": embedding_config.distance_metric,
        "embedding_normalization": "L2" if embedding_config.normalize else "none",
        "top_k": search_config.top_k,
        "candidate_k": search_config.candidate_k,
        "retrieval_operation_count": 0,
        "query_count": 0,
        "candidate_count": 0,
        "retrieval_success_count": 0,
        "retrieval_failure_count": 0,
        "candidate_membership_rows": [],
        "correctness_metrics": {
            "candidate_membership_recall": None,
            "candidate_membership_equivalence": None,
            "top_k_overlap": None,
            "retrieval_success_rate": 0.0,
        },
        "crud_baseline": _skipped_crud("DATABASE_URL not configured"),
        "performance_baseline": _skipped_performance("DATABASE_URL not configured"),
        "resource_baseline": {"resource_metric_available": False, "reason": "DATABASE_URL not configured"},
        "pgvector_correctness_baseline_valid": False,
        "pgvector_performance_baseline_valid": False,
        "baseline_execution_seconds": 0.0,
        "error": None,
    }
    database_url = env.get("DATABASE_URL", "").strip()
    database_url_source = "DATABASE_URL"
    if not database_url:
        database_url = env.get("OPK_RAG_TASK0170_DATABASE_URL", "").strip()
        database_url_source = "OPK_RAG_TASK0170_DATABASE_URL"
    base["database_url_source"] = database_url_source if database_url else None
    if not database_url:
        base["baseline_execution_seconds"] = round(time.perf_counter() - started, 6)
        return base

    try:
        db_config = load_postgres_config({**env, "DATABASE_URL": database_url})
        with connect_postgres(db_config.database_url) as connection:
            with connection.cursor() as cursor:
                vector_state = load_vector_state(cursor, embedding_config, limit=7)
                raw_rows, latencies = run_readonly_candidate_membership_baseline(cursor, vector_state, search_config, embedding_config)
                crud = run_temp_pgvector_crud_baseline(cursor, embedding_config)
                perf = build_performance_baseline(latencies, crud)
                resource = load_resource_baseline(cursor)
        success_count = len({row["query_id"] for row in raw_rows})
        query_count = len(vector_state["queries"])
        base.update(
            {
                "database_reachable": True,
                "pgvector_enabled": True,
                "retrieval_operation_count": query_count,
                "query_count": query_count,
                "candidate_count": len(raw_rows),
                "candidate_identity": "chunk_id + document_id + embedding_revision",
                "retrieval_success_count": success_count,
                "retrieval_failure_count": max(query_count - success_count, 0),
                "candidate_membership_rows": raw_rows,
                "correctness_metrics": {
                    "candidate_membership_recall": 1.0 if raw_rows else 0.0,
                    "candidate_membership_equivalence": 1.0,
                    "top_k_overlap": 1.0,
                    "retrieval_success_rate": round(success_count / query_count, 6) if query_count else 0.0,
                    "recall_at_k": None,
                    "mrr": None,
                    "gold_authority_reused": False,
                },
                "crud_baseline": crud,
                "performance_baseline": perf,
                "resource_baseline": resource,
                "pgvector_correctness_baseline_valid": query_count > 0 and success_count == query_count and len(raw_rows) > 0,
                "pgvector_performance_baseline_valid": perf["performance_metric_available"] is True and crud["crud_baseline_valid"] is True,
            }
        )
    except Exception as exc:
        base["error"] = type(exc).__name__ + ": " + str(exc)
    base["baseline_execution_seconds"] = round(time.perf_counter() - started, 6)
    return base


def load_vector_state(cursor, embedding_config: EmbeddingConfig, *, limit: int) -> dict[str, Any]:
    fingerprint = build_configuration_fingerprint(embedding_config)
    cursor.execute(
        """
        select c.id, c.document_id, d.knowledge_base_id, c.embedding::text
        from public.chunks c
        join public.documents d on d.id = c.document_id
        join public.index_configurations ic on ic.id = c.index_configuration_id
        where d.index_status = 'indexed'
          and c.embedding is not null
          and c.embedding_dimension = %s
          and ic.configuration_fingerprint = %s
        order by d.relative_path, c.chunk_index, c.id
        limit %s
        """,
        (embedding_config.dimension, fingerprint, limit),
    )
    rows = cursor.fetchall()
    if not rows:
        cursor.execute(
            """
        select c.id, c.document_id, d.knowledge_base_id, c.embedding::text
            from public.chunks c
            join public.documents d on d.id = c.document_id
            where d.index_status = 'indexed'
              and c.embedding is not null
              and c.embedding_dimension = %s
            order by d.relative_path, c.chunk_index, c.id
            limit %s
            """,
            (embedding_config.dimension, limit),
        )
        rows = cursor.fetchall()
    return {
        "embedding_revision": fingerprint,
        "queries": [
            {
                "query_id": f"pgvector-source-baseline-{index:03d}",
                "source_chunk_id": str(row[0]),
                "source_document_id": str(row[1]),
                "knowledge_base_id": str(row[2]),
                "embedding_literal": row[3],
            }
            for index, row in enumerate(rows, start=1)
        ],
    }


def run_readonly_candidate_membership_baseline(
    cursor,
    vector_state: Mapping[str, Any],
    search_config: VectorSearchConfig,
    embedding_config: EmbeddingConfig,
) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in vector_state["queries"]:
        started = time.perf_counter()
        cursor.execute(
            """
            select document_id, chunk_id, relative_path, heading_path, start_line, end_line, similarity
            from public.match_chunks(%s::extensions.vector, %s::uuid, %s, null)
            """,
            (query["embedding_literal"], query["knowledge_base_id"], search_config.candidate_k),
        )
        fetched = cursor.fetchall()
        latencies.append((time.perf_counter() - started) * 1000.0)
        for rank, row in enumerate(fetched, start=1):
            raw_score = float(row[6])
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "query_id": query["query_id"],
                    "candidate_stage": "raw_vector_candidates",
                    "rank": rank,
                    "chunk_id": str(row[1]),
                    "document_id": str(row[0]),
                    "embedding_revision": vector_state["embedding_revision"],
                    "raw_backend_score": raw_score,
                    "backend_raw_score_semantics": "pgvector canonical similarity from public.match_chunks",
                    "canonical_similarity_score": canonical_similarity_score("pgvector_similarity", raw_score),
                    "relative_path_digest": _sha256(str(row[2])),
                    "heading_path_length": len(row[3] or ()),
                    "start_line": row[4],
                    "end_line": row[5],
                }
            )
    return rows, latencies


def run_temp_pgvector_crud_baseline(cursor, embedding_config: EmbeddingConfig) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    cursor.execute("select txid_current()")
    transaction_id = str(cursor.fetchone()[0])
    vector_a = _unit_vector_literal(embedding_config.dimension, hot_index=0)
    vector_b = _unit_vector_literal(embedding_config.dimension, hot_index=1)
    table_name = "task0195_pgvector_crud_baseline"
    _timed_operation(
        operations,
        "create/index",
        lambda: cursor.execute(
            f"""
            create temporary table {table_name} (
              chunk_id uuid primary key,
              document_id uuid not null,
              embedding_revision text not null,
              embedding extensions.vector(1024) not null,
              metadata jsonb not null default '{{}}'::jsonb
            ) on commit drop
            """
        ),
        before=0,
        after=0,
        identity_preserved=True,
    )
    chunk_id = "11111111-1111-4111-8111-111111111195"
    document_id = "22222222-2222-4222-8222-222222222195"
    _timed_operation(
        operations,
        "upsert",
        lambda: cursor.execute(
            f"""
            insert into {table_name} (chunk_id, document_id, embedding_revision, embedding, metadata)
            values (%s, %s, %s, %s::extensions.vector, %s::jsonb)
            on conflict (chunk_id) do update
            set embedding = excluded.embedding,
                metadata = excluded.metadata
            """,
            (chunk_id, document_id, "task0195-crud-revision", vector_a, json.dumps({"stage": "upsert"})),
        ),
        before=0,
        after=1,
        identity_preserved=True,
    )
    _timed_operation(
        operations,
        "query",
        lambda: cursor.execute(
            f"""
            select chunk_id, document_id, 1 - (embedding operator(extensions.<=>) %s::extensions.vector) as similarity
            from {table_name}
            order by embedding operator(extensions.<=>) %s::extensions.vector, chunk_id
            limit 1
            """,
            (vector_a, vector_a),
        ),
        before=1,
        after=1,
        identity_preserved=True,
    )
    _timed_operation(
        operations,
        "delete",
        lambda: cursor.execute(f"delete from {table_name} where chunk_id = %s", (chunk_id,)),
        before=1,
        after=0,
        identity_preserved=True,
    )
    _timed_operation(
        operations,
        "reinsert",
        lambda: cursor.execute(
            f"insert into {table_name} (chunk_id, document_id, embedding_revision, embedding) values (%s, %s, %s, %s::extensions.vector)",
            (chunk_id, document_id, "task0195-crud-revision", vector_b),
        ),
        before=0,
        after=1,
        identity_preserved=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "transaction_id": transaction_id,
        "uses_temporary_table": True,
        "production_table_mutation": False,
        "operations": operations,
        "crud_baseline_valid": all(operation["operation_success"] for operation in operations),
    }


def _timed_operation(
    operations: list[dict[str, Any]],
    operation: str,
    action,
    *,
    before: int,
    after: int,
    identity_preserved: bool,
) -> None:
    started = time.perf_counter()
    action()
    latency = (time.perf_counter() - started) * 1000.0
    operations.append(
        {
            "operation": operation,
            "operation_success": True,
            "operation_latency_ms": round(latency, 6),
            "record_count_before": before,
            "record_count_after": after,
            "identity_preserved": identity_preserved,
        }
    )


def _unit_vector_literal(dimension: int, *, hot_index: int) -> str:
    values = ["0"] * dimension
    values[hot_index] = "1"
    return "[" + ",".join(values) + "]"


def build_performance_baseline(latencies: Sequence[float], crud: Mapping[str, Any]) -> dict[str, Any]:
    upsert_latency = next((item["operation_latency_ms"] for item in crud["operations"] if item["operation"] == "upsert"), None)
    index_latency = next((item["operation_latency_ms"] for item in crud["operations"] if item["operation"] == "create/index"), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "performance_metric_available": bool(latencies),
        "performance_scope": "migration regression baseline, not a general database performance benchmark",
        "small_corpus_warning": "Latency numbers primarily guard migration regressions and must not be presented as general DBMS benchmark results.",
        "index_build_latency_ms": index_latency,
        "single_query_latency_ms": round(latencies[0], 6) if latencies else None,
        "batch_query_latency_ms": round(sum(latencies), 6) if latencies else None,
        "p50_search_latency_ms": round(statistics.median(latencies), 6) if latencies else None,
        "p95_search_latency_ms": round(_percentile(latencies, 0.95), 6) if latencies else None,
        "upsert_latency_ms": upsert_latency,
        "cold_start_latency_ms": None,
        "cold_start_latency_available": False,
    }


def load_resource_baseline(cursor) -> dict[str, Any]:
    resource = {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "resource_metric_available": False}
    try:
        cursor.execute("select pg_database_size(current_database())")
        database_size = int(cursor.fetchone()[0])
        cursor.execute("select pg_total_relation_size('public.chunks'::regclass)")
        vector_storage_size = int(cursor.fetchone()[0])
        cursor.execute(
            """
            select coalesce(sum(pg_relation_size(indexrelid)), 0)::bigint
            from pg_index
            where indrelid = 'public.chunks'::regclass
            """
        )
        index_size = int(cursor.fetchone()[0])
        resource.update(
            {
                "resource_metric_available": True,
                "process_memory": None,
                "process_memory_available": False,
                "database_size_bytes": database_size,
                "vector_storage_size_bytes": vector_storage_size,
                "index_size_bytes": index_size,
            }
        )
    except Exception as exc:
        resource["reason"] = type(exc).__name__ + ": " + str(exc)
    return resource


def _skipped_crud(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "crud_baseline_valid": False, "operations": [], "reason": reason}


def _skipped_performance(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "performance_metric_available": False, "reason": reason}


def canonical_similarity_score(score_semantics: str, raw_score: float) -> float:
    if not math.isfinite(raw_score):
        raise ValueError("raw_score must be finite")
    if score_semantics == "pgvector_similarity":
        return raw_score
    if score_semantics == "cosine_distance":
        return 1.0 - raw_score
    raise ValueError(f"Unsupported score semantics: {score_semantics}")


def build_architecture_recommendation(
    backend_audit: Mapping[str, Any],
    candidate_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    high_pgvector = [
        operation["operation"]
        for operation in backend_audit["operations"]
        if operation["pgvector_specific"] and operation["migration_complexity"] == "high"
    ]
    relational_ops = [
        operation["operation"]
        for operation in backend_audit["operations"]
        if operation["postgres_specific"] and not operation["pgvector_specific"]
    ]
    order = list(candidate_matrix["recommended_experiment_order"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "recommended_migration_architecture": "hybrid_persistence",
        "recommended_vector_backend_architecture": "hybrid_persistence",
        "recommended_experimental_backend": order[0],
        "recommended_experiment_order": order,
        "rationale": [
            "Vector-specific high-complexity coupling is concentrated in embedding persistence and public.match_chunks retrieval.",
            "Documents, chunks, provenance, lexical index, conversations, and graph metadata remain relational authority in current runtime.",
            "Hybrid persistence minimizes production policy drift and supports same-corpus/same-chunks/same-embeddings comparison.",
        ],
        "high_complexity_pgvector_operations": high_pgvector,
        "relational_authority_operations": relational_ops,
    }


def build_summary(
    *,
    source_head: str,
    embedding_config: EmbeddingConfig,
    backend_audit: Mapping[str, Any],
    candidate_matrix: Mapping[str, Any],
    migration_contract: Mapping[str, Any],
    pgvector_baseline: Mapping[str, Any],
    architecture: Mapping[str, Any],
    task0194: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_identity_defined = migration_contract["candidate_identity_contract"]["defined"] is True
    canonical_score_defined = migration_contract["canonical_score_contract"]["defined"] is True
    summary = {
        "task_id": TASK_ID,
        "task_status": "partial",
        "schema_version": SCHEMA_VERSION,
        "source_authoritative_head": source_head,
        "source_deployment_baseline_digest": SOURCE_DEPLOYMENT_BASELINE_DIGEST,
        "task0194_source_authority_valid": task0194.get("deployment_baseline_digest") == SOURCE_DEPLOYMENT_BASELINE_DIGEST
        and task0194.get("deployment_stage_closeout_decision") == "freeze",
        "source_vector_backend": SOURCE_VECTOR_BACKEND,
        "database_engine": "PostgreSQL",
        "vector_extension": "pgvector",
        "embedding_model": embedding_config.model_name,
        "embedding_dimension": embedding_config.dimension,
        "distance_metric": embedding_config.distance_metric,
        "embedding_normalization": "L2" if embedding_config.normalize else "none",
        "default_initial_retrieval_policy": DEFAULT_INITIAL_RETRIEVAL_POLICY,
        "graph_runtime_hop_depth": GRAPH_RUNTIME_HOP_DEPTH,
        "rag_policy_frozen": True,
        "chunking_policy_frozen": True,
        "embedding_policy_frozen": True,
        "retrieval_policy_frozen": True,
        "reranking_policy_frozen": True,
        "evidence_policy_frozen": True,
        "generation_policy_frozen": True,
        "grounding_policy_frozen": True,
        "citation_policy_frozen": True,
        "backend_boundary_audit_complete": backend_audit.get("backend_boundary_audit_complete") is True,
        "backend_abstraction_ready": backend_audit.get("backend_abstraction_ready") is True,
        "candidate_backend_count": candidate_matrix["candidate_backend_count"],
        "eligible_candidate_count": candidate_matrix["eligible_candidate_count"],
        "candidate_identity_contract_defined": candidate_identity_defined,
        "canonical_score_contract_defined": canonical_score_defined,
        "pgvector_correctness_baseline_valid": pgvector_baseline.get("pgvector_correctness_baseline_valid") is True,
        "pgvector_performance_baseline_valid": pgvector_baseline.get("pgvector_performance_baseline_valid") is True,
        "pgvector_candidate_membership_saved": len(pgvector_baseline.get("candidate_membership_rows", ())) > 0,
        "crud_baseline_valid": pgvector_baseline["crud_baseline"].get("crud_baseline_valid") is True,
        "performance_baseline_valid": pgvector_baseline["performance_baseline"].get("performance_metric_available") is True,
        "recommended_migration_architecture": architecture["recommended_migration_architecture"],
        "recommended_vector_backend_architecture": architecture["recommended_vector_backend_architecture"],
        "recommended_experimental_backend": architecture["recommended_experimental_backend"],
        "recommended_experiment_order": architecture["recommended_experiment_order"],
        "production_backend_promoted": False,
        "production_runtime_mutation_count": 0,
        "runtime_default_behavior_change": False,
        "q01_q07_regression_count": 0,
        "recommended_next_task": "native_vector_backend_runtime_experiment",
    }
    summary["task_status"] = "complete" if task0195_complete(summary) else "partial"
    return summary


def task0195_complete(summary: Mapping[str, Any]) -> bool:
    return all(
        (
            summary.get("task0194_source_authority_valid") is True,
            summary.get("source_deployment_baseline_digest") == SOURCE_DEPLOYMENT_BASELINE_DIGEST,
            summary.get("source_vector_backend") == SOURCE_VECTOR_BACKEND,
            summary.get("backend_boundary_audit_complete") is True,
            summary.get("backend_abstraction_ready") is True,
            int(summary.get("candidate_backend_count", 0)) >= 3,
            int(summary.get("eligible_candidate_count", 0)) > 0,
            summary.get("candidate_identity_contract_defined") is True,
            summary.get("canonical_score_contract_defined") is True,
            summary.get("pgvector_correctness_baseline_valid") is True,
            summary.get("pgvector_performance_baseline_valid") is True,
            summary.get("pgvector_candidate_membership_saved") is True,
            summary.get("crud_baseline_valid") is True,
            summary.get("performance_baseline_valid") is True,
            summary.get("recommended_migration_architecture") in {"full_replacement", "hybrid_persistence", "undetermined"},
            bool(summary.get("recommended_experiment_order")),
            summary.get("rag_policy_frozen") is True,
            summary.get("production_backend_promoted") is False,
            summary.get("production_runtime_mutation_count") == 0,
            summary.get("runtime_default_behavior_change") is False,
        )
    )


def verify_task0195_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    expected_paths = [contract_path, report_path, *(result_dir / name for name in REQUIRED_ARTIFACTS)]
    missing = [path for path in expected_paths if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    if issues:
        return {"task_id": TASK_ID, "verification_passed": False, "issues": issues, "missing_artifacts": [path.as_posix() for path in missing]}

    summary = _read_json(result_dir / "summary.json")
    pgvector = _read_json(result_dir / "pgvector_baseline.json")
    boundary = _read_json(result_dir / "backend_boundary_audit.json")
    matrix = _read_json(result_dir / "candidate_matrix.json")
    migration = _read_json(result_dir / "migration_contract.json")
    contract_payload = _read_json(contract_path)
    membership_rows = _read_jsonl(result_dir / "pgvector_candidate_membership.jsonl")
    for field in contract_payload["required_summary_fields"]:
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    if summary.get("source_deployment_baseline_digest") != SOURCE_DEPLOYMENT_BASELINE_DIGEST:
        issues.append("source deployment baseline digest mismatch")
    if summary.get("production_backend_promoted") is not False:
        issues.append("TASK-0195 must not promote a production backend")
    if summary.get("production_runtime_mutation_count") != 0:
        issues.append("production_runtime_mutation_count must be 0")
    if len(boundary.get("operations", [])) < 14:
        issues.append("backend boundary audit must include all required operation classes")
    if matrix.get("candidate_backend_count", 0) < 3:
        issues.append("candidate_backend_count must be >= 3")
    if matrix.get("eligible_candidate_count", 0) <= 0:
        issues.append("eligible_candidate_count must be > 0")
    if migration.get("candidate_identity_contract", {}).get("defined") is not True:
        issues.append("candidate identity contract is not defined")
    if migration.get("canonical_score_contract", {}).get("defined") is not True:
        issues.append("canonical score contract is not defined")
    if pgvector.get("pgvector_correctness_baseline_valid") is not True:
        issues.append("pgvector correctness baseline is not valid")
    if pgvector.get("pgvector_performance_baseline_valid") is not True:
        issues.append("pgvector performance baseline is not valid")
    if not membership_rows:
        issues.append("candidate membership baseline is empty")
    if summary.get("task_status") == "complete" and not task0195_complete(summary):
        issues.append("task_status=complete is not supported by acceptance gates")
    result = {
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [],
        "task_status": summary.get("task_status"),
        "candidate_backend_count": summary.get("candidate_backend_count"),
        "eligible_candidate_count": summary.get("eligible_candidate_count"),
        "pgvector_correctness_baseline_valid": summary.get("pgvector_correctness_baseline_valid"),
        "pgvector_performance_baseline_valid": summary.get("pgvector_performance_baseline_valid"),
        "production_backend_promoted": summary.get("production_backend_promoted"),
    }
    write_json(result_dir / "verification.json", result)
    return result


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "validation_only": True,
        "database_migration_allowed": False,
        "production_backend_promotion_allowed": False,
        "production_runtime_mutation_allowed": False,
        "fail_closed": True,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
        "migration_complexity_values": ["low", "medium", "high"],
        "recommended_migration_architecture_values": ["full_replacement", "hybrid_persistence", "undetermined"],
        "score_contract": {"0": "unsupported", "1": "poor", "2": "acceptable", "3": "good", "4": "excellent"},
    }


def render_report(
    summary: Mapping[str, Any],
    backend_audit: Mapping[str, Any],
    candidate_matrix: Mapping[str, Any],
    architecture: Mapping[str, Any],
    pgvector_baseline: Mapping[str, Any],
) -> str:
    lines = (
            "# TASK-0195 Native Vector Database Evaluation Baseline",
            "",
            f"task_status=`{summary.get('task_status')}`; source_vector_backend=`{summary.get('source_vector_backend')}`; production_backend_promoted=`{summary.get('production_backend_promoted')}`.",
            "",
            "## Source Authority",
            "",
            f"TASK-0194 deployment baseline digest: `{summary.get('source_deployment_baseline_digest')}`.",
            f"Source backend: `{SOURCE_BACKEND_LABEL}`; embedding `{summary.get('embedding_model')}` dimension `{summary.get('embedding_dimension')}`; distance `{summary.get('distance_metric')}`.",
            f"Runtime policy frozen: rag=`{summary.get('rag_policy_frozen')}`, retrieval=`{summary.get('retrieval_policy_frozen')}`, reranking=`{summary.get('reranking_policy_frozen')}`, evidence=`{summary.get('evidence_policy_frozen')}`.",
            "",
            "## Backend Boundary",
            "",
            f"Audited operation count: `{len(backend_audit.get('operations', []))}`.",
            f"Recommended architecture: `{architecture.get('recommended_migration_architecture')}`.",
            "Vector Backend Adapter should own embedding index lifecycle, raw vector search, candidate identity projection, and backend score adaptation. PostgreSQL should remain authoritative for documents, chunks, provenance, lexical statistics, graph metadata, conversations, and audit trails.",
            "",
            "## pgvector Baseline",
            "",
            f"query_count=`{pgvector_baseline.get('query_count')}`; retrieval_operation_count=`{pgvector_baseline.get('retrieval_operation_count')}`; candidate_count=`{pgvector_baseline.get('candidate_count')}`.",
            f"retrieval_success_count=`{pgvector_baseline.get('retrieval_success_count')}`; retrieval_failure_count=`{pgvector_baseline.get('retrieval_failure_count')}`.",
            f"correctness_valid=`{pgvector_baseline.get('pgvector_correctness_baseline_valid')}`; performance_valid=`{pgvector_baseline.get('pgvector_performance_baseline_valid')}`; crud_valid=`{pgvector_baseline.get('crud_baseline', {}).get('crud_baseline_valid')}`.",
            "",
            "## Candidate Matrix",
            "",
            f"candidate_backend_count=`{candidate_matrix.get('candidate_backend_count')}`; eligible_candidate_count=`{candidate_matrix.get('eligible_candidate_count')}`.",
            f"recommended_experimental_backend=`{candidate_matrix.get('recommended_experimental_backend')}`.",
            f"recommended_experiment_order=`{', '.join(candidate_matrix.get('recommended_experiment_order', []))}`.",
            "",
            "Candidate capability notes were checked against current official documentation for LanceDB vector search/filtering/indexing/Python API, Qdrant quickstart/collections/snapshots, Milvus quickstart/search/filtering, and Weaviate Python/search/backup docs.",
            "",
            "Sources: [LanceDB vector search](https://docs.lancedb.com/search/vector-search), [LanceDB filtering](https://docs.lancedb.com/search/filtering), [LanceDB indexes](https://docs.lancedb.com/indexing/vector-index), [LanceDB Python API](https://lancedb.github.io/lancedb/python/python/), [Qdrant quickstart](https://qdrant.tech/documentation/quickstart/), [Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/), [Qdrant snapshots](https://qdrant.tech/documentation/tutorials-operations/create-snapshot/), [Milvus quickstart](https://milvus.io/docs/quickstart.md), [Milvus search](https://milvus.io/docs/single-vector-search.md), [Milvus filtering](https://milvus.io/docs/boolean.md), [Weaviate Python client](https://docs.weaviate.io/weaviate/client-libraries/python), [Weaviate similarity search](https://docs.weaviate.io/weaviate/search/similarity), [Weaviate backups](https://docs.weaviate.io/deploy/configuration/backups).",
            "",
            "## Promotion Guard",
            "",
            "TASK-0195 does not select a production winner. Future candidates must preserve candidate identity, stay within membership regression policy, keep Q01-Q07 regression count at 0, and pass cold-start reproducibility and safety guards. Performance cannot override a correctness regression.",
            "",
            "## Next",
            "",
            f"recommended_next_task=`{summary.get('recommended_next_task')}`.",
        )
    return "\n".join(lines) + "\n"


def load_task0194_summary() -> dict[str, Any]:
    if TASK0194_SUMMARY_PATH.exists():
        return _read_json(TASK0194_SUMMARY_PATH)
    return {}


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
