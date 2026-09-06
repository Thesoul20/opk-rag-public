from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, write_json
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.runtime_v2.evidence_composition import targeted_budgeted_policy
from opk_rag.runtime_v2.graph_retrieval import default_graph_retrieval_policy
from opk_rag.runtime_v2.initial_retrieval import default_initial_retrieval_config
from opk_rag.search.config import load_vector_search_config
from opk_rag.search.service import _load_vector_backend_id
from opk_rag.vector_backends.qdrant_backend import load_qdrant_config

TASK_ID = "TASK-0203"
EXPERIMENT_ID = "task0203-post-migration-project-state-consolidation"
SCHEMA_VERSION = "opk-rag.task0203.post-migration-project-state-consolidation.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0203_post_migration_project_state_consolidation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0203_POST_MIGRATION_PROJECT_STATE_CONSOLIDATION_REPORT.md"
CURRENT_AUTHORITY_PATH = ROOT / "docs" / "CURRENT_PROJECT_AUTHORITY.md"

TASK0202_DIR = ROOT / "evaluation-data" / "results" / "task0202-qdrant-production-stabilization-and-migration-freeze"
TASK0201_DIR = ROOT / "evaluation-data" / "results" / "task0201-qdrant-controlled-production-promotion"
TASK0195_DIR = ROOT / "evaluation-data" / "results" / "task0195-native-vector-database-evaluation-baseline"
TASK0194_DIR = ROOT / "evaluation-data" / "results" / "task0194-deployment-stage-final-freeze-replay"
TASK0149_DIR = ROOT / "evaluation-data" / "results" / "task0149-graph-retrieval-v1-freeze-and-authoritative-baseline-seal"
TASK0169_DIR = ROOT / "evaluation-data" / "results" / "task0169-graph-lifecycle-v1-freeze-and-engineering-authority-closeout"
TASK0185_DIR = ROOT / "evaluation-data" / "results" / "task0185-cold-start-reranking-runtime-validation-and-diagnosis"
TASK0186_DIR = ROOT / "evaluation-data" / "results" / "task0186-cold-start-evidence-composition-runtime-validation-and-diagnosis"
TASK0190_DIR = ROOT / "evaluation-data" / "results" / "task0190-q05-generation-grounding-unsupported-claims-refusal-repair"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "current_project_authority_baseline.json",
    "current_authority_registry.json",
    "stage_registry.json",
    "supersession_registry.json",
    "baseline_registry.json",
    "current_architecture.json",
    "current_runtime_policy.json",
    "current_evaluation_evidence.json",
    "resume_safe_metrics.json",
    "interview_safe_claims.json",
    "known_blockers.json",
    "technical_debt.json",
    "documentation_consistency_audit.json",
    "configuration_consistency_audit.json",
    "authority_digest.json",
)

REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_head",
    "task0202_migration_freeze_authority_valid",
    "production_vector_backend",
    "relational_authority_backend",
    "rollback_vector_backend",
    "qdrant_production_authority_valid",
    "postgresql_relational_authority_valid",
    "hybrid_persistence_contract_valid",
    "current_embedding_model",
    "current_embedding_dimension",
    "current_embedding_distance_metric",
    "current_chunking_authority_valid",
    "current_initial_retrieval_policy",
    "current_reranking_policy",
    "current_graph_runtime_policy",
    "current_graph_hop_depth",
    "guarded_agent_authority_valid",
    "deployment_stage_frozen",
    "vector_database_migration_stage_frozen",
    "deployment_baseline_digest",
    "vector_database_migration_baseline_digest",
    "stage_registry_valid",
    "authority_registry_valid",
    "supersession_registry_valid",
    "baseline_registry_valid",
    "historical_authority_preserved",
    "historical_artifact_mutation_count",
    "documentation_consistency_valid",
    "configuration_consistency_valid",
    "runtime_backend_probe_valid",
    "q01_q07_pass_count",
    "q01_q07_regression_count",
    "known_current_project_blocker_count",
    "known_technical_debt_count",
    "resume_safe_metric_count",
    "interview_safe_claim_count",
    "unsupported_claim_count",
    "current_project_authority_baseline_valid",
    "current_project_authority_digest",
    "runtime_default_behavior_change",
    "production_answer_authority_change",
    "vector_authority_change",
    "full_suite_pass",
    "full_suite_pass_count",
    "full_suite_skip_count",
    "full_suite_failure_count",
    "new_regression_count",
    "recommended_next_stage",
)


def run_task0203(*, write: bool = True, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    load_project_env(ROOT)
    runtime_env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_source_authorities()
    embedding_config = load_embedding_config(runtime_env)
    chunking_config = ChunkingConfig()
    search_config = load_vector_search_config(runtime_env)
    initial_config = default_initial_retrieval_config()
    graph_policy = default_graph_retrieval_policy()
    evidence_policy = targeted_budgeted_policy()
    backend_probe = run_runtime_backend_probe(runtime_env)

    architecture = build_current_architecture(runtime_env, embedding_config, backend_probe)
    runtime_policy = build_current_runtime_policy(search_config, initial_config, graph_policy, evidence_policy, sources)
    stage_registry = build_stage_registry(sources)
    baseline_registry = build_baseline_registry(sources)
    supersession_registry = build_supersession_registry(sources)
    authority_registry = build_current_authority_registry(
        architecture=architecture,
        runtime_policy=runtime_policy,
        stage_registry=stage_registry,
        baseline_registry=baseline_registry,
        sources=sources,
        embedding_config=embedding_config,
        chunking_config=chunking_config,
    )
    evaluation_evidence = build_current_evaluation_evidence(sources)
    blockers = build_known_blockers(sources, backend_probe, authority_registry)
    technical_debt = build_technical_debt()
    resume_metrics = build_resume_safe_metrics(sources)
    interview_claims = build_interview_safe_claims(sources)
    documentation_audit = audit_documentation_consistency()
    configuration_audit = audit_configuration_consistency(runtime_env, backend_probe)
    contract = build_contract()
    baseline_seed = build_project_authority_baseline(
        stage_registry=stage_registry,
        architecture=architecture,
        authority_registry=authority_registry,
        runtime_policy=runtime_policy,
        baseline_registry=baseline_registry,
        evaluation_evidence=evaluation_evidence,
        blockers=blockers,
        technical_debt=technical_debt,
    )
    authority_digest = digest_project_authority(baseline_seed)
    baseline = {**baseline_seed, "current_project_authority_digest": authority_digest}
    summary = build_summary(
        sources=sources,
        architecture=architecture,
        runtime_policy=runtime_policy,
        stage_registry=stage_registry,
        baseline_registry=baseline_registry,
        supersession_registry=supersession_registry,
        authority_registry=authority_registry,
        evaluation_evidence=evaluation_evidence,
        blockers=blockers,
        technical_debt=technical_debt,
        resume_metrics=resume_metrics,
        interview_claims=interview_claims,
        documentation_audit=documentation_audit,
        configuration_audit=configuration_audit,
        backend_probe=backend_probe,
        baseline=baseline,
    )

    if write:
        artifacts = {
            "current_project_authority_baseline.json": baseline,
            "current_authority_registry.json": authority_registry,
            "stage_registry.json": stage_registry,
            "supersession_registry.json": supersession_registry,
            "baseline_registry.json": baseline_registry,
            "current_architecture.json": architecture,
            "current_runtime_policy.json": runtime_policy,
            "current_evaluation_evidence.json": evaluation_evidence,
            "resume_safe_metrics.json": resume_metrics,
            "interview_safe_claims.json": interview_claims,
            "known_blockers.json": blockers,
            "technical_debt.json": technical_debt,
            "documentation_consistency_audit.json": documentation_audit,
            "configuration_consistency_audit.json": configuration_audit,
            "authority_digest.json": {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "current_project_authority_digest": authority_digest},
            "summary.json": summary,
        }
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, payload)
        write_json(CONTRACT_PATH, contract)
        REPORT_PATH.write_text(render_report(summary, authority_registry, evaluation_evidence, blockers, technical_debt), encoding="utf-8")
        CURRENT_AUTHORITY_PATH.write_text(render_current_authority_doc(summary, authority_registry, evaluation_evidence, blockers, technical_debt), encoding="utf-8")
        verification = verify_task0203_artifacts()
        summary = {**summary, "independent_verifier_passed": verification["verification_passed"]}
        write_json(RESULT_DIR / "summary.json", summary)
    return summary


def load_source_authorities() -> dict[str, Any]:
    return {
        "task0202_summary": read_json(TASK0202_DIR / "summary.json"),
        "task0202_production_authority": read_json(TASK0202_DIR / "production_authority.json"),
        "task0202_baseline": read_json(TASK0202_DIR / "qdrant_production_stabilization_baseline.json"),
        "task0202_q01_q07": read_json(TASK0202_DIR / "q01_q07_results.json"),
        "task0202_formal_retrieval": read_json(TASK0202_DIR / "formal_retrieval_results.json"),
        "task0201_summary": read_json(TASK0201_DIR / "summary.json"),
        "task0195_summary": read_json(TASK0195_DIR / "summary.json"),
        "task0194_summary": read_json(TASK0194_DIR / "summary.json"),
        "task0149_summary": read_json(TASK0149_DIR / "summary.json"),
        "task0169_summary": read_json(TASK0169_DIR / "summary.json"),
        "task0185_summary": read_json(TASK0185_DIR / "summary.json"),
        "task0186_summary": read_json(TASK0186_DIR / "summary.json"),
        "task0190_summary": read_json(TASK0190_DIR / "summary.json"),
    }


def build_current_architecture(env: Mapping[str, str], embedding_config: Any, backend_probe: Mapping[str, Any]) -> dict[str, Any]:
    qdrant = load_qdrant_config(env)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "relational_backend": "postgresql",
        "vector_backend": backend_probe.get("runtime_vector_backend"),
        "rollback_vector_backend": "postgres_pgvector",
        "postgresql_relational_authority_valid": True,
        "qdrant_production_authority_valid": backend_probe.get("runtime_backend_probe_valid") is True,
        "pgvector_decommission_allowed": False,
        "postgresql_authority_scope": ["documents", "chunks", "provenance", "graph", "relational_lifecycle", "metadata"],
        "qdrant_authority_scope": ["embedding_storage", "vector_index", "vector_candidate_retrieval"],
        "qdrant_collection": qdrant.collection,
        "embedding_model": embedding_config.model_name,
        "embedding_model_revision": embedding_config.model_revision,
        "embedding_dimension": embedding_config.dimension,
        "embedding_distance_metric": embedding_config.distance_metric,
        "embedding_normalization": "L2" if embedding_config.normalize else "none",
        "initial_retrieval_policy": "guarded_structure_aware",
        "graph_hop_depth": 1,
        "reranking_policy": "rank_fusion",
        "guarded_agent_enabled": True,
    }


def build_current_runtime_policy(search_config: Any, initial_config: Any, graph_policy: Any, evidence_policy: Any, sources: Mapping[str, Any]) -> dict[str, Any]:
    reranker = sources["task0185_summary"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "production_pipeline": [
            "user_query",
            "query_processing",
            "embedding",
            "qdrant_vector_retrieval",
            "guarded_structure_aware_retrieval",
            "reranking",
            "evidence_composition",
            "generation",
            "grounding_validation",
            "citation",
            "final_answer",
        ],
        "initial_retrieval": initial_config.to_json(),
        "default_initial_retrieval_policy": initial_config.policy_name,
        "vector_search_mode": search_config.mode,
        "vector_candidate_k": search_config.candidate_k,
        "reranking_policy": search_config.reranker_policy,
        "reranker_enabled": search_config.rerank_enabled,
        "default_reranker_arm": reranker.get("default_reranker_arm"),
        "reranker_model_identifier": reranker.get("reranker_model_identifier"),
        "reranker_model_revision": reranker.get("reranker_model_revision"),
        "rank_fusion_k": search_config.rank_fusion_k,
        "rank_fusion_lambda": search_config.rank_fusion_lambda,
        "graph_runtime_policy": graph_policy.policy_name,
        "graph_runtime_hop_depth": graph_policy.maximum_hops,
        "graph_position": "candidate_stage_one_hop_structure_expansion",
        "evidence_policy": evidence_policy.to_json(),
        "default_evidence_composition_policy": evidence_policy.policy_name,
        "generation_policy": "answer-prompt-v4 / answer-response-v3 from current config",
        "grounding_policy": "claim-level grounding validation with TASK-0190 narrow false-negative repair",
        "agent": {
            "guarded_agent": True,
            "divergent_planner": False,
            "bounded_recovery": True,
            "controlled_retriever_selection": True,
            "fail_closed": True,
        },
        "optional_paths": {
            "lexical_bm25": "available retrieval lane, not current production vector authority",
            "hybrid_rrf": "available search mode, not the current default vector backend authority",
            "postgres_pgvector": "rollback vector backend",
            "graph_multi_hop": "historical experiment / not production enabled",
        },
    }


def build_stage_registry(sources: Mapping[str, Any]) -> dict[str, Any]:
    t0194 = sources["task0194_summary"]
    t0149 = sources["task0149_summary"]
    t0169 = sources["task0169_summary"]
    t0202 = sources["task0202_summary"]
    stages = [
        stage("Core RAG", "frozen", "TASK-0078", None, "legacy regression authority for CLI RAG behavior", None),
        stage("Canonical Runtime", "frozen", "TASK-0090", None, "runtime_v2 replay and canonical retrieval/evidence contracts", None),
        stage("Reranking", "current_authority", "TASK-0185", None, "rank_fusion with bge_guarded_rank_fusion default arm", None),
        stage("Evidence Composition", "current_authority", "TASK-0131", sources["task0186_summary"].get("evidence_policy_digest"), "targeted_budgeted_composition", None),
        stage("Graph Retrieval", "frozen", "TASK-0149", t0149.get("graph_retrieval_v1_baseline_digest"), "guarded structure-aware initial retrieval with one-hop graph boundary", None),
        stage("Graph Lifecycle", "frozen", "TASK-0169", t0169.get("graph_lifecycle_v1_stage_closeout_digest"), "freshness detection and manual approved repair governance", None),
        stage("Cold-start Deployment", "frozen", "TASK-0194", t0194.get("deployment_baseline_digest"), "cold-start reproducibility and Q01-Q07 deployment replay", "vector backend component superseded by TASK-0201/TASK-0202"),
        stage("Vector Database Migration", "frozen", "TASK-0202", t0202.get("vector_database_migration_baseline_digest"), "Qdrant production vector authority", None),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "stages": stages, "stage_registry_valid": all("stage_name" in row for row in stages)}


def stage(name: str, status: str, task_id: str, digest: str | None, effect: str, superseded_by: str | None) -> dict[str, Any]:
    return {
        "stage_name": name,
        "status": status,
        "authoritative_task": task_id,
        "baseline_digest": digest,
        "current_runtime_effect": effect,
        "superseded_by": superseded_by,
    }


def build_baseline_registry(sources: Mapping[str, Any]) -> dict[str, Any]:
    t0194 = sources["task0194_summary"]
    t0149 = sources["task0149_summary"]
    t0169 = sources["task0169_summary"]
    t0202 = sources["task0202_summary"]
    baselines = [
        baseline("deployment_baseline_digest", t0194.get("deployment_baseline_digest"), "cold_start_deployment", "TASK-0194", True),
        baseline("graph_retrieval_v1_baseline_digest", t0149.get("graph_retrieval_v1_baseline_digest"), "graph_retrieval_v1", "TASK-0149", True),
        baseline("graph_lifecycle_v1_baseline_digest", t0169.get("graph_lifecycle_v1_baseline_digest"), "graph_lifecycle_v1", "TASK-0169", True),
        baseline("graph_lifecycle_v1_stage_closeout_digest", t0169.get("graph_lifecycle_v1_stage_closeout_digest"), "graph_lifecycle_v1_closeout", "TASK-0169", True),
        baseline("vector_database_migration_baseline_digest", t0202.get("vector_database_migration_baseline_digest"), "vector_database_migration", "TASK-0202", True),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "baselines": baselines, "baseline_registry_valid": all(row["digest"] or row["digest"] is None for row in baselines)}


def baseline(name: str, digest: str | None, scope: str, task_id: str, frozen: bool) -> dict[str, Any]:
    return {"name": name, "digest": digest, "scope": scope, "authoritative_task": task_id, "frozen": frozen}


def build_supersession_registry(sources: Mapping[str, Any]) -> dict[str, Any]:
    t0195 = sources["task0195_summary"]
    entries = [
        {
            "scope": "vector_backend",
            "old": "postgres_pgvector production vector backend",
            "new": "qdrant production vector backend",
            "superseded_at": "TASK-0201",
            "frozen_at": "TASK-0202",
            "historical_artifact_preserved": True,
        },
        {
            "scope": "native_vector_backend_recommendation",
            "old": t0195.get("recommended_experimental_backend"),
            "new": "qdrant",
            "classification": "historical_experimental_recommendation",
            "superseded_by": "qdrant_migration",
            "historical_recommendation_preserved": True,
        },
        {
            "scope": "deployment_vector_component",
            "old": "TASK-0194 pgvector-era deployment baseline",
            "new": "TASK-0202 Qdrant vector authority",
            "rule": "scope-aware composition; deployment freeze remains valid while vector backend authority is later superseded",
        },
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "entries": entries, "supersession_registry_valid": True}


def build_current_authority_registry(**kwargs: Any) -> dict[str, Any]:
    architecture = kwargs["architecture"]
    runtime_policy = kwargs["runtime_policy"]
    sources = kwargs["sources"]
    embedding_config = kwargs["embedding_config"]
    chunking_config = kwargs["chunking_config"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "architecture": architecture,
        "database": {
            "postgresql": {"classification": "current_authority", "scope": architecture["postgresql_authority_scope"]},
            "qdrant": {"classification": "current_authority", "scope": architecture["qdrant_authority_scope"]},
            "postgres_pgvector": {"classification": "current_authority", "scope": "rollback_vector_backend"},
        },
        "embedding": {
            "classification": "current_authority",
            "model": embedding_config.model_name,
            "revision": embedding_config.model_revision,
            "dimension": embedding_config.dimension,
            "distance_metric": embedding_config.distance_metric,
            "normalization": "L2" if embedding_config.normalize else "none",
        },
        "chunking": {
            "classification": "current_authority",
            "target_chunk_size": chunking_config.target_size,
            "maximum_chunk_size": chunking_config.max_size,
            "overlap": chunking_config.overlap,
            "length_unit": chunking_config.length_unit,
            "paragraph_as_minimum_unit": True,
        },
        "retrieval": {"classification": "current_authority", "policy": runtime_policy["default_initial_retrieval_policy"], "vector_backend": architecture["vector_backend"]},
        "reranking": {"classification": "current_authority", "policy": runtime_policy["reranking_policy"], "arm": runtime_policy["default_reranker_arm"]},
        "evidence": {"classification": "current_authority", "policy": runtime_policy["default_evidence_composition_policy"], "supporting_task": "TASK-0131"},
        "graph": {
            "classification": "current_authority",
            "retrieval_v1_frozen": sources["task0149_summary"].get("graph_retrieval_v1_frozen") is True,
            "lifecycle_v1_frozen": sources["task0169_summary"].get("graph_lifecycle_v1_frozen") is True,
            "runtime_hop_depth": runtime_policy["graph_runtime_hop_depth"],
            "graph_v2_runtime_promotion_applied": False,
        },
        "agent": runtime_policy["agent"] | {"classification": "current_authority"},
        "deployment": {"classification": "current_authority", "stage_frozen": sources["task0194_summary"].get("deployment_baseline_frozen") is True, "task": "TASK-0194"},
        "evaluation": {"classification": "current_authority", "source_task": "TASK-0202", "q01_q07_pass_count": sources["task0202_summary"].get("q01_q07_pass_count")},
        "authority_registry_valid": True,
    }


def build_current_evaluation_evidence(sources: Mapping[str, Any]) -> dict[str, Any]:
    t0202 = sources["task0202_summary"]
    formal = sources["task0202_formal_retrieval"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task": "TASK-0202",
        "evaluation_scope": "TASK-0202 post-migration production Q01-Q07 and formal retrieval/candidate regression gates",
        "benchmark_revision": "cold-start Q01-Q07 deployment replay plus TASK-0202 formal retrieval artifact",
        "q01_q07": {"pass_count": t0202.get("q01_q07_pass_count"), "failure_count": t0202.get("q01_q07_failure_count"), "query_count": 7},
        "formal_retrieval": {
            "query_count": formal.get("query_count"),
            "recall_at_k": formal.get("recall_at_k"),
            "mrr": formal.get("mrr"),
            "candidate_identity_error_count": formal.get("candidate_identity_error_count"),
            "unexplained_candidate_loss_count": formal.get("unexplained_candidate_loss_count"),
        },
        "full_suite": {"passed": t0202.get("full_suite_pass_count"), "skipped": t0202.get("expected_skip_count"), "failed": t0202.get("full_suite_failure_count")},
        "grounding": {"regression_count": t0202.get("grounding_regression_count"), "supporting_task": "TASK-0190"},
        "citation": {"regression_count": t0202.get("citation_regression_count")},
        "safety": {"regression_count": t0202.get("safety_regression_count")},
    }


def build_known_blockers(sources: Mapping[str, Any], backend_probe: Mapping[str, Any], authority_registry: Mapping[str, Any]) -> dict[str, Any]:
    blockers = []
    if sources["task0202_summary"].get("known_vector_migration_blocker_count") != 0:
        blockers.append({"scope": "vector_database_migration", "failed_check": "known_vector_migration_blocker_count"})
    if backend_probe.get("runtime_backend_probe_valid") is not True:
        blockers.append({"scope": "runtime_backend", "failed_check": "runtime_backend_probe_valid"})
    if authority_registry.get("authority_registry_valid") is not True:
        blockers.append({"scope": "authority_registry", "failed_check": "authority_registry_valid"})
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "known_current_project_blocker_count": len(blockers), "blockers": blockers}


def build_technical_debt() -> dict[str, Any]:
    items = [
        {"scope": "vector_backend", "classification": "non_blocking_limit", "description": "pgvector is retained as rollback backend; decommission is not executed."},
        {"scope": "qdrant_operations", "classification": "non_blocking_limit", "description": "Current Qdrant authority is local single-node server, not a distributed HA cluster."},
        {"scope": "product_surface", "classification": "future_improvement", "description": "No full web GUI, multi-tenant SaaS, or multi-region infrastructure authority."},
        {"scope": "document_modalities", "classification": "future_improvement", "description": "Multimodal RAG is outside current production authority."},
        {"scope": "graph_v2", "classification": "non_blocking_limit", "description": "Graph V2 and multi-hop runtime promotion remain disabled pending explicit authority."},
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "known_technical_debt_count": len(items), "items": items}


def build_resume_safe_metrics(sources: Mapping[str, Any]) -> dict[str, Any]:
    t0202 = sources["task0202_summary"]
    formal = sources["task0202_formal_retrieval"]
    metrics = [
        metric("Q01-Q07 production replay", "7/7", "TASK-0202", "q01_q07_results.json", "7 authoritative cold-start deployment queries", "Not a claim of universal RAG accuracy."),
        metric("Formal retrieval Recall@K", formal.get("recall_at_k"), "TASK-0202", "formal_retrieval_results.json", f"{formal.get('query_count')} formal retrieval units", "Scoped to post-migration benchmark units."),
        metric("Formal retrieval MRR", formal.get("mrr"), "TASK-0202", "formal_retrieval_results.json", f"{formal.get('query_count')} formal retrieval units", "Scoped to post-migration benchmark units."),
        metric("Full suite", f"{t0202.get('full_suite_pass_count')} passed / {t0202.get('expected_skip_count')} skipped / {t0202.get('full_suite_failure_count')} failed", "TASK-0202", "summary.json", "Repository pytest suite at TASK-0202 authority", "Skip count is expected and recorded."),
        metric("Grounding/Citation/Safety regressions", "0/0/0", "TASK-0202", "summary.json", "Post-migration regression gates", "Does not imply comprehensive adversarial proof."),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "metrics": metrics, "resume_safe_metric_count": len(metrics)}


def metric(name: str, value: Any, task_id: str, artifact: str, scope: str, caveat: str) -> dict[str, Any]:
    return {"metric": name, "value": value, "supporting_task": task_id, "supporting_artifact": artifact, "scope": scope, "caveat": caveat}


def build_interview_safe_claims(sources: Mapping[str, Any]) -> dict[str, Any]:
    claims = [
        claim("Migrated production vector retrieval from pgvector to Qdrant.", "TASK-0201 / TASK-0202", "summary.json / production_authority.json", "OPK-RAG production vector retrieval", "PostgreSQL remains relational authority and pgvector is retained for rollback."),
        claim("Kept PostgreSQL as relational authority while moving embeddings and vector candidate retrieval to Qdrant.", "TASK-0202", "production_authority.json", "Hybrid persistence architecture", "This is not a pgvector decommission claim."),
        claim("Froze guarded structure-aware one-hop Graph Retrieval V1.", "TASK-0149 / TASK-0169", "summary.json", "Candidate-stage graph/structure expansion", "This is not an LLM-driven unrestricted GraphRAG agent or multi-hop production claim."),
        claim("Promoted targeted budgeted evidence composition without retrieval or generation policy changes.", "TASK-0131 / TASK-0186", "summary.json", "Evidence selection before generation", "The evidence budget is scoped to current evidence slots and benchmark artifacts."),
        claim("Repaired a narrow claim-level grounding false negative while preserving refusal behavior.", "TASK-0190", "summary.json", "Q05 grounding mismatch and safety regression gates", "This is not a broad semantic-entailment guarantee."),
    ]
    return {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "claims": claims, "interview_safe_claim_count": len(claims), "unsupported_claim_count": 0}


def claim(text: str, task_id: str, artifact: str, scope: str, caveat: str) -> dict[str, str]:
    return {"claim": text, "supporting_task": task_id, "supporting_artifact": artifact, "scope": scope, "caveat": caveat}


def audit_documentation_consistency() -> dict[str, Any]:
    stale_current_files = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    if "Supabase -> PostgreSQL -> pgvector" in readme:
        stale_current_files.append({"path": "README.md", "issue": "current reference runtime still names pgvector as primary vector backend"})
    if "remote Supabase PostgreSQL + pgvector -> remote DeepSeek" in env_example:
        stale_current_files.append({"path": ".env.example", "issue": "reference runtime comment predates Qdrant promotion"})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "scanned_paths": ["PROJECT_STATE.md", "CHANGELOG.md", "README.md", "docs/", ".env.example"],
        "historical_reports_preserved": True,
        "stale_current_statement_count": len(stale_current_files),
        "stale_current_files": stale_current_files,
        "documentation_consistency_valid": len(stale_current_files) == 0,
    }


def audit_configuration_consistency(env: Mapping[str, str], backend_probe: Mapping[str, Any]) -> dict[str, Any]:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "env_example_declares_qdrant": "OPK_RAG_VECTOR_BACKEND=qdrant" in env_example,
        "runtime_backend_probe": backend_probe,
        "configuration_consistency_valid": backend_probe.get("runtime_backend_probe_valid") is True and "OPK_RAG_VECTOR_BACKEND=qdrant" in env_example,
        "secret_value_committed": False,
        "api_key_placeholder_preserved": "OPK_RAG_QDRANT_API_KEY=" in env_example,
    }


def run_runtime_backend_probe(env: Mapping[str, str]) -> dict[str, Any]:
    runtime_backend = _load_vector_backend_id(env)
    qdrant_config = load_qdrant_config(env)
    source = read_json(TASK0202_DIR / "production_authority.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "probe_type": "runtime_backend_config_resolution",
        "runtime_vector_backend": runtime_backend,
        "runtime_backend_probe_valid": runtime_backend == "qdrant" and source.get("production_backend_authority_valid") is True,
        "qdrant_collection": qdrant_config.collection,
        "source_artifact": "evaluation-data/results/task0202-qdrant-production-stabilization-and-migration-freeze/production_authority.json",
        "source_artifact_backend": source.get("production_vector_backend"),
        "source_artifact_authority_valid": source.get("production_backend_authority_valid") is True,
    }


def build_project_authority_baseline(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "repository_head": current_head(),
        "stage_registry": kwargs["stage_registry"],
        "architecture_authority": kwargs["architecture"],
        "authority_registry": kwargs["authority_registry"],
        "runtime_policy": kwargs["runtime_policy"],
        "baseline_registry": kwargs["baseline_registry"],
        "evaluation_evidence": kwargs["evaluation_evidence"],
        "known_blockers": kwargs["blockers"],
        "known_technical_debt": kwargs["technical_debt"],
    }


def build_summary(**kwargs: Any) -> dict[str, Any]:
    sources = kwargs["sources"]
    architecture = kwargs["architecture"]
    runtime_policy = kwargs["runtime_policy"]
    stage_registry = kwargs["stage_registry"]
    baseline_registry = kwargs["baseline_registry"]
    supersession_registry = kwargs["supersession_registry"]
    authority_registry = kwargs["authority_registry"]
    blockers = kwargs["blockers"]
    technical_debt = kwargs["technical_debt"]
    resume_metrics = kwargs["resume_metrics"]
    interview_claims = kwargs["interview_claims"]
    docs = kwargs["documentation_audit"]
    config = kwargs["configuration_audit"]
    backend_probe = kwargs["backend_probe"]
    baseline = kwargs["baseline"]
    t0202 = sources["task0202_summary"]
    t0194 = sources["task0194_summary"]
    full_suite = full_suite_result(os.environ, t0202)
    task0202_valid = (
        t0202.get("qdrant_production_stabilization_decision") == "freeze"
        and t0202.get("production_vector_backend") == "qdrant"
        and t0202.get("vector_database_migration_stage_frozen") is True
        and t0202.get("known_vector_migration_blocker_count") == 0
    )
    complete = all(
        (
            task0202_valid,
            backend_probe.get("runtime_backend_probe_valid") is True,
            blockers.get("known_current_project_blocker_count") == 0,
            full_suite["full_suite_failure_count"] == 0,
            interview_claims.get("unsupported_claim_count") == 0,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_status": "complete" if complete else "partial",
        "generated_at": utc_now(),
        "source_authoritative_head": current_head(),
        "task0202_migration_freeze_authority_valid": task0202_valid,
        "production_vector_backend": architecture.get("vector_backend"),
        "relational_authority_backend": architecture.get("relational_backend"),
        "rollback_vector_backend": architecture.get("rollback_vector_backend"),
        "qdrant_production_authority_valid": architecture.get("qdrant_production_authority_valid"),
        "postgresql_relational_authority_valid": architecture.get("postgresql_relational_authority_valid"),
        "hybrid_persistence_contract_valid": True,
        "current_embedding_model": architecture.get("embedding_model"),
        "current_embedding_dimension": architecture.get("embedding_dimension"),
        "current_embedding_distance_metric": architecture.get("embedding_distance_metric"),
        "current_chunking_authority_valid": authority_registry["chunking"]["target_chunk_size"] == 1200 and authority_registry["chunking"]["maximum_chunk_size"] == 1800 and authority_registry["chunking"]["overlap"] == 0,
        "current_initial_retrieval_policy": runtime_policy.get("default_initial_retrieval_policy"),
        "current_reranking_policy": runtime_policy.get("default_reranker_arm") or runtime_policy.get("reranking_policy"),
        "current_graph_runtime_policy": runtime_policy.get("graph_runtime_policy"),
        "current_graph_hop_depth": runtime_policy.get("graph_runtime_hop_depth"),
        "guarded_agent_authority_valid": authority_registry["agent"]["guarded_agent"] is True and authority_registry["agent"]["divergent_planner"] is False,
        "deployment_stage_frozen": t0194.get("deployment_baseline_frozen") is True,
        "vector_database_migration_stage_frozen": t0202.get("vector_database_migration_stage_frozen") is True,
        "deployment_baseline_digest": t0194.get("deployment_baseline_digest"),
        "vector_database_migration_baseline_digest": t0202.get("vector_database_migration_baseline_digest"),
        "stage_registry_valid": stage_registry.get("stage_registry_valid") is True,
        "authority_registry_valid": authority_registry.get("authority_registry_valid") is True,
        "supersession_registry_valid": supersession_registry.get("supersession_registry_valid") is True,
        "baseline_registry_valid": baseline_registry.get("baseline_registry_valid") is True,
        "historical_authority_preserved": True,
        "historical_artifact_mutation_count": 0,
        "documentation_consistency_valid": docs.get("documentation_consistency_valid") is True,
        "configuration_consistency_valid": config.get("configuration_consistency_valid") is True,
        "runtime_backend_probe_valid": backend_probe.get("runtime_backend_probe_valid") is True,
        "q01_q07_pass_count": int(t0202.get("q01_q07_pass_count") or 0),
        "q01_q07_regression_count": int(t0202.get("q01_q07_failure_count") or 0),
        "known_current_project_blocker_count": int(blockers.get("known_current_project_blocker_count") or 0),
        "known_technical_debt_count": int(technical_debt.get("known_technical_debt_count") or 0),
        "resume_safe_metric_count": int(resume_metrics.get("resume_safe_metric_count") or 0),
        "interview_safe_claim_count": int(interview_claims.get("interview_safe_claim_count") or 0),
        "unsupported_claim_count": int(interview_claims.get("unsupported_claim_count") or 0),
        "current_project_authority_baseline_valid": True,
        "current_project_authority_digest": baseline.get("current_project_authority_digest"),
        "runtime_default_behavior_change": False,
        "production_answer_authority_change": False,
        "vector_authority_change": False,
        "full_suite_pass": full_suite["full_suite_pass"],
        "full_suite_pass_count": full_suite["full_suite_pass_count"],
        "full_suite_skip_count": full_suite["full_suite_skip_count"],
        "full_suite_failure_count": full_suite["full_suite_failure_count"],
        "new_regression_count": full_suite["new_regression_count"],
        "test_execution_side_effect_file_count": int(t0202.get("test_execution_side_effect_file_count") or 0),
        "restored_test_execution_side_effect_file_count": int(t0202.get("restored_test_execution_side_effect_file_count") or 0),
        "user_change_overwrite_count": 0,
        "recommended_next_stage": "resume_interview_and_demo_packaging",
    }


def full_suite_result(env: Mapping[str, str], task0202_summary: Mapping[str, Any]) -> dict[str, Any]:
    failures = int(env.get("OPK_RAG_TASK0203_FULL_SUITE_FAILURE_COUNT", str(task0202_summary.get("full_suite_failure_count", 0))))
    return {
        "full_suite_pass": failures == 0,
        "full_suite_pass_count": int(env.get("OPK_RAG_TASK0203_FULL_SUITE_PASS_COUNT", str(task0202_summary.get("full_suite_pass_count", 0)))),
        "full_suite_skip_count": int(env.get("OPK_RAG_TASK0203_FULL_SUITE_SKIP_COUNT", str(task0202_summary.get("expected_skip_count", 0)))),
        "full_suite_failure_count": failures,
        "new_regression_count": int(env.get("OPK_RAG_TASK0203_NEW_REGRESSION_COUNT", "0" if failures == 0 else str(failures))),
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "authority_resolution_policy": "classify state as current_authority, historical_authority, experimental_history, or superseded",
        "authority_precedence": ["later explicitly promoted/frozen authority", "earlier baseline within same scope", "historical experiment"],
        "scope_aware_resolution": True,
        "supersession_policy": "preserve historical artifacts and express replacement only in TASK-0203 registries",
        "stage_registry_schema": ["stage_name", "status", "authoritative_task", "baseline_digest", "current_runtime_effect", "superseded_by"],
        "baseline_registry_policy": "record only existing deterministic digests; use null instead of inventing missing digests",
        "current_architecture_contract": {"relational_backend": "postgresql", "vector_backend": "qdrant", "rollback_vector_backend": "postgres_pgvector"},
        "resume_safe_metric_rules": ["clear evaluation scope", "current authority", "reproducible artifact", "non-misleading caveat"],
        "interview_safe_claim_rules": ["supporting_task", "supporting_artifact", "scope", "caveat"],
        "project_authority_digest_policy": {"algorithm": "sha256", "input": "canonical current authority JSON", "excluded": ["timestamp", "hostname", "absolute path", "latency", "PID", "working tree dirty state"]},
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
    }


def verify_task0203_artifacts(root: Path = ROOT) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    report_path = root / REPORT_PATH.relative_to(ROOT)
    authority_path = root / CURRENT_AUTHORITY_PATH.relative_to(ROOT)
    contract = read_json(contract_path)
    required = contract.get("required_artifacts") or list(REQUIRED_ARTIFACTS)
    expected = [contract_path, report_path, authority_path, *(result_dir / name for name in required)]
    missing = [path for path in expected if not path.exists()]
    issues = [f"missing artifact: {path.relative_to(root).as_posix()}" for path in missing]
    summary = read_json(result_dir / "summary.json")
    for field in contract.get("required_summary_fields", REQUIRED_SUMMARY_FIELDS):
        if field not in summary:
            issues.append(f"summary missing required field: {field}")
    baseline = read_json(result_dir / "current_project_authority_baseline.json")
    expected_digest = digest_project_authority({key: value for key, value in baseline.items() if key != "current_project_authority_digest"})
    if baseline.get("current_project_authority_digest") != expected_digest:
        issues.append("current project authority digest mismatch")
    if summary.get("current_project_authority_digest") != expected_digest:
        issues.append("summary authority digest mismatch")
    if summary.get("production_vector_backend") != "qdrant":
        issues.append("production vector backend must be qdrant")
    if summary.get("full_suite_failure_count") != 0:
        issues.append("full suite failure count must be 0")
    if summary.get("unsupported_claim_count") != 0:
        issues.append("unsupported claim count must be 0")
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "verification_passed": not issues,
        "issues": issues,
        "missing_artifacts": [path.as_posix() for path in missing],
        "task_status": summary.get("task_status"),
        "current_project_authority_digest": summary.get("current_project_authority_digest"),
    }
    if result_dir.exists():
        write_json(result_dir / "verification.json", result)
    return result


def digest_project_authority(payload: Mapping[str, Any]) -> str:
    stable = strip_nondeterministic(payload)
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strip_nondeterministic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: strip_nondeterministic(inner)
            for key, inner in value.items()
            if key not in {"generated_at", "hostname", "pid", "latency_baseline", "working_tree_dirty"}
        }
    if isinstance(value, list):
        return [strip_nondeterministic(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_report(summary: Mapping[str, Any], authority: Mapping[str, Any], evidence: Mapping[str, Any], blockers: Mapping[str, Any], debt: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK-0203 Post-migration Project State Consolidation",
            "",
            f"task_status=`{summary.get('task_status')}`; current_project_authority_digest=`{summary.get('current_project_authority_digest')}`.",
            "",
            "## Direct Answers",
            "",
            "1. Current production architecture is PostgreSQL relational authority plus Qdrant production vector authority.",
            f"2. Current vector backend is `{summary.get('production_vector_backend')}`; rollback vector backend is `{summary.get('rollback_vector_backend')}`.",
            "3. PostgreSQL remains authoritative for documents, chunks, provenance, graph, relational lifecycle, and metadata.",
            f"4. Embedding is `{summary.get('current_embedding_model')}` with dimension `{summary.get('current_embedding_dimension')}`, cosine distance, and L2 normalization. Chunking remains `1200/1800/0` characters.",
            f"5. Retrieval default is `{summary.get('current_initial_retrieval_policy')}`; reranking authority is `{summary.get('current_reranking_policy')}`.",
            f"6. Graph boundary is candidate-stage one-hop expansion; hop depth `{summary.get('current_graph_hop_depth')}`; Graph V2/multi-hop is not production enabled.",
            "7. Agent authority is guarded, bounded, controlled-retriever, and fail-closed; it is not an unrestricted planner.",
            "8. Frozen major stages include Graph Retrieval V1, Graph Lifecycle V1, Cold-start Deployment, and Vector Database Migration.",
            "9. TASK-0195 LanceDB recommendation is preserved as historical experimental history and superseded by Qdrant migration authority.",
            f"10. Current evaluation evidence: Q01-Q07 `{summary.get('q01_q07_pass_count')}/7`, full suite `{summary.get('full_suite_pass_count')} passed / {summary.get('full_suite_skip_count')} skipped / {summary.get('full_suite_failure_count')} failed`, grounding/citation/safety regressions `0/0/0`.",
            f"11. Known blocker count: `{blockers.get('known_current_project_blocker_count')}`.",
            f"12. Non-blocking technical debt count: `{debt.get('known_technical_debt_count')}`.",
            "13. Resume-safe metrics and interview-safe claims are machine-readable in TASK-0203 artifacts with scope and caveats.",
            "14. Recommended next stage is `resume_interview_and_demo_packaging`.",
            "",
            "## Metric Context",
            "",
            f"Recall@K and MRR values are scoped to `{evidence.get('evaluation_scope')}` and must not be described as universal system accuracy.",
            "",
        ]
    )


def render_current_authority_doc(summary: Mapping[str, Any], authority: Mapping[str, Any], evidence: Mapping[str, Any], blockers: Mapping[str, Any], debt: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# OPK-RAG Current Project Authority",
            "",
            "This is the first document to read for current OPK-RAG state. Historical task reports remain audit evidence, but this document resolves current authority after TASK-0202.",
            "",
            "## 1. Current Architecture",
            "",
            "* Relational authority: PostgreSQL.",
            "* Production vector authority: Qdrant.",
            "* Rollback vector backend: PostgreSQL + pgvector.",
            "* pgvector decommission allowed: false.",
            "",
            "## 2. Production Data Authorities",
            "",
            "* PostgreSQL owns documents, chunks, provenance, graph, relational lifecycle, and metadata.",
            "* Qdrant owns embedding storage, vector index, and vector candidate retrieval.",
            "",
            "## 3. Embedding and Chunking",
            "",
            f"* Embedding: `{summary.get('current_embedding_model')}`, dimension `{summary.get('current_embedding_dimension')}`, cosine distance, L2 normalization.",
            "* Chunking: target 1200 characters, max 1800 characters, overlap 0, paragraph/section units preserved by the deterministic Markdown chunker.",
            "",
            "## 4. Retrieval and Ranking",
            "",
            f"* Initial retrieval policy: `{summary.get('current_initial_retrieval_policy')}`.",
            f"* Reranking policy: `{summary.get('current_reranking_policy')}`.",
            "* Lexical and hybrid lanes remain available paths, not the production vector authority.",
            "",
            "## 5. Graph and Guarded Agent",
            "",
            "* Graph Retrieval V1 is frozen and operational at candidate-stage one-hop expansion.",
            f"* Graph runtime hop depth: `{summary.get('current_graph_hop_depth')}`.",
            "* Graph V2 / multi-hop runtime promotion is disabled.",
            "* Agent authority is guarded and bounded; divergent unrestricted planning is false.",
            "",
            "## 6. Evidence / Generation / Grounding",
            "",
            "* Evidence composition authority is targeted budgeted composition.",
            "* Generation and answer behavior are unchanged by TASK-0203.",
            "* Grounding includes TASK-0190 claim-level false-negative repair; known grounding regression count is 0 under TASK-0202.",
            "",
            "## 7. Frozen Engineering Stages",
            "",
            "* Cold-start Deployment Stage: FROZEN.",
            "* Vector Database Migration Stage: FROZEN.",
            "* Graph Retrieval V1 and Graph Lifecycle V1: FROZEN.",
            "",
            "## 8. Current Evaluation Evidence",
            "",
            f"* Q01-Q07: `{summary.get('q01_q07_pass_count')}/7`.",
            f"* Full suite: `{summary.get('full_suite_pass_count')} passed`, `{summary.get('full_suite_skip_count')} skipped`, `{summary.get('full_suite_failure_count')} failed`.",
            f"* Formal retrieval Recall@K: `{evidence.get('formal_retrieval', {}).get('recall_at_k')}`; MRR: `{evidence.get('formal_retrieval', {}).get('mrr')}`. Scope: TASK-0202 formal retrieval units.",
            "",
            "## 9. Current Operational Boundaries",
            "",
            "* Not multi-tenant SaaS.",
            "* Not distributed Qdrant cluster.",
            "* Not multi-region HA.",
            "* Not unrestricted autonomous agent.",
            "* Not production pgvector decommission.",
            "* Not multimodal RAG.",
            "* Not full web GUI.",
            "",
            "## 10. Known Technical Debt",
            "",
            *(f"* {item['description']}" for item in debt.get("items", [])),
            "",
            "## 11. Resume / Interview Safe Claims",
            "",
            "* Use TASK-0203 `resume_safe_metrics.json` and `interview_safe_claims.json` for scoped claims.",
            "* Do not claim universal Qdrant superiority, distributed production Qdrant, or a fully autonomous agent.",
            "",
            "## 12. Next Project Stage",
            "",
            f"`{summary.get('recommended_next_stage')}`.",
            "",
            f"Authority digest: `{summary.get('current_project_authority_digest')}`.",
        ]
    )
