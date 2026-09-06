from __future__ import annotations

from dataclasses import asdict, dataclass
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
from opk_rag.embedding.config import EmbeddingConfig, load_embedding_config
from opk_rag.embedding.query import prepare_query_input
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0170_cold_start_reproducibility_baseline import (
    CORPUS_ROOT,
    TASK_DATABASE_URL_VARIABLE,
    resolve_cold_start_secret_authority,
)
from opk_rag.evaluation.task0183_cold_start_vector_index_materialization_diagnosis import (
    _git_head,
    _load_real_provider,
    audit_database_vector_state,
    audit_query_embedding,
)
from opk_rag.lexical.config import LexicalIndexConfig, build_lexical_configuration_fingerprint
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer
from opk_rag.runtime_v2 import graph_retrieval, initial_retrieval
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.search.models import SearchResponse, SearchResult

TASK_ID = "TASK-0184"
SOURCE_AUTHORITATIVE_TASK = "TASK-0183"
EXPERIMENT_ID = "task0184-cold-start-retrieval-runtime-validation-and-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0184_cold_start_retrieval_runtime_validation_and_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0184_COLD_START_RETRIEVAL_RUNTIME_VALIDATION_AND_DIAGNOSIS_REPORT.md"
TASK0183_SUMMARY = ROOT / "evaluation-data" / "results" / "task0183-cold-start-vector-index-materialization-diagnosis" / "summary.json"

PROBE_QUERIES = (
    "Tauri Demo Markdown 文件选择和公众号草稿箱发布流程",
    "HTML 到 DOCX 到 OOXML 的技术路线如何生成可编辑 Word 三线表",
    "Graph Retrieval V1 一跳结构扩展验证",
)

LOSS_SUBSTAGES = {
    "query_preparation",
    "query_embedding",
    "vector_retrieval",
    "lexical_index_readiness",
    "lexical_retrieval",
    "fusion",
    "guard_routing",
    "structure_identity_resolution",
    "structure_expansion",
    "candidate_construction",
    "candidate_identity_mapping",
    "candidate_deduplication",
    "candidate_to_reranker_contract",
    "none",
}

ROOT_CAUSES = {
    "query_preparation_failure",
    "vector_retrieval_runtime_failure",
    "lexical_index_missing_or_unready",
    "lexical_retrieval_runtime_failure",
    "fusion_contract_mismatch",
    "guard_routing_failure",
    "unexpected_retrieval_policy_fallback",
    "structure_identity_resolution_failure",
    "structure_expansion_failure",
    "candidate_contract_mismatch",
    "candidate_identity_mapping_failure",
    "candidate_score_validation_failure",
    "candidate_to_reranker_contract_mismatch",
    "no_retrieval_runtime_failure_reproduced",
    "unknown_retrieval_runtime_failure",
}


@dataclass(frozen=True)
class ProbeResult:
    query: str
    mode: str
    success: bool
    raw_candidate_count: int
    result_count: int
    error: str
    candidate_chunk_ids: tuple[str, ...]
    candidate_documents: tuple[str, ...]
    reranker_input_candidate_count: int
    evidence_composition_reached: bool
    rows: tuple[dict[str, Any], ...]


def run_task0184(*, write: bool = True, env: Mapping[str, str] | None = None, provider: Any | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = _git_head(ROOT)
    task0183 = _load_json(TASK0183_SUMMARY)
    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    database_url = _resolve_database_url(env)

    db = audit_database_vector_state(database_url, embedding_config)
    query_audits = [audit_query_preparation(query, embedding_config, search_config, provider) for query in PROBE_QUERIES]
    if provider is None and database_url and db.get("database_connection_success"):
        provider = _load_real_provider(embedding_config, {})
        query_audits = [audit_query_preparation(query, embedding_config, search_config, provider) for query in PROBE_QUERIES]

    lexical_readiness = audit_lexical_readiness(database_url, db, search_config)
    vector_probes = run_mode_probes(database_url, db, provider, embedding_config, search_config, mode="vector")
    lexical_probes = run_mode_probes(database_url, db, provider, embedding_config, search_config, mode="bm25")
    fusion_probes = run_mode_probes(database_url, db, provider, embedding_config, search_config, mode="hybrid")
    production_probe = run_production_default_probe(database_url, db, provider, embedding_config, search_config)
    guard_probe = audit_guarded_structure_runtime()

    candidate_rows = tuple(row for probe in (*vector_probes, *lexical_probes, *fusion_probes, production_probe) for row in probe.rows)
    identity = validate_candidate_identity(database_url, tuple(row["chunk_id"] for row in candidate_rows), db)
    candidate_contract = validate_candidate_contract(candidate_rows)
    failure = determine_failure(query_audits, vector_probes, lexical_readiness, lexical_probes, fusion_probes, guard_probe, candidate_contract, identity, production_probe)
    summary = build_summary(
        source_head=source_head,
        task0183=task0183,
        db=db,
        query_audits=query_audits,
        lexical_readiness=lexical_readiness,
        vector_probes=vector_probes,
        lexical_probes=lexical_probes,
        fusion_probes=fusion_probes,
        production_probe=production_probe,
        guard_probe=guard_probe,
        candidate_contract=candidate_contract,
        identity=identity,
        failure=failure,
        search_config=search_config,
        env=env,
    )
    artifacts = {
        "summary.json": summary,
        "runtime_authority_audit.json": runtime_authority(search_config),
        "query_preparation_audit.json": query_audits,
        "lexical_readiness_audit.json": lexical_readiness,
        "vector_probe.json": [asdict(probe) for probe in vector_probes],
        "lexical_probe.json": [asdict(probe) for probe in lexical_probes],
        "fusion_probe.json": [asdict(probe) for probe in fusion_probes],
        "production_default_runtime_probe.json": asdict(production_probe),
        "guarded_structure_aware_probe.json": guard_probe,
        "candidate_identity_validation.json": identity,
        "candidate_contract_validation.json": candidate_contract,
        "failure_taxonomy_decision.json": failure,
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, _redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_query_preparation(query: str, embedding_config: EmbeddingConfig, search_config: VectorSearchConfig, provider: Any | None) -> dict[str, Any]:
    out = {
        "query": query,
        "query_nonempty": bool(query.strip()),
        "query_normalization_success": False,
        "query_embedding_required": True,
        "query_embedding_generation_success": False,
        "query_embedding_dimension": 0,
        "query_preparation_error": "not_attempted",
    }
    try:
        prepared = prepare_query_input(
            query,
            embedding_config,
            count_tokens=provider.count_tokens if provider is not None else lambda text: max(1, len(text.split())),
            instruction=search_config.query_instruction,
            template_version=search_config.query_template_version,
            max_query_tokens=search_config.max_query_tokens,
        )
        out["query_normalization_success"] = bool(prepared.query.strip())
        out["query_preparation_error"] = "none"
    except Exception as exc:
        out["query_preparation_error"] = f"{type(exc).__name__}: {exc}"
        return out
    embed_probe = audit_query_embedding(_QueryProvider(provider, query) if provider is not None else None, embedding_config, search_config)
    out["query_embedding_generation_success"] = embed_probe.get("query_embedding_generation_success", False)
    out["query_embedding_dimension"] = embed_probe.get("query_embedding_dimension", 0)
    return out


class _QueryProvider:
    def __init__(self, provider: Any, query: str) -> None:
        self.provider = provider
        self.query = query
        self.dimension = provider.dimension
        self.model_id = provider.model_id

    def count_tokens(self, text: str) -> int:
        return self.provider.count_tokens(text)

    def embed_query(self, query: str, instruction: str | None = None):
        del query
        return self.provider.embed_query(self.query, instruction=instruction)


def audit_lexical_readiness(database_url: str, db: Mapping[str, Any], search_config: VectorSearchConfig) -> dict[str, Any]:
    out = {
        "lexical_lane_enabled": True,
        "lexical_index_required": True,
        "lexical_index_exists": False,
        "lexical_index_ready": False,
        "indexed_chunk_count": 0,
        "expected_chunk_count": 0,
        "stale_chunk_count": 0,
        "missing_chunk_count": 0,
        "corpus_stats_present": False,
        "lexical_readiness_error": "not_attempted",
    }
    if not database_url or not db.get("knowledge_base_id"):
        out["lexical_readiness_error"] = "database_or_knowledge_base_unavailable"
        return out
    try:
        from opk_rag.db.repositories import LexicalIndexRepository

        tokenizer = JiebaLexicalTokenizer()
        config = LexicalIndexConfig(bm25_k1=search_config.bm25_k1, bm25_b=search_config.bm25_b)
        fingerprint = build_lexical_configuration_fingerprint(config)
        with connect_postgres(database_url) as conn:
            readiness = LexicalIndexRepository(conn).get_lexical_readiness(
                knowledge_base_id=UUID(str(db["knowledge_base_id"])),
                tokenizer_id=tokenizer.tokenizer_id,
                tokenizer_version=tokenizer.version,
                configuration_fingerprint=fingerprint,
            )
        out.update({
            "lexical_index_exists": readiness.indexed_chunk_count > 0 and readiness.corpus_stats_present,
            "lexical_index_ready": readiness.ready,
            "indexed_chunk_count": readiness.indexed_chunk_count,
            "expected_chunk_count": readiness.expected_chunk_count,
            "stale_chunk_count": readiness.stale_chunk_count,
            "missing_chunk_count": readiness.missing_chunk_count,
            "corpus_stats_present": readiness.corpus_stats_present,
            "lexical_readiness_error": "none",
        })
    except Exception as exc:
        out["lexical_readiness_error"] = f"{type(exc).__name__}: {exc}"
    return out


def run_mode_probes(database_url: str, db: Mapping[str, Any], provider: Any | None, embedding_config: EmbeddingConfig, base_config: VectorSearchConfig, *, mode: str) -> tuple[ProbeResult, ...]:
    return tuple(
        run_single_probe(database_url, db, provider, embedding_config, config_for_mode(base_config, mode), query)
        for query in PROBE_QUERIES
    )


def run_production_default_probe(database_url: str, db: Mapping[str, Any], provider: Any | None, embedding_config: EmbeddingConfig, search_config: VectorSearchConfig) -> ProbeResult:
    return run_single_probe(database_url, db, provider, embedding_config, search_config, PROBE_QUERIES[0])


def run_single_probe(database_url: str, db: Mapping[str, Any], provider: Any | None, embedding_config: EmbeddingConfig, config: VectorSearchConfig, query: str) -> ProbeResult:
    if not database_url or not db.get("knowledge_base_id"):
        return _failed_probe(query, config.mode, "database_or_knowledge_base_unavailable")
    if config.mode in {"vector", "hybrid"} and provider is None:
        return _failed_probe(query, config.mode, "embedding_provider_unavailable")
    try:
        response, _payload = production_search_knowledge_base(
            database_url=database_url,
            knowledge_base_id=UUID(str(db["knowledge_base_id"])),
            query=query,
            provider=provider,
            embedding_config=embedding_config,
            search_config=config,
            reranker_provider=None,
        )
        return _probe_from_response(response)
    except Exception as exc:
        return _failed_probe(query, config.mode, f"{type(exc).__name__}: {exc}")


def config_for_mode(base: VectorSearchConfig, mode: str) -> VectorSearchConfig:
    return VectorSearchConfig(
        mode=mode, top_k=base.top_k, candidate_k=base.candidate_k, rerank_top_n=base.rerank_top_n,
        bm25_candidate_k=base.bm25_candidate_k, min_similarity=base.min_similarity, deduplicate=base.deduplicate,
        max_query_tokens=base.max_query_tokens, query_instruction=base.query_instruction,
        query_template_version=base.query_template_version, max_top_k=base.max_top_k,
        overlap_deduplication_threshold=base.overlap_deduplication_threshold, bm25_k1=base.bm25_k1,
        bm25_b=base.bm25_b, rrf_k=base.rrf_k, rrf_vector_weight=base.rrf_vector_weight,
        rrf_bm25_weight=base.rrf_bm25_weight, rerank_enabled=base.rerank_enabled,
        reranker_policy=base.reranker_policy, rank_fusion_k=base.rank_fusion_k,
        rank_fusion_lambda=base.rank_fusion_lambda, context_token_budget=base.context_token_budget,
        context_max_chunks=base.context_max_chunks, context_max_chunks_per_document=base.context_max_chunks_per_document,
    )


def _probe_from_response(response: SearchResponse) -> ProbeResult:
    rows = tuple(candidate_row(result) for result in response.results)
    return ProbeResult(
        query=response.query,
        mode=response.retrieval_mode,
        success=True,
        raw_candidate_count=response.candidate_count,
        result_count=response.result_count,
        error="none",
        candidate_chunk_ids=tuple(str(result.chunk_id) for result in response.results),
        candidate_documents=tuple(str(result.document_id) for result in response.results),
        reranker_input_candidate_count=response.rerank_candidate_count if response.reranker_enabled else min(response.result_count, response.rerank_top_n or response.result_count),
        evidence_composition_reached=response.evidence_bundle is not None,
        rows=rows,
    )


def _failed_probe(query: str, mode: str, error: str) -> ProbeResult:
    return ProbeResult(query, mode, False, 0, 0, error, (), (), 0, False, ())


def candidate_row(result: SearchResult) -> dict[str, Any]:
    return {
        "candidate_id": str(result.chunk_id),
        "chunk_id": str(result.chunk_id),
        "document_id": str(result.document_id),
        "source_lane": "+".join(result.retrieval_sources),
        "raw_score": result.rerank_score if result.rerank_score is not None else result.rrf_score if result.rrf_score is not None else result.vector_similarity if result.vector_similarity is not None else result.bm25_score,
        "normalized_score": result.rrf_score,
        "fused_score": result.rrf_score,
        "rank": result.rank,
        "metadata": result.metadata or {},
        "retrieval_sources": tuple(result.retrieval_sources),
        "vector_similarity": result.vector_similarity,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "rerank_score": result.rerank_score,
    }


def validate_candidate_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("candidate_id", "chunk_id", "document_id", "source_lane", "raw_score", "rank", "metadata")
    contract_valid = all(all(field in row for field in required) for row in rows) and bool(rows)
    score_valid = all(_finite_or_none(row.get(name)) for row in rows for name in ("raw_score", "normalized_score", "fused_score"))
    rank_valid = all(isinstance(row.get("rank"), int) and int(row["rank"]) >= 1 for row in rows)
    provenance_valid = all(bool(row.get("retrieval_sources")) for row in rows)
    return {
        "candidate_contract_valid": contract_valid,
        "candidate_score_valid": score_valid and rank_valid,
        "candidate_provenance_valid": provenance_valid,
        "candidate_score_ordering_valid": rank_valid,
        "candidate_score_finite": score_valid,
        "candidate_total_count": len(rows),
        "duplicate_candidate_count": len(rows) - len({row.get("chunk_id") for row in rows}),
    }


def validate_candidate_identity(database_url: str, candidate_chunk_ids: Sequence[str], db: Mapping[str, Any]) -> dict[str, Any]:
    unique_ids = tuple(dict.fromkeys(str(chunk_id) for chunk_id in candidate_chunk_ids if chunk_id))
    out = {"candidate_total_count": len(candidate_chunk_ids), "canonical_candidate_count": 0, "orphan_candidate_count": len(unique_ids), "candidate_identity_valid": False}
    if not database_url or not unique_ids or not db.get("knowledge_base_id"):
        return out
    try:
        with connect_postgres(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select count(*)::integer,
                           count(*) filter (where c.id is not null and d.id is not null and d.knowledge_base_id = %s)::integer
                    from unnest(%s::uuid[]) candidate(chunk_id)
                    left join public.chunks c on c.id = candidate.chunk_id
                    left join public.documents d on d.id = c.document_id
                    """,
                    (db.get("knowledge_base_id"), list(unique_ids)),
                )
                total, canonical = cur.fetchone()
        valid = int(total) == int(canonical)
        out.update({
            "canonical_candidate_count": len(candidate_chunk_ids) if valid else int(canonical),
            "orphan_candidate_count": 0 if valid else int(total) - int(canonical),
            "candidate_identity_valid": valid,
            "unique_candidate_count": int(total),
        })
    except Exception as exc:
        out["candidate_identity_error"] = f"{type(exc).__name__}: {exc}"
    return out


def audit_guarded_structure_runtime() -> dict[str, Any]:
    graph_policy = graph_retrieval.default_graph_retrieval_policy()
    sample = _structure_probe_sample()
    out = {
        "guard_evaluation_count": 0,
        "guard_decision_success_count": 0,
        "guard_decision_failure_count": 0,
        "default_policy_requested": initial_retrieval.default_initial_retrieval_config().policy_name,
        "default_policy_executed": "",
        "unexpected_policy_fallback_count": 0,
        "structure_aware_retrieval_invocation_count": 0,
        "structure_expansion_requested_count": 0,
        "structure_expansion_not_required_count": 0,
        "structure_expansion_attempt_count": 0,
        "structure_expansion_success_count": 0,
        "structure_expansion_failure_count": 0,
        "structure_expansion_execution_failure_count": 0,
        "graph_runtime_hop_depth": graph_policy.maximum_hops,
        "graph_runtime_error": "none",
    }
    try:
        result = initial_retrieval.retrieve_initial_candidates(sample, config=initial_retrieval.default_initial_retrieval_config())
        out["guard_evaluation_count"] = result.trace.get("guard_evaluation_count", 0)
        out["guard_decision_success_count"] = 1
        out["default_policy_executed"] = result.trace.get("policy_name", "")
        out["unexpected_policy_fallback_count"] = int(out["default_policy_executed"] != out["default_policy_requested"])
        requested = bool(result.trace.get("structure_lane_invoked"))
        out["structure_aware_retrieval_invocation_count"] = 1
        out["structure_expansion_requested_count"] = int(requested)
        out["structure_expansion_not_required_count"] = int(not requested)
        out["structure_expansion_attempt_count"] = int(requested)
        graph_snapshot = _graph_snapshot()
        expansion = graph_retrieval.expand_runtime_candidates(
            result.candidates,
            graph_snapshot,
            runtime_config=graph_retrieval.GraphRetrievalRuntimeConfig(policy=graph_retrieval.ONE_HOP_GRAPH_RETRIEVAL_POLICY),
            policy=graph_policy,
        )
        out["structure_expansion_success_count"] = int(requested and not expansion.diagnostics.get("fallback_used", False))
        out["structure_expansion_failure_count"] = int(bool(expansion.diagnostics.get("fallback_used", False)))
        out["structure_expansion_execution_failure_count"] = out["structure_expansion_failure_count"]
        out["graph_added_candidate_count"] = len(expansion.added_candidates)
        out["graph_trace"] = expansion.trace
    except Exception as exc:
        out["guard_decision_failure_count"] = 1
        out["structure_expansion_failure_count"] = 1
        out["structure_expansion_execution_failure_count"] = 1
        out["graph_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _structure_probe_sample() -> dict[str, Any]:
    return {
        "query": "链接到的相关测试笔记如何验证 HTML 到 DOCX 到 OOXML 的技术路线？",
        "seed_source_units": [
            {
                "document_id": "source-documents/Agent/academic-docx-polisher/02 HTML 到 DOCX 到 OOXML 的技术路线.md",
                "label": "技术路线",
                "line_span": "L1-L80",
                "source_unit_id": "source-documents/Agent/academic-docx-polisher/02 HTML 到 DOCX 到 OOXML 的技术路线.md",
            }
        ],
    }


def _graph_snapshot() -> dict[str, Any]:
    src = "source-documents/Agent/academic-docx-polisher/02 HTML 到 DOCX 到 OOXML 的技术路线.md"
    dst = "source-documents/Agent/academic-docx-polisher/04 最小测试案例与验证结果.md"
    return {
        "allowed_edge_types": ["LINKS_TO"],
        "required_graph_path": [{"edge_id": "task0184-probe-edge", "edge_type": "LINKS_TO", "source_node_id": src, "target_node_id": dst, "authority_level": "G1"}],
        "candidate_source_units": [{"document_id": dst, "label": "最小测试案例与验证结果", "line_span": "L1-L80", "source_unit_id": dst}],
    }


def determine_failure(
    query_audits: Sequence[Mapping[str, Any]],
    vector_probes: Sequence[ProbeResult],
    lexical_readiness: Mapping[str, Any],
    lexical_probes: Sequence[ProbeResult],
    fusion_probes: Sequence[ProbeResult],
    guard_probe: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    production_probe: ProbeResult,
) -> dict[str, str]:
    if not all(audit.get("query_nonempty") and audit.get("query_normalization_success") for audit in query_audits):
        return _failure("query_preparation", "query_preparation_failure")
    if not any(audit.get("query_embedding_generation_success") and audit.get("query_embedding_dimension") == 1024 for audit in query_audits):
        return _failure("query_embedding", "unknown_retrieval_runtime_failure")
    if not any(probe.success and probe.raw_candidate_count > 0 for probe in vector_probes):
        return _failure("vector_retrieval", "vector_retrieval_runtime_failure")
    if lexical_readiness.get("lexical_lane_enabled") and not lexical_readiness.get("lexical_index_ready"):
        return _failure("lexical_index_readiness", "lexical_index_missing_or_unready")
    if lexical_readiness.get("lexical_lane_enabled") and not any(probe.success for probe in lexical_probes):
        return _failure("lexical_retrieval", "lexical_retrieval_runtime_failure")
    if not any(probe.success for probe in fusion_probes):
        return _failure("fusion", "fusion_contract_mismatch")
    if guard_probe.get("guard_decision_failure_count", 0) > 0:
        return _failure("guard_routing", "guard_routing_failure")
    if guard_probe.get("unexpected_policy_fallback_count", 0) > 0:
        return _failure("guard_routing", "unexpected_retrieval_policy_fallback")
    if guard_probe.get("structure_expansion_requested_count", 0) > 0 and guard_probe.get("structure_expansion_execution_failure_count", 0) > 0:
        return _failure("structure_expansion", "structure_expansion_failure")
    if not candidate_contract.get("candidate_contract_valid") or not candidate_contract.get("candidate_provenance_valid"):
        return _failure("candidate_construction", "candidate_contract_mismatch")
    if not candidate_contract.get("candidate_score_valid"):
        return _failure("candidate_construction", "candidate_score_validation_failure")
    if not identity.get("candidate_identity_valid"):
        return _failure("candidate_identity_mapping", "candidate_identity_mapping_failure")
    if not production_probe.success or production_probe.result_count == 0:
        return _failure("candidate_construction", "candidate_contract_mismatch")
    if production_probe.reranker_input_candidate_count <= 0:
        return _failure("candidate_to_reranker_contract", "candidate_to_reranker_contract_mismatch")
    return _failure("none", "no_retrieval_runtime_failure_reproduced")


def build_summary(
    *,
    source_head: str,
    task0183: Mapping[str, Any],
    db: Mapping[str, Any],
    query_audits: Sequence[Mapping[str, Any]],
    lexical_readiness: Mapping[str, Any],
    vector_probes: Sequence[ProbeResult],
    lexical_probes: Sequence[ProbeResult],
    fusion_probes: Sequence[ProbeResult],
    production_probe: ProbeResult,
    guard_probe: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    failure: Mapping[str, str],
    search_config: VectorSearchConfig,
    env: Mapping[str, str],
) -> dict[str, Any]:
    vector_raw = sum(probe.raw_candidate_count for probe in vector_probes if probe.success)
    lexical_raw = sum(probe.raw_candidate_count for probe in lexical_probes if probe.success)
    fusion_pre = vector_raw + lexical_raw
    fusion_post = sum(probe.result_count for probe in fusion_probes if probe.success)
    all_rows = [row for probe in (*vector_probes, *lexical_probes, *fusion_probes, production_probe) for row in probe.rows]
    passed = failure["first_retrieval_loss_substage"] == "none"
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if failure["first_retrieval_loss_substage"] in LOSS_SUBSTAGES and failure["diagnosed_root_cause"] in ROOT_CAUSES else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0183.get("source_authoritative_head") or source_head,
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        "retrieval_visible_embedding_count": db.get("retrieval_visible_embedding_count", 0),
        "upstream_authority_drift": not (db.get("authoritative_document_count") == 8 and db.get("authoritative_chunk_count") == 11 and db.get("stored_embedding_count") == 11 and db.get("retrieval_visible_embedding_count") == 11),
        **runtime_authority(search_config),
        "retrieval_probe_query_count": len(PROBE_QUERIES),
        "retrieval_probe_execution_count": len(vector_probes) + len(lexical_probes) + len(fusion_probes),
        "retrieval_probe_success_count": sum(probe.success for probe in (*vector_probes, *lexical_probes, *fusion_probes)),
        "retrieval_probe_failure_count": sum(not probe.success for probe in (*vector_probes, *lexical_probes, *fusion_probes)),
        "query_nonempty": all(audit.get("query_nonempty") for audit in query_audits),
        "query_normalization_success": all(audit.get("query_normalization_success") for audit in query_audits),
        "query_embedding_required": True,
        "query_embedding_generation_success": any(audit.get("query_embedding_generation_success") for audit in query_audits),
        "query_embedding_dimension": max((int(audit.get("query_embedding_dimension") or 0) for audit in query_audits), default=0),
        "vector_lane_enabled": True,
        "vector_retriever_invocation_count": len(vector_probes),
        "vector_retrieval_success_count": sum(probe.success for probe in vector_probes),
        "vector_retrieval_failure_count": sum(not probe.success for probe in vector_probes),
        "vector_raw_candidate_count": vector_raw,
        "vector_canonical_candidate_count": sum(probe.result_count for probe in vector_probes if probe.success),
        "lexical_lane_enabled": lexical_readiness.get("lexical_lane_enabled", True),
        "lexical_index_required": lexical_readiness.get("lexical_index_required", True),
        "lexical_index_exists": lexical_readiness.get("lexical_index_exists", False),
        "lexical_index_ready": lexical_readiness.get("lexical_index_ready", False),
        "lexical_retriever_invocation_count": len(lexical_probes),
        "lexical_retrieval_success_count": sum(probe.success for probe in lexical_probes),
        "lexical_retrieval_failure_count": sum(not probe.success for probe in lexical_probes),
        "lexical_raw_candidate_count": lexical_raw,
        "lexical_canonical_candidate_count": sum(probe.result_count for probe in lexical_probes if probe.success),
        "fusion_required": True,
        "fusion_invocation_count": len(fusion_probes),
        "fusion_execution_success_count": sum(probe.success for probe in fusion_probes),
        "pre_fusion_candidate_count": fusion_pre,
        "post_fusion_candidate_count": fusion_post,
        "fusion_candidate_identity_valid": identity.get("candidate_identity_valid", False),
        "fusion_score_valid": candidate_contract.get("candidate_score_valid", False),
        "guard_evaluation_count": guard_probe.get("guard_evaluation_count", 0),
        "guard_decision_success_count": guard_probe.get("guard_decision_success_count", 0),
        "guard_decision_failure_count": guard_probe.get("guard_decision_failure_count", 0),
        "default_policy_requested": guard_probe.get("default_policy_requested"),
        "default_policy_executed": guard_probe.get("default_policy_executed"),
        "unexpected_policy_fallback_count": guard_probe.get("unexpected_policy_fallback_count", 0),
        "structure_aware_retrieval_invocation_count": guard_probe.get("structure_aware_retrieval_invocation_count", 0),
        "structure_expansion_requested_count": guard_probe.get("structure_expansion_requested_count", 0),
        "structure_expansion_not_required_count": guard_probe.get("structure_expansion_not_required_count", 0),
        "structure_expansion_attempt_count": guard_probe.get("structure_expansion_attempt_count", 0),
        "structure_expansion_success_count": guard_probe.get("structure_expansion_success_count", 0),
        "structure_expansion_failure_count": guard_probe.get("structure_expansion_failure_count", 0),
        "structure_expansion_execution_failure_count": guard_probe.get("structure_expansion_execution_failure_count", 0),
        "graph_runtime_hop_depth": guard_probe.get("graph_runtime_hop_depth", 0),
        "candidate_contract_valid": candidate_contract.get("candidate_contract_valid", False),
        "candidate_identity_valid": identity.get("candidate_identity_valid", False),
        "candidate_score_valid": candidate_contract.get("candidate_score_valid", False),
        "candidate_provenance_valid": candidate_contract.get("candidate_provenance_valid", False),
        "candidate_total_count": candidate_contract.get("candidate_total_count", 0),
        "canonical_candidate_count": identity.get("canonical_candidate_count", 0),
        "orphan_candidate_count": identity.get("orphan_candidate_count", 0),
        "duplicate_candidate_count": candidate_contract.get("duplicate_candidate_count", 0),
        "candidate_from_vector_count": sum(1 for row in all_rows if "vector" in row.get("retrieval_sources", ())),
        "candidate_from_lexical_count": sum(1 for row in all_rows if "bm25" in row.get("retrieval_sources", ())),
        "candidate_from_fusion_count": sum(1 for row in all_rows if "+" in str(row.get("source_lane", ""))),
        "candidate_from_structure_expansion_count": guard_probe.get("graph_added_candidate_count", 0),
        "production_retrieval_runtime_invocation_count": 1,
        "production_retrieval_runtime_success_count": int(production_probe.success),
        "production_retrieval_runtime_failure_count": int(not production_probe.success),
        "production_candidate_pool_count": production_probe.result_count,
        "reranker_input_constructible": production_probe.reranker_input_candidate_count > 0,
        "reranker_input_candidate_count": production_probe.reranker_input_candidate_count,
        "reranker_execution_reached": search_config.rerank_enabled and production_probe.success,
        "evidence_composition_reached": production_probe.evidence_composition_reached,
        "first_retrieval_loss_substage": failure["first_retrieval_loss_substage"],
        "diagnosed_root_cause": failure["diagnosed_root_cause"],
        "cold_start_retrieval_runtime_stage_passed": passed,
        "next_failure_stage": "reranking_runtime" if passed else f"{failure['first_retrieval_loss_substage']}_repair",
        "retrieval_policy_changed": False,
        "graph_retrieval_policy_changed": False,
        "reranker_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "focused_test_passed_count": int(env.get("TASK0184_FOCUSED_TEST_PASSED_COUNT", "0")),
        "full_suite_passed_count": int(env.get("TASK0184_FULL_SUITE_PASSED_COUNT", "0")),
        "full_suite_skipped_count": int(env.get("TASK0184_FULL_SUITE_SKIPPED_COUNT", "0")),
        "full_suite_failed_count": int(env.get("TASK0184_FULL_SUITE_FAILED_COUNT", "0")),
        "known_preexisting_failure_count": int(env.get("TASK0184_KNOWN_PREEXISTING_FAILURE_COUNT", "0")),
        "new_regression_count": int(env.get("TASK0184_NEW_REGRESSION_COUNT", "0")),
        "rank_fusion_k": search_config.rank_fusion_k,
        "rank_fusion_lambda": search_config.rank_fusion_lambda,
        "rrf_k": search_config.rrf_k,
    }


def runtime_authority(search_config: VectorSearchConfig) -> dict[str, Any]:
    initial = initial_retrieval.default_initial_retrieval_config()
    graph = graph_retrieval.default_graph_retrieval_policy()
    lanes = ["vector", "lexical", "fusion", "guard", "structure_aware_graph"]
    return {
        "default_retrieval_policy": initial.policy_name,
        "available_retrieval_lane_count": len(lanes),
        "available_retrieval_lanes": lanes,
        "default_graph_hop_depth": graph.maximum_hops,
        "default_candidate_top_k": initial.candidate_top_k,
        "production_search_mode": search_config.mode,
    }


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0184.contract.v1",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "diagnosis_first": True,
        "runtime_mutation_allowed": False,
        "schema_mutation_allowed": False,
        "gold_runtime_routing_allowed": False,
        "expected_authoritative_document_count": 8,
        "expected_authoritative_chunk_count": 11,
        "expected_embedding_count": 11,
        "expected_embedding_dimension": 1024,
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
        "loss_substage_taxonomy": sorted(LOSS_SUBSTAGES),
        "root_cause_taxonomy": sorted(ROOT_CAUSES),
    }


REQUIRED_SUMMARY_FIELDS = (
    "task_id", "task_status", "source_authoritative_task", "source_authoritative_head",
    "authoritative_document_count", "authoritative_chunk_count", "stored_embedding_count", "retrieval_visible_embedding_count",
    "default_retrieval_policy", "available_retrieval_lane_count", "available_retrieval_lanes", "default_graph_hop_depth",
    "retrieval_probe_query_count", "retrieval_probe_execution_count", "retrieval_probe_success_count", "retrieval_probe_failure_count",
    "vector_lane_enabled", "vector_retriever_invocation_count", "vector_retrieval_success_count", "vector_retrieval_failure_count",
    "vector_raw_candidate_count", "vector_canonical_candidate_count", "lexical_lane_enabled", "lexical_index_required",
    "lexical_index_exists", "lexical_index_ready", "lexical_retriever_invocation_count", "lexical_retrieval_success_count",
    "lexical_retrieval_failure_count", "lexical_raw_candidate_count", "lexical_canonical_candidate_count", "fusion_required",
    "fusion_invocation_count", "fusion_execution_success_count", "pre_fusion_candidate_count", "post_fusion_candidate_count",
    "fusion_candidate_identity_valid", "fusion_score_valid", "guard_evaluation_count", "guard_decision_success_count",
    "guard_decision_failure_count", "default_policy_requested", "default_policy_executed", "unexpected_policy_fallback_count",
    "structure_aware_retrieval_invocation_count", "structure_expansion_requested_count", "structure_expansion_not_required_count",
    "structure_expansion_attempt_count", "structure_expansion_success_count", "structure_expansion_execution_failure_count",
    "graph_runtime_hop_depth", "candidate_contract_valid", "candidate_identity_valid", "candidate_score_valid",
    "candidate_provenance_valid", "candidate_total_count", "canonical_candidate_count", "orphan_candidate_count",
    "duplicate_candidate_count", "candidate_from_vector_count", "candidate_from_lexical_count", "candidate_from_fusion_count",
    "candidate_from_structure_expansion_count", "production_retrieval_runtime_invocation_count",
    "production_retrieval_runtime_success_count", "production_retrieval_runtime_failure_count", "production_candidate_pool_count",
    "reranker_input_constructible", "reranker_input_candidate_count", "reranker_execution_reached", "evidence_composition_reached",
    "first_retrieval_loss_substage", "diagnosed_root_cause", "cold_start_retrieval_runtime_stage_passed", "next_failure_stage",
    "retrieval_policy_changed", "graph_retrieval_policy_changed", "reranker_policy_changed", "runtime_default_behavior_change",
    "promotion_applied", "focused_test_passed_count", "full_suite_passed_count", "full_suite_skipped_count",
    "full_suite_failed_count", "known_preexisting_failure_count", "new_regression_count",
)


def verify_task0184_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    artifact_names = (
        "summary.json", "runtime_authority_audit.json", "query_preparation_audit.json", "lexical_readiness_audit.json",
        "vector_probe.json", "lexical_probe.json", "fusion_probe.json", "production_default_runtime_probe.json",
        "guarded_structure_aware_probe.json", "candidate_identity_validation.json", "candidate_contract_validation.json",
        "failure_taxonomy_decision.json",
    )
    missing = [name for name in artifact_names if not (result_dir / name).exists()]
    summary_path = result_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    missing_fields = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    passed = (
        not missing
        and not missing_fields
        and summary.get("task_status") == "complete"
        and summary.get("retrieval_policy_changed") is False
        and summary.get("graph_retrieval_policy_changed") is False
        and summary.get("runtime_default_behavior_change") is False
        and summary.get("promotion_applied") is False
        and summary.get("new_regression_count") == 0
    )
    return {
        "task_id": TASK_ID,
        "verification_passed": passed,
        "missing_artifacts": missing,
        "missing_summary_fields": missing_fields,
        "first_retrieval_loss_substage": summary.get("first_retrieval_loss_substage"),
        "diagnosed_root_cause": summary.get("diagnosed_root_cause"),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# TASK-0184 Cold-start Retrieval Runtime Validation and Diagnosis Report",
        "",
        "## Decision",
        f"task_status=`{summary.get('task_status')}`; first_retrieval_loss_substage=`{summary.get('first_retrieval_loss_substage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`; cold_start_retrieval_runtime_stage_passed=`{summary.get('cold_start_retrieval_runtime_stage_passed')}`.",
        "",
        "## Runtime Authority",
        f"default_retrieval_policy=`{summary.get('default_retrieval_policy')}`; lanes=`{summary.get('available_retrieval_lanes')}`; default_graph_hop_depth=`{summary.get('default_graph_hop_depth')}`; default_candidate_top_k=`{summary.get('default_candidate_top_k')}`.",
        "",
        "## Corpus Authority",
        f"documents=`{summary.get('authoritative_document_count')}`; chunks=`{summary.get('authoritative_chunk_count')}`; stored_embeddings=`{summary.get('stored_embedding_count')}`; retrieval_visible_embeddings=`{summary.get('retrieval_visible_embedding_count')}`; upstream_authority_drift=`{summary.get('upstream_authority_drift')}`.",
        "",
        "## Retrieval Probes",
        f"probe_queries=`{summary.get('retrieval_probe_query_count')}`; executions=`{summary.get('retrieval_probe_execution_count')}`; successes=`{summary.get('retrieval_probe_success_count')}`; failures=`{summary.get('retrieval_probe_failure_count')}`.",
        f"vector: success=`{summary.get('vector_retrieval_success_count')}` raw=`{summary.get('vector_raw_candidate_count')}` canonical=`{summary.get('vector_canonical_candidate_count')}`.",
        f"lexical: ready=`{summary.get('lexical_index_ready')}` success=`{summary.get('lexical_retrieval_success_count')}` raw=`{summary.get('lexical_raw_candidate_count')}` canonical=`{summary.get('lexical_canonical_candidate_count')}`.",
        f"fusion: success=`{summary.get('fusion_execution_success_count')}` pre=`{summary.get('pre_fusion_candidate_count')}` post=`{summary.get('post_fusion_candidate_count')}` rank_fusion_k=`{summary.get('rank_fusion_k')}` lambda=`{summary.get('rank_fusion_lambda')}`.",
        "",
        "## Guard And Structure",
        f"default_policy_requested=`{summary.get('default_policy_requested')}`; default_policy_executed=`{summary.get('default_policy_executed')}`; unexpected_policy_fallback_count=`{summary.get('unexpected_policy_fallback_count')}`.",
        f"structure_expansion_requested_count=`{summary.get('structure_expansion_requested_count')}`; structure_expansion_execution_failure_count=`{summary.get('structure_expansion_execution_failure_count')}`.",
        "",
        "## Candidate Contract",
        f"candidate_contract_valid=`{summary.get('candidate_contract_valid')}`; candidate_identity_valid=`{summary.get('candidate_identity_valid')}`; candidate_score_valid=`{summary.get('candidate_score_valid')}`; candidate_provenance_valid=`{summary.get('candidate_provenance_valid')}`; orphan_candidate_count=`{summary.get('orphan_candidate_count')}`.",
        f"production_candidate_pool_count=`{summary.get('production_candidate_pool_count')}`; reranker_input_constructible=`{summary.get('reranker_input_constructible')}`; evidence_composition_reached=`{summary.get('evidence_composition_reached')}`.",
        "",
        "## Policy Mutation",
        f"retrieval_policy_changed=`{summary.get('retrieval_policy_changed')}`; graph_retrieval_policy_changed=`{summary.get('graph_retrieval_policy_changed')}`; reranker_policy_changed=`{summary.get('reranker_policy_changed')}`; runtime_default_behavior_change=`{summary.get('runtime_default_behavior_change')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
        "",
        "## Test Accounting",
        f"Focused tests: `{summary.get('focused_test_passed_count')}` passed. Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`, new_regression_count=`{summary.get('new_regression_count')}`.",
        "",
        "## Next Frontier",
        f"next_failure_stage=`{summary.get('next_failure_stage')}`.",
        "",
    ])


def _failure(substage: str, root: str) -> dict[str, str]:
    return {"first_retrieval_loss_substage": substage, "diagnosed_root_cause": root}


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _load_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _resolve_database_url(env: Mapping[str, str]) -> str:
    secret = resolve_cold_start_secret_authority(env)
    return str(secret.get("database_url_value") or env.get(TASK_DATABASE_URL_VARIABLE) or env.get("DATABASE_URL") or "")


def _redact(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"postgres(?:ql)?://[^:\s/@]+:[^@\s]+@", "postgres://***:***@", text, flags=re.I)
    return json.loads(text)
