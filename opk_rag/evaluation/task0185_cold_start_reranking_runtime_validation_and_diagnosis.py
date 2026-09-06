from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import UUID

from opk_rag.core_tools.tools import search_knowledge_base as production_search_knowledge_base
from opk_rag.embedding.config import load_embedding_config
from opk_rag.evaluation.task0091_reranker_replay_benchmark import ROOT, read_json, write_json
from opk_rag.evaluation.task0183_cold_start_vector_index_materialization_diagnosis import (
    _load_real_provider,
    audit_database_vector_state,
)
from opk_rag.evaluation.task0184_cold_start_retrieval_runtime_validation_and_diagnosis import (
    PROBE_QUERIES,
    _load_json,
    _redact,
    _resolve_database_url,
    candidate_row,
    validate_candidate_identity,
)
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import RerankerConfig, load_reranker_config
from opk_rag.reranking.input import RERANKER_SCORE_SEMANTICS, render_reranker_document
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.search.models import SearchResult

TASK_ID = "TASK-0185"
SOURCE_AUTHORITATIVE_TASK = "TASK-0184"
EXPERIMENT_ID = "task0185-cold-start-reranking-runtime-validation-and-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0185_cold_start_reranking_runtime_validation_and_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0185_COLD_START_RERANKING_RUNTIME_VALIDATION_AND_DIAGNOSIS_REPORT.md"
TASK0184_SUMMARY = ROOT / "evaluation-data" / "results" / "task0184-cold-start-retrieval-runtime-validation-and-diagnosis" / "summary.json"

LOSS_SUBSTAGES = {
    "candidate_to_reranker_input",
    "reranker_input_validation",
    "reranker_model_authority",
    "reranker_model_loading",
    "reranker_execution",
    "rerank_output_validation",
    "rank_fusion",
    "ranked_candidate_construction",
    "ranked_candidate_identity_mapping",
    "ranked_candidate_score_validation",
    "ranked_candidate_to_evidence_contract",
    "none",
}

ROOT_CAUSES = {
    "candidate_reranker_contract_mismatch",
    "reranker_input_validation_failure",
    "reranker_model_unavailable",
    "reranker_model_cache_missing_or_incomplete",
    "reranker_model_load_failure",
    "reranker_runtime_execution_failure",
    "reranker_output_count_mismatch",
    "reranker_score_validation_failure",
    "rank_fusion_runtime_failure",
    "unexpected_reranker_fallback",
    "ranked_candidate_contract_mismatch",
    "ranked_candidate_identity_mapping_failure",
    "candidate_to_evidence_contract_mismatch",
    "no_reranking_runtime_failure_reproduced",
    "unknown_reranking_runtime_failure",
}


@dataclass(frozen=True)
class RerankingProbeResult:
    query: str
    success: bool
    error: str
    candidate_pool_input_count: int
    reranker_input_candidate_count: int
    ranked_candidate_count: int
    raw_rerank_output_count: int
    rerank_score_invalid_count: int
    rerank_ordering_valid: bool
    evidence_input_constructible: bool
    evidence_input_ranked_candidate_count: int
    fallback_used: bool
    ranked_rows: tuple[dict[str, Any], ...]


class CountingRerankerProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.invocation_count = 0
        self.batch_count = 0
        self.invocation_candidate_count = 0
        self.execution_success_count = 0
        self.execution_failure_count = 0
        self.raw_scores: list[float] = []
        self.last_error = "none"

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def model_revision(self) -> str | None:
        return self.provider.model_revision

    @property
    def input_template_version(self) -> str:
        return self.provider.input_template_version

    @property
    def max_pair_tokens(self) -> int:
        return self.provider.max_pair_tokens

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        self.invocation_count += 1
        self.batch_count += 1
        self.invocation_candidate_count += len(documents)
        try:
            scores = tuple(float(score) for score in self.provider.score_pairs(query, documents))
        except Exception as exc:
            self.execution_failure_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.execution_success_count += 1
        self.raw_scores.extend(scores)
        return scores

    def count_pair_tokens(self, query: str, document: str) -> int:
        return self.provider.count_pair_tokens(query, document)

    def prepare_pair_metadata(self, query: str, document: str):
        return self.provider.prepare_pair_metadata(query, document)


def run_task0185(
    *,
    write: bool = True,
    env: Mapping[str, str] | None = None,
    embedding_provider: Any | None = None,
    reranker_provider: Any | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = _git_head(ROOT)
    task0184 = _load_json(TASK0184_SUMMARY)
    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    reranker_config = load_reranker_config(env)
    database_url = _resolve_database_url(env)

    db = audit_database_vector_state(database_url, embedding_config)
    if embedding_provider is None and database_url and db.get("database_connection_success"):
        embedding_provider = _load_real_provider(embedding_config, {})
    candidate_pool = audit_candidate_pool(database_url, db, embedding_provider, embedding_config, search_config)
    input_contract = audit_reranker_input_contract(candidate_pool.get("candidate_rows", ()))
    model_authority = audit_reranker_model_authority(reranker_config)
    provider_probe = build_reranker_provider(reranker_config, reranker_provider)
    probes = run_reranking_probes(
        database_url,
        db,
        embedding_provider,
        embedding_config,
        search_config,
        provider_probe.get("provider"),
    )
    all_rows = tuple(row for probe in probes for row in probe.ranked_rows)
    ranked_contract = validate_ranked_candidate_contract(all_rows)
    identity = validate_candidate_identity(database_url, tuple(row["chunk_id"] for row in all_rows), db)
    membership = audit_probe_membership(probes)
    evidence = audit_evidence_boundary(probes)
    failure = determine_failure(candidate_pool, input_contract, model_authority, provider_probe, probes, ranked_contract, identity, membership, evidence)
    summary = build_summary(
        source_head=source_head,
        task0184=task0184,
        db=db,
        search_config=search_config,
        reranker_config=reranker_config,
        candidate_pool=candidate_pool,
        input_contract=input_contract,
        model_authority=model_authority,
        provider_probe=provider_probe,
        probes=probes,
        ranked_contract=ranked_contract,
        identity=identity,
        membership=membership,
        evidence=evidence,
        failure=failure,
        env=env,
    )
    artifacts = {
        "summary.json": summary,
        "runtime_authority_audit.json": runtime_authority(search_config, reranker_config),
        "candidate_pool_audit.json": _without_candidate_content(candidate_pool),
        "reranker_input_contract_audit.json": input_contract,
        "reranker_model_authority_audit.json": model_authority,
        "reranker_invocation_probe.json": [asdict(probe) for probe in probes],
        "ranked_candidate_contract_validation.json": ranked_contract,
        "candidate_identity_validation.json": identity,
        "candidate_membership_audit.json": membership,
        "evidence_input_boundary_audit.json": evidence,
        "failure_taxonomy_decision.json": failure,
        "contract.json": contract(),
    }
    if write:
        for name, payload in artifacts.items():
            write_json(RESULT_DIR / name, _redact(payload))
        write_json(CONTRACT_PATH, contract())
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return summary


def audit_candidate_pool(database_url: str, db: Mapping[str, Any], provider: Any | None, embedding_config: Any, search_config: VectorSearchConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "production_candidate_pool_audited": False,
        "production_candidate_pool_count": 0,
        "candidate_contract_valid": False,
        "candidate_identity_valid": False,
        "candidate_rows": (),
        "candidate_pool_error": "not_attempted",
    }
    if not database_url or provider is None or not db.get("knowledge_base_id"):
        out["candidate_pool_error"] = "database_or_embedding_provider_unavailable"
        return out
    try:
        config = VectorSearchConfig(
            mode=search_config.mode,
            top_k=search_config.rerank_top_n,
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
            bm25_k1=search_config.bm25_k1,
            bm25_b=search_config.bm25_b,
            rrf_k=search_config.rrf_k,
            rrf_vector_weight=search_config.rrf_vector_weight,
            rrf_bm25_weight=search_config.rrf_bm25_weight,
            rerank_enabled=False,
            reranker_policy=search_config.reranker_policy,
            rank_fusion_k=search_config.rank_fusion_k,
            rank_fusion_lambda=search_config.rank_fusion_lambda,
            context_token_budget=search_config.context_token_budget,
            context_max_chunks=search_config.context_max_chunks,
            context_max_chunks_per_document=search_config.context_max_chunks_per_document,
        )
        response, _payload = production_search_knowledge_base(
            database_url=database_url,
            knowledge_base_id=UUID(str(db["knowledge_base_id"])),
            query=PROBE_QUERIES[0],
            provider=provider,
            embedding_config=embedding_config,
            search_config=config,
            reranker_provider=None,
        )
        rows = tuple(candidate_row(result) | {"content": result.content, "heading_path": result.heading_path} for result in response.results)
        identity = validate_candidate_identity(database_url, tuple(row["chunk_id"] for row in rows), db)
        out.update(
            {
                "production_candidate_pool_audited": True,
                "production_candidate_pool_count": len(rows),
                "candidate_contract_valid": bool(rows),
                "candidate_identity_valid": identity.get("candidate_identity_valid", False),
                "candidate_rows": rows,
                "candidate_pool_error": "none",
            }
        )
    except Exception as exc:
        out["candidate_pool_error"] = f"{type(exc).__name__}: {exc}"
    return out


def audit_reranker_input_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = []
    for row in rows:
        document = render_reranker_document(tuple(row.get("heading_path") or ()), str(row.get("content") or ""))
        checks = {
            "candidate identity": bool(row.get("candidate_id")),
            "query text": True,
            "candidate text": bool(document.strip()),
            "chunk identity": bool(row.get("chunk_id")),
            "document identity": bool(row.get("document_id")),
            "retrieval score": _finite_or_none(row.get("raw_score")),
            "provenance": bool(row.get("retrieval_sources")),
            "metadata": "metadata" in row,
        }
        for prop, compatible in checks.items():
            if not compatible:
                mismatches.append({"candidate_id": row.get("candidate_id"), "property": prop, "compatible": False})
    return {
        "reranker_input_contract_valid": bool(rows) and not mismatches,
        "reranker_input_contract_mismatch_count": len(mismatches),
        "first_reranker_input_contract_mismatch": mismatches[0] if mismatches else None,
        "contract_table": [
            {"property": "candidate identity", "candidate": "candidate_id", "reranker_required": "stable identity", "compatible": not any(m["property"] == "candidate identity" for m in mismatches)},
            {"property": "query text", "candidate": "probe query", "reranker_required": "nonempty query", "compatible": True},
            {"property": "candidate text", "candidate": "heading_path + content", "reranker_required": "nonempty document", "compatible": not any(m["property"] == "candidate text" for m in mismatches)},
            {"property": "chunk identity", "candidate": "chunk_id", "reranker_required": "preserved identity", "compatible": not any(m["property"] == "chunk identity" for m in mismatches)},
            {"property": "document identity", "candidate": "document_id", "reranker_required": "preserved provenance", "compatible": not any(m["property"] == "document identity" for m in mismatches)},
            {"property": "retrieval score", "candidate": "raw_score", "reranker_required": "finite or null accepted for ordering tie-breaks", "compatible": not any(m["property"] == "retrieval score" for m in mismatches)},
            {"property": "provenance", "candidate": "retrieval_sources", "reranker_required": "source lane trace", "compatible": not any(m["property"] == "provenance" for m in mismatches)},
            {"property": "metadata", "candidate": "metadata", "reranker_required": "optional mapping", "compatible": not any(m["property"] == "metadata" for m in mismatches)},
        ],
    }


def audit_reranker_model_authority(config: RerankerConfig) -> dict[str, Any]:
    snapshot = _resolve_hf_snapshot(config)
    required = ("config.json", "tokenizer.json")
    existing = [name for name in required if snapshot and (snapshot / name).exists()]
    model_files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin")) if snapshot else []
    incomplete_markers = list(snapshot.glob("*.incomplete")) if snapshot else []
    exists = snapshot is not None and snapshot.exists()
    complete = exists and len(existing) == len(required) and bool(model_files) and not incomplete_markers
    return {
        "reranker_model_identifier": config.model_name,
        "reranker_model_revision": config.model_revision,
        "reranker_model_cache_exists": exists,
        "reranker_model_cache_complete": complete,
        "reranker_model_authority_valid": complete or not config.local_files_only,
        "reranker_cache_snapshot_path_present": bool(snapshot),
        "reranker_provider": config.provider,
        "reranker_batch_size": config.batch_size,
        "reranker_max_pair_tokens": config.max_pair_tokens,
        "reranker_local_files_only": config.local_files_only,
    }


def build_reranker_provider(config: RerankerConfig, provider: Any | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "provider": None,
        "reranker_model_load_attempted": False,
        "reranker_model_load_success": False,
        "reranker_model_load_failure_class": "not_attempted",
    }
    try:
        base_provider = provider if provider is not None else BgeLocalRerankerProvider(config)
        out["provider"] = CountingRerankerProvider(base_provider)
        out["reranker_model_load_attempted"] = True
        out["reranker_model_load_success"] = True
        out["reranker_model_load_failure_class"] = "none"
    except Exception as exc:
        out["reranker_model_load_attempted"] = True
        out["reranker_model_load_failure_class"] = type(exc).__name__
    return out


def run_reranking_probes(
    database_url: str,
    db: Mapping[str, Any],
    embedding_provider: Any | None,
    embedding_config: Any,
    search_config: VectorSearchConfig,
    reranker_provider: CountingRerankerProvider | None,
) -> tuple[RerankingProbeResult, ...]:
    probe_config = config_for_rerank_probe(search_config)
    return tuple(
        run_single_reranking_probe(database_url, db, embedding_provider, embedding_config, probe_config, reranker_provider, query)
        for query in PROBE_QUERIES
    )


def config_for_rerank_probe(search_config: VectorSearchConfig) -> VectorSearchConfig:
    return VectorSearchConfig(
        mode=search_config.mode,
        top_k=search_config.rerank_top_n,
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
        bm25_k1=search_config.bm25_k1,
        bm25_b=search_config.bm25_b,
        rrf_k=search_config.rrf_k,
        rrf_vector_weight=search_config.rrf_vector_weight,
        rrf_bm25_weight=search_config.rrf_bm25_weight,
        rerank_enabled=search_config.rerank_enabled,
        reranker_policy=search_config.reranker_policy,
        rank_fusion_k=search_config.rank_fusion_k,
        rank_fusion_lambda=search_config.rank_fusion_lambda,
        context_token_budget=search_config.context_token_budget,
        context_max_chunks=search_config.context_max_chunks,
        context_max_chunks_per_document=search_config.context_max_chunks_per_document,
    )


def run_single_reranking_probe(
    database_url: str,
    db: Mapping[str, Any],
    embedding_provider: Any | None,
    embedding_config: Any,
    search_config: VectorSearchConfig,
    reranker_provider: CountingRerankerProvider | None,
    query: str,
) -> RerankingProbeResult:
    if not database_url or embedding_provider is None or not db.get("knowledge_base_id"):
        return _failed_probe(query, "database_or_embedding_provider_unavailable")
    before_success = reranker_provider.execution_success_count if reranker_provider else 0
    before_failure = reranker_provider.execution_failure_count if reranker_provider else 0
    try:
        response, _payload = production_search_knowledge_base(
            database_url=database_url,
            knowledge_base_id=UUID(str(db["knowledge_base_id"])),
            query=query,
            provider=embedding_provider,
            embedding_config=embedding_config,
            search_config=search_config,
            reranker_provider=reranker_provider,
        )
    except Exception as exc:
        return _failed_probe(query, f"{type(exc).__name__}: {exc}")
    rows = tuple(ranked_candidate_row(result) for result in response.results)
    invalid_scores = sum(1 for row in rows if row.get("rerank_score") is None or not _finite_or_none(row.get("rerank_score")))
    fallback_used = bool(search_config.rerank_enabled and reranker_provider and reranker_provider.execution_failure_count > before_failure)
    success = response.result_count > 0 and invalid_scores == 0 and bool(reranker_provider and reranker_provider.execution_success_count > before_success)
    return RerankingProbeResult(
        query=query,
        success=success,
        error="none" if success else (reranker_provider.last_error if fallback_used and reranker_provider else "reranker_scores_missing_or_empty"),
        candidate_pool_input_count=response.candidate_count,
        reranker_input_candidate_count=response.rerank_candidate_count,
        ranked_candidate_count=response.result_count,
        raw_rerank_output_count=len(rows) - invalid_scores,
        rerank_score_invalid_count=invalid_scores,
        rerank_ordering_valid=validate_ordering(rows, search_config),
        evidence_input_constructible=response.evidence_bundle is not None and len(response.evidence_bundle.items) > 0,
        evidence_input_ranked_candidate_count=0 if response.evidence_bundle is None else len(response.evidence_bundle.items),
        fallback_used=fallback_used,
        ranked_rows=rows,
    )


def ranked_candidate_row(result: SearchResult) -> dict[str, Any]:
    row = candidate_row(result)
    row.update(
        {
            "final_score": result.rerank_score if result.rerank_score is not None else result.rrf_score or result.vector_similarity or result.bm25_score,
            "original_rank": result.original_rank,
            "rerank_rank": result.rerank_rank,
            "reranker_pair_token_count": result.reranker_pair_token_count,
            "reranker_original_pair_token_count": result.reranker_original_pair_token_count,
            "reranker_input_truncated": result.reranker_input_truncated,
            "selected_for_context": result.selected_for_context,
            "context_rank": result.context_rank,
        }
    )
    return row


def validate_ordering(rows: Sequence[Mapping[str, Any]], config: VectorSearchConfig) -> bool:
    ranks = [row.get("rank") for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        return False
    if config.reranker_policy != "rank_fusion":
        return False
    return all(row.get("rerank_rank") is None or int(row["rerank_rank"]) >= 1 for row in rows)


def validate_ranked_candidate_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("candidate_id", "chunk_id", "document_id", "retrieval_sources", "rank", "rerank_score", "final_score")
    contract_valid = bool(rows) and all(all(field in row for field in required) for row in rows)
    identity_valid = all(bool(row.get("candidate_id")) and row.get("candidate_id") == row.get("chunk_id") and bool(row.get("document_id")) for row in rows)
    score_valid = all(_finite_or_none(row.get("rerank_score")) and _finite_or_none(row.get("final_score")) for row in rows)
    provenance_valid = all(bool(row.get("retrieval_sources")) for row in rows)
    return {
        "ranked_candidate_contract_valid": contract_valid,
        "ranked_candidate_identity_valid": identity_valid,
        "ranked_candidate_score_valid": score_valid,
        "ranked_candidate_provenance_valid": provenance_valid,
        "ranked_candidate_count": len(rows),
    }


def audit_candidate_membership(input_rows: Sequence[Mapping[str, Any]], output_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    before = sorted({str(row.get("chunk_id")) for row in input_rows if row.get("chunk_id")})
    after = sorted({str(row.get("chunk_id")) for row in output_rows if row.get("chunk_id")})
    change_count = len(set(before).symmetric_difference(after))
    return {
        "candidate_membership_before_rerank": len(before),
        "candidate_membership_after_rerank": len(after),
        "candidate_membership_change_count": change_count,
        "candidate_membership_change_expected": False,
    }


def audit_probe_membership(probes: Sequence[RerankingProbeResult]) -> dict[str, Any]:
    before = sum(probe.reranker_input_candidate_count for probe in probes)
    after = sum(probe.ranked_candidate_count for probe in probes)
    return {
        "candidate_membership_before_rerank": before,
        "candidate_membership_after_rerank": after,
        "candidate_membership_change_count": abs(before - after),
        "candidate_membership_change_expected": False,
    }


def audit_evidence_boundary(probes: Sequence[RerankingProbeResult]) -> dict[str, Any]:
    count = sum(probe.evidence_input_ranked_candidate_count for probe in probes)
    return {
        "evidence_input_constructible": any(probe.evidence_input_constructible for probe in probes),
        "evidence_input_ranked_candidate_count": count,
        "evidence_composition_reached": any(probe.evidence_input_constructible for probe in probes),
    }


def determine_failure(
    candidate_pool: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    provider_probe: Mapping[str, Any],
    probes: Sequence[RerankingProbeResult],
    ranked_contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    membership: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    if candidate_pool.get("production_candidate_pool_count", 0) <= 0:
        return _failure("candidate_to_reranker_input", "candidate_reranker_contract_mismatch")
    if not input_contract.get("reranker_input_contract_valid"):
        return _failure("reranker_input_validation", "reranker_input_validation_failure")
    if not model_authority.get("reranker_model_authority_valid"):
        return _failure("reranker_model_authority", "reranker_model_cache_missing_or_incomplete")
    if not provider_probe.get("reranker_model_load_success"):
        return _failure("reranker_model_loading", "reranker_model_load_failure")
    if not any(probe.success for probe in probes):
        if any(probe.fallback_used for probe in probes):
            return _failure("reranker_execution", "reranker_runtime_execution_failure")
        return _failure("reranker_execution", "unknown_reranking_runtime_failure")
    if sum(probe.rerank_score_invalid_count for probe in probes) > 0:
        return _failure("rerank_output_validation", "reranker_score_validation_failure")
    if not all(probe.rerank_ordering_valid for probe in probes):
        return _failure("rank_fusion", "rank_fusion_runtime_failure")
    if not ranked_contract.get("ranked_candidate_contract_valid"):
        return _failure("ranked_candidate_construction", "ranked_candidate_contract_mismatch")
    if not ranked_contract.get("ranked_candidate_score_valid"):
        return _failure("ranked_candidate_score_validation", "reranker_score_validation_failure")
    if not ranked_contract.get("ranked_candidate_identity_valid") or not identity.get("candidate_identity_valid"):
        return _failure("ranked_candidate_identity_mapping", "ranked_candidate_identity_mapping_failure")
    if membership.get("candidate_membership_change_count", 0) > 0 and not membership.get("candidate_membership_change_expected"):
        return _failure("ranked_candidate_identity_mapping", "ranked_candidate_identity_mapping_failure")
    if not evidence.get("evidence_input_constructible"):
        return _failure("ranked_candidate_to_evidence_contract", "candidate_to_evidence_contract_mismatch")
    return _failure("none", "no_reranking_runtime_failure_reproduced")


def build_summary(
    *,
    source_head: str,
    task0184: Mapping[str, Any],
    db: Mapping[str, Any],
    search_config: VectorSearchConfig,
    reranker_config: RerankerConfig,
    candidate_pool: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    provider_probe: Mapping[str, Any],
    probes: Sequence[RerankingProbeResult],
    ranked_contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    membership: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure: Mapping[str, str],
    env: Mapping[str, str],
) -> dict[str, Any]:
    passed = failure["first_reranking_loss_substage"] == "none"
    provider = provider_probe.get("provider")
    input_rows = tuple(candidate_pool.get("candidate_rows", ()))
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if failure["first_reranking_loss_substage"] in LOSS_SUBSTAGES and failure["diagnosed_root_cause"] in ROOT_CAUSES else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0184.get("source_authoritative_head") or source_head,
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        "production_candidate_pool_count": candidate_pool.get("production_candidate_pool_count", 0),
        **runtime_authority(search_config, reranker_config),
        "reranking_probe_query_count": len(PROBE_QUERIES),
        "reranking_probe_execution_count": len(probes),
        "reranking_probe_success_count": sum(probe.success for probe in probes),
        "reranking_probe_failure_count": sum(not probe.success for probe in probes),
        "candidate_pool_input_count": sum(probe.candidate_pool_input_count for probe in probes),
        "reranker_eligible_candidate_count": candidate_pool.get("production_candidate_pool_count", 0),
        "reranker_input_candidate_count": sum(probe.reranker_input_candidate_count for probe in probes),
        "reranker_input_rejected_candidate_count": max(
            0,
            candidate_pool.get("production_candidate_pool_count", 0) - min(candidate_pool.get("production_candidate_pool_count", 0), search_config.rerank_top_n),
        ),
        **input_contract,
        "reranker_query_text_nonempty": all(bool(query.strip()) for query in PROBE_QUERIES),
        "reranker_candidate_text_nonempty_count": sum(1 for row in input_rows if render_reranker_document(tuple(row.get("heading_path") or ()), str(row.get("content") or "")).strip()),
        "reranker_candidate_text_empty_count": sum(1 for row in input_rows if not render_reranker_document(tuple(row.get("heading_path") or ()), str(row.get("content") or "")).strip()),
        **{key: model_authority[key] for key in ("reranker_model_cache_exists", "reranker_model_cache_complete", "reranker_model_authority_valid")},
        "reranker_model_load_attempted": provider_probe.get("reranker_model_load_attempted", False),
        "reranker_model_load_success": provider_probe.get("reranker_model_load_success", False),
        "reranker_model_load_failure_class": provider_probe.get("reranker_model_load_failure_class", "not_attempted"),
        "reranker_invocation_count": provider.invocation_count if provider else 0,
        "reranker_batch_count": provider.batch_count if provider else 0,
        "reranker_invocation_candidate_count": provider.invocation_candidate_count if provider else 0,
        "reranker_execution_success_count": provider.execution_success_count if provider else 0,
        "reranker_execution_failure_count": provider.execution_failure_count if provider else 0,
        "raw_rerank_output_count": sum(probe.raw_rerank_output_count for probe in probes),
        "rerank_score_finite_count": sum(probe.raw_rerank_output_count for probe in probes),
        "rerank_score_invalid_count": sum(probe.rerank_score_invalid_count for probe in probes),
        "rerank_ordering_valid": bool(probes) and all(probe.rerank_ordering_valid for probe in probes),
        "guarded_rank_fusion_enabled": search_config.rerank_enabled and search_config.reranker_policy == "rank_fusion",
        "guarded_rank_fusion_invocation_count": sum(1 for probe in probes if probe.raw_rerank_output_count > 0),
        "guarded_rank_fusion_success_count": sum(1 for probe in probes if probe.success and probe.rerank_ordering_valid),
        "rank_fusion_k": search_config.rank_fusion_k,
        "rank_fusion_lambda": search_config.rank_fusion_lambda,
        "reranker_fallback_count": sum(probe.fallback_used for probe in probes),
        "expected_reranker_fallback_count": 0,
        "unexpected_reranker_fallback_count": sum(probe.fallback_used for probe in probes),
        "ranked_candidate_count": ranked_contract.get("ranked_candidate_count", 0),
        **ranked_contract,
        "reranker_identity_preserved": identity.get("candidate_identity_valid", False) and membership.get("candidate_membership_change_count", 1) == 0,
        "reranked_orphan_candidate_count": identity.get("orphan_candidate_count", 0),
        **membership,
        **evidence,
        "first_reranking_loss_substage": failure["first_reranking_loss_substage"],
        "diagnosed_root_cause": failure["diagnosed_root_cause"],
        "cold_start_reranking_stage_passed": passed,
        "next_failure_stage": "evidence_composition" if passed else f"{failure['first_reranking_loss_substage']}_repair",
        "retrieval_policy_changed": False,
        "reranking_policy_changed": False,
        "evidence_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "focused_test_passed_count": int(env.get("TASK0185_FOCUSED_TEST_PASSED_COUNT", "0")),
        "full_suite_passed_count": int(env.get("TASK0185_FULL_SUITE_PASSED_COUNT", "0")),
        "full_suite_skipped_count": int(env.get("TASK0185_FULL_SUITE_SKIPPED_COUNT", "0")),
        "full_suite_failed_count": int(env.get("TASK0185_FULL_SUITE_FAILED_COUNT", "0")),
        "known_preexisting_failure_count": int(env.get("TASK0185_KNOWN_PREEXISTING_FAILURE_COUNT", "0")),
        "new_regression_count": int(env.get("TASK0185_NEW_REGRESSION_COUNT", "0")),
    }


def runtime_authority(search_config: VectorSearchConfig, reranker_config: RerankerConfig) -> dict[str, Any]:
    return {
        "default_reranking_policy": search_config.reranker_policy,
        "default_reranker_arm": "bge_guarded_rank_fusion" if search_config.rerank_enabled and reranker_config.provider == "local_bge_cross_encoder" and search_config.reranker_policy == "rank_fusion" else search_config.reranker_policy,
        "reranker_enabled": search_config.rerank_enabled,
        "reranker_guard_enabled": search_config.reranker_policy == "rank_fusion",
        "reranker_model_identifier": reranker_config.model_name,
        "reranker_model_revision": reranker_config.model_revision,
        "reranker_fusion_policy": search_config.reranker_policy,
        "reranker_guard_policy": "fail_to_baseline_retrieval_on_reranker_exception",
        "reranker_score_semantics": RERANKER_SCORE_SEMANTICS,
    }


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0185.contract.v1",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "diagnosis_first": True,
        "runtime_mutation_allowed": False,
        "gold_runtime_routing_allowed": False,
        "expected_authoritative_document_count": 8,
        "expected_authoritative_chunk_count": 11,
        "expected_embedding_count": 11,
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
        "loss_substage_taxonomy": sorted(LOSS_SUBSTAGES),
        "root_cause_taxonomy": sorted(ROOT_CAUSES),
    }


REQUIRED_SUMMARY_FIELDS = (
    "task_id",
    "task_status",
    "source_authoritative_task",
    "source_authoritative_head",
    "authoritative_document_count",
    "authoritative_chunk_count",
    "stored_embedding_count",
    "production_candidate_pool_count",
    "default_reranking_policy",
    "default_reranker_arm",
    "reranker_enabled",
    "reranker_guard_enabled",
    "reranker_model_identifier",
    "reranker_model_revision",
    "reranking_probe_query_count",
    "reranking_probe_execution_count",
    "reranking_probe_success_count",
    "reranking_probe_failure_count",
    "candidate_pool_input_count",
    "reranker_eligible_candidate_count",
    "reranker_input_candidate_count",
    "reranker_input_rejected_candidate_count",
    "reranker_input_contract_valid",
    "reranker_input_contract_mismatch_count",
    "first_reranker_input_contract_mismatch",
    "reranker_query_text_nonempty",
    "reranker_candidate_text_nonempty_count",
    "reranker_candidate_text_empty_count",
    "reranker_model_cache_exists",
    "reranker_model_cache_complete",
    "reranker_model_authority_valid",
    "reranker_model_load_attempted",
    "reranker_model_load_success",
    "reranker_model_load_failure_class",
    "reranker_invocation_count",
    "reranker_batch_count",
    "reranker_invocation_candidate_count",
    "reranker_execution_success_count",
    "reranker_execution_failure_count",
    "raw_rerank_output_count",
    "rerank_score_finite_count",
    "rerank_score_invalid_count",
    "rerank_ordering_valid",
    "guarded_rank_fusion_enabled",
    "guarded_rank_fusion_invocation_count",
    "guarded_rank_fusion_success_count",
    "rank_fusion_k",
    "rank_fusion_lambda",
    "reranker_fallback_count",
    "expected_reranker_fallback_count",
    "unexpected_reranker_fallback_count",
    "ranked_candidate_count",
    "ranked_candidate_contract_valid",
    "ranked_candidate_identity_valid",
    "ranked_candidate_score_valid",
    "ranked_candidate_provenance_valid",
    "reranker_identity_preserved",
    "reranked_orphan_candidate_count",
    "candidate_membership_before_rerank",
    "candidate_membership_after_rerank",
    "candidate_membership_change_count",
    "candidate_membership_change_expected",
    "evidence_input_constructible",
    "evidence_input_ranked_candidate_count",
    "evidence_composition_reached",
    "first_reranking_loss_substage",
    "diagnosed_root_cause",
    "cold_start_reranking_stage_passed",
    "next_failure_stage",
    "retrieval_policy_changed",
    "reranking_policy_changed",
    "evidence_policy_changed",
    "runtime_default_behavior_change",
    "promotion_applied",
    "focused_test_passed_count",
    "full_suite_passed_count",
    "full_suite_skipped_count",
    "full_suite_failed_count",
    "known_preexisting_failure_count",
    "new_regression_count",
)


def verify_task0185_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    artifact_names = (
        "summary.json",
        "runtime_authority_audit.json",
        "candidate_pool_audit.json",
        "reranker_input_contract_audit.json",
        "reranker_model_authority_audit.json",
        "reranker_invocation_probe.json",
        "ranked_candidate_contract_validation.json",
        "candidate_identity_validation.json",
        "candidate_membership_audit.json",
        "evidence_input_boundary_audit.json",
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
        and summary.get("reranking_policy_changed") is False
        and summary.get("evidence_policy_changed") is False
        and summary.get("runtime_default_behavior_change") is False
        and summary.get("promotion_applied") is False
        and summary.get("new_regression_count") == 0
    )
    return {
        "task_id": TASK_ID,
        "verification_passed": passed,
        "missing_artifacts": missing,
        "missing_summary_fields": missing_fields,
        "first_reranking_loss_substage": summary.get("first_reranking_loss_substage"),
        "diagnosed_root_cause": summary.get("diagnosed_root_cause"),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK-0185 Cold-start Reranking Runtime Validation and Diagnosis Report",
            "",
            "## Decision",
            f"task_status=`{summary.get('task_status')}`; first_reranking_loss_substage=`{summary.get('first_reranking_loss_substage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`; cold_start_reranking_stage_passed=`{summary.get('cold_start_reranking_stage_passed')}`.",
            "",
            "## Runtime Authority",
            f"default_reranker_arm=`{summary.get('default_reranker_arm')}`; policy=`{summary.get('default_reranking_policy')}`; enabled=`{summary.get('reranker_enabled')}`; guard_enabled=`{summary.get('reranker_guard_enabled')}`.",
            f"model=`{summary.get('reranker_model_identifier')}`; revision=`{summary.get('reranker_model_revision')}`; rank_fusion_k=`{summary.get('rank_fusion_k')}`; lambda=`{summary.get('rank_fusion_lambda')}`.",
            "",
            "## Corpus And Candidate Pool",
            f"documents=`{summary.get('authoritative_document_count')}`; chunks=`{summary.get('authoritative_chunk_count')}`; stored_embeddings=`{summary.get('stored_embedding_count')}`; production_candidate_pool_count=`{summary.get('production_candidate_pool_count')}`.",
            "",
            "## Reranker Input And Model",
            f"input_contract_valid=`{summary.get('reranker_input_contract_valid')}`; mismatches=`{summary.get('reranker_input_contract_mismatch_count')}`; candidate_text_empty=`{summary.get('reranker_candidate_text_empty_count')}`.",
            f"cache_exists=`{summary.get('reranker_model_cache_exists')}`; cache_complete=`{summary.get('reranker_model_cache_complete')}`; authority_valid=`{summary.get('reranker_model_authority_valid')}`; model_load_success=`{summary.get('reranker_model_load_success')}`.",
            "",
            "## Invocation And Ranking",
            f"probe_queries=`{summary.get('reranking_probe_query_count')}`; executions=`{summary.get('reranking_probe_execution_count')}`; successes=`{summary.get('reranking_probe_success_count')}`; failures=`{summary.get('reranking_probe_failure_count')}`.",
            f"invocations=`{summary.get('reranker_invocation_count')}`; candidates=`{summary.get('reranker_invocation_candidate_count')}`; raw_outputs=`{summary.get('raw_rerank_output_count')}`; invalid_scores=`{summary.get('rerank_score_invalid_count')}`; ordering_valid=`{summary.get('rerank_ordering_valid')}`.",
            f"rank_fusion_success_count=`{summary.get('guarded_rank_fusion_success_count')}`; unexpected_fallbacks=`{summary.get('unexpected_reranker_fallback_count')}`.",
            "",
            "## Ranked Candidate Boundary",
            f"ranked_candidate_count=`{summary.get('ranked_candidate_count')}`; contract_valid=`{summary.get('ranked_candidate_contract_valid')}`; identity_valid=`{summary.get('ranked_candidate_identity_valid')}`; score_valid=`{summary.get('ranked_candidate_score_valid')}`; provenance_valid=`{summary.get('ranked_candidate_provenance_valid')}`.",
            f"identity_preserved=`{summary.get('reranker_identity_preserved')}`; membership_change_count=`{summary.get('candidate_membership_change_count')}`; orphan_count=`{summary.get('reranked_orphan_candidate_count')}`.",
            "",
            "## Evidence Boundary",
            f"evidence_input_constructible=`{summary.get('evidence_input_constructible')}`; evidence_input_ranked_candidate_count=`{summary.get('evidence_input_ranked_candidate_count')}`; evidence_composition_reached=`{summary.get('evidence_composition_reached')}`.",
            "",
            "## Policy Mutation",
            f"retrieval_policy_changed=`{summary.get('retrieval_policy_changed')}`; reranking_policy_changed=`{summary.get('reranking_policy_changed')}`; evidence_policy_changed=`{summary.get('evidence_policy_changed')}`; runtime_default_behavior_change=`{summary.get('runtime_default_behavior_change')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Test Accounting",
            f"Focused tests: `{summary.get('focused_test_passed_count')}` passed. Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`, new_regression_count=`{summary.get('new_regression_count')}`.",
            "",
            "## Next Frontier",
            f"next_failure_stage=`{summary.get('next_failure_stage')}`.",
            "",
        ]
    )


def _failed_probe(query: str, error: str) -> RerankingProbeResult:
    return RerankingProbeResult(query, False, error, 0, 0, 0, 0, 0, False, False, 0, False, ())


def _failure(substage: str, root: str) -> dict[str, str]:
    return {"first_reranking_loss_substage": substage, "diagnosed_root_cause": root}


def _finite_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def _resolve_hf_snapshot(config: RerankerConfig) -> Path | None:
    root = config.cache_dir or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    model_dir = root / f"models--{config.model_name.replace('/', '--')}"
    if config.model_revision:
        snapshot = model_dir / "snapshots" / config.model_revision
        if snapshot.exists():
            return snapshot
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        candidates = sorted((path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.name)
        return candidates[-1] if candidates else snapshots
    return model_dir


def _without_candidate_content(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    redacted["candidate_rows"] = [
        {key: value for key, value in row.items() if key not in {"content"}}
        for row in payload.get("candidate_rows", ())
    ]
    return redacted


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"
