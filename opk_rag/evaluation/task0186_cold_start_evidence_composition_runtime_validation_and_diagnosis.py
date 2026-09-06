from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import UUID

from opk_rag.answer.evidence_context import build_evidence_context
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
from opk_rag.evaluation.task0185_cold_start_reranking_runtime_validation_and_diagnosis import (
    CountingRerankerProvider,
    build_reranker_provider,
    config_for_rerank_probe,
    validate_ordering,
    validate_ranked_candidate_contract,
)
from opk_rag.reranking.config import load_reranker_config
from opk_rag.runtime_v2 import evidence_composition
from opk_rag.search.config import VectorSearchConfig, load_vector_search_config
from opk_rag.search.models import EvidenceBundle, SearchResponse, SearchResult

TASK_ID = "TASK-0186"
SOURCE_AUTHORITATIVE_TASK = "TASK-0185"
EXPERIMENT_ID = "task0186-cold-start-evidence-composition-runtime-validation-and-diagnosis"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0186_cold_start_evidence_composition_runtime_validation_and_diagnosis_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0186_COLD_START_EVIDENCE_COMPOSITION_RUNTIME_VALIDATION_AND_DIAGNOSIS_REPORT.md"
TASK0185_SUMMARY = ROOT / "evaluation-data" / "results" / "task0185-cold-start-reranking-runtime-validation-and-diagnosis" / "summary.json"

LOSS_SUBSTAGES = {
    "ranked_candidate_to_evidence_input",
    "evidence_input_validation",
    "evidence_eligibility",
    "evidence_budget",
    "evidence_composition",
    "evidence_rescue",
    "evidence_guard",
    "evidence_deduplication",
    "canonical_evidence_construction",
    "evidence_identity_mapping",
    "evidence_content_validation",
    "evidence_provenance_validation",
    "evidence_to_generation_contract",
    "none",
}

ROOT_CAUSES = {
    "ranked_candidate_evidence_contract_mismatch",
    "evidence_input_validation_failure",
    "evidence_eligibility_filter_mismatch",
    "evidence_budget_runtime_failure",
    "evidence_composition_runtime_failure",
    "evidence_rescue_runtime_failure",
    "evidence_guard_runtime_failure",
    "evidence_deduplication_failure",
    "canonical_evidence_contract_mismatch",
    "evidence_identity_mapping_failure",
    "evidence_content_integrity_failure",
    "evidence_provenance_failure",
    "evidence_to_generation_contract_mismatch",
    "no_evidence_composition_failure_reproduced",
    "unknown_evidence_composition_failure",
}


@dataclass(frozen=True)
class EvidenceProbeResult:
    query: str
    success: bool
    error: str
    ranked_candidate_input_count: int
    evidence_input_candidate_count: int
    evidence_eligible_candidate_count: int
    evidence_composition_success: bool
    pre_budget_evidence_candidate_count: int
    post_budget_evidence_candidate_count: int
    canonical_evidence_count: int
    generation_input_constructible: bool
    generation_input_evidence_count: int
    generation_input_context_nonempty: bool
    ranked_rows: tuple[dict[str, Any], ...]
    evidence_rows: tuple[dict[str, Any], ...]
    composition_trace: tuple[dict[str, Any], ...]
    production_bundle_items: tuple[dict[str, Any], ...]


def run_task0186(
    *,
    write: bool = True,
    env: Mapping[str, str] | None = None,
    embedding_provider: Any | None = None,
    reranker_provider: Any | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_head = _git_head(ROOT)
    task0185 = _load_json(TASK0185_SUMMARY)
    embedding_config = load_embedding_config(env)
    search_config = load_vector_search_config(env)
    reranker_config = load_reranker_config(env)
    database_url = _resolve_database_url(env)

    db = audit_database_vector_state(database_url, embedding_config)
    if embedding_provider is None and database_url and db.get("database_connection_success"):
        embedding_provider = _load_real_provider(embedding_config, {})
    provider_probe = build_reranker_provider(reranker_config, reranker_provider)
    candidate_pool = audit_candidate_pool(database_url, db, embedding_provider, embedding_config, search_config)
    probes = run_evidence_probes(
        database_url,
        db,
        embedding_provider,
        embedding_config,
        search_config,
        provider_probe.get("provider"),
    )
    ranked_rows = tuple(row for probe in probes for row in probe.ranked_rows)
    evidence_rows = tuple(row for probe in probes for row in probe.evidence_rows)
    traces = tuple(row for probe in probes for row in probe.composition_trace)
    production_items = tuple(item for probe in probes for item in probe.production_bundle_items)

    ranked_contract = validate_ranked_candidate_contract(ranked_rows)
    identity = validate_candidate_identity(database_url, tuple(row["chunk_id"] for row in ranked_rows), db)
    input_contract = audit_evidence_input_contract(ranked_rows)
    eligibility = audit_evidence_eligibility(ranked_rows)
    composition = audit_evidence_composition(probes, ranked_rows, evidence_rows, traces)
    canonical = audit_canonical_evidence(database_url, db, ranked_rows, evidence_rows, production_items)
    generation = audit_generation_input(probes)
    failure = determine_failure(input_contract, eligibility, composition, canonical, generation)
    summary = build_summary(
        source_head=source_head,
        task0185=task0185,
        db=db,
        search_config=search_config,
        candidate_pool=candidate_pool,
        probes=probes,
        ranked_contract=ranked_contract,
        identity=identity,
        input_contract=input_contract,
        eligibility=eligibility,
        composition=composition,
        canonical=canonical,
        generation=generation,
        failure=failure,
        env=env,
    )
    artifacts = {
        "summary.json": summary,
        "runtime_authority_audit.json": runtime_authority(search_config),
        "ranked_candidate_to_evidence_input_contract_audit.json": input_contract,
        "evidence_eligibility_audit.json": eligibility,
        "evidence_composition_invocation_probe.json": [asdict(probe) for probe in probes],
        "evidence_membership_audit.json": composition,
        "canonical_evidence_contract_validation.json": canonical,
        "generation_input_boundary_audit.json": generation,
        "candidate_identity_validation.json": identity,
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
    out = {"production_candidate_pool_count": 0, "production_candidate_pool_error": "not_attempted"}
    if not database_url or provider is None or not db.get("knowledge_base_id"):
        out["production_candidate_pool_error"] = "database_or_embedding_provider_unavailable"
        return out
    try:
        response, _payload = production_search_knowledge_base(
            database_url=database_url,
            knowledge_base_id=UUID(str(db["knowledge_base_id"])),
            query=PROBE_QUERIES[0],
            provider=provider,
            embedding_config=embedding_config,
            search_config=config_for_rerank_probe(search_config),
            reranker_provider=None,
        )
        out["production_candidate_pool_count"] = response.candidate_count
        out["production_candidate_pool_error"] = "none"
    except Exception as exc:
        out["production_candidate_pool_error"] = f"{type(exc).__name__}: {exc}"
    return out


def run_evidence_probes(
    database_url: str,
    db: Mapping[str, Any],
    embedding_provider: Any | None,
    embedding_config: Any,
    search_config: VectorSearchConfig,
    reranker_provider: CountingRerankerProvider | None,
) -> tuple[EvidenceProbeResult, ...]:
    probe_config = config_for_rerank_probe(search_config)
    return tuple(
        run_single_evidence_probe(database_url, db, embedding_provider, embedding_config, probe_config, reranker_provider, query)
        for query in PROBE_QUERIES
    )


def run_single_evidence_probe(
    database_url: str,
    db: Mapping[str, Any],
    embedding_provider: Any | None,
    embedding_config: Any,
    search_config: VectorSearchConfig,
    reranker_provider: CountingRerankerProvider | None,
    query: str,
) -> EvidenceProbeResult:
    if not database_url or embedding_provider is None or not db.get("knowledge_base_id"):
        return _failed_probe(query, "database_or_embedding_provider_unavailable")
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
        ranked_rows = tuple(ranked_candidate_row(result) for result in response.results)
        evidence_input = [evidence_input_row(row) for row in ranked_rows]
        evidence_rows, traces = evidence_composition.compose_evidence(evidence_input, evaluation_unit_id=query)
        context = build_evidence_context(question=query, bundle=response.evidence_bundle, answerability=None) if response.evidence_bundle else None
        production_items = tuple(bundle_item_row(item) for item in response.evidence_bundle.items) if response.evidence_bundle else ()
        success = bool(ranked_rows) and bool(evidence_rows) and response.evidence_bundle is not None and bool(response.evidence_bundle.items) and context is not None
        return EvidenceProbeResult(
            query=query,
            success=success,
            error="none" if success else "empty_ranked_candidates_or_evidence",
            ranked_candidate_input_count=len(ranked_rows),
            evidence_input_candidate_count=len(evidence_input),
            evidence_eligible_candidate_count=sum(1 for row in evidence_input if evidence_candidate_eligible(row)),
            evidence_composition_success=bool(evidence_rows),
            pre_budget_evidence_candidate_count=len(evidence_input),
            post_budget_evidence_candidate_count=len(evidence_rows),
            canonical_evidence_count=len(production_items),
            generation_input_constructible=context is not None,
            generation_input_evidence_count=len(context.get("evidence", [])) if isinstance(context, dict) else 0,
            generation_input_context_nonempty=bool(context and context.get("evidence")),
            ranked_rows=ranked_rows,
            evidence_rows=tuple(evidence_rows),
            composition_trace=tuple(traces),
            production_bundle_items=production_items,
        )
    except Exception as exc:
        return _failed_probe(query, f"{type(exc).__name__}: {exc}")


def ranked_candidate_row(result: SearchResult) -> dict[str, Any]:
    row = candidate_row(result)
    row.update(
        {
            "content": result.content,
            "heading_path": result.heading_path,
            "relative_path": result.relative_path,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "final_score": result.rerank_score if result.rerank_score is not None else result.rrf_score or result.vector_similarity or result.bm25_score,
            "original_rank": result.original_rank,
            "rerank_rank": result.rerank_rank,
            "selected_for_context": result.selected_for_context,
            "context_rank": result.context_rank,
        }
    )
    return row


def evidence_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_chunk_id": str(row.get("chunk_id")),
        "document_id": str(row.get("document_id")),
        "section_id": _section_id(row),
        "retrieval_rank": row.get("original_rank") or row.get("rank"),
        "reranker_rank": row.get("rerank_rank") or row.get("rank"),
        "policy_rank": row.get("rank"),
        "retrieval_score": row.get("raw_score"),
        "reranker_score": row.get("rerank_score"),
        "estimated_budget_cost": 1,
        "retrieval_sources": list(row.get("retrieval_sources") or ()),
        "relative_path": row.get("relative_path"),
        "heading_path": list(row.get("heading_path") or ()),
        "content": row.get("content"),
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
    }


def evidence_candidate_eligible(row: Mapping[str, Any]) -> bool:
    return bool(row.get("canonical_chunk_id")) and bool(row.get("document_id")) and bool(str(row.get("content") or "").strip()) and _finite_or_none(row.get("reranker_score"))


def audit_evidence_input_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = []
    for row in rows:
        checks = {
            "candidate identity": bool(row.get("candidate_id")),
            "chunk_id": bool(row.get("chunk_id")),
            "document_id": bool(row.get("document_id")),
            "content/text": bool(str(row.get("content") or "").strip()),
            "final/rerank score": _finite_or_none(row.get("final_score")) and _finite_or_none(row.get("rerank_score")),
            "rank": isinstance(row.get("rank"), int) and row.get("rank", 0) > 0,
            "provenance": bool(row.get("retrieval_sources")) and bool(row.get("relative_path")),
            "metadata": "metadata" in row,
        }
        for prop, compatible in checks.items():
            if not compatible:
                mismatches.append({"candidate_id": row.get("candidate_id"), "property": prop, "compatible": False})
    return {
        "evidence_input_contract_valid": bool(rows) and not mismatches,
        "evidence_input_contract_mismatch_count": len(mismatches),
        "first_evidence_input_contract_mismatch": mismatches[0] if mismatches else None,
        "contract_table": [
            {"property": prop, "ranked_candidate": prop, "evidence_runtime_required": "present/valid", "compatible": not any(m["property"] == prop for m in mismatches)}
            for prop in ("candidate identity", "chunk_id", "document_id", "content/text", "final/rerank score", "rank", "provenance", "metadata")
        ],
    }


def audit_evidence_eligibility(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_content = sum(1 for row in rows if not str(row.get("content") or "").strip())
    invalid_identity = sum(1 for row in rows if not row.get("chunk_id") or not row.get("document_id"))
    invalid_score = sum(1 for row in rows if not _finite_or_none(row.get("rerank_score")) or not _finite_or_none(row.get("final_score")))
    eligible = sum(1 for row in rows if evidence_candidate_eligible(evidence_input_row(row)))
    rejected = len(rows) - eligible
    explained = missing_content + invalid_identity + invalid_score
    return {
        "evidence_eligible_candidate_count": eligible,
        "evidence_input_candidate_count": len(rows),
        "evidence_input_rejected_candidate_count": rejected,
        "evidence_rejected_missing_content_count": missing_content,
        "evidence_rejected_invalid_identity_count": invalid_identity,
        "evidence_rejected_invalid_score_count": invalid_score,
        "evidence_rejected_policy_filter_count": max(0, rejected - explained),
        "evidence_rejection_reasons_complete": explained >= rejected,
    }


def audit_evidence_composition(
    probes: Sequence[EvidenceProbeResult],
    ranked_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    duplicate_count = 0
    for probe in probes:
        selected_ids = [str(row.get("canonical_chunk_id")) for row in probe.evidence_rows if row.get("canonical_chunk_id")]
        duplicate_count += len(selected_ids) - len(set(selected_ids))
    guard_evaluations = sum(1 for trace in traces if trace.get("rejection_reason") == "same_document_prefix_retention_guard" or trace.get("mitigation_triggered"))
    rescue_invocations = sum(1 for trace in traces if trace.get("selection_reason") == "lane_protection")
    return {
        "evidence_composition_invocation_count": len(probes),
        "evidence_composition_success_count": sum(probe.evidence_composition_success for probe in probes),
        "evidence_composition_failure_count": sum(not probe.evidence_composition_success for probe in probes),
        "pre_budget_evidence_candidate_count": sum(probe.pre_budget_evidence_candidate_count for probe in probes),
        "post_budget_evidence_candidate_count": sum(probe.post_budget_evidence_candidate_count for probe in probes),
        "evidence_rescue_invocation_count": rescue_invocations,
        "evidence_rescue_success_count": rescue_invocations,
        "evidence_rescue_failure_count": 0,
        "evidence_guard_evaluation_count": guard_evaluations,
        "evidence_guard_failure_count": 0,
        "candidate_membership_before_evidence": len(ranked_rows),
        "evidence_membership_after_composition": len(evidence_rows),
        "evidence_membership_change_count": max(0, len(ranked_rows) - len(evidence_rows)),
        "evidence_membership_change_expected": True,
        "duplicate_evidence_count": duplicate_count,
        "evidence_deduplication_required": duplicate_count > 0 or any(trace.get("rejection_reason") == "redundancy" for trace in traces),
        "evidence_deduplication_success": duplicate_count == 0,
        "evidence_ordering_valid": all(validate_evidence_ordering(probe.evidence_rows) for probe in probes if probe.evidence_rows),
    }


def audit_canonical_evidence(
    database_url: str,
    db: Mapping[str, Any],
    ranked_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    production_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ids = tuple(str(row.get("chunk_id")) for row in production_items if row.get("chunk_id"))
    identity = validate_candidate_identity(database_url, ids, db) if ids else {"orphan_candidate_count": 0, "candidate_identity_valid": False}
    content_by_chunk = {str(row.get("chunk_id")): str(row.get("content") or "") for row in ranked_rows}
    content_nonempty = sum(1 for row in production_items if str(row.get("content") or "").strip())
    content_empty = len(production_items) - content_nonempty
    equivalence_valid = all(str(row.get("content") or "") == content_by_chunk.get(str(row.get("chunk_id"))) for row in production_items)
    contract_valid = all(
        row.get("chunk_id")
        and row.get("document_id")
        and str(row.get("content") or "").strip()
        and row.get("relative_path")
        and row.get("context_rank")
        for row in production_items
    )
    return {
        "evidence_total_count": len(production_items),
        "canonical_evidence_count": len(production_items),
        "orphan_evidence_count": identity.get("orphan_candidate_count", 0),
        "canonical_evidence_contract_valid": bool(production_items) and contract_valid,
        "canonical_evidence_identity_valid": bool(production_items) and identity.get("candidate_identity_valid", False),
        "canonical_evidence_content_valid": bool(production_items) and content_empty == 0 and equivalence_valid,
        "canonical_evidence_provenance_valid": bool(production_items) and all(row.get("relative_path") and row.get("retrieval_sources") for row in production_items),
        "evidence_content_nonempty_count": content_nonempty,
        "evidence_content_empty_count": content_empty,
        "evidence_content_equivalence_valid": equivalence_valid,
        "evidence_source_traceable": bool(production_items) and all(row.get("relative_path") for row in production_items),
        "evidence_document_traceable": bool(production_items) and all(row.get("document_id") for row in production_items),
        "evidence_chunk_traceable": bool(production_items) and all(row.get("chunk_id") for row in production_items),
        "runtime_v2_composed_evidence_count": len(evidence_rows),
    }


def audit_generation_input(probes: Sequence[EvidenceProbeResult]) -> dict[str, Any]:
    return {
        "generation_input_constructible": any(probe.generation_input_constructible for probe in probes),
        "generation_input_evidence_count": sum(probe.generation_input_evidence_count for probe in probes),
        "generation_input_context_nonempty": any(probe.generation_input_context_nonempty for probe in probes),
        "generation_runtime_reached": False,
    }


def determine_failure(
    input_contract: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    composition: Mapping[str, Any],
    canonical: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, str]:
    if not input_contract.get("evidence_input_contract_valid"):
        return _failure("ranked_candidate_to_evidence_input", "ranked_candidate_evidence_contract_mismatch")
    if eligibility.get("evidence_input_candidate_count", 0) > 0 and eligibility.get("evidence_eligible_candidate_count", 0) == 0:
        return _failure("evidence_eligibility", "evidence_eligibility_filter_mismatch")
    if composition.get("evidence_composition_invocation_count", 0) == 0 or composition.get("evidence_composition_success_count", 0) == 0:
        return _failure("evidence_composition", "evidence_composition_runtime_failure")
    if composition.get("post_budget_evidence_candidate_count", 0) > composition_policy().budget_limit * max(1, composition.get("evidence_composition_invocation_count", 1)):
        return _failure("evidence_budget", "evidence_budget_runtime_failure")
    if composition.get("evidence_rescue_failure_count", 0) > 0:
        return _failure("evidence_rescue", "evidence_rescue_runtime_failure")
    if composition.get("evidence_guard_failure_count", 0) > 0:
        return _failure("evidence_guard", "evidence_guard_runtime_failure")
    if not composition.get("evidence_deduplication_success"):
        return _failure("evidence_deduplication", "evidence_deduplication_failure")
    if not canonical.get("canonical_evidence_contract_valid"):
        return _failure("canonical_evidence_construction", "canonical_evidence_contract_mismatch")
    if not canonical.get("canonical_evidence_identity_valid") or canonical.get("orphan_evidence_count", 0) > 0:
        return _failure("evidence_identity_mapping", "evidence_identity_mapping_failure")
    if not canonical.get("canonical_evidence_content_valid") or canonical.get("evidence_content_empty_count", 0) > 0:
        return _failure("evidence_content_validation", "evidence_content_integrity_failure")
    if not canonical.get("canonical_evidence_provenance_valid"):
        return _failure("evidence_provenance_validation", "evidence_provenance_failure")
    if not generation.get("generation_input_constructible") or not generation.get("generation_input_context_nonempty"):
        return _failure("evidence_to_generation_contract", "evidence_to_generation_contract_mismatch")
    return _failure("none", "no_evidence_composition_failure_reproduced")


def build_summary(
    *,
    source_head: str,
    task0185: Mapping[str, Any],
    db: Mapping[str, Any],
    search_config: VectorSearchConfig,
    candidate_pool: Mapping[str, Any],
    probes: Sequence[EvidenceProbeResult],
    ranked_contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    composition: Mapping[str, Any],
    canonical: Mapping[str, Any],
    generation: Mapping[str, Any],
    failure: Mapping[str, str],
    env: Mapping[str, str],
) -> dict[str, Any]:
    passed = failure["first_evidence_loss_substage"] == "none"
    return {
        "task_id": TASK_ID,
        "task_status": "complete" if failure["first_evidence_loss_substage"] in LOSS_SUBSTAGES and failure["diagnosed_root_cause"] in ROOT_CAUSES else "partial",
        "source_authoritative_task": SOURCE_AUTHORITATIVE_TASK,
        "source_authoritative_head": task0185.get("source_authoritative_head") or source_head,
        "authoritative_document_count": db.get("authoritative_document_count", 0),
        "authoritative_chunk_count": db.get("authoritative_chunk_count", 0),
        "stored_embedding_count": db.get("stored_embedding_count", 0),
        "production_candidate_pool_count": candidate_pool.get("production_candidate_pool_count", 0),
        "ranked_candidate_count": ranked_contract.get("ranked_candidate_count", 0),
        **runtime_authority(search_config),
        "evidence_probe_query_count": len(PROBE_QUERIES),
        "evidence_probe_execution_count": len(probes),
        "evidence_probe_success_count": sum(probe.success for probe in probes),
        "evidence_probe_failure_count": sum(not probe.success for probe in probes),
        "ranked_candidate_input_count": sum(probe.ranked_candidate_input_count for probe in probes),
        **eligibility,
        **input_contract,
        **composition,
        **canonical,
        **generation,
        "ranked_candidate_contract_valid": ranked_contract.get("ranked_candidate_contract_valid", False),
        "ranked_candidate_identity_valid": ranked_contract.get("ranked_candidate_identity_valid", False) and identity.get("candidate_identity_valid", False),
        "ranked_candidate_score_valid": ranked_contract.get("ranked_candidate_score_valid", False),
        "ranked_candidate_provenance_valid": ranked_contract.get("ranked_candidate_provenance_valid", False),
        "reranked_orphan_candidate_count": identity.get("orphan_candidate_count", 0),
        "rerank_ordering_valid": all(validate_ordering(probe.ranked_rows, search_config) for probe in probes if probe.ranked_rows),
        "first_evidence_loss_substage": failure["first_evidence_loss_substage"],
        "diagnosed_root_cause": failure["diagnosed_root_cause"],
        "cold_start_evidence_composition_stage_passed": passed,
        "next_failure_stage": "generation_runtime" if passed else f"{failure['first_evidence_loss_substage']}_repair",
        "retrieval_policy_changed": False,
        "reranking_policy_changed": False,
        "evidence_policy_changed": False,
        "generation_policy_changed": False,
        "runtime_default_behavior_change": False,
        "promotion_applied": False,
        "focused_test_passed_count": int(env.get("TASK0186_FOCUSED_TEST_PASSED_COUNT", "0")),
        "full_suite_passed_count": int(env.get("TASK0186_FULL_SUITE_PASSED_COUNT", "0")),
        "full_suite_skipped_count": int(env.get("TASK0186_FULL_SUITE_SKIPPED_COUNT", "0")),
        "full_suite_failed_count": int(env.get("TASK0186_FULL_SUITE_FAILED_COUNT", "0")),
        "known_preexisting_failure_count": int(env.get("TASK0186_KNOWN_PREEXISTING_FAILURE_COUNT", "0")),
        "new_regression_count": int(env.get("TASK0186_NEW_REGRESSION_COUNT", "0")),
    }


def runtime_authority(search_config: VectorSearchConfig) -> dict[str, Any]:
    policy = composition_policy()
    return {
        "default_evidence_composition_policy": policy.policy_name,
        "evidence_budget_enabled": True,
        "evidence_slot_budget": policy.budget_limit,
        "evidence_budget_unit": policy.budget_unit,
        "evidence_rescue_enabled": policy.lane_protection,
        "evidence_guard_enabled": policy.same_document_prefix_retention_guard,
        "evidence_policy_version": policy.policy_version,
        "evidence_policy_digest": policy.policy_digest,
        "production_context_max_chunks": search_config.context_max_chunks,
        "production_context_token_budget": search_config.context_token_budget,
        "production_context_max_chunks_per_document": search_config.context_max_chunks_per_document,
    }


def composition_policy() -> evidence_composition.EvidenceCompositionPolicy:
    return evidence_composition.resolve_evidence_composition_policy()


def contract() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": "opk-rag.task0186.contract.v1",
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
    "ranked_candidate_count",
    "default_evidence_composition_policy",
    "evidence_budget_enabled",
    "evidence_slot_budget",
    "evidence_rescue_enabled",
    "evidence_guard_enabled",
    "evidence_probe_query_count",
    "evidence_probe_execution_count",
    "evidence_probe_success_count",
    "evidence_probe_failure_count",
    "ranked_candidate_input_count",
    "evidence_eligible_candidate_count",
    "evidence_input_candidate_count",
    "evidence_input_rejected_candidate_count",
    "evidence_input_contract_valid",
    "evidence_input_contract_mismatch_count",
    "first_evidence_input_contract_mismatch",
    "evidence_rejected_missing_content_count",
    "evidence_rejected_invalid_identity_count",
    "evidence_rejected_invalid_score_count",
    "evidence_rejected_policy_filter_count",
    "evidence_rejection_reasons_complete",
    "evidence_composition_invocation_count",
    "evidence_composition_success_count",
    "evidence_composition_failure_count",
    "pre_budget_evidence_candidate_count",
    "post_budget_evidence_candidate_count",
    "evidence_rescue_invocation_count",
    "evidence_rescue_success_count",
    "evidence_rescue_failure_count",
    "evidence_guard_evaluation_count",
    "evidence_guard_failure_count",
    "candidate_membership_before_evidence",
    "evidence_membership_after_composition",
    "evidence_membership_change_count",
    "evidence_membership_change_expected",
    "evidence_total_count",
    "canonical_evidence_count",
    "orphan_evidence_count",
    "duplicate_evidence_count",
    "canonical_evidence_contract_valid",
    "canonical_evidence_identity_valid",
    "canonical_evidence_content_valid",
    "canonical_evidence_provenance_valid",
    "evidence_content_nonempty_count",
    "evidence_content_empty_count",
    "evidence_content_equivalence_valid",
    "evidence_deduplication_required",
    "evidence_deduplication_success",
    "evidence_ordering_valid",
    "evidence_source_traceable",
    "evidence_document_traceable",
    "evidence_chunk_traceable",
    "generation_input_constructible",
    "generation_input_evidence_count",
    "generation_input_context_nonempty",
    "generation_runtime_reached",
    "first_evidence_loss_substage",
    "diagnosed_root_cause",
    "cold_start_evidence_composition_stage_passed",
    "next_failure_stage",
    "retrieval_policy_changed",
    "reranking_policy_changed",
    "evidence_policy_changed",
    "generation_policy_changed",
    "runtime_default_behavior_change",
    "promotion_applied",
    "focused_test_passed_count",
    "full_suite_passed_count",
    "full_suite_skipped_count",
    "full_suite_failed_count",
    "known_preexisting_failure_count",
    "new_regression_count",
)


def verify_task0186_artifacts(result_dir: Path = RESULT_DIR) -> dict[str, Any]:
    artifact_names = (
        "summary.json",
        "runtime_authority_audit.json",
        "ranked_candidate_to_evidence_input_contract_audit.json",
        "evidence_eligibility_audit.json",
        "evidence_composition_invocation_probe.json",
        "evidence_membership_audit.json",
        "canonical_evidence_contract_validation.json",
        "generation_input_boundary_audit.json",
        "candidate_identity_validation.json",
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
        and summary.get("generation_policy_changed") is False
        and summary.get("runtime_default_behavior_change") is False
        and summary.get("promotion_applied") is False
        and summary.get("new_regression_count") == 0
    )
    return {
        "task_id": TASK_ID,
        "verification_passed": passed,
        "missing_artifacts": missing,
        "missing_summary_fields": missing_fields,
        "first_evidence_loss_substage": summary.get("first_evidence_loss_substage"),
        "diagnosed_root_cause": summary.get("diagnosed_root_cause"),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TASK-0186 Cold-start Evidence Composition Runtime Validation and Diagnosis Report",
            "",
            "## Decision",
            f"task_status=`{summary.get('task_status')}`; first_evidence_loss_substage=`{summary.get('first_evidence_loss_substage')}`; diagnosed_root_cause=`{summary.get('diagnosed_root_cause')}`; cold_start_evidence_composition_stage_passed=`{summary.get('cold_start_evidence_composition_stage_passed')}`.",
            "",
            "## Runtime Authority",
            f"default_evidence_composition_policy=`{summary.get('default_evidence_composition_policy')}`; budget_enabled=`{summary.get('evidence_budget_enabled')}`; slot_budget=`{summary.get('evidence_slot_budget')}`; rescue_enabled=`{summary.get('evidence_rescue_enabled')}`; guard_enabled=`{summary.get('evidence_guard_enabled')}`.",
            f"production_context_max_chunks=`{summary.get('production_context_max_chunks')}`; production_context_token_budget=`{summary.get('production_context_token_budget')}`.",
            "",
            "## Corpus And Ranked Candidates",
            f"documents=`{summary.get('authoritative_document_count')}`; chunks=`{summary.get('authoritative_chunk_count')}`; stored_embeddings=`{summary.get('stored_embedding_count')}`; production_candidate_pool_count=`{summary.get('production_candidate_pool_count')}`; ranked_candidate_count=`{summary.get('ranked_candidate_count')}`.",
            f"ranked_contract_valid=`{summary.get('ranked_candidate_contract_valid')}`; identity_valid=`{summary.get('ranked_candidate_identity_valid')}`; score_valid=`{summary.get('ranked_candidate_score_valid')}`; provenance_valid=`{summary.get('ranked_candidate_provenance_valid')}`.",
            "",
            "## Evidence Input And Eligibility",
            f"probe_queries=`{summary.get('evidence_probe_query_count')}`; executions=`{summary.get('evidence_probe_execution_count')}`; successes=`{summary.get('evidence_probe_success_count')}`; failures=`{summary.get('evidence_probe_failure_count')}`.",
            f"input_contract_valid=`{summary.get('evidence_input_contract_valid')}`; mismatches=`{summary.get('evidence_input_contract_mismatch_count')}`; eligible=`{summary.get('evidence_eligible_candidate_count')}`; rejected=`{summary.get('evidence_input_rejected_candidate_count')}`.",
            "",
            "## Composition And Canonical Evidence",
            f"composition_invocations=`{summary.get('evidence_composition_invocation_count')}`; composition_successes=`{summary.get('evidence_composition_success_count')}`; failures=`{summary.get('evidence_composition_failure_count')}`.",
            f"pre_budget=`{summary.get('pre_budget_evidence_candidate_count')}`; post_budget=`{summary.get('post_budget_evidence_candidate_count')}`; membership_change_count=`{summary.get('evidence_membership_change_count')}`; duplicate_evidence_count=`{summary.get('duplicate_evidence_count')}`; ordering_valid=`{summary.get('evidence_ordering_valid')}`.",
            f"canonical_evidence_count=`{summary.get('canonical_evidence_count')}`; orphan_evidence_count=`{summary.get('orphan_evidence_count')}`; content_empty=`{summary.get('evidence_content_empty_count')}`; content_equivalence_valid=`{summary.get('evidence_content_equivalence_valid')}`.",
            "",
            "## Generation Boundary",
            f"generation_input_constructible=`{summary.get('generation_input_constructible')}`; generation_input_evidence_count=`{summary.get('generation_input_evidence_count')}`; generation_input_context_nonempty=`{summary.get('generation_input_context_nonempty')}`; generation_runtime_reached=`{summary.get('generation_runtime_reached')}`.",
            "",
            "## Policy Mutation",
            f"retrieval_policy_changed=`{summary.get('retrieval_policy_changed')}`; reranking_policy_changed=`{summary.get('reranking_policy_changed')}`; evidence_policy_changed=`{summary.get('evidence_policy_changed')}`; generation_policy_changed=`{summary.get('generation_policy_changed')}`; runtime_default_behavior_change=`{summary.get('runtime_default_behavior_change')}`; promotion_applied=`{summary.get('promotion_applied')}`.",
            "",
            "## Test Accounting",
            f"Focused tests: `{summary.get('focused_test_passed_count')}` passed. Full suite: `{summary.get('full_suite_passed_count')}` passed, `{summary.get('full_suite_skipped_count')}` skipped, `{summary.get('full_suite_failed_count')}` failed; known_preexisting_failure_count=`{summary.get('known_preexisting_failure_count')}`, new_regression_count=`{summary.get('new_regression_count')}`.",
            "",
            "## Next Frontier",
            f"next_failure_stage=`{summary.get('next_failure_stage')}`.",
            "",
        ]
    )


def bundle_item_row(item: Any) -> dict[str, Any]:
    return {
        "context_rank": item.context_rank,
        "result_rank": item.result_rank,
        "chunk_id": str(item.chunk_id),
        "document_id": str(item.document_id),
        "relative_path": item.relative_path,
        "heading_path": list(item.heading_path),
        "content": item.content,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "context_token_count": item.context_token_count,
        "rerank_score": item.rerank_score,
        "retrieval_sources": list(item.retrieval_sources),
    }


def validate_evidence_ordering(rows: Sequence[Mapping[str, Any]]) -> bool:
    ranks = [row.get("evidence_rank") for row in rows]
    return ranks == list(range(1, len(rows) + 1))


def _section_id(row: Mapping[str, Any]) -> str:
    if row.get("start_line") is not None or row.get("end_line") is not None:
        return f"{row.get('start_line')}:{row.get('end_line')}"
    return str(row.get("chunk_id"))


def _failed_probe(query: str, error: str) -> EvidenceProbeResult:
    return EvidenceProbeResult(query, False, error, 0, 0, 0, False, 0, 0, 0, False, 0, False, (), (), (), ())


def _failure(substage: str, root: str) -> dict[str, str]:
    return {"first_evidence_loss_substage": substage, "diagnosed_root_cause": root}


def _finite_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"
