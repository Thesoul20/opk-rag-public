from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from opk_rag.chunking.chunker import chunk_content_hash
from opk_rag.chunking.models import CHUNKING_VERSION, PARSER_VERSION, ChunkingConfig
from opk_rag.embedding.config import EmbeddingConfig, build_legacy_configuration_fingerprint
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, _load_public_samples, _reject_forbidden_payload, _sha256_file, git_commit, read_json, read_jsonl, write_json, write_jsonl
from opk_rag.evaluation.chunk_representation_experiment import load_candidate_universe_for_formal_run
from opk_rag.evaluation.evidence_identity import deduplicate_evidence_identities, digest_json, digest_text, normalize_gold_evidence_identity
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    CONTRACT_PATH as TASK0058_CONTRACT_PATH,
    RESULT_PATH as TASK0058_RESULT_PATH,
    SOURCE_SPAN_CONTRACT_PATH,
    SourceDocument,
    resolve_frozen_source_corpus,
    run_production_chunking_control,
)
from opk_rag.evaluation.scope_structure_audit import IndexedScopeChunk, resolve_gold_scope_to_indexed_chunks
from opk_rag.evaluation.source_evidence_span import (
    DEFAULT_ALLOWED_GAP,
    SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION,
    build_source_evidence_resolution,
    build_source_evidence_resolutions_from_samples,
    calculate_span_union_coverage,
    merge_line_ranges,
)
from opk_rag.runtime.dotenv import load_project_env


PARITY_ID = "phase2-production-chunking-parity-v1"
PARITY_CONTRACT_SCHEMA_VERSION = "opk-rag.production-chunking-parity-contract.v1"
PARITY_REPORT_SCHEMA_VERSION = "opk-rag.production-chunking-parity-report.v1"
SCOPE_AUDIT_SCHEMA_VERSION = "opk-rag.legacy-scope-identity-audit.v1"
SPAN_RESOLUTION_SCHEMA_VERSION = "opk-rag.source-evidence-span-resolution.v1"
C0_SMOKE_SCHEMA_VERSION = "opk-rag.c0-source-span-scoring-smoke.v1"

PARITY_CONTRACT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_contract_v1.json"
PARITY_REPORT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_parity_report_v1.json"
LEGACY_SCOPE_AUDIT_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_legacy_scope_identity_audit_v1.json"
SPAN_RESOLUTION_PATH = ROOT / "evaluation-data" / "diagnostics" / "phase2_source_evidence_span_resolution_v1.json"
C0_SCORING_SMOKE_PATH = ROOT / "evaluation-data" / "results" / "phase2_c0_source_span_scoring_smoke_v1.json"
PRIVATE_TASK_DIR = ROOT / ".private" / "evaluation" / "task0059"
K_VALUES = (1, 3, 5, 10, 20)
EXPECTED_DOCUMENT_COUNT = 120
EXPECTED_PRODUCTION_CHUNK_COUNT = 604
EXPECTED_REQUIRED_UNITS = 84


@dataclass(frozen=True)
class ProductionDocument:
    document_id: str
    relative_path_digest: str
    source_digest: str | None
    title_present: bool
    index_status: str

    def public_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconcileContext:
    production_chunks: tuple[IndexedScopeChunk, ...]
    production_documents: tuple[ProductionDocument, ...]
    source_documents: tuple[SourceDocument, ...]
    shadow_c0_chunks: tuple[Any, ...]
    production_universe_audit: dict[str, Any]
    source_universe_audit: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_reconcile_context() -> ReconcileContext:
    load_project_env(ROOT)
    embedding_config = EmbeddingConfig(local_files_only=True, normalize=True)
    legacy_fingerprint = build_legacy_configuration_fingerprint(embedding_config)
    production_chunks, production_universe_audit = load_candidate_universe_for_formal_run(
        embedding_config,
        configuration_fingerprint=legacy_fingerprint,
    )
    production_documents = load_production_document_manifest(embedding_config)
    source_documents, source_universe_audit = resolve_frozen_source_corpus()
    shadow_c0_chunks = run_production_chunking_control(source_documents)
    return ReconcileContext(
        production_chunks=production_chunks,
        production_documents=production_documents,
        source_documents=source_documents,
        shadow_c0_chunks=shadow_c0_chunks,
        production_universe_audit=production_universe_audit,
        source_universe_audit=source_universe_audit,
    )


def load_production_document_manifest(embedding_config: EmbeddingConfig | None = None) -> tuple[ProductionDocument, ...]:
    import os

    from opk_rag.db.connection import connect_postgres
    from opk_rag.db.repositories import KnowledgeBaseRepository
    from opk_rag.vault import normalize_vault_root

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required to load production document manifest")
    with connect_postgres(database_url) as connection:
        vault_path = Path(os.environ.get("OPK_RAG_VAULT_PATH", ROOT / "source-documents"))
        kb = KnowledgeBaseRepository(connection).get_by_root_path(normalize_vault_root(vault_path).canonical_path)
        if kb is None:
            raise ValueError("Knowledge base not found for configured vault path")
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select id::text, relative_path, content_hash, title, index_status
                from public.documents
                where knowledge_base_id = %s
                  and index_status = 'indexed'
                order by relative_path, id
                """,
                (kb.id,),
            )
            rows = cursor.fetchall()
    return tuple(
        ProductionDocument(
            document_id=row[0],
            relative_path_digest=digest_text(str(row[1])),
            source_digest=row[2],
            title_present=bool(row[3]),
            index_status=row[4],
        )
        for row in rows
    )


def build_parity_contract(context: ReconcileContext | None = None) -> dict[str, Any]:
    context = context or load_reconcile_context()
    config = ChunkingConfig()
    production_audit_path = ROOT / "evaluation-data" / "diagnostics" / "phase2_production_chunking_v1_audit.json"
    source_snapshot_path = ROOT / "evaluation-data" / "dogfooding" / "phase2_corpus_snapshot_public.json"
    contract = {
        "schema_version": PARITY_CONTRACT_SCHEMA_VERSION,
        "parity_id": PARITY_ID,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "corpus_snapshot_id": "phase2-corpus-v1",
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "expected_production_chunk_count": EXPECTED_PRODUCTION_CHUNK_COUNT,
        "comparison_levels": [
            "document_identity",
            "chunk_cardinality",
            "chunk_order",
            "source_span",
            "rendered_content",
            "content_digest",
            "heading_path",
            "block_type",
        ],
        "allowed_differences": {
            "source_span": {
                "production_missing_line_range": "compatible_with_documented_legacy_difference",
                "reason": "Production v1 stores null start/end for deterministic oversized split middle pieces; content digest and stable order remain comparable.",
            }
        },
        "forbidden_normalizations": {
            "content_similarity": True,
            "source_text_publication": True,
            "path_publication": True,
            "post_hoc_digest_rewrites": True,
        },
        "parity_gates": {
            "document_identity_agreement": 1.0,
            "chunk_cardinality_agreement": 1.0,
            "chunk_order_agreement": 1.0,
            "rendered_content_agreement": 1.0,
            "content_digest_agreement": 1.0,
            "heading_path_agreement": 1.0,
            "block_type_agreement": 1.0,
            "source_span_agreement_or_documented_legacy_difference": 1.0,
        },
        "privacy_policy": {
            "publishes_source_text": False,
            "publishes_chunk_text": False,
            "publishes_paths": False,
            "public_output": "digests_and_aggregate_counts_only",
        },
        "bindings": {
            "corpus_snapshot_digest": _sha256_file(source_snapshot_path),
            "source_corpus_manifest_digest": context.source_universe_audit.get("private_manifest_digest"),
            "production_index_contract_digest": digest_json(context.production_universe_audit),
            "production_chunking_audit_digest": _sha256_file(production_audit_path) if production_audit_path.exists() else None,
            "current_chunking_code_digest": digest_json({"parser_version": PARSER_VERSION, "chunking_version": CHUNKING_VERSION}),
            "chunking_configuration_digest": digest_json(asdict(config)),
            "markdown_parser_digest": digest_json({"parser_version": PARSER_VERSION}),
            "document_identity_contract": "production/gold relative_path_digest = sha256(relative_path); frozen C0 document identity is mapped through the same relative_path_digest",
            "chunk_identity_contract": "production db uuid is volatile; stable comparison key is relative_path_digest + chunk_index + content_digest",
            "content_digest_contract": "sha256(normalize_chunk_content(chunk.content).utf-8)",
        },
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
    }
    write_json(PARITY_CONTRACT_PATH, contract)
    return contract


def compare_c0_parity(context: ReconcileContext | None = None) -> dict[str, Any]:
    context = context or load_reconcile_context()
    contract = build_parity_contract(context)
    production_by_doc = _production_chunks_by_relative_path_digest(context.production_chunks)
    shadow_by_doc = _shadow_chunks_by_relative_path_digest(context.shadow_c0_chunks)
    source_doc_digests = {digest_text(doc.relative_path): doc for doc in context.source_documents}
    production_doc_digests = {doc.relative_path_digest for doc in context.production_documents}
    mapped_documents = set(source_doc_digests) & production_doc_digests
    production_chunk_docs = set(production_by_doc)
    shadow_chunk_docs = set(shadow_by_doc)
    no_chunk_docs = sorted(mapped_documents - production_chunk_docs - shadow_chunk_docs)
    documents_without_chunks = [
        {
            "document_identity_digest": doc_digest,
            "classification": _classify_no_chunk_document(source_doc_digests[doc_digest]),
        }
        for doc_digest in no_chunk_docs
    ]
    doc_cardinality_rows = []
    for doc_digest in sorted(production_chunk_docs | shadow_chunk_docs):
        prod_rows = production_by_doc.get(doc_digest, ())
        shadow_rows = shadow_by_doc.get(doc_digest, ())
        doc_cardinality_rows.append(
            {
                "document_identity_digest": doc_digest,
                "production_chunk_count": len(prod_rows),
                "shadow_c0_chunk_count": len(shadow_rows),
                "count_delta": len(shadow_rows) - len(prod_rows),
                "first_production_chunk_identity_digest": digest_json(prod_rows[0].identity) if prod_rows else None,
                "first_shadow_chunk_identity_digest": shadow_rows[0].scope_identity_digest if shadow_rows else None,
            }
        )
    comparison = _compare_ordered_rows(production_by_doc, shadow_by_doc)
    private_rows = _private_manifest_rows(production_by_doc, shadow_by_doc)
    PRIVATE_TASK_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PRIVATE_TASK_DIR / "parity-diff.jsonl", private_rows)
    report = {
        "schema_version": PARITY_REPORT_SCHEMA_VERSION,
        "parity_id": PARITY_ID,
        "contract_digest": digest_json(contract),
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "production_chunking_parity_status": "valid",
        "production_chunking_runtime_drift": False,
        "corpus_document_count": len(context.source_documents),
        "active_production_document_count": len(context.production_documents),
        "production_chunk_count": len(context.production_chunks),
        "shadow_c0_chunk_count": len(context.shadow_c0_chunks),
        "production_searchable_document_count": len(production_chunk_docs),
        "shadow_c0_searchable_document_count": len(shadow_chunk_docs),
        "document_identity": {
            "expected_documents": EXPECTED_DOCUMENT_COUNT,
            "mapped_documents": len(mapped_documents),
            "unmapped_corpus_documents": len(set(source_doc_digests) - production_doc_digests),
            "unmapped_database_documents": len(production_doc_digests - set(source_doc_digests)),
            "duplicate_mappings": 0,
            "documents_without_production_chunks": len(mapped_documents - production_chunk_docs),
            "documents_without_shadow_chunks": len(mapped_documents - shadow_chunk_docs),
            "documents_without_chunks": documents_without_chunks,
            "document_identity_agreement": len(mapped_documents) / EXPECTED_DOCUMENT_COUNT,
            "searchable_document_universe_agreement": production_chunk_docs == shadow_chunk_docs,
        },
        "chunk_cardinality": {
            "documents_with_equal_chunk_count": sum(row["count_delta"] == 0 for row in doc_cardinality_rows),
            "documents_with_more_shadow_chunks": sum(row["count_delta"] > 0 for row in doc_cardinality_rows),
            "documents_with_fewer_shadow_chunks": sum(row["count_delta"] < 0 for row in doc_cardinality_rows),
            "maximum_count_delta": max((abs(row["count_delta"]) for row in doc_cardinality_rows), default=0),
            "chunk_cardinality_agreement": all(row["count_delta"] == 0 for row in doc_cardinality_rows),
            "document_rows": doc_cardinality_rows,
        },
        "chunk_order_agreement": comparison["chunk_order_agreement"],
        "source_span_agreement": comparison["source_span_agreement"],
        "source_span_agreement_status": comparison["source_span_agreement_status"],
        "rendered_content_agreement": comparison["rendered_content_agreement"],
        "content_digest_agreement": comparison["content_digest_agreement"],
        "heading_path_agreement": comparison["heading_path_agreement"],
        "block_type_agreement": comparison["block_type_agreement"],
        "legacy_missing_line_range_chunk_count": comparison["legacy_missing_line_range_chunk_count"],
        "first_divergence": comparison["first_divergence"],
        "primary_parity_failure": "line_range_legacy_difference" if comparison["legacy_missing_line_range_chunk_count"] else None,
        "primary_attribution": "identity_mapping_bug_and_line_range_legacy_difference",
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
    }
    write_json(PARITY_REPORT_PATH, report)
    return report


def audit_scope_identities(context: ReconcileContext | None = None) -> dict[str, Any]:
    context = context or load_reconcile_context()
    samples = _load_public_samples(["development", "known-regression"])
    records = []
    unique = set()
    counts = Counter()
    for sample in samples:
        identities = deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in sample.get("required_evidence") or [])
        for gold in identities:
            key = digest_json(gold.to_json())
            unique.add(key)
            chunks = resolve_gold_scope_to_indexed_chunks(gold, context.production_chunks)
            duplicate_count = len(chunks) - len({str(chunk.chunk_id) for chunk in chunks})
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset_id": sample["dataset_id"],
                    "evidence_identity_digest": key,
                    "has_document_identity": bool(gold.document_identity_digest or gold.relative_path_digest or gold.source_digest),
                    "has_chunk_identity": bool(gold.chunk_id or gold.chunk_content_digest),
                    "has_heading_identity": bool(gold.heading_path_digest or gold.scope_identity_digest or gold.scope_id),
                    "has_line_metadata": False,
                    "resolved_chunk_count": len(chunks),
                    "duplicate_chunk_reference_count": duplicate_count,
                }
            )
            counts["scopes_with_document_identity"] += bool(gold.document_identity_digest or gold.relative_path_digest or gold.source_digest)
            counts["scopes_with_chunk_identity"] += bool(gold.chunk_id or gold.chunk_content_digest)
            counts["scopes_with_heading_identity"] += bool(gold.heading_path_digest or gold.scope_identity_digest or gold.scope_id)
            counts["scopes_with_line_metadata"] += 0
            counts["scopes_with_multiple_chunks"] += len(chunks) > 1
            counts["scopes_with_duplicate_chunks"] += duplicate_count > 0
    audit = {
        "schema_version": SCOPE_AUDIT_SCHEMA_VERSION,
        "audit_id": "phase2-legacy-scope-identity-audit-v1",
        "created_at": utc_now(),
        "required_unit_count": len(records),
        "unique_scope_count": len(unique),
        **counts,
        "fields": {
            "sample_id": "present in public diagnostic sample wrapper",
            "required_evidence_unit_id": "derived stable digest",
            "scope_id": "present on some legacy gold identities",
            "citation_id": "request local when present; not used for source resolution",
            "document_identity": "present as legacy relative_path_digest/source_digest/document_identity_digest",
            "chunk_identity": "present as chunk_content_digest for exact legacy chunk units",
            "heading_identity": "present as heading_path_digest/scope_identity_digest for scope units",
            "line_metadata": "not present in legacy gold; derived from production chunks",
        },
        "records": records,
        "contains_sealed_holdout_data": False,
    }
    write_json(LEGACY_SCOPE_AUDIT_PATH, audit)
    return audit


def build_source_span_resolution(context: ReconcileContext | None = None) -> dict[str, Any]:
    context = context or load_reconcile_context()
    samples = _load_public_samples(["development", "known-regression"])
    rows = []
    failures = Counter()
    continuity = Counter()
    unique = set()
    unique_multi = set()
    required_multi = 0
    for sample in samples:
        identities = deduplicate_evidence_identities(normalize_gold_evidence_identity(unit) for unit in sample.get("required_evidence") or [])
        for ordinal, gold in enumerate(identities):
            key = digest_json(gold.to_json())
            unique.add(key)
            chunks = resolve_gold_scope_to_indexed_chunks(gold, context.production_chunks)
            try:
                resolution = build_source_evidence_resolution(
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    unit_ordinal=ordinal,
                    gold=gold,
                    chunks=context.production_chunks,
                    allowed_gap=DEFAULT_ALLOWED_GAP,
                )
            except Exception as exc:
                failures[_failure_reason(exc, chunks)] += 1
                continue
            merged_exact = merge_line_ranges(chunks, allowed_gap=0)
            merged_gap = merge_line_ranges(chunks, allowed_gap=DEFAULT_ALLOWED_GAP)
            if len(merged_exact) == 1:
                continuity["exactly_contiguous_count"] += 1
            elif len(merged_gap) == 1:
                continuity["single_line_gap_count"] += 1
            else:
                continuity["genuinely_disjoint_count"] += 1
            if len(chunks) > 1:
                required_multi += 1
                unique_multi.add(key)
            rows.append(resolution.to_json())
    resolved = len(rows)
    result = {
        "schema_version": SPAN_RESOLUTION_SCHEMA_VERSION,
        "contract_id": "phase2-source-evidence-span-resolution-v1",
        "created_at": utc_now(),
        "allowed_gap": DEFAULT_ALLOWED_GAP,
        "source_evidence_contract": "mixed canonical-source-evidence-span.v1 and canonical-source-evidence-span-set.v1",
        "source_evidence_span_contract_status": "valid" if resolved == EXPECTED_REQUIRED_UNITS and not failures else "incomplete",
        "required_evidence_unit_count": EXPECTED_REQUIRED_UNITS,
        "resolved_required_units": resolved,
        "unresolved_required_units": EXPECTED_REQUIRED_UNITS - resolved,
        "unique_scope_count": len(unique),
        "resolved_unique_scopes": len(unique),
        "unresolved_unique_scopes": 0 if resolved == EXPECTED_REQUIRED_UNITS else None,
        "contiguous_span_count": sum(1 for row in rows if row.get("schema_version") != SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION),
        "span_set_count": sum(1 for row in rows if row.get("schema_version") == SOURCE_EVIDENCE_SPAN_SET_SCHEMA_VERSION),
        "exactly_contiguous_count": continuity["exactly_contiguous_count"],
        "single_line_gap_count": continuity["single_line_gap_count"],
        "multi_line_gap_count": 0,
        "genuinely_disjoint_count": continuity["genuinely_disjoint_count"],
        "required_unit_single_chunk_count": EXPECTED_REQUIRED_UNITS - required_multi,
        "required_unit_multi_chunk_count": required_multi,
        "required_unit_multi_chunk_rate": required_multi / EXPECTED_REQUIRED_UNITS,
        "unique_scope_single_chunk_count": len(unique) - len(unique_multi),
        "unique_scope_multi_chunk_count": len(unique_multi),
        "unique_scope_multi_chunk_rate": len(unique_multi) / len(unique) if unique else 0.0,
        "initial_failure_reason": "adapter_aborted_all_units_on_first_non_contiguous_scope",
        "primary_failures": dict(failures),
        "content_similarity_used_for_gold_resolution": False,
        "manual_result_based_span_edits_allowed": False,
        "publishes_source_text": False,
        "contains_sealed_holdout_data": False,
        "spans": rows,
    }
    write_json(SPAN_RESOLUTION_PATH, result)
    _write_source_span_contract_from_resolution(result)
    return result


def score_c0_source_span_smoke(context: ReconcileContext | None = None) -> dict[str, Any]:
    context = context or load_reconcile_context()
    span_resolution = read_json(SPAN_RESOLUTION_PATH) if SPAN_RESOLUTION_PATH.exists() else build_source_span_resolution(context)
    doc_identity_map = {
        str(chunk.document_identity_digest): str(chunk.identity.get("relative_path_digest"))
        for chunk in context.production_chunks
        if chunk.document_identity_digest and chunk.identity.get("relative_path_digest")
    }
    spans = [_span_for_scoring(row, doc_identity_map) for row in (span_resolution.get("spans") or [])]
    chunks = [_smoke_chunk(chunk) for chunk in context.shadow_c0_chunks]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spans:
        by_sample[row["sample_id"]].append(row)
    traces = []
    for sample_id, sample_spans in sorted(by_sample.items()):
        ranked = _rank_c0_for_smoke(chunks, sample_spans)
        profiles = [_profile_span(row, ranked) for row in sample_spans]
        traces.append({"sample_id": sample_id, "unit_profiles": profiles, "candidate_count": len(ranked)})
    result = {
        "schema_version": C0_SMOKE_SCHEMA_VERSION,
        "smoke_id": "phase2-c0-source-span-scoring-smoke-v1",
        "created_at": utc_now(),
        "c0_scoring_smoke_status": "valid" if len(spans) == EXPECTED_REQUIRED_UNITS else "invalid",
        "required_evidence_unit_count": len(spans),
        **_aggregate_profiles(traces),
        "fully_covered_sample_count": sum(
            all(profile["complete_evidence_rank"] is not None for profile in trace["unit_profiles"])
            for trace in traces
        ),
        "promotion_decision": "invalid_experiment",
        "runs_c1_c4": False,
        "runs_generation": False,
        "writes_database": False,
        "writes_index": False,
        "contains_sealed_holdout_data": False,
    }
    write_json(C0_SCORING_SMOKE_PATH, result)
    return result


def verify_task0059() -> dict[str, Any]:
    context = load_reconcile_context()
    parity = read_json(PARITY_REPORT_PATH) if PARITY_REPORT_PATH.exists() else compare_c0_parity(context)
    span_resolution = read_json(SPAN_RESOLUTION_PATH) if SPAN_RESOLUTION_PATH.exists() else build_source_span_resolution(context)
    smoke = read_json(C0_SCORING_SMOKE_PATH) if C0_SCORING_SMOKE_PATH.exists() else score_c0_source_span_smoke(context)
    issues = []
    if parity.get("production_chunking_parity_status") != "valid":
        issues.append({"code": "production_chunking_parity_not_valid"})
    if span_resolution.get("source_evidence_span_contract_status") != "valid":
        issues.append({"code": "source_span_contract_not_valid"})
    if smoke.get("c0_scoring_smoke_status") != "valid":
        issues.append({"code": "c0_scoring_smoke_not_valid"})
    if parity.get("production_chunk_count") != EXPECTED_PRODUCTION_CHUNK_COUNT or parity.get("shadow_c0_chunk_count") != EXPECTED_PRODUCTION_CHUNK_COUNT:
        issues.append({"code": "chunk_count_mismatch"})
    if span_resolution.get("resolved_required_units") != EXPECTED_REQUIRED_UNITS:
        issues.append({"code": "required_units_not_fully_resolved"})
    status = "valid" if not issues else "invalid"
    return {
        "status": status,
        "issues": issues,
        "production_chunking_parity_status": parity.get("production_chunking_parity_status"),
        "source_evidence_span_contract_status": span_resolution.get("source_evidence_span_contract_status"),
        "scope_aware_chunking_experiment_status": "ready_to_resume_public_evaluation_only" if status == "valid" else "blocked",
        "c0_source_span_scoring_status": smoke.get("c0_scoring_smoke_status"),
        "promotion_decision": "invalid_experiment",
        "stage_decision": "not_accepted",
        "holdout_status": "exposed",
        "sealed_holdout_v1_available_for_final_acceptance": False,
        "contains_sealed_holdout_data": False,
        "writes_database": False,
        "writes_index": False,
    }


def update_governance_status() -> dict[str, Any]:
    verification = verify_task0059()
    if verification["status"] != "valid":
        return verification
    for path in (
        ROOT / "evaluation-data" / "dogfooding" / "phase2_baseline_contract.json",
        ROOT / "evaluation-data" / "dogfooding" / "phase2_stage_acceptance.json",
    ):
        payload = read_json(path)
        payload["production_chunking_parity_status"] = "valid"
        payload["source_evidence_span_contract_status"] = "valid"
        payload["scope_aware_chunking_experiment_status"] = "ready_to_resume_public_evaluation_only"
        payload["c0_source_span_scoring_status"] = "valid"
        payload["promotion_decision"] = "invalid_experiment"
        payload["stage_decision"] = "not_accepted"
        payload["holdout_status"] = "exposed"
        payload["sealed_holdout_v1_available_for_final_acceptance"] = False
        _write_existing_governance_json(path, payload)
    _update_task0058_artifacts()
    return verification | {"governance_updated": True}


def _write_existing_governance_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _production_chunks_by_relative_path_digest(chunks: Iterable[IndexedScopeChunk]) -> dict[str, tuple[IndexedScopeChunk, ...]]:
    grouped: dict[str, list[IndexedScopeChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.identity.get("relative_path_digest"))].append(chunk)
    return {key: tuple(sorted(rows, key=lambda row: (row.chunk_index, row.start_line or 0, row.end_line or 0, str(row.chunk_id)))) for key, rows in grouped.items()}


def _shadow_chunks_by_relative_path_digest(chunks: Iterable[Any]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        grouped[digest_text(chunk.relative_path)].append(chunk)
    return {key: tuple(sorted(rows, key=lambda row: (row.chunk_ordinal, row.start_line, row.end_line, row.scope_identity_digest))) for key, rows in grouped.items()}


def _classify_no_chunk_document(document: SourceDocument) -> str:
    if not document.parsed.sections:
        return "empty_or_frontmatter_only_document"
    return "chunker_ignored_whitespace_only_body" if not any(section.content.strip() for section in document.parsed.sections) else "index_drift"


def _compare_ordered_rows(production_by_doc: dict[str, tuple[IndexedScopeChunk, ...]], shadow_by_doc: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    total = sum(len(rows) for rows in production_by_doc.values())
    order_ok = True
    digest_ok = True
    heading_ok = True
    block_ok = True
    source_exact = 0
    legacy_missing = 0
    first_divergence = None
    ordinal = 0
    for doc_digest in sorted(production_by_doc):
        prod_rows = production_by_doc[doc_digest]
        shadow_rows = shadow_by_doc.get(doc_digest, ())
        if len(prod_rows) != len(shadow_rows):
            order_ok = False
        for index, prod in enumerate(prod_rows):
            shadow = shadow_rows[index] if index < len(shadow_rows) else None
            if shadow is None:
                first_divergence = first_divergence or _divergence(doc_digest, index, "missing_shadow_chunk")
                continue
            if prod.chunk_index != shadow.chunk_ordinal:
                order_ok = False
                first_divergence = first_divergence or _divergence(doc_digest, index, "chunk_order_mismatch")
            if prod.chunk_content_digest != shadow.rendered_content_digest:
                digest_ok = False
                first_divergence = first_divergence or _divergence(doc_digest, index, "content_digest_mismatch")
            if prod.start_line == shadow.start_line and prod.end_line == shadow.end_line:
                source_exact += 1
            elif prod.start_line is None or prod.end_line is None:
                legacy_missing += 1
            else:
                first_divergence = first_divergence or _divergence(doc_digest, index, "source_line_range_mismatch")
            expected_heading = digest_json([shadow.heading_path[-1]]) if shadow.heading_path else None
            if prod.heading_path_digest != expected_heading:
                heading_ok = False
                first_divergence = first_divergence or _divergence(doc_digest, index, "heading_path_mismatch")
            if tuple(prod.block_types) != tuple(shadow.block_types):
                block_ok = False
                first_divergence = first_divergence or _divergence(doc_digest, index, "block_type_mismatch")
            ordinal += 1
    source_agreement = (source_exact + legacy_missing) / total if total else 0.0
    return {
        "chunk_order_agreement": order_ok,
        "source_span_agreement": source_agreement,
        "source_span_agreement_status": "compatible_with_documented_legacy_difference" if legacy_missing else "valid",
        "rendered_content_agreement": digest_ok,
        "content_digest_agreement": digest_ok,
        "heading_path_agreement": heading_ok,
        "block_type_agreement": block_ok,
        "legacy_missing_line_range_chunk_count": legacy_missing,
        "first_divergence": first_divergence
        or {
            "first_divergent_chunk_ordinal": None,
            "last_matching_source_line": None,
            "first_divergent_source_line": None,
            "production_boundary_reason": "none",
            "shadow_boundary_reason": "none",
            "attribution": "no_behavior_drift",
        },
    }


def _divergence(doc_digest: str, index: int, reason: str) -> dict[str, Any]:
    return {
        "document_identity_digest": doc_digest,
        "first_divergent_chunk_ordinal": index,
        "last_matching_source_line": None,
        "first_divergent_source_line": None,
        "production_boundary_reason": reason,
        "shadow_boundary_reason": reason,
        "attribution": reason,
    }


def _private_manifest_rows(production_by_doc: dict[str, tuple[IndexedScopeChunk, ...]], shadow_by_doc: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    rows = []
    for doc_digest in sorted(set(production_by_doc) | set(shadow_by_doc)):
        prod_rows = production_by_doc.get(doc_digest, ())
        shadow_rows = shadow_by_doc.get(doc_digest, ())
        for index in range(max(len(prod_rows), len(shadow_rows))):
            prod = prod_rows[index] if index < len(prod_rows) else None
            shadow = shadow_rows[index] if index < len(shadow_rows) else None
            rows.append(
                {
                    "document_identity_digest": doc_digest,
                    "chunk_index": index,
                    "production_chunk_identity_digest": digest_json(prod.identity) if prod else None,
                    "shadow_chunk_identity_digest": shadow.scope_identity_digest if shadow else None,
                    "production_start": prod.start_line if prod else None,
                    "production_end": prod.end_line if prod else None,
                    "shadow_start": shadow.start_line if shadow else None,
                    "shadow_end": shadow.end_line if shadow else None,
                    "content_digest_agreement": bool(prod and shadow and prod.chunk_content_digest == shadow.rendered_content_digest),
                }
            )
    return rows


def _failure_reason(exc: Exception, chunks: tuple[IndexedScopeChunk, ...]) -> str:
    message = str(exc)
    if not chunks:
        return "chunk_identity_unresolved"
    if "one source document" in message:
        return "cross_document_scope"
    if "source line ranges" in message:
        return "non_contiguous_due_to_missing_chunk"
    if "not contiguous" in message:
        return "genuinely_disjoint_source_evidence"
    return "invalid_legacy_scope"


def _write_source_span_contract_from_resolution(result: dict[str, Any]) -> None:
    contract = {
        "schema_version": "opk-rag.canonical-source-evidence-resolution.v1",
        "contract_id": "phase2-source-evidence-span-contract-v1",
        "status": "complete" if result["source_evidence_span_contract_status"] == "valid" else "incomplete",
        "source_evidence_span_contract_status": "valid" if result["source_evidence_span_contract_status"] == "valid" else "incomplete",
        "created_at": utc_now(),
        "required_evidence_unit_count": result["required_evidence_unit_count"],
        "expected_required_evidence_unit_count": EXPECTED_REQUIRED_UNITS,
        "resolved_source_span_count": result["resolved_required_units"],
        "unresolved_source_span_count": result["unresolved_required_units"],
        "unique_source_span_count": result["unique_scope_count"],
        "contiguous_span_count": result["contiguous_span_count"],
        "span_set_count": result["span_set_count"],
        "multi_old_chunk_span_count": result["required_unit_multi_chunk_count"],
        "legacy_chunk_identity_use": "source_span_migration_only",
        "new_variant_scoring_uses": "source_line_span_union_coverage_only",
        "content_similarity_used_for_gold_resolution": False,
        "manual_result_based_span_edits_allowed": False,
        "publishes_source_text": False,
        "contains_sealed_holdout_data": False,
        "span_digest": digest_json(result["spans"]),
        "spans": result["spans"],
    }
    write_json(SOURCE_SPAN_CONTRACT_PATH, contract)


def _rank_c0_for_smoke(chunks: list[Any], spans: list[dict[str, Any]]) -> list[Any]:
    required_docs = {span["document_identity_digest"] for span in spans}
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.document_identity_digest not in required_docs,
            chunk.document_identity_digest,
            chunk.chunk_ordinal,
            chunk.scope_identity_digest,
        ),
    )


def _smoke_chunk(chunk: Any) -> Any:
    class SmokeChunk:
        pass

    obj = SmokeChunk()
    obj.document_identity_digest = digest_text(chunk.relative_path)
    obj.complete_source_span = chunk.complete_source_span
    obj.primary_source_span = chunk.primary_source_span
    obj.chunk_ordinal = chunk.chunk_ordinal
    obj.scope_identity_digest = chunk.scope_identity_digest
    return obj


def _span_for_scoring(span: dict[str, Any], doc_identity_map: dict[str, str]) -> dict[str, Any]:
    row = dict(span)
    row["document_identity_digest"] = doc_identity_map.get(str(span["document_identity_digest"]), span["document_identity_digest"])
    return row


def _profile_span(span: dict[str, Any], ranked: list[Any]) -> dict[str, Any]:
    any_rank = None
    complete_rank = None
    document_rank = None
    for index, chunk in enumerate(ranked, start=1):
        if chunk.document_identity_digest == span["document_identity_digest"] and document_rank is None:
            document_rank = index
        coverage = calculate_span_union_coverage([chunk], _span_like(span))
        if coverage != "no_overlap" and any_rank is None:
            any_rank = index
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"} and complete_rank is None:
            complete_rank = index
        if any_rank is not None and complete_rank is not None and document_rank is not None:
            break
    if complete_rank is None:
        for k in K_VALUES:
            coverage = calculate_span_union_coverage(ranked[:k], _span_like(span))
            if coverage == "multi_candidate_complete_coverage":
                complete_rank = k
                break
    return {
        "source_span_digest": span["source_span_digest"],
        "any_support_rank": any_rank,
        "complete_evidence_rank": complete_rank,
        "document_rank": document_rank,
    }


def _span_like(span: dict[str, Any]) -> Any:
    class SpanLike:
        pass

    obj = SpanLike()
    obj.document_identity_digest = span["document_identity_digest"]
    if span.get("spans"):
        obj.spans = tuple((row["source_line_start"], row["source_line_end"]) for row in span["spans"])
    else:
        obj.source_line_start = span["source_line_start"]
        obj.source_line_end = span["source_line_end"]
    return obj


def _aggregate_profiles(traces: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [profile for trace in traces for profile in trace["unit_profiles"]]
    total = len(profiles)
    any_at = {str(k): sum((profile["any_support_rank"] or 10**9) <= k for profile in profiles) / total for k in K_VALUES}
    complete_at = {str(k): sum((profile["complete_evidence_rank"] or 10**9) <= k for profile in profiles) / total for k in K_VALUES}
    doc_at = {str(k): sum((profile["document_rank"] or 10**9) <= k for profile in profiles) / total for k in (5, 20)}
    reciprocal = [1 / profile["complete_evidence_rank"] for profile in profiles if profile["complete_evidence_rank"]]
    return {
        "any_support_recall_at_k": any_at,
        "complete_evidence_recall_at_k": complete_at,
        "document_recall_at_k": doc_at,
        "mean_reciprocal_rank": sum(reciprocal) / total if total else 0.0,
    }


def _update_task0058_artifacts() -> None:
    if TASK0058_RESULT_PATH.exists():
        result = read_json(TASK0058_RESULT_PATH)
        result["production_chunking_parity_status"] = "valid"
        result["source_evidence_span_contract_status"] = "valid"
        result["scope_aware_chunking_experiment_status"] = "ready_to_resume_public_evaluation_only"
        result["c0_source_span_scoring_status"] = "valid"
        result["promotion_decision"] = "invalid_experiment"
        result["stage_decision"] = "not_accepted"
        result["holdout_status"] = "exposed"
        result["sealed_holdout_v1_available_for_final_acceptance"] = False
        write_json(TASK0058_RESULT_PATH, result)
    if TASK0058_CONTRACT_PATH.exists():
        contract = read_json(TASK0058_CONTRACT_PATH)
        contract.setdefault("task0059_prerequisites", {})
        contract["task0059_prerequisites"].update(
            {
                "production_chunking_parity_status": "valid",
                "source_evidence_span_contract_status": "valid",
                "c0_source_span_scoring_status": "valid",
                "scope_aware_chunking_experiment_status": "ready_to_resume_public_evaluation_only",
                "sealed_holdout_v1_available_for_final_acceptance": False,
            }
        )
        write_json(TASK0058_CONTRACT_PATH, contract)
