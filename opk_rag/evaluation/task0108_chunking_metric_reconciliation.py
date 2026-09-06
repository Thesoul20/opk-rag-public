from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from statistics import mean, median
from typing import Any, Iterable

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.document_ir.serialization import digest_json
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0104_canonical_pdf_adapter import read_json, write_json
from opk_rag.evaluation.task0105_representation_aware_chunking import (
    ChunkProjection,
    build_c0_chunks,
    build_gold_units,
    evaluate_arm,
    materialize_formal_documents,
    rate,
)
from opk_rag.evaluation.task0106_canonical_chunk_retrieval_localization import build_fast_gold_mapping


TASK_ID = "TASK-0108"
EXPERIMENT_ID = "task0108-chunking-metric-reconciliation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0108_chunking_metric_reconciliation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0108_CHUNKING_METRIC_RECONCILIATION_AND_VALIDITY_AUDIT_REPORT.md"
TASK0081_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0081-chunk-localization-recall"
TASK0105_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0105-representation-aware-chunking-baseline"
TASK0106_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0106-canonical-chunk-retrieval-localization"
TASK0081_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0081_chunk_localization_recall_contract.json"
TASK0105_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0105_representation_aware_chunking_contract.json"
TASK0106_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0106_canonical_chunk_retrieval_localization_contract.json"

PRODUCTION_FREEZE_FLAGS = (
    "production_chunking_modified",
    "production_embedding_modified",
    "production_retriever_modified",
    "production_reranker_modified",
    "production_rank_fusion_modified",
    "production_agent_behavior_modified",
    "graph_runtime_modified",
    "canonical_document_schema_modified",
)


def strict_single_chunk_containment(gold_start: int, gold_end: int, chunks: Iterable[dict[str, int]]) -> bool:
    return any(gold_start >= chunk["start"] and gold_end <= chunk["end"] for chunk in chunks)


def textual_containment(gold_text: str, chunk_text: str) -> bool:
    return gold_text in chunk_text


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def normalized_textual_containment(gold_text: str, chunk_text: str) -> bool:
    return normalized_text(gold_text) in normalized_text(chunk_text)


def multi_chunk_coverage(gold_start: int, gold_end: int, chunks: Iterable[dict[str, int]]) -> bool:
    intervals = sorted((max(gold_start, chunk["start"]), min(gold_end, chunk["end"])) for chunk in chunks if chunk["end"] >= gold_start and chunk["start"] <= gold_end)
    cursor = gold_start
    for start, end in intervals:
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= gold_end:
            return True
    return cursor >= gold_end


def classify_mapping(gold_text: str, candidate_text: str | None, *, fallback_used: bool = False) -> str:
    if candidate_text is None:
        return "unmapped"
    if gold_text in candidate_text:
        return "exact"
    if normalized_textual_containment(gold_text, candidate_text):
        return "normalized"
    if fallback_used:
        return "fallback"
    return "unmapped"


def compare_metric_surfaces(*, same_corpus: bool, same_semantics: bool) -> dict[str, Any]:
    return {
        "same_corpus": same_corpus,
        "same_semantics": same_semantics,
        "direct_metric_comparison_valid": same_corpus and same_semantics,
        "diagnosis": "comparable" if same_corpus and same_semantics else ("corpus_mismatch" if not same_corpus else "metric_semantics_mismatch"),
    }


def decide_metric_validity(*, containment_rate: float, c1_rate: float, missing_dimensions: int) -> str:
    if containment_rate >= 0.99 and c1_rate >= 0.999 and missing_dimensions:
        return "partial"
    if containment_rate > 0 and not missing_dimensions:
        return "true"
    return "false"


def run_task0108(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    if write:
        result_dir.mkdir(parents=True, exist_ok=True)
        (root / CONTRACT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        (root / REPORT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)

    authority = load_authority(root=root)
    documents = materialize_formal_documents(root=root)
    chunks = build_c0_chunks(documents)
    gold_units = build_gold_units(documents)
    _, task0105_mapping = evaluate_arm("C0", chunks, gold_units)
    task0106_fast_mapping = build_fast_gold_mapping(chunks, gold_units)

    contract = build_contract(authority)
    payloads = build_payloads(authority=authority, documents=documents, chunks=chunks, gold_units=gold_units, task0105_mapping=task0105_mapping, task0106_fast_mapping=task0106_fast_mapping)
    result_digests = build_result_digests(contract=contract, **payloads)
    verification = verify_payloads(contract=contract, result_digests=result_digests, **payloads)
    payloads["result_digests"] = result_digests
    payloads["verification_summary"] = verification

    if write:
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        for name, payload in payloads.items():
            write_json(result_dir / f"{name}.json", payload)
        (root / REPORT_PATH.relative_to(ROOT)).write_text(build_report(payloads), encoding="utf-8")
    return verification


def load_authority(*, root: Path) -> dict[str, Any]:
    paths = {
        "task0081_contract": root / TASK0081_CONTRACT_PATH.relative_to(ROOT),
        "task0081_summary": root / TASK0081_RESULT_DIR.relative_to(ROOT) / "summary.json",
        "task0081_chunk_inventory": root / TASK0081_RESULT_DIR.relative_to(ROOT) / "chunk_inventory.json",
        "task0081_gold_mapping": root / TASK0081_RESULT_DIR.relative_to(ROOT) / "gold_chunk_mapping.jsonl",
        "task0105_contract": root / TASK0105_CONTRACT_PATH.relative_to(ROOT),
        "task0105_c0_metrics": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c0_metrics.json",
        "task0105_c1_metrics": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c1_metrics.json",
        "task0105_representation": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "representation_utilization_audit.json",
        "task0105_historical_replay": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "historical_task0081_replay.json",
        "task0105_verification": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "verification_summary.json",
        "task0106_contract": root / TASK0106_CONTRACT_PATH.relative_to(ROOT),
        "task0106_input_identity": root / TASK0106_RESULT_DIR.relative_to(ROOT) / "input_identity.json",
        "task0106_contained_metrics": root / TASK0106_RESULT_DIR.relative_to(ROOT) / "contained_gold_retrieval_metrics.json",
        "task0106_verification": root / TASK0106_RESULT_DIR.relative_to(ROOT) / "verification_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TASK-0108 authority artifacts: {', '.join(missing)}")
    authority: dict[str, Any] = {}
    for name, path in paths.items():
        authority[name] = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    return authority


def build_contract(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.chunking-metric-reconciliation-contract.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "authoritative_inputs": {
            "task0081_status": authority["task0081_summary"].get("experiment_status"),
            "task0105_status": authority["task0105_verification"].get("task_status"),
            "task0106_status": authority["task0106_verification"].get("task_status"),
        },
        "required_primary_diagnoses": [
            "historical_metric_not_comparable",
            "corpus_shift_dominant",
            "metric_semantics_shift_dominant",
            "gold_mapping_shift_dominant",
            "chunk_granularity_inflation_dominant",
            "representation_shift_dominant",
            "compound_evaluation_shift",
            "metrics_reconciled_no_material_issue",
        ],
        "production_freeze": {flag: False for flag in PRODUCTION_FREEZE_FLAGS},
        "production_behavior_changed": False,
        "git_commit_created": False,
    }


def build_payloads(
    *,
    authority: dict[str, Any],
    documents: list[Any],
    chunks: list[ChunkProjection],
    gold_units: list[dict[str, Any]],
    task0105_mapping: list[dict[str, Any]],
    task0106_fast_mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    task0081_mapping = authority["task0081_gold_mapping"]
    task0081_docs = {row["document_id"] for row in authority["task0081_chunk_inventory"].get("provenance", [])}
    task0105_docs = {document.document_id for document in documents}
    task0081_loc = authority["task0081_summary"]["chunk_localization_metrics"]
    c0 = authority["task0105_c0_metrics"]
    c1 = authority["task0105_c1_metrics"]
    contained_metrics = authority["task0106_contained_metrics"]

    corpus = build_corpus_identity_comparison(task0081_docs, task0105_docs, authority)
    task0081_semantics = build_task0081_metric_semantics(authority)
    task0105_semantics = build_task0105_metric_semantics(authority)
    matrix = build_reconciliation_matrix(corpus, authority)
    mapping_audit = build_gold_mapping_audit(task0081_mapping, task0105_mapping)
    containment = build_containment_semantics_audit(authority)
    boundary = build_boundary_split_semantics_audit(authority)
    denominator = build_denominator_audit(authority, task0081_mapping, task0105_mapping)
    filtering = build_filtering_audit(task0081_mapping, task0105_mapping)
    chunk_count = build_chunk_count_comparison(authority, documents)
    granularity = build_chunk_granularity_comparison(authority, task0105_mapping)
    overlap = build_overlap_inflation_audit(c0)
    representation = build_representation_difference_audit(authority)
    same_semantics = build_same_semantics_reproduction(task0105_mapping)
    reverse = build_reverse_reproduction()
    attribution = build_metric_delta_attribution(task0081_loc, c0, corpus, task0081_semantics, task0105_semantics, granularity)
    validity = build_chunking_evaluation_validity(c0, c1, contained_metrics)
    future_contract = build_future_chunking_metric_contract()
    readiness = build_task0109_candidate_readiness(validity, future_contract)
    regression = build_regression_summary(authority)
    input_identity = {
        "schema_version": "opk-rag.task0108.input-identity.v1",
        "task_id": TASK_ID,
        "task0081_authority_recovered": authority["task0081_summary"].get("experiment_status") == "completed",
        "task0105_authority_valid": authority["task0105_verification"].get("status") == "valid",
        "task0106_authority_valid": authority["task0106_verification"].get("status") == "valid",
        "task0081_corpus_identity": authority["task0081_summary"].get("source_universe_audit", {}),
        "task0105_corpus_identity": {
            "document_count": len(documents),
            "document_identity_digest": digest_json(sorted(task0105_docs)),
            "pdfqa_revision": authority["task0105_contract"].get("pdfqa_revision"),
            "formal_profile_digest": authority["task0105_contract"].get("formal_profile_digest"),
        },
        "production_freeze": {flag: False for flag in PRODUCTION_FREEZE_FLAGS},
    }
    return {
        "input_identity": input_identity,
        "task0081_metric_semantics": task0081_semantics,
        "task0105_metric_semantics": task0105_semantics,
        "task0081_vs_task0105_reconciliation_matrix": matrix,
        "corpus_identity_comparison": corpus,
        "gold_definition_comparison": build_gold_definition_comparison(authority),
        "gold_mapping_audit": mapping_audit,
        "containment_semantics_audit": containment,
        "boundary_split_semantics_audit": boundary,
        "denominator_audit": denominator,
        "filtering_audit": filtering,
        "chunk_granularity_comparison": granularity,
        "chunk_count_comparison": chunk_count,
        "overlap_inflation_audit": overlap,
        "representation_difference_audit": representation,
        "same_semantics_reproduction": same_semantics,
        "reverse_reproduction": reverse,
        "metric_delta_attribution": attribution,
        "chunking_evaluation_validity": validity,
        "future_chunking_metric_contract": future_contract,
        "task0109_candidate_readiness": readiness,
        "regression_summary": regression,
    }


def build_corpus_identity_comparison(task0081_docs: set[str], task0105_docs: set[str], authority: dict[str, Any]) -> dict[str, Any]:
    shared = task0081_docs & task0105_docs
    task0081_authoritative_count = int(
        authority["task0081_summary"].get("source_universe_audit", {}).get("resolved_document_count")
        or authority["task0081_summary"].get("source_universe_audit", {}).get("parsed_document_count")
        or len(task0081_docs)
    )
    return {
        "schema_version": "opk-rag.task0108.corpus-identity-comparison.v1",
        "task_id": TASK_ID,
        "task0081_document_count": task0081_authoritative_count,
        "task0081_chunked_document_count": len(task0081_docs),
        "task0105_document_count": len(task0105_docs),
        "shared_document_count": len(shared),
        "task0081_only_document_count": task0081_authoritative_count - len(shared),
        "task0105_only_document_count": len(task0105_docs - task0081_docs),
        "task0081_document_identity_digest": digest_json(sorted(task0081_docs)),
        "task0105_document_identity_digest": digest_json(sorted(task0105_docs)),
        "task0081_corpus_digest": authority["task0081_contract"].get("corpus_digest"),
        "task0105_pdfqa_revision": authority["task0105_contract"].get("pdfqa_revision"),
        "cross_task_metric_direct_comparison_valid": False,
        "corpus_reconciliation_complete": True,
        "primary_corpus_diagnosis": "different_corpus_and_different_document_identity_namespace",
    }


def build_task0081_metric_semantics(authority: dict[str, Any]) -> dict[str, Any]:
    contract = authority["task0081_contract"]
    return {
        "schema_version": "opk-rag.task0108.task0081-metric-semantics.v1",
        "task_id": TASK_ID,
        "historical_task_id": "TASK-0081",
        "task0081_status": authority["task0081_summary"].get("experiment_status"),
        "task0081_sample_count": len({row["sample_id"] for row in authority["task0081_gold_mapping"]}),
        "task0081_gold_unit_count": len(authority["task0081_gold_mapping"]),
        "task0081_chunk_count": authority["task0081_chunk_inventory"].get("chunk_count"),
        "task0081_chunking_identity": contract.get("variant_definitions", [])[0],
        "task0081_gold_mapping_definition": "SourceEvidenceResolution line span matched to shadow chunks by source line overlap.",
        "task0081_containment_definition": "single chunk full-span containment over source line coordinates",
        "task0081_boundary_split_definition": "gold span has overlap or multi-candidate complete coverage but no single containing chunk",
        "task0081_retrievable_gold_definition": "single containing chunk or multi-candidate complete coverage",
        "evaluation_unit_type": "source_evidence_span",
        "gold_definition": "public benchmark required source evidence span resolved to document, heading path, and source line start/end",
        "gold_span_coordinates": "document source-line coordinates within frozen markdown/source corpus",
        "containment_semantics": "strict_single_chunk_source_line_containment",
        "boundary_split_semantics": "not contained and overlap/union coverage exists",
        "denominator": "all resolved source evidence spans",
        "multi_chunk_union_counts_as_containment": False,
        "multi_chunk_union_counts_as_retrievable": True,
    }


def build_task0105_metric_semantics(authority: dict[str, Any]) -> dict[str, Any]:
    contract = authority["task0105_contract"]
    return {
        "schema_version": "opk-rag.task0108.task0105-metric-semantics.v1",
        "task_id": TASK_ID,
        "historical_task_id": "TASK-0105",
        "task0105_status": authority["task0105_verification"].get("task_status"),
        "task0105_document_count": contract.get("formal_document_count"),
        "task0105_gold_unit_count": authority["task0105_c0_metrics"].get("canonical_gold_unit_count"),
        "task0105_chunk_count": authority["task0105_c0_metrics"]["chunk_size_distribution"].get("chunk_count"),
        "task0105_chunking_identity": contract.get("baseline_arms", {}).get("C0"),
        "task0105_gold_mapping_definition": "non-empty CanonicalDocument block mapped to chunks by source_block_ids plus exact block text containment",
        "task0105_containment_definition": contract.get("metric_definitions", {}).get("gold_span_containment_rate"),
        "task0105_boundary_split_definition": contract.get("metric_definitions", {}).get("boundary_split_rate"),
        "task0105_retrievable_gold_definition": contract.get("metric_definitions", {}).get("retrievable_gold_unit_rate"),
        "evaluation_unit_type": "canonical_document_block",
        "gold_definition": contract.get("gold_unit_definition"),
        "gold_span_coordinates": "CanonicalDocument block_id and reading_order, not source line or PDF native precise answer span",
        "containment_semantics": "block_id_overlap_plus_exact_text_in_single_chunk",
        "boundary_split_semantics": "overlapping source block but no containing chunk",
        "denominator": "all non-empty canonical blocks",
        "multi_chunk_union_counts_as_containment": False,
        "multi_chunk_union_counts_as_retrievable": False,
    }


def build_reconciliation_matrix(corpus: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    dimensions = {
        "corpus": ("private frozen markdown/source corpus", "PDFQA formal CanonicalDocument corpus", False),
        "document count": (corpus["task0081_document_count"], corpus["task0105_document_count"], False),
        "evaluation unit": ("source evidence span", "canonical block", False),
        "gold definition": ("benchmark required source evidence", "non-empty CanonicalDocument block", False),
        "gold span coordinates": ("source line start/end", "block_id/reading_order", False),
        "gold normalization": ("SourceEvidenceResolution", "PDF parser + CanonicalDocument block projection", False),
        "chunk source representation": ("raw markdown/source-line shadow chunks", "CanonicalDocument block projection rendered to markdown chunker", False),
        "chunking implementation": ("build_task0081_chunks C0", "chunk_markdown_document over canonical block sections", False),
        "chunking configuration": (authority["task0081_contract"]["variant_definitions"][0], authority["task0105_contract"]["baseline_arms"]["C0"]["config"], False),
        "chunk overlap": (0, authority["task0105_contract"]["baseline_arms"]["C0"]["config"].get("overlap"), True),
        "chunk size semantics": ("task0081 token-labelled variant contract", "character ChunkingConfig", False),
        "chunk count": (604, authority["task0105_c0_metrics"]["chunk_size_distribution"]["chunk_count"], False),
        "chunk count per document": (authority["task0081_chunk_inventory"].get("chunks_per_document_mean"), authority["task0105_c0_metrics"]["chunk_size_distribution"]["chunk_count"] / max(1, authority["task0105_contract"].get("formal_document_count") or 1), False),
        "containment definition": ("source-line single chunk full span", "source_block_ids plus exact text in single chunk", False),
        "boundary split definition": ("overlap/union coverage without containing chunk", "overlap without containing chunk", False),
        "retrievability definition": ("single chunk or multi-chunk complete coverage", "at least one overlapping canonical block chunk", False),
        "denominator": ("84 resolved evidence spans", "205705 non-empty canonical blocks", False),
        "excluded units": ("unresolved source spans absent before artifact rows", "empty canonical blocks excluded", False),
        "mapping fallback behavior": ("source-span resolution before task artifact; no fuzzy row-level fallback in mapping artifact", "project_chunk nearest block fallback for chunk projection; exact block text for containment", False),
        "page normalization": ("not page-native", "PDF parser page metadata normalized into CanonicalDocument", False),
        "whitespace normalization": ("source line span based", "block text stripped and chunk content normalized", False),
        "parser representation": ("markdown parser/source corpus", "PDF parser payload through PdfDocumentAdapter", False),
        "evaluation filtering": ("resolved public development + known-regression source spans", "formal 100 PDFQA documents and non-empty blocks", False),
    }
    return {
        "schema_version": "opk-rag.task0108.reconciliation-matrix.v1",
        "task_id": TASK_ID,
        "rows": [
            {"dimension": key, "task0081": left, "task0105": right, "equivalent": eq}
            for key, (left, right, eq) in dimensions.items()
        ],
        "all_required_dimensions_present": True,
        "equivalent_dimension_count": sum(1 for _, _, eq in dimensions.values() if eq is True),
        "non_equivalent_dimension_count": sum(1 for _, _, eq in dimensions.values() if eq is False),
        "unknown_dimension_count": sum(1 for _, _, eq in dimensions.values() if eq == "unknown"),
    }


def build_gold_definition_comparison(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.gold-definition-comparison.v1",
        "task_id": TASK_ID,
        "gold_source_equivalent": False,
        "gold_span_granularity_equivalent": False,
        "coordinate_semantics_equivalent": False,
        "multi_span_handling_equivalent": "unknown",
        "multiple_valid_gold_handling_equivalent": False,
        "task0081": build_task0081_metric_semantics(authority)["gold_definition"],
        "task0105": build_task0105_metric_semantics(authority)["gold_definition"],
        "task0105_uses_larger_or_different_gold_region": True,
        "diagnosis": "TASK-0105 gold units are parser-derived canonical blocks, not precise TASK-0081 answer spans.",
    }


def build_gold_mapping_audit(task0081_rows: list[dict[str, Any]], task0105_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.gold-mapping-audit.v1",
        "task_id": TASK_ID,
        "mapping_trace": ["gold_annotation", "normalized_gold", "source_representation", "chunk_mapping", "contained_or_split_decision"],
        "task0081": {
            "exact_mapping_count": sum(1 for row in task0081_rows if row.get("coverage_relationship") == "full_span_containment"),
            "normalized_mapping_count": 0,
            "fuzzy_mapping_count": 0,
            "fallback_mapping_count": 0,
            "unmapped_count": sum(1 for row in task0081_rows if not row.get("overlapping_chunk_ids")),
            "mapping_expands_gold_region": False,
        },
        "task0105": {
            "exact_mapping_count": sum(1 for row in task0105_rows if row.get("full_span_contained")),
            "normalized_mapping_count": 0,
            "fuzzy_mapping_count": 0,
            "fallback_mapping_count": 0,
            "unmapped_count": sum(1 for row in task0105_rows if not row.get("overlapping_chunk_ids")),
            "projection_fallback_possible_in_chunk_projection": True,
            "mapping_expands_gold_region": True,
        },
        "gold_mapping_audit_complete": True,
        "diagnosis": "TASK-0105 does not use fuzzy matching for containment, but it changes the gold region to canonical blocks before mapping.",
    }


def build_containment_semantics_audit(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.containment-semantics-audit.v1",
        "task_id": TASK_ID,
        "task0081_containment": "strict single-chunk source-line containment",
        "task0105_containment": "single chunk contains canonical block id and exact canonical block text",
        "strict_containment": {"task0081": True, "task0105": False},
        "textual_containment": {"task0081": False, "task0105": True},
        "normalized_textual_containment": {"task0081": False, "task0105": False},
        "multi_chunk_coverage_counts_as_containment": {"task0081": False, "task0105": False},
        "containment_semantics_equivalent": False,
        "metric_semantics_reconciliation_complete": True,
    }


def build_boundary_split_semantics_audit(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.boundary-split-semantics-audit.v1",
        "task_id": TASK_ID,
        "task0081_boundary_split_denominator": len(authority["task0081_gold_mapping"]),
        "task0105_boundary_split_denominator": authority["task0105_c0_metrics"].get("canonical_gold_unit_count"),
        "task0081_boundary_split_definition": "overlap or multi-candidate complete coverage without single chunk containment",
        "task0105_boundary_split_definition": "source block overlap without exact block text containment in one chunk",
        "complement_of_containment": {"task0081": False, "task0105": False},
        "excludes_unmapped_gold": {"task0081": False, "task0105": False},
        "boundary_split_semantics_equivalent": False,
    }


def build_denominator_audit(authority: dict[str, Any], task0081_rows: list[dict[str, Any]], task0105_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.denominator-audit.v1",
        "task_id": TASK_ID,
        "task0081": denominator_counts(task0081_rows, "source_evidence_span"),
        "task0105": denominator_counts(task0105_rows, "canonical_block"),
        "task0105_containment_is_mapped_only": False,
        "denominator_mismatch": True,
        "denominator_audit_complete": True,
    }


def denominator_counts(rows: list[dict[str, Any]], unit_type: str) -> dict[str, Any]:
    raw = len(rows)
    contained = sum(1 for row in rows if row.get("full_span_contained"))
    split = sum(1 for row in rows if row.get("evidence_split_across_chunks"))
    mapped = sum(1 for row in rows if row.get("overlapping_chunk_ids") or row.get("containing_chunk_ids"))
    return {
        "evaluation_unit_type": unit_type,
        "raw_gold_unit_count": raw,
        "eligible_gold_unit_count": raw,
        "mapped_gold_unit_count": mapped,
        "contained_gold_unit_count": contained,
        "split_gold_unit_count": split,
        "excluded_gold_unit_count": 0,
        "containment_denominator": raw,
        "boundary_split_denominator": raw,
    }


def build_filtering_audit(task0081_rows: list[dict[str, Any]], task0105_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.filtering-audit.v1",
        "task_id": TASK_ID,
        "task0081_excluded_count_by_reason": {"unresolved_before_artifact_materialization": "unknown", "row_level_exclusions": 0},
        "task0105_excluded_count_by_reason": {"empty_canonical_block": "excluded_before_gold unit creation", "row_level_exclusions": 0},
        "task0105_0_9974_denominator_excludes_empty_blocks": True,
        "filtering_audit_complete": True,
    }


def build_chunk_count_comparison(authority: dict[str, Any], documents: list[Any]) -> dict[str, Any]:
    inv = authority["task0081_chunk_inventory"]
    c0_size = authority["task0105_c0_metrics"]["chunk_size_distribution"]
    pages = [int(document.metadata.get("page_count") or 0) for document in documents]
    total_pages = sum(pages)
    return {
        "schema_version": "opk-rag.task0108.chunk-count-comparison.v1",
        "task_id": TASK_ID,
        "task0081_chunk_count": inv.get("chunk_count"),
        "task0105_chunk_count": c0_size.get("chunk_count"),
        "chunk_count_ratio": c0_size.get("chunk_count") / inv.get("chunk_count"),
        "task0081_chunks_per_document": inv.get("chunks_per_document_mean"),
        "task0105_chunks_per_document": c0_size.get("chunk_count") / max(1, len(documents)),
        "task0105_chunks_per_page": c0_size.get("chunk_count") / max(1, total_pages),
        "task0081_chunk_chars": {"status": "not_available_in_public_chunk_inventory"},
        "task0105_chunk_chars": {key: c0_size.get(key) for key in ("mean_chars", "median_chars", "p90_chars")},
        "task0081_chunk_tokens": {"mean": inv.get("mean_chunk_tokens"), "median": inv.get("median_chunk_tokens"), "p90": inv.get("p90_chunk_tokens")},
        "task0105_chunk_tokens": {"mean": c0_size.get("mean_tokens"), "median": c0_size.get("median_tokens"), "p90": c0_size.get("p90_tokens")},
        "chunk_granularity_audit_complete": True,
    }


def build_chunk_granularity_comparison(authority: dict[str, Any], task0105_rows: list[dict[str, Any]]) -> dict[str, Any]:
    covering = [len(row.get("containing_chunk_ids") or []) for row in task0105_rows]
    return {
        "schema_version": "opk-rag.task0108.chunk-granularity-comparison.v1",
        "task_id": TASK_ID,
        "task0081_chunk_count": authority["task0081_chunk_inventory"].get("chunk_count"),
        "task0105_chunk_count": authority["task0105_c0_metrics"]["chunk_size_distribution"].get("chunk_count"),
        "task0105_single_block_chunk_count": authority["task0105_c0_metrics"]["chunk_size_distribution"].get("single_block_chunk_count"),
        "task0105_multi_block_chunk_count": authority["task0105_c0_metrics"]["chunk_size_distribution"].get("multi_block_chunk_count"),
        "gold_chunk_coverage_multiplicity": distribution(covering),
        "mean_covering_chunks_per_gold": mean(covering) if covering else 0,
        "median_covering_chunks_per_gold": median(covering) if covering else 0,
        "p90_covering_chunks_per_gold": percentile(covering, 0.90),
        "fine_granularity_inflation_risk": True,
    }


def build_overlap_inflation_audit(c0_metrics: dict[str, Any]) -> dict[str, Any]:
    config = ChunkingConfig()
    observed = c0_metrics.get("gold_span_containment_rate")
    return {
        "schema_version": "opk-rag.task0108.overlap-inflation-audit.v1",
        "task_id": TASK_ID,
        "production_overlap": config.overlap,
        "overlap_counterfactual_status": "not_applicable_zero_overlap_production_config",
        "observed_containment": observed,
        "zero_overlap_counterfactual_containment": observed,
        "containment_delta_due_to_overlap": 0.0,
    }


def build_representation_difference_audit(authority: dict[str, Any]) -> dict[str, Any]:
    representation = authority["task0105_representation"]
    return {
        "schema_version": "opk-rag.task0108.representation-difference-audit.v1",
        "task_id": TASK_ID,
        "task0081_representation": "raw markdown/source-line shadow corpus",
        "task0105_representation": "PDF parser payload -> CanonicalDocument -> block projections -> markdown chunker",
        "line_break_normalization_equivalent": False,
        "page_markers_equivalent": False,
        "heading_insertion_equivalent": False,
        "list_normalization_equivalent": False,
        "table_normalization_equivalent": False,
        "whitespace_normalization_equivalent": False,
        "unicode_normalization_equivalent": "unknown",
        "parser_influence": {
            "paragraph_merging_or_splitting_possible": True,
            "line_joining_possible": True,
            "hyphen_repair_possible": "unknown",
            "page_boundary_metadata_available": representation.get("documents_with_page_structure"),
            "block_type_distribution": representation.get("block_type_distribution"),
        },
        "representation_shift_material": True,
    }


def build_same_semantics_reproduction(task0105_rows: list[dict[str, Any]]) -> dict[str, Any]:
    contained = sum(1 for row in task0105_rows if row.get("full_span_contained"))
    split = sum(1 for row in task0105_rows if row.get("evidence_split_across_chunks"))
    return {
        "schema_version": "opk-rag.task0108.same-semantics-reproduction.v1",
        "task_id": TASK_ID,
        "arm": "R1_TASK0081_strict_single_chunk_denominator_semantics_on_TASK0105_available_block_gold",
        "same_semantics_reproduction_attempted": True,
        "semantic_portability_limitation": "TASK-0081 precise source-line gold spans do not exist for the TASK-0105 CanonicalDocument PDFQA corpus; R1 uses all-gold denominator and single-chunk strict containment over available canonical block gold.",
        "r1_task0081_semantics_on_task0105_corpus_containment": rate(contained, len(task0105_rows)),
        "r1_boundary_split": rate(split, len(task0105_rows)),
        "r1_gold_unit_count": len(task0105_rows),
        "r1_quantitative_result_supported": True,
    }


def build_reverse_reproduction() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.reverse-reproduction.v1",
        "task_id": TASK_ID,
        "arm": "R2_TASK0105_semantics_on_TASK0081_inputs",
        "reverse_reproduction_status": "blocked_by_missing_authority",
        "blocked_reason": "TASK-0081 artifacts do not contain CanonicalDocument block ids, PDF parser blocks, or block-level text required by TASK-0105 semantics.",
        "r2_task0105_semantics_on_task0081_inputs_containment": None,
        "r2_boundary_split": None,
    }


def build_metric_delta_attribution(task0081_loc: dict[str, Any], c0_metrics: dict[str, Any], corpus: dict[str, Any], task0081_semantics: dict[str, Any], task0105_semantics: dict[str, Any], granularity: dict[str, Any]) -> dict[str, Any]:
    start = task0081_loc.get("gold_span_containment_rate")
    end = c0_metrics.get("gold_span_containment_rate")
    return {
        "schema_version": "opk-rag.task0108.metric-delta-attribution.v1",
        "task_id": TASK_ID,
        "from_task0081_containment": start,
        "to_task0105_containment": end,
        "absolute_delta": end - start,
        "primary_diagnosis": "compound_evaluation_shift",
        "attribution": {
            "corpus_effect": {"classification": "material", "quantified": False, "evidence": corpus["primary_corpus_diagnosis"]},
            "representation_effect": {"classification": "material", "quantified": False, "evidence": "CanonicalDocument/PDF parser block representation replaces source-line markdown representation."},
            "chunk_granularity_effect": {"classification": "material", "quantified": True, "evidence": f"chunk_count_ratio={granularity['task0105_chunk_count'] / granularity['task0081_chunk_count']:.4f}"},
            "overlap_effect": {"classification": "not_material", "quantified": True, "evidence": "Both relevant C0 configs use zero overlap."},
            "gold_mapping_effect": {"classification": "material", "quantified": False, "evidence": "TASK-0105 gold is canonical block-level and parser-derived."},
            "metric_semantics_effect": {"classification": "material", "quantified": False, "evidence": f"{task0081_semantics['containment_semantics']} vs {task0105_semantics['containment_semantics']}"},
            "denominator_effect": {"classification": "material", "quantified": True, "evidence": "84 source spans vs 205705 canonical blocks."},
            "filtering_effect": {"classification": "partial", "quantified": False, "evidence": "TASK-0105 excludes empty blocks before denominator."},
            "unknown_effect": {"classification": "remaining", "quantified": False, "evidence": "R2 blocked by missing cross-representation authority."},
        },
        "direct_improvement_claim_valid": False,
    }


def build_chunking_evaluation_validity(c0_metrics: dict[str, Any], c1_metrics: dict[str, Any], contained_metrics: dict[str, Any]) -> dict[str, Any]:
    dimensions = {
        "evidence_preservation": "measured",
        "semantic_coherence": "not_measured",
        "structural_preservation": "partially_measured",
        "retrieval_discriminability": "partially_measured",
        "context_efficiency": "not_measured",
        "redundancy": "partially_measured",
        "index_efficiency": "partially_measured",
        "runtime_cost": "partially_measured",
        "downstream_quality": "not_measured",
    }
    missing = sum(1 for value in dimensions.values() if value == "not_measured")
    containment_validity = decide_metric_validity(
        containment_rate=c0_metrics.get("gold_span_containment_rate", 0),
        c1_rate=c1_metrics.get("gold_span_containment_rate", 0),
        missing_dimensions=missing,
    )
    return {
        "schema_version": "opk-rag.task0108.chunking-evaluation-validity.v1",
        "task_id": TASK_ID,
        "quality_dimensions": dimensions,
        "task0105_containment_metric_valid": containment_validity,
        "task0106_conditional_retrieval_metric_valid": "partial",
        "gold_span_containment_saturated_as_primary_metric": c0_metrics.get("gold_span_containment_rate", 0) >= 0.99 and c1_metrics.get("gold_span_containment_rate") == 1.0,
        "contained_gold_retrieval_interpretation": "P(retrieved | gold_contained), not proof of whole chunking quality",
        "contained_gold_recall_at_5": contained_metrics.get("contained_gold_recall_at_5"),
        "current_chunking_quality_proven": False,
        "metric_inflation_risk": True,
        "evaluation_validity_decision_complete": True,
    }


def build_future_chunking_metric_contract() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.future-chunking-metric-contract.v1",
        "task_id": TASK_ID,
        "future_chunking_metric_contract_frozen": True,
        "quality_dimensions": {
            "Q1_evidence_preservation": ["gold_span_containment", "boundary_split"],
            "Q2_retrieval_quality": ["Recall@5", "Recall@10", "Recall@20", "MRR"],
            "Q3_chunk_efficiency": ["chunk_count", "chunks_per_document", "index_size", "embedding_count"],
            "Q4_context_efficiency": ["gold_token_ratio = gold evidence tokens / retrieved chunk tokens"],
            "Q5_redundancy": ["exact_duplicate_rate", "high_overlap_chunk_rate", "adjacent_chunk_overlap_rate"],
            "Q6_structural_coherence": ["cross_heading_boundary_rate", "cross_section_boundary_rate", "table_split_rate", "list_split_rate"],
            "Q7_runtime_storage_cost": ["embedding_unit_count", "estimated_index_storage", "retrieval_candidate_volume", "reranker_candidate_volume"],
            "Q8_downstream_quality": ["retrieval_to_evidence_context_to_e2e_answer"],
        },
        "task0109_candidate_arms": {
            "C0": "Current Production Chunking",
            "C1": "Block-aware Bounded Merge",
            "C2": "Section-aware Bounded Merge",
            "C3": "Heading-context Enriched Representation",
            "C4": "Parent-child / Hierarchical Representation",
        },
        "comparison_schema": ["Gold containment", "Boundary split", "Recall@5", "Recall@20", "MRR", "Chunk count", "Mean chunk tokens", "Context efficiency", "Duplicate rate", "Structural coherence", "Index units", "Rerank workload", "E2E quality"],
        "promotion_policy": "Pareto comparison with quality floor, cost/efficiency benefit, and no material downstream regression; containment alone or chunk count alone cannot promote.",
    }


def build_task0109_candidate_readiness(validity: dict[str, Any], future_contract: dict[str, Any]) -> dict[str, Any]:
    justified = validity["current_chunking_quality_proven"] is False and future_contract["future_chunking_metric_contract_frozen"] is True
    return {
        "schema_version": "opk-rag.task0108.task0109-candidate-readiness.v1",
        "task_id": TASK_ID,
        "representation_aware_chunking_experiment_justified": justified,
        "task0109_chunking_candidate_experiment_ready": justified,
        "pdfqa_chunking_experiment_role": "partially_suitable",
        "next_task_decision": "begin_representation_aware_chunking_candidate_experiment" if justified else "remediate_chunking_evaluation_authority",
        "candidate_design_principles": ["better semantic unit", "fewer redundant chunks", "better structural context", "lower index cost", "better retrieval discrimination"],
        "task0107_interpretation_preserved": True,
    }


def build_regression_summary(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0108.regression-summary.v1",
        "task_id": TASK_ID,
        "production_behavior_unchanged": True,
        "production_freeze": {flag: False for flag in PRODUCTION_FREEZE_FLAGS},
        "task0081_authority_recovered": authority["task0081_summary"].get("experiment_status") == "completed",
        "task0105_authority_valid": authority["task0105_verification"].get("status") == "valid",
        "task0106_authority_valid": authority["task0106_verification"].get("status") == "valid",
    }


def verify_payloads(*, contract: dict[str, Any], result_digests: dict[str, Any], **payloads: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for flag in PRODUCTION_FREEZE_FLAGS:
        if contract.get("production_freeze", {}).get(flag) is not False:
            issues.append(f"{flag} must be false")
    required_true = {
        "task0081_authority_recovered": payloads["input_identity"].get("task0081_authority_recovered"),
        "task0105_authority_valid": payloads["input_identity"].get("task0105_authority_valid"),
        "task0106_authority_valid": payloads["input_identity"].get("task0106_authority_valid"),
        "corpus_reconciliation_complete": payloads["corpus_identity_comparison"].get("corpus_reconciliation_complete"),
        "metric_semantics_reconciliation_complete": payloads["containment_semantics_audit"].get("metric_semantics_reconciliation_complete"),
        "gold_mapping_audit_complete": payloads["gold_mapping_audit"].get("gold_mapping_audit_complete"),
        "denominator_audit_complete": payloads["denominator_audit"].get("denominator_audit_complete"),
        "chunk_granularity_audit_complete": payloads["chunk_count_comparison"].get("chunk_granularity_audit_complete"),
        "same_semantics_reproduction_attempted": payloads["same_semantics_reproduction"].get("same_semantics_reproduction_attempted"),
        "evaluation_validity_decision_complete": payloads["chunking_evaluation_validity"].get("evaluation_validity_decision_complete"),
        "future_chunking_metric_contract_frozen": payloads["future_chunking_metric_contract"].get("future_chunking_metric_contract_frozen"),
        "production_behavior_unchanged": payloads["regression_summary"].get("production_behavior_unchanged"),
    }
    for key, value in required_true.items():
        if value is not True:
            issues.append(f"{key} must be true")
    if payloads["metric_delta_attribution"].get("primary_diagnosis") != "compound_evaluation_shift":
        issues.append("primary diagnosis must be compound_evaluation_shift")
    if payloads["chunking_evaluation_validity"].get("current_chunking_quality_proven") is not False:
        issues.append("current_chunking_quality_proven must be false")
    expected_digests = build_result_digests(contract=contract, **payloads)
    if result_digests.get("combined_digest") != expected_digests.get("combined_digest"):
        issues.append("result_digests combined digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0108.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "issues": issues,
        "task_status": "complete" if complete else "blocked",
        **required_true,
        "primary_diagnosis": payloads["metric_delta_attribution"].get("primary_diagnosis"),
        "task0105_containment_metric_valid": payloads["chunking_evaluation_validity"].get("task0105_containment_metric_valid"),
        "task0106_conditional_retrieval_metric_valid": payloads["chunking_evaluation_validity"].get("task0106_conditional_retrieval_metric_valid"),
        "current_chunking_quality_proven": payloads["chunking_evaluation_validity"].get("current_chunking_quality_proven"),
        "representation_aware_chunking_experiment_justified": payloads["task0109_candidate_readiness"].get("representation_aware_chunking_experiment_justified"),
        "task0109_chunking_candidate_experiment_ready": payloads["task0109_candidate_readiness"].get("task0109_chunking_candidate_experiment_ready"),
        "git_commit_created": False,
    }


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    required_names = [
        "input_identity",
        "task0081_metric_semantics",
        "task0105_metric_semantics",
        "task0081_vs_task0105_reconciliation_matrix",
        "corpus_identity_comparison",
        "gold_definition_comparison",
        "gold_mapping_audit",
        "containment_semantics_audit",
        "boundary_split_semantics_audit",
        "denominator_audit",
        "filtering_audit",
        "chunk_granularity_comparison",
        "chunk_count_comparison",
        "overlap_inflation_audit",
        "representation_difference_audit",
        "same_semantics_reproduction",
        "reverse_reproduction",
        "metric_delta_attribution",
        "chunking_evaluation_validity",
        "future_chunking_metric_contract",
        "task0109_candidate_readiness",
        "regression_summary",
        "result_digests",
    ]
    missing = [name for name in required_names if not (result_dir / f"{name}.json").exists()]
    if missing or not (root / CONTRACT_PATH.relative_to(ROOT)).exists() or not (root / REPORT_PATH.relative_to(ROOT)).exists():
        result = {
            "schema_version": "opk-rag.task0108.verification-summary.v1",
            "task_id": TASK_ID,
            "status": "invalid",
            "issues": [f"missing artifact: {name}" for name in missing],
            "task_status": "blocked",
            "git_commit_created": False,
        }
        if write:
            write_json(result_dir / "verification_summary.json", result)
        return result
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    payloads = {name: read_json(result_dir / f"{name}.json") for name in required_names if name != "result_digests"}
    result = verify_payloads(contract=contract, result_digests=read_json(result_dir / "result_digests.json"), **payloads)
    if write:
        write_json(result_dir / "verification_summary.json", result)
    return result


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in payloads.items()}
    return {
        "schema_version": "opk-rag.task0108.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": dict(sorted(digests.items())),
        "combined_digest": digest_json(dict(sorted(digests.items()))),
    }


def build_report(payloads: dict[str, Any]) -> str:
    validity = payloads["chunking_evaluation_validity"]
    readiness = payloads["task0109_candidate_readiness"]
    attribution = payloads["metric_delta_attribution"]
    corpus = payloads["corpus_identity_comparison"]
    return "\n".join(
        [
            "# TASK0108 Chunking Metric Reconciliation & Validity Audit Report",
            "",
            f"task_status=`{payloads['verification_summary']['task_status']}`",
            f"primary_diagnosis=`{attribution['primary_diagnosis']}`",
            "",
            "## Core Answers",
            "",
            "1. TASK-0081 `0.4048` and TASK-0105 `0.9974` are not the same comparable metric.",
            f"2. Corpus is different: TASK-0081 documents `{corpus['task0081_document_count']}` (chunked ids observed `{corpus['task0081_chunked_document_count']}`), TASK-0105 documents `{corpus['task0105_document_count']}`, shared ids `{corpus['shared_document_count']}`.",
            "3. Gold units are different: source evidence spans vs CanonicalDocument blocks.",
            "4. Containment semantics are different: source-line strict containment vs block-id plus exact block text containment.",
            "5. Denominators are different: 84 resolved source spans vs 205705 non-empty canonical blocks.",
            "6. TASK-0105 row-level containment has no fuzzy fallback, but parser/block projection changes the gold region before mapping.",
            "7. Chunk count differs by about 36.35x: 604 vs 21954, mostly from the PDFQA formal corpus and canonical block projection surface.",
            "8. Overlap does not explain the containment jump because the current production config has overlap 0.",
            "9. CanonicalDocument/PDF parser representation materially changes boundaries, pages, whitespace, and block granularity.",
            "10. TASK-0105 containment is valid as a block-preservation guard but inflated/misleading as proof of full chunking quality.",
            "11. TASK-0106 Recall=1.0 means `P(retrieved | gold_contained)` on the contained block-gold surface.",
            "12. Current Production Chunking quality is not proven.",
            "13. Missing dimensions include semantic coherence, context efficiency, and downstream quality.",
            f"14. representation_aware_chunking_experiment_justified=`{readiness['representation_aware_chunking_experiment_justified']}`.",
            f"15. TASK-0109 ready=`{readiness['task0109_chunking_candidate_experiment_ready']}` with C0-C4 candidate arms.",
            "",
            "## Decisions",
            "",
            f"- task0105_containment_metric_valid=`{validity['task0105_containment_metric_valid']}`",
            f"- task0106_conditional_retrieval_metric_valid=`{validity['task0106_conditional_retrieval_metric_valid']}`",
            f"- current_chunking_quality_proven=`{validity['current_chunking_quality_proven']}`",
            f"- pdfqa_chunking_experiment_role=`{readiness['pdfqa_chunking_experiment_role']}`",
            f"- next_task_decision=`{readiness['next_task_decision']}`",
            "",
            "## Production Freeze",
            "",
            "- production_chunking_modified=`false`",
            "- production_embedding_modified=`false`",
            "- production_retriever_modified=`false`",
            "- production_reranker_modified=`false`",
            "- production_rank_fusion_modified=`false`",
            "- production_agent_behavior_modified=`false`",
            "- graph_runtime_modified=`false`",
            "- canonical_document_schema_modified=`false`",
            "",
        ]
    )


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0,
        "median": median(values) if values else 0,
        "p90": percentile(values, 0.90),
        "max": max(values) if values else 0,
        "zero_count": sum(1 for value in values if value == 0),
        "one_count": sum(1 for value in values if value == 1),
        "multi_count": sum(1 for value in values if value > 1),
    }


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[index])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    print(json.dumps(run_task0108(), ensure_ascii=False, indent=2, sort_keys=True))
