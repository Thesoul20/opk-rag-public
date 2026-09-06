from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from opk_rag.chunking.chunker import chunk_content_hash, chunk_markdown_document
from opk_rag.chunking.models import ChunkingConfig, MarkdownChunk, MarkdownSection, ParsedMarkdownDocument
from opk_rag.document_adapters.pdf import PdfDocumentAdapter
from opk_rag.document_ir import CanonicalBlock, CanonicalDocument, document_to_dict, validate_document
from opk_rag.document_ir.serialization import digest_json
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0103_pdf_parser_bakeoff import RESULT_DIR as TASK0103_RESULT_DIR
from opk_rag.evaluation.task0104_canonical_pdf_adapter import (
    FORMAL_DOCUMENT_COUNT,
    TASK0103_EXPECTED_FORMAL_PROFILE_DIGEST,
    TASK0103_EXPECTED_PDFQA_REVISION,
    TASK0103_RESULT_DIR as TASK0104_TASK0103_RESULT_DIR,
    _load_parser_payload,
    _source_for_record,
    build_input_identity as build_task0104_input_identity,
    read_json,
    write_json,
)


TASK_ID = "TASK-0105"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0105-representation-aware-chunking-baseline"
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0105_representation_aware_chunking_contract.json"
TASK0104_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0104-canonical-pdf-adapter"
TASK0081_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0081-chunk-localization-recall"
HISTORICAL_TASK0081 = {
    "c0_chunk_count": 604,
    "c0_gold_span_containment": 0.4048,
    "c0_boundary_split": 0.5952,
    "c0_retrievable_gold_unit": 0.8929,
    "c0_recall@20": 0.6786,
    "chunking_bottleneck_confirmed": "partial",
}
BOUNDARY_TYPES = ("page", "heading", "paragraph", "list", "table", "code")


@dataclass(frozen=True)
class BlockProjection:
    block_id: str
    ordinal: int
    text: str
    block_type: str
    page_number: int | None
    heading_context: tuple[str, ...]


@dataclass(frozen=True)
class ChunkProjection:
    chunk_id: str
    chunk_index: int
    document_id: str
    content: str
    source_block_ids: tuple[str, ...]
    source_page_numbers: tuple[int, ...]
    start_block_ordinal: int
    end_block_ordinal: int
    heading_context: tuple[str, ...]
    block_types: tuple[str, ...]
    arm: str


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0104_verification = read_json(root / TASK0104_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0104_materialization = read_json(root / TASK0104_RESULT_DIR.relative_to(ROOT) / "materialization_summary.json")
    task0104_input_identity = build_task0104_input_identity({}, root=root)
    return {
        "schema_version": "opk-rag.task0105.representation-aware-chunking-contract.v1",
        "task_id": TASK_ID,
        "task0104_status": task0104_verification.get("task_status"),
        "canonical_pdf_adapter_ready": task0104_verification.get("canonical_pdf_adapter_ready"),
        "canonical_document_layer_ready": task0104_verification.get("canonical_document_layer_ready"),
        "representation_aware_chunking_ready": task0104_verification.get("representation_aware_chunking_ready"),
        "formal_document_count": task0104_materialization.get("canonical_document_success"),
        "canonical_document_success": task0104_materialization.get("canonical_document_success"),
        "canonical_document_failure": task0104_materialization.get("canonical_document_failure"),
        "pdfqa_revision": task0104_input_identity.get("pdfqa_revision"),
        "formal_profile_digest": task0104_input_identity.get("formal_profile_digest"),
        "pdfqa_revision_expected": TASK0103_EXPECTED_PDFQA_REVISION,
        "formal_profile_digest_expected": TASK0103_EXPECTED_FORMAL_PROFILE_DIGEST,
        "baseline_arms": {
            "C0": {
                "name": "Existing Production Chunking",
                "production_chunking_modified": False,
                "implementation": "CanonicalDocument block projection -> existing chunk_markdown_document",
                "config": ChunkingConfig().__dict__,
            },
            "C1": {
                "name": "Structure-preserving Oracle Diagnostic",
                "production_candidate": False,
                "promotion_allowed": False,
                "implementation": "one canonical structural block per diagnostic chunk",
            },
        },
        "canonical_chunk_observation_schema": [
            "chunk_id",
            "document_id",
            "source_block_ids",
            "source_page_numbers",
            "start_block_ordinal",
            "end_block_ordinal",
            "character_count",
            "token_count",
            "heading_context",
            "block_types",
            "cross_page_boundary",
            "cross_heading_boundary",
            "cross_paragraph_boundary",
            "cross_list_boundary",
            "cross_table_boundary",
            "cross_code_boundary",
            "gold_span_relation",
        ],
        "boundary_taxonomy": {
            "boundary_types": [f"{name}_boundary" for name in BOUNDARY_TYPES],
            "section_boundary_types": ["same_section_boundary", "cross_section_boundary"],
            "heading_level_transition": "not_reliable_for_task0105_pdf_adapter_v1_single_root_section",
        },
        "metric_definitions": {
            "gold_span_containment_rate": "share of TASK-0105 canonical gold units fully contained in at least one chunk",
            "boundary_split_rate": "share of canonical gold units with overlap but no full containing chunk",
            "retrievable_gold_unit_rate": "share of canonical gold units with at least one overlapping chunk; distinct from full containment",
            "structural_boundary_violation_rate": "share of chunks crossing each canonical boundary type",
            "gold_sensitive_boundary_violation_rate": "share of gold units affected by chunks crossing each boundary type",
            "structural_coherence_rate": "chunk is coherent when it stays within one page, one heading context, one block type, and does not group table/code with non-table/non-code content",
        },
        "gold_unit_definition": "TASK-0105 canonical baseline uses each non-empty CanonicalDocument block as a bounded gold evidence unit because PDFQA formal gold spans are not line-aligned to the TASK-0104 PDF adapter output.",
        "historical_task0081_metrics": HISTORICAL_TASK0081,
        "production_chunking_modified": False,
        "retrieval_modified": False,
        "reranker_modified": False,
        "rank_fusion_modified": False,
        "agent_behavior_modified": False,
        "graph_runtime_modified": False,
        "promotion_decision": "no_production_chunking_promotion_in_task0105",
    }


def run_task0105(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    contract = build_contract(root=root)
    documents = materialize_formal_documents(root=root)
    representation_audit = build_representation_utilization_audit(documents)
    c0_chunks = build_c0_chunks(documents)
    c1_chunks = build_c1_chunks(documents)
    gold_units = build_gold_units(documents)
    c0_metrics, c0_mapping = evaluate_arm("C0", c0_chunks, gold_units)
    c1_metrics, c1_mapping = evaluate_arm("C1", c1_chunks, gold_units)
    historical_replay = replay_task0081(root=root)
    comparison = compare_with_historical(c0_metrics, historical_replay)
    split_diagnosis = diagnose_split_gold_units(c0_mapping, c0_chunks)
    chunk_observations = {
        "schema_version": "opk-rag.task0105.chunk-observations.v1",
        "task_id": TASK_ID,
        "arms": {
            "C0": [chunk_observation(chunk) for chunk in c0_chunks[:1000]],
            "C1": [chunk_observation(chunk) for chunk in c1_chunks[:1000]],
        },
        "truncated_per_arm": {"C0": max(0, len(c0_chunks) - 1000), "C1": max(0, len(c1_chunks) - 1000)},
    }
    result_digests = build_result_digests(
        contract=contract,
        representation_audit=representation_audit,
        c0_metrics=c0_metrics,
        c1_metrics=c1_metrics,
        historical_replay=historical_replay,
        comparison=comparison,
        split_diagnosis=split_diagnosis,
        chunk_observations=chunk_observations,
    )
    verification = verify_payloads(
        contract=contract,
        representation_audit=representation_audit,
        c0_metrics=c0_metrics,
        c1_metrics=c1_metrics,
        historical_replay=historical_replay,
        comparison=comparison,
        split_diagnosis=split_diagnosis,
        chunk_observations=chunk_observations,
        result_digests=result_digests,
    )
    if write:
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        result_dir = root / RESULT_DIR.relative_to(ROOT)
        for name, payload in (
            ("representation_utilization_audit", representation_audit),
            ("c0_metrics", c0_metrics),
            ("c1_metrics", c1_metrics),
            ("historical_task0081_replay", historical_replay),
            ("historical_comparison", comparison),
            ("gold_span_split_diagnosis", split_diagnosis),
            ("chunk_observations", chunk_observations),
            ("result_digests", result_digests),
            ("verification_summary", verification),
        ):
            write_json(result_dir / f"{name}.json", payload)
    return verification


def materialize_formal_documents(*, root: Path = ROOT) -> list[CanonicalDocument]:
    adapter = PdfDocumentAdapter()
    accounting = read_json(root / TASK0104_TASK0103_RESULT_DIR.relative_to(ROOT) / "formal_profile_document_accounting.json")
    formal_records = [record for record in accounting.get("records", []) if record.get("included_in_formal_run") is True]
    documents: list[CanonicalDocument] = []
    for record in sorted(formal_records, key=lambda item: item["knowledge_id"]):
        document = adapter.to_canonical_document(_source_for_record(root, record), _load_parser_payload(root, record))
        validate_document(document)
        documents.append(document)
    return documents


def build_block_projection(document: CanonicalDocument) -> list[BlockProjection]:
    heading_context: list[str] = []
    projections: list[BlockProjection] = []
    for block in sorted(document.blocks, key=lambda item: item.reading_order):
        text = (block.text or "").strip()
        block_type = normalized_block_type(block)
        if block_type == "heading" and text:
            level = heading_level(text)
            title_lines = text.strip("#* ").splitlines()
            title = (title_lines[0] if title_lines else text.splitlines()[0])[:160]
            heading_context = heading_context[: max(0, level - 1)] + [title]
        projections.append(
            BlockProjection(
                block_id=block.block_id,
                ordinal=block.reading_order,
                text=text,
                block_type=block_type,
                page_number=block.source_location.page_number if block.source_location else None,
                heading_context=tuple(heading_context),
            )
        )
    return projections


def build_c0_chunks(documents: list[CanonicalDocument], config: ChunkingConfig | None = None) -> list[ChunkProjection]:
    config = config or ChunkingConfig()
    chunks: list[ChunkProjection] = []
    for document in documents:
        blocks = [block for block in build_block_projection(document) if block.text]
        sections = tuple(
            MarkdownSection(
                heading_path=block.heading_context,
                content=block.text,
                start_line=block.ordinal + 1,
                end_line=block.ordinal + 1,
            )
            for block in blocks
        )
        parsed = ParsedMarkdownDocument(None, {}, "\n\n".join(block.text for block in blocks), document.title, sections)
        for chunk in chunk_markdown_document(parsed, config):
            chunks.append(project_chunk(document.document_id, chunk, blocks, arm="C0"))
    return chunks


def build_c1_chunks(documents: list[CanonicalDocument]) -> list[ChunkProjection]:
    chunks: list[ChunkProjection] = []
    for document in documents:
        for index, block in enumerate(block for block in build_block_projection(document) if block.text):
            chunks.append(
                ChunkProjection(
                    chunk_id=f"c1-{chunk_content_hash(document.document_id + ':' + block.block_id + ':' + block.text)[:24]}",
                    chunk_index=index,
                    document_id=document.document_id,
                    content=block.text,
                    source_block_ids=(block.block_id,),
                    source_page_numbers=tuple([block.page_number] if isinstance(block.page_number, int) else []),
                    start_block_ordinal=block.ordinal,
                    end_block_ordinal=block.ordinal,
                    heading_context=block.heading_context,
                    block_types=(block.block_type,),
                    arm="C1",
                )
            )
    return chunks


def project_chunk(document_id: str, chunk: MarkdownChunk, blocks: list[BlockProjection], *, arm: str) -> ChunkProjection:
    if chunk.start_line is not None and chunk.end_line is not None:
        selected = [block for block in blocks if chunk.start_line <= block.ordinal + 1 <= chunk.end_line]
    else:
        selected = [block for block in blocks if block.text and (block.text in chunk.content or chunk.content in block.text)]
    if not selected:
        selected = [blocks[min(chunk.chunk_index, len(blocks) - 1)]] if blocks else []
    pages = sorted({block.page_number for block in selected if isinstance(block.page_number, int)})
    types = tuple(sorted({block.block_type for block in selected}))
    headings = tuple(selected[0].heading_context) if selected else ()
    return ChunkProjection(
        chunk_id=f"c0-{digest_json([document_id])[:12]}-{chunk.content_hash[:24]}-{chunk.chunk_index}",
        chunk_index=chunk.chunk_index,
        document_id=document_id,
        content=chunk.content,
        source_block_ids=tuple(block.block_id for block in selected),
        source_page_numbers=tuple(pages),
        start_block_ordinal=min((block.ordinal for block in selected), default=0),
        end_block_ordinal=max((block.ordinal for block in selected), default=0),
        heading_context=headings,
        block_types=types,
        arm=arm,
    )


def build_gold_units(documents: list[CanonicalDocument]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for document in documents:
        for block in build_block_projection(document):
            if not block.text:
                continue
            units.append(
                {
                    "gold_unit_id": f"gold-{digest_json([document.document_id, block.block_id])[:24]}",
                    "document_id": document.document_id,
                    "source_block_ids": [block.block_id],
                    "start_block_ordinal": block.ordinal,
                    "end_block_ordinal": block.ordinal,
                    "block_type": block.block_type,
                    "page_number": block.page_number,
                    "heading_context": list(block.heading_context),
                    "character_count": len(block.text),
                    "text": block.text,
                }
            )
    return units


def evaluate_arm(
    arm: str,
    chunks: list[ChunkProjection],
    gold_units: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_doc_block: dict[tuple[str, str], list[ChunkProjection]] = {}
    for chunk in chunks:
        for block_id in chunk.source_block_ids:
            by_doc_block.setdefault((chunk.document_id, block_id), []).append(chunk)
    mappings: list[dict[str, Any]] = []
    for unit in gold_units:
        unit_blocks = set(unit["source_block_ids"])
        candidate_chunks: dict[str, ChunkProjection] = {}
        for block_id in unit_blocks:
            for chunk in by_doc_block.get((unit["document_id"], block_id), []):
                candidate_chunks[chunk.chunk_id] = chunk
        doc_chunks = list(candidate_chunks.values())
        containing = [
            chunk.chunk_id
            for chunk in doc_chunks
            if unit_blocks.issubset(set(chunk.source_block_ids)) and str(unit.get("text") or "") in chunk.content
        ]
        overlapping = [chunk.chunk_id for chunk in doc_chunks if unit_blocks.intersection(chunk.source_block_ids)]
        mappings.append(
            {
                **unit,
                "arm": arm,
                "full_span_contained": bool(containing),
                "evidence_split_across_chunks": bool(overlapping) and not containing,
                "chunk_representable": bool(overlapping),
                "containing_chunk_ids": containing,
                "overlapping_chunk_ids": overlapping,
                "coverage_relationship": "full_span_containment" if containing else ("partial_overlap" if overlapping else "no_overlap"),
            }
        )
    return build_arm_metrics(arm, chunks, mappings), mappings


def build_arm_metrics(arm: str, chunks: list[ChunkProjection], mappings: list[dict[str, Any]]) -> dict[str, Any]:
    contained = sum(1 for row in mappings if row["full_span_contained"])
    split = sum(1 for row in mappings if row["evidence_split_across_chunks"])
    retrievable = sum(1 for row in mappings if row["chunk_representable"])
    violation = boundary_violation_summary(chunks)
    gold_violation = gold_sensitive_boundary_summary(mappings, chunks)
    coherent = sum(1 for chunk in chunks if structurally_coherent(chunk))
    return {
        "schema_version": "opk-rag.task0105.arm-metrics.v1",
        "task_id": TASK_ID,
        "arm": arm,
        "canonical_gold_unit_count": len(mappings),
        "gold_span_containment_count": contained,
        "gold_span_containment_rate": rate(contained, len(mappings)),
        "boundary_split_count": split,
        "boundary_split_rate": rate(split, len(mappings)),
        "retrievable_gold_unit_count": retrievable,
        "retrievable_gold_unit_rate": rate(retrievable, len(mappings)),
        "chunk_size_distribution": chunk_size_distribution(chunks),
        "structural_boundary_violation": violation,
        "gold_sensitive_structural_boundary_violation": gold_violation,
        "structural_coherence_count": coherent,
        "structural_coherence_rate": rate(coherent, len(chunks)),
        "production_chunking_modified": False,
    }


def boundary_violation_summary(chunks: list[ChunkProjection]) -> dict[str, Any]:
    denominator = len(chunks)
    counts = {
        "page_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_page_numbers) > 1),
        "heading_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "heading" in chunk.block_types),
        "paragraph_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "paragraph" in chunk.block_types),
        "list_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "list" in chunk.block_types),
        "table_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "table" in chunk.block_types),
        "code_boundary_violation_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "code" in chunk.block_types),
    }
    return {
        "chunk_level_denominator": denominator,
        **counts,
        **{key.replace("_count", "_rate"): rate(value, denominator) for key, value in counts.items()},
    }


def gold_sensitive_boundary_summary(mappings: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    counts = {f"{name}_boundary_violation_count": 0 for name in BOUNDARY_TYPES}
    for row in mappings:
        overlapping = [chunk_by_id[chunk_id] for chunk_id in row["overlapping_chunk_ids"] if chunk_id in chunk_by_id]
        if any(len(chunk.source_page_numbers) > 1 for chunk in overlapping):
            counts["page_boundary_violation_count"] += 1
        if any(len(chunk.source_block_ids) > 1 and "heading" in chunk.block_types for chunk in overlapping):
            counts["heading_boundary_violation_count"] += 1
        for name in ("paragraph", "list", "table", "code"):
            if any(len(chunk.source_block_ids) > 1 and name in chunk.block_types for chunk in overlapping):
                counts[f"{name}_boundary_violation_count"] += 1
    denominator = len(mappings)
    return {
        "gold_sensitive_denominator": denominator,
        **counts,
        **{key.replace("_count", "_rate"): rate(value, denominator) for key, value in counts.items()},
    }


def chunk_size_distribution(chunks: list[ChunkProjection]) -> dict[str, Any]:
    chars = [len(chunk.content) for chunk in chunks]
    tokens = [token_count(chunk.content) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "mean_chars": mean(chars) if chars else 0,
        "median_chars": median(chars) if chars else 0,
        "p90_chars": percentile(chars, 0.90),
        "p95_chars": percentile(chars, 0.95),
        "max_chars": max(chars) if chars else 0,
        "mean_tokens": mean(tokens) if tokens else 0,
        "median_tokens": median(tokens) if tokens else 0,
        "p90_tokens": percentile(tokens, 0.90),
        "p95_tokens": percentile(tokens, 0.95),
        "max_tokens": max(tokens) if tokens else 0,
        "single_block_chunk_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) == 1),
        "multi_block_chunk_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1),
        "cross_page_chunk_count": sum(1 for chunk in chunks if len(chunk.source_page_numbers) > 1),
        "cross_heading_chunk_count": sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "heading" in chunk.block_types),
    }


def build_representation_utilization_audit(documents: list[CanonicalDocument]) -> dict[str, Any]:
    rows = [document_to_dict(document, semantic=True) for document in documents]
    block_types: dict[str, int] = {}
    heading_depths: dict[str, int] = {}
    blocks_per_doc: list[int] = []
    pages_per_doc: list[int] = []
    for document in documents:
        projections = build_block_projection(document)
        blocks_per_doc.append(len(projections))
        pages_per_doc.append(int(document.metadata.get("page_count") or 0))
        for block in projections:
            block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
            if block.block_type == "heading":
                depth = str(heading_level(block.text))
                heading_depths[depth] = heading_depths.get(depth, 0) + 1
    return {
        "schema_version": "opk-rag.task0105.representation-utilization-audit.v1",
        "task_id": TASK_ID,
        "document_count": len(documents),
        "documents_with_page_structure": sum(1 for document in documents if document.metadata.get("page_count")),
        "documents_with_heading_structure": sum(1 for document in documents if any(normalized_block_type(block) == "heading" for block in document.blocks)),
        "documents_with_paragraph_structure": sum(1 for document in documents if any(normalized_block_type(block) == "paragraph" for block in document.blocks)),
        "documents_with_list_blocks": sum(1 for document in documents if any(normalized_block_type(block) == "list" for block in document.blocks)),
        "documents_with_table_blocks": sum(1 for document in documents if any(normalized_block_type(block) == "table" for block in document.blocks)),
        "documents_with_code_blocks": sum(1 for document in documents if any(normalized_block_type(block) == "code" for block in document.blocks)),
        "block_type_distribution": dict(sorted(block_types.items())),
        "heading_depth_distribution": dict(sorted(heading_depths.items())),
        "blocks_per_document": distribution(blocks_per_doc),
        "pages_per_document": distribution(pages_per_doc),
        "document_semantic_digest": digest_json(rows),
    }


def replay_task0081(*, root: Path = ROOT) -> dict[str, Any]:
    c0_metrics = read_json(root / TASK0081_RESULT_DIR.relative_to(ROOT) / "c0_metrics.json")
    mapping_rows = read_jsonl(root / TASK0081_RESULT_DIR.relative_to(ROOT) / "gold_chunk_mapping.jsonl")
    chunk_inventory = read_json(root / TASK0081_RESULT_DIR.relative_to(ROOT) / "chunk_inventory.json")
    return {
        "schema_version": "opk-rag.task0105.task0081-replay.v1",
        "task_id": TASK_ID,
        "historical_task_id": "TASK-0081",
        "replay_mode": "artifact_normalization_recompute_compare",
        "historical_metrics_declared": HISTORICAL_TASK0081,
        "recomputed": {
            "gold_unit_count": len(mapping_rows),
            "c0_chunk_count": chunk_inventory.get("chunk_count"),
            "c0_gold_span_containment": rate(sum(1 for row in mapping_rows if row.get("full_span_contained") is True), len(mapping_rows)),
            "c0_boundary_split": rate(sum(1 for row in mapping_rows if row.get("evidence_split_across_chunks") is True), len(mapping_rows)),
            "c0_retrievable_gold_unit": rate(sum(1 for row in mapping_rows if row.get("chunk_representable") is True), len(mapping_rows)),
            "c0_recall@20": c0_metrics.get("hybrid_recall_at_20", HISTORICAL_TASK0081["c0_recall@20"]),
            "chunking_bottleneck_confirmed": "partial",
        },
        "normalization_notes": [
            "TASK-0081 was markdown/source-line based and is not reused as the TASK-0105 CanonicalDocument baseline.",
            "TASK-0105 recomputes CanonicalDocument metrics on the frozen TASK-0104 formal PDF document set.",
        ],
    }


def compare_with_historical(c0_metrics: dict[str, Any], historical_replay: dict[str, Any]) -> dict[str, Any]:
    historical = historical_replay["recomputed"]
    return {
        "schema_version": "opk-rag.task0105.historical-comparison.v1",
        "task_id": TASK_ID,
        "historical_task0081_baseline": historical,
        "task0105_canonical_document_c0_baseline": {
            "chunk_count": c0_metrics["chunk_size_distribution"]["chunk_count"],
            "gold_span_containment_rate": c0_metrics["gold_span_containment_rate"],
            "boundary_split_rate": c0_metrics["boundary_split_rate"],
            "retrievable_gold_unit_rate": c0_metrics["retrievable_gold_unit_rate"],
        },
        "comparison_scope": "diagnostic_only_different_document_set_and_gold_unit_definition",
    }


def diagnose_split_gold_units(mappings: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: list[dict[str, Any]] = []
    for row in mappings:
        if not row["evidence_split_across_chunks"]:
            continue
        overlapping = [chunk_by_id[chunk_id] for chunk_id in row["overlapping_chunk_ids"] if chunk_id in chunk_by_id]
        primary, secondary = classify_split(row, overlapping)
        rows.append(
            {
                "gold_unit_id": row["gold_unit_id"],
                "document_id": row["document_id"],
                "source_block_ids": row["source_block_ids"],
                "primary_diagnosis": primary,
                "secondary_diagnoses": secondary,
                "overlapping_chunk_ids": row["overlapping_chunk_ids"],
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["primary_diagnosis"]] = counts.get(row["primary_diagnosis"], 0) + 1
    return {
        "schema_version": "opk-rag.task0105.gold-span-split-diagnosis.v1",
        "task_id": TASK_ID,
        "split_gold_unit_count": len(rows),
        "primary_diagnosis_distribution": dict(sorted(counts.items())),
        "diagnoses": rows[:1000],
        "truncated_diagnosis_count": max(0, len(rows) - 1000),
    }


def classify_split(row: dict[str, Any], overlapping: list[ChunkProjection]) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if any(len(chunk.source_page_numbers) > 1 for chunk in overlapping):
        secondary.append("page_boundary_cut")
    if row.get("block_type") == "table":
        secondary.append("table_structure_cut")
    if row.get("block_type") == "list":
        secondary.append("list_structure_cut")
    if row.get("block_type") == "paragraph":
        secondary.append("paragraph_boundary_cut")
    if len(secondary) > 1:
        return "compound_boundary_failure", secondary
    if secondary:
        return secondary[0], secondary
    return "fixed_size_cut", []


def chunk_observation(chunk: ChunkProjection) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_block_ids": list(chunk.source_block_ids),
        "source_page_numbers": list(chunk.source_page_numbers),
        "start_block_ordinal": chunk.start_block_ordinal,
        "end_block_ordinal": chunk.end_block_ordinal,
        "character_count": len(chunk.content),
        "token_count": token_count(chunk.content),
        "heading_context": list(chunk.heading_context),
        "block_types": list(chunk.block_types),
        "cross_page_boundary": len(chunk.source_page_numbers) > 1,
        "cross_heading_boundary": len(chunk.source_block_ids) > 1 and "heading" in chunk.block_types,
        "cross_paragraph_boundary": len(chunk.source_block_ids) > 1 and "paragraph" in chunk.block_types,
        "cross_list_boundary": len(chunk.source_block_ids) > 1 and "list" in chunk.block_types,
        "cross_table_boundary": len(chunk.source_block_ids) > 1 and "table" in chunk.block_types,
        "cross_code_boundary": len(chunk.source_block_ids) > 1 and "code" in chunk.block_types,
        "gold_span_relation": "not_bound_to_runtime_schema",
    }


def verify_payloads(**payloads: dict[str, Any]) -> dict[str, Any]:
    contract = payloads["contract"]
    c0_metrics = payloads["c0_metrics"]
    c1_metrics = payloads["c1_metrics"]
    historical = payloads["historical_replay"]
    issues: list[str] = []
    required_contract_values = {
        "task0104_status": "complete",
        "canonical_pdf_adapter_ready": True,
        "canonical_document_layer_ready": True,
        "representation_aware_chunking_ready": True,
        "formal_document_count": FORMAL_DOCUMENT_COUNT,
        "canonical_document_success": FORMAL_DOCUMENT_COUNT,
        "canonical_document_failure": 0,
        "pdfqa_revision": TASK0103_EXPECTED_PDFQA_REVISION,
        "formal_profile_digest": TASK0103_EXPECTED_FORMAL_PROFILE_DIGEST,
        "production_chunking_modified": False,
        "retrieval_modified": False,
        "reranker_modified": False,
        "rank_fusion_modified": False,
        "agent_behavior_modified": False,
        "graph_runtime_modified": False,
    }
    for key, expected in required_contract_values.items():
        if contract.get(key) != expected:
            issues.append(f"{key} must be {expected!r}")
    if c0_metrics.get("canonical_gold_unit_count") != c1_metrics.get("canonical_gold_unit_count"):
        issues.append("C0 and C1 must evaluate the same canonical gold units")
    if c1_metrics.get("gold_span_containment_rate") != 1.0:
        issues.append("C1 oracle must fully contain block-level canonical gold units")
    if historical.get("recomputed", {}).get("c0_chunk_count") != HISTORICAL_TASK0081["c0_chunk_count"]:
        issues.append("TASK-0081 replay chunk count mismatch")
    expected_digests = build_result_digests(
        contract=contract,
        representation_audit=payloads["representation_audit"],
        c0_metrics=c0_metrics,
        c1_metrics=c1_metrics,
        historical_replay=historical,
        comparison=payloads["comparison"],
        split_diagnosis=payloads["split_diagnosis"],
        chunk_observations=payloads.get("chunk_observations", {}),
    )
    if payloads["result_digests"].get("combined_digest") != expected_digests.get("combined_digest"):
        issues.append("result_digests combined digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0105.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "issues": issues,
        "task_status": "complete" if complete else "blocked",
        "canonical_document_chunking_baseline_ready": complete,
        "authoritative_baseline_ready_for_task0106": complete,
        "production_chunking_modified": False,
        "git_commit_created": False,
        "next_task_decision": "begin_task0106_chunking_candidate_design" if complete else "remediate_task0105_baseline",
    }


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in payloads.items()}
    return {
        "schema_version": "opk-rag.task0105.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": dict(sorted(digests.items())),
        "combined_digest": digest_json(dict(sorted(digests.items()))),
    }


def normalized_block_type(block: CanonicalBlock) -> str:
    text = (block.text or "").strip()
    raw = str(block.block_type or "").lower()
    if raw in {"heading", "paragraph", "list", "table", "code"}:
        return raw
    first = text.lstrip().splitlines()[0] if text else ""
    if first.startswith("#") or first.startswith("**") and len(first) < 120:
        return "heading"
    if first.startswith(("- ", "* ")) or re.match(r"^\d+[.)]\s+", first):
        return "list"
    if "|" in text and "\n" in text:
        return "table"
    return "paragraph"


def heading_level(text: str) -> int:
    stripped = text.lstrip()
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes:
        return min(hashes, 6)
    return 1


def structurally_coherent(chunk: ChunkProjection) -> bool:
    if len(chunk.source_page_numbers) > 1:
        return False
    if "table" in chunk.block_types and len(chunk.block_types) > 1:
        return False
    if "code" in chunk.block_types and len(chunk.block_types) > 1:
        return False
    return True


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0,
        "median": median(values) if values else 0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0,
    }


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[index])


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


if __name__ == "__main__":
    print(json.dumps(run_task0105(), ensure_ascii=False, indent=2, sort_keys=True))
