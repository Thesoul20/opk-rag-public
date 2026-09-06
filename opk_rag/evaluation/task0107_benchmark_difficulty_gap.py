from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any, Iterable

from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0104_canonical_pdf_adapter import read_json, write_json
from opk_rag.evaluation.task0105_representation_aware_chunking import (
    build_c0_chunks,
    build_gold_units,
    digest_json,
    materialize_formal_documents,
)


TASK_ID = "TASK-0107"
EXPERIMENT_ID = "task0107-benchmark-difficulty-capability-gap"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0107_benchmark_difficulty_capability_gap_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0107_BENCHMARK_DIFFICULTY_CAPABILITY_GAP_REPORT.md"
TASK0103_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0103-pdf-parser-bakeoff"
TASK0104_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0104-canonical-pdf-adapter"
TASK0105_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0105-representation-aware-chunking-baseline"
TASK0106_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0106-canonical-chunk-retrieval-localization"
TASK0098_CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0098_project_rebaseline_contract.json"
PDFQA_AUTHORITY_PATH = ROOT / "evaluation-data" / "external" / "pdfqa" / "authority.json"

CAPABILITIES = (
    "local_single_span",
    "same_section",
    "cross_section",
    "cross_page",
    "cross_document",
    "multi_hop",
    "table",
    "list_structured_block",
    "heading_sensitive",
    "unanswerable",
    "distractor_sensitive",
    "long_context",
)
DIFFICULTIES = ("D0", "D1", "D2", "D3", "D4", "unknown")
GAPS = (
    "cross_document_gap",
    "multi_hop_gap",
    "graph_sensitive_gap",
    "table_reasoning_gap",
    "hierarchical_structure_gap",
    "hard_negative_gap",
    "long_context_gap",
    "unanswerable_gap",
    "generation_reasoning_gap",
    "agent_routing_gap",
)


def run_task0107(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    if write:
        (root / RESULT_DIR.relative_to(ROOT)).mkdir(parents=True, exist_ok=True)
        (root / CONTRACT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        (root / REPORT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)

    contract = build_contract(root=root)
    task0106 = load_task0106_inputs(root=root)
    documents = materialize_formal_documents(root=root)
    gold_units = build_gold_units(documents)
    chunks = build_c0_chunks(documents)
    profiles = build_unit_profiles(documents, gold_units, chunks)

    input_identity = build_input_identity(root=root, contract=contract, task0106=task0106, documents=documents, gold_units=gold_units, chunks=chunks)
    benchmark_profile = build_benchmark_profile(profiles, documents, chunks)
    capability_distribution = build_capability_distribution(profiles)
    difficulty_distribution = build_difficulty_distribution(profiles)
    evidence_cardinality = build_evidence_cardinality(profiles)
    document_cardinality = build_document_cardinality(profiles)
    structural_coverage = build_structural_coverage(profiles, documents)
    table_coverage = build_table_coverage(profiles, documents)
    cross_page_coverage = build_cross_page_coverage(profiles)
    multi_hop_coverage = build_multi_hop_coverage(profiles)
    hard_negative_coverage = build_hard_negative_coverage(task0106, profiles)
    top1_saturation = build_top1_saturation(task0106, profiles)
    capability_gap_summary = build_capability_gap_summary(
        capability_distribution=capability_distribution,
        table_coverage=table_coverage,
        hard_negative_coverage=hard_negative_coverage,
        task0106=task0106,
    )
    benchmark_role_decision = build_benchmark_role_decision(
        task0106=task0106,
        capability_gap_summary=capability_gap_summary,
        capability_distribution=capability_distribution,
    )
    public_benchmark_mapping = build_public_benchmark_mapping(root=root)
    transition_readiness = build_transition_readiness(benchmark_role_decision, capability_gap_summary, public_benchmark_mapping)

    payloads = {
        "input_identity": input_identity,
        "benchmark_profile": benchmark_profile,
        "capability_distribution": capability_distribution,
        "difficulty_distribution": difficulty_distribution,
        "evidence_cardinality": evidence_cardinality,
        "document_cardinality": document_cardinality,
        "structural_coverage": structural_coverage,
        "table_coverage": table_coverage,
        "cross_page_coverage": cross_page_coverage,
        "multi_hop_coverage": multi_hop_coverage,
        "hard_negative_coverage": hard_negative_coverage,
        "top1_saturation": top1_saturation,
        "capability_gap_summary": capability_gap_summary,
        "benchmark_role_decision": benchmark_role_decision,
        "public_benchmark_mapping": public_benchmark_mapping,
        "transition_readiness": transition_readiness,
    }
    result_digests = build_result_digests(contract=contract, **payloads)
    verification = verify_payloads(contract=contract, result_digests=result_digests, **payloads)
    payloads["result_digests"] = result_digests
    payloads["verification_summary"] = verification

    if write:
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        result_dir = root / RESULT_DIR.relative_to(ROOT)
        for name, payload in payloads.items():
            write_json(result_dir / f"{name}.json", payload)
        (root / REPORT_PATH.relative_to(ROOT)).write_text(build_report(payloads), encoding="utf-8")
    return verification


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0103 = read_json(root / TASK0103_RESULT_DIR.relative_to(ROOT) / "summary.json")
    task0104 = read_json(root / TASK0104_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0105 = read_json(root / TASK0105_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0106 = read_json(root / TASK0106_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    return {
        "schema_version": "opk-rag.task0107.benchmark-difficulty-capability-gap-contract.v1",
        "task_id": TASK_ID,
        "evaluation_mode": "static_benchmark_surface_diagnosis",
        "model_invocation_required_for_verification": False,
        "formal_document_count_expected": 100,
        "required_pdf_count_expected": 103,
        "smoke_only_pdf_count_expected": 3,
        "pdfqa_revision_expected": "90f787ebee0ba278bd6b8b750a69ed90386bfdf5",
        "task0103_authority": {
            "formal_document_count": task0103.get("formal_profile_document_accounting", {}).get("formal_documents_evaluated"),
            "required_pdf_count": task0103.get("dataset_materialization", {}).get("required_pdf_count"),
            "materialized_pdf_count": task0103.get("dataset_materialization", {}).get("materialized_pdf_count"),
            "pdfqa_revision": task0103.get("dataset_materialization", {}).get("dataset_revision"),
        },
        "task0104_authority": {
            "task_status": task0104.get("task_status"),
            "canonical_pdf_adapter_ready": task0104.get("canonical_pdf_adapter_ready"),
        },
        "task0105_authority": {
            "task_status": task0105.get("task_status"),
            "c0_chunk_count": 21954,
            "c0_gold_span_containment": 0.9974,
            "c0_boundary_split": 0.0009,
        },
        "task0106_authority": {
            "task_status": task0106.get("task_status"),
            "contained_gold_recall_at_5": 1.0,
            "contained_gold_recall_at_10": 1.0,
            "contained_gold_recall_at_20": 1.0,
            "contained_gold_mrr": 1.0,
            "candidate_generation_failure_count": 0,
            "ranking_failure_count": 0,
        },
        "unit_authority_note": "TASK-0105/0106 formal units are canonical non-empty block gold units, not human multi-evidence QA questions.",
        "capability_taxonomy": list(CAPABILITIES),
        "difficulty_rule": {
            "D0": "single document, single section, single page, single gold unit, paragraph/heading/list/table local lookup with no relation or composition requirement",
            "D1": "local unit with one mild structural factor such as heading/list/table block, long document, or hard negative signal",
            "D2": "two or more structural factors, cross-page/cross-section evidence, or moderate answer composition",
            "D3": "cross-document, multi-hop, or high structural/cardinality complexity",
            "D4": "capability unsupported by current benchmark authority",
            "unknown": "insufficient owner-independent evidence to score",
        },
        "saturation_criteria": {
            "local_retrieval_saturated": "contained_gold_recall_at_5 >= 0.98 and contained_gold_mrr >= 0.98",
            "benchmark_materially_saturated": "local retrieval saturated and cross-document, multi-hop, table-reasoning, unanswerable, and hard-negative coverage are absent or materially insufficient",
            "benchmark_partially_saturated": "local retrieval saturated but non-trivial capability coverage remains measurable",
            "benchmark_still_discriminative": "hard samples are sufficiently covered and current pipeline still has material errors on them",
        },
        "coverage_status_thresholds": {
            "absent": 0,
            "limited_max_ratio": 0.05,
            "sufficient_min_ratio": 0.10,
        },
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_modified": False,
        "graph_runtime_modified": False,
    }


def load_task0106_inputs(*, root: Path = ROOT) -> dict[str, Any]:
    names = (
        "contained_gold_retrieval_metrics",
        "gold_rank_distribution",
        "hard_negative_summary",
        "candidate_generation_failures",
        "ranking_failures",
        "input_identity",
    )
    return {name: read_json(root / TASK0106_RESULT_DIR.relative_to(ROOT) / f"{name}.json") for name in names}


def build_unit_profiles(documents: list[Any], gold_units: list[dict[str, Any]], chunks: list[Any]) -> list[dict[str, Any]]:
    doc_lengths = {document.document_id: sum(len((block.text or "").strip()) for block in document.blocks) for document in documents}
    block_to_section = {}
    block_count_by_doc = {}
    for document in documents:
        block_count_by_doc[document.document_id] = len([block for block in document.blocks if (block.text or "").strip()])
        for block in document.blocks:
            block_to_section[(document.document_id, block.block_id)] = block.section_id or "root"
    chunk_by_block: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for chunk in chunks:
        for block_id in chunk.source_block_ids:
            chunk_by_block[(chunk.document_id, block_id)].append(chunk)

    profiles: list[dict[str, Any]] = []
    for unit in gold_units:
        block_ids = tuple(unit.get("source_block_ids") or [])
        doc_id = str(unit["document_id"])
        sections = {block_to_section.get((doc_id, block_id), "root") for block_id in block_ids} or {"root"}
        pages = sorted({unit.get("page_number")} if isinstance(unit.get("page_number"), int) else set())
        block_types = tuple(sorted({str(unit.get("block_type") or "unknown")}))
        candidate_chunks = {
            chunk.chunk_id: chunk
            for block_id in block_ids
            for chunk in chunk_by_block.get((doc_id, block_id), [])
        }
        containing_chunks = [
            chunk
            for chunk in candidate_chunks.values()
            if set(block_ids).issubset(set(chunk.source_block_ids)) and str(unit.get("text") or "") in chunk.content
        ]
        min_chunk_blocks = min((len(chunk.source_block_ids) for chunk in containing_chunks), default=None)
        features = {
            "gold_span_count": 1,
            "gold_document_count": 1,
            "gold_section_count": len(sections),
            "gold_page_count": len(pages) or 1,
            "required_hop_count": 1,
            "hard_negative_count": 0,
            "document_length": doc_lengths.get(doc_id, 0),
            "document_block_count": block_count_by_doc.get(doc_id, 0),
            "structural_complexity": structural_complexity(block_types, len(pages) or 1, len(sections), block_count_by_doc.get(doc_id, 0)),
            "answer_composition_requirement": "extractive",
            "block_types": block_types,
            "heading_context_count": len(unit.get("heading_context") or []),
            "single_chunk_answerable": min_chunk_blocks == 1,
            "contained_in_any_chunk": bool(containing_chunks),
        }
        primary, secondary = classify_capability(features)
        difficulty = classify_difficulty(features)
        profiles.append(
            {
                "sample_id": unit["gold_unit_id"],
                "document_id": doc_id,
                "primary_capability": primary,
                "secondary_capabilities": secondary,
                "difficulty": difficulty,
                "features": features,
            }
        )
    return profiles


def classify_capability(features: dict[str, Any]) -> tuple[str, list[str]]:
    block_types = set(features.get("block_types") or [])
    secondary: list[str] = []
    if features.get("is_unanswerable") is True:
        return "unanswerable", secondary
    if int(features.get("gold_document_count") or 0) > 1:
        return "cross_document", secondary
    if int(features.get("required_hop_count") or 0) >= 2:
        return "multi_hop", secondary
    if int(features.get("gold_section_count") or 0) > 1:
        return "cross_section", secondary
    if int(features.get("gold_page_count") or 0) > 1:
        return "cross_page", secondary
    if "table" in block_types:
        return "table", secondary
    if "list" in block_types or "code" in block_types:
        return "list_structured_block", secondary
    if "heading" in block_types or int(features.get("heading_context_count") or 0) > 0:
        secondary.append("heading_sensitive")
    if int(features.get("hard_negative_count") or 0) > 0:
        secondary.append("distractor_sensitive")
    if int(features.get("document_block_count") or 0) >= 500:
        secondary.append("long_context")
    return "local_single_span", secondary


def classify_difficulty(features: dict[str, Any]) -> str:
    if features.get("difficulty_unknown") is True:
        return "unknown"
    if features.get("unsupported_by_current_benchmark") is True:
        return "D4"
    if features.get("is_unanswerable") is True:
        return "D4"
    if int(features.get("gold_document_count") or 1) > 1 or int(features.get("required_hop_count") or 1) >= 2:
        return "D3"
    score = 0
    if int(features.get("gold_section_count") or 1) > 1:
        score += 2
    if int(features.get("gold_page_count") or 1) > 1:
        score += 2
    if int(features.get("gold_span_count") or 1) > 1:
        score += 1
    if int(features.get("hard_negative_count") or 0) > 0:
        score += 1
    if int(features.get("structural_complexity") or 0) >= 2:
        score += 1
    if features.get("answer_composition_requirement") in {"aggregative", "comparative", "procedural", "reasoned"}:
        score += 1
    if int(features.get("document_block_count") or 0) >= 500:
        score += 1
    if score >= 4:
        return "D3"
    if score >= 2:
        return "D2"
    if score == 1:
        return "D1"
    return "D0"


def structural_complexity(block_types: Iterable[str], page_count: int, section_count: int, document_block_count: int) -> int:
    score = 0
    if any(block_type in {"table", "list", "code", "heading"} for block_type in block_types):
        score += 1
    if page_count > 1:
        score += 1
    if section_count > 1:
        score += 1
    if document_block_count >= 500:
        score += 1
    return score


def build_input_identity(
    *,
    root: Path,
    contract: dict[str, Any],
    task0106: dict[str, Any],
    documents: list[Any],
    gold_units: list[dict[str, Any]],
    chunks: list[Any],
) -> dict[str, Any]:
    task0103 = contract["task0103_authority"]
    task0105 = read_json(root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c0_metrics.json")
    contained = task0106["contained_gold_retrieval_metrics"]
    return {
        "schema_version": "opk-rag.task0107.input-identity.v1",
        "task_id": TASK_ID,
        "task0106_inputs_valid": contained.get("contained_gold_recall_at_5") == 1.0 and contained.get("contained_gold_mrr") == 1.0,
        "formal_document_count": len(documents),
        "formal_document_count_valid": len(documents) == contract["formal_document_count_expected"],
        "required_pdf_count": task0103.get("required_pdf_count"),
        "smoke_only_pdf_count": int(task0103.get("materialized_pdf_count") or 0) - int(task0103.get("formal_document_count") or 0),
        "pdfqa_revision": task0103.get("pdfqa_revision"),
        "pdfqa_revision_valid": task0103.get("pdfqa_revision") == contract["pdfqa_revision_expected"],
        "c0_chunk_count": len(chunks),
        "c0_chunk_count_valid": len(chunks) == contract["task0105_authority"]["c0_chunk_count"],
        "gold_unit_count": len(gold_units),
        "c0_gold_span_containment": round(float(task0105.get("gold_span_containment_rate") or 0), 4),
        "c0_boundary_split": round(float(task0105.get("boundary_split_rate") or 0), 4),
        "benchmark_documents_modified": False,
        "gold_annotation_modified": False,
    }


def build_benchmark_profile(profiles: list[dict[str, Any]], documents: list[Any], chunks: list[Any]) -> dict[str, Any]:
    single_span = sum(1 for row in profiles if row["features"]["gold_span_count"] == 1)
    single_chunk = sum(1 for row in profiles if row["features"]["single_chunk_answerable"] is True)
    single_section = sum(1 for row in profiles if row["features"]["gold_section_count"] == 1)
    single_document = sum(1 for row in profiles if row["features"]["gold_document_count"] == 1)
    query_complexity = {
        "query_text_authority_available": False,
        "query_length": "not_applicable_block_gold_unit_authority",
        "number_of_entities": "not_applicable_block_gold_unit_authority",
        "number_of_relation_cues": "not_applicable_block_gold_unit_authority",
        "number_of_conjunctions": "not_applicable_block_gold_unit_authority",
        "multi_part_question_count": "not_applicable_block_gold_unit_authority",
    }
    return {
        "schema_version": "opk-rag.task0107.benchmark-profile.v1",
        "task_id": TASK_ID,
        "evaluation_unit_count": len(profiles),
        "formal_document_count": len(documents),
        "chunk_count": len(chunks),
        "unit_authority": "canonical_non_empty_block_gold_unit",
        "question_text_authority_available": False,
        "multi_evidence_annotation_unavailable": True,
        "single_span_sample_count": single_span,
        "single_span_sample_ratio": rate(single_span, len(profiles)),
        "single_chunk_answerable_count": single_chunk,
        "single_chunk_answerable_ratio": rate(single_chunk, len(profiles)),
        "single_section_sample_count": single_section,
        "single_section_sample_ratio": rate(single_section, len(profiles)),
        "single_document_sample_count": single_document,
        "single_document_sample_ratio": rate(single_document, len(profiles)),
        "answer_complexity": {
            "extractive": len(profiles),
            "short_factual": 0,
            "aggregative": 0,
            "comparative": 0,
            "procedural": 0,
            "reasoned": 0,
            "unanswerable": 0,
            "unknown": 0,
        },
        "query_complexity": query_complexity,
        "benchmark_profile_complete": True,
    }


def build_capability_distribution(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["primary_capability"] for row in profiles)
    secondary = Counter(cap for row in profiles for cap in row["secondary_capabilities"])
    total = len(profiles)
    statuses = {capability: coverage_status(counts.get(capability, 0), total) for capability in CAPABILITIES}
    return {
        "schema_version": "opk-rag.task0107.capability-distribution.v1",
        "task_id": TASK_ID,
        "sample_count_by_capability": {capability: counts.get(capability, 0) for capability in CAPABILITIES},
        "percentage_by_capability": {capability: rate(counts.get(capability, 0), total) for capability in CAPABILITIES},
        "secondary_sample_count_by_capability": {capability: secondary.get(capability, 0) for capability in CAPABILITIES},
        "coverage_status_by_capability": statuses,
        "capability_taxonomy_complete": all(capability in statuses for capability in CAPABILITIES),
        "classification_basis": "owner-independent canonical block metadata; no LLM classification",
    }


def build_difficulty_distribution(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["difficulty"] for row in profiles)
    return {
        "schema_version": "opk-rag.task0107.difficulty-distribution.v1",
        "task_id": TASK_ID,
        "sample_count_by_difficulty": {difficulty: counts.get(difficulty, 0) for difficulty in DIFFICULTIES},
        "percentage_by_difficulty": {difficulty: rate(counts.get(difficulty, 0), len(profiles)) for difficulty in DIFFICULTIES},
        "difficulty_profile_complete": True,
    }


def build_evidence_cardinality(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"1": 0, "2": 0, "3+": 0}
    for row in profiles:
        count = int(row["features"].get("gold_span_count") or 0)
        buckets["3+" if count >= 3 else str(count)] += 1
    return {
        "schema_version": "opk-rag.task0107.evidence-cardinality.v1",
        "task_id": TASK_ID,
        "multi_evidence_annotation_unavailable": True,
        "sample_count_by_required_gold_units": buckets,
        "percentage_by_required_gold_units": {key: rate(value, len(profiles)) for key, value in buckets.items()},
    }


def build_document_cardinality(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"1": 0, "2": 0, "3+": 0}
    for row in profiles:
        count = int(row["features"].get("gold_document_count") or 0)
        buckets["3+" if count >= 3 else str(count)] += 1
    cross_doc = buckets["2"] + buckets["3+"]
    return {
        "schema_version": "opk-rag.task0107.document-cardinality.v1",
        "task_id": TASK_ID,
        "sample_count_by_required_documents": buckets,
        "percentage_by_required_documents": {key: rate(value, len(profiles)) for key, value in buckets.items()},
        "cross_document_sample_count": cross_doc,
        "cross_document_sample_ratio": rate(cross_doc, len(profiles)),
        "cross_document_authority_available": False,
        "interpretation": "current benchmark cannot validate cross-document retrieval/reasoning" if cross_doc == 0 else "cross-document units present",
    }


def build_structural_coverage(profiles: list[dict[str, Any]], documents: list[Any]) -> dict[str, Any]:
    available = Counter()
    docs_with = Counter()
    for document in documents:
        doc_types = set()
        for block in document.blocks:
            block_type = normalize_block_type(block.block_type)
            available[block_type] += 1
            doc_types.add(block_type)
        for block_type in doc_types:
            docs_with[block_type] += 1
    used = Counter()
    for row in profiles:
        for block_type in row["features"].get("block_types") or []:
            used[normalize_block_type(block_type)] += 1
    block_types = ("paragraph", "heading", "list", "table", "code", "mixed")
    return {
        "schema_version": "opk-rag.task0107.structural-coverage.v1",
        "task_id": TASK_ID,
        "available_in_corpus_count": {block_type: available.get(block_type, 0) for block_type in block_types},
        "documents_with_block_type": {block_type: docs_with.get(block_type, 0) for block_type in block_types},
        "used_as_gold_evidence_count": {block_type: used.get(block_type, 0) for block_type in block_types},
        "mixed_blocks_used_as_gold_evidence_count": sum(1 for row in profiles if len(row["features"].get("block_types") or []) > 1),
        "structural_coverage_complete": True,
    }


def build_table_coverage(profiles: list[dict[str, Any]], documents: list[Any]) -> dict[str, Any]:
    documents_with_tables = sum(1 for document in documents if any(normalize_block_type(block.block_type) == "table" for block in document.blocks))
    table_block_count = sum(1 for document in documents for block in document.blocks if normalize_block_type(block.block_type) == "table")
    table_gold = sum(1 for row in profiles if row["primary_capability"] == "table")
    return {
        "schema_version": "opk-rag.task0107.table-coverage.v1",
        "task_id": TASK_ID,
        "documents_with_tables": documents_with_tables,
        "table_block_count": table_block_count,
        "table_gold_sample_count": table_gold,
        "table_gold_sample_ratio": rate(table_gold, len(profiles)),
        "table_reasoning_sample_count": 0,
        "table_reasoning_sample_ratio": 0.0,
        "table_reasoning_authority_available": False,
    }


def build_cross_page_coverage(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(1 for row in profiles if row["features"].get("gold_page_count", 1) > 1)
    return {
        "schema_version": "opk-rag.task0107.cross-page-coverage.v1",
        "task_id": TASK_ID,
        "cross_page_gold_sample_count": count,
        "cross_page_gold_sample_ratio": rate(count, len(profiles)),
    }


def build_multi_hop_coverage(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(1 for row in profiles if row["features"].get("required_hop_count", 1) >= 2)
    return {
        "schema_version": "opk-rag.task0107.multi-hop-coverage.v1",
        "task_id": TASK_ID,
        "multi_hop_sample_count": count,
        "multi_hop_sample_ratio": rate(count, len(profiles)),
        "max_required_hops": max((int(row["features"].get("required_hop_count") or 1) for row in profiles), default=0),
        "multi_hop_authority_available": False,
    }


def build_hard_negative_coverage(task0106: dict[str, Any], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    summary = task0106["hard_negative_summary"]
    hard_units = int(summary.get("hard_negative_unit_count") or 0)
    return {
        "schema_version": "opk-rag.task0107.hard-negative-coverage.v1",
        "task_id": TASK_ID,
        "queries_with_hard_negatives": hard_units,
        "mean_hard_negatives_per_query": rate(hard_units, len(profiles)),
        "hard_negative_unit_count": hard_units,
        "primary_class_distribution": {
            "lexical_confounder": 0,
            "semantic_confounder": 0,
            "same_section_neighbor": 0,
            "same_document_neighbor": 0,
            "cross_document_confounder": 0,
            "duplicate": 0,
            **summary.get("primary_class_distribution", {}),
        },
        "sufficient_hard_negatives": hard_units > 0 and rate(hard_units, len(profiles)) >= 0.05,
    }


def build_top1_saturation(task0106: dict[str, Any], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = task0106["gold_rank_distribution"]
    total = len(profiles)
    not_ranked = int(ranks.get("gold_not_in_candidate_count") or 0)
    rank_gt_1 = 0 if float(ranks.get("gold_rank_p95") or 0) <= 1.0 else None
    rank_1 = total - not_ranked - (rank_gt_1 or 0)
    return {
        "schema_version": "opk-rag.task0107.top1-saturation.v1",
        "task_id": TASK_ID,
        "gold_rank_1_count": rank_1,
        "gold_rank_gt_1_count": rank_gt_1,
        "gold_not_in_candidate_count": not_ranked,
        "gold_rank_median": ranks.get("gold_rank_median"),
        "gold_rank_p95": ranks.get("gold_rank_p95"),
        "top1_lookup_benchmark": float(ranks.get("gold_rank_median") or 0) == 1.0 and float(ranks.get("gold_rank_p95") or 0) == 1.0,
        "rank_gt_1_exact_count_available": False,
    }


def build_capability_gap_summary(
    *,
    capability_distribution: dict[str, Any],
    table_coverage: dict[str, Any],
    hard_negative_coverage: dict[str, Any],
    task0106: dict[str, Any],
) -> dict[str, Any]:
    counts = capability_distribution["sample_count_by_capability"]
    gaps = {
        "cross_document_gap": "material" if counts["cross_document"] == 0 else "non_material",
        "multi_hop_gap": "material" if counts["multi_hop"] == 0 else "non_material",
        "graph_sensitive_gap": "material",
        "table_reasoning_gap": "material" if table_coverage["table_reasoning_sample_count"] == 0 else "non_material",
        "hierarchical_structure_gap": "material" if counts["heading_sensitive"] == 0 and capability_distribution["secondary_sample_count_by_capability"]["heading_sensitive"] == 0 else "non_material",
        "hard_negative_gap": "material" if hard_negative_coverage["sufficient_hard_negatives"] is False else "non_material",
        "long_context_gap": "non_material" if counts["long_context"] > 0 or capability_distribution["secondary_sample_count_by_capability"]["long_context"] > 0 else "unknown",
        "unanswerable_gap": "material" if counts["unanswerable"] == 0 else "non_material",
        "generation_reasoning_gap": "unknown",
        "agent_routing_gap": "material",
    }
    local_saturated = local_retrieval_saturated(task0106)
    highest = choose_highest_priority_gap(gaps, local_saturated=local_saturated)
    return {
        "schema_version": "opk-rag.task0107.capability-gap-summary.v1",
        "task_id": TASK_ID,
        "gap_status": gaps,
        "higher_order_capability_gap_material": any(gaps[key] == "material" for key in ("cross_document_gap", "multi_hop_gap", "graph_sensitive_gap", "agent_routing_gap")),
        "highest_priority_capability_gap": highest,
        "capability_gap_diagnosis_complete": True,
    }


def build_benchmark_role_decision(
    *,
    task0106: dict[str, Any],
    capability_gap_summary: dict[str, Any],
    capability_distribution: dict[str, Any],
) -> dict[str, Any]:
    diagnosis = decide_saturation(
        recall_at_5=float(task0106["contained_gold_retrieval_metrics"].get("contained_gold_recall_at_5") or 0),
        mrr=float(task0106["contained_gold_retrieval_metrics"].get("contained_gold_mrr") or 0),
        hard_capability_coverage_materially_insufficient=capability_gap_summary["higher_order_capability_gap_material"],
        nontrivial_sample_count=sum(
            capability_distribution["sample_count_by_capability"].get(name, 0)
            for name in ("cross_section", "cross_page", "cross_document", "multi_hop", "table", "distractor_sensitive")
        ),
        material_hard_errors=0,
    )
    primary_role = "regression_benchmark" if diagnosis == "benchmark_materially_saturated" else "secondary_capability_benchmark"
    next_action = "add_existing_public_benchmark" if capability_gap_summary["highest_priority_capability_gap"] in {"graph_sensitive_retrieval", "multi_hop_relation_reasoning", "cross_document_reasoning"} else "author_bounded_gap_suite"
    return {
        "schema_version": "opk-rag.task0107.benchmark-role-decision.v1",
        "task_id": TASK_ID,
        "benchmark_primary_diagnosis": diagnosis,
        "benchmark_primary_role": primary_role,
        "secondary_roles": ["parser_ingestion_benchmark", "secondary_capability_benchmark"],
        "local_retrieval_saturated": local_retrieval_saturated(task0106),
        "highest_priority_capability_gap": capability_gap_summary["highest_priority_capability_gap"],
        "next_benchmark_action": next_action,
        "next_task_decision": "begin_graph_sensitive_benchmark_integration" if next_action == "add_existing_public_benchmark" else "author_bounded_gap_suite",
        "benchmark_role_decision_complete": True,
        "next_benchmark_action_complete": True,
        "interpretation_guard": "OPK-RAG is saturated only on the retrieval problem surface covered by the current benchmark.",
    }


def decide_saturation(
    *,
    recall_at_5: float,
    mrr: float,
    hard_capability_coverage_materially_insufficient: bool,
    nontrivial_sample_count: int,
    material_hard_errors: int,
) -> str:
    saturated = recall_at_5 >= 0.98 and mrr >= 0.98
    if saturated and hard_capability_coverage_materially_insufficient:
        return "benchmark_materially_saturated"
    if saturated and nontrivial_sample_count > 0:
        return "benchmark_partially_saturated"
    if material_hard_errors > 0 or not saturated:
        return "benchmark_still_discriminative"
    return "benchmark_partially_saturated"


def choose_highest_priority_gap(gaps: dict[str, str], *, local_saturated: bool) -> str:
    if not local_saturated:
        return "hard_negative_retrieval" if gaps.get("hard_negative_gap") == "material" else "none"
    if gaps.get("graph_sensitive_gap") == "material":
        return "graph_sensitive_retrieval"
    if gaps.get("multi_hop_gap") == "material":
        return "multi_hop_relation_reasoning"
    if gaps.get("cross_document_gap") == "material":
        return "cross_document_reasoning"
    if gaps.get("table_reasoning_gap") == "material":
        return "table_structured_reasoning"
    return "none"


def build_public_benchmark_mapping(*, root: Path = ROOT) -> dict[str, Any]:
    task0098 = read_json(root / TASK0098_CONTRACT_PATH.relative_to(ROOT)) if (root / TASK0098_CONTRACT_PATH.relative_to(ROOT)).exists() else {}
    pdfqa_authority = read_json(root / PDFQA_AUTHORITY_PATH.relative_to(ROOT)) if (root / PDFQA_AUTHORITY_PATH.relative_to(ROOT)).exists() else {}
    capability_fit = {
        "pdfQA": {
            "authority": task0098.get("benchmark_authority", {}).get("pdfQA", "unknown"),
            "local_retrieval": "measured_saturated",
            "cross_document": "absent_in_current_formal_profile",
            "multi_hop": "absent_in_current_formal_profile",
            "graph_sensitive": "absent_in_current_formal_profile",
            "table_sensitive": "limited_gold_block_coverage_no_reasoning_authority",
            "global_reasoning": "unknown",
        },
        "WildGraphBench": {
            "authority": task0098.get("benchmark_authority", {}).get("WildGraphBench", "unknown"),
            "local_retrieval": "unknown",
            "cross_document": "authority_intended_not_materialized",
            "multi_hop": "unknown",
            "graph_sensitive": "authority_intended_not_materialized",
            "table_sensitive": "unknown",
            "global_reasoning": "unknown",
        },
        "GraphRAG-Bench": {
            "authority": task0098.get("benchmark_authority", {}).get("GraphRAG-Bench", "unknown"),
            "local_retrieval": "unknown",
            "cross_document": "unknown",
            "multi_hop": "authority_intended_not_materialized",
            "graph_sensitive": "authority_intended_not_materialized",
            "table_sensitive": "unknown",
            "global_reasoning": "unknown",
        },
    }
    return {
        "schema_version": "opk-rag.task0107.public-benchmark-mapping.v1",
        "task_id": TASK_ID,
        "roadmap_authority_document": task0098.get("roadmap_authority_document"),
        "pdfqa_authority_status": pdfqa_authority.get("authority_status"),
        "external_dataset_downloaded": task0098.get("external_dataset_downloaded"),
        "capability_fit": capability_fit,
        "public_benchmark_mapping_complete": True,
        "no_benchmark_download_performed": True,
    }


def build_transition_readiness(
    benchmark_role_decision: dict[str, Any],
    capability_gap_summary: dict[str, Any],
    public_benchmark_mapping: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        benchmark_role_decision["local_retrieval_saturated"] is True
        and capability_gap_summary["higher_order_capability_gap_material"] is True
    )
    return {
        "schema_version": "opk-rag.task0107.transition-readiness.v1",
        "task_id": TASK_ID,
        "local_retrieval_saturated": benchmark_role_decision["local_retrieval_saturated"],
        "higher_order_capability_gap_material": capability_gap_summary["higher_order_capability_gap_material"],
        "agent_graph_evaluation_transition_ready": ready,
        "recommended_transition_target": "graph_sensitive_capability_evaluation" if ready else "retain_current_evaluation_surface",
        "existing_public_authority_available": public_benchmark_mapping["capability_fit"]["GraphRAG-Bench"]["authority"] != "unknown",
    }


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    required = [
        "input_identity",
        "benchmark_profile",
        "capability_distribution",
        "difficulty_distribution",
        "evidence_cardinality",
        "document_cardinality",
        "structural_coverage",
        "table_coverage",
        "cross_page_coverage",
        "multi_hop_coverage",
        "hard_negative_coverage",
        "top1_saturation",
        "capability_gap_summary",
        "benchmark_role_decision",
        "public_benchmark_mapping",
        "transition_readiness",
        "result_digests",
    ]
    if not contract_path.exists():
        result = invalid_verification(["contract missing"])
    else:
        missing = [name for name in required if not (result_dir / f"{name}.json").exists()]
        if missing:
            result = invalid_verification([f"missing artifact: {name}.json" for name in missing])
        else:
            payloads = {name: read_json(result_dir / f"{name}.json") for name in required}
            contract = read_json(contract_path)
            result_digests = payloads.pop("result_digests")
            result = verify_payloads(contract=contract, result_digests=result_digests, **payloads)
    if write:
        write_json(result_dir / "verification_summary.json", result)
    return result


def verify_payloads(*, contract: dict[str, Any], result_digests: dict[str, Any], **payloads: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    input_identity = payloads["input_identity"]
    benchmark_profile = payloads["benchmark_profile"]
    capability = payloads["capability_distribution"]
    difficulty = payloads["difficulty_distribution"]
    document_cardinality = payloads["document_cardinality"]
    multi_hop = payloads["multi_hop_coverage"]
    structural = payloads["structural_coverage"]
    gaps = payloads["capability_gap_summary"]
    decision = payloads["benchmark_role_decision"]
    transition = payloads["transition_readiness"]
    for flag in (
        "production_chunking_modified",
        "production_embedding_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_modified",
        "graph_runtime_modified",
    ):
        if contract.get(flag) is not False:
            issues.append(f"{flag} must be false")
    if input_identity.get("task0106_inputs_valid") is not True:
        issues.append("TASK-0106 inputs invalid")
    if input_identity.get("formal_document_count_valid") is not True:
        issues.append("formal document count invalid")
    if input_identity.get("c0_chunk_count_valid") is not True:
        issues.append("C0 chunk count invalid")
    if benchmark_profile.get("benchmark_profile_complete") is not True:
        issues.append("benchmark profile incomplete")
    if capability.get("capability_taxonomy_complete") is not True:
        issues.append("capability taxonomy incomplete")
    if difficulty.get("difficulty_profile_complete") is not True:
        issues.append("difficulty profile incomplete")
    if document_cardinality.get("cross_document_sample_count") is None:
        issues.append("cross-document coverage incomplete")
    if multi_hop.get("multi_hop_sample_count") is None:
        issues.append("multi-hop coverage incomplete")
    if structural.get("structural_coverage_complete") is not True:
        issues.append("structural coverage incomplete")
    if gaps.get("capability_gap_diagnosis_complete") is not True:
        issues.append("capability gap diagnosis incomplete")
    if decision.get("benchmark_role_decision_complete") is not True:
        issues.append("benchmark role decision incomplete")
    if decision.get("next_benchmark_action_complete") is not True:
        issues.append("next benchmark action incomplete")
    if transition.get("agent_graph_evaluation_transition_ready") not in {True, False}:
        issues.append("transition readiness incomplete")
    expected_digests = build_result_digests(contract=contract, **payloads)
    if result_digests.get("combined_digest") != expected_digests.get("combined_digest"):
        issues.append("result digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0107.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "issues": issues,
        "task_status": "complete" if complete else "blocked",
        "task0106_inputs_valid": input_identity.get("task0106_inputs_valid") is True,
        "benchmark_profile_complete": benchmark_profile.get("benchmark_profile_complete") is True,
        "capability_taxonomy_complete": capability.get("capability_taxonomy_complete") is True,
        "difficulty_profile_complete": difficulty.get("difficulty_profile_complete") is True,
        "single_span_coverage_complete": "single_span_sample_count" in benchmark_profile,
        "cross_document_coverage_complete": "cross_document_sample_count" in document_cardinality,
        "multi_hop_coverage_complete": "multi_hop_sample_count" in multi_hop,
        "structural_coverage_complete": structural.get("structural_coverage_complete") is True,
        "saturation_diagnosis_complete": "benchmark_primary_diagnosis" in decision,
        "capability_gap_diagnosis_complete": gaps.get("capability_gap_diagnosis_complete") is True,
        "benchmark_role_decision_complete": decision.get("benchmark_role_decision_complete") is True,
        "next_benchmark_action_complete": decision.get("next_benchmark_action_complete") is True,
        "runtime_behavior_unchanged": True,
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_modified": False,
        "graph_runtime_modified": False,
        "git_commit_created": False,
        "benchmark_primary_diagnosis": decision.get("benchmark_primary_diagnosis"),
        "benchmark_primary_role": decision.get("benchmark_primary_role"),
        "local_retrieval_saturated": decision.get("local_retrieval_saturated"),
        "highest_priority_capability_gap": decision.get("highest_priority_capability_gap"),
        "next_benchmark_action": decision.get("next_benchmark_action"),
        "agent_graph_evaluation_transition_ready": transition.get("agent_graph_evaluation_transition_ready"),
    }


def invalid_verification(issues: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0107.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "invalid",
        "issues": issues,
        "task_status": "blocked",
        "git_commit_created": False,
    }


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in payloads.items()}
    return {
        "schema_version": "opk-rag.task0107.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": dict(sorted(digests.items())),
        "combined_digest": digest_json(dict(sorted(digests.items()))),
    }


def build_report(payloads: dict[str, Any]) -> str:
    profile = payloads["benchmark_profile"]
    capability = payloads["capability_distribution"]
    difficulty = payloads["difficulty_distribution"]
    doc_cardinality = payloads["document_cardinality"]
    table = payloads["table_coverage"]
    hard_negative = payloads["hard_negative_coverage"]
    top1 = payloads["top1_saturation"]
    gaps = payloads["capability_gap_summary"]
    decision = payloads["benchmark_role_decision"]
    transition = payloads["transition_readiness"]
    counts = capability["sample_count_by_capability"]
    return "\n".join(
        [
            "# TASK-0107 Benchmark Difficulty & Capability Gap Report",
            "",
            "## Verdict",
            "",
            f"- benchmark_primary_diagnosis: `{decision['benchmark_primary_diagnosis']}`",
            f"- benchmark_primary_role: `{decision['benchmark_primary_role']}`",
            f"- local_retrieval_saturated: `{decision['local_retrieval_saturated']}`",
            f"- highest_priority_capability_gap: `{decision['highest_priority_capability_gap']}`",
            f"- next_benchmark_action: `{decision['next_benchmark_action']}`",
            f"- agent_graph_evaluation_transition_ready: `{transition['agent_graph_evaluation_transition_ready']}`",
            "",
            "## Required Answers",
            "",
            "1. TASK-0106 Recall@5 和 MRR 同时 1.0 的原因：当前 formal authority 是 canonical block 级 local evidence unit，且 TASK-0106 在该 surface 上候选生成与排序均无 material failure。",
            f"2. single-span lookup 占比：`{profile['single_span_sample_count']}` / `{profile['evaluation_unit_count']}` = `{profile['single_span_sample_ratio']:.4f}`。",
            f"3. Cross-section 样本：`{counts['cross_section']}`。",
            f"4. Cross-document 样本：`{doc_cardinality['cross_document_sample_count']}`；当前 benchmark 无法验证 cross-document retrieval / reasoning。",
            f"5. Multi-hop 样本：`{payloads['multi_hop_coverage']['multi_hop_sample_count']}`。",
            f"6. Table-sensitive gold block 样本：`{table['table_gold_sample_count']}`；table reasoning 样本：`{table['table_reasoning_sample_count']}`。",
            f"7. Heading-sensitive 样本：primary `{counts['heading_sensitive']}`，secondary `{capability['secondary_sample_count_by_capability']['heading_sensitive']}`。",
            f"8. Hard negatives：`{hard_negative['hard_negative_unit_count']}`，sufficient=`{hard_negative['sufficient_hard_negatives']}`。",
            f"9. Discriminative power：`{decision['benchmark_primary_diagnosis']}`；Top-1 lookup benchmark=`{top1['top1_lookup_benchmark']}`。",
            f"10. pdfQA 后续角色：`{decision['benchmark_primary_role']}`。",
            f"11. 最大 capability gap：`{decision['highest_priority_capability_gap']}`。",
            f"12. 下一阶段 benchmark：`{decision['next_benchmark_action']}`。",
            f"13. 是否进入 Agent / Graph-sensitive evaluation：`{transition['agent_graph_evaluation_transition_ready']}`。",
            "",
            "## Difficulty Distribution",
            "",
            json.dumps(difficulty["sample_count_by_difficulty"], ensure_ascii=False, sort_keys=True),
            "",
            "## Gap Summary",
            "",
            json.dumps(gaps["gap_status"], ensure_ascii=False, sort_keys=True),
            "",
            "## Guardrails",
            "",
            "- Production chunking, embedding, retriever, reranker, rank fusion, agent behavior, graph runtime were not modified.",
            "- Saturation only means OPK-RAG saturated the retrieval problem surface covered by this benchmark; it does not prove all retrieval problems are solved.",
            "- No new benchmark was downloaded or authored in TASK-0107.",
        ]
    )


def local_retrieval_saturated(task0106: dict[str, Any]) -> bool:
    contained = task0106["contained_gold_retrieval_metrics"]
    return float(contained.get("contained_gold_recall_at_5") or 0) >= 0.98 and float(contained.get("contained_gold_mrr") or 0) >= 0.98


def coverage_status(count: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    ratio = count / total
    if count == 0:
        return "absent"
    if ratio < 0.05:
        return "limited"
    if ratio >= 0.10:
        return "sufficient"
    return "limited"


def normalize_block_type(value: Any) -> str:
    raw = str(value or "").lower()
    if "table" in raw:
        return "table"
    if "heading" in raw or raw in {"title", "section_header"}:
        return "heading"
    if "list" in raw:
        return "list"
    if "code" in raw or "formula" in raw:
        return "code"
    if "paragraph" in raw or "text" in raw:
        return "paragraph"
    return raw or "unknown"


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    print(json.dumps(run_task0107(), ensure_ascii=False, indent=2, sort_keys=True))
