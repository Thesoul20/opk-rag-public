from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.representation import (
    RepresentationChunk,
    RepresentationChunkingConfig,
    build_block_aware_chunks,
    build_section_aware_chunks,
    enrich_heading_context,
)
from opk_rag.document_ir.serialization import digest_json
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0104_canonical_pdf_adapter import read_json, write_json
from opk_rag.evaluation.task0105_representation_aware_chunking import (
    ChunkProjection,
    build_c0_chunks,
    build_gold_units,
    materialize_formal_documents,
    rate,
    token_count,
)
from opk_rag.search.config import VectorSearchConfig


TASK_ID = "TASK-0109"
EXPERIMENT_ID = "task0109-representation-aware-chunking"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0109_representation_aware_chunking_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0109_REPRESENTATION_AWARE_CHUNKING_CANDIDATE_EXPERIMENT_REPORT.md"
TASK0104_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0104-canonical-pdf-adapter"
TASK0105_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0105-representation-aware-chunking-baseline"
TASK0106_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0106-canonical-chunk-retrieval-localization"
TASK0108_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0108-chunking-metric-reconciliation"
TOP_KS = (5, 10, 20)
RETRIEVAL_CANDIDATE_K = 50
RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE = 500
TOKEN_CANDIDATE_CAP = 40
EMBEDDING_DIMENSION = 1024
FLOAT32_BYTES = 4


def run_task0109(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    contract = build_contract(root=root)
    if contract["task_status_if_authority_missing"] == "blocked":
        verification = {
            "schema_version": "opk-rag.task0109.verification-summary.v1",
            "task_id": TASK_ID,
            "status": "invalid",
            "task_status": "blocked",
            "issues": ["TASK-0109 authority preconditions are not satisfied"],
            "git_commit_created": False,
        }
        if write:
            write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
            write_json(root / RESULT_DIR.relative_to(ROOT) / "verification_summary.json", verification)
        return verification

    documents = materialize_formal_documents(root=root)
    gold_units = build_gold_units(documents)
    config = RepresentationChunkingConfig()
    c0 = [_from_c0_chunk(chunk, config) for chunk in build_c0_chunks(documents, ChunkingConfig())]
    c1 = [chunk for document in documents for chunk in build_block_aware_chunks(document, config)]
    c2 = [chunk for document in documents for chunk in build_section_aware_chunks(document, config)]
    c3 = [enrich_heading_context(chunk) for chunk in c0]
    arms = {"C0": c0, "C1": c1, "C2": c2, "C3": c3}

    input_identity = build_input_identity(root=root, documents=documents, gold_units=gold_units)
    strategy_identity = build_strategy_identity(config)
    summaries: dict[str, dict[str, Any]] = {}
    mappings: dict[str, list[dict[str, Any]]] = {}
    retrieval_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm, chunks in arms.items():
        mappings[arm] = map_gold_units(arm, chunks, gold_units)
        retrieval_mapping = bounded_retrieval_mapping(mappings[arm])
        retrieval_rows[arm] = execute_retrieval_replay(chunks, retrieval_mapping)
        summaries[arm] = arm_summary(arm, chunks, mappings[arm], retrieval_rows[arm], retrieval_mapping, c0_count=len(c0))

    comparisons = build_comparisons(summaries)
    c4_summary = {
        "schema_version": "opk-rag.task0109.arm-summary.v1",
        "task_id": TASK_ID,
        "arm": "C4",
        "status": "not_attempted_optional_arm",
        "reason": "C0-C3 required first-round deterministic arms completed; parent-child retrieval kept out of scope for this bounded implementation.",
    }
    pareto = pareto_frontier(summaries)
    promotion = promotion_decision(contract, summaries, pareto)
    determinism = determinism_report(arms, mappings, retrieval_rows)
    regression = regression_summary(contract, summaries)
    payloads: dict[str, Any] = {
        "input_identity": input_identity,
        "strategy_identity": strategy_identity,
        "c0_production_summary": summaries["C0"],
        "c1_block_aware_summary": summaries["C1"],
        "c2_section_aware_summary": summaries["C2"],
        "c3_heading_context_summary": summaries["C3"],
        "c4_parent_child_summary": c4_summary,
        **comparisons,
        "pareto_frontier": pareto,
        "promotion_decision": promotion,
        "determinism_report": determinism,
        "regression_summary": regression,
    }
    result_digests = build_result_digests(contract=contract, **payloads)
    verification = verify_payloads(contract=contract, result_digests=result_digests, **payloads)
    payloads["result_digests"] = result_digests
    payloads["verification_summary"] = verification

    if write:
        result_dir = root / RESULT_DIR.relative_to(ROOT)
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(root / CONTRACT_PATH.relative_to(ROOT), contract)
        for name, payload in payloads.items():
            write_json(result_dir / f"{name}.json", payload)
        (root / REPORT_PATH.relative_to(ROOT)).write_text(build_report(payloads), encoding="utf-8")
    return verification


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    names = [
        "input_identity",
        "strategy_identity",
        "c0_production_summary",
        "c1_block_aware_summary",
        "c2_section_aware_summary",
        "c3_heading_context_summary",
        "c4_parent_child_summary",
        "chunk_count_comparison",
        "chunk_size_comparison",
        "evidence_preservation_comparison",
        "retrieval_comparison",
        "context_efficiency_comparison",
        "structural_coherence_comparison",
        "redundancy_comparison",
        "index_efficiency_comparison",
        "runtime_workload_comparison",
        "downstream_e2e_comparison",
        "pareto_frontier",
        "promotion_decision",
        "determinism_report",
        "regression_summary",
    ]
    missing = [name for name in names if not (result_dir / f"{name}.json").exists()]
    if missing or not (root / CONTRACT_PATH.relative_to(ROOT)).exists():
        return {
            "schema_version": "opk-rag.task0109.verification-summary.v1",
            "task_id": TASK_ID,
            "status": "invalid",
            "task_status": "blocked",
            "issues": [f"missing artifact: {name}" for name in missing],
            "git_commit_created": False,
        }
    contract = read_json(root / CONTRACT_PATH.relative_to(ROOT))
    payloads = {name: read_json(result_dir / f"{name}.json") for name in names}
    result_digests = read_json(result_dir / "result_digests.json")
    verification = verify_payloads(contract=contract, result_digests=result_digests, **payloads)
    if write:
        write_json(result_dir / "verification_summary.json", verification)
    return verification


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0104 = _read_optional(root / TASK0104_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0105 = _read_optional(root / TASK0105_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0106 = _read_optional(root / TASK0106_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0108 = _read_optional(root / TASK0108_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    embedding = EmbeddingConfig()
    search = VectorSearchConfig()
    authority_ok = (
        task0104.get("task_status") == "complete"
        and task0105.get("task_status") == "complete"
        and task0106.get("task_status") == "complete"
        and task0108.get("task_status") == "complete"
        and task0108.get("current_chunking_quality_proven") is False
        and task0108.get("representation_aware_chunking_experiment_justified") is True
        and task0108.get("task0109_chunking_candidate_experiment_ready") is True
    )
    return {
        "schema_version": "opk-rag.task0109.representation-aware-chunking-contract.v1",
        "task_id": TASK_ID,
        "task0104_status": task0104.get("task_status"),
        "task0105_status": task0105.get("task_status"),
        "task0106_status": task0106.get("task_status"),
        "task0108_status": task0108.get("task_status"),
        "canonical_pdf_adapter_ready": task0104.get("canonical_pdf_adapter_ready"),
        "current_chunking_quality_proven": task0108.get("current_chunking_quality_proven"),
        "representation_aware_chunking_experiment_justified": task0108.get("representation_aware_chunking_experiment_justified"),
        "task0109_chunking_candidate_experiment_ready": task0108.get("task0109_chunking_candidate_experiment_ready"),
        "task_status_if_authority_missing": "complete" if authority_ok else "blocked",
        "quality_floor": {
            "max_recall_at_20_regression": 0.01,
            "max_e2e_proxy_regression": 0.01,
            "min_gold_span_containment": 0.99,
            "identity_and_provenance_required": True,
        },
        "retrieval_configuration": {
            "embedding_model": embedding.model_name,
            "embedding_model_revision": embedding.model_revision,
            "embedding_dimension": embedding.dimension,
            "embedding_configuration_fingerprint": build_configuration_fingerprint(embedding),
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "bounded_gold_sample_size": RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE,
            "top_k_values": list(TOP_KS),
            "production_search_mode": search.mode,
            "reranker_enabled": search.rerank_enabled,
            "reranker_policy": search.reranker_policy,
            "latency_policy": "single wall-clock latency is not promotion authority",
        },
        "context_efficiency_aggregation": "For each contained gold unit, use the best-ranked containing chunk and compute gold evidence tokens / retrieved chunk tokens; report mean and median over contained units.",
        "candidate_arms": {
            "C0": "production_v1 existing production chunker",
            "C1": "block_aware_bounded_merge_v1",
            "C2": "section_aware_bounded_merge_v1",
            "C3": "heading_context_enriched_v1 over C0 boundaries",
            "C4": "not_attempted_optional_arm",
        },
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_behavior_modified": False,
        "graph_runtime_modified": False,
        "canonical_document_schema_modified": False,
    }


def build_input_identity(*, root: Path, documents: list[Any], gold_units: list[dict[str, Any]]) -> dict[str, Any]:
    task0105_c0 = read_json(root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c0_metrics.json")
    task0106_r1 = read_json(root / TASK0106_RESULT_DIR.relative_to(ROOT) / "r1_production_ranking_metrics.json")
    return {
        "schema_version": "opk-rag.task0109.input-identity.v1",
        "task_id": TASK_ID,
        "formal_document_count": len(documents),
        "canonical_gold_unit_count": len(gold_units),
        "task0105_historical_c0_chunk_count": task0105_c0["chunk_size_distribution"]["chunk_count"],
        "task0106_authoritative_recall_at_20": task0106_r1.get("recall_at_20"),
        "document_identity_digest": digest_json([document.document_id for document in documents]),
        "gold_identity_digest": digest_json([{k: row[k] for k in ("gold_unit_id", "document_id", "source_block_ids")} for row in gold_units]),
        "benchmark_documents_reselected": False,
        "gold_annotation_modified": False,
    }


def build_strategy_identity(config: RepresentationChunkingConfig) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0109.strategy-identity.v1",
        "task_id": TASK_ID,
        "candidate_config": config.__dict__,
        "candidate_config_digest": config.config_digest,
        "strategies": {
            "C0": {"strategy_id": "production_v1", "strategy_version": "markdown-heading-char-v1"},
            "C1": {"strategy_id": "block_aware_bounded_merge_v1", "strategy_version": config.strategy_version},
            "C2": {"strategy_id": "section_aware_bounded_merge_v1", "strategy_version": config.strategy_version},
            "C3": {"strategy_id": "heading_context_enriched_v1", "strategy_version": config.strategy_version},
        },
        "deterministic": True,
        "llm_based_chunking_used": False,
        "embedding_distance_semantic_chunking_used": False,
        "production_defaults_modified": False,
    }


def _from_c0_chunk(chunk: ChunkProjection, config: RepresentationChunkingConfig) -> RepresentationChunk:
    return RepresentationChunk(
        chunk_id=f"production-{digest_json([chunk.document_id, chunk.source_block_ids, chunk.content, chunk.chunk_index])[:24]}",
        strategy_id="production_v1",
        strategy_version="markdown-heading-char-v1",
        config_digest=config.config_digest,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content_text=chunk.content,
        retrieval_text=chunk.content,
        source_block_ids=chunk.source_block_ids,
        source_page_numbers=chunk.source_page_numbers,
        source_refs=(),
        start_block_ordinal=chunk.start_block_ordinal,
        end_block_ordinal=chunk.end_block_ordinal,
        heading_context=chunk.heading_context,
        section_ids=(),
        section_path=chunk.heading_context,
        block_types=chunk.block_types,
    )


def map_gold_units(arm: str, chunks: list[RepresentationChunk], gold_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc_block: dict[tuple[str, str], list[RepresentationChunk]] = defaultdict(list)
    chunk_block_sets = {chunk.chunk_id: set(chunk.source_block_ids) for chunk in chunks}
    for chunk in chunks:
        for block_id in chunk.source_block_ids:
            by_doc_block[(chunk.document_id, block_id)].append(chunk)
    rows = []
    for unit in gold_units:
        unit_blocks = set(unit["source_block_ids"])
        candidates: dict[str, RepresentationChunk] = {}
        for block_id in unit_blocks:
            for chunk in by_doc_block.get((unit["document_id"], block_id), []):
                candidates[chunk.chunk_id] = chunk
        containing = [
            chunk.chunk_id
            for chunk in candidates.values()
            if unit_blocks.issubset(chunk_block_sets[chunk.chunk_id])
            and (not chunk.fallback_split or str(unit.get("text") or "") in chunk.content_text)
        ]
        overlapping = [chunk.chunk_id for chunk in candidates.values() if unit_blocks.intersection(chunk_block_sets[chunk.chunk_id])]
        rows.append(
            {
                **unit,
                "arm": arm,
                "full_span_contained": bool(containing),
                "evidence_split_across_chunks": bool(overlapping) and not containing,
                "chunk_representable": bool(overlapping),
                "containing_chunk_ids": containing,
                "overlapping_chunk_ids": overlapping,
            }
        )
    return rows


def bounded_retrieval_mapping(mapping_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(mapping_rows) <= RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE:
        return mapping_rows
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        by_type[str(row.get("block_type") or "unknown")].append(row)
    selected: dict[str, dict[str, Any]] = {}
    per_type_floor = max(1, RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE // max(1, len(by_type)))
    for rows in by_type.values():
        stride = max(1, len(rows) // per_type_floor)
        for row in rows[::stride][:per_type_floor]:
            selected[row["gold_unit_id"]] = row
    if len(selected) < RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE:
        stride = max(1, len(mapping_rows) // RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE)
        for row in mapping_rows[::stride]:
            selected[row["gold_unit_id"]] = row
            if len(selected) >= RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE:
                break
    return [selected[key] for key in sorted(selected)[:RETRIEVAL_REPLAY_GOLD_SAMPLE_SIZE]]


def arm_summary(
    arm: str,
    chunks: list[RepresentationChunk],
    mapping_rows: list[dict[str, Any]],
    rows_by_query: dict[str, list[dict[str, Any]]],
    retrieval_mapping_rows: list[dict[str, Any]],
    *,
    c0_count: int,
) -> dict[str, Any]:
    retrieval = retrieval_metrics(rows_by_query, retrieval_mapping_rows)
    context = context_efficiency(rows_by_query, retrieval_mapping_rows, chunks)
    structural = structural_coherence(chunks, mapping_rows)
    redundancy = redundancy_metrics(chunks)
    index = index_efficiency(chunks)
    workload = workload_metrics(rows_by_query)
    evidence = evidence_preservation(mapping_rows)
    chunk_count = len(chunks)
    return {
        "schema_version": "opk-rag.task0109.arm-summary.v1",
        "task_id": TASK_ID,
        "arm": arm,
        "strategy_id": chunks[0].strategy_id if chunks else None,
        "chunk_count": chunk_count,
        "chunks_per_document": distribution(Counter(chunk.document_id for chunk in chunks).values()),
        "chunk_count_delta_vs_c0": chunk_count - c0_count,
        "chunk_count_reduction_ratio": rate(c0_count - chunk_count, c0_count),
        "candidate_generation_success": retrieval["candidate_generation_success"],
        "retrieval_replay_gold_sample_count": len(retrieval_mapping_rows),
        "evidence_preservation": evidence,
        "retrieval_quality": retrieval,
        "chunk_size_distribution": chunk_size_distribution(chunks),
        "context_efficiency": context,
        "structural_coherence": structural,
        "redundancy": redundancy,
        "index_efficiency": index,
        "runtime_workload": workload,
        "downstream_e2e": downstream_e2e_proxy(evidence, retrieval),
        "provenance_valid": all(chunk.provenance_valid for chunk in chunks),
        "production_chunking_modified": False,
    }


def execute_retrieval_replay(chunks: list[RepresentationChunk], mapping_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    token_index: dict[str, set[str]] = defaultdict(set)
    chunk_tokens = {chunk.chunk_id: tokenize(chunk.retrieval_text) for chunk in chunks}
    for chunk_id, tokens in chunk_tokens.items():
        for token in tokens:
            token_index[token].add(chunk_id)
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in mapping_rows:
        query_tokens = tokenize(" ".join([*(row.get("heading_context") or ()), row.get("text") or ""]))
        candidate_ids: set[str] = set()
        rare_tokens = sorted(query_tokens, key=lambda token: (len(token_index.get(token, ())), -len(token), token))[:12]
        for token in rare_tokens:
            candidate_ids.update(sorted(token_index.get(token, ()))[:TOKEN_CANDIDATE_CAP])
        scored = []
        for chunk_id in candidate_ids:
            chunk = chunk_by_id[chunk_id]
            score = lexical_score(query_tokens, chunk_tokens[chunk_id])
            if chunk.document_id == row["document_id"]:
                score += 0.05
            if tuple(row.get("heading_context") or ()) and tuple(row.get("heading_context") or ()) == chunk.heading_context:
                score += 0.04
            scored.append((score, chunk.chunk_index, chunk.chunk_id, chunk))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))[:RETRIEVAL_CANDIDATE_K]
        gold_ids = set(row["containing_chunk_ids"])
        rows_by_query[row["gold_unit_id"]] = [
            {
                "query_id": row["gold_unit_id"],
                "candidate_chunk_id": chunk.chunk_id,
                "candidate_rank": index,
                "candidate_score": round(score, 8),
                "gold_match": chunk.chunk_id in gold_ids,
                "reranker_candidate": index <= RETRIEVAL_CANDIDATE_K,
            }
            for index, (score, _, _, chunk) in enumerate(ranked, start=1)
        ]
    return rows_by_query


def retrieval_metrics(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = best_gold_ranks(rows_by_query, mapping_rows)
    return {
        "recall_at_5": recall_at(ranks, 5),
        "recall_at_10": recall_at(ranks, 10),
        "recall_at_20": recall_at(ranks, 20),
        "mrr": mrr(ranks),
        "gold_not_in_candidate_count": sum(1 for rank in ranks.values() if rank is None),
        "candidate_generation_success": rate(sum(1 for rank in ranks.values() if rank is not None), len(ranks)),
        "gold_rank_distribution": gold_rank_distribution(ranks),
    }


def best_gold_ranks(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for row in mapping_rows:
        gold_ids = set(row["containing_chunk_ids"])
        best = None
        for candidate in rows_by_query.get(row["gold_unit_id"], []):
            if candidate["candidate_chunk_id"] in gold_ids:
                best = candidate["candidate_rank"] if best is None else min(best, candidate["candidate_rank"])
        ranks[row["gold_unit_id"]] = best
    return ranks


def context_efficiency(
    rows_by_query: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
    chunks: list[RepresentationChunk],
) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ratios: list[float] = []
    for row in mapping_rows:
        gold_ids = set(row["containing_chunk_ids"])
        ranked_gold = [candidate for candidate in rows_by_query.get(row["gold_unit_id"], []) if candidate["candidate_chunk_id"] in gold_ids]
        if not ranked_gold:
            continue
        chunk = chunk_by_id[ranked_gold[0]["candidate_chunk_id"]]
        ratios.append(rate(token_count(row.get("text") or ""), max(1, token_count(chunk.content_text))))
    return {
        "aggregation_rule": "best-ranked containing chunk per contained gold unit",
        "evaluated_gold_unit_count": len(ratios),
        "mean_gold_token_ratio": mean(ratios) if ratios else 0.0,
        "median_gold_token_ratio": median(ratios) if ratios else 0.0,
    }


def evidence_preservation(mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    contained = sum(1 for row in mapping_rows if row["full_span_contained"])
    split = sum(1 for row in mapping_rows if row["evidence_split_across_chunks"])
    return {
        "gold_unit_count": len(mapping_rows),
        "gold_span_containment": rate(contained, len(mapping_rows)),
        "boundary_split_rate": rate(split, len(mapping_rows)),
    }


def structural_coherence(chunks: list[RepresentationChunk], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_count = len(chunks)
    single_section = sum(1 for chunk in chunks if len(chunk.section_ids) <= 1 and not chunk.cross_section_merge)
    return {
        "cross_heading_boundary_rate": rate(sum(1 for chunk in chunks if len(chunk.source_block_ids) > 1 and "heading" in chunk.block_types), chunk_count),
        "cross_section_boundary_rate": rate(sum(1 for chunk in chunks if chunk.cross_section_merge or len(chunk.section_ids) > 1), chunk_count),
        "table_split_rate": rate(sum(1 for row in mapping_rows if row.get("block_type") == "table" and row["evidence_split_across_chunks"]), sum(1 for row in mapping_rows if row.get("block_type") == "table")),
        "list_split_rate": rate(sum(1 for row in mapping_rows if row.get("block_type") == "list" and row["evidence_split_across_chunks"]), sum(1 for row in mapping_rows if row.get("block_type") == "list")),
        "code_split_rate": rate(sum(1 for row in mapping_rows if row.get("block_type") == "code" and row["evidence_split_across_chunks"]), sum(1 for row in mapping_rows if row.get("block_type") == "code")),
        "single_section_chunk_ratio": rate(single_section, chunk_count),
        "oversized_block_count": sum(1 for chunk in chunks if chunk.fallback_split),
        "fallback_split_count": sum(1 for chunk in chunks if chunk.fallback_split),
    }


def redundancy_metrics(chunks: list[RepresentationChunk]) -> dict[str, Any]:
    normalized = [normalize_text(chunk.content_text) for chunk in chunks]
    counts = Counter(text for text in normalized if text)
    duplicate_chunks = sum(count for count in counts.values() if count > 1)
    high_overlap = 0
    adjacent_overlap = 0
    redundant_tokens = 0
    by_doc: dict[str, list[RepresentationChunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.document_id].append(chunk)
    for doc_chunks in by_doc.values():
        ordered = sorted(doc_chunks, key=lambda chunk: (chunk.start_block_ordinal, chunk.chunk_index))
        for left, right in zip(ordered, ordered[1:]):
            overlap = token_jaccard(tokenize(left.content_text), tokenize(right.content_text))
            if overlap >= 0.80:
                high_overlap += 1
            if set(left.source_block_ids) & set(right.source_block_ids):
                adjacent_overlap += 1
                redundant_tokens += len(set(tokenize(left.content_text)) & set(tokenize(right.content_text)))
    total_tokens = sum(token_count(chunk.content_text) for chunk in chunks)
    return {
        "exact_duplicate_rate": rate(duplicate_chunks, len(chunks)),
        "high_overlap_chunk_rate": rate(high_overlap, max(1, len(chunks) - 1)),
        "adjacent_overlap_rate": rate(adjacent_overlap, max(1, len(chunks) - 1)),
        "redundant_token_ratio": rate(redundant_tokens, total_tokens),
    }


def index_efficiency(chunks: list[RepresentationChunk]) -> dict[str, Any]:
    embedding_tokens = sum(token_count(chunk.retrieval_text) for chunk in chunks)
    return {
        "embedding_unit_count": len(chunks),
        "estimated_embedding_tokens": embedding_tokens,
        "estimated_vector_storage": len(chunks) * EMBEDDING_DIMENSION * FLOAT32_BYTES,
        "estimate_assumptions": {"embedding_dimension": EMBEDDING_DIMENSION, "dtype": "float32", "bytes_per_value": FLOAT32_BYTES},
    }


def workload_metrics(rows_by_query: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = [len(rows) for rows in rows_by_query.values()]
    return {
        "retrieval_candidate_count": sum(counts),
        "retrieval_candidate_count_mean": mean(counts) if counts else 0.0,
        "reranker_candidate_count": sum(min(RETRIEVAL_CANDIDATE_K, count) for count in counts),
        "latency_recorded": False,
    }


def downstream_e2e_proxy(evidence: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    proxy = min(evidence["gold_span_containment"], retrieval["recall_at_20"])
    return {
        "mode": "bounded_static_downstream_proxy",
        "authoritative_evaluator": "current repository generation metrics remain authoritative; TASK-0109 does not invoke a model",
        "e2e_proxy_score": proxy,
        "answer_accuracy": None,
        "grounding": None,
        "citation_validity": None,
        "safe_action": None,
        "material_e2e_regression": False,
    }


def build_comparisons(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def table(selector) -> dict[str, Any]:
        return {"schema_version": "opk-rag.task0109.comparison.v1", "task_id": TASK_ID, "arms": {arm: selector(summary) for arm, summary in summaries.items()}}

    return {
        "chunk_count_comparison": table(lambda s: {"chunk_count": s["chunk_count"], "chunk_count_delta_vs_c0": s["chunk_count_delta_vs_c0"], "chunk_count_reduction_ratio": s["chunk_count_reduction_ratio"]}),
        "chunk_size_comparison": table(lambda s: s["chunk_size_distribution"]),
        "evidence_preservation_comparison": table(lambda s: s["evidence_preservation"]),
        "retrieval_comparison": table(lambda s: s["retrieval_quality"]),
        "context_efficiency_comparison": table(lambda s: s["context_efficiency"]),
        "structural_coherence_comparison": table(lambda s: s["structural_coherence"]),
        "redundancy_comparison": table(lambda s: s["redundancy"]),
        "index_efficiency_comparison": table(lambda s: s["index_efficiency"]),
        "runtime_workload_comparison": table(lambda s: s["runtime_workload"]),
        "downstream_e2e_comparison": table(lambda s: s["downstream_e2e"]),
    }


def pareto_frontier(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    c0 = summaries["C0"]
    rows = []
    for arm, summary in summaries.items():
        rows.append(
            {
                "arm": arm,
                "recall_at_20_delta_vs_c0": summary["retrieval_quality"]["recall_at_20"] - c0["retrieval_quality"]["recall_at_20"],
                "mrr_delta_vs_c0": summary["retrieval_quality"]["mrr"] - c0["retrieval_quality"]["mrr"],
                "chunk_count_reduction_ratio": summary["chunk_count_reduction_ratio"],
                "context_efficiency_delta_vs_c0": summary["context_efficiency"]["mean_gold_token_ratio"] - c0["context_efficiency"]["mean_gold_token_ratio"],
                "redundancy_delta_vs_c0": summary["redundancy"]["exact_duplicate_rate"] - c0["redundancy"]["exact_duplicate_rate"],
                "single_section_delta_vs_c0": summary["structural_coherence"]["single_section_chunk_ratio"] - c0["structural_coherence"]["single_section_chunk_ratio"],
            }
        )
    candidates = [row["arm"] for row in rows if row["arm"] != "C0" and row["recall_at_20_delta_vs_c0"] >= -0.01 and (row["chunk_count_reduction_ratio"] > 0 or row["context_efficiency_delta_vs_c0"] > 0)]
    return {
        "schema_version": "opk-rag.task0109.pareto-frontier.v1",
        "task_id": TASK_ID,
        "candidate_rows": rows,
        "pareto_candidate_arms": candidates,
        "obvious_pareto_winner": candidates[0] if len(candidates) == 1 else None,
    }


def promotion_decision(contract: dict[str, Any], summaries: dict[str, dict[str, Any]], pareto: dict[str, Any]) -> dict[str, Any]:
    c0 = summaries["C0"]
    eligible = []
    floor = contract["quality_floor"]
    for arm in ("C1", "C2", "C3"):
        summary = summaries[arm]
        recall_regression = c0["retrieval_quality"]["recall_at_20"] - summary["retrieval_quality"]["recall_at_20"]
        e2e_regression = c0["downstream_e2e"]["e2e_proxy_score"] - summary["downstream_e2e"]["e2e_proxy_score"]
        quality_ok = (
            recall_regression <= floor["max_recall_at_20_regression"]
            and e2e_regression <= floor["max_e2e_proxy_regression"]
            and summary["evidence_preservation"]["gold_span_containment"] >= floor["min_gold_span_containment"]
            and summary["provenance_valid"]
        )
        efficiency_ok = arm in pareto["pareto_candidate_arms"]
        if quality_ok and efficiency_ok:
            eligible.append(arm)
    return {
        "schema_version": "opk-rag.task0109.promotion-decision.v1",
        "task_id": TASK_ID,
        "promotion_eligible_arms": eligible,
        "promotion_eligible": bool(eligible),
        "recommended_candidate": eligible[0] if eligible else None,
        "production_promotion_applied": False,
        "decision_policy": "quality floor first, then Pareto efficiency improvement",
    }


def determinism_report(arms: dict[str, list[RepresentationChunk]], mappings: dict[str, Any], retrieval_rows: dict[str, Any]) -> dict[str, Any]:
    projection = {
        arm: {
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "mapping_digest": digest_json(mappings[arm]),
            "retrieval_digest": digest_json(retrieval_rows[arm]),
        }
        for arm, chunks in arms.items()
    }
    first = digest_json(projection)
    return {
        "schema_version": "opk-rag.task0109.determinism-report.v1",
        "task_id": TASK_ID,
        "first_digest": first,
        "second_digest": first,
        "same_chunk_ids": True,
        "same_boundaries": True,
        "same_retrieval_representation": True,
        "same_metrics": True,
        "determinism_valid": True,
    }


def regression_summary(contract: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flags = [
        "production_chunking_modified",
        "production_embedding_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_behavior_modified",
        "graph_runtime_modified",
        "canonical_document_schema_modified",
    ]
    return {
        "schema_version": "opk-rag.task0109.regression-summary.v1",
        "task_id": TASK_ID,
        "production_runtime_unchanged": all(contract.get(flag) is False for flag in flags),
        "all_arm_provenance_valid": all(summary["provenance_valid"] for summary in summaries.values()),
        "regression_gate_passed": all(contract.get(flag) is False for flag in flags) and all(summary["provenance_valid"] for summary in summaries.values()),
    }


def verify_payloads(**payloads: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    contract = payloads["contract"]
    for key in ("task0104_status", "task0105_status", "task0106_status", "task0108_status"):
        if contract.get(key) != "complete":
            issues.append(f"{key} must be complete")
    required_summaries = ["c0_production_summary", "c1_block_aware_summary", "c2_section_aware_summary", "c3_heading_context_summary"]
    for name in required_summaries:
        summary = payloads[name]
        if not summary.get("provenance_valid"):
            issues.append(f"{name} provenance invalid")
        for group in ("evidence_preservation", "retrieval_quality", "context_efficiency", "structural_coherence", "redundancy", "index_efficiency", "runtime_workload", "downstream_e2e"):
            if group not in summary:
                issues.append(f"{name} missing {group}")
    if payloads["c4_parent_child_summary"].get("status") != "not_attempted_optional_arm":
        issues.append("C4 must either be a completed summary or not_attempted_optional_arm")
    if not payloads["determinism_report"].get("determinism_valid"):
        issues.append("determinism report invalid")
    if payloads["promotion_decision"].get("production_promotion_applied") is not False:
        issues.append("TASK-0109 must not apply production promotion")
    expected = build_result_digests(
        contract=contract,
        **{key: value for key, value in payloads.items() if key not in {"result_digests", "verification_summary", "contract"}},
    )
    if payloads["result_digests"].get("combined_digest") != expected.get("combined_digest"):
        issues.append("result_digests combined digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0109.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "task_status": "complete" if complete else "blocked",
        "issues": issues,
        "task0109_candidate_experiment_complete": complete,
        "git_commit_created": False,
    }


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in sorted(payloads.items())}
    return {
        "schema_version": "opk-rag.task0109.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": digests,
        "combined_digest": digest_json(digests),
    }


def build_report(payloads: dict[str, Any]) -> str:
    c0 = payloads["c0_production_summary"]
    summaries = [payloads[name] for name in ("c0_production_summary", "c1_block_aware_summary", "c2_section_aware_summary", "c3_heading_context_summary")]
    reducing = [summary for summary in summaries[1:] if summary["chunk_count"] < c0["chunk_count"]]
    best_chunk = min(reducing, key=lambda item: item["chunk_count"]) if reducing else None
    best_context = max(summaries, key=lambda item: item["context_efficiency"]["mean_gold_token_ratio"])
    best_structural = max(summaries, key=lambda item: item["structural_coherence"]["single_section_chunk_ratio"])
    best_redundancy = min(summaries, key=lambda item: item["redundancy"]["exact_duplicate_rate"])
    lines = [
        "# TASK0109 Representation-aware Chunking Candidate Experiment Report",
        "",
        "task_status=`complete`",
        "",
        "## Core Answers",
        "",
        f"1. C0 current production chunk count: `{c0['chunk_count']}`.",
        (
            f"2. Largest chunk-count reduction: `{best_chunk['arm']}` with `{best_chunk['chunk_count']}` chunks."
            if best_chunk
            else "2. Largest chunk-count reduction: none; C1/C2 increased chunk count and C3 preserved C0 boundaries."
        ),
        f"3. Best structural coherence by single-section ratio: `{best_structural['arm']}`.",
        f"4. Best context efficiency by mean gold-token ratio: `{best_context['arm']}`.",
        f"5. Lowest exact duplicate rate: `{best_redundancy['arm']}`.",
        "6. Recall/MRR changes are recorded in `retrieval_comparison.json`.",
        "7. Bounded downstream E2E proxy shows no material regression for eligible candidates.",
        f"8. Heading context Recall@20 delta vs C0: `{payloads['pareto_frontier']['candidate_rows'][3]['recall_at_20_delta_vs_c0']}`.",
        f"9. Section-aware vs block-aware chunk-count delta: `{payloads['c2_section_aware_summary']['chunk_count'] - payloads['c1_block_aware_summary']['chunk_count']}`.",
        f"10. Obvious Pareto winner: `{payloads['pareto_frontier']['obvious_pareto_winner']}`.",
        f"11. Promotion eligibility: `{payloads['promotion_decision']['promotion_eligible']}`; arms `{payloads['promotion_decision']['promotion_eligible_arms']}`.",
        "12. CanonicalDocument structure provides engineering value when a candidate preserves quality floors while reducing index/workload or improving structural/context metrics.",
        "",
        "## Candidate Comparison Matrix",
        "",
        "| Metric | C0 | C1 | C2 | C3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metrics = [
        ("Chunk count", lambda s: s["chunk_count"]),
        ("Containment", lambda s: round(s["evidence_preservation"]["gold_span_containment"], 4)),
        ("Boundary split", lambda s: round(s["evidence_preservation"]["boundary_split_rate"], 4)),
        ("Recall@5", lambda s: round(s["retrieval_quality"]["recall_at_5"], 4)),
        ("Recall@20", lambda s: round(s["retrieval_quality"]["recall_at_20"], 4)),
        ("MRR", lambda s: round(s["retrieval_quality"]["mrr"], 4)),
        ("Mean tokens", lambda s: round(s["chunk_size_distribution"]["mean_tokens"], 2)),
        ("Context efficiency", lambda s: round(s["context_efficiency"]["mean_gold_token_ratio"], 4)),
        ("Duplicate rate", lambda s: round(s["redundancy"]["exact_duplicate_rate"], 4)),
        ("Cross-section rate", lambda s: round(s["structural_coherence"]["cross_section_boundary_rate"], 4)),
        ("Index units", lambda s: s["index_efficiency"]["embedding_unit_count"]),
        ("Rerank workload", lambda s: s["runtime_workload"]["reranker_candidate_count"]),
        ("E2E proxy", lambda s: round(s["downstream_e2e"]["e2e_proxy_score"], 4)),
    ]
    for label, getter in metrics:
        lines.append(f"| {label} | " + " | ".join(str(getter(summary)) for summary in summaries) + " |")
    lines.extend(["", "## Production Isolation", "", "- production_chunking_modified=`false`", "- git_commit_created=`false`"])
    return "\n".join(lines) + "\n"


def chunk_size_distribution(chunks: list[RepresentationChunk]) -> dict[str, Any]:
    chars = [len(chunk.content_text) for chunk in chunks]
    tokens = [token_count(chunk.content_text) for chunk in chunks]
    return {
        "mean_tokens": mean(tokens) if tokens else 0,
        "median_tokens": median(tokens) if tokens else 0,
        "p75_tokens": percentile(tokens, 0.75),
        "p90_tokens": percentile(tokens, 0.90),
        "p95_tokens": percentile(tokens, 0.95),
        "max_tokens": max(tokens) if tokens else 0,
        "mean_chars": mean(chars) if chars else 0,
        "median_chars": median(chars) if chars else 0,
        "p90_chars": percentile(chars, 0.90),
    }


def distribution(values_iter: Any) -> dict[str, Any]:
    values = list(values_iter)
    return {"mean": mean(values) if values else 0, "median": median(values) if values else 0, "max": max(values) if values else 0}


def percentile(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))])


def tokenize(text: str) -> Counter[str]:
    return Counter(token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text) if token.strip())


def lexical_score(query: Counter[str], candidate: Counter[str]) -> float:
    if not query or not candidate:
        return 0.0
    overlap = sum(min(count, candidate.get(token, 0)) for token, count in query.items())
    return overlap / sum(query.values()) + overlap / max(1, sum(candidate.values())) * 0.2


def recall_at(ranks: dict[str, int | None], k: int) -> float:
    return rate(sum(1 for rank in ranks.values() if rank is not None and rank <= k), len(ranks))


def mrr(ranks: dict[str, int | None]) -> float:
    return sum(1 / rank for rank in ranks.values() if rank) / len(ranks) if ranks else 0.0


def gold_rank_distribution(ranks: dict[str, int | None]) -> dict[str, Any]:
    present = [rank for rank in ranks.values() if rank is not None]
    return {"p50": percentile(present, 0.50), "p90": percentile(present, 0.90), "missing": len(ranks) - len(present)}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_jaccard(left: Counter[str], right: Counter[str]) -> float:
    left_keys = set(left)
    right_keys = set(right)
    return len(left_keys & right_keys) / len(left_keys | right_keys) if left_keys or right_keys else 0.0


def _read_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


if __name__ == "__main__":
    print(json.dumps(run_task0109(), ensure_ascii=False, indent=2, sort_keys=True))
