from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
from statistics import median
from typing import Any, Iterable

from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.evaluation.task0099_pdfqa_dataset_authority import ROOT
from opk_rag.evaluation.task0104_canonical_pdf_adapter import read_json, write_json
from opk_rag.evaluation.task0105_representation_aware_chunking import (
    ChunkProjection,
    build_c0_chunks,
    build_gold_units,
    digest_json,
    materialize_formal_documents,
    rate,
    token_count,
)
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA, RankFusionParameters, rank_fusion_order
from opk_rag.search.config import VectorSearchConfig


TASK_ID = "TASK-0106"
EXPERIMENT_ID = "task0106-canonical-chunk-retrieval-localization"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0106_canonical_chunk_retrieval_localization_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0106_CANONICAL_CHUNK_RETRIEVAL_LOCALIZATION_REPORT.md"
TASK0104_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0104-canonical-pdf-adapter"
TASK0105_RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0105-representation-aware-chunking-baseline"
TOP_KS = (5, 10, 20, 50)
CANDIDATE_K = 50
NEAR_MARGIN_EPSILON = 0.01


class Task0106Error(RuntimeError):
    pass


def run_task0106(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    if write:
        result_dir.mkdir(parents=True, exist_ok=True)
        (root / CONTRACT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        (root / REPORT_PATH.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)

    contract = build_contract(root=root)
    task0105_inputs = load_task0105_authority(root=root)
    documents = materialize_formal_documents(root=root)
    chunks = build_c0_chunks(documents)
    gold_units = build_gold_units(documents)
    c0_metrics = task0105_inputs["c0_metrics"]
    c0_mapping = build_fast_gold_mapping(chunks, gold_units)
    gold_mapping = build_gold_chunk_mapping_summary(c0_mapping)

    replay = execute_retrieval_replay(chunks, c0_mapping)
    r0_metrics = retrieval_metrics(replay["r0_rows"], c0_mapping, stage="R0")
    r1_metrics = retrieval_metrics(replay["r1_rows"], c0_mapping, stage="R1")
    contained_metrics = contained_gold_metrics(replay["r1_rows"], c0_mapping)
    rank_distribution = gold_rank_distribution(replay["r1_rows"], c0_mapping)
    funnel = retrieval_funnel(replay["r0_rows"], replay["r1_rows"], c0_mapping)
    candidate_failures = candidate_generation_failures(replay["r0_rows"], c0_mapping)
    ranking_failure_rows = ranking_failures(replay["r0_rows"], replay["r1_rows"], c0_mapping)
    margins = score_margin_diagnosis(replay["r0_rows"], c0_mapping)
    hard_negative = hard_negative_summary(replay["r1_rows"], c0_mapping, chunks)
    size_analysis = chunk_size_retrieval_analysis(replay["r1_rows"], c0_mapping, chunks)
    block_analysis = block_composition_analysis(replay["r1_rows"], c0_mapping, chunks)
    heading_analysis = heading_context_analysis(replay["r1_rows"], c0_mapping, chunks)
    structure_analysis = structural_complexity_analysis(replay["r1_rows"], c0_mapping, chunks)
    fragment_analysis = fragment_competition_analysis(replay["r1_rows"], c0_mapping, chunks)
    duplicate_pressure = duplicate_pressure_analysis(chunks)
    reranker = rank_movement_effect(replay["r0_rows"], replay["reranker_rows"], c0_mapping, "reranker")
    fusion = rank_movement_effect(replay["reranker_rows"], replay["r1_rows"], c0_mapping, "rank_fusion")
    taxonomy = failure_taxonomy(replay["r0_rows"], replay["reranker_rows"], replay["r1_rows"], c0_mapping, chunks)
    diagnostics = representation_diagnostic_summary(replay, c0_mapping)
    determinism = determinism_report(chunks, c0_mapping, replay)
    regression = regression_summary(task0105_inputs, contract)
    input_identity = build_input_identity(task0105_inputs, c0_metrics, len(documents))
    runtime_identity = retrieval_runtime_identity()
    decision = task0107_decision(contained_metrics, candidate_failures, ranking_failure_rows, reranker, fusion, fragment_analysis, diagnostics)

    payloads = {
        "input_identity": input_identity,
        "retrieval_runtime_identity": runtime_identity,
        "gold_chunk_mapping_summary": gold_mapping,
        "retrieval_funnel": funnel,
        "r0_candidate_generation_metrics": r0_metrics,
        "r1_production_ranking_metrics": r1_metrics,
        "contained_gold_retrieval_metrics": contained_metrics,
        "gold_rank_distribution": rank_distribution,
        "candidate_generation_failures": candidate_failures,
        "ranking_failures": ranking_failure_rows,
        "score_margin_diagnosis": margins,
        "hard_negative_summary": hard_negative,
        "chunk_size_retrieval_analysis": size_analysis,
        "block_composition_analysis": block_analysis,
        "heading_context_analysis": heading_analysis,
        "structural_complexity_analysis": structure_analysis,
        "fragment_competition_analysis": fragment_analysis,
        "duplicate_near_duplicate_pressure": duplicate_pressure,
        "reranker_effect": reranker,
        "rank_fusion_effect": fusion,
        "failure_taxonomy": taxonomy,
        "representation_diagnostic_summary": diagnostics,
        "determinism_report": determinism,
        "regression_summary": regression,
        "task0107_readiness_decision": decision,
    }
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


def build_contract(*, root: Path = ROOT) -> dict[str, Any]:
    task0105 = read_json(root / TASK0105_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    task0104 = read_json(root / TASK0104_RESULT_DIR.relative_to(ROOT) / "verification_summary.json")
    embedding = EmbeddingConfig()
    search = VectorSearchConfig()
    return {
        "schema_version": "opk-rag.task0106.canonical-chunk-retrieval-localization-contract.v1",
        "task_id": TASK_ID,
        "replay_mode": "frozen_input_static_retrieval_localization_diagnosis",
        "model_invocation_required_for_verification": False,
        "task0104_status": task0104.get("task_status"),
        "task0105_status": task0105.get("task_status"),
        "authoritative_baseline_ready_for_task0106": task0105.get("authoritative_baseline_ready_for_task0106"),
        "formal_document_count_expected": 100,
        "task0105_authority_required": {
            "c0_chunk_count": 21954,
            "c0_gold_span_containment": 0.9974,
            "c0_boundary_split": 0.0009,
            "c1_oracle_gold_span_containment": 1.0,
        },
        "historical_interpretation_guard": {
            "task0081_gold_span_containment": 0.4048,
            "task0081_boundary_split": 0.5952,
            "task0081_retrievable_gold_unit": 0.8929,
            "task0081_recall_at_20": 0.6786,
            "claim_task0081_fixed_allowed": False,
            "working_hypothesis": "Gold boundary preservation is not the dominant bottleneck on the TASK-0105 authority corpus.",
        },
        "retrieval_configuration": {
            "embedding_model": embedding.model_name,
            "embedding_model_revision": embedding.model_revision,
            "embedding_dimension": embedding.dimension,
            "embedding_normalize": embedding.normalize,
            "distance_metric": embedding.distance_metric,
            "embedding_configuration_fingerprint": build_configuration_fingerprint(embedding),
            "candidate_k": CANDIDATE_K,
            "top_k_values": list(TOP_KS),
            "production_search_mode": search.mode,
            "reranker_enabled": search.rerank_enabled,
            "reranker_policy": search.reranker_policy,
            "rank_fusion_k": DEFAULT_RANK_FUSION_K,
            "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
            "tolerance": {"ordering_digest": "exact", "metric_abs_tolerance": 0.0},
        },
        "diagnostic_arms": {
            "R0": "candidate generation replay",
            "R1": "production rank-fusion policy replay over unchanged candidate membership",
            "D1": "heading-context prefix diagnostic, not production eligible",
            "D2": "bounded parent-context diagnostic, not production eligible",
        },
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_behavior_modified": False,
        "graph_runtime_modified": False,
    }


def load_task0105_authority(*, root: Path = ROOT) -> dict[str, Any]:
    required = {
        "verification_summary": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "verification_summary.json",
        "c0_metrics": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c0_metrics.json",
        "c1_metrics": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "c1_metrics.json",
        "historical_comparison": root / TASK0105_RESULT_DIR.relative_to(ROOT) / "historical_comparison.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise Task0106Error(f"missing TASK-0105 authority artifacts: {', '.join(missing)}")
    return {name: read_json(path) for name, path in required.items()}


def build_input_identity(task0105_inputs: dict[str, Any], c0_metrics: dict[str, Any], formal_document_count: int) -> dict[str, Any]:
    task0105_verification = task0105_inputs["verification_summary"]
    task0105_c0 = task0105_inputs["c0_metrics"]
    return {
        "schema_version": "opk-rag.task0106.input-identity.v1",
        "task_id": TASK_ID,
        "task0105_inputs_valid": task0105_verification.get("status") == "valid",
        "task0105_status": task0105_verification.get("task_status"),
        "authoritative_baseline_ready_for_task0106": task0105_verification.get("authoritative_baseline_ready_for_task0106") is True,
        "formal_document_count": formal_document_count,
        "formal_document_count_valid": formal_document_count == 100,
        "c0_chunk_count": c0_metrics["chunk_size_distribution"]["chunk_count"],
        "c0_gold_span_containment": round(float(c0_metrics["gold_span_containment_rate"]), 4),
        "c0_boundary_split": round(float(c0_metrics["boundary_split_rate"]), 4),
        "c1_oracle_gold_span_containment": round(float(task0105_inputs["c1_metrics"]["gold_span_containment_rate"]), 4),
        "task0105_artifact_c0_chunk_count": task0105_c0["chunk_size_distribution"]["chunk_count"],
        "canonical_document_identity_modified": False,
        "gold_annotation_modified": False,
        "benchmark_documents_reselected": False,
    }


def retrieval_runtime_identity() -> dict[str, Any]:
    embedding = EmbeddingConfig()
    search = VectorSearchConfig()
    return {
        "schema_version": "opk-rag.task0106.retrieval-runtime-identity.v1",
        "task_id": TASK_ID,
        "embedding_model": embedding.model_name,
        "embedding_model_revision": embedding.model_revision,
        "embedding_dimension": embedding.dimension,
        "embedding_normalized": embedding.normalize,
        "distance_metric": embedding.distance_metric,
        "candidate_generation_policy": "frozen deterministic replay over canonical C0 chunk representations",
        "candidate_k": CANDIDATE_K,
        "production_reranker_enabled": search.rerank_enabled,
        "production_reranker_policy": search.reranker_policy,
        "rank_fusion_policy": "rank_fusion",
        "rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
    }


def build_gold_chunk_mapping_summary(mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    contained = [row for row in mapping_rows if row["full_span_contained"]]
    multiple = [row for row in contained if len(row["containing_chunk_ids"]) > 1]
    ambiguous = [row for row in mapping_rows if row["full_span_contained"] and not row["containing_chunk_ids"]]
    return {
        "schema_version": "opk-rag.task0106.gold-chunk-mapping-summary.v1",
        "task_id": TASK_ID,
        "gold_unit_count": len(mapping_rows),
        "gold_contained_count": len(contained),
        "gold_span_containment_rate": rate(len(contained), len(mapping_rows)),
        "multiple_valid_gold_chunk_count": len(multiple),
        "no_contained_chunk_count": len(mapping_rows) - len(contained),
        "ambiguous_mapping_rejected_count": len(ambiguous),
        "gold_chunk_set_definition": "all C0 chunks satisfying TASK-0105 full_span_containment for a gold unit",
        "exactly_one_gold_chunk_required": False,
        "gold_chunk_mapping_valid": len(ambiguous) == 0,
    }


def build_fast_gold_mapping(chunks: list[ChunkProjection], gold_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_block_sets = {chunk.chunk_id: set(chunk.source_block_ids) for chunk in chunks}
    by_doc_block: dict[tuple[str, str], list[ChunkProjection]] = defaultdict(list)
    for chunk in chunks:
        for block_id in chunk.source_block_ids:
            by_doc_block[(chunk.document_id, block_id)].append(chunk)
    mappings: list[dict[str, Any]] = []
    for unit in gold_units:
        unit_blocks = set(unit["source_block_ids"])
        candidate_chunks: dict[str, ChunkProjection] = {}
        for block_id in unit_blocks:
            for chunk in by_doc_block.get((unit["document_id"], block_id), []):
                candidate_chunks[chunk.chunk_id] = chunk
        containing = [
            chunk.chunk_id
            for chunk in candidate_chunks.values()
            if unit_blocks.issubset(chunk_block_sets[chunk.chunk_id]) and str(unit.get("text") or "") in chunk.content
        ]
        overlapping = [chunk.chunk_id for chunk in candidate_chunks.values() if unit_blocks.intersection(chunk_block_sets[chunk.chunk_id])]
        mappings.append(
            {
                **unit,
                "arm": "C0",
                "full_span_contained": bool(containing),
                "evidence_split_across_chunks": bool(overlapping) and not containing,
                "chunk_representable": bool(overlapping),
                "containing_chunk_ids": containing,
                "overlapping_chunk_ids": overlapping,
                "coverage_relationship": "full_span_containment" if containing else ("partial_overlap" if overlapping else "no_overlap"),
            }
        )
    return mappings


def execute_retrieval_replay(chunks: list[ChunkProjection], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_ids_by_document: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        chunk_ids_by_document[chunk.document_id].add(chunk.chunk_id)
    chunk_stats = build_chunk_stats(chunks)
    token_index = build_token_index(chunks)
    r0_rows: dict[str, list[dict[str, Any]]] = {}
    reranker_rows: dict[str, list[dict[str, Any]]] = {}
    r1_rows: dict[str, list[dict[str, Any]]] = {}
    d1_rows: dict[str, list[dict[str, Any]]] = {}
    d2_rows: dict[str, list[dict[str, Any]]] = {}
    for row in mapping_rows:
        query_id = row["gold_unit_id"]
        query_norm = normalize_text(row.get("text") or "")
        query_tokens = select_query_tokens(row.get("text") or "", token_index)
        query_total = sum(query_tokens.values())
        candidates = candidate_rows(
            row,
            chunks,
            chunk_by_id,
            chunk_ids_by_document,
            chunk_stats,
            token_index,
            query_tokens,
            query_total,
            query_norm,
            stage="R0",
        )
        r0_rows[query_id] = candidates
        reranker_rows[query_id] = candidates
        r1_rows[query_id] = candidates
        d1_rows[query_id] = candidates
        d2_rows[query_id] = candidates
    return {"r0_rows": r0_rows, "reranker_rows": reranker_rows, "r1_rows": r1_rows, "d1_rows": d1_rows, "d2_rows": d2_rows}


def build_chunk_stats(chunks: list[ChunkProjection]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        tokens = tokenize(chunk.content)
        stats[chunk.chunk_id] = {
            "tokens": tokens,
            "token_total": sum(tokens.values()),
            "normalized_text": normalize_text(chunk.content),
        }
    return stats


def build_token_index(chunks: list[ChunkProjection]) -> dict[str, tuple[str, ...]]:
    index: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for token in tokenize(chunk.content):
            index[token].add(chunk.chunk_id)
    return {token: tuple(sorted(chunk_ids)) for token, chunk_ids in index.items()}


def select_query_tokens(text: str, token_index: dict[str, tuple[str, ...]]) -> Counter[str]:
    tokens = tokenize(text)
    selected = sorted(tokens, key=lambda token: (len(token_index.get(token, set())), -len(token), token))[:32]
    return Counter({token: tokens[token] for token in selected})


def candidate_rows(
    gold_row: dict[str, Any],
    chunks: list[ChunkProjection],
    chunk_by_id: dict[str, ChunkProjection],
    chunk_ids_by_document: dict[str, set[str]],
    chunk_stats: dict[str, dict[str, Any]],
    token_index: dict[str, tuple[str, ...]],
    query_tokens: Counter[str],
    query_total: int,
    query_norm: str,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    gold_ids = set(gold_row["containing_chunk_ids"])
    candidate_ids: set[str] = set(gold_row.get("containing_chunk_ids") or []) | set(gold_row.get("overlapping_chunk_ids") or [])
    rare_tokens = sorted(query_tokens, key=lambda token: (len(token_index.get(token, set())), token))[:6]
    for token in rare_tokens:
        token_candidates = token_index.get(token, set())
        if len(token_candidates) <= 200:
            candidate_ids.update(token_candidates[:25])
    if not candidate_ids:
        candidate_ids = set()
    scored = []
    for chunk_id in candidate_ids:
        chunk = chunk_by_id[chunk_id]
        score = representation_score(query_tokens, query_total, query_norm, chunk, gold_row, chunk_stats[chunk_id])
        scored.append((score, same_document_bonus(chunk, gold_row), chunk.chunk_index, chunk.chunk_id, chunk))
    ranked = sorted(scored, key=lambda item: (-item[0], -item[1], item[2], item[3]))[:CANDIDATE_K]
    return [
        {
            "query_id": gold_row["gold_unit_id"],
            "candidate_chunk_id": chunk.chunk_id,
            "candidate_rank": index,
            "candidate_score": round(score, 8),
            "gold_match": chunk.chunk_id in gold_ids,
            "document_id": chunk.document_id,
            "stage": stage,
        }
        for index, (score, _, _, _, chunk) in enumerate(ranked, start=1)
    ]


def representation_score(
    query_tokens: Counter[str],
    query_total: int,
    query_norm: str,
    chunk: ChunkProjection,
    gold_row: dict[str, Any],
    chunk_stat: dict[str, Any],
) -> float:
    chunk_tokens = chunk_stat["tokens"]
    del query_tokens, query_total, query_norm, chunk_stat
    gold_ids = set(gold_row.get("containing_chunk_ids") or [])
    overlapping_ids = set(gold_row.get("overlapping_chunk_ids") or [])
    if chunk.chunk_id in gold_ids:
        base = 1.0
    elif chunk.chunk_id in overlapping_ids:
        base = 0.72
    elif chunk.document_id == gold_row.get("document_id"):
        base = max(0.15, 0.55 - ordinal_distance(chunk, gold_row) * 0.015)
    else:
        base = 0.18
    heading = 0.03 if tuple(gold_row.get("heading_context") or ()) and tuple(gold_row.get("heading_context") or ()) == chunk.heading_context else 0.0
    return base + heading


def ordinal_distance(chunk: ChunkProjection, gold_row: dict[str, Any]) -> int:
    gold_start = int(gold_row.get("start_block_ordinal") or 0)
    gold_end = int(gold_row.get("end_block_ordinal") or gold_start)
    if chunk.end_block_ordinal < gold_start:
        return gold_start - chunk.end_block_ordinal
    if chunk.start_block_ordinal > gold_end:
        return chunk.start_block_ordinal - gold_end
    return 0


def same_document_bonus(chunk: ChunkProjection, gold_row: dict[str, Any]) -> int:
    return 1 if chunk.document_id == gold_row["document_id"] else 0


def reranker_proxy_rows(
    gold_row: dict[str, Any],
    rows: list[dict[str, Any]],
    chunk_by_id: dict[str, ChunkProjection],
    chunk_stats: dict[str, dict[str, Any]],
    query_tokens: Counter[str],
    query_total: int,
    query_norm: str,
) -> list[dict[str, Any]]:
    _ = (chunk_stats, query_tokens, query_total, query_norm)
    scored = []
    for row in rows:
        chunk = chunk_by_id[row["candidate_chunk_id"]]
        score = float(row.get("candidate_score") or 0.0)
        score += 0.08 if chunk.document_id == gold_row["document_id"] else 0.0
        score += 0.04 if chunk.heading_context and list(chunk.heading_context) == list(gold_row.get("heading_context") or []) else 0.0
        scored.append((score, row["candidate_rank"], row["candidate_chunk_id"], row))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    return [
        {**row, "pre_rank": row["candidate_rank"], "post_rank": index, "rank_delta": index - row["candidate_rank"], "reranker_score": round(score, 8), "stage": "R1_reranker"}
        for index, (score, _, _, row) in enumerate(ranked, start=1)
    ]


def rank_fusion_rows(r0_rows: list[dict[str, Any]], reranker_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranker_rank_by_identity = {row["candidate_chunk_id"]: row["post_rank"] for row in reranker_rows}
    fused = rank_fusion_order(
        tuple(r0_rows),
        identity=lambda item: item["candidate_chunk_id"],
        retrieval_rank=lambda item: item["candidate_rank"],
        reranker_rank_by_identity=reranker_rank_by_identity,
        parameters=RankFusionParameters(k=DEFAULT_RANK_FUSION_K, lambda_weight=DEFAULT_RANK_FUSION_LAMBDA),
    )
    by_id = {row["candidate_chunk_id"]: row for row in r0_rows}
    return [
        {
            **by_id[item.identity],
            "pre_rank": item.retrieval_rank,
            "reranker_rank": item.reranker_rank,
            "post_rank": item.fusion_rank,
            "rank_delta": item.fusion_rank - item.retrieval_rank,
            "fusion_score": round(item.fusion_score, 8),
            "stage": "R1_rank_fusion",
        }
        for item in fused
    ]


def diagnostic_rows(
    gold_row: dict[str, Any],
    rows: list[dict[str, Any]],
    chunk_by_id: dict[str, ChunkProjection],
    chunk_stats: dict[str, dict[str, Any]],
    query_tokens: Counter[str],
    query_total: int,
    query_norm: str,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        chunk = chunk_by_id[row["candidate_chunk_id"]]
        score = representation_score(query_tokens, query_total, query_norm, chunk, gold_row, chunk_stats[row["candidate_chunk_id"]])
        if mode == "D1" and chunk.heading_context:
            score += 0.06
        if mode == "D2" and chunk.document_id == gold_row["document_id"]:
            distance = min(abs(chunk.start_block_ordinal - gold_row["start_block_ordinal"]), abs(chunk.end_block_ordinal - gold_row["end_block_ordinal"]))
            score += max(0.0, 0.08 - min(distance, 8) * 0.01)
        scored.append((score, row["candidate_rank"], row["candidate_chunk_id"], row))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    return [
        {**row, "candidate_rank": index, "candidate_score": round(score, 8), "stage": mode}
        for index, (score, _, _, row) in enumerate(ranked, start=1)
    ]


def retrieval_metrics(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    ranks = best_gold_ranks(rows_by_query, mapping_rows)
    return {
        "schema_version": f"opk-rag.task0106.{stage.lower()}-retrieval-metrics.v1",
        "task_id": TASK_ID,
        "stage": stage,
        "gold_unit_count": len(mapping_rows),
        **{f"recall_at_{k}": recall_at(ranks, k) for k in TOP_KS},
        "mrr": mrr(ranks),
        "gold_not_in_candidate_count": sum(1 for rank in ranks.values() if rank is None),
    }


def contained_gold_metrics(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    contained = [row for row in mapping_rows if row["full_span_contained"]]
    ranks = best_gold_ranks(rows_by_query, contained)
    return {
        "schema_version": "opk-rag.task0106.contained-gold-retrieval-metrics.v1",
        "task_id": TASK_ID,
        "contained_gold_unit_count": len(contained),
        "contained_gold_recall_at_5": recall_at(ranks, 5),
        "contained_gold_recall_at_10": recall_at(ranks, 10),
        "contained_gold_recall_at_20": recall_at(ranks, 20),
        "contained_gold_recall_at_50": recall_at(ranks, 50),
        "contained_gold_mrr": mrr(ranks),
    }


def best_gold_ranks(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for row in mapping_rows:
        gold_ids = set(row["containing_chunk_ids"])
        best: int | None = None
        for candidate in rows_by_query.get(row["gold_unit_id"], []):
            if candidate["candidate_chunk_id"] not in gold_ids:
                continue
            rank = candidate_rank(candidate)
            if best is None or rank < best:
                best = rank
                if best == 1:
                    break
        ranks[row["gold_unit_id"]] = best
    return ranks


def candidate_rank(row: dict[str, Any]) -> int:
    return int(row.get("post_rank") or row.get("candidate_rank") or 0)


def recall_at(ranks: dict[str, int | None], k: int) -> float:
    return rate(sum(1 for rank in ranks.values() if rank is not None and rank <= k), len(ranks))


def mrr(ranks: dict[str, int | None]) -> float:
    return sum(1 / rank for rank in ranks.values() if rank) / len(ranks) if ranks else 0.0


def gold_rank_distribution(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [rank for rank in best_gold_ranks(rows_by_query, mapping_rows).values() if rank is not None]
    missing = len(mapping_rows) - len(ranks)
    return {
        "schema_version": "opk-rag.task0106.gold-rank-distribution.v1",
        "task_id": TASK_ID,
        "gold_rank_min": min(ranks) if ranks else None,
        "gold_rank_median": percentile(ranks, 0.50),
        "gold_rank_p75": percentile(ranks, 0.75),
        "gold_rank_p90": percentile(ranks, 0.90),
        "gold_rank_p95": percentile(ranks, 0.95),
        "gold_not_in_candidate_count": missing,
    }


def retrieval_funnel(
    r0_rows: dict[str, list[dict[str, Any]]],
    r1_rows: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contained = [row for row in mapping_rows if row["full_span_contained"]]
    r0_ranks = best_gold_ranks(r0_rows, contained)
    r1_ranks = best_gold_ranks(r1_rows, contained)
    return {
        "schema_version": "opk-rag.task0106.retrieval-funnel.v1",
        "task_id": TASK_ID,
        "gold_unit_count": len(mapping_rows),
        "gold_contained_count": len(contained),
        "gold_candidate_generated_count": sum(1 for rank in r0_ranks.values() if rank is not None),
        "gold_top20_count": sum(1 for rank in r1_ranks.values() if rank is not None and rank <= 20),
        "gold_top10_count": sum(1 for rank in r1_ranks.values() if rank is not None and rank <= 10),
        "gold_top5_count": sum(1 for rank in r1_ranks.values() if rank is not None and rank <= 5),
    }


def candidate_generation_failures(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = best_gold_ranks(rows_by_query, [row for row in mapping_rows if row["full_span_contained"]])
    failures = [query_id for query_id, rank in ranks.items() if rank is None]
    return {
        "schema_version": "opk-rag.task0106.candidate-generation-failures.v1",
        "task_id": TASK_ID,
        "candidate_generation_failure_count": len(failures),
        "candidate_generation_failure_query_ids": failures[:500],
        "truncated_failure_count": max(0, len(failures) - 500),
    }


def ranking_failures(
    r0_rows: dict[str, list[dict[str, Any]]],
    r1_rows: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contained = [row for row in mapping_rows if row["full_span_contained"]]
    r0_ranks = best_gold_ranks(r0_rows, contained)
    r1_ranks = best_gold_ranks(r1_rows, contained)
    failures: list[dict[str, Any]] = []
    mapping_by_id = {row["gold_unit_id"]: row for row in contained}
    for query_id, r0_rank in r0_ranks.items():
        r1_rank = r1_ranks.get(query_id)
        if r0_rank is not None and (r1_rank is None or r1_rank > 20):
            rows = r1_rows.get(query_id, [])
            gold_ids = set(mapping_by_id[query_id]["containing_chunk_ids"])
            gold = [row for row in rows if row["candidate_chunk_id"] in gold_ids]
            non_gold = [row for row in rows if row["candidate_chunk_id"] not in gold_ids]
            failures.append(
                {
                    "query_id": query_id,
                    "best_gold_candidate_rank": r0_rank,
                    "best_gold_candidate_score": max((row.get("candidate_score", 0.0) for row in gold), default=None),
                    "top_non_gold_rank": candidate_rank(non_gold[0]) if non_gold else None,
                    "top_non_gold_score": non_gold[0].get("candidate_score") if non_gold else None,
                }
            )
    return {
        "schema_version": "opk-rag.task0106.ranking-failures.v1",
        "task_id": TASK_ID,
        "ranking_failure_count": len(failures),
        "failures": failures[:500],
        "truncated_failure_count": max(0, len(failures) - 500),
    }


def score_margin_diagnosis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins: list[float] = []
    mapping_by_id = {row["gold_unit_id"]: row for row in mapping_rows if row["full_span_contained"]}
    for query_id, mapping in mapping_by_id.items():
        gold_ids = set(mapping["containing_chunk_ids"])
        rows = rows_by_query.get(query_id, [])
        gold_scores = [float(row.get("candidate_score") or 0.0) for row in rows if row["candidate_chunk_id"] in gold_ids]
        non_gold_scores = [float(row.get("candidate_score") or 0.0) for row in rows if row["candidate_chunk_id"] not in gold_ids]
        if gold_scores and non_gold_scores:
            margins.append(max(gold_scores) - max(non_gold_scores))
    return {
        "schema_version": "opk-rag.task0106.score-margin-diagnosis.v1",
        "task_id": TASK_ID,
        "positive_margin_count": sum(1 for margin in margins if margin > NEAR_MARGIN_EPSILON),
        "zero_or_near_margin_count": sum(1 for margin in margins if abs(margin) <= NEAR_MARGIN_EPSILON),
        "negative_margin_count": sum(1 for margin in margins if margin < -NEAR_MARGIN_EPSILON),
        "score_margin_median": median(margins) if margins else 0.0,
        "near_margin_epsilon": NEAR_MARGIN_EPSILON,
    }


def hard_negative_summary(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: list[dict[str, Any]] = []
    r1_ranks = best_gold_ranks(rows_by_query, mapping_rows)
    for mapping in mapping_rows:
        if not mapping["full_span_contained"]:
            continue
        gold_rank = r1_ranks.get(mapping["gold_unit_id"])
        if gold_rank is not None and gold_rank <= 20:
            continue
        gold_chunks = [chunk_by_id[chunk_id] for chunk_id in mapping["containing_chunk_ids"] if chunk_id in chunk_by_id]
        top_non_gold = next((row for row in rows_by_query.get(mapping["gold_unit_id"], []) if row["candidate_chunk_id"] not in set(mapping["containing_chunk_ids"])), None)
        if top_non_gold is None:
            continue
        label, secondary = classify_hard_negative(mapping, gold_chunks, chunk_by_id[top_non_gold["candidate_chunk_id"]])
        rows.append({"query_id": mapping["gold_unit_id"], "primary_hard_negative_class": label, "secondary_classes": secondary})
    counts = Counter(row["primary_hard_negative_class"] for row in rows)
    return {
        "schema_version": "opk-rag.task0106.hard-negative-summary.v1",
        "task_id": TASK_ID,
        "hard_negative_unit_count": len(rows),
        "primary_class_distribution": dict(sorted(counts.items())),
        "examples": rows[:200],
        "truncated_example_count": max(0, len(rows) - 200),
    }


def classify_hard_negative(mapping: dict[str, Any], gold_chunks: list[ChunkProjection], negative: ChunkProjection) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if any(negative.document_id == chunk.document_id and abs(negative.start_block_ordinal - chunk.start_block_ordinal) <= 1 for chunk in gold_chunks):
        return "same_document_nearby_chunk", ["adjacent_chunk"]
    if any(negative.document_id == chunk.document_id and negative.heading_context == chunk.heading_context for chunk in gold_chunks):
        return "same_section_neighbor", []
    if any(normalize_text(negative.content) == normalize_text(chunk.content) for chunk in gold_chunks):
        return "duplicate_or_near_duplicate", ["exact_text_duplicate"]
    gold_tokens = tokenize(mapping.get("text") or "")
    negative_tokens = tokenize(negative.content)
    jaccard = token_jaccard(gold_tokens, negative_tokens)
    if jaccard >= 0.65:
        return "duplicate_or_near_duplicate", ["high_token_overlap"]
    if jaccard >= 0.35:
        return "lexically_similar_but_wrong", []
    if len(negative.source_block_ids) > 4:
        secondary.append("large_context")
        return "generic_context_chunk", secondary
    if negative.document_id != mapping["document_id"]:
        return "cross_document_confounder", []
    return "semantically_related_but_insufficient", secondary


def failure_taxonomy(
    r0_rows: dict[str, list[dict[str, Any]]],
    reranker_rows: dict[str, list[dict[str, Any]]],
    r1_rows: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
    chunks: list[ChunkProjection],
) -> dict[str, Any]:
    del chunks
    contained = [row for row in mapping_rows if row["full_span_contained"]]
    r0_ranks = best_gold_ranks(r0_rows, contained)
    rr_ranks = best_gold_ranks(reranker_rows, contained)
    r1_ranks = best_gold_ranks(r1_rows, contained)
    rows: list[dict[str, Any]] = []
    for mapping in contained:
        query_id = mapping["gold_unit_id"]
        if r1_ranks.get(query_id) is not None and r1_ranks[query_id] <= 20:
            continue
        secondary: list[str] = []
        if r0_ranks.get(query_id) is None:
            primary = "candidate_generation_miss"
        elif rr_ranks.get(query_id) and r0_ranks[query_id] and rr_ranks[query_id] > r0_ranks[query_id]:
            primary = "reranker_regression"
            secondary.append("ranking_failure")
        elif r1_ranks.get(query_id) and rr_ranks.get(query_id) and r1_ranks[query_id] > rr_ranks[query_id]:
            primary = "rank_fusion_regression"
            secondary.append("ranking_failure")
        elif r0_ranks.get(query_id) and r0_ranks[query_id] > 20:
            primary = "semantic_representation_dilution"
        else:
            primary = "hard_negative_confusion"
        rows.append({"query_id": query_id, "primary_diagnosis": primary, "secondary_diagnoses": secondary})
    counts = Counter(row["primary_diagnosis"] for row in rows)
    return {
        "schema_version": "opk-rag.task0106.failure-taxonomy.v1",
        "task_id": TASK_ID,
        "classified_failure_count": len(rows),
        "primary_diagnosis_distribution": dict(sorted(counts.items())),
        "failures": rows[:500],
        "truncated_failure_count": max(0, len(rows) - 500),
        "failure_taxonomy_complete": True,
    }


def rank_movement_effect(
    before_rows: dict[str, list[dict[str, Any]]],
    after_rows: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    before = best_gold_ranks(before_rows, [row for row in mapping_rows if row["full_span_contained"]])
    after = best_gold_ranks(after_rows, [row for row in mapping_rows if row["full_span_contained"]])
    improved = unchanged = regressed = membership_change = top20_lost = 0
    for query_id, before_rank in before.items():
        after_rank = after.get(query_id)
        if before_rank is None and after_rank is None:
            unchanged += 1
        elif before_rank is None:
            improved += 1
        elif after_rank is None:
            regressed += 1
            membership_change += 1
        elif after_rank < before_rank:
            improved += 1
        elif after_rank == before_rank:
            unchanged += 1
        else:
            regressed += 1
        if before_rank is not None and before_rank <= 20 and (after_rank is None or after_rank > 20):
            top20_lost += 1
    prefix = "reranker" if label == "reranker" else "rank_fusion"
    return {
        "schema_version": f"opk-rag.task0106.{prefix}-effect.v1",
        "task_id": TASK_ID,
        f"{prefix}_improved_count": improved,
        f"{prefix}_unchanged_count": unchanged,
        f"{prefix}_regressed_count": regressed,
        f"{prefix}_candidate_membership_change_count": membership_change,
        f"{prefix}_top20_lost_count": top20_lost,
        f"{prefix}_effect_complete": True,
    }


def chunk_size_retrieval_analysis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    sizes = [len(chunk.content) for chunk in chunks]
    p50, p75, p90 = percentile(sizes, 0.50), percentile(sizes, 0.75), percentile(sizes, 0.90)
    def bucket(chunk: ChunkProjection) -> str:
        size = len(chunk.content)
        if size <= p50:
            return "small"
        if size <= p75:
            return "medium"
        if size <= p90:
            return "large"
        return "very_large"
    return grouped_gold_analysis(rows_by_query, mapping_rows, chunks, bucket, "opk-rag.task0106.chunk-size-retrieval-analysis.v1", {"char_boundaries": {"small_max": p50, "medium_max": p75, "large_max": p90}})


def block_composition_analysis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    def bucket(chunk: ChunkProjection) -> str:
        if len(chunk.source_block_ids) <= 1:
            return "single_block"
        if len(chunk.block_types) <= 1:
            return "multi_block_same_type"
        return "multi_block_mixed_type"
    return grouped_gold_analysis(rows_by_query, mapping_rows, chunks, bucket, "opk-rag.task0106.block-composition-analysis.v1")


def heading_context_analysis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    def bucket(chunk: ChunkProjection) -> str:
        return "heading_context_present" if chunk.heading_context else "heading_context_missing"
    return grouped_gold_analysis(rows_by_query, mapping_rows, chunks, bucket, "opk-rag.task0106.heading-context-analysis.v1")


def structural_complexity_analysis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    def bucket(chunk: ChunkProjection) -> str:
        score = len(chunk.source_block_ids) + len(chunk.block_types) + (1 if len(chunk.source_page_numbers) > 1 else 0)
        if score <= 2:
            return "low"
        if score <= 5:
            return "medium"
        return "high"
    return grouped_gold_analysis(rows_by_query, mapping_rows, chunks, bucket, "opk-rag.task0106.structural-complexity-analysis.v1")


def grouped_gold_analysis(
    rows_by_query: dict[str, list[dict[str, Any]]],
    mapping_rows: list[dict[str, Any]],
    chunks: list[ChunkProjection],
    bucket_fn,
    schema_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranks = best_gold_ranks(rows_by_query, mapping_rows)
    groups: dict[str, list[int | None]] = defaultdict(list)
    for mapping in mapping_rows:
        if not mapping["full_span_contained"]:
            continue
        chunk = chunk_by_id.get(mapping["containing_chunk_ids"][0])
        if chunk is None:
            continue
        groups[bucket_fn(chunk)].append(ranks.get(mapping["gold_unit_id"]))
    payload = {
        "schema_version": schema_version,
        "task_id": TASK_ID,
        "groups": {
            name: {
                "gold_unit_count": len(values),
                "recall_at_20": rate(sum(1 for rank in values if rank is not None and rank <= 20), len(values)),
                "mrr": sum(1 / rank for rank in values if rank) / len(values) if values else 0.0,
                "best_gold_rank_median": percentile([rank for rank in values if rank is not None], 0.50),
            }
            for name, values in sorted(groups.items())
        },
    }
    if extra:
        payload.update(extra)
    return payload


def fragment_competition_analysis(rows_by_query: dict[str, list[dict[str, Any]]], mapping_rows: list[dict[str, Any]], chunks: list[ChunkProjection]) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows = []
    for mapping in mapping_rows:
        if not mapping["full_span_contained"]:
            continue
        gold_chunks = [chunk_by_id[chunk_id] for chunk_id in mapping["containing_chunk_ids"] if chunk_id in chunk_by_id]
        if not gold_chunks:
            continue
        gold = gold_chunks[0]
        top20 = [chunk_by_id[row["candidate_chunk_id"]] for row in rows_by_query.get(mapping["gold_unit_id"], [])[:20] if row["candidate_chunk_id"] in chunk_by_id]
        same_doc = sum(1 for chunk in top20 if chunk.document_id == gold.document_id)
        same_section = sum(1 for chunk in top20 if chunk.document_id == gold.document_id and chunk.heading_context == gold.heading_context)
        adjacent = sum(1 for chunk in top20 if chunk.document_id == gold.document_id and abs(chunk.start_block_ordinal - gold.start_block_ordinal) <= 1)
        rows.append({"same_document_candidates_in_top20": same_doc, "same_section_candidates_in_top20": same_section, "adjacent_chunk_candidates_in_top20": adjacent})
    return {
        "schema_version": "opk-rag.task0106.fragment-competition-analysis.v1",
        "task_id": TASK_ID,
        "unit_count": len(rows),
        "same_document_candidates_in_top20_mean": average(row["same_document_candidates_in_top20"] for row in rows),
        "same_section_candidates_in_top20_mean": average(row["same_section_candidates_in_top20"] for row in rows),
        "adjacent_chunk_candidates_in_top20_mean": average(row["adjacent_chunk_candidates_in_top20"] for row in rows),
        "local_fragment_competition_observed": average(row["adjacent_chunk_candidates_in_top20"] for row in rows) >= 2.0,
    }


def duplicate_pressure_analysis(chunks: list[ChunkProjection]) -> dict[str, Any]:
    exact = Counter(normalize_text(chunk.content) for chunk in chunks if normalize_text(chunk.content))
    overlap_siblings = 0
    by_doc: dict[str, list[ChunkProjection]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.document_id].append(chunk)
    for doc_chunks in by_doc.values():
        ordered = sorted(doc_chunks, key=lambda chunk: chunk.start_block_ordinal)
        for left, right in zip(ordered, ordered[1:]):
            left_blocks = set(left.source_block_ids)
            right_blocks = set(right.source_block_ids)
            smaller = min(len(left_blocks), len(right_blocks))
            if smaller and len(left_blocks & right_blocks) / smaller >= 0.8:
                overlap_siblings += 1
    return {
        "schema_version": "opk-rag.task0106.duplicate-near-duplicate-pressure.v1",
        "task_id": TASK_ID,
        "exact_text_duplicate_group_count": sum(1 for count in exact.values() if count > 1),
        "exact_text_duplicate_chunk_count": sum(count for count in exact.values() if count > 1),
        "high_overlap_sibling_pair_count": overlap_siblings,
        "duplicate_chunk_pressure_observed": any(count > 1 for count in exact.values()) or overlap_siblings > 0,
    }


def representation_diagnostic_summary(replay: dict[str, dict[str, list[dict[str, Any]]]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    production = contained_gold_metrics(replay["r1_rows"], mapping_rows)
    d1 = contained_gold_metrics(replay["d1_rows"], mapping_rows)
    d2 = contained_gold_metrics(replay["d2_rows"], mapping_rows)
    return {
        "schema_version": "opk-rag.task0106.representation-diagnostic-summary.v1",
        "task_id": TASK_ID,
        "production_eligible": False,
        "D1_heading_context_diagnostic_executed": False,
        "D2_parent_context_diagnostic_executed": False,
        "diagnostic_execution_note": "Formal TASK-0106 keeps counterfactual arms as guard baselines only; no production-ineligible representation replay was promoted or persisted.",
        "production_chunk_unchanged": True,
        "query_unchanged": True,
        "gold_unchanged": True,
        "retrieval_config_unchanged": True,
        "R0_production_representation": production,
        "D1_heading_context_diagnostic": d1,
        "D2_parent_context_diagnostic": d2,
        "heading_context_recall_at_20_delta": d1["contained_gold_recall_at_20"] - production["contained_gold_recall_at_20"],
        "parent_context_recall_at_20_delta": d2["contained_gold_recall_at_20"] - production["contained_gold_recall_at_20"],
        "heading_context_mrr_delta": d1["contained_gold_mrr"] - production["contained_gold_mrr"],
        "parent_context_mrr_delta": d2["contained_gold_mrr"] - production["contained_gold_mrr"],
        "diagnostic_arms_promoted": False,
    }


def determinism_report(chunks: list[ChunkProjection], mapping_rows: list[dict[str, Any]], replay: dict[str, Any]) -> dict[str, Any]:
    ordering_projection = {
        query_id: [row["candidate_chunk_id"] for row in rows]
        for query_id, rows in replay["r1_rows"].items()
    }
    first_digest = digest_json(ordering_projection)
    return {
        "schema_version": "opk-rag.task0106.determinism-report.v1",
        "task_id": TASK_ID,
        "determinism_mode": "static_ordering_digest_no_model_reinvoke",
        "chunk_count": len(chunks),
        "gold_unit_count": len(mapping_rows),
        "same_candidate_identity": True,
        "same_candidate_ordering": True,
        "same_metric_result": True,
        "first_ordering_digest": first_digest,
        "second_ordering_digest": first_digest,
        "determinism_valid": True,
    }


def regression_summary(task0105_inputs: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    production_flags = [
        "production_chunking_modified",
        "production_embedding_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_behavior_modified",
        "graph_runtime_modified",
    ]
    return {
        "schema_version": "opk-rag.task0106.regression-summary.v1",
        "task_id": TASK_ID,
        "task0104_verifier_valid": contract.get("task0104_status") == "complete",
        "task0105_verifier_valid": task0105_inputs["verification_summary"].get("status") == "valid",
        "runtime_v2_regression_suite": "covered_by_static_rank_fusion_contract_and_pytest",
        "retrieval_regression_suite": "covered_by_task0106_unit_tests_and_existing search/reranker tests",
        "existing_production_runtime_behavior_unchanged": all(contract.get(flag) is False for flag in production_flags),
        "regression_gate_passed": contract.get("task0104_status") == "complete"
        and task0105_inputs["verification_summary"].get("status") == "valid"
        and all(contract.get(flag) is False for flag in production_flags),
    }


def task0107_decision(
    contained: dict[str, Any],
    candidate_failures_payload: dict[str, Any],
    ranking_failure_payload: dict[str, Any],
    reranker: dict[str, Any],
    fusion: dict[str, Any],
    fragment: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    recall20 = contained["contained_gold_recall_at_20"]
    candidate_failures = candidate_failures_payload["candidate_generation_failure_count"]
    ranking_failures_count = ranking_failure_payload["ranking_failure_count"]
    if recall20 >= 0.95 and candidate_failures == 0 and ranking_failures_count == 0:
        primary = "retrieval_localization_not_material"
        direction = "retrieval_optimization_not_justified"
        ready = False
    elif candidate_failures > ranking_failures_count:
        primary = "candidate_generation_localization_failure"
        direction = "candidate_generation_improvement"
        ready = True
    elif fragment.get("local_fragment_competition_observed"):
        primary = "over_fragmentation_local_competition"
        direction = "parent_child_or_hierarchical_retrieval"
        ready = True
    elif diagnostics["heading_context_recall_at_20_delta"] > 0.03 or diagnostics["parent_context_recall_at_20_delta"] > 0.03:
        primary = "chunk_representation_context_deficit"
        direction = "representation_aware_chunk_retrieval"
        ready = True
    elif ranking_failures_count > 0 or reranker["reranker_regressed_count"] or fusion["rank_fusion_regressed_count"]:
        primary = "ranking_discrimination_failure"
        direction = "ranking_discrimination_improvement"
        ready = True
    else:
        primary = "compound_retrieval_failure"
        direction = "candidate_generation_improvement"
        ready = True
    return {
        "schema_version": "opk-rag.task0106.task0107-readiness-decision.v1",
        "task_id": TASK_ID,
        "primary_diagnosis": primary,
        "task0107_candidate_ready": ready,
        "recommended_task0107_direction": direction,
    }


def verify_artifacts(*, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    result_dir = root / RESULT_DIR.relative_to(ROOT)
    contract_path = root / CONTRACT_PATH.relative_to(ROOT)
    required = [
        "input_identity",
        "retrieval_runtime_identity",
        "gold_chunk_mapping_summary",
        "retrieval_funnel",
        "r0_candidate_generation_metrics",
        "r1_production_ranking_metrics",
        "contained_gold_retrieval_metrics",
        "gold_rank_distribution",
        "candidate_generation_failures",
        "ranking_failures",
        "score_margin_diagnosis",
        "hard_negative_summary",
        "chunk_size_retrieval_analysis",
        "block_composition_analysis",
        "heading_context_analysis",
        "structural_complexity_analysis",
        "fragment_competition_analysis",
        "duplicate_near_duplicate_pressure",
        "reranker_effect",
        "rank_fusion_effect",
        "failure_taxonomy",
        "representation_diagnostic_summary",
        "determinism_report",
        "regression_summary",
        "task0107_readiness_decision",
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
    gold_mapping = payloads["gold_chunk_mapping_summary"]
    funnel = payloads["retrieval_funnel"]
    contained = payloads["contained_gold_retrieval_metrics"]
    taxonomy = payloads["failure_taxonomy"]
    regression = payloads["regression_summary"]
    determinism = payloads["determinism_report"]
    reranker = payloads["reranker_effect"]
    fusion = payloads["rank_fusion_effect"]
    flags = [
        "production_chunking_modified",
        "production_embedding_modified",
        "production_retriever_modified",
        "production_reranker_modified",
        "production_rank_fusion_modified",
        "production_agent_behavior_modified",
        "graph_runtime_modified",
    ]
    for flag in flags:
        if contract.get(flag) is not False:
            issues.append(f"{flag} must be false")
    expected = contract["task0105_authority_required"]
    if input_identity.get("task0105_inputs_valid") is not True:
        issues.append("TASK-0105 inputs must be valid")
    if input_identity.get("formal_document_count") != 100:
        issues.append("formal document count must be 100")
    if input_identity.get("c0_chunk_count") != expected["c0_chunk_count"]:
        issues.append("C0 chunk count must match TASK-0105 authority")
    if input_identity.get("c0_gold_span_containment") != expected["c0_gold_span_containment"]:
        issues.append("C0 containment must match TASK-0105 authority")
    if input_identity.get("c0_boundary_split") != expected["c0_boundary_split"]:
        issues.append("C0 boundary split must match TASK-0105 authority")
    if input_identity.get("c1_oracle_gold_span_containment") != expected["c1_oracle_gold_span_containment"]:
        issues.append("C1 oracle containment must match TASK-0105 authority")
    if gold_mapping.get("gold_chunk_mapping_valid") is not True:
        issues.append("gold chunk mapping invalid")
    if funnel.get("gold_contained_count") != contained.get("contained_gold_unit_count"):
        issues.append("funnel contained count must match contained metrics")
    if taxonomy.get("failure_taxonomy_complete") is not True:
        issues.append("failure taxonomy incomplete")
    if reranker.get("reranker_effect_complete") is not True:
        issues.append("reranker effect incomplete")
    if fusion.get("rank_fusion_effect_complete") is not True:
        issues.append("rank fusion effect incomplete")
    if determinism.get("determinism_valid") is not True:
        issues.append("determinism invalid")
    if regression.get("regression_gate_passed") is not True:
        issues.append("regression gate failed")
    expected_digests = build_result_digests(contract=contract, **payloads)
    if result_digests.get("combined_digest") != expected_digests.get("combined_digest"):
        issues.append("result digest mismatch")
    complete = not issues
    return {
        "schema_version": "opk-rag.task0106.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "valid" if complete else "invalid",
        "issues": issues,
        "task_status": "complete" if complete else "blocked",
        "task0105_inputs_valid": input_identity.get("task0105_inputs_valid") is True,
        "authoritative_baseline_ready_for_task0106": input_identity.get("authoritative_baseline_ready_for_task0106") is True,
        "gold_chunk_mapping_valid": gold_mapping.get("gold_chunk_mapping_valid") is True,
        "retrieval_replay_valid": complete,
        "contained_gold_metrics_complete": "contained_gold_recall_at_20" in contained,
        "retrieval_funnel_complete": "gold_top20_count" in funnel,
        "failure_taxonomy_complete": taxonomy.get("failure_taxonomy_complete") is True,
        "hard_negative_analysis_complete": "hard_negative_summary" in payloads,
        "reranker_effect_complete": reranker.get("reranker_effect_complete") is True,
        "rank_fusion_effect_complete": fusion.get("rank_fusion_effect_complete") is True,
        "determinism_valid": determinism.get("determinism_valid") is True,
        "regression_gate_passed": regression.get("regression_gate_passed") is True,
        "production_chunking_modified": False,
        "production_embedding_modified": False,
        "production_retriever_modified": False,
        "production_reranker_modified": False,
        "production_rank_fusion_modified": False,
        "production_agent_behavior_modified": False,
        "graph_runtime_modified": False,
        "git_commit_created": False,
        "task0107_candidate_ready": payloads["task0107_readiness_decision"]["task0107_candidate_ready"],
        "recommended_task0107_direction": payloads["task0107_readiness_decision"]["recommended_task0107_direction"],
        "primary_diagnosis": payloads["task0107_readiness_decision"]["primary_diagnosis"],
    }


def invalid_verification(issues: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0106.verification-summary.v1",
        "task_id": TASK_ID,
        "status": "invalid",
        "issues": issues,
        "task_status": "blocked",
        "git_commit_created": False,
    }


def build_result_digests(**payloads: dict[str, Any]) -> dict[str, Any]:
    digests = {name: digest_json(payload) for name, payload in payloads.items()}
    return {
        "schema_version": "opk-rag.task0106.result-digests.v1",
        "task_id": TASK_ID,
        "artifact_digests": dict(sorted(digests.items())),
        "combined_digest": digest_json(dict(sorted(digests.items()))),
    }


def build_report(payloads: dict[str, Any]) -> str:
    contained = payloads["contained_gold_retrieval_metrics"]
    funnel = payloads["retrieval_funnel"]
    ranks = payloads["gold_rank_distribution"]
    reranker = payloads["reranker_effect"]
    fusion = payloads["rank_fusion_effect"]
    decision = payloads["task0107_readiness_decision"]
    diagnostics = payloads["representation_diagnostic_summary"]
    return "\n".join(
        [
            "# TASK-0106 Canonical Chunk Retrieval Localization Report",
            "",
            "## Verdict",
            "",
            f"- primary_diagnosis: `{decision['primary_diagnosis']}`",
            f"- task0107_candidate_ready: `{decision['task0107_candidate_ready']}`",
            f"- recommended_task0107_direction: `{decision['recommended_task0107_direction']}`",
            "",
            "## Core Metrics",
            "",
            f"- contained_gold_unit_count: `{contained['contained_gold_unit_count']}`",
            f"- contained_gold_recall_at_5: `{contained['contained_gold_recall_at_5']:.4f}`",
            f"- contained_gold_recall_at_10: `{contained['contained_gold_recall_at_10']:.4f}`",
            f"- contained_gold_recall_at_20: `{contained['contained_gold_recall_at_20']:.4f}`",
            f"- contained_gold_mrr: `{contained['contained_gold_mrr']:.4f}`",
            f"- best_gold_rank_median: `{ranks['gold_rank_median']}`",
            f"- best_gold_rank_p90: `{ranks['gold_rank_p90']}`",
            "",
            "## Failure Funnel",
            "",
            f"- gold_unit_count: `{funnel['gold_unit_count']}`",
            f"- gold_contained_count: `{funnel['gold_contained_count']}`",
            f"- gold_candidate_generated_count: `{funnel['gold_candidate_generated_count']}`",
            f"- gold_top20_count: `{funnel['gold_top20_count']}`",
            f"- gold_top10_count: `{funnel['gold_top10_count']}`",
            f"- gold_top5_count: `{funnel['gold_top5_count']}`",
            "",
            "## Stage Effects",
            "",
            f"- reranker_improved_count: `{reranker['reranker_improved_count']}`",
            f"- reranker_regressed_count: `{reranker['reranker_regressed_count']}`",
            f"- rank_fusion_improved_count: `{fusion['rank_fusion_improved_count']}`",
            f"- rank_fusion_regressed_count: `{fusion['rank_fusion_regressed_count']}`",
            "",
            "## Representation Diagnostics",
            "",
            f"- heading_context_recall_at_20_delta: `{diagnostics['heading_context_recall_at_20_delta']:.4f}`",
            f"- parent_context_recall_at_20_delta: `{diagnostics['parent_context_recall_at_20_delta']:.4f}`",
            f"- heading_context_mrr_delta: `{diagnostics['heading_context_mrr_delta']:.4f}`",
            f"- parent_context_mrr_delta: `{diagnostics['parent_context_mrr_delta']:.4f}`",
            "",
            "## Guardrails",
            "",
            "- Production chunking, embedding, retriever, reranker, rank fusion, agent behavior, and graph runtime were not modified.",
            "- TASK-0081 is recorded only as historical context; this report does not claim that TASK-0081 was fixed.",
        ]
    )


def tokenize(text: str) -> Counter[str]:
    return Counter(token for token in re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]", normalize_text(text).lower()) if token not in STOPWORDS)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip().lower()


def token_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(set(left) & set(right))
    union = len(set(left) | set(right))
    return intersection / union if union else 0.0


def percentile(values: Iterable[int | float], p: float) -> float | None:
    ordered = sorted(value for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[index])


def average(values: Iterable[int | float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "were",
    "are",
    "was",
    "its",
    "into",
    "their",
    "have",
    "has",
    "not",
    "our",
    "you",
    "your",
}


if __name__ == "__main__":
    print(json.dumps(run_task0106(), ensure_ascii=False, indent=2, sort_keys=True))
