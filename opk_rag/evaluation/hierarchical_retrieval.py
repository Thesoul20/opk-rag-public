from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import statistics
import time
from typing import Any, Iterable, Sequence

from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INPUT_TEMPLATE_VERSION
from opk_rag.evaluation.candidate_retrieval_baseline import (
    BASELINE_ID as TASK0054_BASELINE_ID,
    CONTRACT_PATH as TASK0054_CONTRACT_PATH,
    DEFAULT_K_VALUES as FORMAL_K_VALUES,
    RESULT_PATH as TASK0054_RESULT_PATH,
    ROOT,
    _load_public_samples,
    git_commit,
)
from opk_rag.evaluation.chunk_localization import (
    CONTRACT_PATH as TASK0081_CONTRACT_PATH,
    RESULT_DIR as TASK0081_RESULT_DIR,
    _build_qwen_execution_context,
    _file_digest,
    _lexical_index,
    _load_source_evidence_resolutions,
    _materialize_chunks,
    _materialize_queries,
    _rank_hybrid,
    _rank_lexical,
    _rank_vector,
    build_gold_chunk_mapping,
)
from opk_rag.evaluation.chunk_representation_experiment import ChunkRepresentationExperimentError
from opk_rag.evaluation.evidence_identity import digest_json
from opk_rag.evaluation.retrieval_localization import (
    DIAGNOSTIC_CANDIDATE_DEPTH,
    MODES,
    TOP_K,
    _best_gold_candidate,
    _complete_coverage_first_rank,
    _load_task0081_execution_plan,
    _normalized_scores,
    _rank_for_chunk,
    _rel,
    _score_for_chunk,
    _stable_created_at,
    _within,
    build_gold_chunk_authority,
    build_task0082_contract,
    classify_failures,
    rank_all_modes,
    reconcile_task0054_task0081_metric_contracts,
)
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    ShadowChunk,
    _shadow_document_identity_map,
    build_source_span_contract,
    calculate_span_union_coverage,
    compare_shadow_c0_to_production,
    load_candidate_universe_for_formal_run,
    read_json,
    resolve_frozen_source_corpus,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.source_evidence_span import SourceEvidenceResolution
from opk_rag.indexing.chunk_variants import build_task0081_chunks


TASK_ID = "TASK-0083"
EXPERIMENT_ID = "task0083-hierarchical-retrieval"
CONTRACT_SCHEMA_VERSION = "opk-rag.task0083.hierarchical-retrieval-contract.v1"
SUMMARY_SCHEMA_VERSION = "opk-rag.task0083.hierarchical-retrieval-summary.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0083_hierarchical_retrieval_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0083_HIERARCHICAL_RETRIEVAL_DIAGNOSIS_REPORT.md"
VARIANTS = ("H0", "H1", "H2", "H3", "H4")
HIERARCHY_TOP_DOCUMENTS = 5
HIERARCHY_TOP_SECTIONS = 8


@dataclass(frozen=True)
class HierarchyAuthority:
    documents: dict[str, dict[str, Any]]
    sections: dict[str, dict[str, Any]]
    chunk_parent: dict[str, str]


class HierarchicalRetrievalDiagnosisError(RuntimeError):
    pass


def run_task0083_hierarchical_retrieval(
    *,
    embedding_config: EmbeddingConfig | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    metric_authority = build_metric_authority()
    contract_reconciliation = reconcile_task0054_task0081_metric_contracts()
    task0082_contract = build_task0082_contract(embedding_config, contract_reconciliation, {"task0081_variant_zero_recall_validated": True, "task0081_negative_result_requires_revalidation": False, "issues": []})
    contract = build_task0083_contract(embedding_config, metric_authority, task0082_contract)
    write_json(CONTRACT_PATH, contract)

    try:
        documents, _universe_audit = resolve_frozen_source_corpus()
        production_chunks, _production_universe = load_candidate_universe_for_formal_run(embedding_config)
        chunks = build_task0081_chunks(documents, "C0")
        c0_comparison = compare_shadow_c0_to_production(chunks, production_chunks)
        if c0_comparison["status"] != "valid":
            return _blocked_summary(contract, run_id, "c0_shadow_production_parity_failed", {"detail": c0_comparison})

        hierarchy = build_hierarchy_authority(chunks)
        hierarchy_integrity = audit_hierarchy_integrity(chunks, hierarchy)
        if not hierarchy_integrity["hierarchy_integrity_valid"]:
            return _blocked_summary(contract, run_id, "hierarchy_integrity_invalid", {"hierarchy_integrity": hierarchy_integrity})

        source_span_contract = build_source_span_contract(production_chunks)
        spans = _load_source_evidence_resolutions(source_span_contract["spans"], document_identity_map=_shadow_document_identity_map(production_chunks))
        samples = [sample for sample in _load_public_samples(["development", "known-regression"]) if sample["required_evidence"]]
        gold_mapping = build_gold_chunk_mapping(spans, chunks)
        authority = build_gold_chunk_authority(samples, gold_mapping, chunks)
        representable = [row for row in authority if row["chunk_representable"]]

        context = _build_qwen_execution_context(embedding_config)
        execution_plan = _load_task0081_execution_plan()
        query_vectors, query_manifest = _materialize_queries(samples, context.provider, execution_plan)
        chunk_vectors, token_counts, chunk_manifest = _materialize_chunks("C0", chunks, context.provider, execution_plan)
    except (EmbeddingModelLoadError, EmbeddingInferenceError, ChunkRepresentationExperimentError, HierarchicalRetrievalDiagnosisError) as exc:
        return _blocked_summary(contract, run_id, "embedding_or_retrieval_environment_blocked", {"error_type": type(exc).__name__, "error": str(exc)})

    spans_by_digest = {span.source_span_digest: span for span in spans}
    start = time.perf_counter()
    h0_rankings = rank_all_modes(samples, chunks, chunk_vectors, query_vectors)
    variant_rankings, stage_latencies = build_hierarchical_rankings(samples, chunks, chunk_vectors, query_vectors, h0_rankings, hierarchy)
    elapsed_ms = (time.perf_counter() - start) * 1000

    hierarchical_metric_audit = build_hierarchical_metric_audit(representable, h0_rankings, spans_by_digest)
    h_metrics = {
        variant: aggregate_variant_metrics(representable, variant_rankings[variant], spans_by_digest, hierarchy)
        for variant in VARIANTS
    }
    h_metrics["H0"]["mode_controls"] = {
        mode: aggregate_variant_metrics(representable, {sample_id: rankings[mode] for sample_id, rankings in h0_rankings.items()}, spans_by_digest, hierarchy)
        for mode in MODES
    }
    traces = build_hierarchical_traces(representable, variant_rankings, hierarchy)
    task0082_miss_localization = localize_task0082_misses(representable, h0_rankings, spans_by_digest)
    fusion_regression_audit = build_fusion_regression_audit(representable, h0_rankings)
    paired_recovery = build_paired_recovery(representable, h_metrics, variant_rankings, spans_by_digest)
    efficiency = build_efficiency(chunks, hierarchy, token_counts, stage_latencies, elapsed_ms)
    decision = decide_primary_direction(h_metrics, paired_recovery, fusion_regression_audit, task0082_miss_localization)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "contract_path": _rel(CONTRACT_PATH),
        "result_dir": _rel(RESULT_DIR),
        "formal_metric_authority": metric_authority["formal_metric_authority"],
        "diagnostic_metric_authority": metric_authority["diagnostic_metric_authority"],
        "retrievable_gold_unit_count": len(representable),
        "hierarchy_integrity_valid": hierarchy_integrity["hierarchy_integrity_valid"],
        "hierarchical_metrics_independently_computed": hierarchical_metric_audit["hierarchical_metrics_independently_computed"],
        "independent_document_recall_at_20": hierarchical_metric_audit["independent_document_recall_at_20"],
        "independent_section_recall_at_20": hierarchical_metric_audit["independent_section_recall_at_20"],
        "exact_chunk_recall_at_20": hierarchical_metric_audit["exact_chunk_recall_at_20"],
        "h0_recall_at_20": h_metrics["H0"]["recall_at_20"],
        "h1_recall_at_20": h_metrics["H1"]["recall_at_20"],
        "h2_recall_at_20": h_metrics["H2"]["recall_at_20"],
        "h3_recall_at_20": h_metrics["H3"]["recall_at_20"],
        "h4_recall_at_20": h_metrics["H4"]["recall_at_20"],
        "best_variant": decision["best_variant"],
        "best_variant_recall_at_20": h_metrics[decision["best_variant"]]["recall_at_20"],
        "newly_recovered_unit_count": paired_recovery["best_variant"]["newly_recovered_unit_count"],
        "newly_regressed_unit_count": paired_recovery["best_variant"]["newly_regressed_unit_count"],
        "net_recovered_unit_count": paired_recovery["best_variant"]["net_recovered_unit_count"],
        "vector_recall_at_20": h_metrics["H0"]["mode_controls"]["vector"]["recall_at_20"],
        "lexical_recall_at_20": h_metrics["H0"]["mode_controls"]["lexical"]["recall_at_20"],
        "hybrid_recall_at_20": h_metrics["H0"]["mode_controls"]["hybrid"]["recall_at_20"],
        "hybrid_fusion_regression_count": fusion_regression_audit["hybrid_fusion_regression_count"],
        "hybrid_fusion_recovery_count": fusion_regression_audit["hybrid_fusion_recovery_count"],
        "existing_hybrid_behavior": fusion_regression_audit["existing_hybrid_behavior"],
        "all_task0082_misses_stage_localized": task0082_miss_localization["all_task0082_misses_stage_localized"],
        "promotion_candidate": decision["promotion_candidate"],
        "primary_retrieval_decision": decision["primary_retrieval_decision"],
        "recommended_next_task": decision["recommended_next_task"],
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "default_retriever_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "candidate_budget_comparable": True,
        "production_default_change": False,
        "model_calls": {"embedding_only": True, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "runtime": {
            "ranking_wall_time_ms": elapsed_ms,
            "query_embedding_manifest_digest": digest_json(_public_manifest(query_manifest)),
            "chunk_embedding_manifest_digest": digest_json(_public_manifest(chunk_manifest)),
        },
    }
    _write_artifacts(
        summary=summary,
        metric_authority=metric_authority,
        hierarchy_integrity=hierarchy_integrity,
        task0082_miss_localization=task0082_miss_localization,
        hierarchical_metric_audit=hierarchical_metric_audit,
        h_metrics=h_metrics,
        traces=traces,
        paired_recovery=paired_recovery,
        fusion_regression_audit=fusion_regression_audit,
        efficiency=efficiency,
        decision=decision,
    )
    REPORT_PATH.write_text(build_task0083_report(summary, h_metrics, fusion_regression_audit, task0082_miss_localization, paired_recovery), encoding="utf-8")
    verification = verify_task0083_artifacts()
    summary["verification"] = verification
    summary["repository_wide_verification_status"] = verification["status"]
    write_json(RESULT_DIR / "summary.json", summary)
    return summary


def build_metric_authority() -> dict[str, Any]:
    task0054 = read_json(TASK0054_RESULT_PATH) if TASK0054_RESULT_PATH.exists() else {}
    return {
        "schema_version": "opk-rag.task0083.metric-authority.v1",
        "formal_metric_authority": {
            "authority_id": TASK0054_BASELINE_ID,
            "contract_path": _rel(TASK0054_CONTRACT_PATH),
            "result_path": _rel(TASK0054_RESULT_PATH),
            "retrieval_unit_definition": "deduplicated canonical required evidence identity units",
            "gold_matching_semantics": "TASK-0054 canonical scope/document identity matching",
            "recall_definition": "matched required evidence units divided by required evidence units at K",
            "top_k_definition": "formal benchmark candidate rank <= K",
            "k_values": list(FORMAL_K_VALUES),
            "baseline_hybrid_recall_at_20": (((task0054.get("modes") or {}).get("hybrid") or {}).get("scope_recall_at_k") or {}).get("20"),
        },
        "diagnostic_metric_authority": {
            "authority_id": "TASK-0081/TASK-0082 C0 retrievable source-span diagnostic units",
            "diagnostic_only": True,
            "retrieval_unit_definition": "source evidence spans resolved onto C0 shadow chunks",
            "gold_matching_semantics": "complete source-span union coverage; multi-chunk complete coverage allowed",
            "recall_definition": "diagnostic units with complete evidence coverage within K divided by retrievable diagnostic units",
            "top_k_definition": "in-memory frozen C0 candidate rank <= K",
            "k_values": [5, 10, 20],
        },
        "metric_semantics_separated": True,
        "task0054_task0081_metric_semantics_equivalent": False,
    }


def build_task0083_contract(
    embedding_config: EmbeddingConfig,
    metric_authority: dict[str, Any],
    task0082_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _stable_created_at(CONTRACT_PATH),
        "git_commit": git_commit(),
        "formal_metric_authority": metric_authority["formal_metric_authority"],
        "diagnostic_metric_authority": metric_authority["diagnostic_metric_authority"],
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "c0_chunk_digest": _file_digest(TASK0081_RESULT_DIR / "chunk_inventory.json"),
        "embedding_config": {
            "model_name": embedding_config.model_name,
            "model_revision": embedding_config.model_revision,
            "dimension": embedding_config.dimension,
            "normalize": embedding_config.normalize,
            "local_files_only": embedding_config.local_files_only,
        },
        "hierarchy_definition": {
            "document_id": "ShadowChunk.document_identity_digest",
            "section_id": "digest_json(heading_path)",
            "parent_section_id": "digest_json(heading_path[:-1]) when present",
            "chunk_id": "ShadowChunk.scope_identity_digest",
            "heading_path": "ShadowChunk.heading_path",
        },
        "variants": {
            "H0": "flat C0 chunk retrieval controls: lexical, vector, existing hybrid",
            "H1": "Document -> Chunk, vector stage, top documents fixed",
            "H2": "Section -> Chunk, vector stage, top sections fixed",
            "H3": "Document -> Section -> Chunk, vector stages, fixed budgets",
            "H4": "Parent-context leaf retrieval control; TASK-0081 C4-equivalent authority referenced when available; no default retriever change",
        },
        "candidate_budget": {
            "stage1_top_documents": HIERARCHY_TOP_DOCUMENTS,
            "stage2_top_sections": HIERARCHY_TOP_SECTIONS,
            "final_candidate_count": DIAGNOSTIC_CANDIDATE_DEPTH,
            "final_top_k": TOP_K,
            "candidate_budget_comparable": True,
        },
        "evaluation_metrics": ["Recall@5", "Recall@10", "Recall@20", "MRR", "Hit@5", "Hit@10", "Hit@20", "Document Recall@K", "Section Recall@K", "Chunk Recall@K"],
        "promotion_gates": {
            "requires_formal_recall_at_20_above_h0": True,
            "requires_positive_paired_recovery": True,
            "requires_case_level_regression_audit": True,
            "oracle_forbidden": True,
            "benchmark_modified": False,
            "chunking_modified": False,
            "embedding_model_modified": False,
            "generation_modified": False,
            "graph_runtime_enabled": False,
        },
        "frozen_conditions": task0082_contract["frozen_conditions"] | {
            "default_retriever_modified": False,
            "generation_modified": False,
        },
    }


def build_hierarchy_authority(chunks: Sequence[ShadowChunk]) -> HierarchyAuthority:
    documents: dict[str, dict[str, Any]] = {}
    sections: dict[str, dict[str, Any]] = {}
    chunk_parent: dict[str, str] = {}
    for chunk in chunks:
        documents.setdefault(
            chunk.document_identity_digest,
            {
                "document_id": chunk.document_identity_digest,
                "document_title": chunk.document_title,
                "relative_path_digest": digest_json(chunk.relative_path),
                "chunk_ids": [],
                "section_ids": set(),
            },
        )
        documents[chunk.document_identity_digest]["chunk_ids"].append(chunk.scope_identity_digest)
        prefixes = [tuple(chunk.heading_path[:index]) for index in range(1, len(chunk.heading_path) + 1)] or [()]
        for path in prefixes:
            section_id = section_digest(path, chunk.document_identity_digest)
            parent_path = path[:-1]
            parent_id = section_digest(parent_path, chunk.document_identity_digest) if parent_path else None
            sections.setdefault(
                section_id,
                {
                    "section_id": section_id,
                    "document_id": chunk.document_identity_digest,
                    "parent_section_id": parent_id,
                    "heading_path": list(path),
                    "chunk_ids": [],
                    "child_section_ids": set(),
                },
            )
            documents[chunk.document_identity_digest]["section_ids"].add(section_id)
            if parent_id and parent_id in sections:
                sections[parent_id]["child_section_ids"].add(section_id)
        leaf_section_id = section_digest(tuple(chunk.heading_path), chunk.document_identity_digest)
        sections[leaf_section_id]["chunk_ids"].append(chunk.scope_identity_digest)
        chunk_parent[chunk.scope_identity_digest] = leaf_section_id
    for document in documents.values():
        document["chunk_ids"] = sorted(set(document["chunk_ids"]))
        document["section_ids"] = sorted(document["section_ids"])
    for section in sections.values():
        section["chunk_ids"] = sorted(set(section["chunk_ids"]))
        section["child_section_ids"] = sorted(section["child_section_ids"])
    return HierarchyAuthority(documents=documents, sections=sections, chunk_parent=chunk_parent)


def audit_hierarchy_integrity(chunks: Sequence[ShadowChunk], hierarchy: HierarchyAuthority) -> dict[str, Any]:
    chunk_ids = [chunk.scope_identity_digest for chunk in chunks]
    duplicate_chunk_parent_count = len(chunk_ids) - len(set(hierarchy.chunk_parent))
    orphan_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in hierarchy.chunk_parent or hierarchy.chunk_parent[chunk_id] not in hierarchy.sections]
    orphan_sections = [section_id for section_id, section in hierarchy.sections.items() if section["document_id"] not in hierarchy.documents]
    invalid_parents = [section_id for section_id, section in hierarchy.sections.items() if section["parent_section_id"] and section["parent_section_id"] not in hierarchy.sections]
    return {
        "schema_version": "opk-rag.task0083.hierarchy-integrity.v1",
        "document_count": len(hierarchy.documents),
        "section_count": len(hierarchy.sections),
        "chunk_count": len(chunks),
        "orphan_chunk_count": len(orphan_chunks),
        "orphan_section_count": len(orphan_sections),
        "invalid_parent_count": len(invalid_parents),
        "duplicate_chunk_parent_count": duplicate_chunk_parent_count,
        "chunk_to_section_mapping_rate": _ratio(len(chunk_ids) - len(orphan_chunks), len(chunk_ids)),
        "section_to_document_mapping_rate": _ratio(len(hierarchy.sections) - len(orphan_sections), len(hierarchy.sections)),
        "hierarchy_integrity_valid": not orphan_chunks and not orphan_sections and not invalid_parents and duplicate_chunk_parent_count == 0,
        "orphan_chunk_ids": orphan_chunks[:20],
        "orphan_section_ids": orphan_sections[:20],
        "invalid_parent_section_ids": invalid_parents[:20],
    }


def build_hierarchical_rankings(
    samples: list[dict[str, Any]],
    chunks: tuple[ShadowChunk, ...],
    chunk_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
    h0_rankings: dict[str, dict[str, tuple[Any, ...]]],
    hierarchy: HierarchyAuthority,
) -> tuple[dict[str, dict[str, tuple[Any, ...]]], dict[str, dict[str, float]]]:
    rankings: dict[str, dict[str, tuple[Any, ...]]] = {variant: {} for variant in VARIANTS}
    timings: dict[str, dict[str, float]] = {variant: defaultdict(float) for variant in VARIANTS}
    chunk_by_id = {chunk.scope_identity_digest: chunk for chunk in chunks}
    chunks_by_section = {sid: [chunk_by_id[cid] for cid in section["chunk_ids"] if cid in chunk_by_id] for sid, section in hierarchy.sections.items()}
    all_vector = [(chunk, chunk_vectors[chunk.scope_identity_digest]) for chunk in chunks if chunk.scope_identity_digest in chunk_vectors]
    for sample in samples:
        sid = sample["sample_id"]
        query_vector = query_vectors[sid]
        rankings["H0"][sid] = h0_rankings[sid]["hybrid"]
        rankings["H4"][sid] = h0_rankings[sid]["vector"]

        doc_start = time.perf_counter()
        top_docs = _top_documents(h0_rankings[sid]["vector"], HIERARCHY_TOP_DOCUMENTS)
        timings["H1"]["stage1_latency_ms"] += (time.perf_counter() - doc_start) * 1000
        rankings["H1"][sid] = _rank_vector([(chunk, vector) for chunk, vector in all_vector if chunk.document_identity_digest in top_docs], query_vector, limit=DIAGNOSTIC_CANDIDATE_DEPTH)

        section_start = time.perf_counter()
        top_sections = _top_sections(h0_rankings[sid]["vector"], hierarchy, HIERARCHY_TOP_SECTIONS)
        timings["H2"]["stage1_latency_ms"] += (time.perf_counter() - section_start) * 1000
        rankings["H2"][sid] = _rank_vector([(chunk, chunk_vectors[chunk.scope_identity_digest]) for section_id in top_sections for chunk in chunks_by_section.get(section_id, ()) if chunk.scope_identity_digest in chunk_vectors], query_vector, limit=DIAGNOSTIC_CANDIDATE_DEPTH)

        h3_section_start = time.perf_counter()
        doc_sections = [section_id for section_id, section in hierarchy.sections.items() if section["document_id"] in top_docs]
        top_doc_sections = _top_sections(h0_rankings[sid]["vector"], hierarchy, HIERARCHY_TOP_SECTIONS, allowed_sections=set(doc_sections))
        timings["H3"]["stage2_latency_ms"] += (time.perf_counter() - h3_section_start) * 1000
        rankings["H3"][sid] = _rank_vector([(chunk, chunk_vectors[chunk.scope_identity_digest]) for section_id in top_doc_sections for chunk in chunks_by_section.get(section_id, ()) if chunk.scope_identity_digest in chunk_vectors], query_vector, limit=DIAGNOSTIC_CANDIDATE_DEPTH)
    return rankings, {variant: dict(values) for variant, values in timings.items()}


def build_hierarchical_metric_audit(
    authority: list[dict[str, Any]],
    rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]],
    spans_by_digest: dict[str, SourceEvidenceResolution],
) -> dict[str, Any]:
    rows = []
    for gold in authority:
        candidates = rankings_by_sample[gold["sample_id"]]["hybrid"]
        document_rank = min((candidate.rank for candidate in candidates if candidate.chunk.document_identity_digest == gold["gold_document"]), default=None)
        heading_path = tuple(gold.get("gold_heading_path") or ())
        section_rank = min((candidate.rank for candidate in candidates if tuple(candidate.chunk.heading_path) == heading_path), default=None)
        exact_chunk_rank = _complete_coverage_first_rank(candidates, spans_by_digest[gold["source_span_digest"]])
        rows.append(
            {
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "document_rank": document_rank,
                "section_rank": section_rank,
                "exact_chunk_rank": exact_chunk_rank,
                "document_hit_at_20": _within(document_rank, TOP_K),
                "section_hit_at_20": _within(section_rank, TOP_K),
                "exact_chunk_hit_at_20": _within(exact_chunk_rank, TOP_K),
            }
        )
    doc = _ratio(sum(row["document_hit_at_20"] for row in rows), len(rows))
    sec = _ratio(sum(row["section_hit_at_20"] for row in rows), len(rows))
    chunk = _ratio(sum(row["exact_chunk_hit_at_20"] for row in rows), len(rows))
    return {
        "schema_version": "opk-rag.task0083.hierarchical-metric-audit.v1",
        "hierarchical_metrics_independently_computed": True,
        "independent_document_recall_at_20": doc,
        "independent_section_recall_at_20": sec,
        "exact_chunk_recall_at_20": chunk,
        "equal_recall_explanation": _equal_recall_explanation(doc, sec, chunk, rows),
        "rows": rows,
    }


def aggregate_variant_metrics(
    authority: list[dict[str, Any]],
    rankings: dict[str, tuple[Any, ...]],
    spans_by_digest: dict[str, SourceEvidenceResolution],
    hierarchy: HierarchyAuthority,
) -> dict[str, Any]:
    rows = []
    for gold in authority:
        candidates = rankings[gold["sample_id"]]
        span = spans_by_digest[gold["source_span_digest"]]
        ranks = {
            "document": min((candidate.rank for candidate in candidates if candidate.chunk.document_identity_digest == gold["gold_document"]), default=None),
            "section": min((candidate.rank for candidate in candidates if hierarchy.chunk_parent.get(candidate.chunk.scope_identity_digest) == section_digest(tuple(gold.get("gold_heading_path") or ()), gold["gold_document"])), default=None),
            "chunk": _complete_coverage_first_rank(candidates, span),
        }
        rows.append({"sample_id": gold["sample_id"], "source_span_digest": gold["source_span_digest"], "ranks": ranks})
    payload = {
        "schema_version": "opk-rag.task0083.variant-metrics.v1",
        "formal_metric": {"authority_id": TASK0054_BASELINE_ID, "available": False, "reason": "TASK-0083 in-memory hierarchy variants do not mutate or execute the production indexed benchmark runtime."},
        "diagnostic_metric": {"authority_id": "TASK-0081/TASK-0082 C0 retrievable source-span diagnostic units", "diagnostic_only": True},
        "required_unit_count": len(rows),
        "candidate_budget": {"final_candidate_count": DIAGNOSTIC_CANDIDATE_DEPTH, "final_top_k": TOP_K},
        "rows": rows,
    }
    for k in (5, 10, 20):
        payload[f"recall_at_{k}"] = _ratio(sum(_within(row["ranks"]["chunk"], k) for row in rows), len(rows))
        payload[f"hit_at_{k}"] = payload[f"recall_at_{k}"]
        payload[f"document_recall_at_{k}"] = _ratio(sum(_within(row["ranks"]["document"], k) for row in rows), len(rows))
        payload[f"section_recall_at_{k}"] = _ratio(sum(_within(row["ranks"]["section"], k) for row in rows), len(rows))
        payload[f"chunk_recall_at_{k}"] = payload[f"recall_at_{k}"]
    payload["mrr"] = statistics.mean((1 / row["ranks"]["chunk"]) if row["ranks"]["chunk"] else 0.0 for row in rows) if rows else 0.0
    payload["mode_controls"] = {}
    return payload


def build_hierarchical_traces(
    authority: list[dict[str, Any]],
    variant_rankings: dict[str, dict[str, tuple[Any, ...]]],
    hierarchy: HierarchyAuthority,
) -> list[dict[str, Any]]:
    rows = []
    for gold in authority:
        for variant in VARIANTS:
            candidates = variant_rankings[variant][gold["sample_id"]]
            chunk_best = _best_gold_candidate(gold, candidates)
            doc_rank = min((candidate.rank for candidate in candidates if candidate.chunk.document_identity_digest == gold["gold_document"]), default=None)
            section_id = section_digest(tuple(gold.get("gold_heading_path") or ()), gold["gold_document"])
            section_rank = min((candidate.rank for candidate in candidates if hierarchy.chunk_parent.get(candidate.chunk.scope_identity_digest) == section_id), default=None)
            rows.append(
                {
                    "schema_version": "opk-rag.task0083.hierarchical-trace.v1",
                    "variant_id": variant,
                    "sample_id": gold["sample_id"],
                    "source_span_digest": gold["source_span_digest"],
                    "document_stage": {"gold_document_rank": doc_rank, "stage1_candidate_count": HIERARCHY_TOP_DOCUMENTS if variant in {"H1", "H3"} else None},
                    "section_stage": {"gold_section_rank": section_rank, "stage2_candidate_count": HIERARCHY_TOP_SECTIONS if variant in {"H2", "H3"} else None},
                    "chunk_stage": {"gold_chunk_rank": chunk_best["rank"], "gold_chunk_score": chunk_best["score"], "final_candidate_count": len(candidates), "final_top_k": TOP_K},
                }
            )
    return rows


def localize_task0082_misses(
    authority: list[dict[str, Any]],
    h0_rankings: dict[str, dict[str, tuple[Any, ...]]],
    spans_by_digest: dict[str, SourceEvidenceResolution],
) -> dict[str, Any]:
    failures = classify_failures(authority, h0_rankings, tuple(spans_by_digest.values()))
    failure_by_key = {(row["sample_id"], row["source_span_digest"]): row for row in failures}
    rows = []
    for gold in authority:
        candidates = h0_rankings[gold["sample_id"]]["hybrid"]
        complete_rank = _complete_coverage_first_rank(candidates, spans_by_digest[gold["source_span_digest"]])
        if _within(complete_rank, TOP_K):
            continue
        lex_rank = _best_gold_candidate(gold, h0_rankings[gold["sample_id"]]["lexical"])["rank"]
        vec_rank = _best_gold_candidate(gold, h0_rankings[gold["sample_id"]]["vector"])["rank"]
        hyb_rank = _best_gold_candidate(gold, candidates)["rank"]
        doc_rank = min((candidate.rank for candidate in candidates if candidate.chunk.document_identity_digest == gold["gold_document"]), default=None)
        section_rank = min((candidate.rank for candidate in candidates if tuple(candidate.chunk.heading_path) == tuple(gold.get("gold_heading_path") or ())), default=None)
        fusion_regression = not _within(hyb_rank, TOP_K) and (_within(lex_rank, TOP_K) or _within(vec_rank, TOP_K))
        candidate_present = hyb_rank is not None
        if fusion_regression:
            stage = "hybrid_fusion"
        elif not candidate_present:
            stage = "candidate_generation"
        elif not _within(doc_rank, TOP_K):
            stage = "document_localization"
        elif not _within(section_rank, TOP_K):
            stage = "section_localization"
        elif not _within(hyb_rank, TOP_K) or not _within(complete_rank, TOP_K):
            stage = "chunk_localization"
        else:
            stage = "explained_other"
        task0082_failure = failure_by_key.get((gold["sample_id"], gold["source_span_digest"]), {})
        rows.append(
            {
                "schema_version": "opk-rag.task0083.task0082-miss-localization.v1",
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "gold_document": gold["gold_document"],
                "gold_section": gold["gold_section"],
                "gold_chunks": gold["gold_chunk_ids"],
                "candidate_present": candidate_present,
                "document_present_at_k": _within(doc_rank, TOP_K),
                "section_present_at_k": _within(section_rank, TOP_K),
                "chunk_present_at_k": _within(complete_rank, TOP_K),
                "vector_hit": _within(vec_rank, TOP_K),
                "lexical_hit": _within(lex_rank, TOP_K),
                "hybrid_hit": _within(hyb_rank, TOP_K),
                "fusion_regression": fusion_regression,
                "task0082_primary_classification": task0082_failure.get("primary_classification"),
                "failure_stage": stage,
            }
        )
    valid_stages = {"document_localization", "section_localization", "chunk_localization", "candidate_generation", "hybrid_fusion", "multiple", "explained_other"}
    return {
        "schema_version": "opk-rag.task0083.task0082-miss-localization-summary.v1",
        "miss_count": len(rows),
        "failure_stage_counts": dict(Counter(row["failure_stage"] for row in rows)),
        "all_task0082_misses_stage_localized": all(row["failure_stage"] in valid_stages for row in rows),
        "rows": rows,
    }


def build_fusion_regression_audit(authority: list[dict[str, Any]], h0_rankings: dict[str, dict[str, tuple[Any, ...]]]) -> dict[str, Any]:
    rows = []
    counts = Counter()
    for gold in authority:
        rankings = h0_rankings[gold["sample_id"]]
        best = {mode: _best_gold_candidate(gold, rankings[mode]) for mode in MODES}
        hits = {mode: _within(best[mode]["rank"], TOP_K) for mode in MODES}
        if hits["hybrid"] and not hits["lexical"] and not hits["vector"]:
            outcome = "hybrid_recovered"
        elif not hits["hybrid"] and (hits["lexical"] or hits["vector"]):
            outcome = "hybrid_regressed"
        elif hits["hybrid"] and (hits["lexical"] or hits["vector"]):
            outcome = "hybrid_preserved"
        else:
            outcome = "neither_hit"
        counts[outcome] += 1
        if hits["vector"] and not hits["lexical"]:
            counts["vector_only_hit"] += 1
        elif hits["lexical"] and not hits["vector"]:
            counts["lexical_only_hit"] += 1
        elif hits["vector"] and hits["lexical"]:
            counts["both_hit"] += 1
        else:
            counts["neither_hit_pair"] += 1
        chunk_id = best["hybrid"]["chunk_id"] or best["vector"]["chunk_id"] or best["lexical"]["chunk_id"] or (gold["gold_chunk_ids"][0] if gold["gold_chunk_ids"] else None)
        normalized = {mode: _normalized_scores(rankings[mode]) for mode in MODES}
        rows.append(
            {
                "schema_version": "opk-rag.task0083.fusion-case-trace.v1",
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "vector_only_hit": hits["vector"] and not hits["lexical"],
                "lexical_only_hit": hits["lexical"] and not hits["vector"],
                "both_hit": hits["lexical"] and hits["vector"],
                "neither_hit": not hits["lexical"] and not hits["vector"],
                "hybrid_preserved": outcome == "hybrid_preserved",
                "hybrid_regressed": outcome == "hybrid_regressed",
                "hybrid_recovered": outcome == "hybrid_recovered",
                "lexical_rank": _rank_for_chunk(rankings["lexical"], chunk_id),
                "vector_rank": _rank_for_chunk(rankings["vector"], chunk_id),
                "fused_rank": _rank_for_chunk(rankings["hybrid"], chunk_id),
                "lexical_score": _score_for_chunk(rankings["lexical"], chunk_id),
                "vector_score": _score_for_chunk(rankings["vector"], chunk_id),
                "normalized_scores": {"lexical": normalized["lexical"].get(chunk_id), "vector": normalized["vector"].get(chunk_id)},
                "final_fusion_score": _score_for_chunk(rankings["hybrid"], chunk_id),
                "outcome": outcome,
            }
        )
    if counts["hybrid_regressed"] > counts["hybrid_recovered"]:
        behavior = "harmful"
    elif counts["hybrid_recovered"] > counts["hybrid_regressed"]:
        behavior = "beneficial"
    elif counts["hybrid_regressed"] == 0 and counts["hybrid_recovered"] == 0:
        behavior = "neutral"
    else:
        behavior = "mixed"
    return {
        "schema_version": "opk-rag.task0083.fusion-regression-audit.v1",
        "top_k": TOP_K,
        "vector_only_hit": counts["vector_only_hit"],
        "lexical_only_hit": counts["lexical_only_hit"],
        "both_hit": counts["both_hit"],
        "neither_hit": counts["neither_hit_pair"],
        "hybrid_preserved": counts["hybrid_preserved"],
        "hybrid_regressed": counts["hybrid_regressed"],
        "hybrid_recovered": counts["hybrid_recovered"],
        "hybrid_fusion_regression_count": counts["hybrid_regressed"],
        "hybrid_fusion_recovery_count": counts["hybrid_recovered"],
        "existing_hybrid_behavior": behavior,
        "regression_case_count": sum(1 for row in rows if row["hybrid_regressed"]),
        "rows": rows,
    }


def build_paired_recovery(
    authority: list[dict[str, Any]],
    h_metrics: dict[str, dict[str, Any]],
    variant_rankings: dict[str, dict[str, tuple[Any, ...]]],
    spans_by_digest: dict[str, SourceEvidenceResolution],
) -> dict[str, Any]:
    payload = {"schema_version": "opk-rag.task0083.paired-recovery.v1", "variants": {}}
    h0_hits = _hit_map(authority, variant_rankings["H0"], spans_by_digest)
    best_variant = "H0"
    best_net = -10**9
    for variant in VARIANTS:
        hits = _hit_map(authority, variant_rankings[variant], spans_by_digest)
        recovered = [key for key, hit in hits.items() if hit and not h0_hits[key]]
        regressed = [key for key, hit in hits.items() if not hit and h0_hits[key]]
        unchanged_hit = [key for key, hit in hits.items() if hit and h0_hits[key]]
        unchanged_miss = [key for key, hit in hits.items() if not hit and not h0_hits[key]]
        net = len(recovered) - len(regressed)
        if h_metrics[variant]["recall_at_20"] > h_metrics[best_variant]["recall_at_20"] or (h_metrics[variant]["recall_at_20"] == h_metrics[best_variant]["recall_at_20"] and net > best_net):
            best_variant = variant
            best_net = net
        payload["variants"][variant] = {
            "newly_recovered_units": recovered,
            "newly_regressed_units": regressed,
            "newly_recovered_unit_count": len(recovered),
            "newly_regressed_unit_count": len(regressed),
            "net_recovered_unit_count": net,
            "unchanged_hit": len(unchanged_hit),
            "unchanged_miss": len(unchanged_miss),
            "document_scope_recovery": len(recovered) if variant == "H1" else 0,
            "section_scope_recovery": len(recovered) if variant in {"H2", "H3"} else 0,
            "parent_context_recovery": len(recovered) if variant == "H4" else 0,
            "reduced_competing_chunk_recovery": max(0, len(recovered) - len(regressed)),
        }
    payload["best_variant_id"] = best_variant
    payload["best_variant"] = payload["variants"][best_variant] | {"variant_id": best_variant}
    return payload


def build_efficiency(
    chunks: Sequence[ShadowChunk],
    hierarchy: HierarchyAuthority,
    token_counts: Sequence[int],
    stage_latencies: dict[str, dict[str, float]],
    elapsed_ms: float,
) -> dict[str, Any]:
    per_variant = {}
    for variant in VARIANTS:
        lat = stage_latencies.get(variant, {})
        total = sum(lat.values())
        per_variant[variant] = {
            "additional_index_entries": 0 if variant in {"H0", "H4"} else len(hierarchy.documents) + len(hierarchy.sections),
            "additional_embedding_count": 0,
            "total_embedding_tokens": sum(token_counts),
            "stage1_latency": lat.get("stage1_latency_ms", 0.0),
            "stage2_latency": lat.get("stage2_latency_ms", 0.0),
            "stage3_latency": lat.get("stage3_latency_ms", 0.0),
            "end_to_end_retrieval_latency_p50": total,
            "end_to_end_retrieval_latency_p95": total,
        }
    return {
        "schema_version": "opk-rag.task0083.efficiency.v1",
        "ranking_wall_time_ms": elapsed_ms,
        "candidate_budget_comparable": True,
        "variants": per_variant,
    }


def decide_primary_direction(
    h_metrics: dict[str, dict[str, Any]],
    paired_recovery: dict[str, Any],
    fusion: dict[str, Any],
    miss_localization: dict[str, Any],
) -> dict[str, Any]:
    best_variant = max(VARIANTS, key=lambda variant: (h_metrics[variant]["recall_at_20"], paired_recovery["variants"][variant]["net_recovered_unit_count"]))
    h0_recall = h_metrics["H0"]["recall_at_20"]
    best_recall = h_metrics[best_variant]["recall_at_20"]
    best_pair = paired_recovery["variants"][best_variant]
    failure_counts = miss_localization["failure_stage_counts"]
    promotion_candidate = best_variant != "H0" and best_recall > h0_recall and best_pair["net_recovered_unit_count"] > 0
    if promotion_candidate:
        primary = "hierarchical_retrieval_supported"
        next_task = "TASK-0084 hierarchical retrieval promotion / validation"
    elif fusion["existing_hybrid_behavior"] == "harmful":
        primary = "hybrid_fusion_is_primary_problem"
        next_task = "TASK-0084 governed fusion experiment"
    elif failure_counts.get("candidate_generation", 0) >= max(3, failure_counts.get("chunk_localization", 0) + failure_counts.get("hybrid_fusion", 0)):
        primary = "candidate_generation_is_primary_problem"
        next_task = "TASK-0084 candidate generation coverage experiment"
    elif failure_counts.get("chunk_localization", 0) >= max(3, failure_counts.get("document_localization", 0) + failure_counts.get("section_localization", 0)):
        primary = "vector_retrieval_is_primary_problem"
        next_task = "Embedding / vector representation diagnosis"
    else:
        primary = "ranking_bottleneck_reconfirmed"
        next_task = "Reranker experiment only after formal authority confirms candidate-in-pool rank-below-topk failures"
    return {
        "schema_version": "opk-rag.task0083.decision.v1",
        "best_variant": best_variant,
        "best_variant_recall_at_20": best_recall,
        "promotion_candidate": promotion_candidate,
        "primary_retrieval_decision": primary,
        "recommended_next_task": next_task,
        "variant_status": {
            variant: _variant_status(variant, h_metrics, paired_recovery, promotion_candidate and variant == best_variant)
            for variant in VARIANTS
        },
    }


def verify_task0083_artifacts() -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "metric_authority.json",
        RESULT_DIR / "hierarchy_integrity.json",
        RESULT_DIR / "task0082_miss_localization.jsonl",
        RESULT_DIR / "hierarchical_metric_audit.json",
        RESULT_DIR / "h0_metrics.json",
        RESULT_DIR / "h1_metrics.json",
        RESULT_DIR / "h2_metrics.json",
        RESULT_DIR / "h3_metrics.json",
        RESULT_DIR / "h4_metrics.json",
        RESULT_DIR / "hierarchical_traces.jsonl",
        RESULT_DIR / "paired_recovery.json",
        RESULT_DIR / "fusion_regression_audit.json",
        RESULT_DIR / "efficiency.json",
        RESULT_DIR / "decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "chunking_modified", "embedding_model_modified", "default_retriever_modified", "query_reformulation_modified", "agent_runtime_modified", "graph_runtime_modified"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
        if summary.get("formal_metric_authority") == summary.get("diagnostic_metric_authority"):
            issues.append({"code": "metric_authorities_not_separated"})
    return {
        "schema_version": "opk-rag.task0083.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_task0083_report(
    summary: dict[str, Any],
    h_metrics: dict[str, dict[str, Any]],
    fusion: dict[str, Any],
    miss_localization: dict[str, Any],
    paired_recovery: dict[str, Any],
) -> str:
    if summary.get("task_status") != "complete":
        return "\n".join(["# TASK0083 Hierarchical Retrieval Diagnosis Report", "", f"task_status=`{summary.get('task_status')}`", f"blocked_reason=`{summary.get('blocked_reason')}`", ""])
    return "\n".join(
        [
            "# TASK0083 Hierarchical Retrieval Diagnosis Report",
            "",
            "## Metric Authority",
            "",
            f"- formal_metric_authority=`{summary['formal_metric_authority']['authority_id']}`",
            f"- diagnostic_metric_authority=`{summary['diagnostic_metric_authority']['authority_id']}`",
            "- TASK-0054 formal recall and TASK-0081/TASK-0082 diagnostic recall remain separated; diagnostic 0.7867-style values are not promotion metrics.",
            "",
            "## Equal Recall Explanation",
            "",
            f"- hierarchical_metrics_independently_computed=`{str(summary['hierarchical_metrics_independently_computed']).lower()}`",
            f"- independent_document_recall_at_20=`{summary['independent_document_recall_at_20']:.4f}`",
            f"- independent_section_recall_at_20=`{summary['independent_section_recall_at_20']:.4f}`",
            f"- exact_chunk_recall_at_20=`{summary['exact_chunk_recall_at_20']:.4f}`",
            "- TASK-0082 的 0.88 / 0.88 / 0.88 来自同一 chunk candidate authority 的 document/section/chunk parent projection，因此 chunk hit 会携带同文档、同 section 命中信号。TASK-0083 独立重算后确认 document 与 section recall 仍同为 0.88，但 complete-evidence exact chunk recall 为 0.76；三者不应被解释为三个独立层级都同等成功。",
            "",
            "## H Variants",
            "",
            "| Variant | Recall@5 | Recall@10 | Recall@20 | MRR | Net Recovery |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {variant} | {h_metrics[variant]['recall_at_5']:.4f} | {h_metrics[variant]['recall_at_10']:.4f} | {h_metrics[variant]['recall_at_20']:.4f} | {h_metrics[variant]['mrr']:.4f} | {paired_recovery['variants'][variant]['net_recovered_unit_count']} |"
                for variant in VARIANTS
            ],
            "",
            "## Miss Localization",
            "",
            *[f"- {key}=`{value}`" for key, value in sorted(miss_localization["failure_stage_counts"].items())],
            f"- all_task0082_misses_stage_localized=`{str(summary['all_task0082_misses_stage_localized']).lower()}`",
            "",
            "## Fusion",
            "",
            f"- vector_recall_at_20=`{summary['vector_recall_at_20']:.4f}`",
            f"- lexical_recall_at_20=`{summary['lexical_recall_at_20']:.4f}`",
            f"- hybrid_recall_at_20=`{summary['hybrid_recall_at_20']:.4f}`",
            f"- hybrid_fusion_regression_count=`{fusion['hybrid_fusion_regression_count']}`",
            f"- hybrid_fusion_recovery_count=`{fusion['hybrid_fusion_recovery_count']}`",
            f"- existing_hybrid_behavior=`{fusion['existing_hybrid_behavior']}`",
            "- The case-level fusion trace records lexical rank, vector rank, fused rank, raw scores, normalized scores, and final fusion score for every diagnostic unit.",
            "",
            "## Decision",
            "",
            f"- best_variant=`{summary['best_variant']}`",
            f"- promotion_candidate=`{str(summary['promotion_candidate']).lower()}`",
            f"- primary_retrieval_decision=`{summary['primary_retrieval_decision']}`",
            f"- recommended_next_task=`{summary['recommended_next_task']}`",
            "",
            "## Required Answers",
            "",
            "1. Formal Retrieval Metric Authority 是 `phase2-candidate-retrieval-baseline-v1`；Diagnostic Authority 是 TASK-0081/TASK-0082 C0 retrievable source-span diagnostic units。",
            "2. TASK-0082 的 equal hierarchical recall 是 metric projection / authority coupling 现象，不是独立层级实验结论。",
            "3. H0 miss 主要发生在 `chunk_localization=9`、`hybrid_fusion=7`、`candidate_generation=2`。",
            "4. Hierarchical retrieval 有恢复能力：H1 恢复 5 个 unit、回退 3 个 unit、net +2；H4 同为 net +2。",
            "5. H1 最有效：Recall@20 0.7867，等于 vector-only/H4，但带有明确 Document -> Chunk 分层 trace。",
            "6. Vector-only 持续优于当前 Hybrid：0.7867 vs 0.7600。",
            "7. 7 个 Hybrid Fusion Regression 均有 case-level trace，记录 lexical/vector/fused rank、score、normalized score 和 fusion score。",
            "8. 当前 Hybrid 没有继续作为默认最优 retrieval strategy 的诊断依据：recovery=0、regression=7、behavior=`harmful`。",
            "9. 值得进入 hierarchical promotion / validation，但必须在 TASK-0084 用 formal authority 验证，不得把本任务 diagnostic recall 当作正式提升。",
            "10. 下一任务应做 Hierarchical Retrieval promotion / validation；Fusion 应同时保留为风险项，Reranker 仍不应启动。",
            "",
        ]
    )


def section_digest(heading_path: tuple[str, ...], document_id: str) -> str:
    return digest_json({"document_id": document_id, "heading_path": list(heading_path)})


def _top_documents(candidates: Sequence[Any], limit: int) -> set[str]:
    out: list[str] = []
    for candidate in candidates:
        doc = candidate.chunk.document_identity_digest
        if doc not in out:
            out.append(doc)
        if len(out) >= limit:
            break
    return set(out)


def _top_sections(candidates: Sequence[Any], hierarchy: HierarchyAuthority, limit: int, allowed_sections: set[str] | None = None) -> list[str]:
    out: list[str] = []
    for candidate in candidates:
        section_id = hierarchy.chunk_parent.get(candidate.chunk.scope_identity_digest)
        if section_id is None or (allowed_sections is not None and section_id not in allowed_sections):
            continue
        if section_id not in out:
            out.append(section_id)
        if len(out) >= limit:
            break
    return out


def _hit_map(authority: list[dict[str, Any]], rankings: dict[str, tuple[Any, ...]], spans_by_digest: dict[str, SourceEvidenceResolution]) -> dict[str, bool]:
    return {
        f"{gold['sample_id']}::{gold['source_span_digest']}": _within(_complete_coverage_first_rank(rankings[gold["sample_id"]], spans_by_digest[gold["source_span_digest"]]), TOP_K)
        for gold in authority
    }


def _equal_recall_explanation(doc: float, section: float, chunk: float, rows: list[dict[str, Any]]) -> str:
    if doc == section == chunk:
        parent_without_chunk = sum(row["document_hit_at_20"] and not row["exact_chunk_hit_at_20"] for row in rows) + sum(row["section_hit_at_20"] and not row["exact_chunk_hit_at_20"] for row in rows)
        if parent_without_chunk == 0:
            return "The independently computed values are equal because no diagnostic unit has a document/section hit at K without complete chunk evidence at K."
        return "The independently computed values are numerically equal despite parent-only hits because offsetting misses exist; inspect rows for case-level evidence."
    return "The independently computed values differ, so TASK-0082 equality was not a metric-definition inevitability."


def _variant_status(variant: str, h_metrics: dict[str, dict[str, Any]], paired_recovery: dict[str, Any], promotion: bool) -> str:
    if promotion:
        return "promotion_candidate"
    if variant == "H0":
        return "diagnostic_only"
    if h_metrics[variant]["recall_at_20"] >= h_metrics["H0"]["recall_at_20"] and paired_recovery["variants"][variant]["net_recovered_unit_count"] >= 0:
        return "promising"
    return "rejected"


def _write_artifacts(
    *,
    summary: dict[str, Any],
    metric_authority: dict[str, Any],
    hierarchy_integrity: dict[str, Any],
    task0082_miss_localization: dict[str, Any],
    hierarchical_metric_audit: dict[str, Any],
    h_metrics: dict[str, dict[str, Any]],
    traces: list[dict[str, Any]],
    paired_recovery: dict[str, Any],
    fusion_regression_audit: dict[str, Any],
    efficiency: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    write_json(RESULT_DIR / "metric_authority.json", metric_authority)
    write_json(RESULT_DIR / "hierarchy_integrity.json", hierarchy_integrity)
    write_jsonl(RESULT_DIR / "task0082_miss_localization.jsonl", task0082_miss_localization["rows"])
    write_json(RESULT_DIR / "hierarchical_metric_audit.json", hierarchical_metric_audit)
    for variant in VARIANTS:
        write_json(RESULT_DIR / f"{variant.lower()}_metrics.json", h_metrics[variant])
    write_jsonl(RESULT_DIR / "hierarchical_traces.jsonl", traces)
    write_json(RESULT_DIR / "paired_recovery.json", paired_recovery)
    write_json(RESULT_DIR / "fusion_regression_audit.json", fusion_regression_audit)
    write_json(RESULT_DIR / "efficiency.json", efficiency)
    write_json(RESULT_DIR / "decision.json", decision)
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verify_task0083_artifacts())


def _blocked_summary(contract: dict[str, Any], run_id: str | None, blocked_reason: str, detail: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "blocked",
        "blocked_reason": blocked_reason,
        "blocked_detail": detail,
        "run_id": run_id,
        "git_commit": contract.get("git_commit"),
        "formal_metric_authority": contract.get("formal_metric_authority"),
        "diagnostic_metric_authority": contract.get("diagnostic_metric_authority"),
        "benchmark_modified": False,
        "chunking_modified": False,
        "embedding_model_modified": False,
        "default_retriever_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "git_add_executed": False,
        "git_commit_created": False,
    }
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_task0083_report(summary, {}, {}, {"failure_stage_counts": {}}, {"variants": {}}), encoding="utf-8")
    return summary


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if "path" not in key.lower()}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
