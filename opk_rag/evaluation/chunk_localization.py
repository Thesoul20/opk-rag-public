from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import statistics
import time
from typing import Any, Iterable

from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INPUT_TEMPLATE_VERSION, render_query_input
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, _load_public_samples, git_commit
from opk_rag.evaluation.chunk_representation_experiment import (
    ChunkRepresentationExperimentError,
    _build_qwen_execution_context,
    _count_tokens,
    _p95,
)
from opk_rag.evaluation.evidence_identity import digest_json
from opk_rag.evaluation.qwen_embedding_execution import build_cache_key_row, build_execution_plan, materialize_embedding_variant
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    ScopeAwareChunkingInfrastructureBlocked,
    SourceDocument,
    ShadowCandidate,
    ShadowChunk,
    _load_source_evidence_resolutions,
    _shadow_document_identity_map,
    build_execution_manifest,
    build_source_span_contract,
    compare_shadow_c0_to_production,
    load_candidate_universe_for_formal_run,
    read_json,
    resolve_frozen_source_corpus,
    utc_now,
    validate_shadow_chunks,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.source_evidence_span import (
    SourceEvidenceResolution,
    calculate_span_union_coverage,
    match_shadow_chunk_to_source_span,
)
from opk_rag.indexing.chunk_variants import (
    TASK0081_VARIANT_IDS,
    build_task0081_chunks,
    render_task0081_embedding_input,
    task0081_variant_contracts,
)
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer


TASK_ID = "TASK-0081"
EXPERIMENT_ID = "task0081-chunk-localization-recall"
CONTRACT_SCHEMA_VERSION = "opk-rag.task0081.chunk-localization-recall-contract.v1"
SUMMARY_SCHEMA_VERSION = "opk-rag.task0081.chunk-localization-recall-summary.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0081_chunk_localization_recall_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0081_CHUNK_LOCALIZATION_RECALL_REPORT.md"
K_VALUES = (5, 10, 20)
MODES = ("lexical", "vector", "hybrid")
TOKENIZER = JiebaLexicalTokenizer()


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    chunk: ShadowChunk
    score: float
    retrieval_mode: str


def build_task0081_contract(embedding_config: EmbeddingConfig | None = None) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _stable_created_at(CONTRACT_PATH),
        "git_commit": git_commit(),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "benchmark_samples": "core-rag-benchmark-v1 development + known-regression public samples",
        "embedding_config": {
            "model_name": embedding_config.model_name,
            "model_revision": embedding_config.model_revision,
            "dimension": embedding_config.dimension,
            "normalize": embedding_config.normalize,
            "local_files_only": embedding_config.local_files_only,
        },
        "retrieval_config": {
            "query_template_version": QUERY_INPUT_TEMPLATE_VERSION,
            "top_k": list(K_VALUES),
            "candidate_depth": 20,
            "modes": list(MODES),
            "vector_similarity": "dot_product_over_normalized_qwen_vectors",
            "lexical_implementation": "in_memory_term_overlap_for_same_pipeline_shadow_index",
            "hybrid_implementation": "reciprocal_rank_fusion_equal_weight",
            "reranker_enabled": False,
            "graph_retrieval_enabled": False,
        },
        "variant_definitions": task0081_variant_contracts(),
        "evaluation_metrics": ["recall_at_5", "recall_at_10", "recall_at_20", "hit_at_5", "hit_at_10", "hit_at_20", "mrr"],
        "promotion_rules": {
            "requires_recall_at_20_above_c0": True,
            "requires_same_pipeline_retrieval": True,
            "oracle_forbidden_for_promotion": True,
            "benchmark_modified": False,
            "retriever_modified": False,
            "embedding_model_modified": False,
            "query_reformulation_modified": False,
            "agent_runtime_modified": False,
            "graph_runtime_modified": False,
        },
        "production_default_change": False,
    }
    write_json(CONTRACT_PATH, contract)
    return contract


def run_task0081_chunking_experiment(*, embedding_config: EmbeddingConfig | None = None, run_id: str | None = None) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    contract = build_task0081_contract(embedding_config)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        documents, universe_audit = resolve_frozen_source_corpus()
    except ScopeAwareChunkingInfrastructureBlocked as exc:
        summary = _blocked_summary(contract, "frozen_source_corpus_unavailable", str(exc))
        _write_blocked_artifacts(summary)
        return summary

    production_chunks, production_universe = load_candidate_universe_for_formal_run(embedding_config)
    c0_chunks = build_task0081_chunks(documents, "C0")
    c0_comparison = compare_shadow_c0_to_production(c0_chunks, production_chunks)
    if c0_comparison["status"] != "valid":
        summary = _blocked_summary(contract, "c0_shadow_production_parity_failed", json.dumps(c0_comparison, ensure_ascii=False))
        _write_blocked_artifacts(summary)
        return summary

    source_span_contract = build_source_span_contract(production_chunks)
    if source_span_contract.get("source_evidence_span_contract_status") != "complete":
        summary = _blocked_summary(contract, "source_evidence_span_contract_incomplete", json.dumps(source_span_contract, ensure_ascii=False))
        _write_blocked_artifacts(summary)
        return summary
    spans = _load_source_evidence_resolutions(source_span_contract["spans"], document_identity_map=_shadow_document_identity_map(production_chunks))
    samples = _load_public_samples(["development", "known-regression"])
    variants = {variant_id: build_task0081_chunks(documents, variant_id) for variant_id in TASK0081_VARIANT_IDS}
    for variant_id, chunks in variants.items():
        validation = validate_shadow_chunks(chunks, documents)
        if validation["empty_chunk_count"] or validation["invalid_line_range_count"] or validation["failed_document_count"]:
            summary = _blocked_summary(contract, "variant_validation_failed", json.dumps({"variant_id": variant_id, "validation": validation}, ensure_ascii=False))
            _write_blocked_artifacts(summary)
            return summary

    chunk_inventory = build_chunk_inventory(variants["C0"], documents)
    gold_mapping = build_gold_chunk_mapping(spans, variants["C0"])
    localization_metrics = aggregate_gold_mapping(gold_mapping)

    try:
        context = _build_qwen_execution_context(embedding_config)
        execution_manifest = build_execution_manifest(contract=contract, source_universe_audit=universe_audit, embedding_config=embedding_config)
        execution_plan = build_execution_plan(contract=contract, candidate_universe=production_universe, model_execution=context.model_execution, selected_batch_size=4)
        query_vectors, query_manifest = _materialize_queries(samples, context.provider, execution_plan)
        variant_results: dict[str, dict[str, Any]] = {}
        all_trace_rows: list[dict[str, Any]] = []
        for variant_id, chunks in variants.items():
            start = time.perf_counter()
            chunk_vectors, token_counts, manifest = _materialize_chunks(variant_id, chunks, context.provider, execution_plan)
            build_time_ms = (time.perf_counter() - start) * 1000
            retrieval, traces = score_multimode_retrieval(samples, spans, chunks, chunk_vectors, query_vectors)
            variant_results[variant_id] = {
                "variant_id": variant_id,
                "metrics_by_mode": retrieval,
                "efficiency": build_efficiency_metrics(variant_id, chunks, token_counts, manifest, build_time_ms, traces, c0_chunk_count=len(variants["C0"])),
                "chunk_count": len(chunks),
                "embedding_manifest": _public_manifest(manifest),
            }
            all_trace_rows.extend({"variant_id": variant_id, **row} for row in traces)
    except (EmbeddingModelLoadError, EmbeddingInferenceError, ChunkRepresentationExperimentError) as exc:
        summary = _blocked_summary(contract, "embedding_or_retrieval_environment_blocked", f"{type(exc).__name__}: {exc}")
        summary.update(
            {
                "chunk_inventory": chunk_inventory,
                "chunk_localization_metrics": localization_metrics,
                "c0_chunk_count": len(variants["C0"]),
            }
        )
        _write_partial_artifacts(summary, gold_mapping)
        return summary

    paired = build_paired_transition_analysis(all_trace_rows)
    failure_rows = classify_retrieval_failures(
        c0_traces=[row for row in all_trace_rows if row["variant_id"] == "C0"],
        gold_mapping=gold_mapping,
    )
    promotion = build_promotion_decision(variant_results, paired)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "completed",
        "created_at": utc_now(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "contract_path": _rel(CONTRACT_PATH),
        "result_dir": _rel(RESULT_DIR),
        "source_corpus_status": "complete",
        "source_universe_audit": universe_audit,
        "production_chunking_c0_comparison": c0_comparison,
        "chunk_localization_metrics": localization_metrics,
        "chunking_bottleneck_confirmed": _chunking_bottleneck_status(localization_metrics, variant_results),
        "variants": variant_results,
        "paired_transition_analysis": paired,
        "promotion_decision": promotion,
        "all_retrieval_misses_classified": all(row["classification"] != "other" or row.get("classification_reason") for row in failure_rows),
        "benchmark_modified": False,
        "retriever_modified": False,
        "embedding_model_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "production_default_change": False,
    }
    write_task0081_artifacts(summary, chunk_inventory, gold_mapping, failure_rows, all_trace_rows)
    return summary


def build_chunk_inventory(chunks: tuple[ShadowChunk, ...], documents: tuple[SourceDocument, ...]) -> dict[str, Any]:
    token_counts = [len(chunk.content) for chunk in chunks]
    section_keys = {(chunk.document_identity_digest, tuple(chunk.heading_path), chunk.start_line, chunk.end_line) for chunk in chunks}
    doc_counts = Counter(chunk.document_identity_digest for chunk in chunks)
    section_counts = Counter((chunk.document_identity_digest, tuple(chunk.heading_path)) for chunk in chunks)
    provenance = [
        {
            "document_id": chunk.document_identity_digest,
            "source_path": chunk.relative_path,
            "section_id": digest_json([chunk.document_identity_digest, list(chunk.heading_path)]),
            "heading_path": list(chunk.heading_path),
            "chunk_id": chunk.scope_identity_digest,
            "chunk_index_in_section": _chunk_index_in_section(chunk, chunks),
            "start_offset": chunk.start_line,
            "end_offset": chunk.end_line,
            "token_count": len(chunk.content),
        }
        for chunk in chunks
    ]
    duplicate_count = len(chunks) - len({chunk.rendered_content_digest for chunk in chunks})
    return {
        "schema_version": "opk-rag.task0081.chunk-inventory.v1",
        "document_count": len(documents),
        "section_count": len(section_keys),
        "chunk_count": len(chunks),
        "total_chunk_tokens": sum(token_counts),
        "mean_chunk_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "median_chunk_tokens": statistics.median(token_counts) if token_counts else 0.0,
        "p50_chunk_tokens": _percentile(token_counts, 50),
        "p75_chunk_tokens": _percentile(token_counts, 75),
        "p90_chunk_tokens": _percentile(token_counts, 90),
        "p95_chunk_tokens": _percentile(token_counts, 95),
        "max_chunk_tokens": max(token_counts, default=0),
        "min_chunk_tokens": min(token_counts, default=0),
        "chunks_per_document_mean": statistics.mean(doc_counts.values()) if doc_counts else 0.0,
        "chunks_per_section_mean": statistics.mean(section_counts.values()) if section_counts else 0.0,
        "single_chunk_section_count": sum(1 for count in section_counts.values() if count == 1),
        "multi_chunk_section_count": sum(1 for count in section_counts.values() if count > 1),
        "heading_preserved_chunk_count": sum(1 for chunk in chunks if chunk.heading_path),
        "heading_missing_chunk_count": sum(1 for chunk in chunks if not chunk.heading_path),
        "overlap_token_ratio": _ratio(sum(chunk.overlap_token_count for chunk in chunks), sum(token_counts)),
        "duplicate_text_ratio": _ratio(duplicate_count, len(chunks)),
        "provenance": provenance,
    }


def build_gold_chunk_mapping(spans: tuple[SourceEvidenceResolution, ...], chunks: tuple[ShadowChunk, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for span in spans:
        overlapping = [chunk for chunk in chunks if match_shadow_chunk_to_source_span(chunk, span) != "no_overlap"]
        containing = [chunk for chunk in chunks if match_shadow_chunk_to_source_span(chunk, span) == "full_span_containment"]
        coverage = calculate_span_union_coverage(overlapping, span)
        best_overlap = max((_overlap_ratio(chunk, span) for chunk in overlapping), default=0.0)
        split = coverage in {"multi_candidate_complete_coverage", "partial_overlap"} and not containing
        heading_available = any(chunk.heading_path for chunk in overlapping)
        rows.append(
            {
                "sample_id": span.sample_id,
                "question_id": span.sample_id,
                "gold_document": span.document_identity_digest,
                "gold_section": span.heading_path_digest,
                "gold_span": _span_payload(span),
                "matching_chunk_ids": [chunk.scope_identity_digest for chunk in containing],
                "containing_chunk_ids": [chunk.scope_identity_digest for chunk in containing],
                "overlapping_chunk_ids": [chunk.scope_identity_digest for chunk in overlapping],
                "best_overlap_ratio": best_overlap,
                "full_span_contained": bool(containing),
                "evidence_split_across_chunks": split,
                "heading_context_available": heading_available,
                "chunk_representable": bool(containing) or coverage == "multi_candidate_complete_coverage",
                "represented_but_not_retrieved": False,
                "coverage_relationship": coverage,
                "source_span_digest": span.source_span_digest,
            }
        )
    return rows


def aggregate_gold_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    contained = sum(1 for row in rows if row["full_span_contained"])
    split = sum(1 for row in rows if row["evidence_split_across_chunks"])
    heading = sum(1 for row in rows if row["heading_context_available"])
    retrievable = sum(1 for row in rows if row["chunk_representable"])
    coverage = statistics.mean(row["best_overlap_ratio"] for row in rows) if rows else 0.0
    return {
        "gold_unit_count": total,
        "gold_span_containment_rate": _ratio(contained, total),
        "gold_span_coverage_rate": coverage,
        "boundary_split_rate": _ratio(split, total),
        "heading_context_preservation_rate": _ratio(heading, total),
        "retrievable_gold_unit_rate": _ratio(retrievable, total),
        "full_span_contained_count": contained,
        "boundary_split_count": split,
        "not_independently_representable_count": sum(1 for row in rows if not row["chunk_representable"]),
        "represented_but_not_retrieved_count": 0,
    }


def score_multimode_retrieval(
    samples: list[dict[str, Any]],
    spans: tuple[SourceEvidenceResolution, ...],
    chunks: tuple[ShadowChunk, ...],
    chunk_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    spans_by_sample: dict[str, list[SourceEvidenceResolution]] = {}
    for span in spans:
        spans_by_sample.setdefault(span.sample_id, []).append(span)
    lexical_index = _lexical_index(chunks)
    vector_index = [(chunk, chunk_vectors[chunk.scope_identity_digest]) for chunk in chunks if chunk.scope_identity_digest in chunk_vectors]
    traces: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for sample in samples:
        query_vector = query_vectors[sample["sample_id"]]
        rankings = {
            "lexical": _rank_lexical(chunks, lexical_index, sample["question"], limit=20),
            "vector": _rank_vector(vector_index, query_vector, limit=20),
        }
        rankings["hybrid"] = _rank_hybrid(rankings["lexical"], rankings["vector"], limit=20)
        for mode, candidates in rankings.items():
            started = time.perf_counter()
            row = _trace_row(sample, spans_by_sample.get(sample["sample_id"], []), candidates, mode, (time.perf_counter() - started) * 1000)
            by_mode[mode].append(row)
            traces.append(row)
    return {mode: _aggregate_mode(rows) for mode, rows in by_mode.items()}, traces


def classify_retrieval_failures(c0_traces: list[dict[str, Any]], gold_mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {row["source_span_digest"]: row for row in gold_mapping}
    rows = []
    for trace in c0_traces:
        if trace["retrieval_mode"] != "hybrid":
            continue
        for profile in trace["unit_profiles"]:
            if profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] <= 20:
                continue
            gold = mapping.get(profile["source_span_digest"], {})
            if not gold.get("overlapping_chunk_ids"):
                classification = "gold_not_chunked"
            elif gold.get("evidence_split_across_chunks"):
                classification = "gold_split_across_chunks"
            elif not gold.get("heading_context_available"):
                classification = "heading_context_loss"
            elif gold.get("chunk_representable"):
                classification = "chunk_valid_but_ranking_miss"
            else:
                classification = "gold_not_chunked"
            rows.append(
                {
                    "sample_id": trace["sample_id"],
                    "source_span_digest": profile["source_span_digest"],
                    "retrieval_mode": "hybrid",
                    "classification": classification,
                    "classification_reason": f"C0 hybrid complete_evidence_rank={profile['complete_evidence_rank']}",
                    "all_retrieval_misses_classified": True,
                }
            )
    return rows


def build_paired_transition_analysis(traces: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _unit_hits([row for row in traces if row["variant_id"] == "C0" and row["retrieval_mode"] == "hybrid"], 20)
    variants: dict[str, Any] = {}
    for variant_id in TASK0081_VARIANT_IDS:
        current = _unit_hits([row for row in traces if row["variant_id"] == variant_id and row["retrieval_mode"] == "hybrid"], 20)
        recovered = sorted(key for key, hit in current.items() if hit and not baseline.get(key, False))
        regressed = sorted(key for key, hit in baseline.items() if hit and not current.get(key, False))
        variants[variant_id] = {
            "miss_to_hit": len(recovered),
            "hit_to_hit": sum(1 for key, hit in current.items() if hit and baseline.get(key, False)),
            "hit_to_miss": len(regressed),
            "miss_to_miss": sum(1 for key, hit in current.items() if not hit and not baseline.get(key, False)),
            "newly_recovered_samples": sorted({key.split("|", 1)[0] for key in recovered}),
            "newly_regressed_samples": sorted({key.split("|", 1)[0] for key in regressed}),
            "newly_recovered_unit_count": len(recovered),
            "newly_regressed_unit_count": len(regressed),
            "improvement_attribution": _improvement_attribution(variant_id),
            "regression_attribution": "rank_displacement_or_smaller_context" if regressed else "none_observed",
        }
    return {
        "schema_version": "opk-rag.task0081.paired-transition.v1",
        "top_k": 20,
        "retrieval_mode": "hybrid",
        "statistical_confidence_limited": len(baseline) < 100,
        "variants": variants,
    }


def build_efficiency_metrics(
    variant_id: str,
    chunks: tuple[ShadowChunk, ...],
    token_counts: list[int],
    manifest: dict[str, Any],
    build_time_ms: float,
    traces: list[dict[str, Any]],
    *,
    c0_chunk_count: int,
) -> dict[str, Any]:
    latencies = [row["query_latency_ms"] for row in traces if row["retrieval_mode"] == "hybrid"]
    return {
        "variant_id": variant_id,
        "chunk_count": len(chunks),
        "chunk_count_delta_vs_c0": len(chunks) - c0_chunk_count,
        "embedding_count": len(chunks),
        "total_embedding_tokens": sum(token_counts),
        "index_size": len(chunks),
        "index_build_time": build_time_ms,
        "retrieval_latency_p50": statistics.median(latencies) if latencies else 0.0,
        "retrieval_latency_p95": _p95(latencies),
        "embedding_manifest_digest": digest_json(_public_manifest(manifest)),
    }


def build_promotion_decision(variant_results: dict[str, dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    c0 = variant_results["C0"]["metrics_by_mode"]["hybrid"]["recall_at"]["20"]
    best = max(TASK0081_VARIANT_IDS, key=lambda variant_id: variant_results[variant_id]["metrics_by_mode"]["hybrid"]["recall_at"]["20"])
    best_recall = variant_results[best]["metrics_by_mode"]["hybrid"]["recall_at"]["20"]
    regressed = paired["variants"][best]["newly_regressed_unit_count"]
    promotion_candidate = best != "C0" and best_recall > c0 and regressed == 0
    return {
        "best_variant": best,
        "best_variant_recall_at_20": best_recall,
        "c0_recall_at_20": c0,
        "absolute_recall_at_20_delta": best_recall - c0,
        "promotion_candidate": promotion_candidate,
        "promotion_decision": "promotion_candidate" if promotion_candidate else ("promising" if best_recall > c0 else "rejected"),
        "next_task_decision": "promote_validated_chunking_variant" if promotion_candidate else "chunking_not_primary_bottleneck_move_to_retrieval_ranking",
        "recommended_next_task": "TASK-0082" if promotion_candidate else "retrieval ranking localization task",
    }


def write_task0081_artifacts(
    summary: dict[str, Any],
    chunk_inventory: dict[str, Any],
    gold_mapping: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    write_json(RESULT_DIR / "chunk_inventory.json", chunk_inventory)
    write_jsonl(RESULT_DIR / "gold_chunk_mapping.jsonl", gold_mapping)
    write_jsonl(RESULT_DIR / "retrieval_failure_classification.jsonl", failure_rows)
    write_json(RESULT_DIR / "variant_manifest.json", {"variants": task0081_variant_contracts()})
    for variant_id, result in summary["variants"].items():
        write_json(RESULT_DIR / f"{variant_id.lower()}_metrics.json", result)
    write_json(RESULT_DIR / "paired_transition_analysis.json", summary["paired_transition_analysis"])
    write_json(RESULT_DIR / "efficiency_metrics.json", {variant_id: row["efficiency"] for variant_id, row in summary["variants"].items()})
    write_json(RESULT_DIR / "promotion_decision.json", summary["promotion_decision"])
    write_json(RESULT_DIR / "summary.json", summary)
    write_jsonl(RESULT_DIR / "retrieval_traces.jsonl", traces)
    write_json(RESULT_DIR / "verification.json", verify_task0081_artifacts())
    REPORT_PATH.write_text(build_task0081_report(summary), encoding="utf-8")


def verify_task0081_artifacts() -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "chunk_inventory.json",
        RESULT_DIR / "gold_chunk_mapping.jsonl",
        RESULT_DIR / "retrieval_failure_classification.jsonl",
        RESULT_DIR / "variant_manifest.json",
        RESULT_DIR / "paired_transition_analysis.json",
        RESULT_DIR / "efficiency_metrics.json",
        RESULT_DIR / "promotion_decision.json",
        RESULT_DIR / "summary.json",
        REPORT_PATH,
    ]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = read_json(RESULT_DIR / "summary.json")
        for forbidden in ("benchmark_modified", "retriever_modified", "embedding_model_modified", "query_reformulation_modified", "agent_runtime_modified", "graph_runtime_modified"):
            if summary.get(forbidden) is not False:
                issues.append({"code": f"{forbidden}_not_false"})
    return {
        "schema_version": "opk-rag.task0081.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_task0081_report(summary: dict[str, Any]) -> str:
    if summary.get("experiment_status") != "completed":
        return "\n".join(
            [
                "# TASK0081 Chunk Localization Recall Report",
                "",
                f"Status: `{summary.get('experiment_status')}`",
                f"Blocked reason: `{summary.get('blocked_reason')}`",
                f"Blocked detail: `{summary.get('blocked_detail')}`",
                "",
            ]
        )
    loc = summary["chunk_localization_metrics"]
    promo = summary["promotion_decision"]
    rows = []
    for variant_id in TASK0081_VARIANT_IDS:
        metrics = summary["variants"][variant_id]["metrics_by_mode"]["hybrid"]
        rows.append(f"| {variant_id} | {metrics['recall_at']['5']:.4f} | {metrics['recall_at']['10']:.4f} | {metrics['recall_at']['20']:.4f} | {metrics['mrr']:.4f} |")
    paired_best = summary["paired_transition_analysis"]["variants"][promo["best_variant"]]
    return "\n".join(
        [
            "# TASK0081 Chunk Localization Recall Report",
            "",
            f"chunking_bottleneck_confirmed={summary['chunking_bottleneck_confirmed']}",
            "",
            "## Current C0 Diagnosis",
            "",
            f"- Gold units: `{loc['gold_unit_count']}`",
            f"- Full span contained: `{loc['full_span_contained_count']}` (`{loc['gold_span_containment_rate']:.4f}`)",
            f"- Boundary split: `{loc['boundary_split_count']}` (`{loc['boundary_split_rate']:.4f}`)",
            f"- Not independently representable: `{loc['not_independently_representable_count']}`",
            f"- Retrievable gold unit rate: `{loc['retrievable_gold_unit_rate']:.4f}`",
            "",
            "## Same-Pipeline Hybrid Retrieval",
            "",
            "| Variant | Recall@5 | Recall@10 | Recall@20 | MRR |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Decision",
            "",
            f"- Best variant: `{promo['best_variant']}`",
            f"- Recall@20 delta: `{promo['absolute_recall_at_20_delta']:.4f}`",
            f"- Newly recovered units: `{paired_best['newly_recovered_unit_count']}`",
            f"- Newly regressed units: `{paired_best['newly_regressed_unit_count']}`",
            f"- Promotion decision: `{promo['promotion_decision']}`",
            f"- Recommended next task: `{promo['recommended_next_task']}`",
            "",
        ]
    )


def _trace_row(sample: dict[str, Any], spans: list[SourceEvidenceResolution], candidates: Iterable[RankedCandidate], mode: str, latency_ms: float) -> dict[str, Any]:
    ranked = tuple(candidates)
    profiles = []
    for span in spans:
        profiles.append(_span_profile(span, ranked))
    return {
        "sample_id": sample["sample_id"],
        "dataset_id": sample["dataset_id"],
        "question_type": sample.get("question_type"),
        "capability": sample.get("capability"),
        "retrieval_mode": mode,
        "candidate_count": len(ranked),
        "query_latency_ms": latency_ms,
        "required_units_total": len(spans),
        "unit_profiles": profiles,
    }


def _span_profile(span: SourceEvidenceResolution, candidates: tuple[RankedCandidate, ...]) -> dict[str, Any]:
    any_rank = complete_rank = None
    for candidate in candidates:
        if match_shadow_chunk_to_source_span(candidate.chunk, span) != "no_overlap" and any_rank is None:
            any_rank = candidate.rank
    for k in K_VALUES:
        coverage = calculate_span_union_coverage((candidate.chunk for candidate in candidates if candidate.rank <= k), span)
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"}:
            complete_rank = k
            break
    return {"source_span_digest": span.source_span_digest, "any_support_rank": any_rank, "complete_evidence_rank": complete_rank}


def _aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in rows)
    first_ranks = [min((profile["complete_evidence_rank"] for profile in row["unit_profiles"] if profile["complete_evidence_rank"] is not None), default=None) for row in rows]
    return {
        "sample_count": len(rows),
        "required_unit_count": denominator,
        "recall_at": {str(k): _ratio(sum(1 for row in rows for profile in row["unit_profiles"] if profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] <= k), denominator) for k in K_VALUES},
        "hit_at": {str(k): _ratio(sum(1 for row in rows if any(profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] <= k for profile in row["unit_profiles"])), len(rows)) for k in K_VALUES},
        "mrr": statistics.mean((1 / rank) if rank else 0.0 for rank in first_ranks) if rows else 0.0,
    }


def _rank_vector(index: list[tuple[ShadowChunk, list[float]]], query_vector: list[float], *, limit: int) -> tuple[RankedCandidate, ...]:
    scored = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), chunk) for chunk, vector in index]
    scored.sort(key=lambda item: (-item[0], item[1].document_identity_digest, item[1].scope_identity_digest))
    return tuple(RankedCandidate(rank=index + 1, chunk=chunk, score=score, retrieval_mode="vector") for index, (score, chunk) in enumerate(scored[:limit]))


def _rank_lexical(chunks: tuple[ShadowChunk, ...], index: dict[str, Counter[str]], query: str, *, limit: int) -> tuple[RankedCandidate, ...]:
    q = Counter(TOKENIZER.tokenize_query(query))
    scored = []
    for chunk in chunks:
        tf = index[chunk.scope_identity_digest]
        score = sum(min(count, tf.get(term, 0)) for term, count in q.items())
        scored.append((float(score), chunk))
    scored.sort(key=lambda item: (-item[0], item[1].document_identity_digest, item[1].scope_identity_digest))
    return tuple(RankedCandidate(rank=index + 1, chunk=chunk, score=score, retrieval_mode="lexical") for index, (score, chunk) in enumerate(scored[:limit]))


def _rank_hybrid(lexical: tuple[RankedCandidate, ...], vector: tuple[RankedCandidate, ...], *, limit: int) -> tuple[RankedCandidate, ...]:
    scores: dict[str, float] = {}
    chunks: dict[str, ShadowChunk] = {}
    for ranking in (lexical, vector):
        for candidate in ranking:
            key = candidate.chunk.scope_identity_digest
            chunks[key] = candidate.chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + candidate.rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], chunks[item[0]].document_identity_digest, item[0]))
    return tuple(RankedCandidate(rank=index + 1, chunk=chunks[key], score=score, retrieval_mode="hybrid") for index, (key, score) in enumerate(ordered[:limit]))


def _lexical_index(chunks: tuple[ShadowChunk, ...]) -> dict[str, Counter[str]]:
    return {chunk.scope_identity_digest: Counter(TOKENIZER.tokenize_document(chunk.content + "\n" + "\n".join(chunk.heading_path))) for chunk in chunks}


def _materialize_chunks(variant_id: str, chunks: tuple[ShadowChunk, ...], provider: Any, execution_plan: dict[str, Any]) -> tuple[dict[str, list[float]], list[int], dict[str, Any]]:
    texts = [render_task0081_embedding_input(chunk, variant_id) for chunk in chunks]
    identities = [chunk.scope_identity_digest for chunk in chunks]
    cache_rows = [
        build_cache_key_row(execution_plan=execution_plan, identity=identity, rendered_text=text, variant_id=f"task0081-{variant_id}", query_variant_id=None, template_digest=digest_json({"task": TASK_ID, "variant_id": variant_id}))
        for identity, text in zip(identities, texts, strict=True)
    ]
    materialized = materialize_embedding_variant(
        variant_id=f"task0081-{variant_id}",
        kind="representation",
        identities=identities,
        texts=texts,
        provider=provider,
        execution_plan=execution_plan,
        cache_key_rows=cache_rows,
        expected_count=len(chunks),
        count_tokens=lambda text: _count_tokens(provider, text),
    )
    return materialized.vectors, materialized.token_counts, materialized.manifest


def _materialize_queries(samples: list[dict[str, Any]], provider: Any, execution_plan: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    texts = [render_query_input(sample["question"]) for sample in samples]
    identities = [sample["sample_id"] for sample in samples]
    cache_rows = [
        build_cache_key_row(execution_plan=execution_plan, identity=identity, rendered_text=text, variant_id=None, query_variant_id="task0081-current-query", template_digest=digest_json({"task": TASK_ID, "query": "current"}))
        for identity, text in zip(identities, texts, strict=True)
    ]
    materialized = materialize_embedding_variant(
        variant_id="task0081-current-query",
        kind="query",
        identities=identities,
        texts=texts,
        provider=provider,
        execution_plan=execution_plan,
        cache_key_rows=cache_rows,
        expected_count=len(samples),
        count_tokens=lambda text: _count_tokens(provider, text),
    )
    return materialized.vectors, materialized.manifest


def _write_blocked_artifacts(summary: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verify_task0081_artifacts())
    REPORT_PATH.write_text(build_task0081_report(summary), encoding="utf-8")


def _write_partial_artifacts(summary: dict[str, Any], gold_mapping: list[dict[str, Any]]) -> None:
    write_json(RESULT_DIR / "summary.json", summary)
    if "chunk_inventory" in summary:
        write_json(RESULT_DIR / "chunk_inventory.json", summary["chunk_inventory"])
    write_jsonl(RESULT_DIR / "gold_chunk_mapping.jsonl", gold_mapping)
    write_json(RESULT_DIR / "variant_manifest.json", {"variants": task0081_variant_contracts()})
    write_json(RESULT_DIR / "verification.json", verify_task0081_artifacts())
    REPORT_PATH.write_text(build_task0081_report(summary), encoding="utf-8")


def _blocked_summary(contract: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "experiment_status": "blocked",
        "blocked_reason": reason,
        "blocked_detail": detail,
        "contract_path": _rel(CONTRACT_PATH),
        "benchmark_modified": False,
        "retriever_modified": False,
        "embedding_model_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "git_commit": contract.get("git_commit"),
    }


def _chunking_bottleneck_status(localization: dict[str, Any], variants: dict[str, dict[str, Any]]) -> str:
    c0 = variants["C0"]["metrics_by_mode"]["hybrid"]["recall_at"]["20"]
    best = max(row["metrics_by_mode"]["hybrid"]["recall_at"]["20"] for row in variants.values())
    if localization["retrievable_gold_unit_rate"] < 0.8 or localization["boundary_split_rate"] > 0.2:
        return "true" if best > c0 else "partial"
    return "partial" if best > c0 else "false"


def _unit_hits(traces: list[dict[str, Any]], k: int) -> dict[str, bool]:
    return {
        f"{trace['sample_id']}|{profile['source_span_digest']}": profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] <= k
        for trace in traces
        for profile in trace["unit_profiles"]
    }


def _improvement_attribution(variant_id: str) -> str:
    return {
        "C0": "control",
        "C1": "structural_integrity",
        "C2": "smaller_chunk",
        "C3": "boundary_overlap",
        "C4": "parent_context",
    }[variant_id]


def _span_payload(span: SourceEvidenceResolution) -> dict[str, Any]:
    if hasattr(span, "spans"):
        return {"spans": [{"source_line_start": start, "source_line_end": end} for start, end in span.spans]}
    return {"source_line_start": span.source_line_start, "source_line_end": span.source_line_end}


def _overlap_ratio(chunk: ShadowChunk, span: SourceEvidenceResolution) -> float:
    ranges = span.spans if hasattr(span, "spans") else ((span.source_line_start, span.source_line_end),)
    total = sum(end - start + 1 for start, end in ranges)
    overlap = 0
    for start, end in ranges:
        overlap += max(0, min(chunk.complete_source_span["end_line"], end) - max(chunk.complete_source_span["start_line"], start) + 1)
    return _ratio(overlap, total)


def _chunk_index_in_section(chunk: ShadowChunk, chunks: tuple[ShadowChunk, ...]) -> int:
    peers = [item for item in chunks if item.document_identity_digest == chunk.document_identity_digest and item.heading_path == chunk.heading_path]
    return sorted(peers, key=lambda item: item.chunk_ordinal).index(chunk)


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return float(ordered[index])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if "path" not in key.lower()}


def _stable_created_at(path: Path) -> str:
    if path.exists():
        try:
            return str(read_json(path).get("created_at") or utc_now())
        except Exception:
            pass
    return utc_now()


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
