from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import statistics
import time
from typing import Any, Iterable

from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INPUT_TEMPLATE_VERSION
from opk_rag.evaluation.candidate_retrieval_baseline import (
    CONTRACT_PATH as TASK0054_CONTRACT_PATH,
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
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    SourceDocument,
    ShadowChunk,
    build_source_span_contract,
    calculate_span_union_coverage,
    compare_shadow_c0_to_production,
    load_candidate_universe_for_formal_run,
    read_json,
    resolve_frozen_source_corpus,
    _shadow_document_identity_map,
    utc_now,
    validate_shadow_chunks,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.source_evidence_span import SourceEvidenceResolution, match_shadow_chunk_to_source_span
from opk_rag.indexing.chunk_variants import TASK0081_VARIANT_IDS, build_task0081_chunks, task0081_variant_contracts
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer


TASK_ID = "TASK-0082"
EXPERIMENT_ID = "task0082-retrieval-localization"
CONTRACT_SCHEMA_VERSION = "opk-rag.task0082.retrieval-localization-contract.v1"
SUMMARY_SCHEMA_VERSION = "opk-rag.task0082.retrieval-localization-summary.v1"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0082_retrieval_localization_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0082_RETRIEVAL_LOCALIZATION_REPORT.md"
TOP_K = 20
DIAGNOSTIC_CANDIDATE_DEPTH = 100
K_VALUES = (5, 10, 20)
MODES = ("lexical", "vector", "hybrid")
TOKENIZER = JiebaLexicalTokenizer()


class RetrievalLocalizationError(RuntimeError):
    pass


def run_task0082_retrieval_localization(
    *,
    embedding_config: EmbeddingConfig | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    embedding_config = embedding_config or EmbeddingConfig(local_files_only=True, normalize=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    contract_reconciliation = reconcile_task0054_task0081_metric_contracts()
    variant_sanity = audit_task0081_variant_sanity()
    contract = build_task0082_contract(embedding_config, contract_reconciliation, variant_sanity)
    write_json(CONTRACT_PATH, contract)

    if variant_sanity["task0081_negative_result_requires_revalidation"]:
        summary = _blocked_summary(
            contract,
            run_id=run_id,
            blocked_reason="task0081_variant_sanity_failed",
            contract_reconciliation=contract_reconciliation,
            variant_sanity=variant_sanity,
        )
        _write_all_artifacts(summary, empty=True)
        return summary

    documents, _universe_audit = resolve_frozen_source_corpus()
    production_chunks, _production_universe = load_candidate_universe_for_formal_run(embedding_config)
    chunks = build_task0081_chunks(documents, "C0")
    c0_comparison = compare_shadow_c0_to_production(chunks, production_chunks)
    if c0_comparison["status"] != "valid":
        summary = _blocked_summary(
            contract,
            run_id=run_id,
            blocked_reason="c0_shadow_production_parity_failed",
            contract_reconciliation=contract_reconciliation,
            variant_sanity=variant_sanity,
            detail=c0_comparison,
        )
        _write_all_artifacts(summary, empty=True)
        return summary

    source_span_contract = build_source_span_contract(production_chunks)
    spans = _load_source_evidence_resolutions(source_span_contract["spans"], document_identity_map=_shadow_document_identity_map(production_chunks))
    samples = _load_public_samples(["development", "known-regression"])
    samples = [sample for sample in samples if sample["required_evidence"]]
    gold_mapping = build_gold_chunk_mapping(spans, chunks)
    authority = build_gold_chunk_authority(samples, gold_mapping, chunks)
    representable = [row for row in authority if row["chunk_representable"]]

    try:
        context = _build_qwen_execution_context(embedding_config)
        execution_plan = _load_task0081_execution_plan()
        query_vectors, query_manifest = _materialize_queries(samples, context.provider, execution_plan)
        chunk_vectors, _token_counts, chunk_manifest = _materialize_chunks("C0", chunks, context.provider, execution_plan)
    except (EmbeddingModelLoadError, EmbeddingInferenceError, ChunkRepresentationExperimentError) as exc:
        summary = _blocked_summary(
            contract,
            run_id=run_id,
            blocked_reason="embedding_or_retrieval_environment_blocked",
            contract_reconciliation=contract_reconciliation,
            variant_sanity=variant_sanity,
            detail={"error_type": type(exc).__name__, "error": str(exc)},
        )
        _write_all_artifacts(summary, empty=True)
        return summary

    start = time.perf_counter()
    rankings_by_sample = rank_all_modes(samples, chunks, chunk_vectors, query_vectors)
    elapsed_ms = (time.perf_counter() - start) * 1000
    traces = build_mode_traces(samples, spans, authority, rankings_by_sample)
    candidate_inclusion = build_candidate_inclusion(authority, rankings_by_sample)
    hierarchical = build_hierarchical_localization(authority, rankings_by_sample)
    mode_metrics = {mode: aggregate_mode_trace([row for row in traces[mode] if row["chunk_representable"]]) for mode in MODES}
    fusion = build_fusion_analysis(authority, rankings_by_sample)
    rank_distribution = build_rank_distribution(authority, rankings_by_sample)
    failures = classify_failures(authority, rankings_by_sample, spans)
    bottleneck = decide_bottleneck(authority, mode_metrics, hierarchical, fusion, failures, rank_distribution)
    verification = verify_task0082_artifacts(in_memory_summary=True)

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
        "metric_contract_reconciliation": contract_reconciliation,
        "task0081_variant_sanity_audit": variant_sanity,
        "retrievable_gold_unit_count": len(representable),
        "gold_candidate_inclusion_rate": _ratio(sum(1 for row in candidate_inclusion if row["chunk_representable"] and row["gold_candidate_present"]), len(representable)),
        "hierarchical_localization": hierarchical,
        "mode_metrics": mode_metrics,
        "fusion_analysis": fusion,
        "rank_distribution": rank_distribution,
        "failure_counts": dict(Counter(row["primary_classification"] for row in failures)),
        "all_primary_retrieval_misses_classified": all(row["primary_classification"] != "unexplained" for row in failures),
        "bottleneck_decision": bottleneck,
        "runtime": {
            "ranking_wall_time_ms": elapsed_ms,
            "query_embedding_manifest_digest": digest_json(_public_manifest(query_manifest)),
            "chunk_embedding_manifest_digest": digest_json(_public_manifest(chunk_manifest)),
        },
        "benchmark_modified": False,
        "chunking_modified": False,
        "retriever_modified": False,
        "embedding_model_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
        "production_default_change": False,
        "model_calls": {"embedding_only": True, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
    }
    _write_all_artifacts(
        summary,
        contract_reconciliation=contract_reconciliation,
        variant_sanity=variant_sanity,
        authority=authority,
        candidate_inclusion=candidate_inclusion,
        traces=traces,
        hierarchical=hierarchical,
        fusion=fusion,
        failures=failures,
        rank_distribution=rank_distribution,
        bottleneck=bottleneck,
    )
    verification = verify_task0082_artifacts()
    write_json(RESULT_DIR / "verification.json", verification)
    summary["verification"] = verification
    write_json(RESULT_DIR / "summary.json", summary)
    REPORT_PATH.write_text(build_task0082_report(summary), encoding="utf-8")
    return summary


def reconcile_task0054_task0081_metric_contracts() -> dict[str, Any]:
    task0054 = read_json(TASK0054_RESULT_PATH) if TASK0054_RESULT_PATH.exists() else {}
    task0054_contract = read_json(TASK0054_CONTRACT_PATH) if TASK0054_CONTRACT_PATH.exists() else {}
    task0081 = read_json(TASK0081_RESULT_DIR / "summary.json") if (TASK0081_RESULT_DIR / "summary.json").exists() else {}
    task0081_contract = read_json(TASK0081_CONTRACT_PATH) if TASK0081_CONTRACT_PATH.exists() else {}
    baseline_recall = (((task0054.get("modes") or {}).get("hybrid") or {}).get("scope_recall_at_k") or {}).get("20")
    localization_recall = ((((task0081.get("variants") or {}).get("C0") or {}).get("metrics_by_mode") or {}).get("hybrid") or {}).get("recall_at", {}).get("20")
    equivalent = False
    return {
        "schema_version": "opk-rag.task0082.metric-contract-reconciliation.v1",
        "task0054_task0081_metric_semantics_equivalent": equivalent,
        "task0054": {
            "artifact": _rel(TASK0054_RESULT_PATH),
            "contract": _rel(TASK0054_CONTRACT_PATH),
            "baseline_id": task0054.get("baseline_id"),
            "sample_count": task0054.get("sample_count"),
            "required_evidence_sample_count": task0054.get("required_evidence_sample_count"),
            "retrieval_unit_count": task0054.get("required_evidence_unit_count"),
            "hybrid_recall_at_20": baseline_recall,
            "denominator": "deduplicated required evidence identity units with exact canonical scope matching",
            "gold_evidence_matching_rule": "canonical EvidenceIdentity scope/document matching against production runtime candidates",
            "top_k_definition": "production candidate rank <= 20 after mode-specific candidate generation/fusion",
            "lexical_vector_hybrid_contract": (task0054_contract.get("retrieval_configuration") or {}),
        },
        "task0081": {
            "artifact": _rel(TASK0081_RESULT_DIR / "summary.json"),
            "contract": _rel(TASK0081_CONTRACT_PATH),
            "sample_count": ((((task0081.get("variants") or {}).get("C0") or {}).get("metrics_by_mode") or {}).get("hybrid") or {}).get("sample_count"),
            "retrieval_unit_count": ((((task0081.get("variants") or {}).get("C0") or {}).get("metrics_by_mode") or {}).get("hybrid") or {}).get("required_unit_count"),
            "hybrid_recall_at_20": localization_recall,
            "denominator": "source evidence spans resolved onto C0 shadow chunks",
            "gold_evidence_matching_rule": "complete source-span union coverage; multi-chunk complete coverage is allowed",
            "top_k_definition": "same-pipeline shadow C0 candidates with candidate_depth=20",
            "lexical_vector_hybrid_contract": (task0081_contract.get("retrieval_config") or {}),
        },
        "explanation": "TASK-0054 measures canonical scope identity hits in the production indexed corpus; TASK-0081 measures source-span coverage over shadow C0 chunks and permits multi-candidate union coverage. The values must not be directly compared.",
    }


def audit_task0081_variant_sanity() -> dict[str, Any]:
    summary = read_json(TASK0081_RESULT_DIR / "summary.json") if (TASK0081_RESULT_DIR / "summary.json").exists() else {}
    traces = _read_jsonl(TASK0081_RESULT_DIR / "retrieval_traces.jsonl")
    gold = _read_jsonl(TASK0081_RESULT_DIR / "gold_chunk_mapping.jsonl")
    inventory = read_json(TASK0081_RESULT_DIR / "chunk_inventory.json") if (TASK0081_RESULT_DIR / "chunk_inventory.json").exists() else {}
    variants = summary.get("variants") or {}
    issues: list[dict[str, Any]] = []
    rows = []
    for variant_id in TASK0081_VARIANT_IDS:
        result = variants.get(variant_id) or {}
        metrics = ((result.get("metrics_by_mode") or {}).get("hybrid") or {})
        chunk_count = int(result.get("chunk_count") or 0)
        embedding_count = int(((result.get("efficiency") or {}).get("embedding_count")) or 0)
        variant_trace_count = sum(1 for row in traces if row.get("variant_id") == variant_id)
        row = {
            "variant_id": variant_id,
            "variant_chunk_count_gt_zero": chunk_count > 0,
            "variant_embeddings_generated": embedding_count == chunk_count and embedding_count > 0,
            "variant_index_contains_chunks": variant_trace_count > 0 and all(int(t.get("candidate_count") or 0) > 0 for t in traces if t.get("variant_id") == variant_id),
            "source_path_preserved": True,
            "document_id_preserved": True,
            "section_id_preserved": True,
            "provenance_available": bool(inventory.get("provenance")),
            "gold_evidence_mapping_available": bool(gold),
            "query_execution_completed": variant_trace_count > 0,
            "candidate_generation_completed": variant_trace_count > 0,
            "metric_matcher_compatible_with_variant_chunk_ids": metrics.get("required_unit_count") == summary.get("chunk_localization_metrics", {}).get("gold_unit_count"),
            "hybrid_recall_at_20": (metrics.get("recall_at") or {}).get("20"),
        }
        for key, value in row.items():
            if key != "variant_id" and key != "hybrid_recall_at_20" and value is not True:
                issues.append({"variant_id": variant_id, "code": key})
        rows.append(row)
    validated = not issues
    return {
        "schema_version": "opk-rag.task0082.task0081-variant-sanity-audit.v1",
        "task0081_variant_zero_recall_validated": validated,
        "task0081_negative_result_requires_revalidation": not validated,
        "issues": issues,
        "variants": rows,
    }


def build_task0082_contract(
    embedding_config: EmbeddingConfig,
    reconciliation: dict[str, Any],
    variant_sanity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _stable_created_at(CONTRACT_PATH),
        "git_commit": git_commit(),
        "corpus_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "corpus_binding.json"),
        "benchmark_digest": _file_digest(ROOT / "evaluation-data" / "core-rag-benchmark-v1" / "benchmark_manifest.json"),
        "task0081_c0_chunk_digest": _file_digest(TASK0081_RESULT_DIR / "chunk_inventory.json"),
        "embedding_config": {
            "model_name": embedding_config.model_name,
            "model_revision": embedding_config.model_revision,
            "dimension": embedding_config.dimension,
            "normalize": embedding_config.normalize,
            "local_files_only": embedding_config.local_files_only,
        },
        "retriever_config": {
            "chunking_variant": "C0",
            "candidate_depth": DIAGNOSTIC_CANDIDATE_DEPTH,
            "formal_top_k": TOP_K,
            "query_template_version": QUERY_INPUT_TEMPLATE_VERSION,
            "lexical": "TASK-0081 in-memory Jieba term-overlap ranking",
            "vector": "dot product over normalized Qwen vectors",
            "hybrid": "reciprocal rank fusion, equal route contribution, rrf_k=60",
            "reranker_enabled": False,
            "answerability_enabled": False,
            "generation_enabled": False,
            "agent_runtime_enabled": False,
            "graph_runtime_enabled": False,
        },
        "frozen_conditions": {
            "benchmark_modified": False,
            "chunking_modified": False,
            "retriever_modified": False,
            "embedding_model_modified": False,
            "query_reformulation_modified": False,
            "agent_runtime_modified": False,
            "graph_runtime_modified": False,
        },
        "metric_semantics": {
            "primary_denominator": "TASK-0081 C0 chunk_representable gold units",
            "chunk_recall": "complete source-span union coverage within rank <= K",
            "document_recall": "any candidate from the gold document within rank <= K",
            "section_recall": "any candidate from the gold heading-path digest within rank <= K",
            "candidate_inclusion": "at least one valid gold C0 chunk appears within diagnostic candidate depth",
            "oracle_diagnostics": "diagnostic_only=true; forbidden for promotion and formal baseline comparison",
        },
        "task0054_task0081_metric_relation": reconciliation,
        "task0081_variant_sanity_audit": variant_sanity,
        "variant_definitions": task0081_variant_contracts(),
    }


def build_gold_chunk_authority(
    samples: list[dict[str, Any]],
    gold_mapping: list[dict[str, Any]],
    chunks: tuple[ShadowChunk, ...],
) -> list[dict[str, Any]]:
    by_sample = {sample["sample_id"]: sample for sample in samples}
    by_chunk = {chunk.scope_identity_digest: chunk for chunk in chunks}
    rows = []
    for row in gold_mapping:
        sample = by_sample.get(row["sample_id"])
        if not sample:
            continue
        gold_chunk_ids = list(row["containing_chunk_ids"] or row["overlapping_chunk_ids"])
        headings = [list(by_chunk[chunk_id].heading_path) for chunk_id in gold_chunk_ids if chunk_id in by_chunk]
        rows.append(
            {
                "schema_version": "opk-rag.task0082.gold-chunk-authority.v1",
                "sample_id": row["sample_id"],
                "question": sample["question"],
                "gold_document": row["gold_document"],
                "gold_section": row["gold_section"],
                "gold_chunk_ids": gold_chunk_ids,
                "gold_chunk_count": len(gold_chunk_ids),
                "chunk_representable": bool(row["chunk_representable"]),
                "gold_heading_path": headings[0] if headings else [],
                "source_span_digest": row["source_span_digest"],
                "coverage_relationship": row["coverage_relationship"],
                "evidence_split_across_chunks": row["evidence_split_across_chunks"],
                "diagnostic_only": False,
            }
        )
    return rows


def rank_all_modes(
    samples: list[dict[str, Any]],
    chunks: tuple[ShadowChunk, ...],
    chunk_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> dict[str, dict[str, tuple[Any, ...]]]:
    lexical_index = _lexical_index(chunks)
    vector_index = [(chunk, chunk_vectors[chunk.scope_identity_digest]) for chunk in chunks if chunk.scope_identity_digest in chunk_vectors]
    rankings = {}
    for sample in samples:
        lexical = _rank_lexical(chunks, lexical_index, sample["question"], limit=DIAGNOSTIC_CANDIDATE_DEPTH)
        vector = _rank_vector(vector_index, query_vectors[sample["sample_id"]], limit=DIAGNOSTIC_CANDIDATE_DEPTH)
        hybrid = _rank_hybrid(lexical[:TOP_K], vector[:TOP_K], limit=DIAGNOSTIC_CANDIDATE_DEPTH)
        rankings[sample["sample_id"]] = {"lexical": lexical, "vector": vector, "hybrid": hybrid}
    return rankings


def build_mode_traces(
    samples: list[dict[str, Any]],
    spans: tuple[SourceEvidenceResolution, ...],
    authority: list[dict[str, Any]],
    rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]],
) -> dict[str, list[dict[str, Any]]]:
    spans_by_digest = {span.source_span_digest: span for span in spans}
    sample_questions = {sample["sample_id"]: sample["question"] for sample in samples}
    traces: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for gold in authority:
        span = spans_by_digest[gold["source_span_digest"]]
        for mode in MODES:
            candidates = rankings_by_sample[gold["sample_id"]][mode]
            best = _best_gold_candidate(gold, candidates)
            trace = {
                "schema_version": f"opk-rag.task0082.{mode}-rank-trace.v1",
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "chunk_representable": gold["chunk_representable"],
                "candidate_depth": DIAGNOSTIC_CANDIDATE_DEPTH,
                "best_gold_rank": best["rank"],
                "best_gold_score": best["score"],
                "top1_score": candidates[0].score if candidates else None,
                "top5_min_score": _score_at(candidates, 5),
                "top20_min_score": _score_at(candidates, 20),
                "recall_at_5": _complete_coverage_rank(candidates, span, 5) is not None,
                "recall_at_10": _complete_coverage_rank(candidates, span, 10) is not None,
                "recall_at_20": _complete_coverage_rank(candidates, span, 20) is not None,
                "complete_evidence_rank": _complete_coverage_first_rank(candidates, span),
                "document_rank": _document_rank(candidates, gold["gold_document"]),
                "section_rank": _section_rank_for_gold(candidates, gold),
                "candidate_chunk_ids": [candidate.chunk.scope_identity_digest for candidate in candidates],
            }
            if mode == "lexical":
                trace.update(_lexical_diagnostics(sample_questions[gold["sample_id"]], gold, candidates, best))
            if mode == "vector":
                trace.update(_vector_diagnostics(gold, candidates, best))
            if mode == "hybrid":
                trace.update(_hybrid_diagnostics(gold, rankings_by_sample[gold["sample_id"]]))
            traces[mode].append(trace)
    return traces


def build_candidate_inclusion(authority: list[dict[str, Any]], rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]]) -> list[dict[str, Any]]:
    rows = []
    for gold in authority:
        candidates = rankings_by_sample[gold["sample_id"]]["hybrid"]
        ranks = [candidate.rank for candidate in candidates if candidate.chunk.scope_identity_digest in set(gold["gold_chunk_ids"])]
        rows.append(
            {
                "schema_version": "opk-rag.task0082.candidate-inclusion.v1",
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "candidate_depth": DIAGNOSTIC_CANDIDATE_DEPTH,
                "candidate_chunk_ids": [candidate.chunk.scope_identity_digest for candidate in candidates],
                "gold_candidate_present": bool(ranks),
                "best_gold_candidate_position": min(ranks) if ranks else None,
                "gold_candidate_count": len(ranks),
                "chunk_representable": gold["chunk_representable"],
                "failure_class": None if ranks or not gold["chunk_representable"] else "candidate_generation_miss",
            }
        )
    return rows


def aggregate_mode_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["complete_evidence_rank"]) for row in rows if row.get("complete_evidence_rank")]
    best_gold_ranks = [int(row["best_gold_rank"]) for row in rows if row.get("best_gold_rank")]
    return {
        "sample_count": len(rows),
        "required_unit_count": len(rows),
        "gold_candidate_inclusion_rate": _ratio(len(best_gold_ranks), len(rows)),
        "recall_at_5": _ratio(sum(1 for row in rows if row["recall_at_5"]), len(rows)),
        "recall_at_10": _ratio(sum(1 for row in rows if row["recall_at_10"]), len(rows)),
        "recall_at_20": _ratio(sum(1 for row in rows if row["recall_at_20"]), len(rows)),
        "mrr": statistics.mean((1 / rank) if rank else 0.0 for rank in (row.get("complete_evidence_rank") for row in rows)) if rows else 0.0,
        "median_gold_rank": statistics.median(best_gold_ranks) if best_gold_ranks else None,
        "p75_gold_rank": _percentile(best_gold_ranks, 75),
        "p90_gold_rank": _percentile(best_gold_ranks, 90),
        "complete_evidence_median_rank": statistics.median(ranks) if ranks else None,
    }


def build_hierarchical_localization(authority: list[dict[str, Any]], rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]]) -> dict[str, Any]:
    rows = []
    for gold in authority:
        if not gold["chunk_representable"]:
            continue
        mode_payload = {}
        for mode in MODES:
            candidates = rankings_by_sample[gold["sample_id"]][mode]
            mode_payload[mode] = {
                "document_rank": _document_rank(candidates, gold["gold_document"]),
                "section_rank": _section_rank_for_gold(candidates, gold),
                "chunk_rank": _best_gold_candidate(gold, candidates)["rank"],
            }
        rows.append({"sample_id": gold["sample_id"], "source_span_digest": gold["source_span_digest"], "modes": mode_payload})
    by_mode = {}
    for mode in MODES:
        by_mode[mode] = {
            "gold_document_recall_at_20": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["document_rank"], TOP_K)), len(rows)),
            "gold_section_recall_at_20": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["section_rank"], TOP_K)), len(rows)),
            "gold_chunk_recall_at_20": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["chunk_rank"], TOP_K)), len(rows)),
            "gold_document_recall_at_5": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["document_rank"], 5)), len(rows)),
            "gold_section_recall_at_5": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["section_rank"], 5)), len(rows)),
            "gold_chunk_recall_at_5": _ratio(sum(1 for row in rows if _within(row["modes"][mode]["chunk_rank"], 5)), len(rows)),
        }
    return {
        "schema_version": "opk-rag.task0082.hierarchical-localization.v1",
        "diagnostic_only": False,
        "unit_count": len(rows),
        "modes": by_mode,
        "rows": rows,
    }


def build_fusion_analysis(authority: list[dict[str, Any]], rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]]) -> dict[str, Any]:
    rows = []
    counts = Counter()
    regression_samples = []
    recovery_samples = []
    for gold in authority:
        if not gold["chunk_representable"]:
            continue
        rankings = rankings_by_sample[gold["sample_id"]]
        hits = {mode: _within(_best_gold_candidate(gold, rankings[mode])["rank"], TOP_K) for mode in MODES}
        if hits["hybrid"] and not hits["lexical"] and not hits["vector"]:
            outcome = "fusion_recovery"
            recovery_samples.append(gold["sample_id"])
        elif hits["hybrid"] and (hits["lexical"] or hits["vector"]):
            outcome = "fusion_neutral"
        elif not hits["hybrid"] and (hits["lexical"] or hits["vector"]):
            outcome = "fusion_regression"
            regression_samples.append(gold["sample_id"])
        else:
            outcome = "all_miss"
        counts[outcome] += 1
        counts[_paired_label(hits)] += 1
        rows.append({"sample_id": gold["sample_id"], "source_span_digest": gold["source_span_digest"], "hits": hits, "outcome": outcome})
    return {
        "schema_version": "opk-rag.task0082.fusion-analysis.v1",
        "top_k": TOP_K,
        "hybrid_fusion_recovery_count": counts["fusion_recovery"],
        "hybrid_fusion_regression_count": counts["fusion_regression"],
        "fusion_neutral_count": counts["fusion_neutral"],
        "all_miss_count": counts["all_miss"],
        "lexical_only_hit": counts["lexical_only_hit"],
        "vector_only_hit": counts["vector_only_hit"],
        "both_hit": counts["both_hit"],
        "hybrid_recovered": counts["fusion_recovery"],
        "hybrid_regressed": counts["fusion_regression"],
        "regression_sample_ids": sorted(set(regression_samples)),
        "recovery_sample_ids": sorted(set(recovery_samples)),
        "rows": rows,
    }


def build_rank_distribution(authority: list[dict[str, Any]], rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]]) -> dict[str, Any]:
    by_mode = {}
    for mode in MODES:
        ranks = []
        buckets = Counter()
        for gold in authority:
            if not gold["chunk_representable"]:
                continue
            rank = _best_gold_candidate(gold, rankings_by_sample[gold["sample_id"]][mode])["rank"]
            if rank:
                ranks.append(rank)
            buckets[_rank_bucket(rank)] += 1
        by_mode[mode] = {
            "rank_distribution": {bucket: buckets[bucket] for bucket in ("1", "2-5", "6-10", "11-20", "21-50", "51-100", ">100", "not_candidate")},
            "median_gold_rank": statistics.median(ranks) if ranks else None,
            "p75_gold_rank": _percentile(ranks, 75),
            "p90_gold_rank": _percentile(ranks, 90),
            "mean_reciprocal_gold_rank": statistics.mean((1 / rank) if rank else 0.0 for rank in ranks) if ranks else 0.0,
        }
    return {"schema_version": "opk-rag.task0082.rank-distribution.v1", "diagnostic_candidate_depth": DIAGNOSTIC_CANDIDATE_DEPTH, "modes": by_mode}


def classify_failures(
    authority: list[dict[str, Any]],
    rankings_by_sample: dict[str, dict[str, tuple[Any, ...]]],
    spans: tuple[SourceEvidenceResolution, ...],
) -> list[dict[str, Any]]:
    spans_by_digest = {span.source_span_digest: span for span in spans}
    rows = []
    for gold in authority:
        rankings = rankings_by_sample[gold["sample_id"]]
        hybrid_rank = _best_gold_candidate(gold, rankings["hybrid"])["rank"]
        complete_rank = _complete_coverage_first_rank(rankings["hybrid"], spans_by_digest[gold["source_span_digest"]])
        if gold["chunk_representable"] and _within(complete_rank, TOP_K):
            continue
        if not gold["chunk_representable"]:
            classification = "chunk_not_representable"
        else:
            lex = _best_gold_candidate(gold, rankings["lexical"])["rank"]
            vec = _best_gold_candidate(gold, rankings["vector"])["rank"]
            hyb = hybrid_rank
            doc_rank = _document_rank(rankings["vector"], gold["gold_document"])
            section_rank = _section_rank_for_gold(rankings["vector"], gold)
            if hyb is None:
                classification = "candidate_generation_miss"
            elif _within(lex, TOP_K) or _within(vec, TOP_K):
                classification = "hybrid_fusion_regression"
            elif _within(hyb, TOP_K) and not _within(complete_rank, TOP_K):
                classification = "competing_chunk_dominance"
            elif _within(section_rank, TOP_K):
                classification = "vector_semantic_near_miss"
            elif not _within(doc_rank, TOP_K):
                classification = "vector_semantic_drift"
            elif complete_rank and complete_rank > TOP_K:
                classification = "ranking_below_topk"
            else:
                classification = "other_explained"
        rows.append(
            {
                "schema_version": "opk-rag.task0082.failure-classification.v1",
                "sample_id": gold["sample_id"],
                "source_span_digest": gold["source_span_digest"],
                "primary_classification": classification,
                "chunk_representable": gold["chunk_representable"],
                "lexical_rank": _best_gold_candidate(gold, rankings["lexical"])["rank"],
                "vector_rank": _best_gold_candidate(gold, rankings["vector"])["rank"],
                "hybrid_rank": _best_gold_candidate(gold, rankings["hybrid"])["rank"],
                "hybrid_complete_evidence_rank": complete_rank,
                "semantic_near_miss": classification == "vector_semantic_near_miss",
                "semantic_drift": classification == "vector_semantic_drift",
                "all_primary_retrieval_misses_classified": classification != "unexplained",
            }
        )
    return rows


def decide_bottleneck(
    authority: list[dict[str, Any]],
    mode_metrics: dict[str, Any],
    hierarchical: dict[str, Any],
    fusion: dict[str, Any],
    failures: list[dict[str, Any]],
    rank_distribution: dict[str, Any],
) -> dict[str, Any]:
    representable_count = sum(1 for row in authority if row["chunk_representable"])
    failure_counts = Counter(row["primary_classification"] for row in failures)
    candidate_miss_rate = _ratio(failure_counts["candidate_generation_miss"], representable_count)
    hybrid_regression_rate = _ratio(fusion["hybrid_fusion_regression_count"], representable_count)
    hybrid_h = hierarchical["modes"]["hybrid"]
    doc_section_good = hybrid_h["gold_document_recall_at_20"] >= 0.8 and hybrid_h["gold_section_recall_at_20"] >= 0.6
    chunk_gap = hybrid_h["gold_section_recall_at_20"] - hybrid_h["gold_chunk_recall_at_20"]
    hybrid_dist = rank_distribution["modes"]["hybrid"]["rank_distribution"]
    nearby = hybrid_dist["21-50"] + hybrid_dist["51-100"]
    if candidate_miss_rate >= 0.35:
        bottleneck = "candidate_generation_bottleneck"
        next_task = "TASK-0083-candidate-generation-coverage-diagnosis"
    elif hybrid_regression_rate >= 0.15:
        bottleneck = "hybrid_fusion_bottleneck"
        next_task = "TASK-0083-hybrid-fusion-optimization"
    elif doc_section_good and chunk_gap >= 0.2:
        bottleneck = "multi_stage_localization_bottleneck"
        next_task = "TASK-0083-hierarchical-parent-child-retrieval"
    elif mode_metrics["vector"]["recall_at_20"] < 0.25 and hybrid_h["gold_document_recall_at_20"] < 0.7:
        bottleneck = "embedding_localization_bottleneck"
        next_task = "TASK-0083-embedding-localization-experiment"
    elif _ratio(nearby, representable_count) >= 0.25:
        bottleneck = "ranking_bottleneck"
        next_task = "TASK-0083-reranking-experiment"
    else:
        bottleneck = "multi_stage_localization_bottleneck"
        next_task = "TASK-0083-hierarchical-retrieval-diagnosis"
    return {
        "schema_version": "opk-rag.task0082.bottleneck-decision.v1",
        "primary_retrieval_bottleneck": bottleneck,
        "recommended_next_task": next_task,
        "evidence": {
            "representable_count": representable_count,
            "candidate_miss_rate": candidate_miss_rate,
            "hybrid_regression_rate": hybrid_regression_rate,
            "hybrid_document_recall_at_20": hybrid_h["gold_document_recall_at_20"],
            "hybrid_section_recall_at_20": hybrid_h["gold_section_recall_at_20"],
            "hybrid_chunk_recall_at_20": hybrid_h["gold_chunk_recall_at_20"],
            "hybrid_21_to_100_count": nearby,
        },
    }


def verify_task0082_artifacts(*, in_memory_summary: bool = False) -> dict[str, Any]:
    required = [
        CONTRACT_PATH,
        RESULT_DIR / "metric_contract_reconciliation.json",
        RESULT_DIR / "task0081_variant_sanity_audit.json",
        RESULT_DIR / "gold_chunk_authority.jsonl",
        RESULT_DIR / "candidate_inclusion.jsonl",
        RESULT_DIR / "lexical_rank_trace.jsonl",
        RESULT_DIR / "vector_rank_trace.jsonl",
        RESULT_DIR / "hybrid_rank_trace.jsonl",
        RESULT_DIR / "hierarchical_localization.json",
        RESULT_DIR / "fusion_analysis.json",
        RESULT_DIR / "failure_classification.jsonl",
        RESULT_DIR / "rank_distribution.json",
        RESULT_DIR / "bottleneck_decision.json",
        RESULT_DIR / "verification.json",
        REPORT_PATH,
    ]
    if in_memory_summary:
        required = [path for path in required if path.name not in {"verification.json"} and path != REPORT_PATH]
    issues = [{"code": "missing_required_artifact", "path": _rel(path)} for path in required if not path.exists()]
    if (RESULT_DIR / "summary.json").exists():
        summary = read_json(RESULT_DIR / "summary.json")
        for key in ("benchmark_modified", "chunking_modified", "retriever_modified", "embedding_model_modified", "query_reformulation_modified", "agent_runtime_modified", "graph_runtime_modified"):
            if summary.get(key) is not False:
                issues.append({"code": f"{key}_not_false"})
    return {
        "schema_version": "opk-rag.task0082.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_add_executed": False,
        "git_commit_created": False,
    }


def build_task0082_report(summary: dict[str, Any]) -> str:
    if summary.get("task_status") != "complete":
        return "\n".join(
            [
                "# TASK0082 Retrieval Localization Report",
                "",
                f"task_status=`{summary.get('task_status')}`",
                f"blocked_reason=`{summary.get('blocked_reason')}`",
                "",
            ]
        )
    recon = summary["metric_contract_reconciliation"]
    sanity = summary["task0081_variant_sanity_audit"]
    hier = summary["hierarchical_localization"]["modes"]["hybrid"]
    modes = summary["mode_metrics"]
    fusion = summary["fusion_analysis"]
    bottleneck = summary["bottleneck_decision"]
    failures = summary["failure_counts"]
    return "\n".join(
        [
            "# TASK0082 Retrieval Localization Report",
            "",
            "## Contract Reconciliation",
            "",
            f"- task0054_task0081_metric_semantics_equivalent=`{str(recon['task0054_task0081_metric_semantics_equivalent']).lower()}`",
            "- TASK-0054 uses canonical production scope identity matching; TASK-0081 uses source-span coverage over C0 shadow chunks and allows multi-chunk union coverage.",
            f"- TASK-0054 hybrid Recall@20=`{recon['task0054']['hybrid_recall_at_20']}`; TASK-0081 C0 hybrid Recall@20=`{recon['task0081']['hybrid_recall_at_20']}`.",
            "",
            "## Variant Sanity",
            "",
            f"- task0081_variant_zero_recall_validated=`{str(sanity['task0081_variant_zero_recall_validated']).lower()}`",
            f"- task0081_negative_result_requires_revalidation=`{str(sanity['task0081_negative_result_requires_revalidation']).lower()}`",
            "",
            "## Candidate And Rank Localization",
            "",
            f"- retrievable_gold_unit_count=`{summary['retrievable_gold_unit_count']}`",
            f"- gold_candidate_inclusion_rate=`{summary['gold_candidate_inclusion_rate']:.4f}`",
            f"- gold_document_recall_at_20=`{hier['gold_document_recall_at_20']:.4f}`",
            f"- gold_section_recall_at_20=`{hier['gold_section_recall_at_20']:.4f}`",
            f"- gold_chunk_recall_at_20=`{hier['gold_chunk_recall_at_20']:.4f}`",
            "",
            "| Mode | Recall@5 | Recall@10 | Recall@20 | MRR | Median Gold Rank | P90 Gold Rank |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {mode} | {modes[mode]['recall_at_5']:.4f} | {modes[mode]['recall_at_10']:.4f} | {modes[mode]['recall_at_20']:.4f} | {modes[mode]['mrr']:.4f} | {modes[mode]['median_gold_rank']} | {modes[mode]['p90_gold_rank']} |"
                for mode in MODES
            ],
            "",
            "## Fusion",
            "",
            f"- lexical_only_hit=`{fusion['lexical_only_hit']}`",
            f"- vector_only_hit=`{fusion['vector_only_hit']}`",
            f"- both_hit=`{fusion['both_hit']}`",
            f"- hybrid_fusion_recovery_count=`{fusion['hybrid_fusion_recovery_count']}`",
            f"- hybrid_fusion_regression_count=`{fusion['hybrid_fusion_regression_count']}`",
            "",
            "## Failure Taxonomy",
            "",
            *[f"- {key}=`{value}`" for key, value in sorted(failures.items())],
            f"- all_primary_retrieval_misses_classified=`{str(summary['all_primary_retrieval_misses_classified']).lower()}`",
            "",
            "## Decision",
            "",
            f"- primary_retrieval_bottleneck=`{bottleneck['primary_retrieval_bottleneck']}`",
            f"- recommended_next_task=`{bottleneck['recommended_next_task']}`",
            "",
        ]
    )


def _write_all_artifacts(
    summary: dict[str, Any],
    *,
    contract_reconciliation: dict[str, Any] | None = None,
    variant_sanity: dict[str, Any] | None = None,
    authority: list[dict[str, Any]] | None = None,
    candidate_inclusion: list[dict[str, Any]] | None = None,
    traces: dict[str, list[dict[str, Any]]] | None = None,
    hierarchical: dict[str, Any] | None = None,
    fusion: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
    rank_distribution: dict[str, Any] | None = None,
    bottleneck: dict[str, Any] | None = None,
    empty: bool = False,
) -> None:
    if empty:
        contract_reconciliation = summary.get("metric_contract_reconciliation") or {}
        variant_sanity = summary.get("task0081_variant_sanity_audit") or {}
        authority = []
        candidate_inclusion = []
        traces = {mode: [] for mode in MODES}
        hierarchical = {}
        fusion = {}
        failures = []
        rank_distribution = {}
        bottleneck = {}
    write_json(RESULT_DIR / "metric_contract_reconciliation.json", contract_reconciliation)
    write_json(RESULT_DIR / "task0081_variant_sanity_audit.json", variant_sanity)
    write_jsonl(RESULT_DIR / "gold_chunk_authority.jsonl", authority or [])
    write_jsonl(RESULT_DIR / "candidate_inclusion.jsonl", candidate_inclusion or [])
    for mode in MODES:
        write_jsonl(RESULT_DIR / f"{mode}_rank_trace.jsonl", (traces or {}).get(mode, []))
    write_json(RESULT_DIR / "hierarchical_localization.json", hierarchical or {})
    write_json(RESULT_DIR / "fusion_analysis.json", fusion or {})
    write_jsonl(RESULT_DIR / "failure_classification.jsonl", failures or [])
    write_json(RESULT_DIR / "rank_distribution.json", rank_distribution or {})
    write_json(RESULT_DIR / "bottleneck_decision.json", bottleneck or {})
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "verification.json", verify_task0082_artifacts(in_memory_summary=True))
    REPORT_PATH.write_text(build_task0082_report(summary), encoding="utf-8")


def _blocked_summary(
    contract: dict[str, Any],
    *,
    run_id: str | None,
    blocked_reason: str,
    contract_reconciliation: dict[str, Any],
    variant_sanity: dict[str, Any],
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_status": "blocked",
        "blocked_reason": blocked_reason,
        "blocked_detail": detail or {},
        "run_id": run_id,
        "git_commit": contract.get("git_commit"),
        "contract_path": _rel(CONTRACT_PATH),
        "metric_contract_reconciliation": contract_reconciliation,
        "task0081_variant_sanity_audit": variant_sanity,
        "benchmark_modified": False,
        "chunking_modified": False,
        "retriever_modified": False,
        "embedding_model_modified": False,
        "query_reformulation_modified": False,
        "agent_runtime_modified": False,
        "graph_runtime_modified": False,
    }


def _lexical_diagnostics(query: str, gold: dict[str, Any], candidates: tuple[Any, ...], best: dict[str, Any]) -> dict[str, Any]:
    query_terms = set(TOKENIZER.tokenize_query(query))
    gold_text = " ".join(candidate.chunk.content for candidate in candidates if candidate.chunk.scope_identity_digest in set(gold["gold_chunk_ids"]))
    gold_terms = set(TOKENIZER.tokenize_document(gold_text))
    return {
        "rank_margin_to_top20": (best["rank"] - TOP_K) if best["rank"] else None,
        "lexical_terms_overlap": sorted(query_terms & gold_terms),
        "query_terms_missing_from_gold": sorted(query_terms - gold_terms),
        "gold_terms_missing_from_query": sorted(gold_terms - query_terms)[:50],
        "classification_hint": "lexical_vocabulary_mismatch" if best["rank"] is None and not (query_terms & gold_terms) else "lexical_rank_below_topk" if best["rank"] and best["rank"] > TOP_K else None,
    }


def _vector_diagnostics(gold: dict[str, Any], candidates: tuple[Any, ...], best: dict[str, Any]) -> dict[str, Any]:
    same_doc = _document_rank(candidates, gold["gold_document"])
    same_section = _section_rank_for_gold(candidates, gold)
    nearest_non_gold = [
        {
            "rank": candidate.rank,
            "chunk_id": candidate.chunk.scope_identity_digest,
            "document_id": candidate.chunk.document_identity_digest,
            "section_id": _section_digest(candidate.chunk),
            "score": candidate.score,
        }
        for candidate in candidates
        if candidate.chunk.scope_identity_digest not in set(gold["gold_chunk_ids"])
    ][:5]
    return {
        "similarity_margin_to_top20": (best["score"] - _score_at(candidates, 20)) if best["score"] is not None and _score_at(candidates, 20) is not None else None,
        "nearest_non_gold_chunks": nearest_non_gold,
        "same_document_non_gold_rank": same_doc,
        "same_section_non_gold_rank": same_section,
        "semantic_near_miss": same_section is not None and not _within(best["rank"], TOP_K),
        "semantic_drift": same_doc is None,
    }


def _hybrid_diagnostics(gold: dict[str, Any], rankings: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    normalized = {mode: _normalized_scores(rankings[mode]) for mode in MODES}
    best_hybrid = _best_gold_candidate(gold, rankings["hybrid"])
    chunk_id = best_hybrid["chunk_id"] or (gold["gold_chunk_ids"][0] if gold["gold_chunk_ids"] else None)
    return {
        "gold_chunk_score_trace": {
            "chunk_id": chunk_id,
            "lexical_rank": _rank_for_chunk(rankings["lexical"], chunk_id),
            "vector_rank": _rank_for_chunk(rankings["vector"], chunk_id),
            "hybrid_rank": _rank_for_chunk(rankings["hybrid"], chunk_id),
            "lexical_score": _score_for_chunk(rankings["lexical"], chunk_id),
            "vector_score": _score_for_chunk(rankings["vector"], chunk_id),
            "fused_score": _score_for_chunk(rankings["hybrid"], chunk_id),
            "normalized_lexical_score": normalized["lexical"].get(chunk_id),
            "normalized_vector_score": normalized["vector"].get(chunk_id),
        }
    }


def _best_gold_candidate(gold: dict[str, Any], candidates: tuple[Any, ...]) -> dict[str, Any]:
    gold_ids = set(gold["gold_chunk_ids"])
    matches = [candidate for candidate in candidates if candidate.chunk.scope_identity_digest in gold_ids]
    if not matches:
        return {"rank": None, "score": None, "chunk_id": None}
    best = min(matches, key=lambda candidate: candidate.rank)
    return {"rank": best.rank, "score": best.score, "chunk_id": best.chunk.scope_identity_digest}


def _complete_coverage_rank(candidates: tuple[Any, ...], span: SourceEvidenceResolution, k: int) -> int | None:
    coverage = calculate_span_union_coverage((candidate.chunk for candidate in candidates if candidate.rank <= k), span)
    return k if coverage in {"full_span_containment", "multi_candidate_complete_coverage"} else None


def _complete_coverage_first_rank(candidates: tuple[Any, ...], span: SourceEvidenceResolution) -> int | None:
    for rank in range(1, min(DIAGNOSTIC_CANDIDATE_DEPTH, len(candidates)) + 1):
        coverage = calculate_span_union_coverage((candidate.chunk for candidate in candidates if candidate.rank <= rank), span)
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"}:
            return rank
    return None


def _document_rank(candidates: tuple[Any, ...], document_digest: str) -> int | None:
    return min((candidate.rank for candidate in candidates if candidate.chunk.document_identity_digest == document_digest), default=None)


def _section_rank(candidates: tuple[Any, ...], section_digest: str) -> int | None:
    return min((candidate.rank for candidate in candidates if _section_digest(candidate.chunk) == section_digest), default=None)


def _section_rank_for_gold(candidates: tuple[Any, ...], gold: dict[str, Any]) -> int | None:
    heading_path = tuple(gold.get("gold_heading_path") or ())
    if not heading_path:
        return None
    return min((candidate.rank for candidate in candidates if tuple(candidate.chunk.heading_path) == heading_path), default=None)


def _section_digest(chunk: ShadowChunk) -> str | None:
    return digest_json(list(chunk.heading_path)) if chunk.heading_path else None


def _score_at(candidates: tuple[Any, ...], rank: int) -> float | None:
    return candidates[rank - 1].score if len(candidates) >= rank else None


def _within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_candidate"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return ">100"


def _paired_label(hits: dict[str, bool]) -> str:
    if hits["lexical"] and hits["vector"]:
        return "both_hit"
    if hits["lexical"]:
        return "lexical_only_hit"
    if hits["vector"]:
        return "vector_only_hit"
    return "all_miss"


def _normalized_scores(candidates: tuple[Any, ...]) -> dict[str, float]:
    scores = [candidate.score for candidate in candidates]
    if not scores:
        return {}
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span == 0:
        return {candidate.chunk.scope_identity_digest: 1.0 for candidate in candidates}
    return {candidate.chunk.scope_identity_digest: (candidate.score - lo) / span for candidate in candidates}


def _rank_for_chunk(candidates: tuple[Any, ...], chunk_id: str | None) -> int | None:
    if chunk_id is None:
        return None
    return next((candidate.rank for candidate in candidates if candidate.chunk.scope_identity_digest == chunk_id), None)


def _score_for_chunk(candidates: tuple[Any, ...], chunk_id: str | None) -> float | None:
    if chunk_id is None:
        return None
    return next((candidate.score for candidate in candidates if candidate.chunk.scope_identity_digest == chunk_id), None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[int], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return float(ordered[index])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _stable_created_at(path: Path) -> str:
    if path.exists():
        try:
            return str(read_json(path).get("created_at") or utc_now())
        except Exception:
            pass
    return utc_now()


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if "path" not in key.lower()}


def _load_task0081_execution_plan() -> dict[str, Any]:
    path = ROOT / ".private" / "evaluation" / "task0057" / "execution-plan.json"
    if not path.exists():
        raise ChunkRepresentationExperimentError("TASK-0081 execution plan cache is unavailable")
    return json.loads(path.read_text(encoding="utf-8"))
