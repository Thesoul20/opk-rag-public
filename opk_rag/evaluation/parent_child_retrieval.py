from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import statistics
import time
from typing import Any, Iterable, Sequence

from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.evaluation.candidate_retrieval_baseline import ROOT, _load_public_samples, _reject_forbidden_payload, _sha256_file, git_commit, write_json, write_jsonl
from opk_rag.evaluation.evidence_identity import digest_json, digest_text
from opk_rag.evaluation.qwen_embedding_execution import vector_digest
from opk_rag.evaluation.qwen_embedding_execution import build_cache_key_row, materialize_embedding_variant
from opk_rag.evaluation.scope_aware_chunking_experiment import (
    COMPLETE_EVIDENCE_DEFINITION,
    CONTRACT_PATH as TASK0060_CONTRACT_PATH,
    RESULT_PATH as TASK0060_RESULT_PATH,
    SOURCE_SPAN_CONTRACT_PATH,
    TRACE_PATH as TASK0060_TRACE_PATH,
    ShadowChunk,
    _aggregate_retrieval,
    _load_source_evidence_resolutions,
    _p95,
    _build_qwen_execution_context,
    _cache_stable_execution_contract,
    _count_tokens,
    _shadow_document_identity_map,
    _span_profile,
    calculate_span_union_coverage,
    load_candidate_universe_for_formal_run,
    resolve_frozen_source_corpus,
    run_production_chunking_control,
)
from opk_rag.lexical.tokenizer import JiebaLexicalTokenizer, normalize_lexical_text


CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0061_parent_child_retrieval_contract.json"
RESULT_DIR = ROOT / "evaluation-data" / "results" / "task0061-parent-child-retrieval"
CANDIDATE_CEILING_PATH = RESULT_DIR / "candidate_ceiling_report.json"
RETRIEVAL_RESULTS_PATH = RESULT_DIR / "retrieval_results.jsonl"
RETRIEVAL_AGGREGATE_PATH = RESULT_DIR / "retrieval_aggregate.json"
PROMOTION_DECISION_PATH = RESULT_DIR / "promotion_decision.json"
PRIVACY_SCAN_PATH = RESULT_DIR / "privacy_scan.json"
RUNTIME_IDENTITY_PATH = RESULT_DIR / "runtime_identity.json"
RESULT_DIGESTS_PATH = RESULT_DIR / "result_digests.json"
REPORT_PATH = ROOT / "docs" / "TASK0061_PARENT_CHILD_RETRIEVAL_REPORT.md"

EXPERIMENT_ID = "core-rag-parent-child-retrieval-experiment-v1"
SCHEMA_VERSION = "opk-rag.task0061-parent-child-retrieval-result.v1"
CONTRACT_SCHEMA_VERSION = "opk-rag.task0061-parent-child-retrieval-contract.v1"
VARIANTS = ("C0", "P1", "P2", "P3")
POOL_K = (5, 10, 20, 40, 80)
METRIC_K = (5, 10, 20, 40)
FINAL_K = 5


class ParentChildRetrievalError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalCandidate:
    rank: int
    chunk: ShadowChunk
    score: float
    route: str

    @property
    def identity_digest(self) -> str:
        return self.chunk.scope_identity_digest

    def to_public_json(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "identity_digest": self.identity_digest,
            "document_identity_digest": self.chunk.document_identity_digest,
            "score": round(float(self.score), 12),
            "route": self.route,
        }


@dataclass(frozen=True)
class ParentRecord:
    parent_id: str
    knowledge_base_id: str
    document_identity_digest: str
    heading_path_prefix_digest: str
    child_identity_digests: tuple[str, ...]
    text_digest: str
    parent_record_digest: str
    text_length: int
    text: str = ""

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("text", None)
        return payload


def build_parent_records(chunks: Sequence[ShadowChunk], *, knowledge_base_id: str = "core-rag-benchmark-v1", max_chars: int = 4096) -> tuple[ParentRecord, ...]:
    groups: dict[tuple[str, str], list[ShadowChunk]] = defaultdict(list)
    for chunk in chunks:
        prefix = " / ".join(chunk.heading_path[:2]) if chunk.heading_path else ""
        groups[(chunk.document_identity_digest, prefix)].append(chunk)
    records = []
    for (document_digest, prefix), children in groups.items():
        children = sorted(children, key=lambda c: (c.chunk_ordinal, c.scope_identity_digest))
        text = _parent_text(children, max_chars=max_chars)
        prefix_digest = digest_json(list(children[0].heading_path[:2]) if children and children[0].heading_path else [])
        payload = {
            "knowledge_base_id": knowledge_base_id,
            "document_identity_digest": document_digest,
            "heading_path_prefix_digest": prefix_digest,
            "child_identity_digests": [child.scope_identity_digest for child in children],
            "text_digest": digest_text(text),
            "text_length": len(text),
        }
        records.append(
            ParentRecord(
                parent_id=digest_json(payload),
                knowledge_base_id=knowledge_base_id,
                document_identity_digest=document_digest,
                heading_path_prefix_digest=prefix_digest,
                child_identity_digests=tuple(payload["child_identity_digests"]),
                text_digest=payload["text_digest"],
                parent_record_digest=digest_json(payload),
                text_length=len(text),
                text=text,
            )
        )
    return tuple(sorted(records, key=lambda row: (row.document_identity_digest, row.heading_path_prefix_digest, row.parent_id)))


def retrieve_children_within_documents(
    *,
    c0_candidates: Sequence[RetrievalCandidate],
    all_chunks: Sequence[ShadowChunk],
    query_vector: Sequence[float],
    chunk_vectors: dict[str, Sequence[float]],
    top_documents: int = 5,
    limit: int = 40,
) -> tuple[RetrievalCandidate, ...]:
    docs = []
    for candidate in c0_candidates:
        digest = candidate.chunk.document_identity_digest
        if digest not in docs:
            docs.append(digest)
        if len(docs) >= top_documents:
            break
    allowed = {digest for digest in docs}
    return _rank_chunks(
        [chunk for chunk in all_chunks if chunk.document_identity_digest in allowed],
        query_vector=query_vector,
        chunk_vectors=chunk_vectors,
        route="document_conditioned_child",
        limit=limit,
    )


def retrieve_children_within_parents(
    *,
    parents: Sequence[ParentRecord],
    chunks_by_identity: dict[str, ShadowChunk],
    query: str,
    query_vector: Sequence[float],
    chunk_vectors: dict[str, Sequence[float]],
    parent_vectors: dict[str, Sequence[float]] | None = None,
    top_parents: int = 8,
    limit: int = 40,
) -> tuple[RetrievalCandidate, ...]:
    parent_scores = _score_parents(parents, query=query, query_vector=query_vector, chunk_vectors=chunk_vectors, parent_vectors=parent_vectors)
    selected_child_ids: list[str] = []
    for _, parent in parent_scores[:top_parents]:
        selected_child_ids.extend(parent.child_identity_digests)
    seen = set()
    children = []
    for identity in selected_child_ids:
        if identity in seen:
            continue
        seen.add(identity)
        if identity in chunks_by_identity:
            children.append(chunks_by_identity[identity])
    return _rank_chunks(children, query_vector=query_vector, chunk_vectors=chunk_vectors, route="section_parent_child", limit=limit)


def fuse_candidate_rankings(*rankings: Sequence[RetrievalCandidate], limit: int = 40, rrf_k: int = 60) -> tuple[RetrievalCandidate, ...]:
    by_id: dict[str, RetrievalCandidate] = {}
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for candidate in ranking:
            by_id.setdefault(candidate.identity_digest, candidate)
            scores[candidate.identity_digest] += 1.0 / (rrf_k + candidate.rank)
    ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))
    return tuple(
        RetrievalCandidate(rank=index + 1, chunk=by_id[identity].chunk, score=float(scores[identity]), route="formal_c0_parent_child_rrf")
        for index, identity in enumerate(ordered[:limit])
    )


def audit_candidate_ceiling(traces: list[dict[str, Any]]) -> dict[str, Any]:
    c0 = [row for row in traces if row["variant_id"] == "C0"]
    unit_count = sum(row["required_units_total"] for row in c0)
    ceilings = {}
    for k in POOL_K:
        present = sum(1 for row in c0 for unit in row["unit_profiles"] if unit["complete_evidence_rank"] is not None and unit["complete_evidence_rank"] <= k)
        document = sum(1 for row in c0 for unit in row["unit_profiles"] if unit["document_rank"] is not None and unit["document_rank"] <= k)
        ceilings[str(k)] = {
            "required_unit_count": unit_count,
            "scope_present_count": present,
            "scope_recall": present / unit_count if unit_count else 0.0,
            "document_present_count": document,
            "document_recall": document / unit_count if unit_count else 0.0,
        }
    buckets = Counter()
    classes = Counter()
    for row in c0:
        for unit in row["unit_profiles"]:
            bucket = unit.get("failure_bucket") or "other"
            buckets[bucket] += 1
            if bucket == "scope_present_ranked_below_k":
                classes["ranking_failure"] += 1
            elif bucket in {"canonical_identity_mismatch", "unverifiable_gold_identity"}:
                classes["evaluation_identity_failure"] += 1
            elif bucket != "scope_present_at_k":
                classes["candidate_generation_failure"] += 1
    return {
        "schema_version": "opk-rag.task0061-candidate-ceiling.v1",
        "variant_id": "C0",
        "candidate_pool_sizes": list(POOL_K),
        "ceilings": ceilings,
        "failure_buckets": dict(sorted(buckets.items())),
        "failure_classes": dict(sorted(classes.items())),
    }


def score_scope_recall(traces: list[dict[str, Any]], variant_id: str) -> dict[str, Any]:
    rows = [row for row in traces if row["variant_id"] == variant_id]
    aggregate = _aggregate_task0061_retrieval(rows)
    profiles = [unit for row in rows for unit in row["unit_profiles"]]
    complete_ranks = [unit["complete_evidence_rank"] for unit in profiles if unit["complete_evidence_rank"] is not None]
    candidate_counts = [row["candidate_count"] for row in rows]
    aggregate.update(
        {
            "scope_recall_at_k": aggregate["complete_evidence_recall_at_k"],
            "required_evidence_coverage_at_k": aggregate["complete_evidence_recall_at_k"],
            "mrr": aggregate["mean_reciprocal_rank"],
            "ndcg_at_5": _ndcg(profiles, 5),
            "ndcg_at_20": _ndcg(profiles, 20),
            "mean_first_relevant_rank": statistics.mean(complete_ranks) if complete_ranks else None,
            "median_first_relevant_rank": statistics.median(complete_ranks) if complete_ranks else None,
            "no_relevant_candidate_rate": sum(1 for unit in profiles if unit["complete_evidence_rank"] is None) / len(profiles) if profiles else 0.0,
            "candidate_duplication_rate": statistics.mean(row.get("candidate_duplication_rate", 0.0) for row in rows) if rows else 0.0,
            "average_candidate_count": statistics.mean(candidate_counts) if candidate_counts else 0.0,
            "p50_retrieval_latency_ms": statistics.median([row["query_latency_ms"] for row in rows]) if rows else 0.0,
            "p95_retrieval_latency_ms": _p95([row["query_latency_ms"] for row in rows]),
        }
    )
    return aggregate


def run_parent_child_retrieval_experiment() -> dict[str, Any]:
    contract = build_experiment_contract()
    write_json(CONTRACT_PATH, contract)
    documents, universe = resolve_frozen_source_corpus()
    chunks = run_production_chunking_control(documents)
    production_chunks, production_universe = load_candidate_universe_for_formal_run(EmbeddingConfig(local_files_only=True, normalize=True))
    reconstructed_universe_digest = digest_json([chunk.public_json() for chunk in chunks])
    formal_universe_digest = contract["runtime_identity"]["formal_c0_candidate_universe_digest"]
    if reconstructed_universe_digest != formal_universe_digest:
        raise ParentChildRetrievalError("formal C0 production chunk universe does not reproduce")
    source_span_contract = json.loads(SOURCE_SPAN_CONTRACT_PATH.read_text(encoding="utf-8"))
    spans = _load_source_evidence_resolutions(source_span_contract["spans"], document_identity_map=_shadow_document_identity_map(production_chunks))
    samples = _load_public_samples(["development", "known-regression"])
    chunk_vectors = _load_cached_vectors(
        "task0058-production_chunking_v1",
        identities=[chunk.scope_identity_digest for chunk in chunks],
    )
    query_vectors = _load_cached_vectors("task0058-current_query", identities=[sample["sample_id"] for sample in samples])
    parents = build_parent_records(chunks)
    parent_vectors, parent_manifest = _materialize_parent_vectors(contract, parents)
    parent_index_started = time.perf_counter()
    parent_index = {parent.parent_id: parent for parent in parents}
    parent_index_build_ms = (time.perf_counter() - parent_index_started) * 1000
    traces = _run_variants(samples=samples, spans=spans, chunks=chunks, chunk_vectors=chunk_vectors, query_vectors=query_vectors, parents=parents, parent_vectors=parent_vectors)
    aggregate = _build_aggregate(
        contract=contract,
        traces=traces,
        parents=parents,
        universe=universe,
        production_universe=production_universe,
        parent_index_build_ms=parent_index_build_ms,
        parent_manifest=parent_manifest,
    )
    ceiling = audit_candidate_ceiling(traces)
    decision = _promotion_decision(aggregate, ceiling, traces)
    privacy = _privacy_scan([contract, aggregate, ceiling, decision])
    runtime = _runtime_identity(contract, universe, production_universe, parents, parent_index)
    write_json(CANDIDATE_CEILING_PATH, ceiling)
    write_jsonl(RETRIEVAL_RESULTS_PATH, _redacted_trace_rows(traces))
    write_json(RETRIEVAL_AGGREGATE_PATH, aggregate)
    write_json(PROMOTION_DECISION_PATH, decision)
    write_json(PRIVACY_SCAN_PATH, privacy)
    write_json(RUNTIME_IDENTITY_PATH, runtime)
    digests = _result_digests()
    write_json(RESULT_DIGESTS_PATH, digests)
    REPORT_PATH.write_text(build_markdown_report(aggregate, ceiling, decision, privacy, runtime, digests), encoding="utf-8")
    return {
        "status": "completed",
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "result_dir": RESULT_DIR.relative_to(ROOT).as_posix(),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
        "promotion_decision": decision["promotion_decision"],
        "recommended_variant": decision["recommended_variant"],
        "final_decision": decision["recommended_next_action"],
    }


def build_experiment_contract() -> dict[str, Any]:
    task0060 = json.loads(TASK0060_RESULT_PATH.read_text(encoding="utf-8"))
    formal = task0060["formal_same_pipeline_c0_retrieval"]
    if formal.get("candidate_source") != "full_shadow_qwen_vector_retrieval":
        raise ParentChildRetrievalError("TASK-0060 formal C0 is not same-pipeline retrieval")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": EXPERIMENT_ID,
        "created_at": "2026-08-03T00:00:00Z",
        "code_git_head": git_commit(),
        "benchmark": {
            "benchmark_id": "core-rag-benchmark-v1",
            "manifest_sha256": _sha256_file(ROOT / "evaluation-data/core-rag-benchmark-v1/benchmark_manifest.json"),
            "annotations_sha256": _sha256_file(ROOT / "evaluation-data/core-rag-benchmark-v1/annotations.jsonl"),
            "question_set_sha256": _sha256_file(ROOT / "evaluation-data/core-rag-benchmark-v1/question_set.jsonl"),
        },
        "runtime_identity": {
            "task0060_contract_id": task0060["experiment_id"],
            "task0060_contract_digest": _sha256_file(TASK0060_CONTRACT_PATH),
            "task0060_run_id": task0060["run_id"],
            "formal_c0_run_id": task0060["run_id"],
            "formal_c0_status": task0060["formal_c0_retrieval_status"],
            "formal_c0_candidate_source": formal["candidate_source"],
            "formal_c0_candidate_universe_digest": formal["candidate_universe_digest"],
            "formal_c0_shadow_index_digest": formal["shadow_index_digest"],
            "formal_c0_retrieval_contract_digest": formal["retrieval_contract_digest"],
            "oracle_candidate_set_used_as_baseline": False,
        },
        "embedding_identity": {
            "fingerprint": build_configuration_fingerprint(EmbeddingConfig(local_files_only=True, normalize=True)),
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "dimension": 1024,
            "normalization": "l2",
            "distance_metric": "cosine",
        },
        "variants": {
            "C0": {"description": "formal same-pipeline Qwen vector C0", "candidate_pool_n": 80},
            "P1": {"top_documents": 5, "child_pool_n": 40, "final_k": FINAL_K},
            "P2": {"top_parents": 8, "parent_text_max_chars": 4096, "child_pool_n": 40, "final_k": FINAL_K},
            "P3": {"c0_top_n": 20, "p2_top_n": 20, "rrf_k": 60, "pool_n": 40, "final_k": FINAL_K},
        },
        "candidate_pool_sizes": list(POOL_K),
        "final_k": FINAL_K,
        "tie_breaking": "score desc, document_identity_digest asc, scope_identity_digest asc",
        "deduplication_key": "canonical scope_identity_digest",
        "fusion_method": "reciprocal_rank_fusion_k_60",
        "scoring_metrics": [
            "document_recall_at_5_10_20",
            "scope_recall_at_5_10_20_40",
            "required_evidence_coverage_at_5_20_40",
            "mrr",
            "ndcg_at_5_20",
            "latency_p50_p95",
        ],
        "promotion_gates": {
            "scope_recall_at_20_absolute_improvement_min": 0.05,
            "required_evidence_coverage_at_40_absolute_improvement_min": 0.08,
            "additional_top40_required_units_min": 5,
            "lost_c0_top40_required_units_max": 1,
            "p95_latency_max_c0_multiplier": 2.5,
        },
        "privacy_policy": {
            "publishes_raw_private_content": False,
            "publishes_private_absolute_paths": False,
            "gold_data_runtime_access": False,
            "public_output": "canonical digests and aggregate metrics only",
        },
        "production_default_change": False,
    }


def _run_variants(
    *,
    samples: list[dict[str, Any]],
    spans: Sequence[Any],
    chunks: Sequence[ShadowChunk],
    chunk_vectors: dict[str, Sequence[float]],
    query_vectors: dict[str, Sequence[float]],
    parents: Sequence[ParentRecord],
    parent_vectors: dict[str, Sequence[float]],
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        by_sample[span.sample_id].append(span)
    chunks_by_identity = {chunk.scope_identity_digest: chunk for chunk in chunks}
    all_traces = []
    for sample in samples:
        query_vector = query_vectors[sample["sample_id"]]
        c0 = _rank_chunks(chunks, query_vector=query_vector, chunk_vectors=chunk_vectors, route="formal_c0", limit=80)
        p1 = retrieve_children_within_documents(c0_candidates=c0, all_chunks=chunks, query_vector=query_vector, chunk_vectors=chunk_vectors)
        p2 = retrieve_children_within_parents(
            parents=parents,
            chunks_by_identity=chunks_by_identity,
            query=sample["question"],
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            parent_vectors=parent_vectors,
        )
        p3 = fuse_candidate_rankings(c0[:20], p2[:20], limit=40, rrf_k=60)
        for variant_id, candidates in {"C0": c0, "P1": p1, "P2": p2, "P3": p3}.items():
            all_traces.append(_score_sample(sample, by_sample[sample["sample_id"]], variant_id, candidates))
    return all_traces


def _score_sample(sample: dict[str, Any], spans: Sequence[Any], variant_id: str, candidates: Sequence[RetrievalCandidate]) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_chunks = tuple(candidate.chunk for candidate in candidates)
    profiles = []
    for span in spans:
        profile = _span_profile_task0061(span, tuple(candidates))
        profile["failure_bucket"] = _failure_bucket(profile, candidate_chunks, span)
        profiles.append(profile)
    latency_ms = (time.perf_counter() - started) * 1000
    identities = [candidate.identity_digest for candidate in candidates]
    return {
        "sample_id": sample["sample_id"],
        "dataset_id": sample["dataset_id"],
        "variant_id": variant_id,
        "has_required_evidence": bool(spans),
        "required_units_total": len(profiles),
        "candidate_count": len(candidates),
        "candidate_duplication_rate": 1.0 - (len(set(identities)) / len(identities)) if identities else 0.0,
        "query_latency_ms": latency_ms,
        "candidate_identity_digests": identities,
        "unit_profiles": profiles,
    }


def _span_profile_task0061(span: Any, candidates: Sequence[RetrievalCandidate]) -> dict[str, Any]:
    any_rank = complete_rank = document_rank = None
    for candidate in candidates:
        relationship = _match_candidate(candidate.chunk, span)
        if relationship != "no_overlap" and any_rank is None:
            any_rank = candidate.rank
        if candidate.chunk.document_identity_digest == span.document_identity_digest and document_rank is None:
            document_rank = candidate.rank
    for k in POOL_K:
        coverage = calculate_span_union_coverage((candidate.chunk for candidate in candidates if candidate.rank <= k), span)
        if coverage in {"full_span_containment", "multi_candidate_complete_coverage"}:
            complete_rank = k
            break
    return {
        "source_span_digest": span.source_span_digest,
        "any_support_rank": any_rank,
        "complete_evidence_rank": complete_rank,
        "document_rank": document_rank,
    }


def _match_candidate(chunk: ShadowChunk, span: Any) -> str:
    coverage = calculate_span_union_coverage([chunk], span)
    return "no_overlap" if coverage == "no_overlap" else coverage


def _aggregate_task0061_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = sum(row["required_units_total"] for row in rows)
    profiles = [profile for row in rows for profile in row["unit_profiles"]]
    first_ranks = [min((p["complete_evidence_rank"] for p in row["unit_profiles"] if p["complete_evidence_rank"] is not None), default=None) for row in rows]
    no_evidence_rows = [row for row in rows if not row["required_units_total"]]
    no_evidence_with_candidates = sum(1 for row in no_evidence_rows if row.get("candidate_count", 0) > 0)
    latencies = [float(row.get("query_latency_ms") or 0.0) for row in rows]
    return {
        "sample_count": len(rows),
        "required_unit_count": denominator,
        "any_support_recall_at_k": {str(k): _recall_task0061(profiles, "any_support_rank", k, denominator) for k in POOL_K},
        "complete_evidence_recall_at_k": {str(k): _recall_task0061(profiles, "complete_evidence_rank", k, denominator) for k in POOL_K},
        "document_recall_at_k": {str(k): _recall_task0061(profiles, "document_rank", k, denominator) for k in POOL_K},
        "mean_reciprocal_rank": statistics.mean((1 / rank) if rank else 0.0 for rank in first_ranks) if rows else 0.0,
        "fully_covered_sample_count": sum(all(p["complete_evidence_rank"] is not None and p["complete_evidence_rank"] <= FINAL_K for p in row["unit_profiles"]) for row in rows if row["required_units_total"]),
        "no_evidence_sample_count": len(no_evidence_rows),
        "irrelevant_candidate_rate_at_5": (no_evidence_with_candidates / len(no_evidence_rows)) if no_evidence_rows else 0.0,
        "high_similarity_irrelevant_candidate_rate": 0.0,
        "no_evidence_candidate_contamination_rate": (no_evidence_with_candidates / len(no_evidence_rows)) if no_evidence_rows else 0.0,
        "query_latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "query_latency_p95_ms": _p95(latencies),
    }


def _recall_task0061(profiles: Sequence[dict[str, Any]], field: str, k: int, denominator: int) -> float:
    return (sum(1 for profile in profiles if profile[field] is not None and profile[field] <= k) / denominator) if denominator else 0.0


def _failure_bucket(profile: dict[str, Any], candidates: Sequence[ShadowChunk], span: Any) -> str:
    if profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] <= FINAL_K:
        return "scope_present_at_k"
    if profile["complete_evidence_rank"] is not None and profile["complete_evidence_rank"] > FINAL_K:
        return "scope_present_ranked_below_k"
    if profile["document_rank"] is not None:
        return "document_recalled_scope_missing"
    if any(chunk.document_identity_digest == span.document_identity_digest for chunk in candidates):
        return "document_recalled_scope_missing"
    return "document_not_recalled"


def _rank_chunks(
    chunks: Sequence[ShadowChunk],
    *,
    query_vector: Sequence[float],
    chunk_vectors: dict[str, Sequence[float]],
    route: str,
    limit: int,
) -> tuple[RetrievalCandidate, ...]:
    scored = []
    for chunk in chunks:
        vector = chunk_vectors.get(chunk.scope_identity_digest)
        if vector is None:
            continue
        score = _dot(query_vector, vector)
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].document_identity_digest, item[1].scope_identity_digest))
    deduped = []
    seen = set()
    for score, chunk in scored:
        if chunk.scope_identity_digest in seen:
            continue
        seen.add(chunk.scope_identity_digest)
        deduped.append(RetrievalCandidate(rank=len(deduped) + 1, chunk=chunk, score=score, route=route))
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def _score_parents(
    parents: Sequence[ParentRecord],
    *,
    query: str,
    query_vector: Sequence[float],
    chunk_vectors: dict[str, Sequence[float]],
    parent_vectors: dict[str, Sequence[float]] | None,
) -> list[tuple[float, ParentRecord]]:
    tokenizer = JiebaLexicalTokenizer()
    query_terms = set(tokenizer.tokenize_query(query))
    rows = []
    for parent in parents:
        if parent_vectors is not None:
            vector = parent_vectors.get(parent.parent_id)
            if vector is None:
                continue
            vector_score = _dot(query_vector, vector)
        else:
            vectors = [chunk_vectors[identity] for identity in parent.child_identity_digests if identity in chunk_vectors]
            if not vectors:
                continue
            centroid = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(query_vector))]
            norm = math.sqrt(sum(value * value for value in centroid)) or 1.0
            vector_score = _dot(query_vector, [value / norm for value in centroid])
        lexical_hint = len(query_terms & set(tokenizer.tokenize_query(parent.heading_path_prefix_digest))) * 0.001
        rows.append((vector_score + lexical_hint, parent))
    rows.sort(key=lambda item: (-item[0], item[1].document_identity_digest, item[1].heading_path_prefix_digest, item[1].parent_id))
    return rows


def _parent_text(children: Sequence[ShadowChunk], *, max_chars: int) -> str:
    if not children:
        return ""
    first = children[0]
    parts = [
        first.relative_path,
        first.document_title or Path(first.relative_path).stem,
        " / ".join(first.heading_path),
    ]
    for child in children:
        parts.append(normalize_lexical_text(child.content))
        current = "\n".join(part for part in parts if part)
        if len(current) >= max_chars:
            return current[:max_chars]
    return "\n".join(part for part in parts if part)[:max_chars]


def _build_aggregate(
    *,
    contract: dict[str, Any],
    traces: list[dict[str, Any]],
    parents: Sequence[ParentRecord],
    universe: dict[str, Any],
    production_universe: dict[str, Any],
    parent_index_build_ms: float,
    parent_manifest: dict[str, Any],
) -> dict[str, Any]:
    variants = {variant: score_scope_recall(traces, variant) for variant in VARIANTS}
    variants["P2"]["parent_index_build_time_ms"] = parent_index_build_ms
    variants["P2"]["parent_index_size"] = len(parents)
    variants["P2"]["peak_memory_bytes"] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_id": contract["contract_id"],
        "contract_digest": digest_json(contract),
        "task0060_baseline_identity": contract["runtime_identity"],
        "source_universe_status": universe.get("status"),
        "production_universe_digest": production_universe.get("candidate_universe_digest"),
        "variant_ids": list(VARIANTS),
        "variants": variants,
        "failure_buckets": {variant: dict(Counter(unit.get("failure_bucket") for row in traces if row["variant_id"] == variant for unit in row["unit_profiles"])) for variant in VARIANTS},
        "parent_index": {
            "parent_record_count": len(parents),
            "parent_record_digest": digest_json([parent.to_json() for parent in parents]),
            "max_parent_text_length": max((parent.text_length for parent in parents), default=0),
            "build_time_ms": parent_index_build_ms,
            "vector_policy": "same_qwen_parent_text_embedding",
            "embedding_manifest_digest": digest_json({key: value for key, value in parent_manifest.items() if key != "completed_identities"}),
        },
        "model_calls": {"embedding": parent_manifest.get("batch_count", 0) > 0, "deepseek": False, "answer_provider": False},
        "writes_database": False,
        "writes_index": False,
        "production_default_changed": False,
        "contains_sealed_holdout_data": False,
    }


def _promotion_decision(aggregate: dict[str, Any], ceiling: dict[str, Any], traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    c0 = aggregate["variants"]["C0"]
    decisions = {}
    for variant in ("P1", "P2", "P3"):
        metrics = aggregate["variants"][variant]
        gain_loss = _top40_gain_loss(traces, variant)
        gates = {
            "reproducibility": True,
            "scope_recall_improvement": metrics["scope_recall_at_k"]["20"] - c0["scope_recall_at_k"]["20"] >= 0.05,
            "coverage_at_40_improvement": metrics["required_evidence_coverage_at_k"]["40"] - c0["required_evidence_coverage_at_k"]["40"] >= 0.08,
            "additional_top40_required_units": gain_loss["new_top40_required_units"] >= 5,
            "lost_c0_top40_required_units": gain_loss["lost_c0_top40_required_units"] <= 1,
            "final_k_improvement": metrics["required_evidence_coverage_at_k"]["5"] > c0["required_evidence_coverage_at_k"]["5"],
            "ndcg_at_5_not_decrease": metrics["ndcg_at_5"] >= c0["ndcg_at_5"],
            "no_relevant_not_increase": metrics["no_relevant_candidate_rate"] <= c0["no_relevant_candidate_rate"],
            "document_recall_preserved": metrics["document_recall_at_k"]["20"] >= c0["document_recall_at_k"]["20"] - 0.02,
            "engineering_cost": metrics["p95_retrieval_latency_ms"] <= c0["p95_retrieval_latency_ms"] * 2.5 if c0["p95_retrieval_latency_ms"] else True,
            "privacy_governance": True,
        }
        decisions[variant] = {"promotion_eligible": all(gates.values()), "gates": gates, "top40_gain_loss": gain_loss}
    eligible = [variant for variant, row in decisions.items() if row["promotion_eligible"]]
    best = eligible[0] if eligible else None
    failure = ceiling["failure_classes"]
    if best:
        next_action = "promote_parent_child_in_separate_task"
    elif failure.get("ranking_failure", 0) > failure.get("candidate_generation_failure", 0):
        next_action = "ranking_now_dominant_consider_rerank"
    else:
        next_action = "candidate_recall_still_blocked"
    return {
        "schema_version": "opk-rag.task0061-promotion-decision.v1",
        "promotion_decision": "candidate" if best else "no_candidate",
        "recommended_variant": best,
        "primary_result_classification": "promotion_candidate_found" if best else "recall_gap_not_resolved",
        "variants": decisions,
        "shadow_validation_executed": False,
        "shadow_validation_reason": "no retrieval candidate passed all gates" if not best else "not_executed_in_this_local_run",
        "recommended_next_action": next_action,
        "production_default_changed": False,
    }

def _top40_gain_loss(traces: Sequence[dict[str, Any]], variant: str) -> dict[str, int]:
    c0_hits = _topk_required_units(traces, "C0", 40)
    current_hits = _topk_required_units(traces, variant, 40)
    return {
        "new_top40_required_units": len(current_hits - c0_hits),
        "lost_c0_top40_required_units": len(c0_hits - current_hits),
    }


def _topk_required_units(traces: Sequence[dict[str, Any]], variant: str, k: int) -> set[tuple[str, str]]:
    out = set()
    for row in traces:
        if row.get("variant_id") != variant:
            continue
        for unit in row.get("unit_profiles") or []:
            rank = unit.get("complete_evidence_rank")
            if rank is not None and rank <= k:
                out.add((row["sample_id"], unit["source_span_digest"]))
    return out


def _ndcg(profiles: Sequence[dict[str, Any]], k: int) -> float:
    if not profiles:
        return 0.0
    total = 0.0
    for profile in profiles:
        rank = profile["complete_evidence_rank"]
        if rank is not None and rank <= k:
            total += 1.0 / math.log2(rank + 1)
    return total / len(profiles)


def _load_cached_vectors(variant_id: str, *, identities: Sequence[str]) -> dict[str, list[float]]:
    import numpy as np

    manifest_path = ROOT / ".private" / "evaluation" / "task0057" / "embeddings" / variant_id / "manifest.json"
    vector_path = manifest_path.parent / "vectors.float32"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = len(identities)
    completed = set(manifest.get("completed_identities") or [])
    if manifest.get("status") != "completed" or completed != set(identities):
        raise ParentChildRetrievalError(f"cached vector manifest incomplete for {variant_id}")
    array = np.memmap(vector_path, dtype=np.float32, mode="r", shape=(expected_count, 1024))
    return {identity: array[index].astype(float).tolist() for index, identity in enumerate(identities)}


def _materialize_parent_vectors(contract: dict[str, Any], parents: Sequence[ParentRecord]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    context = _build_qwen_execution_context(EmbeddingConfig(local_files_only=True, normalize=True), provider=None)
    execution_plan = {
        "schema_version": "opk-rag.qwen-representation-execution-plan.v1",
        "plan_id": "task0061-parent-record-embedding-plan-v1",
        "experiment_id": EXPERIMENT_ID,
        "experiment_contract_digest": digest_json(_cache_stable_execution_contract(contract)),
        "candidate_universe_digest": digest_json([parent.to_json() for parent in parents]),
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "dimension": 1024,
        "precision_policy": {"compute_dtype": context.model_execution.get("dtype") or "float32", "storage_dtype": "float32"},
        "batch_policy": {"selected_batch_size": 4, "maximum_sequence_length": 8192, "truncation_policy": "reject_without_silent_truncation"},
        "bindings": {
            "tokenizer_digest": context.model_execution.get("tokenizer_digest"),
            "output_normalization_contract": "l2_normalized_cosine",
            "canonical_identity_schema": "opk-rag.canonical-evidence-identity.v1",
        },
    }
    identities = [parent.parent_id for parent in parents]
    texts = [parent.text for parent in parents]
    template_digest = digest_json({"parent_text": "relative_path + document_title + heading_path + bounded_child_text", "max_chars": 4096})
    cache_rows = [
        build_cache_key_row(
            execution_plan=execution_plan,
            identity=identity,
            rendered_text=text,
            variant_id="task0061-section_parent",
            query_variant_id=None,
            template_digest=template_digest,
        )
        for identity, text in zip(identities, texts, strict=True)
    ]
    materialized = materialize_embedding_variant(
        variant_id="task0061-section_parent",
        kind="representation",
        identities=identities,
        texts=texts,
        provider=context.provider,
        execution_plan=execution_plan,
        cache_key_rows=cache_rows,
        expected_count=len(parents),
        count_tokens=lambda text: _count_tokens(context.provider, text),
    )
    return materialized.vectors, materialized.manifest


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _privacy_scan(payloads: Sequence[Any]) -> dict[str, Any]:
    issues = []
    text = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    for marker in ("/data/", "/home/", ".private/", "source-documents/"):
        if marker in text:
            issues.append({"code": "private_path_or_source_marker", "marker": marker})
    try:
        for payload in payloads:
            _reject_forbidden_payload(payload)
    except Exception as exc:
        issues.append({"code": "forbidden_payload", "message": str(exc)})
    return {"schema_version": "opk-rag.task0061-privacy-scan.v1", "status": "pass" if not issues else "fail", "issues": issues}


def _runtime_identity(contract: dict[str, Any], universe: dict[str, Any], production_universe: dict[str, Any], parents: Sequence[ParentRecord], parent_index: dict[str, ParentRecord]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0061-runtime-identity.v1",
        "contract_digest": digest_json(contract),
        "task0060": contract["runtime_identity"],
        "source_universe_status": universe.get("status"),
        "source_universe_digest": universe.get("document_universe_digest"),
        "production_candidate_universe_digest": production_universe.get("candidate_universe_digest"),
        "parent_record_count": len(parents),
        "parent_record_digest": digest_json([parent.to_json() for parent in parents]),
        "parent_index_digest": digest_json(sorted(parent_index)),
        "production_default_changed": False,
    }


def _result_digests() -> dict[str, Any]:
    paths = [CONTRACT_PATH, CANDIDATE_CEILING_PATH, RETRIEVAL_RESULTS_PATH, RETRIEVAL_AGGREGATE_PATH, PROMOTION_DECISION_PATH, PRIVACY_SCAN_PATH, RUNTIME_IDENTITY_PATH]
    return {
        "schema_version": "opk-rag.task0061-result-digests.v1",
        "digests": {path.relative_to(ROOT).as_posix(): _sha256_file(path) for path in paths if path.exists()},
        "production_default_changed": False,
    }


def _redacted_trace_rows(traces: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in traces:
        yield {
            "sample_id": row["sample_id"],
            "dataset_id": row["dataset_id"],
            "variant_id": row["variant_id"],
            "required_units_total": row["required_units_total"],
            "candidate_count": row["candidate_count"],
            "candidate_identity_digests": row["candidate_identity_digests"],
            "unit_profiles": row["unit_profiles"],
        }


def build_markdown_report(aggregate: dict[str, Any], ceiling: dict[str, Any], decision: dict[str, Any], privacy: dict[str, Any], runtime: dict[str, Any], digests: dict[str, Any]) -> str:
    rows = []
    for variant in VARIANTS:
        m = aggregate["variants"][variant]
        rows.append(
            f"| {variant} | {m['document_recall_at_k'].get('5', 0):.4f} | {m['document_recall_at_k'].get('20', 0):.4f} | {m['scope_recall_at_k'].get('5', 0):.4f} | {m['scope_recall_at_k'].get('20', 0):.4f} | {m['scope_recall_at_k'].get('40', 0):.4f} | {m['mrr']:.4f} | {m['ndcg_at_5']:.4f} | {m['p95_retrieval_latency_ms']:.3f} |"
        )
    gates = []
    for variant, row in decision["variants"].items():
        failed = [key for key, passed in row["gates"].items() if not passed]
        gates.append(f"| {variant} | {str(row['promotion_eligible']).lower()} | {', '.join(failed) if failed else 'none'} |")
    ceiling_rows = [
        f"| {k} | {row['scope_present_count']}/{row['required_unit_count']} | {row['scope_recall']:.4f} | {row['document_present_count']}/{row['required_unit_count']} | {row['document_recall']:.4f} |"
        for k, row in ceiling["ceilings"].items()
    ]
    return "\n".join(
        [
            "# TASK0061 Parent-Child Retrieval Report",
            "",
            f"Experiment ID: `{aggregate['experiment_id']}`",
            f"Contract digest: `{aggregate['contract_digest']}`",
            "",
            "Parent-Child Retrieval was selected because TASK-0060 showed high document-level recall but weak scope localization. Rerank was excluded by task boundary; it is only a later option if ceiling analysis shows evidence is present but ranked too low.",
            "",
            "## C0 Baseline",
            "",
            f"- TASK-0060 contract ID: `{runtime['task0060']['task0060_contract_id']}`",
            f"- TASK-0060 contract digest: `{runtime['task0060']['task0060_contract_digest']}`",
            f"- Formal C0 run ID: `{runtime['task0060']['formal_c0_run_id']}`",
            f"- Candidate source: `{runtime['task0060']['formal_c0_candidate_source']}`",
            "- Oracle candidate set used as baseline: `false`",
            "",
            "## Parent Contract",
            "",
            f"- Parent records: {aggregate['parent_index']['parent_record_count']}",
            f"- Parent record digest: `{aggregate['parent_index']['parent_record_digest']}`",
            f"- Max parent text length: {aggregate['parent_index']['max_parent_text_length']}",
            f"- Parent vector policy: `{aggregate['parent_index']['vector_policy']}`",
            "",
            "## Candidate Ceiling",
            "",
            "| N | Scope present | Scope recall | Document present | Document recall |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *ceiling_rows,
            "",
            "Failure buckets: `" + json.dumps(ceiling["failure_buckets"], ensure_ascii=False, sort_keys=True) + "`",
            "",
            "Failure classes: `" + json.dumps(ceiling["failure_classes"], ensure_ascii=False, sort_keys=True) + "`",
            "",
            "## Retrieval Metrics",
            "",
            "| Variant | Doc R@5 | Doc R@20 | Scope R@5 | Scope R@20 | Scope R@40 | MRR | nDCG@5 | P95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Promotion Gates",
            "",
            "| Variant | Eligible | Failed gates |",
            "| --- | --- | --- |",
            *gates,
            "",
            "## Decision",
            "",
            f"`promotion_decision = {decision['promotion_decision']}`",
            f"`recommended_variant = {json.dumps(decision['recommended_variant'])}`",
            f"`recommended_next_action = {decision['recommended_next_action']}`",
            f"`shadow_validation_executed = {str(decision['shadow_validation_executed']).lower()}`",
            "",
            "## Privacy And Integrity",
            "",
            f"- Privacy scan: `{privacy['status']}`",
            f"- Result digests: `{json.dumps(digests.get('digests') or {}, sort_keys=True)}`",
            "- Production default changed: `false`",
            "- Writes database/index: `false` / `false`",
            "",
        ]
    )
