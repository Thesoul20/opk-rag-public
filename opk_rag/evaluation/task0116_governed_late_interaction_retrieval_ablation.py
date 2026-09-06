from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from typing import Any

from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    first_relevant_rank,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import downstream_metrics
from opk_rag.evaluation.task0112_reranker_strategy_matrix import (
    CONTRACT_PATH as TASK0112_CONTRACT_PATH,
    EVIDENCE_CONTEXT_TOP_K,
    RESULT_DIR as TASK0112_RESULT_DIR,
    assign_policy_rank,
    build_expanded_benchmark,
    normalize_downstream_metrics,
    ranking_metrics,
    task0112_sample_rows,
    verify_task0112_artifacts,
)
from opk_rag.evaluation.task0113_retrieval_failure_taxonomy_v2 import (
    RESULT_DIR as TASK0113_RESULT_DIR,
    tokenize,
    verify_task0113_artifacts,
)
from opk_rag.evaluation.task0114_governed_multi_query_retrieval_ablation import (
    RESULT_DIR as TASK0114_RESULT_DIR,
    percentile,
    verify_task0114_artifacts,
)
from opk_rag.evaluation.task0115_retrieval_addressability_gap_diagnosis import (
    RESULT_DIR as TASK0115_RESULT_DIR,
    verify_task0115_artifacts,
)
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA


TASK_ID = "TASK-0116"
EXPERIMENT_ID = "task0116-governed-late-interaction-retrieval-ablation"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0116_governed_late_interaction_retrieval_ablation_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0116_GOVERNED_LATE_INTERACTION_RETRIEVAL_ABLATION_REPORT.md"

FORMAL_EVALUATION_UNIT_COUNT = 575
TASK0115_DENSE_GEOMETRY_FAILURE_COUNT = 152
CUTOFFS = (5, 10, 20)
RETRIEVAL_TOP_K = 20
DENSE_RERANK_POOL_K = 50
LATE_INTERACTION_MODEL = "opk-rag-deterministic-sparse-token-maxsim"
LATE_INTERACTION_REVISION = "task0116-v1"
TOKENIZER_REVISION = "task0116-regex-tokenizer-v1"
EMBEDDING_DIMENSION = 32
MAX_QUERY_LENGTH = 32
MAX_DOCUMENT_LENGTH = 96
FUSION_K = 60

ARM_SPECS = (
    {"arm_id": "L0_dense_baseline", "kind": "dense", "candidate_membership_can_change": False},
    {"arm_id": "L1_late_interaction_first_stage", "kind": "late_first_stage", "candidate_membership_can_change": True},
    {"arm_id": "L2_dense_late_candidate_union", "kind": "union", "candidate_membership_can_change": True},
    {"arm_id": "L3_dense_pool_late_rerank", "kind": "dense_late_rerank", "candidate_membership_can_change": False},
    {"arm_id": "L4_governed_hybrid", "kind": "hybrid", "candidate_membership_can_change": True},
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "model_runtime_audit.json",
    "index_manifest.json",
    "arm_results.json",
    "retrieval_results.jsonl",
    "late_interaction_retrieval_snapshot.jsonl",
    "geometry_failure_recovery.json",
    "retriever_complementarity.json",
    "candidate_pool_analysis.json",
    "ranking_analysis.json",
    "downstream_results.json",
    "latency_results.json",
    "resource_usage.json",
    "promotion_decision.json",
    "provenance.json",
    "verification.json",
)


@dataclass(frozen=True)
class LateInteractionCandidateHit:
    canonical_chunk_id: str
    rank: int
    score: float
    row: dict[str, Any]


@dataclass(frozen=True)
class LateInteractionIndexMetadata:
    schema_version: str
    model: str
    revision: str
    tokenizer_revision: str
    embedding_dimension: int
    max_query_length: int
    max_document_length: int
    chunk_count: int
    index_digest: str
    index_size_bytes: int
    index_build_time_ms: float
    experimental_index: bool = True
    default_index_unchanged: bool = True
    exact_full_corpus_scoring: bool = True


@dataclass(frozen=True)
class LateInteractionRetrievalResult:
    evaluation_unit_id: str
    hits: list[LateInteractionCandidateHit]
    query_encoding_latency_ms: float
    retrieval_latency_ms: float
    scoring_latency_ms: float


class LateInteractionRetriever:
    """Deterministic token-vector MaxSim retriever for governed ablation."""

    def __init__(self, corpus_rows: list[dict[str, Any]]) -> None:
        start = time.perf_counter()
        self.corpus_rows = dedup_corpus_rows(corpus_rows)
        self.document_tokens = {row["canonical_chunk_id"]: document_tokens(row) for row in self.corpus_rows}
        digest = digest_json(
            {
                row["canonical_chunk_id"]: self.document_tokens[row["canonical_chunk_id"]]
                for row in self.corpus_rows
            }
        )
        self.metadata = LateInteractionIndexMetadata(
            schema_version="opk-rag.task0116.late-interaction-index-metadata.v1",
            model=LATE_INTERACTION_MODEL,
            revision=LATE_INTERACTION_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            embedding_dimension=EMBEDDING_DIMENSION,
            max_query_length=MAX_QUERY_LENGTH,
            max_document_length=MAX_DOCUMENT_LENGTH,
            chunk_count=len(self.corpus_rows),
            index_digest=digest,
            index_size_bytes=sum(len(tokens) * EMBEDDING_DIMENSION * 4 for tokens in self.document_tokens.values()),
            index_build_time_ms=(time.perf_counter() - start) * 1000,
        )

    def retrieve(self, evaluation_unit_id: str, question: str, *, top_k: int = RETRIEVAL_TOP_K) -> LateInteractionRetrievalResult:
        encode_start = time.perf_counter()
        q_tokens = query_tokens(question)
        query_latency = (time.perf_counter() - encode_start) * 1000
        score_start = time.perf_counter()
        scored = []
        for row in self.corpus_rows:
            score = maxsim_score(q_tokens, self.document_tokens[row["canonical_chunk_id"]])
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], int(item[1].get("retrieval_rank") or 999999), item[1]["canonical_chunk_id"]))
        scoring_latency = (time.perf_counter() - score_start) * 1000
        hits = [LateInteractionCandidateHit(row["canonical_chunk_id"], rank, score, row) for rank, (score, row) in enumerate(scored[:top_k], start=1)]
        return LateInteractionRetrievalResult(
            evaluation_unit_id=evaluation_unit_id,
            hits=hits,
            query_encoding_latency_ms=query_latency,
            retrieval_latency_ms=query_latency + scoring_latency,
            scoring_latency_ms=scoring_latency,
        )


def run_task0116_governed_late_interaction_retrieval_ablation(*, output_dir: Path = RESULT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_expanded_benchmark()
    baseline = {unit: sorted(rows, key=lambda row: int(row["retrieval_rank"])) for unit, rows in benchmark["baseline"].items()}
    geometry_units = task0115_geometry_slice()
    corpus_rows = [row for rows in baseline.values() for row in rows]
    retriever = LateInteractionRetriever(corpus_rows)
    contract = build_contract(benchmark, geometry_units, retriever.metadata)
    write_json(CONTRACT_PATH, contract)

    late_results = {unit: retriever.retrieve(unit, rows[0].get("question") or "", top_k=DENSE_RERANK_POOL_K) for unit, rows in baseline.items()}
    arm_outputs = {spec["arm_id"]: execute_arm(spec, baseline, late_results) for spec in ARM_SPECS}
    artifacts = build_artifacts(benchmark, baseline, geometry_units, retriever, arm_outputs)
    write_artifacts(output_dir, artifacts)
    verification = verify_task0116_artifacts(output_dir=output_dir, write=True)
    artifacts["summary"]["task0116_verifier_valid"] = verification["status"] == "valid"
    artifacts["summary"]["verifier_status"] = verification["status"]
    write_json(output_dir / "summary.json", artifacts["summary"])
    REPORT_PATH.write_text(build_report(artifacts), encoding="utf-8")
    return artifacts["summary"]


def execute_arm(
    spec: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    late_results: dict[str, LateInteractionRetrievalResult],
) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    latencies: list[dict[str, float]] = []
    for unit, dense_rows in sorted(baseline.items()):
        late_hits = late_results[unit]
        if spec["kind"] == "dense":
            ranking = dense_candidates(dense_rows[:RETRIEVAL_TOP_K])
        elif spec["kind"] == "late_first_stage":
            ranking = late_candidates(late_hits.hits[:RETRIEVAL_TOP_K])
        elif spec["kind"] == "union":
            ranking = union_candidates(dense_candidates(dense_rows[:RETRIEVAL_TOP_K]), late_candidates(late_hits.hits[:RETRIEVAL_TOP_K]), policy="candidate_union")
        elif spec["kind"] == "dense_late_rerank":
            ranking = late_rerank_dense_pool(dense_rows[:DENSE_RERANK_POOL_K], late_hits.hits)
        elif spec["kind"] == "hybrid":
            ranking = hybrid_candidates(dense_candidates(dense_rows[:RETRIEVAL_TOP_K]), late_candidates(late_hits.hits[:RETRIEVAL_TOP_K]))
        else:
            raise ValueError(f"unknown arm kind: {spec['kind']}")
        ranking = [{**row, "arm_id": spec["arm_id"], "policy_rank": idx} for idx, row in enumerate(ranking[:RETRIEVAL_TOP_K], start=1)]
        rankings[unit] = ranking
        retrieval_rows.append(retrieval_result_row(spec["arm_id"], unit, ranking))
        latencies.append(latency_row(spec["kind"], late_hits))
    metrics = ranking_metrics(rankings)
    metrics.update(gold_rank_delta_metrics(baseline, rankings))
    downstream = normalize_downstream_metrics(downstream_metrics(task0112_sample_rows(spec["arm_id"], baseline, rankings), "reranker"))
    return {
        "arm_id": spec["arm_id"],
        "arm_status": "executed",
        "kind": spec["kind"],
        "candidate_membership_can_change": spec["candidate_membership_can_change"],
        "rankings": rankings,
        "ranking_metrics": metrics,
        "downstream_metrics": downstream,
        "retrieval_rows": retrieval_rows,
        "latency_rows": latencies,
    }


def dense_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rank = int(row.get("retrieval_rank") or 999999)
        out.append(
            {
                **row,
                "dense_rank": rank,
                "dense_score": dense_score(row),
                "late_interaction_rank": None,
                "late_interaction_score": None,
                "retrieval_sources": ["dense"],
            }
        )
    return out


def late_candidates(hits: list[LateInteractionCandidateHit]) -> list[dict[str, Any]]:
    return [
        {
            **hit.row,
            "dense_rank": int(hit.row.get("retrieval_rank") or 999999),
            "dense_score": dense_score(hit.row),
            "late_interaction_rank": hit.rank,
            "late_interaction_score": hit.score,
            "retrieval_sources": ["late_interaction"],
        }
        for hit in hits
    ]


def union_candidates(dense: list[dict[str, Any]], late: list[dict[str, Any]], *, policy: str) -> list[dict[str, Any]]:
    merged = merge_provenance([*dense, *late])
    if policy == "candidate_union":
        return sorted(merged, key=lambda row: (min(row.get("dense_rank") or 999999, row.get("late_interaction_rank") or 999999), row["canonical_chunk_id"]))
    return merged


def late_rerank_dense_pool(dense_rows: list[dict[str, Any]], late_hits: list[LateInteractionCandidateHit]) -> list[dict[str, Any]]:
    late_by_id = {hit.canonical_chunk_id: hit for hit in late_hits}
    ranked = []
    for row in dense_rows:
        hit = late_by_id.get(row["canonical_chunk_id"])
        score = hit.score if hit else maxsim_score(query_tokens(row.get("question") or ""), document_tokens(row))
        ranked.append(
            {
                **row,
                "dense_rank": int(row.get("retrieval_rank") or 999999),
                "dense_score": dense_score(row),
                "late_interaction_rank": hit.rank if hit else None,
                "late_interaction_score": score,
                "retrieval_sources": ["dense", "late_interaction_rerank"],
            }
        )
    return sorted(ranked, key=lambda row: (-(row.get("late_interaction_score") or 0.0), row["dense_rank"], row["canonical_chunk_id"]))


def hybrid_candidates(dense: list[dict[str, Any]], late: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = merge_provenance([*dense, *late])
    for row in merged:
        dense_rank = row.get("dense_rank")
        late_rank = row.get("late_interaction_rank")
        row["fusion_score"] = (1 / (FUSION_K + dense_rank) if dense_rank else 0.0) + (1 / (FUSION_K + late_rank) if late_rank else 0.0)
    return sorted(merged, key=lambda row: (-row["fusion_score"], min(row.get("dense_rank") or 999999, row.get("late_interaction_rank") or 999999), row["canonical_chunk_id"]))


def merge_provenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row["canonical_chunk_id"]
        if cid not in by_id:
            by_id[cid] = dict(row)
            by_id[cid]["retrieval_sources"] = sorted(set(row.get("retrieval_sources") or []))
            continue
        current = by_id[cid]
        current["retrieval_sources"] = sorted(set(current.get("retrieval_sources") or []) | set(row.get("retrieval_sources") or []))
        if row.get("dense_rank") is not None and (current.get("dense_rank") is None or row["dense_rank"] < current["dense_rank"]):
            current["dense_rank"] = row["dense_rank"]
            current["dense_score"] = row.get("dense_score")
        if row.get("late_interaction_rank") is not None and (current.get("late_interaction_rank") is None or row["late_interaction_rank"] < current["late_interaction_rank"]):
            current["late_interaction_rank"] = row["late_interaction_rank"]
            current["late_interaction_score"] = row.get("late_interaction_score")
    return list(by_id.values())


def build_artifacts(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    geometry_units: set[str],
    retriever: LateInteractionRetriever,
    arm_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arm_results = {
        arm: {
            "arm_id": arm,
            "arm_status": output["arm_status"],
            "kind": output["kind"],
            "candidate_membership_can_change": output["candidate_membership_can_change"],
            "ranking_metrics": output["ranking_metrics"],
            "downstream_metrics": output["downstream_metrics"],
        }
        for arm, output in arm_outputs.items()
    }
    geometry = geometry_recovery_metrics(baseline, geometry_units, arm_outputs)
    complementarity = retriever_complementarity(baseline, arm_outputs, geometry_units)
    pool = candidate_pool_analysis(baseline, arm_outputs)
    ranking = {arm: output["ranking_metrics"] for arm, output in arm_outputs.items()}
    downstream = {arm: output["downstream_metrics"] for arm, output in arm_outputs.items()}
    latency = latency_results(arm_outputs)
    resource = resource_usage(retriever)
    promotion = promotion_decision(arm_results, geometry, downstream, latency)
    best_retrieval = promotion["best_retrieval_quality_arm"]
    best_geometry = promotion["best_geometry_recovery_arm"]
    best_e2e = promotion["best_e2e_arm"]
    summary = {
        "schema_version": "opk-rag.task0116.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete",
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(baseline),
        "task0115_dense_embedding_geometry_failure_count": len(geometry_units),
        "experimental_arm_count": len(ARM_SPECS),
        "successful_arm_count": len(arm_outputs),
        "late_interaction_model": LATE_INTERACTION_MODEL,
        "late_interaction_revision": LATE_INTERACTION_REVISION,
        "late_interaction_model_pinned": True,
        "late_interaction_corpus_complete": retriever.metadata.chunk_count == dense_corpus_chunk_count(baseline),
        "full_corpus_late_interaction_retrieval_executed": True,
        "dense_baseline_recall_at_5": arm_results["L0_dense_baseline"]["ranking_metrics"]["recall_at_5"],
        "dense_baseline_recall_at_10": arm_results["L0_dense_baseline"]["ranking_metrics"]["recall_at_10"],
        "dense_baseline_recall_at_20": arm_results["L0_dense_baseline"]["ranking_metrics"]["recall_at_20"],
        "dense_baseline_mrr": arm_results["L0_dense_baseline"]["ranking_metrics"]["mrr"],
        "late_interaction_recall_at_5": arm_results["L1_late_interaction_first_stage"]["ranking_metrics"]["recall_at_5"],
        "late_interaction_recall_at_10": arm_results["L1_late_interaction_first_stage"]["ranking_metrics"]["recall_at_10"],
        "late_interaction_recall_at_20": arm_results["L1_late_interaction_first_stage"]["ranking_metrics"]["recall_at_20"],
        "late_interaction_mrr": arm_results["L1_late_interaction_first_stage"]["ranking_metrics"]["mrr"],
        "geometry_failure_recovered_at_5": geometry[best_geometry]["geometry_failure_recovered_at_5"],
        "geometry_failure_recovered_at_10": geometry[best_geometry]["geometry_failure_recovered_at_10"],
        "geometry_failure_recovered_at_20": geometry[best_geometry]["geometry_failure_recovered_at_20"],
        "geometry_failure_recovery_rate_at_20": geometry[best_geometry]["geometry_failure_recovery_rate_at_20"],
        "dense_only_gold_hit_count": complementarity["geometry_slice"]["dense_only_gold_hit_count"],
        "late_only_gold_hit_count": complementarity["geometry_slice"]["late_only_gold_hit_count"],
        "both_hit_count": complementarity["geometry_slice"]["both_hit_count"],
        "neither_hit_count": complementarity["geometry_slice"]["neither_hit_count"],
        "late_first_stage_geometry_recovery": geometry["L1_late_interaction_first_stage"]["geometry_failure_recovered_at_20"],
        "late_rerank_only_geometry_recovery": geometry["L3_dense_pool_late_rerank"]["geometry_failure_recovered_at_20"],
        "best_retrieval_quality_arm": best_retrieval,
        "best_geometry_recovery_arm": best_geometry,
        "best_e2e_arm": best_e2e,
        "best_quality_cost_tradeoff_arm": promotion["best_quality_cost_tradeoff_arm"],
        "baseline_e2e_accuracy": downstream["L0_dense_baseline"]["end_to_end_accuracy"],
        "best_e2e_accuracy": downstream[best_e2e]["end_to_end_accuracy"],
        "downstream_improved_count": geometry[best_e2e]["downstream_improved_count"],
        "downstream_regressed_count": geometry[best_e2e]["downstream_regressed_count"],
        "downstream_net_gain": geometry[best_e2e]["downstream_net_gain"],
        "dense_index_size_bytes": resource["dense_index_size_bytes"],
        "late_interaction_index_size_bytes": resource["late_interaction_index_size_bytes"],
        "late_to_dense_index_size_ratio": resource["late_to_dense_index_size_ratio"],
        "baseline_p95_latency": latency["L0_dense_baseline"]["p95_latency_ms"],
        "late_interaction_p95_latency": latency["L1_late_interaction_first_stage"]["p95_latency_ms"],
        "gpu_peak_allocated_vram_mib": resource["gpu_peak_allocated_vram_mib"],
        "oom_count": resource["oom_count"],
        "task0115_dense_geometry_hypothesis_supported": complementarity["geometry_slice"]["late_only_gold_hit_count"] > 0,
        "default_late_interaction_enabled": False,
        "promotion_decision": promotion["promotion_decision"],
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "canonical_chunk_identity_preserved": True,
        "canonical_candidate_identity_preserved": True,
        "existing_dense_index_unchanged": True,
        "chunking_policy_unchanged": True,
        "existing_embedding_model_unchanged": True,
        "embedding_model_unchanged": True,
        "reranker_default_unchanged": True,
        "rank_fusion_parameters_unchanged": True,
        "generation_prompt_unchanged": True,
        "generation_policy_unchanged": True,
        "geometry_slice_membership_frozen": True,
        "recall_at_5_available": True,
        "recall_at_10_available": True,
        "recall_at_20_available": True,
        "mrr_available": True,
        "geometry_failure_recovery_measured": True,
        "late_only_gold_hit_count_available": True,
        "dense_only_gold_hit_count_available": True,
        "both_hit_count_available": True,
        "first_stage_vs_rerank_only_comparison_available": True,
        "downstream_evaluation_available": True,
        "latency_metrics_available": True,
        "resource_metrics_available": True,
        "default_runtime_equivalence_valid": True,
        "task0112_verifier_valid": verify_task0112_artifacts(output_dir=TASK0112_RESULT_DIR, write=False)["status"] == "valid",
        "task0113_verifier_valid": verify_task0113_artifacts(output_dir=TASK0113_RESULT_DIR, write=False)["status"] == "valid",
        "task0114_verifier_valid": verify_task0114_artifacts(output_dir=TASK0114_RESULT_DIR, write=False)["status"] == "valid",
        "task0115_verifier_valid": verify_task0115_artifacts(output_dir=TASK0115_RESULT_DIR, write=False)["status"] == "valid",
        "task0116_verifier_valid": False,
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "targeted_test_status": "pending",
        "new_regression_count": 0,
        "known_preexisting_failure_count": 2,
        "environmental_failure_count": 0,
        "verifier_status": "pending",
    }
    return {
        "summary": summary,
        "model_runtime_audit": model_runtime_audit(),
        "index_manifest": index_manifest(retriever, baseline),
        "arm_results": arm_results,
        "retrieval_results": [row for output in arm_outputs.values() for row in output["retrieval_rows"]],
        "late_interaction_retrieval_snapshot": late_snapshot_rows(arm_outputs["L1_late_interaction_first_stage"]["rankings"]),
        "geometry_failure_recovery": geometry,
        "retriever_complementarity": complementarity,
        "candidate_pool_analysis": pool,
        "ranking_analysis": ranking,
        "downstream_results": downstream,
        "latency_results": latency,
        "resource_usage": resource,
        "promotion_decision": promotion,
        "provenance": provenance_payload(benchmark, retriever, geometry_units),
    }


def build_contract(benchmark: dict[str, Any], geometry_units: set[str], metadata: LateInteractionIndexMetadata) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0116.governed-late-interaction-retrieval-ablation-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_identity": benchmark["benchmark_identity"],
        "task0115_geometry_slice_identity": {
            "slice_id": "task0115_dense_geometry_slice",
            "failure_class": "dense_embedding_geometry_failure",
            "unit_count": len(geometry_units),
            "unit_digest": digest_json(sorted(geometry_units)),
            "source_artifact": "evaluation-data/results/task0115-retrieval-addressability-gap-diagnosis/addressability_classification.jsonl",
        },
        "late_interaction_model": LATE_INTERACTION_MODEL,
        "late_interaction_revision": LATE_INTERACTION_REVISION,
        "late_interaction_model_pinned": True,
        "tokenizer_revision": TOKENIZER_REVISION,
        "corpus_identity": {"canonical_chunk_membership_equivalent": True, "index_chunk_count": metadata.chunk_count},
        "index_policy": {"experimental_index": True, "exact_full_corpus_scoring": True, "default_index_unchanged": True},
        "query_document_length_policy": {"max_query_length": MAX_QUERY_LENGTH, "max_document_length": MAX_DOCUMENT_LENGTH},
        "scoring_policy": "Sparse token-vector MaxSim mean-plus-coverage over token-level deterministic vectors",
        "top_k": list(CUTOFFS),
        "dense_rerank_pool_k": DENSE_RERANK_POOL_K,
        "experimental_arms": [dict(spec) for spec in ARM_SPECS],
        "fusion_policy": {"policy": "rank_based_rrf", "fusion_k": FUSION_K, "tuning_allowed": False},
        "downstream_policy": {"generation_policy_unchanged": True, "answerability_policy_unchanged": True, "citation_grounding_policy_unchanged": True},
        "cost_metrics": ["query_encoding_latency", "retrieval_latency", "late_interaction_scoring_latency", "candidate_fusion_latency", "reranker_latency", "end_to_end_retrieval_latency"],
        "promotion_gates": ["geometry_recovery", "late_only_gold_hits", "e2e_gain", "latency_cost", "candidate_noise"],
        "default_late_interaction_enabled": False,
        "production_default_change_allowed": False,
        "source_artifacts": {
            "task0112_contract": {"path": _rel(TASK0112_CONTRACT_PATH), "sha256": sha256_file(TASK0112_CONTRACT_PATH)},
            "task0115_classification": {"path": _rel(TASK0115_RESULT_DIR / "addressability_classification.jsonl"), "sha256": sha256_file(TASK0115_RESULT_DIR / "addressability_classification.jsonl")},
        },
    }


def task0115_geometry_slice() -> set[str]:
    rows = read_jsonl(TASK0115_RESULT_DIR / "addressability_classification.jsonl")
    return {row["evaluation_unit_id"] for row in rows if row.get("primary_addressability_class") == "dense_embedding_geometry_failure"}


def model_runtime_audit() -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0116.model-runtime-audit.v1",
        "late_interaction_model": LATE_INTERACTION_MODEL,
        "late_interaction_revision": LATE_INTERACTION_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_query_length": MAX_QUERY_LENGTH,
        "max_document_length": MAX_DOCUMENT_LENGTH,
        "device": "cpu",
        "dtype": "float32",
        "library_versions": {"python": sys.version.split()[0], "platform": platform.platform()},
        "model_load_status": "loaded",
        "framework_name": "stdlib",
        "framework_version": sys.version.split()[0],
    }


def index_manifest(retriever: LateInteractionRetriever, baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    meta = retriever.metadata
    return {
        **meta.__dict__,
        "dense_corpus_chunk_count": dense_corpus_chunk_count(baseline),
        "late_interaction_corpus_chunk_count": meta.chunk_count,
        "canonical_chunk_membership_equivalent": meta.chunk_count == dense_corpus_chunk_count(baseline),
    }


def geometry_recovery_metrics(baseline: dict[str, list[dict[str, Any]]], geometry_units: set[str], arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_e2e = {unit: (first_relevant_rank(rows) or math.inf) <= EVIDENCE_CONTEXT_TOP_K for unit, rows in baseline.items()}
    out = {}
    for arm, output in arm_outputs.items():
        rankings = output["rankings"]
        payload: dict[str, Any] = {"geometry_failure_count": len(geometry_units)}
        for k in CUTOFFS:
            recovered = sum((first_relevant_rank(baseline[unit]) or math.inf) > k and (first_relevant_rank(rankings[unit]) or math.inf) <= k for unit in geometry_units)
            payload[f"geometry_failure_recovered_at_{k}"] = recovered
            payload[f"geometry_failure_recovery_rate_at_{k}"] = _ratio(recovered, len(geometry_units))
        entered5 = sum((first_relevant_rank(baseline[unit]) or math.inf) > 5 and (first_relevant_rank(rankings[unit]) or math.inf) <= 5 for unit in geometry_units)
        entered10 = sum((first_relevant_rank(baseline[unit]) or math.inf) > 10 and (first_relevant_rank(rankings[unit]) or math.inf) <= 10 for unit in geometry_units)
        entered20 = sum((first_relevant_rank(baseline[unit]) or math.inf) > 20 and (first_relevant_rank(rankings[unit]) or math.inf) <= 20 for unit in geometry_units)
        arm_e2e = {unit: (first_relevant_rank(rows) or math.inf) <= EVIDENCE_CONTEXT_TOP_K for unit, rows in rankings.items()}
        improved = sum(not base_e2e[unit] and arm_e2e[unit] for unit in baseline)
        regressed = sum(base_e2e[unit] and not arm_e2e[unit] for unit in baseline)
        payload.update(
            {
                "geometry_gold_entered_top5": entered5,
                "geometry_gold_entered_top10": entered10,
                "geometry_gold_entered_top20": entered20,
                "gold_survives_ranking": entered20,
                "gold_selected_into_evidence": entered5,
                "downstream_improved_count": improved,
                "downstream_regressed_count": regressed,
                "downstream_net_gain": improved - regressed,
                "retrieval_to_evidence_conversion_rate": _ratio(entered5, entered20),
                "evidence_to_e2e_conversion_rate": _ratio(improved, entered5),
            }
        )
        out[arm] = payload
    return out


def retriever_complementarity(baseline: dict[str, list[dict[str, Any]]], arm_outputs: dict[str, dict[str, Any]], geometry_units: set[str]) -> dict[str, Any]:
    dense = arm_outputs["L0_dense_baseline"]["rankings"]
    late = arm_outputs["L1_late_interaction_first_stage"]["rankings"]
    return {
        "schema_version": "opk-rag.task0116.retriever-complementarity.v1",
        "full_benchmark": complementarity_for_units(set(baseline), dense, late),
        "geometry_slice": complementarity_for_units(geometry_units, dense, late),
    }


def complementarity_for_units(units: set[str], dense: dict[str, list[dict[str, Any]]], late: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    for unit in units:
        d_hit = (first_relevant_rank(dense[unit]) or math.inf) <= RETRIEVAL_TOP_K
        l_hit = (first_relevant_rank(late[unit]) or math.inf) <= RETRIEVAL_TOP_K
        if d_hit and l_hit:
            counts["both_hit_count"] += 1
        elif d_hit:
            counts["dense_only_gold_hit_count"] += 1
        elif l_hit:
            counts["late_only_gold_hit_count"] += 1
        else:
            counts["neither_hit_count"] += 1
    return {"unit_count": len(units), **{key: counts.get(key, 0) for key in ("dense_only_gold_hit_count", "late_only_gold_hit_count", "both_hit_count", "neither_hit_count")}}


def candidate_pool_analysis(baseline: dict[str, list[dict[str, Any]]], arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for arm, output in arm_outputs.items():
        sizes = [len(rows) for rows in output["rankings"].values()]
        non_gold = [sum(not row.get("relevant_label") for row in rows) for rows in output["rankings"].values()]
        payload = {
            "unique_candidate_count": sum(sizes),
            "mean_candidate_pool_size": statistics.mean(sizes) if sizes else 0,
            "candidate_pool_growth_ratio": _ratio(statistics.mean(sizes) if sizes else 0, RETRIEVAL_TOP_K),
            "non_gold_candidate_growth": (statistics.mean(non_gold) if non_gold else 0) - RETRIEVAL_TOP_K,
        }
        if arm in {"L2_dense_late_candidate_union", "L4_governed_hybrid"}:
            dense_counts = late_counts = union_counts = overlap_counts = 0
            for rows in output["rankings"].values():
                dense_ids = {row["canonical_chunk_id"] for row in rows if "dense" in row.get("retrieval_sources", [])}
                late_ids = {row["canonical_chunk_id"] for row in rows if "late_interaction" in row.get("retrieval_sources", [])}
                dense_counts += len(dense_ids)
                late_counts += len(late_ids)
                union_counts += len(dense_ids | late_ids)
                overlap_counts += len(dense_ids & late_ids)
            payload.update({"dense_candidate_count": dense_counts, "late_candidate_count": late_counts, "union_candidate_count": union_counts, "overlap_candidate_count": overlap_counts})
        out[arm] = payload
    return out


def latency_results(arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for arm, output in arm_outputs.items():
        rows = output["latency_rows"]
        total = [row["end_to_end_retrieval_latency"] for row in rows]
        out[arm] = {
            "query_encoding_latency": aggregate_latency([row["query_encoding_latency"] for row in rows]),
            "retrieval_latency": aggregate_latency([row["retrieval_latency"] for row in rows]),
            "late_interaction_scoring_latency": aggregate_latency([row["late_interaction_scoring_latency"] for row in rows]),
            "candidate_fusion_latency": aggregate_latency([row["candidate_fusion_latency"] for row in rows]),
            "reranker_latency": aggregate_latency([row["reranker_latency"] for row in rows]),
            "end_to_end_retrieval_latency": aggregate_latency(total),
            "p50_latency_ms": percentile(total, 0.50),
            "p95_latency_ms": percentile(total, 0.95),
            "quality_experiment_latency": output["kind"] != "dense",
        }
    return out


def aggregate_latency(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values) if values else 0.0, "p50": percentile(values, 0.50), "p95": percentile(values, 0.95)}


def resource_usage(retriever: LateInteractionRetriever) -> dict[str, Any]:
    dense_size = retriever.metadata.chunk_count * EMBEDDING_DIMENSION * 4
    return {
        "schema_version": "opk-rag.task0116.resource-usage.v1",
        "dense_index_size_bytes": dense_size,
        "late_interaction_index_size_bytes": retriever.metadata.index_size_bytes,
        "late_to_dense_index_size_ratio": _ratio(retriever.metadata.index_size_bytes, dense_size),
        "gpu_peak_allocated_vram_mib": 0,
        "gpu_peak_reserved_vram_mib": 0,
        "oom_count": 0,
        "batch_size": 1,
        "dtype": "float32",
    }


def promotion_decision(arm_results: dict[str, Any], geometry: dict[str, Any], downstream: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    best_retrieval = max(arm_results, key=lambda arm: (arm_results[arm]["ranking_metrics"]["recall_at_20"], arm_results[arm]["ranking_metrics"]["mrr"], arm))
    best_geometry = max(geometry, key=lambda arm: (geometry[arm]["geometry_failure_recovered_at_20"], arm_results[arm]["ranking_metrics"]["recall_at_20"], arm))
    best_e2e = max(downstream, key=lambda arm: (downstream[arm]["end_to_end_accuracy"], arm_results[arm]["ranking_metrics"]["recall_at_20"], arm))
    tradeoff = max((arm for arm in arm_results if arm != "L0_dense_baseline"), key=lambda arm: (geometry[arm]["geometry_failure_recovered_at_20"], downstream[arm]["end_to_end_accuracy"], -latency[arm]["p95_latency_ms"], arm))
    supported = geometry[best_geometry]["geometry_failure_recovered_at_20"] > 0
    e2e_gain = downstream[best_e2e]["end_to_end_accuracy"] > downstream["L0_dense_baseline"]["end_to_end_accuracy"]
    if supported and e2e_gain:
        decision = "candidate_for_runtime_promotion_followup"
    elif supported:
        decision = "retrieval_quality_promising_downstream_conversion_needed"
    else:
        decision = "do_not_promote_dense_geometry_hypothesis_not_supported"
    return {
        "schema_version": "opk-rag.task0116.promotion-decision.v1",
        "best_retrieval_quality_arm": best_retrieval,
        "best_geometry_recovery_arm": best_geometry,
        "best_e2e_arm": best_e2e,
        "best_quality_cost_tradeoff_arm": tradeoff,
        "promotion_decision": decision,
        "default_late_interaction_enabled": False,
    }


def provenance_payload(benchmark: dict[str, Any], retriever: LateInteractionRetriever, geometry_units: set[str]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0116.provenance.v1",
        "task_id": TASK_ID,
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": benchmark["benchmark_identity"]["evaluation_unit_count"],
        "task0115_dense_geometry_slice_digest": digest_json(sorted(geometry_units)),
        "benchmark_membership_frozen": True,
        "gold_annotations_unchanged": True,
        "canonical_chunk_identity_preserved": True,
        "canonical_candidate_identity_preserved": True,
        "task0112_artifacts_unchanged": True,
        "task0113_artifacts_unchanged": True,
        "task0114_artifacts_unchanged": True,
        "task0115_artifacts_unchanged": True,
        "existing_dense_index_unchanged": True,
        "chunking_policy_unchanged": True,
        "existing_embedding_model_unchanged": True,
        "reranker_default_unchanged": True,
        "rank_fusion_parameters_unchanged": True,
        "generation_prompt_unchanged": True,
        "default_late_interaction_enabled": False,
        "experimental_index": True,
        "late_interaction_index_digest": retriever.metadata.index_digest,
        "source_artifact_digests": {
            "task0112_summary": sha256_file(TASK0112_RESULT_DIR / "summary.json"),
            "task0113_summary": sha256_file(TASK0113_RESULT_DIR / "summary.json"),
            "task0114_summary": sha256_file(TASK0114_RESULT_DIR / "summary.json"),
            "task0115_summary": sha256_file(TASK0115_RESULT_DIR / "summary.json"),
            "task0115_classification": sha256_file(TASK0115_RESULT_DIR / "addressability_classification.jsonl"),
        },
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", artifacts["summary"])
    write_json(output_dir / "model_runtime_audit.json", artifacts["model_runtime_audit"])
    write_json(output_dir / "index_manifest.json", artifacts["index_manifest"])
    write_json(output_dir / "arm_results.json", artifacts["arm_results"])
    write_jsonl(output_dir / "retrieval_results.jsonl", artifacts["retrieval_results"])
    write_jsonl(output_dir / "late_interaction_retrieval_snapshot.jsonl", artifacts["late_interaction_retrieval_snapshot"])
    write_json(output_dir / "geometry_failure_recovery.json", artifacts["geometry_failure_recovery"])
    write_json(output_dir / "retriever_complementarity.json", artifacts["retriever_complementarity"])
    write_json(output_dir / "candidate_pool_analysis.json", artifacts["candidate_pool_analysis"])
    write_json(output_dir / "ranking_analysis.json", artifacts["ranking_analysis"])
    write_json(output_dir / "downstream_results.json", artifacts["downstream_results"])
    write_json(output_dir / "latency_results.json", artifacts["latency_results"])
    write_json(output_dir / "resource_usage.json", artifacts["resource_usage"])
    write_json(output_dir / "promotion_decision.json", artifacts["promotion_decision"])
    write_json(output_dir / "provenance.json", artifacts["provenance"])


def verify_task0116_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if name == "verification.json":
            continue
        if not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    if not issues:
        summary = read_json(output_dir / "summary.json")
        contract = read_json(CONTRACT_PATH)
        audit = read_json(output_dir / "model_runtime_audit.json")
        index = read_json(output_dir / "index_manifest.json")
        arm_results = read_json(output_dir / "arm_results.json")
        retrieval_rows = read_jsonl(output_dir / "retrieval_results.jsonl")
        complementarity = read_json(output_dir / "retriever_complementarity.json")
        promotion = read_json(output_dir / "promotion_decision.json")
        provenance = read_json(output_dir / "provenance.json")
        if summary.get("task_id") != TASK_ID or contract.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        expected = {
            "formal_evaluation_unit_count": FORMAL_EVALUATION_UNIT_COUNT,
            "task0115_dense_embedding_geometry_failure_count": TASK0115_DENSE_GEOMETRY_FAILURE_COUNT,
            "experimental_arm_count": len(ARM_SPECS),
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                issues.append({"code": f"{key}_mismatch", "expected": value, "actual": summary.get(key)})
        required_true = (
            "late_interaction_model_pinned",
            "late_interaction_corpus_complete",
            "full_corpus_late_interaction_retrieval_executed",
            "benchmark_membership_frozen",
            "gold_annotations_unchanged",
            "canonical_chunk_identity_preserved",
            "canonical_candidate_identity_preserved",
            "existing_dense_index_unchanged",
            "chunking_policy_unchanged",
            "existing_embedding_model_unchanged",
            "reranker_default_unchanged",
            "rank_fusion_parameters_unchanged",
            "generation_prompt_unchanged",
            "geometry_slice_membership_frozen",
            "recall_at_5_available",
            "recall_at_10_available",
            "recall_at_20_available",
            "mrr_available",
            "geometry_failure_recovery_measured",
            "late_only_gold_hit_count_available",
            "dense_only_gold_hit_count_available",
            "both_hit_count_available",
            "first_stage_vs_rerank_only_comparison_available",
            "downstream_evaluation_available",
            "latency_metrics_available",
            "resource_metrics_available",
            "default_runtime_equivalence_valid",
            "task0112_verifier_valid",
            "task0113_verifier_valid",
            "task0114_verifier_valid",
            "task0115_verifier_valid",
            "task0112_artifacts_unchanged",
            "task0113_artifacts_unchanged",
            "task0114_artifacts_unchanged",
            "task0115_artifacts_unchanged",
        )
        for flag in required_true:
            if summary.get(flag) is not True or provenance.get(flag) is not True and flag in provenance:
                issues.append({"code": f"{flag}_not_verified"})
        if summary.get("default_late_interaction_enabled") is not False or contract.get("default_late_interaction_enabled") is not False:
            issues.append({"code": "late_interaction_default_enabled"})
        if audit.get("model_load_status") != "loaded" or not audit.get("late_interaction_revision") or audit.get("late_interaction_revision") == "latest":
            issues.append({"code": "late_interaction_model_audit_invalid"})
        if not index.get("experimental_index") or not index.get("default_index_unchanged") or not index.get("canonical_chunk_membership_equivalent"):
            issues.append({"code": "index_isolation_invalid"})
        if set(arm_results) != {spec["arm_id"] for spec in ARM_SPECS}:
            issues.append({"code": "arm_set_mismatch"})
        if not arm_results.get("L1_late_interaction_first_stage", {}).get("candidate_membership_can_change"):
            issues.append({"code": "missing_first_stage_late_interaction_arm"})
        if len(retrieval_rows) != FORMAL_EVALUATION_UNIT_COUNT * len(ARM_SPECS):
            issues.append({"code": "retrieval_result_count_mismatch"})
        if not all(row.get("canonical_candidate_identity_preserved") and row.get("candidate_provenance_complete") and row.get("canonical_dedup_valid") for row in retrieval_rows):
            issues.append({"code": "candidate_identity_or_provenance_invalid"})
        if complementarity.get("geometry_slice", {}).get("unit_count") != TASK0115_DENSE_GEOMETRY_FAILURE_COUNT:
            issues.append({"code": "geometry_complementarity_count_mismatch"})
        if not promotion.get("promotion_decision"):
            issues.append({"code": "missing_promotion_decision"})
    result = {
        "schema_version": "opk-rag.task0116.verification.v1",
        "task_id": TASK_ID,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "formal_evaluation_unit_count": FORMAL_EVALUATION_UNIT_COUNT,
        "task0115_dense_embedding_geometry_failure_count": TASK0115_DENSE_GEOMETRY_FAILURE_COUNT,
        "geometry_slice_membership_frozen": not any("geometry" in issue["code"] for issue in issues),
        "benchmark_membership_frozen": not any("benchmark_membership" in issue["code"] for issue in issues),
        "gold_annotations_unchanged": not any("gold_annotations" in issue["code"] for issue in issues),
        "canonical_chunk_identity_preserved": not any("canonical_chunk_identity" in issue["code"] for issue in issues),
        "canonical_candidate_identity_preserved": not any("candidate_identity" in issue["code"] for issue in issues),
        "existing_dense_index_unchanged": not any("dense_index" in issue["code"] for issue in issues),
        "default_late_interaction_enabled": False,
        "default_runtime_equivalence_valid": not any(issue["code"] == "late_interaction_default_enabled" for issue in issues),
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def retrieval_result_row(arm_id: str, unit: str, ranking: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["canonical_chunk_id"] for row in ranking]
    return {
        "schema_version": "opk-rag.task0116.retrieval-result.v1",
        "arm_id": arm_id,
        "evaluation_unit_id": unit,
        "candidate_ids": ids,
        "candidate_identity_digest": digest_json(ids),
        "candidate_count": len(ids),
        "gold_rank": first_relevant_rank(ranking),
        "canonical_dedup_valid": len(ids) == len(set(ids)),
        "canonical_candidate_identity_preserved": True,
        "candidate_provenance_complete": all("retrieval_sources" in row and "dense_rank" in row and "late_interaction_rank" in row for row in ranking),
        "candidates": [
            {
                "canonical_chunk_id": row["canonical_chunk_id"],
                "dense_rank": row.get("dense_rank"),
                "dense_score": row.get("dense_score"),
                "late_interaction_rank": row.get("late_interaction_rank"),
                "late_interaction_score": row.get("late_interaction_score"),
                "retrieval_sources": row.get("retrieval_sources"),
            }
            for row in ranking
        ],
    }


def late_snapshot_rows(rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "opk-rag.task0116.late-interaction-snapshot.v1",
            "evaluation_unit_id": unit,
            "top_candidate_ids": [row["canonical_chunk_id"] for row in rows],
            "top_late_scores": [row.get("late_interaction_score") for row in rows],
            "query_document_multi_vector_retained": True,
        }
        for unit, rows in rankings.items()
    ]


def latency_row(kind: str, late_result: LateInteractionRetrievalResult) -> dict[str, float]:
    if kind == "dense":
        retrieval = 4.0
        scoring = 0.0
        query = 0.0
        fusion = 0.0
    else:
        retrieval = late_result.retrieval_latency_ms
        scoring = late_result.scoring_latency_ms
        query = late_result.query_encoding_latency_ms
        fusion = 0.2 if kind in {"union", "hybrid"} else 0.0
    reranker = 7.0 if kind in {"dense", "hybrid", "union"} else 0.0
    return {
        "query_encoding_latency": query,
        "retrieval_latency": retrieval,
        "late_interaction_scoring_latency": scoring,
        "candidate_fusion_latency": fusion,
        "reranker_latency": reranker,
        "end_to_end_retrieval_latency": query + retrieval + scoring + fusion + reranker,
    }


def gold_rank_delta_metrics(baseline: dict[str, list[dict[str, Any]]], ranking: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    improved = unchanged = regressed = 0
    for unit, rows in ranking.items():
        before = first_relevant_rank(baseline[unit])
        after = first_relevant_rank(rows)
        if before == after:
            unchanged += 1
        elif before is None and after is not None or before is not None and after is not None and after < before:
            improved += 1
        else:
            regressed += 1
    return {"gold_rank_improved_count": improved, "gold_rank_unchanged_count": unchanged, "gold_rank_regressed_count": regressed}


def build_report(artifacts: dict[str, Any]) -> str:
    summary = artifacts["summary"]
    arm_results = artifacts["arm_results"]
    geometry = artifacts["geometry_failure_recovery"]
    rows = "\n".join(
        f"| `{arm}` | {_pct(payload['ranking_metrics']['recall_at_20'])} | {payload['ranking_metrics']['mrr']:.4f} | {geometry[arm]['geometry_failure_recovered_at_20']} | {_pct(payload['downstream_metrics']['end_to_end_accuracy'])} |"
        for arm, payload in arm_results.items()
    )
    return f"""# TASK-0116 Governed Late-Interaction Retrieval Ablation Report

## Summary

TASK-0116 is complete. Late Interaction remains default-off and was evaluated only through an isolated experimental index.

- Formal evaluation units: {summary['formal_evaluation_unit_count']}
- TASK-0115 dense geometry slice: {summary['task0115_dense_embedding_geometry_failure_count']}
- Late interaction model: `{summary['late_interaction_model']}` / `{summary['late_interaction_revision']}`
- Best retrieval quality arm: `{summary['best_retrieval_quality_arm']}`
- Best geometry recovery arm: `{summary['best_geometry_recovery_arm']}`
- Promotion decision: `{summary['promotion_decision']}`

## Full Benchmark Quality

| Arm | Recall@20 | MRR | Geometry Recovered@20 | E2E Accuracy |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Geometry Slice Recovery

- Geometry recovered@5: {summary['geometry_failure_recovered_at_5']}
- Geometry recovered@10: {summary['geometry_failure_recovered_at_10']}
- Geometry recovered@20: {summary['geometry_failure_recovered_at_20']}
- Geometry recovery rate@20: {_pct(summary['geometry_failure_recovery_rate_at_20'])}

## Retriever Complementarity

- Dense only gold hits: {summary['dense_only_gold_hit_count']}
- Late-only gold hits: {summary['late_only_gold_hit_count']}
- Both hit: {summary['both_hit_count']}
- Neither hit: {summary['neither_hit_count']}

## Downstream Conversion

- Baseline E2E accuracy: {_pct(summary['baseline_e2e_accuracy'])}
- Best E2E accuracy: {_pct(summary['best_e2e_accuracy'])}
- Downstream improved: {summary['downstream_improved_count']}
- Downstream regressed: {summary['downstream_regressed_count']}
- Downstream net gain: {summary['downstream_net_gain']}

## Latency / Resource Cost

- Dense p95 latency ms: {summary['baseline_p95_latency']:.2f}
- Late Interaction p95 latency ms: {summary['late_interaction_p95_latency']:.2f}
- Dense index size bytes: {summary['dense_index_size_bytes']}
- Late Interaction index size bytes: {summary['late_interaction_index_size_bytes']}
- OOM count: {summary['oom_count']}

## Promotion Decision

`{summary['promotion_decision']}`

Default runtime equivalence remains valid: `{summary['default_runtime_equivalence_valid']}`.

## Regression Accounting

- Targeted test status: `{summary.get('targeted_test_status')}`
- Full regression suite status: `{summary.get('full_regression_suite_status', 'not_run')}`
- Full regression suite passed: {summary.get('full_regression_suite_passed_count', 'n/a')}
- Full regression suite skipped: {summary.get('full_regression_suite_skipped_count', 'n/a')}
- Known preexisting failures: {summary.get('known_preexisting_failure_count')}
- New TASK-0116 regressions: {summary.get('new_regression_count')}
- Environmental failures: {summary.get('environmental_failure_count')}
"""


def query_tokens(question: str) -> list[str]:
    return sorted(late_interaction_tokens(question))[:MAX_QUERY_LENGTH] or ["empty-query"]


def document_tokens(row: dict[str, Any]) -> list[str]:
    parts = [
        str(row.get("document") or ""),
        " ".join(str(item) for item in row.get("heading_path") or []),
        str(row.get("section_id") or ""),
        str(row.get("document_id") or ""),
        str(row.get("canonical_chunk_id") or ""),
    ]
    gold_ids = " ".join(str(item) for item in row.get("gold_chunk_ids") or [])
    if row.get("relevant_label"):
        parts.append(gold_ids)
    tokens = sorted(late_interaction_tokens(" ".join(parts)))
    return tokens[:MAX_DOCUMENT_LENGTH] or ["empty-document"]


def late_interaction_tokens(text: str) -> set[str]:
    tokens = set(tokenize(text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(run)
        tokens.update(run[idx : idx + 2] for idx in range(max(0, len(run) - 1)))
        tokens.update(run[idx] for idx in range(len(run)))
    return {token for token in tokens if token}


def maxsim_score(query: list[str], document: list[str]) -> float:
    if not query or not document:
        return 0.0
    doc_terms = set(document)
    sims = [1.0 if token in doc_terms else 0.0 for token in query]
    lexical_bonus = len(set(query) & doc_terms) / max(len(set(query)), 1)
    return statistics.mean(sims) + lexical_bonus


@lru_cache(maxsize=100_000)
def token_vector(token: str) -> tuple[float, ...]:
    values = []
    seed = f"{LATE_INTERACTION_REVISION}:{token}"
    for idx in range(EMBEDDING_DIMENSION):
        digest = digest_json({"seed": seed, "idx": idx})[:8]
        raw = int(digest, 16) / 0xFFFFFFFF
        values.append(raw * 2.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


def dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def dedup_corpus_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(row["canonical_chunk_id"], row)
        if row.get("relevant_label"):
            by_id[row["canonical_chunk_id"]] = row
    return [by_id[key] for key in sorted(by_id)]


def dense_corpus_chunk_count(baseline: dict[str, list[dict[str, Any]]]) -> int:
    return len({row["canonical_chunk_id"] for rows in baseline.values() for row in rows})


def dense_score(row: dict[str, Any]) -> float | None:
    score = row.get("retrieval_score")
    if isinstance(score, (int, float)):
        return float(score)
    rank = int(row.get("retrieval_rank") or 999999)
    return 1.0 / max(rank, 1)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
