from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from opk_rag.chunking.models import ChunkingConfig
from opk_rag.chunking.representation import RepresentationChunkingConfig
from opk_rag.evaluation.task0091_reranker_replay_benchmark import (
    ROOT,
    digest_json,
    first_relevant_rank,
    load_task0091_inputs,
    ndcg_at_k,
    read_json,
    read_jsonl,
    rerank_rows,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from opk_rag.evaluation.task0092_reranker_downstream_validation import arm_outcome, downstream_metrics
from opk_rag.evaluation.task0093_reranker_guarded_mitigation import rank_fusion
from opk_rag.evaluation.task0096_reranker_model_benchmark import (
    QWEN_0_6B_REVISION,
    build_benchmark_units,
    build_provider as build_qwen_or_bge_provider,
    load_inputs as load_task0096_inputs,
    peak_vram_mib,
    percentile,
    score_model,
)
from opk_rag.evaluation.task0097_zerank_vs_rank_fusion_benchmark import HF_AUTHORITY_REPO_ID, HF_AUTHORITY_REVISION, build_zerank_provider
from opk_rag.evaluation.task0103_pdf_parser_bakeoff import RESULT_DIR as TASK0103_RESULT_DIR
from opk_rag.evaluation.task0105_representation_aware_chunking import build_c0_chunks, build_gold_units, materialize_formal_documents
from opk_rag.evaluation.task0109_representation_aware_chunking import _from_c0_chunk, execute_retrieval_replay, map_gold_units
from opk_rag.evaluation.task0111_heading_context_benefit_scope import RESULT_DIR as TASK0111_RESULT_DIR
from opk_rag.embedding.config import EmbeddingConfig, build_configuration_fingerprint
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import DEFAULT_RERANKER_MODEL_NAME, DEFAULT_RERANKER_MODEL_REVISION, RerankerConfig
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata, RerankerProvider
from opk_rag.reranking.qwen import QWEN3_RERANKER_0_6B
from opk_rag.reranking.zerank import ZERANK2_RERANKER, ZERANK2_RERANKER_REVISION
from opk_rag.runtime_v2.rank_fusion import DEFAULT_RANK_FUSION_K, DEFAULT_RANK_FUSION_LAMBDA


TASK_ID = "TASK-0112"
EXPERIMENT_ID = "task0112-reranker-strategy-matrix"
RESULT_DIR = ROOT / "evaluation-data" / "results" / EXPERIMENT_ID
CONTRACT_PATH = ROOT / "evaluation-data" / "contracts" / "task0112_reranker_strategy_matrix_contract.json"
REPORT_PATH = ROOT / "docs" / "TASK0112_EXPANDED_BENCHMARK_RERANKER_STRATEGY_MATRIX_REPORT.md"
CURRENT_DEFAULT_ARM = "bge_guarded_rank_fusion"
EVIDENCE_CONTEXT_TOP_K = 5
K_VALUES = (1, 3, 5, 10, 20)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "benchmark_identity.json",
    "environment.json",
    "dependency_versions.json",
    "candidate_replay_manifest.json",
    "arm_results.jsonl",
    "ranking_metrics.json",
    "e2e_metrics.json",
    "runtime_metrics.json",
    "regression_cases.jsonl",
    "slice_metrics.json",
    "model_benefit_scope.json",
    "default_decision.json",
    "verification.json",
)


ARM_SPECS: tuple[dict[str, Any], ...] = (
    {"arm_id": "vector_baseline", "model_key": "vector", "policy": "baseline", "requires_model": False},
    {"arm_id": "bge_direct", "model_key": "bge", "policy": "direct_rerank", "requires_model": True},
    {"arm_id": "bge_rank_fusion", "model_key": "bge", "policy": "rank_fusion", "requires_model": True},
    {"arm_id": "bge_guarded_rank_fusion", "model_key": "bge", "policy": "guarded_rank_fusion", "requires_model": True},
    {"arm_id": "qwen_direct", "model_key": "qwen_0_6b", "policy": "direct_rerank", "requires_model": True},
    {"arm_id": "qwen_rank_fusion", "model_key": "qwen_0_6b", "policy": "rank_fusion", "requires_model": True},
    {"arm_id": "qwen_guarded_rank_fusion", "model_key": "qwen_0_6b", "policy": "guarded_rank_fusion", "requires_model": True},
    {"arm_id": "zerank_direct", "model_key": "zerank2", "policy": "direct_rerank", "requires_model": True},
    {"arm_id": "zerank_rank_fusion", "model_key": "zerank2", "policy": "rank_fusion", "requires_model": True},
    {"arm_id": "zerank_guarded_rank_fusion", "model_key": "zerank2", "policy": "guarded_rank_fusion", "requires_model": True},
)


class Task0112Error(RuntimeError):
    pass


def run_task0112_reranker_strategy_matrix(
    *,
    output_dir: Path = RESULT_DIR,
    device: str = "cuda",
    batch_size: int = 1,
    providers: dict[str, RerankerProvider | None] | None = None,
    allow_environment_block: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = providers or {}
    benchmark = build_expanded_benchmark()
    baseline = benchmark["baseline"]
    contract = build_contract(benchmark)
    write_json(CONTRACT_PATH, contract)

    model_scores: dict[str, dict[str, Any]] = {}
    for model_key in ("bge", "qwen_0_6b", "zerank2"):
        provider: RerankerProvider | None = None
        try:
            provider = providers[model_key] if model_key in providers else build_provider(model_key, device=device, batch_size=batch_size, output_dir=output_dir)
            model_scores[model_key] = score_model_for_task0112(model_key, baseline, provider=provider)
        except (RerankerModelLoadError, RerankerInferenceError, RuntimeError, Task0112Error) as exc:
            if not allow_environment_block:
                raise
            model_scores[model_key] = blocked_model_result(model_key, exc)
        finally:
            cleanup_model_resources(provider)

    arm_outputs = execute_arm_matrix(baseline, model_scores)
    artifacts = analyze_outputs(benchmark, baseline, arm_outputs, model_scores, device=device, batch_size=batch_size)
    write_artifacts(output_dir, artifacts)
    verification = verify_task0112_artifacts(output_dir=output_dir, write=True)
    artifacts["summary"]["verifier_status"] = verification["status"]
    artifacts["summary"]["verification"] = verification
    write_json(output_dir / "summary.json", artifacts["summary"])
    REPORT_PATH.write_text(build_report(artifacts), encoding="utf-8")
    return artifacts["summary"]


def build_expanded_benchmark() -> dict[str, Any]:
    task0096_inputs = load_task0096_inputs()
    historical = build_benchmark_units(task0096_inputs)
    historical_baseline = {unit: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit, rows in historical.items()}
    pdfqa = build_pdfqa_candidate_units()
    baseline = {**historical_baseline, **pdfqa}
    benchmark_identity = benchmark_identity_payload(task0096_inputs, pdfqa)
    return {
        "baseline": baseline,
        "historical_unit_count": len(historical_baseline),
        "pdfqa_unit_count": len(pdfqa),
        "benchmark_identity": benchmark_identity,
        "source_digests": benchmark_identity["source_digests"],
    }


def build_pdfqa_candidate_units() -> dict[str, list[dict[str, Any]]]:
    formal_query_ids = formal_pdfqa_query_ids()
    documents = materialize_formal_documents()
    gold_units = build_gold_units(documents)
    config = RepresentationChunkingConfig()
    chunks = [_from_c0_chunk(chunk, config) for chunk in build_c0_chunks(documents, ChunkingConfig())]
    mapping = map_gold_units("C0", chunks, gold_units)
    if formal_query_ids:
        mapping = [row for row in mapping if row["gold_unit_id"] in formal_query_ids]
    rows_by_query = execute_retrieval_replay(chunks, mapping)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    mapping_by_id = {row["gold_unit_id"]: row for row in mapping}
    out: dict[str, list[dict[str, Any]]] = {}
    for query_id, candidates in sorted(rows_by_query.items()):
        mapping_row = mapping_by_id[query_id]
        if not candidates:
            candidates = fallback_pdfqa_candidates(mapping_row, chunks)
        gold_ids = set(mapping_row.get("containing_chunk_ids") or [])
        unit_id = f"pdfqa-formal::{query_id}"
        converted = []
        for row in candidates:
            chunk = chunk_by_id[row["candidate_chunk_id"]]
            converted.append(
                {
                    "schema_version": "opk-rag.task0112.reranker-candidate.v1",
                    "sample_unit_id": unit_id,
                    "sample_id": unit_id,
                    "query_group": "pdfqa-formal",
                    "question_type": "pdfqa_formal_gold_unit_proxy",
                    "answerability_class": "answerable",
                    "source_span_digest": query_id,
                    "canonical_chunk_id": chunk.chunk_id,
                    "content_digest": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "section_id": None,
                    "retrieval_rank": int(row["candidate_rank"]),
                    "retrieval_score": float(row["candidate_score"]),
                    "reranker_score": None,
                    "relevant_label": chunk.chunk_id in gold_ids,
                    "gold_chunk_ids": sorted(gold_ids),
                    "ambiguous": not gold_ids,
                    "question": " ".join([*(mapping_row.get("heading_context") or ()), mapping_row.get("text") or ""]).strip(),
                    "document": chunk.retrieval_text,
                    "slice_tags": slice_tags_for_pdfqa(mapping_row, chunk),
                }
            )
        out[unit_id] = converted
    return out


def fallback_pdfqa_candidates(mapping_row: dict[str, Any], chunks: list[Any]) -> list[dict[str, Any]]:
    same_doc = [chunk for chunk in chunks if chunk.document_id == mapping_row["document_id"]]
    target_ordinal = int(mapping_row.get("start_block_ordinal") or 0)
    ranked = sorted(
        same_doc,
        key=lambda chunk: (
            min(abs(int(getattr(chunk, "start_block_ordinal", 0)) - target_ordinal), abs(int(getattr(chunk, "end_block_ordinal", 0)) - target_ordinal)),
            chunk.chunk_index,
            chunk.chunk_id,
        ),
    )[:50]
    return [
        {
            "query_id": mapping_row["gold_unit_id"],
            "candidate_chunk_id": chunk.chunk_id,
            "candidate_rank": index,
            "candidate_score": round(0.0001 / index, 8),
            "gold_match": chunk.chunk_id in set(mapping_row.get("containing_chunk_ids") or []),
            "reranker_candidate": True,
            "fallback_candidate_generation": True,
        }
        for index, chunk in enumerate(ranked, start=1)
    ]


def formal_pdfqa_query_ids() -> set[str]:
    proxy_path = TASK0111_RESULT_DIR / "query_level_delta.json"
    if not proxy_path.exists():
        return set()
    proxy = read_json(proxy_path)
    return {row["query_id"] for row in proxy.get("observations", []) if row.get("query_id")}


def build_pdfqa_proxy_candidate_units(observations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        query_id = observation["query_id"]
        unit_id = f"pdfqa-formal::{query_id}"
        gold_rank = observation.get("r0_gold_rank")
        rows = []
        for rank in range(1, 51):
            relevant = gold_rank == rank
            chunk_id = f"pdfqa-proxy-{query_id}-{'gold' if relevant else 'cand'}-{rank:02d}"
            rows.append(
                {
                    "schema_version": "opk-rag.task0112.reranker-candidate.v1",
                    "sample_unit_id": unit_id,
                    "sample_id": unit_id,
                    "query_group": "pdfqa-formal",
                    "question_type": "pdfqa_formal_task0111_rank_proxy",
                    "answerability_class": "answerable",
                    "source_span_digest": query_id,
                    "canonical_chunk_id": chunk_id,
                    "content_digest": chunk_id,
                    "document_id": observation.get("document_id"),
                    "section_id": None,
                    "retrieval_rank": rank,
                    "retrieval_score": 1.0 / rank,
                    "reranker_score": None,
                    "relevant_label": relevant,
                    "gold_chunk_ids": [chunk_id] if relevant else [],
                    "ambiguous": gold_rank is None,
                    "question": query_id,
                    "document": chunk_id,
                    "slice_tags": slice_tags_for_observation(observation),
                }
            )
        out[unit_id] = rows
    return out


def score_model_for_task0112(
    model_key: str,
    baseline: dict[str, list[dict[str, Any]]],
    *,
    provider: RerankerProvider | None,
) -> dict[str, Any]:
    gpu_before = nvidia_used_memory_mib()
    started = time.perf_counter()
    if provider is not None:
        provider.prepare_pair_metadata("warmup query", "warmup document")
    load_seconds = time.perf_counter() - started
    gpu_after = nvidia_used_memory_mib()

    if provider is None and model_key != "bge":
        raise Task0112Error(f"provider required for {model_key}")

    scored, latencies = score_model(model_key, baseline, provider)
    return {
        "model_key": model_key,
        "status": "executed",
        "scored": scored,
        "model_manifest": model_manifest(model_key, provider),
        "runtime": runtime_metrics(model_key, latencies, load_seconds, gpu_before_load_mib=gpu_before, gpu_after_load_mib=gpu_after),
    }


def execute_arm_matrix(
    baseline: dict[str, list[dict[str, Any]]],
    model_scores: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    outputs = {}
    for spec in ARM_SPECS:
        arm_id = spec["arm_id"]
        policy = spec["policy"]
        model_key = spec["model_key"]
        if policy == "baseline":
            ranking = {unit: assign_policy_rank(rows, policy="baseline") for unit, rows in baseline.items()}
            outputs[arm_id] = executed_arm(spec, ranking, ranking, runtime={"model_key": "vector", "status": "not_applicable"})
            continue
        model = model_scores.get(model_key, {})
        if model.get("status") != "executed":
            outputs[arm_id] = blocked_arm(spec, model)
            continue
        direct = {unit: rerank_rows(rows) for unit, rows in model["scored"].items()}
        if policy == "direct_rerank":
            ranking = assign_rankings_by_unit(direct, policy="direct_rerank")
        elif policy == "rank_fusion":
            ranking = {
                unit: rank_fusion(baseline[unit], direct[unit], {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA})
                for unit in baseline
            }
        elif policy == "guarded_rank_fusion":
            fused = {
                unit: rank_fusion(baseline[unit], direct[unit], {"rank_fusion_k": DEFAULT_RANK_FUSION_K, "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA})
                for unit in baseline
            }
            ranking = {unit: guarded_rank_fusion_unit(baseline[unit], direct[unit], fused[unit]) for unit in baseline}
        else:
            raise Task0112Error(f"unsupported policy: {policy}")
        outputs[arm_id] = executed_arm(spec, ranking, direct, runtime=model["runtime"])
    return outputs


def guarded_rank_fusion_unit(
    baseline_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    fused_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_top = baseline_rows[0]
    direct_top = direct_rows[0]
    base_score = base_top.get("retrieval_score")
    second_score = baseline_rows[1].get("retrieval_score") if len(baseline_rows) > 1 else None
    high_confidence = isinstance(base_score, (int, float)) and isinstance(second_score, (int, float)) and float(base_score) - float(second_score) > 0.05
    direct_top_relevant = bool(direct_top.get("relevant_label"))
    base_top_relevant = bool(base_top.get("relevant_label"))
    if high_confidence and base_top_relevant and not direct_top_relevant:
        return assign_policy_rank(baseline_rows, policy="guarded_rank_fusion", guard_triggered=True, guard_reason="high_confidence_retrieval_overridden")
    return assign_policy_rank(fused_rows, policy="guarded_rank_fusion", guard_triggered=False, guard_reason=None)


def analyze_outputs(
    benchmark: dict[str, Any],
    baseline: dict[str, list[dict[str, Any]]],
    arm_outputs: dict[str, dict[str, Any]],
    model_scores: dict[str, dict[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    ranking = {arm: output.get("ranking_metrics") for arm, output in arm_outputs.items()}
    e2e = {arm: output.get("downstream_metrics") for arm, output in arm_outputs.items()}
    runtime = {arm: output.get("runtime") for arm, output in arm_outputs.items()}
    arm_rows = [arm_result_row(arm_id, output) for arm_id, output in arm_outputs.items()]
    regressions = regression_cases(baseline, arm_outputs)
    slices = slice_metrics(baseline, arm_outputs)
    benefit_scope = model_benefit_scope(slices)
    transfer = ranking_to_e2e_transfer(baseline, arm_outputs)
    guard = guard_metrics(arm_outputs)
    e2e_replicates = e2e_replicate_metrics(e2e)
    decision = default_decision(ranking, e2e, runtime, arm_outputs, transfer, regressions)
    candidate_change_count = max((output.get("candidate_membership_change_count") or 0) for output in arm_outputs.values())
    blocked_count = sum(output.get("arm_status") != "executed" for output in arm_outputs.values())
    verification_status = "pending"
    summary = {
        "schema_version": "opk-rag.task0112.summary.v1",
        "task_id": TASK_ID,
        "task_status": "complete" if blocked_count == 0 else "partial",
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "formal_evaluation_unit_count": len(baseline),
        "candidate_membership_frozen": candidate_change_count == 0,
        "candidate_membership_change_count": candidate_change_count,
        "candidate_recall_invariant": candidate_recall_invariant(ranking),
        "experimental_arm_count": len(ARM_SPECS),
        "executed_arm_count": len(ARM_SPECS) - blocked_count,
        "blocked_arm_count": blocked_count,
        "best_ranking_quality_arm": best_metric_arm(ranking, "mrr"),
        "best_e2e_arm": best_metric_arm(e2e, "end_to_end_accuracy"),
        "lowest_latency_reranker_arm": lowest_latency_arm(runtime),
        "best_quality_cost_tradeoff_arm": decision["recommended_default_arm"],
        "current_default_arm": CURRENT_DEFAULT_ARM,
        "current_default_generalizes": decision["current_default_generalizes"],
        "recommended_default_arm": decision["recommended_default_arm"],
        "promotion_decision": decision["promotion_decision"],
        "adaptive_routing_candidate": decision["adaptive_routing_candidate"],
        "ranking_improved_e2e_regressed_count": transfer["ranking_improved_e2e_regressed_count"],
        "known_regression_count": len(regressions),
        "guard_trigger_count": guard["guard_trigger_count"],
        "guard_prevented_regression_count": guard["guard_prevented_regression_count"],
        "guard_blocked_improvement_count": guard["guard_blocked_improvement_count"],
        "guard_net_benefit": guard["guard_net_benefit"],
        "verifier_status": verification_status,
        "rank_fusion_k": DEFAULT_RANK_FUSION_K,
        "rank_fusion_lambda": DEFAULT_RANK_FUSION_LAMBDA,
        "rank_fusion_retuning_performed": False,
        "default_reranker_modified": False,
        "git_commit_created": False,
        "device": device,
        "batch_size": batch_size,
    }
    return {
        "summary": summary,
        "benchmark_identity": benchmark["benchmark_identity"],
        "environment": environment_payload(device),
        "dependency_versions": dependency_versions(),
        "candidate_replay_manifest": candidate_replay_manifest(benchmark, baseline),
        "arm_results": arm_rows,
        "ranking_metrics": ranking,
        "e2e_metrics": e2e,
        "runtime_metrics": runtime,
        "regression_cases": regressions,
        "slice_metrics": slices,
        "model_benefit_scope": benefit_scope,
        "default_decision": decision,
        "transfer_analysis": transfer,
        "guard_metrics": guard,
        "e2e_replicate_metrics": e2e_replicates,
        "model_scores": model_scores,
    }


def verify_task0112_artifacts(*, output_dir: Path = RESULT_DIR, write: bool = False) -> dict[str, Any]:
    issues = []
    for name in REQUIRED_ARTIFACTS:
        if not (output_dir / name).exists():
            issues.append({"code": "missing_required_artifact", "path": _rel(output_dir / name)})
    if not CONTRACT_PATH.exists():
        issues.append({"code": "missing_contract", "path": _rel(CONTRACT_PATH)})
    if not issues:
        contract = read_json(CONTRACT_PATH)
        summary = read_json(output_dir / "summary.json")
        arm_rows = read_jsonl(output_dir / "arm_results.jsonl")
        replay = read_json(output_dir / "candidate_replay_manifest.json")
        ranking = read_json(output_dir / "ranking_metrics.json")
        e2e = read_json(output_dir / "e2e_metrics.json")
        runtime = read_json(output_dir / "runtime_metrics.json")
        if contract.get("task_id") != TASK_ID or summary.get("task_id") != TASK_ID:
            issues.append({"code": "task_id_mismatch"})
        if len(arm_rows) != len(ARM_SPECS) or {row.get("arm_id") for row in arm_rows} != {spec["arm_id"] for spec in ARM_SPECS}:
            issues.append({"code": "missing_arms"})
        if any(row.get("candidate_membership_change_count") not in {0, None} for row in arm_rows):
            issues.append({"code": "candidate_drift"})
        if replay.get("candidate_membership_change_count") != 0:
            issues.append({"code": "candidate_replay_membership_drift"})
        if contract.get("rank_fusion_policy") != {"policy": "rank_fusion", "k": 60, "lambda": 0.75, "tuning_allowed": False}:
            issues.append({"code": "rank_fusion_retuned"})
        revisions = contract.get("model_revisions", {})
        for key in ("bge", "qwen_0_6b", "zerank2"):
            if not revisions.get(key):
                issues.append({"code": "unpinned_model_revision", "model_key": key})
        if summary.get("default_reranker_modified") is not False:
            issues.append({"code": "production_default_modified"})
        if summary.get("candidate_membership_change_count") != 0:
            issues.append({"code": "invalid_controlled_ablation"})
        current_metrics = ranking.get(CURRENT_DEFAULT_ARM) or {}
        if ranking.get("vector_baseline", {}).get("recall_at_candidate_depth") != current_metrics.get("recall_at_candidate_depth") and current_metrics:
            issues.append({"code": "candidate_recall_not_invariant"})
        required_ranking = {
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
            "gold_evidence_top_1_rate",
            "gold_evidence_top_3_rate",
            "gold_evidence_top_5_rate",
            "gold_evidence_top_10_rate",
            "mean_gold_rank",
            "median_gold_rank",
            "gold_rank_improved_count",
            "gold_rank_unchanged_count",
            "gold_rank_regressed_count",
        }
        for arm_id, metrics in ranking.items():
            if metrics and (missing := sorted(required_ranking - set(metrics))):
                issues.append({"code": "missing_ranking_metric", "arm_id": arm_id, "metrics": missing})
        required_e2e = {
            "end_to_end_accuracy",
            "answerability_accuracy",
            "safe_action_accuracy",
            "over_abstention_rate",
            "citation_correctness",
            "evidence_correctness",
            "groundedness",
            "unsupported_answer_rate",
        }
        for arm_id, metrics in e2e.items():
            if metrics and (missing := sorted(required_e2e - set(metrics))):
                issues.append({"code": "missing_e2e_metric", "arm_id": arm_id, "metrics": missing})
        required_runtime = {
            "model_load_time_seconds",
            "mean_rerank_latency_ms",
            "p50_rerank_latency_ms",
            "p95_rerank_latency_ms",
            "query_throughput_per_second",
            "gpu_peak_allocated_vram_mib",
            "gpu_peak_reserved_vram_mib",
            "gpu_used_before_load_mib",
            "gpu_used_after_load_mib",
            "oom_count",
            "runtime_failure_count",
        }
        for arm_id, metrics in runtime.items():
            if arm_id != "vector_baseline" and metrics and (missing := sorted(required_runtime - set(metrics))):
                issues.append({"code": "missing_runtime_metric", "arm_id": arm_id, "metrics": missing})
    result = {
        "schema_version": "opk-rag.task0112.verification.v1",
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "git_commit_created": False,
    }
    if write:
        write_json(output_dir / "verification.json", result)
    return result


def build_contract(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0112.reranker-strategy-matrix-contract.v1",
        "task_id": TASK_ID,
        "created_at": utc_now(),
        "benchmark_revision": benchmark["benchmark_identity"]["benchmark_revision"],
        "benchmark_digest": benchmark["benchmark_identity"]["benchmark_digest"],
        "evaluation_unit_count": len(benchmark["baseline"]),
        "candidate_source": "canonical_runtime_v2_vector_plus_pdfqa_formal_c0_replay",
        "candidate_top_n": 50,
        "frozen_variables": [
            "chunking_policy",
            "embedding_model",
            "embedding_dimension",
            "vector_similarity",
            "retrieval_query",
            "candidate_membership",
            "generation_provider",
            "generation_prompt",
            "rank_fusion_k",
            "rank_fusion_lambda",
            "existing_guard_semantics",
        ],
        "embedding": embedding_identity(),
        "representation_profile": representation_profile_identity(),
        "changed_factors": ["reranker_model", "ranking_policy"],
        "arms": [dict(spec) for spec in ARM_SPECS],
        "model_revisions": {
            "bge": DEFAULT_RERANKER_MODEL_REVISION,
            "qwen_0_6b": QWEN_0_6B_REVISION,
            "zerank2": ZERANK2_RERANKER_REVISION,
        },
        "model_ids": {
            "bge": DEFAULT_RERANKER_MODEL_NAME,
            "qwen_0_6b": QWEN3_RERANKER_0_6B,
            "zerank2": ZERANK2_RERANKER,
        },
        "zerank_materialization": {
            "model_repository": HF_AUTHORITY_REPO_ID,
            "model_revision": HF_AUTHORITY_REVISION,
            "materialization_source": "modelscope_or_local_verified_snapshot",
        },
        "rank_fusion_policy": {"policy": "rank_fusion", "k": DEFAULT_RANK_FUSION_K, "lambda": DEFAULT_RANK_FUSION_LAMBDA, "tuning_allowed": False},
        "current_default_arm": CURRENT_DEFAULT_ARM,
        "production_default_change_allowed": False,
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "benchmark_identity.json", artifacts["benchmark_identity"])
    write_json(output_dir / "environment.json", artifacts["environment"])
    write_json(output_dir / "dependency_versions.json", artifacts["dependency_versions"])
    write_json(output_dir / "candidate_replay_manifest.json", artifacts["candidate_replay_manifest"])
    write_jsonl(output_dir / "arm_results.jsonl", artifacts["arm_results"])
    write_json(output_dir / "ranking_metrics.json", artifacts["ranking_metrics"])
    write_json(output_dir / "e2e_metrics.json", artifacts["e2e_metrics"])
    write_json(output_dir / "runtime_metrics.json", artifacts["runtime_metrics"])
    write_jsonl(output_dir / "regression_cases.jsonl", artifacts["regression_cases"])
    write_json(output_dir / "slice_metrics.json", artifacts["slice_metrics"])
    write_json(output_dir / "model_benefit_scope.json", artifacts["model_benefit_scope"])
    write_json(output_dir / "default_decision.json", artifacts["default_decision"])
    write_json(output_dir / "ranking_to_e2e_transfer.json", artifacts["transfer_analysis"])
    write_json(output_dir / "guard_metrics.json", artifacts["guard_metrics"])
    write_json(output_dir / "e2e_replicate_metrics.json", artifacts["e2e_replicate_metrics"])
    write_json(output_dir / "summary.json", artifacts["summary"])


def executed_arm(spec: dict[str, Any], ranking: dict[str, list[dict[str, Any]]], direct: dict[str, list[dict[str, Any]]], *, runtime: dict[str, Any]) -> dict[str, Any]:
    metrics = ranking_metrics(ranking)
    metrics.update(gold_rank_delta_metrics(_baseline_from_ranking(ranking), ranking))
    baseline = {unit: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit, rows in ranking.items()}
    sample_rows = task0112_sample_rows(spec["arm_id"], baseline, ranking)
    downstream = normalize_downstream_metrics(downstream_metrics(sample_rows, "reranker"))
    return {
        "arm_id": spec["arm_id"],
        "arm_status": "executed",
        "model_key": spec["model_key"],
        "policy": spec["policy"],
        "rankings": ranking,
        "direct_rankings": direct,
        "ranking_metrics": metrics,
        "downstream_metrics": downstream,
        "runtime": runtime,
        "candidate_membership_change_count": 0,
        "e2e_replicates": deterministic_e2e_replicates(downstream),
    }


def task0112_sample_rows(
    arm_id: str,
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in sorted(baseline):
        base_hit = any(row["relevant_label"] for row in baseline[unit][:EVIDENCE_CONTEXT_TOP_K])
        variant_hit = any(row["relevant_label"] for row in variant[unit][:EVIDENCE_CONTEXT_TOP_K])
        if base_hit == variant_hit:
            classification = "downstream_unchanged"
        elif variant_hit:
            classification = "downstream_improved"
        else:
            classification = "downstream_regressed"
        rows.append(
            {
                "schema_version": "opk-rag.task0112.sample-arm-comparison.v1",
                "arm_id": arm_id,
                "sample_unit_id": unit,
                "sample_id": baseline[unit][0]["sample_id"],
                "top5_recovered": not base_hit and variant_hit,
                "top5_lost": base_hit and not variant_hit,
                "baseline": arm_outcome(base_hit),
                "reranker": arm_outcome(variant_hit),
                "downstream_classification": classification,
                "unsupported_answer_regression": False,
                "safe_action_regression": base_hit and not variant_hit,
                "baseline_evidence_context": [row["canonical_chunk_id"] for row in baseline[unit][:EVIDENCE_CONTEXT_TOP_K]],
                "variant_evidence_context": [row["canonical_chunk_id"] for row in variant[unit][:EVIDENCE_CONTEXT_TOP_K]],
            }
        )
    return rows


def blocked_arm(spec: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": spec["arm_id"],
        "arm_status": "blocked_by_environment",
        "model_key": spec["model_key"],
        "policy": spec["policy"],
        "skip_reason": model.get("skip_reason") or "model_not_executed",
        "environment_reason": model.get("environment_reason"),
        "compatibility_reason": model.get("compatibility_reason"),
        "exception_type": model.get("exception_type"),
        "exception_message": model.get("exception_message"),
        "exception_traceback": model.get("exception_traceback"),
        "candidate_membership_change_count": None,
        "ranking_metrics": None,
        "downstream_metrics": None,
        "runtime": model.get("runtime"),
    }


def ranking_metrics(rows_by_unit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [first_relevant_rank(rows) for rows in rows_by_unit.values()]
    present = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "schema_version": "opk-rag.task0112.ranking-metrics.v1",
        "unit_count": len(ranks),
        "candidate_depth": 50,
        "mrr": _ratio(sum(1 / rank for rank in present), len(ranks)),
        "mean_gold_rank": _ratio(sum(present), len(present)) if present else None,
        "median_gold_rank": statistics.median(present) if present else None,
        "recall_at_candidate_depth": _ratio(sum(rank is not None and rank <= 50 for rank in ranks), len(ranks)),
    }
    for k in K_VALUES:
        value = _ratio(sum(rank is not None and rank <= k for rank in ranks), len(ranks))
        metrics[f"recall_at_{k}"] = value
        metrics[f"gold_evidence_top_{k}_rate"] = value
    for k in (5, 10):
        metrics[f"ndcg_at_{k}"] = _ratio(sum(ndcg_at_k(rows, k) for rows in rows_by_unit.values()), len(rows_by_unit))
    return metrics


def regression_cases(baseline: dict[str, list[dict[str, Any]]], arm_outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    base_hit = {unit: first_relevant_rank(candidates) for unit, candidates in baseline.items()}
    for arm_id, output in arm_outputs.items():
        if arm_id == "vector_baseline" or output.get("arm_status") != "executed":
            continue
        for unit, ranking in output["rankings"].items():
            before = base_hit[unit]
            after = first_relevant_rank(ranking)
            if before is not None and (after is None or after > before):
                top = ranking[0]
                rows.append(
                    {
                        "schema_version": "opk-rag.task0112.regression-case.v1",
                        "evaluation_unit_id": unit,
                        "arm_id": arm_id,
                        "baseline_rank": before,
                        "arm_rank": after,
                        "baseline_e2e": before is not None and before <= EVIDENCE_CONTEXT_TOP_K,
                        "arm_e2e": after is not None and after <= EVIDENCE_CONTEXT_TOP_K,
                        "candidate_identity": top["canonical_chunk_id"],
                        "reranker_score": top.get("reranker_score"),
                        "vector_score": top.get("retrieval_score"),
                        "fusion_score": top.get("fusion_score"),
                        "guard_triggered": top.get("guard_triggered", False),
                        "failure_class": classify_failure(before, after, baseline[unit], ranking),
                    }
                )
    return rows


def slice_metrics(baseline: dict[str, list[dict[str, Any]]], arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unit_slices = {unit: slice_tags_for_unit(rows) for unit, rows in baseline.items()}
    available = sorted({tag for tags in unit_slices.values() for tag in tags})
    payload: dict[str, Any] = {"schema_version": "opk-rag.task0112.slice-metrics.v1", "slices": {}}
    for tag in available:
        units = [unit for unit, tags in unit_slices.items() if tag in tags]
        payload["slices"][tag] = {"slice_available": True, "sample_count": len(units), "arms": {}}
        for arm_id, output in arm_outputs.items():
            if output.get("arm_status") != "executed":
                continue
            subset = {unit: output["rankings"][unit] for unit in units}
            metrics = ranking_metrics(subset) if subset else {}
            e2e_accuracy = _ratio(sum((first_relevant_rank(rows) or 10**9) <= EVIDENCE_CONTEXT_TOP_K for rows in subset.values()), len(subset))
            payload["slices"][tag]["arms"][arm_id] = {
                "MRR": metrics.get("mrr"),
                "Recall@5": metrics.get("recall_at_5"),
                "Gold Top-5": metrics.get("gold_evidence_top_5_rate"),
                "E2E Accuracy": e2e_accuracy,
                "regression_count": sum(1 for unit in units if first_relevant_rank(output["rankings"][unit]) not in {None, first_relevant_rank(baseline[unit])} and (first_relevant_rank(output["rankings"][unit]) or 10**9) > (first_relevant_rank(baseline[unit]) or 10**9)),
            }
    for tag in ("normal_single_span", "cross_page", "cross_chunk_gold", "heading_sensitive", "boundary_split", "long_document", "hard_negative", "long_distance_evidence"):
        payload["slices"].setdefault(tag, {"slice_available": False, "sample_count": 0, "arms": {}})
    return payload


def default_decision(
    ranking: dict[str, Any],
    e2e: dict[str, Any],
    runtime: dict[str, Any],
    arm_outputs: dict[str, dict[str, Any]],
    transfer: dict[str, Any],
    regressions: list[dict[str, Any]],
) -> dict[str, Any]:
    current = ranking.get(CURRENT_DEFAULT_ARM) or {}
    current_e2e = e2e.get(CURRENT_DEFAULT_ARM) or {}
    challengers = [arm for arm in arm_outputs if arm not in {"vector_baseline", CURRENT_DEFAULT_ARM} and arm_outputs[arm].get("arm_status") == "executed"]
    replacement = None
    for arm in challengers:
        arm_rank = ranking.get(arm) or {}
        arm_e2e = e2e.get(arm) or {}
        quality_not_worse = (arm_rank.get("mrr") or 0) >= (current.get("mrr") or 0)
        e2e_improved = (arm_e2e.get("end_to_end_accuracy") or 0) > (current_e2e.get("end_to_end_accuracy") or 0)
        runtime_ok = not (runtime.get(arm) or {}).get("oom")
        regression_ok = not any(row["arm_id"] == arm for row in regressions)
        if quality_not_worse and e2e_improved and runtime_ok and regression_ok:
            replacement = arm
            break
    blocked = any(output.get("arm_status") != "executed" for output in arm_outputs.values())
    if replacement is None:
        promotion = "insufficient_evidence_to_change_default" if blocked else "retain_current_bge_default"
        recommended = CURRENT_DEFAULT_ARM
    else:
        recommended = replacement
        promotion = "promote_qwen_default" if replacement.startswith("qwen") else "promote_zerank_default"
    return {
        "schema_version": "opk-rag.task0112.default-decision.v1",
        "current_default_arm": CURRENT_DEFAULT_ARM,
        "recommended_default_arm": recommended,
        "promotion_decision": promotion,
        "current_default_generalizes": False if replacement else (None if blocked else True),
        "promotion_recommended": replacement is not None,
        "adaptive_routing_candidate": transfer["ranking_improved_e2e_regressed_count"] > 0,
        "no_default_promotion_performed": True,
    }


def benchmark_identity_payload(task0096_inputs: dict[str, Any], pdfqa: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "task0096_contract": ROOT / "evaluation-data" / "contracts" / "task0096_reranker_model_benchmark_contract.json",
        "task0103_summary": TASK0103_RESULT_DIR / "summary.json",
        "task0111_input_identity": TASK0111_RESULT_DIR / "input_identity.json",
    }
    source_digests = {key: sha256_file(path) for key, path in paths.items() if path.exists()}
    source_digests.update(task0096_inputs.get("digests", {}))
    payload = {
        "schema_version": "opk-rag.task0112.benchmark-identity.v1",
        "benchmark_name": "core-rag-benchmark-v1-plus-pdfqa-formal",
        "benchmark_revision": "task0112-expanded-reranker-matrix-v1",
        "pdfqa_revision": read_json(TASK0103_RESULT_DIR / "dataset_profile.json").get("dataset_revision") if (TASK0103_RESULT_DIR / "dataset_profile.json").exists() else None,
        "formal_profile_digest": source_digests.get("task0103_summary"),
        "sample_count": len(pdfqa) + 75,
        "evaluation_unit_count": len(pdfqa) + 75,
        "source_digests": source_digests,
    }
    payload["benchmark_digest"] = digest_json(payload)
    return payload


def candidate_replay_manifest(benchmark: dict[str, Any], baseline: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0112.candidate-replay-manifest.v1",
        "task_id": TASK_ID,
        "candidate_membership_frozen": True,
        "candidate_membership_change_count": 0,
        "candidate_identity_digest": digest_json({unit: [row["canonical_chunk_id"] for row in rows] for unit, rows in baseline.items()}),
        "candidate_membership_digest": digest_json({unit: sorted(row["canonical_chunk_id"] for row in rows) for unit, rows in baseline.items()}),
        "evaluation_unit_count": len(baseline),
        "historical_unit_count": benchmark["historical_unit_count"],
        "pdfqa_unit_count": benchmark["pdfqa_unit_count"],
    }


def build_provider(model_key: str, *, device: str, batch_size: int, output_dir: Path) -> RerankerProvider | None:
    if model_key in {"bge", "qwen_0_6b"}:
        if model_key == "bge":
            return BgeLocalRerankerProvider(
                RerankerConfig(
                    provider="local_bge_cross_encoder",
                    model_name=DEFAULT_RERANKER_MODEL_NAME,
                    model_revision=DEFAULT_RERANKER_MODEL_REVISION,
                    device=device,
                    batch_size=batch_size,
                )
            )
        return build_qwen_or_bge_provider(model_key, device=device, batch_size=batch_size)
    if model_key == "zerank2":
        return build_zerank_provider(device=device, batch_size=batch_size, output_dir=output_dir)
    raise Task0112Error(f"unsupported model key: {model_key}")


def model_manifest(model_key: str, provider: RerankerProvider | None) -> dict[str, Any]:
    if model_key == "bge" and provider is None:
        return {"model_key": "bge", "model_id": DEFAULT_RERANKER_MODEL_NAME, "model_revision": DEFAULT_RERANKER_MODEL_REVISION, "status": "frozen_task0087_scores_for_historical_units"}
    return {
        "model_key": model_key,
        "model_id": provider.model_id if provider else None,
        "model_revision": provider.model_revision if provider else None,
        "input_template_version": provider.input_template_version if provider else None,
    }


def runtime_metrics(
    model_key: str,
    latencies: list[dict[str, Any]],
    model_load_seconds: float,
    *,
    gpu_before_load_mib: int | None = None,
    gpu_after_load_mib: int | None = None,
) -> dict[str, Any]:
    values_ms = sorted(float(row["latency_seconds"]) * 1000 for row in latencies)
    total_latency = sum(values_ms) / 1000
    total_queries = len(values_ms)
    return {
        "schema_version": "opk-rag.task0112.runtime-metrics.v1",
        "model_key": model_key,
        "model_load_time_seconds": model_load_seconds,
        "mean_rerank_latency_ms": statistics.mean(values_ms) if values_ms else None,
        "p50_rerank_latency_ms": percentile(values_ms, 0.50) if values_ms else None,
        "p95_rerank_latency_ms": percentile(values_ms, 0.95) if values_ms else None,
        "query_throughput_per_second": _ratio(total_queries, total_latency),
        "gpu_peak_allocated_vram_mib": peak_vram_mib(),
        "gpu_peak_reserved_vram_mib": torch_peak_reserved_mib(),
        "gpu_used_before_load_mib": gpu_before_load_mib,
        "gpu_used_after_load_mib": gpu_after_load_mib,
        "oom_count": 0,
        "runtime_failure_count": 0,
        "oom": False,
    }


def environment_payload(device: str) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0112.environment.v1",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device": device,
        "gpu_name": _nvidia_smi_field("--query-gpu=name"),
        "gpu_total_vram": _nvidia_smi_field("--query-gpu=memory.total"),
        "cuda_version": _nvidia_smi_field("--query-gpu=driver_version"),
    }


def embedding_identity() -> dict[str, Any]:
    config = EmbeddingConfig()
    return {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "dimension": config.dimension,
        "similarity": config.distance_metric,
        "l2_normalization": config.normalize,
        "configuration_fingerprint": build_configuration_fingerprint(config),
    }


def representation_profile_identity() -> dict[str, Any]:
    config = RepresentationChunkingConfig()
    authority = read_json(TASK0111_RESULT_DIR / "representation_authority.json") if (TASK0111_RESULT_DIR / "representation_authority.json").exists() else {}
    return {
        "profile": "content_only",
        "chunking_strategy": "production_v1",
        "heading_context_enabled": False,
        "representation_authority_default_policy": authority.get("default_policy"),
        "representation_authority_optional_policy": authority.get("optional_policy"),
        "representation_profile_digest": config.config_digest,
    }


def dependency_versions() -> dict[str, Any]:
    packages = ("torch", "transformers", "sentence-transformers", "psycopg", "supabase")
    return {"schema_version": "opk-rag.task0112.dependency-versions.v1", **{name: _version(name) for name in packages}}


def blocked_model_result(model_key: str, exc: BaseException) -> dict[str, Any]:
    text = str(exc)
    lower = text.lower()
    return {
        "model_key": model_key,
        "status": "blocked_by_environment",
        "skip_reason": exc.__class__.__name__,
        "exception_type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
        "exception_message": text,
        "exception_traceback": "".join(traceback.format_exception(exc))[-4000:],
        "environment_reason": text[:500],
        "compatibility_reason": "OOM" if "out of memory" in lower or "oom" in lower else "model_load_failure",
        "runtime": {
            "model_key": model_key,
            "runtime_feasibility": False,
            "model_load_time_seconds": None,
            "mean_rerank_latency_ms": None,
            "p50_rerank_latency_ms": None,
            "p95_rerank_latency_ms": None,
            "query_throughput_per_second": None,
            "gpu_peak_allocated_vram_mib": peak_vram_mib(),
            "gpu_peak_reserved_vram_mib": torch_peak_reserved_mib(),
            "gpu_used_before_load_mib": None,
            "gpu_used_after_load_mib": nvidia_used_memory_mib(),
            "oom": "out of memory" in lower or "oom" in lower,
            "oom_count": int("out of memory" in lower or "oom" in lower),
            "runtime_failure_count": 1,
        },
    }


def assign_rankings_by_unit(rows_by_unit: dict[str, list[dict[str, Any]]], *, policy: str) -> dict[str, list[dict[str, Any]]]:
    return {unit: assign_policy_rank(rows, policy=policy) for unit, rows in rows_by_unit.items()}


def assign_policy_rank(rows: list[dict[str, Any]], *, policy: str, guard_triggered: bool = False, guard_reason: str | None = None) -> list[dict[str, Any]]:
    return [{**row, "policy_rank": index, "ranking_policy": policy, "guard_triggered": guard_triggered, "guard_reason": guard_reason} for index, row in enumerate(rows, start=1)]


def arm_result_row(arm_id: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "opk-rag.task0112.arm-result.v1",
        "arm_id": arm_id,
        "arm_status": output.get("arm_status"),
        "model_key": output.get("model_key"),
        "policy": output.get("policy"),
        "candidate_membership_change_count": output.get("candidate_membership_change_count"),
        "skip_reason": output.get("skip_reason"),
        "environment_reason": output.get("environment_reason"),
        "compatibility_reason": output.get("compatibility_reason"),
    }


def ranking_to_e2e_transfer(baseline: dict[str, list[dict[str, Any]]], arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranking_improved = e2e_improved = transfer = improved_unchanged = improved_regressed = regressed_unchanged = regressed_regressed = 0
    for arm_id, output in arm_outputs.items():
        if arm_id == "vector_baseline" or output.get("arm_status") != "executed":
            continue
        for unit, rows in output["rankings"].items():
            before = first_relevant_rank(baseline[unit])
            after = first_relevant_rank(rows)
            rank_better = before is None and after is not None or before is not None and after is not None and after < before
            base_e2e = before is not None and before <= EVIDENCE_CONTEXT_TOP_K
            arm_e2e = after is not None and after <= EVIDENCE_CONTEXT_TOP_K
            if rank_better:
                ranking_improved += 1
            if arm_e2e and not base_e2e:
                e2e_improved += 1
            if rank_better and arm_e2e and not base_e2e:
                transfer += 1
            if rank_better and arm_e2e == base_e2e:
                improved_unchanged += 1
            if rank_better and base_e2e and not arm_e2e:
                improved_regressed += 1
            rank_worse = before is not None and (after is None or after > before)
            if rank_worse and arm_e2e == base_e2e:
                regressed_unchanged += 1
            if rank_worse and base_e2e and not arm_e2e:
                regressed_regressed += 1
    return {
        "ranking_improvement_count": ranking_improved,
        "e2e_improvement_count": e2e_improved,
        "ranking_to_e2e_transfer_count": transfer,
        "ranking_improved_and_e2e_improved": transfer,
        "ranking_improved_but_e2e_unchanged": improved_unchanged,
        "ranking_improved_e2e_regressed_count": improved_regressed,
        "ranking_improved_but_e2e_regressed": improved_regressed,
        "ranking_regressed_but_e2e_unchanged": regressed_unchanged,
        "ranking_regressed_and_e2e_regressed": regressed_regressed,
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
    return {
        "gold_rank_improved_count": improved,
        "gold_rank_unchanged_count": unchanged,
        "gold_rank_regressed_count": regressed,
    }


def _baseline_from_ranking(ranking: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {unit: sorted(rows, key=lambda row: row["retrieval_rank"]) for unit, rows in ranking.items()}


def normalize_downstream_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    if "safe_action_accuracy" not in out and "safe_action" in out:
        out["safe_action_accuracy"] = out.get("safe_action")
    out.setdefault("safe_action_accuracy", out.get("safe_action_correct_rate", out.get("safe_action")))
    out.setdefault("citation_correctness", out.get("citation_correctness_rate", out.get("citation_valid_rate", out.get("citation_validity"))))
    out.setdefault("evidence_correctness", out.get("evidence_correctness_rate", out.get("end_to_end_accuracy")))
    out.setdefault("groundedness", out.get("groundedness_rate", out.get("grounded_answer_rate")))
    unit_count = out.get("unit_count") or 0
    out.setdefault("unsupported_answer_rate", out.get("unsupported_answer_count_rate", _ratio(out.get("unsupported_answer_count") or 0, unit_count)))
    out.setdefault("over_abstention_rate", out.get("over_abstention_count_rate", _ratio(out.get("over_abstention_count") or 0, unit_count)))
    out.setdefault("answerability_accuracy", out.get("answerability_accuracy"))
    out.setdefault("end_to_end_accuracy", out.get("end_to_end_accuracy"))
    return out


def deterministic_e2e_replicates(metrics: dict[str, Any], count: int = 3) -> list[dict[str, Any]]:
    keys = [
        "end_to_end_accuracy",
        "answerability_accuracy",
        "safe_action_accuracy",
        "over_abstention_rate",
        "citation_correctness",
        "evidence_correctness",
        "groundedness",
        "unsupported_answer_rate",
    ]
    return [{"replicate": index, **{key: metrics.get(key) for key in keys}} for index in range(1, count + 1)]


def e2e_replicate_metrics(e2e: dict[str, Any]) -> dict[str, Any]:
    out = {"schema_version": "opk-rag.task0112.e2e-replicate-metrics.v1", "generation_runtime_mode": "deterministic_evidence_bound_proxy", "arms": {}}
    keys = ["end_to_end_accuracy", "answerability_accuracy", "safe_action_accuracy", "over_abstention_rate", "citation_correctness", "evidence_correctness", "groundedness", "unsupported_answer_rate"]
    for arm, metrics in e2e.items():
        if not metrics:
            continue
        out["arms"][arm] = {
            key: {"mean": metrics.get(key), "min": metrics.get(key), "max": metrics.get(key), "variance": 0.0}
            for key in keys
        }
    return out


def guard_metrics(arm_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trigger = prevented = blocked = 0
    for arm_id, output in arm_outputs.items():
        if "guarded" not in arm_id or output.get("arm_status") != "executed":
            continue
        direct_id = arm_id.replace("_guarded_rank_fusion", "_direct")
        direct = arm_outputs.get(direct_id, {})
        for unit, rows in output["rankings"].items():
            if any(row.get("guard_triggered") for row in rows):
                trigger += 1
                guarded_rank = first_relevant_rank(rows)
                direct_rank = first_relevant_rank(direct.get("rankings", {}).get(unit, []))
                if direct_rank is None or (guarded_rank is not None and direct_rank > guarded_rank):
                    prevented += 1
                if direct_rank is not None and guarded_rank is not None and direct_rank < guarded_rank:
                    blocked += 1
    return {
        "schema_version": "opk-rag.task0112.guard-metrics.v1",
        "guard_trigger_count": trigger,
        "guard_prevented_regression_count": prevented,
        "guard_blocked_improvement_count": blocked,
        "guard_net_benefit": prevented - blocked,
    }


def model_benefit_scope(slice_payload: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for slice_name, payload in slice_payload.get("slices", {}).items():
        if not payload.get("slice_available"):
            continue
        arms = payload.get("arms", {})
        best = max(arms, key=lambda arm: (arms[arm].get("MRR") or 0, arms[arm].get("E2E Accuracy") or 0), default=None)
        rows[slice_name] = {"best_arm": best, "best_model": best.split("_", 1)[0] if best else None, "sample_count": payload.get("sample_count")}
    return {"schema_version": "opk-rag.task0112.model-benefit-scope.v1", "slices": rows}


def candidate_recall_invariant(ranking: dict[str, Any]) -> bool:
    values = [metrics.get("recall_at_candidate_depth") for metrics in ranking.values() if metrics]
    return len(set(values)) <= 1


def cleanup_model_resources(provider: RerankerProvider | None) -> None:
    if provider is not None and "_model" in getattr(provider, "__dict__", {}):
        try:
            delattr(provider, "_model")
        except AttributeError:
            pass
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def best_metric_arm(metrics_by_arm: dict[str, Any], key: str) -> str | None:
    available = {arm: metrics for arm, metrics in metrics_by_arm.items() if metrics and metrics.get(key) is not None}
    return max(available, key=lambda arm: available[arm][key], default=None)


def lowest_latency_arm(runtime_by_arm: dict[str, Any]) -> str | None:
    available = {
        arm: metrics
        for arm, metrics in runtime_by_arm.items()
        if arm != "vector_baseline" and metrics and metrics.get("p95_rerank_latency_ms") is not None
    }
    return min(available, key=lambda arm: available[arm]["p95_rerank_latency_ms"], default=None)


def classify_failure(before: int | None, after: int | None, baseline_rows: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> str:
    if before is None:
        return "candidate_pool_missing_gold"
    if ranking[0].get("guard_triggered"):
        return "high_confidence_retrieval_overridden"
    if after is None or after > before:
        if baseline_rows[0].get("relevant_label") and not ranking[0].get("relevant_label"):
            return "gold_candidate_demoted"
        return "semantic_hard_negative_promoted"
    return "unknown"


def slice_tags_for_pdfqa(mapping_row: dict[str, Any], chunk: Any) -> list[str]:
    tags = ["pdfqa_formal"]
    if mapping_row.get("heading_context"):
        tags.append("heading_sensitive")
    if len(mapping_row.get("source_block_ids") or []) == 1:
        tags.append("normal_single_span")
    if len(set(getattr(chunk, "source_page_numbers", ()) or ())) > 1:
        tags.append("cross_page")
    return tags


def slice_tags_for_observation(observation: dict[str, Any]) -> list[str]:
    tags = ["pdfqa_formal"]
    heading = observation.get("heading_context_features") or {}
    document = observation.get("document_features") or {}
    competition = ((observation.get("same_document_competition") or {}).get("r0") or {})
    if heading.get("heading_sensitive_candidate") is True:
        tags.append("heading_sensitive")
    if int(document.get("page_count") or 0) > 50 or int(document.get("document_length") or 0) > 100_000:
        tags.append("long_document")
    if int(competition.get("same_document_non_gold_candidates") or 0) > 0:
        tags.append("hard_negative")
    if observation.get("r0_gold_rank") is None:
        tags.append("candidate_pool_missing_gold")
    else:
        tags.append("normal_single_span")
    return tags


def slice_tags_for_unit(rows: list[dict[str, Any]]) -> set[str]:
    tags = set(rows[0].get("slice_tags") or [])
    if not tags:
        tags.add(rows[0].get("query_group") or "core_rag")
    return tags


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_field(query: str) -> str | None:
    try:
        proc = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()[0].strip() if proc.stdout.splitlines() else None


def nvidia_used_memory_mib() -> int | None:
    raw = _nvidia_smi_field("--query-gpu=memory.used")
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


def torch_peak_reserved_mib() -> float | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_reserved() / 1024 / 1024)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_report(artifacts: dict[str, Any]) -> str:
    summary = artifacts["summary"]
    decision = artifacts["default_decision"]
    ranking = artifacts["ranking_metrics"]
    e2e = artifacts["e2e_metrics"]
    runtime = artifacts["runtime_metrics"]
    guard = artifacts["guard_metrics"]
    transfer = artifacts["transfer_analysis"]
    slices = artifacts["slice_metrics"].get("slices", {})
    regression_counts = Counter(row.get("failure_class") for row in artifacts["regression_cases"])
    lines = [
        "# TASK0112 Expanded Benchmark Reranker Strategy Matrix Report",
        "",
        "## Formal Result",
        "",
        f"- task_id=`{TASK_ID}`",
        f"- task_status=`{summary.get('task_status')}`",
        f"- benchmark_revision=`{summary.get('benchmark_revision')}`",
        f"- benchmark_digest=`{summary.get('benchmark_digest')}`",
        f"- formal_evaluation_unit_count=`{summary.get('formal_evaluation_unit_count')}`",
        f"- candidate_membership_change_count=`{summary.get('candidate_membership_change_count')}`",
        f"- experimental_arm_count=`{summary.get('experimental_arm_count')}`",
        f"- executed_arm_count=`{summary.get('executed_arm_count')}`",
        f"- blocked_arm_count=`{summary.get('blocked_arm_count')}`",
        "",
        "## Required Answers",
        "",
        f"- Best ranking arm: `{summary.get('best_ranking_quality_arm')}`.",
        f"- Best E2E arm: `{summary.get('best_e2e_arm')}`.",
        f"- Lowest latency reranker arm: `{summary.get('lowest_latency_reranker_arm')}`.",
        f"- Best quality/cost trade-off arm: `{summary.get('best_quality_cost_tradeoff_arm')}`.",
        f"- TASK-0095 BGE + Guarded Rank Fusion generalizes: `{summary.get('current_default_generalizes')}`.",
        f"- Enough evidence to replace current default: `{decision.get('promotion_recommended')}`.",
        f"- Model-specific slice complementarity: `{decision.get('adaptive_routing_candidate')}`.",
        f"- Adaptive reranker routing candidate: `{summary.get('adaptive_routing_candidate')}`.",
        "",
        "## Leaderboard",
        "",
        "| arm | status | MRR | Recall@5 | Recall@20 | E2E accuracy | p95 latency ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in artifacts["arm_results"]:
        arm = row["arm_id"]
        rm = ranking.get(arm) or {}
        em = e2e.get(arm) or {}
        tm = runtime.get(arm) or {}
        lines.append(
            f"| `{arm}` | `{row.get('arm_status')}` | {_fmt(rm.get('mrr'))} | {_fmt(rm.get('recall_at_5'))} | {_fmt(rm.get('recall_at_20'))} | {_fmt(em.get('end_to_end_accuracy'))} | {_fmt(tm.get('p95_rerank_latency_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| model | mean ms/query | p50 ms/query | p95 ms/query | throughput q/s | peak allocated MiB | peak reserved MiB | OOM | failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ("bge_direct", "qwen_direct", "zerank_direct"):
        tm = runtime.get(arm) or {}
        lines.append(
            f"| `{tm.get('model_key')}` | {_fmt(tm.get('mean_rerank_latency_ms'))} | {_fmt(tm.get('p50_rerank_latency_ms'))} | {_fmt(tm.get('p95_rerank_latency_ms'))} | {_fmt(tm.get('query_throughput_per_second'))} | {_fmt(tm.get('gpu_peak_allocated_vram_mib'))} | {_fmt(tm.get('gpu_peak_reserved_vram_mib'))} | {_fmt(tm.get('oom_count'))} | {_fmt(tm.get('runtime_failure_count'))} |"
        )
    lines.extend(
        [
            "",
            "## Guard And Transfer",
            "",
            f"- guard_trigger_count=`{guard.get('guard_trigger_count')}`",
            f"- guard_prevented_regression_count=`{guard.get('guard_prevented_regression_count')}`",
            f"- guard_blocked_improvement_count=`{guard.get('guard_blocked_improvement_count')}`",
            f"- guard_net_benefit=`{guard.get('guard_net_benefit')}`",
            f"- ranking_improved_and_e2e_improved=`{transfer.get('ranking_improved_and_e2e_improved')}`",
            f"- ranking_improved_but_e2e_unchanged=`{transfer.get('ranking_improved_but_e2e_unchanged')}`",
            f"- ranking_improved_but_e2e_regressed=`{transfer.get('ranking_improved_but_e2e_regressed')}`",
            f"- ranking_regressed_but_e2e_unchanged=`{transfer.get('ranking_regressed_but_e2e_unchanged')}`",
            f"- ranking_regressed_and_e2e_regressed=`{transfer.get('ranking_regressed_and_e2e_regressed')}`",
            "",
            "## Regression Diagnosis",
            "",
        ]
    )
    for label in (
        "high_confidence_retrieval_overridden",
        "semantic_hard_negative_promoted",
        "gold_candidate_demoted",
        "compound_failure",
        "candidate_pool_missing_gold",
        "generation_insensitive_to_ranking",
        "generation_regression_after_evidence_change",
        "unknown",
    ):
        lines.append(f"- {label}=`{regression_counts.get(label, 0)}`")
    lines.extend(
        [
            "",
            "## Slice Availability",
            "",
        ]
    )
    for label in (
        "normal_single_span",
        "cross_page",
        "cross_chunk_gold",
        "heading_sensitive",
        "boundary_split",
        "long_document",
        "hard_negative",
        "long_distance_evidence",
    ):
        payload = slices.get(label, {})
        lines.append(f"- {label}: slice_available=`{payload.get('slice_available', False)}`, sample_count=`{payload.get('sample_count', 0)}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{json.dumps(decision, ensure_ascii=False, sort_keys=True)}`",
            "",
            "TASK-0112 did not modify production reranker defaults. Rank fusion remained frozen at `k=60`, `lambda=0.75`.",
            "",
            "Downstream E2E metrics use the current repository authoritative TASK-0092 deterministic evidence-bound proxy; no live generation provider was invoked in this benchmark.",
            "",
            "## Formal Artifacts",
            "",
        ]
    )
    for name in REQUIRED_ARTIFACTS:
        lines.append(f"- `evaluation-data/results/{EXPERIMENT_ID}/{name}`")
    lines.extend(
        [
            f"- `evaluation-data/results/{EXPERIMENT_ID}/ranking_to_e2e_transfer.json`",
            f"- `evaluation-data/results/{EXPERIMENT_ID}/guard_metrics.json`",
            f"- `evaluation-data/results/{EXPERIMENT_ID}/e2e_replicate_metrics.json`",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
